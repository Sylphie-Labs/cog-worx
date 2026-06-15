"""Scope-promotion surface — structural routing of agent-scope claims to world/user scopes.

Routes active agent-scope claims to world or user scopes via developer-authored
``PromotionRule`` objects.  Routing decisions are ENTIRELY structural (field values, counts,
numeric thresholds) — the claim's payload text content is NEVER read or used for routing
(CANON S9: structure over prompting; no model-as-routing-signal).

Pod 2.7 carry-forward CF-4 (from Pod 2.4): automated agent→user/world promotion.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cogworx.claims.provenance import EpistemicType
from cogworx.substrate.entity_kg import EntityKG, ScoredClaim

if TYPE_CHECKING:
    from cogworx.substrate.coherence import CoherenceStore

__all__ = [
    "PromotionRule",
    "ScopePromoter",
    "SourceKind",
]

SourceKind = str
"""Source-kind prefix convention (e.g. 'human', 'tool', 'system').

A framework-minted ``source_id`` has the form ``f"source:{kind}:{ref}"``.  The promoter checks
that at least one evidence event's ``source_id`` starts with ``f"source:{kind}:"``.  Source ids
are always framework-assigned (S9), so this prefix test is safe as a structural routing signal.
"""


@dataclass(frozen=True)
class PromotionRule:
    """Developer-authored promotion rule.  Config-over-code: devs register these, not models.

    All matching criteria are structural — no claim payload text is read for routing (S9).

    Attributes:
        target: Name of the target scope (must be a key in ``ScopePromoter.sinks``).
        predicate_norm: When not ``None``, the claim's normalised predicate must equal this.
        subject_norm: When not ``None``, the claim's normalised subject must equal this.
        min_distinct_sources: Minimum count of distinct ``source_id`` values across the claim's
            evidence events.  Uses the deduped source count for positive-polarity evidence.
        require_source_kind: When not ``None``, at least one evidence event's ``source_id`` must
            start with ``f"source:{require_source_kind}:"``.  Ensures human-attested claims only
            when set to ``"human"``.  ``None`` = no source-kind requirement.
        min_confidence_lcb: Minimum confidence LCB floor.  LCB = posterior_mean - sqrt(variance)
            (z approx 1.0).  Zero (default) = no floor.
        require_min_rank: Minimum ``EPISTEMIC_RANK`` floor.  Zero (default) = any level.
    """

    target: str
    predicate_norm: str | None = None
    subject_norm: str | None = None
    min_distinct_sources: int = 1
    require_source_kind: SourceKind | None = "human"
    min_confidence_lcb: float = 0.0
    require_min_rank: int = 0


@runtime_checkable
class _AssertFactSink(Protocol):
    """Structural protocol for promotion sinks — the only surface the promoter needs."""

    async def assert_fact(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        object_is_entity: bool,
        epistemic_type: EpistemicType,
        source: object,
        evidence_type: str,
        polarity: str,
        run_id: str | None,
        stage: str | None,
        created_by: str,
    ) -> str: ...


# Import EPISTEMIC_RANK lazily to avoid a circular import at module load (coherence.upgrade
# imports from this package's namespace too).  Defer to first use via _rank_of helper.
def _rank_of(epistemic_type: str) -> int:
    from cogworx.coherence.upgrade import EPISTEMIC_RANK

    return EPISTEMIC_RANK.get(epistemic_type, 0)


def _confidence_lcb(scored: ScoredClaim) -> float:
    """Lower confidence bound: posterior_mean - sqrt(variance) (z approx 1.0)."""
    conf = scored.confidence
    return conf.confidence - math.sqrt(conf.variance)


def _normalize(s: str) -> str:
    """Strip and casefold for predicate/subject comparison."""
    return s.strip().lower()


class ScopePromoter:
    """Routes active agent-scope claims to world/user scopes via structural rules.

    NEVER reads claim payload text content for routing decisions (CANON S9).  All routing is
    purely structural: field values, evidence counts, numeric thresholds, and framework-assigned
    source-id prefixes.

    After promoting a claim, ``copy_evidence`` is called on the store to carry the provenance
    chain forward onto the newly minted target claim.
    """

    def __init__(
        self,
        rules: Sequence[PromotionRule],
        sinks: Mapping[str, _AssertFactSink],
        store: CoherenceStore,
        source_kg: EntityKG,
    ) -> None:
        """
        Args:
            rules: Ordered list of developer-authored promotion rules.
            sinks: Map of target scope name → sink for that scope.  In production these are
                :class:`~cogworx.knowledge.scoped_kg.ScopedKG` instances; in tests any object
                that satisfies the :class:`_AssertFactSink` structural protocol is accepted.
            store: Coherence store for ``copy_evidence`` after promotion.
            source_kg: The entity KG that owns the source claims (used to read evidence events).
        """
        self._rules = tuple(rules)
        self._sinks = dict(sinks)
        self._store = store
        self._source_kg = source_kg

    async def promote_for_subject(
        self,
        claims: Sequence[ScoredClaim],
    ) -> int:
        """Route active claims to target scopes that match any registered rule.

        Routing is PURELY STRUCTURAL — the claim's payload text is NEVER read or used as a
        routing signal (CANON S9: no model self-report, no text-as-control).  Only field values
        (predicate, subject), evidence counts, numeric thresholds, and framework-assigned
        source-id kind prefixes drive routing decisions.

        For each active claim, for each rule:
        - All matching conditions must pass.
        - On match: the target ``sink.assert_fact(...)`` mints the promoted claim, then
          ``store.copy_evidence(orig_id, new_id, id_prefix=f"promo:{rule.target}")`` carries the
          evidence provenance chain forward.

        Note: ``assert_fact`` and ``copy_evidence`` are not atomic.  A crash between them leaves
        a promoted claim with no evidence (Beta(1,1) prior confidence).  This is self-healing:
        the source subject's dirty mark survives (``commit_reconciliation`` did not run), so the
        next tick re-runs ``promote_for_subject``: ``assert_fact`` is idempotent (same claim id),
        ``copy_evidence`` fills in the missing evidence.  The orphan window is bounded by one tick
        interval.  This mirrors the ClaimExtractor D6 at-least-once-cost pattern.

        Returns:
            Count of promotions performed (one per matching (claim, rule) pair).
        """
        from cogworx.knowledge.source_registry import SourceDeclaration

        promotions = 0

        for scored in claims:
            claim = scored.claim

            # Skip inactive claims — only promote active ones.
            if getattr(claim, "status", "active") != "active":
                continue

            # Fetch evidence events for structural analysis (source counts, kind check).
            evidence_events = await self._source_kg.evidence_for(claim.id)

            # Precompute values used by rules to avoid redundant work per-rule.
            claim_rank = _rank_of(claim.epistemic_type)
            lcb = _confidence_lcb(scored)
            pred_norm = _normalize(claim.predicate or "") if claim.predicate else ""
            subj_norm = _normalize(claim.subject)

            # Distinct positive-polarity source_ids for min_distinct_sources check.
            distinct_sources: set[str] = {
                ev.source_id for ev in evidence_events if ev.polarity == "+"
            }

            for rule in self._rules:
                if rule.target not in self._sinks:
                    continue

                # predicate_norm filter (S9-safe: structural field, not text content).
                if rule.predicate_norm is not None and pred_norm != _normalize(rule.predicate_norm):
                    continue

                # subject_norm filter (S9-safe: structural field).
                if rule.subject_norm is not None and subj_norm != _normalize(rule.subject_norm):
                    continue

                # Epistemic rank floor.
                if claim_rank < rule.require_min_rank:
                    continue

                # Confidence LCB floor.
                if lcb < rule.min_confidence_lcb:
                    continue

                # Distinct source count.
                if len(distinct_sources) < rule.min_distinct_sources:
                    continue

                # Source-kind gate: at least one evidence source_id starts with the kind prefix.
                # source_ids are framework-assigned (S9-safe prefix check).
                if rule.require_source_kind is not None:
                    prefix = f"source:{rule.require_source_kind}:"
                    if not any(ev.source_id.startswith(prefix) for ev in evidence_events):
                        continue

                # All conditions passed — promote to the target scope.
                sink = self._sinks[rule.target]
                promotion_source = SourceDeclaration(
                    kind="system",
                    ref=f"scope-promoter:{rule.target}",
                )
                new_claim_id = await sink.assert_fact(
                    subject=claim.subject,
                    predicate=claim.predicate or "",
                    obj=claim.payload,
                    object_is_entity=claim.object_entity is not None,
                    epistemic_type=claim.epistemic_type,
                    source=promotion_source,
                    evidence_type="tool_proof",
                    polarity="+",
                    run_id=None,
                    stage=None,
                    created_by=f"scope-promoter:{rule.target}",
                )
                await self._store.copy_evidence(
                    claim.id,
                    new_claim_id,
                    id_prefix=f"promo:{rule.target}",
                )
                promotions += 1

        return promotions
