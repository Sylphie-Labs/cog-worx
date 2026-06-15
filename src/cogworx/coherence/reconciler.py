"""Coherence reconciler — async/batched contradiction detection and resolution (Pod 2.7, U6).

The :class:`CoherenceReconciler` is a sweeper-shaped async object.  Call :meth:`tick` repeatedly
to drain the dirty-subject queue.  Model calls are off the write path (CANON S1); the reconciler
runs in the background, never on-commit.

Single-instance discipline (like all sweepers in cog-worx): cross-process exclusion is a
documented ops-pod carry-forward (CF-F).
"""

from __future__ import annotations

import datetime
import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass

from cogworx.claims.provenance import Provenance
from cogworx.coherence.config import CoherenceConfig
from cogworx.coherence.entrenchment import decide_resolution, entrenchment_of
from cogworx.coherence.mus import BudgetExceeded, CallBudget, find_mus
from cogworx.coherence.oracle import ConsistencyOracle
from cogworx.coherence.pairs import candidate_pairs
from cogworx.coherence.promotion import ScopePromoter
from cogworx.substrate.coherence import (
    AdjudicationRecord,
    CoherenceStore,
    Defeat,
    ReconciliationOutcome,
)
from cogworx.substrate.entity_kg import EntityKG, ScoredClaim

__all__ = ["CoherenceReconciler", "ReconcilerStats"]

_log = logging.getLogger(__name__)


@dataclass
class ReconcilerStats:
    """Per-tick statistics for the coherence reconciler sweep."""

    subjects_processed: int = 0
    subjects_cleared: int = 0
    """Subjects with no candidate conflict pairs — dirty mark cleared, nothing to resolve."""
    subjects_defeated: int = 0
    """Subjects where at least one defeat was committed."""
    subjects_escalated: int = 0
    """Subjects where at least one adjudication was escalated."""
    subjects_skipped: int = 0
    """Subjects that hit the poison guard (max_attempts) or subject-too-large."""
    subjects_failed: int = 0
    """Subjects where an unhandled exception occurred — dirty mark preserved for retry."""
    oracle_calls: int = 0
    """Total oracle calls across all subjects this tick."""
    promotions: int = 0
    """Total scope promotions performed this tick."""


def _adj_id_for_claims(claim_ids: tuple[str, ...]) -> str:
    """Deterministic adjudication id: ``"adj:" + sha256(sorted ids joined by \\x1f)[:32]``."""
    key = "\x1f".join(sorted(claim_ids))
    return "adj:" + hashlib.sha256(key.encode()).hexdigest()[:32]


def _system_provenance(now: datetime.datetime) -> Provenance:
    """Provenance for purely structural adjudication outcomes (no oracle call)."""
    return Provenance(source="system", confidence=1.0, recorded_at=now)


def _inference_provenance(now: datetime.datetime) -> Provenance:
    """Provenance for oracle-backed adjudication outcomes (S5 — model-derived)."""
    return Provenance(source="inference", confidence=1.0, recorded_at=now)


class CoherenceReconciler:
    """Sweeper-shaped async reconciler that drains the dirty-subject queue.

    Call :meth:`tick` repeatedly (e.g. from a background sweeper loop) to process up to
    ``config.batch_limit`` dirty subjects per call.  Each subject is processed independently;
    a per-subject exception leaves the dirty mark in place for the next tick.

    Model calls are off the write path (CANON S1).  Single-instance discipline (like all
    sweepers in cog-worx): cross-process exclusion is a documented ops-pod carry-forward (CF-F).
    """

    def __init__(
        self,
        *,
        entity_kg: EntityKG,
        store: CoherenceStore,
        oracle: ConsistencyOracle,
        config: CoherenceConfig | None = None,
        promoter: ScopePromoter | None = None,
        now: Callable[[], datetime.datetime] = lambda: datetime.datetime.now(datetime.UTC),
    ) -> None:
        self._entity_kg = entity_kg
        self._store = store
        self._oracle = oracle
        self._config = config if config is not None else CoherenceConfig()
        self._promoter = promoter
        self._now = now

    async def tick(self) -> ReconcilerStats:
        """Process up to ``config.batch_limit`` dirty subjects.

        Returns a :class:`ReconcilerStats` with per-tick counters.  Never raises — per-subject
        exceptions are caught, logged, and counted in ``stats.subjects_failed``.
        """
        stats = ReconcilerStats()
        dirty_subjects = await self._store.claim_dirty_subjects(limit=self._config.batch_limit)

        for subject in dirty_subjects:
            stats.subjects_processed += 1
            try:
                await self._process_subject(subject, stats)
            except Exception:
                _log.exception(
                    "CoherenceReconciler: per-subject failure for key=%r scope=%r subject_norm=%r",
                    subject.key,
                    subject.scope,
                    subject.subject_norm,
                )
                stats.subjects_failed += 1

        return stats

    async def _process_subject(
        self,
        subject: object,  # DirtySubject — imported via store contract
        stats: ReconcilerStats,
    ) -> None:
        from cogworx.substrate.coherence import DirtySubject

        assert isinstance(subject, DirtySubject)
        now = self._now()

        # -- Poison guard: too many attempts → escalate and clear ----------------------------
        if subject.attempts >= self._config.max_attempts:
            _log.debug(
                "Poison guard: subject key=%r attempts=%d >= max_attempts=%d",
                subject.key,
                subject.attempts,
                self._config.max_attempts,
            )
            adj_id = _adj_id_for_claims(())
            # Use a stable id based on the subject key so the MERGE is idempotent per subject.
            adj_id = "adj:" + hashlib.sha256(f"poison\x1f{subject.key}".encode()).hexdigest()[:32]
            adj = AdjudicationRecord(
                id=adj_id,
                scope=subject.scope,
                subject_norm=subject.subject_norm,
                claim_ids=(),
                verdict="consistent",
                resolution="escalated",
                winner_id=None,
                loser_ids=(),
                margin=None,
                oracle_calls=0,
                escalation_reason="max-attempts-exceeded",
                provenance=_system_provenance(now),
                recorded_at=now,
            )
            outcome = ReconciliationOutcome(
                adjudications=(adj,),
                defeats=(),
                contradictions=(),
                recorded_at=now,
            )
            await self._store.commit_reconciliation(
                outcome, dirty_key=subject.key, observed_epoch=subject.epoch
            )
            stats.subjects_skipped += 1
            return

        # -- Bump attempts BEFORE doing any work so a crash leaves an accurate count ----------
        await self._store.bump_dirty_attempts(subject.key)

        # -- Fetch claims for this subject ---------------------------------------------------
        fetch_limit = self._config.max_claims_per_subject + 1
        scored_claims = await self._entity_kg.claims_about(
            subject.subject_norm,
            scope=subject.scope,
            limit=fetch_limit,
        )

        if len(scored_claims) > self._config.max_claims_per_subject:
            _log.debug(
                "Subject too large: key=%r claims=%d > max=%d",
                subject.key,
                len(scored_claims),
                self._config.max_claims_per_subject,
            )
            adj_id = "adj:" + hashlib.sha256(f"toolarge\x1f{subject.key}".encode()).hexdigest()[:32]
            adj = AdjudicationRecord(
                id=adj_id,
                scope=subject.scope,
                subject_norm=subject.subject_norm,
                claim_ids=(),
                verdict="consistent",
                resolution="escalated",
                winner_id=None,
                loser_ids=(),
                margin=None,
                oracle_calls=0,
                escalation_reason="subject-too-large",
                provenance=_system_provenance(now),
                recorded_at=now,
            )
            outcome = ReconciliationOutcome(
                adjudications=(adj,),
                defeats=(),
                contradictions=(),
                recorded_at=now,
            )
            await self._store.commit_reconciliation(
                outcome, dirty_key=subject.key, observed_epoch=subject.epoch
            )
            stats.subjects_skipped += 1
            return

        observed_epoch = subject.epoch

        # -- No-good cache: adjudication ids already committed for this subject ---------------
        cached_ids = set(
            await self._store.adjudication_ids_for_subject(subject.scope, subject.subject_norm)
        )

        # candidate_pairs expects plain Claim objects
        claims = [sc.claim for sc in scored_claims]
        report = candidate_pairs(
            claims,
            cosine_threshold=self._config.cosine_threshold,
            skip_adjudication_ids=cached_ids,
        )

        if not report.pairs:
            # Nothing left to adjudicate — run promoter then clear dirty mark.
            if self._promoter is not None:
                active_claims = [sc for sc in scored_claims if sc.claim.status == "active"]
                promotions = await self._promoter.promote_for_subject(active_claims)
                stats.promotions += promotions
            outcome = ReconciliationOutcome(
                adjudications=(),
                defeats=(),
                contradictions=(),
                recorded_at=now,
            )
            await self._store.commit_reconciliation(
                outcome, dirty_key=subject.key, observed_epoch=observed_epoch
            )
            stats.subjects_cleared += 1
            return

        # -- MUS loop -------------------------------------------------------------------------
        # Build a working set of claims by id from the candidate set.
        candidate_id_set = set(report.candidate_claim_ids)
        working_set: list[ScoredClaim] = [
            sc for sc in scored_claims if sc.claim.id in candidate_id_set
        ]

        adjudications: list[AdjudicationRecord] = []
        defeats: list[Defeat] = []
        contradictions: list[tuple[str, str]] = []
        oracle_calls_this_subject = 0
        had_defeat = False
        had_escalation = False

        for _iteration in range(self._config.max_mus_per_subject):
            if not working_set:
                break

            remaining_budget = self._config.max_oracle_calls_per_subject - oracle_calls_this_subject
            budget = CallBudget(remaining_budget)

            # Sort ascending by entrenchment (lowest entrenchment first — QuickXplain bias).
            def _ent_key(sc: ScoredClaim) -> tuple[int, float, str]:
                ent = entrenchment_of(sc.claim, sc.confidence)
                return (ent.rank, ent.lcb, ent.claim_id)

            working_set.sort(key=_ent_key)
            working_claims = [sc.claim for sc in working_set]

            try:
                mus_result = await find_mus(working_claims, self._oracle, budget=budget)
            except BudgetExceeded:
                # Budget hit during MUS search — escalate this subject.
                escalation_adj_id = _adj_id_for_claims(tuple(sorted(c.id for c in working_claims)))
                adj = AdjudicationRecord(
                    id=escalation_adj_id,
                    scope=subject.scope,
                    subject_norm=subject.subject_norm,
                    claim_ids=tuple(sorted(c.id for c in working_claims)),
                    verdict="inconsistent",
                    resolution="escalated",
                    winner_id=None,
                    loser_ids=(),
                    margin=None,
                    oracle_calls=budget.used,
                    escalation_reason="budget-exhausted",
                    provenance=_system_provenance(now),
                    recorded_at=now,
                )
                adjudications.append(adj)
                oracle_calls_this_subject += budget.used
                had_escalation = True
                break

            oracle_calls_this_subject += mus_result.oracle_calls

            if not mus_result.mus:
                # Working set is consistent — no more conflicts in this working set.
                break

            mus_claim_ids = set(mus_result.mus)
            mus_claims = [c for c in working_claims if c.id in mus_claim_ids]

            # Build adjudication id from the MUS claim ids.
            adj_id = _adj_id_for_claims(tuple(mus_result.mus))

            if not mus_result.verified:
                # MUS could not be verified inconsistent — escalate.
                adj = AdjudicationRecord(
                    id=adj_id,
                    scope=subject.scope,
                    subject_norm=subject.subject_norm,
                    claim_ids=mus_result.mus,
                    verdict="inconsistent",
                    resolution="escalated",
                    winner_id=None,
                    loser_ids=(),
                    margin=None,
                    oracle_calls=mus_result.oracle_calls,
                    escalation_reason="unverified-mus",
                    provenance=_inference_provenance(now),
                    recorded_at=now,
                )
                adjudications.append(adj)
                had_escalation = True
                break

            # Compute entrenchment for MUS claims using their scored confidence.
            mus_scored = {sc.claim.id: sc for sc in working_set if sc.claim.id in mus_claim_ids}
            entrenchment_map = {
                cid: entrenchment_of(sc.claim, sc.confidence) for cid, sc in mus_scored.items()
            }

            resolution = decide_resolution(
                mus_claims,
                entrenchment_map,
                margin=self._config.margin,
            )

            # Build AdjudicationRecord from the resolution outcome.
            verdict = (
                "inconsistent"
                if resolution.kind in ("revision-defeat", "update-supersession", "escalated")
                else "consistent"
            )

            adj = AdjudicationRecord(
                id=adj_id,
                scope=subject.scope,
                subject_norm=subject.subject_norm,
                claim_ids=mus_result.mus,
                verdict=verdict,
                resolution=resolution.kind,
                winner_id=resolution.winner_id,
                loser_ids=resolution.loser_ids,
                margin=resolution.margin,
                oracle_calls=mus_result.oracle_calls,
                escalation_reason=resolution.escalation_reason,
                provenance=_inference_provenance(now),
                recorded_at=now,
            )
            adjudications.append(adj)

            if resolution.kind == "escalated":
                # Record contradiction pairs for each pair in the MUS, then stop looping.
                for i in range(len(mus_claims)):
                    for j in range(i + 1, len(mus_claims)):
                        contradictions.append((mus_claims[i].id, mus_claims[j].id))
                had_escalation = True
                break

            # Defeat: build Defeat records and remove losers from working set.
            if resolution.kind in ("revision-defeat", "update-supersession"):
                assert resolution.winner_id is not None
                defeat_kind = "update" if resolution.kind == "update-supersession" else "revision"
                for loser_id in resolution.loser_ids:
                    defeat = Defeat(
                        winner_id=resolution.winner_id,
                        loser_id=loser_id,
                        kind=defeat_kind,
                        adjudication_id=adj_id,
                        set_valid_to=resolution.set_valid_to,
                    )
                    defeats.append(defeat)
                    # Record contradiction pair for the defeat.
                    contradictions.append((resolution.winner_id, loser_id))
                had_defeat = True

                # Remove defeated ids from working set for next iteration.
                loser_set = set(resolution.loser_ids)
                working_set = [sc for sc in working_set if sc.claim.id not in loser_set]

        stats.oracle_calls += oracle_calls_this_subject

        if had_defeat:
            stats.subjects_defeated += 1
        if had_escalation:
            stats.subjects_escalated += 1
        if not had_defeat and not had_escalation:
            stats.subjects_cleared += 1

        # -- Run scope promoter (if configured) ----------------------------------------------
        if self._promoter is not None:
            active_claims = [sc for sc in scored_claims if sc.claim.status == "active"]
            promotions = await self._promoter.promote_for_subject(active_claims)
            stats.promotions += promotions

        # -- Commit everything atomically ----------------------------------------------------
        outcome = ReconciliationOutcome(
            adjudications=tuple(adjudications),
            defeats=tuple(defeats),
            contradictions=tuple(contradictions),
            recorded_at=now,
        )
        await self._store.commit_reconciliation(
            outcome, dirty_key=subject.key, observed_epoch=observed_epoch
        )
