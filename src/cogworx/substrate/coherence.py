"""Coherence-reconciler seam — the thin internal Protocol for tests/mocks (CANON S3).

This module defines the typed shapes that the Pod 2.7 coherence reconciler writes and reads.
The real implementation is :class:`cogworx.adapters.neo4j_entity_kg.Neo4jEntityKG` (which also
implements :class:`CoherenceStore`). The in-memory double is
:class:`cogworx.testing.doubles.InMemoryEntityKG`.

Design invariants:
  - :class:`AdjudicationRecord` is APPEND-ONLY — it is written ON CREATE (MERGE-with-epoch-guard)
    and never updated. It doubles as a no-good cache: re-running the same claim-set produces the
    same ``id`` and the MERGE is a no-op.
  - :class:`Defeat` records a SUPERSEDES edge. ``set_valid_to`` carries the loser's expiry in
    update-supersession mode; ``None`` in revision-defeat mode (the loser is defeated but not
    time-bounded — the predicate itself was wrong, not just stale).
  - ``DirtySubject`` nodes are MERGE-idempotent; ``epoch`` monotonically increments on every
    re-mark. ``commit_reconciliation`` deletes the dirty node ONLY when ``epoch`` matches the
    value observed at reconcile-start (epoch guard), so a write that arrived mid-flight is never
    silently swallowed.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import EpistemicType, Provenance
from cogworx.knowledge.evidence import EvidenceEvent

__all__ = [
    "AdjudicationRecord",
    "AdjudicationVerdict",
    "CoherenceStore",
    "Defeat",
    "DirtyKey",
    "DirtySubject",
    "ReconciliationOutcome",
    "ResolutionKind",
]

# Composite dirty-mark key: f"{scope}\x1f{subject_norm}"
# The INFORMATION SEPARATOR ONE (\x1f) is chosen because it cannot appear in scope or subject_norm
# values (both are user-supplied strings that are NFC-normalised and stripped).
DirtyKey = str

AdjudicationVerdict = Literal["consistent", "inconsistent"]
ResolutionKind = Literal["none", "update-supersession", "revision-defeat", "escalated"]


class DirtySubject(BaseModel):
    """A subject that has unreconciled claims and is queued for the coherence reconciler.

    ``epoch`` is a monotonic re-mark counter — it increments on every write that touches the
    subject.  The reconciler observes the epoch at claim-time and deletes the dirty node ONLY when
    ``epoch`` still matches (epoch guard in ``commit_reconciliation``), so a concurrent write that
    bumps the epoch prevents a stale reconciliation from silently clearing the dirty mark.
    """

    model_config = ConfigDict(frozen=True)

    key: DirtyKey
    scope: str
    subject_norm: str
    epoch: int
    marked_at: datetime
    attempts: int


class AdjudicationRecord(BaseModel):
    """Append-only event written by the reconciler for each batch of claims it adjudicated.

    ``id`` is deterministic: ``"adj:" + sha256(sorted claim_ids joined by \\x1f)[:32]`` — the same
    claim-set produces the same id regardless of order, so MERGE is idempotent and serves as a
    no-good cache (re-running the same disagreement is a no-op after the first write).

    ``oracle_calls`` counts LLM invocations during adjudication (S1 tracking — kept for eval;
    never used as a control signal).
    ``escalation_reason`` is non-None only when ``resolution == "escalated"``.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    scope: str
    subject_norm: str
    claim_ids: tuple[str, ...]
    verdict: AdjudicationVerdict
    resolution: ResolutionKind
    winner_id: str | None
    loser_ids: tuple[str, ...]
    margin: float | None
    oracle_calls: int
    escalation_reason: str | None
    provenance: Provenance
    recorded_at: datetime


class Defeat(BaseModel):
    """A single SUPERSEDES defeat record — one winner, one loser.

    ``kind`` distinguishes the two resolution modes:
      ``"update"``  — the winner is a more-recent fact; the loser is time-bounded
                      (``set_valid_to = winner.valid_from``).
      ``"revision"`` — the loser's predicate was wrong; it is defeated but not time-bounded
                       (``set_valid_to = None``).
    ``adjudication_id`` back-links to the :class:`AdjudicationRecord` that produced this defeat.
    """

    model_config = ConfigDict(frozen=True)

    winner_id: str
    loser_id: str
    kind: Literal["update", "revision"]
    adjudication_id: str
    set_valid_to: datetime | None


class ReconciliationOutcome(BaseModel):
    """The full outcome of one reconciliation pass over a dirty subject.

    Produced by the reconciler logic (Unit 2+) and persisted atomically by
    ``CoherenceStore.commit_reconciliation``.
    """

    model_config = ConfigDict(frozen=True)

    adjudications: tuple[AdjudicationRecord, ...]
    defeats: tuple[Defeat, ...]
    contradictions: tuple[tuple[str, str], ...]
    recorded_at: datetime


class CoherenceStore(Protocol):
    """Additive coherence-reconciler surface on the Neo4j entity-KG seam (CANON S3).

    These methods are implemented alongside the EntityKG contract on
    :class:`~cogworx.adapters.neo4j_entity_kg.Neo4jEntityKG` and
    :class:`~cogworx.testing.doubles.InMemoryEntityKG`.  They are a SEPARATE Protocol so callers
    that only need the coherence surface can narrow their type annotation without depending on the
    full EntityKG Protocol.

    ``commit_reconciliation`` is the only write that mutates claim status and edges. All other
    writes (``write_claim``, ``add_evidence``, ``invalidate_claim``, ``project_claims``) produce
    dirty marks only; the reconciler processes them asynchronously (S1, off-write-path).
    """

    async def claim_dirty_subjects(self, *, limit: int = 16) -> Sequence[DirtySubject]:
        """Return up to ``limit`` dirty subjects, ordered by ``marked_at`` ASC (oldest-first).

        The reconciler processes in FIFO order so recently-written subjects do not starve
        subjects that have been dirty for a long time.
        """
        ...

    async def bump_dirty_attempts(self, key: DirtyKey) -> int:
        """Atomically increment the attempt counter for the given dirty key; return the new count.

        Called before each reconciliation attempt so that a reconciler that crashes mid-pass
        leaves an accurate attempt count. The epoch is NOT bumped (attempts are bookkeeping only).
        """
        ...

    async def adjudication_ids_for_subject(self, scope: str, subject_norm: str) -> Sequence[str]:
        """Return the ids of all Adjudication nodes recorded for this (scope, subject_norm).

        Used by the reconciler to check the no-good cache before invoking the oracle.
        """
        ...

    async def commit_reconciliation(
        self,
        outcome: ReconciliationOutcome,
        *,
        dirty_key: DirtyKey,
        observed_epoch: int,
    ) -> None:
        """Atomically persist one reconciliation outcome and clear the dirty mark.

        ONE transaction:
          1. MERGE each :class:`AdjudicationRecord` (ON CREATE only — immutable-on-match) with
             ``[:ADJUDICATES]`` edges to each claim in ``claim_ids``.
          2. MERGE ``[:CONTRADICTS]`` edges for each pair in ``outcome.contradictions``
             (idempotent, semantically undirected).
          3. For each :class:`Defeat`:
             - MERGE ``[:SUPERSEDES]`` edge (ON CREATE — first-wins; sets kind/adjudication_id/
               recorded_at).
             - SET ``loser.status = 'defeasibly-defeated'``.
             - SET ``loser.defeated_by = winner_id`` (coalesce — first-defeat-wins, never cleared).
             - If ``set_valid_to`` is not ``None``: SET ``loser.valid_to`` only if currently NULL
               (first-invalidation-wins, mirrors ``invalidate_claim``).
          4. DELETE the :class:`DirtySubject` WHERE ``key = dirty_key AND epoch = observed_epoch``
             (epoch guard: a write that arrived mid-flight bumps the epoch and prevents premature
             clearance — the subject stays dirty for the next reconciler tick).

        A reconciliation outcome that cannot be applied atomically must be retried from scratch;
        the epoch guard ensures idempotent retry is safe.
        """
        ...

    async def current_epistemic_level(self, claim_id: str) -> EpistemicType | None:
        """Return the claim's current ``epistemic_type``, or ``None`` if the claim is not found.

        Used by :func:`cogworx.coherence.upgrade.epistemic_upgrade` as a pre-call rank guard so
        the validation layer can raise ``ValueError`` on explicit-downgrade attempts and return
        ``applied=False`` on same-or-lower targets WITHOUT calling ``apply_epistemic_upgrade``
        (the store must not be called on a no-op path).
        """
        ...

    async def apply_epistemic_upgrade(
        self,
        claim_id: str,
        *,
        new_level: EpistemicType,
        evidence: EvidenceEvent,
        actor: str,
        recorded_at: datetime,
    ) -> bool:
        """Promote a claim's epistemic level with full provenance (Pod 2.7 upgrade surface).

        ONE transaction:
          - Rank-guard IN-TXN: read the current ``epistemic_type``; abort (return ``False``) if
            the stored rank is already >= ``new_level`` (rank order: observation < inference <
            confirmed).  The guard is inside the transaction so a concurrent upgrade loses the
            race rather than writing a stale level.
          - MERGE a ``[:HAS_EVIDENCE]`` edge for ``evidence`` (the justification evidence).
          - SET ``claim.epistemic_type = new_level``.
          - MERGE an ``(:EpistemicUpgrade {uid})`` audit node with ``[:UPGRADES]->(:Claim)`` and
            ``[:JUSTIFIED_BY]->(:Evidence)`` edges. ``uid = f"upgrade:{claim_id}:{new_level}"``.
          - Dirty-mark ``(scope, subject_norm)`` in the same transaction.

        Returns ``True`` iff the level changed; ``False`` iff it was already at or above
        ``new_level`` (idempotent — no audit node written on a no-op).
        """
        ...

    async def copy_evidence(self, from_claim_id: str, to_claim_id: str, *, id_prefix: str) -> int:
        """Copy all evidence events from one claim to another, prefixing each copied id.

        Each evidence event ``e`` on ``from_claim_id`` is MERGED onto ``to_claim_id`` with
        ``id = f"{id_prefix}:{e.id}"``.  MERGE ensures idempotency so calling this twice with the
        same ``id_prefix`` produces the same result (no duplicate evidence events).

        Returns the count of evidence events now present on ``to_claim_id`` that were sourced from
        this copy (i.e. the number of events matched from ``from_claim_id``).
        """
        ...
