"""Entity-KG seam — the thin internal Protocol for tests/mocks (CANON S3).

This is the engine-shaped seam for the entity knowledge graph on Neo4j. It is the test/mock surface
only — not a backend-portability layer (S3 violation). The real implementation is
:class:`cogworx.adapters.neo4j_entity_kg.Neo4jEntityKG`.

Graph semantics (enforced by BOTH the real adapter AND :class:`InMemoryEntityKG`):
  - Every (:Claim) is owned by exactly one (:Entity) via [:HAS_CLAIM] (subject side).
  - When a claim's object is itself an entity, a [:REFERS_TO] edge connects (:Claim)->(:Entity).
  - [:HAS_EVIDENCE] edges carry immutable :Evidence events; they ACCUMULATE (CREATE, never MERGE).
  - [:DERIVED_FROM] edges record inference lineage (same convention as the base GraphStore seam).
  - [:CONTRADICTS] is MERGE-idempotent and semantically undirected; stored as one directed edge.
  - Claims are invalidated by setting valid_to (bi-temporal, never deleted).
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Claim
from cogworx.knowledge.confidence import ClaimConfidence
from cogworx.knowledge.evidence import EvidenceEvent

__all__ = [
    "EntityKG",
    "ScoredClaim",
]


class ScoredClaim(BaseModel):
    """A claim with derived confidence + weakest-link lineage score, optionally with similarity."""

    model_config = ConfigDict(frozen=True)

    claim: Claim
    confidence: ClaimConfidence
    lineage_min_confidence: float
    """min(own confidence, each DERIVED_FROM ancestor's confidence) — the weakest-link rule.
    Equals own confidence when the claim has no ancestors."""
    similarity: float | None = None
    """Raw cosine similarity in [-1, 1] when the claim was retrieved via the vector channel."""


@runtime_checkable
class EntityKG(Protocol):
    """The Neo4j entity-KG seam (S3 — thin internal test/mock surface, NOT a portability layer).

    CONTRACT SEMANTICS (enforced by both real adapter and InMemoryEntityKG):

    upsert_claim
      DISABLED. The entity KG's only write surface is write_claim. Implementations MUST raise
      ``NotImplementedError`` — the Phase-0 base upsert bypasses identity discipline, immutable-on-
      match, and bi-temporal honesty. Any code that calls upsert_claim on an entity-KG instance is
      a bug.

    write_claim
      Raises ``ValueError`` (before any I/O) if ``claim.id`` does not match
      ``claim_id_for(subject, predicate or "", object_entity if it is not None else payload)``
      (``is not None``, not truthiness — a degenerate empty-string object_entity is still the
      object). Identity discipline is structural. Always records the evidence event, including
      when the claim node already existed (the accumulation path).

    Epistemic level is FIRST-WRITE-WINS (S5 boundary)
      ``epistemic_type`` is not part of claim identity, so a re-derivation of the same triple at a
      DIFFERENT level (e.g. an ``observation`` arriving after an ``inference``) accumulates its
      evidence under the FIRST writer's level — the stored level never silently changes, which is
      exactly S5's "never silently merged": the first level sticks until an EXPLICIT epistemic
      upgrade surface (the Pod 2.7 coherence reconciler) promotes it with provenance. Callers that
      need the distinction NOW must mint a distinct predicate. Evidence types (``tool_proof`` vs
      ``recall``) carry the structural truth-weight in the meantime.

    add_evidence / invalidate_claim
      Raise ``ValueError`` on an unknown ``claim_id``.

    invalidate_claim
      Sets ``valid_to`` ONLY if currently ``None`` (first invalidation wins; later calls are
      no-ops, not errors). Never deletes.

    claims_about
      Returns claims where ``entity`` is subject ([:HAS_CLAIM]) OR object ([:REFERS_TO]).
      ``as_of`` filter: valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of).
      Confidence derived at read (never stored). Ordered newest-first by ingest_time, capped by
      ``limit``.
      Skeleton nodes (claim nodes with payload IS NULL, created as DERIVED_FROM placeholders)
      are excluded from all results.

    claims_by_similarity
      Uses the native vector index. Callers supply and receive RAW cosine scores (range [-1, 1]).
      The adapter translates in/out of Neo4j's (1+cos)/2 storage format internally.

    resolution_candidates
      Exact-match on (subject_norm, predicate_norm) ordered newest-first, up to ``k``.
      If that yields fewer than ``k`` and an ``embedding`` is given, fills from the vector channel
      (min raw cosine 0.70), deduped by id. Returns plain Claims (no confidence; the judge pod
      decides). Does NOT raise on unknown subject.

    Datetime contract
      Naive datetimes are interpreted as UTC (not rejected). Tz-aware datetimes are converted to
      UTC. Round-trips are instant-preserving but offset-normalized to +00:00. All stored ISO
      strings share the +00:00 offset so lexicographic order == temporal order.

    source_id contract (S9)
      ``EvidenceEvent.source_id`` MUST be assigned by framework code (run id, tool id, document
      URL, user id) and NEVER taken from model output. A model-chosen source_id is an S9 violation
      — self-reported identity gating confidence. Structural enforcement is deferred to the
      ops/coherence pod; this contract is the interim discipline.
    """

    async def write_claim(self, claim: Claim, *, evidence: EvidenceEvent) -> str:
        """MERGE the claim node and CREATE one evidence event. Returns the canonical claim id.

        Raises ``ValueError`` if ``claim.id`` fails identity-discipline check.
        """
        ...

    async def add_evidence(self, claim_id: str, event: EvidenceEvent) -> None:
        """Append a new evidence event to an existing claim.

        Raises ``ValueError`` if ``claim_id`` is not known.
        """
        ...

    async def get_claim(self, claim_id: str) -> Claim | None:
        """Return the claim by id, or ``None``."""
        ...

    async def evidence_for(self, claim_id: str) -> Sequence[EvidenceEvent]:
        """Return all evidence events for the given claim, in creation order."""
        ...

    async def claims_about(
        self,
        entity: str,
        *,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> Sequence[ScoredClaim]:
        """Return scored claims where ``entity`` is subject or object, newest-first."""
        ...

    async def claims_by_similarity(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
    ) -> Sequence[ScoredClaim]:
        """Return up to ``k`` scored claims nearest to ``embedding`` (raw cosine >= min_score)."""
        ...

    async def resolution_candidates(
        self,
        subject: str,
        predicate: str,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
    ) -> Sequence[Claim]:
        """Return candidate claims for judge-based resolution (no confidence scoring)."""
        ...

    async def write_contradiction(self, claim_id_a: str, claim_id_b: str) -> None:
        """Record a CONTRADICTS edge between two claims (idempotent MERGE)."""
        ...

    async def contradictions_of(self, claim_id: str) -> Sequence[Claim]:
        """Return all claims that contradict the given claim."""
        ...

    async def invalidate_claim(self, claim_id: str, *, valid_to: datetime) -> None:
        """Set valid_to on the claim (first-invalidation-wins; later calls are no-ops).

        Raises ``ValueError`` if ``claim_id`` is not known.
        """
        ...
