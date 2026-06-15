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
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Claim
from cogworx.knowledge.confidence import ClaimConfidence
from cogworx.knowledge.evidence import EvidenceEvent
from cogworx.substrate.journal import ProjectionCursor

__all__ = [
    "ClaimProjection",
    "EntityKG",
    "ScoredClaim",
]


@dataclass(frozen=True)
class ClaimProjection:
    """One (claim, evidence) pair for atomic batch projection into the entity KG."""

    claim: Claim
    evidence: EvidenceEvent


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
    text_score: float | None = None
    """BM25 score when the claim was retrieved via the full-text channel."""


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
      exactly S5's "never silently merged": the first level sticks until an EXPLICIT upgrade via
      ``CoherenceStore.apply_epistemic_upgrade`` (Pod 2.7), which records provenance and marks the
      subject dirty. Callers that need the distinction NOW must mint a distinct predicate. Evidence
      types (``tool_proof`` vs ``recall``) carry the structural truth-weight in the meantime.

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
        Also marks ``(scope, subject_norm)`` dirty in the same transaction (Pod 2.7 coherence).
        """
        ...

    async def add_evidence(self, claim_id: str, event: EvidenceEvent) -> None:
        """Append a new evidence event to an existing claim.

        Raises ``ValueError`` if ``claim_id`` is not known.
        Also marks ``(scope, subject_norm)`` dirty in the same transaction (Pod 2.7 coherence).
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
        scope: str | None = None,
    ) -> Sequence[ScoredClaim]:
        """Return scored claims where ``entity`` is subject or object, newest-first.

        ``scope``: when not ``None``, restricts results to claims whose ``scope`` field equals
        the given string (using ``coalesce(c.scope, 'agent')`` so pre-2.4 nodes without a
        ``scope`` property are treated as ``'agent'``).  ``None`` (default) returns all scopes.
        """
        ...

    async def claims_by_similarity(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
        scope: str | None = None,
    ) -> Sequence[ScoredClaim]:
        """Return up to ``k`` scored claims nearest to ``embedding`` (raw cosine >= min_score).

        ``scope``: when not ``None``, post-filters results to the given scope after an over-fetch
        from the vector index (the index cannot pre-filter by scope).  The adapter fetches
        ``max(4 * k, 64)`` candidates and trims to ``k`` in Python.  If fewer than ``k`` survive
        the scope filter, the short result is returned as-is (v1: no re-query).  ``None`` returns
        all scopes.
        """
        ...

    async def claims_full_text(
        self,
        query: str,
        *,
        k: int = 10,
        scope: str | None = None,
        as_of: datetime | None = None,
    ) -> Sequence[ScoredClaim]:
        """BM25 full-text search over claim subject/predicate/payload.

        Score is implementation-defined positive; only descending rank is contractual.
        Empty/whitespace-only query returns []. Skeleton nodes excluded (same as claims_about).
        scope/as_of: same over-fetch + post-filter discipline as claims_by_similarity.
        """
        ...

    async def resolution_candidates(
        self,
        subject: str,
        predicate: str,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        scope: str | None = None,
    ) -> Sequence[Claim]:
        """Return candidate claims for judge-based resolution (no confidence scoring).

        ``scope``: when not ``None``, restricts both the exact-topic pass and the vector-fallback
        pass to claims whose ``scope`` matches (``coalesce(c.scope, 'agent')``).  ``None`` returns
        candidates across all scopes.
        """
        ...

    async def write_contradiction(self, claim_id_a: str, claim_id_b: str) -> None:
        """Record a CONTRADICTS edge between two claims (idempotent MERGE)."""
        ...

    async def contradictions_of(self, claim_id: str) -> Sequence[Claim]:
        """Return all claims that contradict the given claim."""
        ...

    async def invalidate_claim(self, claim_id: str, *, valid_to: datetime) -> None:
        """Set valid_to on the claim (first-invalidation-wins; later calls are no-ops).

        Raises ``ValueError`` if ``claim_id`` is not known. Never deletes.
        Also marks ``(scope, subject_norm)`` dirty in the same transaction (Pod 2.7 coherence).
        """
        ...

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        """Return the consumer's Neo4j (:ProjectionCursor) position, or None on cold start.

        Used by the ClaimExtractor to read its own watermark before a tick.  The cursor is
        written by :meth:`project_claims` in the SAME atomic Neo4j transaction as the claims,
        so read and write always see a consistent view of which steps have been extracted.
        """
        ...

    async def project_claims(
        self,
        consumer: str,
        writes: Sequence[ClaimProjection],
        progress: ProjectionCursor | None,
    ) -> None:
        """ATOMICALLY write a batch of (claim, evidence) pairs and advance the cursor.

        ONE managed Neo4j transaction: for each ClaimProjection, applies write_claim semantics
        (identity-discipline check + coalesce-populate MERGE + evidence CREATE). Then advances
        the Neo4j (:ProjectionCursor {consumer: consumer}) to ordinal_max(current, progress).

        A crash before commit leaves NO claims and NO cursor advance — exactly-once state even
        when the upstream model (the ClaimExtractor) is non-deterministic (D6).

        progress=None: no cursor advance (useful for one-shot imports).
        writes=[]: a no-op (cursor still advances if progress is not None).

        Each claim write also marks ``(scope, subject_norm)`` dirty in the same transaction
        (Pod 2.7 coherence).

        Raises ValueError if any claim.id fails identity-discipline check.
        """
        ...
