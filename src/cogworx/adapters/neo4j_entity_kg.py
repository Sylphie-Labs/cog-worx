"""Neo4j adapter for the entity knowledge graph (CANON S3, S5).

Extends :class:`cogworx.adapters.neo4j_graph.Neo4jGraphStore` (shares driver, config, aclose,
reset, and the raw GraphStore seam) and fully implements the :class:`EntityKG` Protocol.

Schema additions over the base adapter (applied by :meth:`ensure_schema`):
  - UNIQUE (:Entity {name}) — subject/object entities are point-looked-up by name.
  - UNIQUE (:Evidence {id}) — evidence events are globally unique.
  - Composite index on (:Claim {subject_norm, predicate_norm}) — drives resolution candidates.
  - Optional VECTOR INDEX ``claim_embedding`` on (:Claim {embedding}) with cosine similarity.

Evidence events ACCUMULATE (CREATE, never MERGE). Confidence is NEVER stored — it is re-derived at
read time from all accumulated evidence via :func:`cogworx.knowledge.confidence.claim_confidence`.

Vector score translation (CANON S3, tess convention): callers think in RAW cosine [-1, 1]; Neo4j's
index stores (1+cos)/2 ∈ [0, 1]. This adapter translates in/out transparently.

Invalidation: claims are never deleted. ``invalidate_claim`` sets ``valid_to`` only if currently
NULL (first-invalidation-wins). Later calls are no-ops, not errors.

Datetime contract: ALL persisted datetimes (valid_from, valid_to, ingest_time, recorded_at) and
ALL query parameters are UTC-normalised via :func:`to_utc` before storage or comparison. Naive
datetimes are interpreted as UTC (attached timezone.utc); tz-aware datetimes are converted to UTC.
All stored ISO strings share the +00:00 offset so lexicographic order == temporal order.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_graph import Neo4jGraphStore, _as_epistemic, _as_source
from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.confidence import ClaimConfidence, claim_confidence
from cogworx.knowledge.evidence import EvidenceEvent
from cogworx.knowledge.identity import claim_id_for, normalize_topic_part
from cogworx.substrate.entity_kg import ScoredClaim

if TYPE_CHECKING:
    from neo4j import AsyncManagedTransaction, Record

__all__ = ["Neo4jEntityKG"]


# ---------------------------------------------------------------------------
# UTC datetime normalization
# ---------------------------------------------------------------------------


def to_utc(dt: datetime) -> datetime:
    """Normalize a datetime to UTC.

    - Tz-aware: convert to UTC and strip the fold flag.
    - Naive: interpret AS UTC (attach timezone.utc without conversion).

    All datetimes persisted by the entity-KG adapter and all query parameters pass through this
    function so stored ISO strings share the +00:00 offset. This guarantees that lexicographic
    order == temporal order for the as_of filter and ORDER BY ingest_time comparisons.
    """
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_ENTITY_CONSTRAINT = """
CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
FOR (e:Entity) REQUIRE e.name IS UNIQUE
"""

_EVIDENCE_CONSTRAINT = """
CREATE CONSTRAINT evidence_id_unique IF NOT EXISTS
FOR (ev:Evidence) REQUIRE ev.id IS UNIQUE
"""

_CLAIM_TOPIC_INDEX = """
CREATE INDEX claim_topic_composite IF NOT EXISTS
FOR (c:Claim) ON (c.subject_norm, c.predicate_norm)
"""

# Vector index is created only when ensure_schema is called with embedding_dim; the base
# claim_id unique constraint is inherited from Neo4jGraphStore.ensure_schema().
_CLAIM_VECTOR_INDEX = """
CREATE VECTOR INDEX claim_embedding IF NOT EXISTS
FOR (c:Claim) ON (c.embedding)
OPTIONS {{indexConfig: {{
  `vector.dimensions`:  {dim},
  `vector.similarity_function`: 'cosine'
}}}}
"""

# ---------------------------------------------------------------------------
# Write-path Cypher
# ---------------------------------------------------------------------------

# Two write variants: with and without object_entity. Avoids conditional MERGE-or-skip in Cypher.
#
# Skeleton-population strategy (Fix 2): DERIVED_FROM edges may MERGE a placeholder (:Claim {id})
# before the real claim arrives (out-of-order writes are normal; cycles require them). When the
# real claim arrives we must populate ALL fields — but must NOT overwrite an already-populated
# field or clear a valid_to that was set by invalidate_claim.
#
# Solution: replace ON CREATE/ON MATCH split with coalesce() on every field.
#   - If the node is NEW (all fields null) → every coalesce picks the incoming value.
#   - If the node is a SKELETON (payload IS NULL) → every coalesce populates it fully.
#   - If the node is POPULATED (payload IS NOT NULL) → coalesce keeps the existing non-null value;
#     the only exception is embedding which may legitimately be null on an existing node.
#   - valid_to coalesces like the rest: a claim may legitimately be BORN with a known validity
#     end (valid_to is part of the Claim contract), so the first population honours it; once set
#     (by birth or by invalidate_claim) coalesce never clears or replaces it — re-writes cannot
#     resurrect an invalidated claim, and invalidate_claim stays first-wins.
#
# "Populated" marker = c.payload IS NOT NULL (payload is required on every real claim; a skeleton
# created only by DERIVED_FROM link will have payload = null). Read queries filter skeletons via
# WHERE c.payload IS NOT NULL so they never surface in results.
_WRITE_CLAIM_WITH_ENTITY_CYPHER = """
MERGE (s:Entity {name: $subject})
MERGE (o:Entity {name: $object_entity})
MERGE (c:Claim {id: $claim_id})
SET c.subject          = coalesce(c.subject, $subject),
    c.predicate        = coalesce(c.predicate, $predicate),
    c.payload          = coalesce(c.payload, $payload),
    c.object_entity    = coalesce(c.object_entity, $object_entity),
    c.epistemic_type   = coalesce(c.epistemic_type, $epistemic_type),
    c.valid_from       = coalesce(c.valid_from, $valid_from),
    c.valid_to         = coalesce(c.valid_to, $valid_to),
    c.ingest_time      = coalesce(c.ingest_time, $ingest_time),
    c.created_by       = coalesce(c.created_by, $created_by),
    c.embedding        = coalesce(c.embedding, $embedding),
    c.subject_norm     = coalesce(c.subject_norm, $subject_norm),
    c.predicate_norm   = coalesce(c.predicate_norm, $predicate_norm),
    c.prov_source      = coalesce(c.prov_source, $prov_source),
    c.prov_source_ref  = coalesce(c.prov_source_ref, $prov_source_ref),
    c.prov_confidence  = coalesce(c.prov_confidence, $prov_confidence),
    c.prov_evidence    = coalesce(c.prov_evidence, $prov_evidence),
    c.prov_recorded_at = coalesce(c.prov_recorded_at, $prov_recorded_at)
MERGE (s)-[:HAS_CLAIM]->(c)
MERGE (c)-[:REFERS_TO]->(o)
CREATE (ev:Evidence {
  id:               $ev_id,
  type:             $ev_type,
  polarity:         $ev_polarity,
  source_id:        $ev_source_id,
  source_authority: $ev_source_authority,
  base_weight:      $ev_base_weight,
  run_id:           $ev_run_id,
  stage:            $ev_stage,
  recorded_at:      $ev_recorded_at
})
CREATE (c)-[:HAS_EVIDENCE]->(ev)
RETURN c.id AS claim_id
"""

_WRITE_CLAIM_NO_ENTITY_CYPHER = """
MERGE (s:Entity {name: $subject})
MERGE (c:Claim {id: $claim_id})
SET c.subject          = coalesce(c.subject, $subject),
    c.predicate        = coalesce(c.predicate, $predicate),
    c.payload          = coalesce(c.payload, $payload),
    c.object_entity    = CASE WHEN c.object_entity IS NOT NULL THEN c.object_entity ELSE null END,
    c.epistemic_type   = coalesce(c.epistemic_type, $epistemic_type),
    c.valid_from       = coalesce(c.valid_from, $valid_from),
    c.valid_to         = coalesce(c.valid_to, $valid_to),
    c.ingest_time      = coalesce(c.ingest_time, $ingest_time),
    c.created_by       = coalesce(c.created_by, $created_by),
    c.embedding        = coalesce(c.embedding, $embedding),
    c.subject_norm     = coalesce(c.subject_norm, $subject_norm),
    c.predicate_norm   = coalesce(c.predicate_norm, $predicate_norm),
    c.prov_source      = coalesce(c.prov_source, $prov_source),
    c.prov_source_ref  = coalesce(c.prov_source_ref, $prov_source_ref),
    c.prov_confidence  = coalesce(c.prov_confidence, $prov_confidence),
    c.prov_evidence    = coalesce(c.prov_evidence, $prov_evidence),
    c.prov_recorded_at = coalesce(c.prov_recorded_at, $prov_recorded_at)
MERGE (s)-[:HAS_CLAIM]->(c)
CREATE (ev:Evidence {
  id:               $ev_id,
  type:             $ev_type,
  polarity:         $ev_polarity,
  source_id:        $ev_source_id,
  source_authority: $ev_source_authority,
  base_weight:      $ev_base_weight,
  run_id:           $ev_run_id,
  stage:            $ev_stage,
  recorded_at:      $ev_recorded_at
})
CREATE (c)-[:HAS_EVIDENCE]->(ev)
RETURN c.id AS claim_id
"""

# DERIVED_FROM is written in the same write transaction as the claim so a partial state (claim
# exists, lineage absent) never appears. Parent placeholder is MERGE-d — it may not exist yet.
_LINK_DERIVED_FROM_CYPHER = """
MATCH (child:Claim {id: $child_id})
MERGE (parent:Claim {id: $parent_id})
MERGE (child)-[:DERIVED_FROM]->(parent)
"""

# Add one evidence event to an existing claim. MATCH (not MERGE) so we detect unknown claim_ids.
_ADD_EVIDENCE_CYPHER = """
MATCH (c:Claim {id: $claim_id})
CREATE (ev:Evidence {
  id:               $ev_id,
  type:             $ev_type,
  polarity:         $ev_polarity,
  source_id:        $ev_source_id,
  source_authority: $ev_source_authority,
  base_weight:      $ev_base_weight,
  run_id:           $ev_run_id,
  stage:            $ev_stage,
  recorded_at:      $ev_recorded_at
})
CREATE (c)-[:HAS_EVIDENCE]->(ev)
RETURN ev.id AS ev_id
"""

# Retrieve all evidence events for one claim in creation order.
_EVIDENCE_FOR_CYPHER = """
MATCH (c:Claim {id: $claim_id})-[:HAS_EVIDENCE]->(ev:Evidence)
RETURN ev.id               AS id,
       ev.type             AS type,
       ev.polarity         AS polarity,
       ev.source_id        AS source_id,
       ev.source_authority AS source_authority,
       ev.base_weight      AS base_weight,
       ev.run_id           AS run_id,
       ev.stage            AS stage,
       ev.recorded_at      AS recorded_at
ORDER BY ev.recorded_at ASC
"""

# ---------------------------------------------------------------------------
# Read-path Cypher — claims_about (entity as subject OR object)
# ---------------------------------------------------------------------------

# Returns claim rows + own evidence + ancestor evidence (depth *1..5). The depth cap prevents
# runaway traversals on deep lineage graphs. DISTINCT on ancestor aggregation collapses diamond
# lineage paths so the same ancestor's evidence is not double-counted in Python.
_CLAIMS_ABOUT_BASE = """
MATCH (e:Entity {{name: $entity}})
OPTIONAL MATCH (e)-[:HAS_CLAIM]->(c1:Claim)
OPTIONAL MATCH (c2:Claim)-[:REFERS_TO]->(e)
WITH collect(DISTINCT c1) + collect(DISTINCT c2) AS claims_raw
UNWIND claims_raw AS c
WITH DISTINCT c
WHERE c IS NOT NULL AND c.payload IS NOT NULL{as_of_filter}
OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(ev:Evidence)
WITH c, collect({{
  id:               ev.id,
  type:             ev.type,
  polarity:         ev.polarity,
  source_id:        ev.source_id,
  source_authority: ev.source_authority,
  base_weight:      ev.base_weight,
  run_id:           ev.run_id,
  stage:            ev.stage,
  recorded_at:      ev.recorded_at
}}) AS own_evidence
OPTIONAL MATCH (c)-[:DERIVED_FROM*1..5]->(anc:Claim)
OPTIONAL MATCH (anc)-[:HAS_EVIDENCE]->(anc_ev:Evidence)
WITH c, own_evidence, anc,
     collect({{
       id:               anc_ev.id,
       type:             anc_ev.type,
       polarity:         anc_ev.polarity,
       source_id:        anc_ev.source_id,
       source_authority: anc_ev.source_authority,
       base_weight:      anc_ev.base_weight,
       run_id:           anc_ev.run_id,
       stage:            anc_ev.stage,
       recorded_at:      anc_ev.recorded_at
     }}) AS anc_evidence
WITH c, own_evidence,
     collect(DISTINCT {{ claim_id: anc.id, evidence: anc_evidence }}) AS ancestors
RETURN c.id             AS claim_id,
       c.subject        AS subject,
       c.predicate      AS predicate,
       c.payload        AS payload,
       c.object_entity  AS object_entity,
       c.epistemic_type AS epistemic_type,
       c.valid_from     AS valid_from,
       c.valid_to       AS valid_to,
       c.ingest_time    AS ingest_time,
       c.created_by     AS created_by,
       c.embedding      AS embedding,
       c.prov_source    AS prov_source,
       c.prov_source_ref    AS prov_source_ref,
       c.prov_confidence    AS prov_confidence,
       c.prov_evidence      AS prov_evidence,
       c.prov_recorded_at   AS prov_recorded_at,
       own_evidence     AS own_evidence,
       ancestors        AS ancestors
ORDER BY c.ingest_time DESC
LIMIT $limit
"""

# as_of filter clause — injected when the caller passes an as_of datetime.
# Appended to the existing "WHERE c IS NOT NULL" so the format placeholder becomes
# "WHERE c IS NOT NULL AND c.valid_from <= ..." (one WHERE clause, not two).
_AS_OF_FILTER = " AND c.valid_from <= $as_of AND (c.valid_to IS NULL OR c.valid_to > $as_of)"

_CLAIMS_ABOUT_CYPHER = _CLAIMS_ABOUT_BASE.format(as_of_filter="")
_CLAIMS_ABOUT_AS_OF_CYPHER = _CLAIMS_ABOUT_BASE.format(as_of_filter=_AS_OF_FILTER)

# ---------------------------------------------------------------------------
# Read-path Cypher — claims_by_similarity (vector index)
# ---------------------------------------------------------------------------

# Callers think in raw cosine [-1,1]; Neo4j stores (1+cos)/2 — translate in/out.
# Note: this string is NOT a Python format template — braces are literal Cypher map syntax.
_CLAIMS_BY_SIMILARITY_CYPHER = """
CALL db.index.vector.queryNodes('claim_embedding', $k, $embedding)
YIELD node AS c, score
WHERE score >= $min_neo4j_score AND c.payload IS NOT NULL
OPTIONAL MATCH (c)-[:HAS_EVIDENCE]->(ev:Evidence)
WITH c, score, collect({
  id:               ev.id,
  type:             ev.type,
  polarity:         ev.polarity,
  source_id:        ev.source_id,
  source_authority: ev.source_authority,
  base_weight:      ev.base_weight,
  run_id:           ev.run_id,
  stage:            ev.stage,
  recorded_at:      ev.recorded_at
}) AS own_evidence
OPTIONAL MATCH (c)-[:DERIVED_FROM*1..5]->(anc:Claim)
OPTIONAL MATCH (anc)-[:HAS_EVIDENCE]->(anc_ev:Evidence)
WITH c, score, own_evidence, anc,
     collect({
       id:               anc_ev.id,
       type:             anc_ev.type,
       polarity:         anc_ev.polarity,
       source_id:        anc_ev.source_id,
       source_authority: anc_ev.source_authority,
       base_weight:      anc_ev.base_weight,
       run_id:           anc_ev.run_id,
       stage:            anc_ev.stage,
       recorded_at:      anc_ev.recorded_at
     }) AS anc_evidence
WITH c, score, own_evidence,
     collect(DISTINCT { claim_id: anc.id, evidence: anc_evidence }) AS ancestors
RETURN c.id             AS claim_id,
       c.subject        AS subject,
       c.predicate      AS predicate,
       c.payload        AS payload,
       c.object_entity  AS object_entity,
       c.epistemic_type AS epistemic_type,
       c.valid_from     AS valid_from,
       c.valid_to       AS valid_to,
       c.ingest_time    AS ingest_time,
       c.created_by     AS created_by,
       c.embedding      AS embedding,
       c.prov_source    AS prov_source,
       c.prov_source_ref    AS prov_source_ref,
       c.prov_confidence    AS prov_confidence,
       c.prov_evidence      AS prov_evidence,
       c.prov_recorded_at   AS prov_recorded_at,
       own_evidence     AS own_evidence,
       ancestors        AS ancestors,
       2.0 * score - 1.0 AS raw_similarity
ORDER BY score DESC
"""

# ---------------------------------------------------------------------------
# Read-path Cypher — resolution_candidates
# ---------------------------------------------------------------------------

_RESOLUTION_BY_TOPIC_CYPHER = """
MATCH (c:Claim)
WHERE c.subject_norm   = $subject_norm
  AND c.predicate_norm = $predicate_norm
  AND c.payload IS NOT NULL
RETURN c.id             AS claim_id,
       c.subject        AS subject,
       c.predicate      AS predicate,
       c.payload        AS payload,
       c.object_entity  AS object_entity,
       c.epistemic_type AS epistemic_type,
       c.valid_from     AS valid_from,
       c.valid_to       AS valid_to,
       c.ingest_time    AS ingest_time,
       c.created_by     AS created_by,
       c.embedding      AS embedding,
       c.prov_source    AS prov_source,
       c.prov_source_ref    AS prov_source_ref,
       c.prov_confidence    AS prov_confidence,
       c.prov_evidence      AS prov_evidence,
       c.prov_recorded_at   AS prov_recorded_at
ORDER BY c.ingest_time DESC
LIMIT $k
"""

# Vector fallback for resolution_candidates — min_score 0.70 raw cosine.
_RESOLUTION_BY_VECTOR_CYPHER = """
CALL db.index.vector.queryNodes('claim_embedding', $k, $embedding)
YIELD node AS c, score
WHERE score >= $min_neo4j_score AND c.payload IS NOT NULL
RETURN c.id             AS claim_id,
       c.subject        AS subject,
       c.predicate      AS predicate,
       c.payload        AS payload,
       c.object_entity  AS object_entity,
       c.epistemic_type AS epistemic_type,
       c.valid_from     AS valid_from,
       c.valid_to       AS valid_to,
       c.ingest_time    AS ingest_time,
       c.created_by     AS created_by,
       c.embedding      AS embedding,
       c.prov_source    AS prov_source,
       c.prov_source_ref    AS prov_source_ref,
       c.prov_confidence    AS prov_confidence,
       c.prov_evidence      AS prov_evidence,
       c.prov_recorded_at   AS prov_recorded_at
ORDER BY score DESC
"""

# ---------------------------------------------------------------------------
# Contradiction Cypher
# ---------------------------------------------------------------------------

# Semantically undirected — store one directed edge, query with undirected pattern.
_WRITE_CONTRADICTS_CYPHER = """
MATCH (a:Claim {id: $claim_id_a})
MATCH (b:Claim {id: $claim_id_b})
MERGE (a)-[:CONTRADICTS]->(b)
"""

_CONTRADICTIONS_OF_CYPHER = """
MATCH (anchor:Claim {id: $claim_id})-[:CONTRADICTS]-(other:Claim)
WHERE other.payload IS NOT NULL
RETURN DISTINCT
       other.id             AS claim_id,
       other.subject        AS subject,
       other.predicate      AS predicate,
       other.payload        AS payload,
       other.object_entity  AS object_entity,
       other.epistemic_type AS epistemic_type,
       other.valid_from     AS valid_from,
       other.valid_to       AS valid_to,
       other.ingest_time    AS ingest_time,
       other.created_by     AS created_by,
       other.embedding      AS embedding,
       other.prov_source    AS prov_source,
       other.prov_source_ref    AS prov_source_ref,
       other.prov_confidence    AS prov_confidence,
       other.prov_evidence      AS prov_evidence,
       other.prov_recorded_at   AS prov_recorded_at
"""

# ---------------------------------------------------------------------------
# Invalidation Cypher
# ---------------------------------------------------------------------------

# First-invalidation-wins: only sets valid_to when currently null. RETURN row so the caller
# can detect unknown claim_id (None row = no such claim → ValueError in Python).
_INVALIDATE_CLAIM_CYPHER = """
MATCH (c:Claim {id: $claim_id})
SET c.valid_to = CASE WHEN c.valid_to IS NULL THEN $valid_to ELSE c.valid_to END
RETURN c.id AS claim_id
"""

# Reset: wipe Entity + Evidence nodes in addition to base Claim wipe (integration fixture).
_RESET_ENTITY_CYPHER = "MATCH (e:Entity) DETACH DELETE e"
_RESET_EVIDENCE_CYPHER = "MATCH (ev:Evidence) DETACH DELETE ev"

# Min raw cosine for the vector fallback in resolution_candidates. Mirrors tess.kg convention.
_RESOLUTION_MIN_RAW_COSINE: float = 0.70


def _entity_kg_record_to_claim(record: Any) -> Claim:
    """Deserialise a Cypher result row from the entity-KG read queries into a Claim.

    All entity-KG Cypher queries alias the claim id as ``claim_id`` (not ``id``) so that the
    column name is self-documenting in result sets that also contain other ids (e.g. ancestor
    ``claim_id``). This function normalises ``claim_id`` → ``id`` before field extraction so
    callers do not need to know about the alias.
    """
    data = record.data() if hasattr(record, "data") else dict(record)
    # Normalise the id key: entity-KG Cypher uses "claim_id"; base Cypher uses "id".
    if "id" not in data and "claim_id" in data:
        data = dict(data)
        data["id"] = data["claim_id"]
    embedding_raw = data["embedding"]
    embedding = tuple(float(v) for v in embedding_raw) if embedding_raw else None
    provenance = Provenance(
        source=_as_source(data["prov_source"]),
        source_ref=data["prov_source_ref"],
        confidence=float(data["prov_confidence"]),
        evidence=tuple(data["prov_evidence"] or ()),
        recorded_at=datetime.fromisoformat(data["prov_recorded_at"]),
    )
    valid_to_raw = data["valid_to"]
    return Claim(
        id=str(data["id"]),
        subject=str(data["subject"]),
        predicate=data["predicate"],
        payload=str(data["payload"]),
        object_entity=data.get("object_entity"),
        epistemic_type=_as_epistemic(data["epistemic_type"]),
        provenance=provenance,
        valid_from=datetime.fromisoformat(data["valid_from"]),
        valid_to=datetime.fromisoformat(valid_to_raw) if valid_to_raw is not None else None,
        ingest_time=datetime.fromisoformat(data["ingest_time"]),
        created_by=str(data["created_by"]),
        embedding=embedding,
    )


class Neo4jEntityKG(Neo4jGraphStore):
    """Entity knowledge graph on Neo4j — event-sourced claims with Beta confidence.

    Extends :class:`~cogworx.adapters.neo4j_graph.Neo4jGraphStore` (inherits driver lifecycle,
    ``aclose``, ``upsert_claim``, ``get_claim``, ``neighbors``) and implements the
    :class:`~cogworx.substrate.entity_kg.EntityKG` Protocol.

    Embeddings are always PASSED IN by callers — this class performs zero model calls (S1).
    """

    def __init__(
        self,
        *,
        settings: SubstrateSettings | None = None,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        super().__init__(settings=settings, uri=uri, user=user, password=password)

    async def ensure_schema(self, *, embedding_dim: int | None = None) -> None:
        """Idempotently create constraints + indexes for the entity-KG schema.

        Calls the base constraint (UNIQUE :Claim {id}) first, then adds the entity-KG additions.
        Pass ``embedding_dim`` to also create the native VECTOR INDEX on (:Claim {embedding}).
        """
        await super().ensure_schema()

        async def _write_ddl(tx: AsyncManagedTransaction) -> None:
            await tx.run(_ENTITY_CONSTRAINT)
            await tx.run(_EVIDENCE_CONSTRAINT)
            await tx.run(_CLAIM_TOPIC_INDEX)

        async with self._connection.session() as session:
            await session.execute_write(_write_ddl)

        if embedding_dim is not None:
            vector_cypher = _CLAIM_VECTOR_INDEX.format(dim=embedding_dim)

            async def _write_vector(tx: AsyncManagedTransaction) -> None:
                await tx.run(vector_cypher)

            async with self._connection.session() as session:
                await session.execute_write(_write_vector)

    async def reset(self) -> None:
        """Drop all :Claim, :Entity, and :Evidence nodes. Used by per-case integration fixtures."""
        await super().reset()

        async def _wipe(tx: AsyncManagedTransaction) -> None:
            await tx.run(_RESET_ENTITY_CYPHER)
            await tx.run(_RESET_EVIDENCE_CYPHER)

        async with self._connection.session() as session:
            await session.execute_write(_wipe)

    # -----------------------------------------------------------------------
    # Sealed write-seam (FIX 1 — upsert_claim back-door)
    # -----------------------------------------------------------------------

    async def upsert_claim(self, claim: Claim) -> str:
        """Sealed: the entity KG's only write surface is write_claim.

        Raises ``NotImplementedError`` unconditionally. The inherited Phase-0
        ``Neo4jGraphStore.upsert_claim`` performs a blanket SET of every field (including
        valid_to → resurrects invalidated claims; epistemic_type → silent S5 merge) with no
        identity discipline. Using it on an entity-KG node bypasses:
          - Identity discipline (claim.id == claim_id_for(...) check in write_claim).
          - Immutable-on-match semantics (coalesce keeps the first real write's fields).
          - Bi-temporal honesty (valid_to = first-invalidation-wins, never overwritten).

        Use ``write_claim`` instead.
        """
        raise NotImplementedError(
            "Neo4jEntityKG.upsert_claim is disabled. "
            "The entity KG's only write surface is write_claim (identity discipline + "
            "immutable-on-match + bi-temporal honesty). "
            "The inherited Phase-0 Neo4jGraphStore.upsert_claim bypasses all three invariants."
        )

    # -----------------------------------------------------------------------
    # Record deserialization
    # -----------------------------------------------------------------------

    @staticmethod
    def _record_to_claim(record: Any) -> Claim:
        """Override the base class method to accept the ``claim_id`` alias used by all entity-KG
        Cypher queries (which return ``c.id AS claim_id`` rather than ``c.id AS id``).

        Replicates the base extraction, normalising ``claim_id`` → ``id`` before field reads.
        """
        return _entity_kg_record_to_claim(record)

    # -----------------------------------------------------------------------
    # EntityKG Protocol implementation
    # -----------------------------------------------------------------------

    async def write_claim(self, claim: Claim, *, evidence: EvidenceEvent) -> str:
        """MERGE the claim node and CREATE one evidence event. Returns the canonical claim id.

        Identity discipline: raises ``ValueError`` before any I/O if ``claim.id`` does not match
        ``claim_id_for(subject, predicate or "", object_entity or payload)``.
        Always records the evidence, even when the claim node already existed.
        """
        _assert_identity(claim)

        cypher = (
            _WRITE_CLAIM_WITH_ENTITY_CYPHER
            if claim.object_entity is not None
            else _WRITE_CLAIM_NO_ENTITY_CYPHER
        )
        params = _write_claim_params(claim, evidence)

        async def _work(tx: AsyncManagedTransaction) -> None:
            await tx.run(cypher, **params)
            # Link DERIVED_FROM edges within the same transaction so partial states cannot appear.
            for parent_id in claim.provenance.evidence:
                await tx.run(_LINK_DERIVED_FROM_CYPHER, child_id=claim.id, parent_id=parent_id)

        async with self._connection.session() as session:
            await session.execute_write(_work)
        return claim.id

    async def add_evidence(self, claim_id: str, event: EvidenceEvent) -> None:
        """Append a new evidence event to an existing claim.

        Raises ``ValueError`` if ``claim_id`` is unknown.
        """
        params = _ev_params(event)
        params["claim_id"] = claim_id

        async def _work(tx: AsyncManagedTransaction) -> Record | None:
            result = await tx.run(_ADD_EVIDENCE_CYPHER, **params)
            return await result.single()

        async with self._connection.session() as session:
            record = await session.execute_write(_work)
        if record is None:
            raise ValueError(f"add_evidence: unknown claim_id {claim_id!r}")

    async def evidence_for(self, claim_id: str) -> Sequence[EvidenceEvent]:
        """Return all evidence events for the claim, in creation order."""

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(_EVIDENCE_FOR_CYPHER, claim_id=claim_id)
            return [r async for r in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)
        return tuple(_record_to_evidence(r) for r in records)

    async def claims_about(
        self,
        entity: str,
        *,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> Sequence[ScoredClaim]:
        """Return scored claims where ``entity`` is subject or object, newest-first.

        ``as_of`` filter: valid_from <= as_of AND (valid_to IS NULL OR valid_to > as_of).
        Confidence is derived at read — never stored.
        """
        cypher = _CLAIMS_ABOUT_AS_OF_CYPHER if as_of is not None else _CLAIMS_ABOUT_CYPHER
        params: dict[str, Any] = {"entity": entity, "limit": limit}
        if as_of is not None:
            # UTC-normalise so the ISO string comparison in Cypher is correct regardless of the
            # caller's offset. Stored datetimes are always +00:00 after FIX 4 normalisation.
            params["as_of"] = to_utc(as_of).isoformat()

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(cypher, **params)
            return [r async for r in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)
        return tuple(_record_to_scored_claim(r) for r in records)

    async def claims_by_similarity(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
    ) -> Sequence[ScoredClaim]:
        """Return up to ``k`` scored claims nearest to ``embedding`` (raw cosine >= min_score).

        Translates min_score (raw cosine) → Neo4j (1+cos)/2 on the way in, and converts returned
        Neo4j scores → raw cosine on the way out (already done in Cypher: 2*score-1).
        """
        min_neo4j = (1.0 + min_score) / 2.0
        params: dict[str, Any] = {
            "embedding": list(embedding),
            "k": k,
            "min_neo4j_score": min_neo4j,
        }

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(_CLAIMS_BY_SIMILARITY_CYPHER, **params)
            return [r async for r in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)
        return tuple(_record_to_scored_claim(r, similarity_key="raw_similarity") for r in records)

    async def resolution_candidates(
        self,
        subject: str,
        predicate: str,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
    ) -> Sequence[Claim]:
        """Return candidate claims for judge-based resolution (plain Claims, no confidence).

        Primary: exact match on (subject_norm, predicate_norm), newest-first, up to ``k``.
        If primary yields fewer than ``k`` AND ``embedding`` is given, fills from the vector
        channel (min raw cosine 0.70), deduped by id, still capped at ``k``.
        """
        subject_norm = normalize_topic_part(subject)
        predicate_norm = normalize_topic_part(predicate)

        async def _topic_work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(
                _RESOLUTION_BY_TOPIC_CYPHER,
                subject_norm=subject_norm,
                predicate_norm=predicate_norm,
                k=k,
            )
            return [r async for r in result]

        async with self._connection.session() as session:
            topic_records = await session.execute_read(_topic_work)

        seen: set[str] = set()
        candidates: list[Claim] = []
        for r in topic_records:
            claim = self._record_to_claim(r)
            seen.add(claim.id)
            candidates.append(claim)

        if len(candidates) < k and embedding is not None:
            need = k - len(candidates)
            min_neo4j = (1.0 + _RESOLUTION_MIN_RAW_COSINE) / 2.0

            async def _vector_work(tx: AsyncManagedTransaction) -> list[Record]:
                result = await tx.run(
                    _RESOLUTION_BY_VECTOR_CYPHER,
                    embedding=list(embedding),
                    k=k,
                    min_neo4j_score=min_neo4j,
                )
                return [r async for r in result]

            async with self._connection.session() as session:
                vector_records = await session.execute_read(_vector_work)

            for r in vector_records:
                claim = self._record_to_claim(r)
                if claim.id not in seen:
                    seen.add(claim.id)
                    candidates.append(claim)
                    if len(candidates) >= k or len(candidates) - len(topic_records) >= need:
                        break

        return tuple(candidates[:k])

    async def write_contradiction(self, claim_id_a: str, claim_id_b: str) -> None:
        """Record a CONTRADICTS edge (idempotent MERGE, semantically undirected)."""

        async def _work(tx: AsyncManagedTransaction) -> None:
            await tx.run(_WRITE_CONTRADICTS_CYPHER, claim_id_a=claim_id_a, claim_id_b=claim_id_b)

        async with self._connection.session() as session:
            await session.execute_write(_work)

    async def contradictions_of(self, claim_id: str) -> Sequence[Claim]:
        """Return all claims that contradict the given claim (undirected traversal)."""

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(_CONTRADICTIONS_OF_CYPHER, claim_id=claim_id)
            return [r async for r in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)
        return tuple(self._record_to_claim(r) for r in records)

    async def invalidate_claim(self, claim_id: str, *, valid_to: datetime) -> None:
        """Set valid_to on the claim (first-invalidation-wins; later calls are no-ops).

        Raises ``ValueError`` if ``claim_id`` is not known. Never deletes.
        valid_to is UTC-normalised before storage.
        """

        async def _work(tx: AsyncManagedTransaction) -> Record | None:
            result = await tx.run(
                _INVALIDATE_CLAIM_CYPHER,
                claim_id=claim_id,
                valid_to=to_utc(valid_to).isoformat(),
            )
            return await result.single()

        async with self._connection.session() as session:
            record = await session.execute_write(_work)
        if record is None:
            raise ValueError(f"invalidate_claim: unknown claim_id {claim_id!r}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _assert_identity(claim: Claim) -> None:
    """Enforce identity discipline: claim.id must equal claim_id_for(...).

    Raises ``ValueError`` before any I/O so caller receives a loud error rather than a silently
    misrouted evidence event.
    """
    object_repr = claim.object_entity if claim.object_entity is not None else claim.payload
    expected = claim_id_for(claim.subject, claim.predicate or "", object_repr)
    if claim.id != expected:
        raise ValueError(
            f"claim.id {claim.id!r} does not match expected "
            f"claim_id_for({claim.subject!r}, {claim.predicate!r}, {object_repr!r}) = {expected!r}"
        )


def _ev_params(ev: EvidenceEvent) -> dict[str, Any]:
    """Flatten one EvidenceEvent into the ``ev_*`` Cypher parameters.

    recorded_at is UTC-normalised so all evidence timestamps share the +00:00 offset.
    """
    return {
        "ev_id": ev.id,
        "ev_type": ev.type,
        "ev_polarity": ev.polarity,
        "ev_source_id": ev.source_id,
        "ev_source_authority": ev.source_authority,
        "ev_base_weight": ev.base_weight,
        "ev_run_id": ev.run_id,
        "ev_stage": ev.stage,
        "ev_recorded_at": to_utc(ev.recorded_at).isoformat(),
    }


def _write_claim_params(claim: Claim, ev: EvidenceEvent) -> dict[str, Any]:
    """Build the full parameter dict for the write_claim Cypher variants.

    All datetimes are UTC-normalised via to_utc() so stored ISO strings share +00:00 offset and
    lexicographic order matches temporal order (required for the as_of filter correctness).
    """
    prov = claim.provenance
    params: dict[str, Any] = {
        "claim_id": claim.id,
        "subject": claim.subject,
        "predicate": claim.predicate,
        "payload": claim.payload,
        "object_entity": claim.object_entity,
        "epistemic_type": claim.epistemic_type,
        "valid_from": to_utc(claim.valid_from).isoformat(),
        "valid_to": to_utc(claim.valid_to).isoformat() if claim.valid_to is not None else None,
        "ingest_time": to_utc(claim.ingest_time).isoformat(),
        "created_by": claim.created_by,
        "embedding": list(claim.embedding) if claim.embedding is not None else None,
        "subject_norm": normalize_topic_part(claim.subject),
        "predicate_norm": normalize_topic_part(claim.predicate or ""),
        "prov_source": prov.source,
        "prov_source_ref": prov.source_ref,
        "prov_confidence": prov.confidence,
        "prov_evidence": list(prov.evidence),
        "prov_recorded_at": to_utc(prov.recorded_at).isoformat(),
    }
    params.update(_ev_params(ev))
    return params


def _evidence_items_from_rows(rows: list[dict[str, Any]]) -> list[EvidenceEvent]:
    """Convert Cypher-collected evidence map-rows to EvidenceEvent objects.

    Rows with null ``type`` (from OPTIONAL MATCH with no evidence) are skipped so an evidence-less
    ancestor doesn't inject a null-valued item into claim_confidence.
    """
    events: list[EvidenceEvent] = []
    for row in rows:
        if row.get("type") is None:
            continue
        events.append(
            EvidenceEvent(
                id=str(row["id"]) if row.get("id") else "",
                type=row["type"],
                polarity=row["polarity"],
                source_id=str(row["source_id"]),
                source_authority=float(row["source_authority"]),
                base_weight=float(row["base_weight"]),
                run_id=row.get("run_id"),
                stage=row.get("stage"),
                recorded_at=datetime.fromisoformat(str(row["recorded_at"]))
                if row.get("recorded_at")
                else datetime.min,
            )
        )
    return events


def _derive_confidence(record: Any) -> tuple[ClaimConfidence, float]:
    """Derive own confidence and lineage_min_confidence from a claim result row.

    Returns (own_confidence, lineage_min_confidence). Ancestor nodes that are placeholder-only
    (no evidence, no properties beyond id) are treated as prior-only confidence.
    """
    own_rows: list[dict[str, Any]] = list(record["own_evidence"]) if record["own_evidence"] else []
    own_events = _evidence_items_from_rows(own_rows)
    own_conf = claim_confidence(own_events)

    ancestor_confidences: list[float] = []
    for anc in record["ancestors"] or []:
        if anc.get("claim_id") is None:
            continue
        anc_rows = list(anc["evidence"]) if anc.get("evidence") else []
        anc_events = _evidence_items_from_rows(anc_rows)
        anc_conf = claim_confidence(anc_events)
        ancestor_confidences.append(anc_conf.confidence)

    lineage_min = (
        min(own_conf.confidence, *ancestor_confidences)
        if ancestor_confidences
        else own_conf.confidence
    )
    return own_conf, lineage_min


def _record_to_scored_claim(
    record: Any,
    *,
    similarity_key: str | None = None,
) -> ScoredClaim:
    """Rebuild a ScoredClaim from a Cypher result row that includes own_evidence + ancestors."""
    claim = _entity_kg_record_to_claim(record)
    own_conf, lineage_min = _derive_confidence(record)
    similarity: float | None = None
    if similarity_key is not None:
        raw = record.get(similarity_key)
        similarity = float(raw) if raw is not None else None
    return ScoredClaim(
        claim=claim,
        confidence=own_conf,
        lineage_min_confidence=lineage_min,
        similarity=similarity,
    )


def _record_to_evidence(record: Any) -> EvidenceEvent:
    """Rebuild one EvidenceEvent from a Cypher result row."""
    data = record.data() if hasattr(record, "data") else dict(record)
    return EvidenceEvent(
        id=str(data["id"]),
        type=data["type"],
        polarity=data["polarity"],
        source_id=str(data["source_id"]),
        source_authority=float(data["source_authority"]),
        base_weight=float(data["base_weight"]),
        run_id=data.get("run_id"),
        stage=data.get("stage"),
        recorded_at=datetime.fromisoformat(str(data["recorded_at"])),
    )
