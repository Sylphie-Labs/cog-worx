"""Neo4j adapter for the graph-knowledge seam (CANON S3, S5).

Implements :class:`cogworx.substrate.graph_store.GraphStore` against a real Neo4j 5+ instance using
the async driver. This is the graph knowledge layer ONLY (S3): the durable journal lives in
TimescaleDB and the latent space in pgvector — neither belongs here.

Provenance + epistemic type ride on every claim node (S5). Provenance is stored as flattened
``prov_*`` properties on the ``:Claim`` node rather than a separate ``(:Provenance)`` node: a
``Provenance`` is a single immutable value-object owned 1:1 by its claim, with no independent query
surface in Phase 0, so a separate node + ``HAS_PROVENANCE`` hop would add a write and a traversal
for zero benefit. (Contrast tess's ``(:Evidence)`` node, which is event-sourced and accumulates
many-per-claim — a genuinely different cardinality.) The provenance ``evidence`` tuple is the one
part that becomes graph structure: each id is a ``[:DERIVED_FROM]`` edge so "why did this surface"
stays a traversal, which is what :meth:`Neo4jGraphStore.neighbors` walks.

Datetimes round-trip as ISO-8601 strings (tz-aware) so :meth:`Neo4jGraphStore.get_claim` rebuilds
the exact ``Claim`` the caller persisted without Neo4j temporal coercion surprises.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

from neo4j import AsyncGraphDatabase

from cogworx.adapters.config import SubstrateSettings
from cogworx.claims.provenance import Claim, EpistemicType, Provenance, ProvenanceSource

if TYPE_CHECKING:
    from collections.abc import Sequence

    from neo4j import AsyncDriver, AsyncManagedTransaction, Record

__all__ = ["Neo4jGraphStore"]


# Schema: one UNIQUE constraint so MERGE on (:Claim {id}) is a point-lookup and idempotent.
# Recall indexes (native vector / full-text BM25) are Phase 2 — not created here.
_ENSURE_CONSTRAINT_CYPHER = """
CREATE CONSTRAINT claim_id_unique IF NOT EXISTS
FOR (c:Claim) REQUIRE c.id IS UNIQUE
"""

# Idempotent upsert. MERGE on the unique id, SET all scalar properties (provenance flattened to
# prov_* — S5), then MERGE one [:DERIVED_FROM] edge per evidence id so lineage is a traversal.
# UNWIND over an empty list is a no-op, so the evidence-less case needs no separate query.
_UPSERT_CLAIM_CYPHER = """
MERGE (c:Claim {id: $id})
SET c.subject           = $subject,
    c.predicate         = $predicate,
    c.payload           = $payload,
    c.epistemic_type    = $epistemic_type,
    c.valid_from        = $valid_from,
    c.valid_to          = $valid_to,
    c.ingest_time       = $ingest_time,
    c.created_by        = $created_by,
    c.embedding         = $embedding,
    c.prov_source       = $prov_source,
    c.prov_source_ref   = $prov_source_ref,
    c.prov_confidence   = $prov_confidence,
    c.prov_evidence     = $prov_evidence,
    c.prov_recorded_at  = $prov_recorded_at
WITH c
UNWIND $prov_evidence AS evidence_id
MERGE (e:Claim {id: evidence_id})
MERGE (c)-[:DERIVED_FROM]->(e)
"""

# All persisted properties, named so get_claim/neighbors share one projection + one rebuild path.
_CLAIM_RETURN = """
       c.id              AS id,
       c.subject         AS subject,
       c.predicate       AS predicate,
       c.payload         AS payload,
       c.epistemic_type  AS epistemic_type,
       c.valid_from      AS valid_from,
       c.valid_to        AS valid_to,
       c.ingest_time     AS ingest_time,
       c.created_by      AS created_by,
       c.embedding       AS embedding,
       c.prov_source     AS prov_source,
       c.prov_source_ref AS prov_source_ref,
       c.prov_confidence AS prov_confidence,
       c.prov_evidence   AS prov_evidence,
       c.prov_recorded_at AS prov_recorded_at
"""

_GET_CLAIM_CYPHER = f"""
MATCH (c:Claim {{id: $id}})
RETURN{_CLAIM_RETURN}
"""

# Neighbors: claims one [:DERIVED_FROM] hop away in EITHER direction (this claim's evidence, plus
# claims that cite this one as evidence). Phase 0 keeps recall thin; full fusion is Phase 2.
_NEIGHBORS_CYPHER = f"""
MATCH (c:Claim {{id: $id}})-[:DERIVED_FROM]-(n:Claim)
RETURN DISTINCT{_CLAIM_RETURN}
LIMIT $limit
"""

_RESET_CYPHER = "MATCH (c:Claim) DETACH DELETE c"


def _iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


class Neo4jGraphStore:
    """A ``GraphStore`` backed by Neo4j 5+ (async driver). Graph knowledge layer only (S3)."""

    def __init__(
        self,
        *,
        settings: SubstrateSettings | None = None,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
    ) -> None:
        resolved = settings if settings is not None else SubstrateSettings()
        self._uri = uri if uri is not None else resolved.neo4j_uri
        self._user = user if user is not None else resolved.neo4j_user
        self._password = password if password is not None else resolved.neo4j_password
        self._driver: AsyncDriver | None = None

    @property
    def _connection(self) -> AsyncDriver:
        """Lazily construct the async driver so __init__ does no I/O."""
        if self._driver is None:
            self._driver = AsyncGraphDatabase.driver(self._uri, auth=(self._user, self._password))
        return self._driver

    async def ensure_schema(self) -> None:
        """Idempotently create the UNIQUE constraint on ``(:Claim {id})``."""

        async def _work(tx: AsyncManagedTransaction) -> None:
            await tx.run(_ENSURE_CONSTRAINT_CYPHER)

        async with self._connection.session() as session:
            await session.execute_write(_work)

    async def upsert_claim(self, claim: Claim) -> str:
        """MERGE the claim on its id (idempotent), flatten provenance, link evidence. Returns id."""
        params = self._claim_to_params(claim)

        async def _work(tx: AsyncManagedTransaction) -> None:
            await tx.run(_UPSERT_CLAIM_CYPHER, **params)

        async with self._connection.session() as session:
            await session.execute_write(_work)
        return claim.id

    async def get_claim(self, claim_id: str) -> Claim | None:
        """Rebuild the full ``Claim`` (provenance + evidence) by id, or ``None``."""

        async def _work(tx: AsyncManagedTransaction) -> Record | None:
            result = await tx.run(_GET_CLAIM_CYPHER, id=claim_id)
            return await result.single()

        async with self._connection.session() as session:
            record = await session.execute_read(_work)
        return self._record_to_claim(record) if record is not None else None

    async def neighbors(self, claim_id: str, *, limit: int = 20) -> Sequence[Claim]:
        """Claims one ``[:DERIVED_FROM]`` hop away (either direction), up to ``limit``."""

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(_NEIGHBORS_CYPHER, id=claim_id, limit=limit)
            return [record async for record in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)
        return tuple(self._record_to_claim(record) for record in records)

    async def reset(self) -> None:
        """Drop every ``:Claim`` (and its edges). Used by the per-case integration fixture."""

        async def _work(tx: AsyncManagedTransaction) -> None:
            await tx.run(_RESET_CYPHER)

        async with self._connection.session() as session:
            await session.execute_write(_work)

    async def aclose(self) -> None:
        """Close the async driver if it was constructed."""
        if self._driver is not None:
            await self._driver.close()
            self._driver = None

    @staticmethod
    def _claim_to_params(claim: Claim) -> dict[str, Any]:
        prov = claim.provenance
        return {
            "id": claim.id,
            "subject": claim.subject,
            "predicate": claim.predicate,
            "payload": claim.payload,
            "epistemic_type": claim.epistemic_type,
            "valid_from": claim.valid_from.isoformat(),
            "valid_to": _iso_or_none(claim.valid_to),
            "ingest_time": claim.ingest_time.isoformat(),
            "created_by": claim.created_by,
            "embedding": list(claim.embedding) if claim.embedding is not None else None,
            "prov_source": prov.source,
            "prov_source_ref": prov.source_ref,
            "prov_confidence": prov.confidence,
            "prov_evidence": list(prov.evidence),
            "prov_recorded_at": prov.recorded_at.isoformat(),
        }

    @staticmethod
    def _record_to_claim(record: Record) -> Claim:
        data = record.data()
        embedding_raw = data["embedding"]
        embedding = tuple(float(value) for value in embedding_raw) if embedding_raw else None
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
            epistemic_type=_as_epistemic(data["epistemic_type"]),
            provenance=provenance,
            valid_from=datetime.fromisoformat(data["valid_from"]),
            valid_to=datetime.fromisoformat(valid_to_raw) if valid_to_raw is not None else None,
            ingest_time=datetime.fromisoformat(data["ingest_time"]),
            created_by=str(data["created_by"]),
            embedding=embedding,
        )


# Explicit maps from the persisted string back to the closed Literal. Spelling each member as a
# literal lets mypy --strict narrow without a `# type: ignore` and rejects drift at the boundary.
_EPISTEMIC_TYPES: dict[str, EpistemicType] = {
    "observation": "observation",
    "inference": "inference",
    "confirmed": "confirmed",
}
_PROVENANCE_SOURCES: dict[str, ProvenanceSource] = {
    "human": "human",
    "sensor": "sensor",
    "tool": "tool",
    "extraction": "extraction",
    "reflection": "reflection",
    "inference": "inference",
}


def _as_epistemic(value: Any) -> EpistemicType:
    text = str(value)
    epistemic = _EPISTEMIC_TYPES.get(text)
    if epistemic is None:
        raise ValueError(f"unknown epistemic_type persisted in Neo4j: {text!r}")
    return epistemic


def _as_source(value: Any) -> ProvenanceSource:
    text = str(value)
    source = _PROVENANCE_SOURCES.get(text)
    if source is None:
        raise ValueError(f"unknown provenance source persisted in Neo4j: {text!r}")
    return source
