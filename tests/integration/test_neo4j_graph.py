"""Integration tests for the Neo4j graph adapter (CANON S3, S5).

These hit a REAL Neo4j (``docker compose up -d`` first) and ERROR — not silently pass — when it is
unreachable. They only run under ``-m integration``. Each case gets an ephemeral, wiped graph.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_graph import Neo4jGraphStore
from cogworx.claims.provenance import Claim, Provenance

pytestmark = pytest.mark.integration


def _claim(claim_id: str, *, evidence: tuple[str, ...] = ()) -> Claim:
    now = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
    return Claim(
        id=claim_id,
        subject="pluto",
        predicate="has_mass",
        payload=f"payload-{claim_id}",
        epistemic_type="inference",
        provenance=Provenance(
            source="extraction",
            source_ref="doc-42",
            confidence=0.8,
            evidence=evidence,
            recorded_at=now,
        ),
        valid_from=now,
        valid_to=None,
        ingest_time=now,
        created_by="stage:research",
        embedding=(0.1, 0.2, 0.3),
    )


@pytest.fixture
async def store(settings: SubstrateSettings) -> AsyncIterator[Neo4jGraphStore]:
    adapter = Neo4jGraphStore(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


async def test_upsert_roundtrips_claim(store: Neo4jGraphStore) -> None:
    claim = _claim("claim:a", evidence=("claim:evidence-1",))
    returned_id = await store.upsert_claim(claim)
    assert returned_id == claim.id

    fetched = await store.get_claim("claim:a")
    assert fetched is not None
    assert fetched.subject == "pluto"
    assert fetched.payload == "payload-claim:a"
    assert fetched.epistemic_type == "inference"
    assert fetched.provenance.source == "extraction"
    assert fetched.provenance.confidence == pytest.approx(0.8)
    assert fetched.provenance.evidence == ("claim:evidence-1",)
    assert fetched.embedding == pytest.approx((0.1, 0.2, 0.3))


async def test_upsert_is_idempotent(store: Neo4jGraphStore) -> None:
    claim = _claim("claim:dup")
    await store.upsert_claim(claim)
    await store.upsert_claim(claim)

    # One node survives two upserts: re-fetch is single, and a neighbor on it sees exactly one edge.
    a = _claim("claim:a", evidence=("claim:dup",))
    await store.upsert_claim(a)
    await store.upsert_claim(a)
    neighbors = await store.neighbors("claim:dup")
    assert [n.id for n in neighbors] == ["claim:a"]


async def test_neighbors_follows_derived_from(store: Neo4jGraphStore) -> None:
    a = _claim("claim:A")
    b = _claim("claim:B", evidence=("claim:A",))
    await store.upsert_claim(a)
    await store.upsert_claim(b)

    neighbors = await store.neighbors("claim:B")
    assert "claim:A" in {n.id for n in neighbors}


async def test_get_unknown_returns_none(store: Neo4jGraphStore) -> None:
    assert await store.get_claim("claim:missing") is None
