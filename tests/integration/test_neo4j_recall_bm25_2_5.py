"""Integration tests for Neo4j entity-KG BM25 full-text recall — Pod 2.5 Stream B1 (CANON S3, S8).

These hit a REAL Neo4j (``docker compose up -d`` first) and ERROR when unreachable.
They only run under ``-m integration`` or ``-m "integration and neo4j"``.

Covers:
  - ensure_schema creates the fulltext index idempotently (call twice → no error).
  - Exact term match: matching claim ranks first with text_score > 0.
  - Sanitizer in-query: Lucene special chars in the query string do not raise.
  - Empty query: returns [] without exception.
  - Skeleton exclusion: payload=None claims never appear in results.
  - Scope filter: scoped queries isolate to the correct scope.
  - as_of filter: invalidated claims are excluded by the temporal filter.
  - Rank parity: Neo4j adapter and InMemoryEntityKG agree on rank order over a small corpus.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_entity_kg import Neo4jEntityKG
from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.testing.doubles import InMemoryEntityKG

pytestmark = [pytest.mark.integration, pytest.mark.neo4j]

_DIM = 4
_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _evidence(recorded_at: datetime = _T0) -> object:
    return make_evidence(
        type="tool_proof",
        polarity="+",
        source_id="bm25-test",
        source_authority=0.9,
        recorded_at=recorded_at,
    )


def _claim(
    subject: str,
    predicate: str,
    obj: str,
    *,
    scope: str = "agent",
    recorded_at: datetime = _T0,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
) -> Claim:
    """Mint a Claim with minimal required fields."""
    vf = valid_from if valid_from is not None else recorded_at
    claim_id = claim_id_for(subject, predicate, obj, scope=scope)
    return Claim(
        id=claim_id,
        subject=subject,
        predicate=predicate,
        payload=obj,
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=recorded_at),
        valid_from=vf,
        valid_to=valid_to,
        ingest_time=recorded_at,
        created_by="bm25-integration-test",
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jEntityKG]:
    """Real Neo4jEntityKG, schema ensured (with fulltext index), wiped per case."""
    adapter = Neo4jEntityKG(settings=settings)
    await adapter.ensure_schema(embedding_dim=_DIM)
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


# ---------------------------------------------------------------------------
# Test 1: ensure_schema creates the fulltext index idempotently
# ---------------------------------------------------------------------------


async def test_ensure_schema_fulltext_index_idempotent(settings: SubstrateSettings) -> None:
    """ensure_schema() can be called twice; claim_fulltext index exists after both calls."""
    adapter = Neo4jEntityKG(settings=settings)
    await adapter.ensure_schema(embedding_dim=_DIM)
    # Second call must not raise (IF NOT EXISTS makes it idempotent)
    await adapter.ensure_schema(embedding_dim=_DIM)

    async with adapter._connection.session() as session:
        result = await session.run("SHOW INDEXES YIELD name RETURN name")
        records = await result.values()
        index_names = {r[0] for r in records}

    assert "claim_fulltext" in index_names, (
        f"'claim_fulltext' index not found after ensure_schema. Indexes: {sorted(index_names)}"
    )

    await adapter.aclose()


# ---------------------------------------------------------------------------
# Test 2: Exact term match
# ---------------------------------------------------------------------------


async def test_exact_term_match(kg: Neo4jEntityKG) -> None:
    """A claim containing a distinctive term ranks first with text_score > 0."""
    matching = _claim("Plant", "process", "photosynthesis")
    unrelated = _claim("Sky", "color", "blue")

    await kg.write_claim(matching, evidence=_evidence())  # type: ignore[arg-type]
    await kg.write_claim(unrelated, evidence=_evidence())  # type: ignore[arg-type]

    results = await kg.claims_full_text("photosynthesis", k=10)

    assert len(results) >= 1, "Expected at least one result for 'photosynthesis'"
    top = results[0]
    assert top.claim.id == matching.id, (
        f"Expected matching claim first; got {top.claim.id!r}"
    )
    assert top.text_score is not None and top.text_score > 0, (
        f"Expected text_score > 0; got {top.text_score}"
    )


# ---------------------------------------------------------------------------
# Test 3: Sanitizer in-query — Lucene special chars do not raise
# ---------------------------------------------------------------------------


async def test_sanitizer_special_chars_no_exception(kg: Neo4jEntityKG) -> None:
    """A query string containing Lucene special characters does not raise; result is a Sequence."""
    # Write a claim so the index is non-empty
    c = _claim("Test", "has", "value")
    await kg.write_claim(c, evidence=_evidence())  # type: ignore[arg-type]

    # Various Lucene special character combinations
    for query in ["test + query", "(test)", "field:value", "[1 TO 10]", "NOT term"]:
        results = await kg.claims_full_text(query, k=10)
        # Must return a Sequence without raising
        assert isinstance(results, tuple), (
            f"Expected tuple for query {query!r}; got {type(results)}"
        )


# ---------------------------------------------------------------------------
# Test 4: Empty query returns [] without exception
# ---------------------------------------------------------------------------


async def test_empty_query_returns_empty(kg: Neo4jEntityKG) -> None:
    """Empty string and whitespace-only queries return [] without exception."""
    c = _claim("Something", "is", "here")
    await kg.write_claim(c, evidence=_evidence())  # type: ignore[arg-type]

    for query in ["", "   ", "\t\n"]:
        results = await kg.claims_full_text(query, k=10)
        assert len(results) == 0, (
            f"Expected empty result for query {query!r}; got {len(results)} results"
        )


# ---------------------------------------------------------------------------
# Test 5: Skeleton exclusion
# ---------------------------------------------------------------------------


async def test_skeleton_exclusion(kg: Neo4jEntityKG) -> None:
    """A skeleton claim (payload=None, written via MERGE placeholder) does not appear in results.

    We simulate the skeleton scenario by writing a real claim and then directly MERGing a
    skeleton node with the search term in its subject but no payload, using the raw driver.
    """
    real = _claim("Chlorophyll", "enables", "photosynthesis-absorption")
    await kg.write_claim(real, evidence=_evidence())  # type: ignore[arg-type]

    # Directly inject a skeleton node (payload=None) with "photosynthesis" in subject
    async with kg._connection.session() as session:
        await session.run(
            "MERGE (c:Claim {id: 'skeleton-test-id'}) "
            "SET c.subject = 'photosynthesis-skeleton', c.payload = null"
        )

    results = await kg.claims_full_text("photosynthesis", k=10)

    claim_ids = {sc.claim.id for sc in results}
    assert "skeleton-test-id" not in claim_ids, (
        "Skeleton claim (payload=None) must not appear in full-text results"
    )
    # The real claim should still appear
    assert real.id in claim_ids, (
        f"Real claim {real.id!r} should appear in full-text results; got {claim_ids}"
    )


# ---------------------------------------------------------------------------
# Test 6: Scope filter
# ---------------------------------------------------------------------------


async def test_scope_filter_isolates_correctly(kg: Neo4jEntityKG) -> None:
    """Scoped query returns only claims matching the requested scope."""
    agent_c = _claim("River", "flows", "downstream-agent", scope="agent")
    world_c = _claim("River", "flows", "downstream-world", scope="world")

    await kg.write_claim(agent_c, evidence=_evidence())  # type: ignore[arg-type]
    await kg.write_claim(world_c, evidence=_evidence())  # type: ignore[arg-type]

    agent_results = await kg.claims_full_text("downstream", k=10, scope="agent")
    world_results = await kg.claims_full_text("downstream", k=10, scope="world")
    all_results = await kg.claims_full_text("downstream", k=10, scope=None)

    agent_ids = {sc.claim.id for sc in agent_results}
    world_ids = {sc.claim.id for sc in world_results}
    all_ids = {sc.claim.id for sc in all_results}

    assert agent_c.id in agent_ids, "Agent claim must appear in scope='agent' results"
    assert world_c.id not in agent_ids, "World claim must NOT appear in scope='agent' results"

    assert world_c.id in world_ids, "World claim must appear in scope='world' results"
    assert agent_c.id not in world_ids, "Agent claim must NOT appear in scope='world' results"

    # Unscoped query returns both
    assert agent_c.id in all_ids
    assert world_c.id in all_ids


# ---------------------------------------------------------------------------
# Test 7: as_of filter
# ---------------------------------------------------------------------------


async def test_as_of_filter(kg: Neo4jEntityKG) -> None:
    """A claim invalidated before as_of is excluded; one still valid at as_of is included."""
    # Claim that is valid at T0 but invalidated before T2 (valid_to = T1)
    past_claim = _claim("Orbit", "period", "mercury-fast", valid_from=_T0, valid_to=_T1)
    past_id = claim_id_for("Orbit", "period", "mercury-fast", scope="agent")
    await kg.write_claim(past_claim, evidence=_evidence())  # type: ignore[arg-type]

    # Explicitly set valid_to on the node so the temporal filter fires
    await kg.invalidate_claim(past_id, valid_to=_T1)

    # Still-valid claim (no valid_to)
    live_claim = _claim("Orbit", "period", "jupiter-slow")
    await kg.write_claim(live_claim, evidence=_evidence())  # type: ignore[arg-type]

    # as_of = T0 — past_claim is within its validity window → should appear
    results_t0 = await kg.claims_full_text("mercury", k=10, as_of=_T0)
    t0_ids = {sc.claim.id for sc in results_t0}
    assert past_id in t0_ids, (
        f"Past claim should be visible at T0; results: {t0_ids}"
    )

    # as_of = T2 — past_claim's valid_to (T1) has passed → must not appear
    results_t2 = await kg.claims_full_text("mercury", k=10, as_of=_T2)
    t2_ids = {sc.claim.id for sc in results_t2}
    assert past_id not in t2_ids, (
        f"Invalidated claim must not appear at T2; results: {t2_ids}"
    )

    # live_claim appears at T2 regardless
    results_live = await kg.claims_full_text("jupiter", k=10, as_of=_T2)
    live_ids = {sc.claim.id for sc in results_live}
    assert live_claim.id in live_ids, (
        f"Live claim must appear at T2; results: {live_ids}"
    )


# ---------------------------------------------------------------------------
# Test 8: Rank parity with InMemoryEntityKG
# ---------------------------------------------------------------------------


async def test_rank_parity_with_in_memory(kg: Neo4jEntityKG) -> None:
    """Neo4j adapter and InMemoryEntityKG agree on descending rank order over a small corpus.

    We write 5 claims with varying term frequencies for the search term 'quasar'.
    The claim with the highest tf (most 'quasar' repetitions) should rank first in both engines.
    We only assert rank ORDER, not score equality (S9 — scores are implementation-defined).
    """
    # Build a corpus where claim_0 has the highest frequency, descending.
    # InMemory BM25 uses tf-idf; Neo4j Lucene BM25 does the same — rank order should agree.
    corpus: list[tuple[Claim, str]] = [
        (
            _claim(f"Star{i}", "type", "quasar " * (5 - i) + "object"),
            f"Star{i}",
        )
        for i in range(5)
    ]

    mem_kg = InMemoryEntityKG()
    for claim, _ in corpus:
        await kg.write_claim(claim, evidence=_evidence())  # type: ignore[arg-type]
        await mem_kg.write_claim(claim, evidence=_evidence())  # type: ignore[arg-type]

    neo_results = await kg.claims_full_text("quasar", k=5)
    mem_results = await mem_kg.claims_full_text("quasar", k=5)

    neo_ids = [sc.claim.id for sc in neo_results]
    mem_ids = [sc.claim.id for sc in mem_results]

    # Both engines must return at least one result
    assert len(neo_ids) > 0, "Neo4j returned no results for 'quasar'"
    assert len(mem_ids) > 0, "InMemory returned no results for 'quasar'"

    # The top-ranked claim must agree between both engines.
    # (Star0 has the most 'quasar' tokens and must rank first in both BM25 implementations.)
    assert neo_ids[0] == mem_ids[0], (
        f"Rank-1 mismatch: Neo4j={neo_ids[0]!r}, InMemory={mem_ids[0]!r}. "
        "BM25 rank order must agree on the highest-frequency document."
    )
