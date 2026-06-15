"""Integration tests for Neo4j entity-KG scope support — Pod 2.4 (CANON S3, S5, S7).

These hit a REAL Neo4j (``docker compose up -d`` first) and ERROR when unreachable.
They only run under ``-m integration`` or ``-m "integration and neo4j"``.

Covers:
  - ensure_schema creates scope indexes idempotently (run twice → no error).
  - World-scope and agent-scope claims are correctly isolated on scoped reads.
  - Shared entity: world claim + user claim both on "Paris" → exactly one :Entity {name:"Paris"}.
  - Evidence segregation: same triple in agent vs world scope → independent evidence lists.
  - Migration-free: raw write_claim (no scope kwarg → default "agent") visible in scope="agent"
    and scope=None reads; invisible in scope="world".
  - ScopedKG factory functions (world_model / user_model) wire correctly to Neo4jEntityKG.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_entity_kg import Neo4jEntityKG
from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.scoped_kg import user_model, world_model
from cogworx.knowledge.scopes import ScopeRegistry
from cogworx.knowledge.source_registry import SourceDeclaration, SourceRegistry

pytestmark = [pytest.mark.integration, pytest.mark.neo4j]

_DIM = 4  # embedding dimension matching other integration tests
_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _source(
    kind: str = "tool", ref: str = "scope-test", authority: float = 0.9
) -> SourceDeclaration:
    return SourceRegistry().declare(kind, ref, authority=authority)  # type: ignore[arg-type]


def _evidence(
    source: SourceDeclaration | None = None, recorded_at: datetime = _T0
) -> EvidenceEvent:
    src = source or _source()
    return make_evidence(
        type="tool_proof",
        polarity="+",
        source_id=src.source_id,
        source_authority=src.source_authority,
        recorded_at=recorded_at,
    )


def _bare_claim(
    subject: str,
    predicate: str,
    obj: str,
    *,
    scope: str = "agent",
    recorded_at: datetime = _T0,
) -> Claim:
    """Mint a Claim directly (no ScopedKG) — used for migration-free tests."""
    claim_id = claim_id_for(subject, predicate, obj, scope=scope)
    return Claim(
        id=claim_id,
        subject=subject,
        predicate=predicate,
        payload=obj,
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=recorded_at),
        valid_from=recorded_at,
        ingest_time=recorded_at,
        created_by="integration-test",
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jEntityKG]:
    """Real Neo4jEntityKG, schema ensured (with scope indexes), wiped per case."""
    adapter = Neo4jEntityKG(settings=settings)
    await adapter.ensure_schema(embedding_dim=_DIM)
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


# ---------------------------------------------------------------------------
# Test: ensure_schema creates scope indexes idempotently
# ---------------------------------------------------------------------------


async def test_ensure_schema_scope_indexes_idempotent(settings: SubstrateSettings) -> None:
    """ensure_schema() can be called twice without error; scope indexes are created both times."""
    adapter = Neo4jEntityKG(settings=settings)
    # First call: create indexes
    await adapter.ensure_schema(embedding_dim=_DIM)
    # Second call: idempotent (IF NOT EXISTS clauses)
    await adapter.ensure_schema(embedding_dim=_DIM)

    # Verify the scope indexes exist by querying Neo4j's index listing
    assert adapter._driver is not None
    async with adapter._driver.session() as session:
        result = await session.run("SHOW INDEXES YIELD name RETURN name")
        records = await result.values()
        index_names = {r[0] for r in records}

    assert "claim_scope" in index_names, (
        f"'claim_scope' index not found after ensure_schema. Indexes: {sorted(index_names)}"
    )
    assert "claim_scope_subject" in index_names, (
        f"'claim_scope_subject' index not found after ensure_schema. Indexes: {sorted(index_names)}"
    )

    await adapter.aclose()


# ---------------------------------------------------------------------------
# Test: World-scope and agent-scope isolation
# ---------------------------------------------------------------------------


async def test_scope_isolation_world_vs_agent(kg: Neo4jEntityKG) -> None:
    """World-scope and agent-scope claims on the same entity are correctly isolated in reads."""
    source = _source()

    # Write world-scope claim
    world_claim = _bare_claim("Paris", "capital_of", "France", scope="world")
    await kg.write_claim(world_claim, evidence=_evidence(source))

    # Write agent-scope claim (same subject, different predicate/object to avoid id collision)
    agent_claim = _bare_claim("Paris", "factoid", "famous", scope="agent")
    await kg.write_claim(agent_claim, evidence=_evidence(source))

    # Scoped reads
    world_hits = await kg.claims_about("Paris", scope="world")
    agent_hits = await kg.claims_about("Paris", scope="agent")
    all_hits = await kg.claims_about("Paris", scope=None)

    assert len(world_hits) == 1, f"Expected 1 world claim, got {len(world_hits)}"
    assert len(agent_hits) == 1, f"Expected 1 agent claim, got {len(agent_hits)}"
    assert len(all_hits) == 2, f"Expected 2 total claims, got {len(all_hits)}"

    assert world_hits[0].claim.scope == "world"
    assert agent_hits[0].claim.scope == "agent"


# ---------------------------------------------------------------------------
# Test: Shared entity — world + user claims both on "Paris" → one :Entity node
# ---------------------------------------------------------------------------


async def test_shared_entity_single_node(kg: Neo4jEntityKG) -> None:
    """World claim and user claim both on 'Paris' result in exactly one :Entity {name:'Paris'}."""
    source = _source()

    world_claim = _bare_claim("Paris", "capital_of", "France", scope="world")
    user_claim = _bare_claim("Paris", "home_city_of", "jim", scope="user:jim")

    await kg.write_claim(world_claim, evidence=_evidence(source))
    await kg.write_claim(user_claim, evidence=_evidence(source))

    # Count :Entity nodes with name="Paris" — must be exactly 1 (MERGE semantics)
    assert kg._driver is not None
    async with kg._driver.session() as session:
        result = await session.run(
            "MATCH (e:Entity {name: $name}) RETURN count(e) AS cnt",
            name="Paris",
        )
        record = await result.single()

    assert record is not None
    entity_count = record["cnt"]
    assert entity_count == 1, (
        f"Expected exactly 1 :Entity {{name:'Paris'}}, found {entity_count}. "
        "MERGE should deduplicate entities across scopes."
    )

    # Both claims still appear in the unscoped read
    all_hits = await kg.claims_about("Paris", scope=None)
    claim_ids = {sc.claim.id for sc in all_hits}
    assert world_claim.id in claim_ids
    assert user_claim.id in claim_ids


# ---------------------------------------------------------------------------
# Test: Evidence segregation — agent vs world claims have independent evidence lists
# ---------------------------------------------------------------------------


async def test_evidence_segregation_agent_vs_world(kg: Neo4jEntityKG) -> None:
    """Agent-scope and world-scope claims for the same triple have independent evidence lists.

    11 evidence events on the agent claim; the world claim must have exactly 1.
    """
    source = _source()

    agent_claim = _bare_claim("X", "is", "Y", scope="agent")
    world_claim = _bare_claim("X", "is", "Y", scope="world")

    # Write both (different IDs due to scope-in-identity)
    assert agent_claim.id != world_claim.id

    await kg.write_claim(agent_claim, evidence=_evidence(source))  # 1 event
    await kg.write_claim(world_claim, evidence=_evidence(source))  # 1 event

    # Add 10 more events to agent claim only
    for _ in range(10):
        ev = make_evidence(
            type="corroboration",
            polarity="+",
            source_id=source.source_id,
            source_authority=source.source_authority,
            recorded_at=_T0,
        )
        await kg.add_evidence(agent_claim.id, ev)

    agent_ev = await kg.evidence_for(agent_claim.id)
    world_ev = await kg.evidence_for(world_claim.id)

    assert len(agent_ev) == 11, f"Agent claim should have 11 events, got {len(agent_ev)}"
    assert len(world_ev) == 1, f"World claim should have exactly 1 event, got {len(world_ev)}"


# ---------------------------------------------------------------------------
# Test: Migration-free — raw write_claim with default scope="agent"
# ---------------------------------------------------------------------------


async def test_migration_free_default_scope(kg: Neo4jEntityKG) -> None:
    """A claim written via raw write_claim (no explicit scope → default 'agent') appears in
    scope='agent' and scope=None reads, but NOT in scope='world' reads."""
    source = _source()

    # Write claim without explicit scope kwarg — uses Claim's default scope="agent"
    default_claim = Claim(
        id=claim_id_for("Migration", "test", "value"),  # no scope arg → DEFAULT_SCOPE
        subject="Migration",
        predicate="test",
        payload="value",
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_T0),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="migration-test",
        # scope field uses default "agent" from Claim model
    )
    await kg.write_claim(default_claim, evidence=_evidence(source))

    # Must appear in scope="agent" read
    agent_hits = await kg.claims_about("Migration", scope="agent")
    assert len(agent_hits) == 1, (
        f"Default-scope claim must appear in scope='agent' reads; got {len(agent_hits)}"
    )
    assert agent_hits[0].claim.scope == "agent"

    # Must appear in unscoped read (scope=None)
    unscoped_hits = await kg.claims_about("Migration", scope=None)
    assert len(unscoped_hits) == 1

    # Must NOT appear in scope="world" read
    world_hits = await kg.claims_about("Migration", scope="world")
    assert len(world_hits) == 0, (
        f"Default-scope claim must NOT appear in scope='world' reads; got {len(world_hits)}"
    )


# ---------------------------------------------------------------------------
# Test: ScopedKG factory functions wire correctly to Neo4jEntityKG
# ---------------------------------------------------------------------------


async def test_scoped_kg_world_model_wires_to_neo4j(kg: Neo4jEntityKG) -> None:
    """world_model() wired to a real Neo4jEntityKG writes and reads correctly."""
    registry = ScopeRegistry()
    source = _source()
    wm = world_model(kg, registry, owner="neo4j-agent")

    claim_id = await wm.assert_fact(
        subject="Tokyo",
        predicate="capital_of",
        obj="Japan",
        epistemic_type="confirmed",
        source=source,
        created_by="neo4j-agent",
    )

    # Verify via ScopedKG view
    fetched = await wm.get_claim(claim_id)
    assert fetched is not None
    assert fetched.scope == "world"
    assert fetched.subject == "Tokyo"

    # Verify scope isolation: agent-scope view returns None for this claim
    agent_result = await kg.get_claim(claim_id)
    assert agent_result is not None  # Raw KG always returns the claim
    assert agent_result.scope == "world"

    # claims_about via ScopedKG filters to world only
    world_hits = await wm.claims_about("Tokyo")
    assert len(world_hits) == 1
    assert world_hits[0].claim.scope == "world"


async def test_scoped_kg_user_model_wires_to_neo4j(kg: Neo4jEntityKG) -> None:
    """user_model() wired to a real Neo4jEntityKG writes and reads correctly."""
    registry = ScopeRegistry()
    source = _source()
    um = user_model(kg, registry, "alice", owner="neo4j-agent")

    claim_id = await um.assert_fact(
        subject="Alice",
        predicate="lives_in",
        obj="Berlin",
        epistemic_type="inference",
        source=source,
        created_by="neo4j-agent",
    )

    fetched = await um.get_claim(claim_id)
    assert fetched is not None
    assert fetched.scope == "user:alice"

    # World model view cannot see this claim
    wm = world_model(kg, registry, owner="neo4j-agent-w")
    assert await wm.get_claim(claim_id) is None

    # User claims_about is scope-filtered
    user_hits = await um.claims_about("Alice")
    assert len(user_hits) == 1
    assert user_hits[0].claim.scope == "user:alice"


async def test_scoped_kg_cross_scope_operations_raise_on_neo4j(kg: Neo4jEntityKG) -> None:
    """add_evidence and invalidate on out-of-scope claims raise ValueError against Neo4j."""
    registry = ScopeRegistry()
    source = _source()

    wm = world_model(kg, registry, owner="neo4j-agent-w")
    um = user_model(kg, registry, "bob", owner="neo4j-agent-b")

    world_id = await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="neo4j-agent-w",
    )

    # add_evidence from user model → ValueError
    ev = make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source.source_id,
        source_authority=source.source_authority,
        recorded_at=_T0,
    )
    with pytest.raises(ValueError):
        await um.add_evidence(world_id, ev)

    # invalidate from user model → ValueError
    with pytest.raises(ValueError):
        await um.invalidate(world_id)
