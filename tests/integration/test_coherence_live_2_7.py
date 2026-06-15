"""Integration tests for CoherenceReconciler over the live Neo4j substrate — Pod 2.7.

Requires a live Neo4j instance (``docker compose up -d``).
Marked ``@pytest.mark.integration``; skipped by default in CI.

Covers:
1. test_full_write_dirty_tick_graph_state: write 2 claims via Neo4jEntityKG, verify dirty mark
   created, tick the reconciler with TableOracle, verify dirty mark gone and adjudication exists.
2. test_crash_between_oracle_and_commit: a failing store wrapper that raises on
   commit_reconciliation → re-tick produces identical final graph state (idempotency via MERGE).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_entity_kg import Neo4jEntityKG
from cogworx.claims.provenance import Claim, Provenance
from cogworx.coherence.config import CoherenceConfig
from cogworx.coherence.reconciler import CoherenceReconciler
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.substrate.coherence import ReconciliationOutcome
from cogworx.testing.fake_oracle import TableOracle

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)

_DIM = 4


# ---------------------------------------------------------------------------
# Event loop policy (Windows psycopg 3 / Neo4j driver requires selector loop)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Per-test fixture: fresh Neo4jEntityKG
# ---------------------------------------------------------------------------


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jEntityKG]:
    """Real Neo4jEntityKG, schema ensured, wiped per case."""
    adapter = Neo4jEntityKG(settings=settings)
    await adapter.ensure_schema(embedding_dim=_DIM)
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    valid_from: datetime = _T0,
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope="agent")
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="observation",
        provenance=_prov(),
        valid_from=valid_from,
        ingest_time=valid_from,
        created_by="test",
        scope="agent",
    )


def _ev(event_id: str, *, source_id: str = "src1") -> object:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source_id,
        source_authority=0.8,
        recorded_at=_T0,
        event_id=event_id,
    )


# ---------------------------------------------------------------------------
# IL-1: full write → dirty → tick → graph clean
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_full_write_dirty_tick_graph_state(kg: Neo4jEntityKG) -> None:
    """Write 2 claims → dirty mark exists → tick reconciler → dirty gone, adjudication present."""
    claim_a = _make_claim("IntegAlice", "status", "active", valid_from=_T0)
    claim_b = _make_claim("IntegAlice", "status", "inactive", valid_from=_T1)
    await kg.write_claim(claim_a, evidence=_ev("ev-a"))  # type: ignore[arg-type]
    await kg.write_claim(claim_b, evidence=_ev("ev-b"))  # type: ignore[arg-type]

    # Dirty mark should exist.
    dirty_before = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty_before) >= 1, "Expected at least one dirty subject after writes"

    oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
    config = CoherenceConfig(batch_limit=4)
    rec = CoherenceReconciler(
        entity_kg=kg,
        store=kg,  # type: ignore[arg-type]  # Neo4jEntityKG missing current_epistemic_level (CF)
        oracle=oracle,
        config=config,
        now=lambda: _T2,
    )

    stats = await rec.tick()

    assert stats.subjects_processed >= 1
    assert stats.subjects_failed == 0

    # Dirty mark should be cleared.
    dirty_after = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty_after) == 0, f"Expected dirty cleared after tick, got {dirty_after}"

    # At least one adjudication should have been written.
    from cogworx.knowledge.identity import normalize_topic_part

    adj_ids = await kg.adjudication_ids_for_subject("agent", normalize_topic_part("IntegAlice"))
    assert len(adj_ids) >= 1, "Expected at least one adjudication node after reconciliation"


# ---------------------------------------------------------------------------
# IL-2: crash between oracle and commit → re-tick idempotent
# ---------------------------------------------------------------------------


class _FailOnFirstCommitKG(Neo4jEntityKG):
    """Subclass that raises on the first commit_reconciliation call, then succeeds."""

    def __init__(self, settings: SubstrateSettings) -> None:
        super().__init__(settings=settings)
        self._commit_calls: int = 0

    async def commit_reconciliation(
        self,
        outcome: ReconciliationOutcome,
        *,
        dirty_key: str,
        observed_epoch: int,
    ) -> None:
        self._commit_calls += 1
        if self._commit_calls == 1:
            raise RuntimeError("Simulated crash before commit")
        await super().commit_reconciliation(
            outcome, dirty_key=dirty_key, observed_epoch=observed_epoch
        )


@pytest.mark.asyncio
async def test_crash_between_oracle_and_commit(settings: SubstrateSettings) -> None:
    """Crash on first commit_reconciliation → second tick resolves idempotently."""
    adapter = Neo4jEntityKG(settings=settings)
    await adapter.ensure_schema(embedding_dim=_DIM)
    await adapter.reset()

    try:
        claim_a = _make_claim("IntegBob", "score", "high", valid_from=_T0)
        claim_b = _make_claim("IntegBob", "score", "low", valid_from=_T1)
        await adapter.write_claim(claim_a, evidence=_ev("ev-c"))  # type: ignore[arg-type]
        await adapter.write_claim(claim_b, evidence=_ev("ev-d"))  # type: ignore[arg-type]

        # Build a wrapper that fails on first commit.
        fail_kg = _FailOnFirstCommitKG(settings=settings)
        # Ensure schema (no-op — adapter already did it) and share the same graph data.
        await fail_kg.ensure_schema(embedding_dim=_DIM)
        # The fail_kg points at the same Neo4j instance (same settings), same data.

        oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
        config = CoherenceConfig(batch_limit=4)
        rec = CoherenceReconciler(
            entity_kg=fail_kg,
            store=fail_kg,  # type: ignore[arg-type]  # CF: Neo4jEntityKG missing current_epistemic_level
            oracle=oracle,
            config=config,
            now=lambda: _T2,
        )

        # First tick: crash on commit → subject_failed=1, dirty mark NOT cleared (attempts bumped).
        stats1 = await rec.tick()
        assert stats1.subjects_failed == 1

        # Second tick: commit succeeds → adjudication written.
        stats2 = await rec.tick()
        assert stats2.subjects_failed == 0
        assert stats2.subjects_processed >= 1

        # Dirty queue is clear after second tick.
        dirty_final = await fail_kg.claim_dirty_subjects(limit=10)
        assert len(dirty_final) == 0, f"Expected dirty cleared after second tick, got {dirty_final}"

        # Verify at least one adjudication node.
        from cogworx.knowledge.identity import normalize_topic_part

        adj_ids = await fail_kg.adjudication_ids_for_subject(
            "agent", normalize_topic_part("IntegBob")
        )
        assert len(adj_ids) >= 1

        # Third tick: second reconciliation of the same adjudication is idempotent (MERGE).
        # Re-mark dirty so the reconciler has something to process.
        await adapter.write_claim(claim_a, evidence=_ev("ev-e", source_id="src-extra"))  # type: ignore[arg-type]
        stats3 = await rec.tick()
        assert stats3.subjects_failed == 0

        # Adjudication count must not have grown — MERGE is idempotent.
        adj_ids_after = await fail_kg.adjudication_ids_for_subject(
            "agent", normalize_topic_part("IntegBob")
        )
        assert set(adj_ids_after) == set(adj_ids), (
            f"Adjudication ids must not grow on idempotent re-tick. "
            f"Before={set(adj_ids)}, after={set(adj_ids_after)}"
        )

    finally:
        await adapter.aclose()
        await fail_kg.aclose()
