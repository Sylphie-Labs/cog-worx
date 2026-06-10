"""Live migration of a PRE-2.1 journal-steps table to the ``commit_xid`` projection ordering (S6).

The one load-bearing path with no other coverage: the standard ``journal`` fixture recreates the
table from ``_SCHEMA`` (which ALREADY has ``commit_xid``), so ``_MIGRATE_COMMIT_XID``'s ALTER never
runs anywhere else. This builds the genuine pre-2.1 shape (the Phase-1-gate steps table: NO
``commit_xid`` column, only the ``cogworx_journal_steps_order`` index), inserts committed steps,
then asserts ``ensure_schema()`` migrates it correctly AND that the projector drains the backlog
under the TIED-ORDINAL regime — every backfilled row shares the migration txn's single xid, so the
``(run_id, step_index)`` keyset walk (the ONLY thing that separates them) is exercised on real
Postgres. Marked ``integration``; ERRORs (not silently passes) when Timescale is unreachable.

The pre-2.1 rows carry MIXED-CASE ``run_id`` values so the ``COLLATE "C"`` byte-ordering tie-break
is exercised together with the tied ordinal — uppercase letters sort before lowercase under
codepoint order, which is exactly the cross-collation seam FIX 1 pinned.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

import psycopg
import pytest
from neo4j import AsyncManagedTransaction
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_procedural_kg import Neo4jProceduralKG
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.knowledge.procedural_registry import ProcedureRegistry
from cogworx.loop.result import Done, StageResult
from cogworx.loop.state import RunStatus
from cogworx.runtime.projector import DEFAULT_PROJECTION_CONSUMER, TrialProjector

pytestmark = pytest.mark.integration

_RESULT_ADAPTER: TypeAdapter[StageResult] = TypeAdapter(StageResult)

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_PATHWAY = "math_pathway"
_STAGE = "solve_stage"

# Mixed-case run_ids on purpose: under COLLATE "C" byte order, "RunA" (0x52) sorts BEFORE "runA"
# (0x72), so the tie-break path FIX 1 pinned is genuinely exercised when all rows share one xid.
_RUN_IDS = ("RunA", "runA", "RunB", "runB", "Run-C")
# 2 steps per run x 5 runs = 10 committed steps; with batch_limit below this the projector MUST page
# through the tied ordinals (every backfilled row shares the ALTER txn's single xid).
_STEPS_PER_RUN = 2
_TOTAL_STEPS = len(_RUN_IDS) * _STEPS_PER_RUN
_BATCH_LIMIT = 4

# The pre-2.1 (Phase-1-gate) steps table: NO commit_xid column, the run-grain order index only. We
# also create the intermediate-dev band index so the migration's DROP IF EXISTS is proven to fire.
_PRE_2_1_SCHEMA = """
DROP TABLE IF EXISTS cogworx_journal_steps CASCADE;

CREATE TABLE cogworx_journal_steps (
    run_id text NOT NULL,
    step_index bigint NOT NULL,
    stage_name text NOT NULL,
    result jsonb NOT NULL,
    committed_at timestamptz NOT NULL,
    CONSTRAINT cogworx_journal_steps_run_step UNIQUE (run_id, step_index)
);

CREATE INDEX IF NOT EXISTS cogworx_journal_steps_order
    ON cogworx_journal_steps (run_id, step_index);

CREATE INDEX IF NOT EXISTS cogworx_journal_steps_committed
    ON cogworx_journal_steps (committed_at, run_id, step_index);
"""


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # Async psycopg 3 cannot run on Windows' default ProactorEventLoop; it needs a selector loop.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    adapter = TimescaleJournal(settings=settings)
    # Ensure the auxiliary tables (runs/timers/attempts/human_inputs) exist + are clean, then
    # REPLACE the steps table with the pre-2.1 shape so the migration ALTER actually has work to do.
    await adapter.ensure_schema()
    await adapter.reset()
    conn = await adapter._connection()
    await conn.execute(_PRE_2_1_SCHEMA)
    try:
        yield adapter
    finally:
        await adapter.aclose()


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jProceduralKG]:
    adapter = Neo4jProceduralKG(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


def _registry() -> ProcedureRegistry:
    registry = ProcedureRegistry()
    registry.declare(_PATHWAY, _STAGE, problem_type="word problem")
    return registry


def _decl(registry: ProcedureRegistry) -> tuple[str, str]:
    decl = registry.get(_PATHWAY, _STAGE)
    assert decl is not None
    return decl.procedure_id, decl.problem_type


def _stamped_result(registry: ProcedureRegistry, outcome: str) -> StageResult:
    procedure_id, problem_type = _decl(registry)
    return Done(
        output=Artifact(
            kind="solution",
            produced_by=_STAGE,
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_T0),
            data={"outcome": outcome, "procedure_id": procedure_id, "problem_type": problem_type},
        )
    )


async def _insert_pre_2_1_steps(
    conn: psycopg.AsyncConnection[Any], registry: ProcedureRegistry
) -> set[str]:
    """INSERT committed steps the pre-2.1 way (no commit_xid). Return the expected trial_id set."""
    expected: set[str] = set()
    for run_id in _RUN_IDS:
        await conn.execute(
            "INSERT INTO cogworx_journal_runs "
            "(run_id, session_id, status, pathway_id, pathway_version, pathway_fingerprint) "
            "VALUES (%s, %s, %s, %s, %s, %s) ON CONFLICT (run_id) DO NOTHING",
            (run_id, f"session:{run_id}", RunStatus.RUNNING.value, _PATHWAY, 1, "fp"),
        )
        for step_index in range(_STEPS_PER_RUN):
            result = _stamped_result(registry, "success" if step_index == 0 else "failure")
            result_json = _RESULT_ADAPTER.dump_python(result, mode="json")
            await conn.execute(
                "INSERT INTO cogworx_journal_steps "
                "(run_id, step_index, stage_name, result, committed_at) "
                "VALUES (%s, %s, %s, %s, %s)",
                (run_id, step_index, _STAGE, Jsonb(result_json), _T0),
            )
            expected.add(f"{run_id}:{step_index}")
    return expected


async def _column_exists(conn: psycopg.AsyncConnection[Any], column: str) -> bool:
    cur = await conn.execute(
        "SELECT 1 FROM information_schema.columns "
        "WHERE table_name = 'cogworx_journal_steps' AND column_name = %s",
        (column,),
    )
    return await cur.fetchone() is not None


async def _index_exists(conn: psycopg.AsyncConnection[Any], index: str) -> bool:
    cur = await conn.execute(
        "SELECT 1 FROM pg_indexes WHERE tablename = 'cogworx_journal_steps' AND indexname = %s",
        (index,),
    )
    return await cur.fetchone() is not None


async def _distinct_commit_xids(conn: psycopg.AsyncConnection[Any]) -> list[int]:
    cur = await conn.execute("SELECT DISTINCT commit_xid FROM cogworx_journal_steps ORDER BY 1")
    return [row[0] for row in await cur.fetchall()]


async def test_migration_alters_backfills_and_reindexes(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    """(a)+(b)+(c): ALTER adds commit_xid backfilled to ONE shared xid, reindexes, and the projector
    drains the backlog to EXACT set-equality through the tied-ordinal keyset walk."""
    registry = _registry()
    conn = await journal._connection()
    expected_trials = await _insert_pre_2_1_steps(conn, registry)

    assert not await _column_exists(conn, "commit_xid")
    assert await _index_exists(conn, "cogworx_journal_steps_committed")
    assert not await _index_exists(conn, "cogworx_journal_steps_commit_xid")

    await journal.ensure_schema()

    assert await _column_exists(conn, "commit_xid")
    # The ALTER backfilled EVERY existing row to the migration txn's single xid (DEFAULT evaluated
    # once per row but inside ONE txn -> one xid). This is the tied-ordinal regime in the wild.
    shared_xids = await _distinct_commit_xids(conn)
    assert len(shared_xids) == 1, f"expected one shared backfill xid, got {shared_xids}"
    cur = await conn.execute("SELECT count(*) FROM cogworx_journal_steps")
    row = await cur.fetchone()
    assert row is not None and row[0] == _TOTAL_STEPS
    # New index present, old band index dropped by the migration.
    assert await _index_exists(conn, "cogworx_journal_steps_commit_xid")
    assert not await _index_exists(conn, "cogworx_journal_steps_committed")

    projector = TrialProjector(
        journal=journal, procedural_kg=kg, registry=registry, batch_limit=_BATCH_LIMIT
    )
    total = 0
    ticks = 0
    while True:
        projected = await projector.tick()
        if projected == 0:
            break
        total += projected
        ticks += 1
        assert ticks < _TOTAL_STEPS, "projector failed to make progress over tied ordinals"

    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    actual_trials = {t.trial_id for t in trials}
    # EXACT set-equality: every committed step -> exactly one Trial, zero missing, zero dup. Because
    # all rows share one ordinal, this is the (run_id, step_index) tie-break keyset walk on real PG.
    assert actual_trials == expected_trials
    assert len(trials) == _TOTAL_STEPS
    assert total == _TOTAL_STEPS
    # The paging proves the tie-break actually drove pagination (>1 tick to drain TOTAL > batch).
    assert ticks >= -(-_TOTAL_STEPS // _BATCH_LIMIT)


async def test_ensure_schema_is_idempotent_after_migration(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    """(d): re-running ensure_schema() after a migration is a no-op — no error, no double-migrate,
    the backfill xid stays a single value and the column/index set is unchanged."""
    registry = _registry()
    conn = await journal._connection()
    await _insert_pre_2_1_steps(conn, registry)

    await journal.ensure_schema()
    xids_first = await _distinct_commit_xids(conn)

    await journal.ensure_schema()
    await journal.ensure_schema()
    xids_again = await _distinct_commit_xids(conn)

    assert xids_first == xids_again
    assert len(xids_again) == 1
    assert await _column_exists(conn, "commit_xid")
    assert await _index_exists(conn, "cogworx_journal_steps_commit_xid")
    assert not await _index_exists(conn, "cogworx_journal_steps_committed")


async def test_old_cursor_without_commit_ordinal_cold_reprojects_idempotently(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    """(e): an old ``(:ProjectionCursor)`` lacking ``commit_ordinal`` -> ``read_cursor`` returns
    None -> cold re-projection is a no-op END-STATE (same Trial set; MERGE first-write-wins)."""
    registry = _registry()
    conn = await journal._connection()
    expected_trials = await _insert_pre_2_1_steps(conn, registry)
    await journal.ensure_schema()

    projector = TrialProjector(
        journal=journal, procedural_kg=kg, registry=registry, batch_limit=_BATCH_LIMIT
    )
    while await projector.tick():
        pass

    procedure_id, problem_type = _decl(registry)
    trials_before = {t.trial_id for t in await kg.trials_for(procedure_id, problem_type)}
    post_before = await kg.posterior(procedure_id, problem_type)
    assert trials_before == expected_trials

    # Simulate a pre-commit_xid cursor node: it has the consumer but NO commit_ordinal property.
    async def _orphan_cursor(tx: AsyncManagedTransaction) -> None:
        await tx.run(
            "MATCH (c:ProjectionCursor {consumer: $consumer}) "
            "REMOVE c.commit_ordinal, c.cursor_run_id, c.cursor_step_index",
            consumer=DEFAULT_PROJECTION_CONSUMER,
        )

    async with kg._connection.session() as session:
        await session.execute_write(_orphan_cursor)

    # read_cursor now returns None (commit_ordinal is gone) -> the next tick cold-reprojects.
    assert await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER) is None
    while await projector.tick():
        pass

    trials_after = {t.trial_id for t in await kg.trials_for(procedure_id, problem_type)}
    post_after = await kg.posterior(procedure_id, problem_type)
    # Cold re-projection converges to the SAME end-state: MERGE on trial_id is first-write-wins, so
    # re-walking every committed step adds nothing and changes no posterior (a4).
    assert trials_after == trials_before
    assert (post_after.alpha, post_after.beta, post_after.n_trials) == (
        post_before.alpha,
        post_before.beta,
        post_before.n_trials,
    )
