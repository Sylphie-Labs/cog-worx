"""Pod 2.1 commit_xid projection-ordering spike (CANON S12) — the falsifiable RACE tests.

The structural fix (mythos, Jim option a): the trial projector orders committed steps by the
DB-assigned ``commit_xid`` and reads with a VISIBILITY FENCE
(``commit_xid < pg_snapshot_xmin(pg_current_snapshot())``) so it never consumes a row an in-flight
txn could still slot below the cursor. The sequence-visibility race is only constructible on LIVE
Postgres (two real txns, one held open mid-commit), so these are integration-tier and ERROR — not
silently pass — when the cluster is unreachable.

Each claim carries a BUG-INJECTION NEGATIVE CONTROL (the standing red-team rule): the unfenced
predicate (``TimescaleJournal._read_projection(..., fenced=False)``) is shown to consume a row a
held txn still sits below, advance the cursor past it, and PERMANENTLY lose that row once the held
txn commits — proving the fence is load-bearing, not decorative.

  S1 — race falsifier + unfenced negative control.
  S2 — xid/commit-order inversions, both directions, zero loss.
  S3 — DO-NOTHING replays interleaved with real commits: exactly-one-Trial-per-step (a1),
       first-write payloads (a5), no stall at burned-xid gaps; + the static-grep + runtime probe
       that no statement names ``commit_xid``.
  S4 — crash mid-tick (between read and project_batch): re-run -> byte-identical (a4/a6).
  S5 — soak (slow): N writers, each commit inside a txn held open 0-200ms; at quiescence assert SET
       EQUALITY between committed procedure steps and Trials; cursor monotone throughout.
  S6 — honest cost: an unrelated open write txn DELAYS (does not lose) a step; it lands at txn end.
"""

from __future__ import annotations

import asyncio
import random
import sys
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import psycopg
import pytest
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
from cogworx.substrate.journal import ProjectionCursor, StepRecord

_RESULT_ADAPTER: TypeAdapter[StageResult] = TypeAdapter(StageResult)

pytestmark = [pytest.mark.spike, pytest.mark.integration]

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_PATHWAY = "math_pathway"
_STAGE = "solve_stage"


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # Async psycopg 3 cannot run on Windows' default ProactorEventLoop; it needs a selector loop.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
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


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _stamped(registry: ProcedureRegistry) -> Artifact:
    procedure_id, problem_type = _decl(registry)
    return Artifact(
        kind="solution",
        produced_by=_STAGE,
        provenance=_prov(),
        data={"outcome": "success", "procedure_id": procedure_id, "problem_type": problem_type},
    )


async def _start(journal: TimescaleJournal, run_id: str) -> None:
    await journal.start_run(
        run_id,
        f"session:{run_id}",
        pathway_id=_PATHWAY,
        pathway_version=1,
        pathway_fingerprint="fp",
    )


def _step(registry: ProcedureRegistry, run_id: str, step_index: int = 0) -> StepRecord:
    return StepRecord(
        run_id=run_id,
        step_index=step_index,
        stage_name=_STAGE,
        result=Done(output=_stamped(registry)),
        committed_at=_T0,
    )


def _projector(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, registry: ProcedureRegistry
) -> TrialProjector:
    return TrialProjector(journal=journal, procedural_kg=kg, registry=registry)


async def _insert_in_open_txn(
    dsn: str, registry: ProcedureRegistry, run_id: str
) -> psycopg.AsyncConnection[Any]:
    """Open a NON-autocommit connection, INSERT a journal step, and HOLD the txn open (no commit).

    Forces xid assignment by reading ``pg_current_xact_id()`` inside the txn before the INSERT, so
    the row's ``commit_xid`` DEFAULT lands on a real (in-flight) xid and the held txn pins the
    snapshot ``xmin`` at that xid for any concurrent reader. The returned connection is the caller's
    to commit or roll back.
    """
    conn: psycopg.AsyncConnection[Any] = await psycopg.AsyncConnection.connect(
        dsn, autocommit=False
    )
    await conn.execute(
        "INSERT INTO cogworx_journal_runs "
        "(run_id, session_id, status, pathway_id, pathway_version, "
        "pathway_fingerprint) VALUES (%s, %s, %s, %s, %s, %s) "
        "ON CONFLICT (run_id) DO NOTHING",
        (run_id, f"session:{run_id}", RunStatus.RUNNING.value, _PATHWAY, 1, "fp"),
    )
    await conn.execute("SELECT pg_current_xact_id()")
    result_json = _RESULT_ADAPTER.dump_python(Done(output=_stamped(registry)), mode="json")
    await conn.execute(
        "INSERT INTO cogworx_journal_steps "
        "(run_id, step_index, stage_name, result, committed_at) VALUES (%s, %s, %s, %s, %s) "
        "ON CONFLICT (run_id, step_index) DO NOTHING",
        (run_id, 0, _STAGE, Jsonb(result_json), _T0),
    )
    return conn


# ---------------------------------------------------------------------------
# S1 — the race falsifier + the unfenced negative control
# ---------------------------------------------------------------------------


async def test_s1_fence_holds_back_row_below_an_in_flight_txn(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    registry = _registry()
    # Conn A: explicit txn inserts (runA, 0) and holds open (its xid pins the snapshot xmin).
    conn_a = await _insert_in_open_txn(settings.pg_dsn, registry, "runA")
    try:
        # Conn B (the journal's autocommit conn): commit (runB, 0) — a HIGHER, committed xid.
        await _start(journal, "runB")
        await journal.commit_step(_step(registry, "runB"))

        # A tick now: B's row is committed but its commit_xid is >= the open A txn's xmin, so the
        # fence holds it back. A's row is uncommitted -> invisible. ZERO rows; cursor unchanged.
        projected = await _projector(journal, kg, registry).tick()
        assert projected == 0
        assert await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER) is None
    finally:
        await conn_a.commit()
        await conn_a.close()

    # A committed -> both rows are now below xmin: the next tick projects both, exactly once.
    projected = await _projector(journal, kg, registry).tick()
    assert projected == 2
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert sorted(t.trial_id for t in trials) == ["runA:0", "runB:0"]


async def test_s1_negative_control_unfenced_read_permanently_loses_held_row(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    # NEGATIVE CONTROL: the UNFENCED predicate consumes B (the higher, committed xid) while A is
    # still open below it, advances the cursor past A's xid, and then PERMANENTLY loses A once A
    # commits —
    # because the cursor is now strictly above A's keyset. Proves the fence is load-bearing.
    registry = _registry()
    conn_a = await _insert_in_open_txn(settings.pg_dsn, registry, "runA")
    a_committed = False
    try:
        await _start(journal, "runB")
        await journal.commit_step(_step(registry, "runB"))

        # UNFENCED read sees B (committed) and advances the cursor to B's commit_xid — past A's xid.
        unfenced = await journal._read_projection(None, limit=256, fenced=True)  # sanity: fenced=0
        assert unfenced == ()
        unfenced = await journal._read_projection(None, limit=256, fenced=False)
        assert [ps.record.run_id for ps in unfenced] == ["runB"]
        lost_cursor = ProjectionCursor(
            commit_ordinal=unfenced[-1].commit_ordinal,
            run_id=unfenced[-1].record.run_id,
            step_index=unfenced[-1].record.step_index,
        )
    finally:
        await conn_a.commit()
        a_committed = True
        await conn_a.close()
    assert a_committed

    # A is committed but its commit_xid is BELOW the advanced cursor -> the strict keyset excludes
    # it forever. The harness detects the loss: a forward read from lost_cursor never returns runA.
    after = await journal.committed_steps_after(lost_cursor, limit=256)
    assert "runA" not in [ps.record.run_id for ps in after]


# ---------------------------------------------------------------------------
# S2 — xid/commit-order inversions, both directions, zero loss
# ---------------------------------------------------------------------------


async def test_s2_inversion_held_takes_xid_early_then_b_commits(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    # A takes its xid EARLY (pg_current_xact_id before B exists), B commits with a higher xid first.
    # A still sits below xmin until it commits; nothing is consumed until A drains. Zero loss.
    registry = _registry()
    conn_a = await _insert_in_open_txn(settings.pg_dsn, registry, "runA")
    try:
        await _start(journal, "runB")
        await journal.commit_step(_step(registry, "runB"))
        assert await _projector(journal, kg, registry).tick() == 0
    finally:
        await conn_a.commit()
        await conn_a.close()
    await _projector(journal, kg, registry).tick()
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert sorted(t.trial_id for t in trials) == ["runA:0", "runB:0"]


async def test_s2_inversion_held_inserts_after_b_commits(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    # The other direction: B commits FIRST (fully visible, below any future xmin), THEN a held txn
    # opens and inserts A with a HIGHER xid. B is consumed immediately (correct — it is below the
    # cursor-less horizon); A is held back by its own in-flight xid and lands only after it commits.
    # The invariant is ZERO LOSS, not a stall: B projected now, A projected after commit, each once.
    registry = _registry()
    await _start(journal, "runB")
    await journal.commit_step(_step(registry, "runB"))
    conn_a = await _insert_in_open_txn(settings.pg_dsn, registry, "runA")
    try:
        # B lands now; A is still in-flight so it is NOT consumed (and cannot be lost — the cursor
        # only advanced to B, which is below A's xid).
        assert await _projector(journal, kg, registry).tick() == 1
        procedure_id, problem_type = _decl(registry)
        assert sorted(t.trial_id for t in await kg.trials_for(procedure_id, problem_type)) == [
            "runB:0"
        ]
    finally:
        await conn_a.commit()
        await conn_a.close()
    await _projector(journal, kg, registry).tick()
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert sorted(t.trial_id for t in trials) == ["runA:0", "runB:0"]


# ---------------------------------------------------------------------------
# S3 — DO-NOTHING replays, exactly-once, no stall at burned-xid gaps, a5 static + runtime probe
# ---------------------------------------------------------------------------


async def test_s3_replays_interleaved_exactly_one_trial_per_step(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    # Interleave N DO-NOTHING replayed commits (idempotent re-commits of the SAME step) between real
    # commits. The replays burn xids (ON CONFLICT DO NOTHING still advances the xid counter) -> gaps
    # in the commit_xid sequence. Assert exactly-one-Trial-per-committed-step (a1) and no stall at
    # the burned-xid gaps (the cursor walks straight over them).
    registry = _registry()
    reals = ["r0", "r1", "r2", "r3"]
    for i, run_id in enumerate(reals):
        await _start(journal, run_id)
        await journal.commit_step(_step(registry, run_id))
        # Replay the SAME step several times (DO NOTHING) and re-commit an earlier one — burns xids.
        for _ in range(3):
            await journal.commit_step(_step(registry, run_id))
        if i > 0:
            await journal.commit_step(_step(registry, reals[i - 1]))

    total = 0
    for _ in range(10):
        total += await _projector(journal, kg, registry).tick()
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert sorted(t.trial_id for t in trials) == [f"{r}:0" for r in reals]
    assert total == len(reals)  # exactly one Trial per committed step, no stall at xid gaps


async def test_s3_a5_no_statement_names_commit_xid_static_grep() -> None:
    # a5 STATIC: no INSERT/UPDATE statement may NAME commit_xid — it is DB-self-populated only. Scan
    # the adapter source; commit_xid may appear ONLY in DDL (CREATE/ALTER ... DEFAULT), the read
    # SELECT/WHERE/ORDER BY (projection), and the index. It must NEVER appear in an INSERT column
    # list or an UPDATE SET.
    src = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "cogworx"
        / "adapters"
        / "timescale_journal.py"
    )
    text = src.read_text(encoding="utf-8")
    lowered = text.lower()
    # Crude but mutation-resistant: there is exactly one INSERT into cogworx_journal_steps, and its
    # column list must not contain commit_xid.
    insert_marker = "insert into cogworx_journal_steps"
    start = lowered.index(insert_marker)
    column_list = lowered[start : start + 200]
    assert "commit_xid" not in column_list, (
        "commit_xid named in the journal-steps INSERT column list (a5 violation): it must "
        "self-populate via the DB DEFAULT, never be written by a statement."
    )
    assert "set commit_xid" not in lowered, "commit_xid named in an UPDATE SET (a5 violation)."


async def test_s3_a5_runtime_probe_commit_step_self_populates_commit_xid(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    # a5 RUNTIME: commit_step writes ZERO columns for commit_xid, yet the row lands with a non-null,
    # positive commit_xid (the DEFAULT fired inside the writing txn). Probe the raw column.
    registry = _registry()
    await _start(journal, "runX")
    await journal.commit_step(_step(registry, "runX"))
    conn: psycopg.AsyncConnection[Any] = await psycopg.AsyncConnection.connect(
        settings.pg_dsn, autocommit=True
    )
    try:
        cur = await conn.execute(
            "SELECT commit_xid FROM cogworx_journal_steps WHERE run_id = %s", ("runX",)
        )
        row = await cur.fetchone()
    finally:
        await conn.close()
    assert row is not None
    assert isinstance(row[0], int) and row[0] > 0


# ---------------------------------------------------------------------------
# S4 — crash mid-tick (between read and project_batch): re-run -> byte-identical (a4/a6)
# ---------------------------------------------------------------------------


async def test_s4_crash_between_read_and_project_batch_is_byte_identical(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    registry = _registry()
    for run_id in ("c0", "c1", "c2"):
        await _start(journal, run_id)
        await journal.commit_step(_step(registry, run_id))

    # Simulate a crash AFTER the journal read but BEFORE project_batch: read, then discard (no-op).
    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    scanned = await journal.committed_steps_after(cursor, limit=256)
    assert len(scanned) == 3
    # Crash here: nothing was projected, the cursor never advanced.
    assert await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER) is None

    # Re-run from scratch: the projector re-reads the same rows and projects them idempotently.
    projector = _projector(journal, kg, registry)
    await projector.tick()
    procedure_id, problem_type = _decl(registry)
    post_a = await kg.posterior(procedure_id, problem_type)
    trials_a = sorted(t.trial_id for t in await kg.trials_for(procedure_id, problem_type))
    cursor_a = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)

    await projector.tick()
    await projector.tick()
    post_b = await kg.posterior(procedure_id, problem_type)
    trials_b = sorted(t.trial_id for t in await kg.trials_for(procedure_id, problem_type))
    cursor_b = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)

    assert trials_a == trials_b == ["c0:0", "c1:0", "c2:0"]
    assert (post_a.alpha, post_a.beta, post_a.n_trials) == (
        post_b.alpha,
        post_b.beta,
        post_b.n_trials,
    )
    assert cursor_a == cursor_b


# ---------------------------------------------------------------------------
# S5 — soak (slow): writers commit inside txns held open 0-200ms; SET EQUALITY at quiescence
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.parametrize("seed", [1, 7, 42])
async def test_s5_soak_set_equality_and_monotone_cursor(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings, seed: int
) -> None:
    registry = _registry()
    rng = random.Random(seed)
    writers, steps = 8, 50
    expected: set[str] = set()

    async def _writer(w: int) -> None:
        conn: psycopg.AsyncConnection[Any] = await psycopg.AsyncConnection.connect(
            settings.pg_dsn, autocommit=False
        )
        try:
            for s in range(steps):
                run_id = f"s{seed}w{w}r{s}"
                await conn.execute(
                    "INSERT INTO cogworx_journal_runs (run_id, session_id, status, pathway_id, "
                    "pathway_version, pathway_fingerprint) VALUES (%s,%s,%s,%s,%s,%s) "
                    "ON CONFLICT (run_id) DO NOTHING",
                    (run_id, f"sess:{run_id}", RunStatus.RUNNING.value, _PATHWAY, 1, "fp"),
                )
                result_json = _RESULT_ADAPTER.dump_python(
                    Done(output=_stamped(registry)), mode="json"
                )
                await conn.execute(
                    "INSERT INTO cogworx_journal_steps "
                    "(run_id, step_index, stage_name, result, committed_at) "
                    "VALUES (%s,%s,%s,%s,%s) ON CONFLICT (run_id, step_index) DO NOTHING",
                    (run_id, 0, _STAGE, Jsonb(result_json), _T0),
                )
                expected.add(f"{run_id}:0")
                await asyncio.sleep(rng.uniform(0.0, 0.2))
                await conn.commit()
        finally:
            await conn.close()

    last_ordinal = 0

    async def _ticker() -> None:
        nonlocal last_ordinal
        # The ticker drives the projector on its OWN autocommit journal conn (NOT the fixture conn,
        # NOT DDL — the schema already exists, and a concurrent ALTER would block on the open
        # writer txns). Read-only ticks never lock against the writers.
        tick_journal = TimescaleJournal(settings=settings)
        projector = TrialProjector(
            journal=tick_journal, procedural_kg=kg, registry=registry, batch_limit=7
        )
        try:
            for _ in range(400):
                await projector.tick()
                cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
                if cursor is not None:
                    assert cursor.commit_ordinal >= last_ordinal  # monotone throughout
                    last_ordinal = cursor.commit_ordinal
                await asyncio.sleep(0.01)
        finally:
            await tick_journal.aclose()

    await asyncio.gather(*[_writer(w) for w in range(writers)], _ticker())

    # Drain any tail after the writers quiesce.
    drain = _projector(journal, kg, registry)
    for _ in range(50):
        if await drain.tick() == 0:
            break

    procedure_id, problem_type = _decl(registry)
    trials = {t.trial_id for t in await kg.trials_for(procedure_id, problem_type)}
    assert trials == expected  # SET EQUALITY: zero missing, zero dup
    assert len(expected) == writers * steps


# ---------------------------------------------------------------------------
# S6 — honest cost: an unrelated open write txn DELAYS (does not lose) a step
# ---------------------------------------------------------------------------


async def test_s6_unrelated_open_txn_delays_not_loses(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, settings: SubstrateSettings
) -> None:
    # An unrelated open write txn on a SCRATCH table pins xmin low. A step committed AFTER it is
    # DELAYED behind the horizon (fenced) until the unrelated txn ends, then it lands. This pins
    # the horizon-stall latency mode (the honest cost of the fence): availability, not
    # correctness, is what gives.
    registry = _registry()
    scratch = f"cogworx_spike_scratch_{uuid.uuid4().hex[:8]}"
    blocker: psycopg.AsyncConnection[Any] = await psycopg.AsyncConnection.connect(
        settings.pg_dsn, autocommit=False
    )
    try:
        await blocker.execute(f"CREATE TABLE IF NOT EXISTS {scratch} (id bigint)")
        await blocker.commit()
        await blocker.execute("SELECT pg_current_xact_id()")
        await blocker.execute(f"INSERT INTO {scratch} (id) VALUES (1)")  # txn now open, xmin pinned

        await _start(journal, "delayed")
        await journal.commit_step(_step(registry, "delayed"))
        # Fenced behind the blocker's xmin -> DELAYED, not lost.
        assert await _projector(journal, kg, registry).tick() == 0
        assert await kg.get_trial("delayed:0") is None
    finally:
        await blocker.commit()  # blocker ends -> horizon advances
        await blocker.execute(f"DROP TABLE IF EXISTS {scratch}")
        await blocker.commit()
        await blocker.close()

    # Once the unrelated txn ends, the delayed step LANDS.
    await _projector(journal, kg, registry).tick()
    assert await kg.get_trial("delayed:0") is not None
