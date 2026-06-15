"""Phase 3 gate live integration tests — durable taint on real Postgres/TimescaleDB.

Requires the docker-compose Postgres/TimescaleDB stack (``docker compose up -d``).
No model API key needed — these tests make zero model calls.

SC-13: Durable taint bit survives engine death (verified from a fresh connection after the taint).
SC-14: set_run_tainted contract — idempotent, unknown-run silent, ensure_schema idempotent.
SC-15: Approval HITL round-trip on real Postgres across an engine death.
       Proves taint rehydration from the DB into a fresh Engine instance's gate (B1 durable taint),
       the unapproved-probe-blocked falsifier (if rehydration failed the probe would silently pass),
       exactly-once side effects across the engine death (S6), first-answer-wins idempotency, and
       the H3 crash-window resume path (answer written + RUNNING status set, process died, cold
       resume picks up and completes correctly).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import sys
from collections.abc import AsyncIterator, Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.capability.base import PermissionTier
from cogworx.capability.policy import ApprovalRequired, StageToolPolicy
from cogworx.capability.registry import Registry
from cogworx.capability.router import dispatch_one
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import AwaitHuman, Degraded, Done, Transition
from cogworx.loop.state import RunStatus
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.context import RunContext
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import RunState
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel

pytestmark = pytest.mark.integration


# ---------------------------------------------------------------------------
# Event-loop policy (Windows psycopg 3 requires the selector loop)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Per-test fixture: fresh TimescaleJournal with schema ensured and data wiped
# ---------------------------------------------------------------------------


@pytest.fixture
async def journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    """A ``TimescaleJournal`` with schema ensured and all tables wiped per test case."""
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


async def _start(
    j: TimescaleJournal,
    run_id: str,
    *,
    session_id: str = "test-sess",
    pathway_id: str = "p",
    pathway_version: int = 1,
    pathway_fingerprint: str = "fp",
) -> None:
    await j.start_run(
        run_id,
        session_id,
        pathway_id=pathway_id,
        pathway_version=pathway_version,
        pathway_fingerprint=pathway_fingerprint,
    )


# ===========================================================================
# SC-13 — Durable taint bit survives engine death (fresh connection)
# ===========================================================================


async def test_sc13_durable_taint_survives_engine_swap(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """Taint bit persists across a fresh TimescaleJournal connection (engine-death simulation).

    This is the core SC-13 invariant: the taint bit must be in Postgres, not only in
    journal_A's in-process connection state.  A fresh journal_B on the same DSN must read
    tainted=True from the database.
    """
    await _start(journal, "sc13-r1")
    await journal.set_run_tainted("sc13-r1")

    # Simulate engine death: a brand-new journal instance opens its own connection.
    journal_b = TimescaleJournal(settings=settings)
    try:
        state: RunState | None = await journal_b.load_run("sc13-r1")
        assert state is not None, "sc13-r1 must exist in the database"
        assert state.tainted is True, (
            "SC-13: taint bit must be True when read from a fresh journal connection"
        )
        assert state.run_id == "sc13-r1"
        assert state.status is RunStatus.RUNNING
    finally:
        await journal_b.aclose()


async def test_sc13_negative_control_untainted_run(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """A run that was never tainted loads with tainted=False from a fresh connection.

    Negative control: confirms the taint bit is not True by default, so SC-13 positive
    test is meaningful.
    """
    await _start(journal, "sc13-clean")
    # Deliberately do NOT call set_run_tainted.

    journal_b = TimescaleJournal(settings=settings)
    try:
        state = await journal_b.load_run("sc13-clean")
        assert state is not None, "sc13-clean must exist in the database"
        assert state.tainted is False, (
            "SC-13 negative: an untainted run must load with tainted=False"
        )
    finally:
        await journal_b.aclose()


async def test_sc13_raw_sql_probe(journal: TimescaleJournal, settings: SubstrateSettings) -> None:
    """Direct SQL query confirms the tainted column is True without going through load_run.

    This eliminates any possibility that load_run is returning a Python-constructed value
    rather than the actual database column.  The psycopg connection here is opened
    independently of the journal — it is a raw substrate probe.
    """
    await _start(journal, "sc13-raw")
    await journal.set_run_tainted("sc13-raw")

    conn = await psycopg.AsyncConnection.connect(settings.pg_dsn, autocommit=True)
    try:
        cursor = await conn.execute(
            "SELECT tainted FROM cogworx_journal_runs WHERE run_id = %s",
            ("sc13-raw",),
        )
        row = await cursor.fetchone()
        assert row is not None, "sc13-raw must have a row in cogworx_journal_runs"
        assert row[0] is True, (
            "SC-13 raw SQL: tainted column must be boolean True in Postgres, not a Python object"
        )
    finally:
        await conn.close()


async def test_sc13_taint_does_not_bleed_across_runs(
    journal: TimescaleJournal,
) -> None:
    """Tainting one run does not affect a sibling run in the same schema.

    Regression guard: verifies that set_run_tainted is keyed by run_id and does not
    perform a table-wide UPDATE.
    """
    await _start(journal, "sc13-tainted-run")
    await _start(journal, "sc13-clean-run")

    await journal.set_run_tainted("sc13-tainted-run")
    # sc13-clean-run is never tainted.

    tainted_state = await journal.load_run("sc13-tainted-run")
    clean_state = await journal.load_run("sc13-clean-run")

    assert tainted_state is not None
    assert tainted_state.tainted is True, "the tainted run must have tainted=True"

    assert clean_state is not None
    assert clean_state.tainted is False, (
        "SC-13 bleed: the sibling run must NOT be tainted — set_run_tainted must be keyed by run_id"
    )


# ===========================================================================
# SC-14 — set_run_tainted contract on live Postgres
# ===========================================================================


async def test_sc14_set_run_tainted_idempotent(journal: TimescaleJournal) -> None:
    """Calling set_run_tainted twice raises no error; the bit stays True after both calls.

    The contract: set_run_tainted is a monotonic False->True flip.  Calling it a second
    time must be a silent no-op at the database row level, not an error.
    """
    await _start(journal, "sc14-idem")
    await journal.set_run_tainted("sc14-idem")
    await journal.set_run_tainted("sc14-idem")  # second call — must not raise

    state = await journal.load_run("sc14-idem")
    assert state is not None
    assert state.tainted is True, (
        "SC-14 idempotent: tainted must still be True after a duplicate set_run_tainted call"
    )


async def test_sc14_unknown_run_id_silent(journal: TimescaleJournal) -> None:
    """set_run_tainted on an unknown run_id is silent — updates 0 rows, raises no exception.

    The TimescaleJournal contract (see substrate/journal.py Journal.set_run_tainted docstring):
    'If the run is unknown this silently does nothing.'
    """
    # This must complete without raising any exception.
    await journal.set_run_tainted("run-that-does-not-exist-sc14")

    # Confirm nothing was inserted by accident.
    state = await journal.load_run("run-that-does-not-exist-sc14")
    assert state is None, (
        "SC-14 unknown: set_run_tainted must not create a row for an unknown run_id"
    )


async def test_sc14_migration_adds_column_idempotently(
    settings: SubstrateSettings,
) -> None:
    """ensure_schema on an existing schema is idempotent — calling it twice raises no error.

    This pins the 'ALTER TABLE ADD COLUMN IF NOT EXISTS' idiom in _MIGRATE_TAINTED: a second
    call must be a no-op, not an 'column already exists' error.
    """
    j = TimescaleJournal(settings=settings)
    await j.ensure_schema()
    await j.reset()
    await j.ensure_schema()  # second ensure_schema on the same tables — must not raise
    await j.ensure_schema()  # third call for extra confidence

    # After three ensure_schema calls, the tainted column must still be queryable.
    await _start(j, "sc14-migrate-probe")
    await j.set_run_tainted("sc14-migrate-probe")
    state = await j.load_run("sc14-migrate-probe")
    assert state is not None
    assert state.tainted is True, (
        "SC-14 migration: tainted column must be functional after repeated ensure_schema calls"
    )

    await j.aclose()


async def test_sc14_tainted_column_present_in_schema(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """The tainted column exists in cogworx_journal_runs and has the correct default.

    Probes information_schema to confirm the column was created by the migration and
    defaults to false, so pre-existing rows are safely backfilled.
    """
    conn = await psycopg.AsyncConnection.connect(settings.pg_dsn, autocommit=True)
    try:
        cursor = await conn.execute(
            "SELECT column_name, data_type, column_default "
            "FROM information_schema.columns "
            "WHERE table_schema = 'public' "
            "  AND table_name = 'cogworx_journal_runs' "
            "  AND column_name = 'tainted'",
        )
        row = await cursor.fetchone()
        assert row is not None, "SC-14 schema: 'tainted' column must exist in cogworx_journal_runs"
        column_name: str = row[0]
        data_type: str = row[1]
        column_default: str | None = row[2]

        assert column_name == "tainted", "column name must be 'tainted'"
        assert data_type == "boolean", (
            f"SC-14 schema: tainted must be a boolean column, got {data_type!r}"
        )
        assert column_default is not None, (
            "SC-14 schema: tainted must have a DEFAULT so pre-existing rows backfill to false"
        )
        assert "false" in column_default.lower(), (
            f"SC-14 schema: tainted DEFAULT must be false, got {column_default!r}"
        )
    finally:
        await conn.close()


async def test_sc14_untainted_run_has_false_in_db(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """A freshly started run has tainted=false in the raw Postgres row.

    Pins the DEFAULT false on the column: before any set_run_tainted call, the DB row
    must carry boolean false (not NULL), so there is no three-valued-logic ambiguity.
    """
    await _start(journal, "sc14-default-false")

    conn = await psycopg.AsyncConnection.connect(settings.pg_dsn, autocommit=True)
    try:
        cursor = await conn.execute(
            "SELECT tainted FROM cogworx_journal_runs WHERE run_id = %s",
            ("sc14-default-false",),
        )
        row = await cursor.fetchone()
        assert row is not None, "sc14-default-false must have a row in cogworx_journal_runs"
        assert row[0] is False, (
            "SC-14 default: a freshly inserted run must have tainted=False (not NULL) in Postgres"
        )
    finally:
        await conn.close()


# ===========================================================================
# SC-15 — Approval HITL round-trip on real Postgres across an engine death
# ===========================================================================
#
# Pathway (3 stages):
#   taint_stage (step 0): dispatches fetch_ext (external) → latches taint → Transition to act
#   act_stage   (step 1): catches ApprovalRequired from send → AwaitHuman (parks AWAITING_HUMAN)
#   finish_stage(step 2): reads human input from step 1; before dispatch_approved, fires an
#                          unapproved probe to falsify taint rehydration; routes on decision
#
# The falsifier: if the engine failed to seed tainted=True from the DB into the fresh gate,
# dispatch_one(..., approved=False) on the trifecta cap would NOT raise ApprovalRequired — the
# gate would be in a clean state and send would fire. The probe_blocked flag in the Done output
# carries this falsification result as a first-class assertion target.
#
# The four tests cover:
#   SC-15-A (approve)  — full two-engine sequence, send fires exactly once, taint rehydrated
#   SC-15-B (deny)     — full two-engine sequence, send never fires, taint rehydrated
#   SC-15-C (dup)      — duplicate answer, first-answer-wins, terminal state unchanged
#   SC-15-D (H3 crash) — answer written + RUNNING set manually, cold resume completes correctly


_SC15_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
_SC15_EMPTY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _sc15_counter_clock() -> Any:
    """An injected clock that advances 1s per call — avoids datetime.now() (S6)."""
    state = {"tick": 0}

    def clock() -> datetime:
        moment = _SC15_EPOCH + timedelta(seconds=state["tick"])
        state["tick"] += 1
        return moment

    return clock


def _sc15_prov(source: str = "system") -> Provenance:
    return Provenance(source=source, confidence=1.0, recorded_at=_SC15_EPOCH)


def _sc15_artifact(kind: str = "step", **data: Any) -> Artifact:
    return Artifact(kind=kind, produced_by="test", provenance=_sc15_prov(), data=dict(data))


def _sc15_initial_artifact() -> Artifact:
    return _sc15_artifact("start")


# ---------------------------------------------------------------------------
# Stage definitions for SC-15
# ---------------------------------------------------------------------------


class _SC15TaintStage:
    """Step 0: dispatches fetch_ext (external) to latch taint -> Transition to act."""

    name = "taint"
    transitions: tuple[str, ...] = ("act",)
    tool_policy = StageToolPolicy(
        allowed_tiers=frozenset({"external"}),
        taint_drops_external=False,
    )

    def __init__(self, reg: Registry) -> None:
        self._reg = reg

    async def run(self, ctx: Any) -> Transition:
        assert ctx._gate is not None
        await dispatch_one(ctx._gate, self._reg, "fetch_ext", {}, approved=False)
        return Transition(to="act", output=_sc15_artifact("tainted"))


class _SC15ActStage:
    """Step 1: catches ApprovalRequired from send -> parks AWAITING_HUMAN."""

    name = "act"
    transitions: tuple[str, ...] = ("finish",)
    tool_policy = StageToolPolicy(
        allowed_tiers=frozenset({"external"}),
        taint_drops_external=False,
    )

    def __init__(self, reg: Registry) -> None:
        self._reg = reg

    async def run(self, ctx: Any) -> AwaitHuman:
        assert ctx._gate is not None
        with contextlib.suppress(ApprovalRequired):
            await dispatch_one(ctx._gate, self._reg, "send", {}, approved=False)
        return AwaitHuman(
            question="Approve the send action?",
            to="finish",
            output=_sc15_artifact("awaiting"),
        )


class _SC15FinishStage:
    """Step 2: reads human input, fires unapproved probe (falsifier), routes on decision.

    The probe (dispatch_one with approved=False on the trifecta cap) is the taint-rehydration
    falsifier: if the engine loaded tainted=False from DB instead of True, the probe would NOT
    raise ApprovalRequired, send_cap.invoke would be called (call_count++ before dispatch_approved),
    and probe_blocked would be False. Any of those conditions failing is a rehydration bug.
    """

    name = "finish"
    transitions: tuple[str, ...] = ()
    # Must allow external tier so dispatch_approved("send") passes check_dispatch.
    tool_policy = StageToolPolicy(
        allowed_tiers=frozenset({"external"}),
        taint_drops_external=False,
    )

    def __init__(self, reg_ref_holder: list[Registry | None]) -> None:
        self._reg_ref_holder = reg_ref_holder

    async def run(self, ctx: Any) -> Done | Degraded:
        # step 1 is where act_stage committed its AwaitHuman.
        answer = await ctx.read_human_input(1)
        if answer is None:
            return Degraded(reason="no human input", output=_sc15_artifact("no-input"))
        payload = answer.data or {}
        decision: str = str(payload.get("decision", "no"))

        # Unapproved probe — the taint-rehydration falsifier.
        # If the fresh gate was not seeded tainted=True from the DB, this dispatch would NOT
        # raise ApprovalRequired (the gate would be clean) and probe_blocked would be False.
        probe_blocked = False
        reg = self._reg_ref_holder[0]
        assert reg is not None, "reg_ref_holder must be populated before engine runs"
        assert ctx._gate is not None
        try:
            await dispatch_one(ctx._gate, reg, "send", {}, approved=False)
        except ApprovalRequired:
            probe_blocked = True

        if decision == "yes":
            assert isinstance(ctx, RunContext)
            await ctx.dispatch_approved("send", {})
            return Done(
                output=_sc15_artifact("approved", probe_blocked=probe_blocked, decision=decision)
            )
        else:
            return Degraded(
                reason="human denied",
                output=_sc15_artifact("denied", probe_blocked=probe_blocked, decision=decision),
            )


# ---------------------------------------------------------------------------
# Factory helpers
# ---------------------------------------------------------------------------


def _build_sc15_fixtures(
    reg_ref_holder: list[Registry | None],
) -> tuple[Registry, PathwayRegistry, Any, Any]:
    """Build fresh SC-15 fixtures.

    ``reg_ref_holder[0]`` is populated with the registry so the finish stage's unapproved probe
    can reference it.  Build fresh per engine to avoid cross-contamination between engines A and B
    (each engine needs its own stage instances and call counters).

    Returns (reg, pathways, send_cap, fetch_cap).
    """

    class _FetchExtCap:
        name = "fetch_ext"
        tier: PermissionTier = "external"
        description = "fetch external data"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _SC15_EMPTY_SCHEMA
            self.call_count = 0

        async def invoke(self, args: Any) -> Any:
            self.call_count += 1
            return "external-payload"

    class _SendCap:
        name = "send"
        tier: PermissionTier = "external"
        description = "send msg"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _SC15_EMPTY_SCHEMA
            self.call_count = 0

        async def invoke(self, args: Any) -> Any:
            self.call_count += 1
            return "sent"

    fetch_cap = _FetchExtCap()
    send_cap = _SendCap()
    reg = Registry()
    reg.register(fetch_cap, tags=())
    reg.register(send_cap, tags=("consequential", "irreversible"))

    # Populate the holder so the finish stage can access the registry for its probe.
    reg_ref_holder[0] = reg

    taint_stage = _SC15TaintStage(reg)
    act_stage = _SC15ActStage(reg)
    finish_stage = _SC15FinishStage(reg_ref_holder)

    graph = StageGraph([taint_stage, act_stage, finish_stage], entry="taint")
    pathways = PathwayRegistry()
    pathways.register("sc15-path", graph)

    return reg, pathways, send_cap, fetch_cap


def _build_sc15_engine(
    journal: TimescaleJournal,
    reg: Registry,
    pathways: PathwayRegistry,
    *,
    clock: Any,
) -> Engine:
    """Build a model-free engine using ReplayModel (zero responses needed -- no model calls)."""
    model_reg = ModelRegistry()
    model_reg.register("default", ReplayModel([]))
    return Engine(
        models=model_reg,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        registry=reg,
        pathways=pathways,
        clock=clock,
    )


# ---------------------------------------------------------------------------
# SC-15-A: Approve path -- full two-engine HITL sequence
# ---------------------------------------------------------------------------


async def test_sc15_approve_cold_engine_full_hitl_sequence(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """Full engine-level approval HITL on real Postgres across an engine death.

    SC-15-A invariants:
    - Engine A parks at AwaitHuman; taint bit is in the DB row (raw SQL probe).
    - Engine B (fresh connections, fresh stage instances, same DSN) resumes with decision='yes'.
    - Taint rehydrated from DB into fresh gate: unapproved probe raises ApprovalRequired
      (probe_blocked=True in the finish-stage output).
    - dispatch_approved fires send exactly once (send_cap_b.call_count == 1).
    - Step 0 replayed from journal -- fetch_cap_b.call_count == 0 (S6 exactly-once).
    - HITL answer row exists in cogworx_human_inputs with decision='yes'.
    - Zero model calls end-to-end (ReplayModel with no scripted responses).
    """
    run_id = "sc15-approve"

    # --- Engine A: run until AwaitHuman ---
    holder_a: list[Registry | None] = [None]
    reg_a, pathways_a, send_cap_a, fetch_cap_a = _build_sc15_fixtures(holder_a)
    engine_a = _build_sc15_engine(journal, reg_a, pathways_a, clock=_sc15_counter_clock())

    await engine_a.run(
        run_id=run_id,
        session_id="sc15-sess-a",
        pathway_id="sc15-path",
        initial=_sc15_initial_artifact(),
    )

    # Verify parked state from DB (via load_run).
    parked: RunState | None = await journal.load_run(run_id)
    assert parked is not None, "sc15-approve must exist in the database after engine A"
    assert parked.status is RunStatus.AWAITING_HUMAN, (
        f"SC-15-A: engine A must park at AWAITING_HUMAN; got {parked.status!r}"
    )
    assert parked.tainted is True, (
        "SC-15-A: taint bit must be True after dispatching fetch_ext in taint_stage"
    )
    assert len(parked.steps) == 2, (
        f"SC-15-A: engine A must commit exactly 2 steps (taint + act); got {len(parked.steps)}"
    )
    names_a = [s.stage_name for s in parked.steps]
    assert names_a == ["taint", "act"], (
        f"SC-15-A: stage sequence must be ['taint', 'act']; got {names_a}"
    )

    # Raw SQL probe: confirm the tainted column is boolean True in Postgres -- not a Python-side
    # reconstruction.  This closes the gap where load_run could return a Python-default.
    conn_probe = await psycopg.AsyncConnection.connect(settings.pg_dsn, autocommit=True)
    try:
        raw_cursor = await conn_probe.execute(
            "SELECT tainted FROM cogworx_journal_runs WHERE run_id = %s",
            (run_id,),
        )
        raw_row = await raw_cursor.fetchone()
        assert raw_row is not None, "SC-15-A raw SQL: run row must exist"
        assert raw_row[0] is True, (
            "SC-15-A raw SQL: tainted column must be boolean True in Postgres"
        )
    finally:
        await conn_probe.close()

    # Verify engine A side-effect counts before simulated death.
    assert fetch_cap_a.call_count == 1, (
        f"SC-15-A: fetch_ext must be invoked once by engine A; got {fetch_cap_a.call_count}"
    )
    assert send_cap_a.call_count == 0, (
        "SC-15-A: send must NOT be invoked by engine A (ApprovalRequired caught); "
        f"got {send_cap_a.call_count}"
    )

    # --- Engine A "dies": build Engine B from scratch ---
    holder_b: list[Registry | None] = [None]
    reg_b, pathways_b, send_cap_b, fetch_cap_b = _build_sc15_fixtures(holder_b)
    journal_b = TimescaleJournal(settings=settings)
    await journal_b.ensure_schema()
    try:
        engine_b = _build_sc15_engine(journal_b, reg_b, pathways_b, clock=_sc15_counter_clock())

        # Provide human approval (payload is a dict; engine wraps it in an Artifact internally).
        final: RunState = await engine_b.provide_human_input(run_id, payload={"decision": "yes"})
    finally:
        await journal_b.aclose()

    # Engine state sequence.
    assert final.status is RunStatus.COMPLETED, (
        f"SC-15-A: approve path must reach COMPLETED; got {final.status!r}"
    )
    assert len(final.steps) == 3, (
        f"SC-15-A: must have 3 committed steps (taint + act + finish); got {len(final.steps)}"
    )
    names_b = [s.stage_name for s in final.steps]
    assert names_b == ["taint", "act", "finish"], (
        f"SC-15-A: stage sequence must be ['taint', 'act', 'finish']; got {names_b}"
    )
    assert final.tainted is True, "SC-15-A: tainted must remain True in the final RunState"

    # Taint rehydration falsifier: the unapproved probe in the finish stage was blocked.
    finish_result = final.steps[2].result
    assert finish_result.kind == "done", (
        f"SC-15-A: finish stage must produce Done; got {finish_result.kind!r}"
    )
    assert finish_result.output is not None
    assert finish_result.output.data.get("probe_blocked") is True, (
        "SC-15-A taint-rehydration falsifier: probe_blocked must be True -- "
        "if False, the fresh engine gate was NOT seeded tainted=True from the DB"
    )

    # Exactly-once side effects across the engine death (S6).
    assert send_cap_b.call_count == 1, (
        "SC-15-A: send must be invoked exactly once by engine B (dispatch_approved); "
        f"got {send_cap_b.call_count}"
    )
    assert fetch_cap_b.call_count == 0, (
        "SC-15-A: fetch_ext must NOT be re-invoked by engine B (step 0 replayed S6); "
        f"got {fetch_cap_b.call_count}"
    )

    # HITL answer row in cogworx_human_inputs.
    conn_hitl = await psycopg.AsyncConnection.connect(settings.pg_dsn, autocommit=True)
    try:
        hitl_cursor = await conn_hitl.execute(
            "SELECT answer FROM cogworx_human_inputs WHERE run_id = %s AND step_index = %s",
            (run_id, 1),
        )
        hitl_row = await hitl_cursor.fetchone()
        assert hitl_row is not None, (
            "SC-15-A: cogworx_human_inputs must have a row for (sc15-approve, step 1)"
        )
        hitl_data: Any = hitl_row[0]
        if isinstance(hitl_data, str):
            hitl_data = json.loads(hitl_data)
        # The engine stores payload={"decision": "yes"} in Artifact.data.
        assert hitl_data.get("data", {}).get("decision") == "yes", (
            f"SC-15-A: persisted HITL answer must have decision='yes'; got {hitl_data!r}"
        )
    finally:
        await conn_hitl.close()


# ---------------------------------------------------------------------------
# SC-15-B: Deny path -- taint rehydrated, send never fires
# ---------------------------------------------------------------------------


async def test_sc15_deny_cold_engine_degraded_send_never_fires(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """Human denies -> run DEGRADED, send never fires, taint rehydration falsifier holds.

    SC-15-B invariants:
    - Same two-engine structure as SC-15-A.
    - decision='no' -> finish_stage returns Degraded.
    - send_cap_b.call_count == 0 (dispatch_approved is never called on deny path).
    - probe_blocked is True (taint was rehydrated from DB into engine B's fresh gate).
    - Final status DEGRADED, tainted=True.
    """
    run_id = "sc15-deny"

    # Engine A: park at AwaitHuman.
    holder_a: list[Registry | None] = [None]
    reg_a, pathways_a, send_cap_a, fetch_cap_a = _build_sc15_fixtures(holder_a)
    engine_a = _build_sc15_engine(journal, reg_a, pathways_a, clock=_sc15_counter_clock())

    await engine_a.run(
        run_id=run_id,
        session_id="sc15-sess-deny",
        pathway_id="sc15-path",
        initial=_sc15_initial_artifact(),
    )

    parked: RunState | None = await journal.load_run(run_id)
    assert parked is not None
    assert parked.status is RunStatus.AWAITING_HUMAN
    assert parked.tainted is True

    assert fetch_cap_a.call_count == 1
    assert send_cap_a.call_count == 0

    # Engine A "dies": build Engine B from scratch.
    holder_b: list[Registry | None] = [None]
    reg_b, pathways_b, send_cap_b, fetch_cap_b = _build_sc15_fixtures(holder_b)
    journal_b = TimescaleJournal(settings=settings)
    await journal_b.ensure_schema()
    try:
        engine_b = _build_sc15_engine(journal_b, reg_b, pathways_b, clock=_sc15_counter_clock())
        final: RunState = await engine_b.provide_human_input(run_id, payload={"decision": "no"})
    finally:
        await journal_b.aclose()

    assert final.status is RunStatus.DEGRADED, (
        f"SC-15-B: deny path must reach DEGRADED; got {final.status!r}"
    )
    assert final.tainted is True, "SC-15-B: tainted must remain True after deny"

    # send must never fire on the deny path.
    assert send_cap_b.call_count == 0, (
        f"SC-15-B: send must NOT be invoked on deny path; got {send_cap_b.call_count}"
    )
    assert fetch_cap_b.call_count == 0, (
        "SC-15-B: fetch_ext must NOT be re-invoked (step 0 replayed S6); "
        f"got {fetch_cap_b.call_count}"
    )

    # Taint rehydration falsifier: probe_blocked must be True even on deny.
    finish_result = final.steps[2].result
    assert finish_result.kind == "degraded", (
        f"SC-15-B: finish stage must produce Degraded on deny; got {finish_result.kind!r}"
    )
    assert finish_result.output.data.get("probe_blocked") is True, (
        "SC-15-B taint-rehydration falsifier: probe_blocked must be True on deny path -- "
        "taint must be rehydrated from DB into engine B's fresh gate"
    )

    # Verify from fresh load_run that the run is permanently DEGRADED.
    reloaded: RunState | None = await journal.load_run(run_id)
    assert reloaded is not None
    assert reloaded.status is RunStatus.DEGRADED
    assert reloaded.tainted is True


# ---------------------------------------------------------------------------
# SC-15-C: Duplicate answer -- first-answer-wins, terminal state unchanged
# ---------------------------------------------------------------------------


async def test_sc15_duplicate_answer_first_wins_terminal_noop(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """Duplicate provide_human_input on a completed run is a silent no-op (first-answer-wins).

    SC-15-C invariants:
    - Engine A parks at AwaitHuman.
    - Engine B approves ('yes') -> run COMPLETED, send fires once.
    - Engine C (fresh, same DSN) calls provide_human_input with decision='no'.
    - provide_human_input on a COMPLETED run returns the current state unchanged (no-op).
    - send_cap_c.call_count == 0 (no re-execution).
    - read_human_input returns the FIRST answer ('yes'), not the duplicate 'no'.
    """
    run_id = "sc15-dup"

    # Engine A: park.
    holder_a: list[Registry | None] = [None]
    reg_a, pathways_a, _send_a, _fetch_a = _build_sc15_fixtures(holder_a)
    engine_a = _build_sc15_engine(journal, reg_a, pathways_a, clock=_sc15_counter_clock())

    await engine_a.run(
        run_id=run_id,
        session_id="sc15-sess-dup",
        pathway_id="sc15-path",
        initial=_sc15_initial_artifact(),
    )

    parked: RunState | None = await journal.load_run(run_id)
    assert parked is not None and parked.status is RunStatus.AWAITING_HUMAN

    # Engine B: approve -- first answer.
    holder_b: list[Registry | None] = [None]
    reg_b, pathways_b, send_cap_b, _fetch_b = _build_sc15_fixtures(holder_b)
    journal_b = TimescaleJournal(settings=settings)
    await journal_b.ensure_schema()
    try:
        engine_b = _build_sc15_engine(journal_b, reg_b, pathways_b, clock=_sc15_counter_clock())
        final_b: RunState = await engine_b.provide_human_input(run_id, payload={"decision": "yes"})
    finally:
        await journal_b.aclose()

    assert final_b.status is RunStatus.COMPLETED, (
        f"SC-15-C: first provide_human_input must reach COMPLETED; got {final_b.status!r}"
    )
    assert send_cap_b.call_count == 1

    # Engine C: duplicate answer with 'no' on an already-COMPLETED run.
    holder_c: list[Registry | None] = [None]
    reg_c, pathways_c, send_cap_c, _fetch_c2 = _build_sc15_fixtures(holder_c)
    journal_c = TimescaleJournal(settings=settings)
    await journal_c.ensure_schema()
    try:
        engine_c = _build_sc15_engine(journal_c, reg_c, pathways_c, clock=_sc15_counter_clock())
        state_c: RunState = await engine_c.provide_human_input(run_id, payload={"decision": "no"})
    finally:
        await journal_c.aclose()

    # provide_human_input on a non-AWAITING_HUMAN run returns the current state unchanged.
    assert state_c.status is RunStatus.COMPLETED, (
        "SC-15-C: provide_human_input on COMPLETED run must return COMPLETED; "
        f"got {state_c.status!r}"
    )
    assert send_cap_c.call_count == 0, (
        f"SC-15-C: send must NOT be re-invoked on duplicate answer; got {send_cap_c.call_count}"
    )

    # First-answer-wins: the persisted answer must still be 'yes'.
    persisted: Artifact | None = await journal.read_human_input(run_id, 1)
    assert persisted is not None, "SC-15-C: human input row must exist"
    assert persisted.data.get("decision") == "yes", (
        f"SC-15-C first-answer-wins: persisted decision must be 'yes'; got {persisted.data!r}"
    )


# ---------------------------------------------------------------------------
# SC-15-D: H3 crash window -- answer written + RUNNING set, cold resume
# ---------------------------------------------------------------------------


async def test_sc15_crash_between_record_and_cas_resume_path(
    journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """Crash between record_human_input and CAS flip: cold resume picks up and completes (H3).

    The H3 crash window (from engine.py docstring):
      1. record_human_input persists the answer (H3: BEFORE the CAS).
      2. Process dies before compare_and_set_run_status can flip AWAITING_HUMAN -> RUNNING.

    To simulate this:
      - Engine A parks at AWAITING_HUMAN.
      - We manually call journal.record_human_input + journal.set_run_status(RUNNING) to replicate
        the state after the H3 crash: answer is committed, status is RUNNING (as the CAS would
        have set it), but the drive never ran. (set_run_status is not CAS; it force-sets RUNNING
        regardless -- matching the post-crash journal state.)
      - Engine B calls resume(run_id): RUNNING status means resume() will actually re-drive
        (unlike AWAITING_HUMAN which resume() returns immediately). The re-drive replays committed
        steps 0 and 1, then runs finish_stage which reads the pre-committed answer and routes.

    SC-15-D invariants:
    - final.status is COMPLETED.
    - send_cap_b.call_count == 1 (dispatch_approved fired on the approve answer).
    - probe_blocked is True (taint rehydrated from DB).
    - fetch_cap_b.call_count == 0 (step 0 replayed S6).
    - Zero model calls.
    """
    run_id = "sc15-crash"

    # Engine A: park at AwaitHuman.
    holder_a: list[Registry | None] = [None]
    reg_a, pathways_a, _send_a, _fetch_a = _build_sc15_fixtures(holder_a)
    engine_a = _build_sc15_engine(journal, reg_a, pathways_a, clock=_sc15_counter_clock())

    await engine_a.run(
        run_id=run_id,
        session_id="sc15-sess-crash",
        pathway_id="sc15-path",
        initial=_sc15_initial_artifact(),
    )

    parked: RunState | None = await journal.load_run(run_id)
    assert parked is not None and parked.status is RunStatus.AWAITING_HUMAN
    assert parked.tainted is True

    # Simulate H3 crash window: record the answer then force RUNNING (without CAS, simulating the
    # state left after record_human_input succeeded but the process died before completing the CAS).
    # The Artifact constructor requires kind/produced_by/provenance -- we build a minimal one that
    # matches the engine's own payload wrapping (data={"decision": "yes"}).
    answer_artifact = Artifact(
        kind="human-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_SC15_EPOCH),
        data={"decision": "yes"},
    )
    await journal.record_human_input(run_id, 1, answer_artifact)
    await journal.set_run_status(run_id, RunStatus.RUNNING)

    # Verify the crash state is set up correctly before engine B resumes.
    crash_state: RunState | None = await journal.load_run(run_id)
    assert crash_state is not None
    assert crash_state.status is RunStatus.RUNNING, (
        "SC-15-D setup: run must be in RUNNING after simulated crash"
    )

    # Engine B: fresh instance -- cold resume on RUNNING run re-drives from entry.
    holder_b: list[Registry | None] = [None]
    reg_b, pathways_b, send_cap_b, fetch_cap_b = _build_sc15_fixtures(holder_b)
    journal_b = TimescaleJournal(settings=settings)
    await journal_b.ensure_schema()
    try:
        engine_b = _build_sc15_engine(journal_b, reg_b, pathways_b, clock=_sc15_counter_clock())
        final: RunState = await engine_b.resume(run_id)
    finally:
        await journal_b.aclose()

    assert final.status is RunStatus.COMPLETED, (
        f"SC-15-D: H3 crash resume must reach COMPLETED; got {final.status!r}"
    )
    assert len(final.steps) == 3, f"SC-15-D: must have 3 committed steps; got {len(final.steps)}"
    assert [s.stage_name for s in final.steps] == ["taint", "act", "finish"], (
        f"SC-15-D: stage sequence must be ['taint', 'act', 'finish']; got "
        f"{[s.stage_name for s in final.steps]}"
    )
    assert final.tainted is True

    # Exactly-once: only send (approved) fires; fetch is replayed not re-executed.
    assert send_cap_b.call_count == 1, (
        f"SC-15-D: send must fire exactly once via dispatch_approved; got {send_cap_b.call_count}"
    )
    assert fetch_cap_b.call_count == 0, (
        f"SC-15-D: fetch_ext must NOT re-fire (step 0 replayed S6); got {fetch_cap_b.call_count}"
    )

    # Taint rehydration falsifier: probe_blocked must be True.
    finish_result = final.steps[2].result
    assert finish_result.kind == "done", (
        f"SC-15-D: finish stage must produce Done; got {finish_result.kind!r}"
    )
    assert finish_result.output is not None
    assert finish_result.output.data.get("probe_blocked") is True, (
        "SC-15-D taint-rehydration falsifier: probe_blocked must be True -- "
        "taint must be rehydrated from DB into engine B's fresh gate on cold resume"
    )
