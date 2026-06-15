"""Fire-and-forget-with-feedback — Phase-1 pod 1.4 spike suite (CANON S6, S7, S1, S5, S9).

WRITTEN TESTS-FIRST (S12): these target a contract that does NOT yet exist, so the module fails to
COLLECT on the not-yet-built symbols:
  - ``cogworx.runtime.handle.RunHandle`` — attribute ``run_id: str``; ``async def result()
    -> RunState`` (awaits the backing task, resolves at first park-or-terminal); property
    ``done: bool`` (whether the backing task finished). Thin wrapper over
    ``asyncio.Task[RunState]``.
  - ``Engine.start(*, run_id, session_id, pathway_id, initial, pathway_version=1) -> RunHandle``
    — schedules ``self.run(...)`` on a tracked ``asyncio.Task`` (held in an Engine-owned set,
    discarded on done), returns IMMEDIATELY without awaiting terminal.
  - ``Engine.aclose() -> None`` (async) — drains all outstanding background tasks so the suite
    emits no pending-task warnings.

That RED state is intended — it pins the seam the fire-and-forget pathway must satisfy before any
implementation hardens it (S12 spike-before-harden).

Graph used in most tests: ``gated`` (no model, parks on an asyncio.Event controlled by the test) ->
``work`` (the ONE model-bearing stage) -> ``done_stage`` (terminal, no model). The gate lets the
test assert in-flight state before releasing the run to terminal.

Each criterion names the mutation it kills — a test that does not name a mutation is incomplete.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.coordination.events import Event, EventType
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayError, PathwayRegistry
from cogworx.loop.result import AwaitHuman, Done, StageResult, Transition, Wait
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
)
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine

# The API under test — will ImportError / AttributeError until the python-expert builds it.
from cogworx.runtime.handle import RunHandle
from cogworx.substrate.journal import Journal, RunState
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    CommitSpyJournal,
    CrashAfterStepJournal,
    SimulatedCrash,
    assert_resume_never_recalls_model,
    assert_run_writes_carry_provenance,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_FF_PATHWAY_ID = "ff-pathway"
_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# Step indices in the 3-stage gated graph: gated=0, work=1, done_stage=2.
_WORK_STEP_INDEX = 1


# ---------------------------------------------------------------------------
# Provenance helpers
# ---------------------------------------------------------------------------


def _provenanced(kind: str, produced_by: str, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by=produced_by,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_T0),
        data={"text": text},
    )


def _ff_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_T0),
        data={"text": "fire and forget"},
    )


# ---------------------------------------------------------------------------
# A model that blocks on a gate — the exact pattern from _GatedModel in
# test_timers_sweeper.py, kept local to this module for legibility.
# ---------------------------------------------------------------------------


class _GatedModel:
    """Blocks ``complete`` on an ``asyncio.Event``; counts calls like ``ReplayModel``.

    ``entered`` is set the moment ``complete`` is called (before the gate) so the test
    can synchronise on the run being in-flight inside the stage. ``gate.set()`` releases
    the blocked call and returns the canned response.
    """

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.call_count: int = 0
        self._capabilities = ModelCapabilities(structured_output=True, tools=True)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        self.call_count += 1
        self.entered.set()
        await self.gate.wait()
        return ModelResponse(text="ff-answer", model_id="gated", finish_reason="stop")

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# The fire-and-forget graph
#
# gated (no model, gates on asyncio.Event) -> work (one model call) -> done_stage (terminal)
#
# GatedStage: stores a reference to the asyncio.Event and awaits it before returning
# Transition. This keeps the run mid-flight inside the stage until the test releases the gate.
# ---------------------------------------------------------------------------


class GatedStage:
    """Parks the run inside the stage (not in the journal) by awaiting an asyncio.Event.

    Returns a ``Transition`` to ``work`` once the gate is released. This models a run that is
    RUNNING but has NOT yet committed its first real step — so the test can assert ``done is False``
    and the journal has no COMPLETED status before releasing.

    Mutation killed by FF1: ``start()`` awaiting terminal (the gate would never be checked because
    start blocks; or the stage would never park inside itself because the engine awaits terminal).
    """

    name: str = "gated"
    transitions: tuple[str, ...] = ("work",)

    def __init__(self, gate: asyncio.Event) -> None:
        self._gate = gate

    async def run(self, ctx: StageContext) -> StageResult:
        await self._gate.wait()
        return Transition(to="work", output=_provenanced("gated", "gated"))


class WorkStage:
    """The one model-bearing stage — placed AFTER the gate so the no-recompute proof spans it."""

    name: str = "work"
    transitions: tuple[str, ...] = ("done_stage",)

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="ff work")])
        ctx.budget.record(response.usage)
        return Transition(
            to="done_stage",
            output=_provenanced("response", "work", response.text or ""),
        )


class DoneStage:
    """Terminal stage, no model call."""

    name: str = "done_stage"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_provenanced("done", "done_stage"))


def _build_ff_graph(gate: asyncio.Event) -> StageGraph:
    return StageGraph([GatedStage(gate), WorkStage(), DoneStage()], entry="gated")


def _ff_pathways(gate: asyncio.Event) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_FF_PATHWAY_ID, _build_ff_graph(gate))
    return registry


def _fixed_clock(now: datetime = _T0) -> Callable[[], datetime]:
    return lambda: now


_CLOCK_AT_T0 = _fixed_clock(_T0)


def _make_engine(
    pathways: PathwayRegistry,
    journal: Journal,
    model: ReplayModel | _GatedModel,
    *,
    event_sink: Callable[[Event], None] | None = None,
) -> Engine:
    _registry = ModelRegistry()
    _registry.register_factory("default", lambda g, m=model: BudgetGuardedModel(m, g))
    return Engine(
        models=_registry,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_CLOCK_AT_T0,
        event_sink=event_sink,
    )


def _scripted_model() -> ReplayModel:
    return ReplayModel([ModelResponse(text="ff-answer", model_id="replay", finish_reason="stop")])


def _no_model() -> ReplayModel:
    """A model that raises on ANY call."""
    return ReplayModel([])


# ---------------------------------------------------------------------------
# FF1 — start returns immediately; result() resolves to terminal
#
# With a gated stage, engine.start() must return a RunHandle BEFORE the stage gate is released.
# After releasing, await handle.result() must return COMPLETED. The committed step sequence from
# the background path must be IDENTICAL to a synchronous engine.run() on the same pathway.
#
# Mutation killed:
#   - start() awaiting to terminal: handle.done would be True immediately / run COMPLETED before
#     gate release; the test catches this because it asserts done is False BEFORE releasing.
#   - background path committing differently from sync: the step-sequence equality assertion fails.
# ---------------------------------------------------------------------------


async def test_ff1_start_returns_before_terminal_result_resolves_completed() -> None:
    """``start`` returns a ``RunHandle`` while the run is mid-flight; ``result()`` resolves."""
    gate = asyncio.Event()
    journal = InMemoryJournal()
    model = _scripted_model()
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, journal, model)

    handle = engine.start(
        run_id="ff1",
        session_id="ff1-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )

    # The handle must exist immediately and be a RunHandle.
    assert isinstance(handle, RunHandle)
    assert handle.run_id == "ff1"

    # The run must NOT be COMPLETED yet (the gate is still closed; gated stage is parked).
    # Allow the event loop to tick so the background task starts.
    await asyncio.sleep(0)
    assert handle.done is False

    # The journal run status must NOT be COMPLETED while the gate is closed.
    # (It may be RUNNING or even not yet started — that is fine; it must not be COMPLETED.)
    mid_status = await journal.get_run_status("ff1")
    assert mid_status is not RunStatus.COMPLETED, (
        "FF1 mutation: start() awaited to terminal before gate release; handle.done must be "
        "False while the gated stage is parked"
    )

    # Release the gate — the background task can now complete.
    gate.set()

    # Await the result; must resolve to COMPLETED.
    state = await handle.result()
    assert state.status is RunStatus.COMPLETED
    assert handle.done is True

    # The committed step sequence from the background path must match a FRESH synchronous run.
    gate2 = asyncio.Event()
    gate2.set()  # release immediately so sync run completes in-line
    pathways2 = _ff_pathways(gate2)
    journal2 = InMemoryJournal()
    model2 = _scripted_model()
    engine2 = _make_engine(pathways2, journal2, model2)
    sync_state = await engine2.run(
        run_id="ff1-sync",
        session_id="ff1-sync-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )

    # Stage sequence and count must be identical; step_index sequence must be identical.
    ff_stages = tuple(s.stage_name for s in state.steps)
    sync_stages = tuple(s.stage_name for s in sync_state.steps)
    assert ff_stages == sync_stages, (
        f"FF1 mutation: background path committed different stages than sync path "
        f"({ff_stages!r} vs {sync_stages!r})"
    )
    ff_indices = tuple(s.step_index for s in state.steps)
    sync_indices = tuple(s.step_index for s in sync_state.steps)
    assert ff_indices == sync_indices, (
        f"FF1 mutation: background step_index sequence differs from sync ({ff_indices!r} vs "
        f"{sync_indices!r})"
    )

    await engine.aclose()
    await engine2.aclose()


# ---------------------------------------------------------------------------
# FF2 — feedback on the event sink
#
# After await handle.result(), the captured events for that run_id include at minimum
# RUN_STARTED … RUN_COMPLETED (the with-feedback channel).
#
# Mutation killed: start() bypassing run()'s event emission — feedback would be absent.
# ---------------------------------------------------------------------------


async def test_ff2_feedback_on_event_sink() -> None:
    """Events RUN_STARTED and RUN_COMPLETED are emitted into the sink for the background run."""
    gate = asyncio.Event()
    gate.set()  # release immediately; we only need to verify events were emitted

    events: list[Event] = []

    def _sink(ev: Event) -> None:
        events.append(ev)

    journal = InMemoryJournal()
    model = _scripted_model()
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, journal, model, event_sink=_sink)

    handle = engine.start(
        run_id="ff2",
        session_id="ff2-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )
    state = await handle.result()
    assert state.status is RunStatus.COMPLETED

    run_event_types = {ev.type for ev in events if ev.run_id == "ff2"}
    assert EventType.RUN_STARTED in run_event_types, (
        "FF2 mutation: start() bypassed run()'s event emission; RUN_STARTED must appear in the "
        "event sink for the background run"
    )
    assert EventType.RUN_COMPLETED in run_event_types, (
        "FF2 mutation: start() bypassed run()'s event emission; RUN_COMPLETED must appear in "
        "the event sink for the background run"
    )

    await engine.aclose()


# ---------------------------------------------------------------------------
# FF3 — parks compose with 1.3 (AwaitHuman)
#
# A fire-and-forget run whose pathway hits AwaitHuman:
#   - await handle.result() resolves to AWAITING_HUMAN (first park, NOT hanging forever).
#   - await engine.provide_human_input(run_id, payload=...) drives it to COMPLETED.
#
# Mutation killed: result() blocking past the park — the test would hang forever.
# ---------------------------------------------------------------------------


class _AskForFeedbackStage:
    """Parks the run AWAITING_HUMAN; must declare 'decide' in transitions."""

    name: str = "ask_ff"
    transitions: tuple[str, ...] = ("decide_ff",)

    async def run(self, ctx: StageContext) -> StageResult:
        return AwaitHuman(
            question="Approve?",
            to="decide_ff",
            output=_provenanced("question", "ask_ff"),
        )


class _DecideStage:
    """Reads the answer from the journal and routes to terminal_ff."""

    name: str = "decide_ff"
    transitions: tuple[str, ...] = ("terminal_ff",)

    def __init__(self) -> None:
        self.was_called: bool = False
        self.received_answer: Artifact | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        self.was_called = True
        self.received_answer = await ctx.read_human_input(0)
        return Transition(
            to="terminal_ff",
            output=_provenanced("decided", "decide_ff"),
        )


class _TerminalFfStage:
    """Terminal stage on the approve path."""

    name: str = "terminal_ff"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_provenanced("terminal", "terminal_ff"))


_FF_HUMAN_PATHWAY_ID = "ff-human-pathway"


def _ff_human_pathways(decide: _DecideStage | None = None) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(
        _FF_HUMAN_PATHWAY_ID,
        StageGraph(
            [_AskForFeedbackStage(), decide or _DecideStage(), _TerminalFfStage()],
            entry="ask_ff",
        ),
    )
    return registry


async def test_ff3_parks_compose_with_1_3_await_human() -> None:
    """A fire-and-forget run hitting AwaitHuman resolves result() to AWAITING_HUMAN (not hung)."""
    journal = InMemoryJournal()
    decide = _DecideStage()
    pathways = _ff_human_pathways(decide)
    engine = _make_engine(pathways, journal, _no_model())

    handle = engine.start(
        run_id="ff3",
        session_id="ff3-sess",
        pathway_id=_FF_HUMAN_PATHWAY_ID,
        initial=_ff_initial(),
    )

    # result() must yield at the park, NOT block past it.  If it blocked, this would hang.
    parked = await handle.result()
    assert parked.status is RunStatus.AWAITING_HUMAN, (
        "FF3 mutation: result() blocked past the AwaitHuman park; it must resolve at first "
        "park-or-terminal (AWAITING_HUMAN here)"
    )

    # The human provides the answer via the existing 1.3 API.
    final = await engine.provide_human_input("ff3", payload={"decision": "approve"})
    assert final.status is RunStatus.COMPLETED

    assert decide.was_called
    assert decide.received_answer is not None

    await engine.aclose()


# ---------------------------------------------------------------------------
# FF4 — durability unchanged / crash-resume
#
# Start a run on a CrashAfterStepJournal (crashes after "work" commits); the background
# task raises SimulatedCrash. A FRESH Engine over the SAME shared journal + ReplayModel([])
# resumes via engine.resume(run_id) and completes with ZERO model re-calls.
#
# Mutation killed: the background drive making the run non-resumable or committing differently
# from a regular run() (resume would fail or call_count would be > 0).
# ---------------------------------------------------------------------------


async def test_ff4_crash_resume_background_task_no_model_recall() -> None:
    """Background task crash leaves a durable prefix; cold resume completes exactly-once."""
    gate = asyncio.Event()
    gate.set()  # immediate; crash happens after work commits

    shared = InMemoryJournal()
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="work")
    model = _scripted_model()
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, crash_journal, model)

    handle = engine.start(
        run_id="ff4",
        session_id="ff4-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )

    # The background task should raise SimulatedCrash wrapped in a task exception.
    # We await it and expect either SimulatedCrash to propagate or the task to be cancelled.
    with contextlib.suppress(SimulatedCrash, Exception):
        await handle.result()
    # The crash propagated as expected; the shared journal holds the durable prefix.

    # The shared journal (inner of crash_journal) must hold the "work" step committed before crash.
    mid = await shared.load_run("ff4")
    assert mid is not None
    work_committed = next((s for s in mid.steps if s.stage_name == "work"), None)
    assert work_committed is not None, (
        "FF4 mutation: the background drive did not durably commit 'work' before crashing; "
        "the step must be in the shared journal so cold resume can replay it"
    )

    # Cold resume on a FRESH engine + zero-response model: any re-call raises ReplayExhaustedError.
    zero = ReplayModel([])
    engine_b = _make_engine(pathways, shared, zero)
    final = await engine_b.resume("ff4")

    assert final.status is RunStatus.COMPLETED
    assert zero.call_count == 0, (
        f"FF4 / S6 mutation: cold resume after background-task crash re-called the model "
        f"{zero.call_count} time(s); committed 'work' must replay from the journal"
    )
    stage_seq = tuple(s.stage_name for s in final.steps)
    assert stage_seq == ("gated", "work", "done_stage"), (
        f"FF4 mutation: cold resume produced unexpected stage sequence {stage_seq!r}"
    )
    assert len({s.step_index for s in final.steps}) == len(final.steps), (
        "FF4 mutation: cold resume produced duplicate step indices (double-commit)"
    )

    await engine.aclose()
    await engine_b.aclose()


# ---------------------------------------------------------------------------
# FF5a — aclose drains a task that is STILL PENDING when aclose is called
#
# The 1.4 red-team (M4) showed the original FF5a was vacuous: it awaited handle.result() FIRST,
# so the registry was already empty before aclose ran (and the "Task destroyed but pending!"
# message goes to the asyncio logger, not the warnings system, so recwarn could never fire).
# This version calls aclose() WHILE the background task is parked on a closed gate: a correct
# aclose must still be draining (not returned) until the gate opens.
#
# Mutation killed: aclose() returning without awaiting a pending task; an empty/no-op aclose.
# ---------------------------------------------------------------------------


async def test_ff5a_no_orphaned_task_after_aclose() -> None:
    """``aclose`` awaits a still-pending background task; the registry is empty afterwards."""
    gate = asyncio.Event()  # stays CLOSED until aclose is already in flight
    journal = InMemoryJournal()
    model = _scripted_model()
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, journal, model)

    handle = engine.start(
        run_id="ff5a",
        session_id="ff5a-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )
    await asyncio.sleep(0)
    assert handle.done is False

    closer = asyncio.create_task(engine.aclose())
    await asyncio.sleep(0)
    assert closer.done() is False, (
        "FF5a mutation: aclose() returned while a background task was still parked on its gate"
    )

    gate.set()
    await asyncio.wait_for(closer, timeout=5)

    assert handle.done is True, (
        "FF5a mutation: aclose() returned without draining the pending background task"
    )
    assert (await handle.result()).status is RunStatus.COMPLETED

    # The run_id-keyed registry must be empty after the drain (no AttributeError-tolerant
    # getattr default here — if the attribute is renamed, this must fail LOUD, not pass vacuously).
    registry: dict[str, asyncio.Task[RunState]] = engine._bg
    assert len(registry) == 0, (
        f"FF5a mutation: {len(registry)} task(s) remain registered after aclose()"
    )


# ---------------------------------------------------------------------------
# FF5b — no double-drive: concurrent start + resume of the same run_id
#
# engine.start(run_id) starts a background run; a concurrent engine.resume(same_run_id) before
# terminal must NOT drive the model-bearing 'work' stage twice. The mutual-exclusion primitive is
# the IN-PROCESS drive mutex (``Engine._driving`` set membership, registered by run() for the
# duration of the drive) — NOT a journal CAS: a bare resume of a RUNNING run takes no CAS at all
# (that gap, cross-instance, is the deferred run-lease/reaper pod).
# Assert: model.call_count == 1 across both concurrent drives.
#
# Mutation killed: double-drive (resume driving an already-RUNNING run into the model stage
# again). The resume is wrapped in wait_for so a guard regression fails RED in seconds — without
# it, a regressed resume blocks on the same model gate this test only opens later (an infinite
# CI hang, which is how the missing guard was first caught).
# ---------------------------------------------------------------------------


async def test_ff5b_no_double_drive_concurrent_start_and_resume() -> None:
    """Concurrent start + resume cannot both execute the model-bearing stage."""
    # Use a gated model so start's background task parks inside 'work'.complete;
    # this guarantees the concurrent resume arrives while the run is RUNNING mid-stage.
    gate = asyncio.Event()
    journal = InMemoryJournal()
    gated_model = _GatedModel()

    # We need the gated GRAPH (GatedStage blocks on event, then WorkStage calls the model).
    # BUT WorkStage calls ctx.model.complete — which is gated_model.
    # We want: gated stage releases immediately so 'work' is entered and blocks inside
    # model.complete.
    gate.set()  # GatedStage releases immediately; WorkStage will block on gated_model.gate

    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, journal, gated_model)

    # start schedules the background task; it runs through gated (releases instantly) into work
    # and then blocks inside gated_model.complete waiting for gated_model.gate.
    handle = engine.start(
        run_id="ff5b",
        session_id="ff5b-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )

    # Wait until work is inside the model call (run is RUNNING mid-stage).
    await gated_model.entered.wait()
    assert await journal.get_run_status("ff5b") is RunStatus.RUNNING

    # A concurrent resume of the SAME run_id: must be a no-op (the run is in ``_driving``).
    # wait_for: a regressed guard makes resume block on the still-closed model gate — fail RED
    # at 5s instead of hanging the suite forever.
    resume_state = await asyncio.wait_for(engine.resume("ff5b"), timeout=5)
    # resume sees the in-flight drive: it must return the current state without driving again.
    assert resume_state.status is RunStatus.RUNNING, (
        "FF5b mutation: resume drove into the model-bearing stage while start's background task "
        "was already RUNNING inside it; exactly one driver must win the in-flight run"
    )

    # Release gated_model so the background task completes.
    gated_model.gate.set()
    final = await handle.result()
    assert final.status is RunStatus.COMPLETED

    # The model was called EXACTLY once: only the background task's 'work' stage ran.
    assert gated_model.call_count == 1, (
        f"FF5b mutation: double-drive — model was called {gated_model.call_count} time(s); "
        "the in-process drive mutex must prevent the concurrent resume from re-executing 'work'"
    )

    await engine.aclose()


# ---------------------------------------------------------------------------
# FF6 — aclose drains multiple concurrent background tasks on ONE engine
#
# The 1.4 red-team (M4) showed the original FF6 (one engine PER run) was vacuous: no registry
# ever held more than one task, so an aclose() that awaited only one task survived the suite.
# This version puts three pending runs on ONE engine and releases their gates one at a time:
# after only the first gate opens, a correct aclose must STILL be draining the other two.
#
# Mutation killed: aclose() that awaits only the first / one task, leaving the rest pending.
# ---------------------------------------------------------------------------


async def test_ff6_aclose_drains_multiple_concurrent_tasks() -> None:
    """One engine, three pending background runs — ``aclose`` must drain ALL of them."""
    n_runs = 3
    gates = [asyncio.Event() for _ in range(n_runs)]

    # One registry, one pathway per gate, ONE engine over a single journal.
    registry = PathwayRegistry()
    for i, gate in enumerate(gates):
        registry.register(f"{_FF_PATHWAY_ID}-{i}", _build_ff_graph(gate))
    journal = InMemoryJournal()
    model = ReplayModel(
        [
            ModelResponse(text=f"answer-{i}", model_id="replay", finish_reason="stop")
            for i in range(n_runs)
        ]
    )
    engine = _make_engine(registry, journal, model)

    handles = [
        engine.start(
            run_id=f"ff6-{i}",
            session_id=f"ff6-sess-{i}",
            pathway_id=f"{_FF_PATHWAY_ID}-{i}",
            initial=_ff_initial(),
        )
        for i in range(n_runs)
    ]

    # None are done yet (gates closed).
    await asyncio.sleep(0)
    for h in handles:
        assert h.done is False, "FF6: gate is closed, handle should not be done yet"

    closer = asyncio.create_task(engine.aclose())

    # Release ONLY the first gate and let its run finish completely.
    gates[0].set()
    await asyncio.wait_for(handles[0].result(), timeout=5)
    await asyncio.sleep(0)
    assert closer.done() is False, (
        "FF6 mutation: aclose() returned after draining only ONE task; two runs are still "
        "parked on closed gates"
    )

    # Release the rest; aclose must now complete.
    gates[1].set()
    gates[2].set()
    await asyncio.wait_for(closer, timeout=5)

    for h in handles:
        assert h.done is True, (
            "FF6 mutation: aclose() did not drain every background task; a handle is still pending"
        )
        assert (await h.result()).status is RunStatus.COMPLETED


async def test_ff6b_aclose_drains_a_run_started_mid_drain() -> None:
    """``aclose`` also drains a run ``start``-ed AFTER the drain began (the docstring's claim).

    Mutation killed: a single-pass aclose that snapshots the pending set once — a mid-drain
    start would be orphaned, and the closer would return while its task is still parked.
    """
    gate_a = asyncio.Event()
    gate_b = asyncio.Event()
    registry = PathwayRegistry()
    registry.register(f"{_FF_PATHWAY_ID}-a", _build_ff_graph(gate_a))
    registry.register(f"{_FF_PATHWAY_ID}-b", _build_ff_graph(gate_b))
    journal = InMemoryJournal()
    model = ReplayModel(
        [
            ModelResponse(text="answer-a", model_id="replay", finish_reason="stop"),
            ModelResponse(text="answer-b", model_id="replay", finish_reason="stop"),
        ]
    )
    engine = _make_engine(registry, journal, model)

    handle_a = engine.start(
        run_id="ff6b-a",
        session_id="ff6b-sess-a",
        pathway_id=f"{_FF_PATHWAY_ID}-a",
        initial=_ff_initial(),
    )
    closer = asyncio.create_task(engine.aclose())
    await asyncio.sleep(0)
    assert closer.done() is False

    # Mid-drain start: a single-pass aclose would never see this task.
    handle_b = engine.start(
        run_id="ff6b-b",
        session_id="ff6b-sess-b",
        pathway_id=f"{_FF_PATHWAY_ID}-b",
        initial=_ff_initial(),
    )

    # Finish run A completely; a single-pass aclose would return here with B still parked.
    gate_a.set()
    await asyncio.wait_for(handle_a.result(), timeout=5)
    await asyncio.sleep(0)
    assert closer.done() is False, (
        "FF6b mutation: aclose() returned after its initial snapshot drained, orphaning the "
        "run started mid-drain"
    )

    gate_b.set()
    await asyncio.wait_for(closer, timeout=5)
    assert handle_b.done is True
    assert (await handle_b.result()).status is RunStatus.COMPLETED


# ---------------------------------------------------------------------------
# FF7 — double-start of the same run_id is refused LOUD (RT-1.4 H1)
#
# Two start() calls with one run_id would attach two live drivers to the same run: start_run is
# idempotent (ON CONFLICT DO NOTHING), so the journal would not object while both drive the same
# uncommitted, model-bearing stage (reproduced double model call in the red-team). start() must
# refuse synchronously while a driver is live — and allow a NEW start once the run finished
# (which S6-replays the COMPLETED run without re-calling the model).
#
# Mutation killed: start() not consulting the registry (two live drivers); set-not-dict
# refcount bug (first task's done-callback evicting the second's registration).
# ---------------------------------------------------------------------------


async def test_ff7_double_start_same_run_id_refused() -> None:
    """A second ``start`` of a live run_id raises ``ValueError``; the first driver is intact."""
    gate = asyncio.Event()  # closed: the first run parks inside the gated stage
    journal = InMemoryJournal()
    model = _scripted_model()
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, journal, model)

    handle = engine.start(
        run_id="ff7",
        session_id="ff7-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )
    await asyncio.sleep(0)

    with pytest.raises(ValueError, match="already has a live driver"):
        engine.start(
            run_id="ff7",
            session_id="ff7-sess-2",
            pathway_id=_FF_PATHWAY_ID,
            initial=_ff_initial(),
        )

    # The refusal must not have disturbed the original driver.
    gate.set()
    final = await handle.result()
    assert final.status is RunStatus.COMPLETED
    assert model.call_count == 1, (
        f"FF7 mutation: double-start attached a second driver — the model-bearing stage "
        f"executed {model.call_count} time(s)"
    )

    # Once the run finished, its slot is freed: a NEW start of the same run_id is permitted and
    # S6-replays the COMPLETED run (zero further model calls — ReplayModel([]) would raise).
    handle2 = engine.start(
        run_id="ff7",
        session_id="ff7-sess-3",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )
    replayed = await handle2.result()
    assert replayed.status is RunStatus.COMPLETED
    assert model.call_count == 1, (
        "FF7 / S6 mutation: restarting a COMPLETED run re-called the model instead of replaying"
    )

    await engine.aclose()


# ---------------------------------------------------------------------------
# FF8 — an unknown pathway fails AT start(), not in the background (RT-1.4 H2)
#
# run() resolves the pathway BEFORE start_run journals anything. Backgrounded, that failure
# window silently vanishes the run: the task dies pre-journal, the handle is dropped (the whole
# point of fire-and-forget), aclose absorbs the exception — nothing was ever journaled, so
# nothing is resumable or sweepable. start() must therefore validate the pathway SYNCHRONOUSLY.
#
# Mutation killed: start() skipping the fail-fast lookup (the PathwayError moves into the task).
# ---------------------------------------------------------------------------


async def test_ff8_unknown_pathway_raises_at_start_not_in_background() -> None:
    """``start`` with an unknown pathway raises ``PathwayError`` at the call site."""
    journal = InMemoryJournal()
    engine = _make_engine(_ff_pathways(asyncio.Event()), journal, _no_model())

    with pytest.raises(PathwayError):
        engine.start(
            run_id="ff8",
            session_id="ff8-sess",
            pathway_id="no-such-pathway",
            initial=_ff_initial(),
        )

    # Nothing journaled, nothing registered — and in particular no background task whose
    # pre-journal death would have been a silently lost run.
    assert await journal.load_run("ff8") is None
    assert engine._bg == {}, "FF8 mutation: a task was registered for a refused start"
    await engine.aclose()


# ---------------------------------------------------------------------------
# FF9 — crash FEEDBACK: an unhandled background crash emits RUN_CRASHED; the SAME engine
# recovers (RT-1.4 H2 feedback channel + M4 eviction mutant)
#
# Fire-and-forget-WITH-FEEDBACK: when the backgrounded run dies, the caller who dropped the
# handle must still learn about it on the event sink. And the crash must free the in-process
# driver slot so THIS engine (not just a fresh one) can resume the journaled prefix.
#
# Mutation killed: done-callback not emitting on crash; handle.result() swallowing the
# exception (asserted via pytest.raises, not suppress); the done-callback eviction deleted
# (same-engine resume would return a stale RUNNING state instead of driving — the red-team's
# surviving no-eviction mutant).
# ---------------------------------------------------------------------------


async def test_ff9_background_crash_emits_run_crashed_and_same_engine_recovers() -> None:
    """A crashed background run surfaces as ``RUN_CRASHED`` + ``result()`` re-raise; same-engine
    resume then completes from the journaled prefix with zero model re-calls."""
    gate = asyncio.Event()
    gate.set()  # crash happens AFTER 'work' commits, not in the gated stage

    shared = InMemoryJournal()
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="work")
    model = _scripted_model()
    events: list[Event] = []
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, crash_journal, model, event_sink=events.append)

    handle = engine.start(
        run_id="ff9",
        session_id="ff9-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )

    # result() must RE-RAISE the crash (no suppress here — a result() that swallowed the
    # exception must fail this test).
    with pytest.raises(SimulatedCrash):
        await handle.result()

    # Done-callbacks run a tick after task completion.
    await asyncio.sleep(0)
    crashed = [e for e in events if e.type is EventType.RUN_CRASHED]
    assert len(crashed) == 1, (
        f"FF9 mutation: expected exactly one RUN_CRASHED feedback event, got {len(crashed)} — "
        "a dropped handle means the sink is the ONLY way the caller learns about the crash"
    )
    assert crashed[0].run_id == "ff9"
    assert crashed[0].attributes["error_type"] == "SimulatedCrash"

    # The crash freed the driver slot: the SAME engine resumes the journaled prefix. 'work' is
    # already committed, so the resume replays it (S6) — zero further model calls.
    final = await engine.resume("ff9")
    assert final.status is RunStatus.COMPLETED, (
        "FF9 mutation: same-engine resume after a background crash did not drive — the crashed "
        "task's registration was not evicted (stale in-process driver slot)"
    )
    assert model.call_count == 1, (
        f"FF9 / S6 mutation: resume re-called the model ({model.call_count} calls total)"
    )

    await engine.aclose()


# ---------------------------------------------------------------------------
# FF10 — park-then-sweep race: a sweeper re-drive and a concurrent resume cannot both
# execute the model-bearing stage (RT-1.4 H3, reproduced attack C)
#
# A backgrounded run parks WAITING; its task completes and leaves the registry. The sweeper's
# fire_timer then re-drives it. The red-team showed a concurrent resume() slipped past the old
# start()-scoped guard (fire_timer registered nothing) and double-executed 'work'. The drive
# mutex must cover EVERY drive entrypoint, registered BEFORE the CAS.
#
# Mutation killed: fire_timer not registering in the drive mutex (resume double-drives).
# NOT raced here: registration-AFTER-the-CAS (the InMemory journal's awaits resolve too fast to
# interleave a resume into that window deterministically) — that ordering is enforced by
# construction instead: no await sits between the membership check and the add in fire_timer.
# ---------------------------------------------------------------------------


class _WaitOnceStage:
    """Parks the run on a durable timer on first execution; replays as a plain advance."""

    name: str = "wait_stage"
    transitions: tuple[str, ...] = ("work",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Wait(
            to="work",
            wake_at=ctx.clock() + timedelta(seconds=60),
            output=_provenanced("wait", "wait_stage"),
        )


async def test_ff10_sweeper_redrive_vs_resume_single_driver() -> None:
    """While ``fire_timer`` drives a woken run, a concurrent ``resume`` is a no-op."""
    journal = InMemoryJournal()
    gated_model = _GatedModel()
    registry = PathwayRegistry()
    registry.register(
        "ff-wait", StageGraph([_WaitOnceStage(), WorkStage(), DoneStage()], entry="wait_stage")
    )
    engine = _make_engine(registry, journal, gated_model)

    # Drive to the durable park (no model call yet).
    parked = await engine.run(
        run_id="ff10",
        session_id="ff10-sess",
        pathway_id="ff-wait",
        initial=_ff_initial(),
    )
    assert parked.status is RunStatus.WAITING
    assert gated_model.call_count == 0

    # The sweeper wakes the run: fire_timer replays the Wait and drives INTO 'work', where the
    # gated model parks it mid-stage.
    fire = asyncio.create_task(engine.fire_timer("ff10"))
    await asyncio.wait_for(gated_model.entered.wait(), timeout=5)

    # A concurrent resume of the woken run must be a no-op (wait_for: a regressed guard blocks
    # on the same closed model gate — fail RED, not a CI hang).
    resumed = await asyncio.wait_for(engine.resume("ff10"), timeout=5)
    assert resumed.status is RunStatus.RUNNING, (
        "FF10 mutation: resume drove a run the sweeper was already driving"
    )

    gated_model.gate.set()
    final = await asyncio.wait_for(fire, timeout=5)
    assert final.status is RunStatus.COMPLETED
    assert gated_model.call_count == 1, (
        f"FF10 mutation: sweeper-vs-resume double-drive — the model-bearing stage executed "
        f"{gated_model.call_count} time(s); the drive mutex must cover fire_timer"
    )


# ---------------------------------------------------------------------------
# FF11 — two concurrent in-process resumes of an orphaned RUNNING run: one driver wins
# (RT-1.4 H3, reproduced attack E)
#
# A crash leaves the run RUNNING in the journal with an uncommitted stage. Two concurrent
# resume() calls on the SAME engine must not both execute it — the first to register in the
# drive mutex drives; the second returns the current state. (The same race across two ENGINE
# INSTANCES remains the documented run-lease/reaper carry-forward.)
#
# Mutation killed: resume not registering in the drive mutex (both resumes drive 'work').
# ---------------------------------------------------------------------------


async def test_ff11_concurrent_resumes_same_engine_single_driver() -> None:
    """Of two concurrent same-engine resumes of an orphaned RUNNING run, exactly one drives."""
    gate = asyncio.Event()
    gate.set()  # the gated STAGE releases immediately; the crash is what orphans the run

    shared = InMemoryJournal()
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="gated")
    gated_model = _GatedModel()
    pathways = _ff_pathways(gate)
    engine = _make_engine(pathways, crash_journal, gated_model)

    # Orphan the run: crash right after 'gated' commits — journal status stays RUNNING.
    with pytest.raises(SimulatedCrash):
        await engine.run(
            run_id="ff11",
            session_id="ff11-sess",
            pathway_id=_FF_PATHWAY_ID,
            initial=_ff_initial(),
        )
    assert await shared.get_run_status("ff11") is RunStatus.RUNNING
    assert gated_model.call_count == 0

    # First resume replays 'gated' and parks inside the model-bearing 'work'.
    first = asyncio.create_task(engine.resume("ff11"))
    await asyncio.wait_for(gated_model.entered.wait(), timeout=5)

    # Second concurrent resume on the SAME engine: must return without driving.
    second = await asyncio.wait_for(engine.resume("ff11"), timeout=5)
    assert second.status is RunStatus.RUNNING, (
        "FF11 mutation: a second same-engine resume drove an already-driven RUNNING run"
    )

    gated_model.gate.set()
    final = await asyncio.wait_for(first, timeout=5)
    assert final.status is RunStatus.COMPLETED
    assert gated_model.call_count == 1, (
        f"FF11 mutation: concurrent same-engine resumes double-drove 'work' "
        f"({gated_model.call_count} model calls); the drive mutex must cover resume itself"
    )


# ---------------------------------------------------------------------------
# S6 invariant — background drive replays identically (assert_resume_never_recalls_model)
#
# A fire-and-forget run over a CrashAfterStepJournal; cold resume on a zero-response model.
# Reuses the canonical S6 harness from invariants.py directly.
#
# Mutation killed: the background drive committing differently from a regular run, making the S6
# harness's pre/post-resume step comparison fail; or a crash in the background task leaving the
# journal in a state that makes cold resume call the model again.
# ---------------------------------------------------------------------------


async def test_s6_background_drive_replays_identically() -> None:
    """S6: assert_resume_never_recalls_model applies to the fire-and-forget background path."""
    gate = asyncio.Event()
    gate.set()

    shared = InMemoryJournal()
    scripted = _scripted_model()
    pathways = _ff_pathways(gate)

    def _build_engine(journal: Journal, model: ReplayModel) -> Engine:
        return _make_engine(pathways, journal, model)

    await assert_resume_never_recalls_model(
        build_engine=_build_engine,
        initial=_ff_initial(),
        pathway_id=_FF_PATHWAY_ID,
        crash_after_stage="work",
        shared_journal=shared,
        scripted_model=scripted,
        run_id="s6-ff",
        session_id="s6-ff-sess",
    )


# ---------------------------------------------------------------------------
# S1 + S5 light check — write path untouched by fire-and-forget scheduling
#
# The CommitSpyJournal ensures no model call happens on any commit_step. A quick
# assert_run_writes_carry_provenance confirms the output artifacts are provenance-bearing.
# (Heavy S1/S5 coverage already lives in prior pods; this is the belt-and-suspenders check
# that start() did not introduce a new model-on-write-path regression.)
#
# Mutation killed: Engine.start() wrapping the run() call in a way that inserts a model call
# on the commit path (e.g. by summarizing on commit inside the scheduling wrapper).
# ---------------------------------------------------------------------------


async def test_s1_s5_start_write_path_unchanged() -> None:
    """S1: no model call on the write path when driven via start(); S5: outputs carry provenance."""
    gate = asyncio.Event()
    gate.set()

    inner = InMemoryJournal()
    model = _scripted_model()
    pathways = _ff_pathways(gate)
    spy = CommitSpyJournal(inner=inner, model=model)
    engine = _make_engine(pathways, spy, model)

    handle = engine.start(
        run_id="s1s5-ff",
        session_id="s1s5-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )
    # CommitSpyJournal raises InvariantViolation inside the task if any commit calls the model.
    # Propagation: handle.result() re-raises the task's exception.
    state = await handle.result()
    assert state.status is RunStatus.COMPLETED

    # Also verify via the loaded run state (the spy delegates to inner).
    final = await inner.load_run("s1s5-ff")
    assert final is not None
    assert_run_writes_carry_provenance(final)

    await engine.aclose()


# ---------------------------------------------------------------------------
# RunHandle contract — structural validation (no running engine needed)
# ---------------------------------------------------------------------------


async def test_run_handle_has_run_id_attribute() -> None:
    """``RunHandle`` must expose ``run_id: str`` as a public attribute.

    Mutation killed: ``run_id`` stored only privately — the fire-and-forget caller cannot
    correlate feedback events to the handle without it. (The original structural check
    ``hasattr(cls, "__annotations__")`` was vacuously true for EVERY class — RT-1.4 L7.)
    """

    async def _never() -> RunState:
        raise NotImplementedError

    task = asyncio.create_task(_never())
    task.cancel()
    handle = RunHandle("the-run", task)
    assert handle.run_id == "the-run", "RunHandle must expose the run_id it was built with"
    with contextlib.suppress(asyncio.CancelledError):
        await task


def test_run_handle_has_done_property() -> None:
    """``RunHandle`` must expose ``done: bool`` as a property or attribute.

    Mutation killed: ``done`` returning an ``asyncio.Future`` or requiring an await — the caller
    must be able to poll ``h.done`` synchronously to check task status.
    """
    # Structural: confirm done is not an async method.
    done_attr = getattr(RunHandle, "done", None)
    assert done_attr is not None, "RunHandle must have a 'done' attribute or property"
    # If it is a function/coroutine it would require await; it must be a property (or bool).
    assert not inspect.iscoroutinefunction(done_attr), (
        "RunHandle.done must be a synchronous property, not an async method"
    )


async def test_run_handle_result_is_coroutine() -> None:
    """``RunHandle.result`` must be an async method (awaitable).

    Mutation killed: ``result`` returning a plain ``RunState`` synchronously — callers need to
    ``await h.result()`` so they co-operatively yield to the event loop while the task finishes.
    """
    assert inspect.iscoroutinefunction(RunHandle.result), (
        "RunHandle.result must be an async method so callers can await the background task"
    )


async def test_engine_start_method_exists() -> None:
    """``Engine.start`` must exist as a method (no-op structure check pre-impl).

    Mutation killed: the python-expert adding the method under a different name or not at all.
    """
    assert hasattr(Engine, "start"), (
        "Engine must have a 'start' method for fire-and-forget scheduling"
    )
    # It must be a regular (not async) method — start returns immediately.
    assert not inspect.iscoroutinefunction(Engine.start), (
        "Engine.start must be a synchronous method that returns a RunHandle immediately; "
        "making it async would force the caller to await and defeats fire-and-forget semantics"
    )


async def test_engine_aclose_method_is_async() -> None:
    """``Engine.aclose`` must be an async method.

    Mutation killed: aclose returning synchronously (asyncio.gather on tasks requires await;
    a sync aclose cannot safely drain background tasks).
    """
    assert hasattr(Engine, "aclose"), (
        "Engine must have an 'aclose' async method for draining background tasks"
    )
    assert inspect.iscoroutinefunction(Engine.aclose), (
        "Engine.aclose must be an async method; draining asyncio.Tasks requires await"
    )


# ---------------------------------------------------------------------------
# F1 regression test — start() builds exactly one context per run
# ---------------------------------------------------------------------------


async def test_start_builds_exactly_one_context() -> None:
    """F1 regression: ``start()`` drives the same ``run()`` coroutine that calls ``_build_context``
    exactly once — no additional context is minted in ``start()`` itself.

    A counting factory records every guard that is vended by the registry's ``assemble`` call
    (which happens inside ``_build_context``).  A clean ``start()`` + ``result()`` must produce
    exactly ONE mint (one drive, one guard, one context).

    Mutation killed: ``start()`` calling ``_build_context`` itself in addition to ``run()`` —
    that would register the guard twice and mint == 2.
    """
    gate = asyncio.Event()
    gate.set()  # no gating — let the run complete straight to terminal
    journal = InMemoryJournal()
    model = _scripted_model()
    pathways = _ff_pathways(gate)

    mints: list[object] = []
    counting_registry = ModelRegistry()
    counting_registry.register_factory(
        "default",
        lambda g, m=model: mints.append(g) or BudgetGuardedModel(m, g),
    )
    engine = Engine(
        models=counting_registry,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_CLOCK_AT_T0,
    )

    handle = engine.start(
        run_id="ctx-count",
        session_id="ctx-count-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )
    await handle.result()

    assert len(mints) == 1, (
        f"F1 regression: expected exactly 1 guard minted (1 drive = 1 context), got {len(mints)}; "
        "start() must NOT call _build_context() itself — that creates a ghost context with no drive"
    )

    await engine.aclose()


async def test_ff9_background_crash_mints_one_context_run_crashed_lands_on_sink() -> None:
    """F1 + FF9 combined: crash path mints exactly one context; RUN_CRASHED lands on the sink.

    The guard is minted by ``run()`` (inside ``_build_context``), NOT by ``start()``.  Even on a
    crash, ``mints`` must hold exactly one entry.  And the RUN_CRASHED event — now emitted via
    ``_emit_engine`` (no live RunContext) — must still reach the event sink.

    Mutation killed:
    - start() minting a context (mints > 1 on crash path).
    - _emit_engine not routing to the sink (crashed event drops).
    """
    gate = asyncio.Event()
    gate.set()

    shared = InMemoryJournal()
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="work")
    model = _scripted_model()
    events: list[Event] = []
    pathways = _ff_pathways(gate)

    mints: list[object] = []
    counting_registry = ModelRegistry()
    counting_registry.register_factory(
        "default",
        lambda g, m=model: mints.append(g) or BudgetGuardedModel(m, g),
    )
    engine = Engine(
        models=counting_registry,
        journal=crash_journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_CLOCK_AT_T0,
        event_sink=events.append,
    )

    handle = engine.start(
        run_id="ff9-ctx",
        session_id="ff9-ctx-sess",
        pathway_id=_FF_PATHWAY_ID,
        initial=_ff_initial(),
    )

    with pytest.raises(SimulatedCrash):
        await handle.result()

    await asyncio.sleep(0)

    # Exactly one guard minted — start() does not call _build_context() itself.
    assert len(mints) == 1, (
        f"F1 regression on crash path: expected 1 guard minted, got {len(mints)}"
    )

    # RUN_CRASHED must land on the sink via _emit_engine (no live ctx).
    crashed = [e for e in events if e.type is EventType.RUN_CRASHED]
    assert len(crashed) == 1, (
        f"RUN_CRASHED must reach the sink even with no live RunContext; got {len(crashed)} events"
    )
    assert crashed[0].run_id == "ff9-ctx"

    await engine.aclose()
