"""Durable timers + sweeper + pause/resume — Phase-1 pod 1.1 spike/chaos suite (CANON S6, S5, S1).

WRITTEN TESTS-FIRST (S12): these target a contract that does NOT yet exist, so the module fails to
COLLECT (import/attribute errors on ``Wait``, ``RunStatus.WAITING``/``PAUSED``, the new ``Journal``
lease/cancel methods, ``Engine.fire_timer``/``pause``/``unpause``, and ``cogworx.runtime.sweeper``).
That RED state is intended — it pins the seam a Wait-bearing pathway must satisfy before any impl
hardens it (S12 spike-before-harden).

The graph mirrors ``test_durability_chaos.py``: ``intake`` (no model) -> ``wait`` (returns a 5th
``StageResult`` kind, ``Wait(to="poll", wake_at=clock()+delay)``) -> ``poll`` (the ONE model-bearing
stage, kept AFTER the wait so the no-re-call proof spans the timer fire) -> ``close`` (terminal,
no-model). All clocks are injected so ``wake_at`` arithmetic is deterministic — never wall-clock.

Each criterion is mutation-resistant: a destructive-delete lease, a re-fire that re-calls the
model, a Wait that re-arms ``set_timer`` on replay, or a resume that advances a WAITING/PAUSED run
must make exactly one of these tests fail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition, Wait
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
)
from cogworx.runtime.engine import Engine
from cogworx.runtime.sweeper import Sweeper
from cogworx.substrate.journal import Journal, RunState, StepRecord, Timer
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    CrashAfterStepJournal,
    SimulatedCrash,
    assert_run_writes_carry_provenance,
)

_TIMER_PATHWAY_ID = "timer-chaos"

# A fixed base instant + delay so wake_at is deterministic (clock injected everywhere).
_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_DELAY = timedelta(minutes=5)
_WAKE_AT = _T0 + _DELAY
_BEFORE_WAKE = _WAKE_AT - timedelta(seconds=1)
_AFTER_WAKE = _WAKE_AT + timedelta(seconds=1)
_LEASE_TTL = timedelta(minutes=1)

# Positional step indices in the timer graph: intake=0, wait=1, poll=2, close=3.
_POLL_STEP_INDEX = 2


# --------------------------------------------------------------------------------------------------
# The timer-bearing chaos graph (mirrors build_chaos_graph + _make_build_engine in the S6 suite)
# --------------------------------------------------------------------------------------------------


def _provenanced(kind: str, produced_by: str, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by=produced_by,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_T0),
        data={"text": text},
    )


class WaitStage:
    """Returns the 5th ``StageResult`` kind: ``Wait(to="poll", wake_at=clock()+delay)``.

    ``to`` MUST be declared in ``transitions`` or ``StageGraph`` rejects the graph (the Wait target
    has to be a reachable edge like any transition). The ``wake_at`` is derived from the injected
    engine clock the stage is handed via ``ctx`` — deterministic, never wall-clock.
    """

    name: str = "wait"
    transitions: tuple[str, ...] = ("poll",)

    def __init__(self, *, wake_at: datetime) -> None:
        self._wake_at = wake_at

    async def run(self, ctx: StageContext) -> StageResult:
        return Wait(
            to="poll",
            wake_at=self._wake_at,
            output=_provenanced("wait", "wait", "armed"),
        )


class PollStage:
    """The ONE model-bearing stage, AFTER the wait — the no-re-call proof spans the timer fire."""

    name: str = "poll"
    transitions: tuple[str, ...] = ("close",)

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Poll after the wait.")]
        )
        ctx.budget.record(response.usage)
        return Transition(to="close", output=_provenanced("response", "poll", response.text or ""))


class CloseStage:
    """Terminal, no-model stage: echoes the text the (committed) poll step produced."""

    name: str = "close"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        prior = await ctx.journal.read_step(ctx.run_id, _POLL_STEP_INDEX)
        text = ""
        if prior is not None:
            output = getattr(prior.result, "output", None)
            if output is not None:
                text = str(output.data.get("text", ""))
        return Done(output=_provenanced("closed", "close", text))


class _IntakeToWait:
    """Intake that transitions to ``wait`` (the shared ``IntakeStage`` hardwires ``to="respond"``).

    The 4-stage timer graph's second stage is ``wait``, so the entry stage must edge to ``wait`` or
    ``StageGraph`` rejects the graph at construction. Mirrors how ``test_durability_chaos.py``
    defines its own ``RespondToCloseStage`` rather than reusing a stage whose ``to`` target does not
    exist in its graph.
    """

    name: str = "intake"
    transitions: tuple[str, ...] = ("wait",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="wait", output=_provenanced("intake", "intake"))


def build_timer_graph(*, wake_at: datetime = _WAKE_AT) -> StageGraph:
    return StageGraph(
        [_IntakeToWait(), WaitStage(wake_at=wake_at), PollStage(), CloseStage()], entry="intake"
    )


def _timer_pathways(*, wake_at: datetime = _WAKE_AT) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_TIMER_PATHWAY_ID, build_timer_graph(wake_at=wake_at))
    return registry


def _fixed_clock(now: datetime) -> Callable[[], datetime]:
    return lambda: now


_CLOCK_AT_T0 = _fixed_clock(_T0)


def _make_build_engine(
    pathways: PathwayRegistry, *, clock: Callable[[], datetime] = _CLOCK_AT_T0
) -> Callable[[Journal, ReplayModel], Engine]:
    def build(journal: Journal, model: ReplayModel) -> Engine:
        return Engine(
            model=model,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            clock=clock,
        )

    return build


def _timer_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_T0),
        data={"text": "hello"},
    )


def _scripted_model() -> ReplayModel:
    return ReplayModel(
        [ModelResponse(text="post-wait answer", model_id="replay", finish_reason="stop")]
    )


# --------------------------------------------------------------------------------------------------
# A counting wrapper for C6 (spies on set_timer; delegates everything else to the inner journal)
# --------------------------------------------------------------------------------------------------


class SetTimerSpyJournal:
    """Counts ``set_timer`` calls; delegates every other method to the inner ``Journal``.

    C6 wraps the SHARED inner journal in this spy across run + fire_timer + resume and asserts
    ``set_timer`` is called exactly ONCE (the fresh Wait commit) and NEVER again on replay of the
    committed Wait.
    """

    def __init__(self, inner: Journal) -> None:
        self._inner = inner
        self.set_timer_count = 0

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        await self._inner.start_run(
            run_id,
            session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint,
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._inner.set_run_status(run_id, status)

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def commit_step(self, record: StepRecord) -> None:
        await self._inner.commit_step(record)

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

    async def load_run(self, run_id: str) -> RunState | None:
        return await self._inner.load_run(run_id)

    async def set_timer(self, timer: Timer) -> None:
        self.set_timer_count += 1
        await self._inner.set_timer(timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return await self._inner.due_timers(now)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        return await self._inner.claim_due_timers(now, lease_ttl=lease_ttl)

    async def cancel_timer(self, timer_id: str) -> None:
        await self._inner.cancel_timer(timer_id)

    async def cancel_timers_for_run(self, run_id: str) -> None:
        await self._inner.cancel_timers_for_run(run_id)

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return await self._inner.get_run_status(run_id)


class PauseAfterStepJournal:
    """A ``Journal`` wrapper that flips the run to PAUSED right after a chosen stage commits.

    Models an EXTERNAL pause arriving mid-drive (e.g. another process calling ``pause`` between
    steps): the step is durably committed, then the run status is set to PAUSED via the inner
    journal. The engine's top-of-loop ``get_run_status`` check must observe PAUSED and break BEFORE
    running the next stage — so the drive stops cooperatively at a step boundary, with the
    just-committed step's transition preserved and resumed on ``unpause``. Delegates the rest.
    """

    def __init__(self, *, inner: Journal, pause_after_stage: str) -> None:
        self._inner = inner
        self._pause_after_stage = pause_after_stage

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        await self._inner.start_run(
            run_id,
            session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint,
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._inner.set_run_status(run_id, status)

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def commit_step(self, record: StepRecord) -> None:
        await self._inner.commit_step(record)
        if record.stage_name == self._pause_after_stage:
            # An external pause lands the instant this step's commit is durable.
            await self._inner.set_run_status(record.run_id, RunStatus.PAUSED)

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

    async def load_run(self, run_id: str) -> RunState | None:
        return await self._inner.load_run(run_id)

    async def set_timer(self, timer: Timer) -> None:
        await self._inner.set_timer(timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return await self._inner.due_timers(now)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        return await self._inner.claim_due_timers(now, lease_ttl=lease_ttl)

    async def cancel_timer(self, timer_id: str) -> None:
        await self._inner.cancel_timer(timer_id)

    async def cancel_timers_for_run(self, run_id: str) -> None:
        await self._inner.cancel_timers_for_run(run_id)

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return await self._inner.get_run_status(run_id)


# --------------------------------------------------------------------------------------------------
# C1 — round-trip + sweeper advance
# --------------------------------------------------------------------------------------------------


async def test_c1_wait_round_trip_and_sweeper_advance() -> None:
    """Run hits Wait -> WAITING + one timer; sweeper fires only at/after wake -> COMPLETED."""
    journal = InMemoryJournal()
    model = _scripted_model()
    engine = _make_build_engine(_timer_pathways())(journal, model)

    state = await engine.run(
        run_id="c1",
        session_id="c1-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    # The run parks at the Wait — non-terminal WAITING, no model call yet (poll is AFTER the wait).
    assert state.status is RunStatus.WAITING
    assert model.call_count == 0
    assert tuple(step.stage_name for step in state.steps) == ("intake", "wait")

    # Exactly one timer present, keyed (run_id:step_index) and due at wake_at.
    timers = await journal.due_timers(_AFTER_WAKE)
    assert len(timers) == 1
    assert timers[0].timer_id == "c1:1"
    assert timers[0].wake_at == _WAKE_AT

    sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(_BEFORE_WAKE))

    # Before wake: nothing due -> fires 0, still WAITING, still no model call.
    fired_before = await sweeper.tick(_BEFORE_WAKE, lease_ttl=_LEASE_TTL)
    assert fired_before == 0
    assert (await journal.get_run_status("c1")) is RunStatus.WAITING
    assert model.call_count == 0

    # At/after wake: fires exactly 1, run advances through poll -> close to COMPLETED.
    fired_after = await sweeper.tick(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert fired_after == 1

    final = await journal.load_run("c1")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "wait", "poll", "close")
    assert model.call_count == 1  # poll ran exactly once, only after the timer fired
    last_result = final.steps[-1].result
    assert isinstance(last_result, Done)
    assert last_result.output.data["text"] == "post-wait answer"


# --------------------------------------------------------------------------------------------------
# C2 — exactly-once advance across a crash AFTER the post-wait stage commits (S6 core)
# --------------------------------------------------------------------------------------------------


async def test_c2_exactly_once_advance_across_crash_after_poll() -> None:
    """Fire the timer, crash after ``poll`` commits; cold resume completes exactly-once."""
    shared = InMemoryJournal()
    pathways = _timer_pathways()
    scripted = _scripted_model()

    # Drive to the Wait (engine A, intact journal).
    engine_a = _make_build_engine(pathways)(shared, scripted)
    waiting = await engine_a.run(
        run_id="c2",
        session_id="c2-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    assert waiting.status is RunStatus.WAITING
    assert scripted.call_count == 0

    # Lease the due timer, then fire on an engine whose journal crashes AFTER poll durably commits.
    leased = await shared.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert len(leased) == 1
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="poll")
    engine_crash = _make_build_engine(pathways)(crash_journal, scripted)
    crashed = False
    try:
        await engine_crash.fire_timer("c2")
    except SimulatedCrash:
        crashed = True
    assert crashed
    assert scripted.call_count == 1  # poll's model call happened exactly once, before the crash

    mid = await shared.load_run("c2")
    assert mid is not None
    assert mid.status is RunStatus.RUNNING  # crashed mid-flight, NOT terminal

    # Cold resume on a FRESH engine + zero-response model: any re-call raises ReplayExhaustedError.
    zero = ReplayModel([])
    engine_b = _make_build_engine(pathways)(shared, zero)
    final = await engine_b.resume("c2")

    assert final.status is RunStatus.COMPLETED
    stage_seq = tuple(step.stage_name for step in final.steps)
    assert stage_seq == ("intake", "wait", "poll", "close")  # no dupes -> exactly-once per index
    assert len(set(step.step_index for step in final.steps)) == len(final.steps)
    assert zero.call_count == 0  # committed poll replayed, never recomputed
    last_result = final.steps[-1].result
    assert isinstance(last_result, Done)
    assert last_result.output.data["text"] == "post-wait answer"


# --------------------------------------------------------------------------------------------------
# C3 — at-least-once fire / exactly-once effect: crash BEFORE the lease -> timer survives
# --------------------------------------------------------------------------------------------------


async def test_c3_crash_before_lease_timer_survives_then_fires_once() -> None:
    """A crash before any lease leaves the timer unclaimed; the next tick fires it exactly-once."""
    journal = InMemoryJournal()
    model = _scripted_model()
    engine = _make_build_engine(_timer_pathways())(journal, model)

    await engine.run(
        run_id="c3",
        session_id="c3-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    # Simulate a sweeper process dying BEFORE it leased: the timer is still present + unclaimed.
    surviving = await journal.due_timers(_AFTER_WAKE)
    assert len(surviving) == 1
    assert surviving[0].claimed_at is None

    # The next tick claims + fires it exactly once -> run completes.
    sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(_AFTER_WAKE))
    fired = await sweeper.tick(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert fired == 1

    final = await journal.load_run("c3")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "wait", "poll", "close")
    assert model.call_count == 1


# --------------------------------------------------------------------------------------------------
# C4 — concurrent-claim single-delivery (in-memory lease analog)
# --------------------------------------------------------------------------------------------------


async def test_c4_back_to_back_claim_leases_once_in_memory() -> None:
    """Two back-to-back ``claim_due_timers`` over the same due set: the second returns empty.

    The first call LEASES the timer (sets ``claimed_at``); a second call at the same instant sees a
    fresh (non-stale) lease and returns nothing -> the timer is delivered ONCE. This is the
    in-memory analog of single-delivery; the true PG race is exercised in the integration suite.
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    engine = _make_build_engine(_timer_pathways())(journal, model)
    await engine.run(
        run_id="c4",
        session_id="c4-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )

    first = await journal.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    second = await journal.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)

    assert len(first) == 1
    assert first[0].timer_id == "c4:1"
    assert len(second) == 0  # the fresh lease blocks a second delivery
    # The lease does NOT delete: the timer still exists (re-leaseable once the lease goes stale).
    assert len(await journal.due_timers(_AFTER_WAKE)) == 1


# --------------------------------------------------------------------------------------------------
# C5 — pause/unpause across processes
# --------------------------------------------------------------------------------------------------


async def test_c5_cooperative_pause_mid_drive_then_unpause_across_fresh_engine() -> None:
    """An external pause lands mid-drive -> the loop breaks at the next step boundary (PAUSED).

    Uses a NON-waiting 3-stage graph (intake -> poll(model) -> close) and a
    ``PauseAfterStepJournal`` that flips the run to PAUSED the instant ``poll`` commits — modelling
    another process calling ``pause`` between steps. The engine's top-of-loop ``get_run_status``
    check must then observe PAUSED and break BEFORE running ``close``, so the run is left PAUSED
    with ``close`` NOT yet run (the just-committed ``poll`` transition is preserved). A fresh engine
    ``resume`` must NOT advance a PAUSED run; ``unpause`` re-drives to terminal, replaying the
    committed ``poll`` with no model re-call. This is the cooperative mid-drive pause the architect
    validated — not pausing a finished run.
    """
    registry = PathwayRegistry()

    class _IntakeToPoll:
        name: str = "intake"
        transitions: tuple[str, ...] = ("poll",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Transition(to="poll", output=_provenanced("intake", "intake"))

    # poll is at index 1 here, so close reads index 1 — dedicated close for this no-wait graph.
    class _CloseReadsPoll:
        name: str = "close"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            prior = await ctx.journal.read_step(ctx.run_id, 1)
            text = ""
            if prior is not None:
                output = getattr(prior.result, "output", None)
                if output is not None:
                    text = str(output.data.get("text", ""))
            return Done(output=_provenanced("closed", "close", text))

    registry.register(
        "pause-pw", StageGraph([_IntakeToPoll(), PollStage(), _CloseReadsPoll()], entry="intake")
    )

    shared = InMemoryJournal()
    model = _scripted_model()
    build = _make_build_engine(registry)

    # Engine A drives through a journal that flips PAUSED the instant poll commits. The drive must
    # stop cooperatively at the NEXT step boundary — close never runs.
    pausing = PauseAfterStepJournal(inner=shared, pause_after_stage="poll")
    engine_a = build(pausing, model)
    paused_state = await engine_a.run(
        run_id="c5",
        session_id="c5-sess",
        pathway_id="pause-pw",
        initial=_timer_initial(),
    )
    assert paused_state.status is RunStatus.PAUSED
    # poll committed (model ran once) but close did NOT run — the loop broke at the boundary.
    assert tuple(step.stage_name for step in paused_state.steps) == ("intake", "poll")
    assert model.call_count == 1
    assert (await shared.get_run_status("c5")) is RunStatus.PAUSED

    # A FRESH engine resume must NOT advance a PAUSED run (non-advancing, no model re-call).
    zero = ReplayModel([])
    engine_b = build(shared, zero)
    resumed = await engine_b.resume("c5")
    assert resumed.status is RunStatus.PAUSED
    assert zero.call_count == 0

    # unpause re-drives to terminal on the fresh engine: intake + poll REPLAYED (no model re-call),
    # only close runs anew.
    final = await engine_b.unpause("c5")
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "poll", "close")
    assert zero.call_count == 0  # the committed poll was replayed, never recomputed
    last_result = final.steps[-1].result
    assert isinstance(last_result, Done)
    assert last_result.output.data["text"] == "post-wait answer"


# --------------------------------------------------------------------------------------------------
# C5b — engine.pause()/unpause() directly: the conditional CAS, not the journal-wrapper path
# --------------------------------------------------------------------------------------------------


async def test_c5b_engine_pause_cas_lands_paused_then_unpause_completes() -> None:
    """``engine.pause()`` WHILE a stage is gated -> True, run lands PAUSED at the next boundary.

    C5 exercises the engine's top-of-loop PAUSED detection via a journal wrapper that flips the
    status on commit; this test drives ``engine.pause()`` ITSELF (the conditional CAS). The run is
    driven in an ``asyncio`` task with ``poll`` gated on a ``_GatedModel`` event (the F6 pattern):
    while ``poll`` is parked inside the model call, the run is RUNNING, so the RUNNING->PAUSED CAS
    wins and returns True. Releasing the gate lets ``poll`` commit; the engine's top-of-loop
    ``get_run_status`` then observes PAUSED and breaks BEFORE ``close`` runs. ``unpause`` re-drives
    to COMPLETED, replaying the committed ``poll`` with NO model re-call. Deterministic: the gate is
    an ``asyncio.Event`` and the clock is fixed — no wall-clock, no sleep races.
    """
    registry = PathwayRegistry()

    class _IntakeToPoll:
        name: str = "intake"
        transitions: tuple[str, ...] = ("poll",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Transition(to="poll", output=_provenanced("intake", "intake"))

    class _CloseReadsPoll:
        name: str = "close"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            prior = await ctx.journal.read_step(ctx.run_id, 1)
            text = ""
            if prior is not None:
                output = getattr(prior.result, "output", None)
                if output is not None:
                    text = str(output.data.get("text", ""))
            return Done(output=_provenanced("closed", "close", text))

    registry.register(
        "pause-cas-pw",
        StageGraph([_IntakeToPoll(), PollStage(), _CloseReadsPoll()], entry="intake"),
    )

    journal = InMemoryJournal()
    model = _GatedModel()
    engine = Engine(
        model=model,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=registry,
        clock=_CLOCK_AT_T0,
    )

    drive = asyncio.create_task(
        engine.run(
            run_id="c5b",
            session_id="c5b-sess",
            pathway_id="pause-cas-pw",
            initial=_timer_initial(),
        )
    )

    # Wait until ``poll`` is parked inside the gated model call: the run is now RUNNING.
    await model.entered.wait()
    assert (await journal.get_run_status("c5b")) is RunStatus.RUNNING

    # pause() WHILE the stage is gated: the RUNNING->PAUSED CAS wins -> True.
    paused = await engine.pause("c5b")
    assert paused is True

    # Release the gate: poll commits, the top-of-loop check sees PAUSED and breaks before close.
    model.gate.set()
    paused_state = await drive

    assert paused_state.status is RunStatus.PAUSED
    assert tuple(step.stage_name for step in paused_state.steps) == ("intake", "poll")
    assert model.call_count == 1  # poll's gated model call happened exactly once
    assert (await journal.get_run_status("c5b")) is RunStatus.PAUSED

    # unpause drives to COMPLETED: intake + poll REPLAYED (no model re-call), only close runs anew.
    final = await engine.unpause("c5b")
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "poll", "close")
    assert model.call_count == 1  # the committed poll was replayed, never recomputed
    last_result = final.steps[-1].result
    assert isinstance(last_result, Done)
    assert last_result.output.data["text"] == "post-wait answer"

    # pause() on a now-COMPLETED run is a no-op: every CAS loses -> False (the F2 distinction).
    assert (await engine.pause("c5b")) is False


# --------------------------------------------------------------------------------------------------
# C6 — replay safety of Wait: a committed Wait replays as an advance, NEVER re-arms set_timer
# --------------------------------------------------------------------------------------------------


async def test_c6_committed_wait_never_re_arms_set_timer_on_replay() -> None:
    """``set_timer`` fires exactly ONCE (fresh Wait) across run + fire + resume — not on replay."""
    inner = InMemoryJournal()
    spy = SetTimerSpyJournal(inner)
    pathways = _timer_pathways()
    model = _scripted_model()
    build = _make_build_engine(pathways)
    engine = build(spy, model)

    # Run to the Wait: ONE set_timer (the fresh arm).
    await engine.run(
        run_id="c6",
        session_id="c6-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    assert spy.set_timer_count == 1

    # A resume while still WAITING must NOT advance and must NOT re-arm the timer.
    resumed_waiting = await engine.resume("c6")
    assert resumed_waiting.status is RunStatus.WAITING
    assert spy.set_timer_count == 1

    # Fire the timer: the committed Wait REPLAYS as an advance (cancel timer + go to ``to``),
    # re-running poll/close. The replayed Wait must NOT call set_timer again.
    await inner.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    await engine.fire_timer("c6")
    assert spy.set_timer_count == 1

    # A final resume of the completed run replays the committed Wait once more — still no re-arm.
    await engine.resume("c6")
    assert spy.set_timer_count == 1

    final = await inner.load_run("c6")
    assert final is not None
    assert final.status is RunStatus.COMPLETED


# --------------------------------------------------------------------------------------------------
# C7 — crash BETWEEN lease and fire (the architect-flagged stuck-run hole) — REQUIRED
# --------------------------------------------------------------------------------------------------


async def test_c7_crash_between_lease_and_fire_re_leases_and_completes() -> None:
    """Lease a timer, then die before firing; a stale-lease tick re-leases + fires -> terminal.

    A destructive-DELETE claim would have LOST the timer here and left the run stuck WAITING.
    This test encodes that distinction: the timer must STILL EXIST right after the abandoned lease
    (the lease sets ``claimed_at``, it does NOT delete), and a later tick past
    ``wake_at + lease_ttl`` must re-lease and fire it.
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    engine = _make_build_engine(_timer_pathways())(journal, model)

    await engine.run(
        run_id="c7",
        session_id="c7-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )

    # Lease the timer, then simulate the process dying BEFORE fire_timer makes any progress.
    leased = await journal.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert len(leased) == 1
    assert leased[0].claimed_at is not None  # the lease stamped claimed_at — it did not delete

    # The run is still WAITING (fire never ran) and the timer STILL EXISTS (lease != delete).
    assert (await journal.get_run_status("c7")) is RunStatus.WAITING
    survived = await journal.due_timers(_AFTER_WAKE)
    assert len(survived) == 1
    assert survived[0].timer_id == "c7:1"

    # A claim at the SAME instant must see the fresh (non-stale) lease and return nothing.
    not_yet_stale = await journal.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert len(not_yet_stale) == 0

    # Once the lease is stale (now >= wake_at + lease_ttl), a tick re-leases + fires it -> terminal.
    stale_now = _WAKE_AT + _LEASE_TTL + timedelta(seconds=1)
    sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(stale_now))
    fired = await sweeper.tick(stale_now, lease_ttl=_LEASE_TTL)
    assert fired == 1

    final = await journal.load_run("c7")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "wait", "poll", "close")
    assert model.call_count == 1  # the abandoned attempt never reached poll; only the re-fire did


# --------------------------------------------------------------------------------------------------
# F6 — two concurrent fire_timer on the SAME WAITING run: model called EXACTLY once (the CAS guard)
# --------------------------------------------------------------------------------------------------


class _GatedModel:
    """A ``Model`` whose first ``complete`` blocks on a gate so two fires interleave at the poll.

    Counts calls like ``ReplayModel`` but PARKS the first call on an ``asyncio.Event`` (after
    recording it) until the test releases the gate — guaranteeing both concurrent ``fire_timer``
    coroutines reach the UNCOMMITTED ``poll`` before either commits. WITHOUT the run-status CAS both
    fires execute ``poll`` and ``call_count == 2`` (the red-teamer's reproduced double-call); WITH
    the CAS the loser never reaches ``poll`` and ``call_count == 1``.
    """

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.entered = asyncio.Event()
        self.call_count = 0
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
        return ModelResponse(text="post-wait answer", model_id="gated", finish_reason="stop")


async def test_f6_concurrent_fire_timer_calls_model_exactly_once() -> None:
    """Two ``fire_timer`` on the SAME WAITING run -> model called ONCE, run COMPLETED, no dup index.

    The run-status CAS (WAITING->RUNNING) is the run-level mutual exclusion: only the winner drives
    the uncommitted, model-bearing ``poll``; the loser no-ops. This test FAILS against the pre-CAS
    engine (both drivers execute ``poll`` -> ``call_count == 2``) and passes after.
    """
    journal = InMemoryJournal()
    model = _GatedModel()
    engine = Engine(
        model=model,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_timer_pathways(),
        clock=_CLOCK_AT_T0,
    )
    await engine.run(
        run_id="f6",
        session_id="f6-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    assert (await journal.get_run_status("f6")) is RunStatus.WAITING

    # Lease the timer once (mirrors the sweeper) so both fires race purely on the run-status CAS.
    await journal.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)

    async def _release_once_entered() -> None:
        # Let both fires reach the poll await (only the CAS winner gets there), then open the gate.
        await model.entered.wait()
        model.gate.set()

    first, second, _ = await asyncio.gather(
        engine.fire_timer("f6"),
        engine.fire_timer("f6"),
        _release_once_entered(),
    )

    assert model.call_count == 1
    final = await journal.load_run("f6")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    stage_seq = tuple(step.stage_name for step in final.steps)
    assert stage_seq == ("intake", "wait", "poll", "close")
    assert len(set(step.step_index for step in final.steps)) == len(final.steps)
    # At least one fire drove to a terminal state; neither produced a duplicate step.
    assert RunStatus.COMPLETED in (first.status, second.status)


# --------------------------------------------------------------------------------------------------
# Invariants — wire the timer pathway through the auto-applied suites (S1, S6, S5)
# --------------------------------------------------------------------------------------------------


async def test_s5_wait_output_carries_provenance() -> None:
    """S5: every committed output, INCLUDING the new ``Wait`` step's output, carries provenance."""
    journal = InMemoryJournal()
    model = _scripted_model()
    engine = _make_build_engine(_timer_pathways())(journal, model)
    state = await engine.run(
        run_id="s5-wait",
        session_id="s5-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    assert state.status is RunStatus.WAITING
    # The committed Wait step is present and its output is provenance-bearing.
    wait_step = next(step for step in state.steps if step.stage_name == "wait")
    assert isinstance(wait_step.result, Wait)
    assert wait_step.result.output.provenance is not None
    assert_run_writes_carry_provenance(state)


async def test_s1_no_model_on_fire_timer_write_path() -> None:
    """S1: firing the timer through a CommitSpyJournal never calls the model on a commit path.

    Imported lazily to keep this assertion local; the spy trips if any commit increments the model
    call count. The poll model call happens INSIDE the stage run, never inside a commit.
    """
    from cogworx.testing.invariants import CommitSpyJournal

    inner = InMemoryJournal()
    model = _scripted_model()
    spy = CommitSpyJournal(inner=inner, model=model)
    pathways = _timer_pathways()
    engine = _make_build_engine(pathways)(spy, model)
    await engine.run(
        run_id="s1-wait",
        session_id="s1-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )
    await inner.claim_due_timers(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    # fire_timer re-drives through the same spy journal; no commit may invoke the model.
    await engine.fire_timer("s1-wait")
    final = await inner.load_run("s1-wait")
    assert final is not None
    assert final.status is RunStatus.COMPLETED


async def test_s6_resume_of_waiting_run_never_recalls_model() -> None:
    """S6: a WAITING run resumed on a FRESH zero-response engine does not advance or re-call."""
    shared = InMemoryJournal()
    pathways = _timer_pathways()
    scripted = _scripted_model()
    engine_a = _make_build_engine(pathways)(shared, scripted)
    await engine_a.run(
        run_id="s6-wait",
        session_id="s6-sess",
        pathway_id=_TIMER_PATHWAY_ID,
        initial=_timer_initial(),
    )

    zero = ReplayModel([])
    engine_b = _make_build_engine(pathways)(shared, zero)
    resumed = await engine_b.resume("s6-wait")
    assert resumed.status is RunStatus.WAITING  # WAITING returned as-is, non-advancing
    assert zero.call_count == 0
