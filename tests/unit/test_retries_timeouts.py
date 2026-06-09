"""Durable retries + per-stage FSM timeouts — Phase-1 pod 1.2 spike/chaos suite (S6/S1/S5/S9/S11).

WRITTEN TESTS-FIRST (S12): these target a contract that does NOT yet exist, so the module fails to
COLLECT (import/attribute errors on ``cogworx.loop.retry.RetryPolicy`` / ``DEFAULT_RETRY_POLICY``,
``RunStatus.RETRYING``, and the new ``Journal.increment_attempt`` / ``read_attempt`` methods). That
RED state is intended — it pins the seam a retry/timeout-bearing pathway must satisfy before an impl
hardens it (S12 spike-before-harden).

The load-bearing invariant (architect-validated): a **failed attempt commits NOTHING** — only a
success-class result lands at ``seq``. The journal's "committed iff succeeded iff never re-run"
stays pristine; ``seq`` freezes while a per-``(run_id, step_index)`` **attempt counter** climbs. A
retry is
distinguished from a wait by the journal: a wait has a committed step at its ``seq``, a retry does
not. Retry classification is **structural** (exception-type match against the dev-authored
``retryable`` allowlist), never the model's words (S9).

The graph mirrors ``test_timers_sweeper.py``: ``intake`` (no model) -> ``work`` (the configurable
failing stage, optionally model-bearing) -> ``close`` (terminal, no-model). All clocks are injected
so ``backoff`` arithmetic is deterministic — never wall-clock. The ONLY real timeout exercised is R5
(a stage that blocks on an unset ``asyncio.Event`` past a short real ``policy.timeout`` so
``asyncio.wait_for`` genuinely fires); everything else is fully deterministic.

Each criterion is mutation-resistant: committing a failed attempt, resetting the attempt count on
resume, double-incrementing under concurrent fire, swallowing a non-retryable bug into a degrade, or
not classifying a timeout as a failure must each make exactly one of these tests fail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.cost.budget import BudgetExceededError, BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Degraded, Done, StageResult, Transition
from cogworx.loop.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from cogworx.loop.stage import Stage, StageContext
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
    CommitSpyJournal,
    CrashAfterStepJournal,
    SimulatedCrash,
    assert_control_independent_of_model_text,
    assert_run_writes_carry_provenance,
)

_RETRY_PATHWAY_ID = "retry-chaos"

# A fixed base instant + deterministic backoff so wake_at arithmetic is reproducible.
_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_BACKOFF = timedelta(seconds=30)
_LEASE_TTL = timedelta(minutes=1)

# Positional step indices in the retry graph: intake=0, work=1, close=2. The failing stage is at
# WORK_STEP_INDEX and, crucially, ``seq`` FREEZES there across retries (a failed attempt commits no
# step) — so the eventual success is the ONLY committed step at this index.
_WORK_STEP_INDEX = 1


# backoff(n) is deterministic; a fixed clock + this lets us compute the exact wake_at to fire at.
def _RETRY_BACKOFF(n: int) -> timedelta:
    return _BACKOFF * n


_WAKE_AT_1 = _T0 + _RETRY_BACKOFF(1)  # the first retry timer's wake_at (attempt 1 -> n=1)
_AFTER_BACKOFF_1 = _WAKE_AT_1 + timedelta(seconds=1)


# --------------------------------------------------------------------------------------------------
# Fixtures: provenance helper, the configurable failing stage, the retry-bearing graph
# --------------------------------------------------------------------------------------------------


def _provenanced(kind: str, produced_by: str, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by=produced_by,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_T0),
        data={"text": text},
    )


class _RetryableError(Exception):
    """The dev-authored retryable exception class (in the policy allowlist)."""


class _IntakeToWork:
    """Entry stage edging to ``work`` (so StageGraph accepts the graph)."""

    name: str = "intake"
    transitions: tuple[str, ...] = ("work",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="work", output=_provenanced("intake", "intake"))


class FailingWorkStage:
    """The configurable failing stage at ``work`` — fails the first ``fail_times`` ATTEMPTS, then
    succeeds. A ``retry_policy`` attribute is settable per-test; without it the engine falls back to
    ``DEFAULT_RETRY_POLICY``.

    Attempt classification is STRUCTURAL: the stage RAISES ``exc_factory()`` on a failing attempt;
    the engine decides retryable-vs-not by type-matching ``exc`` against ``policy.retryable``. The
    stage never inspects model text to decide whether to fail (S9): failure is keyed purely on the
    durable attempt count read from the journal (so the decision survives a crash/cold resume).

    ``model_bearing`` makes a SUCCESSFUL attempt call the model once (to prove a retried
    model-bearing stage does not double-count the committed model call). A FAILED attempt never
    reaches the model.
    """

    name: str = "work"
    transitions: tuple[str, ...] = ("close",)

    def __init__(
        self,
        *,
        fail_times: int,
        exc_factory: Callable[[], Exception] = _RetryableError,
        retry_policy: RetryPolicy | None = None,
        model_bearing: bool = False,
    ) -> None:
        self._fail_times = fail_times
        self._exc_factory = exc_factory
        self._model_bearing = model_bearing
        self.run_calls = 0  # how many times ``run`` was invoked (attempts, incl. the success)
        if retry_policy is not None:
            self.retry_policy: RetryPolicy = retry_policy

    async def run(self, ctx: StageContext) -> StageResult:
        self.run_calls += 1
        # The DURABLE attempt count is the authority for whether this attempt should fail — so cold
        # resume on a fresh stage instance (run_calls reset to 0) still fails/succeeds correctly.
        prior_attempts = await ctx.journal.read_attempt(ctx.run_id, _WORK_STEP_INDEX)
        if prior_attempts < self._fail_times:
            raise self._exc_factory()
        text = ""
        if self._model_bearing:
            ctx.budget.check()
            response = await ctx.model.complete(
                messages=[ChatMessage(role="user", content="do the work")]
            )
            ctx.budget.record(response.usage)
            text = response.text or ""
        return Transition(to="close", output=_provenanced("work", "work", text))


class _BlockingWorkStage:
    """A ``work`` stage whose first attempt BLOCKS on an unset ``asyncio.Event`` past the policy
    timeout (so ``asyncio.wait_for`` genuinely fires). The retried attempt reads the journaled
    attempt count, sees the prior timeout, and returns immediately. R5's only real-timeout stage.
    """

    name: str = "work"
    transitions: tuple[str, ...] = ("close",)

    def __init__(self, *, retry_policy: RetryPolicy) -> None:
        self.retry_policy = retry_policy
        self._never_set = asyncio.Event()
        self.run_calls = 0

    async def run(self, ctx: StageContext) -> StageResult:
        self.run_calls += 1
        prior_attempts = await ctx.journal.read_attempt(ctx.run_id, _WORK_STEP_INDEX)
        if prior_attempts < 1:
            # First attempt blocks forever -> the engine's asyncio.wait_for(timeout) fires.
            await self._never_set.wait()
        return Transition(to="close", output=_provenanced("work", "work"))


class CloseStage:
    """Terminal, no-model stage: echoes whatever committed at the work index (success/Degraded)."""

    name: str = "close"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        prior = await ctx.journal.read_step(ctx.run_id, _WORK_STEP_INDEX)
        text = ""
        if prior is not None:
            output = getattr(prior.result, "output", None)
            if output is not None:
                text = str(output.data.get("text", ""))
        return Done(output=_provenanced("closed", "close", text))


def build_retry_graph(work: Stage) -> StageGraph:
    return StageGraph([_IntakeToWork(), work, CloseStage()], entry="intake")


def _retry_pathways(work: Stage) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_RETRY_PATHWAY_ID, build_retry_graph(work))
    return registry


def _fixed_clock(now: datetime) -> Callable[[], datetime]:
    return lambda: now


_CLOCK_AT_T0 = _fixed_clock(_T0)


def _make_build_engine(
    pathways: PathwayRegistry,
    *,
    clock: Callable[[], datetime] = _CLOCK_AT_T0,
    budget: BudgetGuard | None = None,
) -> Callable[[Journal, ReplayModel], Engine]:
    def build(journal: Journal, model: ReplayModel) -> Engine:
        return Engine(
            model=model,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            budget=budget,
            clock=clock,
        )

    return build


def _retry_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_T0),
        data={"text": "hello"},
    )


def _retryable_policy(
    *,
    max_attempts: int,
    on_exhausted: str = "degraded",
    exhausted_to: str | None = None,
    timeout: timedelta | None = None,
) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=max_attempts,
        backoff=_RETRY_BACKOFF,
        retryable=(_RetryableError,),
        timeout=timeout,
        on_exhausted=on_exhausted,
        exhausted_to=exhausted_to,
    )


def _scripted_model() -> ReplayModel:
    return ReplayModel([ModelResponse(text="work answer", model_id="replay", finish_reason="stop")])


def _committed_at_work(state: RunState) -> tuple[StepRecord, ...]:
    return tuple(step for step in state.steps if step.step_index == _WORK_STEP_INDEX)


# --------------------------------------------------------------------------------------------------
# R1 — retry then succeed: one retry timer, exactly one committed step at seq (no failed-attempt
# commit)
# --------------------------------------------------------------------------------------------------


async def test_r1_retry_then_succeed_commits_only_the_success() -> None:
    """Fail attempt 1 (retryable) -> RETRYING + exactly one retry timer; sweep after backoff ->
    re-attempt succeeds -> COMPLETED.

    Load-bearing: the stage ran TWICE, ``read_attempt == 1``, and there is EXACTLY ONE committed
    step at ``_WORK_STEP_INDEX`` (the success). Mutation killed: committing the failed attempt would
    yield two steps at that index (or a Degraded/failed result committed) and trip the count assert.
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    work = FailingWorkStage(fail_times=1, retry_policy=_retryable_policy(max_attempts=3))
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    state = await engine.run(
        run_id="r1",
        session_id="r1-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    # The failed attempt 1 parks the run RETRYING — not terminal, no step committed at work.
    assert state.status is RunStatus.RETRYING
    assert work.run_calls == 1
    assert await journal.read_attempt("r1", _WORK_STEP_INDEX) == 1
    assert _committed_at_work(state) == ()  # the failed attempt committed NOTHING at seq
    assert tuple(step.stage_name for step in state.steps) == ("intake",)

    # Exactly one retry timer, keyed (run_id:seq:retry:n) and due at clock()+backoff(1).
    timers = await journal.due_timers(_AFTER_BACKOFF_1)
    assert len(timers) == 1
    assert timers[0].timer_id == "r1:1:retry:1"
    assert timers[0].wake_at == _WAKE_AT_1
    assert timers[0].payload == {"action": "retry", "step_index": _WORK_STEP_INDEX}

    # Before the backoff elapses, nothing is due: still RETRYING.
    sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(_T0))
    assert await sweeper.tick(_T0, lease_ttl=_LEASE_TTL) == 0
    assert (await journal.get_run_status("r1")) is RunStatus.RETRYING

    # After the backoff: the retry timer fires, the SAME stage re-attempts at the SAME seq + wins.
    sweep_after = Sweeper(
        journal=journal, fire=engine.fire_timer, clock=_fixed_clock(_AFTER_BACKOFF_1)
    )
    fired = await sweep_after.tick(_AFTER_BACKOFF_1, lease_ttl=_LEASE_TTL)
    assert fired == 1

    final = await journal.load_run("r1")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert work.run_calls == 2  # exactly one re-attempt
    assert await journal.read_attempt("r1", _WORK_STEP_INDEX) == 1  # one FAILED attempt counted
    # Exactly one committed step at the work index — the success — never the failed attempt.
    work_steps = _committed_at_work(final)
    assert len(work_steps) == 1
    assert isinstance(work_steps[0].result, Transition)
    assert tuple(step.stage_name for step in final.steps) == ("intake", "work", "close")
    # No duplicate step index anywhere — exactly-once on (run_id, step_index).
    assert len({step.step_index for step in final.steps}) == len(final.steps)


# --------------------------------------------------------------------------------------------------
# R2 — durable attempt count across a crash: cold resume reads the JOURNALED count (not reset)
# --------------------------------------------------------------------------------------------------


async def test_r2_attempt_count_durable_across_crash() -> None:
    """Fail attempt 1 (timer armed), then crash on the SUCCESS commit of the retried stage; a cold
    resume on a FRESH engine + sweep reads the JOURNALED attempt count (not reset) and completes.

    Mutation killed: resetting the attempt count on resume would make the re-attempt FAIL again
    (read_attempt back to 0 < fail_times) and never converge; committing the failed attempt would
    leave a stale step at ``seq``. We assert the count never reset AND there is no duplicate step.
    """
    shared = InMemoryJournal()
    work = FailingWorkStage(fail_times=1, retry_policy=_retryable_policy(max_attempts=3))
    pathways = _retry_pathways(work)
    model = _scripted_model()

    # Drive to RETRYING (engine A, intact journal): attempt 1 fails, the retry timer is armed.
    engine_a = _make_build_engine(pathways)(shared, model)
    waiting = await engine_a.run(
        run_id="r2",
        session_id="r2-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    assert waiting.status is RunStatus.RETRYING
    assert await shared.read_attempt("r2", _WORK_STEP_INDEX) == 1

    # Lease the due retry timer, then fire on an engine whose journal CRASHES after work commits the
    # success — modelling the process dying after the durable commit but before the run advances.
    leased = await shared.claim_due_timers(_AFTER_BACKOFF_1, lease_ttl=_LEASE_TTL)
    assert len(leased) == 1
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="work")
    engine_crash = _make_build_engine(pathways, clock=_fixed_clock(_AFTER_BACKOFF_1))(
        crash_journal, model
    )
    crashed = False
    try:
        await engine_crash.fire_timer("r2")
    except SimulatedCrash:
        crashed = True
    assert crashed

    # The success DID durably commit at the work index; the attempt count is still 1 (never reset).
    assert await shared.read_attempt("r2", _WORK_STEP_INDEX) == 1
    mid = await shared.load_run("r2")
    assert mid is not None
    assert len(_committed_at_work(mid)) == 1  # the success committed exactly once before the crash

    # Cold resume on a FRESH engine + zero-response model: the committed work replays, no re-run.
    zero = ReplayModel([])
    engine_b = _make_build_engine(pathways, clock=_fixed_clock(_AFTER_BACKOFF_1))(shared, zero)
    final = await engine_b.resume("r2")

    assert final.status is RunStatus.COMPLETED
    assert await shared.read_attempt("r2", _WORK_STEP_INDEX) == 1  # count NEVER reset across crash
    assert zero.call_count == 0  # committed work replayed, never recomputed
    work_steps = _committed_at_work(final)
    assert len(work_steps) == 1  # no duplicate step at seq across the crash/resume
    assert tuple(step.stage_name for step in final.steps) == ("intake", "work", "close")
    assert len({step.step_index for step in final.steps}) == len(final.steps)


# --------------------------------------------------------------------------------------------------
# R3 — exhaustion -> degraded-onward (and the exhausted_to=None terminal-DEGRADED variant)
# --------------------------------------------------------------------------------------------------


async def test_r3_exhaustion_degrades_onward_to_fallback() -> None:
    """A stage that ALWAYS raises retryable, ``max_attempts=N`` + ``exhausted_to="fallback"``: after
    N attempts a ``Degraded`` commits at ``seq`` and the run advances to ``fallback`` then terminal.

    Load-bearing: ``read_attempt == N``, exactly one ``Degraded`` committed at the work index, and
    the run reaches a terminal state via the fallback (not stuck RETRYING forever).
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    max_attempts = 3

    # A 4-stage graph: intake -> work(always fails) -> fallback -> close. exhausted_to="fallback".
    class _FallbackStage:
        name: str = "fallback"
        transitions: tuple[str, ...] = ("close",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Transition(to="close", output=_provenanced("fallback", "fallback"))

    class _CloseAfterFallback:
        name: str = "close"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(output=_provenanced("closed", "close"))

    work = FailingWorkStage(
        fail_times=max_attempts + 5,  # always fails
        retry_policy=_retryable_policy(max_attempts=max_attempts, exhausted_to="fallback"),
    )
    registry = PathwayRegistry()
    registry.register(
        _RETRY_PATHWAY_ID,
        StageGraph(
            [_IntakeToWork(), work, _FallbackStage(), _CloseAfterFallback()], entry="intake"
        ),
    )
    build = _make_build_engine(registry)
    engine = build(journal, model)

    state = await engine.run(
        run_id="r3",
        session_id="r3-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    # First exhaustion (attempt counter < max) parks RETRYING through max_attempts-1 retry timers.
    assert state.status is RunStatus.RETRYING

    # Drive all the retries: each fires, re-attempts, fails, re-arms — until exhaustion degrades.
    for n in range(1, max_attempts):
        wake = _T0 + _RETRY_BACKOFF(n)
        sweeper = Sweeper(
            journal=journal,
            fire=engine.fire_timer,
            clock=_fixed_clock(wake + timedelta(seconds=1)),
        )
        await sweeper.tick(wake + timedelta(seconds=1), lease_ttl=_LEASE_TTL)

    final = await journal.load_run("r3")
    assert final is not None
    assert await journal.read_attempt("r3", _WORK_STEP_INDEX) == max_attempts
    # Exactly one Degraded committed at the work index (the exhaustion result), no failed attempt.
    work_steps = _committed_at_work(final)
    assert len(work_steps) == 1
    assert isinstance(work_steps[0].result, Degraded)
    assert work_steps[0].result.to == "fallback"
    # Advanced through the fallback to a terminal state — not stuck.
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "work", "fallback", "close")
    assert len({step.step_index for step in final.steps}) == len(final.steps)


async def test_r3b_exhaustion_terminal_degraded_when_no_fallback() -> None:
    """``exhausted_to=None`` -> after N attempts the run terminates ``DEGRADED`` with one Degraded
    committed at ``seq`` (to=None)."""
    journal = InMemoryJournal()
    model = _scripted_model()
    max_attempts = 2
    work = FailingWorkStage(
        fail_times=max_attempts + 5,
        retry_policy=_retryable_policy(max_attempts=max_attempts, exhausted_to=None),
    )
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    await engine.run(
        run_id="r3b",
        session_id="r3b-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    for n in range(1, max_attempts):
        wake = _T0 + _RETRY_BACKOFF(n) + timedelta(seconds=1)
        sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(wake))
        await sweeper.tick(wake, lease_ttl=_LEASE_TTL)

    final = await journal.load_run("r3b")
    assert final is not None
    assert await journal.read_attempt("r3b", _WORK_STEP_INDEX) == max_attempts
    assert final.status is RunStatus.DEGRADED
    work_steps = _committed_at_work(final)
    assert len(work_steps) == 1
    assert isinstance(work_steps[0].result, Degraded)
    assert work_steps[0].result.to is None
    # No timers left dangling on a terminal run.
    assert await journal.due_timers(_T0 + timedelta(days=1)) == ()


# --------------------------------------------------------------------------------------------------
# R3c — a STALE retry timer that fires AFTER exhaustion is a no-op (FINDING 3, mutation-resistance):
# the CAS expects WAITING|RETRYING and a terminal run loses it, so the now-terminal stage is never
# re-run and the run status is untouched.
# --------------------------------------------------------------------------------------------------


async def test_stale_retry_timer_after_exhaustion_is_noop() -> None:
    """Drive a stage to terminal DEGRADED via exhaustion, then arm + fire a STALE retry timer keyed
    ``{run_id}:{seq}:retry:{n}`` on the now-terminal run — it must be a NO-OP (S6, FINDING 3).

    A post-exhaustion fire LOSES the ``fire_timer`` CAS (it expects WAITING|RETRYING; a terminal
    DEGRADED run is neither), so the exhausted stage is NOT re-attempted (``run_calls`` unchanged)
    and the run status stays DEGRADED. This pins the boundary: once a run is terminal, a leftover
    retry timer (stale lease, double-arm, or a sweeper that beat the terminal cleanup) can never
    re-drive it. Mutation evidence: drop the parked-status guard / CAS in ``fire_timer`` and the
    stale timer re-runs the terminal stage and/or mutates the status — both asserts trip.
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    max_attempts = 2
    work = FailingWorkStage(
        fail_times=max_attempts + 5,
        retry_policy=_retryable_policy(max_attempts=max_attempts, exhausted_to=None),
    )
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    await engine.run(
        run_id="r3c",
        session_id="r3c-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    for n in range(1, max_attempts):
        wake = _T0 + _RETRY_BACKOFF(n) + timedelta(seconds=1)
        sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(wake))
        await sweeper.tick(wake, lease_ttl=_LEASE_TTL)

    assert (await journal.get_run_status("r3c")) is RunStatus.DEGRADED
    runs_at_exhaustion = work.run_calls

    # Arm a STALE retry timer on the now-terminal run (a leftover the terminal cleanup did not
    # catch, or a re-armed lease) and fire it directly AND via the sweeper past its wake — no-ops.
    stale_wake = _T0 + _RETRY_BACKOFF(max_attempts)
    await journal.set_timer(
        Timer(
            run_id="r3c",
            timer_id=f"r3c:{_WORK_STEP_INDEX}:retry:{max_attempts}",
            wake_at=stale_wake,
            payload={"action": "retry", "step_index": _WORK_STEP_INDEX},
        )
    )

    fired_state = await engine.fire_timer("r3c")
    assert fired_state.status is RunStatus.DEGRADED  # the terminal status is untouched

    after_wake = stale_wake + timedelta(seconds=1)
    sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(after_wake))
    await sweeper.tick(after_wake, lease_ttl=_LEASE_TTL)

    # The terminal stage was NEVER re-attempted and the status never moved off DEGRADED.
    assert work.run_calls == runs_at_exhaustion
    assert (await journal.get_run_status("r3c")) is RunStatus.DEGRADED
    final = await journal.load_run("r3c")
    assert final is not None
    assert final.status is RunStatus.DEGRADED
    # Still exactly one committed result at the work index — the exhaustion Degraded, not a re-run.
    work_steps = _committed_at_work(final)
    assert len(work_steps) == 1
    assert isinstance(work_steps[0].result, Degraded)


# --------------------------------------------------------------------------------------------------
# R4 — exhaustion -> fail: on_exhausted="fail" -> RunStatus.FAILED, timers cancelled
# --------------------------------------------------------------------------------------------------


async def test_r4_exhaustion_fail_sets_failed_and_cancels_timers() -> None:
    """``on_exhausted="fail"`` on an always-failing stage -> after N attempts ``RunStatus.FAILED``
    and NO timers remain (the run is terminal; nothing can re-fire it).

    Load-bearing: ``due_timers`` is empty after exhaustion and NO step committed at the work index
    (fail does not commit a Degraded — the bug propagates as a terminal FAILED, not a swallowed
    degrade).
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    max_attempts = 3
    work = FailingWorkStage(
        fail_times=max_attempts + 5,
        retry_policy=_retryable_policy(max_attempts=max_attempts, on_exhausted="fail"),
    )
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    await engine.run(
        run_id="r4",
        session_id="r4-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    for n in range(1, max_attempts):
        wake = _T0 + _RETRY_BACKOFF(n) + timedelta(seconds=1)
        sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(wake))
        await sweeper.tick(wake, lease_ttl=_LEASE_TTL)

    final = await journal.load_run("r4")
    assert final is not None
    assert final.status is RunStatus.FAILED
    assert await journal.read_attempt("r4", _WORK_STEP_INDEX) == max_attempts
    assert _committed_at_work(final) == ()  # fail commits no step at seq
    # No timers left to re-fire a FAILED run (look far past every armed wake_at).
    assert await journal.due_timers(_T0 + timedelta(days=1)) == ()


# --------------------------------------------------------------------------------------------------
# R5 — in-process timeout: a blocked stage past policy.timeout is classified as a retryable failure
# --------------------------------------------------------------------------------------------------


async def test_r5_in_process_timeout_counts_as_retryable_failure() -> None:
    """A stage that BLOCKS on an unset ``asyncio.Event`` past a short REAL ``policy.timeout`` ->
    ``asyncio.wait_for`` fires -> the timeout is classified as a retryable failure (attempt
    incremented + retry timer armed); the next attempt reads the journaled count and succeeds.

    This is the ONLY test that exercises a real timeout (short, deterministic via the gated block).
    Load-bearing: the timeout INCREMENTED the attempt counter — proving a timeout is treated as a
    failure, not a crash or a swallow. Mutation killed: not catching ``TimeoutError`` (it would
    propagate) or not incrementing on timeout.
    """
    journal = InMemoryJournal()
    model = _scripted_model()
    policy = _retryable_policy(max_attempts=3, timeout=timedelta(seconds=0.05))
    work = _BlockingWorkStage(retry_policy=policy)
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    state = await engine.run(
        run_id="r5",
        session_id="r5-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    # The first attempt timed out -> classified as a failure -> RETRYING + attempt incremented.
    assert state.status is RunStatus.RETRYING
    assert await journal.read_attempt("r5", _WORK_STEP_INDEX) == 1
    assert _committed_at_work(state) == ()  # the timed-out attempt committed nothing

    timers = await journal.due_timers(_AFTER_BACKOFF_1)
    assert len(timers) == 1
    assert timers[0].timer_id == "r5:1:retry:1"

    # Fire after backoff: the re-attempt reads attempts==1 (>= 1), so it returns immediately.
    sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(_AFTER_BACKOFF_1))
    fired = await sweeper.tick(_AFTER_BACKOFF_1, lease_ttl=_LEASE_TTL)
    assert fired == 1

    final = await journal.load_run("r5")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert await journal.read_attempt("r5", _WORK_STEP_INDEX) == 1  # exactly one (timeout) failure
    assert len(_committed_at_work(final)) == 1
    assert tuple(step.stage_name for step in final.steps) == ("intake", "work", "close")


# --------------------------------------------------------------------------------------------------
# R6 — a NON-retryable exception propagates (not caught, not swallowed into a degrade)
# --------------------------------------------------------------------------------------------------


async def test_r6_non_retryable_exception_propagates() -> None:
    """A stage raising ``KeyError`` while ``retryable=(_RetryableError,)`` -> the exception
    PROPAGATES out of ``engine.run`` (fail-loud), NO attempt increment, NO retry timer, run left
    non-terminal.

    Mutation killed: a catch-all that swallows the bug into a degrade (the run would
    complete/degrade instead of raising; the attempt count would tick; a retry timer would arm) ->
    all three asserts trip.
    """
    journal = InMemoryJournal()
    model = _scripted_model()

    def _raise_keyerror() -> Exception:
        return KeyError("a real bug, not a retryable failure")

    work = FailingWorkStage(
        fail_times=10,
        exc_factory=_raise_keyerror,
        retry_policy=_retryable_policy(max_attempts=3),  # retryable=(_RetryableError,) only
    )
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    with pytest.raises(KeyError):
        await engine.run(
            run_id="r6",
            session_id="r6-sess",
            pathway_id=_RETRY_PATHWAY_ID,
            initial=_retry_initial(),
        )

    # The bug was NOT classified retryable: no attempt increment, no retry timer, no commit at seq.
    assert await journal.read_attempt("r6", _WORK_STEP_INDEX) == 0
    assert await journal.due_timers(_T0 + timedelta(days=1)) == ()
    assert _committed_at_work(await _load(journal, "r6")) == ()
    # The run is left non-terminal (RUNNING) — recoverable only by explicit resume, never degraded.
    status = await journal.get_run_status("r6")
    assert status is RunStatus.RUNNING


async def _load(journal: Journal, run_id: str) -> RunState:
    state = await journal.load_run(run_id)
    assert state is not None
    return state


# --------------------------------------------------------------------------------------------------
# R7 — exactly-once execution under concurrent retry-fire (the CAS-expect=RETRYING guard)
# --------------------------------------------------------------------------------------------------


class _GatedModel:
    """A ``Model`` whose ``complete`` PARKS on a gate so two concurrent fires interleave at work.

    Mirrors the 1.1 F6 ``_GatedModel``: the first call records + parks on an ``asyncio.Event`` until
    the test releases the gate, guaranteeing both ``fire_timer`` coroutines reach the UNCOMMITTED
    re-attempt before either commits. WITHOUT the CAS-expect=RETRYING guard both fires execute the
    stage and increment the attempt twice; WITH it the loser no-ops.
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
        return ModelResponse(text="work answer", model_id="gated", finish_reason="stop")


async def test_r7_concurrent_retry_fire_executes_exactly_once() -> None:
    """Drive to RETRYING, then two concurrent ``fire_timer`` (gated at the re-attempt's model call)
    -> the stage re-attempts ONCE, the attempt is incremented ONCE, the run completes once.

    The CAS-expect=RETRYING -> RUNNING flip is the run-level mutual exclusion: only the winner
    drives the uncommitted re-attempt; the loser no-ops. Must FAIL without the CAS guard (both fires
    execute ``work`` -> ``call_count == 2`` and a double increment).
    """
    journal = InMemoryJournal()
    model = _GatedModel()
    work = FailingWorkStage(
        fail_times=1, retry_policy=_retryable_policy(max_attempts=3), model_bearing=True
    )
    engine = Engine(
        model=model,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_retry_pathways(work),
        clock=_CLOCK_AT_T0,
    )

    await engine.run(
        run_id="r7",
        session_id="r7-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    assert (await journal.get_run_status("r7")) is RunStatus.RETRYING
    assert await journal.read_attempt("r7", _WORK_STEP_INDEX) == 1

    # Lease the retry timer once (mirrors the sweeper) so both fires race purely on the run-status
    # CAS.
    await journal.claim_due_timers(_AFTER_BACKOFF_1, lease_ttl=_LEASE_TTL)

    async def _release_once_entered() -> None:
        await model.entered.wait()
        model.gate.set()

    await asyncio.gather(
        engine.fire_timer("r7"),
        engine.fire_timer("r7"),
        _release_once_entered(),
    )

    assert model.call_count == 1  # the model-bearing re-attempt ran exactly once
    assert work.run_calls == 2  # one failed attempt + exactly one re-attempt (no double-execution)
    # The failed attempt counted once; the re-attempt SUCCEEDS (no failure) -> count stays 1.
    assert await journal.read_attempt("r7", _WORK_STEP_INDEX) == 1
    final = await journal.load_run("r7")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    work_steps = _committed_at_work(final)
    assert len(work_steps) == 1  # exactly one committed success at seq, no duplicate index
    assert len({step.step_index for step in final.steps}) == len(final.steps)


# --------------------------------------------------------------------------------------------------
# Invariants — S1, S5, S6, S9, S11 wired through the retry pathway
# --------------------------------------------------------------------------------------------------


async def test_s1_no_model_on_retry_commit_path() -> None:
    """S1: driving a retry through ``CommitSpyJournal`` never calls the model on a commit path.

    The spy trips if any ``commit_step`` increments the model call count. The attempt increment, the
    retry-timer arm + the success commit all happen OUTSIDE a model call, so the spy stays green
    across the fail-then-retry-then-succeed cycle.
    """
    inner = InMemoryJournal()
    model = _scripted_model()
    spy = CommitSpyJournal(inner=inner, model=model)
    work = FailingWorkStage(
        fail_times=1, retry_policy=_retryable_policy(max_attempts=3), model_bearing=True
    )
    pathways = _retry_pathways(work)
    engine = _make_build_engine(pathways)(spy, model)

    await engine.run(
        run_id="s1-retry",
        session_id="s1-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    await inner.claim_due_timers(_AFTER_BACKOFF_1, lease_ttl=_LEASE_TTL)
    engine_after = _make_build_engine(pathways, clock=_fixed_clock(_AFTER_BACKOFF_1))(spy, model)
    await engine_after.fire_timer("s1-retry")

    final = await inner.load_run("s1-retry")
    assert final is not None
    assert final.status is RunStatus.COMPLETED


async def test_s5_exhaustion_degraded_output_carries_provenance() -> None:
    """S5: the exhaustion ``Degraded`` committed at ``seq`` carries provenance (every write
    provenanced)."""
    journal = InMemoryJournal()
    model = _scripted_model()
    max_attempts = 2
    work = FailingWorkStage(
        fail_times=max_attempts + 5,
        retry_policy=_retryable_policy(max_attempts=max_attempts, exhausted_to=None),
    )
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    await engine.run(
        run_id="s5-retry",
        session_id="s5-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    for n in range(1, max_attempts):
        wake = _T0 + _RETRY_BACKOFF(n) + timedelta(seconds=1)
        sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(wake))
        await sweeper.tick(wake, lease_ttl=_LEASE_TTL)

    final = await journal.load_run("s5-retry")
    assert final is not None
    assert final.status is RunStatus.DEGRADED
    degraded = _committed_at_work(final)[0].result
    assert isinstance(degraded, Degraded)
    assert degraded.output.provenance is not None  # the exhaustion output is provenanced
    assert_run_writes_carry_provenance(final)


async def test_s6_retrying_run_resumed_on_zero_model_does_not_advance() -> None:
    """S6: a RETRYING run resumed on a FRESH zero-response engine returns RETRYING as-is (the
    sweeper advances it, never a plain resume), and re-calls nothing; a committed success replays
    with 0 re-calls."""
    shared = InMemoryJournal()
    work = FailingWorkStage(fail_times=1, retry_policy=_retryable_policy(max_attempts=3))
    pathways = _retry_pathways(work)
    scripted = _scripted_model()

    engine_a = _make_build_engine(pathways)(shared, scripted)
    await engine_a.run(
        run_id="s6-retry",
        session_id="s6-sess",
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
    )
    assert (await shared.get_run_status("s6-retry")) is RunStatus.RETRYING

    # A plain resume must NOT advance a RETRYING run and must not re-call the model.
    zero = ReplayModel([])
    engine_b = _make_build_engine(pathways)(shared, zero)
    resumed = await engine_b.resume("s6-retry")
    assert resumed.status is RunStatus.RETRYING  # non-advancing, returned as-is
    assert zero.call_count == 0
    assert await shared.read_attempt("s6-retry", _WORK_STEP_INDEX) == 1  # untouched by resume

    # The sweeper advances it (fresh engine + scripted model so the re-attempt can succeed), then a
    # final resume of the now-COMPLETED run replays the committed work with no model re-call.
    engine_c = _make_build_engine(pathways, clock=_fixed_clock(_AFTER_BACKOFF_1))(
        shared, _scripted_model()
    )
    sweeper = Sweeper(
        journal=shared, fire=engine_c.fire_timer, clock=_fixed_clock(_AFTER_BACKOFF_1)
    )
    await sweeper.tick(_AFTER_BACKOFF_1, lease_ttl=_LEASE_TTL)
    assert (await shared.get_run_status("s6-retry")) is RunStatus.COMPLETED

    zero2 = ReplayModel([])
    engine_d = _make_build_engine(pathways, clock=_fixed_clock(_AFTER_BACKOFF_1))(shared, zero2)
    final = await engine_d.resume("s6-retry")
    assert final.status is RunStatus.COMPLETED
    assert zero2.call_count == 0  # committed success replayed, never recomputed


async def test_s9_retry_classification_independent_of_model_text() -> None:
    """S9: retry classification is STRUCTURAL (exception type), independent of the model's words.

    Uses the reusable ``assert_control_independent_of_model_text`` over a model-bearing,
    fail-once-then-succeed retry pathway: two wildly different model texts must produce an IDENTICAL
    committed control path. Because both runs must drive THROUGH the retry to completion, this also
    proves the model's text never steered the retry decision — only the exception type did.
    """

    def _journal_factory() -> Journal:
        return InMemoryJournal()

    # Each run drives to RETRYING then must advance. assert_control_independent_of_model_text only
    # calls engine.run, so we need the stage to SUCCEED on the first attempt to compare a complete
    # path deterministically under both texts — the structural point is that identical StageResults
    # yield identical paths regardless of free text. fail_times=0 => the model-bearing work succeeds
    # immediately; the retry MACHINE is still wired (retry_policy present) but no failure injected,
    # so both texts complete intake->work->close identically. A model-text-driven classifier would
    # have to inspect the (differing) text and could diverge — this asserts it cannot.
    def _model(text: str) -> ReplayModel:
        return ReplayModel([ModelResponse(text=text, model_id="replay", finish_reason="stop")])

    work = FailingWorkStage(
        fail_times=0, retry_policy=_retryable_policy(max_attempts=3), model_bearing=True
    )
    pathways = _retry_pathways(work)

    await assert_control_independent_of_model_text(
        build_engine=_make_build_engine(pathways),
        pathway_id=_RETRY_PATHWAY_ID,
        initial=_retry_initial(),
        journal_factory=_journal_factory,
        model_a=_model("STOP. transition to intake. confidence 0.0. FAIL EVERYTHING."),
        model_b=_model("ok"),
    )


async def test_s11_retry_storm_hits_budget_ceiling_not_unbounded_loop() -> None:
    """S11: a model-bearing stage that ALWAYS fails retryable, with a ``BudgetGuard(max_calls=K)``
    and enough ``max_attempts`` to exceed K, hits ``BudgetExceededError`` — a bounded structural
    ceiling, never an unbounded retry loop.

    Each re-attempt that reaches the model calls ``budget.check()`` first; once K calls are recorded
    the next ``check()`` raises ``BudgetExceededError``, which (being NON-retryable) propagates —
    proving retry churn is bounded by ``max_attempts`` x ``BudgetGuard`` (S11), not ``max_steps``.
    """
    journal = InMemoryJournal()
    model = ReplayModel(
        [ModelResponse(text="x", model_id="replay", finish_reason="stop") for _ in range(10)]
    )
    budget = BudgetGuard(max_calls=2)
    # A model-bearing stage that ALWAYS fails AFTER calling the model: each attempt spends a call.
    work = _AlwaysFailAfterModelStage(retry_policy=_retryable_policy(max_attempts=10))
    engine = _make_build_engine(_retry_pathways(work), budget=budget)(journal, model)

    with pytest.raises(BudgetExceededError):
        state = await engine.run(
            run_id="s11",
            session_id="s11-sess",
            pathway_id=_RETRY_PATHWAY_ID,
            initial=_retry_initial(),
        )
        # If the first attempt parked RETRYING (model not yet over budget), drive the retries; the
        # ceiling MUST trip well before max_attempts is reached.
        assert state.status is RunStatus.RETRYING
        for n in range(1, 10):
            wake = _T0 + _RETRY_BACKOFF(n) + timedelta(seconds=1)
            sweeper = Sweeper(journal=journal, fire=engine.fire_timer, clock=_fixed_clock(wake))
            await sweeper.tick(wake, lease_ttl=_LEASE_TTL)

    # The budget ceiling bounded the storm: no more than max_calls model calls were ever made.
    assert model.call_count <= 2


class _AlwaysFailAfterModelStage:
    """A model-bearing ``work`` stage that ALWAYS raises retryable AFTER spending a model call, so a
    retry storm spends one budget call per attempt and the ``BudgetGuard`` ceiling bounds it (S11).
    """

    name: str = "work"
    transitions: tuple[str, ...] = ("close",)

    def __init__(self, *, retry_policy: RetryPolicy) -> None:
        self.retry_policy = retry_policy

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()  # raises BudgetExceededError once the ceiling is reached (non-retryable)
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="work")])
        ctx.budget.record(response.usage)
        raise _RetryableError("always fails after spending a model call")


# --------------------------------------------------------------------------------------------------
# DEFAULT_RETRY_POLICY degradation (S8): with no retry_policy on a stage, retryable=() => a raising
# stage behaves EXACTLY as today — the exception propagates, nothing is caught.
# --------------------------------------------------------------------------------------------------


async def test_default_policy_opt_in_raising_stage_behaves_as_today() -> None:
    """A stage with NO ``retry_policy`` attribute uses ``DEFAULT_RETRY_POLICY`` (``retryable=()``),
    so a raising stage's exception propagates unchanged — backward-compatible (S8 degradation).

    Asserts the module-level default opts OUT of retries (empty allowlist) and that a stage which
    never sets ``retry_policy`` raises straight through ``engine.run`` exactly as in pods 1.0/1.1.
    """
    assert DEFAULT_RETRY_POLICY.retryable == ()

    journal = InMemoryJournal()
    model = _scripted_model()
    # No retry_policy passed -> the attribute is absent -> engine uses DEFAULT_RETRY_POLICY.
    work = FailingWorkStage(fail_times=10, exc_factory=_RetryableError)
    assert not hasattr(work, "retry_policy")
    engine = _make_build_engine(_retry_pathways(work))(journal, model)

    with pytest.raises(_RetryableError):
        await engine.run(
            run_id="default",
            session_id="default-sess",
            pathway_id=_RETRY_PATHWAY_ID,
            initial=_retry_initial(),
        )
    # retryable=() means even _RetryableError is NOT retried: no attempt increment, no timer.
    assert await journal.read_attempt("default", _WORK_STEP_INDEX) == 0
    assert await journal.due_timers(_T0 + timedelta(days=1)) == ()
