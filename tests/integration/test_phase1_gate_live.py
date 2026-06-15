"""⛓ The Phase-1 gate — END-TO-END operation-pod durability on the LIVE Timescale journal (S6).

Spike Suite 1 proved the 1.0 core on the live substrate (kill-mid-run, cold + cyclic resume); the
other integration files prove the live PRIMITIVES (timer-lease atomicity, attempt-counter
atomicity, human-input first-answer-wins). What was still missing — and what the ROADMAP's
"the full gate spans the operation pods" line means — is the END-TO-END feature paths of pods
1.1-1.4 against real Postgres/Timescale, with a FRESH Engine instance at every wake/re-drive hop
so each hop is a true cold, journal-rehydrated resume:

- G1 (1.1): ``Wait`` parks the run with a DURABLE timer row → a ``Sweeper`` on a FRESH engine
  leases it and fires → the committed ``Wait`` replays as an advance → the run completes.
- G2 (1.1): pause beats the sweeper — a PAUSED run's fired timer no-ops (CAS loses), and a later
  ``unpause`` on a FRESH engine advances past the committed ``Wait`` immediately.
- G3 (1.2): a flaky stage fails twice → each failure commits NOTHING, climbs the durable attempt
  counter, parks ``RETRYING`` on a real backoff timer → a FRESH engine's sweeper re-drives the
  SAME frozen ``seq`` each time → the third attempt succeeds → exactly one success commit.
- G4 (1.4): ``start()`` backgrounds a run over the live journal → it crashes mid-drive →
  ``handle.result()`` re-raises AND ``RUN_CRASHED`` lands on the sink → the journal still holds
  the committed prefix ``RUNNING`` → a FRESH engine cold-resumes it to ``COMPLETED``.

Every engine here carries ``ReplayModel([])`` — ANY model call raises — so the whole gate doubles
as the S6 no-model-on-replay invariant at the live tier (and S1: these paths are model-free by
construction).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.coordination.events import Event, EventType
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition, Wait
from cogworx.loop.retry import RetryPolicy
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.runtime.sweeper import Sweeper
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import CrashAfterStepJournal, SimulatedCrash

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_WAKE_DELAY = timedelta(seconds=60)
_AFTER_WAKE = _NOW + _WAKE_DELAY + timedelta(seconds=60)
_LEASE_TTL = timedelta(seconds=30)


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def prepared_journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    """A ``TimescaleJournal`` with schema ensured and data wiped per case (the shared pattern)."""
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


def _provenanced(kind: str, produced_by: str, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by=produced_by,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_NOW),
        data={"text": text},
    )


def _initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_NOW),
        data={"text": "go"},
    )


# ---------------------------------------------------------------------------
# Stages (all model-free: ReplayModel([]) makes any model call raise loud)
# ---------------------------------------------------------------------------


class _SleepStage:
    """Parks the run on a durable timer; replays as a plain advance (1.1)."""

    name: str = "sleep"
    transitions: tuple[str, ...] = ("after",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Wait(
            to="after",
            wake_at=ctx.clock() + _WAKE_DELAY,
            output=_provenanced("wait", "sleep"),
        )


class _AfterStage:
    name: str = "after"
    transitions: tuple[str, ...] = ("finish",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="finish", output=_provenanced("after", "after"))


class _FinishStage:
    name: str = "finish"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_provenanced("done", "finish"))


class _FlakyError(Exception):
    """The dev-authored retryable failure class for G3 (S9 structural classification)."""


class _FlakyStage:
    """Fails until the shared execution counter reaches 3; then transitions onward (1.2).

    The counter is SHARED across the fresh per-hop engines (each hop builds its own registry and
    stage instances from the same list) so the journal — not in-process state — is what carries
    the attempt count between hops.
    """

    name: str = "flaky"
    transitions: tuple[str, ...] = ("finish",)
    retry_policy: RetryPolicy = RetryPolicy(
        max_attempts=3,
        backoff=lambda n: timedelta(seconds=10 * n),
        retryable=(_FlakyError,),
    )

    def __init__(self, executions: list[int]) -> None:
        self._executions = executions

    async def run(self, ctx: StageContext) -> StageResult:
        self._executions.append(1)
        if len(self._executions) < 3:
            raise _FlakyError(f"transient failure #{len(self._executions)}")
        return Transition(to="finish", output=_provenanced("flaky", "flaky"))


_WAIT_PATHWAY = "gate-wait"
_RETRY_PATHWAY = "gate-retry"
_FF_PATHWAY = "gate-ff"


def _wait_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(
        _WAIT_PATHWAY, StageGraph([_SleepStage(), _AfterStage(), _FinishStage()], entry="sleep")
    )
    return registry


def _retry_pathways(executions: list[int]) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(
        _RETRY_PATHWAY, StageGraph([_FlakyStage(executions), _FinishStage()], entry="flaky")
    )
    return registry


def _ff_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_FF_PATHWAY, StageGraph([_AfterStage(), _FinishStage()], entry="after"))
    return registry


def _engine(
    journal: Journal,
    pathways: PathwayRegistry,
    *,
    clock: datetime = _NOW,
    event_sink: list[Event] | None = None,
) -> Engine:
    """A FRESH engine per call — every hop is a cold, registry+journal rehydrated drive."""
    _replay = ReplayModel([])
    _reg = ModelRegistry()
    _reg.register("default", _replay)
    return Engine(
        models=_reg,  # ANY model call raises: the live S6/S1 invariant
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=lambda: clock,
        event_sink=event_sink.append if event_sink is not None else None,
    )


# ---------------------------------------------------------------------------
# G1 — 1.1 end-to-end: Wait -> durable timer row -> fresh-engine Sweeper wake -> COMPLETED
# ---------------------------------------------------------------------------


async def test_g1_wait_sweeper_cold_wake_completes(prepared_journal: TimescaleJournal) -> None:
    """A WAITING run's durable timer is leased and fired by a Sweeper on a FRESH engine."""
    parked = await _engine(prepared_journal, _wait_pathways()).run(
        run_id="g1",
        session_id="g1-sess",
        pathway_id=_WAIT_PATHWAY,
        initial=_initial(),
    )
    assert parked.status is RunStatus.WAITING
    assert tuple(s.stage_name for s in parked.steps) == ("sleep",)

    # The wake is a different process in production: a FRESH engine + sweeper, later clock.
    waker = _engine(prepared_journal, _wait_pathways(), clock=_AFTER_WAKE)
    sweeper = Sweeper(journal=prepared_journal, fire=waker.fire_timer, clock=lambda: _AFTER_WAKE)
    fired = await sweeper.tick(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert fired == 1, "the durable timer row must be due, leased, and fired exactly once"

    final = await prepared_journal.load_run("g1")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert tuple(s.stage_name for s in final.steps) == ("sleep", "after", "finish")
    # The committed Wait was REPLAYED (one 'sleep' step), and a second tick finds nothing.
    assert await sweeper.tick(_AFTER_WAKE, lease_ttl=_LEASE_TTL) == 0


# ---------------------------------------------------------------------------
# G2 — 1.1 end-to-end: pause beats the sweeper; unpause advances past the committed Wait
# ---------------------------------------------------------------------------


async def test_g2_pause_beats_sweeper_then_unpause_advances(
    prepared_journal: TimescaleJournal,
) -> None:
    """A PAUSED run's fired timer no-ops (CAS loses); unpause on a FRESH engine completes it."""
    starter = _engine(prepared_journal, _wait_pathways())
    parked = await starter.run(
        run_id="g2",
        session_id="g2-sess",
        pathway_id=_WAIT_PATHWAY,
        initial=_initial(),
    )
    assert parked.status is RunStatus.WAITING
    assert await starter.pause("g2") is True
    assert await prepared_journal.get_run_status("g2") is RunStatus.PAUSED

    # The sweeper still leases the armed timer, but fire_timer must no-op on a PAUSED run.
    waker = _engine(prepared_journal, _wait_pathways(), clock=_AFTER_WAKE)
    sweeper = Sweeper(journal=prepared_journal, fire=waker.fire_timer, clock=lambda: _AFTER_WAKE)
    await sweeper.tick(_AFTER_WAKE, lease_ttl=_LEASE_TTL)
    assert await prepared_journal.get_run_status("g2") is RunStatus.PAUSED, (
        "G2: a fired timer advanced a PAUSED run — the WAITING->RUNNING CAS must lose on PAUSED"
    )

    # Unpause on ANOTHER fresh engine: the committed Wait replays as an immediate advance.
    final = await _engine(prepared_journal, _wait_pathways(), clock=_AFTER_WAKE).unpause("g2")
    assert final.status is RunStatus.COMPLETED
    assert tuple(s.stage_name for s in final.steps) == ("sleep", "after", "finish")


# ---------------------------------------------------------------------------
# G3 — 1.2 end-to-end: two durable retries across FRESH engines, success at the frozen seq
# ---------------------------------------------------------------------------


async def test_g3_durable_retries_across_fresh_engines(
    prepared_journal: TimescaleJournal,
) -> None:
    """Fail -> RETRYING on a real backoff timer -> fresh-engine re-drive at the SAME seq, twice."""
    executions: list[int] = []

    parked = await _engine(prepared_journal, _retry_pathways(executions)).run(
        run_id="g3",
        session_id="g3-sess",
        pathway_id=_RETRY_PATHWAY,
        initial=_initial(),
    )
    assert parked.status is RunStatus.RETRYING
    assert len(parked.steps) == 0, "G3 / S6: a FAILED attempt must commit NOTHING"
    assert await prepared_journal.read_attempt("g3", 0) == 1

    # First re-drive: a fresh engine's sweeper fires the backoff timer; the re-attempt fails
    # again at the SAME frozen seq and re-parks RETRYING on a new timer.
    hop1 = _engine(prepared_journal, _retry_pathways(executions), clock=_AFTER_WAKE)
    sweeper1 = Sweeper(journal=prepared_journal, fire=hop1.fire_timer, clock=lambda: _AFTER_WAKE)
    assert await sweeper1.tick(_AFTER_WAKE, lease_ttl=_LEASE_TTL) == 1
    assert await prepared_journal.get_run_status("g3") is RunStatus.RETRYING
    assert await prepared_journal.read_attempt("g3", 0) == 2, (
        "G3 / S6: the attempt counter must climb durably across fresh-engine re-drives"
    )

    # Second re-drive: the third execution succeeds; exactly ONE success commit lands at seq 0.
    much_later = _AFTER_WAKE + timedelta(minutes=10)
    hop2 = _engine(prepared_journal, _retry_pathways(executions), clock=much_later)
    sweeper2 = Sweeper(journal=prepared_journal, fire=hop2.fire_timer, clock=lambda: much_later)
    assert await sweeper2.tick(much_later, lease_ttl=_LEASE_TTL) == 1

    final = await prepared_journal.load_run("g3")
    assert final is not None
    assert final.status is RunStatus.COMPLETED
    assert tuple(s.stage_name for s in final.steps) == ("flaky", "finish")
    assert [s.step_index for s in final.steps] == [0, 1], (
        "G3 / S6: retries must FREEZE seq — the success lands at the failed attempts' index"
    )
    assert len(executions) == 3, "two failures + one success, each executed exactly once"


# ---------------------------------------------------------------------------
# G4 — 1.4 end-to-end: backgrounded crash over the live journal -> feedback -> cold resume
# ---------------------------------------------------------------------------


async def test_g4_fire_and_forget_crash_feedback_cold_resume(
    prepared_journal: TimescaleJournal,
) -> None:
    """start() over live Timescale: crash mid-drive -> RUN_CRASHED + re-raise -> cold COMPLETED."""
    crashing = CrashAfterStepJournal(inner=prepared_journal, crash_after_stage="after")
    events: list[Event] = []
    engine_a = _engine(crashing, _ff_pathways(), event_sink=events)

    handle = engine_a.start(
        run_id="g4",
        session_id="g4-sess",
        pathway_id=_FF_PATHWAY,
        initial=_initial(),
    )
    with pytest.raises(SimulatedCrash):
        await handle.result()
    await asyncio.sleep(0)  # done-callbacks run on the next loop tick

    crashed = [e for e in events if e.type is EventType.RUN_CRASHED]
    assert len(crashed) == 1 and crashed[0].run_id == "g4", (
        "G4: the dropped-handle caller's ONLY crash signal is the RUN_CRASHED feedback event"
    )

    # The LIVE journal holds the committed prefix, status RUNNING (crashed mid-drive).
    mid = await prepared_journal.load_run("g4")
    assert mid is not None
    assert mid.status is RunStatus.RUNNING
    assert tuple(s.stage_name for s in mid.steps) == ("after",)

    # Cold resume on a FRESH engine over the BARE live journal: replay the prefix, run only the
    # uncommitted stage, complete. ReplayModel([]) proves zero model re-calls (S6).
    final = await _engine(prepared_journal, _ff_pathways()).resume("g4")
    assert final.status is RunStatus.COMPLETED
    assert tuple(s.stage_name for s in final.steps) == ("after", "finish")
    assert len({s.step_index for s in final.steps}) == 2, "no double-commit at any seq"

    await engine_a.aclose()
