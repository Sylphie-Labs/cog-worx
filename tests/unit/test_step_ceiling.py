"""The structural step ceiling: an unbounded cyclic pathway terminates FAILED (CANON S6, S9, S11).

Positional step keying makes a cyclic pathway durable (each visit commits a distinct step), but a
pathway whose stages cycle forever would otherwise drive without end. ``Engine.max_steps`` is the
structural ceiling: a run that would loop past it is FAILED (the PERSISTED status is the authority)
and the loop emits ``RUN_FAILED`` through its sink — the runner witnesses the runaway, not the test.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.coordination.events import Event, EventType
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.runtime.engine import Clock, Engine
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel

_CYCLE_PATHWAY_ID = "cycle"
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _counter_clock() -> Clock:
    state = {"tick": 0}

    def clock() -> datetime:
        moment = _EPOCH + timedelta(seconds=state["tick"])
        state["tick"] += 1
        return moment

    return clock


def _artifact(stage: str) -> Artifact:
    return Artifact(
        kind="cycle",
        produced_by=stage,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_EPOCH),
    )


class _PingStage:
    """No-model stage that ALWAYS transitions to ``pong`` (half of the unbounded cycle).

    Declares an edge to the terminal ``end`` too, so the graph passes the by-construction
    termination check — but ``run`` never returns that edge, so the loop never ends on its own.
    """

    name: str = "ping"
    transitions: tuple[str, ...] = ("pong", "end")

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="pong", output=_artifact("ping"))


class _PongStage:
    """No-model stage that transitions back to ``ping`` forever (the other half of the cycle).

    A terminal ``end`` stage is reachable from ``ping`` (so the graph passes the by-construction
    termination check) but the stages NEVER return the edge to it at runtime — the loop cycles
    ping->pong->ping... until the engine's step ceiling fires.
    """

    name: str = "pong"
    transitions: tuple[str, ...] = ("ping",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="ping", output=_artifact("pong"))


class _EndStage:
    """A terminal stage that exists for the structural reachability check but is never entered."""

    name: str = "end"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_artifact("end"))


def _build_cyclic_graph() -> StageGraph:
    # ping -> pong -> ping (the runaway cycle); ping -> end gives a reachable terminal so the graph
    # is constructible, but no stage ever returns the edge to ``end`` at runtime.
    return StageGraph([_PingStage(), _PongStage(), _EndStage()], entry="ping")


def _cyclic_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_CYCLE_PATHWAY_ID, _build_cyclic_graph())
    return registry


def _initial() -> Artifact:
    return _artifact("input")


async def test_unbounded_cycle_terminates_failed_at_step_ceiling() -> None:
    """An unbounded cyclic pathway is FAILED at a small ``max_steps`` and emits ``RUN_FAILED``."""
    sink: list[Event] = []
    max_steps = 8
    engine = Engine(
        model=ReplayModel([]),  # the cycle is model-free; any model call would exhaust.
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_cyclic_pathways(),
        event_sink=sink.append,
        clock=_counter_clock(),
        max_steps=max_steps,
    )

    state = await engine.run(
        run_id="cycle-1",
        session_id="cycle-sess",
        pathway_id=_CYCLE_PATHWAY_ID,
        initial=_initial(),
    )

    # The persisted run status is FAILED — the ceiling stopped the runaway, the loop did not hang.
    assert state.status is RunStatus.FAILED
    # The ceiling fired before committing the (max_steps)-th step: exactly max_steps committed
    # steps, each a distinct position despite revisiting the same two stage names (per-visit key).
    assert len(state.steps) == max_steps
    assert tuple(step.step_index for step in state.steps) == tuple(range(max_steps))
    assert tuple(step.stage_name for step in state.steps) == ("ping", "pong") * (max_steps // 2)

    # The loop itself emitted RUN_FAILED (the engine witnessed the runaway, not the test).
    emitted = [event.type for event in sink]
    assert EventType.RUN_FAILED in emitted
    assert EventType.RUN_COMPLETED not in emitted
