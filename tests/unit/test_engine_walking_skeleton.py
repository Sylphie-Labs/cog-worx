"""Walking-skeleton proof: the engine drives the reference agent to a clean terminal run.

Proves the S1/S6 shape structurally: exactly one model call (the respond stage), exactly two
committed steps, and the boundary-checked spine events. The ``ReplayModel`` is the spy.
"""

from __future__ import annotations

from cogworx.coordination.events import Event, EventType, validate_event_boundary
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done
from cogworx.loop.state import RunStatus
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import RunState
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.reference_agent import REFERENCE_PATHWAY_ID, reference_initial


async def test_reference_agent_runs_to_completion(
    replay_model: ReplayModel,
    in_memory_journal: InMemoryJournal,
    in_memory_graph: InMemoryGraphStore,
    in_memory_latent: InMemoryLatentStore,
    pathways: PathwayRegistry,
) -> None:
    sink: list[Event] = []
    engine = Engine(
        model=replay_model,
        journal=in_memory_journal,
        graph_store=in_memory_graph,
        latent=in_memory_latent,
        pathways=pathways,
        event_sink=sink.append,
    )

    state: RunState = await engine.run(
        run_id="run-1",
        session_id="sess-1",
        pathway_id=REFERENCE_PATHWAY_ID,
        initial=reference_initial(),
    )

    assert state.status is RunStatus.COMPLETED
    assert len(state.steps) == 2
    assert tuple(step.stage_name for step in state.steps) == ("intake", "respond")

    assert replay_model.call_count == 1
    assert replay_model.calls[0].messages[0].content == "Respond to the intake."

    final_result = state.steps[-1].result
    assert isinstance(final_result, Done)
    assert final_result.output.data["text"] == "hello from replay"

    for event in sink:
        validate_event_boundary(event)
    emitted = [event.type for event in sink]
    assert EventType.RUN_STARTED in emitted
    assert EventType.STEP_COMMITTED in emitted
    assert EventType.RUN_COMPLETED in emitted
    assert emitted.count(EventType.STEP_COMMITTED) == 2
