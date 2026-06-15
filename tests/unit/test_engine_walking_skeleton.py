"""Walking-skeleton proof: the engine drives the reference agent to a clean terminal run.

Proves the S1/S6 shape structurally: exactly one model call (the respond stage), exactly two
committed steps, and the boundary-checked spine events. The ``ReplayModel`` is the spy.
"""

from __future__ import annotations

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.coordination.events import Event, EventType, validate_event_boundary
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.ladder import StructuredOutputError, StructuredOutputModel
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import RunState
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.reference_agent import REFERENCE_PATHWAY_ID, reference_initial

# ---------------------------------------------------------------------------
# Schema stage helpers for ladder tests
# ---------------------------------------------------------------------------

_SCHEMA_PATHWAY_ID = "schema-test"
_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}

_T0_EPOCH = __import__("datetime").datetime(2026, 1, 1, tzinfo=__import__("datetime").timezone.utc)


def _schema_artifact(stage: str) -> Artifact:
    return Artifact(
        kind="test",
        produced_by=stage,
        provenance=Provenance(source="system", confidence=1.0, recorded_at=_T0_EPOCH),
    )


class _IntakeToSchema:
    name: str = "intake"
    transitions: tuple[str, ...] = ("schema_stage",)

    async def run(self, ctx: StageContext) -> StageResult:
        from cogworx.loop.result import Transition

        return Transition(to="schema_stage", output=_schema_artifact("intake"))


class _SchemaStage:
    """Calls ``ctx.model.complete`` with a JSON schema and returns Done with the response text."""

    name: str = "schema_stage"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Answer me.")],
            json_schema=_SCHEMA,  # type: ignore[arg-type]
        )
        return Done(
            output=Artifact(
                kind="answer",
                produced_by="schema_stage",
                provenance=Provenance(source="inference", confidence=1.0, recorded_at=_T0_EPOCH),
                data={"text": response.text or ""},
            )
        )


def _schema_pathways() -> PathwayRegistry:
    graph = StageGraph([_IntakeToSchema(), _SchemaStage()], entry="intake")
    reg = PathwayRegistry()
    reg.register(_SCHEMA_PATHWAY_ID, graph, version=1)
    return reg


async def test_reference_agent_runs_to_completion(
    replay_model: ReplayModel,
    in_memory_journal: InMemoryJournal,
    in_memory_graph: InMemoryGraphStore,
    in_memory_latent: InMemoryLatentStore,
    pathways: PathwayRegistry,
) -> None:
    sink: list[Event] = []
    registry = ModelRegistry()
    registry.register_factory(
        "default",
        lambda g, m=replay_model: StructuredOutputModel(BudgetGuardedModel(m, g)),
    )
    engine = Engine(
        models=registry,
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


# ---------------------------------------------------------------------------
# F3 new test 2 — ladder validates schema on the drive path
# ---------------------------------------------------------------------------


async def test_engine_ladder_validates_schema_on_drive_path() -> None:
    """``StructuredOutputModel`` validates the schema on the drive path (S4/S8/S9).

    Factory: ``StructuredOutputModel(BudgetGuardedModel(inner, g))`` — the full production stack.
    ``inner`` returns valid JSON ``{"answer": "42"}``.  The rung selection is based solely on
    ``capabilities.structured_output=True`` (native rung, constrained decoding).  The response text
    is validated framework-side: it must conform to ``_SCHEMA`` before being returned.

    Asserts:
    - run COMPLETED.
    - ``inner.calls[0].json_schema`` was the schema dict passed to ``complete``.
    - The artifact data parses to ``{"answer": "42"}``.
    """
    import json

    inner = ReplayModel(
        [ModelResponse(text='{"answer": "42"}', model_id="replay", finish_reason="stop")]
    )
    registry = ModelRegistry()
    registry.register_factory(
        "default",
        lambda g, m=inner: StructuredOutputModel(BudgetGuardedModel(m, g)),
    )
    engine = Engine(
        models=registry,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_schema_pathways(),
    )

    state = await engine.run(
        run_id="schema-ok",
        session_id="schema-sess",
        pathway_id=_SCHEMA_PATHWAY_ID,
        initial=_schema_artifact("input"),
    )

    assert state.status is RunStatus.COMPLETED
    # The schema was forwarded to the inner model (native rung — structured_output=True).
    assert inner.calls[0].json_schema == dict(_SCHEMA)
    # The artifact text is the validated JSON string.
    artifact_text = state.steps[-1].result.output.data["text"]  # type: ignore[union-attr]
    assert json.loads(artifact_text) == {"answer": "42"}


# ---------------------------------------------------------------------------
# F3 new test 3 — ladder rejects invalid schema output
# ---------------------------------------------------------------------------


async def test_engine_ladder_rejects_invalid_schema_output() -> None:
    """``StructuredOutputModel`` raises ``StructuredOutputError`` for a type-violating response.

    ``inner`` returns ``{"answer": 7}`` (integer, not string) — a schema violation.  The ladder's
    native rung validates the JSON structurally (S9) and raises ``StructuredOutputError``.

    Asserts:
    - ``pytest.raises(StructuredOutputError)`` around ``engine.run``.
    - ``inner.call_count == 1`` — the model was called exactly once (no second ladder rung for
      the native path; the schema was delegated via ``json_schema`` and the response was invalid).
    """
    inner = ReplayModel(
        [ModelResponse(text='{"answer": 7}', model_id="replay", finish_reason="stop")]
    )
    registry = ModelRegistry()
    registry.register_factory(
        "default",
        lambda g, m=inner: StructuredOutputModel(BudgetGuardedModel(m, g)),
    )
    engine = Engine(
        models=registry,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_schema_pathways(),
    )

    with pytest.raises(StructuredOutputError):
        await engine.run(
            run_id="schema-bad",
            session_id="schema-sess",
            pathway_id=_SCHEMA_PATHWAY_ID,
            initial=_schema_artifact("input"),
        )

    assert inner.call_count == 1
