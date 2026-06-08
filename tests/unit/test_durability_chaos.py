"""The kill-mid-run durability/chaos test (CANON S6) — gates Phase 1.

A 3-stage chaos graph (``intake`` → ``respond`` → ``close``) is driven until a simulated crash right
after the model-bearing ``respond`` stage durably commits. A fresh engine resumes over the SAME
journal with a ZERO-response model: if resume re-called the model, ``ReplayExhaustedError`` would
fire. It does not. Resume replays intake+respond from the journal (0 model calls), runs the
non-model ``close`` stage, and reaches COMPLETED — carrying the SAME response text ``respond``
produced before the crash, proving the model call was not repeated.
"""

from __future__ import annotations

from datetime import UTC, datetime

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import assert_resume_never_recalls_model
from cogworx.testing.reference_agent import IntakeStage


class RespondToCloseStage:
    """Model-bearing stage that transitions to ``close`` (respond variant for the 3-stage graph)."""

    name: str = "respond"
    transitions: tuple[str, ...] = ("close",)

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Respond to the intake.")]
        )
        ctx.budget.record(response.usage)
        artifact = Artifact(
            kind="response",
            produced_by="respond",
            provenance=Provenance(
                source="inference", confidence=1.0, recorded_at=datetime.now(UTC)
            ),
            data={"text": response.text or ""},
        )
        return Transition(to="close", output=artifact)


class CloseStage:
    """Terminal, no-model stage: echoes the response text it was handed via the journal."""

    name: str = "close"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        # No model call. Read the response text the (already-committed) respond step produced.
        prior = await ctx.journal.read_step(ctx.run_id, "respond")
        text = ""
        if prior is not None:
            output = getattr(prior.result, "output", None)
            if output is not None:
                text = str(output.data.get("text", ""))
        artifact = Artifact(
            kind="closed",
            produced_by="close",
            provenance=Provenance(
                source="inference", confidence=1.0, recorded_at=datetime.now(UTC)
            ),
            data={"text": text},
        )
        return Done(output=artifact)


def build_chaos_graph() -> StageGraph:
    return StageGraph([IntakeStage(), RespondToCloseStage(), CloseStage()], entry="intake")


def _build_engine(journal: Journal, model: ReplayModel) -> Engine:
    return Engine(
        model=model,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
    )


def _chaos_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=datetime.now(UTC)),
        data={"text": "hello"},
    )


def _register_graph(engine: Engine, run_id: str) -> None:
    # In-process resume holds the graph keyed by run_id (engine.run does this internally). For a
    # fresh resuming engine in Phase 0 we seed that map directly; cross-process resume is Phase 1.
    engine._graphs[run_id] = build_chaos_graph()


async def test_kill_after_respond_resumes_exactly_once_no_model_recall() -> None:
    shared_journal = InMemoryJournal()
    scripted = ReplayModel(
        [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
    )

    final = await assert_resume_never_recalls_model(
        build_engine=_build_engine,
        register_graph=_register_graph,
        graph_factory=build_chaos_graph,
        initial=_chaos_initial(),
        crash_after_stage="respond",
        shared_journal=shared_journal,
        scripted_model=scripted,
        run_id="chaos-1",
        session_id="chaos-sess",
    )

    # Resume reached the terminal Done via the non-model close stage.
    assert final.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in final.steps) == ("intake", "respond", "close")

    # The final close output incorporates the SAME text respond produced before the crash — proof
    # the committed model call was replayed, not repeated.
    close_result = final.steps[-1].result
    assert isinstance(close_result, Done)
    assert close_result.output.data["text"] == "durable answer"

    # And the model used during run A was exhausted exactly once (the scripted call), confirming the
    # single model-bearing stage committed before the crash.
    assert scripted.call_count == 1


async def test_resume_of_completed_run_is_idempotent_noop() -> None:
    journal = InMemoryJournal()
    model = ReplayModel(
        [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
    )
    engine = _build_engine(journal, model)
    graph = build_chaos_graph()

    state = await engine.run(
        run_id="chaos-2",
        session_id="chaos-sess",
        graph=graph,
        initial=_chaos_initial(),
    )
    assert state.status is RunStatus.COMPLETED
    calls_after_run = model.call_count
    assert calls_after_run == 1

    # Resuming an already-completed run is a no-op: terminal status returned, no new model calls.
    resumed = await engine.resume("chaos-2")
    assert resumed.status is RunStatus.COMPLETED
    assert resumed.steps == state.steps
    assert model.call_count == calls_after_run
