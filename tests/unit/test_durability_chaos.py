"""The kill-mid-run durability/chaos test (CANON S6) — gates Phase 1.

A 3-stage chaos graph (``intake`` → ``respond`` → ``close``) is driven until a simulated crash right
after the model-bearing ``respond`` stage durably commits. A fresh engine resumes over the SAME
journal with a ZERO-response model: if resume re-called the model, ``ReplayExhaustedError`` would
fire. It does not. Resume re-drives from the graph ENTRY: ``_drive`` walks intake→respond→close, and
for each already-committed stage (``intake``, ``respond``) it reads the journaled step and replays
its stored result WITHOUT re-running the stage or re-calling the model (the replay branch
``if existing is not None`` in ``Engine._drive``); only the uncommitted ``close`` stage actually
runs (it makes no model call). The run reaches COMPLETED carrying the SAME response text ``respond``
produced before the crash, proving the committed model call was replayed, not repeated.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    CrashAfterStepJournal,
    SimulatedCrash,
    assert_resume_never_recalls_model,
)
from cogworx.testing.reference_agent import IntakeStage

_CHAOS_PATHWAY_ID = "chaos"


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
        # No model call. Read the response text the (already-committed) respond step produced. In
        # the chaos graph respond is the second stage, committed at positional step_index 1.
        prior = await ctx.journal.read_step(ctx.run_id, 1)
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


def _chaos_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_CHAOS_PATHWAY_ID, build_chaos_graph())
    return registry


def _make_build_engine(pathways: PathwayRegistry) -> Callable[[Journal, ReplayModel], Engine]:
    """A ``build_engine(journal, model)`` factory baking in the SHARED registry.

    Engine A and the FRESH engine B both come from this factory, so engine B rehydrates the chaos
    graph from the registry via the run's stored pathway pointer — true cold resume, no in-process
    graph carried."""

    def build(journal: Journal, model: ReplayModel) -> Engine:
        registry = ModelRegistry()

        def _factory(g: BudgetGuard) -> BudgetGuardedModel:
            return BudgetGuardedModel(model, g)

        registry.register_factory("default", _factory)
        return Engine(
            models=registry,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
        )

    return build


def _chaos_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=datetime.now(UTC)),
        data={"text": "hello"},
    )


async def test_kill_after_respond_resumes_exactly_once_no_model_recall() -> None:
    shared_journal = InMemoryJournal()
    scripted = ReplayModel(
        [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
    )

    final = await assert_resume_never_recalls_model(
        build_engine=_make_build_engine(_chaos_pathways()),
        initial=_chaos_initial(),
        pathway_id=_CHAOS_PATHWAY_ID,
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


async def test_cold_resume_via_registry_fresh_engine_zero_recalls() -> None:
    """A FRESH ``Engine`` resumes off only (journal + shared registry) — true cold resume (S2, S6).

    Engine A crashes after ``respond`` durably commits (the run is left RUNNING). Engine B is a
    BRAND-NEW ``Engine`` instance — it shares ONLY the ``InMemoryJournal`` and the
    ``PathwayRegistry`` with A; it carries no in-process graph. It rehydrates the chaos graph from
    the registry using the run's stored ``pathway_id`` pointer and resumes to COMPLETED with a
    zero-response model, making ZERO model re-calls.
    """
    shared_journal = InMemoryJournal()
    shared_pathways = _chaos_pathways()  # the SAME registry threaded into both engines
    build_engine = _make_build_engine(shared_pathways)

    # Engine A: run until the durable-commit-then-crash after the model-bearing respond stage.
    scripted = ReplayModel(
        [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
    )
    crash_journal = CrashAfterStepJournal(inner=shared_journal, crash_after_stage="respond")
    engine_a = build_engine(crash_journal, scripted)
    crashed = False
    try:
        await engine_a.run(
            run_id="cold-1",
            session_id="cold-sess",
            pathway_id=_CHAOS_PATHWAY_ID,
            initial=_chaos_initial(),
        )
    except SimulatedCrash:
        crashed = True
    assert crashed
    assert scripted.call_count == 1  # the one model-bearing stage ran exactly once, before crash

    mid = await shared_journal.load_run("cold-1")
    assert mid is not None
    # Crashed mid-flight, NOT terminal — so resume must actually re-drive (not short-circuit).
    assert mid.status is RunStatus.RUNNING

    # Engine B: a FRESH Engine (new instance) over the SAME journal + SAME registry, no graph held.
    zero_model = ReplayModel([])  # any model re-call would raise ReplayExhaustedError
    engine_b = build_engine(shared_journal, zero_model)
    assert engine_b is not engine_a
    resumed = await engine_b.resume("cold-1")

    assert resumed.status is RunStatus.COMPLETED
    assert tuple(step.stage_name for step in resumed.steps) == ("intake", "respond", "close")
    assert zero_model.call_count == 0  # cold resume replayed committed work, re-called no model
    close_result = resumed.steps[-1].result
    assert isinstance(close_result, Done)
    assert close_result.output.data["text"] == "durable answer"


async def test_resume_of_completed_run_is_idempotent_noop() -> None:
    journal = InMemoryJournal()
    model = ReplayModel(
        [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
    )
    engine = _make_build_engine(_chaos_pathways())(journal, model)

    state = await engine.run(
        run_id="chaos-2",
        session_id="chaos-sess",
        pathway_id=_CHAOS_PATHWAY_ID,
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
