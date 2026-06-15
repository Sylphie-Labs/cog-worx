"""Unit tests for RunContext.recall + Engine per-stage memory-policy binding (Pod 2.6).

Covers:
  - RunContext.recall with injector=None → status="unwired", empty context
  - RunContext.recall with injector wired → delegates correctly
  - bind_memory_policy: explicit arg overrides bound policy
  - bind_memory_policy: bound policy used when no explicit arg
  - bind_memory_policy: DEFAULT used when both None
  - bind_memory_policy(None) clears a previously bound policy
  - Engine with recall_stack=None → ctx.recall returns unwired
  - Engine with recall_stack wired → ctx.recall returns result from stack

asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.cost.budget import BudgetGuard
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, InjectedMemory, MemoryPolicy
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.registry import ModelRegistry
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import AssembledContext
from cogworx.recall.stack import RecallStack, default_recall_stack
from cogworx.runtime.context import RunContext
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel, echo_model

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_FIXED_CLOCK = lambda: _EPOCH  # noqa: E731


def _empty_context(budget: int = 2048) -> AssembledContext:
    return AssembledContext(chunks=(), token_count=0, budget=budget, dropped=0)


def _unwired_injected_memory() -> InjectedMemory:
    return InjectedMemory(context=_empty_context(), status="unwired")


def _ok_injected_memory() -> InjectedMemory:
    return InjectedMemory(context=_empty_context(), status="ok")


def _make_run_context(
    injector: MemoryInjector | None = None,
    model: ReplayModel | None = None,
) -> RunContext:
    """Minimal RunContext for testing recall/policy binding."""
    return RunContext(
        run_id="run-test",
        session_id="sess-test",
        model=model or echo_model("reply"),
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
        clock=_FIXED_CLOCK,
        injector=injector,
    )


def _make_mock_injector(return_value: InjectedMemory | None = None) -> MemoryInjector:
    """Build a MemoryInjector with inject replaced by an AsyncMock."""
    stack_mock = MagicMock(spec=RecallStack)
    injector = MemoryInjector(stack_mock, clock=_FIXED_CLOCK)
    injector.inject = AsyncMock(  # type: ignore[method-assign]
        return_value=return_value or _ok_injected_memory()
    )
    return injector


def _make_reference_artifact() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_EPOCH),
        data={"text": "test"},
    )


# ---------------------------------------------------------------------------
# 1. RunContext.recall with no injector → status="unwired", empty context
# ---------------------------------------------------------------------------


async def test_recall_no_injector_returns_unwired() -> None:
    ctx = _make_run_context(injector=None)
    result = await ctx.recall(RecallQuery())
    assert result.status == "unwired"


async def test_recall_no_injector_returns_empty_context() -> None:
    ctx = _make_run_context(injector=None)
    result = await ctx.recall(RecallQuery())
    assert result.context.chunks == ()
    assert result.context.token_count == 0


async def test_recall_no_injector_has_zero_latent_uses() -> None:
    ctx = _make_run_context(injector=None)
    result = await ctx.recall(RecallQuery())
    assert result.latent_uses_recorded == 0


# ---------------------------------------------------------------------------
# 2. RunContext.recall with injector → delegates to injector.inject
# ---------------------------------------------------------------------------


async def test_recall_with_injector_delegates_inject() -> None:
    injector = _make_mock_injector(_ok_injected_memory())
    ctx = _make_run_context(injector=injector)
    result = await ctx.recall(RecallQuery(text="hello"))
    assert result.status == "ok"
    injector.inject.assert_awaited_once()  # type: ignore[attr-defined]


async def test_recall_with_injector_passes_query() -> None:
    injector = _make_mock_injector()
    ctx = _make_run_context(injector=injector)
    query = RecallQuery(text="search term", k=5)
    await ctx.recall(query)
    call_args = injector.inject.call_args  # type: ignore[attr-defined]
    assert call_args.args[0] is query


# ---------------------------------------------------------------------------
# 3. bind_memory_policy: explicit policy arg overrides bound policy
# ---------------------------------------------------------------------------


async def test_recall_explicit_policy_overrides_bound() -> None:
    """An explicit policy= kwarg must win over whatever is bound on ctx."""
    bound_policy = MemoryPolicy(token_budget=111)
    explicit_policy = MemoryPolicy(token_budget=999)

    injector = _make_mock_injector()
    ctx = _make_run_context(injector=injector)
    ctx.bind_memory_policy(bound_policy)

    await ctx.recall(RecallQuery(), policy=explicit_policy)

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"] is explicit_policy


# ---------------------------------------------------------------------------
# 4. bind_memory_policy: bound policy used when no explicit arg
# ---------------------------------------------------------------------------


async def test_recall_bound_policy_used_when_no_explicit_arg() -> None:
    bound_policy = MemoryPolicy(token_budget=512)

    injector = _make_mock_injector()
    ctx = _make_run_context(injector=injector)
    ctx.bind_memory_policy(bound_policy)

    await ctx.recall(RecallQuery())

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"] is bound_policy


# ---------------------------------------------------------------------------
# 5. bind_memory_policy: DEFAULT used when both None
# ---------------------------------------------------------------------------


async def test_recall_default_policy_used_when_both_none() -> None:
    """No explicit arg, no bound policy → DEFAULT_MEMORY_POLICY is forwarded."""
    injector = _make_mock_injector()
    ctx = _make_run_context(injector=injector)
    # No bind_memory_policy called

    await ctx.recall(RecallQuery())

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"] is DEFAULT_MEMORY_POLICY


# ---------------------------------------------------------------------------
# 6. bind_memory_policy(None) clears a previously bound policy
# ---------------------------------------------------------------------------


async def test_bind_memory_policy_none_clears_previous() -> None:
    """bind_memory_policy(None) must revert the context to DEFAULT_MEMORY_POLICY behaviour."""
    bound_policy = MemoryPolicy(token_budget=111)

    injector = _make_mock_injector()
    ctx = _make_run_context(injector=injector)
    ctx.bind_memory_policy(bound_policy)
    # Now clear it
    ctx.bind_memory_policy(None)

    await ctx.recall(RecallQuery())

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    # After clearing, DEFAULT must be used (not the previously bound policy)
    assert call_kwargs["policy"] is DEFAULT_MEMORY_POLICY


async def test_bind_memory_policy_none_clears_and_different_call_uses_default() -> None:
    """Verify two calls: first with bound, then after clear with default."""
    bound = MemoryPolicy(token_budget=256)

    injector = _make_mock_injector()
    ctx = _make_run_context(injector=injector)
    ctx.bind_memory_policy(bound)

    await ctx.recall(RecallQuery())
    first_policy = injector.inject.call_args.kwargs["policy"]  # type: ignore[attr-defined]

    # Clear and call again
    ctx.bind_memory_policy(None)
    await ctx.recall(RecallQuery())
    second_policy = injector.inject.call_args.kwargs["policy"]  # type: ignore[attr-defined]

    assert first_policy is bound
    assert second_policy is DEFAULT_MEMORY_POLICY


# ---------------------------------------------------------------------------
# 7. Engine with recall_stack=None → ctx.recall returns unwired
# ---------------------------------------------------------------------------


class _RecallCaptureStage:
    """A stage that calls ctx.recall and stores the result."""

    name: str = "recall-stage"
    transitions: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.recall_result: InjectedMemory | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        self.recall_result = await ctx.recall(RecallQuery(text="test"))
        artifact = Artifact(
            kind="test-output",
            produced_by="recall-stage",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_EPOCH),
        )
        return Done(output=artifact)


async def test_engine_no_recall_stack_ctx_recall_unwired() -> None:
    stage = _RecallCaptureStage()
    graph = StageGraph([stage], entry="recall-stage")
    pathways = PathwayRegistry()
    pathways.register("test-pathway", graph, version=1)

    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
        recall_stack=None,  # no stack
    )

    await engine.run(
        run_id="run-no-stack",
        session_id="sess-1",
        pathway_id="test-pathway",
        initial=_make_reference_artifact(),
    )

    assert stage.recall_result is not None
    assert stage.recall_result.status == "unwired"
    assert stage.recall_result.context.chunks == ()


# ---------------------------------------------------------------------------
# 8. Engine with recall_stack wired → ctx.recall returns result from stack
# ---------------------------------------------------------------------------


async def test_engine_with_recall_stack_ctx_recall_returns_ok() -> None:
    """With a wired recall_stack, ctx.recall must return status='ok' (even on empty results)."""
    stage = _RecallCaptureStage()
    graph = StageGraph([stage], entry="recall-stage")
    pathways = PathwayRegistry()
    pathways.register("test-pathway", graph, version=1)

    # Build a minimal stack (empty KG — no results, but wired)
    kg = InMemoryEntityKG()
    recall_stack = default_recall_stack(entity_kg=kg)

    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
        recall_stack=recall_stack,
    )

    await engine.run(
        run_id="run-with-stack",
        session_id="sess-1",
        pathway_id="test-pathway",
        initial=_make_reference_artifact(),
    )

    assert stage.recall_result is not None
    # Wired stack with empty KG → ok status (no required_kinds in DEFAULT policy)
    assert stage.recall_result.status == "ok"


# ---------------------------------------------------------------------------
# 9. Engine per-stage binding: memory_policy on a stage is forwarded to ctx
# ---------------------------------------------------------------------------


class _PolicyCheckStage:
    """Stage that exposes memory_policy and captures what policy was forwarded."""

    name: str = "policy-check"
    transitions: tuple[str, ...] = ()
    memory_policy: MemoryPolicy = MemoryPolicy(token_budget=128, required_kinds=("claim",))

    def __init__(self) -> None:
        self.received_policy: MemoryPolicy | None = None
        self.recall_result: InjectedMemory | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        # Call recall without an explicit policy — the bound one (from memory_policy) must be used.
        self.recall_result = await ctx.recall(RecallQuery(text="q"))
        artifact = Artifact(
            kind="test",
            produced_by="policy-check",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_EPOCH),
        )
        return Done(output=artifact)


async def test_engine_stage_memory_policy_bound_before_run() -> None:
    """Engine must call bind_memory_policy(stage.memory_policy) before stage.run()."""
    stage = _PolicyCheckStage()
    graph = StageGraph([stage], entry="policy-check")
    pathways = PathwayRegistry()
    pathways.register("test-pathway", graph, version=1)

    # A wired stack (empty KG): we only care about policy forwarding, not results
    kg = InMemoryEntityKG()
    recall_stack = default_recall_stack(entity_kg=kg)

    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
        recall_stack=recall_stack,
    )

    state = await engine.run(
        run_id="run-policy",
        session_id="sess-1",
        pathway_id="test-pathway",
        initial=_make_reference_artifact(),
    )

    assert state.status is RunStatus.COMPLETED
    assert stage.recall_result is not None
    # The stage has required_kinds=("claim",) but the empty KG returns no results,
    # so status must be "below_floor"
    assert stage.recall_result.status == "below_floor"
    assert "claim" in stage.recall_result.missing_kinds


# ---------------------------------------------------------------------------
# 10. bind_memory_policy clears between stages (no cross-stage leak)
# ---------------------------------------------------------------------------


class _PolicyLeakCheckStageA:
    """First stage with an explicit memory_policy."""

    name: str = "stage-a"
    transitions: tuple[str, ...] = ("stage-b",)
    memory_policy: MemoryPolicy = MemoryPolicy(token_budget=64)

    async def run(self, ctx: StageContext) -> StageResult:
        artifact = Artifact(
            kind="a-output",
            produced_by="stage-a",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_EPOCH),
        )
        return Transition(to="stage-b", output=artifact)


class _PolicyLeakCheckStageB:
    """Second stage with NO memory_policy — must see DEFAULT after stage-a's policy is cleared."""

    name: str = "stage-b"
    transitions: tuple[str, ...] = ()
    captured_policy: MemoryPolicy | None = None  # class var — set during run

    def __init__(self) -> None:
        self.captured_policy = None

    async def run(self, ctx: StageContext) -> StageResult:
        # Access the bound policy via a recall call — capture what was forwarded to inject
        # We can't inspect _bound_policy directly from a stage, so we verify via inject kwarg
        # by sub-classing RunContext — easier: just do a real recall and check status is "ok"
        # (not "below_floor"), confirming DEFAULT is used, not stage-a's policy.
        _result = await ctx.recall(RecallQuery())
        self.__class__.captured_policy = getattr(ctx, "_bound_policy", None)
        artifact = Artifact(
            kind="b-output",
            produced_by="stage-b",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_EPOCH),
        )
        return Done(output=artifact)


async def test_stage_policy_does_not_leak_to_next_stage() -> None:
    """After stage-a completes, stage-b's bound policy must be None (cleared by engine)."""
    stage_a = _PolicyLeakCheckStageA()
    stage_b = _PolicyLeakCheckStageB()
    graph = StageGraph([stage_a, stage_b], entry="stage-a")
    pathways = PathwayRegistry()
    pathways.register("leak-test", graph, version=1)

    kg = InMemoryEntityKG()
    recall_stack = default_recall_stack(entity_kg=kg)

    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
        recall_stack=recall_stack,
    )

    state = await engine.run(
        run_id="run-leak",
        session_id="sess-1",
        pathway_id="leak-test",
        initial=_make_reference_artifact(),
    )

    assert state.status is RunStatus.COMPLETED
    # stage-b has no memory_policy → engine called bind_memory_policy(None) → _bound_policy is None
    assert _PolicyLeakCheckStageB.captured_policy is None
