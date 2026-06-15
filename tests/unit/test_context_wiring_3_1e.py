"""Unit tests for Pod 3.1e — ContextAssembler wired into StageContext / engine.

Covers:
  1. Reference-agent integration: a stage calls assemble_context → feeds assembled messages +
     tools to ctx.model.complete; run COMPLETED; assembled messages have expected banded structure.
  2. Unwired degrade (S8): RunContext with assembler=None → assemble_context returns
     status="unwired" in slot reports; task-only context; run still completes.
  3. Per-stage context_policy binding: a stage with a context_policy attribute → that policy
     is used; binding cleared between stages (no leak).  Mirrors the 2.6 bind_memory_policy test.
  4. S6 determinism: same request → byte-identical assembled messages through the engine path.

asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.context.assembler import ContextAssembler
from cogworx.context.errors import ContextBudgetError
from cogworx.context.types import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_SLOTS,
    AssembledCallContext,
    ContextPolicy,
    ContextRequest,
)
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import Stage, StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.context import RunContext
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import (
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel, echo_model

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC)
_FIXED_CLOCK = lambda: _EPOCH  # noqa: E731


def _make_artifact(stage: str) -> Artifact:
    return Artifact(
        kind="test",
        produced_by=stage,
        provenance=Provenance(source="system", confidence=1.0, recorded_at=_EPOCH),
    )


def _make_run_context(
    assembler: ContextAssembler | None = None,
    model: ReplayModel | None = None,
) -> RunContext:
    """Minimal RunContext for testing assemble_context / policy binding."""
    return RunContext(
        run_id="run-test",
        session_id="sess-test",
        model=model or echo_model("reply"),
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
        clock=_FIXED_CLOCK,
        assembler=assembler,
    )


def _make_engine(
    stages: list[Stage],
    *,
    pathway_id: str = "test-pathway",
    model: ReplayModel | None = None,
) -> tuple[Engine, PathwayRegistry]:
    graph = StageGraph(stages, entry=stages[0].name)
    pathways = PathwayRegistry()
    pathways.register(pathway_id, graph, version=1)

    _m = model or echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
    )
    return engine, pathways


def _initial_artifact() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_EPOCH),
        data={"text": "hello"},
    )


# ---------------------------------------------------------------------------
# Part 1 — Reference-agent integration: assemble_context end-to-end
# ---------------------------------------------------------------------------


class _AssembleContextStage:
    """Stage that calls assemble_context and feeds the assembled messages to model.complete."""

    name: str = "assemble-stage"
    transitions: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.assembled: AssembledCallContext | None = None
        self.model_call_messages: tuple[ChatMessage, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        request = ContextRequest(task="Describe the context assembly.", instructions="Be concise.")
        self.assembled = await ctx.assemble_context(request)
        # Feed assembled messages directly to the model (the key wiring test)
        response = await ctx.model.complete(
            messages=list(self.assembled.messages),
            tools=list(self.assembled.tools),
        )
        self.model_call_messages = self.assembled.messages
        ctx.budget.record(response.usage)
        return Done(output=_make_artifact("assemble-stage"))


async def test_integration_assemble_context_run_completes() -> None:
    """Stage using assemble_context drives to COMPLETED with the engine-wired assembler."""
    stage = _AssembleContextStage()
    # Engine always wires the assembler; give the model a scripted response
    _m = ReplayModel(
        [ModelResponse(text="assembled-reply", model_id="replay", finish_reason="stop")]
    )
    _r = ModelRegistry()
    _r.register("default", _m)
    pathways = PathwayRegistry()
    pathways.register(
        "assemble-pathway",
        StageGraph([stage], entry="assemble-stage"),
        version=1,
    )
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
    )
    state = await engine.run(
        run_id="run-assemble-1",
        session_id="sess-1",
        pathway_id="assemble-pathway",
        initial=_initial_artifact(),
    )
    assert state.status is RunStatus.COMPLETED
    assert stage.assembled is not None


async def test_integration_assembled_messages_banded_structure() -> None:
    """Assembled messages follow banded structure: system (instructions) + user (task)."""
    stage = _AssembleContextStage()
    _m = ReplayModel([ModelResponse(text="ok", model_id="replay", finish_reason="stop")])
    _r = ModelRegistry()
    _r.register("default", _m)
    pathways = PathwayRegistry()
    pathways.register(
        "banded-pathway",
        StageGraph([stage], entry="assemble-stage"),
        version=1,
    )
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
    )
    await engine.run(
        run_id="run-banded",
        session_id="sess-1",
        pathway_id="banded-pathway",
        initial=_initial_artifact(),
    )

    assert stage.assembled is not None
    messages = stage.assembled.messages
    # With instructions="Be concise." → system message should contain it
    # Task is required (tail band) → must appear in messages
    assert len(messages) >= 1
    task_msg = messages[-1]
    assert task_msg.role == "user"
    assert "Describe the context assembly." in task_msg.content

    # If instructions contributed, there should be a system message
    system_msgs = [m for m in messages if m.role == "system"]
    assert len(system_msgs) >= 1
    assert "Be concise." in system_msgs[0].content


async def test_integration_assembled_model_call_uses_assembled_messages() -> None:
    """The model call receives the assembled messages, not raw strings."""
    stage = _AssembleContextStage()
    _m = ReplayModel([ModelResponse(text="ok", model_id="replay", finish_reason="stop")])
    _r = ModelRegistry()
    _r.register("default", _m)
    pathways = PathwayRegistry()
    pathways.register(
        "msg-pathway",
        StageGraph([stage], entry="assemble-stage"),
        version=1,
    )
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
    )
    await engine.run(
        run_id="run-msg",
        session_id="sess-1",
        pathway_id="msg-pathway",
        initial=_initial_artifact(),
    )

    # The model received the assembled messages
    assert _m.call_count == 1
    call = _m.calls[0]
    assert call.messages == stage.model_call_messages


# ---------------------------------------------------------------------------
# Part 2 — Unwired degrade (S8): assembler=None → status="unwired" slot report
# ---------------------------------------------------------------------------


async def test_unwired_assemble_context_returns_unwired_status() -> None:
    """RunContext with assembler=None → first slot report has status='unwired'."""
    ctx = _make_run_context(assembler=None)
    request = ContextRequest(task="Test task.")
    result = await ctx.assemble_context(request)
    assert result.slots[0].status == "unwired"


async def test_unwired_assemble_context_task_in_messages() -> None:
    """Unwired degrade includes the task in the final user message."""
    ctx = _make_run_context(assembler=None)
    request = ContextRequest(task="My task text.")
    result = await ctx.assemble_context(request)
    assert any(m.role == "user" and "My task text." in m.content for m in result.messages)


async def test_unwired_assemble_context_instructions_in_messages() -> None:
    """Unwired degrade includes instructions as a system message when present."""
    ctx = _make_run_context(assembler=None)
    request = ContextRequest(task="Task.", instructions="System instruction.")
    result = await ctx.assemble_context(request)
    system_msgs = [m for m in result.messages if m.role == "system"]
    assert len(system_msgs) == 1
    assert "System instruction." in system_msgs[0].content


async def test_unwired_assemble_context_run_completes() -> None:
    """Engine with no recall_stack still drives to COMPLETED when stage uses assemble_context."""

    class _UnwiredStage:
        name: str = "unwired-stage"
        transitions: tuple[str, ...] = ()

        def __init__(self) -> None:
            self.result: AssembledCallContext | None = None

        async def run(self, ctx: StageContext) -> StageResult:
            self.result = await ctx.assemble_context(ContextRequest(task="Hello."))
            return Done(output=_make_artifact("unwired-stage"))

    stage = _UnwiredStage()
    # Engine always builds the assembler (task-only works without recall_stack)
    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    pathways = PathwayRegistry()
    pathways.register(
        "unwired-pathway",
        StageGraph([stage], entry="unwired-stage"),
        version=1,
    )
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
        recall_stack=None,
    )
    state = await engine.run(
        run_id="run-unwired",
        session_id="sess-1",
        pathway_id="unwired-pathway",
        initial=_initial_artifact(),
    )
    assert state.status is RunStatus.COMPLETED
    assert stage.result is not None
    # Task message present even without assembler wired
    assert any(m.role == "user" and "Hello." in m.content for m in stage.result.messages)


async def test_assemble_context_no_assembler_explicit_policy_used() -> None:
    """Unwired: explicit policy is still respected for budget in the degrade path."""
    ctx = _make_run_context(assembler=None)
    request = ContextRequest(task="Task.")
    tight_policy = ContextPolicy(total_budget=100)
    result = await ctx.assemble_context(request, policy=tight_policy)
    assert result.budget == 100


# ---------------------------------------------------------------------------
# Part 3 — Per-stage context_policy binding (mirrors bind_memory_policy tests)
# ---------------------------------------------------------------------------


async def test_bind_context_policy_sets_policy() -> None:
    """bind_context_policy stores the policy for use in assemble_context."""
    assembler = ContextAssembler(slots=DEFAULT_SLOTS)
    ctx = _make_run_context(assembler=assembler)
    policy = ContextPolicy(total_budget=512)
    ctx.bind_context_policy(policy)
    request = ContextRequest(task="Test.")
    result = await ctx.assemble_context(request)
    # The bound policy's budget should be used
    assert result.budget == 512


async def test_bind_context_policy_none_clears() -> None:
    """bind_context_policy(None) reverts to DEFAULT_CONTEXT_POLICY."""
    assembler = ContextAssembler(slots=DEFAULT_SLOTS)
    ctx = _make_run_context(assembler=assembler)
    ctx.bind_context_policy(ContextPolicy(total_budget=512))
    ctx.bind_context_policy(None)
    request = ContextRequest(task="Test.")
    result = await ctx.assemble_context(request)
    # After clearing, DEFAULT budget should be used
    assert result.budget == DEFAULT_CONTEXT_POLICY.total_budget


async def test_explicit_policy_overrides_bound_context_policy() -> None:
    """Explicit policy= arg wins over bound stage policy."""
    assembler = ContextAssembler(slots=DEFAULT_SLOTS)
    ctx = _make_run_context(assembler=assembler)
    ctx.bind_context_policy(ContextPolicy(total_budget=512))
    explicit = ContextPolicy(total_budget=999)
    request = ContextRequest(task="Test.")
    result = await ctx.assemble_context(request, policy=explicit)
    assert result.budget == 999


async def test_engine_stage_context_policy_bound_before_run() -> None:
    """Engine binds stage.context_policy before stage.run()."""

    class _PolicyCheckCtxStage:
        name: str = "ctx-policy-check"
        transitions: tuple[str, ...] = ()
        context_policy: ContextPolicy = ContextPolicy(total_budget=1024)

        def __init__(self) -> None:
            self.assembled: AssembledCallContext | None = None

        async def run(self, ctx: StageContext) -> StageResult:
            # No explicit policy — should use bound stage policy
            self.assembled = await ctx.assemble_context(ContextRequest(task="Policy test."))
            return Done(output=_make_artifact("ctx-policy-check"))

    stage = _PolicyCheckCtxStage()
    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    pathways = PathwayRegistry()
    pathways.register(
        "ctx-policy-pathway",
        StageGraph([stage], entry="ctx-policy-check"),
        version=1,
    )
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
    )
    state = await engine.run(
        run_id="run-ctx-policy",
        session_id="sess-1",
        pathway_id="ctx-policy-pathway",
        initial=_initial_artifact(),
    )
    assert state.status is RunStatus.COMPLETED
    assert stage.assembled is not None
    # Stage's context_policy (total_budget=1024) must be used
    assert stage.assembled.budget == 1024


async def test_engine_context_policy_no_leak_between_stages() -> None:
    """context_policy from stage-a must NOT leak into stage-b (mirrors D15 invariant)."""

    class _CtxPolicyStageA:
        name: str = "ctx-a"
        transitions: tuple[str, ...] = ("ctx-b",)
        context_policy: ContextPolicy = ContextPolicy(total_budget=256)

        async def run(self, ctx: StageContext) -> StageResult:
            return Transition(to="ctx-b", output=_make_artifact("ctx-a"))

    class _CtxPolicyStageB:
        name: str = "ctx-b"
        transitions: tuple[str, ...] = ()

        def __init__(self) -> None:
            self.assembled: AssembledCallContext | None = None
            self.bound_policy_leaked: ContextPolicy | None = None

        async def run(self, ctx: StageContext) -> StageResult:
            # stage-b has no context_policy; engine should have cleared the binding
            self.assembled = await ctx.assemble_context(ContextRequest(task="No leak test."))
            # Access internal state to verify the bound policy was cleared
            self.bound_policy_leaked = getattr(ctx, "_bound_context_policy", None)
            return Done(output=_make_artifact("ctx-b"))

    stage_a = _CtxPolicyStageA()
    stage_b = _CtxPolicyStageB()
    _m = echo_model("ok")
    _r = ModelRegistry()
    _r.register("default", _m)
    pathways = PathwayRegistry()
    pathways.register(
        "ctx-leak-pathway",
        StageGraph([stage_a, stage_b], entry="ctx-a"),
        version=1,
    )
    engine = Engine(
        models=_r,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_FIXED_CLOCK,
    )
    state = await engine.run(
        run_id="run-ctx-leak",
        session_id="sess-1",
        pathway_id="ctx-leak-pathway",
        initial=_initial_artifact(),
    )
    assert state.status is RunStatus.COMPLETED
    # stage-b has no context_policy → engine called bind_context_policy(None)
    assert stage_b.bound_policy_leaked is None
    # stage-b's assembled context should use DEFAULT budget (not stage-a's 256)
    assert stage_b.assembled is not None
    assert stage_b.assembled.budget == DEFAULT_CONTEXT_POLICY.total_budget


# ---------------------------------------------------------------------------
# Part 4 — S6 determinism: same request → byte-identical assembled messages
# ---------------------------------------------------------------------------


async def test_s6_determinism_same_request_byte_identical_messages() -> None:
    """Same ContextRequest through the engine path produces byte-identical messages (S6)."""

    class _DeterminismCaptureStage:
        name: str = "det-stage"
        transitions: tuple[str, ...] = ()

        def __init__(self) -> None:
            self.captured: AssembledCallContext | None = None

        async def run(self, ctx: StageContext) -> StageResult:
            self.captured = await ctx.assemble_context(
                ContextRequest(
                    task="Determinism test task.",
                    instructions="Determinism instructions.",
                )
            )
            return Done(output=_make_artifact("det-stage"))

    # Run the same pathway twice with the same request → must produce identical messages
    def _build_engine_and_stage() -> tuple[Engine, _DeterminismCaptureStage]:
        stage = _DeterminismCaptureStage()
        _m = echo_model("ok")
        _r = ModelRegistry()
        _r.register("default", _m)
        pathways = PathwayRegistry()
        pathways.register(
            "det-pathway",
            StageGraph([stage], entry="det-stage"),
            version=1,
        )
        eng = Engine(
            models=_r,
            journal=InMemoryJournal(),
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            clock=_FIXED_CLOCK,
        )
        return eng, stage

    engine1, stage1 = _build_engine_and_stage()
    engine2, stage2 = _build_engine_and_stage()

    await engine1.run(
        run_id="run-det-1",
        session_id="sess-1",
        pathway_id="det-pathway",
        initial=_initial_artifact(),
    )
    await engine2.run(
        run_id="run-det-2",
        session_id="sess-1",
        pathway_id="det-pathway",
        initial=_initial_artifact(),
    )

    assert stage1.captured is not None
    assert stage2.captured is not None
    assert stage1.captured.messages == stage2.captured.messages


async def test_s6_determinism_direct_context_assembler() -> None:
    """Same ContextRequest direct to ContextAssembler → byte-identical messages (S6)."""
    assembler = ContextAssembler(slots=DEFAULT_SLOTS)
    request = ContextRequest(task="Determinism direct.", instructions="Direct instructions.")

    result1 = await assembler.assemble(request)
    result2 = await assembler.assemble(request)

    assert result1.messages == result2.messages


# ---------------------------------------------------------------------------
# Part 5 — FIX-NOW 1: degrade path enforces budget identically (S11)
# ---------------------------------------------------------------------------
# S8 licenses degrading content richness (no assembler wired); it does NOT
# license bypassing S11.  The unwired path must enforce the hard token ceiling
# with the SAME resolve_token_counter as the wired path.


async def test_unwired_over_budget_raises_context_budget_error() -> None:
    """Unwired (assembler=None) + task that overflows budget 8192 → ContextBudgetError.

    FIX-NOW 1a: the degrade path must raise, not silently return truncated context.
    task = 'X' * 100000 → far beyond 8192 tokens → ContextBudgetError must propagate.
    """
    ctx = _make_run_context(assembler=None)
    tight_policy = ContextPolicy(total_budget=8192)
    request = ContextRequest(task="X" * 100000)
    with pytest.raises(ContextBudgetError) as exc_info:
        await ctx.assemble_context(request, policy=tight_policy)
    err = exc_info.value
    assert err.required_tokens > err.budget, (
        f"FIX-NOW 1a: required_tokens={err.required_tokens} must exceed budget={err.budget}"
    )


async def test_unwired_over_budget_parity_with_wired() -> None:
    """Parity: the same over-budget request raises on BOTH wired and unwired paths.

    FIX-NOW 1b: both paths are equivalent in their enforcement of S11.
    """
    # A tiny budget that the task alone will exceed.
    tiny_policy = ContextPolicy(total_budget=1)
    # Task that is over budget no matter the counter (1 char ≥ 1 token minimum).
    request = ContextRequest(task="A" * 100)

    # Unwired path must raise.
    ctx_unwired = _make_run_context(assembler=None)
    with pytest.raises(ContextBudgetError):
        await ctx_unwired.assemble_context(request, policy=tiny_policy)

    # Wired path must also raise (ContextAssembler enforces the same ceiling).
    assembler = ContextAssembler(slots=DEFAULT_SLOTS)
    ctx_wired = _make_run_context(assembler=assembler)
    with pytest.raises(ContextBudgetError):
        await ctx_wired.assemble_context(request, policy=tiny_policy)


async def test_unwired_counter_consistency_matches_model_count_tokens() -> None:
    """FIX-NOW 1c: unwired token_count uses model.count_tokens, not len//4.

    A FakeModel with word-count as count_tokens (distinctive — differs from len//4).
    In-budget request: assert unwired token_count matches the model's counter, NOT len//4.
    """

    class _WordCountModel:
        """Counts tokens as number of words (deliberately distinctive from len//4)."""

        def count_tokens(self, text: str) -> int:
            return max(1, len(text.split()))

        async def complete(self, **kwargs: object) -> object:  # pragma: no cover
            raise AssertionError("complete() must not be called")

    model = _WordCountModel()
    task_text = "hello world from cog worx system"  # 6 words
    instructions_text = "be concise"  # 2 words
    # Both messages: task (6 words) + instructions (2 words) = 8 words total.
    expected_token_count = model.count_tokens(instructions_text) + model.count_tokens(task_text)
    # Budget must be generous enough to not raise.
    policy = ContextPolicy(total_budget=expected_token_count + 100)

    ctx = RunContext(
        run_id="test-counter",
        session_id="sess-counter",
        model=model,  # type: ignore[arg-type]
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
        clock=_FIXED_CLOCK,
        assembler=None,
    )

    request = ContextRequest(task=task_text, instructions=instructions_text)
    result = await ctx.assemble_context(request, policy=policy)

    # The token_count should match the word-count model, NOT len//4.
    len4_count = sum(max(1, len(m.content) // 4) for m in result.messages)
    assert result.token_count == expected_token_count, (
        f"FIX-NOW 1c: token_count={result.token_count} must equal "
        f"model.count_tokens sum={expected_token_count}, not len//4={len4_count}"
    )
    # Discriminability guard: the word-count and len//4 must differ for this fixture.
    assert expected_token_count != len4_count, (
        "FIX-NOW 1c: fixture must discriminate word-count from len//4 "
        f"(both are {expected_token_count} — increase task length)"
    )
