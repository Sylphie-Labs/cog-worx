"""Tests for ``BudgetGuardedModel`` (CANON S11, S8, S4).

Verifies the structural pre-call ceiling (S11): ``guard.check()`` fires BEFORE ``inner.complete()``
so the inner model is never invoked on a rejected call.  After a successful call, ``guard.record``
is called with the response's ``Usage``.  ``capabilities`` and ``count_tokens`` pass through
unchanged (S4 — the wrapper is transparent to callers).  The engine-level test confirms that a run
whose budget is exhausted terminates with a propagated ``BudgetExceededError`` (non-retryable) and
NOT a retried or silently degraded outcome (S11/S9 structural, never model-decided).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.cost.budget import BudgetExceededError, BudgetGuard, BudgetPolicy
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    Usage,
)
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _make_response(cost_usd: float = 0.0) -> ModelResponse:
    return ModelResponse(
        text="ok",
        model_id="replay",
        finish_reason="stop",
        usage=Usage(cost_usd=cost_usd),
    )


def _make_messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="hello")]


def _provenance(source: str = "system") -> Provenance:
    return Provenance(source=source, confidence=1.0, recorded_at=_EPOCH)


def _artifact(stage: str, source: str = "system") -> Artifact:
    return Artifact(kind="test", produced_by=stage, provenance=_provenance(source))


# ---------------------------------------------------------------------------
# (a) guard.check() is called BEFORE inner.complete()
# ---------------------------------------------------------------------------


async def test_check_called_before_inner_complete() -> None:
    """``guard.check()`` fires before ``inner.complete()``; a rejected call never reaches the inner
    model (S11 pre-call, S9 structural guard).

    Strategy: pre-exhaust the guard with ``max_calls=0`` (already at the ceiling before any call).
    ``BudgetGuardedModel.complete()`` must raise ``BudgetExceededError`` and the inner model must
    record zero calls.
    """
    inner = ReplayModel([_make_response()])
    guard = BudgetGuard(max_calls=0)  # ceiling already reached
    guarded = BudgetGuardedModel(inner, guard)

    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=_make_messages())

    # The inner model was NEVER called — the guard blocked before delegation.
    assert inner.call_count == 0


# ---------------------------------------------------------------------------
# (b) guard.record() called with the response Usage after a successful call
# ---------------------------------------------------------------------------


async def test_record_called_with_response_usage() -> None:
    """After a successful ``complete``, ``guard.record(response.usage)`` is called so the guard's
    running cost tally stays accurate for subsequent ``check`` calls (S11).
    """
    cost = 0.25
    inner = ReplayModel([_make_response(cost_usd=cost)])
    guard = BudgetGuard(max_usd=1.0)
    # max_usd is set so we must supply an estimator (new contract)
    guarded = BudgetGuardedModel(inner, guard, estimator=lambda _m, _t: 0.0)

    response = await guarded.complete(messages=_make_messages())

    # Guard recorded the cost from the response.
    assert guard.spent_usd == pytest.approx(cost)
    assert guard.calls == 1
    # The response is passed through unmodified.
    assert response.text == "ok"


# ---------------------------------------------------------------------------
# (c) N+1th call raises BudgetExceededError; inner NOT invoked on that call
# ---------------------------------------------------------------------------


async def test_ceiling_hit_on_n_plus_1_call() -> None:
    """The (N+1)-th call raises ``BudgetExceededError`` once ``max_calls`` is reached, and the
    inner model is NOT invoked on that call (S11: hard pre-call ceiling, S9: structural check).

    N = 2: the first two calls succeed; the third is rejected before reaching the inner model.
    """
    n = 2
    # Provide N+1 scripted responses — the (N+1)-th must never be consumed.
    inner = ReplayModel([_make_response() for _ in range(n + 1)])
    guard = BudgetGuard(max_calls=n)
    guarded = BudgetGuardedModel(inner, guard)

    # First N calls succeed.
    for _ in range(n):
        await guarded.complete(messages=_make_messages())

    assert guard.calls == n
    assert inner.call_count == n

    # The (N+1)-th call is rejected pre-call.
    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=_make_messages())

    # Inner call count has NOT increased — the guard blocked before delegating.
    assert inner.call_count == n


# ---------------------------------------------------------------------------
# (d) capabilities and count_tokens pass through unchanged
# ---------------------------------------------------------------------------


def test_capabilities_passthrough() -> None:
    """``capabilities`` delegates to the inner model's property — transparent to callers (S4)."""
    caps = ModelCapabilities(streaming=True, tools=True)
    inner = ReplayModel(capabilities=caps)
    guard = BudgetGuard()
    guarded = BudgetGuardedModel(inner, guard)

    assert guarded.capabilities is caps


def test_count_tokens_passthrough() -> None:
    """``count_tokens`` passes through to the inner model's implementation without guarding (S1 —
    no model call, no cost incurred; guarding it would be a false positive).
    """
    inner = ReplayModel(token_counter=lambda text: len(text))
    guard = BudgetGuard(max_calls=0)  # would block complete() but must NOT block count_tokens
    guarded = BudgetGuardedModel(inner, guard)

    # Should NOT raise even though max_calls=0.
    result = guarded.count_tokens("hello")
    assert result == len("hello")
    # Guard is untouched.
    assert guard.calls == 0


# ---------------------------------------------------------------------------
# (d-extra) satisfies the Model protocol (runtime_checkable)
# ---------------------------------------------------------------------------


def test_budget_guarded_model_satisfies_model_protocol() -> None:
    """``BudgetGuardedModel`` satisfies the ``Model`` runtime-checkable protocol (S4)."""
    from cogworx.model.base import Model

    inner = ReplayModel()
    guard = BudgetGuard()
    guarded = BudgetGuardedModel(inner, guard)

    assert isinstance(guarded, Model)


# ---------------------------------------------------------------------------
# Engine-level test: budget-exhausted run terminates non-retried FAILED
# ---------------------------------------------------------------------------

_BUDGET_PATHWAY_ID = "budget-ceiling"


class _IntakeStage:
    """No-model intake stage — transitions to the model-calling stage."""

    name: str = "intake"
    transitions: tuple[str, ...] = ("call_model",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="call_model", output=_artifact("intake"))


class _CallModelStage:
    """A stage that calls ``ctx.model.complete()`` — hits the budget ceiling via the guarded model.

    This stage does NOT manually call ``ctx.budget.check()`` / ``ctx.budget.record()`` — that is
    the guard's job via ``BudgetGuardedModel``.  The ``BudgetExceededError`` raised by the guard
    propagates as a non-retryable exception, proving S11: the loop bounds the run, not the model.
    """

    name: str = "call_model"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        # ctx.model is the BudgetGuardedModel wired by Engine._build_context.
        # The guard fires here; BudgetExceededError is NOT in the retryable allowlist.
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="hello")])
        return Done(
            output=Artifact(
                kind="response",
                produced_by="call_model",
                provenance=_provenance("inference"),
                data={"text": response.text or ""},
            )
        )


def _budget_pathways() -> PathwayRegistry:
    graph = StageGraph([_IntakeStage(), _CallModelStage()], entry="intake")
    registry = PathwayRegistry()
    registry.register(_BUDGET_PATHWAY_ID, graph, version=1)
    return registry


async def test_engine_budget_exhausted_run_fails_non_retried() -> None:
    """Engine-level S11 proof: a ``BudgetGuard(max_calls=0)`` attached to the engine causes the
    guarded model to raise ``BudgetExceededError`` before the inner model is called.

    - The error propagates as a non-retryable exception (not in the retryable allowlist — S9
      structural classification).
    - The inner model is NEVER invoked (guard blocked before delegation).
    - ``pytest.raises`` confirms the error escapes ``engine.run()`` directly (non-retried FAILED).

    S8 note: the system degrades (BudgetExceededError propagates) rather than silently swallowing
    the error; removing the ``BudgetGuardedModel`` wrapper from ``_build_context`` would allow the
    call through unbounded — degraded-not-dead (lesion-safe).
    """
    inner = ReplayModel(
        [_make_response()]  # has one scripted response but the guard blocks before it's used
    )
    registry = ModelRegistry()
    registry.register_factory("default", lambda g: BudgetGuardedModel(inner, g))
    engine = Engine(
        models=registry,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_budget_pathways(),
        budget_policy=BudgetPolicy(max_calls_per_drive=0),
    )

    with pytest.raises(BudgetExceededError):
        await engine.run(
            run_id="budget-fail",
            session_id="budget-sess",
            pathway_id=_BUDGET_PATHWAY_ID,
            initial=_artifact("input"),
        )

    # The inner model was NEVER called — the BudgetGuardedModel rejected pre-call (S11).
    assert inner.call_count == 0


# ---------------------------------------------------------------------------
# New tests for start_call / record split contract and estimator validation
# ---------------------------------------------------------------------------


async def test_failed_inner_still_counts_call_slot() -> None:
    """A started call that raises (inner raises) still increments the call counter (S11).

    start_call() increments calls BEFORE inner.complete() is invoked.  Even if inner
    raises, the call slot is consumed.
    """
    from collections.abc import Mapping, Sequence
    from typing import Any

    from cogworx.model.base import ModelCapabilities, ModelTier, ToolSpec

    class _RaisingModel:
        """A model that always raises on complete()."""

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities()

        async def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSpec] = (),
            tier: ModelTier = "pro",
            json_schema: Mapping[str, Any] | None = None,
        ) -> ModelResponse:
            raise RuntimeError("inner failed")

        def count_tokens(self, text: str) -> int:
            return 1

    inner = _RaisingModel()
    guard = BudgetGuard(max_calls=2)
    guarded = BudgetGuardedModel(inner, guard)

    with pytest.raises(RuntimeError, match="inner failed"):
        await guarded.complete(messages=_make_messages())

    # The call was started (slot consumed) even though inner raised.
    assert guard.calls == 1
    assert guard.spent_usd == pytest.approx(0.0)


async def test_second_attempt_raises_without_invoking_inner() -> None:
    """After a failed call consumes the only slot, the next call raises BudgetExceededError
    before reaching inner (S11 — hard pre-call ceiling).
    """
    from collections.abc import Mapping, Sequence
    from typing import Any

    from cogworx.model.base import ModelCapabilities, ModelTier, ToolSpec

    call_count = [0]

    class _RaisingModel:
        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities()

        async def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSpec] = (),
            tier: ModelTier = "pro",
            json_schema: Mapping[str, Any] | None = None,
        ) -> ModelResponse:
            call_count[0] += 1
            raise RuntimeError("inner failed")

        def count_tokens(self, text: str) -> int:
            return 1

    inner = _RaisingModel()
    guard = BudgetGuard(max_calls=1)
    guarded = BudgetGuardedModel(inner, guard)

    # First call: start_call increments (calls==1), inner raises.
    with pytest.raises(RuntimeError):
        await guarded.complete(messages=_make_messages())

    assert guard.calls == 1
    assert call_count[0] == 1

    # Second call: ceiling reached — BudgetExceededError BEFORE inner is called.
    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=_make_messages())

    # Inner was NOT called a second time.
    assert call_count[0] == 1


def test_budget_guarded_model_no_estimator_raises_when_max_usd_set() -> None:
    """BudgetGuardedModel raises ValueError at construction when max_usd is set but no estimator."""
    inner = ReplayModel()
    guard = BudgetGuard(max_usd=5.0)
    with pytest.raises(ValueError, match="estimator"):
        BudgetGuardedModel(inner, guard)


def test_budget_guarded_model_no_estimator_ok_when_only_max_calls() -> None:
    """BudgetGuardedModel is fine without an estimator when only max_calls is set (no max_usd)."""
    inner = ReplayModel()
    guard = BudgetGuard(max_calls=5)
    # Should NOT raise
    guarded = BudgetGuardedModel(inner, guard)
    assert guarded is not None


# ---------------------------------------------------------------------------
# F3 new test 1 — guard accumulates across stages within a single drive
# ---------------------------------------------------------------------------

_ACCUM_PATHWAY_ID = "budget-accum"


class _CallAStage:
    """Model-bearing stage; transitions to call_b."""

    name: str = "call_a"
    transitions: tuple[str, ...] = ("call_b",)

    async def run(self, ctx: StageContext) -> StageResult:
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="call a")])
        return Transition(
            to="call_b",
            output=Artifact(
                kind="response",
                produced_by="call_a",
                provenance=_provenance("inference"),
                data={"text": response.text or ""},
            ),
        )


class _CallBStage:
    """Second model-bearing stage — will breach the per-drive ceiling when guard allows 1 call."""

    name: str = "call_b"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="call b")])
        return Done(
            output=Artifact(
                kind="response",
                produced_by="call_b",
                provenance=_provenance("inference"),
                data={"text": response.text or ""},
            )
        )


class _AccumIntakeStage:
    """Intake for the accumulation test — transitions to call_a."""

    name: str = "intake"
    transitions: tuple[str, ...] = ("call_a",)

    async def run(self, ctx: StageContext) -> StageResult:
        from cogworx.loop.result import Transition

        return Transition(to="call_a", output=_artifact("intake"))


def _accum_pathways() -> PathwayRegistry:
    graph = StageGraph(
        [_AccumIntakeStage(), _CallAStage(), _CallBStage()],
        entry="intake",
    )
    reg = PathwayRegistry()
    reg.register(_ACCUM_PATHWAY_ID, graph, version=1)
    return reg


async def test_engine_budget_accumulates_across_stages() -> None:
    """ONE guard spans stages within a drive on the engine path (F3 / S11 proof).

    Pathway: intake (no model) -> call_a (1 model call) -> call_b (1 model call).
    Guard ceiling: ``max_calls_per_drive=1``.

    call_a consumes the single allowed call; call_b hits ``BudgetExceededError``.
    The guard is NOT reset between stages — a single ``_build_context`` call per drive
    produces one guard that accumulates across all stages of that drive segment.

    Proves: the guard is drive-scoped, not stage-scoped.
    """
    inner = ReplayModel([_make_response(), _make_response()])
    registry = ModelRegistry()
    registry.register_factory("default", lambda g, m=inner: BudgetGuardedModel(m, g))
    journal = InMemoryJournal()
    engine = Engine(
        models=registry,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_accum_pathways(),
        budget_policy=BudgetPolicy(max_calls_per_drive=1),
    )

    with pytest.raises(BudgetExceededError):
        await engine.run(
            run_id="accum-1",
            session_id="accum-sess",
            pathway_id=_ACCUM_PATHWAY_ID,
            initial=_artifact("input"),
        )

    # call_a consumed the single allowed call; call_b was blocked pre-call.
    assert inner.call_count == 1

    # Journal holds committed intake (step 0) and call_a (step 1) — call_b never committed.
    loaded = await journal.load_run("accum-1")
    if loaded is not None:
        committed_names = tuple(s.stage_name for s in loaded.steps)
        assert "intake" in committed_names
        assert "call_a" in committed_names
        assert "call_b" not in committed_names
