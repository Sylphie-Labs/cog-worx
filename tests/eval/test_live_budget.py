"""Deterministic unit tests for the live-run cost estimator (Pod 4.4-live L4).

Pure, offline, deterministic — no model, no docker, no journal, no network. Covers:

  - ``make_cost_estimator`` projects the expected USD for a known token count, at both the list
    and promo price bases, spot-checked against ``PriceTable.cost_usd`` directly.
  - a USD ceiling trips PRE-call: the stub's ``complete`` is never awaited.
  - a failed inner call still counts toward ``max_calls`` (``start_call`` increments before the
    failure).
  - ``guard.record`` accrues REAL usage after a successful call (not the projection).
  - end-to-end: a budgeted model run through a few "arm" cells under a generous ceiling never trips.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from cogworx.cost.budget import BudgetExceededError, BudgetGuard
from cogworx.eval._live.budget import build_budgeted_model, make_cost_estimator
from cogworx.eval._live.settings import DEEPSEEK_V4_PRO_LIST_PRICE, DEEPSEEK_V4_PRO_PROMO_PRICE
from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
    Usage,
)
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.testing.fake_model import ReplayModel

_MAX_OUTPUT_TOKENS = 8192


def _messages(text: str) -> list[ChatMessage]:
    return [ChatMessage(role="user", content=text)]


def _count_tokens(text: str) -> int:
    return len(text)


# ===========================================================================
# (1) make_cost_estimator projects the expected USD, spot-checked at list + promo bases
# ===========================================================================


@pytest.mark.parametrize("price_table", [DEEPSEEK_V4_PRO_LIST_PRICE, DEEPSEEK_V4_PRO_PROMO_PRICE])
def test_estimator_matches_price_table_cost_usd(price_table: Any) -> None:
    text = "x" * 100  # _count_tokens -> 100 prompt tokens
    estimator = make_cost_estimator(
        price_table, max_output_tokens=_MAX_OUTPUT_TOKENS, count_tokens=_count_tokens
    )

    projected = estimator(_messages(text), "pro")

    expected_usage = Usage(prompt_tokens=100, completion_tokens=_MAX_OUTPUT_TOKENS)
    assert projected == pytest.approx(price_table.cost_usd(expected_usage, "pro"))
    assert projected > 0.0  # both DeepSeek bases are non-zero — a real ceiling can bite


def test_estimator_sums_tokens_across_all_messages() -> None:
    estimator = make_cost_estimator(
        DEEPSEEK_V4_PRO_LIST_PRICE, max_output_tokens=_MAX_OUTPUT_TOKENS, count_tokens=_count_tokens
    )
    messages = [
        ChatMessage(role="system", content="a" * 10),
        ChatMessage(role="user", content="b" * 20),
    ]

    projected = estimator(messages, "pro")

    expected_usage = Usage(prompt_tokens=30, completion_tokens=_MAX_OUTPUT_TOKENS)
    assert projected == pytest.approx(DEEPSEEK_V4_PRO_LIST_PRICE.cost_usd(expected_usage, "pro"))


def test_estimator_uses_full_max_output_tokens_not_actual_completion_length() -> None:
    """The projection assumes the FULL ceiling, not any particular completion length — the
    S11-honest over-estimate that can only trip the guard early, never late."""
    small_max = make_cost_estimator(
        DEEPSEEK_V4_PRO_LIST_PRICE, max_output_tokens=10, count_tokens=_count_tokens
    )
    large_max = make_cost_estimator(
        DEEPSEEK_V4_PRO_LIST_PRICE, max_output_tokens=10_000, count_tokens=_count_tokens
    )
    messages = _messages("x")

    assert large_max(messages, "pro") > small_max(messages, "pro")


# ===========================================================================
# (2) a USD ceiling trips PRE-call — stub's complete() is NEVER awaited
# ===========================================================================


async def test_usd_ceiling_trips_pre_call_without_invoking_inner() -> None:
    inner = ReplayModel(
        [ModelResponse(text="ok", model_id="replay", finish_reason="stop")],
        token_counter=_count_tokens,
    )
    one_call_usd = make_cost_estimator(
        DEEPSEEK_V4_PRO_LIST_PRICE, max_output_tokens=_MAX_OUTPUT_TOKENS, count_tokens=_count_tokens
    )(_messages("x" * 100), "pro")
    guard = BudgetGuard(max_usd=one_call_usd - 0.0001)  # just below one projected call
    guarded = build_budgeted_model(
        inner,
        guard=guard,
        price_table=DEEPSEEK_V4_PRO_LIST_PRICE,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=_messages("x" * 100))

    assert inner.call_count == 0  # never awaited — the guard blocked before delegation


# ===========================================================================
# (3) a failed inner call still counts toward max_calls (start_call increments before failure)
# ===========================================================================


class _RaisingModel:
    """A ``Model`` double whose ``complete`` always raises, to prove the pre-call slot is still
    consumed (CANON S11: ``start_call`` increments before the inner call, so a subsequently-failed
    call still counts toward ``max_calls``)."""

    def __init__(self) -> None:
        self.attempts = 0

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
        self.attempts += 1
        raise RuntimeError("provider outage")

    def count_tokens(self, text: str) -> int:
        return _count_tokens(text)


async def test_failed_inner_call_still_counts_toward_max_calls() -> None:
    inner = _RaisingModel()
    guard = BudgetGuard(max_calls=1)
    guarded = build_budgeted_model(
        inner,
        guard=guard,
        price_table=DEEPSEEK_V4_PRO_LIST_PRICE,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    with pytest.raises(RuntimeError, match="provider outage"):
        await guarded.complete(messages=_messages("hello"))

    assert guard.calls == 1  # the slot was consumed despite the failure
    assert inner.attempts == 1

    # The next call is refused pre-call — the ceiling is already spent.
    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=_messages("hello"))
    assert inner.attempts == 1  # inner was NOT invoked a second time


# ===========================================================================
# (4) guard.record accrues REAL usage after a successful call, not the projection
# ===========================================================================


async def test_guard_records_real_usage_not_projection() -> None:
    real_cost = 0.0001  # deliberately far below what the conservative projection would assume
    inner = ReplayModel(
        [
            ModelResponse(
                text="ok",
                model_id="replay",
                finish_reason="stop",
                usage=Usage(prompt_tokens=5, completion_tokens=5, cost_usd=real_cost),
            )
        ],
        token_counter=_count_tokens,
    )
    guard = BudgetGuard(max_usd=1.0)
    guarded = build_budgeted_model(
        inner,
        guard=guard,
        price_table=DEEPSEEK_V4_PRO_LIST_PRICE,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    projected_at_call_time = make_cost_estimator(
        DEEPSEEK_V4_PRO_LIST_PRICE, max_output_tokens=_MAX_OUTPUT_TOKENS, count_tokens=_count_tokens
    )(_messages("hello"), "pro")

    await guarded.complete(messages=_messages("hello"))

    assert guard.spent_usd == pytest.approx(real_cost)
    assert guard.spent_usd != pytest.approx(projected_at_call_time)


# ===========================================================================
# (5) end-to-end: a budgeted model runs a few cells under a generous ceiling without tripping
# ===========================================================================


async def test_budgeted_model_runs_several_cells_under_generous_ceiling() -> None:
    responses = [
        ModelResponse(
            text=f"cell-{i}",
            model_id="replay",
            finish_reason="stop",
            usage=Usage(prompt_tokens=5, completion_tokens=5, cost_usd=0.001),
        )
        for i in range(3)
    ]
    inner = ReplayModel(responses, token_counter=_count_tokens)
    guard = BudgetGuard(max_usd=1.0, max_calls=10)  # generous relative to 3 cheap cells
    guarded: BudgetGuardedModel = build_budgeted_model(
        inner,
        guard=guard,
        price_table=DEEPSEEK_V4_PRO_LIST_PRICE,
        max_output_tokens=_MAX_OUTPUT_TOKENS,
    )

    async def run_cell(item_id: int) -> str:
        response = await guarded.complete(messages=_messages(f"item {item_id}"))
        return response.text or ""

    results = [await run_cell(i) for i in range(3)]

    assert results == ["cell-0", "cell-1", "cell-2"]
    assert guard.calls == 3
    assert guard.spent_usd == pytest.approx(0.003)
