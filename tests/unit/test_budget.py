"""S11 contract tests: the ``BudgetGuard`` is a hard, pre-call ceiling."""

from __future__ import annotations

import pytest

from cogworx.cost.budget import BudgetExceededError, BudgetGuard
from cogworx.model.base import Usage


def test_unbounded_guard_never_raises() -> None:
    guard = BudgetGuard()
    for _ in range(100):
        guard.check(projected_usd=1_000.0)
        guard.record(Usage(cost_usd=1_000.0))
    assert guard.remaining_usd() is None


def test_max_calls_blocks_before_exceeding() -> None:
    guard = BudgetGuard(max_calls=2)
    guard.check()
    guard.record(Usage())
    guard.check()
    guard.record(Usage())
    assert guard.calls == 2
    with pytest.raises(BudgetExceededError):
        guard.check()


def test_max_usd_blocks_when_projection_would_exceed() -> None:
    guard = BudgetGuard(max_usd=1.0)
    guard.check(projected_usd=0.6)
    guard.record(Usage(cost_usd=0.6))
    assert guard.spent_usd == pytest.approx(0.6)
    assert guard.remaining_usd() == pytest.approx(0.4)
    with pytest.raises(BudgetExceededError):
        guard.check(projected_usd=0.6)


def test_max_usd_allows_up_to_the_ceiling() -> None:
    guard = BudgetGuard(max_usd=1.0)
    guard.check(projected_usd=1.0)
    guard.record(Usage(cost_usd=1.0))
    assert guard.remaining_usd() == pytest.approx(0.0)
