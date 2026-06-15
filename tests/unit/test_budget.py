"""S11 contract tests: the ``BudgetGuard`` is a hard, pre-call ceiling."""

from __future__ import annotations

import pytest

from cogworx.cost.budget import BudgetExceededError, BudgetGuard
from cogworx.model.base import Usage


def test_unbounded_guard_never_raises() -> None:
    guard = BudgetGuard()
    for _ in range(100):
        guard.start_call(projected_usd=1_000.0)
        guard.record(Usage(cost_usd=1_000.0))
    assert guard.remaining_usd() is None


def test_max_calls_blocks_before_exceeding() -> None:
    guard = BudgetGuard(max_calls=2)
    guard.start_call()
    guard.record(Usage())
    guard.start_call()
    guard.record(Usage())
    assert guard.calls == 2
    with pytest.raises(BudgetExceededError):
        guard.start_call()


def test_max_usd_blocks_when_projection_would_exceed() -> None:
    guard = BudgetGuard(max_usd=1.0)
    guard.start_call(projected_usd=0.6)
    guard.record(Usage(cost_usd=0.6))
    assert guard.spent_usd == pytest.approx(0.6)
    assert guard.remaining_usd() == pytest.approx(0.4)
    with pytest.raises(BudgetExceededError):
        guard.start_call(projected_usd=0.6)


def test_max_usd_allows_up_to_the_ceiling() -> None:
    guard = BudgetGuard(max_usd=1.0)
    guard.start_call(projected_usd=1.0)
    guard.record(Usage(cost_usd=1.0))
    assert guard.remaining_usd() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# New tests for start_call / record split contract
# ---------------------------------------------------------------------------


def test_start_call_at_ceiling_raises_and_does_not_increment() -> None:
    """start_call raises at the ceiling without incrementing calls."""
    guard = BudgetGuard(max_calls=1)
    guard.start_call()  # call 1 — succeeds, calls == 1
    assert guard.calls == 1
    with pytest.raises(BudgetExceededError):
        guard.start_call()  # at ceiling — must raise without incrementing
    assert guard.calls == 1  # NOT 2


def test_start_call_below_ceiling_increments() -> None:
    """start_call below the ceiling increments the call counter."""
    guard = BudgetGuard(max_calls=2)
    guard.start_call()
    assert guard.calls == 1


def test_failed_call_consumes_slot() -> None:
    """A started but never recorded call still consumes a slot (S11)."""
    guard = BudgetGuard(max_calls=2)
    guard.start_call()  # calls == 1 (call started but "failed" — no record)
    assert guard.calls == 1
    guard.record(Usage())  # recording does NOT increment calls
    assert guard.calls == 1
    guard.start_call()  # calls == 2
    assert guard.calls == 2
    with pytest.raises(BudgetExceededError):
        guard.start_call()  # ceiling reached


def test_record_does_not_increment_calls() -> None:
    """record() only accumulates cost — it does NOT increment the call counter."""
    guard = BudgetGuard()
    guard.record(Usage(cost_usd=0.5))
    assert guard.calls == 0
    assert guard.spent_usd == pytest.approx(0.5)


def test_max_usd_property() -> None:
    """max_usd property returns the configured ceiling."""
    assert BudgetGuard(max_usd=5.0).max_usd == pytest.approx(5.0)


def test_max_usd_none_property() -> None:
    """max_usd property returns None when unbounded."""
    assert BudgetGuard().max_usd is None
