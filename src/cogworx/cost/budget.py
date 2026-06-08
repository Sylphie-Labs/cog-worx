"""Structural cost ceiling (CANON S11).

A pre-call guard: the model cannot self-terminate, so cost is bounded by the loop, not by the
model's judgement. Callers MUST ``check()`` before invoking the model and ``record()`` after — the
guard refuses the next call once a ceiling is reached, and the per-run ceiling is hard.
"""

from __future__ import annotations

from cogworx.model.base import Usage


class BudgetExceededError(Exception):
    """Raised by ``BudgetGuard.check`` when the next call would breach a ceiling."""


class BudgetGuard:
    """A pre-call cost/call ceiling. ``None`` on an axis means unbounded on that axis.

    The model never decides to stop: the loop calls ``check()`` before each model invocation and
    ``record()`` after. ``check()`` raises ``BudgetExceededError`` rather than letting the call
    proceed, making the per-run ceiling hard and structural (S11).
    """

    def __init__(self, *, max_usd: float | None = None, max_calls: int | None = None) -> None:
        self._max_usd = max_usd
        self._max_calls = max_calls
        self._spent_usd: float = 0.0
        self._calls: int = 0

    @property
    def spent_usd(self) -> float:
        return self._spent_usd

    @property
    def calls(self) -> int:
        return self._calls

    def remaining_usd(self) -> float | None:
        if self._max_usd is None:
            return None
        return self._max_usd - self._spent_usd

    def check(self, *, projected_usd: float = 0.0) -> None:
        if self._max_calls is not None and self._calls >= self._max_calls:
            raise BudgetExceededError(
                f"call ceiling reached: {self._calls} of {self._max_calls} calls used"
            )
        if self._max_usd is not None and self._spent_usd + projected_usd > self._max_usd:
            raise BudgetExceededError(
                f"usd ceiling reached: spent {self._spent_usd} + projected {projected_usd} "
                f"would exceed {self._max_usd}"
            )

    def record(self, usage: Usage) -> None:
        self._calls += 1
        self._spent_usd += usage.cost_usd


__all__ = [
    "BudgetExceededError",
    "BudgetGuard",
]
