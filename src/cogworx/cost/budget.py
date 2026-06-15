"""Structural cost ceiling (CANON S11).

A pre-call guard: the model cannot self-terminate, so cost is bounded by the loop, not by the
model's judgement.  The split contract:

- ``start_call(projected_usd)`` — pre-call: runs ``check()`` then increments the call counter
  atomically.  A call that subsequently fails still counts toward ``max_calls`` (S11).
- ``record(usage)`` — post-call: adds real cost to the running total.  Does NOT increment the
  call counter.
- ``check(projected_usd)`` — read-only probe; does not mutate state.

The guard refuses the next call once a ceiling is reached, and the per-DRIVE-SEGMENT ceiling is
hard; a paused/resumed/timer-fired run gets a fresh guard per segment. Cumulative per-RUN ceilings
are CF-3.0-B (deferred) — see CANON S11 note.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from cogworx.model.base import Usage


class BudgetExceededError(Exception):
    """Raised by ``BudgetGuard.check`` when the next call would breach a ceiling."""


class BudgetGuard:
    """A pre-call cost/call ceiling. ``None`` on an axis means unbounded on that axis.

    The model never decides to stop: the loop calls ``check()`` before each model invocation and
    ``record()`` after. ``check()`` raises ``BudgetExceededError`` rather than letting the call
    proceed, making the per-drive-segment ceiling hard and structural (S11). A guard IS one segment:
    it accumulates cost for its single drive segment and is not shared across resume boundaries.
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

    @property
    def max_usd(self) -> float | None:
        return self._max_usd

    def start_call(self, *, projected_usd: float = 0.0) -> None:
        """Pre-call reservation: check() then increment the call count atomically.
        A call that subsequently fails still counts toward max_calls (S11)."""
        self.check(projected_usd=projected_usd)
        self._calls += 1

    def record(self, usage: Usage) -> None:
        """Post-call: add real cost. No longer increments the call count."""
        self._spent_usd += usage.cost_usd


class BudgetPolicy(BaseModel):
    """Frozen template that mints a fresh ``BudgetGuard`` per drive (CANON S11).

    ``BudgetGuard`` is stateful and one-shot (it accumulates cost for one drive segment).
    ``BudgetPolicy`` is the immutable template the engine holds; it calls ``new_guard()``
    at the start of each drive to produce an isolated guard for that segment.

    Note — per-DRIVE-SEGMENT ceiling (not cumulative-across-resume):
        Because ``_build_context`` runs per drive, ceilings reset each segment — a
        paused/resumed/timer-fired run gets a fresh ceiling per segment, not one cumulative
        run-lifetime ceiling.  Each segment is still hard-bounded (S11) and S6 replay never
        re-calls the model, so this is defensible.  Cumulative per-RUN ceilings are
        CF-3.0-B (deferred) — see CANON S11 note.
    """

    # CF-3.0-B: cumulative-across-resume ceilings (journal guard totals);
    # concurrent in-flight USD reservation; failed-call billed cost.

    model_config = ConfigDict(frozen=True)

    max_usd_per_drive: float | None = None
    max_calls_per_drive: int | None = None

    def new_guard(self) -> BudgetGuard:
        """Mint a fresh ``BudgetGuard`` for one drive segment."""
        return BudgetGuard(max_usd=self.max_usd_per_drive, max_calls=self.max_calls_per_drive)


__all__ = [
    "BudgetExceededError",
    "BudgetGuard",
    "BudgetPolicy",
]
