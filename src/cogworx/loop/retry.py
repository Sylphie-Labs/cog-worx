"""Durable retry + per-stage timeout policy (CANON S6, S8, S9, S11).

A ``RetryPolicy`` is a frozen, dev-authored value a stage carries to make a FAILED attempt
recoverable without violating the ``(run_id, step_index)`` exactly-once invariant: a failed attempt
commits NOTHING — only a success-class result lands at ``seq``. The journal's "committed ⟺ succeeded
⟺ never re-run" stays pristine while a per-``(run_id, step_index)`` attempt counter climbs and a
durable retry timer (reusing the 1.1 sweeper machinery) re-drives the same uncommitted ``seq`` after
``backoff(n)``.

Classification is STRUCTURAL (S9): the engine decides retryable-vs-not by type-matching the raised
exception against the dev-authored ``retryable`` allowlist — never the model's words. A type NOT in
the allowlist is a BUG and propagates loud (fail-loud-on-bugs); only listed exceptions + an
in-process ``timeout`` enter the retry machine.

On exhaustion (``attempt >= max_attempts``) the policy picks one of two first-class endings:
``"degraded"`` (the S8 default) commits a ``Degraded`` at ``seq`` and routes to ``exhausted_to``
(or terminates ``DEGRADED`` if ``None``); ``"fail"`` sets the run ``FAILED`` (no step committed).
Retry churn is bounded by ``max_attempts`` (per ``seq``) AND the run's ``BudgetGuard`` (S11) — never
by ``max_steps`` (retries freeze ``seq``).

``DEFAULT_RETRY_POLICY`` is OPT-IN: ``retryable=()`` means even a "retryable-looking" exception is
not retried, so a stage that ships no ``retry_policy`` behaves exactly as pods 1.0/1.1 did — a raise
propagates unchanged (backward-compatible, S8 graceful degradation).
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import timedelta
from typing import Literal

from pydantic import BaseModel, ConfigDict


class RetryPolicy(BaseModel):
    """A frozen, dev-authored retry/timeout policy a ``Stage`` may carry.

    - ``max_attempts``: the failure ceiling per ``(run_id, step_index)``. The Nth failure (the
      attempt counter reaching ``max_attempts``) is exhaustion; earlier failures arm a retry timer.
    - ``backoff(n)``: the delay before re-driving after the ``n``-th failure (``n`` is the NEW
      attempt count, starting at 1). Pure + deterministic so ``wake_at`` arithmetic is replay-safe.
    - ``retryable``: the exception-type allowlist. A raised type matching one of these (or an
      in-process ``TimeoutError``) is retried; anything else propagates loud (S9 structural, never
      model-decided).
    - ``timeout``: an in-process ``asyncio.wait_for`` ceiling per attempt; ``None`` => no timeout.
    - ``on_exhausted``: ``"degraded"`` (commit a ``Degraded`` + route onward, the S8 default) or
      ``"fail"`` (set the run ``FAILED``).
    - ``exhausted_to``: the stage the exhaustion ``Degraded`` routes to (``None`` => terminate
      ``DEGRADED``). Only meaningful when ``on_exhausted == "degraded"``.
    """

    model_config = ConfigDict(frozen=True)

    max_attempts: int
    backoff: Callable[[int], timedelta]
    retryable: tuple[type[Exception], ...]
    timeout: timedelta | None = None
    on_exhausted: Literal["degraded", "fail"] = "degraded"
    exhausted_to: str | None = None


def _exponential_backoff(n: int) -> timedelta:
    """Exponential backoff: ``1, 2, 4, …`` seconds for the ``n``-th failure (``n`` starts at 1)."""
    return timedelta(seconds=2 ** (n - 1))


DEFAULT_RETRY_POLICY = RetryPolicy(
    max_attempts=3,
    backoff=_exponential_backoff,
    retryable=(),
    timeout=None,
    on_exhausted="degraded",
    exhausted_to=None,
)
"""The opt-in default: ``retryable=()`` so a stage without an explicit policy behaves as pods
1.0/1.1 (a raise propagates), while the retry MACHINE is wired for any stage that opts in (S8)."""


__all__ = [
    "DEFAULT_RETRY_POLICY",
    "RetryPolicy",
]
