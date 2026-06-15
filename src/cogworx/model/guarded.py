"""Budget-guarded model wrapper (CANON S11, S1, S8, S4).

``BudgetGuardedModel`` wraps any ``Model`` implementation and enforces a structural cost ceiling
(S11) as a pre-call guard: ``guard.check()`` is called BEFORE ``inner.complete()`` so the model
cannot be invoked once the ceiling is reached — the loop, not the model, decides when to stop
(S9/S11).  After a successful call, ``guard.record(response.usage)`` updates the guard's running
totals so successive checks remain accurate.

S8 (Lesion Test): removing this wrapper makes the underlying model run unbounded — degraded, not
dead.  The system degrades gracefully rather than failing hard; stages that call ``ctx.model`` see
exactly the same ``Model`` interface whether the guard is present or not.

S4 (model-agnostic): ``BudgetGuardedModel`` is provider-neutral; it wraps any ``Model`` and is
transparent to callers — ``capabilities`` and ``count_tokens`` pass straight through so no stage
needs to know the guard exists.

S1: ``count_tokens`` is a synchronous, zero-cost utility that is NEVER guarded (no model call, no
cost); guarding it would be a false positive violation of the pre-call ceiling contract.

Wired by ``Engine._build_context`` so every stage transparently gets the guarded model — zero stage
migration.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from cogworx.cost.budget import BudgetGuard
from cogworx.model.base import (
    ChatMessage,
    Model,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
)

__all__ = ["BudgetGuardedModel"]


class BudgetGuardedModel:
    """A ``Model`` decorator that enforces a ``BudgetGuard`` ceiling on every ``complete`` call.

    The guard fires BEFORE ``inner.complete()`` (S11 pre-call) and records usage AFTER a successful
    return.  ``capabilities`` and ``count_tokens`` are pure pass-throughs — the guard imposes no
    overhead on introspection or token counting.

    Satisfies the ``Model`` protocol: any code that accepts a ``Model`` accepts a
    ``BudgetGuardedModel`` without modification.
    """

    def __init__(
        self,
        inner: Model,
        guard: BudgetGuard,
        estimator: Callable[[Sequence[ChatMessage], ModelTier], float] | None = None,
    ) -> None:
        """
        Args:
            inner:      The underlying ``Model`` to delegate to.
            guard:      The ``BudgetGuard`` whose ceiling is enforced pre-call.
            estimator:  Optional callable that projects the USD cost of a call before it is made,
                        used to check the USD ceiling (``guard.check(projected_usd=...)``).
                        Defaults to ``lambda messages, tier: 0.0`` (no projection — only the call
                        ceiling is enforced until a real estimator is wired in).
        """
        self._inner = inner
        self._guard = guard
        if guard.max_usd is not None and estimator is None:
            raise ValueError(
                "BudgetGuardedModel: pass an estimator or use build_model "
                "when guard.max_usd is set."
            )
        self._estimator: Callable[[Sequence[ChatMessage], ModelTier], float] = (
            estimator if estimator is not None else lambda _msgs, _tier: 0.0
        )

    # ------------------------------------------------------------------
    # Model protocol — capabilities + count_tokens are pure pass-throughs
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> ModelCapabilities:
        """Delegate capabilities unchanged — the guard adds no capability of its own."""
        return self._inner.capabilities

    def count_tokens(self, text: str) -> int:
        """Synchronous utility; NEVER guarded (S1 — no model call, no cost)."""
        return self._inner.count_tokens(text)

    # ------------------------------------------------------------------
    # complete — the guarded path
    # ------------------------------------------------------------------

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        """Pre-call guard → inner.complete → record usage → return response.

        Raises:
            BudgetExceededError: if the guard's ceiling is already reached or the projected cost
                would breach the USD limit.  Raised BEFORE ``inner.complete`` is called — the inner
                model is NEVER invoked on a rejected call (S11 hard ceiling, S9 structural guard).
        """
        projected = self._estimator(messages, tier)
        self._guard.start_call(projected_usd=projected)  # reserves slot; raises pre-call
        response = await self._inner.complete(
            messages=messages,
            tools=tools,
            tier=tier,
            json_schema=json_schema,
        )
        self._guard.record(response.usage)  # cost only, on success
        return response
