"""Context assembly error types (Pod 3.1a).

Structural errors raised by the ``ContextAssembler`` during the arbitration phase.

Dependency direction (CANON D3): runtime → context → injection → recall.
This module does NOT import ``cogworx.runtime``.

CANON sections cited:
  - S8  — graceful degradation; these errors are only raised for unrecoverable conditions
           (required slots that cannot fit or whose contributors hard-failed).  All soft
           failures are expressed via ``SlotStatus`` in ``SlotReport``, not exceptions.
  - S11 — cost bounded structurally; ``ContextBudgetError`` is the structural guard that
           prevents a model call when required content cannot fit in the budget.

Contract changelog:
  - 2026-06-12 (Pod 3.1a): initial surface — ContextBudgetError, ContextAssemblyError.
"""

from __future__ import annotations

__all__ = [
    "ContextAssemblyError",
    "ContextBudgetError",
]


class ContextBudgetError(Exception):
    """Required slots exceed the total_budget and cannot be assembled.

    Raised by the assembler (Pod 3.1d) when the token estimate for all ``required`` slots
    exceeds ``ContextPolicy.total_budget``.  This is the S11 structural pre-call guard —
    the model is never called when required context cannot fit (S11 violation otherwise).

    Attributes:
        required_tokens: Estimated tokens needed by required slots.
        budget:          The configured ``total_budget`` in effect.
        slot_names:      Names of the required slots that could not fit.
    """

    def __init__(
        self,
        *,
        required_tokens: int,
        budget: int,
        slot_names: tuple[str, ...],
    ) -> None:
        self.required_tokens = required_tokens
        self.budget = budget
        self.slot_names = slot_names
        super().__init__(
            f"Required slots {slot_names!r} need ~{required_tokens} tokens but budget is "
            f"{budget}; cannot assemble context."
        )


class ContextAssemblyError(Exception):
    """A required-slot contributor raised an exception during contribute().

    Raised by the assembler (Pod 3.1d) when a ``required`` slot's contributor raises rather
    than returning ``SlotContent(status="degraded")``.  Preferred and evictable slot failures
    are handled gracefully (S8) — they do NOT raise this error.

    Attributes:
        slot_name: The name of the slot whose contributor failed.
        cause:     The original exception from the contributor.
    """

    def __init__(self, *, slot_name: str, cause: BaseException) -> None:
        self.slot_name = slot_name
        self.cause = cause
        super().__init__(
            f"Required slot {slot_name!r} contributor raised {type(cause).__name__}: {cause}"
        )
