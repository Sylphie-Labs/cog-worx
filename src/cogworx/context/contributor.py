"""ContextContributor protocol — the seam between the assembler and slot providers (Pod 3.1a).

A contributor is any object that satisfies the ``ContextContributor`` runtime-checkable Protocol.
It receives a ``ContextRequest`` and a ``SlotAllocation`` (its budget envelope) and returns
``SlotContent`` — text chunks plus optional tool specs — for its slot.

Dependency direction (CANON D3): runtime → context → injection → recall.
This module does NOT import ``cogworx.runtime``.

CANON sections cited:
  - S4  — model-agnostic; contributors carry no provider reference.
  - S8  — graceful degradation; contributors SHOULD return ``SlotContent(status="degraded")``
           rather than raising, so the assembler can decide whether to treat the slot as empty or
           propagate ``ContextAssemblyError``.
  - S9  — structure over prompting; contributors are structural, not prompt-injecting bridges.

Contract changelog:
  - 2026-06-12 (Pod 3.1a): initial surface — ContextContributor Protocol.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from cogworx.context.types import ContextRequest, SlotAllocation, SlotContent

__all__ = ["ContextContributor"]


@runtime_checkable
class ContextContributor(Protocol):
    """Async slot-content provider.

    Implementers are registered with the ``ContextAssembler`` (one contributor per slot name).
    The assembler calls ``contribute`` concurrently for independent slots and awaits results
    before arbitrating the assembled context.

    Conformance note (CANON §6.1 C3):
        Adding a *new* method or parameter to this Protocol is a **breaking** change — existing
        implementations will silently stop conforming.
        Route such changes through ``/update-canon``.

    Example minimal implementation::

        class MyRulesContributor:
            async def contribute(
                self, request: ContextRequest, allocation: SlotAllocation
            ) -> SlotContent:
                return SlotContent(
                    chunks=(SlotChunk(text="Be helpful.", key="rules:base", source_slot="rules"),),
                )
    """

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Return content for the contributor's assigned slot.

        Args:
            request:    The full context request for this turn.
            allocation: The budget envelope; ``allocation.max_tokens`` caps token usage for this
                        slot.  ``None`` means the assembler has not yet partitioned the budget
                        (stub phase only — concrete assembler always sets a value).

        Returns:
            ``SlotContent`` describing zero or more chunks and/or tool specs.
            A contributor SHOULD NOT raise — return ``SlotContent(status="degraded", detail=...)``
            on soft failures so the assembler can apply S8 degradation logic.
        """
        ...
