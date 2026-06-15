"""Concrete ContextContributor implementations (Pod 3.1c).

Four contributors ship here, each implementing the ``ContextContributor`` Protocol
(``async def contribute(self, request, allocation) -> SlotContent``):

- :class:`StaticContributor`     — fixed text; placeholder for personality (3.3) and rules (3.4).
- :class:`TaskContributor`       — lifts ``request.task`` or ``request.instructions``.
- :class:`ToolSpecContributor`   — wraps a static ``Sequence[ToolSpec]``; 3.2 replaces the source.
- :class:`MemoryContributor`     — wraps a ``MemoryInjector``; runs recall on ``request.query``.

Failure semantics (CANON S8):
    Contributors raise on **genuine** errors (I/O failure, bad state) so that the assembler (3.1d)
    can decide how to handle them per slot necessity.  Legitimately-absent content (empty text,
    empty tool list, None query) is signalled via ``SlotContent(status="empty")`` — not an
    exception.

Memory diagnostics hand-off (MemoryContributor → assembler):
    ``InjectedMemory`` is surfaced via two channels:

    1. **Primary (3.1d-compatible):** ``self.last_injected_memory`` — the assembler reads this
       via duck-typed ``getattr(contributor, "last_injected_memory", None)`` and attaches the
       result to ``AssembledCallContext.memory``.
    2. **Secondary (auditable):** ``SlotContent.detail`` is set to ``repr(injected)`` when
       chunks are present.  A typed ``SlotContent`` field would be cleaner but is a
       Protocol-breaking change (CANON §6.1 C3).

    ``self.last_injected`` is a property alias for ``last_injected_memory`` kept for convenience.

Dependency direction (CANON D3): runtime → context → injection → recall.
This module does NOT import ``cogworx.runtime``.

CANON sections cited:
  - S1  — model work off the write path; contributors never call the model.
  - S4  — model-agnostic; no provider reference in any contributor.
  - S8  — graceful degradation; empty/absent content → status="empty", not an exception.
  - S9  — structure over prompting; selection logic is purely structural.
  - S11 — cost bounded structurally; MemoryContributor applies min(allocation, policy) ceiling.

Contract changelog:
  - 2026-06-12 (Pod 3.1c): initial surface — StaticContributor, TaskContributor,
    ToolSpecContributor, MemoryContributor.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

from cogworx.context.types import (
    ContextRequest,
    SlotAllocation,
    SlotChunk,
    SlotContent,
)
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import InjectedMemory, MemoryPolicy
from cogworx.model.base import ToolSpec

if TYPE_CHECKING:
    from cogworx.capability.policy import ToolGate

__all__ = [
    "MemoryContributor",
    "RegistryToolContributor",
    "StaticContributor",
    "TaskContributor",
    "ToolSpecContributor",
]


# ---------------------------------------------------------------------------
# StaticContributor
# ---------------------------------------------------------------------------


class StaticContributor:
    """Contributor that holds a fixed piece of text.

    Used as the placeholder for the ``personality`` slot (Pod 3.3) and ``rules`` slot (Pod 3.4),
    both of which ship empty in 3.1.  When text is non-empty a single ``SlotChunk`` is returned;
    when text is ``""`` the slot is reported as ``status="empty"`` with no chunks.

    Args:
        text:        The fixed text to contribute.  Defaults to ``""`` (empty placeholder).
        key:         Stable unique key for the chunk (e.g. ``"rules:internal_1"``).
        source_slot: Name of the slot this contributor serves (e.g. ``"rules"``).
    """

    def __init__(
        self,
        text: str = "",
        *,
        key: str,
        source_slot: str,
    ) -> None:
        self._text = text
        self._key = key
        self._source_slot = source_slot

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Return the static text as a single chunk, or empty if text is ``""``."""
        if not self._text:
            return SlotContent(status="empty")
        return SlotContent(
            chunks=(
                SlotChunk(
                    text=self._text,
                    key=self._key,
                    source_slot=self._source_slot,
                ),
            ),
        )


# ---------------------------------------------------------------------------
# TaskContributor
# ---------------------------------------------------------------------------


class TaskContributor:
    """Contributor that lifts a string field from the ``ContextRequest``.

    Selects either ``request.task`` or ``request.instructions`` via the ``field`` constructor
    argument.  ``"task"`` is always a ``str``; ``"instructions"`` may be ``None`` (meaning "use
    the base instructions" — no per-turn overlay).  Both paths return ``status="empty"`` when the
    source value is ``None`` or ``""``.

    Args:
        field:       Which request field to surface: ``"task"`` | ``"instructions"``.
        key:         Stable unique key for the chunk (e.g. ``"task:main"``).
        source_slot: Name of the slot this contributor serves (e.g. ``"task"``).
    """

    def __init__(
        self,
        field: Literal["task", "instructions"] = "task",
        *,
        key: str,
        source_slot: str,
    ) -> None:
        self._field = field
        self._key = key
        self._source_slot = source_slot

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Return the selected request field as a single chunk, or empty when absent."""
        text: str | None = request.task if self._field == "task" else request.instructions
        if not text:
            return SlotContent(status="empty")
        return SlotContent(
            chunks=(
                SlotChunk(
                    text=text,
                    key=self._key,
                    source_slot=self._source_slot,
                ),
            ),
        )


# ---------------------------------------------------------------------------
# ToolSpecContributor
# ---------------------------------------------------------------------------


class ToolSpecContributor:
    """Contributor that wraps a static list of tool definitions.

    Tools travel in ``SlotContent.tools``, NOT as text chunks.  Pod 3.2 will replace the source
    of these specs (the capability registry); 3.1c just carries a static list.

    Args:
        specs: The tool definitions to contribute.  An empty sequence → ``status="empty"``.
    """

    def __init__(self, specs: Sequence[ToolSpec] = ()) -> None:
        self._specs: tuple[ToolSpec, ...] = tuple(specs)

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Return the static tool specs, or empty when the list is empty."""
        if not self._specs:
            return SlotContent(status="empty")
        return SlotContent(tools=self._specs)


# ---------------------------------------------------------------------------
# RegistryToolContributor (Pod 3.2c)
# ---------------------------------------------------------------------------


class RegistryToolContributor:
    """Contributor that derives tool specs from a ``ToolGate`` at contribute-time (Pod 3.2c).

    Unlike ``ToolSpecContributor`` (static list), this contributor calls
    ``gate.exposed_specs()`` lazily on every ``contribute`` invocation so it always sees the
    stage's just-bound policy and current taint state.  This is the "single source of truth"
    requirement from the mythos design: the assembler and the dispatch chokepoint both derive
    from the same ``ToolGate._effective_tiers()`` call, so exposure and enforcement can never
    diverge (SC-6 coherence invariant).

    ``gate`` is imported lazily via ``TYPE_CHECKING`` at module level to avoid the
    ``context → capability`` runtime-import direction (CANON D3: runtime → context → injection).
    The concrete ``ToolGate`` is passed in at construction time, so no runtime import is needed.

    Args:
        gate: The per-drive ``ToolGate`` whose ``exposed_specs()`` is called on each contribute.
    """

    def __init__(self, gate: ToolGate) -> None:
        self._gate = gate

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Return specs for all currently-allowed capabilities, re-evaluated at call time.

        Calls ``gate.exposed_specs()`` fresh on every invocation so the result reflects the
        stage's active policy (set by ``bind_tool_policy``) and current taint state.
        An empty spec list → ``status="empty"`` (S8 graceful degrade — no tools exposed).
        """
        specs = self._gate.exposed_specs()
        if not specs:
            return SlotContent(status="empty")
        return SlotContent(tools=specs)


# ---------------------------------------------------------------------------
# MemoryContributor
# ---------------------------------------------------------------------------


class MemoryContributor:
    """Contributor that runs a ``MemoryInjector`` for the ``body/memory`` slot.

    On each ``contribute`` call:

    1. If ``request.query`` is ``None`` → return ``status="empty"`` immediately (no recall).
    2. Compute the effective budget = ``min(allocation.max_tokens, policy.token_budget)``
       when ``allocation.max_tokens`` is not ``None``; otherwise use ``policy.token_budget``
       unchanged.  This applies the S11 structural ceiling without touching the allocator's budget.
    3. Run ``injector.inject(query, policy=effective_policy)`` and convert the resulting
       ``InjectedMemory.context.chunks`` into ``SlotChunk``s.
    4. Stash the full ``InjectedMemory`` on ``self.last_injected`` so the assembler (3.1d) can
       attach it to ``AssembledCallContext.memory`` without re-running the injector.

    Diagnostics hand-off (flag for 3.1d):
        ``self.last_injected_memory`` is the primary channel — the assembler reads it via
        duck-typed ``getattr(contributor, "last_injected_memory", None)`` to attach the result
        to ``AssembledCallContext.memory``.  ``self.last_injected`` is an alias kept for
        backwards-compatibility with tests written before the 3.1d assembler landed.
        ``SlotContent.detail`` is set to ``repr(injected)`` as a secondary/auditable channel.
        A typed ``SlotContent`` field would be cleaner but is a Protocol-breaking change (§6.1 C3).

    NEVER calls the model (S1) — the injector uses ``count_tokens`` only (structural budget).

    Args:
        injector: Wired ``MemoryInjector`` (required).
        policy:   ``MemoryPolicy`` governing budget and recall behaviour.  Defaults to the
                  injector's ``DEFAULT_MEMORY_POLICY``.
        source_slot: Name of the slot this contributor serves (default ``"memory"``).
    """

    def __init__(
        self,
        injector: MemoryInjector,
        *,
        policy: MemoryPolicy | None = None,
        source_slot: str = "memory",
    ) -> None:
        self._injector = injector
        self._policy = policy
        self._source_slot = source_slot
        # Mutable diagnostic stash — cleared on each contribute() call.
        # ``last_injected_memory`` is the canonical name read by the assembler (3.1d)
        # via duck-typed getattr.  ``last_injected`` is a convenience alias.
        self.last_injected_memory: InjectedMemory | None = None

    @property
    def last_injected(self) -> InjectedMemory | None:
        """Alias for ``last_injected_memory`` — convenience for test code."""
        return self.last_injected_memory

    @last_injected.setter
    def last_injected(self, value: InjectedMemory | None) -> None:
        self.last_injected_memory = value

    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        """Run the injector for ``request.query``; return chunks as SlotContent."""
        self.last_injected_memory = None  # Clear stale value before each call (Fix 3).
        from cogworx.injection.policy import DEFAULT_MEMORY_POLICY

        if request.policy is not None and "memory" in request.policy.model_fields_set:
            base_policy = request.policy.memory
        else:
            base_policy = self._policy if self._policy is not None else DEFAULT_MEMORY_POLICY
            # CARRY-FORWARD: assembler._default_policy.memory does not reach the contributor
            # when request.policy is None — documented residual seam (Pod 3.1 Fix 4b).

        # Step 1: no query → empty (no recall, S8 graceful degradation)
        if request.query is None:
            self.last_injected_memory = None
            return SlotContent(status="empty")

        # Step 2: apply effective budget = min(allocation.max_tokens, policy.token_budget)
        if allocation.max_tokens is not None:
            effective_budget = min(allocation.max_tokens, base_policy.token_budget)
            effective_policy = base_policy.model_copy(update={"token_budget": effective_budget})
        else:
            effective_policy = base_policy

        # Step 3: run the injector (S1 — count_tokens only, no model call)
        injected: InjectedMemory = await self._injector.inject(
            request.query, policy=effective_policy
        )

        # Step 4: stash for 3.1d assembler hand-off
        # ``last_injected_memory`` is the canonical name read by the 3.1d assembler.
        self.last_injected_memory = injected

        # Convert InjectedMemory.context.chunks → SlotChunks
        slot_chunks: tuple[SlotChunk, ...] = tuple(
            SlotChunk(
                text=chunk.text,
                key=chunk.key,
                source_slot=self._source_slot,
            )
            for chunk in injected.context.chunks
        )

        status = "ok" if slot_chunks else "empty"
        return SlotContent(
            chunks=slot_chunks,
            status=status,
            # Secondary diagnostic channel for 3.1d (see module docstring)
            detail=repr(injected) if slot_chunks else None,
        )
