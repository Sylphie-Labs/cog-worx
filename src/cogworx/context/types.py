"""Context slot types, policy, and assembled-call result (Pod 3.1a).

Provides the pure, frozen value types that describe how context is structured, budgeted, and
assembled before a model call.  No I/O, no model import beyond leaf value types from
``cogworx.model.base`` (``ChatMessage``, ``ToolSpec``).

Dependency direction (CANON D3): runtime → context → injection → recall.
This module does NOT import ``cogworx.runtime``.

CANON sections cited:
  - S1  — model work off the write path; context assembly is off the hot write path.
  - S4  — model-agnostic; context types carry no provider reference.
  - S8  — graceful degradation; SlotStatus encodes degraded/empty as first-class states.
  - S11 — cost bounded structurally; ContextPolicy.total_budget is the pre-call budget ceiling.

Contract changelog:
  - 2026-06-12 (Pod 3.1a): initial surface — SlotBand, SlotNecessity, SlotStatus,
    SlotSpec, SlotChunk, SlotContent, SlotAllocation, ContextPolicy, ContextRequest,
    SlotReport, AssembledCallContext, DEFAULT_CONTEXT_POLICY, DEFAULT_SLOTS.
  - 2026-06-12 (Pod 3.1): token_count SEMANTICS CHANGE — now includes serialised tool-spec
    tokens in addition to message tokens (was message-tokens-only). Callers comparing
    token_count to budget must use the new total-consumption semantics.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, InjectedMemory, MemoryPolicy
from cogworx.model.base import ChatMessage, ToolSpec
from cogworx.recall.query import RecallQuery

__all__ = [
    "DEFAULT_CONTEXT_POLICY",
    "DEFAULT_SLOTS",
    "AssembledCallContext",
    "ContextPolicy",
    "ContextRequest",
    "SlotAllocation",
    "SlotBand",
    "SlotChunk",
    "SlotContent",
    "SlotNecessity",
    "SlotReport",
    "SlotSpec",
    "SlotStatus",
]

# ---------------------------------------------------------------------------
# Literal types (branded narrow strings — no value-type overhead)
# ---------------------------------------------------------------------------

SlotBand = Literal["head", "body", "tail", "tools"]
"""Where in the assembled prompt this slot appears.

- ``head``  — system preamble (rules, personality, instructions).
- ``body``  — mid-prompt injected content (memory, retrieved context).
- ``tail``  — closing user-facing content (task statement).
- ``tools`` — tool definitions channel (assembled separately, not as messages).
"""

SlotNecessity = Literal["required", "preferred", "evictable"]
"""How the assembler may treat this slot when the budget is tight (S11/S8).

- ``required``  — must be present; assembly fails with ``ContextBudgetError`` if it cannot fit.
- ``preferred`` — included when budget allows; gracefully dropped if not (S8).
- ``evictable`` — last to be admitted; first to be dropped under budget pressure.
"""

SlotStatus = Literal["ok", "empty", "degraded", "unwired"]
"""Health of a slot after assembly.

- ``ok``      — content present and within budget.
- ``empty``   — contributor returned no content (zero chunks, zero tools).
- ``degraded``— contributor returned partial content or flagged a soft failure.
- ``unwired`` — no contributor registered for this slot.
"""


# ---------------------------------------------------------------------------
# Slot specification
# ---------------------------------------------------------------------------


class SlotSpec(BaseModel):
    """Declarative specification for a single context slot.

    Slots are assembled in band order (head → body → tail) and, within a band, by ascending
    ``priority``.  The ``min_tokens`` guard enforces a floor for required/preferred slots (S11).
    """

    model_config = ConfigDict(frozen=True)

    name: str
    """Unique slot identifier (e.g. ``"rules"``, ``"memory"``, ``"task"``)."""

    band: SlotBand
    """Prompt position: ``head`` | ``body`` | ``tail`` | ``tools``."""

    necessity: SlotNecessity
    """How the assembler treats this slot under budget pressure."""

    priority: int = 0
    """Tie-breaking within a band; lower values are assembled first."""

    min_tokens: int = 0
    """Minimum token floor; ignored when ``necessity == "evictable"``."""

    role: Literal["system", "user"] = "system"
    """ChatMessage role for chunks emitted by this slot."""


# ---------------------------------------------------------------------------
# Slot content — the output of a contributor
# ---------------------------------------------------------------------------


class SlotChunk(BaseModel):
    """A single unit of text content contributed to a slot."""

    model_config = ConfigDict(frozen=True)

    text: str
    """The rendered text for this chunk."""

    key: str
    """Stable unique key for this chunk (e.g. ``"rules:internal_1"``)."""

    source_slot: str
    """Name of the slot that produced this chunk."""


class SlotContent(BaseModel):
    """Everything a contributor returns for its slot assignment.

    A contributor may return text chunks, tool specs, or both.  An empty ``SlotContent``
    (no chunks, no tools) signals that the slot is genuinely empty for this request, which
    may be an acceptable outcome for preferred/evictable slots (S8).
    """

    model_config = ConfigDict(frozen=True)

    chunks: tuple[SlotChunk, ...] = ()
    """Ordered text chunks to be emitted as ChatMessages."""

    tools: tuple[ToolSpec, ...] = ()
    """Tool definitions contributed by this slot (e.g. the ``tools`` band slot)."""

    status: SlotStatus = "ok"
    """Self-reported health; ``"degraded"`` surfaces soft failures without aborting assembly."""

    detail: str | None = None
    """Optional human-readable explanation for a non-``ok`` status."""


# ---------------------------------------------------------------------------
# Slot allocation — the budget envelope handed to a contributor
# ---------------------------------------------------------------------------


class SlotAllocation(BaseModel):
    """The budget envelope the assembler hands to a contributor for its slot.

    When ``max_tokens`` is ``None`` the contributor has no explicit ceiling and must honour the
    overall ``ContextPolicy.total_budget`` itself; concrete assembler logic (3.1c/3.1d) will
    always populate this from the partitioned budget.
    """

    model_config = ConfigDict(frozen=True)

    max_tokens: int | None = None
    """Maximum tokens the contributor may use; ``None`` = unconstrained (assembler decides)."""


# ---------------------------------------------------------------------------
# Context policy
# ---------------------------------------------------------------------------


class ContextPolicy(BaseModel):
    """Declarative policy governing budget and memory behaviour for one context assembly pass.

    ``total_budget`` is the structural pre-call ceiling (S11); the assembler enforces it before
    the model is called — the model cannot self-terminate.
    """

    model_config = ConfigDict(frozen=True)

    total_budget: int = 8192
    """Hard token ceiling for the assembled call context (S11 — pre-call guard)."""

    memory: MemoryPolicy = DEFAULT_MEMORY_POLICY
    """Policy for the memory/injection slot; subset of ``total_budget``."""


DEFAULT_CONTEXT_POLICY: ContextPolicy = ContextPolicy(
    total_budget=8192,
    memory=DEFAULT_MEMORY_POLICY,
)
"""Ready-to-use default policy: 8 192-token budget, default memory policy."""


# ---------------------------------------------------------------------------
# Context request — the input to the assembler
# ---------------------------------------------------------------------------


class ContextRequest(BaseModel):
    """Everything the caller tells the assembler about what it needs for one turn.

    The assembler uses ``task`` as the required content for the ``tail/task`` slot.
    ``instructions`` overlays per-turn instruction content on top of the base ``head/instructions``
    slot.  ``query`` drives the ``body/memory`` slot via the recall/injection stack.  ``policy``
    overrides ``DEFAULT_CONTEXT_POLICY`` for this request.
    """

    model_config = ConfigDict(frozen=True)

    task: str = Field(min_length=1)
    """User-visible task statement; required input for the ``tail/task`` slot."""

    instructions: str | None = None
    """Per-turn instruction overlay for the ``head/instructions`` slot; ``None`` = use base."""

    query: RecallQuery | None = None
    """Recall query driving the ``body/memory`` slot; ``None`` = skip memory injection."""

    policy: ContextPolicy | None = None
    """Override policy for this request; ``None`` = use the assembler's ``default_policy``."""


# ---------------------------------------------------------------------------
# Diagnostic / report types
# ---------------------------------------------------------------------------


class SlotReport(BaseModel):
    """Per-slot diagnostic record emitted by the assembler after assembly completes."""

    model_config = ConfigDict(frozen=True)

    name: str
    """Slot name (matches ``SlotSpec.name``)."""

    status: SlotStatus
    """Final status of this slot after assembly."""

    tokens: int
    """Tokens consumed by this slot in the assembled context."""

    chunks_admitted: int
    """Number of chunks included in the assembled context."""

    chunks_dropped: int
    """Number of chunks dropped due to budget pressure."""

    evicted: bool
    """``True`` if the entire slot was evicted from the assembled context."""


class AssembledCallContext(BaseModel):
    """The fully assembled context ready to be passed to a model call.

    This is the output of ``ContextAssembler.assemble``.  It carries the complete message
    sequence, tool definitions, token accounting, and per-slot diagnostics.  Frozen so it
    can be safely cached and passed across async boundaries (S6).
    """

    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...]
    """Ordered ChatMessages ready for the model call (S4 — model-agnostic wire type)."""

    tools: tuple[ToolSpec, ...]
    """Tool definitions to pass alongside the messages."""

    token_count: int
    """Total budget consumption of the assembled context: message tokens plus serialised
    tool-spec tokens. Invariant: ``token_count <= budget`` (S11 pre-call guard)."""

    budget: int
    """The ``total_budget`` in effect for this assembly pass (S11 audit trail)."""

    slots: tuple[SlotReport, ...]
    """Per-slot diagnostics; order matches the assembly order (band → priority)."""

    memory: InjectedMemory | None = None
    """The result of the memory injection pass, if one was performed; ``None`` otherwise."""


# ---------------------------------------------------------------------------
# Canonical default slot layout
# ---------------------------------------------------------------------------

DEFAULT_SLOTS: tuple[SlotSpec, ...] = (
    SlotSpec(name="rules", band="head", necessity="required", priority=0),
    SlotSpec(name="personality", band="head", necessity="preferred", priority=1),
    SlotSpec(name="instructions", band="head", necessity="required", priority=2),
    SlotSpec(name="memory", band="body", necessity="evictable", priority=0),
    SlotSpec(name="tools", band="tools", necessity="preferred", priority=0),
    SlotSpec(name="task", band="tail", necessity="required", priority=0),
)
"""The canonical six-slot context layout.

Assembly order (within each band, ascending priority):

+---+---------------+--------+-----------+----------+
| # | name          | band   | necessity | priority |
+===+===============+========+===========+==========+
| 1 | rules         | head   | required  | 0        |
+---+---------------+--------+-----------+----------+
| 2 | personality   | head   | preferred | 1        |
+---+---------------+--------+-----------+----------+
| 3 | instructions  | head   | required  | 2        |
+---+---------------+--------+-----------+----------+
| 4 | memory        | body   | evictable | 0        |
+---+---------------+--------+-----------+----------+
| 5 | tools         | tools  | preferred | 0        |
+---+---------------+--------+-----------+----------+
| 6 | task          | tail   | required  | 0        |
+---+---------------+--------+-----------+----------+
"""
