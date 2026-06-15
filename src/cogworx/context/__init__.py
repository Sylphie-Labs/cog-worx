"""cogworx.context — context assembly types, protocol, assembler skeleton, and contributors.

Provides the pure, frozen value types and the ``ContextContributor`` Protocol that together
define the context-assembly seam.  The concrete ``ContextAssembler.assemble()`` arbitration
logic is implemented in Pod 3.1d; this package provides the complete import surface today.

Public API
----------
Types (all frozen Pydantic value objects):

- :data:`SlotBand`           — ``Literal["head", "body", "tail", "tools"]``
- :data:`SlotNecessity`      — ``Literal["required", "preferred", "evictable"]``
- :data:`SlotStatus`         — ``Literal["ok", "empty", "degraded", "unwired"]``
- :class:`SlotSpec`          — declarative slot specification (name, band, necessity, …)
- :class:`SlotChunk`         — single unit of text contributed to a slot
- :class:`SlotContent`       — a contributor's full response (chunks + tools + status)
- :class:`SlotAllocation`    — budget envelope handed to a contributor
- :class:`ContextPolicy`     — total_budget + memory policy
- :class:`ContextRequest`    — assembler input (task, instructions, query, policy)
- :class:`SlotReport`        — per-slot diagnostic record
- :class:`AssembledCallContext` — fully assembled call-ready context

Constants:

- :data:`DEFAULT_CONTEXT_POLICY` — 8 192-token budget, default memory policy
- :data:`DEFAULT_SLOTS`          — canonical six-slot layout (rules/personality/instructions/
                                   memory/tools/task)

Protocol:

- :class:`ContextContributor` — ``runtime_checkable`` async contributor protocol

Assembler (skeleton):

- :class:`ContextAssembler`   — constructor + ``register``; ``assemble`` raises
                                ``NotImplementedError`` until Pod 3.1d

Errors:

- :class:`ContextBudgetError`   — required slots exceed total_budget (S11 pre-call guard)
- :class:`ContextAssemblyError` — required-slot contributor hard-failed

Concrete contributors (Pod 3.1c):

- :class:`StaticContributor`    — fixed text; placeholder for rules (3.4)
- :class:`TaskContributor`      — lifts ``request.task`` or ``request.instructions``
- :class:`ToolSpecContributor`  — wraps a static ``Sequence[ToolSpec]``
- :class:`MemoryContributor`    — wraps a ``MemoryInjector``; runs recall on ``request.query``

Personality contributors (Pod 3.3):

- :class:`PersonalityAttribute`  — frozen key/value metadata pair
- :class:`PersonalityProfile`    — frozen declarative personality description
- :func:`render_profile`         — pure renderer; canonical six-section template
- :class:`PersonalityContributor` — ``ContextContributor``-compatible; renders once at
                                    construction, injects into the ``personality`` slot

Rules contributors (Pod 3.4):

- :class:`Rule`                  — frozen single rule (text + optional label)
- :class:`RuleSet`               — frozen ordered collection of rules with preamble/header
- :func:`render_rules`           — pure renderer; header + preamble + numbered rules
- :class:`RulesContributor`      — ``ContextContributor``-compatible; renders once at
                                    construction, injects into the required ``rules`` slot;
                                    never truncates (assembler raises ContextBudgetError)

Dependency direction (CANON D3): ``runtime → context → injection → recall``.
This package does NOT import ``cogworx.runtime``.

CANON sections cited: S1, S4, S8, S9, S11.
"""

from __future__ import annotations

from cogworx.context.assembler import ContextAssembler
from cogworx.context.contributor import ContextContributor
from cogworx.context.contributors import (
    MemoryContributor,
    StaticContributor,
    TaskContributor,
    ToolSpecContributor,
)
from cogworx.context.errors import ContextAssemblyError, ContextBudgetError
from cogworx.context.personality import (
    PersonalityAttribute,
    PersonalityContributor,
    PersonalityProfile,
    render_profile,
)
from cogworx.context.rules import (
    Rule,
    RulesContributor,
    RuleSet,
    render_rules,
)
from cogworx.context.types import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_SLOTS,
    AssembledCallContext,
    ContextPolicy,
    ContextRequest,
    SlotAllocation,
    SlotBand,
    SlotChunk,
    SlotContent,
    SlotNecessity,
    SlotReport,
    SlotSpec,
    SlotStatus,
)

__all__ = [
    "DEFAULT_CONTEXT_POLICY",
    "DEFAULT_SLOTS",
    "AssembledCallContext",
    "ContextAssembler",
    "ContextAssemblyError",
    "ContextBudgetError",
    "ContextContributor",
    "ContextPolicy",
    "ContextRequest",
    "MemoryContributor",
    "PersonalityAttribute",
    "PersonalityContributor",
    "PersonalityProfile",
    "Rule",
    "RuleSet",
    "RulesContributor",
    "SlotAllocation",
    "SlotBand",
    "SlotChunk",
    "SlotContent",
    "SlotNecessity",
    "SlotReport",
    "SlotSpec",
    "SlotStatus",
    "StaticContributor",
    "TaskContributor",
    "ToolSpecContributor",
    "render_profile",
    "render_rules",
]
