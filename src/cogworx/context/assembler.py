"""ContextAssembler -- two-phase budget arbitration + banded U-fold rendering (Pod 3.1d).

Dependency direction (CANON D3): runtime -> context -> injection -> recall.
This module does NOT import ``cogworx.runtime``.

CANON sections cited:
  - S1  -- model work off the write path; assembly is a pure value transformation.
           ``assemble`` NEVER calls ``model.complete``; only ``count_tokens`` is permitted.
  - S4  -- model-agnostic; Model reference is optional and duck-typed.
  - S6  -- determinism / replay-friendly: same request + same contributor outputs ->
           BYTE-IDENTICAL message list.  No clock access here (clock stays in the injector).
  - S8  -- graceful degradation; preferred/evictable slot failures are absorbed.
  - S11 -- cost bounded structurally; total_budget enforced pre-call (ContextBudgetError).

Contract changelog:
  - 2026-06-12 (Pod 3.1a): initial surface -- ContextAssembler constructor signature + stub.
    Full assemble() logic deferred to Pod 3.1d.
  - 2026-06-12 (Pod 3.1d): replaced NotImplementedError stub with full two-phase budget
    arbitration + banded U-fold rendering.
  - 2026-06-12 (Pod 3.1): token_count SEMANTICS CHANGE — now includes serialised tool-spec
    tokens in addition to message tokens (was message-tokens-only). Callers comparing
    token_count to budget must use the new total-consumption semantics.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable
from typing import cast

from cogworx.context.contributor import ContextContributor
from cogworx.context.errors import ContextAssemblyError, ContextBudgetError
from cogworx.context.types import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_SLOTS,
    AssembledCallContext,
    ContextPolicy,
    ContextRequest,
    SlotAllocation,
    SlotChunk,
    SlotContent,
    SlotReport,
    SlotSpec,
    SlotStatus,
)
from cogworx.injection.policy import InjectedMemory
from cogworx.model.base import ChatMessage, ToolSpec
from cogworx.recall.assembly import approx_tokens

__all__ = ["ContextAssembler"]

# ---------------------------------------------------------------------------
# Section separator markers (stable + deterministic -- S6)
# ---------------------------------------------------------------------------

_HEAD_SECTION_SEP = "\n\n---\n\n"
"""Separator joining head-band sections (rules / personality / instructions)."""

_BODY_ENVELOPE_START = "<!-- memory:start -->"
_BODY_ENVELOPE_END = "<!-- memory:end -->"
"""Deterministic envelope around the body/memory message (U-fold container)."""

# Band sort order (head=0, body=1, tail=2, tools=3).
_BAND_ORDER: dict[str, int] = {"head": 0, "body": 1, "tail": 2, "tools": 3}


# ---------------------------------------------------------------------------
# Token-counter helper (duck-typed -- S4/S8)
# ---------------------------------------------------------------------------


def _resolve_token_counter(model: object | None) -> Callable[[str], int]:
    """Return a token-counting callable from *model*, falling back to approx_tokens.

    Pure duck-type check -- avoids any runtime import of the ``Model`` class (S4/S8).
    Delegates to ``cogworx.injection.injector.resolve_token_counter`` when available;
    falls back to a local duck-type check + ``approx_tokens``.
    """
    # Delegate to the injection module when available (avoids duplication).
    try:
        from cogworx.injection.injector import (
            resolve_token_counter,
        )

        return resolve_token_counter(model)
    except ImportError:
        pass

    if model is None:
        return approx_tokens
    counter = getattr(model, "count_tokens", None)
    if callable(counter):
        typed: Callable[[str], int] = cast(Callable[[str], int], counter)
        return typed
    return approx_tokens


def _count_tool_tokens(tools: tuple[ToolSpec, ...], count_fn: Callable[[str], int]) -> int:
    """Estimate token cost of serialised tool specs (JSON representation)."""
    if not tools:
        return 0
    return sum(count_fn(json.dumps(t.model_dump(), separators=(",", ":"))) for t in tools)


def _slot_text(content: SlotContent) -> str:
    """Return all chunk texts joined with newlines."""
    return "\n".join(c.text for c in content.chunks)


# ---------------------------------------------------------------------------
# Assembler
# ---------------------------------------------------------------------------


class ContextAssembler:
    """Assembles a fully-budgeted ``AssembledCallContext`` from registered slot contributors.

    Construction:
        *slots* defines the slot layout (defaults to ``DEFAULT_SLOTS``).  One
        ``ContextContributor`` may be registered per slot name via ``register``.  *model* is
        optional and is duck-typed for ``count_tokens``; no runtime ``Model`` import is needed
        (S4/S8).  *default_policy* overrides ``DEFAULT_CONTEXT_POLICY`` for all requests that
        do not supply their own policy.

    Args:
        slots:          Ordered slot specifications.  Defaults to ``DEFAULT_SLOTS``.
        model:          Optional model object; duck-typed for ``count_tokens`` (S4).
        default_policy: Default ``ContextPolicy`` for requests that omit their own.
    """

    def __init__(
        self,
        *,
        slots: tuple[SlotSpec, ...] = DEFAULT_SLOTS,
        model: object | None = None,
        default_policy: ContextPolicy = DEFAULT_CONTEXT_POLICY,
    ) -> None:
        self._slots: tuple[SlotSpec, ...] = slots
        self._model: object | None = model
        self._default_policy: ContextPolicy = default_policy
        self._contributors: dict[str, ContextContributor] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, slot_name: str, contributor: ContextContributor) -> None:
        """Register a contributor for *slot_name*.

        Overwrites any previously registered contributor for the same slot.  Raises
        ``KeyError`` if *slot_name* does not appear in the configured slot layout.

        Args:
            slot_name:   Must match a ``SlotSpec.name`` in ``self._slots``.
            contributor: Any object satisfying the ``ContextContributor`` Protocol.
        """
        valid_names = {s.name for s in self._slots}
        if slot_name not in valid_names:
            raise KeyError(
                f"Slot {slot_name!r} is not in the configured layout "
                f"(known slots: {sorted(valid_names)!r})."
            )
        self._contributors[slot_name] = contributor

    # ------------------------------------------------------------------
    # Assembly
    # ------------------------------------------------------------------

    async def assemble(self, request: ContextRequest) -> AssembledCallContext:
        """Assemble a fully-budgeted ``AssembledCallContext`` for *request*.

        **Two-phase budget arbitration:**

        Phase 1 -- REQUIRED slots are charged first.  If their combined token cost
        (including envelope/separator overhead) exceeds ``total_budget``, a
        ``ContextBudgetError`` is raised.  Required slots are NEVER evicted or truncated.

        Phase 2 -- PREFERRED slots are charged against the remainder.  Personality is
        dropped WHOLE under pressure (half a persona is worse than none).  Tool specs
        are dropped PER-TOOL greedy in declared order (some tools survive under pressure).

        Phase 3 -- EVICTABLE pool: the memory slot receives
        ``min(remainder - memory_envelope, policy.memory.token_budget)`` tokens.
        Its contributor applies per-chunk greedy-skip + U-fold internally.

        **Banded rendering -> messages:**
        - ``head`` band  -> ONE ``system`` ``ChatMessage`` (rules / personality / instructions).
        - ``body`` band  -> ONE ``user`` ``ChatMessage`` (memory), with deterministic envelope.
        - ``tail`` band  -> final ``user`` ``ChatMessage`` (the task).
        - ``tools`` band -> ``tools`` tuple on ``AssembledCallContext`` (out-of-band).

        **Determinism (S6):** same request + same contributor outputs -> byte-identical messages.
        No clock access here -- clock stamping lives inside ``MemoryInjector``.

        Raises:
            ContextBudgetError:   Required slots exceed total_budget.
            ContextAssemblyError: A required-slot contributor raised an exception.
        """
        policy = request.policy if request.policy is not None else self._default_policy
        budget = policy.total_budget
        count_fn = _resolve_token_counter(self._model)

        # ------------------------------------------------------------------
        # Sort slots into (band, priority, name) order -- deterministic (S6).
        # ------------------------------------------------------------------
        ordered_specs: list[SlotSpec] = sorted(
            self._slots,
            key=lambda s: (_BAND_ORDER[s.band], s.priority, s.name),
        )

        required_specs = [s for s in ordered_specs if s.necessity == "required"]
        preferred_specs = [s for s in ordered_specs if s.necessity == "preferred"]
        evictable_specs = [s for s in ordered_specs if s.necessity == "evictable"]

        # ------------------------------------------------------------------
        # Fan-out: call all registered contributors (initial allocation = full budget).
        # Required-slot hard failures -> ContextAssemblyError.
        # Preferred/evictable hard failures -> degraded empty (S8).
        # The "task" slot is synthesised from request.task when no contributor is registered.
        # ------------------------------------------------------------------
        slot_contents: dict[str, SlotContent] = {}
        # Pre-initialize evictable slots as "unwired" so Phase 3 can safely overwrite them.
        for spec in evictable_specs:
            slot_contents[spec.name] = SlotContent(status="unwired")
        for spec in ordered_specs:
            if spec.necessity == "evictable":
                continue  # Evictable content is produced in Phase 3 only.
            contributor = self._contributors.get(spec.name)
            if contributor is None:
                # Task slot: synthesise from request.task when no contributor registered.
                if spec.name == "task" and request.task:
                    slot_contents[spec.name] = SlotContent(
                        chunks=(
                            SlotChunk(
                                text=request.task,
                                key="task:0",
                                source_slot="task",
                            ),
                        ),
                        status="ok",
                    )
                else:
                    slot_contents[spec.name] = SlotContent(status="unwired")
                continue

            allocation = SlotAllocation(max_tokens=budget)
            try:
                content = await asyncio.shield(contributor.contribute(request, allocation))
            except Exception as exc:
                if spec.necessity == "required":
                    raise ContextAssemblyError(slot_name=spec.name, cause=exc) from exc
                slot_contents[spec.name] = SlotContent(
                    status="degraded",
                    detail=f"contributor raised {type(exc).__name__}: {exc}",
                )
                continue

            # Required-slot contract guard: a required contributor that reports ok/degraded
            # but returns no chunks and no tools has silently violated the contract.
            # status="empty" is legitimate (the contributor genuinely has nothing to say);
            # status="ok"/"degraded" with no output is a contributor bug — raise loudly.
            if (
                spec.necessity == "required"
                and content.status in ("ok", "degraded")
                and not content.chunks
                and not content.tools
            ):
                violation = ValueError(
                    f"Required slot {spec.name!r} contributor returned "
                    f"status={content.status!r} with no chunks — "
                    "this is a contributor contract violation"
                )
                raise ContextAssemblyError(slot_name=spec.name, cause=violation)

            # Task slot: fall back to request.task if contributor returned nothing.
            if spec.name == "task" and not content.chunks and request.task:
                slot_contents[spec.name] = SlotContent(
                    chunks=(
                        SlotChunk(
                            text=request.task,
                            key="task:0",
                            source_slot="task",
                        ),
                    ),
                    status="ok",
                )
            else:
                slot_contents[spec.name] = content

        # ------------------------------------------------------------------
        # Phase 1 -- Charge REQUIRED slots.
        # Head-band required slots joined with _HEAD_SECTION_SEP -> one system message.
        # Body-band required slots wrapped in the body envelope.
        # Tail-band required slots emitted as bare text.
        # ------------------------------------------------------------------
        req_head_specs = [s for s in required_specs if s.band == "head"]
        req_body_specs = [s for s in required_specs if s.band == "body"]
        req_tail_specs = [s for s in required_specs if s.band == "tail"]

        head_req_texts = [t for s in req_head_specs if (t := _slot_text(slot_contents[s.name]))]
        head_req_joined = _HEAD_SECTION_SEP.join(head_req_texts)
        required_tokens = count_fn(head_req_joined) if head_req_joined else 0

        for spec in req_body_specs:
            text = _slot_text(slot_contents[spec.name])
            if text:
                envelope = f"{_BODY_ENVELOPE_START}\n{text}\n{_BODY_ENVELOPE_END}"
                required_tokens += count_fn(envelope)

        for spec in req_tail_specs:
            text = _slot_text(slot_contents[spec.name])
            if text:
                required_tokens += count_fn(text)

        if required_tokens > budget:
            raise ContextBudgetError(
                required_tokens=required_tokens,
                budget=budget,
                slot_names=tuple(s.name for s in required_specs),
            )

        remaining = budget - required_tokens

        # ------------------------------------------------------------------
        # Phase 2 -- PREFERRED slots against remainder.
        # Personality: dropped WHOLE if it doesn't fit.
        # Tools: admitted PER-TOOL greedy in declared order.
        # Other preferred: dropped whole if they don't fit.
        # ------------------------------------------------------------------
        preferred_admitted: dict[str, bool] = {}
        admitted_tools: list[ToolSpec] = []

        # Running list of admitted head-section texts (for delta charging).
        # Initialized from non-empty required head texts in head_specs_ordered position order.
        head_specs_ordered: list[SlotSpec] = [s for s in ordered_specs if s.band == "head"]
        admitted_head_sections: list[str] = [
            t
            for s in head_specs_ordered
            if s.necessity == "required"
            for t in [_slot_text(slot_contents[s.name])]
            if t
        ]

        for spec in preferred_specs:
            content = slot_contents[spec.name]

            if not content.chunks and not content.tools:
                # Empty contributor response -> admitted (as empty, zero cost).
                preferred_admitted[spec.name] = True
                continue

            if spec.band == "tools":
                # Per-tool greedy admission in declared order.
                for tool in content.tools:
                    tool_cost = count_fn(json.dumps(tool.model_dump(), separators=(",", ":")))
                    if tool_cost <= remaining:
                        admitted_tools.append(tool)
                        remaining -= tool_cost
                preferred_admitted[spec.name] = True
            elif spec.name == "personality" or spec.band == "head":
                # Head preferred: drop WHOLE under pressure.
                # Charge the incremental delta of joining this section into the head message.
                text = _slot_text(content)
                if not text:
                    preferred_admitted[spec.name] = True
                    continue
                # Find insertion position for this slot among head_specs_ordered.
                # Build candidate_sections: admitted_head_sections with this text inserted at
                # its position.
                candidate_sections: list[str] = []
                inserted = False
                for hspec in head_specs_ordered:
                    if hspec.necessity == "required":
                        t = _slot_text(slot_contents[hspec.name])
                        if t:
                            candidate_sections.append(t)
                    elif hspec.name == spec.name:
                        # This is our candidate slot — insert it here.
                        candidate_sections.append(text)
                        inserted = True
                    elif hspec.necessity == "preferred" and preferred_admitted.get(
                        hspec.name, False
                    ):
                        t = _slot_text(slot_contents[hspec.name])
                        if t:
                            candidate_sections.append(t)
                if not inserted:
                    candidate_sections.append(text)
                # Delta = count of joined candidate minus count of current admitted.
                current_cost = (
                    count_fn(_HEAD_SECTION_SEP.join(admitted_head_sections))
                    if admitted_head_sections
                    else 0
                )
                candidate_cost = (
                    count_fn(_HEAD_SECTION_SEP.join(candidate_sections))
                    if candidate_sections
                    else 0
                )
                delta = candidate_cost - current_cost
                if delta <= remaining:
                    remaining -= delta
                    preferred_admitted[spec.name] = True
                    admitted_head_sections = candidate_sections
                else:
                    preferred_admitted[spec.name] = False
            else:
                # Other preferred slots: drop whole if they don't fit.
                text = _slot_text(content)
                cost = count_fn(text) if text else 0
                if cost <= remaining:
                    remaining -= cost
                    preferred_admitted[spec.name] = True
                else:
                    preferred_admitted[spec.name] = False

        # ------------------------------------------------------------------
        # Phase 3 -- EVICTABLE pool (memory).
        # Memory gets: min(remaining - envelope_overhead, policy.memory.token_budget).
        # Re-call contributor with the refined budget so it can trim internally.
        # ------------------------------------------------------------------
        _MEMORY_ENVELOPE_OVERHEAD = count_fn(f"{_BODY_ENVELOPE_START}\n\n{_BODY_ENVELOPE_END}")
        evictable_budgets: dict[str, int] = {}
        phase3_called: set[str] = set()
        for spec in evictable_specs:
            memory_remainder = max(0, remaining - _MEMORY_ENVELOPE_OVERHEAD)
            alloc = min(memory_remainder, policy.memory.token_budget)
            evictable_budgets[spec.name] = alloc

            contributor = self._contributors.get(spec.name)
            if contributor is None:
                slot_contents[spec.name] = SlotContent(status="unwired")
            elif alloc == 0:
                slot_contents[spec.name] = SlotContent(status="empty")
            else:
                refined_allocation = SlotAllocation(max_tokens=alloc)
                phase3_called.add(spec.name)
                try:
                    content = await asyncio.shield(
                        contributor.contribute(request, refined_allocation)
                    )
                    slot_contents[spec.name] = content
                except Exception as exc:
                    slot_contents[spec.name] = SlotContent(
                        status="degraded",
                        detail=f"contributor raised {type(exc).__name__}: {exc}",
                    )

        # Post-render trim: ensure the rendered memory envelope doesn't exceed the reservation.
        # Memory is EVICTABLE so trimming is safe (NEVER touches required slots).
        for spec in evictable_specs:
            content = slot_contents[spec.name]
            if not content.chunks:
                continue
            text = _slot_text(content)
            if not text:
                continue
            envelope = f"{_BODY_ENVELOPE_START}\n{text}\n{_BODY_ENVELOPE_END}"
            body_cost = count_fn(envelope)
            alloc = evictable_budgets.get(spec.name, 0)
            if body_cost > alloc:
                # Drop chunks from the end until it fits (or becomes empty).
                chunks = list(content.chunks)
                while chunks:
                    chunks.pop()
                    if not chunks:
                        slot_contents[spec.name] = SlotContent(status="empty")
                        break
                    new_text = "\n".join(c.text for c in chunks)
                    new_envelope = f"{_BODY_ENVELOPE_START}\n{new_text}\n{_BODY_ENVELOPE_END}"
                    if count_fn(new_envelope) <= alloc:
                        slot_contents[spec.name] = SlotContent(
                            chunks=tuple(chunks),
                            status=content.status,
                            detail=content.detail,
                        )
                        break

        # ------------------------------------------------------------------
        # Banded rendering -> messages (S6 deterministic).
        # ------------------------------------------------------------------
        messages: list[ChatMessage] = []

        # --- HEAD -> system message ---
        # head_specs_ordered is defined above in Phase 2 (delta-charging init).
        head_sections: list[str] = []
        for spec in head_specs_ordered:
            if spec.necessity == "required":
                text = _slot_text(slot_contents[spec.name])
                if text:
                    head_sections.append(text)
            elif spec.necessity == "preferred":
                if preferred_admitted.get(spec.name, False):
                    text = _slot_text(slot_contents[spec.name])
                    if text:
                        head_sections.append(text)

        if head_sections:
            messages.append(
                ChatMessage(role="system", content=_HEAD_SECTION_SEP.join(head_sections))
            )

        # --- BODY -> user message (memory) ---
        body_specs_ordered = [s for s in ordered_specs if s.band == "body"]
        for spec in body_specs_ordered:
            content = slot_contents[spec.name]
            if spec.necessity in ("required", "evictable"):
                text = _slot_text(content)
                if text:
                    envelope = f"{_BODY_ENVELOPE_START}\n{text}\n{_BODY_ENVELOPE_END}"
                    messages.append(ChatMessage(role="user", content=envelope))
            else:
                # preferred body slot
                if preferred_admitted.get(spec.name, False):
                    text = _slot_text(content)
                    if text:
                        messages.append(ChatMessage(role="user", content=text))

        # --- TAIL -> final user message ---
        tail_specs_ordered = [s for s in ordered_specs if s.band == "tail"]
        for spec in tail_specs_ordered:
            if spec.necessity == "required":
                text = _slot_text(slot_contents[spec.name])
                if text:
                    messages.append(ChatMessage(role="user", content=text))
            elif spec.necessity == "preferred":
                if preferred_admitted.get(spec.name, False):
                    text = _slot_text(slot_contents[spec.name])
                    if text:
                        messages.append(ChatMessage(role="user", content=text))

        # Tools tuple (out-of-band, never a message).
        tools_tuple: tuple[ToolSpec, ...] = tuple(admitted_tools)

        # ------------------------------------------------------------------
        # Token accounting: messages + serialised tool specs (Fix 2, Pod 3.1).
        # token_count = total budget consumption (S11 pre-call invariant).
        # ------------------------------------------------------------------
        token_count = sum(count_fn(m.content) for m in messages) + _count_tool_tokens(
            tools_tuple, count_fn
        )

        # ------------------------------------------------------------------
        # Build SlotReports (diagnostics).
        # ------------------------------------------------------------------
        slot_reports: list[SlotReport] = []
        for spec in ordered_specs:
            content = slot_contents[spec.name]
            status: SlotStatus
            tok: int
            n_admitted: int
            n_dropped: int
            evicted: bool

            if spec.necessity == "required":
                status = content.status if content.status != "ok" else "ok"
                n_admitted = len(content.chunks)
                n_dropped = 0
                evicted = False
                text = _slot_text(content)
                tok = count_fn(text) if text else 0

            elif spec.necessity == "preferred":
                if spec.band == "tools":
                    total_tools = len(content.tools)
                    n_admitted = len(admitted_tools)
                    n_dropped = total_tools - n_admitted
                    evicted = total_tools > 0 and n_admitted == 0
                    tok = _count_tool_tokens(tools_tuple, count_fn)
                    if not content.chunks and not content.tools:
                        status = "empty"
                    elif evicted:
                        status = "degraded"
                    else:
                        status = content.status
                else:
                    admitted = preferred_admitted.get(spec.name, True)
                    n_admitted = len(content.chunks) if admitted else 0
                    n_dropped = 0 if admitted else len(content.chunks)
                    evicted = not admitted and (len(content.chunks) > 0 or bool(content.tools))
                    text = _slot_text(content)
                    tok = count_fn(text) if admitted and text else 0
                    if not content.chunks and not content.tools:
                        status = "empty"
                    elif evicted:
                        status = "degraded"
                    else:
                        status = content.status

            else:
                # evictable (memory)
                text = _slot_text(content)
                tok = count_fn(text) if text else 0
                n_admitted = len(content.chunks)
                n_dropped = 0  # memory contributor handles internal drops
                evicted = n_admitted == 0 and evictable_budgets.get(spec.name, 0) == 0
                status = "empty" if not content.chunks else content.status

            slot_reports.append(
                SlotReport(
                    name=spec.name,
                    status=status,
                    tokens=tok,
                    chunks_admitted=n_admitted,
                    chunks_dropped=n_dropped,
                    evicted=evicted,
                )
            )

        # ------------------------------------------------------------------
        # Extract InjectedMemory from the memory slot's contributor (3.1c hand-off).
        # The MemoryContributor stashes it on ``last_injected_memory`` (duck-typed).
        # Fix 3: only read from slots actually invoked in Phase 3 (phase3_called).
        # ------------------------------------------------------------------
        injected_memory: InjectedMemory | None = None
        for spec in evictable_specs:
            if spec.name not in phase3_called:
                continue
            contributor = self._contributors.get(spec.name)
            if contributor is not None:
                candidate = getattr(contributor, "last_injected_memory", None)
                if isinstance(candidate, InjectedMemory):
                    injected_memory = candidate
                    break

        return AssembledCallContext(
            messages=tuple(messages),
            tools=tools_tuple,
            token_count=token_count,
            budget=budget,
            slots=tuple(slot_reports),
            memory=injected_memory,
        )
