"""Unit tests for cogworx.context types package (Pod 3.1a).

All tests are pure — no I/O, no model calls, no network.  Tests cover:

1. Literal type aliases resolve to the correct string sets.
2. Every frozen Pydantic model enforces immutability.
3. Default-value correctness for every model field.
4. SlotSpec construction with all fields.
5. SlotContent / SlotChunk / SlotAllocation construction.
6. ContextPolicy / DEFAULT_CONTEXT_POLICY correctness.
7. ContextRequest construction (required + optional fields).
8. SlotReport construction + diagnostics.
9. AssembledCallContext construction + optional memory field.
10. DEFAULT_SLOTS canonical layout invariants.
11. ContextContributor Protocol structural check.
12. ContextAssembler constructor, register, and stub assemble().
13. ContextBudgetError / ContextAssemblyError attributes.
14. No cogworx.runtime in sys.modules after importing cogworx.context.
"""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from cogworx.context import (
    DEFAULT_CONTEXT_POLICY,
    DEFAULT_SLOTS,
    AssembledCallContext,
    ContextAssembler,
    ContextAssemblyError,
    ContextBudgetError,
    ContextContributor,
    ContextPolicy,
    ContextRequest,
    SlotAllocation,
    SlotChunk,
    SlotContent,
    SlotReport,
    SlotSpec,
)
from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, MemoryPolicy
from cogworx.model.base import ChatMessage, ToolSpec
from cogworx.recall.query import RecallQuery

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _chunk(name: str = "rules", key: str = "rules:base") -> SlotChunk:
    return SlotChunk(text="some text", key=key, source_slot=name)


def _tool(name: str = "search") -> ToolSpec:
    return ToolSpec(name=name, description="A tool.", input_schema={"type": "object"})


def _message(role: str = "system", content: str = "hello") -> ChatMessage:
    return ChatMessage(role=role, content=content)


def _report(name: str = "rules") -> SlotReport:
    return SlotReport(
        name=name,
        status="ok",
        tokens=42,
        chunks_admitted=1,
        chunks_dropped=0,
        evicted=False,
    )


# ---------------------------------------------------------------------------
# 1. SlotBand, SlotNecessity, SlotStatus — validate via SlotSpec / SlotContent
# ---------------------------------------------------------------------------


def test_slot_band_valid_values() -> None:
    for band in ("head", "body", "tail", "tools"):
        spec = SlotSpec(name="s", band=band, necessity="required")
        assert spec.band == band


def test_slot_band_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        SlotSpec(name="s", band="middle", necessity="required")


def test_slot_necessity_valid_values() -> None:
    for nec in ("required", "preferred", "evictable"):
        spec = SlotSpec(name="s", band="head", necessity=nec)
        assert spec.necessity == nec


def test_slot_necessity_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        SlotSpec(name="s", band="head", necessity="optional")


def test_slot_status_valid_values() -> None:
    for status in ("ok", "empty", "degraded", "unwired"):
        sc = SlotContent(status=status)
        assert sc.status == status


def test_slot_status_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        SlotContent(status="broken")


# ---------------------------------------------------------------------------
# 2. SlotSpec — frozen + all fields
# ---------------------------------------------------------------------------


def test_slot_spec_defaults() -> None:
    spec = SlotSpec(name="task", band="tail", necessity="required")
    assert spec.priority == 0
    assert spec.min_tokens == 0
    assert spec.role == "system"


def test_slot_spec_full_construction() -> None:
    spec = SlotSpec(
        name="memory",
        band="body",
        necessity="evictable",
        priority=5,
        min_tokens=128,
        role="user",
    )
    assert spec.name == "memory"
    assert spec.band == "body"
    assert spec.necessity == "evictable"
    assert spec.priority == 5
    assert spec.min_tokens == 128
    assert spec.role == "user"


def test_slot_spec_is_frozen() -> None:
    spec = SlotSpec(name="rules", band="head", necessity="required")
    with pytest.raises((ValidationError, TypeError)):
        spec.name = "other"  # type: ignore[misc]


def test_slot_spec_role_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        SlotSpec(name="s", band="head", necessity="required", role="tool")


# ---------------------------------------------------------------------------
# 3. SlotChunk — frozen
# ---------------------------------------------------------------------------


def test_slot_chunk_construction() -> None:
    chunk = SlotChunk(text="Be helpful.", key="rules:base", source_slot="rules")
    assert chunk.text == "Be helpful."
    assert chunk.key == "rules:base"
    assert chunk.source_slot == "rules"


def test_slot_chunk_is_frozen() -> None:
    chunk = _chunk()
    with pytest.raises((ValidationError, TypeError)):
        chunk.text = "mutated"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. SlotContent — frozen, defaults
# ---------------------------------------------------------------------------


def test_slot_content_defaults() -> None:
    sc = SlotContent()
    assert sc.chunks == ()
    assert sc.tools == ()
    assert sc.status == "ok"
    assert sc.detail is None


def test_slot_content_with_chunks() -> None:
    ch = _chunk()
    sc = SlotContent(chunks=(ch,))
    assert sc.chunks == (ch,)


def test_slot_content_with_tools() -> None:
    t = _tool()
    sc = SlotContent(tools=(t,))
    assert sc.tools == (t,)


def test_slot_content_degraded() -> None:
    sc = SlotContent(status="degraded", detail="upstream failed")
    assert sc.status == "degraded"
    assert sc.detail == "upstream failed"


def test_slot_content_is_frozen() -> None:
    sc = SlotContent()
    with pytest.raises((ValidationError, TypeError)):
        sc.status = "degraded"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. SlotAllocation — frozen, max_tokens optional
# ---------------------------------------------------------------------------


def test_slot_allocation_default_none() -> None:
    alloc = SlotAllocation()
    assert alloc.max_tokens is None


def test_slot_allocation_with_value() -> None:
    alloc = SlotAllocation(max_tokens=512)
    assert alloc.max_tokens == 512


def test_slot_allocation_is_frozen() -> None:
    alloc = SlotAllocation(max_tokens=512)
    with pytest.raises((ValidationError, TypeError)):
        alloc.max_tokens = 1024  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 6. ContextPolicy + DEFAULT_CONTEXT_POLICY
# ---------------------------------------------------------------------------


def test_context_policy_defaults() -> None:
    policy = ContextPolicy()
    assert policy.total_budget == 8192
    assert policy.memory == DEFAULT_MEMORY_POLICY


def test_context_policy_custom_budget() -> None:
    policy = ContextPolicy(total_budget=4096)
    assert policy.total_budget == 4096


def test_context_policy_custom_memory() -> None:
    mem = MemoryPolicy(token_budget=512)
    policy = ContextPolicy(memory=mem)
    assert policy.memory.token_budget == 512


def test_context_policy_is_frozen() -> None:
    policy = ContextPolicy()
    with pytest.raises((ValidationError, TypeError)):
        policy.total_budget = 999  # type: ignore[misc]


def test_default_context_policy_total_budget() -> None:
    assert DEFAULT_CONTEXT_POLICY.total_budget == 8192


def test_default_context_policy_memory_is_default() -> None:
    assert DEFAULT_CONTEXT_POLICY.memory == DEFAULT_MEMORY_POLICY


# ---------------------------------------------------------------------------
# 7. ContextRequest — required task, all optionals
# ---------------------------------------------------------------------------


def test_context_request_minimal() -> None:
    req = ContextRequest(task="Summarize the docs.")
    assert req.task == "Summarize the docs."
    assert req.instructions is None
    assert req.query is None
    assert req.policy is None


def test_context_request_full() -> None:
    q = RecallQuery(text="recent events")
    p = ContextPolicy(total_budget=4096)
    req = ContextRequest(
        task="Answer the question.",
        instructions="Be concise.",
        query=q,
        policy=p,
    )
    assert req.instructions == "Be concise."
    assert req.query is not None
    assert req.query.text == "recent events"
    assert req.policy is not None
    assert req.policy.total_budget == 4096


def test_context_request_task_required() -> None:
    with pytest.raises(ValidationError):
        ContextRequest()  # type: ignore[call-arg]


def test_context_request_is_frozen() -> None:
    req = ContextRequest(task="t")
    with pytest.raises((ValidationError, TypeError)):
        req.task = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. SlotReport — construction + frozen
# ---------------------------------------------------------------------------


def test_slot_report_construction() -> None:
    r = SlotReport(
        name="memory",
        status="degraded",
        tokens=100,
        chunks_admitted=2,
        chunks_dropped=3,
        evicted=False,
    )
    assert r.name == "memory"
    assert r.status == "degraded"
    assert r.tokens == 100
    assert r.chunks_admitted == 2
    assert r.chunks_dropped == 3
    assert not r.evicted


def test_slot_report_evicted_true() -> None:
    r = SlotReport(
        name="memory",
        status="empty",
        tokens=0,
        chunks_admitted=0,
        chunks_dropped=5,
        evicted=True,
    )
    assert r.evicted is True


def test_slot_report_is_frozen() -> None:
    r = _report()
    with pytest.raises((ValidationError, TypeError)):
        r.name = "other"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 9. AssembledCallContext — construction + optional memory field
# ---------------------------------------------------------------------------


def test_assembled_call_context_minimal() -> None:
    msg = _message()
    ctx = AssembledCallContext(
        messages=(msg,),
        tools=(),
        token_count=10,
        budget=8192,
        slots=(_report(),),
    )
    assert ctx.token_count == 10
    assert ctx.memory is None


def test_assembled_call_context_with_tools() -> None:
    t = _tool()
    ctx = AssembledCallContext(
        messages=(),
        tools=(t,),
        token_count=0,
        budget=8192,
        slots=(),
    )
    assert ctx.tools == (t,)


def test_assembled_call_context_is_frozen() -> None:
    ctx = AssembledCallContext(
        messages=(),
        tools=(),
        token_count=0,
        budget=8192,
        slots=(),
    )
    with pytest.raises((ValidationError, TypeError)):
        ctx.token_count = 99  # type: ignore[misc]


def test_assembled_call_context_memory_field() -> None:
    from cogworx.injection.policy import InjectedMemory
    from cogworx.recall.results import AssembledContext

    assembled = AssembledContext(chunks=(), token_count=0, budget=2048, dropped=0)
    mem = InjectedMemory(context=assembled, status="ok")
    ctx = AssembledCallContext(
        messages=(),
        tools=(),
        token_count=0,
        budget=8192,
        slots=(),
        memory=mem,
    )
    assert ctx.memory is not None
    assert ctx.memory.status == "ok"


# ---------------------------------------------------------------------------
# 10. DEFAULT_SLOTS canonical layout invariants
# ---------------------------------------------------------------------------


def test_default_slots_count() -> None:
    assert len(DEFAULT_SLOTS) == 6


def test_default_slots_names() -> None:
    names = {s.name for s in DEFAULT_SLOTS}
    assert names == {"rules", "personality", "instructions", "memory", "tools", "task"}


def test_default_slots_rules() -> None:
    rules = next(s for s in DEFAULT_SLOTS if s.name == "rules")
    assert rules.band == "head"
    assert rules.necessity == "required"
    assert rules.priority == 0


def test_default_slots_personality() -> None:
    p = next(s for s in DEFAULT_SLOTS if s.name == "personality")
    assert p.band == "head"
    assert p.necessity == "preferred"
    assert p.priority == 1


def test_default_slots_instructions() -> None:
    instr = next(s for s in DEFAULT_SLOTS if s.name == "instructions")
    assert instr.band == "head"
    assert instr.necessity == "required"
    assert instr.priority == 2


def test_default_slots_memory() -> None:
    mem = next(s for s in DEFAULT_SLOTS if s.name == "memory")
    assert mem.band == "body"
    assert mem.necessity == "evictable"
    assert mem.priority == 0


def test_default_slots_tools() -> None:
    tools_slot = next(s for s in DEFAULT_SLOTS if s.name == "tools")
    assert tools_slot.band == "tools"
    assert tools_slot.necessity == "preferred"
    assert tools_slot.priority == 0


def test_default_slots_task() -> None:
    task = next(s for s in DEFAULT_SLOTS if s.name == "task")
    assert task.band == "tail"
    assert task.necessity == "required"
    assert task.priority == 0


def test_default_slots_head_band_priority_order() -> None:
    """Within the head band, rules < personality < instructions by priority."""
    head = sorted(
        (s for s in DEFAULT_SLOTS if s.band == "head"),
        key=lambda s: s.priority,
    )
    assert [s.name for s in head] == ["rules", "personality", "instructions"]


def test_default_slots_all_frozen() -> None:
    for spec in DEFAULT_SLOTS:
        with pytest.raises((ValidationError, TypeError)):
            spec.name = "mutated"  # type: ignore[misc]


def test_default_slots_required_slots_are_rules_instructions_task() -> None:
    required = {s.name for s in DEFAULT_SLOTS if s.necessity == "required"}
    assert required == {"rules", "instructions", "task"}


def test_default_slots_evictable_is_memory() -> None:
    evictable = {s.name for s in DEFAULT_SLOTS if s.necessity == "evictable"}
    assert evictable == {"memory"}


# ---------------------------------------------------------------------------
# 11. ContextContributor Protocol — structural check
# ---------------------------------------------------------------------------


class _GoodContributor:
    async def contribute(
        self,
        request: ContextRequest,
        allocation: SlotAllocation,
    ) -> SlotContent:
        return SlotContent()


class _BadContributor:
    def contribute(self, request: ContextRequest) -> SlotContent:  # missing allocation + async
        return SlotContent()


def test_context_contributor_protocol_accepts_conformant() -> None:
    assert isinstance(_GoodContributor(), ContextContributor)


def test_context_contributor_protocol_rejects_missing_method() -> None:
    class _NoMethod:
        pass

    assert not isinstance(_NoMethod(), ContextContributor)


# ---------------------------------------------------------------------------
# 12. ContextAssembler — constructor, register, stub assemble
# ---------------------------------------------------------------------------


def test_context_assembler_default_constructor() -> None:
    asm = ContextAssembler()
    assert asm._default_policy == DEFAULT_CONTEXT_POLICY
    assert asm._slots == DEFAULT_SLOTS


def test_context_assembler_custom_policy() -> None:
    policy = ContextPolicy(total_budget=4096)
    asm = ContextAssembler(default_policy=policy)
    assert asm._default_policy.total_budget == 4096


def test_context_assembler_custom_slots() -> None:
    custom = (SlotSpec(name="task", band="tail", necessity="required"),)
    asm = ContextAssembler(slots=custom)
    assert asm._slots == custom


def test_context_assembler_register_valid_slot() -> None:
    asm = ContextAssembler()
    contrib = _GoodContributor()
    asm.register("rules", contrib)
    assert asm._contributors["rules"] is contrib


def test_context_assembler_register_unknown_slot_raises() -> None:
    asm = ContextAssembler()
    with pytest.raises(KeyError, match="nonexistent"):
        asm.register("nonexistent", _GoodContributor())


@pytest.mark.asyncio
async def test_context_assembler_assemble_async_returns_assembled_context() -> None:
    """assemble() with no contributors registered returns an AssembledCallContext (Pod 3.1d)."""
    from cogworx.context.types import AssembledCallContext

    asm = ContextAssembler()
    req = ContextRequest(task="Do something.")
    result = await asm.assemble(req)
    assert isinstance(result, AssembledCallContext)


# ---------------------------------------------------------------------------
# 13. Error types — attributes
# ---------------------------------------------------------------------------


def test_context_budget_error_attributes() -> None:
    err = ContextBudgetError(
        required_tokens=10000,
        budget=8192,
        slot_names=("rules", "task"),
    )
    assert err.required_tokens == 10000
    assert err.budget == 8192
    assert err.slot_names == ("rules", "task")
    assert "10000" in str(err)
    assert "8192" in str(err)


def test_context_assembly_error_attributes() -> None:
    cause = ValueError("contributor exploded")
    err = ContextAssemblyError(slot_name="rules", cause=cause)
    assert err.slot_name == "rules"
    assert err.cause is cause
    assert "rules" in str(err)


def test_context_budget_error_is_exception() -> None:
    err = ContextBudgetError(required_tokens=1, budget=0, slot_names=())
    assert isinstance(err, Exception)


def test_context_assembly_error_is_exception() -> None:
    err = ContextAssemblyError(slot_name="task", cause=RuntimeError("boom"))
    assert isinstance(err, Exception)


# ---------------------------------------------------------------------------
# 14. Isolation: cogworx.context must NOT import cogworx.runtime
# ---------------------------------------------------------------------------


_NO_RUNTIME_SCRIPT = """\
import sys
import cogworx.context
assert "cogworx.runtime" not in sys.modules, (
    "import cogworx.context pulled in cogworx.runtime -- "
    "dependency direction violation: context must NOT import runtime (CANON D3)"
)
"""


def test_context_does_not_import_runtime() -> None:
    """cogworx.context must not pull cogworx.runtime into sys.modules (CANON D3)."""
    proc = subprocess.run(
        [sys.executable, "-c", _NO_RUNTIME_SCRIPT],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        "cogworx.context imported cogworx.runtime (dependency direction violation):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
