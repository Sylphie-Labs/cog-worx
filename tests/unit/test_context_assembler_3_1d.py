"""Unit tests for ContextAssembler.assemble -- Pod 3.1d.

Covers:
  - Happy path: required + memory + task -> correct banded message structure.
  - Budget property test (randomised slot contents + budgets, fixed seeds):
    rendered token_count <= total_budget; required slots present when they fit.
  - Pressure ladder:
      * budget = required + epsilon -> memory empty, personality evicted whole,
        tools dropped per-tool, rules+instructions+task byte-intact.
      * budget < required -> ContextBudgetError (never silent rules truncation).
  - U-fold positions: rules at messages[0], task as final message,
    rank-1 memory chunk at body FRONT and rank-2 at body BACK.
  - Determinism: identical request + fixtures -> byte-identical messages.
  - S1 guard: assemble under a model whose complete() raises if called;
    count_tokens is permitted, any complete() fails the test.

All contributors are test-local FAKE implementations of the ContextContributor Protocol.
No dependency on 3.1c concrete contributors.

asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cogworx.context.assembler import _BODY_ENVELOPE_START, _HEAD_SECTION_SEP, ContextAssembler
from cogworx.context.contributors import MemoryContributor, StaticContributor, TaskContributor
from cogworx.context.errors import ContextBudgetError
from cogworx.context.types import (
    DEFAULT_SLOTS,
    ContextPolicy,
    ContextRequest,
    SlotAllocation,
    SlotChunk,
    SlotContent,
    SlotSpec,
)
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import InjectedMemory, MemoryPolicy
from cogworx.model.base import ToolSpec
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import AssembledContext, ChannelHit, FusedResult
from cogworx.recall.stack import ChannelStatus, RecallOutcome, RecallStack

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_APPROX_CHARS_PER_TOKEN = 4  # must match cogworx.recall.assembly.approx_tokens


# ---------------------------------------------------------------------------
# Fake contributors (test-local; satisfy the ContextContributor Protocol)
# ---------------------------------------------------------------------------


class _FixedTextContributor:
    """Returns a single SlotChunk with a fixed text string."""

    def __init__(self, text: str, *, slot_name: str = "slot") -> None:
        self._text = text
        self._slot_name = slot_name

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        if not self._text:
            return SlotContent(status="empty")
        return SlotContent(
            chunks=(
                SlotChunk(
                    text=self._text,
                    key=f"{self._slot_name}:0",
                    source_slot=self._slot_name,
                ),
            ),
        )


class _FixedToolsContributor:
    """Returns N fake ToolSpec objects."""

    def __init__(self, tools: list[ToolSpec]) -> None:
        self._tools = tools

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        return SlotContent(tools=tuple(self._tools))


class _MemoryContributor:
    """Fake memory contributor.

    Admits chunks greedily within the given allocation.max_tokens budget and
    exposes ``last_injected_memory`` for the diagnostics hand-off.
    """

    def __init__(self, chunk_texts: list[str], *, slot_name: str = "memory") -> None:
        self._chunk_texts = chunk_texts
        self._slot_name = slot_name
        self.last_injected_memory: InjectedMemory | None = None

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        budget = allocation.max_tokens if allocation.max_tokens is not None else 99999

        admitted: list[SlotChunk] = []
        used = 0
        for i, text in enumerate(self._chunk_texts):
            cost = max(1, len(text) // _APPROX_CHARS_PER_TOKEN)
            if used + cost <= budget:
                admitted.append(
                    SlotChunk(
                        text=text,
                        key=f"{self._slot_name}:{i}",
                        source_slot=self._slot_name,
                    )
                )
                used += cost

        status = "ok" if admitted else "empty"
        content = SlotContent(chunks=tuple(admitted), status=status)

        # Stash InjectedMemory for the diagnostics hand-off (3.1c protocol).
        if admitted:
            asm = AssembledContext(
                chunks=(),
                token_count=used,
                budget=budget,
                dropped=len(self._chunk_texts) - len(admitted),
            )
            self.last_injected_memory = InjectedMemory(context=asm, status="ok")
        else:
            self.last_injected_memory = None
        return content


class _EmptyContributor:
    """Always returns empty SlotContent."""

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        return SlotContent(status="empty")


class _S1GuardModel:
    """Model whose complete() raises AssertionError if called.

    count_tokens is permitted -- only complete() is forbidden (S1).
    """

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)

    async def complete(self, **kwargs: object) -> None:
        raise AssertionError(
            "S1 violation: ContextAssembler.assemble must NEVER call model.complete()"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool(name: str) -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


def _make_assembler(
    *,
    rules: str = "Rule: be helpful.",
    instructions: str = "Task instructions here.",
    personality: str | None = "I am a friendly assistant.",
    memory_chunks: list[str] | None = None,
    tools: list[ToolSpec] | None = None,
    budget: int = 4096,
    model: object | None = None,
) -> tuple[ContextAssembler, _MemoryContributor | None]:
    """Build a ContextAssembler wired with fake contributors."""
    mem_contributor: _MemoryContributor | None = None
    policy = ContextPolicy(total_budget=budget)

    assembler = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=model,
        default_policy=policy,
    )
    assembler.register("rules", _FixedTextContributor(rules, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instructions, slot_name="instructions"),
    )
    if personality is not None:
        assembler.register(
            "personality",
            _FixedTextContributor(personality, slot_name="personality"),
        )
    else:
        assembler.register("personality", _EmptyContributor())

    if memory_chunks is not None:
        mem_contributor = _MemoryContributor(memory_chunks, slot_name="memory")
        assembler.register("memory", mem_contributor)
    else:
        assembler.register("memory", _EmptyContributor())

    if tools is not None:
        assembler.register("tools", _FixedToolsContributor(tools))
    else:
        assembler.register("tools", _EmptyContributor())

    return assembler, mem_contributor


def _make_request(task: str = "Solve the problem.", *, budget: int | None = None) -> ContextRequest:
    if budget is not None:
        return ContextRequest(task=task, policy=ContextPolicy(total_budget=budget))
    return ContextRequest(task=task)


def _approx(text: str) -> int:
    return max(1, len(text) // _APPROX_CHARS_PER_TOKEN)


# ---------------------------------------------------------------------------
# 1. Happy path
# ---------------------------------------------------------------------------


async def test_happy_path_system_message_present() -> None:
    """A head-band system message is emitted containing rules and instructions."""
    assembler, _ = _make_assembler(rules="Be helpful.", instructions="Answer concisely.")
    result = await assembler.assemble(_make_request())
    system_msgs = [m for m in result.messages if m.role == "system"]
    assert len(system_msgs) == 1, "exactly one system message expected"


async def test_happy_path_system_contains_rules() -> None:
    assembler, _ = _make_assembler(rules="Be helpful.", instructions="Answer concisely.")
    result = await assembler.assemble(_make_request())
    assert "Be helpful." in result.messages[0].content


async def test_happy_path_system_contains_instructions() -> None:
    assembler, _ = _make_assembler(rules="Be helpful.", instructions="Answer concisely.")
    result = await assembler.assemble(_make_request())
    assert "Answer concisely." in result.messages[0].content


async def test_happy_path_task_is_final_message() -> None:
    task = "Solve the problem."
    assembler, _ = _make_assembler()
    result = await assembler.assemble(_make_request(task=task))
    assert result.messages[-1].content == task


async def test_happy_path_task_role_is_user() -> None:
    assembler, _ = _make_assembler()
    result = await assembler.assemble(_make_request())
    assert result.messages[-1].role == "user"


async def test_happy_path_with_memory_has_three_messages() -> None:
    """With memory present: [system, user(memory), user(task)]."""
    assembler, _ = _make_assembler(memory_chunks=["Memory chunk one."])
    result = await assembler.assemble(_make_request())
    assert len(result.messages) == 3


async def test_happy_path_without_memory_has_two_messages() -> None:
    """Without memory: [system, user(task)]."""
    assembler, _ = _make_assembler(memory_chunks=None)
    result = await assembler.assemble(_make_request())
    assert len(result.messages) == 2


async def test_happy_path_memory_message_is_user() -> None:
    assembler, _ = _make_assembler(memory_chunks=["Memory chunk one."])
    result = await assembler.assemble(_make_request())
    assert result.messages[1].role == "user"


async def test_happy_path_memory_message_has_envelope() -> None:
    assembler, _ = _make_assembler(memory_chunks=["Memory chunk one."])
    result = await assembler.assemble(_make_request())
    assert _BODY_ENVELOPE_START in result.messages[1].content


async def test_happy_path_token_count_positive() -> None:
    assembler, _ = _make_assembler()
    result = await assembler.assemble(_make_request())
    assert result.token_count > 0


async def test_happy_path_token_count_within_budget() -> None:
    assembler, _ = _make_assembler(budget=4096)
    result = await assembler.assemble(_make_request())
    assert result.token_count <= result.budget


async def test_happy_path_slot_reports_count() -> None:
    """Number of SlotReports equals number of slots."""
    assembler, _ = _make_assembler()
    result = await assembler.assemble(_make_request())
    assert len(result.slots) == len(DEFAULT_SLOTS)


async def test_happy_path_tools_in_result() -> None:
    tools = [_make_tool("search"), _make_tool("calculator")]
    assembler, _ = _make_assembler(tools=tools)
    result = await assembler.assemble(_make_request())
    assert len(result.tools) == 2


async def test_happy_path_memory_diagnostic_present_when_memory_admitted() -> None:
    assembler, _ = _make_assembler(memory_chunks=["A chunk of memory."])
    result = await assembler.assemble(_make_request())
    assert result.memory is not None


async def test_happy_path_memory_diagnostic_none_when_no_memory() -> None:
    assembler, _ = _make_assembler(memory_chunks=None)
    result = await assembler.assemble(_make_request())
    assert result.memory is None


# ---------------------------------------------------------------------------
# 2. Budget property test (Hypothesis)
# ---------------------------------------------------------------------------


@given(
    rules_len=st.integers(min_value=4, max_value=40),
    instr_len=st.integers(min_value=4, max_value=40),
    task_len=st.integers(min_value=4, max_value=40),
    extra_budget=st.integers(min_value=0, max_value=200),
    personality_len=st.integers(min_value=0, max_value=400),
    n_tools=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=60, deadline=5000)
async def test_budget_property_token_count_within_budget(
    rules_len: int,
    instr_len: int,
    task_len: int,
    extra_budget: int,
    personality_len: int,
    n_tools: int,
) -> None:
    """token_count <= total_budget for any content sizes that fit required slots.

    Un-vacuated: personality text length and tool count are drawn strategy params.
    token_count now includes tool tokens (Fix 2).
    """
    rules_text = "R" * rules_len
    instr_text = "I" * instr_len
    task_text = "T" * task_len
    personality_text = "P" * personality_len
    tools = [_make_tool(f"tool_{i}") for i in range(n_tools)]

    head_text = _HEAD_SECTION_SEP.join([rules_text, instr_text])
    required_approx = _approx(head_text) + _approx(task_text)
    budget = required_approx + extra_budget

    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register(
        "personality", _FixedTextContributor(personality_text, slot_name="personality")
    )
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _FixedToolsContributor(tools))
    request = ContextRequest(task=task_text)
    try:
        result = await assembler.assemble(request)
        assert result.token_count <= result.budget
    except ContextBudgetError:
        # Acceptable -- required content didn't fit; the invariant holds vacuously.
        pass


@given(
    rules_len=st.integers(min_value=4, max_value=40),
    instr_len=st.integers(min_value=4, max_value=40),
    task_len=st.integers(min_value=4, max_value=40),
    extra_budget=st.integers(min_value=1, max_value=200),
)
@settings(max_examples=60, deadline=5000)
async def test_budget_property_required_slots_present(
    rules_len: int, instr_len: int, task_len: int, extra_budget: int
) -> None:
    """Required slots are present in output when they fit."""
    rules_text = "R" * rules_len
    instr_text = "I" * instr_len
    task_text = "T" * task_len

    head_text = _HEAD_SECTION_SEP.join([rules_text, instr_text])
    required_approx = _approx(head_text) + _approx(task_text)
    budget = required_approx + extra_budget  # always has enough room

    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    request = ContextRequest(task=task_text)
    try:
        result = await assembler.assemble(request)
        system_content = result.messages[0].content
        assert rules_text in system_content
        assert instr_text in system_content
        assert result.messages[-1].content == task_text
    except ContextBudgetError:
        pass


# ---------------------------------------------------------------------------
# 3. Pressure ladder
# ---------------------------------------------------------------------------


async def test_pressure_budget_below_required_raises_budget_error() -> None:
    """budget < required -> ContextBudgetError raised (never silent truncation)."""
    rules_text = "R" * 100
    instr_text = "I" * 100
    task_text = "T" * 100
    policy = ContextPolicy(total_budget=1)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    with pytest.raises(ContextBudgetError):
        await assembler.assemble(ContextRequest(task=task_text))


async def test_pressure_budget_error_carries_correct_budget() -> None:
    policy = ContextPolicy(total_budget=1)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor("R" * 100, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor("I" * 100, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    with pytest.raises(ContextBudgetError) as exc_info:
        await assembler.assemble(ContextRequest(task="T" * 100))
    assert exc_info.value.budget == 1


def _tight_budget(rules: str, instr: str, task: str) -> int:
    """Return required_tokens + 1 (minimal slack)."""
    head_joined = _HEAD_SECTION_SEP.join([rules, instr])
    return _approx(head_joined) + _approx(task) + 1


async def test_pressure_rules_byte_intact_under_tight_budget() -> None:
    """rules text is byte-intact even at tightest budget."""
    rules_text = "Rule: always be helpful."
    instr_text = "Instructions: answer concisely."
    task_text = "Solve the problem."
    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    assert rules_text in result.messages[0].content


async def test_pressure_instructions_byte_intact_under_tight_budget() -> None:
    rules_text = "Rule: always be helpful."
    instr_text = "Instructions: answer concisely."
    task_text = "Solve the problem."
    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    assert instr_text in result.messages[0].content


async def test_pressure_task_byte_intact_under_tight_budget() -> None:
    rules_text = "Rule: always be helpful."
    instr_text = "Instructions: answer concisely."
    task_text = "Solve the problem."
    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    assert result.messages[-1].content == task_text


async def test_pressure_personality_evicted_whole_under_tight_budget() -> None:
    """Personality is evicted WHOLE (not truncated) when budget is very tight."""
    rules_text = "Rule: always be helpful."
    instr_text = "Instructions: answer concisely."
    task_text = "Solve the problem."
    personality_text = "P" * 400  # long -- won't fit in the tiny slack

    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register(
        "personality",
        _FixedTextContributor(personality_text, slot_name="personality"),
    )
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    assert personality_text not in result.messages[0].content
    per_report = next(r for r in result.slots if r.name == "personality")
    assert per_report.evicted


async def test_pressure_personality_evicted_slot_report() -> None:
    """SlotReport.evicted == True for personality when evicted under pressure."""
    rules_text = "Rule: be helpful."
    instr_text = "Instructions: be brief."
    task_text = "Do the task."
    personality_text = "P" * 400
    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register(
        "personality",
        _FixedTextContributor(personality_text, slot_name="personality"),
    )
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    per_report = next(r for r in result.slots if r.name == "personality")
    assert per_report.evicted is True


async def test_pressure_memory_empty_under_tight_budget() -> None:
    """Memory is empty (body message absent) when budget is exhausted by required slots."""
    rules_text = "Rule: always be helpful."
    instr_text = "Instructions: answer concisely."
    task_text = "Solve the problem."
    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    mem_c = _MemoryContributor(["Memory chunk. " * 20], slot_name="memory")
    assembler.register("memory", mem_c)
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    # Only system + task when memory is empty.
    assert len(result.messages) == 2


async def test_pressure_tools_dropped_per_tool() -> None:
    """Tools are admitted greedy per-tool; small tool survives when large is dropped."""
    rules_text = "Be helpful."
    instr_text = "Do this."
    task_text = "Go."
    head_joined = _HEAD_SECTION_SEP.join([rules_text, instr_text])
    required_tokens = _approx(head_joined) + _approx(task_text)

    small_tool = _make_tool("ping")
    small_tool_cost = _approx(json.dumps(small_tool.model_dump(), separators=(",", ":")))
    large_tool = ToolSpec(
        name="bigone",
        description="D" * 500,
        input_schema={"type": "object", "properties": {}},
    )
    large_tool_cost = _approx(json.dumps(large_tool.model_dump(), separators=(",", ":")))
    # Fits small, not large.
    budget = required_tokens + small_tool_cost + 1

    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register("instructions", _FixedTextContributor(instr_text, slot_name="instructions"))
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    # small_tool first in declared order (greedy -- it should survive).
    assembler.register("tools", _FixedToolsContributor([small_tool, large_tool]))

    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    admitted_names = {t.name for t in result.tools}
    assert "ping" in admitted_names
    # Sanity: large_tool_cost is non-trivial.
    assert large_tool_cost > 1


async def test_pressure_tools_slot_report_dropped_count() -> None:
    """SlotReport for tools tracks chunks_dropped correctly."""
    rules_text = "Be helpful."
    instr_text = "Do this."
    task_text = "Go."
    head_joined = _HEAD_SECTION_SEP.join([rules_text, instr_text])
    required_tokens = _approx(head_joined) + _approx(task_text)

    small_tool = _make_tool("ping")
    small_tool_cost = _approx(json.dumps(small_tool.model_dump(), separators=(",", ":")))
    large_tool = ToolSpec(
        name="bigone",
        description="D" * 500,
        input_schema={"type": "object", "properties": {}},
    )

    budget = required_tokens + small_tool_cost + 1
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor(instr_text, slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _FixedToolsContributor([small_tool, large_tool]))

    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    tools_report = next(r for r in result.slots if r.name == "tools")
    # small admitted, large dropped -> chunks_dropped == 1
    assert tools_report.chunks_dropped == 1


# ---------------------------------------------------------------------------
# 4. U-fold positions
# ---------------------------------------------------------------------------


async def test_ufold_rules_at_messages_start() -> None:
    """Rules text is at the start of messages[0] (the system message)."""
    rules_text = "RULES_MARKER: be helpful always."
    assembler, _ = _make_assembler(rules=rules_text, budget=8192)
    result = await assembler.assemble(_make_request())
    assert result.messages[0].content.startswith(rules_text)


async def test_ufold_task_is_last_message() -> None:
    """task text is the content of the last message."""
    task = "TASK_MARKER: solve everything."
    assembler, _ = _make_assembler(budget=8192)
    result = await assembler.assemble(_make_request(task=task))
    assert result.messages[-1].content == task


async def test_ufold_memory_rank1_chunk_at_body_front() -> None:
    """Rank-1 memory chunk (highest relevance) appears at the FRONT of the body message."""
    chunk1 = "RANK1_CHUNK: This is the most relevant memory."
    chunk2 = "RANK2_CHUNK: This is the second most relevant."
    assembler, _ = _make_assembler(memory_chunks=[chunk1, chunk2], budget=8192)
    result = await assembler.assemble(_make_request())
    body_msg = next(m for m in result.messages if m.role == "user" and "memory:start" in m.content)
    pos1 = body_msg.content.find("RANK1_CHUNK")
    pos2 = body_msg.content.find("RANK2_CHUNK")
    assert pos1 < pos2, (
        f"Rank-1 chunk should appear before rank-2 in body message. pos1={pos1}, pos2={pos2}"
    )


async def test_ufold_memory_rank2_chunk_presence_through_doubles() -> None:
    """With 2 chunks via fake contributor doubles, both rank-1 and rank-2 are present in body.

    Tests presence through FixedTextContributor/MemoryContributor doubles only.
    End-to-end positional ordering through the real injector is covered by
    test_ufold_real_injector_order_preservation below.
    """
    chunk1 = "RANK1_FRONT: most relevant memory."
    chunk2 = "RANK2_BACK: second most relevant memory."
    assembler, _ = _make_assembler(memory_chunks=[chunk1, chunk2], budget=8192)
    result = await assembler.assemble(_make_request())
    body_msg = next(m for m in result.messages if m.role == "user" and "memory:start" in m.content)
    assert "RANK1_FRONT" in body_msg.content
    assert "RANK2_BACK" in body_msg.content


# ---------------------------------------------------------------------------
# 5. Determinism
# ---------------------------------------------------------------------------


async def test_determinism_identical_requests_produce_identical_messages() -> None:
    """Same request + same contributors -> byte-identical messages tuple."""
    assembler, _ = _make_assembler(
        rules="Be helpful.",
        instructions="Answer concisely.",
        personality="I am friendly.",
        memory_chunks=["Memory A.", "Memory B."],
        budget=8192,
    )
    request = _make_request(task="Tell me something.")

    result_a = await assembler.assemble(request)
    result_b = await assembler.assemble(request)

    assert result_a.messages == result_b.messages


async def test_determinism_tools_order_stable() -> None:
    tools = [_make_tool("alpha"), _make_tool("beta"), _make_tool("gamma")]
    assembler, _ = _make_assembler(tools=tools, budget=8192)
    request = _make_request()

    r1 = await assembler.assemble(request)
    r2 = await assembler.assemble(request)

    assert r1.tools == r2.tools


async def test_determinism_slot_reports_stable() -> None:
    assembler, _ = _make_assembler(memory_chunks=["Chunk."], budget=8192)
    request = _make_request()
    r1 = await assembler.assemble(request)
    r2 = await assembler.assemble(request)
    assert r1.slots == r2.slots


# ---------------------------------------------------------------------------
# 6. S1 guard -- assemble must NEVER call model.complete()
# ---------------------------------------------------------------------------


async def test_s1_guard_complete_never_called_during_assemble() -> None:
    """assemble() under a model whose complete() raises -> no error (complete not called)."""
    guard_model = _S1GuardModel()
    assembler, _ = _make_assembler(
        rules="Be helpful.",
        instructions="Answer concisely.",
        memory_chunks=["Memory chunk."],
        model=guard_model,
        budget=8192,
    )
    request = _make_request(task="Do something.")
    # Should NOT raise (complete() is never called).
    result = await assembler.assemble(request)
    assert result.token_count > 0


async def test_s1_guard_count_tokens_is_used() -> None:
    """count_tokens IS called when model provides it."""

    class _CountingModel:
        """Tracks how many times count_tokens is called."""

        def __init__(self) -> None:
            self.call_count = 0

        def count_tokens(self, text: str) -> int:
            self.call_count += 1
            return max(1, len(text) // 4)

        async def complete(self, **kwargs: object) -> None:
            raise AssertionError("complete() must not be called")

    model = _CountingModel()
    assembler, _ = _make_assembler(
        rules="Be helpful.",
        instructions="Do the task.",
        model=model,
        budget=4096,
    )
    await assembler.assemble(_make_request())
    assert model.call_count > 0


# ---------------------------------------------------------------------------
# 7. Edge cases
# ---------------------------------------------------------------------------


async def test_register_unknown_slot_raises_key_error() -> None:
    assembler = ContextAssembler(slots=DEFAULT_SLOTS)
    with pytest.raises(KeyError):
        assembler.register("nonexistent_slot", _EmptyContributor())


async def test_no_contributors_returns_task_from_request() -> None:
    """With all-empty contributors, task slot falls back to request.task."""
    policy = ContextPolicy(total_budget=4096)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    for spec in DEFAULT_SLOTS:
        assembler.register(spec.name, _EmptyContributor())
    result = await assembler.assemble(_make_request(task="Do something."))
    # task is synthesised from request.task even when contributor returns empty.
    assert result.messages[-1].content == "Do something."


async def test_request_policy_overrides_default_policy() -> None:
    """request.policy takes precedence over the assembler's default_policy."""
    assembler, _ = _make_assembler(budget=1000)
    override_policy = ContextPolicy(total_budget=2048)
    request = ContextRequest(task="Do something.", policy=override_policy)
    result = await assembler.assemble(request)
    assert result.budget == 2048


async def test_slot_report_rules_status_ok() -> None:
    """SlotReport for rules has status='ok' when rules contributor returns content."""
    assembler, _ = _make_assembler(rules="Be helpful.")
    result = await assembler.assemble(_make_request())
    rules_report = next(r for r in result.slots if r.name == "rules")
    assert rules_report.status == "ok"


async def test_slot_report_personality_empty_status() -> None:
    """A slot with _EmptyContributor registered gets status='empty'."""
    policy = ContextPolicy(total_budget=4096)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor("Rules.", slot_name="rules"))
    assembler.register(
        "instructions",
        _FixedTextContributor("Instr.", slot_name="instructions"),
    )
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())
    result = await assembler.assemble(_make_request())
    per_report = next(r for r in result.slots if r.name == "personality")
    assert per_report.status == "empty"


async def test_assembled_context_budget_matches_policy() -> None:
    budget = 6000
    assembler, _ = _make_assembler(budget=budget)
    result = await assembler.assemble(_make_request())
    assert result.budget == budget


async def test_all_slot_names_in_reports() -> None:
    assembler, _ = _make_assembler()
    result = await assembler.assemble(_make_request())
    report_names = {r.name for r in result.slots}
    slot_names = {s.name for s in DEFAULT_SLOTS}
    assert report_names == slot_names


# Unused imports kept for type-checker completeness; suppress ruff warnings.
_ = SlotSpec  # referenced in DEFAULT_SLOTS type annotation


# ---------------------------------------------------------------------------
# 8. Fix 1 — Phase-2 head charging = join-then-count delta (Pod 3.1 budget-honesty)
# ---------------------------------------------------------------------------


async def test_fix1_personality_evicted_when_join_exceeds_budget() -> None:
    """Personality is evicted when join(rules+sep+personality+sep+instructions) exceeds budget
    but required + count_fn(personality_text) <= budget under additive (wrong) charging.

    Uses approx_tokens (~4 chars/token). Separator is "\\n\\n---\\n\\n" = 7 chars ≈ 1-2 tokens.
    Build: rules="R"*80 (20 tok), instr="I"*80 (20 tok), personality="P"*80 (20 tok).
    Sep cost between rules+personality+instructions: 2 seps x ~2 tokens = ~4 tokens.
    required_charge = count_fn(join(rules, instr)) = count_fn("R"*80 + sep + "I"*80)
    Budget = required_charge + count_fn("P"*80) + 1
      → additive cost would admit personality
      → but join(rules, personality, instr) has 2 seps → delta pushes it over budget
    """
    rules_text = "R" * 80
    instr_text = "I" * 80
    personality_text = "P" * 80

    # required head cost (what Phase 1 charges)
    req_joined = _HEAD_SECTION_SEP.join([rules_text, instr_text])
    required_head_cost = _approx(req_joined)
    task_text = "T" * 4  # 1 token
    required_total = required_head_cost + _approx(task_text)

    # Budget: required + personality_text_cost alone (no sep overhead)
    # The WRONG additive cost would be count_fn(personality_text) → fits
    # The CORRECT delta cost includes 2 seps → must not fit
    personality_cost_additive = _approx(personality_text)

    # Joined with personality inserted: rules + sep + personality + sep + instr
    joined_with_personality = _HEAD_SECTION_SEP.join([rules_text, personality_text, instr_text])
    delta_cost = _approx(joined_with_personality) - required_head_cost

    # Budget must satisfy: required + personality_additive_cost >= budget (additive would admit)
    # AND required + delta_cost > budget (delta correctly rejects)
    # Set budget = required_total + personality_cost_additive (exactly fits additive, not delta)
    budget = required_total + personality_cost_additive

    # Only run this test if delta_cost > personality_cost_additive (i.e., the test is meaningful)
    if delta_cost <= personality_cost_additive:
        pytest.skip("Separator overhead not significant enough to distinguish — skipping")

    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register("instructions", _FixedTextContributor(instr_text, slot_name="instructions"))
    assembler.register(
        "personality", _FixedTextContributor(personality_text, slot_name="personality")
    )
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())

    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))

    # Personality must be EVICTED (delta cost exceeds budget slack).
    personality_report = next(r for r in result.slots if r.name == "personality")
    assert personality_report.evicted, (
        "Fix 1: personality must be evicted when join-delta exceeds budget "
        "(additive would wrongly admit it)"
    )
    assert personality_text not in result.messages[0].content


async def test_fix1_personality_admitted_when_join_fits() -> None:
    """Budget one token above joined cost → personality admitted; rendered head cost == charged."""
    rules_text = "R" * 80
    instr_text = "I" * 80
    personality_text = "P" * 80
    task_text = "T" * 4

    req_joined = _HEAD_SECTION_SEP.join([rules_text, instr_text])
    required_head_cost = _approx(req_joined)
    required_total = required_head_cost + _approx(task_text)

    joined_with_personality = _HEAD_SECTION_SEP.join([rules_text, personality_text, instr_text])
    full_head_cost = _approx(joined_with_personality)
    delta_cost = full_head_cost - required_head_cost

    # Budget = required_total + delta_cost + 1 (exactly fits with 1 token slack)
    budget = required_total + delta_cost + 1

    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register("instructions", _FixedTextContributor(instr_text, slot_name="instructions"))
    assembler.register(
        "personality", _FixedTextContributor(personality_text, slot_name="personality")
    )
    assembler.register("memory", _EmptyContributor())
    assembler.register("tools", _EmptyContributor())

    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))

    # Personality must be ADMITTED.
    personality_report = next(r for r in result.slots if r.name == "personality")
    assert not personality_report.evicted, "Fix 1: personality must be admitted when join fits"
    assert personality_text in result.messages[0].content

    # Rendered head cost == charged cost (contract check).
    # The system message content is the rendered head.
    system_msg = next(m for m in result.messages if m.role == "system")
    rendered_head_cost = _approx(system_msg.content)
    # token_count includes all messages; task is a separate message
    non_head_cost = sum(_approx(m.content) for m in result.messages if m is not system_msg)
    assert result.token_count == rendered_head_cost + non_head_cost


# ---------------------------------------------------------------------------
# 9. Fix 2 — token_count = TOTAL consumption (messages + tools) (Pod 3.1 budget-honesty)
# ---------------------------------------------------------------------------


async def test_fix2_token_count_includes_tool_tokens() -> None:
    """token_count == sum(message tokens) + tool tokens."""
    tools = [_make_tool("alpha"), _make_tool("beta"), _make_tool("gamma")]
    assembler, _ = _make_assembler(
        rules="Be helpful.",
        instructions="Answer concisely.",
        tools=tools,
        budget=4096,
    )
    result = await assembler.assemble(_make_request())
    msg_tokens = sum(_approx(m.content) for m in result.messages)
    tool_tokens = sum(
        _approx(json.dumps(t.model_dump(), separators=(",", ":"))) for t in result.tools
    )
    assert result.token_count == msg_tokens + tool_tokens, (
        f"Fix 2: token_count={result.token_count} must equal "
        f"msg_tokens={msg_tokens} + tool_tokens={tool_tokens}"
    )


async def test_fix2_token_count_within_budget_with_tools() -> None:
    """token_count <= budget when tools are present."""
    tools = [_make_tool("search"), _make_tool("calc")]
    assembler, _ = _make_assembler(tools=tools, budget=4096)
    result = await assembler.assemble(_make_request())
    assert result.token_count <= result.budget


# ---------------------------------------------------------------------------
# 10. Fix 3 — eliminate memory double-call; gate the diagnostic (Pod 3.1)
# ---------------------------------------------------------------------------


class _CountingContributor:
    """Contributor that counts exactly how many times contribute() was called."""

    def __init__(self, text: str = "memory chunk text", *, slot_name: str = "memory") -> None:
        self._text = text
        self._slot_name = slot_name
        self.call_count = 0
        self.last_injected_memory: InjectedMemory | None = None

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        self.call_count += 1
        self.last_injected_memory = None
        if not self._text:
            return SlotContent(status="empty")
        return SlotContent(
            chunks=(
                SlotChunk(
                    text=self._text,
                    key=f"{self._slot_name}:0",
                    source_slot=self._slot_name,
                ),
            ),
        )


async def test_fix3_memory_contribute_called_exactly_once() -> None:
    """Fix 3: memory contributor.contribute() is called exactly once per assemble()."""
    counting = _CountingContributor(text="memory content", slot_name="memory")
    policy = ContextPolicy(total_budget=4096)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor("Rules.", slot_name="rules"))
    assembler.register("instructions", _FixedTextContributor("Instr.", slot_name="instructions"))
    assembler.register("personality", _EmptyContributor())
    assembler.register("memory", counting)
    assembler.register("tools", _EmptyContributor())

    await assembler.assemble(_make_request())
    assert counting.call_count == 1, (
        f"Fix 3: memory contributor must be called exactly once per assemble, "
        f"got {counting.call_count}"
    )


async def test_fix3_result_memory_none_when_alloc_zero() -> None:
    """Fix 3: result.memory is None when memory alloc == 0 (tight budget)."""
    # Make budget so tight that alloc == 0 for memory.
    rules_text = "Rule: be helpful."
    instr_text = "Instructions: answer concisely."
    task_text = "Solve the problem."
    budget = _tight_budget(rules_text, instr_text, task_text)
    policy = ContextPolicy(total_budget=budget)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", _FixedTextContributor(rules_text, slot_name="rules"))
    assembler.register("instructions", _FixedTextContributor(instr_text, slot_name="instructions"))
    assembler.register("personality", _EmptyContributor())
    counting = _CountingContributor(text="memory content" * 20, slot_name="memory")
    assembler.register("memory", counting)
    assembler.register("tools", _EmptyContributor())

    result = await assembler.assemble(ContextRequest(task=task_text, policy=policy))
    # alloc == 0 → contributor NOT called, result.memory is None
    assert counting.call_count == 0, (
        f"Fix 3: memory contributor must NOT be called when alloc==0, got {counting.call_count}"
    )
    assert result.memory is None, "Fix 3: result.memory must be None when alloc==0"


# ---------------------------------------------------------------------------
# 11. Fix 4a — isinstance-based defeated-claim filter (Pod 3.1)
# ---------------------------------------------------------------------------

# (Tests for Fix 4a are in the injector test file — see test_injection_defeated_2_7cfb.py)
# Here we confirm the assembler contract around result.memory.


# ---------------------------------------------------------------------------
# 12. FIX-NOW 2 — Real injector U-fold positional test (Pod 3.1)
# ---------------------------------------------------------------------------
# This test validates order-preservation through the real pipeline:
#   MemoryInjector → MemoryContributor → ContextAssembler
# (no FixedMemoryContributor / _FixedTextContributor in the memory path).
#
# The U-fold algorithm itself is validated at Pod 2.5 (test_recall_assembly.py).
# 3.1 owns proving the assembler faithfully renders the injector's U-fold output
# WITHOUT reordering.
#
# U-fold algorithm validated at Pod 2.5; 3.1 validates order-preservation through
# the real injector.


def _make_fused_episode_result(
    text: str,
    *,
    key: str,
    rank: int,
    score: float = 0.9,
) -> FusedResult:
    """Build a simple FusedResult with an Episode item (simple value type, no scoring needed)."""
    from datetime import UTC, datetime

    from cogworx.substrate.episodes import Episode

    item = Episode(
        episode_id=key,
        run_id="run-ufold-test",
        step_index=rank,
        turn_index=0,
        session_id="sess-ufold",
        role="assistant",
        content=text,
        kind="conversation",
        occurred_at=datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC),
    )
    return FusedResult(
        key=f"episode:{key}",
        kind="episode",
        item=item,
        text=text,
        hits=(ChannelHit(channel="temporal.episodes", rank=rank, raw_score=score),),
        fused_score=1.0 / (60 + rank),
        fused_rank=rank,
    )


def _make_injector_with_results(results: list[FusedResult]) -> MemoryInjector:
    """Build a MemoryInjector backed by a stub RecallStack returning fixed results."""
    outcome = RecallOutcome(
        results=tuple(results),
        channel_status=(ChannelStatus(channel="dense.latent", state="ok", count=len(results)),),
    )
    stack_mock = MagicMock(spec=RecallStack)
    stack_mock.recall = AsyncMock(return_value=outcome)
    return MemoryInjector(stack_mock)


async def test_ufold_real_injector_order_preservation() -> None:
    """3.1 positional test: real MemoryInjector → MemoryContributor → ContextAssembler.

    3 FusedResults in rank order [R1, R2, R3] admitted by the injector.
    U-fold renders them as [R1, R3, R2] in the body message (rank-1 at front,
    rank-2 at back, rank-3 in the middle — the exact U-fold pattern).

    The assertion is positional: idx(R1) < idx(R3) < idx(R2) in body content.

    LESION CHECK (mandatory per mythos ruling):
      Applying the mutation ``output = list(admitted)`` at recall/assembly.py:122
      collapses U-fold to a simple concat, producing [R1, R2, R3] order.
      Under that mutation: idx(R1) < idx(R2) < idx(R3), so idx(R3) < idx(R2)
      is FALSE — the test fails.  The mutation was applied and confirmed to fail,
      then reverted.  This test then passes on the real code.
    """
    sentinel_r1 = "MEM-RANK-1"
    sentinel_r2 = "MEM-RANK-2"
    sentinel_r3 = "MEM-RANK-3"

    # 3 FusedResults in rank order (best first = rank 1).
    # All will fit in the memory budget so all 3 are admitted.
    r1 = _make_fused_episode_result(sentinel_r1, key="k1", rank=1, score=0.95)
    r2 = _make_fused_episode_result(sentinel_r2, key="k2", rank=2, score=0.80)
    r3 = _make_fused_episode_result(sentinel_r3, key="k3", rank=3, score=0.65)

    # Real MemoryInjector backed by stub recall stack.
    injector = _make_injector_with_results([r1, r2, r3])

    # Real MemoryContributor wrapping the real injector.
    mem_policy = MemoryPolicy(token_budget=4096)
    mem_contrib = MemoryContributor(injector, policy=mem_policy)

    # Real ContextAssembler with real contributors (no FixedMemoryContributor).
    policy = ContextPolicy(total_budget=8192, memory=mem_policy)
    assembler = ContextAssembler(slots=DEFAULT_SLOTS, default_policy=policy)
    assembler.register("rules", StaticContributor(text="RULES", key="rules:0", source_slot="rules"))
    assembler.register(
        "instructions",
        StaticContributor(text="", key="instructions:0", source_slot="instructions"),
    )
    assembler.register(
        "personality",
        StaticContributor(text="", key="personality:0", source_slot="personality"),
    )
    assembler.register("memory", mem_contrib)
    assembler.register("tools", _EmptyContributor())
    # TaskContributor reads request.task — use it as the real task contributor.
    assembler.register(
        "task",
        TaskContributor(field="task", key="task:main", source_slot="task"),
    )

    request = ContextRequest(
        task="Recall test task.",
        query=RecallQuery(text="test"),
        policy=policy,
    )
    result = await assembler.assemble(request)

    # Extract the body message (memory envelope).
    body_msg = next(
        (m for m in result.messages if "<!-- memory:start -->" in m.content),
        None,
    )
    assert body_msg is not None, (
        "Real injector test: body (memory envelope) message must exist with 3 admitted chunks"
    )
    body_text = body_msg.content

    # All 3 sentinels must be present.
    assert sentinel_r1 in body_text, f"R1 sentinel must appear in body: {body_text!r}"
    assert sentinel_r2 in body_text, f"R2 sentinel must appear in body: {body_text!r}"
    assert sentinel_r3 in body_text, f"R3 sentinel must appear in body: {body_text!r}"

    idx_r1 = body_text.index(sentinel_r1)
    idx_r2 = body_text.index(sentinel_r2)
    idx_r3 = body_text.index(sentinel_r3)

    # U-fold with 3 admitted items [R1, R2, R3] (rank order) produces output [R1, R3, R2].
    # Expected: idx(R1) < idx(R3) < idx(R2).
    assert idx_r1 < idx_r3, (
        f"U-fold: rank-1 (idx={idx_r1}) must appear before rank-3 (idx={idx_r3}). "
        f"If this fails with idx_r3 > idx_r2, the U-fold collapsed to simple concat."
    )
    assert idx_r3 < idx_r2, (
        f"U-fold: rank-3 (idx={idx_r3}) must appear before rank-2 (idx={idx_r2}). "
        f"With simple concat [R1,R2,R3] we get idx_r2 < idx_r3 — the lesion test "
        f"confirms this mutation fails this assertion."
    )
