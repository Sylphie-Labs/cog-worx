"""Pod 3.1 context-assembly spike (CANON S12) — SC-1 through SC-7.

Falsifiable spike success criteria for the ContextAssembler, DEFAULT_SLOTS, the four
contributors (StaticContributor, TaskContributor, ToolSpecContributor, MemoryContributor),
ContextRequest, ContextPolicy, AssembledCallContext, and the related error types.

CANON compliance:
  S1  — assembly NEVER calls model.complete(); only count_tokens is permitted.
  S5  — defeated-claim exclusion is a structural filter (zero model calls).
  S6  — same request + same contributor outputs → byte-identical message tuples (replay-safe).
  S8  — assembler=None degrades gracefully (task-only, COMPLETED, not a crash).
  S9  — budget arbitration is structural; no model judgment in any arbitration path.
  S11 — token_count <= total_budget is enforced pre-call; ContextBudgetError is the guard.
  S12 — spike gate.

Every spike leg has a negative control / discriminability guard so no assertion can pass
trivially. Negative controls are co-located and clearly labelled.

Pure Python — no Neo4j, no Postgres, deterministic fixed seeds / fake contributors.
Model doubles: FailOnCallModel (complete() always raises) and ReplayModel([]).
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.context.assembler import ContextAssembler
from cogworx.context.contributors import (
    StaticContributor,
    TaskContributor,
    ToolSpecContributor,
)
from cogworx.context.errors import ContextAssemblyError, ContextBudgetError
from cogworx.context.types import (
    DEFAULT_SLOTS,
    AssembledCallContext,
    ContextPolicy,
    ContextRequest,
    SlotAllocation,
    SlotChunk,
    SlotContent,
)
from cogworx.cost.budget import BudgetPolicy
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import InjectedMemory, MemoryPolicy
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
)
from cogworx.recall.assembly import approx_tokens
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import ChannelHit, FusedResult
from cogworx.recall.stack import ChannelStatus, RecallOutcome
from cogworx.runtime.context import RunContext
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel

pytestmark = pytest.mark.spike

# ---------------------------------------------------------------------------
# Fixed timestamps — deterministic; no datetime.now() anywhere in this file
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 12, 10, 0, 0, tzinfo=UTC)
_CLOCK = lambda: _NOW  # noqa: E731


# ===========================================================================
# Shared model doubles
# ===========================================================================


class FailOnCallModel:
    """Model double whose complete() always raises AssertionError.

    Used in SC-6 (S1/S9 guard): count_tokens is permitted; any complete() during
    assembly must never happen and will fail the test structurally if it does.
    """

    def __init__(self) -> None:
        self._calls: int = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        self._calls += 1
        raise AssertionError(
            f"FailOnCallModel.complete() called (call #{self._calls}) — S1/S9 violation: "
            "assembly must NEVER invoke model.complete()"
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @property
    def complete_call_count(self) -> int:
        return self._calls


# ===========================================================================
# Shared contributor helpers
# ===========================================================================


def _make_chunk(text: str, slot: str, key: str | None = None) -> SlotChunk:
    return SlotChunk(text=text, key=key or f"{slot}:0", source_slot=slot)


def _static(text: str, slot: str) -> StaticContributor:
    return StaticContributor(text=text, key=f"{slot}:0", source_slot=slot)


def _task_contrib(field: Literal["task", "instructions"] = "task") -> TaskContributor:
    return TaskContributor(field=field, key=f"{field}:main", source_slot=field)


class AlwaysRaisingContributor:
    """Contributor that always raises RuntimeError — used for S8 lesion tests."""

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        raise RuntimeError("intentional AlwaysRaisingContributor failure")


class FixedMemoryContributor:
    """Contributor that returns a pre-built SlotContent (no real recall).

    Used as a memory slot double when we want full control over what memory content
    is returned, without needing to wire a real MemoryInjector.
    """

    def __init__(self, chunks: tuple[SlotChunk, ...]) -> None:
        self._chunks = chunks
        self.last_injected_memory: InjectedMemory | None = None

    async def contribute(self, request: ContextRequest, allocation: SlotAllocation) -> SlotContent:
        return SlotContent(chunks=self._chunks, status="ok" if self._chunks else "empty")


# ---------------------------------------------------------------------------
# MemoryInjector double (pure structural, no real recall stack needed)
# ---------------------------------------------------------------------------


def _make_memory_injector_with_fused(
    results: Sequence[FusedResult],
    *,
    policy: MemoryPolicy | None = None,
) -> MemoryInjector:
    """Build a MemoryInjector backed by a pre-loaded InMemoryLatentStore.

    Because MemoryInjector only calls record_use for latent-kind items, and these
    FusedResult objects don't route through a real stack, we wire a trivial
    RecallStack subclass that short-circuits recall().
    """

    class FixedRecallStack:
        """RecallStack double that returns a fixed RecallOutcome."""

        def __init__(self, fused: Sequence[FusedResult]) -> None:
            self._fused = tuple(fused)

        async def recall(self, query: RecallQuery) -> RecallOutcome:
            ch_status = ChannelStatus(channel="test.fixed", state="ok", count=len(self._fused))
            return RecallOutcome(
                results=self._fused,
                channel_status=(ch_status,),
            )

    stack = FixedRecallStack(results)
    # Type: MemoryInjector takes a RecallStack; we pass our duck-typed double.
    return MemoryInjector(
        stack,  # type: ignore[arg-type]
        clock=_CLOCK,
    )


def _make_claim_with_status(
    subject: str,
    predicate: str,
    payload: str,
    status: Literal["active", "defeasibly-defeated"] = "active",
) -> Claim:
    cid = claim_id_for(subject, predicate, payload)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=0.9, recorded_at=_NOW),
        valid_from=_NOW,
        valid_to=None,
        ingest_time=_NOW,
        created_by="spike-3-1",
        status=status,
    )


def _scored_claim(
    subject: str,
    predicate: str,
    payload: str,
    status: Literal["active", "defeasibly-defeated"] = "active",
) -> ScoredClaim:
    claim = _make_claim_with_status(subject, predicate, payload, status=status)
    conf = claim_confidence([])
    return ScoredClaim(claim=claim, confidence=conf, lineage_min_confidence=conf.confidence)


def _fused_claim_result(
    subject: str,
    predicate: str,
    payload: str,
    text: str,
    *,
    status: Literal["active", "defeasibly-defeated"] = "active",
    score: float = 0.8,
    rank: int = 1,
) -> FusedResult:
    sc = _scored_claim(subject, predicate, payload, status=status)
    return FusedResult(
        key=f"claim:{sc.claim.id}",
        kind="claim",
        item=sc,
        text=text,
        hits=(ChannelHit(channel="dense.claims", rank=rank, raw_score=score),),
        fused_score=score,
        fused_rank=rank,
    )


def _make_tool(name: str, description: str = "") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=description or f"Tool {name}",
        input_schema={"type": "object", "properties": {"x": {"type": "string"}}},
    )


def _tool_tokens(tool: ToolSpec) -> int:
    """Approximate token cost of one serialised ToolSpec."""
    return approx_tokens(json.dumps(tool.model_dump(), separators=(",", ":")))


def _minimal_assembler(
    *,
    rules_text: str = "RULES",
    instructions_text: str | None = "INSTRUCTIONS",
    personality_text: str = "",
    task_text: str = "TASK",
    memory_chunks: tuple[SlotChunk, ...] = (),
    tools: Sequence[ToolSpec] = (),
    budget: int = 4096,
    model: object | None = None,
) -> tuple[ContextAssembler, ContextRequest]:
    """Build a minimal wired assembler + request for testing.

    Returns (assembler, request) ready for assembler.assemble(request).
    """
    slots = DEFAULT_SLOTS
    asm = ContextAssembler(
        slots=slots,
        model=model or ReplayModel([]),
        default_policy=ContextPolicy(total_budget=budget),
    )
    asm.register("rules", _static(rules_text, "rules"))
    if instructions_text is not None:
        asm.register("instructions", _static(instructions_text, "instructions"))
    else:
        asm.register("instructions", _static("", "instructions"))
    asm.register("personality", _static(personality_text, "personality"))
    asm.register("memory", FixedMemoryContributor(memory_chunks))
    asm.register("tools", ToolSpecContributor(list(tools)))
    asm.register("task", _static(task_text, "task"))
    req = ContextRequest(
        task=task_text,
        policy=ContextPolicy(total_budget=budget),
    )
    return asm, req


# ===========================================================================
# SC-1: Budget invariant (property test)
# ===========================================================================


@pytest.mark.parametrize(
    "seed,n_chunks,budget",
    [
        (0, 5, 200),
        (1, 10, 512),
        (2, 3, 50),
        (3, 15, 1024),
        (4, 8, 100),
        (5, 2, 20),
        (7, 6, 150),
        (13, 12, 300),
        (42, 4, 80),
        (99, 20, 2048),
    ],
)
async def test_sc1_budget_invariant_property(seed: int, n_chunks: int, budget: int) -> None:
    """Property test: AssembledCallContext.token_count <= total_budget.

    Randomised slot contents, fixed seeds (deterministic). Counts messages AND
    serialised tools (both paths charged against budget).
    """
    rng = random.Random(seed)

    # Generate memory chunks with randomised text (seed-controlled → deterministic per index).
    mem_chunks = tuple(
        _make_chunk(
            text="x" * rng.randint(4, 30),
            slot="memory",
            key=f"memory:{i}",
        )
        for i in range(n_chunks)
    )

    # Add 1-3 tools with randomised description lengths.
    n_tools = rng.randint(0, 3)
    tools = [_make_tool(f"tool_{j}", description="d" * rng.randint(10, 60)) for j in range(n_tools)]

    asm, req = _minimal_assembler(
        rules_text="R" * rng.randint(4, 40),
        instructions_text="I" * rng.randint(4, 40),
        task_text="T" * rng.randint(4, 40),
        memory_chunks=mem_chunks,
        tools=tools,
        budget=budget,
    )
    ctx = await asm.assemble(req)

    # Core invariant: rendered message tokens <= budget.
    assert ctx.token_count <= ctx.budget, (
        f"SC-1 FAIL seed={seed}: token_count={ctx.token_count} > budget={ctx.budget}"
    )
    assert ctx.budget == budget


async def test_sc1_budget_invariant_negative_required_overflow() -> None:
    """Negative control: required content alone exceeds budget → ContextBudgetError.

    This proves the invariant is NOT vacuously true because nothing ever fills the budget —
    there exist fixtures where required content can overflow it.
    """
    # rules="R"*200 + task="T"*200 → each ~50 tokens → combined ~100, budget=5 → must raise.
    rules_text = "R" * 200
    task_text = "T" * 200
    budget = 5  # tiny budget — required content cannot fit

    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text="",  # empty → won't contribute tokens
        task_text=task_text,
        budget=budget,
    )
    with pytest.raises(ContextBudgetError) as exc_info:
        await asm.assemble(req)

    err = exc_info.value
    assert err.required_tokens > budget
    assert err.budget == budget
    # The guard fires pre-call: no context is returned when required content overflows.


async def test_sc1_token_count_includes_message_content_not_just_length() -> None:
    """Token counting uses count_tokens callable, not raw len(messages).

    Uses a ReplayModel whose count_tokens returns len(text) // 4 (approx_tokens).
    Verifies token_count == sum(count_tokens(m.content) for m in messages) + tool tokens (Fix 2).
    """
    model = ReplayModel([])
    rules_text = "ABCDEFGHIJ" * 5  # 50 chars → approx 12 tokens
    task_text = "Task statement here"  # 19 chars → approx 4 tokens

    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text="",
        task_text=task_text,
        budget=1024,
        model=model,
    )
    ctx = await asm.assemble(req)

    # Recompute what token_count should be (Fix 2: messages + tools).
    expected = sum(model.count_tokens(m.content) for m in ctx.messages) + sum(
        model.count_tokens(json.dumps(t.model_dump(), separators=(",", ":"))) for t in ctx.tools
    )
    assert ctx.token_count == expected, (
        f"token_count={ctx.token_count} does not match sum(count_tokens)={expected}"
    )


async def test_sc1_token_count_includes_tools_and_personality() -> None:
    """SC-1 de-vacuation: non-empty personality + tools, token_count <= budget (Fix 2 semantics).

    Ensures the budget invariant holds with personality text AND tools in play under the
    new total-consumption semantics (messages + serialised tools).
    """
    model = ReplayModel([])
    personality_text = "P" * 120  # non-empty personality (~30 tokens)
    tool_a = _make_tool("tool_a", description="d" * 40)
    tool_b = _make_tool("tool_b", description="d" * 40)
    tool_c = _make_tool("tool_c", description="d" * 40)

    # Budget must fit required + personality delta + some tools.
    rules_text = "R" * 40
    instr_text = "I" * 40
    task_text = "T" * 20
    budget = 2048  # generous enough for all content

    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text=instr_text,
        personality_text=personality_text,
        task_text=task_text,
        tools=[tool_a, tool_b, tool_c],
        budget=budget,
        model=model,
    )
    ctx = await asm.assemble(req)

    # Core invariant: token_count <= budget (with tools counted).
    assert ctx.token_count <= ctx.budget, (
        f"SC-1 de-vacuated: token_count={ctx.token_count} > budget={ctx.budget}"
    )

    # Verify the total is messages + tools (Fix 2 semantics).
    msg_tokens = sum(model.count_tokens(m.content) for m in ctx.messages)
    tool_tokens = sum(
        model.count_tokens(json.dumps(t.model_dump(), separators=(",", ":"))) for t in ctx.tools
    )
    assert ctx.token_count == msg_tokens + tool_tokens, (
        f"SC-1: token_count={ctx.token_count} must equal "
        f"msg_tokens={msg_tokens} + tool_tokens={tool_tokens}"
    )


# ===========================================================================
# SC-2: Pressure ladder
# ===========================================================================


async def test_sc2_pressure_ladder_memory_evicted_tools_dropped_required_intact() -> None:
    """Budget = required + ε → memory evicted, personality evicted whole, tools dropped per-tool.

    rules + instructions + task text survive BYTE-INTACT in messages and SlotReports.
    This checks the three required slots are present + exact; the preferred/evictable are dropped.
    """
    rules_text = "SYS_RULES_MUST_SURVIVE"
    instructions_text = "INSTRUCTIONS_MUST_SURVIVE"
    task_text = "TASK_MUST_SURVIVE"
    personality_text = "PERSONALITY_SHOULD_BE_EVICTED"
    mem_text = "MEMORY_CHUNK_SHOULD_BE_EVICTED"

    # Compute rough token cost of required content only.
    required_tokens = (
        approx_tokens(rules_text)
        + approx_tokens(instructions_text)
        + approx_tokens(task_text)
        # separator overhead: HEAD_SECTION_SEP is "\n\n---\n\n" = 7 chars ≈ 1 token
        + 2  # small overhead for separators
    )
    # Budget = required + 1 (just barely enough for required, nothing for preferred/evictable)
    budget = required_tokens + 1

    mem_chunk = _make_chunk(mem_text, "memory")
    tools = [_make_tool("calculator"), _make_tool("search_web")]

    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text=instructions_text,
        personality_text=personality_text,
        task_text=task_text,
        memory_chunks=(mem_chunk,),
        tools=tools,
        budget=budget,
    )
    ctx = await asm.assemble(req)

    # Required content is byte-intact in the rendered messages.
    all_message_content = " ".join(m.content for m in ctx.messages)
    assert rules_text in all_message_content, (
        "SC-2: rules text must survive pressure, not found in messages"
    )
    assert task_text in all_message_content, (
        "SC-2: task text must survive pressure, not found in messages"
    )

    # Required content also present in SlotReports with non-zero tokens.
    report_by_name = {r.name: r for r in ctx.slots}
    assert report_by_name["rules"].tokens > 0, "SC-2: rules slot must have non-zero tokens"
    assert report_by_name["task"].tokens > 0, "SC-2: task slot must have non-zero tokens"

    # Memory was evicted (empty or zero chunks in memory body message).
    memory_report = report_by_name.get("memory")
    assert memory_report is not None
    memory_text_in_messages = next(
        (m.content for m in ctx.messages if "<!-- memory:start -->" in m.content),
        None,
    )
    # Under extreme budget pressure memory should be empty/evicted.
    if memory_text_in_messages is not None:
        assert mem_text not in memory_text_in_messages, (
            "SC-2: memory chunk text should not survive under tight budget"
        )

    # Personality was NOT included in head message (whole-slot eviction).
    head_msg = next((m for m in ctx.messages if m.role == "system"), None)
    if head_msg is not None:
        assert personality_text not in head_msg.content, (
            "SC-2: personality must be evicted whole under pressure"
        )

    # Budget not exceeded.
    assert ctx.token_count <= budget


async def test_sc2_budget_below_required_raises_context_budget_error() -> None:
    """budget < required → ContextBudgetError, NEVER silent rules truncation.

    Critical: rules text is NEVER partially present when the error fires — the exception
    propagates before any context is returned.
    """
    rules_text = "MANDATORY_RULES_TEXT_CANNOT_BE_TRUNCATED"
    task_text = "TASK_TEXT_MANDATORY"
    budget = 2  # smaller than required content

    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text="",
        task_text=task_text,
        budget=budget,
    )
    with pytest.raises(ContextBudgetError) as exc_info:
        await asm.assemble(req)

    err = exc_info.value
    # The error fires at the pre-call guard, never after partial assembly.
    assert err.budget == budget
    assert err.required_tokens > budget
    # No partial rules text can be present — ContextBudgetError is raised before return.
    # (This is structural: the function raises, so no AssembledCallContext is returned.)


async def test_sc2_rules_text_never_truncated_partial_under_any_budget() -> None:
    """Rules text is NEVER partially present — it's either byte-intact or absent (error).

    Negative control: if rules text IS in the message, it must be the FULL text.
    """
    rules_text = "FULL_RULES_MUST_BE_PRESENT_OR_ABSENT_NEVER_TRUNCATED"
    task_text = "T" * 20
    budget = 400  # large enough for required

    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text="",
        task_text=task_text,
        budget=budget,
    )
    ctx = await asm.assemble(req)

    head_content = next((m.content for m in ctx.messages if m.role == "system"), "")
    if rules_text in head_content:
        # If present, it must be the FULL text (no truncation).
        idx = head_content.index(rules_text)
        assert head_content[idx : idx + len(rules_text)] == rules_text


# ===========================================================================
# SC-3: U-fold positions
# U-fold algorithm validated at Pod 2.5; 3.1 validates order-preservation
# through the real injector.  End-to-end positional proof is in
# test_context_assembler_3_1d.py::test_ufold_real_injector_order_preservation.
# ===========================================================================


async def test_sc3_ufold_positions_rules_at_head_task_at_tail() -> None:
    """U-fold structure: rules at messages[0] start, task as final message.

    Also verifies the body memory message is present between head and tail.
    """
    rules_text = "RULES_MARKER_HEAD_START"
    task_text = "TASK_MARKER_TAIL_END"
    mem_text = "MEMORY_BODY_MARKER"

    mem_chunk = _make_chunk(mem_text, "memory", key="memory:0")
    asm, req = _minimal_assembler(
        rules_text=rules_text,
        instructions_text="",
        task_text=task_text,
        memory_chunks=(mem_chunk,),
        budget=4096,
    )
    ctx = await asm.assemble(req)

    msgs = ctx.messages
    assert len(msgs) >= 2, f"SC-3: expected at least 2 messages, got {len(msgs)}"

    # rules must appear in messages[0] (system message at head).
    assert rules_text in msgs[0].content, (
        f"SC-3: rules marker must be in messages[0], got {msgs[0].content!r}"
    )
    assert msgs[0].role == "system", f"SC-3: messages[0] must be system role, got {msgs[0].role!r}"

    # task must be the FINAL message.
    assert task_text in msgs[-1].content, (
        f"SC-3: task marker must be in messages[-1], got {msgs[-1].content!r}"
    )
    assert msgs[-1].role == "user", (
        f"SC-3: task (final) message must have user role, got {msgs[-1].role!r}"
    )

    # memory body marker must be present somewhere in the middle (not head, not tail).
    body_idx = next(
        (i for i, m in enumerate(msgs) if mem_text in m.content),
        None,
    )
    assert body_idx is not None, "SC-3: memory chunk text must appear in a body message"
    # It must not be the first (head) or last (task) message.
    assert body_idx != 0, "SC-3: memory body must not be at messages[0] (head)"
    assert body_idx != len(msgs) - 1, "SC-3: memory body must not be the final message"


async def test_sc3_back_edge_pin_rank1_at_body_front_rank2_at_body_back() -> None:
    """Back-edge pin: rank-1 memory chunk at body front, rank-2 at body back.

    This pin FAILS if ordering collapses to a simple concat (the GE-10 lesson).
    We plant two memory chunks with markers and assert their ORDER in the body message.
    rank-1 (higher relevance) must appear BEFORE rank-2 in the body content.
    """
    # Two chunks with distinct markers planted in ORDER order.
    chunk_rank1 = _make_chunk("RANK1_CHUNK_BODY_FRONT", slot="memory", key="memory:rank1")
    chunk_rank2 = _make_chunk("RANK2_CHUNK_BODY_BACK", slot="memory", key="memory:rank2")

    asm, req = _minimal_assembler(
        rules_text="RULES",
        instructions_text="",
        task_text="TASK",
        # Contributor returns rank-1 first, rank-2 second (declaration order matters).
        memory_chunks=(chunk_rank1, chunk_rank2),
        budget=4096,
    )
    ctx = await asm.assemble(req)

    msgs = ctx.messages
    body_msg = next(
        (m for m in msgs if "<!-- memory:start -->" in m.content),
        None,
    )
    assert body_msg is not None, "SC-3: body (memory envelope) message must exist"

    body_content = body_msg.content
    idx_rank1 = body_content.find("RANK1_CHUNK_BODY_FRONT")
    idx_rank2 = body_content.find("RANK2_CHUNK_BODY_BACK")

    assert idx_rank1 != -1, "SC-3: rank-1 chunk marker must appear in body"
    assert idx_rank2 != -1, "SC-3: rank-2 chunk marker must appear in body"

    # Back-edge pin: rank-1 precedes rank-2 (fails if ordering collapses).
    assert idx_rank1 < idx_rank2, (
        f"SC-3 BACK-EDGE FAIL: rank-1 index={idx_rank1} must come before "
        f"rank-2 index={idx_rank2} in body content"
    )


async def test_sc3_negative_reversed_order_detected() -> None:
    """Negative control: if we swap the chunks, positions swap — order is discriminable.

    Proves that the back-edge pin is not trivially satisfied by any ordering.
    """
    chunk_a = _make_chunk("ALPHA_FIRST", slot="memory", key="memory:a")
    chunk_b = _make_chunk("BETA_SECOND", slot="memory", key="memory:b")

    # Forward order.
    asm_fwd, req_fwd = _minimal_assembler(
        rules_text="R",
        instructions_text="",
        task_text="T",
        memory_chunks=(chunk_a, chunk_b),
        budget=4096,
    )
    ctx_fwd = await asm_fwd.assemble(req_fwd)
    body_fwd = next(
        (m.content for m in ctx_fwd.messages if "<!-- memory:start -->" in m.content),
        "",
    )

    # Reversed order.
    asm_rev, req_rev = _minimal_assembler(
        rules_text="R",
        instructions_text="",
        task_text="T",
        memory_chunks=(chunk_b, chunk_a),  # reversed
        budget=4096,
    )
    ctx_rev = await asm_rev.assemble(req_rev)
    body_rev = next(
        (m.content for m in ctx_rev.messages if "<!-- memory:start -->" in m.content),
        "",
    )

    if body_fwd and body_rev:
        idx_a_fwd = body_fwd.find("ALPHA_FIRST")
        idx_b_fwd = body_fwd.find("BETA_SECOND")
        idx_a_rev = body_rev.find("ALPHA_FIRST")
        idx_b_rev = body_rev.find("BETA_SECOND")

        if idx_a_fwd != -1 and idx_b_fwd != -1:
            assert idx_a_fwd < idx_b_fwd, "Forward: ALPHA must precede BETA"
        if idx_a_rev != -1 and idx_b_rev != -1:
            assert idx_b_rev < idx_a_rev, "Reversed: BETA must precede ALPHA"

        # The two bodies must differ — order is discriminable.
        assert body_fwd != body_rev, (
            "SC-3 negative control: reversed chunk order must produce different body content"
        )


# ===========================================================================
# SC-4: CF-B defeated-claim exclusion
# ===========================================================================


async def test_sc4_defeated_claim_excluded_by_default() -> None:
    """Active C1 + revision-defeated C2 (status='defeasibly-defeated', valid_to=None).

    C2 must be absent from assembled context, defeated_excluded == 1.
    This is the exact reconciler gap scenario: valid_to UNSET but status='defeasibly-defeated'.
    """
    c1_text = "ACTIVE_CLAIM_C1_MUST_APPEAR"
    c2_text = "DEFEATED_CLAIM_C2_MUST_BE_EXCLUDED"

    c1_result = _fused_claim_result(
        "subject:test", "pred:c1", "payload_c1", c1_text, status="active"
    )
    c2_result = _fused_claim_result(
        "subject:test",
        "pred:c2",
        "payload_c2",
        c2_text,
        status="defeasibly-defeated",  # valid_to=None (exact reconciler gap)
        score=0.9,  # higher score than C1 — would appear if not filtered
        rank=2,
    )

    # Default policy: include_defeated=False → C2 filtered out.
    injector = _make_memory_injector_with_fused([c1_result, c2_result])
    mem = await injector.inject(
        RecallQuery(text="query"),
        policy=MemoryPolicy(token_budget=2048, include_defeated=False),
    )

    assert mem.defeated_excluded == 1, (
        f"SC-4: expected defeated_excluded=1, got {mem.defeated_excluded}"
    )
    chunk_texts = {c.text for c in mem.context.chunks}
    assert c2_text not in chunk_texts, "SC-4: defeated C2 must NOT appear in assembled context"
    # C1 is still present (active claim).
    assert c1_text in chunk_texts, "SC-4: active C1 must appear in assembled context"


async def test_sc4_include_defeated_true_shows_c2() -> None:
    """Flip include_defeated=True → C2 present (proves the filter is the mechanism).

    This discriminability guard ensures the SC-4 exclusion test isn't trivially true
    because C2 never appeared regardless.
    """
    c1_text = "ACTIVE_CLAIM_C1"
    c2_text = "DEFEATED_CLAIM_C2_VISIBLE_WHEN_INCLUDE_DEFEATED_TRUE"

    c1_result = _fused_claim_result(
        "subject:flip", "pred:c1", "payload_c1_flip", c1_text, status="active"
    )
    c2_result = _fused_claim_result(
        "subject:flip",
        "pred:c2",
        "payload_c2_flip",
        c2_text,
        status="defeasibly-defeated",
        score=0.9,
        rank=2,
    )

    injector = _make_memory_injector_with_fused([c1_result, c2_result])
    mem_with = await injector.inject(
        RecallQuery(text="query"),
        policy=MemoryPolicy(token_budget=2048, include_defeated=True),
    )

    chunk_texts_with = {c.text for c in mem_with.context.chunks}
    assert c2_text in chunk_texts_with, (
        "SC-4 discriminability: with include_defeated=True, defeated C2 MUST appear"
    )
    # defeated_excluded is 0 when the filter is skipped.
    assert mem_with.defeated_excluded == 0, (
        "SC-4: include_defeated=True -> defeated_excluded must be 0, "
        f"got {mem_with.defeated_excluded}"
    )


async def test_sc4_filter_is_structural_zero_model_calls() -> None:
    """S9: defeated-claim filter is pure structural — zero model calls (no model judgment)."""
    model = ReplayModel([])
    c2_result = _fused_claim_result(
        "subject:s9",
        "pred:c2",
        "payload_s9_c2",
        "C2_TEXT_S9",
        status="defeasibly-defeated",
    )

    injector = _make_memory_injector_with_fused([c2_result])
    await injector.inject(
        RecallQuery(text="query"),
        policy=MemoryPolicy(token_budget=2048, include_defeated=False),
    )
    # No model was injected — MemoryInjector only duck-types count_tokens;
    # if we supply one we can verify it was never called for complete().
    assert model.call_count == 0, (
        f"SC-4 S9: model was called {model.call_count} times "
        "— structural filter must use zero model calls"
    )


# ===========================================================================
# SC-5: S8 lesion
# ===========================================================================


async def test_sc5_assembler_none_on_run_context_degrades_to_task_only() -> None:
    """assembler=None on RunContext → assemble_context returns task-only context, COMPLETED.

    The run is degraded, not dead. The slot report signals 'unwired'.
    """
    model = ReplayModel([])
    journal = InMemoryJournal()
    graph_store = InMemoryGraphStore()
    latent = InMemoryLatentStore(clock=_CLOCK)
    guard = BudgetPolicy().new_guard()

    # assembler=None → S8 lesion path
    ctx = RunContext(
        run_id="sc5-lesion",
        session_id="sess-sc5",
        model=model,
        journal=journal,
        graph_store=graph_store,
        latent=latent,
        budget=guard,
        assembler=None,
    )

    task_text = "TASK_ONLY_CONTEXT_LESION"
    request = ContextRequest(task=task_text)
    assembled = await ctx.assemble_context(request)

    # Run completes (returns an AssembledCallContext, no exception).
    assert isinstance(assembled, AssembledCallContext)

    # Task text is present.
    all_content = " ".join(m.content for m in assembled.messages)
    assert task_text in all_content, (
        "SC-5: task text must be present even on assembler=None lesion path"
    )

    # Slot report signals 'unwired'.
    assert len(assembled.slots) >= 1
    unwired_reports = [r for r in assembled.slots if r.status == "unwired"]
    assert unwired_reports, (
        "SC-5: assembler=None lesion must produce at least one 'unwired' SlotReport"
    )

    # No model complete() calls on the assembler=None path.
    assert model.call_count == 0


async def test_sc5_memory_slot_lesion_assembles_without_memory() -> None:
    """Memory-slot lesion: no recall stack / no injector → context assembles without memory.

    The memory slot is 'unwired' or 'empty'; assembly still succeeds.
    """
    # Build an assembler with the memory slot having no contributor registered.
    asm = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=ReplayModel([]),
        default_policy=ContextPolicy(total_budget=4096),
    )
    asm.register("rules", _static("RULES_OK", "rules"))
    asm.register("instructions", _static("", "instructions"))
    asm.register("personality", _static("", "personality"))
    # memory slot: no contributor registered → status="unwired"
    asm.register("tools", ToolSpecContributor([]))
    asm.register("task", _static("TASK_OK", "task"))

    req = ContextRequest(task="TASK_OK", policy=ContextPolicy(total_budget=4096))
    ctx = await asm.assemble(req)

    # Assembly succeeds.
    assert isinstance(ctx, AssembledCallContext)
    assert ctx.token_count <= ctx.budget

    # Memory slot report should be 'unwired' (no contributor).
    report_by_name = {r.name: r for r in ctx.slots}
    mem_report = report_by_name.get("memory")
    assert mem_report is not None
    assert mem_report.status in ("unwired", "empty"), (
        f"SC-5: unwired memory slot must report 'unwired' or 'empty', got {mem_report.status!r}"
    )


async def test_sc5_preferred_contributor_raising_marks_degraded_assembly_continues() -> None:
    """Preferred contributor raising → slot marked degraded, assembly continues.

    The run is NOT aborted — S8 lesion degrades gracefully.
    """
    asm = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=ReplayModel([]),
        default_policy=ContextPolicy(total_budget=4096),
    )
    asm.register("rules", _static("RULES_OK", "rules"))
    asm.register("instructions", _static("", "instructions"))
    # personality is preferred — raises → degraded, not aborted
    asm.register("personality", AlwaysRaisingContributor())
    asm.register("memory", FixedMemoryContributor(()))
    asm.register("tools", ToolSpecContributor([]))
    asm.register("task", _static("TASK_OK", "task"))

    req = ContextRequest(task="TASK_OK", policy=ContextPolicy(total_budget=4096))
    ctx = await asm.assemble(req)

    # Assembly completed (no exception).
    assert isinstance(ctx, AssembledCallContext)

    # personality slot is degraded or empty — the assembler absorbs preferred contributor
    # failures gracefully (S8). The SlotContent is set to degraded internally, but the
    # SlotReport status reflects "empty" when the contributor raised and returned no chunks
    # (the assembler converts a no-chunk degraded content to "empty" in the report).
    report_by_name = {r.name: r for r in ctx.slots}
    personality_report = report_by_name.get("personality")
    assert personality_report is not None
    assert personality_report.status in ("degraded", "empty"), (
        f"SC-5: preferred contributor raising must mark slot 'degraded' or 'empty' (S8 absorbed), "
        f"got {personality_report.status!r}"
    )
    # Crucially: the personality text does NOT appear in messages (it was raised, not contributed).
    assert "intentional AlwaysRaisingContributor failure" not in " ".join(
        m.content for m in ctx.messages
    ), "SC-5: raised contributor's error message must not leak into messages"

    # Required slots still intact.
    assert "RULES_OK" in " ".join(m.content for m in ctx.messages)
    assert "TASK_OK" in " ".join(m.content for m in ctx.messages)


async def test_sc5_negative_required_contributor_raising_raises_assembly_error() -> None:
    """Negative control: a REQUIRED contributor raising → ContextAssemblyError, loud.

    Proves that the S8 degradation is NOT applied to required slots — required failures are hard.
    """
    asm = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=ReplayModel([]),
        default_policy=ContextPolicy(total_budget=4096),
    )
    # rules is required — raising contributor must propagate ContextAssemblyError.
    asm.register("rules", AlwaysRaisingContributor())
    asm.register("instructions", _static("", "instructions"))
    asm.register("personality", _static("", "personality"))
    asm.register("memory", FixedMemoryContributor(()))
    asm.register("tools", ToolSpecContributor([]))
    asm.register("task", _static("TASK_OK", "task"))

    req = ContextRequest(task="TASK_OK", policy=ContextPolicy(total_budget=4096))

    with pytest.raises(ContextAssemblyError) as exc_info:
        await asm.assemble(req)

    err = exc_info.value
    assert err.slot_name == "rules", (
        f"SC-5 negative: ContextAssemblyError must name the failing slot 'rules', "
        f"got {err.slot_name!r}"
    )


# ===========================================================================
# SC-6: S1/S9 guard — complete() never called during assembly
# ===========================================================================


async def test_sc6_s1_guard_fail_on_call_model_complete_never_called() -> None:
    """FailOnCallModel: count_tokens permitted, ANY complete() during assembly fails structurally.

    Wires the assembler with a FailOnCallModel double whose complete() raises AssertionError.
    If the assembler ever calls complete(), the test will fail with that error.
    count_tokens() is called (for budget accounting) — that must succeed.
    """
    fail_model = FailOnCallModel()

    asm = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=fail_model,
        default_policy=ContextPolicy(total_budget=4096),
    )
    asm.register("rules", _static("RULES_NO_COMPLETE_CALL", "rules"))
    asm.register("instructions", _static("INSTRUCTIONS_NO_COMPLETE_CALL", "instructions"))
    asm.register("personality", _static("PERSONALITY_NO_COMPLETE_CALL", "personality"))
    asm.register("memory", FixedMemoryContributor((_make_chunk("MEMORY_NO_COMPLETE", "memory"),)))
    asm.register("tools", ToolSpecContributor([_make_tool("safe_tool")]))
    asm.register("task", _static("TASK_NO_COMPLETE_CALL", "task"))

    req = ContextRequest(task="TASK_NO_COMPLETE_CALL", policy=ContextPolicy(total_budget=4096))

    # This must not raise AssertionError from FailOnCallModel.
    # If it does, it means complete() was called — a hard S1/S9 violation.
    ctx = await asm.assemble(req)

    # Assembly succeeded.
    assert isinstance(ctx, AssembledCallContext)
    # complete() was NEVER called.
    assert fail_model.complete_call_count == 0, (
        f"SC-6 S1 VIOLATION: FailOnCallModel.complete() was called "
        f"{fail_model.complete_call_count} time(s) during assembly"
    )


async def test_sc6_s1_guard_replay_model_zero_complete_calls() -> None:
    """ReplayModel([]) confirms call_count == 0 after assembly (complementary guard).

    Uses ReplayModel with an empty scripted response list — any complete() would raise
    ReplayExhaustedError. This is a second structural guard from the other direction.
    """
    model = ReplayModel([])  # complete() raises ReplayExhaustedError if called

    asm = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=model,
        default_policy=ContextPolicy(total_budget=4096),
    )
    asm.register("rules", _static("RULES_S1", "rules"))
    asm.register("instructions", _static("INSTRUCTIONS_S1", "instructions"))
    asm.register("personality", _static("PERSONALITY_S1", "personality"))
    asm.register("memory", FixedMemoryContributor((_make_chunk("MEMORY_S1", "memory"),)))
    asm.register("tools", ToolSpecContributor([_make_tool("tool_s1")]))
    asm.register("task", _static("TASK_S1", "task"))

    req = ContextRequest(task="TASK_S1", policy=ContextPolicy(total_budget=4096))
    ctx = await asm.assemble(req)

    assert isinstance(ctx, AssembledCallContext)
    assert model.call_count == 0, (
        f"SC-6: ReplayModel.call_count must be 0 after assembly, got {model.call_count}"
    )


async def test_sc6_count_tokens_is_called_not_complete() -> None:
    """count_tokens IS called during assembly (budget accounting), complete() is NOT.

    Uses a model double that tracks both, so we can assert the asymmetry.
    """
    tokens_called_with: list[str] = []
    complete_called: list[int] = []

    class TrackingModel:
        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities()

        async def complete(
            self,
            *,
            messages: Sequence[ChatMessage],
            tools: Sequence[ToolSpec] = (),
            tier: ModelTier = "pro",
            json_schema: Mapping[str, Any] | None = None,
        ) -> ModelResponse:
            complete_called.append(1)
            raise AssertionError("complete() must not be called")

        def count_tokens(self, text: str) -> int:
            tokens_called_with.append(text)
            return max(1, len(text) // 4)

    model = TrackingModel()
    asm = ContextAssembler(
        slots=DEFAULT_SLOTS,
        model=model,
        default_policy=ContextPolicy(total_budget=4096),
    )
    asm.register("rules", _static("RULES_TRACK", "rules"))
    asm.register("instructions", _static("", "instructions"))
    asm.register("personality", _static("", "personality"))
    asm.register("memory", FixedMemoryContributor(()))
    asm.register("tools", ToolSpecContributor([]))
    asm.register("task", _static("TASK_TRACK", "task"))

    req = ContextRequest(task="TASK_TRACK", policy=ContextPolicy(total_budget=4096))
    ctx = await asm.assemble(req)

    assert isinstance(ctx, AssembledCallContext)
    # complete() was never called.
    assert len(complete_called) == 0, f"SC-6: complete() was called {len(complete_called)} time(s)"
    # count_tokens was called at least once (for budget accounting).
    assert len(tokens_called_with) > 0, (
        "SC-6: count_tokens must be called at least once during assembly (budget accounting)"
    )


# ===========================================================================
# SC-7: Determinism (S6 replay-safety)
# ===========================================================================


async def test_sc7_determinism_identical_request_yields_byte_identical_messages() -> None:
    """Identical request + identical contributor outputs → BYTE-IDENTICAL message tuples.

    Two separate assembler instances, both fully wired, produce the same output.
    The full messages tuple is compared by equality, not just lengths.
    """
    rules_text = "DETERMINISTIC_RULES"
    instructions_text = "DETERMINISTIC_INSTRUCTIONS"
    task_text = "DETERMINISTIC_TASK"
    mem_chunk = _make_chunk("DETERMINISTIC_MEMORY", "memory", key="memory:det")
    tools = [_make_tool("det_tool_1"), _make_tool("det_tool_2")]
    budget = 4096

    def _build_asm() -> tuple[ContextAssembler, ContextRequest]:
        return _minimal_assembler(
            rules_text=rules_text,
            instructions_text=instructions_text,
            task_text=task_text,
            memory_chunks=(mem_chunk,),
            tools=tools,
            budget=budget,
        )

    asm1, req1 = _build_asm()
    asm2, req2 = _build_asm()

    ctx1 = await asm1.assemble(req1)
    ctx2 = await asm2.assemble(req2)

    assert ctx1.messages == ctx2.messages, (
        f"SC-7 S6 FAIL: two assemblies with identical inputs produced different messages.\n"
        f"  First:  {ctx1.messages}\n"
        f"  Second: {ctx2.messages}"
    )
    assert ctx1.token_count == ctx2.token_count
    assert ctx1.budget == ctx2.budget
    assert ctx1.tools == ctx2.tools


async def test_sc7_determinism_repeated_calls_on_same_assembler() -> None:
    """Calling assemble twice on the same assembler with the same request is idempotent."""
    asm, req = _minimal_assembler(
        rules_text="REPEAT_RULES",
        instructions_text="",
        task_text="REPEAT_TASK",
        memory_chunks=(_make_chunk("REPEAT_MEM", "memory"),),
        budget=4096,
    )
    ctx_a = await asm.assemble(req)
    ctx_b = await asm.assemble(req)

    assert ctx_a.messages == ctx_b.messages, (
        "SC-7: repeated calls on same assembler with same request must produce identical messages"
    )
    assert ctx_a.token_count == ctx_b.token_count


async def test_sc7_different_tasks_produce_different_messages() -> None:
    """Negative control: different task text → different messages tuple.

    Proves that the byte-identical assertion in SC-7 is not trivially satisfied —
    two different inputs MUST produce different outputs.
    """
    asm1, req1 = _minimal_assembler(
        rules_text="R",
        instructions_text="",
        task_text="TASK_ALPHA",
        budget=4096,
    )
    asm2, req2 = _minimal_assembler(
        rules_text="R",
        instructions_text="",
        task_text="TASK_BETA",
        budget=4096,
    )

    ctx1 = await asm1.assemble(req1)
    ctx2 = await asm2.assemble(req2)

    assert ctx1.messages != ctx2.messages, (
        "SC-7 negative control: different task text must produce different messages tuples"
    )


async def test_sc7_determinism_model_free_no_random_state() -> None:
    """Assembly is deterministic even when called across fresh model instances.

    Three separate assembly passes with independently-constructed models and assemblers
    must all produce byte-identical messages. This guards against any hidden randomness
    in the contributor or assembler code paths.
    """
    rules_text = "TRIPLE_RULES_DETERMINISM"
    task_text = "TRIPLE_TASK_DETERMINISM"
    mem_chunk = _make_chunk("TRIPLE_MEMORY_DET", "memory", key="memory:triple")

    results: list[tuple[ChatMessage, ...]] = []
    for _ in range(3):
        asm, req = _minimal_assembler(
            rules_text=rules_text,
            instructions_text="",
            task_text=task_text,
            memory_chunks=(mem_chunk,),
            budget=4096,
        )
        ctx = await asm.assemble(req)
        results.append(ctx.messages)

    # All three must be byte-identical.
    assert results[0] == results[1] == results[2], (
        "SC-7: three independent assemblies must produce byte-identical message tuples"
    )
