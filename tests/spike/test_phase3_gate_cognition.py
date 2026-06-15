"""Phase 3 gate spike (Pod 3.5, CANON S12) — assembled cognition + safety posture.

SC-1 full pipeline band order + 4 lesions   SC-5 approval truth table + no-self-approval
SC-2 ladder rung selection on drive path     SC-6 approval HITL end-to-end (deterministic)
SC-3 S11 pre-call budget exact-count         SC-7 taint persist-before-invoke, fail-closed
SC-4 ToolLoopLimit structural ceiling        SC-8 S1/S6/S9 invariants over the gate pathway

Pure Python — no Neo4j, no Postgres, no live model calls, no network.
"""

from __future__ import annotations

import contextlib
import itertools
import json
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import jsonschema
import pytest

from cogworx.capability.base import PermissionTier
from cogworx.capability.policy import (
    ApprovalRequired,
    StageToolPolicy,
    TaintState,
    ToolGate,
)
from cogworx.capability.registry import Registry
from cogworx.capability.router import (
    ToolLoopLimit,
    dispatch_one,
    run_tool_loop,
)
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.context.personality import PersonalityProfile
from cogworx.context.rules import Rule, RuleSet
from cogworx.context.types import ContextRequest
from cogworx.cost.budget import BudgetExceededError, BudgetGuard
from cogworx.knowledge.evidence import EvidenceEvent
from cogworx.knowledge.identity import claim_id_for
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import AwaitHuman, Degraded, Done, Transition
from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ToolCall,
)
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.ladder import (
    _FORCED_TOOL_NAME,
    StructuredOutputError,
    StructuredOutputModel,
)
from cogworx.model.registry import ModelRegistry
from cogworx.recall.query import RecallQuery
from cogworx.recall.stack import default_recall_stack
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    assert_control_independent_of_model_text,
    assert_no_model_on_write_path,
    assert_resume_never_recalls_model,
)

pytestmark = pytest.mark.spike  # asyncio_mode="auto" → no @pytest.mark.asyncio per test

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)

# Shared empty JSON schema used by all stub capabilities.
_EMPTY_SCHEMA: Mapping[str, Any] = {
    "type": "object",
    "properties": {},
    "additionalProperties": False,
}


def _counter_clock() -> Any:
    """An injected clock that advances 1s per call — avoids datetime.now() (S6)."""
    state = {"tick": 0}

    def clock() -> datetime:
        moment = _EPOCH + timedelta(seconds=state["tick"])
        state["tick"] += 1
        return moment

    return clock


def _prov(source: str = "system") -> Provenance:
    return Provenance(source=source, confidence=1.0, recorded_at=_EPOCH)


def _artifact(kind: str = "step", **data: Any) -> Artifact:
    return Artifact(kind=kind, produced_by="test", provenance=_prov(), data=dict(data))


def _initial_artifact() -> Artifact:
    return _artifact("start")


def _stop(text: str) -> ModelResponse:
    return ModelResponse(text=text, model_id="replay", finish_reason="stop")


def _build_engine(
    journal: InMemoryJournal,
    model: ReplayModel,
    *,
    pathways: PathwayRegistry,
    registry: Registry | None = None,
    personality: PersonalityProfile | None = None,
    rules: RuleSet | None = None,
    recall_stack: Any = None,
    entity_kg: InMemoryEntityKG | None = None,
) -> Engine:
    mr = ModelRegistry()
    mr.register("default", model)
    latent = InMemoryLatentStore()
    return Engine(
        models=mr,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=latent,
        pathways=pathways,
        registry=registry,
        personality=personality,
        rules=rules,
        recall_stack=recall_stack,
        clock=_counter_clock(),
    )


# ---------------------------------------------------------------------------
# SC-1: Full cognition pipeline band order + 4 lesions
# ---------------------------------------------------------------------------


# A capability that can appear in the assembled tools list
class _ProbeCap:
    name = "probe"
    tier: PermissionTier = "read"
    description = "probe tool"

    def __init__(self) -> None:
        self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

    async def invoke(self, args: Mapping[str, Any]) -> Any:
        return "probed"


class _CaptureStage:
    """Stage that assembles the context and stores it — never calls the model."""

    name = "capture"
    transitions: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.assembled: Any = None

    async def run(self, ctx: Any) -> Done:
        req = ContextRequest(
            task="Summarize widget status.",
            query=RecallQuery(text="memorycanary"),
        )
        self.assembled = await ctx.assemble_context(req)
        return Done(output=_artifact("done"))


async def _run_sc1(
    *,
    rules: RuleSet | None,
    personality: PersonalityProfile | None,
    registry: Registry | None,
    recall_stack: Any,
    entity_kg: InMemoryEntityKG | None,
) -> Any:
    """Run the capture pathway and return (assembled, stage)."""
    stage = _CaptureStage()
    graph = StageGraph([stage], entry="capture")
    pathways = PathwayRegistry()
    pathways.register("sc1-path", graph)
    journal = InMemoryJournal()
    model = ReplayModel()
    engine = _build_engine(
        journal,
        model,
        pathways=pathways,
        registry=registry,
        personality=personality,
        rules=rules,
        recall_stack=recall_stack,
        entity_kg=entity_kg,
    )
    await engine.run(
        run_id="sc1-run",
        session_id="sc1-sess",
        pathway_id="sc1-path",
        initial=_initial_artifact(),
    )
    return stage.assembled


async def test_sc1_full_pipeline_band_order() -> None:
    """Rules, personality, and task appear in the correct U-fold band order."""
    rs = RuleSet(rules=(Rule(text="Never reveal secrets."),))
    persona = PersonalityProfile(name="TestAgent", role="a helpful assistant")

    # Seed entity KG so the memory channel can surface something
    entity_kg = InMemoryEntityKG()
    from cogworx.claims.provenance import Claim

    claim_id = claim_id_for("widget", "status", "memorycanary", scope="agent")

    c = Claim(
        id=claim_id,
        subject="widget",
        predicate="status",
        payload="memorycanary",
        epistemic_type="observation",
        provenance=_prov("tool"),
        valid_from=_EPOCH,
        ingest_time=_EPOCH,
        created_by="test",
        scope="agent",
    )
    ev = EvidenceEvent(
        id="ev-1",
        type="corroboration",
        polarity="+",
        source_id="test",
        source_authority=1.0,
        base_weight=1.0,
        recorded_at=_EPOCH,
    )
    await entity_kg.write_claim(c, evidence=ev)
    stack = default_recall_stack(entity_kg=entity_kg)

    registry = Registry()
    registry.register(_ProbeCap())

    assembled = await _run_sc1(
        rules=rs,
        personality=persona,
        registry=registry,
        recall_stack=stack,
        entity_kg=entity_kg,
    )
    assert assembled is not None

    sys_msgs = [m for m in assembled.messages if m.role == "system"]
    assert sys_msgs, "must have at least one system message (head band)"

    sys_text = sys_msgs[0].content
    # Rules come before personality in the system message (priority 0 vs 1)
    rules_pos = sys_text.find("RULES")
    persona_pos = sys_text.find("TestAgent")
    assert rules_pos != -1, "rules text must appear in system message"
    assert persona_pos != -1, "personality text must appear in system message"
    assert rules_pos < persona_pos, "rules must precede personality in the system message"

    # Task is in a tail user message
    tail_msgs = [m for m in assembled.messages if m.role == "user"]
    assert tail_msgs, "must have user messages"
    task_text = assembled.messages[-1].content
    assert "Summarize widget status" in task_text, "task must appear in final user message"


async def test_sc1_lesion_rules() -> None:
    """rules=None → no rules text in system message."""
    assembled = await _run_sc1(
        rules=None,
        personality=PersonalityProfile(name="TestAgent"),
        registry=None,
        recall_stack=None,
        entity_kg=None,
    )
    assert assembled is not None
    sys_msgs = [m for m in assembled.messages if m.role == "system"]
    sys_text = " ".join(m.content for m in sys_msgs)
    # With rules=None, the placeholder static contributor emits "" so "RULES" must be absent
    assert "RULES" not in sys_text, "rules=None must produce no RULES section"


async def test_sc1_lesion_personality() -> None:
    """personality=None → no persona text in system message."""
    rs = RuleSet(rules=(Rule(text="Keep it brief."),))
    assembled = await _run_sc1(
        rules=rs,
        personality=None,
        registry=None,
        recall_stack=None,
        entity_kg=None,
    )
    assert assembled is not None
    sys_msgs = [m for m in assembled.messages if m.role == "system"]
    sys_text = " ".join(m.content for m in sys_msgs)
    # 'You are' is rendered by render_profile — must be absent
    assert "You are" not in sys_text, "personality=None must produce no persona text"


async def test_sc1_lesion_memory() -> None:
    """No recall_stack → memory slot empty (memorycanary keyword absent)."""
    assembled = await _run_sc1(
        rules=None,
        personality=None,
        registry=None,
        recall_stack=None,
        entity_kg=None,
    )
    assert assembled is not None
    all_content = " ".join(m.content for m in assembled.messages)
    assert "memorycanary" not in all_content, (
        "without a recall stack the memory keyword must be absent from assembled context"
    )


async def test_sc1_lesion_registry() -> None:
    """registry=None → assembler wires no RegistryToolContributor → tools tuple is empty."""
    assembled = await _run_sc1(
        rules=None,
        personality=None,
        registry=None,
        recall_stack=None,
        entity_kg=None,
    )
    assert assembled is not None
    assert assembled.tools == (), (
        f"registry=None must produce empty tools tuple, got {assembled.tools!r}"
    )


# ---------------------------------------------------------------------------
# SC-2: Ladder rung selection + framework-side validation
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}

_MSGS: list[ChatMessage] = [ChatMessage(role="user", content="Answer please.")]


async def test_sc2_rung1_native_validates_then_passes() -> None:
    """Rung 1 (structured_output=True): valid JSON → response returned with text matching schema."""
    payload = json.dumps({"answer": "yes"})
    inner = ReplayModel(
        [_stop(payload)],
        capabilities=ModelCapabilities(structured_output=True, tools=False),
    )
    som = StructuredOutputModel(inner)
    resp = await som.complete(messages=_MSGS, json_schema=_SCHEMA)
    parsed = json.loads(resp.text or "")
    assert parsed == {"answer": "yes"}, f"expected valid parsed dict, got {parsed!r}"
    assert inner.call_count == 1


async def test_sc2_rung1_invalid_json_raises_framework_side() -> None:
    """Rung 1: model returns non-JSON text → StructuredOutputError raised, never returned."""
    inner = ReplayModel(
        [_stop("not json at all")],
        capabilities=ModelCapabilities(structured_output=True, tools=False),
    )
    som = StructuredOutputModel(inner)
    with pytest.raises(StructuredOutputError):
        await som.complete(messages=_MSGS, json_schema=_SCHEMA)
    assert inner.call_count == 1


async def test_sc2_rung2_extracts_forced_tool_call() -> None:
    """Rung 2 (tools=True only): tool-call arguments extracted + validated → resp.text is JSON."""
    args = {"answer": "forty-two"}
    tc = ToolCall(id="tc-1", name=_FORCED_TOOL_NAME, arguments=args)
    resp_with_tc = ModelResponse(
        text=None,
        tool_calls=(tc,),
        model_id="replay",
        finish_reason="tool_use",
    )
    inner = ReplayModel(
        [resp_with_tc],
        capabilities=ModelCapabilities(structured_output=False, tools=True),
    )
    som = StructuredOutputModel(inner)
    resp = await som.complete(messages=_MSGS, json_schema=_SCHEMA)
    parsed = json.loads(resp.text or "")
    assert parsed == args, f"rung2 must return validated JSON from tool-call args; got {parsed!r}"
    assert inner.call_count == 1


async def test_sc2_rung3_retries_exactly_max_then_raises() -> None:
    """Rung 3 (no capabilities): bad JSON on all retries → StructuredOutputError.

    call_count must equal max_retries — model called exactly that many times.
    """
    from cogworx.model.ladder import _RetryRung

    max_retries = 3
    bad_responses = [_stop("garbage") for _ in range(max_retries)]
    inner = ReplayModel(
        bad_responses,
        capabilities=ModelCapabilities(structured_output=False, tools=False),
    )
    retry_rung = _RetryRung(max_retries=max_retries)
    som = StructuredOutputModel(inner, ladder=(retry_rung,))
    with pytest.raises(StructuredOutputError):
        await som.complete(messages=_MSGS, json_schema=_SCHEMA)
    assert inner.call_count == max_retries, (
        f"rung3 must call model exactly max_retries={max_retries} times; got {inner.call_count}"
    )


# ---------------------------------------------------------------------------
# SC-3: S11 pre-call budget exact count
# ---------------------------------------------------------------------------


async def test_sc3_call_ceiling_blocks_third_call_counter_exactly_2() -> None:
    """BudgetPolicy(max_calls_per_drive=2): calls 1+2 succeed; call 3 raises BudgetExceededError.

    inner.call_count == 2 (NOT 3) proves pre-call enforcement (S11).
    """
    inner = ReplayModel(
        [_stop("resp-1"), _stop("resp-2"), _stop("resp-3")],
        capabilities=ModelCapabilities(),
    )
    guard = BudgetGuard(max_calls=2)
    guarded = BudgetGuardedModel(inner, guard)

    msgs = [ChatMessage(role="user", content="hello")]

    resp1 = await guarded.complete(messages=msgs)
    assert resp1.text == "resp-1"
    resp2 = await guarded.complete(messages=msgs)
    assert resp2.text == "resp-2"

    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=msgs)

    assert inner.call_count == 2, (
        f"inner model must be called exactly 2 times; got {inner.call_count} (pre-call enforcement)"
    )


async def test_sc3_usd_ceiling_blocks_first_call_counter_zero() -> None:
    """BudgetPolicy with tiny USD ceiling + estimator that projects above it: call 1 is refused.

    inner.call_count == 0 proves the guard fires before the inner model is ever called.
    """
    inner = ReplayModel(
        [_stop("should-never-see-this")],
        capabilities=ModelCapabilities(),
    )
    guard = BudgetGuard(max_usd=0.0001)

    def _big_estimator(msgs: Any, tier: Any) -> float:
        return 1.0

    # estimator always projects above the ceiling
    guarded = BudgetGuardedModel(inner, guard, estimator=_big_estimator)

    msgs = [ChatMessage(role="user", content="hello")]
    with pytest.raises(BudgetExceededError):
        await guarded.complete(messages=msgs)

    assert inner.call_count == 0, (
        f"inner model must not be called at all; got {inner.call_count} (USD pre-call guard failed)"
    )


# ---------------------------------------------------------------------------
# SC-4: ToolLoopLimit ceiling
# ---------------------------------------------------------------------------


class _AlwaysToolModel:
    """Model double that always returns a tool call — drives the loop ceiling."""

    def __init__(self, tool_name: str, args: dict[str, Any]) -> None:
        self._tool_name = tool_name
        self._args = args
        self._call_count = 0

    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(tools=True)

    async def complete(
        self,
        *,
        messages: Any = (),
        tools: Any = (),
        tier: Any = "pro",
        json_schema: Any = None,
    ) -> ModelResponse:
        self._call_count += 1
        tc = ToolCall(
            id=f"tc-{self._call_count}",
            name=self._tool_name,
            arguments=self._args,
        )
        return ModelResponse(
            text=None,
            tool_calls=(tc,),
            model_id="always-tool",
            finish_reason="tool_use",
        )

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)

    @property
    def call_count(self) -> int:
        return self._call_count


async def test_sc4_loop_limit_exactly_4_calls() -> None:
    """run_tool_loop with max_rounds=3 raises ToolLoopLimit; model.call_count == 4 (3 + 1 final)."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    class _CountCap:
        name = "loop_tool"
        tier: PermissionTier = "read"
        description = "always more"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = schema
            self.call_count = 0

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            self.call_count += 1
            return "keep-going"

    cap = _CountCap()
    reg = Registry()
    reg.register(cap)
    gate = ToolGate(reg, policy=StageToolPolicy(allowed_tiers=frozenset({"read"})))
    model = _AlwaysToolModel("loop_tool", {})

    with pytest.raises(ToolLoopLimit) as exc_info:
        await run_tool_loop(
            gate,
            reg,
            model,
            [ChatMessage(role="user", content="start")],
            max_rounds=3,
        )

    # run_tool_loop: for _round in range(max_rounds): [call → has tools → dispatch → append]
    # then one final call after the last feedback → still has tools → ToolLoopLimit
    assert model.call_count == 4, (
        f"model must be called exactly 4 times (max_rounds=3 + 1 final); got {model.call_count}"
    )
    assert exc_info.value.rounds == 3


async def test_sc4_negative_model_stops_returns() -> None:
    """A model that stops after one tool call → clean return, no ToolLoopLimit raised."""
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {},
        "additionalProperties": False,
    }

    class _StopAfterOneCap:
        name = "one_shot"
        tier: PermissionTier = "read"
        description = "one shot"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = schema

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            return "done"

    # First response has a tool call; second response (post-feedback) stops cleanly
    resp_with_tc = ModelResponse(
        text=None,
        tool_calls=(ToolCall(id="tc-1", name="one_shot", arguments={}),),
        model_id="replay",
        finish_reason="tool_use",
    )
    resp_stop = _stop("all done")

    class _OneRoundModel:
        def __init__(self) -> None:
            self._responses = [resp_with_tc, resp_stop]
            self.call_count = 0

        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities(tools=True)

        async def complete(
            self,
            *,
            messages: Any = (),
            tools: Any = (),
            tier: Any = "pro",
            json_schema: Any = None,
        ) -> ModelResponse:
            self.call_count += 1
            return self._responses.pop(0)

        def count_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

    cap = _StopAfterOneCap()
    reg = Registry()
    reg.register(cap)
    gate = ToolGate(reg, policy=StageToolPolicy(allowed_tiers=frozenset({"read"})))
    model = _OneRoundModel()

    result = await run_tool_loop(
        gate,
        reg,
        model,
        [ChatMessage(role="user", content="start")],
        max_rounds=4,
    )
    assert result.text == "all done"
    assert model.call_count == 2  # one round (tool call) + one stop


# ---------------------------------------------------------------------------
# SC-5: Approval truth table + no-self-approval
# ---------------------------------------------------------------------------


def _build_trifecta_gate(*, tainted: bool = False) -> tuple[Registry, ToolGate, Any]:
    """Build a registry + gate for the trifecta tests. Returns (reg, gate, send_cap)."""

    class _SendCap:
        name = "send"
        tier: PermissionTier = "external"
        description = "send msg"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA
            self.call_count = 0

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            self.call_count += 1
            return "sent"

    send_cap = _SendCap()
    reg = Registry()
    reg.register(
        send_cap,
        tags=("consequential", "irreversible"),
    )
    taint_state = TaintState(tainted=tainted)
    gate = ToolGate(
        reg,
        policy=StageToolPolicy(
            allowed_tiers=frozenset({"external"}),
            taint_drops_external=False,  # keep external reachable so approval check fires
        ),
        taint=taint_state,
    )
    return reg, gate, send_cap


@pytest.mark.parametrize(
    "tainted,consequential,irreversible",
    list(itertools.product([False, True], repeat=3)),
)
async def test_sc5_approval_truth_table(
    tainted: bool, consequential: bool, irreversible: bool
) -> None:
    """ApprovalRequired iff tainted AND consequential AND irreversible AND not approved."""

    class _Cap:
        def __init__(self) -> None:
            self.call_count = 0
            self.name = "act"
            self.tier: PermissionTier = "external"
            self.description = "action"
            tags: tuple[str, ...] = ()
            if consequential:
                tags = (*tags, "consequential")
            if irreversible:
                tags = (*tags, "irreversible")
            self._tags = tags
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            self.call_count += 1
            return "ok"

    cap = _Cap()
    reg = Registry()
    reg.register(cap, tags=cap._tags)
    gate = ToolGate(
        reg,
        policy=StageToolPolicy(
            allowed_tiers=frozenset({"external"}),
            taint_drops_external=False,
        ),
        taint=TaintState(tainted=tainted),
    )

    requires_approval = tainted and consequential and irreversible

    if requires_approval:
        with pytest.raises(ApprovalRequired):
            await dispatch_one(gate, reg, "act", {}, approved=False)
        assert cap.call_count == 0, "invoke must not be called when approval required"
    else:
        result = await dispatch_one(gate, reg, "act", {}, approved=False)
        assert result == "ok"
        assert cap.call_count == 1


async def test_sc5_approved_grant_lets_trifecta_through() -> None:
    """(T, T, T) + approved=True → dispatch succeeds, invoke counter == 1."""
    reg, gate, send_cap = _build_trifecta_gate(tainted=True)
    result = await dispatch_one(gate, reg, "send", {}, approved=True)
    assert result == "sent"
    assert send_cap.call_count == 1


async def test_sc5_clean_drive_external_trifecta_runs_and_latches() -> None:
    """(F, T, T) external trifecta on clean drive → runs (no approval).
    After invoke, gate.taint.tainted is True. Immediate second call → ApprovalRequired.
    """
    reg, gate, send_cap = _build_trifecta_gate(tainted=False)

    # First call — drive is clean → succeeds
    result = await dispatch_one(gate, reg, "send", {}, approved=False)
    assert result == "sent"
    assert send_cap.call_count == 1

    # Taint must be latched now (send is external without trusted-output)
    assert gate.taint.tainted is True, "external dispatch must latch taint"

    # Second call — now tainted + consequential + irreversible → ApprovalRequired
    with pytest.raises(ApprovalRequired):
        await dispatch_one(gate, reg, "send", {}, approved=False)
    assert send_cap.call_count == 1, "second call must not reach invoke"


async def test_sc5_model_supplied_approved_in_args_rejected_by_schema() -> None:
    """{approved: True} in args with a schema that has no 'approved' property → ValidationError.

    The model cannot self-approve by injecting approved=True in the args dict;
    schema enforcement (additionalProperties:false) catches it before invocation.
    """
    reg, gate, send_cap = _build_trifecta_gate(tainted=True)
    # The send cap has additionalProperties:false + no 'approved' property
    with pytest.raises(jsonschema.ValidationError):
        await dispatch_one(gate, reg, "send", {"approved": True}, approved=False)
    # Arg validation fires at step ② BEFORE approval check and BEFORE invoke
    assert send_cap.call_count == 0, "invoke must not be called when args fail schema"


# ---------------------------------------------------------------------------
# SC-6: Approval HITL end-to-end (deterministic 3-stage pathway)
# ---------------------------------------------------------------------------

# Pathway: taint_stage → act_stage → finish_stage


class _TaintAndTransitionStage:
    """Stage 0 (step 0): dispatches an external cap to taint the drive, then Transition → act."""

    name = "taint"
    transitions: tuple[str, ...] = ("act",)
    tool_policy = StageToolPolicy(
        allowed_tiers=frozenset({"external"}),
        taint_drops_external=False,
    )

    def __init__(self, reg: Registry) -> None:
        self._reg = reg

    async def run(self, ctx: Any) -> Transition:
        # Dispatch the external cap to latch taint
        assert ctx._gate is not None
        await dispatch_one(ctx._gate, self._reg, "fetch_ext", {}, approved=False)
        return Transition(to="act", output=_artifact("tainted"))


class _ActStage:
    """Stage 1 (step 1): tries to dispatch 'send' → ApprovalRequired → AwaitHuman."""

    name = "act"
    transitions: tuple[str, ...] = ("finish",)
    tool_policy = StageToolPolicy(
        allowed_tiers=frozenset({"external"}),
        taint_drops_external=False,
    )

    def __init__(self, reg: Registry) -> None:
        self._reg = reg

    async def run(self, ctx: Any) -> AwaitHuman:
        assert ctx._gate is not None
        with contextlib.suppress(ApprovalRequired):
            await dispatch_one(ctx._gate, self._reg, "send", {}, approved=False)
        return AwaitHuman(
            question="Approve the send action?",
            to="finish",
            output=_artifact("awaiting"),
        )


class _FinishStage:
    """Stage 2 (step 2): reads human input from step 1 and routes structurally."""

    name = "finish"
    transitions: tuple[str, ...] = ()
    # Must allow external tier so dispatch_approved("send") passes check_dispatch.
    tool_policy = StageToolPolicy(
        allowed_tiers=frozenset({"external"}),
        taint_drops_external=False,
    )

    def __init__(self, send_cap: Any) -> None:
        self._send_cap = send_cap

    async def run(self, ctx: Any) -> Done | Degraded:
        # step 1 is where act_stage committed its AwaitHuman
        answer = await ctx.read_human_input(1)
        if answer is None:
            return Degraded(reason="no human input", output=_artifact("no-input"))
        payload = answer.data or {}
        decision = payload.get("decision", "no")
        if decision == "yes":
            from cogworx.runtime.context import RunContext as _RunContext

            assert isinstance(ctx, _RunContext)
            await ctx.dispatch_approved("send", {})
            return Done(output=_artifact("approved"))
        else:
            return Degraded(reason="human denied", output=_artifact("denied"))


def _build_sc6_engine(
    journal: InMemoryJournal,
    model: ReplayModel,
    *,
    pathways: PathwayRegistry,
    registry: Registry,
) -> Engine:
    mr = ModelRegistry()
    mr.register("default", model)
    return Engine(
        models=mr,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        registry=registry,
        clock=_counter_clock(),
    )


def _build_sc6_fixtures() -> tuple[Registry, PathwayRegistry, Any, Any, Any]:
    """Build the shared trifecta registry + pathway + stage refs."""

    class _FetchExtCap:
        name = "fetch_ext"
        tier: PermissionTier = "external"
        description = "fetch external data"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            return "external-payload"

    class _SendCap:
        def __init__(self) -> None:
            self.name = "send"
            self.tier: PermissionTier = "external"
            self.description = "send msg"
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA
            self.call_count = 0

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            self.call_count += 1
            return "sent"

    send_cap = _SendCap()
    reg = Registry()
    reg.register(_FetchExtCap(), tags=())
    reg.register(send_cap, tags=("consequential", "irreversible"))

    taint_stage = _TaintAndTransitionStage(reg)
    act_stage = _ActStage(reg)
    finish_stage = _FinishStage(send_cap)

    graph = StageGraph([taint_stage, act_stage, finish_stage], entry="taint")
    pathways = PathwayRegistry()
    pathways.register("sc6-path", graph)
    return reg, pathways, send_cap, graph, (taint_stage, act_stage, finish_stage)


async def test_sc6_approve_path_send_invoked_completed() -> None:
    """Human approves → send counter == 1, run COMPLETED."""
    from cogworx.loop.state import RunStatus

    reg, pathways, send_cap, _, _ = _build_sc6_fixtures()
    journal = InMemoryJournal()
    engine = _build_sc6_engine(journal, ReplayModel(), pathways=pathways, registry=reg)

    await engine.run(
        run_id="sc6-run",
        session_id="sc6-sess",
        pathway_id="sc6-path",
        initial=_initial_artifact(),
    )
    state = await journal.load_run("sc6-run")
    assert state is not None
    assert state.status == RunStatus.AWAITING_HUMAN

    final = await engine.provide_human_input("sc6-run", payload={"decision": "yes"})
    assert final.status == RunStatus.COMPLETED, (
        f"approve path must reach COMPLETED, got {final.status!r}"
    )
    assert send_cap.call_count == 1, (
        f"send must be invoked exactly once on approve; got {send_cap.call_count}"
    )


async def test_sc6_deny_path_send_not_invoked_degraded() -> None:
    """Human denies → send counter == 0, run DEGRADED."""
    from cogworx.loop.state import RunStatus

    reg, pathways, send_cap, _, _ = _build_sc6_fixtures()
    journal = InMemoryJournal()
    engine = _build_sc6_engine(journal, ReplayModel(), pathways=pathways, registry=reg)

    await engine.run(
        run_id="sc6d-run",
        session_id="sc6d-sess",
        pathway_id="sc6-path",
        initial=_initial_artifact(),
    )
    state = await journal.load_run("sc6d-run")
    assert state is not None
    assert state.status == RunStatus.AWAITING_HUMAN

    final = await engine.provide_human_input("sc6d-run", payload={"decision": "no"})
    assert final.status == RunStatus.DEGRADED, (
        f"deny path must reach DEGRADED, got {final.status!r}"
    )
    assert send_cap.call_count == 0, f"send must NOT be invoked on deny; got {send_cap.call_count}"


# ---------------------------------------------------------------------------
# SC-7: Taint persist ordering (fail-closed)
# ---------------------------------------------------------------------------


async def test_sc7_persist_before_invoke_even_on_invoke_error() -> None:
    """persist_taint fires BEFORE cap.invoke; even when invoke raises, persist counter == 1."""
    persist_count = [0]
    invoke_count = [0]

    async def _persist() -> None:
        persist_count[0] += 1

    class _BoomCap:
        name = "boom"
        tier: PermissionTier = "external"
        description = "always raises"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            invoke_count[0] += 1
            raise RuntimeError("boom!")

    reg = Registry()
    reg.register(_BoomCap(), tags=())
    taint = TaintState(tainted=False)
    gate = ToolGate(
        reg,
        policy=StageToolPolicy(
            allowed_tiers=frozenset({"external"}),
            taint_drops_external=False,
        ),
        taint=taint,
        persist_taint=_persist,
    )

    with pytest.raises(RuntimeError, match="boom"):
        await dispatch_one(gate, reg, "boom", {}, approved=False)

    assert persist_count[0] == 1, (
        f"persist_taint must fire exactly once (before invoke); got {persist_count[0]}"
    )
    assert taint.tainted is True, "taint latch must be set even when invoke raised"
    assert invoke_count[0] == 1, "invoke WAS called (it raised, but it was called)"


async def test_sc7_persist_failure_prevents_invoke() -> None:
    """If persist_taint raises OSError, invoke is NOT called (fail-closed S6)."""
    invoke_count = [0]

    async def _failing_persist() -> None:
        raise OSError("journal unavailable")

    class _NormalCap:
        name = "act"
        tier: PermissionTier = "external"
        description = "action"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            invoke_count[0] += 1
            return "done"

    reg = Registry()
    reg.register(_NormalCap(), tags=())
    taint = TaintState(tainted=False)
    gate = ToolGate(
        reg,
        policy=StageToolPolicy(
            allowed_tiers=frozenset({"external"}),
            taint_drops_external=False,
        ),
        taint=taint,
        persist_taint=_failing_persist,
    )

    with pytest.raises(OSError, match="journal unavailable"):
        await dispatch_one(gate, reg, "act", {}, approved=False)

    assert invoke_count[0] == 0, (
        f"invoke must NOT be called when persist_taint raises; got {invoke_count[0]}"
    )


async def test_sc7_negative_non_tainting_tool_skips_persist() -> None:
    """A read-tier tool (no untrusted-source tag) never calls persist_taint; taint stays False."""
    persist_count = [0]

    async def _persist() -> None:
        persist_count[0] += 1

    class _ReadCap:
        name = "search"
        tier: PermissionTier = "read"
        description = "search"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            return "result"

    reg = Registry()
    reg.register(_ReadCap(), tags=())  # no untrusted-source, not external
    taint = TaintState(tainted=False)
    gate = ToolGate(
        reg,
        policy=StageToolPolicy(allowed_tiers=frozenset({"read"})),
        taint=taint,
        persist_taint=_persist,
    )

    await dispatch_one(gate, reg, "search", {}, approved=False)

    assert persist_count[0] == 0, (
        "persist_taint must not be called for a non-tainting read-tier tool;"
        f" got {persist_count[0]}"
    )
    assert taint.tainted is False, "taint must remain False after a read-tier dispatch"


# ---------------------------------------------------------------------------
# SC-8: S1/S6/S9 invariants over the assembled gate pathway
# ---------------------------------------------------------------------------


class _ThinkStage:
    """Stage 0: calls the model once + dispatches a probe tool → Transition."""

    name = "think"
    transitions: tuple[str, ...] = ("emit",)

    def __init__(self) -> None:
        self.model_responses_seen: list[str] = []

    async def run(self, ctx: Any) -> Transition:
        resp = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="think about it")]
        )
        self.model_responses_seen.append(resp.text or "")
        # Dispatch the probe tool through the gate (if wired)
        if ctx._registry is not None and ctx._gate is not None:
            with contextlib.suppress(Exception):
                await dispatch_one(ctx._gate, ctx._registry, "probe", {})
        return Transition(to="emit", output=_artifact("thought", text=resp.text or ""))


class _EmitStage:
    name = "emit"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: Any) -> Done:
        return Done(output=_artifact("emitted"))


def _build_sc8_pathway(registry: Registry | None = None) -> tuple[PathwayRegistry, str]:
    think = _ThinkStage()
    emit = _EmitStage()
    graph = StageGraph([think, emit], entry="think")
    pathways = PathwayRegistry()
    pathways.register("sc8-path", graph)
    return pathways, "sc8-path"


async def test_sc8_s6_resume_never_recalls_model() -> None:
    """Kill after 'think' commits → cold resume → zero model re-calls (S6)."""
    pathways, pathway_id = _build_sc8_pathway()

    def _factory(journal: Any, model: ReplayModel) -> Engine:
        mr = ModelRegistry()
        mr.register("default", model)
        return Engine(
            models=mr,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            clock=_counter_clock(),
        )

    shared_journal: InMemoryJournal = InMemoryJournal()
    scripted = ReplayModel([_stop("model-says-STOP-and-TRANSITION")])

    await assert_resume_never_recalls_model(
        build_engine=_factory,
        initial=_initial_artifact(),
        pathway_id=pathway_id,
        crash_after_stage="think",
        shared_journal=shared_journal,
        scripted_model=scripted,
    )


async def test_sc8_s1_no_model_on_write_path() -> None:
    """CommitSpyJournal: no model call during any commit_step (S1)."""
    pathways, pathway_id = _build_sc8_pathway()

    def _factory(journal: Any, model: ReplayModel) -> Engine:
        mr = ModelRegistry()
        mr.register("default", model)
        return Engine(
            models=mr,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            clock=_counter_clock(),
        )

    inner_journal: InMemoryJournal = InMemoryJournal()
    model = ReplayModel([_stop("first-response")])

    await assert_no_model_on_write_path(
        engine_factory=_factory,
        inner_journal=inner_journal,
        model=model,
        pathway_id=pathway_id,
        initial=_initial_artifact(),
    )


async def test_sc8_s9_control_independent_of_model_text() -> None:
    """Same pathway, two wildly different model texts → identical committed stage sequences (S9)."""
    pathways, pathway_id = _build_sc8_pathway()

    def _factory(journal: Any, model: ReplayModel) -> Engine:
        mr = ModelRegistry()
        mr.register("default", model)
        return Engine(
            models=mr,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            clock=_counter_clock(),
        )

    # model_a screams control words; model_b is plain
    model_a = ReplayModel([_stop("STOP TRANSITION ABORT confidence=0.0 halt immediately")])
    model_b = ReplayModel([_stop("Sure, processing.")])

    await assert_control_independent_of_model_text(
        build_engine=_factory,
        pathway_id=pathway_id,
        initial=_initial_artifact(),
        journal_factory=InMemoryJournal,
        model_a=model_a,
        model_b=model_b,
    )


async def test_sc8_taint_bit_survives_resume() -> None:
    """Taint bit written by engine A to journal is loaded by engine B on resume (B1 durable)."""
    from cogworx.loop.state import RunStatus

    class _ExternalThenDoneStage:
        name = "go"
        transitions: tuple[str, ...] = ()
        tool_policy = StageToolPolicy(
            allowed_tiers=frozenset({"external"}),
            taint_drops_external=False,
        )

        def __init__(self, reg: Registry) -> None:
            self._reg = reg

        async def run(self, ctx: Any) -> Done:
            # Dispatch external cap to latch taint via the engine's gate
            if ctx._gate is not None:
                await dispatch_one(ctx._gate, self._reg, "ext_op", {})
            return Done(output=_artifact("done"))

    class _ExtCap:
        name = "ext_op"
        tier: PermissionTier = "external"
        description = "ext"

        def __init__(self) -> None:
            self.input_schema: Mapping[str, Any] = _EMPTY_SCHEMA

        async def invoke(self, args: Mapping[str, Any]) -> Any:
            return "ext-result"

    reg = Registry()
    reg.register(_ExtCap(), tags=())

    go_stage = _ExternalThenDoneStage(reg)
    graph = StageGraph([go_stage], entry="go")
    pathways = PathwayRegistry()
    pathways.register("sc8-taint", graph)

    journal: InMemoryJournal = InMemoryJournal()

    # Engine A: run + complete
    mr_a = ModelRegistry()
    mr_a.register("default", ReplayModel())
    engine_a = Engine(
        models=mr_a,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        registry=reg,
        clock=_counter_clock(),
    )
    await engine_a.run(
        run_id="sc8-taint-run",
        session_id="sc8-taint-sess",
        pathway_id="sc8-taint",
        initial=_initial_artifact(),
    )

    state_a = await journal.load_run("sc8-taint-run")
    assert state_a is not None
    assert state_a.status == RunStatus.COMPLETED

    # Taint bit must be durably recorded in the journal
    assert state_a.tainted is True, (
        "external dispatch must have set run tainted=True in the journal"
    )

    # Engine B: fresh instance, same journal — loads tainted=True from journal
    mr_b = ModelRegistry()
    mr_b.register("default", ReplayModel())
    engine_b = Engine(
        models=mr_b,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        registry=reg,
        clock=_counter_clock(),
    )
    # Resume reads the completed run; the taint bit survives in the journal
    state_b = await engine_b.resume("sc8-taint-run")
    assert state_b.tainted is True, (
        "taint bit must survive in journal and be visible to engine B on resume"
    )
