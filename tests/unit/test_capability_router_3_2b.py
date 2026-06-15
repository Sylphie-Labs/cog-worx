"""Pod 3.2b Feature Test Bundle — route_tool_calls, tool_result_messages, run_tool_loop.

Mutation-resistant assertions throughout: exact status values, exact-count proofs, zero-invoke-
counter checks.  No "absence of exception" or "does not raise" tests — every assertion pins a
concrete observable.

Coverage:
  R1 — successful dispatch → status "ok", content is JSON, invoke called exactly once.
  R2 — unknown tool → status "unknown_tool" (via unavailable), invoke NOT called.
  R3 — disabled/lesioned cap → status "unavailable", invoke NOT called.
  R4 — tier refused → status "refused_tier", invoke NOT called, TierViolation path.
  R5 — invalid args (wrong type) → status "invalid_args", invoke NOT called.
  R6 — invalid args (missing required field) → status "invalid_args", invoke NOT called.
  R7 — invalid args (extra key with additionalProperties: false) → status "invalid_args".
  R8 — tool raises → status "error", invoke was called (error comes from invoke).
  R9 — tool times out → status "timeout", invoke was called and exceeded timeout.
  R10 — tool_result_messages produces one ChatMessage per result with correct role and id.
  R11 — run_tool_loop ceiling: exactly max_rounds model calls then ToolLoopLimit.
  R12 — run_tool_loop terminates early when model returns no tool_calls.
  R13 — taint latched after successful external dispatch (only if ①②③ passed).
  R14 — taint NOT latched on failed dispatch (tier refused, arg invalid, tool error).
  R15 — batch dispatch: multiple tool calls in one round, results parallel to calls.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from pydantic import ValidationError

from cogworx.capability.policy import StageToolPolicy, TaintState, ToolGate
from cogworx.capability.registry import Registry, function_capability
from cogworx.capability.router import (
    ToolLoopLimit,
    ToolResult,
    dispatch_one,
    route_tool_calls,
    run_tool_loop,
    tool_result_messages,
)
from cogworx.model.base import ChatMessage, ModelCapabilities, ModelResponse, ToolCall
from cogworx.testing.fake_model import ReplayModel

# ---------------------------------------------------------------------------
# Fake capabilities and helpers
# ---------------------------------------------------------------------------


_invoke_counter: dict[str, int] = {}


async def _echo(x: str) -> str:
    _invoke_counter["echo"] = _invoke_counter.get("echo", 0) + 1
    return x


async def _always_raise(x: str) -> str:
    _invoke_counter["always_raise"] = _invoke_counter.get("always_raise", 0) + 1
    raise RuntimeError("boom from tool")


async def _slow(x: str) -> str:
    _invoke_counter["slow"] = _invoke_counter.get("slow", 0) + 1
    await asyncio.sleep(10)
    return x


async def _external_clean(x: str) -> str:
    _invoke_counter["ext_clean"] = _invoke_counter.get("ext_clean", 0) + 1
    return x


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    """Clear the global invoke counter before every test."""
    _invoke_counter.clear()


def _make_gate_and_registry(
    *,
    policy: StageToolPolicy | None = None,
    taint: TaintState | None = None,
    include_echo: bool = True,
    include_always_raise: bool = False,
    include_slow: bool = False,
    include_external: bool = False,
    external_tags: tuple[str, ...] = (),
    disable_echo: bool = False,
) -> tuple[ToolGate, Registry]:
    reg = Registry()
    if include_echo:
        cap = function_capability(_echo, name="echo", tier="read")
        reg.register(cap)
    if include_always_raise:
        cap_r = function_capability(_always_raise, name="always_raise", tier="read")
        reg.register(cap_r)
    if include_slow:
        cap_s = function_capability(_slow, name="slow", tier="read")
        reg.register(cap_s)
    if include_external:
        cap_e = function_capability(_external_clean, name="ext_clean", tier="external")
        reg.register(cap_e, tags=external_tags)
    if disable_echo and include_echo:
        reg.disable("echo")
    gate = ToolGate(reg, policy=policy, taint=taint)
    return gate, reg


def _tool_call(name: str, args: dict[str, Any], *, id_: str = "tc1") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args)


# ---------------------------------------------------------------------------
# R1 — successful dispatch → status "ok", content JSON, invoke called once
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_ok_status_and_json_content() -> None:
    gate, reg = _make_gate_and_registry()
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {"x": "hello"})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "ok"
    assert r.error is None
    parsed = json.loads(r.content)
    assert parsed == "hello"
    assert _invoke_counter.get("echo") == 1, "invoke must be called exactly once"


# ---------------------------------------------------------------------------
# R2 — unknown tool → status "unavailable", invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_unknown_tool_unavailable() -> None:
    gate, reg = _make_gate_and_registry()
    results = await route_tool_calls(gate, reg, [_tool_call("does_not_exist", {"x": "v"})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "unavailable", f"expected unavailable, got {r.status}"
    assert r.error is not None
    assert _invoke_counter == {}, "invoke must not be called for unknown tool"


# ---------------------------------------------------------------------------
# R3 — disabled/lesioned cap → status "unavailable", invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_disabled_cap_unavailable() -> None:
    gate, reg = _make_gate_and_registry(disable_echo=True)
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {"x": "v"})])
    assert len(results) == 1
    assert results[0].status == "unavailable"
    assert _invoke_counter == {}, "invoke must not be called for disabled cap"


# ---------------------------------------------------------------------------
# R4 — tier refused → status "refused_tier", invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_tier_refused() -> None:
    gate, reg = _make_gate_and_registry(include_external=True)
    # Default policy: read+write only; ext_clean is external → refused.
    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "refused_tier", f"expected refused_tier, got {r.status}"
    assert r.error is not None
    assert _invoke_counter == {}, "invoke must not be called when tier refused"


# ---------------------------------------------------------------------------
# R5 — invalid args (wrong type) → status "invalid_args", invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_invalid_args_wrong_type() -> None:
    gate, reg = _make_gate_and_registry()
    # _echo expects x: str; passing an int is invalid.
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {"x": 42})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "invalid_args", f"expected invalid_args, got {r.status}"
    assert r.error is not None
    assert _invoke_counter == {}, "invoke must not be called on arg validation failure"


# ---------------------------------------------------------------------------
# R6 — invalid args (missing required field) → status "invalid_args", invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_invalid_args_missing_required() -> None:
    gate, reg = _make_gate_and_registry()
    # _echo requires x; pass empty dict.
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "invalid_args"
    assert _invoke_counter == {}, "invoke must not be called on missing required field"


# ---------------------------------------------------------------------------
# R7 — extra key with additionalProperties: false → status "invalid_args"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_invalid_args_extra_key() -> None:
    gate, reg = _make_gate_and_registry()
    # _echo only declares x; extra key "y" must be rejected by additionalProperties: false.
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {"x": "hi", "y": "injected"})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "invalid_args", (
        f"extra key must be rejected by additionalProperties:false schema, got {r.status}"
    )
    assert _invoke_counter == {}, "invoke must not be called when extra key rejected"


# ---------------------------------------------------------------------------
# R8 — tool raises → status "error", invoke WAS called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_tool_raises_error_status() -> None:
    gate, reg = _make_gate_and_registry(include_always_raise=True)
    results = await route_tool_calls(gate, reg, [_tool_call("always_raise", {"x": "v"})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "error"
    assert r.error is not None
    assert "boom" in (r.error or "")
    assert _invoke_counter.get("always_raise") == 1, "invoke must have been called once"


# ---------------------------------------------------------------------------
# R9 — tool times out → status "timeout"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_tool_timeout() -> None:
    policy = StageToolPolicy(tool_timeout_s=0.01)  # 10 ms — far less than the 10 s sleep
    gate, reg = _make_gate_and_registry(policy=policy, include_slow=True)
    results = await route_tool_calls(gate, reg, [_tool_call("slow", {"x": "v"})])
    assert len(results) == 1
    r = results[0]
    assert r.status == "timeout", f"expected timeout, got {r.status}"
    assert r.error is not None


# ---------------------------------------------------------------------------
# R10 — tool_result_messages
# ---------------------------------------------------------------------------


def test_tool_result_messages_structure() -> None:
    results = [
        ToolResult(
            tool_call_id="id1",
            name="echo",
            status="ok",
            content='"hello"',
            error=None,
            latency_ms=1.0,
        ),
        ToolResult(
            tool_call_id="id2",
            name="echo",
            status="invalid_args",
            content='{"error":"invalid_args"}',
            error="bad args",
            latency_ms=0.5,
        ),
    ]
    msgs = tool_result_messages(results)
    assert len(msgs) == 2, f"expected 2 messages, got {len(msgs)}"
    for msg, res in zip(msgs, results, strict=True):
        assert msg.role == "tool"
        assert msg.tool_call_id == res.tool_call_id
        assert msg.content == res.content


def test_tool_result_messages_empty() -> None:
    assert tool_result_messages([]) == ()


# ---------------------------------------------------------------------------
# R11 — run_tool_loop ceiling: exactly max_rounds model calls then ToolLoopLimit
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_loop_ceiling() -> None:
    """A model that always returns tool_calls hits ToolLoopLimit at exactly max_rounds."""
    gate, reg = _make_gate_and_registry()

    # Build a model that always returns a tool_call.
    def _always_tool_response() -> ModelResponse:
        return ModelResponse(
            text=None,
            tool_calls=(ToolCall(id="tc", name="echo", arguments={"x": "v"}),),
            model_id="fake",
            finish_reason="tool_calls",
        )

    # We need max_rounds + 1 responses (one final check after the last round).
    max_rounds = 3
    # run_tool_loop does: for _round in range(max_rounds): complete → if no tool_calls: return
    # after the loop: one final complete → if still tool_calls: raise ToolLoopLimit.
    # Total complete calls = max_rounds + 1.
    responses = [_always_tool_response() for _ in range(max_rounds + 1)]
    model = ReplayModel(
        responses,
        capabilities=ModelCapabilities(tools=True),
    )

    messages = [ChatMessage(role="user", content="go")]
    with pytest.raises(ToolLoopLimit) as exc_info:
        await run_tool_loop(gate, reg, model, messages, max_rounds=max_rounds)

    assert exc_info.value.rounds == max_rounds
    # Exact model-call count: max_rounds + 1.
    assert model.call_count == max_rounds + 1, (
        f"expected {max_rounds + 1} model calls, got {model.call_count}"
    )


# ---------------------------------------------------------------------------
# R12 — run_tool_loop terminates early when model returns no tool_calls
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_tool_loop_terminates_early() -> None:
    gate, reg = _make_gate_and_registry()

    # First call: tool_calls.  Second call: no tool_calls → done.
    responses = [
        ModelResponse(
            text=None,
            tool_calls=(ToolCall(id="tc1", name="echo", arguments={"x": "v"}),),
            model_id="fake",
            finish_reason="tool_calls",
        ),
        ModelResponse(
            text="final answer",
            tool_calls=(),
            model_id="fake",
            finish_reason="stop",
        ),
    ]
    model = ReplayModel(responses, capabilities=ModelCapabilities(tools=True))

    messages = [ChatMessage(role="user", content="go")]
    result = await run_tool_loop(gate, reg, model, messages, max_rounds=4)

    assert result.text == "final answer"
    assert result.tool_calls == ()
    assert model.call_count == 2, f"expected 2 model calls, got {model.call_count}"


# ---------------------------------------------------------------------------
# R13 — taint latched after successful external dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_taint_latched_after_successful_external() -> None:
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()
    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_external=True, external_tags=()
    )
    assert not taint.tainted
    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert results[0].status == "ok"
    assert taint.tainted, "taint must be latched after successful external dispatch"


@pytest.mark.asyncio
async def test_taint_not_latched_when_trusted_output() -> None:
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()
    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_external=True, external_tags=("trusted-output",)
    )
    await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert not taint.tainted, "trusted-output tag must prevent taint"


# ---------------------------------------------------------------------------
# R14 — taint NOT latched on failed dispatch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_taint_not_latched_on_tier_refused() -> None:
    taint = TaintState()
    gate, reg = _make_gate_and_registry(taint=taint, include_external=True)
    # Default policy: external refused.
    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert results[0].status == "refused_tier"
    assert not taint.tainted, "taint must not latch when dispatch was refused"


@pytest.mark.asyncio
async def test_taint_not_latched_on_invalid_args() -> None:
    taint = TaintState()
    gate, reg = _make_gate_and_registry(taint=taint)
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {})])
    assert results[0].status == "invalid_args"
    assert not taint.tainted, "taint must not latch when arg validation fails"


@pytest.mark.asyncio
async def test_taint_not_latched_on_tool_error() -> None:
    taint = TaintState()
    gate, reg = _make_gate_and_registry(taint=taint, include_always_raise=True)
    results = await route_tool_calls(gate, reg, [_tool_call("always_raise", {"x": "v"})])
    assert results[0].status == "error"
    assert not taint.tainted, "taint must not latch when tool raises (invoke failed)"


# ---------------------------------------------------------------------------
# R15 — batch dispatch: results parallel to calls, correct order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_route_batch_parallel_results() -> None:
    gate, reg = _make_gate_and_registry(include_always_raise=True)
    calls = [
        _tool_call("echo", {"x": "first"}, id_="t1"),
        _tool_call("always_raise", {"x": "v"}, id_="t2"),
        _tool_call("echo", {"x": "third"}, id_="t3"),
    ]
    results = await route_tool_calls(gate, reg, calls)
    assert len(results) == 3, f"expected 3 results, got {len(results)}"
    assert results[0].tool_call_id == "t1"
    assert results[0].status == "ok"
    assert results[1].tool_call_id == "t2"
    assert results[1].status == "error"
    assert results[2].tool_call_id == "t3"
    assert results[2].status == "ok"
    # echo called twice (t1 + t3), always_raise called once (t2).
    assert _invoke_counter.get("echo") == 2, (
        f"echo must be invoked 2 times, got {_invoke_counter.get('echo')}"
    )
    assert _invoke_counter.get("always_raise") == 1


# ---------------------------------------------------------------------------
# Additional: ToolResult is frozen (immutability contract)
# ---------------------------------------------------------------------------


def test_tool_result_frozen() -> None:
    r = ToolResult(
        tool_call_id="id",
        name="echo",
        status="ok",
        content='"v"',
        error=None,
        latency_ms=1.0,
    )
    with pytest.raises(ValidationError):
        r.status = "error"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# B3 — latch BEFORE invoke: raise-mid-invoke still taints; refused dispatch does NOT
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_b3_taint_latched_even_when_invoke_raises() -> None:
    """External cap invoke raises → status=error, but taint IS latched (latch is pre-invoke).

    Mutation killed: post-invoke taint update (taint would never latch on a failed invoke).
    """
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()
    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_external=True, external_tags=()
    )
    # Patch ext_clean to raise after being called.
    original_ext = reg.get("ext_clean")

    class _RaisingCap:
        name = "ext_clean"
        tier = "external"
        input_schema = original_ext.input_schema

        async def invoke(self, args: object) -> str:
            _invoke_counter["ext_clean"] = _invoke_counter.get("ext_clean", 0) + 1
            raise RuntimeError("invoke side-effect then raise")

    reg._capabilities["ext_clean"] = _RaisingCap()

    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert results[0].status == "error", f"expected error, got {results[0].status!r}"
    assert _invoke_counter.get("ext_clean") == 1, "invoke must have been called once"
    # Taint MUST be latched even though invoke raised — latch is pre-invoke (B3).
    assert taint.tainted is True, (
        "taint must be latched even when invoke raises (latch is pre-invoke, B3)"
    )


@pytest.mark.asyncio
async def test_b3_tier_refused_does_not_taint() -> None:
    """A tier-refused dispatch (step ①) must NOT latch taint (latch is after ①②)."""
    taint = TaintState()
    gate, reg = _make_gate_and_registry(taint=taint, include_external=True)
    # Default policy: external refused at step ①.
    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert results[0].status == "refused_tier"
    assert _invoke_counter == {}, "invoke must not be called on tier refusal"
    assert taint.tainted is False, "tier refusal must NOT latch taint (B3)"


@pytest.mark.asyncio
async def test_b3_schema_invalid_does_not_taint() -> None:
    """An invalid-args dispatch (step ②) must NOT latch taint (latch is after ①②)."""
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()
    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_external=True, external_tags=()
    )
    # ext_clean requires x: str; pass an int to fail schema validation at step ②.
    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": 99})])
    assert results[0].status == "invalid_args"
    assert _invoke_counter == {}, "invoke must not be called on schema failure"
    assert taint.tainted is False, "schema refusal must NOT latch taint (B3)"


@pytest.mark.asyncio
async def test_b3_timeout_taints_because_invoke_started() -> None:
    """A timed-out external cap DOES taint — the timeout fires during invoke, AFTER the latch."""
    policy = StageToolPolicy(
        allowed_tiers=frozenset({"read", "write", "external"}), tool_timeout_s=0.01
    )
    taint = TaintState()
    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_slow=False, include_external=True, external_tags=()
    )
    # Make ext_clean sleep so it times out.
    original_ext = reg.get("ext_clean")

    class _SlowExternalCap:
        name = "ext_clean"
        tier = "external"
        input_schema = original_ext.input_schema

        async def invoke(self, args: object) -> str:
            _invoke_counter["ext_clean"] = _invoke_counter.get("ext_clean", 0) + 1
            await asyncio.sleep(10)
            return "unreachable"

    reg._capabilities["ext_clean"] = _SlowExternalCap()

    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert results[0].status == "timeout"
    # Taint IS latched: the latch fires at step ③ (after ①②) BEFORE invoke at step ④.
    # The timeout then cancels the coroutine at step ④, but the latch already fired.
    assert taint.tainted is True, (
        "external cap timeout must still taint (latch fires pre-invoke, B3)"
    )


# ---------------------------------------------------------------------------
# B4 — persist_taint fail-closed: if the durable hook raises, cap.invoke is NEVER reached
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_taint_raises_invoke_never_called_route() -> None:
    """persist_taint() raising aborts dispatch — cap.invoke is never reached.

    Guards against a future refactor that moves persist_taint after invoke or swallows its
    exception.  The durable-taint write is the S6 fail-closed checkpoint: if the journal write
    fails, the tool must NOT execute (a tainted-but-unjournaled run could resume untainted and
    gain back the external tier — a security regression).

    Assertions:
      (a) cap.invoke was never called (invoke-spy count == 0).
      (b) route_tool_calls surfaces the hook exception as status="error".

    Tested via both route_tool_calls (batch router) and dispatch_one (shared chokepoint).
    """
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()

    async def _raising_persist() -> None:
        raise RuntimeError("journal unavailable — persist_taint failed")

    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_external=True, external_tags=()
    )
    # Inject the raising persist_taint hook directly onto the gate's internal slot.
    gate._persist_taint = _raising_persist  # type: ignore[method-assign]

    # --- route_tool_calls path ---
    results = await route_tool_calls(gate, reg, [_tool_call("ext_clean", {"x": "v"})])
    assert len(results) == 1
    # (b) exception must surface as "error" — route_tool_calls maps it via the generic handler.
    assert results[0].status == "error", (
        f"persist_taint raise must surface as 'error', got {results[0].status!r}"
    )
    # (a) invoke must never have been called.
    assert _invoke_counter.get("ext_clean", 0) == 0, (
        "cap.invoke must NOT be called when persist_taint raises (fail-closed, S6)"
    )


@pytest.mark.asyncio
async def test_persist_taint_raises_invoke_never_called_dispatch_one() -> None:
    """dispatch_one propagates the persist_taint exception without calling cap.invoke.

    Covers the shared chokepoint directly: a caller of dispatch_one (e.g. RunContext.dispatch)
    must also see the exception propagate rather than being silently swallowed.
    """
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()

    async def _raising_persist() -> None:
        raise RuntimeError("journal unavailable — persist_taint failed")

    gate, reg = _make_gate_and_registry(
        policy=policy, taint=taint, include_external=True, external_tags=()
    )
    gate._persist_taint = _raising_persist  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="persist_taint failed"):
        await dispatch_one(gate, reg, "ext_clean", {"x": "v"})

    # (a) invoke was never reached.
    assert _invoke_counter.get("ext_clean", 0) == 0, (
        "cap.invoke must NOT be called when persist_taint raises in dispatch_one (fail-closed)"
    )
