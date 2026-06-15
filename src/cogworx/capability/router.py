"""Tool-call routing, dispatch, and the agentic tool loop (CANON S9, S10, S11) — Pod 3.2b.

All validation and routing is deterministic CODE — no model calls inside this module except inside
``run_tool_loop``'s single ``await model.complete(...)`` call (S1).  The model NEVER repairs its
own bad arguments (S9 — structure over prompting).

Design note — ctx.dispatch seam
--------------------------------
``route_tool_calls`` currently takes ``(gate, registry, model)`` directly rather than a full
``RunContext`` so it can be called and unit-tested without importing the runtime package (avoiding
the import cycle that the Pod 2.5/2.6 red-team caught, CF-1).  Pod 3.2c will wire this through
``ctx.dispatch`` by passing ``ctx._registry`` and the gate it builds from the engine's registry —
the exact seam is: ``route_tool_calls(gate, registry, model, tool_calls, *, policy_timeout_s)``.
3.2c needs to pass ``registry=ctx._registry`` (or expose a public accessor) and build / store the
drive-level ``ToolGate`` on the ``RunContext`` or engine, then call ``route_tool_calls`` from
there.  No API changes are needed here; 3.2c just adds the wiring.

Contract changelog:
  - 2026-06-12 (Pod 3.2b): initial — ToolResult, ToolLoopLimit, route_tool_calls,
    tool_result_messages, run_tool_loop.  New module; no existing callers.
  - 2026-06-12 (Pod 3.5b): dispatch_one gains approved: bool = False + step ②.5 (check_approval
    AFTER arg validation, BEFORE taint latch); route_tool_calls re-raises ApprovalRequired rather
    than converting it to a ToolResult (model must not negotiate with the approval gate, S9/S10).
    Additive (defaulted param; no ToolStatus change; existing callers unaffected).
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Sequence
from typing import Any, Literal

import jsonschema
from pydantic import BaseModel, ConfigDict

from cogworx.capability.base import CapabilityUnavailable
from cogworx.capability.policy import ApprovalRequired, TierViolation, ToolGate
from cogworx.capability.registry import Registry
from cogworx.model.base import ChatMessage, Model, ModelResponse, ToolCall


class ToolLoopLimit(Exception):
    """The model continued requesting tool calls beyond the configured ``max_rounds`` ceiling.

    This is a structural ceiling the model cannot extend (S11 spirit).  The caller should route
    this as a degraded result (S8) — the partial tool-call transcript is available on the
    exception for diagnostic / HITL purposes.

    Attributes
    ----------
    rounds:
        Number of rounds executed before the ceiling was hit.
    last_response:
        The ``ModelResponse`` that triggered the limit (still contained ``tool_calls``).
    """

    def __init__(self, rounds: int, last_response: ModelResponse) -> None:
        super().__init__(
            f"Tool loop exceeded {rounds}-round ceiling; "
            f"last response still had {len(last_response.tool_calls)} tool call(s)."
        )
        self.rounds = rounds
        self.last_response = last_response


ToolStatus = Literal[
    "ok",
    "invalid_args",
    "unknown_tool",
    "unavailable",
    "refused_tier",
    "error",
    "timeout",
]


class ToolResult(BaseModel):
    """The outcome of one dispatched tool call (frozen, S9 structural record).

    ``content`` holds the JSON-serialised result on success, or a structured error description
    on failure — always a string so it can be fed back to the model as a ``tool`` message without
    additional serialisation.  ``error`` carries the raw exception message for logging; it is
    ``None`` on ``"ok"``.

    Attributes
    ----------
    tool_call_id:
        Echoes the model's ``ToolCall.id`` so the model can correlate results.
    name:
        The tool name that was requested.
    status:
        One of ``"ok" | "invalid_args" | "unknown_tool" | "unavailable" | "refused_tier" |
        "error" | "timeout"``.
    content:
        JSON-serialised result (on ``"ok"``) or a structured error string (on failure).
        Always non-empty.
    error:
        Raw exception message, or ``None`` on success.
    latency_ms:
        Wall-clock milliseconds from gate-entry to result, including all validation overhead.
    """

    model_config = ConfigDict(frozen=True)

    tool_call_id: str
    name: str
    status: ToolStatus
    content: str
    error: str | None
    latency_ms: float


def _serialize(value: Any) -> str:
    """JSON-serialise a tool return value (tess ``invoke_and_serialize`` pattern).

    Falls back to ``str()`` for non-JSON-serialisable values so the router never raises on an
    unexpected return type — the round-trip fidelity is the tool author's responsibility.
    """
    try:
        return json.dumps(value)
    except (TypeError, ValueError):
        return json.dumps(str(value))


async def dispatch_one(
    gate: ToolGate,
    registry: Registry,
    name: str,
    args: Any,
    *,
    approved: bool = False,
    timeout_s: float | None = None,
) -> Any:
    """Single-capability dispatch through the security gate (the shared chokepoint).

    This is the common implementation used by both ``route_tool_calls`` (batch router) and
    ``RunContext.dispatch`` (single-call path).  Factoring it here keeps the tier-check,
    arg-validation, invoke, and taint-update logic in one place so both callers can NEVER
    diverge on security semantics (SC-6 coherence requirement, mythos Decision 4 #3).

    The ``runtime`` package is allowed to import ``capability`` (runtime → capability is a valid
    directed edge); ``capability`` must never import ``runtime``.  ``dispatch_one`` lives here in
    ``capability.router`` so ``RunContext.dispatch`` can call it without creating a new import edge.

    Pipeline (mirrors ``route_tool_calls`` per-call pipeline):
      ① ``gate.check_dispatch(name)`` — tier check (raises ``TierViolation`` or
        ``CapabilityUnavailable``).
      ② ``jsonschema.validate(args, cap.input_schema)`` — framework-side arg validation (S9;
        raises ``jsonschema.ValidationError``).
      ③ ``gate.taint.update(name, registry)`` — taint latch BEFORE invoke (structural, S9;
        a dispatch refused at ①② must NOT taint; security state must not depend on tool
        outcome — S9).  On the False→True transition, ``await gate.persist_taint()`` writes
        the durable journal bit BEFORE ``cap.invoke`` (S6 fail-closed: if the journal write
        raises, the tool is NOT invoked).
      ④ ``await asyncio.wait_for(cap.invoke(args), timeout=...)`` — tool invocation (raises
        ``TimeoutError`` or a tool-defined exception).

    Parameters
    ----------
    gate:
        The bound ``ToolGate`` for the current stage/drive.
    registry:
        The ``Registry`` from which capabilities are resolved.
    name:
        The capability name to dispatch.
    args:
        The arguments mapping to pass to ``cap.invoke``.
    timeout_s:
        Per-call timeout in seconds.  ``None`` means use ``gate.policy.tool_timeout_s``.

    Returns
    -------
    The raw return value from ``cap.invoke``.

    Raises
    ------
    ``TierViolation``
        Step ①: name is known but its tier is not in the effective set.
    ``CapabilityUnavailable``
        Step ①: name is unknown or disabled/lesioned.
    ``jsonschema.ValidationError``
        Step ②: args fail framework-side schema validation.
    ``TimeoutError``
        Step ③: invoke exceeded the timeout.
    Any exception from ``cap.invoke``
        Step ③: the tool raised an error.
    """
    effective_timeout = timeout_s if timeout_s is not None else gate.policy.tool_timeout_s

    # ① Tier gate (re-validates INDEPENDENTLY of exposure).
    gate.check_dispatch(name)

    # Safe to fetch after check_dispatch passed.
    cap = registry.get(name)

    # ② Framework-side argument validation (S9 — structure over prompting).
    jsonschema.validate(
        instance=args,
        schema=dict(cap.input_schema),
        cls=jsonschema.Draft202012Validator,
    )

    # ②.5 S10 approval gate: tainted ∧ consequential ∧ irreversible ∧ ¬approved → ApprovalRequired.
    # AFTER arg validation: malformed args get structural "invalid_args" feedback, never HITL
    # escalation (S9). BEFORE the taint latch: check_approval reads taint AS OF DISPATCH ENTRY —
    # taint introduced by THIS call cannot have influenced the model's decision to make it.
    # A refused dispatch (ApprovalRequired) must NOT taint and must NOT invoke.
    gate.check_approval(name, approved=approved)

    # ③ Taint latch BEFORE invoke (S9: security state must not depend on tool outcome).
    # A dispatch refused at ① or ② must NOT taint — latching here (after both checks)
    # ensures only a dispatch that would actually reach the tool taints the drive.
    # On the False→True transition: await the durable journal write BEFORE invoke (S6
    # fail-closed: if the journal write raises, the tool is NOT called and the exception
    # propagates to the generic handler).
    was_tainted = gate.taint.tainted
    gate.taint.update(name, registry)
    if not was_tainted and gate.taint.tainted:
        # False→True transition: persist durably before invoking (S6 ordering).
        await gate.persist_taint()

    # ④ Invoke with hard timeout (S11 structural ceiling).
    raw = await asyncio.wait_for(cap.invoke(args), timeout=effective_timeout)

    return raw


async def route_tool_calls(
    gate: ToolGate,
    registry: Registry,
    tool_calls: Sequence[ToolCall],
) -> tuple[ToolResult, ...]:
    """Dispatch a batch of model-requested tool calls through the security gate.

    Per call the dispatch pipeline is:

      ① ``gate.check_dispatch(name)`` — tier check (LOUD ``TierViolation`` on failure).
      ② ``jsonschema.validate(args, cap.input_schema)`` — framework-side arg validation (S9).
      ③ ``gate.taint.update(name, registry)`` — taint latch BEFORE invoke (S9: security state
         must not depend on tool outcome; a refusal at ①② must NOT taint).  On the False→True
         transition, ``await gate.persist_taint()`` durably journals the bit (S6 fail-closed:
         if the journal write raises, the tool is NOT called).
      ④ ``await cap.invoke(args)`` with ``asyncio.wait_for(timeout=policy.tool_timeout_s)``.

    ``cap.invoke`` is NEVER called if ① or ② fail.  Each failure maps to a distinct ``status``
    in the taxonomy so the caller (and the model via ``tool_result_messages``) gets structured
    feedback, not a raw exception.

    Parameters
    ----------
    gate:
        The bound ``ToolGate`` for the current stage/drive.
    registry:
        The ``Registry`` from which capabilities are resolved.
    tool_calls:
        The ``ToolCall`` sequence from the ``ModelResponse``.

    Returns
    -------
    A ``tuple[ToolResult, ...]`` parallel to ``tool_calls`` — one result per call, in order.
    """
    results: list[ToolResult] = []

    for tc in tool_calls:
        t0 = time.monotonic()

        try:
            raw = await dispatch_one(gate, registry, tc.name, tc.arguments)
        except TierViolation as exc:
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    status="refused_tier",
                    content=json.dumps({"error": "tier_refused", "detail": str(exc)}),
                    error=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            )
            continue
        except CapabilityUnavailable as exc:
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    status="unavailable",
                    content=json.dumps({"error": "unavailable", "detail": str(exc)}),
                    error=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            )
            continue
        except jsonschema.ValidationError as exc:
            validation_detail = exc.message
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    status="invalid_args",
                    content=json.dumps({"error": "invalid_args", "detail": validation_detail}),
                    error=validation_detail,
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            )
            continue
        except TimeoutError:
            timeout_s = gate.policy.tool_timeout_s
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    status="timeout",
                    content=json.dumps(
                        {"error": "timeout", "detail": f"exceeded {timeout_s}s limit"}
                    ),
                    error=f"asyncio.TimeoutError after {timeout_s}s",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            )
            continue
        except ApprovalRequired:
            # S9/S10: ApprovalRequired is NOT converted to a ToolResult. Re-raise so the stage
            # (or run_tool_loop caller) can route it as an AwaitHuman transition. The model must
            # not receive "approval_required" feedback it could negotiate with.
            raise
        except Exception as exc:
            results.append(
                ToolResult(
                    tool_call_id=tc.id,
                    name=tc.name,
                    status="error",
                    content=json.dumps(
                        {"error": "tool_error", "detail": str(exc), "type": type(exc).__name__}
                    ),
                    error=str(exc),
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            )
            continue

        results.append(
            ToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                status="ok",
                content=_serialize(raw),
                error=None,
                latency_ms=(time.monotonic() - t0) * 1000,
            )
        )

    return tuple(results)


def tool_result_messages(results: Sequence[ToolResult]) -> tuple[ChatMessage, ...]:
    """Convert ``ToolResult``s to ``ChatMessage(role="tool", ...)`` for the next model call.

    Each result — success or failure — is fed back as a structured ``tool`` role message so the
    model receives machine-readable, schema-consistent feedback regardless of outcome (S9 pattern:
    structure over narrative error messages).
    """
    return tuple(
        ChatMessage(role="tool", tool_call_id=r.tool_call_id, content=r.content) for r in results
    )


async def run_tool_loop(
    gate: ToolGate,
    registry: Registry,
    model: Model,
    initial_messages: Sequence[ChatMessage],
    *,
    max_rounds: int = 4,
) -> ModelResponse:
    """Drive the agentic tool loop until the model stops requesting tools or ``max_rounds`` is hit.

    Loop:
      1. Assemble ``exposed_specs()`` from the gate (tier-filtered; the model sees only
         what it is allowed to call).
      2. ``await model.complete(messages=..., tools=exposed_specs)`` — the ONLY model call.
      3. If no ``tool_calls``, return the response immediately.
      4. Route the tool calls via ``route_tool_calls``.
      5. Append the assistant response + tool-result messages to the conversation.
      6. Repeat from step 2, up to ``max_rounds``.

    If the model is still requesting tools at ``max_rounds``, raises ``ToolLoopLimit`` (S11 —
    structural ceiling the model cannot self-extend).  Model-call cost is bounded upstream by the
    ``BudgetGuardedModel`` wrapper; no additional budget machinery is added here.

    Parameters
    ----------
    gate:
        The bound ``ToolGate`` for the current drive stage.
    registry:
        The capability registry.
    model:
        The ``Model`` to call (should already be wrapped in ``BudgetGuardedModel``).
    initial_messages:
        The conversation so far (system + user + prior turns).
    max_rounds:
        Hard ceiling on tool-call rounds.  Default 4.

    Returns
    -------
    The final ``ModelResponse`` once the model stops requesting tools.
    """
    messages: list[ChatMessage] = list(initial_messages)

    for _round in range(max_rounds):
        specs = gate.exposed_specs()
        response = await model.complete(messages=messages, tools=specs)

        if not response.tool_calls:
            return response

        # Append the assistant's turn (with tool_calls) to the transcript.
        assistant_text = response.text or ""
        messages.append(ChatMessage(role="assistant", content=assistant_text))

        # Route tool calls and feed results back.
        results = await route_tool_calls(gate, registry, response.tool_calls)
        messages.extend(tool_result_messages(results))

    # One final model call after the last round of tool feedback.
    specs = gate.exposed_specs()
    response = await model.complete(messages=messages, tools=specs)
    if response.tool_calls:
        raise ToolLoopLimit(max_rounds, response)
    return response


__all__ = [
    "ToolLoopLimit",
    "ToolResult",
    "ToolStatus",
    "dispatch_one",
    "route_tool_calls",
    "run_tool_loop",
    "tool_result_messages",
]
