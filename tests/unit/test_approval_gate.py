"""Unit tests for the approval gate (CANON S9, S10) — Pod 3.5b.

Security invariants under test:
  SC-U1 — Approval truth table: ApprovalRequired raised iff tainted ∧ consequential ∧
           irreversible ∧ ¬approved (all 8 parameter combinations).
  SC-U2 — approved=True unblocks the (T, T, T) case.
  SC-U3 — A refusal does NOT toggle taint state.
  SC-U4 — A refused dispatch does NOT invoke the tool (invoke counter stays 0).
  SC-U5 — route_tool_calls and run_tool_loop have no ``approved`` parameter
           (model-driven paths cannot self-approve; the absence IS the structural guarantee).
  SC-U6 — A model-supplied ``{"approved": True}`` in args is rejected by schema
           (additionalProperties:false; jsonschema.ValidationError; invoke counter stays 0).

asyncio_mode = "auto" (pyproject.toml) — no per-test decorator needed.
No pytestmark = pytest.mark.spike.
No network, no real substrate.
"""

from __future__ import annotations

import inspect
import itertools
from typing import Any

import jsonschema
import pytest

from cogworx.capability.policy import (
    ApprovalRequired,
    StageToolPolicy,
    TaintState,
    ToolGate,
)
from cogworx.capability.registry import Registry, function_capability
from cogworx.capability.router import dispatch_one, route_tool_calls, run_tool_loop
from cogworx.model.base import ToolCall

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_EXTERNAL_POLICY = StageToolPolicy(
    allowed_tiers=frozenset({"external"}),
    taint_drops_external=False,  # keep external tier even when tainted
)


def _make_registry(*, has_both: bool, name: str = "act") -> tuple[Registry, list[int]]:
    """Build a Registry with one 'external' capability.

    ``has_both=True``  → tags ("consequential", "irreversible") attached.
    ``has_both=False`` → no tags (or only one of the two — never both).

    Returns the registry and a mutable counter list so callers can verify invoke happened.
    """
    counter: list[int] = [0]

    async def _fn() -> str:
        counter[0] += 1
        return "done"

    cap = function_capability(_fn, name=name, tier="external")
    reg = Registry()
    tags: tuple[str, ...] = ("consequential", "irreversible") if has_both else ()
    reg.register(cap, tags=tags)
    return reg, counter


def _make_gate(reg: Registry, *, tainted: bool) -> ToolGate:
    return ToolGate(
        reg,
        policy=_EXTERNAL_POLICY,
        taint=TaintState(tainted=tainted),
    )


# ---------------------------------------------------------------------------
# SC-U1 — Truth table (8 parametrized cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "tainted,has_both,approved",
    list(itertools.product([False, True], [False, True], [False, True])),
)
async def test_truth_table_approval_required(tainted: bool, has_both: bool, approved: bool) -> None:
    """ApprovalRequired is raised iff (tainted=True ∧ has_both=True ∧ approved=False)."""
    reg, _counter = _make_registry(has_both=has_both)
    gate = _make_gate(reg, tainted=tainted)

    should_raise = tainted and has_both and (not approved)

    if should_raise:
        with pytest.raises(ApprovalRequired):
            await dispatch_one(gate, reg, "act", {}, approved=approved)
    else:
        result = await dispatch_one(gate, reg, "act", {}, approved=approved)
        assert result == "done"


# ---------------------------------------------------------------------------
# SC-U2 — approved=True unblocks the (T, T, T) case
# ---------------------------------------------------------------------------


async def test_approved_true_unblocks_ttt() -> None:
    """tainted=True, both tags present, approved=True → dispatch succeeds and returns 'done'."""
    reg, counter = _make_registry(has_both=True)
    gate = _make_gate(reg, tainted=True)

    result = await dispatch_one(gate, reg, "act", {}, approved=True)

    assert result == "done"
    assert counter[0] == 1


# ---------------------------------------------------------------------------
# SC-U3 — Refusal does NOT change taint state
# ---------------------------------------------------------------------------


async def test_refusal_does_not_change_taint() -> None:
    """A refused dispatch (ApprovalRequired) must not alter the taint latch."""
    reg, _counter = _make_registry(has_both=True)
    gate = _make_gate(reg, tainted=True)

    # Confirm pre-condition: already tainted.
    assert gate.taint.tainted is True

    with pytest.raises(ApprovalRequired):
        await dispatch_one(gate, reg, "act", {}, approved=False)

    # Taint state must be unchanged after the refusal.
    assert gate.taint.tainted is True


async def test_refusal_does_not_clear_taint_when_untainted() -> None:
    """A capability without both tags (no refusal) on an untainted gate should taint it."""
    # This is the positive-path check: external cap without trusted-output DOES taint.
    reg, _counter = _make_registry(has_both=False)
    gate = _make_gate(reg, tainted=False)

    await dispatch_one(gate, reg, "act", {}, approved=False)

    # External capability without 'trusted-output' tag should have latched taint.
    assert gate.taint.tainted is True


# ---------------------------------------------------------------------------
# SC-U4 — Refusal does NOT invoke the tool
# ---------------------------------------------------------------------------


async def test_refusal_does_not_invoke() -> None:
    """route_tool_calls with ApprovalRequired must re-raise before invoking the tool."""
    counter: list[int] = [0]

    async def _counting_act() -> str:
        counter[0] += 1
        return "done"

    cap = function_capability(_counting_act, name="act", tier="external")
    reg = Registry()
    reg.register(cap, tags=("consequential", "irreversible"))
    gate = _make_gate(reg, tainted=True)

    with pytest.raises(ApprovalRequired):
        await route_tool_calls(gate, reg, [ToolCall(id="t1", name="act", arguments={})])

    assert counter[0] == 0


# ---------------------------------------------------------------------------
# SC-U5 — No `approved` parameter on route_tool_calls or run_tool_loop
# ---------------------------------------------------------------------------


def test_no_self_approve_signature() -> None:
    """Model-driven paths cannot carry 'approved' — the absence IS the structural guarantee."""
    sig_rtc = inspect.signature(route_tool_calls)
    sig_rtl = inspect.signature(run_tool_loop)

    assert "approved" not in sig_rtc.parameters, (
        "route_tool_calls must not expose an 'approved' parameter"
    )
    assert "approved" not in sig_rtl.parameters, (
        "run_tool_loop must not expose an 'approved' parameter"
    )


# ---------------------------------------------------------------------------
# SC-U6 — Model-supplied {"approved": True} in args rejected by schema
# ---------------------------------------------------------------------------


async def test_model_supplied_approved_arg_rejected_by_schema() -> None:
    """additionalProperties:false blocks {"approved": True} injected by the model.

    A no-param function's hardened schema has no 'approved' property and
    additionalProperties:false, so the extra key fails jsonschema validation
    BEFORE invoke is called (S9: structure over prompting, S10: model cannot self-approve).
    """
    counter: list[int] = [0]

    async def _act() -> str:
        counter[0] += 1
        return "done"

    cap = function_capability(_act, name="act", tier="external")

    # Verify the schema is actually hardened before proceeding.
    schema: dict[str, Any] = dict(cap.input_schema)
    assert schema.get("additionalProperties") is False, (
        "function_capability must produce a schema with additionalProperties:false "
        f"for a no-param function; got: {schema}"
    )

    reg = Registry()
    reg.register(cap, tags=("consequential", "irreversible"))
    # Use tainted=True so the approval check would be reached — but jsonschema validation
    # fires BEFORE check_approval (pipeline step ② then ②.5), so the error should be
    # ValidationError regardless of approval state.
    gate = _make_gate(reg, tainted=True)

    with pytest.raises(jsonschema.ValidationError):
        await dispatch_one(gate, reg, "act", {"approved": True}, approved=False)

    assert counter[0] == 0
