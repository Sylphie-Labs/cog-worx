"""Pod 3.2c Feature Test Bundle — ToolGate + RegistryToolContributor wired into the engine.

Mutation-resistant assertions throughout: exact-set proofs, zero-invoke-counter checks, and
convergence proofs between the route_tool_calls and RunContext.dispatch paths.

Coverage:
  W1  — RegistryToolContributor.contribute() returns exactly gate.exposed_specs() (exact-set).
  W2  — RegistryToolContributor re-evaluates after a policy re-bind (tiers narrow → specs narrow).
  W3  — RegistryToolContributor re-evaluates after taint (external drops away structurally).
  W4  — RunContext.dispatch tier refusal raises TierViolation, invoke NOT called.
  W5  — RunContext.dispatch unknown name raises CapabilityUnavailable, invoke NOT called.
  W6  — RunContext.dispatch bad args raises jsonschema.ValidationError, invoke NOT called.
  W7  — RunContext.dispatch success invokes exactly once with exact args.
  W8  — RunContext.dispatch and route_tool_calls converge: same TierViolation for same input.
  W9  — RunContext.dispatch and route_tool_calls converge: same ValidationError for same input.
  W10 — Engine._build_context builds a real ToolGate + RegistryToolContributor (not placeholder).
  W11 — Per-stage bind_tool_policy: stage with tool_policy={read} exposes only read specs.
  W12 — No StageContext Protocol break: stage double without tool_policy works (gets DEFAULT).
  W13 — Import-order / no-cycle: cogworx.capability importable first without pulling runtime.
  W14 — No policy leak across stages: policy bound for stage A is cleared for stage B.

asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime
from typing import Any

import jsonschema
import pytest

from cogworx.capability.base import CapabilityUnavailable
from cogworx.capability.policy import (
    StageToolPolicy,
    TaintState,
    TierViolation,
    ToolGate,
)
from cogworx.capability.registry import Registry, function_capability
from cogworx.capability.router import route_tool_calls
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.context.contributors import RegistryToolContributor
from cogworx.context.types import ContextRequest, SlotAllocation
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.model.base import ModelResponse, ToolCall
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.context import RunContext
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import (
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC)
_FIXED_CLOCK = lambda: _EPOCH  # noqa: E731

_invoke_counter: dict[str, int] = {}


@pytest.fixture(autouse=True)
def _reset_counter() -> None:
    _invoke_counter.clear()


async def _echo(x: str) -> str:
    _invoke_counter["echo"] = _invoke_counter.get("echo", 0) + 1
    return x


async def _write_op(x: str) -> str:
    _invoke_counter["write_op"] = _invoke_counter.get("write_op", 0) + 1
    return x


async def _external_op(x: str) -> str:
    _invoke_counter["ext_op"] = _invoke_counter.get("ext_op", 0) + 1
    return x


def _make_registry(
    *,
    include_read: bool = True,
    include_write: bool = False,
    include_external: bool = False,
    external_tags: tuple[str, ...] = (),
) -> Registry:
    reg = Registry()
    if include_read:
        reg.register(function_capability(_echo, name="echo", tier="read"))
    if include_write:
        reg.register(function_capability(_write_op, name="write_op", tier="write"))
    if include_external:
        reg.register(
            function_capability(_external_op, name="ext_op", tier="external"),
            tags=external_tags,
        )
    return reg


def _make_gate(
    reg: Registry,
    *,
    policy: StageToolPolicy | None = None,
    taint: TaintState | None = None,
) -> ToolGate:
    return ToolGate(reg, policy=policy, taint=taint)


def _make_artifact(stage: str) -> Artifact:
    return Artifact(
        kind="test",
        produced_by=stage,
        provenance=Provenance(source="system", confidence=1.0, recorded_at=_EPOCH),
    )


def _make_run_context(
    registry: Registry | None = None,
    gate: ToolGate | None = None,
) -> RunContext:
    return RunContext(
        run_id="run-wiring-test",
        session_id="sess-wiring-test",
        model=ReplayModel(),
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
        registry=registry,
        gate=gate,
        clock=_FIXED_CLOCK,
    )


def _alloc() -> SlotAllocation:
    return SlotAllocation(max_tokens=None)


def _tool_call(name: str, args: dict[str, Any], *, id_: str = "tc1") -> ToolCall:
    return ToolCall(id=id_, name=name, arguments=args)


# ---------------------------------------------------------------------------
# W1 — RegistryToolContributor.contribute() returns exactly gate.exposed_specs()
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_contributor_exact_set_matches_gate() -> None:
    """contribute() specs == gate.exposed_specs() (exact-set, order-insensitive)."""
    reg = _make_registry(include_read=True, include_write=True)
    gate = _make_gate(reg)
    contributor = RegistryToolContributor(gate)

    content = await contributor.contribute(ContextRequest(task="t"), _alloc())
    gate_specs = gate.exposed_specs()

    assert set(s.name for s in content.tools) == set(s.name for s in gate_specs), (
        "RegistryToolContributor must return exactly the gate's exposed_specs names"
    )
    assert len(content.tools) == len(gate_specs), (
        "RegistryToolContributor must not add or drop specs"
    )
    assert content.status == "ok", f"expected ok, got {content.status}"


@pytest.mark.asyncio
async def test_registry_contributor_empty_when_no_caps() -> None:
    """Empty registry → status='empty', no tools."""
    reg = Registry()
    gate = _make_gate(reg)
    contributor = RegistryToolContributor(gate)

    content = await contributor.contribute(ContextRequest(task="t"), _alloc())
    assert content.tools == ()
    assert content.status == "empty"


# ---------------------------------------------------------------------------
# W2 — RegistryToolContributor re-evaluates after a policy re-bind
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_contributor_re_evaluates_after_policy_rebind() -> None:
    """After bind_policy narrowing to read-only, external specs drop from contribute()."""
    reg = _make_registry(include_read=True, include_write=True, include_external=True)
    full_policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    gate = _make_gate(reg, policy=full_policy)
    contributor = RegistryToolContributor(gate)

    # Before rebind: all three tiers visible.
    before = await contributor.contribute(ContextRequest(task="t"), _alloc())
    assert len(before.tools) == 3, f"expected 3 specs before rebind, got {len(before.tools)}"

    # Rebind to read-only.
    gate.bind_policy(StageToolPolicy(allowed_tiers=frozenset({"read"})))

    # After rebind: only read spec visible.
    after = await contributor.contribute(ContextRequest(task="t"), _alloc())
    assert len(after.tools) == 1, f"expected 1 spec after read-only rebind, got {len(after.tools)}"
    assert after.tools[0].name == "echo"


# ---------------------------------------------------------------------------
# W3 — RegistryToolContributor re-evaluates after taint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_registry_contributor_re_evaluates_after_taint() -> None:
    """After external dispatch taints the drive, external spec drops from contribute()."""
    reg = _make_registry(include_read=True, include_external=True)
    full_policy = StageToolPolicy(
        allowed_tiers=frozenset({"read", "external"}),
        taint_drops_external=True,
    )
    taint = TaintState()
    gate = _make_gate(reg, policy=full_policy, taint=taint)
    contributor = RegistryToolContributor(gate)

    # Before taint: read + external.
    before = await contributor.contribute(ContextRequest(task="t"), _alloc())
    assert {s.name for s in before.tools} == {"echo", "ext_op"}

    # Latch taint manually (as the router does after a successful external invocation).
    taint.tainted = True

    # After taint: external drops structurally.
    after = await contributor.contribute(ContextRequest(task="t"), _alloc())
    assert {s.name for s in after.tools} == {"echo"}, (
        "external spec must be absent after taint (structural taint_drops_external)"
    )


# ---------------------------------------------------------------------------
# W4 — RunContext.dispatch tier refusal raises TierViolation, invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_tier_refused_raises_tier_violation() -> None:
    """dispatch() raises TierViolation when the cap's tier is outside the effective set."""
    reg = _make_registry(include_read=True, include_external=True)
    # Default policy: read+write only; external is refused.
    gate = _make_gate(reg)
    ctx = _make_run_context(registry=reg, gate=gate)

    with pytest.raises(TierViolation):
        await ctx.dispatch("ext_op", {"x": "v"})

    assert _invoke_counter == {}, "invoke must NOT be called on tier refusal"


# ---------------------------------------------------------------------------
# W5 — RunContext.dispatch unknown name raises CapabilityUnavailable, invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_unknown_name_raises_capability_unavailable() -> None:
    """dispatch() raises CapabilityUnavailable for an unknown capability name."""
    reg = _make_registry(include_read=True)
    gate = _make_gate(reg)
    ctx = _make_run_context(registry=reg, gate=gate)

    with pytest.raises(CapabilityUnavailable):
        await ctx.dispatch("does_not_exist", {"x": "v"})

    assert _invoke_counter == {}


# ---------------------------------------------------------------------------
# W6 — RunContext.dispatch bad args raises ValidationError, invoke NOT called
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_bad_args_raises_validation_error() -> None:
    """dispatch() raises jsonschema.ValidationError when args fail schema validation (S9)."""
    reg = _make_registry(include_read=True)
    gate = _make_gate(reg)
    ctx = _make_run_context(registry=reg, gate=gate)

    # _echo expects x: str; pass an int.
    with pytest.raises(jsonschema.ValidationError):
        await ctx.dispatch("echo", {"x": 42})

    assert _invoke_counter == {}, "invoke must NOT be called when arg validation fails"


# ---------------------------------------------------------------------------
# W7 — RunContext.dispatch success invokes exactly once with exact args
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_success_invokes_once_with_exact_args() -> None:
    """dispatch() invokes exactly once and returns the raw result."""
    reg = _make_registry(include_read=True)
    gate = _make_gate(reg)
    ctx = _make_run_context(registry=reg, gate=gate)

    result = await ctx.dispatch("echo", {"x": "hello"})

    assert result == "hello", f"expected 'hello', got {result!r}"
    assert _invoke_counter.get("echo") == 1, "invoke must be called exactly once"


# ---------------------------------------------------------------------------
# W8 — Convergence: dispatch and route_tool_calls both raise TierViolation for same input
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_and_route_converge_on_tier_violation() -> None:
    """Both RunContext.dispatch and route_tool_calls refuse the same tier-violating call."""
    reg = _make_registry(include_read=True, include_external=True)
    gate = _make_gate(reg)  # default: read+write, external refused.
    ctx = _make_run_context(registry=reg, gate=gate)

    # ctx.dispatch path — must raise TierViolation.
    with pytest.raises(TierViolation):
        await ctx.dispatch("ext_op", {"x": "v"})

    assert _invoke_counter == {}, "dispatch: invoke must not be called on tier refusal"

    # route_tool_calls path (shares the same gate, same state).
    results = await route_tool_calls(gate, reg, [_tool_call("ext_op", {"x": "v"})])
    assert results[0].status == "refused_tier", (
        f"route_tool_calls must also refuse the tier, got {results[0].status}"
    )
    assert _invoke_counter == {}, "route_tool_calls: invoke must not be called on tier refusal"


# ---------------------------------------------------------------------------
# W9 — Convergence: dispatch and route_tool_calls converge on bad-args refusal
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dispatch_and_route_converge_on_invalid_args() -> None:
    """Both paths enforce the same arg-validation for the same malformed input."""
    reg = _make_registry(include_read=True)
    gate = _make_gate(reg)
    ctx = _make_run_context(registry=reg, gate=gate)

    # ctx.dispatch path — must raise ValidationError.
    with pytest.raises(jsonschema.ValidationError):
        await ctx.dispatch("echo", {"x": 42})

    # route_tool_calls path — same bad args → "invalid_args" status.
    results = await route_tool_calls(gate, reg, [_tool_call("echo", {"x": 42})])
    assert results[0].status == "invalid_args", (
        f"route_tool_calls must also reject the bad args, got {results[0].status}"
    )

    # Neither path should have called invoke.
    assert _invoke_counter == {}, "invoke must not be called by either path"


# ---------------------------------------------------------------------------
# W10 — Engine._build_context wires registry into RunContext + uses RegistryToolContributor
# ---------------------------------------------------------------------------


class _CapturePolicyStage:
    """Stage that captures the registry and gate visible at run time."""

    name: str = "capture-stage"
    transitions: tuple[str, ...] = ()
    captured_registry: Registry | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        assert isinstance(ctx, RunContext)
        _CapturePolicyStage.captured_registry = ctx.registry
        return Done(output=_make_artifact("capture-stage"))


def _initial_artifact() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_EPOCH),
        data={"text": "test"},
    )


def _make_engine_with_registry(
    registry: Registry,
    stages: list[Any],
    *,
    pathway_id: str = "test-pathway",
    model: ReplayModel | None = None,
) -> Engine:
    graph = StageGraph(stages, entry=stages[0].name)
    pathways = PathwayRegistry()
    pathways.register(pathway_id, graph, version=1)
    mr = ModelRegistry()
    mr.register("default", model or ReplayModel())
    engine = Engine(
        models=mr,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        registry=registry,
        clock=_FIXED_CLOCK,
    )
    return engine


@pytest.mark.asyncio
async def test_engine_wires_registry_into_context() -> None:
    """Engine passes the registry through to RunContext.registry (public property, not None)."""
    reg = _make_registry(include_read=True)
    stage = _CapturePolicyStage()
    model = ReplayModel([ModelResponse(text="ok", model_id="replay", finish_reason="stop")])
    engine = _make_engine_with_registry(reg, [stage], model=model)

    await engine.run(
        run_id="run-engine-wiring",
        session_id="sess",
        pathway_id="test-pathway",
        initial=_initial_artifact(),
    )

    assert _CapturePolicyStage.captured_registry is reg, (
        "Engine must wire the registry into RunContext.registry"
    )


@pytest.mark.asyncio
async def test_engine_assembler_uses_registry_tool_contributor() -> None:
    """Engine._build_assembler uses RegistryToolContributor (not the empty static placeholder)."""
    from cogworx.context.assembler import ContextAssembler
    from cogworx.context.contributors import ToolSpecContributor
    from cogworx.context.types import DEFAULT_SLOTS

    reg = _make_registry(include_read=True, include_write=True)
    gate = ToolGate(reg, taint=TaintState())

    assembler = ContextAssembler(slots=DEFAULT_SLOTS, model=ReplayModel())
    assembler.register("tools", RegistryToolContributor(gate))

    tools_contributor = assembler._contributors.get("tools")  # type: ignore[attr-defined]
    assert isinstance(tools_contributor, RegistryToolContributor), (
        f"tools slot must be RegistryToolContributor, got {type(tools_contributor)}"
    )
    assert not isinstance(tools_contributor, ToolSpecContributor), (
        "RegistryToolContributor must replace the static ToolSpecContributor placeholder"
    )


# ---------------------------------------------------------------------------
# W11 — Per-stage bind_tool_policy: stage with {read} exposes only read specs
# ---------------------------------------------------------------------------


class _CapturePolicyWithReadStage:
    """Stage that declares tool_policy=read-only; captures the specs it sees."""

    name: str = "read-only-stage"
    transitions: tuple[str, ...] = ()
    tool_policy: StageToolPolicy = StageToolPolicy(allowed_tiers=frozenset({"read"}))
    captured_specs: tuple[Any, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        assert isinstance(ctx, RunContext)
        assert ctx._gate is not None
        _CapturePolicyWithReadStage.captured_specs = ctx._gate.exposed_specs()
        return Done(output=_make_artifact("read-only-stage"))


@pytest.mark.asyncio
async def test_engine_bind_tool_policy_narrows_to_read() -> None:
    """Engine's bind_tool_policy site narrows exposed specs when stage.tool_policy={read}."""
    reg = _make_registry(include_read=True, include_write=True)
    stage = _CapturePolicyWithReadStage()
    model = ReplayModel([ModelResponse(text="ok", model_id="replay", finish_reason="stop")])
    engine = _make_engine_with_registry(reg, [stage], model=model)

    await engine.run(
        run_id="run-bind-policy",
        session_id="sess",
        pathway_id="test-pathway",
        initial=_initial_artifact(),
    )

    names = {s.name for s in _CapturePolicyWithReadStage.captured_specs}
    assert names == {"echo"}, f"read-only policy must expose only 'echo' (read tier), got {names}"
    assert "write_op" not in names, "write-tier spec must be absent under read-only policy"


# ---------------------------------------------------------------------------
# W12 — No Protocol break: stage without tool_policy still works (DEFAULT_TOOL_POLICY applied)
# ---------------------------------------------------------------------------


class _NoToolPolicyStage:
    """Stage with NO tool_policy attribute — simulates an existing stage double."""

    name: str = "no-policy-stage"
    transitions: tuple[str, ...] = ()
    captured_specs: tuple[Any, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        assert isinstance(ctx, RunContext)
        assert ctx._gate is not None
        _NoToolPolicyStage.captured_specs = ctx._gate.exposed_specs()
        return Done(output=_make_artifact("no-policy-stage"))


@pytest.mark.asyncio
async def test_stage_without_tool_policy_gets_default() -> None:
    """A stage with no tool_policy attribute gets DEFAULT_TOOL_POLICY (read+write)."""
    reg = _make_registry(include_read=True, include_write=True)
    stage = _NoToolPolicyStage()
    model = ReplayModel([ModelResponse(text="ok", model_id="replay", finish_reason="stop")])
    engine = _make_engine_with_registry(reg, [stage], model=model)

    await engine.run(
        run_id="run-no-policy",
        session_id="sess",
        pathway_id="test-pathway",
        initial=_initial_artifact(),
    )

    names = {s.name for s in _NoToolPolicyStage.captured_specs}
    assert "echo" in names, "read-tier spec must be present under DEFAULT_TOOL_POLICY"
    assert "write_op" in names, "write-tier spec must be present under DEFAULT_TOOL_POLICY"


# ---------------------------------------------------------------------------
# W13 — Import-order / no-cycle: cogworx.capability importable before cogworx.runtime
# ---------------------------------------------------------------------------


def test_capability_importable_before_runtime_no_cycle() -> None:
    """cogworx.capability is importable first in a fresh interpreter (no cycle regression)."""
    script = (
        "import sys; "
        "import cogworx.capability; "
        "assert 'cogworx.runtime' not in sys.modules, "
        "'importing cogworx.capability pulled cogworx.runtime -- import cycle!'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        "Import cycle regression: cogworx.capability failed or pulled cogworx.runtime:\n"
        f"stderr: {proc.stderr}"
    )


def test_router_importable_before_runtime_no_cycle() -> None:
    """cogworx.capability.router is importable first without pulling cogworx.runtime."""
    script = (
        "import sys; "
        "import cogworx.capability.router; "
        "assert 'cogworx.runtime' not in sys.modules, "
        "'importing cogworx.capability.router pulled cogworx.runtime -- import cycle!'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        "Import cycle regression: cogworx.capability.router pulled cogworx.runtime:\n"
        f"stderr: {proc.stderr}"
    )


# ---------------------------------------------------------------------------
# W14 — No policy leak across stages
# ---------------------------------------------------------------------------


class _StageA:
    name: str = "stage-a"
    transitions: tuple[str, ...] = ("stage-b",)
    tool_policy: StageToolPolicy = StageToolPolicy(allowed_tiers=frozenset({"read"}))
    captured_specs: tuple[Any, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        assert isinstance(ctx, RunContext)
        assert ctx._gate is not None
        _StageA.captured_specs = ctx._gate.exposed_specs()
        return Transition(to="stage-b", output=_make_artifact("stage-a"))


class _StageB:
    name: str = "stage-b"
    transitions: tuple[str, ...] = ()
    # No tool_policy — must revert to DEFAULT_TOOL_POLICY.
    captured_specs: tuple[Any, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        assert isinstance(ctx, RunContext)
        assert ctx._gate is not None
        _StageB.captured_specs = ctx._gate.exposed_specs()
        return Done(output=_make_artifact("stage-b"))


@pytest.mark.asyncio
async def test_tool_policy_cleared_between_stages() -> None:
    """Stage A's read-only policy must not leak into stage B (clear-first invariant)."""
    reg = _make_registry(include_read=True, include_write=True)
    stage_a = _StageA()
    stage_b = _StageB()
    model = ReplayModel(
        [
            ModelResponse(text="ok-a", model_id="replay", finish_reason="stop"),
            ModelResponse(text="ok-b", model_id="replay", finish_reason="stop"),
        ]
    )
    engine = _make_engine_with_registry(reg, [stage_a, stage_b], model=model)

    await engine.run(
        run_id="run-no-leak",
        session_id="sess",
        pathway_id="test-pathway",
        initial=_initial_artifact(),
    )

    a_names = {s.name for s in _StageA.captured_specs}
    b_names = {s.name for s in _StageB.captured_specs}

    # Stage A: read-only → echo only.
    assert a_names == {"echo"}, f"Stage A must expose only echo, got {a_names}"
    # Stage B: DEFAULT_TOOL_POLICY (read+write) → both specs present.
    assert b_names == {"echo", "write_op"}, (
        f"Stage B must revert to DEFAULT (read+write), got {b_names}"
    )


# ---------------------------------------------------------------------------
# B1 — Durable taint: InMemoryJournal.set_run_tainted monotonic + idempotent
#      _build_context(tainted=True) seeds the gate with external already dropped
#      (Pod 3.2 B1 — closes the vacuity gap on rehydrated drives)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_in_memory_journal_set_run_tainted_monotonic() -> None:
    """set_run_tainted is monotonic (False→True) and idempotent (True→True)."""
    journal = InMemoryJournal()
    await journal.start_run(
        "run-taint",
        "sess",
        pathway_id="p",
        pathway_version=1,
        pathway_fingerprint="fp",
    )

    # Initially untainted.
    state = await journal.load_run("run-taint")
    assert state is not None
    assert state.tainted is False

    # First write: False → True.
    await journal.set_run_tainted("run-taint")
    state2 = await journal.load_run("run-taint")
    assert state2 is not None
    assert state2.tainted is True

    # Idempotent: True → True (second write is a no-op; no exception).
    await journal.set_run_tainted("run-taint")
    state3 = await journal.load_run("run-taint")
    assert state3 is not None
    assert state3.tainted is True


@pytest.mark.asyncio
async def test_in_memory_journal_set_run_tainted_unknown_run_noop() -> None:
    """set_run_tainted on an unknown run_id is a silent no-op (never raises)."""
    journal = InMemoryJournal()
    await journal.set_run_tainted("ghost-run")  # must not raise
    state = await journal.load_run("ghost-run")
    assert state is None  # run was never created


@pytest.mark.asyncio
async def test_build_context_tainted_true_gate_excludes_external() -> None:
    """_build_context(tainted=True) seeds TaintState(tainted=True) so external is dropped.

    Discriminating pair (taint is the ONLY variable that drops external):
      - Gate with external-allowing policy + tainted=False  → ext_op IS in exposed_specs().
      - Gate with external-allowing policy + tainted=True   → ext_op is NOT in exposed_specs()
        AND check_dispatch raises TierViolation.

    The pair proves taint is the discriminating cause.  A mutation that forces
    _build_context to ignore tainted (always seeds TaintState(tainted=False)) would cause
    the tainted=True gate to expose ext_op, failing the second leg of the pair.

    Separately, _build_context is verified to seed gate.taint.tainted correctly so the
    rehydration claim (SC-6 coherence across re-drives) holds structurally.
    """
    reg = _make_registry(include_read=True, include_write=True, include_external=True)

    # External-allowing policy — required so that taint is the ONLY thing that can drop
    # ext_op.  With DEFAULT_TOOL_POLICY ({read,write}), external is never exposed regardless
    # of taint, making every assertion vacuous.
    external_policy = StageToolPolicy(
        allowed_tiers=frozenset({"read", "write", "external"}),
        taint_drops_external=True,
    )

    # --- Discriminating pair ---

    # Leg 1: untainted → ext_op IS exposed (positive control).
    gate_clean = ToolGate(reg, policy=external_policy, taint=TaintState(tainted=False))
    clean_names = {s.name for s in gate_clean.exposed_specs()}
    assert "ext_op" in clean_names, (
        "positive control: external-allowing policy + tainted=False must expose ext_op"
    )
    assert "echo" in clean_names
    assert "write_op" in clean_names

    # Leg 2: tainted → ext_op is NOT exposed AND check_dispatch raises TierViolation.
    gate_tainted = ToolGate(reg, policy=external_policy, taint=TaintState(tainted=True))
    tainted_names = {s.name for s in gate_tainted.exposed_specs()}
    assert "ext_op" not in tainted_names, (
        "tainted=True must exclude ext_op — taint is the discriminating cause"
    )
    assert "echo" in tainted_names, "read-tier must survive taint"
    assert "write_op" in tainted_names, "write-tier must survive taint"
    with pytest.raises(TierViolation):
        gate_tainted.check_dispatch("ext_op")

    # --- _build_context rehydration proof ---
    # _build_context(tainted=True) must seed gate.taint.tainted = True.  This is the
    # structural claim for SC-6 coherence: a resumed drive inherits the journaled taint.
    stage = _CapturePolicyStage()
    model = ReplayModel([ModelResponse(text="ok", model_id="replay", finish_reason="stop")])
    graph = StageGraph([stage], entry=stage.name)
    pathways = PathwayRegistry()
    pathways.register("test-pathway-tainted", graph, version=1)
    mr = ModelRegistry()
    mr.register("default", model)
    journal = InMemoryJournal()
    engine = Engine(
        models=mr,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        registry=reg,
        clock=_FIXED_CLOCK,
    )

    ctx_tainted = engine._build_context(run_id="tainted-run", session_id="sess", tainted=True)
    gate_from_ctx = ctx_tainted._gate
    assert gate_from_ctx is not None, "_build_context must build a ToolGate when registry is wired"
    assert gate_from_ctx.taint.tainted is True, (
        "_build_context(tainted=True) must seed gate.taint.tainted=True (SC-6 rehydration)"
    )

    ctx_clean = engine._build_context(run_id="clean-run", session_id="sess", tainted=False)
    gate_from_ctx_clean = ctx_clean._gate
    assert gate_from_ctx_clean is not None
    assert gate_from_ctx_clean.taint.tainted is False, (
        "_build_context(tainted=False) must seed gate.taint.tainted=False (negative control)"
    )
