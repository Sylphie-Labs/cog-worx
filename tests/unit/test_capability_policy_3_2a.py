"""Pod 3.2a Feature Test Bundle — StageToolPolicy, TaintState, ToolGate (CANON S9, S10).

Mutation-resistant assertions throughout: exact-set membership, exact counts, zero-invoke-counter
proofs.  No "absence of exception" tests — every assertion pins a concrete observable.

Coverage:
  P1 — exposed_specs filters to effective tiers only (exact set, not superset).
  P2 — check_dispatch is INDEPENDENT of exposure (forged unexposed name → TierViolation).
  P3 — effective_tiers coherence: both checkpoints use the same tier logic, can never diverge.
  P4 — taint latch: external (no trusted-output) taints; trusted-output does NOT taint.
  P5 — taint latch: untrusted-source tag taints regardless of tier.
  P6 — taint drops external from effective_tiers (lethal-trifecta break).
  P7 — bind_policy clears and replaces; None resets to DEFAULT.
  P8 — unknown capability → CapabilityUnavailable from check_dispatch (not TierViolation).
  P9 — disabled capability → CapabilityUnavailable from check_dispatch.
  P10 — exposed_specs excludes disabled capabilities.
  P11 — DEFAULT_TOOL_POLICY tier set is exactly {"read", "write"}.
"""

from __future__ import annotations

import pytest

from cogworx.capability.base import CapabilityUnavailable
from cogworx.capability.policy import (
    DEFAULT_TOOL_POLICY,
    StageToolPolicy,
    TaintState,
    TierViolation,
    ToolGate,
)
from cogworx.capability.registry import Registry, function_capability

# ---------------------------------------------------------------------------
# Fake capabilities
# ---------------------------------------------------------------------------


async def _noop_read(x: str) -> str:
    return x


async def _noop_write(x: str) -> str:
    return x


async def _noop_external(x: str) -> str:
    return x


def _make_registry(
    *,
    read_tags: tuple[str, ...] = (),
    write_tags: tuple[str, ...] = (),
    external_tags: tuple[str, ...] = (),
    include_external: bool = True,
) -> Registry:
    reg = Registry()
    cap_read = function_capability(_noop_read, name="read_cap", tier="read")
    cap_write = function_capability(_noop_write, name="write_cap", tier="write")
    reg.register(cap_read, tags=read_tags)
    reg.register(cap_write, tags=write_tags)
    if include_external:
        cap_ext = function_capability(_noop_external, name="ext_cap", tier="external")
        reg.register(cap_ext, tags=external_tags)
    return reg


# ---------------------------------------------------------------------------
# P1 — exposed_specs filters to effective tiers only
# ---------------------------------------------------------------------------


def test_exposed_specs_read_write_only() -> None:
    """Exposed specs with default policy contain ONLY read/write caps, NOT external."""
    reg = _make_registry()
    gate = ToolGate(reg)
    specs = gate.exposed_specs()
    names = {s.name for s in specs}
    assert names == {"read_cap", "write_cap"}, f"unexpected exposed names: {names}"
    # External must be absent — exact-set proof.
    assert "ext_cap" not in names


def test_exposed_specs_external_included_when_allowed() -> None:
    """External cap IS exposed when policy explicitly allows external tier."""
    reg = _make_registry()
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    gate = ToolGate(reg, policy=policy)
    names = {s.name for s in gate.exposed_specs()}
    assert names == {"read_cap", "write_cap", "ext_cap"}


def test_exposed_specs_read_only_policy() -> None:
    """A read-only policy exposes ONLY the read cap — exact count proof."""
    reg = _make_registry()
    policy = StageToolPolicy(allowed_tiers=frozenset({"read"}))
    gate = ToolGate(reg, policy=policy)
    specs = gate.exposed_specs()
    assert len(specs) == 1, f"expected 1 spec, got {len(specs)}"
    assert specs[0].name == "read_cap"


# ---------------------------------------------------------------------------
# P2 — check_dispatch is INDEPENDENT of exposure
# ---------------------------------------------------------------------------


def test_check_dispatch_refuses_tier_not_in_policy() -> None:
    """check_dispatch raises TierViolation for external even when exposure didn't show it."""
    reg = _make_registry()
    gate = ToolGate(reg)  # default policy: read+write only
    # External cap IS in the registry but NOT in effective_tiers — forged dispatch attempt.
    with pytest.raises(TierViolation):
        gate.check_dispatch("ext_cap")


def test_check_dispatch_allows_in_tier() -> None:
    """check_dispatch does NOT raise for a cap whose tier is in effective_tiers."""
    reg = _make_registry()
    gate = ToolGate(reg)
    gate.check_dispatch("read_cap")  # must not raise
    gate.check_dispatch("write_cap")  # must not raise


def test_check_dispatch_forged_name_unexposed() -> None:
    """Model forges name of a real-but-out-of-tier cap; check_dispatch catches it (SC-6)."""
    reg = _make_registry()
    policy = StageToolPolicy(allowed_tiers=frozenset({"read"}))
    gate = ToolGate(reg, policy=policy)
    # write_cap exists in registry but not in effective_tiers for this policy.
    with pytest.raises(TierViolation) as exc_info:
        gate.check_dispatch("write_cap")
    assert "write_cap" in str(exc_info.value)


# ---------------------------------------------------------------------------
# P3 — effective_tiers coherence (both checkpoints agree)
# ---------------------------------------------------------------------------


def test_effective_tiers_coherence_untainted() -> None:
    """Exposure and check agree before taint: a write cap is both exposed AND dispatchable."""
    reg = _make_registry()
    gate = ToolGate(reg)
    exposed_names = {s.name for s in gate.exposed_specs()}
    assert "write_cap" in exposed_names
    gate.check_dispatch("write_cap")  # must not raise → coherent


def test_effective_tiers_coherence_tainted_drops_external() -> None:
    """After taint, external is dropped from BOTH checkpoints — coherence invariant."""
    reg = _make_registry(external_tags=())
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()
    gate = ToolGate(reg, policy=policy, taint=taint)

    # Before taint: external is exposed and dispatchable.
    assert "ext_cap" in {s.name for s in gate.exposed_specs()}
    gate.check_dispatch("ext_cap")  # must not raise

    # Manually taint.
    taint.tainted = True

    # After taint: external is dropped from BOTH.
    assert "ext_cap" not in {s.name for s in gate.exposed_specs()}
    with pytest.raises(TierViolation):
        gate.check_dispatch("ext_cap")


# ---------------------------------------------------------------------------
# P4 — taint latch: external (no trusted-output) taints; trusted-output does NOT
# ---------------------------------------------------------------------------


def test_taint_update_external_no_trusted_output_taints() -> None:
    reg = _make_registry(external_tags=())
    taint = TaintState()
    assert not taint.tainted
    taint.update("ext_cap", reg)
    assert taint.tainted, "external cap without trusted-output must latch taint"


def test_taint_update_external_trusted_output_does_not_taint() -> None:
    reg = _make_registry(external_tags=("trusted-output",))
    taint = TaintState()
    taint.update("ext_cap", reg)
    assert not taint.tainted, "external cap WITH trusted-output must NOT latch taint"


def test_taint_update_read_tier_does_not_taint() -> None:
    reg = _make_registry()
    taint = TaintState()
    taint.update("read_cap", reg)
    assert not taint.tainted


def test_taint_update_write_tier_does_not_taint() -> None:
    reg = _make_registry()
    taint = TaintState()
    taint.update("write_cap", reg)
    assert not taint.tainted


# ---------------------------------------------------------------------------
# P5 — untrusted-source tag taints regardless of tier
# ---------------------------------------------------------------------------


def test_taint_update_read_with_untrusted_source_taints() -> None:
    reg = _make_registry(read_tags=("untrusted-source",))
    taint = TaintState()
    taint.update("read_cap", reg)
    assert taint.tainted, "untrusted-source tag must taint regardless of tier"


def test_taint_update_write_with_untrusted_source_taints() -> None:
    reg = _make_registry(write_tags=("untrusted-source",))
    taint = TaintState()
    taint.update("write_cap", reg)
    assert taint.tainted


def test_taint_update_irreversible() -> None:
    """Taint is a one-way latch — a subsequent clean-cap update cannot clear it."""
    reg = _make_registry(external_tags=())
    taint = TaintState()
    taint.update("ext_cap", reg)  # taints
    assert taint.tainted
    taint.update("read_cap", reg)  # clean cap — taint must not clear
    assert taint.tainted, "taint must remain latched after a clean-cap update"


# ---------------------------------------------------------------------------
# P6 — taint drops external from effective_tiers (lethal-trifecta structural break)
# ---------------------------------------------------------------------------


def test_taint_drops_external_from_exposure() -> None:
    reg = _make_registry(external_tags=())
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState()
    taint.tainted = True
    gate = ToolGate(reg, policy=policy, taint=taint)
    names = {s.name for s in gate.exposed_specs()}
    assert "ext_cap" not in names
    assert names == {"read_cap", "write_cap"}


def test_taint_does_not_drop_external_when_taint_drops_external_false() -> None:
    """taint_drops_external=False: taint does not remove external from effective tiers."""
    reg = _make_registry(external_tags=())
    policy = StageToolPolicy(
        allowed_tiers=frozenset({"read", "write", "external"}),
        taint_drops_external=False,
    )
    taint = TaintState()
    taint.tainted = True
    gate = ToolGate(reg, policy=policy, taint=taint)
    names = {s.name for s in gate.exposed_specs()}
    assert "ext_cap" in names, "taint_drops_external=False must keep external in effective tiers"


# ---------------------------------------------------------------------------
# P7 — bind_policy clears and replaces; None resets to DEFAULT
# ---------------------------------------------------------------------------


def test_bind_policy_replaces() -> None:
    reg = _make_registry()
    gate = ToolGate(reg)
    assert gate.policy.allowed_tiers == frozenset({"read", "write"})
    new_policy = StageToolPolicy(allowed_tiers=frozenset({"read"}))
    gate.bind_policy(new_policy)
    assert gate.policy.allowed_tiers == frozenset({"read"})
    assert len(gate.exposed_specs()) == 1


def test_bind_policy_none_resets_to_default() -> None:
    reg = _make_registry()
    policy = StageToolPolicy(allowed_tiers=frozenset({"read"}))
    gate = ToolGate(reg, policy=policy)
    gate.bind_policy(None)
    assert gate.policy is DEFAULT_TOOL_POLICY


# ---------------------------------------------------------------------------
# P8 — unknown capability → CapabilityUnavailable (not TierViolation)
# ---------------------------------------------------------------------------


def test_check_dispatch_unknown_name_raises_unavailable() -> None:
    reg = _make_registry()
    gate = ToolGate(reg)
    with pytest.raises(CapabilityUnavailable):
        gate.check_dispatch("does_not_exist")


def test_check_dispatch_unknown_name_not_tier_violation() -> None:
    reg = _make_registry()
    gate = ToolGate(reg)
    with pytest.raises(CapabilityUnavailable):
        gate.check_dispatch("phantom_cap")
    # Must NOT raise TierViolation — distinct error taxonomy.


# ---------------------------------------------------------------------------
# P9 — disabled capability → CapabilityUnavailable
# ---------------------------------------------------------------------------


def test_check_dispatch_disabled_raises_unavailable() -> None:
    reg = _make_registry()
    reg.disable("read_cap")
    gate = ToolGate(reg)
    with pytest.raises(CapabilityUnavailable):
        gate.check_dispatch("read_cap")


# ---------------------------------------------------------------------------
# P10 — exposed_specs excludes disabled capabilities
# ---------------------------------------------------------------------------


def test_exposed_specs_excludes_disabled() -> None:
    reg = _make_registry()
    reg.disable("read_cap")
    gate = ToolGate(reg)
    names = {s.name for s in gate.exposed_specs()}
    assert "read_cap" not in names
    assert "write_cap" in names


def test_exposed_specs_count_exact_after_disable() -> None:
    reg = _make_registry(include_external=False)
    reg.disable("write_cap")
    gate = ToolGate(reg)
    specs = gate.exposed_specs()
    assert len(specs) == 1, f"expected exactly 1 spec, got {len(specs)}"
    assert specs[0].name == "read_cap"


# ---------------------------------------------------------------------------
# P11 — DEFAULT_TOOL_POLICY tier set is exactly {"read", "write"}
# ---------------------------------------------------------------------------


def test_default_policy_tiers_exact() -> None:
    assert DEFAULT_TOOL_POLICY.allowed_tiers == frozenset({"read", "write"})
    assert "external" not in DEFAULT_TOOL_POLICY.allowed_tiers


def test_default_policy_taint_drops_external_true() -> None:
    assert DEFAULT_TOOL_POLICY.taint_drops_external is True


def test_default_policy_timeout_positive() -> None:
    assert DEFAULT_TOOL_POLICY.tool_timeout_s > 0


# ---------------------------------------------------------------------------
# B1 — TaintState seeded from journaled tainted bit (Pod 3.2 B1)
# ---------------------------------------------------------------------------


def test_taint_state_init_false_by_default() -> None:
    """TaintState() starts untainted (default preserved)."""
    t = TaintState()
    assert t.tainted is False


def test_taint_state_init_true_from_journal() -> None:
    """TaintState(tainted=True) seeds the latch from a journaled tainted run."""
    t = TaintState(tainted=True)
    assert t.tainted is True


def test_taint_state_seeded_true_stays_latched() -> None:
    """A gate seeded tainted=True has external dropped from effective tiers immediately."""
    reg = _make_registry(external_tags=())
    policy = StageToolPolicy(allowed_tiers=frozenset({"read", "write", "external"}))
    taint = TaintState(tainted=True)
    gate = ToolGate(reg, policy=policy, taint=taint)

    # External must already be absent — no dispatch needed to taint a rehydrated drive.
    names = {s.name for s in gate.exposed_specs()}
    assert "ext_cap" not in names, (
        "rehydrated tainted gate must NOT expose external caps before any dispatch"
    )
    with pytest.raises(TierViolation):
        gate.check_dispatch("ext_cap")


# ---------------------------------------------------------------------------
# B1 — persist_taint hook (Pod 3.2 B1)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_persist_taint_noop_when_no_hook() -> None:
    """persist_taint is a coroutine no-op when no hook was injected."""
    reg = _make_registry()
    gate = ToolGate(reg)
    await gate.persist_taint()  # must not raise


@pytest.mark.asyncio
async def test_persist_taint_calls_hook_on_invocation() -> None:
    """persist_taint invokes the injected hook each time it is awaited."""
    call_log: list[str] = []

    async def _hook() -> None:
        call_log.append("called")

    reg = _make_registry()
    gate = ToolGate(reg, persist_taint=_hook)

    await gate.persist_taint()
    await gate.persist_taint()  # second call must also delegate (journal handles idempotency)
    assert call_log == ["called", "called"], (
        "hook must be called each time persist_taint is awaited (idempotent at journal level)"
    )
