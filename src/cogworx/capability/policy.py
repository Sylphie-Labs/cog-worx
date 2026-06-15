"""Stage tool policy + gate (CANON S9, S10, S11) — Pod 3.2a.

Security by structure: the gate exposes only the tools the model is *allowed to call*, validates
tier at dispatch time **independently of exposure** (so a model that forges the name of a real-but-
unexposed tool is caught at the chokepoint), and latches taint from external-tier invocations so the
lethal-trifecta break (S10) is structural, not content-filtered.

Error taxonomy
--------------
``TierViolation``
    A LOUD security refusal — the model named a tool whose permission tier is outside the stage's
    effective tier set.  This is not a degradation signal (S8); it is a structural access violation
    that MUST surface loud so the framework can audit it (S10).  It is distinct from
    ``CapabilityUnavailable`` (which signals "tool not wired / lesioned — degrade gracefully").

``ToolArgumentError``
    The tool arguments supplied by the model failed framework-side jsonschema validation.  The
    framework NEVER repairs or coerces the arguments (S9 — structure over prompting); the call is
    rejected and the validation error is fed back to the model as a structured ``ToolResult``.

Taint trigger discipline (CF-3.2-B):
    Untagged read-tier ingestion does NOT taint; tagging untrusted sources is the integrator's
    obligation until CF-3.2-B (owner: architect) ships a structural tag-at-ingest mechanism.
    The trigger is: ``cap.tier == "external"`` AND ``"trusted-output"`` NOT in tags, OR
    ``"untrusted-source"`` in tags.  Tag-dependent taint is a documented carry-forward, not
    silently fixed.

Contract changelog:
  - 2026-06-12 (Pod 3.2a): initial — StageToolPolicy, TaintState, ToolGate, TierViolation,
    ToolArgumentError.  Additive new module; no existing callers.
  - 2026-06-12 (Pod 3.2 B1/B3): TaintState.__init__ gains ``tainted: bool = False`` so
    rehydrated state can seed the in-memory latch.  ToolGate gains ``persist_taint`` async hook
    (injected by the engine) for durable journal write on the False→True latch transition.
    Additive: no existing callers are broken (defaults preserve current behaviour).
  - 2026-06-12 (Pod 3.5b): ApprovalRequired + ToolGate.check_approval — additive (new exception +
    new method on ToolGate; no existing callers affected).
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable

from pydantic import BaseModel, ConfigDict

from cogworx.capability.base import CapabilityUnavailable, PermissionTier
from cogworx.capability.registry import Registry, RegistryError
from cogworx.model.base import ToolSpec


class ApprovalRequired(Exception):
    """S10 trifecta refusal: a consequential ∧ irreversible capability was dispatched on a
    TAINTED drive without a human approval grant. Loud, structural — the caller routes to
    AwaitHuman (the Pod 1.3 durable HITL transition), never retries silently."""


class TierViolation(Exception):
    """The model requested a tool whose tier is outside the stage's effective allowed set.

    This is a structural security refusal (S10), not a graceful-degradation signal (S8).
    ``CapabilityUnavailable`` means "unavailable — degrade"; ``TierViolation`` means
    "access denied — surface loud."
    """


class ToolArgumentError(Exception):
    """The model-supplied tool arguments failed framework-side jsonschema validation (S9).

    The framework never repairs or coerces arguments (structure over prompting).  The validation
    error is serialised into the ``ToolResult`` and fed back to the model as structured feedback.
    """


class StageToolPolicy(BaseModel):
    """Per-stage tool-access policy (frozen, rebindable via ``ToolGate.bind_policy``).

    Attributes
    ----------
    allowed_tiers:
        The permission tiers the stage may dispatch.  Default ``{"read", "write"}`` per the
        mythos design (``"external"`` is opt-in; a Jim sign-off is pending on widening the
        default — ``DEFAULT_TOOL_POLICY`` is the single named constant so flipping is one line).
    taint_drops_external:
        When ``True`` (default) and the drive is tainted, ``"external"`` is removed from
        ``effective_tiers`` even if it is in ``allowed_tiers``.  This structurally breaks the
        lethal trifecta (S10): once untrusted content touches the drive, exfiltration-capable
        tools drop away automatically.
    tool_timeout_s:
        Per-call hard timeout for ``cap.invoke`` (seconds).  Enforced via ``asyncio.wait_for``
        in the router (S11 structural ceiling — the model cannot extend it).
    """

    model_config = ConfigDict(frozen=True)

    allowed_tiers: frozenset[PermissionTier] = frozenset({"read", "write"})
    taint_drops_external: bool = True
    tool_timeout_s: float = 30.0


DEFAULT_TOOL_POLICY: StageToolPolicy = StageToolPolicy()
"""The default stage tool policy.  Change this single constant to adjust the framework-wide default.

Current default: tiers ``{"read", "write"}`` (external is opt-in); taint drops external; 30 s
per-call timeout.  A Jim sign-off is pending on whether the default should narrow to ``{"read"}``.
"""


class TaintState:
    """Drive-level taint latch — mutable, held by ``ToolGate`` across the drive.

    Taint is determined STRUCTURALLY (S9): a capability taints the drive iff any of:

      1. ``cap.tier == "external"`` AND ``"trusted-output"`` is NOT in the capability's registered
         tags (code-assigned at registration; the model cannot inject or remove tags).
      2. The capability carries the tag ``"untrusted-source"`` (regardless of tier).

    Once latched, taint is never cleared within a drive (irreversible security posture).
    The model/MCP layer cannot observe or influence the taint state directly.

    CF-3.2-B (carry-forward, owner: architect): untagged read-tier ingestion does NOT taint;
    tagging untrusted sources is the integrator's obligation until CF-3.2-B ships a structural
    tag-at-ingest mechanism.  Tag-dependent taint trigger is a documented carry-forward, not
    silently fixed.
    """

    def __init__(self, *, tainted: bool = False) -> None:
        self.tainted: bool = tainted

    def update(self, name: str, registry: Registry) -> None:
        """Latch taint if the dispatched capability meets the taint rule (structural, S9).

        ``name`` is the capability name; ``registry`` is the source of the code-assigned tags.
        Raises ``RegistryError`` if the name is unknown (callers should catch this case before
        calling ``update`` — the router validates the name first).
        """
        if self.tainted:
            return
        try:
            tags = registry.tags_of(name)
        except RegistryError:
            return
        cap = registry.get(name)
        is_external = cap.tier == "external"
        has_trusted_output = "trusted-output" in tags
        has_untrusted_source = "untrusted-source" in tags
        if (is_external and not has_trusted_output) or has_untrusted_source:
            self.tainted = True


class ToolGate:
    """Single security chokepoint for stage-level capability dispatch (CANON S10).

    Holds the registry, the bound policy, and the drive-level taint state.  Both the exposure
    surface (``exposed_specs``) and the dispatch check (``check_dispatch``) derive from the same
    ``effective_tiers`` computation so they can never diverge (SC-6 coherence requirement).

    The policy is rebindable per stage via ``bind_policy``; the taint state is drive-level and
    never reset between stages.

    Args
    ----
    registry:
        The ``Registry`` instance containing all registered capabilities.
    policy:
        Initial policy; defaults to ``DEFAULT_TOOL_POLICY``.  Pass ``None`` to ``bind_policy``
        to reset to the default.
    taint:
        The drive-level ``TaintState`` instance (shared across all stages in the drive).
    """

    def __init__(
        self,
        registry: Registry,
        *,
        policy: StageToolPolicy | None = None,
        taint: TaintState | None = None,
        persist_taint: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._registry = registry
        self._policy: StageToolPolicy = policy if policy is not None else DEFAULT_TOOL_POLICY
        self._taint: TaintState = taint if taint is not None else TaintState()
        # Async hook injected by the engine — awaited BEFORE cap.invoke on the False→True
        # transition (B1 durable ordering: S6).  None when no journal is wired (tests, etc.).
        self._persist_taint: Callable[[], Awaitable[None]] | None = persist_taint

    def bind_policy(self, policy: StageToolPolicy | None) -> None:
        """Rebind the policy for the next stage.  ``None`` resets to ``DEFAULT_TOOL_POLICY``."""
        self._policy = policy if policy is not None else DEFAULT_TOOL_POLICY

    async def persist_taint(self) -> None:
        """Await the injected durable-taint hook (S6 ordering: before invoke, after latch).

        Called by ``dispatch_one`` on the False→True latch transition — BEFORE ``cap.invoke``
        so a crash between journal write and invocation leaves the run correctly tainted on
        resume (fail-closed).  A no-op when no hook was injected (tests without a journal).
        If the hook raises, the exception propagates and ``cap.invoke`` is NOT called.
        """
        if self._persist_taint is not None:
            await self._persist_taint()

    @property
    def policy(self) -> StageToolPolicy:
        return self._policy

    @property
    def taint(self) -> TaintState:
        return self._taint

    def _effective_tiers(self) -> frozenset[PermissionTier]:
        """The tier set currently in effect — the single source of truth for both checkpoints.

        Removes ``"external"`` from the allowed set when the drive is tainted and
        ``taint_drops_external`` is True.  Both ``exposed_specs`` and ``check_dispatch`` call this
        method so they cannot diverge on the tier logic (SC-6 coherence invariant).
        """
        tiers = self._policy.allowed_tiers
        if self._taint.tainted and self._policy.taint_drops_external:
            tiers = tiers - frozenset({"external"})
        return tiers

    def exposed_specs(self) -> tuple[ToolSpec, ...]:
        """Return ``ToolSpec``s for every enabled capability whose tier is in ``effective_tiers``.

        The model is shown ONLY the specs it is allowed to call.  Disabled capabilities and
        out-of-tier capabilities are both invisible to the model (S10 least-privilege).
        """
        effective = self._effective_tiers()
        specs: list[ToolSpec] = []
        for cap in self._registry.list():
            if cap.tier not in effective:
                continue
            description = getattr(cap, "description", "")
            specs.append(
                ToolSpec(
                    name=cap.name,
                    description=description if isinstance(description, str) else "",
                    input_schema=dict(cap.input_schema),
                )
            )
        return tuple(specs)

    def check_dispatch(self, name: str) -> None:
        """Re-validate tier at call time INDEPENDENTLY of exposure.

        This second checkpoint catches the adversarial case where the model forges the name of a
        real-but-unexposed tool (one that exists in the registry but whose tier was not exposed).
        The check is independent of ``exposed_specs`` — it reads ``effective_tiers`` fresh from the
        same ``_effective_tiers()`` method.

        Raises
        ------
        ``CapabilityUnavailable``
            If the name is unknown or disabled/lesioned in the registry.
        ``TierViolation``
            If the capability's tier is outside the effective allowed set (LOUD security refusal).
        """
        try:
            cap = self._registry.get(name)
        except RegistryError as exc:
            raise CapabilityUnavailable(f"capability {name!r} is unavailable: {exc}") from exc
        effective = self._effective_tiers()
        if cap.tier not in effective:
            raise TierViolation(
                f"capability {name!r} has tier {cap.tier!r} which is not in "
                f"the effective allowed tiers {set(effective)!r} for this stage "
                f"(policy={self._policy!r}, tainted={self._taint.tainted!r})"
            )

    def check_approval(self, name: str, *, approved: bool) -> None:
        """Raise ApprovalRequired iff tainted ∧ consequential ∧ irreversible ∧ not approved.

        Reads taint state AS OF DISPATCH ENTRY (before this call's own latch).
        Tags are code-assigned at registration; the model cannot influence them (S9).
        """
        if not (self._taint.tainted and not approved):
            return
        try:
            tags = self._registry.tags_of(name)
        except RegistryError:
            return
        if "consequential" in tags and "irreversible" in tags:
            raise ApprovalRequired(
                f"capability {name!r} requires human approval: drive is tainted and "
                f"capability has 'consequential' + 'irreversible' tags"
            )


__all__ = [
    "DEFAULT_TOOL_POLICY",
    "ApprovalRequired",
    "StageToolPolicy",
    "TaintState",
    "TierViolation",
    "ToolArgumentError",
    "ToolGate",
]
