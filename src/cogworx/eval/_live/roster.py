"""Roster preflight guard for the live GATE run (Pod 4.4-live L2).

This module is the LAST gate before a :class:`~cogworx.eval._live.settings.GateRunSettings` is
handed to anything that makes a network call. :func:`preflight_roster` (1) re-asserts the S11
zero-price refusal on every FILLED role (defense in depth:
:func:`~cogworx.eval._live.settings.load_gate_run_settings` already calls
:func:`~cogworx.eval._live.settings.assert_priced` per role at load time, but a ``GateRunSettings``
can in principle be constructed by another path — this preflight is itself a system boundary, so it
re-checks rather than trusting the loader was used), then (2) reports whether the CONFIGURED roster
could ever support a ``mode="binding"`` verdict, and (3) enforces the mode/state match:
``"bring-up"`` may proceed with a non-empty ``binding_blocked_reasons`` (that IS a bring-up run);
``"binding"`` may never proceed with one.

PREDICATE PARITY (the S9 point — read before touching ``_binding_blocked_reasons``):
  The two family-disjointness checks below are NOT independent judgment calls — they mirror the
  EXACT conditions the live pipeline enforces elsewhere, so this preflight can never rubber-stamp a
  roster that the pipeline itself would refuse:

    - **planter vs. arm family** mirrors
      :class:`cogworx.eval.planting.KInjector`'s build-time check (``planting.py`` ~L592-607): a
      same-family planter raises :class:`~cogworx.eval.planting.SameFamilyFallback`. Here, "the
      arm families" are
      :attr:`~cogworx.eval._live.settings.GateRunSettings.role_families`\\ ``["arm_family"]``
      (drives D / C / C' / B) and, if filled, ``["arm_d_prime"]`` (drives D').
    - **converter panel/adversary disjointness** mirrors
      :meth:`cogworx.eval.conversion.PanelConfig._check_disjoint` (construction-time) and
      :func:`cogworx.eval.conversion.convert_k_pool`'s insufficient-disjoint-families degrade gate
      (``conversion.py`` ~L519-529): the panel + adversary must be internally distinct AND disjoint
      from the forbidden set (planter + arm families).

  Both conditions are re-derived here from the same primitives (family-string set membership /
  internal-duplicate check) the enforcing code uses — never a re-implementation with its own drift
  risk. A test suite change to either enforcing predicate without a matching change here (or vice
  versa) is exactly what the predicate-parity tests in ``tests/eval/test_live_roster.py`` are built
  to catch (construct the SAME collision and assert both the enforcing code raises AND this
  preflight reports the corresponding reason).

Pure stdlib + pydantic; no substrate, no network, no concrete provider import (CANON S1/S2/S4).
Reuses :class:`~cogworx.eval._live.settings.GateRunSettings` /
:func:`~cogworx.eval._live.settings.assert_priced` — reused, never redefined.

Contract changelog (CANON §6.1):
  - 2026-07-02 (Pod 4.4-live L2): initial — ``RosterReport`` / ``RosterUnsound`` /
    ``preflight_roster``. New module; no existing callers. Additive new public surface only.
"""

from __future__ import annotations

from typing import Final

from pydantic import BaseModel, ConfigDict

from cogworx.eval._live.settings import GateRunMode, GateRunSettings, assert_priced

__all__ = [
    "RosterReport",
    "RosterUnsound",
    "preflight_roster",
]

#: Declaration order for every role the schema names (CANON: never omit a deferred role from the
#: schema — see ``settings.py``'s "present-but-empty" note). Also the canonical ordering for
#: ``RosterReport.roles_filled`` / ``roles_deferred``.
_ALL_ROLES: Final[tuple[str, ...]] = (
    "arm_family",
    "planter",
    "converter_panel",
    "converter_adversary",
    "arm_d_prime",
)


class RosterUnsound(RuntimeError):
    """Raised BEFORE any network call when the roster's real state cannot support what ``mode``
    asks for: a ``mode="binding"`` roster with any :attr:`RosterReport.binding_blocked_reasons`,
    or a ``mode="bring-up"`` roster that (inconsistently) has none. Names every blocking reason."""


class RosterReport(BaseModel):
    """The single source of truth for the roster's pre-flight state (Pod 4.4-live L2).

    A driver (L6) logs :attr:`binding_blocked_reasons` at run start and end and stamps them into the
    run manifest; this report is the one place that text is composed.

    Attributes
    ----------
    families_present:
        Sorted, de-duplicated model-family (``provider``) strings across every FILLED role.
    roles_filled:
        Role names with a configured ``ProviderConfig`` (or, for ``converter_panel``, at least one
        entry), in :data:`_ALL_ROLES` order.
    roles_deferred:
        The complement of ``roles_filled`` within :data:`_ALL_ROLES`, same ordering.
    binding_blocked_reasons:
        Human-readable reasons a ``mode="binding"`` verdict is not (yet) reachable with this
        roster. Empty iff every role is filled AND every family-disjointness predicate this
        preflight mirrors passes. A ``mode="bring-up"`` run is expected to carry a non-empty tuple
        here (that is the definition of "bring-up, not binding"); see :func:`preflight_roster`.
    mode:
        The :class:`~cogworx.eval._live.settings.GateRunSettings` mode this report was built from.
    """

    model_config = ConfigDict(frozen=True)

    families_present: tuple[str, ...]
    roles_filled: tuple[str, ...]
    roles_deferred: tuple[str, ...]
    binding_blocked_reasons: tuple[str, ...]
    mode: GateRunMode


def _filled_roles(settings: GateRunSettings) -> tuple[str, ...]:
    filled = ["arm_family"]  # load_gate_run_settings guarantees this role is always filled
    if settings.planter is not None:
        filled.append("planter")
    if settings.converter_panel:
        filled.append("converter_panel")
    if settings.converter_adversary is not None:
        filled.append("converter_adversary")
    if settings.arm_d_prime is not None:
        filled.append("arm_d_prime")
    return tuple(filled)


def _assert_all_priced(settings: GateRunSettings) -> None:
    """Re-assert the S11 zero-price refusal on every FILLED role — refuse before any network use,
    regardless of whether the settings object came from
    :func:`~cogworx.eval._live.settings.load_gate_run_settings` (which already checks this at load
    time) or was constructed some other way."""
    assert_priced(settings.arm_family, role="arm_family")
    if settings.planter is not None:
        assert_priced(settings.planter, role="planter")
    for config in settings.converter_panel:
        assert_priced(config, role="converter_panel")
    if settings.converter_adversary is not None:
        assert_priced(settings.converter_adversary, role="converter_adversary")
    if settings.arm_d_prime is not None:
        assert_priced(settings.arm_d_prime, role="arm_d_prime")


def _arm_families(settings: GateRunSettings) -> frozenset[str]:
    """The families driving the model arms: ``arm_family`` (D / C / C' / B) plus, if filled,
    ``arm_d_prime`` (D')."""
    families = {settings.role_families["arm_family"]}
    d_prime_family = settings.role_families.get("arm_d_prime")
    if d_prime_family is not None:
        families.add(d_prime_family)
    return frozenset(families)


def _binding_blocked_reasons(
    settings: GateRunSettings,
    *,
    roles_deferred: tuple[str, ...],
    families_present: tuple[str, ...],
) -> tuple[str, ...]:
    reasons: list[str] = []
    arm_families = _arm_families(settings)

    # Mirrors planting.SameFamilyFallback (planting.py ~L592-607): the planter family must be
    # disjoint from every arm family.
    planter_family = settings.role_families.get("planter")
    if planter_family is not None and planter_family in arm_families:
        reasons.append(
            f"role 'planter' family {planter_family!r} collides with an arm family "
            f"{sorted(arm_families)!r} — mirrors planting.SameFamilyFallback (would raise at "
            "KInjector construction; CF-4.4c-PLANTER)"
        )

    # Mirrors conversion.PanelConfig._check_disjoint + convert_k_pool's
    # insufficient-disjoint-families degrade gate (conversion.py ~L195-222, ~L519-529): panel +
    # adversary must be internally distinct AND disjoint from the forbidden set (planter + arm
    # families).
    forbidden = set(arm_families)
    if planter_family is not None:
        forbidden.add(planter_family)
    adversary_family = settings.role_families.get("converter_adversary")
    converter_used = tuple(settings.converter_panel_families) + (
        (adversary_family,) if adversary_family is not None else ()
    )
    if converter_used:
        internal_dup = len(set(converter_used)) != len(converter_used)
        clash = set(converter_used) & forbidden
        if internal_dup or clash:
            reasons.append(
                f"converter panel/adversary families {sorted(converter_used)!r} collide with the "
                f"forbidden set (planter + arm families) {sorted(forbidden)!r} — mirrors "
                "conversion.PanelFamilyCollision / convert_k_pool's insufficient-disjoint-families "
                "gate"
            )

    if roles_deferred:
        reasons.insert(
            0,
            f"binding verdict not reachable: {len(families_present)} family present "
            f"({', '.join(families_present)}); {'/'.join(roles_deferred)} deferred",
        )

    return tuple(reasons)


def preflight_roster(settings: GateRunSettings) -> RosterReport:
    """Refuse-before-network-use roster preflight (Pod 4.4-live L2; plan §1 — the roster guard).

    Re-asserts the S11 zero-price refusal on every filled role, then derives
    :attr:`RosterReport.binding_blocked_reasons` from the SAME family-disjointness predicates
    :class:`~cogworx.eval.planting.SameFamilyFallback` /
    :class:`~cogworx.eval.conversion.PanelFamilyCollision` enforce (see the module docstring's
    "PREDICATE PARITY" section).

    Mode/state enforcement:
      - ``mode="bring-up"`` with ``binding_blocked_reasons`` non-empty: PASSES (returns the
        report — that non-empty tuple is exactly what makes this a bring-up run, not a binding
        one).
      - ``mode="bring-up"`` with ``binding_blocked_reasons`` EMPTY: an inconsistent state (a
        bring-up run must never look binding-eligible) — raises :class:`RosterUnsound`.
      - ``mode="binding"`` with any blocked reason: raises :class:`RosterUnsound` BEFORE any
        network call, naming every reason.
      - ``mode="binding"`` with none: PASSES (returns the report).
    """
    _assert_all_priced(settings)

    roles_filled = _filled_roles(settings)
    roles_deferred = tuple(role for role in _ALL_ROLES if role not in roles_filled)
    families_present = tuple(
        sorted(set(settings.role_families.values()) | set(settings.converter_panel_families))
    )

    binding_blocked_reasons = _binding_blocked_reasons(
        settings,
        roles_deferred=roles_deferred,
        families_present=families_present,
    )

    report = RosterReport(
        families_present=families_present,
        roles_filled=roles_filled,
        roles_deferred=roles_deferred,
        binding_blocked_reasons=binding_blocked_reasons,
        mode=settings.mode,
    )

    if settings.mode == "bring-up":
        if not report.binding_blocked_reasons:
            raise RosterUnsound(
                "preflight_roster: mode='bring-up' but binding_blocked_reasons is EMPTY — an "
                "inconsistent state (a bring-up run must never report itself binding-eligible; if "
                "this roster really is binding-sound, load it with mode='binding')."
            )
        return report

    # mode == "binding"
    if report.binding_blocked_reasons:
        raise RosterUnsound(
            "preflight_roster: mode='binding' refused BEFORE any network call — "
            + "; ".join(report.binding_blocked_reasons)
        )
    return report
