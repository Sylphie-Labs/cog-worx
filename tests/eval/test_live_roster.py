"""Deterministic unit tests for the roster preflight guard (Pod 4.4-live L2).

Pure, offline, deterministic — no network, no docker, no journal, no TOML file. Every
``GateRunSettings`` here is fabricated directly (not via ``load_gate_run_settings``) so the tests
can freely construct hypothetical rosters (a 1-family bring-up, a fully-disjoint fabricated
4+-family roster, deliberate family collisions) without needing real provider credentials.

PREDICATE PARITY is the load-bearing property under test (S9): for each family-disjointness rule
the live pipeline enforces (``planting.SameFamilyFallback`` / ``conversion.PanelFamilyCollision``),
a test constructs the SAME collision against the REAL enforcing code and asserts it raises, AND
against a mirrored ``GateRunSettings`` and asserts ``preflight_roster`` reports the corresponding
blocked reason — so if either predicate drifts from the other, a test here fails.
"""

from __future__ import annotations

import pytest

from cogworx.eval._live.roster import RosterReport, RosterUnsound, preflight_roster
from cogworx.eval._live.settings import GateRunMode, GateRunSettings
from cogworx.eval.conversion import PanelConfig, PanelFamilyCollision
from cogworx.eval.planting import KInjector, SameFamilyFallback
from cogworx.model.base import ModelResponse, Usage
from cogworx.model.providers.config import ZERO_PRICE_TABLE, PriceTable, ProviderConfig
from cogworx.testing.fake_model import ReplayModel

# ===========================================================================
# Fabrication helpers — build a GateRunSettings directly, bypassing the TOML loader.
# ===========================================================================

_PRICE = PriceTable(
    pro_input_usd_per_mtok=1.0,
    pro_output_usd_per_mtok=1.0,
    flash_input_usd_per_mtok=1.0,
    flash_output_usd_per_mtok=1.0,
)


def _provider_config(family: str) -> ProviderConfig:
    return ProviderConfig(
        model_pro=f"{family}-pro", model_flash=f"{family}-pro", price_per_mtok=_PRICE
    )


def _settings(
    *,
    mode: GateRunMode = "bring-up",
    arm_family: str = "deepseek",
    arm_d_prime: str | None = None,
    planter: str | None = None,
    converter_panel: tuple[str, ...] = (),
    converter_adversary: str | None = None,
) -> GateRunSettings:
    role_families: dict[str, str] = {"arm_family": arm_family}
    if planter is not None:
        role_families["planter"] = planter
    if converter_adversary is not None:
        role_families["converter_adversary"] = converter_adversary
    if arm_d_prime is not None:
        role_families["arm_d_prime"] = arm_d_prime

    all_families = {arm_family, *converter_panel}
    for extra in (planter, converter_adversary, arm_d_prime):
        if extra is not None:
            all_families.add(extra)

    return GateRunSettings(
        mode=mode,
        price_basis="list",
        resolved_price_tables={family: _PRICE for family in all_families},
        phase_a_max_usd=10.0,
        phase_b_max_usd=10.0,
        arm_family=_provider_config(arm_family),
        planter=_provider_config(planter) if planter is not None else None,
        converter_panel=tuple(_provider_config(f) for f in converter_panel),
        converter_adversary=(
            _provider_config(converter_adversary) if converter_adversary is not None else None
        ),
        arm_d_prime=_provider_config(arm_d_prime) if arm_d_prime is not None else None,
        role_families=role_families,
        converter_panel_families=converter_panel,
    )


# ===========================================================================
# 1-family bring-up
# ===========================================================================


def test_bring_up_one_family_deepseek_reports_expected_state() -> None:
    """The 1-key DeepSeek bring-up: only arm_family filled, everything else deferred."""
    settings = _settings(mode="bring-up", arm_family="deepseek")

    report = preflight_roster(settings)

    assert isinstance(report, RosterReport)
    assert report.mode == "bring-up"
    assert report.families_present == ("deepseek",)
    assert report.roles_filled == ("arm_family",)
    assert report.roles_deferred == (
        "planter",
        "converter_panel",
        "converter_adversary",
        "arm_d_prime",
    )
    assert report.binding_blocked_reasons != ()


def test_bring_up_headline_reason_names_the_specifics() -> None:
    """The headline reason names the actual family count/name and every deferred role — never a
    bare 'blocked' with no specifics (this is what makes the reasons useful in a run manifest)."""
    settings = _settings(mode="bring-up", arm_family="deepseek")

    report = preflight_roster(settings)

    headline = report.binding_blocked_reasons[0]
    assert "deepseek" in headline
    assert "1" in headline
    for role in ("planter", "converter_panel", "converter_adversary", "arm_d_prime"):
        assert role in headline


# ===========================================================================
# Fabricated fully-disjoint roster — binding-eligible
# ===========================================================================


def test_fabricated_disjoint_roster_is_binding_eligible() -> None:
    """A fully-filled, mutually-disjoint roster has NO blocked reasons and is binding-eligible."""
    settings = _settings(
        mode="binding",
        arm_family="deepseek",
        arm_d_prime="anthropic",
        planter="openai",
        converter_panel=("google", "mistral"),
        converter_adversary="cohere",
    )

    report = preflight_roster(settings)

    assert report.binding_blocked_reasons == ()
    assert report.roles_deferred == ()
    assert report.roles_filled == (
        "arm_family",
        "planter",
        "converter_panel",
        "converter_adversary",
        "arm_d_prime",
    )
    assert set(report.families_present) == {
        "deepseek",
        "anthropic",
        "openai",
        "google",
        "mistral",
        "cohere",
    }


# ===========================================================================
# Predicate parity — planter vs. arm family (mirrors planting.SameFamilyFallback)
# ===========================================================================


def test_planter_arm_collision_predicate_parity() -> None:
    """The REAL KInjector raises SameFamilyFallback on a same-family planter; the roster preflight
    over an equivalently-configured GateRunSettings must report the SAME collision."""
    model = ReplayModel(
        [ModelResponse(text="x", model_id="deepseek/x", finish_reason="stop", usage=Usage())]
    )
    with pytest.raises(SameFamilyFallback):
        KInjector(
            model,
            model_family="deepseek",
            model_id="deepseek/x",
            thesis_family="deepseek",
            antithesis_family="openai",
        )

    settings = _settings(mode="bring-up", arm_family="deepseek", planter="deepseek")
    report = preflight_roster(settings)

    collisions = [r for r in report.binding_blocked_reasons if "planter" in r and "collide" in r]
    assert collisions, report.binding_blocked_reasons
    assert "deepseek" in collisions[0]
    assert "SameFamilyFallback" in collisions[0]


def test_planter_disjoint_from_arm_family_no_collision_reason() -> None:
    """Negative control: a planter family disjoint from every arm family raises no collision
    reason (only the ordinary deferred-role reasons for the other roles)."""
    settings = _settings(mode="bring-up", arm_family="deepseek", planter="openai")

    report = preflight_roster(settings)

    collisions = [r for r in report.binding_blocked_reasons if "planter" in r and "collide" in r]
    assert collisions == []


# ===========================================================================
# Predicate parity — converter panel/adversary disjointness (mirrors
# conversion.PanelFamilyCollision / convert_k_pool's insufficient-disjoint-families gate)
# ===========================================================================


def test_converter_panel_collision_predicate_parity() -> None:
    """The REAL PanelConfig raises PanelFamilyCollision when a panel family is in the forbidden set;
    the roster preflight over an equivalently-configured GateRunSettings must report the SAME
    collision."""
    with pytest.raises(PanelFamilyCollision):
        PanelConfig(
            panel_families=("deepseek",),
            adversary_family="qwen",
            forbidden_families=frozenset({"deepseek", "openai"}),
        )

    settings = _settings(
        mode="bring-up",
        arm_family="deepseek",
        converter_panel=("deepseek",),
        converter_adversary="qwen",
    )
    report = preflight_roster(settings)

    collisions = [r for r in report.binding_blocked_reasons if "converter" in r and "collide" in r]
    assert collisions, report.binding_blocked_reasons
    assert "deepseek" in collisions[0]
    assert "PanelFamilyCollision" in collisions[0]


def test_converter_panel_internal_duplicate_collision_predicate_parity() -> None:
    """The REAL PanelConfig also raises on an internal duplicate (panel family == adversary family,
    with neither in the forbidden set); the roster preflight must catch this too."""
    with pytest.raises(PanelFamilyCollision, match="distinct"):
        PanelConfig(
            panel_families=("qwen",),
            adversary_family="qwen",
            forbidden_families=frozenset({"deepseek"}),
        )

    settings = _settings(
        mode="bring-up",
        arm_family="deepseek",
        converter_panel=("qwen",),
        converter_adversary="qwen",
    )
    report = preflight_roster(settings)

    collisions = [r for r in report.binding_blocked_reasons if "converter" in r and "collide" in r]
    assert collisions, report.binding_blocked_reasons


def test_converter_disjoint_from_forbidden_no_collision_reason() -> None:
    """Negative control: a panel/adversary fully disjoint from planter + arm families raises no
    converter collision reason."""
    settings = _settings(
        mode="bring-up",
        arm_family="deepseek",
        planter="openai",
        converter_panel=("google", "mistral"),
        converter_adversary="cohere",
    )

    report = preflight_roster(settings)

    collisions = [r for r in report.binding_blocked_reasons if "converter" in r and "collide" in r]
    assert collisions == []


# ===========================================================================
# Mode/state enforcement
# ===========================================================================


def test_binding_mode_blocked_raises_roster_unsound_naming_every_reason() -> None:
    settings = _settings(mode="binding", arm_family="deepseek")

    with pytest.raises(RosterUnsound) as excinfo:
        preflight_roster(settings)

    message = str(excinfo.value)
    assert "deepseek" in message
    for role in ("planter", "converter_panel", "converter_adversary", "arm_d_prime"):
        assert role in message


def test_binding_mode_clean_roster_passes() -> None:
    settings = _settings(
        mode="binding",
        arm_family="deepseek",
        arm_d_prime="anthropic",
        planter="openai",
        converter_panel=("google", "mistral"),
        converter_adversary="cohere",
    )

    report = preflight_roster(settings)  # must not raise

    assert report.mode == "binding"


def test_bring_up_mode_with_no_blocked_reasons_raises_roster_unsound() -> None:
    """A bring-up run whose roster is ALREADY binding-sound is an inconsistent state — never
    silently green."""
    settings = _settings(
        mode="bring-up",
        arm_family="deepseek",
        arm_d_prime="anthropic",
        planter="openai",
        converter_panel=("google", "mistral"),
        converter_adversary="cohere",
    )

    with pytest.raises(RosterUnsound, match="inconsistent"):
        preflight_roster(settings)


# ===========================================================================
# S11 zero-price refusal (re-asserted at the preflight boundary)
# ===========================================================================


def test_zero_price_role_refused_before_any_network_call() -> None:
    """A GateRunSettings carrying an unpriced role (bypassing the loader's own assert_priced call)
    is refused by the preflight itself — the ceiling can never silently become theater."""
    settings = GateRunSettings(
        mode="bring-up",
        price_basis="list",
        resolved_price_tables={},
        phase_a_max_usd=10.0,
        phase_b_max_usd=10.0,
        arm_family=ProviderConfig(model_pro="x", model_flash="x"),  # defaults to ZERO_PRICE_TABLE
        role_families={"arm_family": "deepseek"},
    )
    assert settings.arm_family.price_per_mtok == ZERO_PRICE_TABLE

    with pytest.raises(ValueError, match="ZERO_PRICE_TABLE"):
        preflight_roster(settings)


# ===========================================================================
# Mutation guard — the loud banner can never be silently dropped
# ===========================================================================


def test_blocked_reasons_banner_never_silently_empty_when_roles_deferred() -> None:
    """Mutation guard: a variant that dropped the loud blocked-reasons banner, or emptied it while
    roles remain deferred, fails here. Every deferred role's name and the family count/name must be
    literally present in the reasons text."""
    settings = _settings(mode="bring-up", arm_family="deepseek")

    report = preflight_roster(settings)

    assert report.binding_blocked_reasons != ()
    joined = " ".join(report.binding_blocked_reasons)
    assert "deepseek" in joined
    for role in ("planter", "converter_panel", "converter_adversary", "arm_d_prime"):
        assert role in joined
