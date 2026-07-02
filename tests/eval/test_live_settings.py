"""Deterministic unit tests for the live GATE-run settings (Pod 4.4-live L0/L1).

Pure, offline, deterministic — no network, no docker, no journal. Covers:

  - L0: the S11 zero-price refusal (``assert_priced``) and the DeepSeek V4 Pro price tables'
    ``cost_usd`` arithmetic at both bases.
  - L1: the ``GateRunSettings`` TOML round-trip — a valid config loads, a secret literal in the
    TOML is refused, a missing env var is a clear error, and ``mode`` parses.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from cogworx.eval._live.settings import (
    DEEPSEEK_V4_PRO_LIST_PRICE,
    DEEPSEEK_V4_PRO_PROMO_PRICE,
    assert_priced,
    load_gate_run_settings,
)
from cogworx.model.base import Usage
from cogworx.model.providers.config import ZERO_PRICE_TABLE, ProviderConfig

# ===========================================================================
# L0 — assert_priced (the S11 zero-price refusal)
# ===========================================================================


def test_assert_priced_refuses_zero_price_table() -> None:
    """A ProviderConfig left at the ZERO_PRICE_TABLE default is refused loudly, naming the role."""
    config = ProviderConfig(model_pro="x", model_flash="x")  # price_per_mtok defaults to zero
    with pytest.raises(ValueError, match="ZERO_PRICE_TABLE"):
        assert_priced(config, role="arm_family")


def test_assert_priced_names_the_role_in_the_message() -> None:
    """The refusal names the offending role so an operator can find the TOML section."""
    config = ProviderConfig(model_pro="x", model_flash="x")
    with pytest.raises(ValueError, match="planter"):
        assert_priced(config, role="planter")


def test_assert_priced_accepts_a_real_price_table() -> None:
    """A ProviderConfig carrying a non-zero PriceTable is NOT refused."""
    config = ProviderConfig(
        model_pro="deepseek-chat",
        model_flash="deepseek-chat",
        price_per_mtok=DEEPSEEK_V4_PRO_LIST_PRICE,
    )
    assert_priced(config, role="arm_family")  # must not raise


def test_zero_price_table_is_still_the_provider_config_default() -> None:
    """Pins the precondition assert_priced defends against: an un-configured ProviderConfig really
    does default to ZERO_PRICE_TABLE (if this ever changes, assert_priced's docstring is stale)."""
    assert ProviderConfig(model_pro="x", model_flash="x").price_per_mtok == ZERO_PRICE_TABLE


# ===========================================================================
# L0 — DeepSeek V4 Pro PriceTable.cost_usd arithmetic (both bases)
# ===========================================================================


def test_deepseek_list_price_cost_usd_known_tokens() -> None:
    """1M prompt + 0.5M completion tokens at LIST price: $1.74 + $1.74 = $3.48."""
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=500_000)
    cost = DEEPSEEK_V4_PRO_LIST_PRICE.cost_usd(usage, "pro")
    assert cost == pytest.approx(1.74 * 1.0 + 3.48 * 0.5)


def test_deepseek_promo_price_cost_usd_known_tokens() -> None:
    """2M prompt + 1M completion tokens at PROMO price: 2*0.435 + 1*0.87 = $1.74."""
    usage = Usage(prompt_tokens=2_000_000, completion_tokens=1_000_000)
    cost = DEEPSEEK_V4_PRO_PROMO_PRICE.cost_usd(usage, "pro")
    assert cost == pytest.approx(0.435 * 2.0 + 0.87 * 1.0)


def test_deepseek_price_tables_fill_both_tiers_identically() -> None:
    """No distinct flash model id is configured for the bring-up — both tiers carry the same V4 Pro
    numbers, so flash and pro cost_usd agree on identical token counts."""
    usage = Usage(prompt_tokens=750_000, completion_tokens=250_000)
    assert DEEPSEEK_V4_PRO_LIST_PRICE.cost_usd(usage, "pro") == DEEPSEEK_V4_PRO_LIST_PRICE.cost_usd(
        usage, "flash"
    )
    assert DEEPSEEK_V4_PRO_PROMO_PRICE.cost_usd(
        usage, "pro"
    ) == DEEPSEEK_V4_PRO_PROMO_PRICE.cost_usd(usage, "flash")


def test_deepseek_promo_is_cheaper_than_list() -> None:
    """Sanity: the promo basis really is a discount off list (75% off per the authored numbers)."""
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert DEEPSEEK_V4_PRO_PROMO_PRICE.cost_usd(usage, "pro") < DEEPSEEK_V4_PRO_LIST_PRICE.cost_usd(
        usage, "pro"
    )


# ===========================================================================
# L1 — GateRunSettings TOML round-trip
# ===========================================================================

_VALID_TOML = """
mode = "bring-up"
price_basis = "list"
phase_a_max_usd = 20.0
phase_b_max_usd = 30.0

[roles.arm_family]
provider = "deepseek"
model_pro = "deepseek-chat"
base_url = "https://api.deepseek.com/v1"
api_key_env = "TEST_DEEPSEEK_API_KEY"
"""


def _write(tmp_path: Path, text: str, name: str = "gate.toml") -> Path:
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_valid_config_loads(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed TOML config loads: mode/price_basis/ceilings parse, arm_family is filled from
    the named env var, and every deferred role resolves empty."""
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    path = _write(tmp_path, _VALID_TOML)

    settings = load_gate_run_settings(path)

    assert settings.mode == "bring-up"
    assert settings.price_basis == "list"
    assert settings.phase_a_max_usd == 20.0
    assert settings.phase_b_max_usd == 30.0
    assert settings.arm_family.model_pro == "deepseek-chat"
    assert settings.arm_family.model_flash == "deepseek-chat"  # defaults to model_pro
    assert settings.arm_family.base_url == "https://api.deepseek.com/v1"
    assert settings.arm_family.api_key is not None
    assert settings.arm_family.api_key.get_secret_value() == "sk-test-123"
    assert settings.arm_family.price_per_mtok == DEEPSEEK_V4_PRO_LIST_PRICE
    assert settings.resolved_price_tables == {"deepseek": DEEPSEEK_V4_PRO_LIST_PRICE}
    # The deferred roles are present-but-empty, never omitted.
    assert settings.planter is None
    assert settings.converter_adversary is None
    assert settings.arm_d_prime is None
    assert settings.converter_panel == ()


def test_promo_basis_resolves_the_promo_price_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    toml_text = _VALID_TOML.replace('price_basis = "list"', 'price_basis = "promo"')
    path = _write(tmp_path, toml_text)

    settings = load_gate_run_settings(path)

    assert settings.price_basis == "promo"
    assert settings.arm_family.price_per_mtok == DEEPSEEK_V4_PRO_PROMO_PRICE
    assert settings.resolved_price_tables == {"deepseek": DEEPSEEK_V4_PRO_PROMO_PRICE}


def test_binding_mode_parses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    toml_text = _VALID_TOML.replace('mode = "bring-up"', 'mode = "binding"')
    path = _write(tmp_path, toml_text)

    settings = load_gate_run_settings(path)

    assert settings.mode == "binding"


def test_invalid_mode_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    toml_text = _VALID_TOML.replace('mode = "bring-up"', 'mode = "sideways"')
    path = _write(tmp_path, toml_text)

    with pytest.raises(ValueError, match="mode"):
        load_gate_run_settings(path)


def test_secret_literal_in_toml_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A literal `api_key` in the TOML (instead of `api_key_env`) is a load-time refusal —
    credentials are ENV-ONLY regardless of whether `api_key_env` is ALSO present."""
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    # Inject the literal into the arm_family table specifically.
    toml_text = _VALID_TOML.replace(
        'api_key_env = "TEST_DEEPSEEK_API_KEY"',
        'api_key_env = "TEST_DEEPSEEK_API_KEY"\napi_key = "sk-literal-secret-leaked-into-toml"',
    )
    path = _write(tmp_path, toml_text)

    with pytest.raises(ValueError, match="ENV-ONLY"):
        load_gate_run_settings(path)


def test_missing_env_var_is_a_clear_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A role naming an env var that is NOT set in the environment gets a clear, role-naming error
    (never a bare KeyError)."""
    monkeypatch.delenv("TEST_DEEPSEEK_API_KEY_UNSET", raising=False)
    toml_text = _VALID_TOML.replace(
        'api_key_env = "TEST_DEEPSEEK_API_KEY"', 'api_key_env = "TEST_DEEPSEEK_API_KEY_UNSET"'
    )
    path = _write(tmp_path, toml_text)

    with pytest.raises(ValueError, match="TEST_DEEPSEEK_API_KEY_UNSET"):
        load_gate_run_settings(path)


def test_missing_arm_family_is_refused(tmp_path: Path) -> None:
    """arm_family MUST be configured — it is the one fillable role in bring-up."""
    toml_text = """
mode = "bring-up"
price_basis = "list"
phase_a_max_usd = 20.0
phase_b_max_usd = 30.0
"""
    path = _write(tmp_path, toml_text)

    with pytest.raises(ValueError, match="arm_family"):
        load_gate_run_settings(path)


def test_unknown_provider_is_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A role naming a provider with no authored PriceTable is refused (S11 — never silently
    defaults to ZERO_PRICE_TABLE)."""
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    toml_text = _VALID_TOML.replace('provider = "deepseek"', 'provider = "some-unpriced-provider"')
    path = _write(tmp_path, toml_text)

    with pytest.raises(ValueError, match="no authored PriceTable"):
        load_gate_run_settings(path)


def test_deferred_roles_can_be_filled(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A role declared 'deferred' in bring-up still loads correctly when its TOML table IS filled
    (the schema doesn't hardcode which roles are empty — it just reflects what the TOML carries)."""
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    monkeypatch.setenv("TEST_PLANTER_API_KEY", "sk-planter-456")
    toml_text = (
        _VALID_TOML
        + """
[roles.planter]
provider = "deepseek"
model_pro = "deepseek-chat"
api_key_env = "TEST_PLANTER_API_KEY"
"""
    )
    path = _write(tmp_path, toml_text)

    settings = load_gate_run_settings(path)

    assert settings.planter is not None
    assert settings.planter.api_key is not None
    assert settings.planter.api_key.get_secret_value() == "sk-planter-456"


def test_converter_panel_list_round_trips(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """converter_panel is a LIST role; multiple TOML entries load into a tuple of configs."""
    monkeypatch.setenv("TEST_DEEPSEEK_API_KEY", "sk-test-123")
    monkeypatch.setenv("TEST_PANEL_KEY_1", "sk-panel-1")
    monkeypatch.setenv("TEST_PANEL_KEY_2", "sk-panel-2")
    toml_text = (
        _VALID_TOML
        + """
[[roles.converter_panel]]
provider = "deepseek"
model_pro = "deepseek-chat"
api_key_env = "TEST_PANEL_KEY_1"

[[roles.converter_panel]]
provider = "deepseek"
model_pro = "deepseek-chat"
api_key_env = "TEST_PANEL_KEY_2"
"""
    )
    path = _write(tmp_path, toml_text)

    settings = load_gate_run_settings(path)

    assert len(settings.converter_panel) == 2
    secrets = {
        c.api_key.get_secret_value() for c in settings.converter_panel if c.api_key is not None
    }
    assert secrets == {"sk-panel-1", "sk-panel-2"}
