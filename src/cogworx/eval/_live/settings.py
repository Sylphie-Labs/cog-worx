"""Live GATE-run configuration: price authoring + the S11 zero-price refusal (Pod 4.4-live L0/L1).

This module is the price-table source-of-truth (closing CF at
``model/providers/config.py:26-27`` as "TOML run-config") and the TOML-loaded
:class:`GateRunSettings` the 1-key DeepSeek V4 Pro bring-up driver reads. It is the FIRST module in
the private :mod:`cogworx.eval._live` composition root (see that package's docstring for the S4/T5
posture).

S11 — the zero-price refusal (the load-bearing fix)
-----------------------------------------------------
``ProviderConfig.price_per_mtok`` defaults to
:data:`~cogworx.model.providers.config.ZERO_PRICE_TABLE` (``model/providers/config.py:142``), which
makes :meth:`~cogworx.model.providers.config.PriceTable.cost_usd` identically ``0.0`` — a
:class:`~cogworx.cost.budget.BudgetGuard` built on that cost NEVER trips, so the ``max_usd`` ceiling
is theater (flagged during live-run scoping, 2026-07-02). :func:`assert_priced` is the structural
refusal: any role whose resolved :class:`~cogworx.model.providers.config.ProviderConfig` still
carries the zero default raises loudly, naming the role, rather than letting an unpriced role run.

Credentials are ENV-ONLY
-------------------------
The TOML run-config never carries a secret literal. Each role names the ENV VAR holding its API key
(``api_key_env``); :func:`load_gate_run_settings` reads the actual secret from ``os.environ`` at
load time. A role table that carries a literal ``api_key`` key is a load-time refusal (regardless of
its value — the wall is the field name, not a heuristic over the string). This module reads
``os.environ`` directly; it does NOT use pydantic ``BaseSettings`` env-magic (``ProviderConfig`` is
explicitly not an env-reading settings type — see its docstring).

The role schema (present-but-empty deferred roles)
-----------------------------------------------------
Only ``roles.arm_family`` is fillable in the 1-key bring-up (``mode = "bring-up"``); the other four
roles the full nightly run will eventually need — ``planter``, ``converter_panel`` (a list),
``converter_adversary``, ``arm_d_prime`` — are declared on :class:`GateRunSettings` NOW (never
omitted from the schema) but resolve to ``None`` / ``()`` when their TOML table is absent or empty.
A role table that IS present but malformed (e.g. missing ``provider``) is a load-time error, not a
silent defer — "present but empty" is the only shape that means "deferred".

Pure stdlib (:mod:`tomllib`) + pydantic; no new dependency (CANON S2). No concrete provider adapter
import — this module builds :class:`~cogworx.model.providers.config.ProviderConfig` value objects
only; wiring them to a live :class:`~cogworx.model.base.Model` is a later module in this package.

Contract changelog (CANON §6.1):
  - 2026-07-02 (Pod 4.4-live L0/L1): initial — DeepSeek V4 Pro list/promo ``PriceTable``\\ s,
    ``assert_priced``, and the TOML-loaded ``GateRunSettings`` (``load_gate_run_settings``). New
    module; no existing callers. Additive new public surface only.
"""

from __future__ import annotations

import os
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, SecretStr

from cogworx.model.providers.config import ZERO_PRICE_TABLE, PriceTable, ProviderConfig

__all__ = [
    "DEEPSEEK_V4_PRO_LIST_PRICE",
    "DEEPSEEK_V4_PRO_PROMO_PRICE",
    "GateRunMode",
    "GateRunSettings",
    "PriceBasis",
    "assert_priced",
    "load_gate_run_settings",
]

PriceBasis = Literal["list", "promo"]
GateRunMode = Literal["bring-up", "binding"]

# ---------------------------------------------------------------------------
# L0 — DeepSeek V4 Pro price authoring (both bases; pro/flash tiers identical —
# no distinct flash model id is configured for the bring-up).
# ---------------------------------------------------------------------------

DEEPSEEK_V4_PRO_LIST_PRICE: PriceTable = PriceTable(
    pro_input_usd_per_mtok=1.74,
    pro_output_usd_per_mtok=3.48,
    flash_input_usd_per_mtok=1.74,
    flash_output_usd_per_mtok=3.48,
)
"""DeepSeek V4 Pro list (rack-rate) price. The default basis (S11-honest): an over-estimate trips
the ``BudgetGuard`` ceiling early, never late, and a promo's expiry can never silently un-bound a
run that assumed it was still active."""

DEEPSEEK_V4_PRO_PROMO_PRICE: PriceTable = PriceTable(
    pro_input_usd_per_mtok=0.435,
    pro_output_usd_per_mtok=0.87,
    flash_input_usd_per_mtok=0.435,
    flash_output_usd_per_mtok=0.87,
)
"""DeepSeek V4 Pro promotional (cache-miss) price — 75% off list. Opt-in only
(``price_basis = "promo"``): using it un-audited risks the ceiling being sized against a rate that
has since expired."""

_KNOWN_PRICE_TABLES: Mapping[str, Mapping[PriceBasis, PriceTable]] = {
    "deepseek": {
        "list": DEEPSEEK_V4_PRO_LIST_PRICE,
        "promo": DEEPSEEK_V4_PRO_PROMO_PRICE,
    },
}


def _price_table_for(provider: str, basis: PriceBasis, *, role: str) -> PriceTable:
    """Look up the authored :class:`PriceTable` for *provider* at *basis*.

    Refuses loudly (naming the *role*) for any provider with no authored table — this is what
    structurally confines the bring-up to DeepSeek: no other provider has an entry in
    :data:`_KNOWN_PRICE_TABLES` yet, so any other ``roles.<name>.provider`` fails here rather than
    silently falling back to :data:`~cogworx.model.providers.config.ZERO_PRICE_TABLE`.
    """
    by_basis = _KNOWN_PRICE_TABLES.get(provider)
    if by_basis is None or basis not in by_basis:
        raise ValueError(
            f"GateRunSettings: role {role!r} names provider {provider!r}, which has no authored "
            f"PriceTable for basis {basis!r}. Author one in cogworx.eval._live.settings before "
            "wiring this role (CANON S11 — an unpriced role must never run)."
        )
    return by_basis[basis]


def assert_priced(config: ProviderConfig, *, role: str) -> None:
    """Refuse loudly a *config* whose price table is the zero default (CANON S11).

    ``cost_usd`` computed off :data:`~cogworx.model.providers.config.ZERO_PRICE_TABLE` is
    identically ``0.0``, so a :class:`~cogworx.cost.budget.BudgetGuard` built on it never trips —
    the ``max_usd`` ceiling silently becomes theater. Raises naming *role* and the resolved
    ``model_pro`` id so the operator can find the offending TOML section immediately.
    """
    if config.price_per_mtok == ZERO_PRICE_TABLE:
        raise ValueError(
            f"GateRunSettings: role {role!r} (model_pro={config.model_pro!r}) carries "
            "ZERO_PRICE_TABLE — cost_usd would be identically 0.0 and BudgetGuard.max_usd would "
            "never trip (CANON S11). Configure a real PriceTable for this role before running."
        )


# ---------------------------------------------------------------------------
# L1 — GateRunSettings (TOML-loaded; credentials ENV-ONLY)
# ---------------------------------------------------------------------------


class GateRunSettings(BaseModel):
    """The Phase-4 GATE run configuration, loaded from a TOML run-config (Pod 4.4-live L1).

    Attributes
    ----------
    mode:
        ``"bring-up"`` (1-key, infra smoke-test — see the live-run scoping memo: the planter must be
        a family distinct from the arms, so a bring-up run is NOT a binding verdict) or
        ``"binding"`` (the full multi-family nightly run).
    price_basis:
        Which authored price basis (``"list"`` or ``"promo"``) was used to resolve every filled
        role's :attr:`~cogworx.model.providers.config.ProviderConfig.price_per_mtok`.
    resolved_price_tables:
        provider name -> the :class:`~cogworx.model.providers.config.PriceTable` actually resolved
        for it this load, keyed by the ``provider`` string named in each filled role's TOML table.
        Carried alongside ``price_basis`` for at-a-glance audit (the same tables are also embedded
        in each role's ``ProviderConfig.price_per_mtok``; this is a flat summary, not a second
        source of truth).
    phase_a_max_usd:
        The Phase-A (corpus-production) :class:`~cogworx.cost.budget.BudgetGuard` ceiling.
    phase_b_max_usd:
        The Phase-B (5-arm run) :class:`~cogworx.cost.budget.BudgetGuard` ceiling. Phase A and
        Phase B mint SEPARATE guards — see the live-run scoping memo.
    arm_family:
        The one role fillable in ``mode = "bring-up"`` — the model family driving the arms.
    planter, converter_adversary, arm_d_prime:
        Deferred roles (present on the schema, ``None`` until a further family's credentials are
        configured).
    converter_panel:
        The deferred multi-model K->O converter panel (empty until configured).
    """

    model_config = ConfigDict(frozen=True)

    mode: GateRunMode
    price_basis: PriceBasis
    resolved_price_tables: Mapping[str, PriceTable]
    phase_a_max_usd: float
    phase_b_max_usd: float
    arm_family: ProviderConfig
    planter: ProviderConfig | None = None
    converter_panel: tuple[ProviderConfig, ...] = ()
    converter_adversary: ProviderConfig | None = None
    arm_d_prime: ProviderConfig | None = None


def _require_mode(raw: Mapping[str, object]) -> GateRunMode:
    value = raw.get("mode")
    if value == "bring-up":
        return "bring-up"
    if value == "binding":
        return "binding"
    raise ValueError(f"GateRunSettings: 'mode' must be 'bring-up' or 'binding', got {value!r}")


def _require_price_basis(raw: Mapping[str, object]) -> PriceBasis:
    value = raw.get("price_basis", "list")
    if value == "list":
        return "list"
    if value == "promo":
        return "promo"
    raise ValueError(f"GateRunSettings: 'price_basis' must be 'list' or 'promo', got {value!r}")


def _require_float(raw: Mapping[str, object], key: str) -> float:
    value = raw.get(key)
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"GateRunSettings: '{key}' must be a number, got {value!r}")
    return float(value)


def _require_str(table: Mapping[str, object], key: str, *, role: str) -> str:
    value = table.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"GateRunSettings: role {role!r} is missing required string field {key!r}")
    return value


def _optional_str(table: Mapping[str, object], key: str) -> str | None:
    value = table.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError(f"GateRunSettings: field {key!r} must be a string, got {value!r}")
    return value


def _role_table(roles: Mapping[str, object], name: str) -> dict[str, object]:
    value = roles.get(name)
    if value is None:
        return {}
    if not isinstance(value, dict):
        got = type(value).__name__
        raise ValueError(f"GateRunSettings: 'roles.{name}' must be a table, got {got}")
    return value


def _role_table_list(roles: Mapping[str, object], name: str) -> list[dict[str, object]]:
    value = roles.get(name)
    if value is None:
        return []
    if not isinstance(value, list):
        got = type(value).__name__
        raise ValueError(f"GateRunSettings: 'roles.{name}' must be an array of tables, got {got}")
    tables: list[dict[str, object]] = []
    for entry in value:
        if not isinstance(entry, dict):
            raise ValueError(f"GateRunSettings: 'roles.{name}' entries must be tables")
        if entry:
            tables.append(entry)
    return tables


def _read_env_secret(env_var: str, *, role: str) -> SecretStr:
    value = os.environ.get(env_var)
    if value is None:
        raise ValueError(
            f"GateRunSettings: role {role!r} names env var {env_var!r} for its API key, but that "
            "variable is not set. Export it before loading this config (credentials are ENV-ONLY)."
        )
    return SecretStr(value)


def _load_role(
    table: Mapping[str, object], *, role: str, price_basis: PriceBasis
) -> tuple[ProviderConfig, str] | None:
    """Load one role's table into a ``(ProviderConfig, provider)`` pair, or ``None`` if deferred
    (an empty table). Refuses a secret literal, an unknown provider, or a missing env var — all
    load-time, all naming *role*."""
    if not table:
        return None
    if "api_key" in table:
        raise ValueError(
            f"GateRunSettings: role {role!r} carries a literal 'api_key' key in the TOML — "
            "credentials are ENV-ONLY. Name the env var via 'api_key_env' instead."
        )
    provider = _require_str(table, "provider", role=role)
    price_table = _price_table_for(provider, price_basis, role=role)
    model_pro = _require_str(table, "model_pro", role=role)
    model_flash = _optional_str(table, "model_flash") or model_pro
    base_url = _optional_str(table, "base_url")
    api_key_env = _require_str(table, "api_key_env", role=role)
    api_key = _read_env_secret(api_key_env, role=role)
    config = ProviderConfig(
        api_key=api_key,
        base_url=base_url,
        model_pro=model_pro,
        model_flash=model_flash,
        price_per_mtok=price_table,
    )
    assert_priced(config, role=role)
    return config, provider


def load_gate_run_settings(path: str | Path) -> GateRunSettings:
    """Load a :class:`GateRunSettings` from a TOML run-config at *path*.

    Credentials are ENV-ONLY: every role names its API key's env var via ``api_key_env``; a
    literal ``api_key`` in the TOML is a load-time refusal. ``roles.arm_family`` MUST be
    configured (the 1-key bring-up role); ``roles.planter`` / ``roles.converter_panel`` /
    ``roles.converter_adversary`` / ``roles.arm_d_prime`` resolve to ``None`` / ``()`` when absent
    or empty (the deferred roles).
    """
    with open(path, "rb") as f:
        raw: dict[str, Any] = tomllib.load(f)

    mode = _require_mode(raw)
    price_basis = _require_price_basis(raw)
    phase_a_max_usd = _require_float(raw, "phase_a_max_usd")
    phase_b_max_usd = _require_float(raw, "phase_b_max_usd")

    roles_raw = raw.get("roles", {})
    if not isinstance(roles_raw, dict):
        got = type(roles_raw).__name__
        raise ValueError(f"GateRunSettings: 'roles' must be a table, got {got}")

    resolved_price_tables: dict[str, PriceTable] = {}

    def _fill(role_name: str, table: dict[str, object]) -> ProviderConfig | None:
        loaded = _load_role(table, role=role_name, price_basis=price_basis)
        if loaded is None:
            return None
        config, provider = loaded
        resolved_price_tables[provider] = config.price_per_mtok
        return config

    arm_family_table = _role_table(roles_raw, "arm_family")
    if not arm_family_table:
        raise ValueError(
            "GateRunSettings: 'roles.arm_family' must be configured — it is the ONE fillable role "
            "in 1-key bring-up (CANON S4/S11)."
        )
    arm_family = _fill("arm_family", arm_family_table)
    assert arm_family is not None  # a non-empty table always yields a config or raises above

    planter = _fill("planter", _role_table(roles_raw, "planter"))
    converter_adversary = _fill(
        "converter_adversary", _role_table(roles_raw, "converter_adversary")
    )
    arm_d_prime = _fill("arm_d_prime", _role_table(roles_raw, "arm_d_prime"))
    converter_panel = tuple(
        config
        for t in _role_table_list(roles_raw, "converter_panel")
        if (config := _fill("converter_panel", t)) is not None
    )

    return GateRunSettings(
        mode=mode,
        price_basis=price_basis,
        resolved_price_tables=resolved_price_tables,
        phase_a_max_usd=phase_a_max_usd,
        phase_b_max_usd=phase_b_max_usd,
        arm_family=arm_family,
        planter=planter,
        converter_panel=converter_panel,
        converter_adversary=converter_adversary,
        arm_d_prime=arm_d_prime,
    )
