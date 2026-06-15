"""Tests for ProviderConfig optional price table (Part 2 simplification)."""

from __future__ import annotations

import pytest

from cogworx.model.base import Usage
from cogworx.model.providers.config import ZERO_PRICE_TABLE, ProviderConfig


def test_provider_config_without_price_per_mtok_defaults_to_zero() -> None:
    """A ProviderConfig built WITHOUT price_per_mtok gets ZERO_PRICE_TABLE."""
    config = ProviderConfig(model_pro="m-pro", model_flash="m-flash")
    assert config.price_per_mtok == ZERO_PRICE_TABLE
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    assert config.price_per_mtok.cost_usd(usage, "pro") == pytest.approx(0.0)
    assert config.price_per_mtok.cost_usd(usage, "flash") == pytest.approx(0.0)


def test_provider_config_with_real_price_table_unchanged() -> None:
    """A ProviderConfig with an explicit price table is unchanged."""
    from cogworx.model.providers.config import PriceTable

    table = PriceTable(
        pro_input_usd_per_mtok=3.0,
        pro_output_usd_per_mtok=15.0,
        flash_input_usd_per_mtok=0.25,
        flash_output_usd_per_mtok=1.25,
    )
    config = ProviderConfig(model_pro="m-pro", model_flash="m-flash", price_per_mtok=table)
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=0)
    assert config.price_per_mtok.cost_usd(usage, "pro") == pytest.approx(3.0)
