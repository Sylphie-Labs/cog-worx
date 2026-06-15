"""Shared provider configuration types (CANON S4, S11).

``PriceTable`` and ``ProviderConfig`` are the shared, provider-agnostic config
surface that ALL model-provider adapters depend on.  Task 3.0d (OpenAI-compat
adapter) and any future provider MUST import from here — never duplicate these
types per-provider.

CANON cross-references
----------------------
- S4  Model-agnostic: thin ``Model`` seam; provider selectable per agent.
- S11 Cost bounded structurally: ``PriceTable.cost_usd`` feeds pre-call budget
      guards so the engine can enforce caps without touching the model.

Design note — price-table source-of-truth
-----------------------------------------
Where the price numbers actually live (in-code constants vs. a YAML/JSON data
file vs. a remote price API) is **open for Jim to decide**.  ``PriceTable`` is
intentionally a plain frozen pydantic model so it can be:

  a) Constructed inline with hardcoded defaults (current approach).
  b) Loaded from a YAML/TOML data file at startup (e.g. ``prices.yaml``).
  c) Fetched from a remote pricing endpoint and cached.

Whichever path is chosen, the type contract here stays unchanged.

# CARRY-FORWARD: Jim to decide price-table source-of-truth (in-code constants,
#   data file, or remote API).  See design note above.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, SecretStr

from cogworx.model.base import ModelTier, Usage

__all__ = [
    "ZERO_PRICE_TABLE",
    "PriceTable",
    "ProviderConfig",
]


class PriceTable(BaseModel):
    """Per-tier USD pricing in dollars per million tokens (frozen value type).

    Fields use *dollars per million tokens* ($/MTok) so the arithmetic is
    legible: 1 MTok == 1_000_000 tokens.

    Attributes
    ----------
    pro_input_usd_per_mtok:
        Input (prompt) price for the ``pro`` tier in $/MTok.
    pro_output_usd_per_mtok:
        Output (completion) price for the ``pro`` tier in $/MTok.
    flash_input_usd_per_mtok:
        Input price for the ``flash`` tier in $/MTok.
    flash_output_usd_per_mtok:
        Output price for the ``flash`` tier in $/MTok.
    """

    model_config = ConfigDict(frozen=True)

    pro_input_usd_per_mtok: float = Field(ge=0)
    pro_output_usd_per_mtok: float = Field(ge=0)
    flash_input_usd_per_mtok: float = Field(ge=0)
    flash_output_usd_per_mtok: float = Field(ge=0)

    def cost_usd(self, usage: Usage, tier: ModelTier) -> float:
        """Return the total USD cost for *usage* at *tier*.

        Parameters
        ----------
        usage:
            Token counts from a completed model call.
        tier:
            Which pricing tier to apply (``"pro"`` or ``"flash"``).

        Returns
        -------
        float
            Total cost in USD (input + output).  Never negative.
        """
        _MTOK = 1_000_000.0
        if tier == "pro":
            input_rate = self.pro_input_usd_per_mtok
            output_rate = self.pro_output_usd_per_mtok
        else:
            input_rate = self.flash_input_usd_per_mtok
            output_rate = self.flash_output_usd_per_mtok
        return (
            usage.prompt_tokens / _MTOK * input_rate + usage.completion_tokens / _MTOK * output_rate
        )


ZERO_PRICE_TABLE: PriceTable = PriceTable(
    pro_input_usd_per_mtok=0.0,
    pro_output_usd_per_mtok=0.0,
    flash_input_usd_per_mtok=0.0,
    flash_output_usd_per_mtok=0.0,
)


class ProviderConfig(BaseModel):
    """Shared provider configuration (NOT a settings object — no env-var reading).

    This is a **plain pydantic value model**, not a ``BaseSettings`` subclass.
    Environment-variable loading is the responsibility of the caller (e.g. a
    settings object in ``cogworx.adapters.config``).  Keep this type free of
    env-read machinery so it is composable and unit-testable without env state.

    Attributes
    ----------
    api_key:
        Provider API key.  ``None`` means "use environment default" (the SDK
        will pick it up from its own env-var, e.g. ``ANTHROPIC_API_KEY``).
    base_url:
        Override the API base URL (useful for proxies / local mocks).
        ``None`` means use the SDK default.
    model_pro:
        Model ID used for the ``"pro"`` tier.
    model_flash:
        Model ID used for the ``"flash"`` tier.
    timeout_s:
        Per-request timeout in seconds passed to the transport layer.
    max_transport_retries:
        Maximum number of transport-level retries (429 / 5xx).  Distinct from
        the engine's semantic retries (CANON S8 graceful degradation): these
        retries never re-run business logic, they only re-send the same HTTP
        request after a backoff.
    price_per_mtok:
        Pricing table used to compute ``Usage.cost_usd`` after each call.
    """

    model_config = ConfigDict(frozen=True)

    api_key: SecretStr | None = None
    base_url: str | None = None
    model_pro: str
    model_flash: str
    timeout_s: float = 60.0
    max_transport_retries: int = 3
    price_per_mtok: PriceTable = ZERO_PRICE_TABLE
    max_output_tokens: int = Field(default=8192, ge=1)
