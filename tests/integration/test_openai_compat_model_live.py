"""Live integration tests for OpenAICompatModel (CANON S4, S9, S11).

These tests hit real endpoints and are SKIPPED by default unless the relevant
environment variables are set.

Supported targets
-----------------
- OpenAI production  — requires ``OPENAI_API_KEY``
- DeepSeek           — requires ``DEEPSEEK_API_KEY``  (DEFERRED — see carry-forward)
- Ollama             — requires ``OLLAMA_BASE_URL``    (DEFERRED — see carry-forward)

Run a specific target::

    pytest tests/integration/test_openai_compat_model_live.py \
        ::test_openai_live_simple_completion -m integration

CARRY-FORWARD
-------------
DeepSeek and Ollama live rows are deferred pending live endpoint access in a
future session.  The skip guards and config stubs are in place; activate them
by removing the ``pytest.skip`` call and ensuring the environment variable is
set.
"""

from __future__ import annotations

import os

import pytest

from cogworx.model.base import ChatMessage, ModelResponse, ToolSpec
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.model.providers.openai_compat import (
    DEEPSEEK_CAPABILITIES,
    OLLAMA_CAPABILITIES,
    OPENAI_CAPABILITIES,
    OpenAICompatModel,
)

pytestmark = pytest.mark.integration

# ---------------------------------------------------------------------------
# OpenAI live tests
# ---------------------------------------------------------------------------

_OPENAI_API_KEY: str | None = os.environ.get("OPENAI_API_KEY")

_OPENAI_PRICE_TABLE = PriceTable(
    pro_input_usd_per_mtok=5.00,
    pro_output_usd_per_mtok=15.00,
    flash_input_usd_per_mtok=0.15,
    flash_output_usd_per_mtok=0.60,
)

_OPENAI_CONFIG = ProviderConfig(
    model_pro="gpt-4o",
    model_flash="gpt-4o-mini",
    price_per_mtok=_OPENAI_PRICE_TABLE,
    max_transport_retries=2,
)


@pytest.fixture(scope="module")
def openai_live_model() -> OpenAICompatModel:
    """Real OpenAICompatModel backed by the OpenAI API."""
    if not _OPENAI_API_KEY:
        pytest.skip("OPENAI_API_KEY not set — skipping live OpenAI integration tests.")
    from pydantic import SecretStr

    config = ProviderConfig(
        model_pro=_OPENAI_CONFIG.model_pro,
        model_flash=_OPENAI_CONFIG.model_flash,
        api_key=SecretStr(_OPENAI_API_KEY),
        price_per_mtok=_OPENAI_PRICE_TABLE,
        max_transport_retries=2,
    )
    return OpenAICompatModel(config, capabilities=OPENAI_CAPABILITIES)


async def test_openai_live_simple_completion(openai_live_model: OpenAICompatModel) -> None:
    """Smoke test: a basic user message returns a non-empty text response."""
    resp: ModelResponse = await openai_live_model.complete(
        messages=[ChatMessage(role="user", content="Reply with exactly one word: hello")],
        tier="flash",
    )
    assert resp.text is not None
    assert len(resp.text) > 0
    assert resp.finish_reason in ("stop", "length")


async def test_openai_live_usage_populated(openai_live_model: OpenAICompatModel) -> None:
    """Usage fields (tokens, cost) must be positive after a real call."""
    resp: ModelResponse = await openai_live_model.complete(
        messages=[ChatMessage(role="user", content="Say hi")],
        tier="flash",
    )
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert resp.usage.cost_usd > 0.0
    assert resp.usage.latency_ms > 0.0


async def test_openai_live_tool_call(openai_live_model: OpenAICompatModel) -> None:
    """The model should call the provided tool when appropriate."""
    weather_tool = ToolSpec(
        name="get_weather",
        description="Get the current weather for a city.",
        input_schema={
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
        },
    )
    resp: ModelResponse = await openai_live_model.complete(
        messages=[ChatMessage(role="user", content="What is the weather in Tokyo?")],
        tools=[weather_tool],
        tier="flash",
    )
    assert len(resp.tool_calls) >= 1
    tc = resp.tool_calls[0]
    assert tc.name == "get_weather"
    assert "city" in tc.arguments


async def test_openai_live_model_id_matches_config(
    openai_live_model: OpenAICompatModel,
) -> None:
    """ModelResponse.model_id must match the config model for the chosen tier."""
    resp: ModelResponse = await openai_live_model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="flash",
    )
    assert resp.model_id == _OPENAI_CONFIG.model_flash


# ---------------------------------------------------------------------------
# DeepSeek live tests (DEFERRED)
# ---------------------------------------------------------------------------

# CARRY-FORWARD: DeepSeek live tests deferred pending live endpoint access.
# To activate: remove the pytest.skip below, ensure DEEPSEEK_API_KEY is set,
# and adjust model IDs / prices as needed.

_DEEPSEEK_API_KEY: str | None = os.environ.get("DEEPSEEK_API_KEY")

_DEEPSEEK_PRICE_TABLE = PriceTable(
    pro_input_usd_per_mtok=0.27,
    pro_output_usd_per_mtok=1.10,
    flash_input_usd_per_mtok=0.14,
    flash_output_usd_per_mtok=0.28,
)

_DEEPSEEK_CONFIG = ProviderConfig(
    model_pro="deepseek-chat",
    model_flash="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    price_per_mtok=_DEEPSEEK_PRICE_TABLE,
    max_transport_retries=2,
)


@pytest.fixture(scope="module")
def deepseek_live_model() -> OpenAICompatModel:
    """Real OpenAICompatModel backed by the DeepSeek API (deferred)."""
    pytest.skip(
        "DeepSeek live integration tests deferred — set DEEPSEEK_API_KEY and "
        "remove this skip to activate.  (CARRY-FORWARD)"
    )
    # Unreachable; here for static analysis.
    from pydantic import SecretStr

    config = ProviderConfig(
        model_pro=_DEEPSEEK_CONFIG.model_pro,
        model_flash=_DEEPSEEK_CONFIG.model_flash,
        api_key=SecretStr(_DEEPSEEK_API_KEY or ""),
        base_url=_DEEPSEEK_CONFIG.base_url,
        price_per_mtok=_DEEPSEEK_PRICE_TABLE,
        max_transport_retries=2,
    )
    return OpenAICompatModel(config, capabilities=DEEPSEEK_CAPABILITIES)


async def test_deepseek_live_simple_completion(deepseek_live_model: OpenAICompatModel) -> None:
    resp: ModelResponse = await deepseek_live_model.complete(
        messages=[ChatMessage(role="user", content="Say hello")],
        tier="flash",
    )
    assert resp.text is not None
    assert len(resp.text) > 0


# ---------------------------------------------------------------------------
# Ollama live tests (DEFERRED)
# ---------------------------------------------------------------------------

# CARRY-FORWARD: Ollama live tests deferred pending local Ollama availability.
# To activate: remove the pytest.skip below, ensure OLLAMA_BASE_URL is set
# (default: http://localhost:11434/v1), and adjust model IDs as needed.

_OLLAMA_BASE_URL: str = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")

_OLLAMA_PRICE_TABLE = PriceTable(
    pro_input_usd_per_mtok=0.0,
    pro_output_usd_per_mtok=0.0,
    flash_input_usd_per_mtok=0.0,
    flash_output_usd_per_mtok=0.0,
)

_OLLAMA_CONFIG = ProviderConfig(
    model_pro="llama3",
    model_flash="llama3",
    base_url=_OLLAMA_BASE_URL,
    price_per_mtok=_OLLAMA_PRICE_TABLE,
    max_transport_retries=1,
)


@pytest.fixture(scope="module")
def ollama_live_model() -> OpenAICompatModel:
    """Real OpenAICompatModel backed by a local Ollama instance (deferred)."""
    pytest.skip(
        "Ollama live integration tests deferred — start Ollama locally and "
        "set OLLAMA_BASE_URL (or accept default http://localhost:11434/v1), "
        "then remove this skip.  (CARRY-FORWARD)"
    )
    # Unreachable; here for static analysis.
    return OpenAICompatModel(_OLLAMA_CONFIG, capabilities=OLLAMA_CAPABILITIES)


async def test_ollama_live_simple_completion(ollama_live_model: OpenAICompatModel) -> None:
    """Smoke test: Ollama responds (degraded capabilities: no tools, no structured output)."""
    resp: ModelResponse = await ollama_live_model.complete(
        messages=[ChatMessage(role="user", content="Say hello")],
        tier="pro",
    )
    assert resp.text is not None
    assert len(resp.text) > 0


async def test_ollama_live_tools_silently_omitted(ollama_live_model: OpenAICompatModel) -> None:
    """Verifies tools are not forwarded to Ollama (capabilities.tools=False)."""
    assert ollama_live_model.capabilities.tools is False
    tool = ToolSpec(
        name="dummy",
        description="A dummy tool",
        input_schema={"type": "object", "properties": {}},
    )
    # Should not raise even though Ollama cannot handle tools.
    resp: ModelResponse = await ollama_live_model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tools=[tool],
        tier="pro",
    )
    assert resp.text is not None
