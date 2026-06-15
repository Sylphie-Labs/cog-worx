"""Live integration tests for ClaudeModel (CANON S4, S9, S11).

These tests hit the real Anthropic API and are SKIPPED by default unless the
environment variable ``ANTHROPIC_API_KEY`` is set.

Run with::

    pytest tests/integration/test_claude_model_live.py -m integration

Or to run everything including integration::

    pytest -m "integration"

Gates
-----
- ``ruff check`` — style/lint
- ``mypy --strict`` — type correctness
- ``pytest`` (unit only, no live key needed) — correctness without network
"""

from __future__ import annotations

import os

import pytest

from cogworx.model.base import ChatMessage, ModelResponse, ToolSpec
from cogworx.model.providers.claude import ClaudeModel
from cogworx.model.providers.config import PriceTable, ProviderConfig

# ---------------------------------------------------------------------------
# Skip guard — no live key, skip all tests in this module.
# ---------------------------------------------------------------------------

_API_KEY: str | None = os.environ.get("ANTHROPIC_API_KEY")
pytestmark = pytest.mark.integration

if not _API_KEY:
    pytest.skip(
        "ANTHROPIC_API_KEY not set — skipping live Claude integration tests.",
        allow_module_level=True,
    )

# ---------------------------------------------------------------------------
# Shared live config (flash tier to keep cost low during CI)
# ---------------------------------------------------------------------------

_PRICE_TABLE = PriceTable(
    pro_input_usd_per_mtok=5.00,
    pro_output_usd_per_mtok=25.00,
    flash_input_usd_per_mtok=1.00,
    flash_output_usd_per_mtok=5.00,
)

_CONFIG = ProviderConfig(
    model_pro="claude-opus-4-8",
    model_flash="claude-haiku-4-5",
    price_per_mtok=_PRICE_TABLE,
    max_transport_retries=2,
)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def live_model() -> ClaudeModel:
    """Real ClaudeModel backed by the Anthropic API."""
    return ClaudeModel(_CONFIG)


async def test_live_simple_completion(live_model: ClaudeModel) -> None:
    """Smoke test: a basic user message returns a non-empty text response."""
    resp: ModelResponse = await live_model.complete(
        messages=[ChatMessage(role="user", content="Reply with exactly one word: hello")],
        tier="flash",  # use cheap tier for live tests
    )
    assert resp.text is not None
    assert len(resp.text) > 0
    assert resp.finish_reason in ("end_turn", "stop", "max_tokens")


async def test_live_usage_populated(live_model: ClaudeModel) -> None:
    """Usage fields (tokens, cost) must be positive after a real call."""
    resp: ModelResponse = await live_model.complete(
        messages=[ChatMessage(role="user", content="Say hi")],
        tier="flash",
    )
    assert resp.usage.prompt_tokens > 0
    assert resp.usage.completion_tokens > 0
    assert resp.usage.cost_usd > 0.0
    assert resp.usage.latency_ms > 0.0


async def test_live_system_prompt_honoured(live_model: ClaudeModel) -> None:
    """A system prompt should influence the model's reply."""
    resp: ModelResponse = await live_model.complete(
        messages=[
            ChatMessage(role="system", content="Always answer in exactly one word."),
            ChatMessage(role="user", content="What colour is the sky?"),
        ],
        tier="flash",
    )
    assert resp.text is not None
    # One-word answer — very loose check (model may add punctuation)
    words = resp.text.strip().split()
    assert 1 <= len(words) <= 5, f"Expected ~1 word, got: {resp.text!r}"


async def test_live_tool_call(live_model: ClaudeModel) -> None:
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
    resp: ModelResponse = await live_model.complete(
        messages=[ChatMessage(role="user", content="What is the weather in Paris?")],
        tools=[weather_tool],
        tier="flash",
    )
    assert len(resp.tool_calls) >= 1
    tc = resp.tool_calls[0]
    assert tc.name == "get_weather"
    assert "city" in tc.arguments


async def test_live_model_id_matches_config(live_model: ClaudeModel) -> None:
    """ModelResponse.model_id must match the config model for the chosen tier."""
    resp: ModelResponse = await live_model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="flash",
    )
    assert resp.model_id == _CONFIG.model_flash


async def test_live_pro_tier(live_model: ClaudeModel) -> None:
    """Pro tier smoke test — uses the more expensive model."""
    resp: ModelResponse = await live_model.complete(
        messages=[ChatMessage(role="user", content="Say exactly: pro tier works")],
        tier="pro",
    )
    assert resp.text is not None
    assert resp.model_id == _CONFIG.model_pro
