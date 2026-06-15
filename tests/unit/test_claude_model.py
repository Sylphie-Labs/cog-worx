"""Unit tests for ClaudeModel (CANON S4, S9, S11).

All tests use a FAKE injected client — zero network calls.  The fake satisfies
the narrow contract ``client.messages.create(**kwargs) -> raw_response`` that
ClaudeModel uses internally.

Coverage targets
----------------
- Static capability declaration (S9: never probe, never self-report).
- Message/tool mapping round-trips (system prompt, tool_use, tool_result).
- Cost computation via PriceTable (S11).
- Transport-retry on simulated 429 / 5xx, bounded by max_transport_retries.
- Model ID selection per tier.
- count_tokens heuristic.
- Protocol satisfaction (isinstance check).
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from cogworx.model.base import (
    ChatMessage,
    Model,
    ModelCapabilities,
    ModelResponse,
    ToolCall,
    ToolSpec,
    Usage,
)
from cogworx.model.providers.claude import ClaudeModel, _backoff, _extract_status
from cogworx.model.providers.config import PriceTable, ProviderConfig

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

PRO_INPUT = 5.0
PRO_OUTPUT = 25.0
FLASH_INPUT = 1.0
FLASH_OUTPUT = 5.0

PRICE_TABLE = PriceTable(
    pro_input_usd_per_mtok=PRO_INPUT,
    pro_output_usd_per_mtok=PRO_OUTPUT,
    flash_input_usd_per_mtok=FLASH_INPUT,
    flash_output_usd_per_mtok=FLASH_OUTPUT,
)

CONFIG = ProviderConfig(
    model_pro="claude-opus-4-8",
    model_flash="claude-haiku-4-5",
    price_per_mtok=PRICE_TABLE,
    max_transport_retries=2,
)


def _make_raw_response(
    *,
    text: str = "hello",
    stop_reason: str = "end_turn",
    input_tokens: int = 100,
    output_tokens: int = 50,
    tool_use_blocks: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics an Anthropic ``Message``."""
    raw = MagicMock()
    raw.stop_reason = stop_reason
    raw.usage = MagicMock()
    raw.usage.input_tokens = input_tokens
    raw.usage.output_tokens = output_tokens

    content_blocks: list[MagicMock] = []

    text_block = MagicMock()
    text_block.type = "text"
    text_block.text = text
    content_blocks.append(text_block)

    for tu in tool_use_blocks or []:
        tb = MagicMock()
        tb.type = "tool_use"
        tb.id = tu["id"]
        tb.name = tu["name"]
        tb.input = tu["input"]
        content_blocks.append(tb)

    raw.content = content_blocks
    return raw


def _make_client(raw: MagicMock | None = None) -> MagicMock:
    """Build a fake client whose messages.create returns *raw*."""
    if raw is None:
        raw = _make_raw_response()
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=raw)
    return client


def _make_model(client: MagicMock | None = None, config: ProviderConfig = CONFIG) -> ClaudeModel:
    if client is None:
        client = _make_client()
    return ClaudeModel(config, client=client)


# ---------------------------------------------------------------------------
# Protocol satisfaction (S9 — static, not probed)
# ---------------------------------------------------------------------------


def test_claude_model_satisfies_model_protocol() -> None:
    """ClaudeModel must satisfy the runtime_checkable Model Protocol."""
    model = _make_model()
    assert isinstance(model, Model)


# ---------------------------------------------------------------------------
# Capability declaration (S9 — STATIC, never probe)
# ---------------------------------------------------------------------------


def test_capabilities_are_static() -> None:
    """capabilities must be the same class-level object every time — no probe."""
    m1 = _make_model()
    m2 = _make_model()
    assert m1.capabilities is ClaudeModel._CAPABILITIES
    assert m2.capabilities is ClaudeModel._CAPABILITIES


def test_capabilities_values() -> None:
    model = _make_model()
    caps: ModelCapabilities = model.capabilities
    assert caps.structured_output is True
    assert caps.tools is True
    assert caps.caching is True
    assert caps.streaming is True
    assert caps.logprobs is False  # not declared


# ---------------------------------------------------------------------------
# Model ID selection by tier
# ---------------------------------------------------------------------------


async def test_pro_tier_uses_model_pro() -> None:
    client = _make_client()
    model = _make_model(client=client)
    await model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="pro",
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == CONFIG.model_pro


async def test_flash_tier_uses_model_flash() -> None:
    client = _make_client()
    model = _make_model(client=client)
    await model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="flash",
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["model"] == CONFIG.model_flash


# ---------------------------------------------------------------------------
# Message mapping
# ---------------------------------------------------------------------------


async def test_system_message_extracted_to_system_param() -> None:
    client = _make_client()
    model = _make_model(client=client)
    await model.complete(
        messages=[
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Hello"),
        ]
    )
    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["system"] == "Be concise."
    # Only the user turn in messages list
    assert any(m["role"] == "user" for m in kwargs["messages"])
    assert not any(m["role"] == "system" for m in kwargs["messages"])


async def test_no_system_message_omits_system_param() -> None:
    client = _make_client()
    model = _make_model(client=client)
    await model.complete(messages=[ChatMessage(role="user", content="Hello")])
    kwargs = client.messages.create.call_args.kwargs
    assert "system" not in kwargs


async def test_tool_role_maps_to_tool_result_block() -> None:
    client = _make_client()
    model = _make_model(client=client)
    await model.complete(
        messages=[
            ChatMessage(role="user", content="Call the tool"),
            ChatMessage(role="tool", content='{"result": 42}', tool_call_id="tc_001"),
        ]
    )
    kwargs = client.messages.create.call_args.kwargs
    # The tool result must appear as a user-role message with tool_result content block.
    tool_result_msg = next(
        (m for m in kwargs["messages"] if m["role"] == "user" and isinstance(m["content"], list)),
        None,
    )
    assert tool_result_msg is not None
    block = tool_result_msg["content"][0]
    assert block["type"] == "tool_result"
    assert block["tool_use_id"] == "tc_001"
    assert block["content"] == '{"result": 42}'


# ---------------------------------------------------------------------------
# Tool mapping
# ---------------------------------------------------------------------------


async def test_tools_sent_to_api() -> None:
    client = _make_client()
    model = _make_model(client=client)
    tool = ToolSpec(
        name="get_weather",
        description="Fetch weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    await model.complete(
        messages=[ChatMessage(role="user", content="Weather?")],
        tools=[tool],
    )
    kwargs = client.messages.create.call_args.kwargs
    assert "tools" in kwargs
    api_tool = kwargs["tools"][0]
    assert api_tool["name"] == "get_weather"
    assert api_tool["description"] == "Fetch weather"
    assert api_tool["input_schema"] == tool.input_schema


async def test_tool_use_response_parsed() -> None:
    raw = _make_raw_response(
        text="",
        stop_reason="tool_use",
        tool_use_blocks=[{"id": "tu_001", "name": "get_weather", "input": {"city": "London"}}],
    )
    # Override text block to empty so only tool_use is meaningful
    raw.content[0].text = ""
    client = _make_client(raw=raw)
    model = _make_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="Weather in London?")]
    )
    assert len(resp.tool_calls) == 1
    tc: ToolCall = resp.tool_calls[0]
    assert tc.id == "tu_001"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "London"}
    assert resp.finish_reason == "tool_use"


# ---------------------------------------------------------------------------
# Cost computation (S11)
# ---------------------------------------------------------------------------


async def test_cost_computed_correctly_pro_tier() -> None:
    raw = _make_raw_response(input_tokens=1_000_000, output_tokens=500_000)
    client = _make_client(raw=raw)
    model = _make_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        tier="pro",
    )
    # 1 MTok input @ $5.00 + 0.5 MTok output @ $25.00 = $5.00 + $12.50 = $17.50
    expected = 1.0 * PRO_INPUT + 0.5 * PRO_OUTPUT
    assert abs(resp.usage.cost_usd - expected) < 1e-9


async def test_cost_computed_correctly_flash_tier() -> None:
    raw = _make_raw_response(input_tokens=2_000_000, output_tokens=1_000_000)
    client = _make_client(raw=raw)
    model = _make_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        tier="flash",
    )
    # 2 MTok input @ $1.00 + 1 MTok output @ $5.00 = $2.00 + $5.00 = $7.00
    expected = 2.0 * FLASH_INPUT + 1.0 * FLASH_OUTPUT
    assert abs(resp.usage.cost_usd - expected) < 1e-9


async def test_usage_token_counts_round_trip() -> None:
    raw = _make_raw_response(input_tokens=123, output_tokens=456)
    client = _make_client(raw=raw)
    model = _make_model(client=client)
    resp: ModelResponse = await model.complete(messages=[ChatMessage(role="user", content="hello")])
    assert resp.usage.prompt_tokens == 123
    assert resp.usage.completion_tokens == 456


# ---------------------------------------------------------------------------
# PriceTable.cost_usd direct tests (S11)
# ---------------------------------------------------------------------------


def test_price_table_cost_usd_pro() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost = PRICE_TABLE.cost_usd(usage, "pro")
    assert cost == PRO_INPUT + PRO_OUTPUT


def test_price_table_cost_usd_flash() -> None:
    usage = Usage(prompt_tokens=1_000_000, completion_tokens=1_000_000)
    cost = PRICE_TABLE.cost_usd(usage, "flash")
    assert cost == FLASH_INPUT + FLASH_OUTPUT


def test_price_table_cost_usd_zero_tokens() -> None:
    usage = Usage(prompt_tokens=0, completion_tokens=0)
    assert PRICE_TABLE.cost_usd(usage, "pro") == 0.0
    assert PRICE_TABLE.cost_usd(usage, "flash") == 0.0


# ---------------------------------------------------------------------------
# Transport retries (429 / 5xx) — DISTINCT from semantic retries
# ---------------------------------------------------------------------------


async def test_retry_on_429_eventually_succeeds() -> None:
    """After two 429s the adapter should succeed on the third attempt."""
    exc_429 = Exception("rate limited")
    exc_429.status_code = 429  # type: ignore[attr-defined]

    raw = _make_raw_response()
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=[exc_429, exc_429, raw])

    config = ProviderConfig(
        model_pro="claude-opus-4-8",
        model_flash="claude-haiku-4-5",
        price_per_mtok=PRICE_TABLE,
        max_transport_retries=2,
    )
    model = ClaudeModel(config, client=client)

    # Patch asyncio.sleep so the test doesn't actually wait.
    original_sleep = asyncio.sleep
    sleep_calls: list[float] = []

    async def fake_sleep(delay: float) -> None:
        sleep_calls.append(delay)

    asyncio.sleep = fake_sleep  # type: ignore[assignment]
    try:
        resp: ModelResponse = await model.complete(
            messages=[ChatMessage(role="user", content="hello")]
        )
    finally:
        asyncio.sleep = original_sleep

    assert resp.text == "hello"
    assert client.messages.create.call_count == 3
    assert len(sleep_calls) == 2  # slept between retries


async def test_retry_exhausted_raises() -> None:
    """When retries are exhausted the original exception must propagate."""
    exc_500 = Exception("server error")
    exc_500.status_code = 500  # type: ignore[attr-defined]

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=exc_500)

    config = ProviderConfig(
        model_pro="claude-opus-4-8",
        model_flash="claude-haiku-4-5",
        price_per_mtok=PRICE_TABLE,
        max_transport_retries=1,
    )
    model = ClaudeModel(config, client=client)

    original_sleep = asyncio.sleep
    asyncio.sleep = AsyncMock()
    try:
        with pytest.raises(Exception, match="server error"):
            await model.complete(messages=[ChatMessage(role="user", content="hello")])
    finally:
        asyncio.sleep = original_sleep

    # Called max_transport_retries + 1 times total
    assert client.messages.create.call_count == 2


async def test_non_retryable_error_raises_immediately() -> None:
    """A 400 error must NOT be retried — it is raised immediately."""
    exc_400 = Exception("bad request")
    exc_400.status_code = 400  # type: ignore[attr-defined]

    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(side_effect=exc_400)

    model = _make_model(client=client)

    with pytest.raises(Exception, match="bad request"):
        await model.complete(messages=[ChatMessage(role="user", content="hello")])

    assert client.messages.create.call_count == 1


# ---------------------------------------------------------------------------
# count_tokens heuristic
# ---------------------------------------------------------------------------


def test_count_tokens_nonempty() -> None:
    model = _make_model()
    # 40-char string → 10 tokens
    assert model.count_tokens("a" * 40) == 10


def test_count_tokens_empty_returns_at_least_one() -> None:
    model = _make_model()
    assert model.count_tokens("") >= 1


def test_count_tokens_short_string_returns_at_least_one() -> None:
    model = _make_model()
    assert model.count_tokens("hi") >= 1


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


def test_backoff_grows_exponentially() -> None:
    assert _backoff(0) == 1.0
    assert _backoff(1) == 2.0
    assert _backoff(2) == 4.0


def test_backoff_capped() -> None:
    # Large attempt number should not exceed the cap.
    assert _backoff(100) == 30.0


# ---------------------------------------------------------------------------
# _extract_status helper
# ---------------------------------------------------------------------------


def test_extract_status_with_status_code() -> None:
    exc = Exception("error")
    exc.status_code = 429  # type: ignore[attr-defined]
    assert _extract_status(exc) == 429


def test_extract_status_without_attribute() -> None:
    assert _extract_status(ValueError("plain")) == 0


# ---------------------------------------------------------------------------
# ProviderConfig frozen / immutability
# ---------------------------------------------------------------------------


def test_provider_config_is_frozen() -> None:
    with pytest.raises(ValidationError):
        CONFIG.model_pro = "something-else"  # type: ignore[misc]


def test_price_table_is_frozen() -> None:
    with pytest.raises(ValidationError):
        PRICE_TABLE.pro_input_usd_per_mtok = 99.0  # type: ignore[misc]


def test_price_table_negative_raises() -> None:
    """PriceTable must reject negative prices (Field ge=0 validators)."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PriceTable(
            pro_input_usd_per_mtok=-1.0,
            pro_output_usd_per_mtok=0.0,
            flash_input_usd_per_mtok=0.0,
            flash_output_usd_per_mtok=0.0,
        )


# ---------------------------------------------------------------------------
# F1: structured output tool unwrapping
# ---------------------------------------------------------------------------


async def test_structured_output_tool_unwrapped_from_response() -> None:
    """When json_schema is set, the _structured_output tool-use block is unwrapped into
    response.text and does NOT appear in response.tool_calls."""
    raw = _make_raw_response(
        text="",
        tool_use_blocks=[{"id": "tu_so", "name": "_structured_output", "input": {"a": 1}}],
    )
    # Override text block to empty so only tool_use is meaningful
    raw.content[0].text = ""
    client = _make_client(raw=raw)
    model = _make_model(client=client)

    resp = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        json_schema={"type": "object"},
    )

    import json as _json

    assert resp.text == _json.dumps({"a": 1})
    # The synthetic tool call must NOT appear in tool_calls
    assert all(tc.name != "_structured_output" for tc in resp.tool_calls)


async def test_structured_output_no_synthetic_tool_passes_through() -> None:
    """When json_schema is set but the model returns only text (no synthetic tool call),
    the text is passed through unchanged — no error raised from the adapter."""
    raw = _make_raw_response(text='{"x": 1}')
    client = _make_client(raw=raw)
    model = _make_model(client=client)

    resp = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        json_schema={"type": "object"},
    )

    assert resp.text == '{"x": 1}'


async def test_non_schema_call_tool_calls_unchanged() -> None:
    """When json_schema is NOT set, real tool_use blocks pass through unchanged."""
    raw = _make_raw_response(
        text="",
        tool_use_blocks=[{"id": "tc_1", "name": "my_tool", "input": {"key": "value"}}],
    )
    raw.content[0].text = ""
    client = _make_client(raw=raw)
    model = _make_model(client=client)

    resp = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        # No json_schema
    )

    assert len(resp.tool_calls) == 1
    assert resp.tool_calls[0].name == "my_tool"
    assert resp.tool_calls[0].arguments == {"key": "value"}


async def test_max_output_tokens_used_in_request() -> None:
    """ClaudeModel uses config.max_output_tokens (not a hardcoded 8192) in the API call."""
    custom_config = ProviderConfig(
        model_pro="claude-opus-4-8",
        model_flash="claude-haiku-4-5",
        price_per_mtok=PRICE_TABLE,
        max_output_tokens=1024,
    )
    client = _make_client()
    model = ClaudeModel(custom_config, client=client)

    await model.complete(messages=[ChatMessage(role="user", content="hello")])

    kwargs = client.messages.create.call_args.kwargs
    assert kwargs["max_tokens"] == 1024
