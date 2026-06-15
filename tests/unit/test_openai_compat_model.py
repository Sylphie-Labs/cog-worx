"""Unit tests for OpenAICompatModel (CANON S4, S9, S11).

All tests use a FAKE injected client — zero network calls.  The fake satisfies
the narrow calling contract ``await client.chat.completions.create(**kwargs)``
that ``OpenAICompatModel`` uses internally.

Coverage targets
----------------
- Static capability declaration (S9: never probe, never self-report).
- Per-target capability constants (OPENAI / DEEPSEEK / OLLAMA).
- Degraded-capabilities path: tools and structured_output silently omitted
  when the capability flag is ``False`` (e.g. Ollama-style config).
- base_url routing: base_url=None → OpenAI default caps; explicit URL requires
  explicit capabilities kwarg.
- Message / tool mapping round-trips (system, tool result, tool calls).
- Cost computation via PriceTable (S11).
- Transport-retry on simulated 429 / 5xx, bounded by max_transport_retries.
- Model ID selection per tier.
- count_tokens heuristic.
- Protocol satisfaction (isinstance check).
- Constructor raises when base_url is set but capabilities is not supplied.

CARRY-FORWARD
-------------
DeepSeek and Ollama *live* integration rows are deferred to a future session
pending live endpoint access.  See ``tests/integration/test_openai_compat_model_live.py``
for the skip-guarded stubs.
"""

from __future__ import annotations

import asyncio
import json
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
)
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.model.providers.openai_compat import (
    DEEPSEEK_CAPABILITIES,
    OLLAMA_CAPABILITIES,
    OPENAI_CAPABILITIES,
    OpenAICompatModel,
    _backoff,
    _extract_status,
)

# ---------------------------------------------------------------------------
# Shared test fixtures
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

# OpenAI-production config (no base_url → defaults to OPENAI_CAPABILITIES).
OPENAI_CONFIG = ProviderConfig(
    model_pro="gpt-4o",
    model_flash="gpt-4o-mini",
    price_per_mtok=PRICE_TABLE,
    max_transport_retries=2,
)

# Ollama-style config: explicit base_url requires explicit capabilities.
OLLAMA_CONFIG = ProviderConfig(
    model_pro="llama3",
    model_flash="llama3",
    base_url="http://localhost:11434/v1",
    price_per_mtok=PRICE_TABLE,
    max_transport_retries=2,
)

# DeepSeek config.
DEEPSEEK_CONFIG = ProviderConfig(
    model_pro="deepseek-chat",
    model_flash="deepseek-chat",
    base_url="https://api.deepseek.com/v1",
    price_per_mtok=PRICE_TABLE,
    max_transport_retries=2,
)


# ---------------------------------------------------------------------------
# Fake client helpers
# ---------------------------------------------------------------------------


def _make_raw_response(
    *,
    text: str = "hello",
    finish_reason: str = "stop",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    tool_calls_data: list[dict[str, Any]] | None = None,
) -> MagicMock:
    """Build a MagicMock that mimics an OpenAI ``ChatCompletion``."""
    raw = MagicMock()

    # usage
    raw.usage = MagicMock()
    raw.usage.prompt_tokens = prompt_tokens
    raw.usage.completion_tokens = completion_tokens

    # choices[0].message
    message = MagicMock()
    message.content = text
    message.tool_calls = None

    if tool_calls_data:
        tc_mocks = []
        for td in tool_calls_data:
            tc = MagicMock()
            tc.id = td["id"]
            tc.function = MagicMock()
            tc.function.name = td["name"]
            tc.function.arguments = json.dumps(td["arguments"])
            tc_mocks.append(tc)
        message.tool_calls = tc_mocks

    choice = MagicMock()
    choice.message = message
    choice.finish_reason = finish_reason

    raw.choices = [choice]
    return raw


def _make_client(raw: MagicMock | None = None) -> MagicMock:
    """Build a fake async OpenAI client."""
    if raw is None:
        raw = _make_raw_response()
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(return_value=raw)
    return client


def _make_openai_model(
    client: MagicMock | None = None,
    config: ProviderConfig = OPENAI_CONFIG,
    capabilities: ModelCapabilities | None = None,
) -> OpenAICompatModel:
    if client is None:
        client = _make_client()
    return OpenAICompatModel(config, capabilities=capabilities, client=client)


def _make_ollama_model(
    client: MagicMock | None = None,
    config: ProviderConfig = OLLAMA_CONFIG,
) -> OpenAICompatModel:
    if client is None:
        client = _make_client()
    return OpenAICompatModel(config, capabilities=OLLAMA_CAPABILITIES, client=client)


# ---------------------------------------------------------------------------
# Protocol satisfaction (S9 — static, not probed)
# ---------------------------------------------------------------------------


def test_openai_compat_model_satisfies_model_protocol() -> None:
    """OpenAICompatModel must satisfy the runtime_checkable Model Protocol."""
    model = _make_openai_model()
    assert isinstance(model, Model)


# ---------------------------------------------------------------------------
# Per-target capability constants (S9)
# ---------------------------------------------------------------------------


def test_openai_capabilities_values() -> None:
    assert OPENAI_CAPABILITIES.structured_output is True
    assert OPENAI_CAPABILITIES.tools is True
    assert OPENAI_CAPABILITIES.streaming is True
    assert OPENAI_CAPABILITIES.logprobs is True
    assert OPENAI_CAPABILITIES.caching is False


def test_deepseek_capabilities_values() -> None:
    assert DEEPSEEK_CAPABILITIES.structured_output is False
    assert DEEPSEEK_CAPABILITIES.tools is True
    assert DEEPSEEK_CAPABILITIES.streaming is True
    assert DEEPSEEK_CAPABILITIES.logprobs is False


def test_ollama_capabilities_values() -> None:
    assert OLLAMA_CAPABILITIES.structured_output is False
    assert OLLAMA_CAPABILITIES.tools is False
    assert OLLAMA_CAPABILITIES.streaming is False
    assert OLLAMA_CAPABILITIES.logprobs is False


# ---------------------------------------------------------------------------
# base_url routing and capability resolution
# ---------------------------------------------------------------------------


def test_no_base_url_defaults_to_openai_capabilities() -> None:
    """No base_url and no capabilities kwarg → OPENAI_CAPABILITIES applied."""
    model = _make_openai_model()
    assert model.capabilities == OPENAI_CAPABILITIES


def test_explicit_capabilities_override_default() -> None:
    """Passing explicit capabilities overrides the default even with no base_url."""
    custom_caps = ModelCapabilities(structured_output=False, tools=False)
    model = _make_openai_model(capabilities=custom_caps)
    assert model.capabilities == custom_caps


def test_base_url_without_capabilities_raises() -> None:
    """Setting base_url without capabilities must raise ValueError (S9 design)."""
    with pytest.raises(ValueError, match=r"capabilities.*must be supplied"):
        OpenAICompatModel(OLLAMA_CONFIG, client=_make_client())


def test_base_url_with_explicit_capabilities_accepted() -> None:
    """Setting base_url WITH explicit capabilities is valid."""
    model = OpenAICompatModel(
        OLLAMA_CONFIG, capabilities=OLLAMA_CAPABILITIES, client=_make_client()
    )
    assert model.capabilities == OLLAMA_CAPABILITIES


def test_deepseek_config_accepted() -> None:
    model = OpenAICompatModel(
        DEEPSEEK_CONFIG, capabilities=DEEPSEEK_CAPABILITIES, client=_make_client()
    )
    assert model.capabilities.tools is True
    assert model.capabilities.structured_output is False


# ---------------------------------------------------------------------------
# Degraded capabilities (Ollama-style: no tools, no structured output)
# ---------------------------------------------------------------------------


async def test_tools_not_sent_when_capability_false() -> None:
    """When tools=False, tool specs are NOT forwarded to the API."""
    client = _make_client()
    model = _make_ollama_model(client=client)
    tool = ToolSpec(
        name="search",
        description="Search the web",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )
    await model.complete(
        messages=[ChatMessage(role="user", content="search something")],
        tools=[tool],
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "tools" not in kwargs, "tools must be omitted for a degraded target"


async def test_structured_output_not_sent_when_capability_false() -> None:
    """When structured_output=False, json_schema is NOT sent to the API."""
    client = _make_client()
    model = _make_ollama_model(client=client)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    await model.complete(
        messages=[ChatMessage(role="user", content="answer me")],
        json_schema=schema,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" not in kwargs, "response_format must be omitted for a degraded target"


async def test_tools_sent_when_capability_true() -> None:
    """When tools=True, tool specs ARE forwarded to the API."""
    client = _make_client()
    model = _make_openai_model(client=client)
    tool = ToolSpec(
        name="get_weather",
        description="Fetch weather",
        input_schema={"type": "object", "properties": {"city": {"type": "string"}}},
    )
    await model.complete(
        messages=[ChatMessage(role="user", content="Weather?")],
        tools=[tool],
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "tools" in kwargs
    api_tool = kwargs["tools"][0]
    assert api_tool["type"] == "function"
    assert api_tool["function"]["name"] == "get_weather"


async def test_structured_output_sent_when_capability_true() -> None:
    """When structured_output=True, response_format IS sent to the API."""
    client = _make_client()
    model = _make_openai_model(client=client)
    schema: dict[str, Any] = {
        "type": "object",
        "properties": {"answer": {"type": "string"}},
    }
    await model.complete(
        messages=[ChatMessage(role="user", content="answer me")],
        json_schema=schema,
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert "response_format" in kwargs
    assert kwargs["response_format"]["type"] == "json_schema"
    assert kwargs["response_format"]["json_schema"]["schema"] == schema


# ---------------------------------------------------------------------------
# Model ID selection by tier
# ---------------------------------------------------------------------------


async def test_pro_tier_uses_model_pro() -> None:
    client = _make_client()
    model = _make_openai_model(client=client)
    await model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="pro",
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == OPENAI_CONFIG.model_pro


async def test_flash_tier_uses_model_flash() -> None:
    client = _make_client()
    model = _make_openai_model(client=client)
    await model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="flash",
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == OPENAI_CONFIG.model_flash


# ---------------------------------------------------------------------------
# Message mapping
# ---------------------------------------------------------------------------


async def test_system_message_maps_to_system_role() -> None:
    """In the OpenAI format, system messages stay in the messages array."""
    client = _make_client()
    model = _make_openai_model(client=client)
    await model.complete(
        messages=[
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="Hello"),
        ]
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    msgs: list[dict[str, Any]] = kwargs["messages"]
    system_msgs = [m for m in msgs if m["role"] == "system"]
    assert len(system_msgs) == 1
    assert system_msgs[0]["content"] == "Be concise."


async def test_tool_role_maps_to_tool_message() -> None:
    """Tool results use role='tool' with tool_call_id in OpenAI format."""
    client = _make_client()
    model = _make_openai_model(client=client)
    await model.complete(
        messages=[
            ChatMessage(role="user", content="Call the tool"),
            ChatMessage(role="tool", content='{"result": 42}', tool_call_id="tc_001"),
        ]
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    tool_msgs = [m for m in kwargs["messages"] if m["role"] == "tool"]
    assert len(tool_msgs) == 1
    assert tool_msgs[0]["tool_call_id"] == "tc_001"
    assert tool_msgs[0]["content"] == '{"result": 42}'


async def test_user_and_assistant_messages_passed_through() -> None:
    client = _make_client()
    model = _make_openai_model(client=client)
    await model.complete(
        messages=[
            ChatMessage(role="user", content="Hello"),
            ChatMessage(role="assistant", content="Hi there"),
            ChatMessage(role="user", content="What's up?"),
        ]
    )
    kwargs = client.chat.completions.create.call_args.kwargs
    msgs: list[dict[str, Any]] = kwargs["messages"]
    roles = [m["role"] for m in msgs]
    assert roles == ["user", "assistant", "user"]


# ---------------------------------------------------------------------------
# Tool call response parsing
# ---------------------------------------------------------------------------


async def test_tool_call_response_parsed() -> None:
    raw = _make_raw_response(
        text="",
        finish_reason="tool_calls",
        tool_calls_data=[{"id": "call_abc", "name": "get_weather", "arguments": {"city": "Paris"}}],
    )
    raw.choices[0].message.content = None
    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="Weather in Paris?")]
    )
    assert len(resp.tool_calls) == 1
    tc: ToolCall = resp.tool_calls[0]
    assert tc.id == "call_abc"
    assert tc.name == "get_weather"
    assert tc.arguments == {"city": "Paris"}
    assert resp.finish_reason == "tool_calls"


async def test_tool_call_arguments_malformed_json_becomes_empty_dict() -> None:
    """Malformed tool-call arguments must degrade to {} rather than raising."""
    raw = _make_raw_response(finish_reason="tool_calls")
    # Override arguments to be invalid JSON.
    tc_mock = MagicMock()
    tc_mock.id = "call_bad"
    tc_mock.function = MagicMock()
    tc_mock.function.name = "bad_tool"
    tc_mock.function.arguments = "not-valid-json"
    raw.choices[0].message.tool_calls = [tc_mock]
    raw.choices[0].message.content = None

    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(messages=[ChatMessage(role="user", content="hi")])
    assert resp.tool_calls[0].arguments == {}


# ---------------------------------------------------------------------------
# Cost computation (S11)
# ---------------------------------------------------------------------------


async def test_cost_computed_correctly_pro_tier() -> None:
    raw = _make_raw_response(prompt_tokens=1_000_000, completion_tokens=500_000)
    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        tier="pro",
    )
    # 1 MTok input @ $5.00 + 0.5 MTok output @ $25.00 = $5.00 + $12.50 = $17.50
    expected = 1.0 * PRO_INPUT + 0.5 * PRO_OUTPUT
    assert abs(resp.usage.cost_usd - expected) < 1e-9


async def test_cost_computed_correctly_flash_tier() -> None:
    raw = _make_raw_response(prompt_tokens=2_000_000, completion_tokens=1_000_000)
    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="hello")],
        tier="flash",
    )
    # 2 MTok input @ $1.00 + 1 MTok output @ $5.00 = $2.00 + $5.00 = $7.00
    expected = 2.0 * FLASH_INPUT + 1.0 * FLASH_OUTPUT
    assert abs(resp.usage.cost_usd - expected) < 1e-9


async def test_usage_token_counts_round_trip() -> None:
    raw = _make_raw_response(prompt_tokens=123, completion_tokens=456)
    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(messages=[ChatMessage(role="user", content="hello")])
    assert resp.usage.prompt_tokens == 123
    assert resp.usage.completion_tokens == 456


async def test_cost_zero_tokens() -> None:
    raw = _make_raw_response(prompt_tokens=0, completion_tokens=0)
    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(messages=[ChatMessage(role="user", content="hello")])
    assert resp.usage.cost_usd == 0.0


# ---------------------------------------------------------------------------
# Transport retries (429 / 5xx) — DISTINCT from semantic retries
# ---------------------------------------------------------------------------


async def test_retry_on_429_eventually_succeeds() -> None:
    """After two 429s the adapter should succeed on the third attempt."""
    exc_429 = Exception("rate limited")
    exc_429.status_code = 429  # type: ignore[attr-defined]

    raw = _make_raw_response()
    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=[exc_429, exc_429, raw])

    config = ProviderConfig(
        model_pro="gpt-4o",
        model_flash="gpt-4o-mini",
        price_per_mtok=PRICE_TABLE,
        max_transport_retries=2,
    )
    model = OpenAICompatModel(config, client=client)

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
    assert client.chat.completions.create.call_count == 3
    assert len(sleep_calls) == 2


async def test_retry_exhausted_raises() -> None:
    """When retries are exhausted the original exception must propagate."""
    exc_500 = Exception("server error")
    exc_500.status_code = 500  # type: ignore[attr-defined]

    client = MagicMock()
    client.chat = MagicMock()
    client.chat.completions = MagicMock()
    client.chat.completions.create = AsyncMock(side_effect=exc_500)

    config = ProviderConfig(
        model_pro="gpt-4o",
        model_flash="gpt-4o-mini",
        price_per_mtok=PRICE_TABLE,
        max_transport_retries=1,
    )
    model = OpenAICompatModel(config, client=client)

    original_sleep = asyncio.sleep
    asyncio.sleep = AsyncMock()
    try:
        with pytest.raises(Exception, match="server error"):
            await model.complete(messages=[ChatMessage(role="user", content="hello")])
    finally:
        asyncio.sleep = original_sleep

    assert client.chat.completions.create.call_count == 2


async def test_non_retryable_error_raises_immediately() -> None:
    """A 400 error must NOT be retried — raised immediately."""
    exc_400 = Exception("bad request")
    exc_400.status_code = 400  # type: ignore[attr-defined]

    client = _make_client()
    client.chat.completions.create = AsyncMock(side_effect=exc_400)
    model = _make_openai_model(client=client)

    with pytest.raises(Exception, match="bad request"):
        await model.complete(messages=[ChatMessage(role="user", content="hello")])

    assert client.chat.completions.create.call_count == 1


# ---------------------------------------------------------------------------
# ModelResponse fields
# ---------------------------------------------------------------------------


async def test_model_id_in_response() -> None:
    client = _make_client()
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="hi")],
        tier="pro",
    )
    assert resp.model_id == OPENAI_CONFIG.model_pro


async def test_finish_reason_in_response() -> None:
    raw = _make_raw_response(finish_reason="length")
    client = _make_client(raw=raw)
    model = _make_openai_model(client=client)
    resp: ModelResponse = await model.complete(
        messages=[ChatMessage(role="user", content="tell me everything")]
    )
    assert resp.finish_reason == "length"


# ---------------------------------------------------------------------------
# count_tokens heuristic
# ---------------------------------------------------------------------------


def test_count_tokens_nonempty() -> None:
    model = _make_openai_model()
    assert model.count_tokens("a" * 40) == 10


def test_count_tokens_empty_returns_at_least_one() -> None:
    model = _make_openai_model()
    assert model.count_tokens("") >= 1


def test_count_tokens_short_string_returns_at_least_one() -> None:
    model = _make_openai_model()
    assert model.count_tokens("hi") >= 1


# ---------------------------------------------------------------------------
# Backoff helper
# ---------------------------------------------------------------------------


def test_backoff_grows_exponentially() -> None:
    assert _backoff(0) == 1.0
    assert _backoff(1) == 2.0
    assert _backoff(2) == 4.0


def test_backoff_capped() -> None:
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
# ProviderConfig / PriceTable frozen (shared types, tested here for coverage)
# ---------------------------------------------------------------------------


def test_provider_config_is_frozen() -> None:
    with pytest.raises(ValidationError):
        OPENAI_CONFIG.model_pro = "something-else"  # type: ignore[misc]


def test_price_table_is_frozen() -> None:
    with pytest.raises(ValidationError):
        PRICE_TABLE.pro_input_usd_per_mtok = 99.0  # type: ignore[misc]
