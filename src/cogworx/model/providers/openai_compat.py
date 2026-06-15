"""OpenAI-compatible model adapter (CANON S4, S9, S11).

Implements the ``Model`` Protocol for any OpenAI-compatible chat-completions
endpoint — OpenAI, DeepSeek, and local Ollama are the primary targets.  The
adapter selects the concrete backend via ``ProviderConfig.base_url``:

- ``base_url=None``  →  OpenAI production (``https://api.openai.com/v1``)
- ``base_url="https://api.deepseek.com/v1"``  →  DeepSeek
- ``base_url="http://localhost:11434/v1"``  →  Ollama (OpenAI-compat shim)

CANON cross-references
----------------------
- S4  Model-agnostic: this file is one provider behind the thin ``Model`` seam.
- S9  Structure over prompting: capabilities are declared STATICALLY, either
      via the per-target constants below or via the ``capabilities`` kwarg on
      the constructor.  They are NEVER probed at runtime (no API introspection,
      no model self-report).  Static declaration is the invariant; do not add
      probe logic.
- S11 Cost bounded structurally: ``Usage.cost_usd`` is computed here from the
      provider's price table so budget guards upstream can enforce caps.
- S1  Model work OFF the write path.
- S8  Graceful degradation: capabilities degrade (structured_output=False,
      tools=False) for targets that lack them; the engine handles the fallback.

Capabilities design (S9 deliberate design point)
-------------------------------------------------
OpenAI-compat endpoints vary widely in capability:

- OpenAI GPT-4o   → structured output, tools, streaming, logprobs, caching
- DeepSeek Chat   → tools, streaming; NO native structured-output enforcement
- Ollama          → typically NO tools, NO structured output (model-dependent)

Because S9 forbids runtime probing, the adapter cannot interrogate the endpoint
to discover its capabilities.  Instead:

1. Three *per-target constants* below document the canonical capability sets for
   the three primary targets.  Callers building a config for a known target
   should pass the matching constant as the ``capabilities`` constructor kwarg.

2. Callers building for a novel / custom endpoint must pass an explicit
   ``ModelCapabilities`` instance — the adapter has no default to guess for an
   unknown URL.

3. The ONLY built-in default is ``OPENAI_CAPABILITIES`` when ``capabilities``
   is ``None`` AND ``base_url`` is ``None`` (i.e. OpenAI production).  This
   keeps the zero-config path ergonomic for the most capable target.

This is a deliberate, documented policy — not an oversight.  Any change to
these defaults is a code change with an accompanying docstring update.

Transport vs. semantic retries
-------------------------------
``OpenAICompatModel`` retries ONLY on transient transport failures (HTTP 429
and 5xx) with bounded exponential backoff.  These retries are invisible to the
caller — the same request is re-sent.  They are distinct from the engine's
*semantic* retries (e.g. output-validation failures, tool-parse errors) which
live in the engine layer and may alter the prompt.  Mixing the two would
violate S8.

Injectable client
-----------------
The ``client`` constructor kwarg accepts any object satisfying the narrow
calling contract used internally:
``await client.chat.completions.create(**kwargs)`` returns an object with
``.choices[0].message`` and ``.usage``.  Passing a fake in tests eliminates
all network calls while exercising the full adapter logic.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Mapping, Sequence
from typing import Any

from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolCall,
    ToolSpec,
    Usage,
)
from cogworx.model.providers.config import ProviderConfig

__all__ = [
    "DEEPSEEK_CAPABILITIES",
    "OLLAMA_CAPABILITIES",
    "OPENAI_CAPABILITIES",
    "OpenAICompatModel",
]

# ---------------------------------------------------------------------------
# Per-target static capability constants (S9 — never probe, never self-report)
# ---------------------------------------------------------------------------

OPENAI_CAPABILITIES: ModelCapabilities = ModelCapabilities(
    structured_output=True,
    tools=True,
    caching=False,  # client-side caching exists but is not a protocol feature
    streaming=True,
    logprobs=True,
)
"""Canonical capabilities for OpenAI production endpoints (GPT-4o class)."""

DEEPSEEK_CAPABILITIES: ModelCapabilities = ModelCapabilities(
    structured_output=False,
    tools=True,
    caching=False,
    streaming=True,
    logprobs=False,
)
"""Canonical capabilities for DeepSeek Chat endpoints.

DeepSeek supports tool calls and streaming but does NOT enforce structured
output natively — the framework falls back to prompt-based JSON extraction
when ``structured_output=False``.
"""

OLLAMA_CAPABILITIES: ModelCapabilities = ModelCapabilities(
    structured_output=False,
    tools=False,
    caching=False,
    streaming=False,
    logprobs=False,
)
"""Canonical capabilities for local Ollama endpoints.

Ollama's OpenAI-compat shim is model-dependent; the conservative baseline is
no tools and no structured output.  Operators running a capable model (e.g.
llama3 with tool support) should pass a custom ``ModelCapabilities`` instance
to the constructor rather than overriding this constant.
"""

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Transport-error status codes that warrant a retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Backoff base (seconds); capped at 30 s.
_BACKOFF_BASE: float = 1.0
_BACKOFF_CAP: float = 30.0


def _backoff(attempt: int) -> float:
    """Exponential backoff capped at ``_BACKOFF_CAP`` seconds."""
    return min(_BACKOFF_BASE * (2.0**attempt), _BACKOFF_CAP)


def _map_messages(
    messages: Sequence[ChatMessage],
) -> list[dict[str, Any]]:
    """Convert framework ``ChatMessage`` objects to the OpenAI chat format.

    The OpenAI-compat API accepts ``system`` as an ordinary role in the
    ``messages`` array (unlike Anthropic's top-level ``system`` param), so
    every message type maps directly without extraction.

    Tool results use ``role="tool"`` with ``tool_call_id`` — this matches the
    OpenAI wire format exactly.
    """
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "tool":
            api_messages.append(
                {
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id or "",
                }
            )
        else:
            api_messages.append({"role": msg.role, "content": msg.content})
    return api_messages


def _map_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Convert framework ``ToolSpec`` objects to OpenAI function-tool dicts."""
    return [
        {
            "type": "function",
            "function": {
                "name": spec.name,
                "description": spec.description,
                "parameters": spec.input_schema,
            },
        }
        for spec in tools
    ]


def _extract_response(raw: Any, model_id: str, usage: Usage) -> ModelResponse:
    """Parse an OpenAI chat-completions response into a ``ModelResponse``."""
    choice = raw.choices[0]
    message = choice.message

    text: str | None = getattr(message, "content", None) or None

    tool_calls: list[ToolCall] = []
    raw_tool_calls = getattr(message, "tool_calls", None) or []
    for tc in raw_tool_calls:
        import json as _json

        args_raw = getattr(tc.function, "arguments", None) or "{}"
        try:
            arguments: dict[str, Any] = _json.loads(args_raw)
        except ValueError:
            arguments = {}
        tool_calls.append(
            ToolCall(
                id=tc.id,
                name=tc.function.name,
                arguments=arguments,
            )
        )

    # Normalise finish_reason — OpenAI uses "stop" / "tool_calls" / "length" etc.
    finish_reason: str = getattr(choice, "finish_reason", None) or "stop"

    return ModelResponse(
        text=text,
        tool_calls=tuple(tool_calls),
        usage=usage,
        model_id=model_id,
        finish_reason=finish_reason,
    )


def _extract_status(exc: BaseException) -> int:
    """Return the HTTP status code from an OpenAI SDK exception, or 0.

    The ``openai`` SDK raises ``openai.APIStatusError`` which carries a
    ``.status_code`` attribute.  Fall back to 0 for non-SDK exceptions.
    """
    return int(getattr(exc, "status_code", 0))


# ---------------------------------------------------------------------------
# OpenAICompatModel
# ---------------------------------------------------------------------------


class OpenAICompatModel:
    """OpenAI-compatible implementation of the ``Model`` Protocol.

    Covers OpenAI, DeepSeek, and Ollama via ``ProviderConfig.base_url``.

    Parameters
    ----------
    config:
        Provider configuration (API key, base URL, model IDs, pricing,
        timeouts …).
    capabilities:
        Static capability declaration for this endpoint (S9 — never probe).
        When ``None`` and ``config.base_url`` is ``None``, defaults to
        ``OPENAI_CAPABILITIES`` (OpenAI production).  For DeepSeek, pass
        ``DEEPSEEK_CAPABILITIES``; for Ollama, pass ``OLLAMA_CAPABILITIES``.
        For a novel endpoint, pass an explicit ``ModelCapabilities`` instance.
        See the module docstring for the full design rationale.
    client:
        Injectable async OpenAI client.  When ``None`` the adapter constructs
        a real ``openai.AsyncOpenAI`` from *config*.  Pass a fake here in unit
        tests to avoid network calls.
    """

    def __init__(
        self,
        config: ProviderConfig,
        *,
        capabilities: ModelCapabilities | None = None,
        client: Any = None,
    ) -> None:
        self._config = config

        # S9 invariant: capabilities are resolved ONCE at construction time from
        # the caller-supplied constant.  No probe, no introspection, ever.
        if capabilities is not None:
            self._capabilities = capabilities
        elif config.base_url is None:
            # Default to full OpenAI capabilities for the production endpoint.
            self._capabilities = OPENAI_CAPABILITIES
        else:
            raise ValueError(
                "OpenAICompatModel: 'capabilities' must be supplied explicitly "
                "when 'config.base_url' is set.  Pass DEEPSEEK_CAPABILITIES, "
                "OLLAMA_CAPABILITIES, or a custom ModelCapabilities instance.  "
                "See the module docstring for the design rationale (CANON S9)."
            )

        if client is not None:
            self._client = client
        else:
            try:
                import openai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The 'openai' package is required for OpenAICompatModel.  "
                    "Install it with: pip install 'cog-worx[providers]'"
                ) from exc

            api_key = config.api_key.get_secret_value() if config.api_key is not None else None
            kwargs: dict[str, Any] = {"timeout": config.timeout_s}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if config.base_url is not None:
                kwargs["base_url"] = config.base_url
            self._client = openai.AsyncOpenAI(**kwargs)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Static capability declaration (CANON S9 — never probe the model)."""
        return self._capabilities

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        """Call the OpenAI-compat endpoint and return a ``ModelResponse``.

        Transport-level retries (429 / 5xx) are handled internally, bounded by
        ``config.max_transport_retries``.  Semantic / business-logic retries
        are the engine's concern and must NOT be added here.

        Parameters
        ----------
        messages:
            Conversation history (system + turns).
        tools:
            Tool definitions available to the model.  Silently ignored when
            ``self.capabilities.tools`` is ``False`` (degraded target).
        tier:
            Model tier (``"pro"`` → ``config.model_pro``,
            ``"flash"`` → ``config.model_flash``).
        json_schema:
            Optional JSON schema for structured output.  When provided AND
            ``self.capabilities.structured_output`` is ``True``, the schema
            is sent via OpenAI's ``response_format`` parameter.  When the
            capability is ``False`` (degraded target), this parameter is
            silently ignored — the engine layer is responsible for falling
            back gracefully (S8 / S4).
        """
        model_id = self._config.model_pro if tier == "pro" else self._config.model_flash
        api_messages = _map_messages(messages)

        kwargs: dict[str, Any] = {
            "model": model_id,
            "messages": api_messages,
        }

        # Tools — only include when the target supports them (S4 degradation).
        if tools and self._capabilities.tools:
            kwargs["tools"] = _map_tools(tools)

        # Structured output — only include when the target supports it (S4 degradation).
        if json_schema is not None and self._capabilities.structured_output:
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_output",
                    "strict": True,
                    "schema": json_schema,
                },
            }

        # Transport retry loop (distinct from engine semantic retries — S8).
        last_exc: BaseException | None = None
        t0 = time.monotonic()
        latency_ms: float = 0.0
        raw: Any = None
        for attempt in range(self._config.max_transport_retries + 1):
            try:
                t0 = time.monotonic()
                raw = await self._client.chat.completions.create(**kwargs)
                latency_ms = (time.monotonic() - t0) * 1000.0
                break
            except Exception as exc:
                status = _extract_status(exc)
                if status in _RETRYABLE_STATUS and attempt < self._config.max_transport_retries:
                    await asyncio.sleep(_backoff(attempt))
                    last_exc = exc
                    continue
                raise
        else:
            # All retries exhausted — re-raise the last exception.
            raise last_exc  # type: ignore[misc]

        # Map raw usage to framework Usage + compute cost (S11).
        raw_usage = getattr(raw, "usage", None)
        prompt_tokens: int = getattr(raw_usage, "prompt_tokens", 0) if raw_usage else 0
        completion_tokens: int = getattr(raw_usage, "completion_tokens", 0) if raw_usage else 0
        usage_pre_cost = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
        )
        cost = self._config.price_per_mtok.cost_usd(usage_pre_cost, tier)
        usage = Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost,
            latency_ms=latency_ms,
        )

        return _extract_response(raw, model_id, usage)

    def count_tokens(self, text: str) -> int:
        """Approximate token count for *text* using a heuristic (4 chars ≈ 1 token).

        This is intentionally a local estimate — no API call — to keep it
        synchronous and free.  For accurate counts use a tiktoken-based counter
        separately.
        """
        return max(1, len(text) // 4)
