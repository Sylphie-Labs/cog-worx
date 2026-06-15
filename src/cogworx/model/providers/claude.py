"""Anthropic Claude model adapter (CANON S4, S9, S11).

Implements the ``Model`` Protocol for Anthropic's Claude API.  The adapter is
intentionally thin — it maps framework value types to the Anthropic SDK and
back, computes cost from ``ProviderConfig.price_per_mtok``, and applies
transport-only retries (429 / 5xx) with exponential backoff.

CANON cross-references
----------------------
- S4  Model-agnostic: this file is one provider behind the thin ``Model`` seam.
- S9  Structure over prompting: capabilities are declared STATICALLY as a class
      attribute and NEVER probed at runtime (no API introspection, no model
      self-report).  Static declaration is the invariant; do not add probe logic.
- S11 Cost bounded structurally: ``Usage.cost_usd`` is computed here from the
      provider's price table so budget guards upstream can enforce caps.
- S1  Model work OFF the write path.

Transport vs. semantic retries
-------------------------------
``ClaudeModel`` retries ONLY on transient transport failures (HTTP 429 and 5xx)
with bounded exponential backoff.  These retries are invisible to the caller —
the same request is re-sent.  They are distinct from the engine's *semantic*
retries (e.g. output-validation failures, tool-parse errors) which live in the
engine layer and may alter the prompt.  Mixing the two would violate S8.

Injectable client
-----------------
The ``client`` constructor kwarg accepts any object that satisfies the narrow
calling contract used internally (``client.messages.create(...)``).  Passing a
fake in tests eliminates all network calls while exercising the full adapter
logic (mapping, cost computation, retry state machine).
"""

from __future__ import annotations

import asyncio
import json
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

__all__ = ["ClaudeModel"]

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

# Transport-error status codes that warrant a retry.
_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})

# Backoff base (seconds); capped at 30 s.
_BACKOFF_BASE: float = 1.0
_BACKOFF_CAP: float = 30.0


_STRUCTURED_OUTPUT_TOOL: str = "_structured_output"


def _backoff(attempt: int) -> float:
    """Exponential backoff capped at ``_BACKOFF_CAP`` seconds."""
    return min(_BACKOFF_BASE * (2.0**attempt), _BACKOFF_CAP)


def _map_messages(
    messages: Sequence[ChatMessage],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Split framework messages into the (system, user/assistant) shape the Anthropic API expects.

    Returns
    -------
    system_prompt:
        The text of the first ``system`` message, or ``None``.
    api_messages:
        The remaining turns as dicts with ``role`` and ``content`` keys.
    """
    system_prompt: str | None = None
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            # Anthropic takes system as a top-level param; use the first one.
            if system_prompt is None:
                system_prompt = msg.content
            # Additional system messages are appended as user turns (best-effort).
            else:
                api_messages.append({"role": "user", "content": msg.content})
        elif msg.role == "tool":
            # Tool results come back as user-role messages in the Anthropic API.
            api_messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": msg.tool_call_id or "",
                            "content": msg.content,
                        }
                    ],
                }
            )
        else:
            api_messages.append({"role": msg.role, "content": msg.content})
    return system_prompt, api_messages


def _map_tools(tools: Sequence[ToolSpec]) -> list[dict[str, Any]]:
    """Convert framework ``ToolSpec`` objects to Anthropic tool dicts."""
    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
        }
        for spec in tools
    ]


def _extract_response(
    raw: Any, model_id: str, usage: Usage, *, structured_tool: str | None = None
) -> ModelResponse:
    """Parse an Anthropic ``Message`` response into a ``ModelResponse``."""
    text: str | None = None
    tool_calls: list[ToolCall] = []

    for block in raw.content:
        btype = block.type
        if btype == "text":
            text = (text or "") + block.text
        elif btype == "thinking":
            # Thinking blocks are silently consumed; we don't expose raw CoT.
            pass
        elif btype == "tool_use":
            tool_calls.append(
                ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                )
            )

    if structured_tool is not None:
        # Find the synthetic tool call
        synthetic = next((tc for tc in tool_calls if tc.name == structured_tool), None)
        if synthetic is not None:
            # Tool payload is authoritative — overwrite any text blocks
            text = _tool_call_to_text(synthetic)
            # Filter out the synthetic tool call so it doesn't leak to callers
            tool_calls = [tc for tc in tool_calls if tc.name != structured_tool]
        # If synthetic is absent, pass through as-is (ladder will validate)

    finish_reason: str = getattr(raw, "stop_reason", None) or "stop"

    return ModelResponse(
        text=text or None,
        tool_calls=tuple(tool_calls),
        usage=usage,
        model_id=model_id,
        finish_reason=finish_reason,
    )


# ---------------------------------------------------------------------------
# ClaudeModel
# ---------------------------------------------------------------------------


class ClaudeModel:
    """Anthropic Claude implementation of the ``Model`` Protocol.

    Parameters
    ----------
    config:
        Provider configuration (API key, model IDs, pricing, timeouts …).
    client:
        Injectable Anthropic ``AsyncAnthropic`` client.  When ``None`` the
        adapter constructs a real client from *config*.  Pass a fake here in
        unit tests to avoid network calls.
    """

    # S9 invariant: capabilities are STATIC.  Never probe, never introspect.
    _CAPABILITIES: ModelCapabilities = ModelCapabilities(
        structured_output=True,
        tools=True,
        caching=True,
        streaming=True,
    )

    def __init__(self, config: ProviderConfig, *, client: Any = None) -> None:
        self._config = config
        if client is not None:
            self._client = client
        else:
            # Import here so the module can be imported without the SDK installed
            # when tests supply a fake client.
            try:
                import anthropic
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    "The 'anthropic' package is required for ClaudeModel.  "
                    "Install it with: pip install 'cog-worx[providers]'"
                ) from exc

            api_key = config.api_key.get_secret_value() if config.api_key is not None else None
            kwargs: dict[str, Any] = {"timeout": config.timeout_s}
            if api_key is not None:
                kwargs["api_key"] = api_key
            if config.base_url is not None:
                kwargs["base_url"] = config.base_url
            self._client = anthropic.AsyncAnthropic(**kwargs)

    @property
    def capabilities(self) -> ModelCapabilities:
        """Static capability declaration (CANON S9 — never probe the model)."""
        return self._CAPABILITIES

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        """Call the Anthropic API and return a ``ModelResponse``.

        Transport-level retries (429 / 5xx) are handled internally, bounded by
        ``config.max_transport_retries``.  Semantic / business-logic retries
        are the engine's concern and must NOT be added here.

        Parameters
        ----------
        messages:
            Conversation history (system + turns).
        tools:
            Tool definitions available to the model for this call.
        tier:
            Model tier to use (``"pro"`` → ``config.model_pro``,
            ``"flash"`` → ``config.model_flash``).
        json_schema:
            Optional JSON schema for structured output.  When provided the
            schema is injected as a tool call so the model produces
            schema-conformant JSON.
        """
        model_id = self._config.model_pro if tier == "pro" else self._config.model_flash
        system_prompt, api_messages = _map_messages(messages)

        # Build the kwargs dict passed to messages.create.
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": self._config.max_output_tokens,
            "messages": api_messages,
        }
        if system_prompt is not None:
            kwargs["system"] = system_prompt

        # Tools — merge user tools with json_schema tool if requested.
        api_tools = _map_tools(tools)
        if json_schema is not None:
            # Inject a synthetic tool that forces structured output via tool_use.
            schema_tool: dict[str, Any] = {
                "name": _STRUCTURED_OUTPUT_TOOL,
                "description": "Return the response in the requested JSON schema.",
                "input_schema": json_schema,
            }
            api_tools = [schema_tool, *api_tools]
            # Force the model to call the schema tool.
            kwargs["tool_choice"] = {"type": "tool", "name": _STRUCTURED_OUTPUT_TOOL}
        if api_tools:
            kwargs["tools"] = api_tools

        # Transport retry loop (distinct from engine semantic retries — S8).
        last_exc: BaseException | None = None
        for attempt in range(self._config.max_transport_retries + 1):
            try:
                t0 = time.monotonic()
                raw = await self._client.messages.create(**kwargs)
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

        # Map raw usage to framework Usage + compute cost.
        raw_usage = getattr(raw, "usage", None)
        prompt_tokens: int = getattr(raw_usage, "input_tokens", 0) if raw_usage else 0
        completion_tokens: int = getattr(raw_usage, "output_tokens", 0) if raw_usage else 0
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

        return _extract_response(
            raw,
            model_id,
            usage,
            structured_tool=_STRUCTURED_OUTPUT_TOOL if json_schema is not None else None,
        )

    def count_tokens(self, text: str) -> int:
        """Approximate token count for *text* using a heuristic (4 chars ≈ 1 token).

        This is intentionally a local estimate — no API call — to keep it
        synchronous and free.  For accurate counts use the Anthropic token-
        counting endpoint separately.
        """
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Helper: extract HTTP status from Anthropic SDK exceptions
# ---------------------------------------------------------------------------


def _extract_status(exc: BaseException) -> int:
    """Return the HTTP status code from an Anthropic SDK exception, or 0."""
    # anthropic.APIStatusError carries .status_code; fall back gracefully.
    return int(getattr(exc, "status_code", 0))


def _tool_call_to_text(tc: ToolCall) -> str:
    """Serialise a ToolCall's arguments to a JSON string (for _structured_output unwrapping)."""
    return json.dumps(tc.arguments)
