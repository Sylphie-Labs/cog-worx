"""Model-agnostic interface (CANON S4).

The framework talks to models through a thin ``Model`` seam; provider is selectable per agent
(Claude / DeepSeek / Ollama / any OpenAI-compatible). Premium capabilities (structured output,
caching, streaming, logprobs, tools) are declared on ``ModelCapabilities`` and degrade gracefully
when a provider lacks them. Generalized from tess ``LLMClient`` (tess/tess/llm.py:141).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

ModelTier = Literal["pro", "flash"]


class ChatMessage(BaseModel):
    model_config = ConfigDict(frozen=True)

    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    arguments: dict[str, Any]


class Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0


class ModelResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    reasoning: str | None = None
    usage: Usage = Field(default_factory=Usage)
    model_id: str
    finish_reason: str


class ModelCapabilities(BaseModel):
    model_config = ConfigDict(frozen=True)

    structured_output: bool = False
    streaming: bool = False
    caching: bool = False
    logprobs: bool = False
    tools: bool = False


@runtime_checkable
class Model(Protocol):
    """The per-agent, provider-agnostic model seam."""

    @property
    def capabilities(self) -> ModelCapabilities: ...

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse: ...

    def count_tokens(self, text: str) -> int: ...


__all__ = [
    "ChatMessage",
    "Model",
    "ModelCapabilities",
    "ModelResponse",
    "ModelTier",
    "ToolCall",
    "ToolSpec",
    "Usage",
]
