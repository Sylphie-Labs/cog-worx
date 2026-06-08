"""Scripted ``Model`` double for the Test Kit (CANON S4, S1, S6).

``ReplayModel`` returns canned ``ModelResponse``s and records every call, so the deterministic
invariant suites can assert "no model call happened on this path" (S1 off-write-path, S6 no model
re-call on replay). Generalized from tess ``StubLLMClient`` (tess/tess/llm.py:301).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ConfigDict

from cogworx.model.base import (
    ChatMessage,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
)


class ReplayExhaustedError(Exception):
    """Raised when ``ReplayModel.complete`` is called with no scripted response left."""


class ReplayCall(BaseModel):
    model_config = ConfigDict(frozen=True)

    messages: tuple[ChatMessage, ...]
    tools: tuple[ToolSpec, ...]
    tier: ModelTier
    json_schema: dict[str, Any] | None = None


class ReplayModel:
    """A scripted ``Model`` that returns canned responses and spies on every call."""

    def __init__(
        self,
        responses: Sequence[ModelResponse] = (),
        *,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        self._responses: list[ModelResponse] = list(responses)
        self._calls: list[ReplayCall] = []
        self._capabilities = capabilities or ModelCapabilities(structured_output=True, tools=True)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._capabilities

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        self._calls.append(
            ReplayCall(
                messages=tuple(messages),
                tools=tuple(tools),
                tier=tier,
                json_schema=dict(json_schema) if json_schema is not None else None,
            )
        )
        if not self._responses:
            raise ReplayExhaustedError(
                f"ReplayModel exhausted: no scripted response for call #{len(self._calls)}"
            )
        return self._responses.pop(0)

    @property
    def calls(self) -> tuple[ReplayCall, ...]:
        return tuple(self._calls)

    @property
    def call_count(self) -> int:
        return len(self._calls)


def echo_model(text: str) -> ReplayModel:
    """A ``ReplayModel`` with a single response echoing ``text``."""

    return ReplayModel([ModelResponse(text=text, model_id="replay", finish_reason="stop")])


__all__ = [
    "ReplayCall",
    "ReplayExhaustedError",
    "ReplayModel",
    "echo_model",
]
