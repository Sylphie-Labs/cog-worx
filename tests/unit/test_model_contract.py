"""S4 contract test: a structural stub satisfies the runtime_checkable ``Model`` Protocol."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cogworx.model.base import (
    ChatMessage,
    Model,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
)


class _StubModel:
    @property
    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities()

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        return ModelResponse(text="ok", model_id="stub", finish_reason="stop")


def test_stub_satisfies_model_protocol() -> None:
    assert isinstance(_StubModel(), Model)
