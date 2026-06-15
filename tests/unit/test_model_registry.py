"""Tests for ``ModelRegistry`` and ``build_model`` (CANON S4, S6, S11).

Verifies:
- ``register`` / ``resolve`` round-trip for pre-assembled models.
- ``register_spec`` / ``resolve_spec`` round-trip for ``ModelSpec`` descriptors.
- Loud ``ModelRegistryError`` on lookup miss (S6 — never silent).
- Loud ``ModelRegistryError`` on duplicate registration.
- ``build_model`` composition order: ``StructuredOutputModel(BudgetGuardedModel(adapter, guard))``.
- Budget guard is innermost: a zero-budget guard raises ``BudgetExceededError`` BEFORE the ladder
  can consult the underlying model (S11 pre-call, S9 structural).
- ``has`` / ``has_spec`` predicates.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import pytest

from cogworx.cost.budget import BudgetExceededError, BudgetGuard
from cogworx.model.base import (
    ChatMessage,
    Model,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolSpec,
    Usage,
)
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.ladder import StructuredOutputModel
from cogworx.model.providers.config import PriceTable, ProviderConfig
from cogworx.model.registry import ModelRegistry, ModelRegistryError, ModelSpec
from cogworx.testing.fake_model import ReplayModel

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_ZERO_PRICE = PriceTable(
    pro_input_usd_per_mtok=0.0,
    pro_output_usd_per_mtok=0.0,
    flash_input_usd_per_mtok=0.0,
    flash_output_usd_per_mtok=0.0,
)

_CLAUDE_CONFIG = ProviderConfig(
    model_pro="claude-3-5-sonnet-20241022",
    model_flash="claude-3-5-haiku-20241022",
    price_per_mtok=_ZERO_PRICE,
)

_OPENAI_CONFIG = ProviderConfig(
    model_pro="gpt-4o",
    model_flash="gpt-4o-mini",
    price_per_mtok=_ZERO_PRICE,
)


def _ok_response() -> ModelResponse:
    return ModelResponse(
        text='{"result": "ok"}',
        model_id="test",
        finish_reason="stop",
        usage=Usage(cost_usd=0.0),
    )


def _messages() -> list[ChatMessage]:
    return [ChatMessage(role="user", content="hello")]


# ---------------------------------------------------------------------------
# Minimal stub adapter that records calls (avoids real provider SDKs in tests)
# ---------------------------------------------------------------------------


class _StubAdapter:
    """Minimal Model-protocol implementation for registry tests (no network calls)."""

    def __init__(self, call_count_ref: list[int] | None = None) -> None:
        self._counter = call_count_ref if call_count_ref is not None else []
        self._caps = ModelCapabilities(structured_output=True, tools=True)

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._caps

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        self._counter.append(1)
        return _ok_response()

    def count_tokens(self, text: str) -> int:
        return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# ModelRegistry — register / resolve round-trip
# ---------------------------------------------------------------------------


def test_register_resolve_roundtrip() -> None:
    """``register`` stores a model; ``resolve`` returns the same instance."""
    registry = ModelRegistry()
    model: Model = ReplayModel()
    registry.register("researcher", model)
    assert registry.resolve("researcher") is model


def test_register_has_and_has_not() -> None:
    """``has`` returns True after registration, False before."""
    registry = ModelRegistry()
    assert not registry.has("researcher")
    registry.register("researcher", ReplayModel())
    assert registry.has("researcher")


def test_resolve_miss_raises_loud_error() -> None:
    """``resolve`` raises ``ModelRegistryError`` on a miss — never silent (CANON S6)."""
    registry = ModelRegistry()
    with pytest.raises(ModelRegistryError, match="researcher"):
        registry.resolve("researcher")


def test_register_duplicate_raises_error() -> None:
    """A second ``register`` for the same profile raises ``ModelRegistryError``."""
    registry = ModelRegistry()
    registry.register("researcher", ReplayModel())
    with pytest.raises(ModelRegistryError, match="researcher"):
        registry.register("researcher", ReplayModel())


# ---------------------------------------------------------------------------
# ModelRegistry — register_spec / resolve_spec round-trip
# ---------------------------------------------------------------------------


def test_register_spec_resolve_spec_roundtrip() -> None:
    """``register_spec`` stores a spec; ``resolve_spec`` returns the same instance."""
    registry = ModelRegistry()
    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    registry.register_spec("researcher", spec)
    assert registry.resolve_spec("researcher") is spec


def test_resolve_spec_miss_raises_loud_error() -> None:
    """``resolve_spec`` raises ``ModelRegistryError`` on a miss — never silent (CANON S6)."""
    registry = ModelRegistry()
    with pytest.raises(ModelRegistryError, match="researcher"):
        registry.resolve_spec("researcher")


def test_register_spec_duplicate_raises_error() -> None:
    """A second ``register_spec`` for the same profile raises ``ModelRegistryError``."""
    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    registry = ModelRegistry()
    registry.register_spec("researcher", spec)
    with pytest.raises(ModelRegistryError, match="researcher"):
        registry.register_spec("researcher", spec)


def test_has_spec_predicate() -> None:
    """``has_spec`` reflects spec registration state."""
    registry = ModelRegistry()
    assert not registry.has_spec("researcher")
    registry.register_spec("researcher", ModelSpec(provider="claude", config=_CLAUDE_CONFIG))
    assert registry.has_spec("researcher")


# ---------------------------------------------------------------------------
# ModelRegistryError — content checks
# ---------------------------------------------------------------------------


def test_resolve_miss_error_lists_registered_profiles() -> None:
    """The miss error message names the registered profiles to aid debugging."""
    registry = ModelRegistry()
    registry.register("writer", ReplayModel())
    with pytest.raises(ModelRegistryError) as exc_info:
        registry.resolve("researcher")
    assert "researcher" in str(exc_info.value)
    assert "writer" in str(exc_info.value)


# ---------------------------------------------------------------------------
# build_model — composition order verification
#
# ``build_model`` constructs real provider SDK clients (ClaudeModel / OpenAICompatModel),
# which require valid credentials at construction time on some SDK versions.  The composition
# ORDER tests — which are the load-bearing invariant (S11 guard is innermost) — are tested via
# manual stack construction with ``_StubAdapter``, ``BudgetGuardedModel``, and
# ``StructuredOutputModel`` directly.  This is the correct seam: the stack wiring is what
# matters, not which provider backs the adapter.
#
# Integration-style tests that exercise ``build_model`` end-to-end (with real SDK clients) live
# in the integration tier and require the ``providers`` extra + real credentials.
# ---------------------------------------------------------------------------


async def test_build_model_guard_innermost_async() -> None:
    """build_model assembles the full stack; a zero-budget guard raises before the adapter.

    Composition order under test:
        StructuredOutputModel        <- outermost
          └─ BudgetGuardedModel      <- middle (guard fires here)
               └─ ClaudeModel        <- innermost (must NOT be called)
    """
    from unittest.mock import AsyncMock, MagicMock

    from cogworx.model.registry import build_model

    fake = MagicMock()
    fake.messages = MagicMock()
    fake.messages.create = AsyncMock()

    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    model = build_model(spec, guard=BudgetGuard(max_calls=0), client=fake)

    with pytest.raises(BudgetExceededError):
        await model.complete(messages=_messages())

    # The fake client was never called — the guard blocked before the adapter.
    assert fake.messages.create.call_count == 0, (
        "Adapter was called despite zero-budget guard: budget guard is not innermost — "
        "composition order is WRONG (CANON S11)"
    )


async def test_build_model_guard_innermost_with_json_schema() -> None:
    """Guard fires BEFORE the ladder attempts a structured-output call via build_model.

    When ``json_schema`` is supplied the ``StructuredOutputModel`` ladder selects a rung and
    calls the underlying model.  With ``max_calls=0`` the ``BudgetGuardedModel`` must raise
    BEFORE the adapter is reached — the ladder never gets to consult the adapter.
    """
    from unittest.mock import AsyncMock, MagicMock

    from cogworx.model.registry import build_model

    fake = MagicMock()
    fake.messages = MagicMock()
    fake.messages.create = AsyncMock()

    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    model = build_model(spec, guard=BudgetGuard(max_calls=0), client=fake)

    schema: dict[str, object] = {
        "type": "object",
        "properties": {"result": {"type": "string"}},
        "required": ["result"],
    }
    with pytest.raises(BudgetExceededError):
        await model.complete(messages=_messages(), json_schema=schema)

    assert fake.messages.create.call_count == 0, (
        "Adapter was called during a json_schema call despite zero-budget guard — "
        "composition order is WRONG (CANON S11)"
    )


async def test_build_model_successful_call_passes_through() -> None:
    """A non-budget-exhausted call flows through all three layers and returns a response."""
    counter: list[int] = []
    adapter = _StubAdapter(counter)
    guard = BudgetGuard(max_calls=2)
    guarded = BudgetGuardedModel(adapter, guard)
    model = StructuredOutputModel(guarded)

    response = await model.complete(messages=_messages())

    assert response.text == '{"result": "ok"}'
    assert counter == [1]  # adapter was called exactly once
    assert guard.calls == 1


# ---------------------------------------------------------------------------
# Capabilities pass-through — verified via manual stack (no real SDK needed)
# ---------------------------------------------------------------------------


def test_capabilities_passthrough_structured_output_false() -> None:
    """When the adapter declares structured_output=False the stack propagates that capability.

    Verifies the S4 / S9 invariant: capabilities flow through both wrappers unchanged.
    """
    caps = ModelCapabilities(structured_output=False, tools=False)
    adapter = ReplayModel(capabilities=caps)
    guard = BudgetGuard()
    guarded = BudgetGuardedModel(adapter, guard)
    model = StructuredOutputModel(guarded)

    assert model.capabilities.structured_output is False
    assert model.capabilities.tools is False


def test_capabilities_passthrough_full_capabilities() -> None:
    """Adapter with full capabilities → all capabilities propagate through the stack."""
    caps = ModelCapabilities(structured_output=True, tools=True, streaming=True, caching=True)
    adapter = ReplayModel(capabilities=caps)
    guard = BudgetGuard()
    guarded = BudgetGuardedModel(adapter, guard)
    model = StructuredOutputModel(guarded)

    assert model.capabilities.structured_output is True
    assert model.capabilities.tools is True
    assert model.capabilities.streaming is True
    assert model.capabilities.caching is True


# ---------------------------------------------------------------------------
# ModelSpec — frozen value type
# ---------------------------------------------------------------------------


def test_model_spec_is_frozen() -> None:
    """``ModelSpec`` is a frozen pydantic model (immutable value type)."""
    import pydantic

    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    with pytest.raises(pydantic.ValidationError):
        spec.provider = "openai_compat"  # type: ignore[misc]


def test_model_spec_claude_no_capabilities_required() -> None:
    """``ModelSpec`` for claude does not require ``capabilities`` (ClaudeModel declares its own)."""
    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    assert spec.capabilities is None


def test_model_spec_openai_compat_with_capabilities() -> None:
    """``ModelSpec`` for openai_compat accepts an explicit ``capabilities``."""
    from cogworx.model.providers.openai_compat import DEEPSEEK_CAPABILITIES

    spec = ModelSpec(
        provider="openai_compat", config=_OPENAI_CONFIG, capabilities=DEEPSEEK_CAPABILITIES
    )
    assert spec.capabilities is DEEPSEEK_CAPABILITIES


# ---------------------------------------------------------------------------
# F9: build_model client kwarg
# ---------------------------------------------------------------------------


def test_build_model_with_client_kwarg() -> None:
    """build_model accepts a ``client=`` kwarg and returns a Model without error."""
    from unittest.mock import MagicMock

    from cogworx.model.base import Model
    from cogworx.model.registry import build_model

    fake = MagicMock()
    fake.messages = MagicMock()

    spec = ModelSpec(provider="claude", config=_CLAUDE_CONFIG)
    model = build_model(spec, guard=BudgetGuard(max_calls=5), client=fake)

    assert isinstance(model, Model)
