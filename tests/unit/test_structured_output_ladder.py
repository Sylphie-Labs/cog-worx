"""Unit tests for StructuredOutputModel and the three-rung degradation ladder (CANON S4, S8, S9).

All tests are pure async — no subprocess, no substrate, no real model calls (S1).
Uses ReplayModel from the Test Kit to drive each rung by capability configuration.

Rung coverage:
* Rung 1 (native)          — capabilities.structured_output=True
* Rung 2 (tool-call)       — capabilities.tools=True, structured_output=False
* Rung 3 (retry-rebind)    — both False
* Rung 3 retry path        — invalid then valid
* Rung 3 exhausted         — raises StructuredOutputError
"""

from __future__ import annotations

import json

import pytest

from cogworx.model.base import ModelCapabilities, ModelResponse, ToolCall, Usage
from cogworx.model.ladder import (
    StructuredOutputError,
    StructuredOutputModel,
)
from cogworx.testing.fake_model import ReplayModel

# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

_SCHEMA: dict[str, object] = {
    "type": "object",
    "properties": {"name": {"type": "string"}},
    "required": ["name"],
    "additionalProperties": False,
}

_VALID_JSON = json.dumps({"name": "alice"})
_INVALID_JSON = json.dumps({"wrong_field": 42})
_NOT_JSON = "this is not JSON at all"


def _caps(
    *,
    structured_output: bool = False,
    tools: bool = False,
) -> ModelCapabilities:
    return ModelCapabilities(structured_output=structured_output, tools=tools)


def _response(text: str | None = None, *, tool_calls: tuple[ToolCall, ...] = ()) -> ModelResponse:
    return ModelResponse(
        text=text,
        tool_calls=tool_calls,
        model_id="replay",
        finish_reason="stop",
        usage=Usage(),
    )


def _tool_call_response(arguments: dict[str, object]) -> ModelResponse:
    """ModelResponse containing a single structured_output tool-call."""
    tc = ToolCall(id="tc1", name="structured_output", arguments=arguments)
    return _response(tool_calls=(tc,))


# ---------------------------------------------------------------------------
# Rung 1 — native structured output
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rung1_native_passes_schema_to_model() -> None:
    """Rung 1 passes json_schema= to the underlying model and the recorded call reflects it."""
    model = ReplayModel(
        [_response(_VALID_JSON)],
        capabilities=_caps(structured_output=True),
    )
    wrapper = StructuredOutputModel(model)
    result = await wrapper.complete(
        messages=[],
        json_schema=_SCHEMA,
    )
    assert result.text == _VALID_JSON
    assert model.call_count == 1
    call = model.calls[0]
    # The schema must have been forwarded to the underlying model.
    assert call.json_schema is not None
    assert call.json_schema.get("type") == "object"


@pytest.mark.asyncio
async def test_rung1_native_framework_validates_output() -> None:
    """Rung 1 raises StructuredOutputError when the model returns invalid JSON despite
    claiming structured-output capability (S9 — never trust the model's self-report)."""
    model = ReplayModel(
        [_response(_INVALID_JSON)],
        capabilities=_caps(structured_output=True),
    )
    wrapper = StructuredOutputModel(model)
    with pytest.raises(StructuredOutputError):
        await wrapper.complete(messages=[], json_schema=_SCHEMA)


@pytest.mark.asyncio
async def test_rung1_not_selected_when_structured_output_false() -> None:
    """When structured_output=False, rung 1 must NOT be selected.

    We confirm by checking that json_schema is NOT forwarded to the model
    (rung 2 deliberately passes json_schema=None to the underlying model).
    """
    # Rung 2 path: tools=True, structured_output=False
    model = ReplayModel(
        [_tool_call_response({"name": "bob"})],
        capabilities=_caps(tools=True, structured_output=False),
    )
    wrapper = StructuredOutputModel(model)
    await wrapper.complete(messages=[], json_schema=_SCHEMA)
    call = model.calls[0]
    # Rung 2 passes json_schema=None (it delivers schema via tool input_schema instead).
    assert call.json_schema is None


# ---------------------------------------------------------------------------
# Rung 2 — constrained tool-call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rung2_tool_call_selected_when_tools_only() -> None:
    """Rung 2 is selected when tools=True but structured_output=False."""
    model = ReplayModel(
        [_tool_call_response({"name": "carol"})],
        capabilities=_caps(tools=True, structured_output=False),
    )
    wrapper = StructuredOutputModel(model)
    result = await wrapper.complete(messages=[], json_schema=_SCHEMA)

    # Result text is the validated arguments serialised as JSON.
    assert result.text is not None
    parsed = json.loads(result.text)
    assert parsed == {"name": "carol"}

    # Model was called with a tool whose input_schema matches the requested schema.
    call = model.calls[0]
    assert len(call.tools) >= 1
    forced = next(t for t in call.tools if t.name == "structured_output")
    assert forced.input_schema.get("type") == "object"


@pytest.mark.asyncio
async def test_rung2_raises_when_model_does_not_call_tool() -> None:
    """Rung 2 raises StructuredOutputError when the model returns text instead of a tool-call."""
    model = ReplayModel(
        [_response(_VALID_JSON)],  # text response, no tool-call
        capabilities=_caps(tools=True, structured_output=False),
    )
    wrapper = StructuredOutputModel(model)
    with pytest.raises(StructuredOutputError):
        await wrapper.complete(messages=[], json_schema=_SCHEMA)


@pytest.mark.asyncio
async def test_rung2_validates_tool_call_arguments() -> None:
    """Rung 2 framework-validates the tool-call arguments (S9)."""
    bad_args: dict[str, object] = {"wrong_field": 99}
    model = ReplayModel(
        [_tool_call_response(bad_args)],
        capabilities=_caps(tools=True, structured_output=False),
    )
    wrapper = StructuredOutputModel(model)
    with pytest.raises(StructuredOutputError):
        await wrapper.complete(messages=[], json_schema=_SCHEMA)


# ---------------------------------------------------------------------------
# Rung 3 — schema-prompted retry
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rung3_selected_when_no_capabilities() -> None:
    """Rung 3 is selected when structured_output=False and tools=False."""
    model = ReplayModel(
        [_response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    result = await wrapper.complete(messages=[], json_schema=_SCHEMA)
    assert result.text == _VALID_JSON
    assert model.call_count == 1


@pytest.mark.asyncio
async def test_rung3_system_prompt_injects_schema() -> None:
    """Rung 3 injects the JSON schema into the system prompt."""
    model = ReplayModel(
        [_response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    await wrapper.complete(messages=[], json_schema=_SCHEMA)

    call = model.calls[0]
    # A system message must have been synthesised containing the schema.
    system_msgs = [m for m in call.messages if m.role == "system"]
    assert system_msgs, "expected an injected system message"
    assert "JSON Schema" in system_msgs[0].content


@pytest.mark.asyncio
async def test_rung3_retry_rebind_on_invalid_then_valid() -> None:
    """Rung 3 re-prompts with the validation error when the first response is invalid,
    and succeeds on the second attempt."""
    model = ReplayModel(
        [_response(_INVALID_JSON), _response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    result = await wrapper.complete(messages=[], json_schema=_SCHEMA)

    assert result.text == _VALID_JSON
    assert model.call_count == 2

    # After F7: no assistant role in retry; a single user turn carries the error
    second_call = model.calls[1]
    roles = [m.role for m in second_call.messages]
    assert "assistant" not in roles  # F7: no synthetic assistant turn
    assert "user" in roles
    user_correction = next(m for m in reversed(second_call.messages) if m.role == "user")
    assert "Validation error" in user_correction.content
    assert "JSON Schema validation" in user_correction.content


@pytest.mark.asyncio
async def test_rung3_exhausted_retries_raise_structured_output_error() -> None:
    """Rung 3 raises StructuredOutputError after exhausting all retry attempts."""
    # Default max_retries=3; provide three invalid responses.
    model = ReplayModel(
        [_response(_INVALID_JSON)] * 3,
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    with pytest.raises(StructuredOutputError, match="exhausted"):
        await wrapper.complete(messages=[], json_schema=_SCHEMA)

    assert model.call_count == 3


@pytest.mark.asyncio
async def test_rung3_not_json_parse_failure_counts_as_invalid() -> None:
    """Rung 3 treats a non-JSON text response as a validation failure (not a crash)."""
    model = ReplayModel(
        [_response(_NOT_JSON), _response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    result = await wrapper.complete(messages=[], json_schema=_SCHEMA)
    assert result.text == _VALID_JSON
    assert model.call_count == 2


# ---------------------------------------------------------------------------
# Pass-through (no json_schema) — ladder is not engaged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_passthrough_when_no_schema() -> None:
    """When json_schema is None the ladder is bypassed entirely."""
    plain_text = "hello world"
    model = ReplayModel(
        [_response(plain_text)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    result = await wrapper.complete(messages=[], json_schema=None)
    assert result.text == plain_text
    call = model.calls[0]
    assert call.json_schema is None


# ---------------------------------------------------------------------------
# Ladder override at construction
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_custom_ladder_override() -> None:
    """A custom ladder can be supplied at construction — the default is not used."""
    from cogworx.model.ladder import LadderRung, _NativeRung

    # Provide only rung 1 in the custom ladder — even though the model has no
    # structured_output capability, the rung's matches() will return False and no
    # rung will match → StructuredOutputError from the "no rung matched" guard.
    custom_ladder: tuple[LadderRung, ...] = (_NativeRung(),)
    model = ReplayModel(
        [],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model, ladder=custom_ladder)
    with pytest.raises(StructuredOutputError, match="No rung"):
        await wrapper.complete(messages=[], json_schema=_SCHEMA)


# ---------------------------------------------------------------------------
# Model Protocol compliance
# ---------------------------------------------------------------------------


def test_structured_output_model_satisfies_model_protocol() -> None:
    """StructuredOutputModel itself satisfies the runtime-checkable Model Protocol."""
    from cogworx.model.base import Model

    model = ReplayModel(capabilities=_caps(structured_output=True))
    wrapper = StructuredOutputModel(model)
    assert isinstance(wrapper, Model)


# ---------------------------------------------------------------------------
# F7: retry message shape — no synthetic assistant turn
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rung3_retry_no_assistant_message_beyond_caller_supplied() -> None:
    """On attempt 2, the ladder appends a user turn only — no assistant-role message (F7)."""
    model = ReplayModel(
        [_response(_INVALID_JSON), _response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    await wrapper.complete(messages=[], json_schema=_SCHEMA)

    second_call = model.calls[1]
    roles = [m.role for m in second_call.messages]
    assert "assistant" not in roles, (
        "F7: ladder must NOT append a synthetic assistant turn on retry"
    )


@pytest.mark.asyncio
async def test_rung3_retry_user_turn_contains_validation_error() -> None:
    """The appended user turn on retry contains the validation error string (F7)."""
    model = ReplayModel(
        [_response(_INVALID_JSON), _response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    await wrapper.complete(messages=[], json_schema=_SCHEMA)

    second_call = model.calls[1]
    user_turn = next(m for m in reversed(second_call.messages) if m.role == "user")
    assert "Validation error" in user_turn.content
    assert "JSON Schema validation" in user_turn.content


@pytest.mark.asyncio
async def test_rung3_retry_no_empty_content_message() -> None:
    """No message appended by the ladder has empty content (F7)."""
    model = ReplayModel(
        [_response(_INVALID_JSON), _response(_VALID_JSON)],
        capabilities=_caps(structured_output=False, tools=False),
    )
    wrapper = StructuredOutputModel(model)
    await wrapper.complete(messages=[], json_schema=_SCHEMA)

    second_call = model.calls[1]
    for msg in second_call.messages:
        assert msg.content, f"Message with role {msg.role!r} has empty content"
