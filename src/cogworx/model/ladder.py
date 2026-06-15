"""Structured-output degradation ladder for the Model seam (CANON S4, S8, S9).

The framework guarantees that ``StructuredOutputModel.complete`` returns output that
has **passed framework-side jsonschema validation**, or raises ``StructuredOutputError``.
Rung selection is a function of ``ModelCapabilities`` *only* — no provider lookup or
``(provider, capability)`` dict (S4).  The model's self-report is never trusted; every
response is validated structurally (S9).

Three-rung degradation ladder (``DEFAULT_STRUCTURED_OUTPUT_LADDER``):

* **Rung 1 — native structured output** (``capabilities.structured_output is True``):
  passes the JSON schema to the model via the ``json_schema`` parameter so the provider
  can use constrained decoding.  Framework still validates the returned JSON.

* **Rung 2 — constrained tool-call** (``capabilities.tools is True``, no native
  structured output): wraps the schema as a single tool's ``input_schema`` and sends it
  as a forced tool-call.  The tool-call arguments are extracted and validated.

* **Rung 3 — schema-prompted retry** (neither capability): embeds the JSON schema in
  the system prompt, parses the text response as JSON, and on validation failure
  re-prompts with the validation error bound in.  Retries are bounded; exhaustion raises
  ``StructuredOutputError``.

The ``StructuredOutputModel`` wrapper satisfies the ``Model`` Protocol unchanged for
non-structured calls.  For calls that supply ``json_schema=``, the ladder intercepts.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import jsonschema

from cogworx.model.base import (
    ChatMessage,
    Model,
    ModelCapabilities,
    ModelResponse,
    ModelTier,
    ToolCall,
    ToolSpec,
)

__all__ = [
    "DEFAULT_STRUCTURED_OUTPUT_LADDER",
    "LadderRung",
    "StructuredOutputError",
    "StructuredOutputModel",
]

# ---------------------------------------------------------------------------
# Error type
# ---------------------------------------------------------------------------


class StructuredOutputError(Exception):
    """Raised when all ladder rungs are exhausted without producing valid output.

    Callers should treat this as a degraded-mode signal (S8) and route to
    ``await-human`` or a ``degraded`` loop transition.
    """


# ---------------------------------------------------------------------------
# LadderRung protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class LadderRung(Protocol):
    """A single rung of the structured-output degradation ladder.

    ``matches`` returns ``True`` when this rung is eligible for the given
    capabilities.  ``execute`` calls the underlying model and returns validated
    JSON text (already serialised), or raises ``StructuredOutputError`` when the
    rung itself is exhausted (e.g. retries).  It propagates any ``StructuredOutputError``
    from inner rungs transparently.
    """

    def matches(self, caps: ModelCapabilities) -> bool: ...

    async def execute(
        self,
        *,
        model: Model,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        tier: ModelTier,
        json_schema: Mapping[str, Any],
    ) -> ModelResponse: ...


# ---------------------------------------------------------------------------
# Rung 1 — native structured output
# ---------------------------------------------------------------------------


class _NativeRung:
    """Rung 1: provider-native structured output (constrained decoding).

    Used when ``capabilities.structured_output`` is True.  Passes the schema
    directly via ``json_schema=`` and validates the returned JSON framework-side
    (S9 — never trust the model's self-report).
    """

    def matches(self, caps: ModelCapabilities) -> bool:
        return caps.structured_output

    async def execute(
        self,
        *,
        model: Model,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        tier: ModelTier,
        json_schema: Mapping[str, Any],
    ) -> ModelResponse:
        response = await model.complete(
            messages=messages,
            tools=tools,
            tier=tier,
            json_schema=json_schema,
        )
        _validate_text(response, json_schema, rung="native")
        return response


# ---------------------------------------------------------------------------
# Rung 2 — constrained tool-call
# ---------------------------------------------------------------------------

_FORCED_TOOL_NAME = "structured_output"


class _ToolCallRung:
    """Rung 2: forced tool-call with the schema as ``input_schema``.

    Used when ``capabilities.tools`` is True but native structured output is not
    available.  The model is offered a single tool whose ``input_schema`` matches
    the requested schema; the first tool-call's arguments are extracted and
    validated framework-side (S9).
    """

    def matches(self, caps: ModelCapabilities) -> bool:
        return caps.tools and not caps.structured_output

    async def execute(
        self,
        *,
        model: Model,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        tier: ModelTier,
        json_schema: Mapping[str, Any],
    ) -> ModelResponse:
        forced_tool = ToolSpec(
            name=_FORCED_TOOL_NAME,
            description=(
                "Emit your response as structured JSON matching the provided schema. "
                "You MUST call this tool with a valid response."
            ),
            input_schema=dict(json_schema),
        )
        # Merge any caller-supplied tools; our forced tool goes first so the model
        # encounters it as the primary option.
        merged_tools = (forced_tool, *tools)

        response = await model.complete(
            messages=messages,
            tools=merged_tools,
            tier=tier,
            json_schema=None,  # schema delivered via tool, not native param
        )

        # Extract the first matching tool-call and validate its arguments.
        tc = _first_tool_call(response, _FORCED_TOOL_NAME)
        if tc is None:
            raise StructuredOutputError(
                f"Rung 2 (tool-call): model did not call '{_FORCED_TOOL_NAME}'. "
                f"finish_reason={response.finish_reason!r}"
            )

        # Serialise the arguments back to text so the response is uniform with
        # rung 1 and 3 (callers get ``response.text`` with the validated JSON).
        validated_text = _validate_args(tc.arguments, json_schema, rung="tool-call")
        return ModelResponse(
            text=validated_text,
            tool_calls=response.tool_calls,
            reasoning=response.reasoning,
            usage=response.usage,
            model_id=response.model_id,
            finish_reason=response.finish_reason,
        )


# ---------------------------------------------------------------------------
# Rung 3 — schema-prompted retry
# ---------------------------------------------------------------------------

_DEFAULT_MAX_RETRIES: int = 3


class _RetryRung:
    """Rung 3: schema embedded in system prompt, with validation-error rebind on failure.

    Used when neither structured output nor tools are available.  Embeds the JSON
    schema in the system prompt, parses the response as JSON, and on failure
    re-prompts with the validation error message bound in (bounded retries, S11).

    Validation is always framework-side — the model's claim that its output is
    valid is not a control signal (S9).
    """

    def __init__(self, max_retries: int = _DEFAULT_MAX_RETRIES) -> None:
        self._max_retries = max_retries

    def matches(self, caps: ModelCapabilities) -> bool:
        return not caps.structured_output and not caps.tools

    async def execute(
        self,
        *,
        model: Model,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec],
        tier: ModelTier,
        json_schema: Mapping[str, Any],
    ) -> ModelResponse:
        schema_text = json.dumps(json_schema, indent=2)
        system_injection = (
            "You MUST respond with a single JSON object (no markdown, no prose) that "
            f"exactly satisfies this JSON Schema:\n\n{schema_text}"
        )

        # Prepend the schema instruction to the system message, or create one.
        augmented = _inject_schema_instruction(list(messages), system_injection)

        last_response: ModelResponse | None = None
        last_error: str = ""

        for attempt in range(self._max_retries):
            call_messages: list[ChatMessage] = list(augmented)
            if attempt > 0 and last_response is not None:
                prior = last_response.text or "(empty response)"
                call_messages.append(
                    ChatMessage(
                        role="user",
                        content=(
                            "Your previous response failed JSON Schema validation.\n"
                            f"Validation error: {last_error}\n"
                            "Previous response (shown for reference only"
                            " — it is INVALID, do not repeat it):\n"
                            f"{prior}\n\n"
                            "Respond again with a single JSON object that satisfies"
                            " the schema. JSON only."
                        ),
                    )
                )

            response = await model.complete(
                messages=call_messages,
                tools=tools,
                tier=tier,
                json_schema=None,
            )
            last_response = response

            error = _check_text(response, json_schema)
            if error is None:
                # Validation passed — return the response as-is.
                return response
            last_error = error

        raise StructuredOutputError(
            f"Rung 3 (retry): exhausted {self._max_retries} attempts. "
            f"Last validation error: {last_error}"
        )


# ---------------------------------------------------------------------------
# Default ladder
# ---------------------------------------------------------------------------

DEFAULT_STRUCTURED_OUTPUT_LADDER: tuple[LadderRung, ...] = (
    _NativeRung(),
    _ToolCallRung(),
    _RetryRung(),
)
"""The canonical three-rung degradation ladder (S4, S8, S9).

Override at ``StructuredOutputModel`` construction time — that override IS the
entire "registry" (no separate registry dict needed; see CANON S4).
"""


# ---------------------------------------------------------------------------
# StructuredOutputModel
# ---------------------------------------------------------------------------


class StructuredOutputModel:
    """Model wrapper that guarantees framework-validated structured output (CANON S4, S8, S9).

    Wraps any ``Model`` implementation and selects the appropriate degradation
    rung at call time based solely on ``capabilities`` (S4 — no provider dict).
    For calls that do not supply ``json_schema=``, the wrapped model is called
    transparently with no ladder overhead.

    Args:
        model:  The underlying ``Model`` implementation to wrap.
        ladder: An ordered tuple of ``LadderRung`` objects.  The first rung
                whose ``matches(capabilities)`` returns ``True`` is used.
                Defaults to ``DEFAULT_STRUCTURED_OUTPUT_LADDER``.
    """

    def __init__(
        self,
        model: Model,
        *,
        ladder: tuple[LadderRung, ...] = DEFAULT_STRUCTURED_OUTPUT_LADDER,
    ) -> None:
        self._model = model
        self._ladder = ladder

    # ------------------------------------------------------------------
    # Model Protocol
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> ModelCapabilities:
        return self._model.capabilities

    async def complete(
        self,
        *,
        messages: Sequence[ChatMessage],
        tools: Sequence[ToolSpec] = (),
        tier: ModelTier = "pro",
        json_schema: Mapping[str, Any] | None = None,
    ) -> ModelResponse:
        """Call the wrapped model, applying the degradation ladder when ``json_schema`` is given.

        Returns a ``ModelResponse`` whose ``text`` field contains framework-validated
        JSON (already serialised).  Raises ``StructuredOutputError`` if all applicable
        ladder rungs fail.
        """
        if json_schema is None:
            # Pass-through: no structured-output guarantee requested.
            return await self._model.complete(
                messages=messages,
                tools=tools,
                tier=tier,
                json_schema=None,
            )

        caps = self._model.capabilities
        for rung in self._ladder:
            if rung.matches(caps):
                return await rung.execute(
                    model=self._model,
                    messages=messages,
                    tools=tools,
                    tier=tier,
                    json_schema=json_schema,
                )

        raise StructuredOutputError(
            "No rung in the ladder matched the model's capabilities: "
            f"{caps!r}.  Add a catch-all rung or supply a custom ladder."
        )

    def count_tokens(self, text: str) -> int:
        return self._model.count_tokens(text)


# ---------------------------------------------------------------------------
# Internal validation helpers (S9 — framework always validates)
# ---------------------------------------------------------------------------


def _validate_text(
    response: ModelResponse,
    schema: Mapping[str, Any],
    *,
    rung: str,
) -> None:
    """Parse ``response.text`` as JSON and validate against ``schema``.

    Raises ``StructuredOutputError`` on parse or validation failure.
    """
    error = _check_text(response, schema)
    if error is not None:
        raise StructuredOutputError(f"Rung {rung!r}: {error}")


def _check_text(
    response: ModelResponse,
    schema: Mapping[str, Any],
) -> str | None:
    """Return a validation error string, or ``None`` if the response is valid."""
    raw = response.text or ""
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        return f"JSON parse error: {exc}"
    try:
        jsonschema.validate(instance=parsed, schema=dict(schema))
    except jsonschema.ValidationError as exc:
        return f"Schema validation error: {exc.message}"
    return None


def _validate_args(
    arguments: dict[str, Any],
    schema: Mapping[str, Any],
    *,
    rung: str,
) -> str:
    """Validate tool-call ``arguments`` against ``schema`` and return serialised JSON.

    Raises ``StructuredOutputError`` on validation failure.
    """
    try:
        jsonschema.validate(instance=arguments, schema=dict(schema))
    except jsonschema.ValidationError as exc:
        raise StructuredOutputError(
            f"Rung {rung!r}: tool-call arguments failed schema validation: {exc.message}"
        ) from exc
    return json.dumps(arguments)


def _first_tool_call(response: ModelResponse, name: str) -> ToolCall | None:
    """Return the first ``ToolCall`` matching ``name``, or ``None``."""
    for tc in response.tool_calls:
        if tc.name == name:
            return tc
    return None


def _inject_schema_instruction(
    messages: list[ChatMessage],
    instruction: str,
) -> list[ChatMessage]:
    """Prepend ``instruction`` to the first system message, or insert one at index 0."""
    for i, msg in enumerate(messages):
        if msg.role == "system":
            combined = f"{instruction}\n\n{msg.content}"
            messages[i] = ChatMessage(role="system", content=combined)
            return messages
    messages.insert(0, ChatMessage(role="system", content=instruction))
    return messages
