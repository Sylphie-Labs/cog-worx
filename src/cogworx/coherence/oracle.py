"""Consistency oracle seam — production implementation + protocol (CANON S4, S9).

The ``ConsistencyOracle`` protocol is the single public seam; routing code MUST read only
``OracleAnswer.consistent`` (the control bit).  ``raw_text`` is stored for audit/log only and MUST
NOT be parsed for control anywhere in the codebase (S9 — structure over prompting).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Claim
from cogworx.model.base import ChatMessage, Model, ModelTier


class OracleAnswer(BaseModel):
    model_config = ConfigDict(frozen=True)

    consistent: bool
    """THE control bit — the ONLY field routing may read (S9)."""
    model_ref: str | None = None
    """Provenance only — which model produced this answer."""
    raw_text: str | None = None
    """Audit/log only — NEVER parsed for control flow."""


class OracleProtocolError(Exception):
    """Raised when the model returns a response that cannot be parsed as CONSISTENT/INCONSISTENT."""


class ConsistencyOracle(Protocol):
    """Structural protocol for any consistency oracle (real or test double)."""

    async def check(self, claims: Sequence[Claim]) -> OracleAnswer: ...


_BINARY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "answer": {"type": "string", "enum": ["CONSISTENT", "INCONSISTENT"]},
    },
    "required": ["answer"],
    "additionalProperties": False,
}


def _render_claim(i: int, claim: Claim) -> str:
    predicate = getattr(claim, "predicate", None) or "—"
    payload = getattr(claim, "payload", str(claim))
    valid_from = getattr(claim, "valid_from", None)
    valid_to = getattr(claim, "valid_to", None)
    subject = getattr(claim, "subject", str(claim))

    time_range = str(valid_from) if valid_from is not None else "unknown"
    if valid_to is not None:
        time_range = f"{time_range}, to {valid_to}"
    return f"{i}. {subject} — {predicate} — {payload} (valid from {time_range})"


class ModelConsistencyOracle:
    """Production consistency oracle — calls Model.complete with a binary constrained prompt.

    If the model declares ``structured_output`` capability, passes the binary JSON schema so the
    provider can use constrained decoding.  Otherwise, falls back to token-match on the text
    response (S4 graceful degradation).
    """

    def __init__(self, model: Model, *, tier: ModelTier = "flash") -> None:
        self._model = model
        self._tier = tier

    async def check(self, claims: Sequence[Claim]) -> OracleAnswer:
        rendered = "\n".join(_render_claim(i + 1, c) for i, c in enumerate(claims))
        system_msg = ChatMessage(
            role="system",
            content=(
                "Given these claims, answer with exactly one token: CONSISTENT or INCONSISTENT."
            ),
        )
        user_msg = ChatMessage(role="user", content=rendered)

        use_structured = self._model.capabilities.structured_output
        response = await self._model.complete(
            messages=[system_msg, user_msg],
            tier=self._tier,
            json_schema=_BINARY_SCHEMA if use_structured else None,
        )

        raw_text = response.text
        model_ref = getattr(response, "model_id", None)

        # Structured output path: parse the JSON answer field.
        if use_structured and raw_text is not None:
            import json

            try:
                parsed = json.loads(raw_text)
                token = str(parsed.get("answer", "")).strip().upper()
            except (json.JSONDecodeError, AttributeError):
                token = raw_text.strip().upper()
        else:
            token = (raw_text or "").strip().upper()

        if token == "CONSISTENT":
            return OracleAnswer(consistent=True, model_ref=model_ref, raw_text=raw_text)
        if token == "INCONSISTENT":
            return OracleAnswer(consistent=False, model_ref=model_ref, raw_text=raw_text)

        raise OracleProtocolError(
            f"Oracle returned unparseable response: {raw_text!r}; "
            "expected exactly 'CONSISTENT' or 'INCONSISTENT'."
        )


__all__ = [
    "ConsistencyOracle",
    "ModelConsistencyOracle",
    "OracleAnswer",
    "OracleProtocolError",
]
