"""S2/S8 contract tests: the discriminated ``StageResult`` parses each kind correctly."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from pydantic import TypeAdapter

from cogworx.loop.result import AwaitHuman, Degraded, Done, StageResult, Transition

_ADAPTER: TypeAdapter[StageResult] = TypeAdapter(StageResult)


def _artifact() -> dict[str, Any]:
    return {
        "kind": "summary",
        "produced_by": "stage-a",
        "provenance": {
            "source": "reflection",
            "confidence": 0.5,
            "recorded_at": datetime(2026, 6, 8, tzinfo=UTC).isoformat(),
        },
    }


def test_transition_parses() -> None:
    result = _ADAPTER.validate_python({"kind": "transition", "to": "next", "output": _artifact()})
    assert isinstance(result, Transition)


def test_done_parses() -> None:
    result = _ADAPTER.validate_python({"kind": "done", "output": _artifact()})
    assert isinstance(result, Done)


def test_await_human_parses() -> None:
    result = _ADAPTER.validate_python({"kind": "await-human", "question": "ok?", "to": "next"})
    assert isinstance(result, AwaitHuman)
    assert result.to == "next"


def test_degraded_parses() -> None:
    result = _ADAPTER.validate_python({"kind": "degraded", "reason": "x", "output": _artifact()})
    assert isinstance(result, Degraded)
