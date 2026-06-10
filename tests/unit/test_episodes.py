"""Unit tests for cogworx.knowledge.episodes (Pod 2.3 — Episodic Memory).

All tests are pure-Python: no database, no model, no network.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.knowledge.episodes import EpisodeKind, Turn, stamp_turns, turns_of
from cogworx.loop.result import Done
from cogworx.substrate.journal import StepRecord

_NOW = datetime(2026, 6, 10, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _artifact(data: dict[str, Any] | None = None) -> Artifact:
    return Artifact(
        kind="output",
        produced_by="stage:test",
        provenance=Provenance(source="system", confidence=1.0, recorded_at=_NOW),
        data=data if data is not None else {},
    )


def _step(data: dict[str, Any] | None = None) -> StepRecord:
    """Build a StepRecord whose Done result carries the given output data."""
    return StepRecord(
        run_id="run-01",
        step_index=0,
        stage_name="test_stage",
        result=Done(output=_artifact(data)),
        committed_at=_NOW,
    )


def _step_no_output() -> StepRecord:
    """Build a StepRecord whose result.output is None (uses AwaitHuman with output=None)."""
    from cogworx.loop.result import AwaitHuman

    return StepRecord(
        run_id="run-01",
        step_index=0,
        stage_name="test_stage",
        result=AwaitHuman(question="confirm?", to="next_stage", output=None),
        committed_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stamp_and_roundtrip() -> None:
    """stamp_turns + turns_of round-trips two turns correctly."""
    user_turn = Turn(role="user", content="Hello", kind="conversation")
    assistant_turn = Turn(role="assistant", content="Hi there", kind="conversation")

    data: dict[str, Any] = {}
    stamp_turns(data, [user_turn, assistant_turn])

    step = _step(data)
    result = turns_of(step)

    assert len(result) == 2
    assert result[0] == user_turn
    assert result[1] == assistant_turn


def test_turns_of_no_stamp() -> None:
    """StepRecord with no 'turns' in output.data returns an empty list."""
    step = _step({"other_key": "value"})
    assert turns_of(step) == []


def test_turns_of_none_output() -> None:
    """StepRecord whose result has output=None returns an empty list."""
    step = _step_no_output()
    assert turns_of(step) == []


def test_stamp_empty_raises() -> None:
    """stamp_turns with an empty list raises ValueError."""
    data: dict[str, Any] = {}
    with pytest.raises(ValueError, match="non-empty"):
        stamp_turns(data, [])


def test_turns_of_malformed_raises() -> None:
    """output.data['turns'] is a string (not a list) → ValueError."""
    step = _step({"turns": "not-a-list"})
    with pytest.raises(ValueError, match="expected a list of dicts"):
        turns_of(step)


def test_turns_of_malformed_item_raises() -> None:
    """output.data['turns'] contains a non-dict item → ValueError."""
    step = _step({"turns": ["not-a-dict"]})
    with pytest.raises(ValueError, match="expected a dict"):
        turns_of(step)


def test_stamp_idempotent() -> None:
    """Repeated stamp_turns calls with the same turns produce identical data."""
    turn = Turn(role="user", content="ping", kind="conversation")

    data: dict[str, Any] = {}
    stamp_turns(data, [turn])
    first = list(data["turns"])

    stamp_turns(data, [turn])
    second = list(data["turns"])

    assert first == second


def test_turn_kinds() -> None:
    """Turn accepts all three EpisodeKind values."""
    kinds: list[EpisodeKind] = ["conversation", "tool_exchange", "system_note"]
    for kind in kinds:
        t = Turn(role="system", content="note", kind=kind)
        assert t.kind == kind


def test_turns_of_malformed_turn_dict_raises() -> None:
    """output.data['turns'] contains a dict that fails Turn validation → ValueError."""
    step = _step({"turns": [{"role": "unknown_role", "content": "x", "kind": "conversation"}]})
    with pytest.raises(ValueError, match="failed Turn validation"):
        turns_of(step)
