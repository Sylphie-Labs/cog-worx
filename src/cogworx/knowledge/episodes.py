"""Turn stamping for episodic memory capture (CANON S1, S3, S6).

Capture is turn-stamping: a stage stamps output.data["turns"] = [Turn(...)].
The EpisodeProjector (runtime/) then materializes episodes from committed journal steps off-path.
Zero new writes on the hot path — the step commit that already happens IS the capture
guarantee (S6).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Literal

from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from cogworx.substrate.journal import StepRecord

EpisodeKind = Literal["conversation", "tool_exchange", "system_note"]


class Turn(BaseModel):
    """A single conversational turn stamped into stage output data."""

    model_config = ConfigDict(frozen=True)

    role: Literal["user", "assistant", "system", "tool"]
    content: str
    kind: EpisodeKind


def stamp_turns(data: dict[str, Any], turns: Sequence[Turn]) -> None:
    """Stamp a list of turns into stage output data dict (output.data).

    Sets ``data["turns"]`` as a list of dicts (``turn.model_dump()`` for each).
    Idempotent: repeated calls with the same turns overwrite with the same value.

    Raises:
        ValueError: if ``turns`` is empty.
    """
    if not turns:
        raise ValueError("stamp_turns: turns must be non-empty")
    data["turns"] = [turn.model_dump() for turn in turns]


def turns_of(step: StepRecord) -> list[Turn]:
    """Parse turns stamped in a committed journal step.

    Returns an empty list if the step's output data has no ``"turns"`` key — not every step
    is a conversation step.

    Raises:
        ValueError: if ``"turns"`` is present but malformed (not a list of dicts, or any item
            fails ``Turn`` validation).
    """
    output = getattr(step.result, "output", None)
    if output is None:
        return []

    raw = output.data.get("turns")
    if raw is None:
        return []

    if not isinstance(raw, list):
        raise ValueError(
            f"turns_of: step {step.run_id!r}:{step.step_index} has output.data['turns'] of type "
            f"{type(raw).__name__!r}; expected a list of dicts"
        )

    turns: list[Turn] = []
    for i, item in enumerate(raw):
        if not isinstance(item, dict):
            raise ValueError(
                f"turns_of: step {step.run_id!r}:{step.step_index} output.data['turns'][{i}] is "
                f"type {type(item).__name__!r}; expected a dict"
            )
        try:
            turns.append(Turn.model_validate(item))
        except Exception as exc:
            raise ValueError(
                f"turns_of: step {step.run_id!r}:{step.step_index} output.data['turns'][{i}] "
                f"failed Turn validation: {exc}"
            ) from exc

    return turns


__all__ = ["EpisodeKind", "Turn", "stamp_turns", "turns_of"]
