"""Spine / loop layer (CANON S2): FSM states and stage composition primitives."""

from __future__ import annotations

from cogworx.loop.stage import (
    AwaitHuman,
    Degraded,
    Done,
    Stage,
    StageContext,
    StageResult,
    Transition,
)
from cogworx.loop.state import RunStatus, StageStatus

__all__ = [
    "AwaitHuman",
    "Degraded",
    "Done",
    "RunStatus",
    "Stage",
    "StageContext",
    "StageResult",
    "StageStatus",
    "Transition",
]
