"""Spine / loop layer (CANON S2): FSM states and stage composition primitives."""

from __future__ import annotations

from cogworx.loop.graph import Loop, StageGraph, StageGraphError
from cogworx.loop.result import AwaitHuman, Degraded, Done, StageResult, Transition
from cogworx.loop.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from cogworx.loop.stage import Stage, StageContext
from cogworx.loop.state import RunStatus, StageStatus

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "AwaitHuman",
    "Degraded",
    "Done",
    "Loop",
    "RetryPolicy",
    "RunStatus",
    "Stage",
    "StageContext",
    "StageGraph",
    "StageGraphError",
    "StageResult",
    "StageStatus",
    "Transition",
]
