"""Spine FSM states (CANON S2, S6, S8).

The loop is a graph of stages by default; each stage and each run carries an explicit status.
``await-human`` and ``degraded`` are first-class transitions, so HITL_WAIT/AWAITING_HUMAN and
DEGRADED are first-class states. ``WAITING`` (parked on a durable timer, advanced only by the
sweeper) and ``PAUSED`` (parked manually, advanced only by ``unpause``) are the two NON-terminal
parked states a plain crash-resume must return as-is (S6) — only the sweeper / ``unpause`` advance
them.
"""

from __future__ import annotations

from enum import StrEnum


class StageStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    HITL_WAIT = "hitl_wait"
    DEGRADED = "degraded"
    WAITING = "waiting"
    PAUSED = "paused"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"
    DEGRADED = "degraded"
    WAITING = "waiting"
    PAUSED = "paused"


__all__ = [
    "RunStatus",
    "StageStatus",
]
