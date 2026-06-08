"""Spine FSM states (CANON S2, S6, S8).

The loop is a graph of stages by default; each stage and each run carries an explicit status.
``await-human`` and ``degraded`` are first-class transitions, so HITL_WAIT/AWAITING_HUMAN and
DEGRADED are first-class states.
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


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"
    DEGRADED = "degraded"


__all__ = [
    "RunStatus",
    "StageStatus",
]
