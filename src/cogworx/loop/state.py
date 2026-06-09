"""Spine FSM states (CANON S2, S6, S8).

The loop is a graph of stages by default; each stage and each run carries an explicit status.
``await-human`` and ``degraded`` are first-class transitions, so HITL_WAIT/AWAITING_HUMAN and
DEGRADED are first-class states. ``WAITING`` (parked on a durable timer, advanced only by the
sweeper), ``RETRYING`` (parked on a durable retry timer after a retryable failure, advanced only by
the sweeper), and ``PAUSED`` (parked manually, advanced only by ``unpause``) are the NON-terminal
parked states a plain crash-resume must return as-is (S6) — only the sweeper / ``unpause`` advance
them. ``StageStatus`` stays UN-persisted: it labels retry/timeout EVENTS only; the FSM is derived
from ``{committed step at seq?}`` and ``{attempt count}``, not materialized (S6 — durable iff
resume needs it).
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
    RETRYING = "retrying"
    TIMED_OUT = "timed_out"


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    AWAITING_HUMAN = "awaiting_human"
    DEGRADED = "degraded"
    WAITING = "waiting"
    PAUSED = "paused"
    RETRYING = "retrying"


__all__ = [
    "RunStatus",
    "StageStatus",
]
