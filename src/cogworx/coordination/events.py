"""Coordination by contract (CANON S7).

cog-worx ships the coordination contracts/types so agents are wireable later, but does not build the
orchestration platform now. The contract is two-channel — the shared substrate (events, broadcast)
and direct point-to-point messaging (requests/replies carry no durable state). Every event type is
owned by exactly one subsystem (write-token-per-entity at the event layer).
"""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from cogworx.claims.provenance import Provenance


class Subsystem(StrEnum):
    SPINE = "spine"
    COGNITION = "cognition"
    MEMORY = "memory"
    PERCEPTION = "perception"
    VERIFICATION = "verification"
    OPERATIONS = "operations"
    COORDINATION = "coordination"


class EventType(StrEnum):
    RUN_STARTED = "run_started"
    RUN_COMPLETED = "run_completed"
    RUN_FAILED = "run_failed"
    STAGE_ENTERED = "stage_entered"
    STAGE_COMPLETED = "stage_completed"
    STAGE_DEGRADED = "stage_degraded"
    STAGE_AWAITING_HUMAN = "stage_awaiting_human"
    STAGE_WAITING = "stage_waiting"
    STAGE_RETRYING = "stage_retrying"
    STAGE_TIMED_OUT = "stage_timed_out"
    STEP_COMMITTED = "step_committed"
    RUN_WAITING = "run_waiting"
    RUN_RETRYING = "run_retrying"
    RUN_PAUSED = "run_paused"
    RUN_RESUMED = "run_resumed"
    CLAIM_WRITTEN = "claim_written"
    CAPABILITY_INVOKED = "capability_invoked"
    BUDGET_EXCEEDED = "budget_exceeded"
    MESSAGE_SENT = "message_sent"
    MESSAGE_REPLIED = "message_replied"
    LESION_ENABLED = "lesion_enabled"
    LESION_DISABLED = "lesion_disabled"
    TIMER_SET = "timer_set"
    TIMER_FIRED = "timer_fired"


class Event(BaseModel):
    """A witnessed coordination event on the shared (broadcast) channel."""

    model_config = ConfigDict(frozen=True)

    id: str
    type: EventType
    timestamp: datetime
    subsystem: Subsystem
    session_id: str
    run_id: str | None = None
    correlation_id: str | None = None
    provenance: Provenance | None = None
    schema_version: int = 1
    attributes: dict[str, Any] = Field(default_factory=dict)


EVENT_BOUNDARY_MAP: dict[EventType, Subsystem] = {
    EventType.RUN_STARTED: Subsystem.SPINE,
    EventType.RUN_COMPLETED: Subsystem.SPINE,
    EventType.RUN_FAILED: Subsystem.SPINE,
    EventType.STAGE_ENTERED: Subsystem.SPINE,
    EventType.STAGE_COMPLETED: Subsystem.SPINE,
    EventType.STAGE_DEGRADED: Subsystem.SPINE,
    EventType.STAGE_AWAITING_HUMAN: Subsystem.SPINE,
    EventType.STAGE_WAITING: Subsystem.SPINE,
    EventType.STAGE_RETRYING: Subsystem.SPINE,
    EventType.STAGE_TIMED_OUT: Subsystem.SPINE,
    EventType.STEP_COMMITTED: Subsystem.SPINE,
    EventType.RUN_WAITING: Subsystem.SPINE,
    EventType.RUN_RETRYING: Subsystem.SPINE,
    EventType.RUN_PAUSED: Subsystem.SPINE,
    EventType.RUN_RESUMED: Subsystem.SPINE,
    EventType.TIMER_SET: Subsystem.SPINE,
    EventType.TIMER_FIRED: Subsystem.SPINE,
    EventType.CLAIM_WRITTEN: Subsystem.MEMORY,
    EventType.CAPABILITY_INVOKED: Subsystem.COGNITION,
    EventType.BUDGET_EXCEEDED: Subsystem.OPERATIONS,
    EventType.LESION_ENABLED: Subsystem.OPERATIONS,
    EventType.LESION_DISABLED: Subsystem.OPERATIONS,
    EventType.MESSAGE_SENT: Subsystem.COORDINATION,
    EventType.MESSAGE_REPLIED: Subsystem.COORDINATION,
}


class EventBoundaryError(Exception):
    """Raised when an event is emitted by a subsystem that does not own its type."""


def validate_event_boundary(event: Event) -> None:
    """Enforce that each event type is owned by exactly one subsystem."""

    owner = EVENT_BOUNDARY_MAP[event.type]
    if event.subsystem != owner:
        raise EventBoundaryError(
            f"event {event.type!r} is owned by {owner!r}, not {event.subsystem!r}"
        )


class Request(BaseModel):
    """Direct point-to-point request (rule ①: carries intent/payload, never durable state)."""

    model_config = ConfigDict(frozen=True)

    id: str
    from_agent: str
    to_agent: str
    intent: str
    payload: dict[str, Any] = Field(default_factory=dict)


class Reply(BaseModel):
    """Direct point-to-point reply (rule ①: carries result, never durable state)."""

    model_config = ConfigDict(frozen=True)

    id: str
    request_id: str
    from_agent: str
    ok: bool = True
    payload: dict[str, Any] = Field(default_factory=dict)


class WriteToken(BaseModel):
    """Exactly one write-token per entity (CANON S7)."""

    model_config = ConfigDict(frozen=True)

    entity_id: str
    holder: str
    granted_at: datetime


class WriteTokenError(Exception):
    """Raised on a write-token violation (two writers to one entity)."""


__all__ = [
    "EVENT_BOUNDARY_MAP",
    "Event",
    "EventBoundaryError",
    "EventType",
    "Reply",
    "Request",
    "Subsystem",
    "WriteToken",
    "WriteTokenError",
    "validate_event_boundary",
]
