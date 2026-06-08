"""S7 contract tests: event-boundary ownership (write-token-per-entity at the event layer)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.coordination.events import (
    EVENT_BOUNDARY_MAP,
    Event,
    EventBoundaryError,
    EventType,
    Subsystem,
    validate_event_boundary,
)


def _event(event_type: EventType, subsystem: Subsystem) -> Event:
    return Event(
        id="evt-1",
        type=event_type,
        timestamp=datetime(2026, 6, 8, tzinfo=UTC),
        subsystem=subsystem,
        session_id="sess-1",
    )


@pytest.mark.parametrize("event_type", list(EventType))
def test_every_event_type_is_owned(event_type: EventType) -> None:
    assert event_type in EVENT_BOUNDARY_MAP


def test_correctly_owned_event_passes() -> None:
    validate_event_boundary(_event(EventType.CLAIM_WRITTEN, Subsystem.MEMORY))


def test_mismatched_subsystem_raises() -> None:
    with pytest.raises(EventBoundaryError):
        validate_event_boundary(_event(EventType.CLAIM_WRITTEN, Subsystem.SPINE))
