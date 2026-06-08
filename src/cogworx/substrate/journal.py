"""The durable journal seam — TimescaleDB (CANON S3, S6).

A step is "done" iff its result is committed here before the runner advances; on resume the runner
reads stored outputs and never re-calls the model. This is the engine-shaped seam for the
time-ordered run journal — not a generic store that flattens engines.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from cogworx.loop.result import StageResult
from cogworx.loop.state import RunStatus


class StepRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    step_id: str
    stage_name: str
    # The committed control decision (S6): a step is "done" iff its StageResult is journaled before
    # the runner advances. On resume the runner reads this and selects the next stage WITHOUT
    # re-running the stage or re-calling the model.
    result: StageResult
    idempotency_key: str
    committed_at: datetime


class RunState(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    status: RunStatus
    current_stage: str | None = None
    steps: tuple[StepRecord, ...] = ()


class Timer(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    wake_at: datetime
    payload: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Journal(Protocol):
    """The durable, exactly-once journal seam over TimescaleDB."""

    async def start_run(self, run_id: str, session_id: str) -> None: ...

    async def commit_step(self, record: StepRecord) -> None:
        """Idempotent on ``(run_id, idempotency_key)``; commit BEFORE the runner advances."""
        ...

    async def read_step(self, run_id: str, step_id: str) -> StepRecord | None: ...

    async def load_run(self, run_id: str) -> RunState | None: ...

    async def set_timer(self, timer: Timer) -> None: ...

    async def due_timers(self, now: datetime) -> Sequence[Timer]: ...


__all__ = [
    "Journal",
    "RunState",
    "StepRecord",
    "Timer",
]
