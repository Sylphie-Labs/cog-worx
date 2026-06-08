"""The durable journal seam — TimescaleDB (CANON S3, S6).

A step is "done" iff its result is committed here before the runner advances; on resume the runner
reads stored outputs and never re-calls the model. This is the engine-shaped seam for the
time-ordered run journal — not a generic store that flattens engines.

Steps are keyed POSITIONALLY by ``(run_id, step_index)`` — a 0-based monotonic position in the run's
drive. Positional keying makes a CYCLIC pathway durable: revisiting the same stage commits a
DISTINCT step at the next index, so exactly-once and replay work per-visit (a stage name is no key).

The run's ``status`` is PERSISTED on the run record — it is the AUTHORITY, NOT derived from the last
step. A step ceiling that FAILED a run, or a future WAITING/PAUSED, is not derivable from a
``StageResult``; the runner sets the status explicitly via ``set_run_status``. The run record also
carries the ``pathway_id`` + ``pathway_version`` pointer so a cold resume can rehydrate the graph
from the pathway registry (S2, S6) instead of relying on in-process state.
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
    step_index: int
    stage_name: str
    # The committed control decision (S6): a step is "done" iff its StageResult is journaled before
    # the runner advances. On resume the runner reads this and selects the next stage WITHOUT
    # re-running the stage or re-calling the model. Exactly-once is on ``(run_id, step_index)``.
    result: StageResult
    committed_at: datetime


class RunState(BaseModel):
    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    status: RunStatus
    pathway_id: str
    pathway_version: int
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

    async def start_run(
        self, run_id: str, session_id: str, *, pathway_id: str, pathway_version: int
    ) -> None:
        """Record the run as RUNNING with its pathway pointer. Idempotent re-start is a no-op."""
        ...

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        """Persist the run's status — the run record is the AUTHORITY (not derived from steps)."""
        ...

    async def commit_step(self, record: StepRecord) -> None:
        """Idempotent on ``(run_id, step_index)``; commit BEFORE the runner advances."""
        ...

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None: ...

    async def load_run(self, run_id: str) -> RunState | None: ...

    async def set_timer(self, timer: Timer) -> None: ...

    async def due_timers(self, now: datetime) -> Sequence[Timer]: ...


__all__ = [
    "Journal",
    "RunState",
    "StepRecord",
    "Timer",
]
