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
from datetime import datetime, timedelta
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from cogworx.claims.provenance import Artifact
from cogworx.loop.result import StageResult
from cogworx.loop.state import RunStatus


class ProjectionCursor(BaseModel):
    """The last committed step a projection consumer has SCANNED, in COMMIT order (S1, S6).

    The total order is the lexicographic tuple ``(commit_ordinal, run_id, step_index)``.
    ``commit_ordinal`` is a per-cluster, MONOTONIC-IN-COMMIT-ORDER integer assigned BY THE DATABASE
    inside the writing txn (the Postgres ``commit_xid``), so it imposes a single total order that is
    visibility-fenced: a row's ordinal cannot be consumed until no in-flight txn could still commit
    BELOW it. This replaces the old wall-clock ``committed_at`` keyset, which was non-monotonic
    across writers and forced a lookback band. ``run_id``/``step_index`` are total-order tie-breaks
    under one ordinal (multiple steps committed in one txn share an ordinal — Phase-1 commits one
    step per txn, so this is latent).

    The stored cursor only ever advances to the tuple-``max`` of its value and a new one (never
    regresses): a crash that loses the cursor re-projects idempotently under the first-write-wins
    MERGE on ``trial_id``.
    """

    model_config = ConfigDict(frozen=True)

    commit_ordinal: int
    """The DB-assigned commit-order integer of the last scanned step (Postgres ``commit_xid``)."""
    run_id: str
    """The run of the last scanned step — first tuple tie-break under one ``commit_ordinal``."""
    step_index: int
    """The positional step index of the last scanned step — second tuple tie-break."""


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


class ProjectedStep(BaseModel):
    """A committed step paired with its DB-assigned ``commit_ordinal`` for a projection read (S1).

    The projection read seam returns these so the consumer advances its :class:`ProjectionCursor` to
    the COMMIT order of the last scanned row WITHOUT widening :class:`StepRecord` (which is the
    replay/timer shape and must stay frozen): the ordinal is a read-side projection concern, not
    part of the durable step. ``commit_ordinal`` is non-optional — a projection read always carries
    it.
    """

    model_config = ConfigDict(frozen=True)

    record: StepRecord
    commit_ordinal: int


class RunState(BaseModel):
    """Contract changelog: 2026-06-12 (Pod 3.2 B1): added ``tainted: bool = False`` — additive
    (defaulted field on a frozen value type). NOTE: the companion ``Journal.set_run_tainted`` method
    is a NEW PROTOCOL MEMBER — C3-**breaking**, handled via the §6 amendment path: **Jim-approved
    this session** (durable lethal-trifecta taint, S10+S6).
    Recorded breaking-but-approved for audit.
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    session_id: str
    status: RunStatus
    pathway_id: str
    pathway_version: int
    pathway_fingerprint: str
    current_stage: str | None = None
    steps: tuple[StepRecord, ...] = ()
    tainted: bool = False


class Timer(BaseModel):
    """A durable timer that parks a run until ``wake_at`` (S6).

    ``timer_id`` is the identity (derived as ``f"{run_id}:{step_index}"`` by the engine) so a given
    ``Wait`` has exactly one timer: ``set_timer`` is idempotent on it and advancing past the
    ``Wait`` can cancel it by derived id. ``claimed_at`` is the LEASE marker — ``claim_due_timers``
    stamps it instead of deleting, so a crash between claim and advance leaves the timer reclaimable
    once the lease goes stale (the at-least-once-fire / exactly-once-advance property).
    """

    model_config = ConfigDict(frozen=True)

    run_id: str
    timer_id: str
    wake_at: datetime
    claimed_at: datetime | None = None
    payload: dict[str, Any] = Field(default_factory=dict)


@runtime_checkable
class Journal(Protocol):
    """The durable, exactly-once journal seam over TimescaleDB."""

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        """Record the run as RUNNING with its pathway pointer + structural fingerprint.

        The ``pathway_fingerprint`` is a hash of the graph's STRUCTURE at start time (see
        ``cogworx.loop.pathway.pathway_fingerprint``); a cold resume compares the rehydrated graph's
        fingerprint against this stored value to catch a structural in-place edit under the same
        ``(pathway_id, version)``. Idempotent re-start is a no-op.
        """
        ...

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        """Persist the run's status — the run record is the AUTHORITY (not derived from steps)."""
        ...

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        """Atomically flip status to ``new`` ONLY IF it currently equals ``expect``; return whether
        it changed — the run-level MUTUAL-EXCLUSION primitive that serializes drivers (S6).

        Backed by a single conditional ``UPDATE … WHERE status = expect RETURNING``, so the DB row
        lock admits exactly ONE winner even across concurrent sweepers racing to drive the same run.
        This is what makes a RUN (not just a commit) exactly-once: a driver must win the CAS before
        it may execute uncommitted stages, so two racers cannot both call the model. A loser gets
        ``False`` and must NOT drive (the run is already being driven, parked, or terminal).

        The CAS serializes drivers (exactly-once EXECUTION even under concurrent sweepers). What is
        DEFERRED is auto-recovery of a driver that crashes mid-drive: the CAS cannot distinguish a
        LIVE RUNNING run from a DEAD one — that needs a run-lease (owner + expiry + heartbeat) and a
        stranded-RUNNING reaper (a future ops pod). Until then a crashed RUNNING run is recovered by
        an explicit ``resume(run_id)``.
        """
        ...

    async def commit_step(self, record: StepRecord) -> None:
        """Idempotent on ``(run_id, step_index)``; commit BEFORE the runner advances."""
        ...

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None: ...

    async def committed_steps_after(
        self, cursor: ProjectionCursor | None, *, limit: int
    ) -> Sequence[ProjectedStep]:
        """Read the next committed-step batch for a COMMIT-ORDER projection consumer (S1).

        The PROJECTION read seam: an off-write-path consumer (e.g. the Pod 2.1 trial projector)
        polls this to turn newly committed steps into its own derived records, advancing its OWN
        :class:`ProjectionCursor` — the journal table stays pure (no per-consumer column, S3).

        TWO-CLAUSE, VISIBILITY-FENCED CONTRACT (the ``commit_xid`` design):

        * FENCE — only rows whose ``commit_ordinal`` is strictly below the snapshot's ``xmin``
          (``pg_snapshot_xmin(pg_current_snapshot())``) are eligible. This FORBIDS consuming any row
          an in-flight txn could still slot BELOW (a row whose ordinal was assigned but not yet
          committed). It is what closes the sequence-visibility race: a consumer never advances its
          cursor past an ordinal that a concurrent writer can still commit beneath.
        * STRICT KEYSET — ``(commit_ordinal, run_id, step_index) > cursor`` (omitted on cold start,
          ``cursor is None``), so the batch resumes STRICTLY past the last scanned row.

        Ordered by ``(commit_ordinal, run_id, step_index)`` — the SAME leading column as the fence
        (a hybrid seq+xid ordering is provably wrong). Capped at ``limit`` (the per-tick batch). No
        lookback, no ceiling: ``commit_ordinal`` is monotonic-in-commit-order, so a single fenced
        forward read is total — there is no late-row band to re-scan.
        """
        ...

    async def load_run(self, run_id: str) -> RunState | None: ...

    async def set_run_tainted(self, run_id: str) -> None:
        """Idempotent monotonic flip of the run's ``tainted`` flag to ``True`` (S6 ordering).

        A single unconditional ``UPDATE … SET tainted = true WHERE run_id = …`` — writing
        ``True`` when already ``True`` is a no-op at the row level (no CAS needed; monotonic
        from False to True only).  If the run is unknown this silently does nothing.

        MUST be awaited BEFORE ``cap.invoke`` in ``dispatch_one`` so that a crash between the
        journal write and the invocation leaves the run correctly tainted on resume (fail-closed:
        the journal write failing propagates without invoking the tool).
        """
        ...

    async def set_timer(self, timer: Timer) -> None:
        """Arm a durable timer; idempotent on ``timer_id`` (a re-armed ``Wait`` is a no-op)."""
        ...

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        """Read all timers due at ``now`` REGARDLESS of lease (observability/tests, not sweep)."""
        ...

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        """Atomically LEASE the due timers — the sweep primitive (S6). It does NOT delete.

        Returns the timers due at ``now`` that are either unclaimed OR whose lease is STALE
        (``claimed_at <= now - lease_ttl``, so a lease exactly ``lease_ttl`` old is expired),
        stamping ``claimed_at = now`` on each as it returns it.
        Two concurrent claims over the same due set therefore deliver each timer to exactly one
        caller (the lease is the single-delivery guard). The timer row SURVIVES the claim: it is
        deleted only when the run advances past its ``Wait`` (``cancel_timer`` on the wait-replay
        branch) or on terminal ``cancel_timers_for_run``. So a process that dies after the claim but
        before the advance does not strand the run — the next sweep past ``lease_ttl`` reclaims and
        re-fires it, and the advance is idempotent (at-least-once fire / exactly-once advance).
        """
        ...

    async def cancel_timer(self, timer_id: str) -> None:
        """Delete a timer by id; idempotent (the engine cancels on the wait-replay advance)."""
        ...

    async def cancel_timers_for_run(self, run_id: str) -> None:
        """Delete every timer for a run; called on terminal cleanup so no straggler can re-fire."""
        ...

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        """Cheap AUTHORITATIVE status poll (no step fan-out) for the cooperative pause check."""
        ...

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        """Atomically bump the FAILURE counter for ``(run_id, step_index)``; return the NEW count.

        Keyed per-``(run_id, step_index)`` and called ONLY on a retryable failure (or timeout) — a
        successful (re-)attempt never increments. The first failure returns 1. Backed by a single
        ``INSERT … VALUES (…, 1, …) ON CONFLICT (run_id, step_index) DO UPDATE SET attempt =
        attempt + 1 … RETURNING attempt`` so N concurrent callers row-lock the conflicting row and
        each gets a DISTINCT value (no lost update) — the durable, exactly-once attempt count that
        survives a crash (never resets, never over-counts; a failed attempt commits no step, so the
        count alone distinguishes a retry from a wait at a frozen ``seq``).
        """
        ...

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        """Read the durable FAILURE count for ``(run_id, step_index)`` (0 if never incremented).

        The authority a re-driven stage reads to know how many prior attempts failed — so a cold
        resume on a fresh stage instance fails/succeeds correctly off the journaled count, never an
        in-process counter (S6).
        """
        ...

    async def record_human_input(self, run_id: str, step_index: int, answer: Artifact) -> None:
        """Idempotent FIRST-ANSWER-WINS persist for the S5 provenance-bearing HITL input.

        The first caller commits the answer; all subsequent callers for the same
        ``(run_id, step_index)`` are silent no-ops (``INSERT … ON CONFLICT (run_id, step_index) DO
        NOTHING``). This is a pure write — no model-heavy work on the hot path (S1). The answer is
        persisted BEFORE the re-drive CAS (H3 hard ordering constraint): if the process dies between
        record and CAS, a re-issued ``provide_human_input`` finds the answer already in the journal
        and completes without overwriting it.
        """
        ...

    async def read_human_input(self, run_id: str, step_index: int) -> Artifact | None:
        """Return the HITL answer committed at ``(run_id, step_index)``, or ``None`` if absent.

        Used by downstream stages via ``ctx.read_human_input`` (the PULL model): a stage reads the
        journaled answer structurally — the engine never pushes ``ctx.human_input`` — so cold resume
        works correctly without the engine re-injecting the answer.
        """
        ...


__all__ = [
    "Artifact",
    "Journal",
    "ProjectedStep",
    "ProjectionCursor",
    "RunState",
    "StepRecord",
    "Timer",
]
