"""The durable journal adapter on Postgres/TimescaleDB (CANON S3, S6).

The real :class:`~cogworx.substrate.journal.Journal` seam: a step is "done" iff its committed
``StageResult`` is journaled here before the runner advances, and exactly-once commits mean a replay
reads the stored output and never re-calls the model (S6). Behaviourally equivalent to
``InMemoryJournal`` — same exactly-once positional keying, same PERSISTED run status, same
deterministic step ordering.

Steps are keyed POSITIONALLY by ``(run_id, step_index)`` (a 0-based monotonic position in the run's
drive), so a cyclic pathway that revisits a stage commits a DISTINCT step per visit. The run's
``status`` is PERSISTED on the run record (the authority — not derived from the last step), and the
run record carries the ``pathway_id`` + ``pathway_version`` pointer for cold resume.

Durable timers ride the same cluster. ``cogworx_timers`` is keyed by ``timer_id`` (idempotent
``set_timer`` via ``ON CONFLICT DO NOTHING``) and carries a ``claimed_at`` LEASE marker. The sweep
primitive ``claim_due_timers`` is a bare atomic ``UPDATE … SET claimed_at = now WHERE wake_at <= now
AND (claimed_at IS NULL OR claimed_at <= now - lease_ttl) RETURNING …`` — no ``FOR UPDATE SKIP
LOCKED`` is needed: under READ COMMITTED the conditional ``UPDATE`` takes a row lock per candidate
row, and a SECOND concurrent claim blocks on that lock, then RE-EVALUATES its ``WHERE`` against the
now-committed ``claimed_at`` once the lock releases — so it sees the fresh (non-stale) lease and
returns 0 rows for that timer. That row-lock-plus-re-check IS single-delivery: two concurrent
sweepers cannot both claim the same timer. The row SURVIVES the claim (it is deleted only by
``cancel_timer`` once the run advances past its ``Wait``, or by ``cancel_timers_for_run`` on a
terminal run), so a crash between claim and advance leaves a stale lease the next sweep reclaims —
the at-least-once-fire / exactly-once-advance contract. Behaviourally identical to
``InMemoryJournal`` (S3 — same lease/cancel/idempotency semantics).

``compare_and_set_run_status`` is the run-level mutual-exclusion primitive: a conditional ``UPDATE …
SET status = new WHERE run_id = id AND status = expect RETURNING`` whose row lock admits exactly one
winning driver even across concurrent sweepers. It is what makes a RUN exactly-once (not merely its
commits), so two racers cannot both execute an uncommitted, model-bearing stage.

``cogworx_step_attempts`` is the durable retry FAILURE counter keyed ``(run_id, step_index)``:
``increment_attempt`` is a single atomic ``INSERT … ON CONFLICT (run_id, step_index) DO UPDATE
attempt = attempt + 1 RETURNING attempt``, so N concurrent callers row-lock the conflicting row and
bump serially, each getting a DISTINCT value (no lost update). The count is incremented ONLY on a
retryable failure/timeout — a failed attempt commits no step, so the count alone distinguishes a
retry (no committed step at a frozen ``seq``) from a wait (a committed ``Wait`` at ``seq``). It
survives a crash (never resets, never over-counts), so a cold resume re-attempts with the journaled
count, not a reset.

Hypertable-vs-exactly-once trade-off: a TimescaleDB hypertable requires the time partitioning column
to appear in every UNIQUE constraint. A UNIQUE on ``(run_id, step_index, committed_at)`` would NOT
give exactly-once — a replay computes a fresh ``committed_at`` and the same position would slip in
twice. S6 correctness (exactly-once) wins over the hypertable here: ``cogworx_journal_steps`` is a
PLAIN table with ``(run_id, step_index)`` UNIQUE so ``ON CONFLICT (run_id, step_index) DO NOTHING``
is true exactly-once. The Phase-1 durability pod can revisit chunking (a separate append-only
metrics hypertable) without weakening this guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from cogworx.adapters.config import SubstrateSettings
from cogworx.claims.provenance import Artifact
from cogworx.loop.result import StageResult
from cogworx.loop.state import RunStatus
from cogworx.substrate.journal import RunState, StepRecord, Timer

_ARTIFACT_ADAPTER: TypeAdapter[Artifact] = TypeAdapter(Artifact)

_RESULT_ADAPTER: TypeAdapter[StageResult] = TypeAdapter(StageResult)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cogworx_journal_runs (
    run_id text PRIMARY KEY,
    session_id text NOT NULL,
    status text NOT NULL,
    pathway_id text NOT NULL,
    pathway_version integer NOT NULL,
    pathway_fingerprint text NOT NULL
);

CREATE TABLE IF NOT EXISTS cogworx_journal_steps (
    run_id text NOT NULL,
    step_index bigint NOT NULL,
    stage_name text NOT NULL,
    result jsonb NOT NULL,
    committed_at timestamptz NOT NULL,
    CONSTRAINT cogworx_journal_steps_run_step UNIQUE (run_id, step_index)
);

CREATE INDEX IF NOT EXISTS cogworx_journal_steps_order
    ON cogworx_journal_steps (run_id, step_index);

CREATE TABLE IF NOT EXISTS cogworx_timers (
    timer_id text PRIMARY KEY,
    run_id text NOT NULL,
    wake_at timestamptz NOT NULL,
    payload jsonb NOT NULL,
    claimed_at timestamptz
);

CREATE INDEX IF NOT EXISTS cogworx_timers_wake ON cogworx_timers (wake_at);
CREATE INDEX IF NOT EXISTS cogworx_timers_run ON cogworx_timers (run_id);

CREATE TABLE IF NOT EXISTS cogworx_step_attempts (
    run_id text NOT NULL,
    step_index bigint NOT NULL,
    attempt integer NOT NULL,
    last_failure_class text,
    last_failed_at timestamptz,
    CONSTRAINT cogworx_step_attempts_pk PRIMARY KEY (run_id, step_index)
);

CREATE TABLE IF NOT EXISTS cogworx_human_inputs (
    run_id text NOT NULL,
    step_index bigint NOT NULL,
    answer jsonb NOT NULL,
    recorded_at timestamptz NOT NULL,
    CONSTRAINT cogworx_human_inputs_pk PRIMARY KEY (run_id, step_index)
);
"""


class TimescaleJournal:
    """A durable, exactly-once :class:`Journal` backed by the Postgres/TimescaleDB cluster."""

    def __init__(
        self, *, settings: SubstrateSettings | None = None, dsn: str | None = None
    ) -> None:
        self._dsn = dsn if dsn is not None else (settings or SubstrateSettings()).pg_dsn
        self._conn: psycopg.AsyncConnection[Any] | None = None

    async def _connection(self) -> psycopg.AsyncConnection[Any]:
        if self._conn is None or self._conn.closed:
            self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

    async def ensure_schema(self) -> None:
        conn = await self._connection()
        await conn.execute(_SCHEMA)

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        conn = await self._connection()
        await conn.execute(
            "INSERT INTO cogworx_journal_runs "
            "(run_id, session_id, status, pathway_id, pathway_version, pathway_fingerprint) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id) DO NOTHING",
            (
                run_id,
                session_id,
                RunStatus.RUNNING.value,
                pathway_id,
                pathway_version,
                pathway_fingerprint,
            ),
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        conn = await self._connection()
        await conn.execute(
            "UPDATE cogworx_journal_runs SET status = %s WHERE run_id = %s",
            (status.value, run_id),
        )

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        conn = await self._connection()
        cursor = await conn.execute(
            "UPDATE cogworx_journal_runs SET status = %s "
            "WHERE run_id = %s AND status = %s RETURNING run_id",
            (new.value, run_id, expect.value),
        )
        return await cursor.fetchone() is not None

    async def commit_step(self, record: StepRecord) -> None:
        conn = await self._connection()
        result_json = _RESULT_ADAPTER.dump_python(record.result, mode="json")
        await conn.execute(
            "INSERT INTO cogworx_journal_steps "
            "(run_id, step_index, stage_name, result, committed_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id, step_index) DO NOTHING",
            (
                record.run_id,
                record.step_index,
                record.stage_name,
                Jsonb(result_json),
                record.committed_at,
            ),
        )

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT run_id, step_index, stage_name, result, committed_at "
            "FROM cogworx_journal_steps WHERE run_id = %s AND step_index = %s",
            (run_id, step_index),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_step(row)

    async def load_run(self, run_id: str) -> RunState | None:
        conn = await self._connection()
        run_cursor = await conn.execute(
            "SELECT session_id, status, pathway_id, pathway_version, pathway_fingerprint "
            "FROM cogworx_journal_runs WHERE run_id = %s",
            (run_id,),
        )
        run_row = await run_cursor.fetchone()
        if run_row is None:
            return None
        session_id: str = run_row[0]
        status = RunStatus(run_row[1])
        pathway_id: str = run_row[2]
        pathway_version: int = run_row[3]
        pathway_fingerprint: str = run_row[4]

        step_cursor = await conn.execute(
            "SELECT run_id, step_index, stage_name, result, committed_at "
            "FROM cogworx_journal_steps WHERE run_id = %s "
            "ORDER BY step_index",
            (run_id,),
        )
        rows = await step_cursor.fetchall()
        steps = tuple(self._row_to_step(row) for row in rows)
        current_stage = steps[-1].stage_name if steps else None
        return RunState(
            run_id=run_id,
            session_id=session_id,
            status=status,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint,
            current_stage=current_stage,
            steps=steps,
        )

    async def set_timer(self, timer: Timer) -> None:
        conn = await self._connection()
        await conn.execute(
            "INSERT INTO cogworx_timers (timer_id, run_id, wake_at, claimed_at, payload) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (timer_id) DO NOTHING",
            (timer.timer_id, timer.run_id, timer.wake_at, timer.claimed_at, Jsonb(timer.payload)),
        )

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT run_id, timer_id, wake_at, claimed_at, payload FROM cogworx_timers "
            "WHERE wake_at <= %s ORDER BY wake_at",
            (now,),
        )
        rows = await cursor.fetchall()
        return tuple(self._row_to_timer(row) for row in rows)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        # Atomic per-row LEASE (not delete): the UPDATE … RETURNING admits exactly one claimant per
        # timer even under concurrent sweepers — a second claim at the same instant sees the fresh
        # (non-stale) lease and skips the row. The row SURVIVES until the advance past the Wait is
        # durably committed (cancel_timer), so a crash between claim and fire is recoverable.
        conn = await self._connection()
        stale_before = now - lease_ttl
        cursor = await conn.execute(
            "UPDATE cogworx_timers SET claimed_at = %s "
            "WHERE wake_at <= %s AND (claimed_at IS NULL OR claimed_at <= %s) "
            "RETURNING run_id, timer_id, wake_at, claimed_at, payload",
            (now, now, stale_before),
        )
        rows = await cursor.fetchall()
        return tuple(self._row_to_timer(row) for row in rows)

    async def cancel_timer(self, timer_id: str) -> None:
        conn = await self._connection()
        await conn.execute("DELETE FROM cogworx_timers WHERE timer_id = %s", (timer_id,))

    async def cancel_timers_for_run(self, run_id: str) -> None:
        conn = await self._connection()
        await conn.execute("DELETE FROM cogworx_timers WHERE run_id = %s", (run_id,))

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT status FROM cogworx_journal_runs WHERE run_id = %s", (run_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return RunStatus(row[0])

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        # The atomic FAILURE counter: a single INSERT … ON CONFLICT DO UPDATE attempt = attempt + 1
        # RETURNING attempt. Under READ COMMITTED the conflicting row is locked per caller, so N
        # concurrent increments serialize and each receives a DISTINCT value (no lost update). The
        # first failure inserts attempt = 1. last_failure_class/last_failed_at are observability —
        # the control signal is the run status + the journal, not these columns.
        conn = await self._connection()
        cursor = await conn.execute(
            "INSERT INTO cogworx_step_attempts "
            "(run_id, step_index, attempt, last_failure_class, last_failed_at) "
            "VALUES (%s, %s, 1, %s, %s) "
            "ON CONFLICT (run_id, step_index) DO UPDATE SET "
            "attempt = cogworx_step_attempts.attempt + 1, "
            "last_failure_class = EXCLUDED.last_failure_class, "
            "last_failed_at = EXCLUDED.last_failed_at "
            "RETURNING attempt",
            (run_id, step_index, None, datetime.now(UTC)),
        )
        row = await cursor.fetchone()
        if row is None:
            raise RuntimeError(
                f"increment_attempt did not RETURN a count for ({run_id!r}, {step_index})"
            )
        attempt: int = row[0]
        return attempt

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT attempt FROM cogworx_step_attempts WHERE run_id = %s AND step_index = %s",
            (run_id, step_index),
        )
        row = await cursor.fetchone()
        if row is None:
            return 0
        count: int = row[0]
        return count

    async def record_human_input(self, run_id: str, step_index: int, answer: Artifact) -> None:
        conn = await self._connection()
        answer_json = _ARTIFACT_ADAPTER.dump_python(answer, mode="json")
        await conn.execute(
            "INSERT INTO cogworx_human_inputs (run_id, step_index, answer, recorded_at) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (run_id, step_index) DO NOTHING",
            (run_id, step_index, Jsonb(answer_json), datetime.now(UTC)),
        )

    async def read_human_input(self, run_id: str, step_index: int) -> Artifact | None:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT answer FROM cogworx_human_inputs WHERE run_id = %s AND step_index = %s",
            (run_id, step_index),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _ARTIFACT_ADAPTER.validate_python(row[0])

    async def reset(self) -> None:
        # Test-only ephemeral isolation: DROP + recreate (not TRUNCATE) so the per-case schema is
        # always current — robust to schema evolution across phases. Never called in production.
        conn = await self._connection()
        await conn.execute(
            "DROP TABLE IF EXISTS "
            "cogworx_journal_steps, cogworx_journal_runs, cogworx_timers, "
            "cogworx_step_attempts, cogworx_human_inputs CASCADE"
        )
        await conn.execute(_SCHEMA)

    async def aclose(self) -> None:
        if self._conn is not None and not self._conn.closed:
            await self._conn.close()
        self._conn = None

    @staticmethod
    def _row_to_step(row: Sequence[Any]) -> StepRecord:
        return StepRecord(
            run_id=row[0],
            step_index=row[1],
            stage_name=row[2],
            result=_RESULT_ADAPTER.validate_python(row[3]),
            committed_at=row[4],
        )

    @staticmethod
    def _row_to_timer(row: Sequence[Any]) -> Timer:
        # Row shape is fixed across due_timers/claim_due_timers:
        # (run_id, timer_id, wake_at, claimed_at, payload).
        return Timer(
            run_id=row[0],
            timer_id=row[1],
            wake_at=row[2],
            claimed_at=row[3],
            payload=row[4],
        )


__all__ = ["TimescaleJournal"]
