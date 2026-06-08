"""The durable journal adapter on Postgres/TimescaleDB (CANON S3, S6).

The real :class:`~cogworx.substrate.journal.Journal` seam: a step is "done" iff its committed
``StageResult`` is journaled here before the runner advances, and exactly-once commits mean a replay
reads the stored output and never re-calls the model (S6). Behaviourally equivalent to
``InMemoryJournal`` — same idempotency on the key, same status derivation, same deterministic step
ordering.

Hypertable-vs-exactly-once trade-off: a TimescaleDB hypertable requires the time partitioning column
to appear in every UNIQUE constraint. A UNIQUE on ``(idempotency_key, committed_at)`` would NOT give
exactly-once — a replay computes a fresh ``committed_at`` and the same key would slip in twice. S6
correctness (exactly-once) wins over the hypertable here: ``cogworx_journal_steps`` is kept a PLAIN
table with ``idempotency_key`` UNIQUE so ``ON CONFLICT (idempotency_key) DO NOTHING`` is true
exactly-once. The Phase-1 durability pod can revisit chunking (e.g. a separate append-only metrics
hypertable) without weakening this guarantee.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any

import psycopg
from psycopg.types.json import Jsonb
from pydantic import TypeAdapter

from cogworx.adapters.config import SubstrateSettings
from cogworx.loop.result import StageResult
from cogworx.loop.state import RunStatus
from cogworx.substrate.journal import RunState, StepRecord, Timer

_RESULT_ADAPTER: TypeAdapter[StageResult] = TypeAdapter(StageResult)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cogworx_journal_runs (
    run_id text PRIMARY KEY,
    session_id text NOT NULL
);

CREATE TABLE IF NOT EXISTS cogworx_journal_steps (
    seq bigserial,
    run_id text NOT NULL,
    step_id text NOT NULL,
    stage_name text NOT NULL,
    result jsonb NOT NULL,
    idempotency_key text NOT NULL,
    committed_at timestamptz NOT NULL,
    CONSTRAINT cogworx_journal_steps_idem UNIQUE (idempotency_key),
    CONSTRAINT cogworx_journal_steps_run_step UNIQUE (run_id, step_id)
);

CREATE INDEX IF NOT EXISTS cogworx_journal_steps_order
    ON cogworx_journal_steps (run_id, committed_at, seq);

CREATE TABLE IF NOT EXISTS cogworx_timers (
    run_id text NOT NULL,
    wake_at timestamptz NOT NULL,
    payload jsonb NOT NULL
);

CREATE INDEX IF NOT EXISTS cogworx_timers_wake ON cogworx_timers (wake_at);
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

    async def start_run(self, run_id: str, session_id: str) -> None:
        conn = await self._connection()
        await conn.execute(
            "INSERT INTO cogworx_journal_runs (run_id, session_id) VALUES (%s, %s) "
            "ON CONFLICT (run_id) DO NOTHING",
            (run_id, session_id),
        )

    async def commit_step(self, record: StepRecord) -> None:
        conn = await self._connection()
        result_json = _RESULT_ADAPTER.dump_python(record.result, mode="json")
        await conn.execute(
            "INSERT INTO cogworx_journal_steps "
            "(run_id, step_id, stage_name, result, idempotency_key, committed_at) "
            "VALUES (%s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (idempotency_key) DO NOTHING",
            (
                record.run_id,
                record.step_id,
                record.stage_name,
                Jsonb(result_json),
                record.idempotency_key,
                record.committed_at,
            ),
        )

    async def read_step(self, run_id: str, step_id: str) -> StepRecord | None:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT run_id, step_id, stage_name, result, idempotency_key, committed_at "
            "FROM cogworx_journal_steps WHERE run_id = %s AND step_id = %s",
            (run_id, step_id),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return self._row_to_step(row)

    async def load_run(self, run_id: str) -> RunState | None:
        conn = await self._connection()
        run_cursor = await conn.execute(
            "SELECT session_id FROM cogworx_journal_runs WHERE run_id = %s",
            (run_id,),
        )
        run_row = await run_cursor.fetchone()
        if run_row is None:
            return None
        session_id: str = run_row[0]

        step_cursor = await conn.execute(
            "SELECT run_id, step_id, stage_name, result, idempotency_key, committed_at "
            "FROM cogworx_journal_steps WHERE run_id = %s "
            "ORDER BY committed_at, seq",
            (run_id,),
        )
        rows = await step_cursor.fetchall()
        steps = tuple(self._row_to_step(row) for row in rows)
        status = self._derive_status(steps)
        current_stage = steps[-1].stage_name if steps else None
        return RunState(
            run_id=run_id,
            session_id=session_id,
            status=status,
            current_stage=current_stage,
            steps=steps,
        )

    async def set_timer(self, timer: Timer) -> None:
        conn = await self._connection()
        await conn.execute(
            "INSERT INTO cogworx_timers (run_id, wake_at, payload) VALUES (%s, %s, %s)",
            (timer.run_id, timer.wake_at, Jsonb(timer.payload)),
        )

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT run_id, wake_at, payload FROM cogworx_timers WHERE wake_at <= %s "
            "ORDER BY wake_at",
            (now,),
        )
        rows = await cursor.fetchall()
        return tuple(Timer(run_id=row[0], wake_at=row[1], payload=row[2]) for row in rows)

    async def reset(self) -> None:
        conn = await self._connection()
        await conn.execute("TRUNCATE cogworx_journal_steps, cogworx_journal_runs, cogworx_timers")

    async def aclose(self) -> None:
        if self._conn is not None and not self._conn.closed:
            await self._conn.close()
        self._conn = None

    @staticmethod
    def _row_to_step(row: Sequence[Any]) -> StepRecord:
        return StepRecord(
            run_id=row[0],
            step_id=row[1],
            stage_name=row[2],
            result=_RESULT_ADAPTER.validate_python(row[3]),
            idempotency_key=row[4],
            committed_at=row[5],
        )

    @staticmethod
    def _derive_status(steps: Sequence[StepRecord]) -> RunStatus:
        if not steps:
            return RunStatus.PENDING
        result = steps[-1].result
        match result.kind:
            case "done":
                return RunStatus.COMPLETED
            case "await-human":
                return RunStatus.AWAITING_HUMAN
            case "degraded":
                return RunStatus.DEGRADED if result.to is None else RunStatus.RUNNING
            case "transition":
                return RunStatus.RUNNING


__all__ = ["TimescaleJournal"]
