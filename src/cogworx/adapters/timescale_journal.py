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
    session_id text NOT NULL,
    status text NOT NULL,
    pathway_id text NOT NULL,
    pathway_version integer NOT NULL
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

    async def start_run(
        self, run_id: str, session_id: str, *, pathway_id: str, pathway_version: int
    ) -> None:
        conn = await self._connection()
        await conn.execute(
            "INSERT INTO cogworx_journal_runs "
            "(run_id, session_id, status, pathway_id, pathway_version) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (run_id) DO NOTHING",
            (run_id, session_id, RunStatus.RUNNING.value, pathway_id, pathway_version),
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        conn = await self._connection()
        await conn.execute(
            "UPDATE cogworx_journal_runs SET status = %s WHERE run_id = %s",
            (status.value, run_id),
        )

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
            "SELECT session_id, status, pathway_id, pathway_version "
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
        # Test-only ephemeral isolation: DROP + recreate (not TRUNCATE) so the per-case schema is
        # always current — robust to schema evolution across phases. Never called in production.
        conn = await self._connection()
        await conn.execute(
            "DROP TABLE IF EXISTS "
            "cogworx_journal_steps, cogworx_journal_runs, cogworx_timers CASCADE"
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


__all__ = ["TimescaleJournal"]
