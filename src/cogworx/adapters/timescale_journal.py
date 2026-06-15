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

Projection ordering — the ``commit_xid`` visibility fence (POSTGRES 13+). Each step row carries a
``commit_xid bigint`` that self-populates inside the writing txn from ``pg_current_xact_id()`` (the
xid8 functions; PG13+ is the floor). It is a per-cluster integer that is MONOTONIC IN COMMIT ORDER,
so the projection read seam (:meth:`committed_steps_after`) imposes a single total order on
committed steps that the multi-writer, injectable wall-clock ``committed_at`` cannot. The read is
VISIBILITY-FENCED: it consumes only rows whose ``commit_xid`` is strictly below
``pg_snapshot_xmin(pg_current_snapshot())`` — i.e. below any xid an in-flight txn could still commit
beneath — so a consumer never advances past an ordinal a concurrent writer can still slot under (the
sequence-visibility race). The fenced column and the ORDER BY leading column are BOTH ``commit_xid``
(a hybrid seq+xid ordering is provably wrong). The xid8 values are handled as BIGINT end-to-end via
the ``::text::bigint`` idiom (never raw xid8). ``commit_step`` writes ZERO columns for this — the
DEFAULT self-populates inside the commit, and an a5-style invariant FORBIDS any statement from
naming ``commit_xid`` in an INSERT/UPDATE (a static grep test + a runtime probe pin it).
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
from cogworx.substrate.journal import (
    ProjectedStep,
    ProjectionCursor,
    RunState,
    StepRecord,
    Timer,
)

_ARTIFACT_ADAPTER: TypeAdapter[Artifact] = TypeAdapter(Artifact)

_RESULT_ADAPTER: TypeAdapter[StageResult] = TypeAdapter(StageResult)


def _to_utc(dt: datetime) -> datetime:
    """Naive → attach UTC; tz-aware → convert to UTC (the substrate datetime contract)."""
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


_SCHEMA = """
CREATE TABLE IF NOT EXISTS cogworx_journal_runs (
    run_id text PRIMARY KEY,
    session_id text NOT NULL,
    status text NOT NULL,
    pathway_id text NOT NULL,
    pathway_version integer NOT NULL,
    pathway_fingerprint text NOT NULL,
    tainted boolean NOT NULL DEFAULT false
);

CREATE TABLE IF NOT EXISTS cogworx_journal_steps (
    run_id text NOT NULL,
    step_index bigint NOT NULL,
    stage_name text NOT NULL,
    result jsonb NOT NULL,
    committed_at timestamptz NOT NULL,
    commit_xid bigint NOT NULL DEFAULT (pg_current_xact_id()::text::bigint),
    CONSTRAINT cogworx_journal_steps_run_step UNIQUE (run_id, step_index)
);

CREATE INDEX IF NOT EXISTS cogworx_journal_steps_order
    ON cogworx_journal_steps (run_id, step_index);

-- The commit_xid projection index is created in _MIGRATE_COMMIT_XID, NOT here: on a pre-2.1
-- upgrade the steps table already exists WITHOUT commit_xid, so CREATE TABLE IF NOT EXISTS is a
-- no-op and an index on commit_xid here would reference a column the migration's ALTER has not yet
-- added (the migration runs after _SCHEMA). Building it in the migration block guarantees the
-- column exists first; on a fresh install the column is in the CREATE TABLE above and the
-- migration creates the index idempotently.

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

# Idempotent migration of a pre-2.1 steps table to the commit_xid projection ordering. The ALTER
# backfills every existing row to the migration txn's own xid (via the DEFAULT), which sorts BELOW
# all future inserts, so the migrated rows keep their relative position under the new total order.
# The old committed_at-keyset band reader is gone, so its composite index is dropped — nothing else
# used it (the run-grain index `cogworx_journal_steps_order` and the per-step UNIQUE remain).
_MIGRATE_COMMIT_XID = """
ALTER TABLE cogworx_journal_steps
    ADD COLUMN IF NOT EXISTS commit_xid bigint NOT NULL
    DEFAULT (pg_current_xact_id()::text::bigint);

CREATE INDEX IF NOT EXISTS cogworx_journal_steps_commit_xid
    ON cogworx_journal_steps (commit_xid, run_id COLLATE "C", step_index);

DROP INDEX IF EXISTS cogworx_journal_steps_committed;
"""

# Idempotent migration of a pre-3.5 runs table to carry the durable lethal-trifecta taint bit
# (S10 + S6). The DEFAULT false backfills every existing row to untainted; the flip to true is a
# monotonic UPDATE in set_run_tainted, written BEFORE cap.invoke (fail-closed, S6 ordering).
_MIGRATE_TAINTED = """
ALTER TABLE cogworx_journal_runs
    ADD COLUMN IF NOT EXISTS tainted boolean NOT NULL DEFAULT false;
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
        await conn.execute(_MIGRATE_COMMIT_XID)
        await conn.execute(_MIGRATE_TAINTED)

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

    async def set_run_tainted(self, run_id: str) -> None:
        # Monotonic False→True flip of the durable lethal-trifecta taint bit (S10 + S6). A single
        # unconditional UPDATE — writing true when already true is a row-level no-op. Awaited BEFORE
        # cap.invoke in dispatch_one (fail-closed). Unknown run_id silently updates zero rows.
        conn = await self._connection()
        await conn.execute(
            "UPDATE cogworx_journal_runs SET tainted = true WHERE run_id = %s",
            (run_id,),
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

    async def committed_steps_after(
        self, cursor: ProjectionCursor | None, *, limit: int
    ) -> Sequence[ProjectedStep]:
        # The COMMIT-ORDER projection read seam (S1), VISIBILITY-FENCED on commit_xid.
        #   * FENCE: commit_xid < pg_snapshot_xmin(pg_current_snapshot())::text::bigint — only rows
        #     below any xid an in-flight txn could still commit beneath are eligible. This closes
        #     the sequence-visibility race (a consumer never advances past an ordinal a concurrent
        #     writer can still slot under).
        #   * STRICT KEYSET: (commit_xid, run_id COLLATE "C", step_index) > cursor (omitted on
        #     cold start). The fenced column and the ORDER BY leading column are BOTH commit_xid
        #     (a hybrid is wrong); run_id is byte-ordered COLLATE "C" to match the Neo4j-side
        #     codepoint cursor.
        # xid8 is read as BIGINT via ::text::bigint end-to-end. Uses the
        # cogworx_journal_steps_commit_xid (commit_xid, run_id COLLATE "C", step_index) index. LIMIT
        # bounds the batch. No lookback/ceiling: commit_xid is monotonic-in-commit-order, so one
        # fenced forward read is total — there is no late-row band.
        return await self._read_projection(cursor, limit=limit, fenced=True)

    async def _read_projection(
        self, cursor: ProjectionCursor | None, *, limit: int, fenced: bool
    ) -> Sequence[ProjectedStep]:
        # Shared projection read. ``fenced=False`` is the SPIKE NEGATIVE CONTROL ONLY (the predicate
        # minus the visibility fence): it consumes rows whose commit_xid is assigned but not yet
        # committed, advancing the cursor past an ordinal an in-flight txn still sits below, which
        # PERMANENTLY loses that row once it commits. Never call it on the projection hot path.
        conn = await self._connection()
        clauses = (
            ["commit_xid < pg_snapshot_xmin(pg_current_snapshot())::text::bigint"] if fenced else []
        )
        params: list[Any] = []
        if cursor is not None:
            # The strict keyset > cursor, written EXPANDED (not a row-value tuple) so the run_id
            # tie-break can be pinned COLLATE "C": the SQL must byte-order run_id IDENTICALLY to
            # the Neo4j-side Python codepoint cursor (ordinal_ge/ordinal_max in procedural_kg.py),
            # else a user on an ICU/locale Postgres collation gets a divergent order from
            # cog-worx's own cursor and re-projects already-seen rows (harmless churn, but a real
            # cross-collation smell on a published OSS package). COLLATE inside a row-value tuple
            # `(a,b,c) > (...)` is NOT honored per-element by Postgres, so the expanded form is
            # required. Stays semantically identical to the tuple keyset; commit_xid/step_index are
            # numeric (no collation). The ORDER BY pins the SAME COLLATE "C" so the predicate and
            # ordering agree, and the cogworx_journal_steps_commit_xid index is built COLLATE "C"
            # to satisfy it.
            clauses.append(
                "(commit_xid > %s "
                'OR (commit_xid = %s AND (run_id COLLATE "C" > %s '
                "OR (run_id = %s AND step_index > %s))))"
            )
            params.extend(
                (
                    cursor.commit_ordinal,
                    cursor.commit_ordinal,
                    cursor.run_id,
                    cursor.run_id,
                    cursor.step_index,
                )
            )
        where = f"WHERE {' AND '.join(clauses)} " if clauses else ""
        params.append(limit)
        db_cursor = await conn.execute(
            "SELECT run_id, step_index, stage_name, result, committed_at, commit_xid "
            "FROM cogworx_journal_steps "
            f"{where}"
            'ORDER BY commit_xid, run_id COLLATE "C", step_index '
            "LIMIT %s",
            tuple(params),
        )
        rows = await db_cursor.fetchall()
        return tuple(
            ProjectedStep(record=self._row_to_step(row), commit_ordinal=row[5]) for row in rows
        )

    async def load_run(self, run_id: str) -> RunState | None:
        conn = await self._connection()
        run_cursor = await conn.execute(
            "SELECT session_id, status, pathway_id, pathway_version, pathway_fingerprint, tainted "
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
        tainted: bool = run_row[5]

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
            tainted=tainted,
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
        # The commit_xid projection index lives in _MIGRATE_COMMIT_XID (not _SCHEMA), so run it here
        # too — on the freshly recreated table the ALTER is a no-op and the CREATE INDEX adds it.
        await conn.execute(_MIGRATE_COMMIT_XID)
        await conn.execute(_MIGRATE_TAINTED)

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
