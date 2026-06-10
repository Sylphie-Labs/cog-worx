"""The Postgres episode-store adapter (CANON S3, Pod 2.3).

Implements the full :class:`~cogworx.substrate.episodes.EpisodeStore` seam: durable, time-ordered
episode storage backed by Postgres. Episodes are immutable projection rows; the
``cogworx_projection_cursors`` table is shared by any consumer that projects from the journal
(EpisodeProjector, ClaimExtractor, …) and is keyed by a ``consumer`` string.

## Key invariants

- **Atomicity:** ``project_episodes`` inserts all episodes AND upserts the cursor in ONE
  transaction — a crash between the two is impossible (S6).
- **Idempotency:** episode insert is ``ON CONFLICT (episode_id) DO NOTHING`` — first-write-wins;
  re-projecting a batch that partially committed is safe.
- **Cursor never regresses:** cursor upsert uses ``ordinal_max`` semantics — the ``ON CONFLICT DO
  UPDATE`` only fires when the incoming ``(commit_ordinal, run_id, step_index)`` tuple is strictly
  greater than the stored one (Postgres row-value comparison). Mirrors the Neo4j cursor guard.
  A re-entrant call with a lower cursor is a no-op (stored value unchanged).
- **occurred_at comes from the journal step:** the clock injected here is unused in practice but
  kept for consistency with other adapters that need a clock for writes (PgLatentStore pattern).

## Schema

    cogworx_episodes (
      episode_id   text PRIMARY KEY,
      run_id       text NOT NULL,
      step_index   int NOT NULL,
      turn_index   int NOT NULL,
      session_id   text NOT NULL,
      role         text NOT NULL,
      content      text NOT NULL,
      kind         text NOT NULL,
      occurred_at  timestamptz NOT NULL
    )

    cogworx_projection_cursors (
      consumer      text PRIMARY KEY,
      commit_ordinal bigint NOT NULL,
      run_id        text NOT NULL,
      step_index    int NOT NULL
    )
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import psycopg

from cogworx.adapters.config import SubstrateSettings
from cogworx.substrate.episodes import Episode
from cogworx.substrate.journal import ProjectionCursor

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_CREATE_EPISODES = """\
CREATE TABLE IF NOT EXISTS cogworx_episodes (
  episode_id   text PRIMARY KEY,
  run_id       text NOT NULL,
  step_index   int NOT NULL,
  turn_index   int NOT NULL,
  session_id   text NOT NULL,
  role         text NOT NULL,
  content      text NOT NULL,
  kind         text NOT NULL,
  occurred_at  timestamptz NOT NULL
)"""

_CREATE_CURSORS = """\
CREATE TABLE IF NOT EXISTS cogworx_projection_cursors (
  consumer      text PRIMARY KEY,
  commit_ordinal bigint NOT NULL,
  run_id        text NOT NULL,
  step_index    int NOT NULL
)"""

# ---------------------------------------------------------------------------
# DML
# ---------------------------------------------------------------------------

_INSERT_EPISODE = """\
INSERT INTO cogworx_episodes
  (episode_id, run_id, step_index, turn_index, session_id, role, content, kind, occurred_at)
VALUES
  (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (episode_id) DO NOTHING"""

_UPSERT_CURSOR = """\
INSERT INTO cogworx_projection_cursors (consumer, commit_ordinal, run_id, step_index)
VALUES (%s, %s, %s, %s)
ON CONFLICT (consumer) DO UPDATE SET
  commit_ordinal = EXCLUDED.commit_ordinal,
  run_id         = EXCLUDED.run_id,
  step_index     = EXCLUDED.step_index
WHERE (EXCLUDED.commit_ordinal, EXCLUDED.run_id, EXCLUDED.step_index) >
      (cogworx_projection_cursors.commit_ordinal,
       cogworx_projection_cursors.run_id,
       cogworx_projection_cursors.step_index)"""

_READ_CURSOR = """\
SELECT commit_ordinal, run_id, step_index
FROM cogworx_projection_cursors
WHERE consumer = %s"""

_EPISODES_FOR_SESSION = """\
SELECT episode_id, run_id, step_index, turn_index, session_id, role, content, kind, occurred_at
FROM cogworx_episodes
WHERE session_id = %s
ORDER BY step_index ASC, turn_index ASC
LIMIT %s"""

_GET_EPISODE = """\
SELECT episode_id, run_id, step_index, turn_index, session_id, role, content, kind, occurred_at
FROM cogworx_episodes
WHERE episode_id = %s"""

_RECENT_EPISODES = """\
SELECT episode_id, run_id, step_index, turn_index, session_id, role, content, kind, occurred_at
FROM cogworx_episodes
WHERE session_id = %s
  AND (%s::timestamptz IS NULL OR occurred_at < %s)
ORDER BY step_index DESC, turn_index DESC
LIMIT %s"""


class PgEpisodeStore:
    """An :class:`~cogworx.substrate.episodes.EpisodeStore` backed by Postgres."""

    def __init__(
        self,
        *,
        settings: SubstrateSettings | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._dsn = (settings or SubstrateSettings()).pg_dsn
        # clock is included for adapter consistency; occurred_at is sourced from the journal step.
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._conn: psycopg.AsyncConnection[Any] | None = None

    async def _connection(self) -> psycopg.AsyncConnection[Any]:
        if self._conn is None or self._conn.closed:
            self._conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
        return self._conn

    async def ensure_schema(self) -> None:
        """Create cogworx_episodes and cogworx_projection_cursors if they do not exist."""
        conn = await self._connection()
        await conn.execute(_CREATE_EPISODES)
        await conn.execute(_CREATE_CURSORS)

    async def project_episodes(
        self,
        consumer: str,
        episodes: Sequence[Episode],
        progress: ProjectionCursor,
    ) -> None:
        """Insert episodes + advance cursor in ONE atomic transaction (S6).

        ON CONFLICT (episode_id) DO NOTHING — idempotent first-write-wins. An empty ``episodes``
        sequence still advances the cursor.
        """
        conn = await self._connection()
        async with conn.transaction():
            for ep in episodes:
                await conn.execute(
                    _INSERT_EPISODE,
                    (
                        ep.episode_id,
                        ep.run_id,
                        ep.step_index,
                        ep.turn_index,
                        ep.session_id,
                        ep.role,
                        ep.content,
                        ep.kind,
                        ep.occurred_at,
                    ),
                )
            await conn.execute(
                _UPSERT_CURSOR,
                (consumer, progress.commit_ordinal, progress.run_id, progress.step_index),
            )

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        """Return the last-scanned cursor for this consumer, or None on cold start."""
        conn = await self._connection()
        cursor = await conn.execute(_READ_CURSOR, (consumer,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return ProjectionCursor(
            commit_ordinal=int(row[0]),
            run_id=str(row[1]),
            step_index=int(row[2]),
        )

    async def episodes_for_session(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> Sequence[Episode]:
        """Return episodes for a session ordered by (step_index, turn_index) ASC."""
        conn = await self._connection()
        cursor = await conn.execute(_EPISODES_FOR_SESSION, (session_id, limit))
        rows = await cursor.fetchall()
        return tuple(_row_to_episode(row) for row in rows)

    async def recent_episodes(
        self,
        session_id: str,
        *,
        limit: int = 20,
        before: datetime | None = None,
    ) -> Sequence[Episode]:
        """Return the most-recent episodes for a session, newest-first.

        Ordered by (step_index, turn_index) DESC. ``before`` restricts to episodes with
        ``occurred_at < before`` (exclusive); ``None`` means no upper bound (S6: read-only).
        """
        if before is not None and before.tzinfo is not None:
            before = before.astimezone(UTC)
        conn = await self._connection()
        cursor = await conn.execute(_RECENT_EPISODES, (session_id, before, before, limit))
        rows = await cursor.fetchall()
        return tuple(_row_to_episode(row) for row in rows)

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Return an episode by id, or None."""
        conn = await self._connection()
        cursor = await conn.execute(_GET_EPISODE, (episode_id,))
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_episode(row)

    async def reset(self) -> None:
        """Drop and recreate the episode tables (test/dev use only)."""
        conn = await self._connection()
        await conn.execute("DROP TABLE IF EXISTS cogworx_episodes")
        await conn.execute("DROP TABLE IF EXISTS cogworx_projection_cursors")
        await self.ensure_schema()

    async def aclose(self) -> None:
        """Close the underlying connection."""
        if self._conn is not None and not self._conn.closed:
            await self._conn.close()
        self._conn = None


def _row_to_episode(row: Any) -> Episode:
    occurred_at: datetime = row[8]
    if occurred_at.tzinfo is None:
        occurred_at = occurred_at.replace(tzinfo=UTC)
    return Episode(
        episode_id=str(row[0]),
        run_id=str(row[1]),
        step_index=int(row[2]),
        turn_index=int(row[3]),
        session_id=str(row[4]),
        role=str(row[5]),
        content=str(row[6]),
        kind=str(row[7]),
        occurred_at=occurred_at,
    )


__all__ = ["PgEpisodeStore"]
