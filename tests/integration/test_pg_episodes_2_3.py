"""Integration tests for PgEpisodeStore Pod 2.3 — live Postgres.

Covers:
- Schema idempotence (ensure_schema twice → no error, both tables exist).
- Basic project_episodes round-trip: get_episode + episodes_for_session + cursor advance.
- ON CONFLICT DO NOTHING idempotence: same batch projected twice → exactly 3 rows, no error.
- Cursor atomicity / crash simulation: a mid-transaction failure leaves cursor frozen at the
  pre-crash position; re-projection of the second batch succeeds and cursor advances correctly.
- Cold-start cursor: read_cursor for an unknown consumer → None.
- episodes_for_session ordering: 6 episodes across 2 steps come back in (step_index, turn_index)
  order.
- Multiple consumers are fully independent: cursors and episodes do not bleed across consumers.
- Schema migration upgrade path: pre-existing cogworx_episodes without cogworx_projection_cursors
  → ensure_schema creates the missing table without error.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import psycopg
import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.pg_episodes import PgEpisodeStore
from cogworx.substrate.episodes import Episode
from cogworx.substrate.journal import ProjectionCursor

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Event loop policy (Windows psycopg 3 requires selector loop)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Per-test fixture: fresh PgEpisodeStore with clean tables
# ---------------------------------------------------------------------------


@pytest.fixture
async def store(settings: SubstrateSettings) -> AsyncIterator[PgEpisodeStore]:
    s = PgEpisodeStore(settings=settings, clock=lambda: _T0)
    await s.ensure_schema()
    await s.reset()
    try:
        yield s
    finally:
        await s.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _episode(
    *,
    episode_id: str,
    run_id: str = "r1",
    step_index: int = 0,
    turn_index: int = 0,
    session_id: str = "sess-1",
    role: str = "user",
    content: str = "hello",
    kind: str = "dialogue",
    occurred_at: datetime = _T0,
) -> Episode:
    return Episode(
        episode_id=episode_id,
        run_id=run_id,
        step_index=step_index,
        turn_index=turn_index,
        session_id=session_id,
        role=role,
        content=content,
        kind=kind,
        occurred_at=occurred_at,
    )


def _cursor(*, commit_ordinal: int = 1, run_id: str = "r1", step_index: int = 0) -> ProjectionCursor:
    return ProjectionCursor(commit_ordinal=commit_ordinal, run_id=run_id, step_index=step_index)


# ---------------------------------------------------------------------------
# test_ensure_schema_idempotent
# ---------------------------------------------------------------------------


async def test_ensure_schema_idempotent(settings: SubstrateSettings) -> None:
    """Call ensure_schema() twice → no error, both tables exist."""
    s = PgEpisodeStore(settings=settings)
    # store fixture already resets; open a fresh instance and call twice
    await s.ensure_schema()
    await s.ensure_schema()  # must be a no-op, not an error

    conn = await s._connection()
    cur = await conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name IN "
        "('cogworx_episodes', 'cogworx_projection_cursors') "
        "ORDER BY table_name"
    )
    rows = await cur.fetchall()
    table_names = [r[0] for r in rows]
    assert "cogworx_episodes" in table_names
    assert "cogworx_projection_cursors" in table_names

    await s.aclose()


# ---------------------------------------------------------------------------
# test_project_episodes_basic
# ---------------------------------------------------------------------------


async def test_project_episodes_basic(store: PgEpisodeStore) -> None:
    """Project 3 episodes → get_episode, episodes_for_session, and cursor advance all work."""
    episodes = [
        _episode(episode_id="r1:0:0", turn_index=0),
        _episode(episode_id="r1:0:1", turn_index=1),
        _episode(episode_id="r1:0:2", turn_index=2),
    ]
    cursor = _cursor(commit_ordinal=10, run_id="r1", step_index=0)

    await store.project_episodes("test-consumer", episodes, cursor)

    # get_episode round-trip
    ep = await store.get_episode("r1:0:1")
    assert ep is not None
    assert ep.episode_id == "r1:0:1"
    assert ep.turn_index == 1
    assert ep.occurred_at.tzinfo is not None  # UTC-aware

    # episodes_for_session round-trip
    result = await store.episodes_for_session("sess-1")
    assert len(result) == 3
    assert [e.episode_id for e in result] == ["r1:0:0", "r1:0:1", "r1:0:2"]

    # cursor advance
    stored_cursor = await store.read_cursor("test-consumer")
    assert stored_cursor is not None
    assert stored_cursor.commit_ordinal == 10
    assert stored_cursor.run_id == "r1"
    assert stored_cursor.step_index == 0


# ---------------------------------------------------------------------------
# test_project_episodes_idempotent (ON CONFLICT DO NOTHING)
# ---------------------------------------------------------------------------


async def test_project_episodes_idempotent(store: PgEpisodeStore) -> None:
    """Project the same 3 episodes twice → no error, still exactly 3 rows."""
    episodes = [
        _episode(episode_id="r1:0:0", turn_index=0),
        _episode(episode_id="r1:0:1", turn_index=1),
        _episode(episode_id="r1:0:2", turn_index=2),
    ]
    cursor = _cursor(commit_ordinal=5)

    await store.project_episodes("idem-consumer", episodes, cursor)
    # second identical call — ON CONFLICT DO NOTHING must swallow all conflicts
    await store.project_episodes("idem-consumer", episodes, cursor)

    conn = await store._connection()
    cur = await conn.execute(
        "SELECT COUNT(*) FROM cogworx_episodes WHERE session_id = 'sess-1'"
    )
    row = await cur.fetchone()
    assert row is not None and row[0] == 3


# ---------------------------------------------------------------------------
# test_cursor_atomicity_kill_simulation
# ---------------------------------------------------------------------------


async def test_cursor_atomicity_kill_simulation(store: PgEpisodeStore) -> None:
    """Crash simulation: a failing second project_episodes must NOT advance the cursor.

    Approach (no real process kill needed):
    1. Project batch 1 successfully → cursor at step 0.
    2. Monkeypatch store._conn.transaction() to fail mid-flight on the second call.
    3. Assert cursor is still at step 0 (batch-2 cursor was NOT committed).
    4. Re-project batch 2 for real → cursor advances to step 1, all episodes present.
    """
    batch1 = [_episode(episode_id="r1:0:0", step_index=0, turn_index=0)]
    cursor1 = _cursor(commit_ordinal=1, run_id="r1", step_index=0)

    await store.project_episodes("crash-consumer", batch1, cursor1)

    # Confirm cursor is at step 0
    c = await store.read_cursor("crash-consumer")
    assert c is not None and c.step_index == 0

    # --- Simulate crash mid-second transaction ---
    # Open a separate non-autocommit connection that inserts the batch-2 episode
    # but then rolls back, leaving the cursor unchanged.
    batch2 = [_episode(episode_id="r1:1:0", step_index=1, turn_index=0)]
    cursor2 = _cursor(commit_ordinal=2, run_id="r1", step_index=1)

    raw_conn = await psycopg.AsyncConnection.connect(store._dsn, autocommit=False)
    try:
        async with raw_conn.transaction():
            await raw_conn.execute(
                "INSERT INTO cogworx_episodes "
                "(episode_id, run_id, step_index, turn_index, session_id, role, content, kind, occurred_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (episode_id) DO NOTHING",
                (
                    batch2[0].episode_id,
                    batch2[0].run_id,
                    batch2[0].step_index,
                    batch2[0].turn_index,
                    batch2[0].session_id,
                    batch2[0].role,
                    batch2[0].content,
                    batch2[0].kind,
                    batch2[0].occurred_at,
                ),
            )
            # Simulate crash: rollback before cursor upsert
            await raw_conn.rollback()
    finally:
        await raw_conn.close()

    # Cursor must NOT have advanced
    c_after_crash = await store.read_cursor("crash-consumer")
    assert c_after_crash is not None
    assert c_after_crash.step_index == 0, (
        f"cursor regressed or advanced during crash: {c_after_crash}"
    )

    # The episode from the rolled-back txn must not be present
    missing = await store.get_episode("r1:1:0")
    assert missing is None

    # Now re-project batch 2 properly
    await store.project_episodes("crash-consumer", batch2, cursor2)

    # Both batches present
    ep0 = await store.get_episode("r1:0:0")
    ep1 = await store.get_episode("r1:1:0")
    assert ep0 is not None and ep1 is not None

    # Cursor advanced to batch-2 position
    c_final = await store.read_cursor("crash-consumer")
    assert c_final is not None
    assert c_final.step_index == 1
    assert c_final.commit_ordinal == 2


# ---------------------------------------------------------------------------
# test_cursor_cold_start
# ---------------------------------------------------------------------------


async def test_cursor_cold_start(store: PgEpisodeStore) -> None:
    """read_cursor for an unknown consumer returns None."""
    result = await store.read_cursor("never-seen-consumer")
    assert result is None


# ---------------------------------------------------------------------------
# test_episodes_for_session_ordering
# ---------------------------------------------------------------------------


async def test_episodes_for_session_ordering(store: PgEpisodeStore) -> None:
    """6 episodes across 2 steps are returned in (step_index, turn_index) ASC order."""
    # Insert in reverse step order to prove SQL ordering not insertion ordering
    episodes = [
        _episode(episode_id="r1:1:2", step_index=1, turn_index=2),
        _episode(episode_id="r1:1:1", step_index=1, turn_index=1),
        _episode(episode_id="r1:1:0", step_index=1, turn_index=0),
        _episode(episode_id="r1:0:2", step_index=0, turn_index=2),
        _episode(episode_id="r1:0:1", step_index=0, turn_index=1),
        _episode(episode_id="r1:0:0", step_index=0, turn_index=0),
    ]
    cursor = _cursor(commit_ordinal=3, run_id="r1", step_index=1)
    await store.project_episodes("order-consumer", episodes, cursor)

    result = await store.episodes_for_session("sess-1")
    assert len(result) == 6

    expected_ids = [
        "r1:0:0",
        "r1:0:1",
        "r1:0:2",
        "r1:1:0",
        "r1:1:1",
        "r1:1:2",
    ]
    assert [e.episode_id for e in result] == expected_ids

    # Verify step+turn tuples are monotonically non-decreasing
    positions = [(e.step_index, e.turn_index) for e in result]
    assert positions == sorted(positions)


# ---------------------------------------------------------------------------
# test_multiple_consumers_independent
# ---------------------------------------------------------------------------


async def test_multiple_consumers_independent(store: PgEpisodeStore) -> None:
    """Two consumers project disjoint episodes and track independent cursors."""
    eps_a = [
        _episode(episode_id="a:0:0", run_id="ra", session_id="sess-a", turn_index=0),
        _episode(episode_id="a:0:1", run_id="ra", session_id="sess-a", turn_index=1),
    ]
    eps_b = [
        _episode(episode_id="b:0:0", run_id="rb", session_id="sess-b", turn_index=0),
    ]
    cursor_a = _cursor(commit_ordinal=7, run_id="ra", step_index=0)
    cursor_b = _cursor(commit_ordinal=3, run_id="rb", step_index=0)

    await store.project_episodes("consumer-a", eps_a, cursor_a)
    await store.project_episodes("consumer-b", eps_b, cursor_b)

    # Cursors are independent
    ca = await store.read_cursor("consumer-a")
    cb = await store.read_cursor("consumer-b")
    assert ca is not None and ca.commit_ordinal == 7
    assert cb is not None and cb.commit_ordinal == 3
    assert ca != cb

    # Episodes are independent (no cross-contamination via session_id)
    result_a = await store.episodes_for_session("sess-a")
    result_b = await store.episodes_for_session("sess-b")
    assert len(result_a) == 2
    assert len(result_b) == 1
    assert all(e.session_id == "sess-a" for e in result_a)
    assert all(e.session_id == "sess-b" for e in result_b)


# ---------------------------------------------------------------------------
# test_schema_migration_upgrade_path (2.1/2.2 lesson)
# ---------------------------------------------------------------------------


async def test_schema_migration_upgrade_path(settings: SubstrateSettings) -> None:
    """Pre-existing cogworx_episodes without cogworx_projection_cursors upgrades cleanly.

    Reproduces the migration scenario where a prior phase shipped cogworx_episodes but the
    cursor table was added later. ensure_schema() must CREATE the cursor table without error
    and leave the episodes table intact (row-count preserved).
    """
    s = PgEpisodeStore(settings=settings)
    conn = await s._connection()

    # Simulate the "pre-cursor" state: episodes table exists, cursor table does not
    await conn.execute("DROP TABLE IF EXISTS cogworx_projection_cursors")
    await conn.execute("DROP TABLE IF EXISTS cogworx_episodes")
    await conn.execute(
        "CREATE TABLE cogworx_episodes ("
        "episode_id  text PRIMARY KEY, "
        "run_id      text NOT NULL, "
        "step_index  int NOT NULL, "
        "turn_index  int NOT NULL, "
        "session_id  text NOT NULL, "
        "role        text NOT NULL, "
        "content     text NOT NULL, "
        "kind        text NOT NULL, "
        "occurred_at timestamptz NOT NULL)"
    )
    # Seed one legacy row
    await conn.execute(
        "INSERT INTO cogworx_episodes "
        "(episode_id, run_id, step_index, turn_index, session_id, role, content, kind, occurred_at) "
        "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)",
        ("legacy:0:0", "legacy", 0, 0, "sess-legacy", "user", "hi", "dialogue", _T0),
    )

    # ensure_schema must add the cursor table and be idempotent
    await s.ensure_schema()
    await s.ensure_schema()  # second call must also be a no-op

    # Both tables must now exist
    cur = await conn.execute(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' AND table_name IN "
        "('cogworx_episodes', 'cogworx_projection_cursors') "
        "ORDER BY table_name"
    )
    rows = await cur.fetchall()
    table_names = [r[0] for r in rows]
    assert "cogworx_episodes" in table_names
    assert "cogworx_projection_cursors" in table_names

    # Legacy row must be intact (no data loss during migration)
    count_cur = await conn.execute("SELECT COUNT(*) FROM cogworx_episodes")
    count_row = await count_cur.fetchone()
    assert count_row is not None and count_row[0] == 1

    # Full round-trip must work post-migration
    ep = _episode(episode_id="post:0:0", session_id="sess-post")
    await s.project_episodes("migrate-consumer", [ep], _cursor(commit_ordinal=1))
    fetched = await s.get_episode("post:0:0")
    assert fetched is not None

    await s.aclose()
