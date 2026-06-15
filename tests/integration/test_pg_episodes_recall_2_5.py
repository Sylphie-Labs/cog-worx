"""Integration tests for PgEpisodeStore.recent_episodes — Pod 2.5 Stream B2 (live Postgres).

Covers:
- Newest-first ordering: step_index DESC, turn_index DESC.
- Turn-index tiebreak within the same step.
- before filter is exclusive: occurred_at must be strictly less than before.
- before=None returns all episodes up to limit.
- limit cap: only the N most recent rows are returned.
- Session isolation: each session's recent_episodes returns only its own rows.
- Empty table / nonexistent session returns [].
- All Episode fields are populated on returned objects.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.pg_episodes import PgEpisodeStore
from cogworx.substrate.episodes import Episode
from cogworx.substrate.journal import ProjectionCursor

pytestmark = [pytest.mark.integration, pytest.mark.postgres]

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


def _cursor(
    *, commit_ordinal: int = 1, run_id: str = "r1", step_index: int = 0
) -> ProjectionCursor:
    return ProjectionCursor(commit_ordinal=commit_ordinal, run_id=run_id, step_index=step_index)


async def _project(store: PgEpisodeStore, episodes: list[Episode], step_index: int = 0) -> None:
    await store.project_episodes(
        "recall-consumer",
        episodes,
        _cursor(commit_ordinal=step_index + 1, step_index=step_index),
    )


# ---------------------------------------------------------------------------
# test_newest_first_ordering
# ---------------------------------------------------------------------------


async def test_newest_first_ordering(store: PgEpisodeStore) -> None:
    """recent_episodes returns episodes in (step_index, turn_index) DESC order."""
    episodes = [
        _episode(episode_id="r1:1:0", step_index=1, turn_index=0),
        _episode(episode_id="r1:2:0", step_index=2, turn_index=0),
        _episode(episode_id="r1:3:0", step_index=3, turn_index=0),
    ]
    await _project(store, episodes, step_index=3)

    result = await store.recent_episodes("sess-1")

    assert len(result) == 3
    assert [e.episode_id for e in result] == ["r1:3:0", "r1:2:0", "r1:1:0"]


# ---------------------------------------------------------------------------
# test_turn_index_tiebreak
# ---------------------------------------------------------------------------


async def test_turn_index_tiebreak(store: PgEpisodeStore) -> None:
    """Two episodes with the same step_index are returned in turn_index DESC order."""
    episodes = [
        _episode(episode_id="r1:5:0", step_index=5, turn_index=0),
        _episode(episode_id="r1:5:1", step_index=5, turn_index=1),
    ]
    await _project(store, episodes, step_index=5)

    result = await store.recent_episodes("sess-1")

    assert len(result) == 2
    assert result[0].episode_id == "r1:5:1"
    assert result[1].episode_id == "r1:5:0"


# ---------------------------------------------------------------------------
# test_before_filter_exclusive
# ---------------------------------------------------------------------------


async def test_before_filter_exclusive(store: PgEpisodeStore) -> None:
    """before is exclusive: occurred_at == before is NOT returned; occurred_at < before IS."""
    t_exact = _T0
    t_plus_one = _T0 + timedelta(seconds=1)

    episodes = [_episode(episode_id="r1:0:0", step_index=0, turn_index=0, occurred_at=t_exact)]
    await _project(store, episodes, step_index=0)

    # before == occurred_at → must NOT return the episode (exclusive)
    result_excluded = await store.recent_episodes("sess-1", before=t_exact)
    assert len(result_excluded) == 0

    # before == occurred_at + 1s → must return the episode
    result_included = await store.recent_episodes("sess-1", before=t_plus_one)
    assert len(result_included) == 1
    assert result_included[0].episode_id == "r1:0:0"


# ---------------------------------------------------------------------------
# test_before_none_returns_all
# ---------------------------------------------------------------------------


async def test_before_none_returns_all(store: PgEpisodeStore) -> None:
    """before=None applies no upper-bound filter and returns all episodes up to limit."""
    episodes = [_episode(episode_id=f"r1:{i}:0", step_index=i, turn_index=0) for i in range(4)]
    await _project(store, episodes, step_index=3)

    result = await store.recent_episodes("sess-1", before=None)
    assert len(result) == 4


# ---------------------------------------------------------------------------
# test_limit_cap
# ---------------------------------------------------------------------------


async def test_limit_cap(store: PgEpisodeStore) -> None:
    """limit=2 returns exactly the 2 most recent episodes."""
    episodes = [_episode(episode_id=f"r1:{i}:0", step_index=i, turn_index=0) for i in range(5)]
    await _project(store, episodes, step_index=4)

    result = await store.recent_episodes("sess-1", limit=2)

    assert len(result) == 2
    # The 2 most recent are step 4, then step 3
    assert result[0].step_index == 4
    assert result[1].step_index == 3


# ---------------------------------------------------------------------------
# test_session_isolation
# ---------------------------------------------------------------------------


async def test_session_isolation(store: PgEpisodeStore) -> None:
    """recent_episodes for one session does not return rows from another session."""
    eps_a = [_episode(episode_id="a:0:0", session_id="sess-a", step_index=0)]
    eps_b = [_episode(episode_id="b:0:0", session_id="sess-b", step_index=0)]

    await store.project_episodes("consumer-a", eps_a, _cursor(commit_ordinal=1, step_index=0))
    await store.project_episodes("consumer-b", eps_b, _cursor(commit_ordinal=2, step_index=0))

    result_a = await store.recent_episodes("sess-a")
    result_b = await store.recent_episodes("sess-b")

    assert len(result_a) == 1 and result_a[0].session_id == "sess-a"
    assert len(result_b) == 1 and result_b[0].session_id == "sess-b"
    assert result_a[0].episode_id != result_b[0].episode_id


# ---------------------------------------------------------------------------
# test_empty_table
# ---------------------------------------------------------------------------


async def test_empty_table(store: PgEpisodeStore) -> None:
    """recent_episodes on a nonexistent session returns [] (S8: empty-table safe)."""
    result = await store.recent_episodes("nonexistent-session")
    assert result == ()


# ---------------------------------------------------------------------------
# test_all_fields_present
# ---------------------------------------------------------------------------


async def test_all_fields_present(store: PgEpisodeStore) -> None:
    """All Episode fields are populated on objects returned by recent_episodes."""
    ep = _episode(
        episode_id="r1:7:3",
        run_id="run-full",
        step_index=7,
        turn_index=3,
        session_id="sess-fields",
        role="assistant",
        content="full content",
        kind="dialogue",
        occurred_at=_T0,
    )
    await store.project_episodes(
        "fields-consumer", [ep], _cursor(commit_ordinal=1, run_id="run-full", step_index=7)
    )

    result = await store.recent_episodes("sess-fields")

    assert len(result) == 1
    r = result[0]
    assert r.episode_id == "r1:7:3"
    assert r.run_id == "run-full"
    assert r.step_index == 7
    assert r.turn_index == 3
    assert r.session_id == "sess-fields"
    assert r.role == "assistant"
    assert r.content == "full content"
    assert r.kind == "dialogue"
    assert r.occurred_at.tzinfo is not None  # UTC-aware
    assert r.occurred_at == _T0
