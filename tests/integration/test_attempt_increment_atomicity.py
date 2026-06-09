"""Integration: the REAL atomic attempt-counter against live Postgres/Timescale (CANON S6).

R1/R2's live counterpart. The in-memory ``increment_attempt`` proves the per-``(run_id,
step_index)`` counter climbs deterministically; the TRUE concurrency property — N callers bumping
the ``(run_id, step_index)`` key without losing an update — can only be proven against real PG.
``increment_attempt`` is backed by a single atomic
``INSERT … ON CONFLICT (run_id, step_index) DO UPDATE attempt = attempt + 1 RETURNING attempt``, so
each concurrent caller row-locks the conflicting row, then increments serially — no lost update, no
double-count, every caller gets a DISTINCT returned value. A non-atomic read-then-write would lose
updates under contention and this test would fail.

Marked ``integration`` so it is deselected unless ``-m integration`` and the stack is up
(``docker compose up -d``). It does not run in unit CI but must be correct when PG is present.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.timescale_journal import TimescaleJournal

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_RUN_ID = "attempt-race"
_STEP_INDEX = 1
_CONCURRENCY = 50


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def prepared_journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


async def test_read_attempt_zero_before_any_increment(
    prepared_journal: TimescaleJournal,
) -> None:
    """``read_attempt`` returns 0 for a ``(run_id, step_index)`` that has never been incremented."""
    assert await prepared_journal.read_attempt(_RUN_ID, _STEP_INDEX) == 0


async def test_increment_attempt_returns_monotonic_count(
    prepared_journal: TimescaleJournal,
) -> None:
    """Sequential ``increment_attempt`` returns 1, 2, 3 … and ``read_attempt`` tracks it."""
    assert await prepared_journal.increment_attempt(_RUN_ID, _STEP_INDEX) == 1
    assert await prepared_journal.increment_attempt(_RUN_ID, _STEP_INDEX) == 2
    assert await prepared_journal.increment_attempt(_RUN_ID, _STEP_INDEX) == 3
    assert await prepared_journal.read_attempt(_RUN_ID, _STEP_INDEX) == 3
    # A DISTINCT (run_id, step_index) key is independent.
    assert await prepared_journal.read_attempt(_RUN_ID, _STEP_INDEX + 1) == 0


async def test_concurrent_increment_attempt_no_lost_update(
    prepared_journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """N concurrent ``increment_attempt`` on the SAME key on SEPARATE connections -> the final count
    equals N and every caller got a DISTINCT value in ``1..N`` (no lost update, no double-count).

    The atomic ``… DO UPDATE attempt = attempt + 1 RETURNING attempt`` row-locks the conflicting row
    so the increments serialize; a non-atomic read-then-write would collide and lose updates.
    """
    others = [TimescaleJournal(settings=settings) for _ in range(_CONCURRENCY)]
    try:
        results = await asyncio.gather(
            *(j.increment_attempt(_RUN_ID, _STEP_INDEX) for j in others)
        )
    finally:
        for j in others:
            await j.aclose()

    # Every caller got a DISTINCT value and the union is exactly 1..N — no lost/duplicated update.
    assert sorted(results) == list(range(1, _CONCURRENCY + 1))
    assert await prepared_journal.read_attempt(_RUN_ID, _STEP_INDEX) == _CONCURRENCY
