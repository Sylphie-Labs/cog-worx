"""Integration: the REAL atomic timer-claim race against live Postgres/Timescale (CANON S6).

C4's live counterpart. The in-memory ``claim_due_timers`` proves single-delivery sequentially; the
TRUE race — two concurrent sweepers leasing the SAME due timers — can only be proven against the
real journal. There is NO ``FOR UPDATE SKIP LOCKED``: the bare atomic ``UPDATE … WHERE … RETURNING``
is single-delivery-safe under READ COMMITTED row locking. The conditional ``UPDATE`` row-locks each
candidate; a second concurrent claim blocks on that lock, then re-evaluates its ``WHERE`` against
the now-committed ``claimed_at`` once the lock releases — so it sees the fresh (non-stale) lease and
gets 0 rows for that timer. This test fires two ``claim_due_timers`` concurrently on SEPARATE
connections via ``asyncio.gather`` and asserts the union delivers each due timer to EXACTLY ONE
caller.

Marked ``integration`` so it is deselected unless ``-m integration`` and the stack is up
(``docker compose up -d``). It does not run in unit CI but must be correct when PG is present.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.substrate.journal import Timer

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_WAKE = _NOW - timedelta(minutes=1)  # already due
_LEASE_TTL = timedelta(minutes=5)
_TIMER_COUNT = 50


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def seeded_journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    for i in range(_TIMER_COUNT):
        await adapter.set_timer(Timer(run_id=f"race-{i}", timer_id=f"race:{i}", wake_at=_WAKE))
    try:
        yield adapter
    finally:
        await adapter.aclose()


async def test_concurrent_claim_due_timers_delivers_each_timer_once(
    seeded_journal: TimescaleJournal, settings: SubstrateSettings
) -> None:
    """Two concurrent sweepers leasing the same due set -> each timer leased by exactly one.

    The seeded journal arms ``_TIMER_COUNT`` due timers. Two ``claim_due_timers`` calls run
    concurrently on SEPARATE connections (a second ``TimescaleJournal``); the per-row lease admits
    exactly one claimant per timer even under contention. A non-atomic read-then-update would
    double-deliver and this test would fail. Asserts the two result sets are DISJOINT, their union
    is the full due set, and no ``timer_id`` appears twice.
    """
    other = TimescaleJournal(settings=settings)
    try:
        first, second = await asyncio.gather(
            seeded_journal.claim_due_timers(_NOW, lease_ttl=_LEASE_TTL),
            other.claim_due_timers(_NOW, lease_ttl=_LEASE_TTL),
        )
    finally:
        await other.aclose()

    first_ids = [timer.timer_id for timer in first]
    second_ids = [timer.timer_id for timer in second]

    # No timer claimed twice within either result set (each call's RETURNING is unique).
    assert len(first_ids) == len(set(first_ids))
    assert len(second_ids) == len(set(second_ids))
    # The two claimants are DISJOINT — no timer delivered to both (the single-delivery guarantee).
    assert set(first_ids).isdisjoint(second_ids)
    # Their union is the full due set: every due timer was delivered to exactly one caller.
    assert set(first_ids) | set(second_ids) == {f"race:{i}" for i in range(_TIMER_COUNT)}

    # Lease-not-delete (the live-PG counterpart of unit C7): every claimed row STILL EXISTS with
    # claimed_at stamped — a destructive-DELETE claim would have removed them and stranded the runs.
    survivors = await seeded_journal.due_timers(_NOW)
    survivor_ids = {timer.timer_id for timer in survivors}
    assert survivor_ids == {f"race:{i}" for i in range(_TIMER_COUNT)}
    assert all(timer.claimed_at is not None for timer in survivors)
