"""The durable-timer sweeper (CANON S1, S6).

The ``Sweeper`` is the OFF-write-path poller that wakes parked runs. It owns NO model and does NO
commits (S1): each ``tick`` atomically LEASES the due timers from the journal and asks the engine to
re-drive each one's run via ``fire_timer``. The lease (not a delete) is what makes the sweep safe to
crash — a leased-but-not-fired timer goes stale and the next tick reclaims it, so a timer fires
at-least-once while the run's advance past its ``Wait`` stays exactly-once (the advance is
idempotent on ``(run_id, step_index)``, S6).

Degradation (S8): with the sweeper switched off, runs that never ``Wait`` still complete normally —
only ``Wait``-bearing runs simply stay ``WAITING`` until a sweeper runs.

Carry-forward (PAUSED-timer churn): a PAUSED run's still-armed timer is re-claimed by the sweeper
every ``lease_ttl`` and the ``fire_timer`` CAS (WAITING->RUNNING) no-ops on it (the run is PAUSED,
not WAITING) — harmless churn, never a model re-call. An optional future optimization is to skip
firing timers whose run is PAUSED, or cancel-on-pause + re-arm-on-unpause.

Carry-forward (orphan retry-timer on a crashed-RUNNING strand — FINDING 2, LOW): a ``fire_timer``
winner that crashes AFTER the CAS->RUNNING but BEFORE ``cancel_timers_for_run`` leaves the run
stranded RUNNING with its retry timer still armed. The sweeper reclaims the stale lease every
``lease_ttl`` and the CAS no-ops (the run is RUNNING, not WAITING|RETRYING) — no re-execution, no
over-count — but the orphan timer RE-FIRES forever until an explicit ``resume``. This is the same
family as the deferred crashed-RUNNING strand (it needs the run-lease + stranded-RUNNING reaper of
the future ops pod), made self-perpetuating by the retry timer. No behavior change here; recorded as
part of that deferred ops-pod carry-forward.

Concurrency: the run-status CAS in the engine (``compare_and_set_run_status``) serializes drivers,
so even with MANY sweepers racing the same due run, exactly one wins the WAITING->RUNNING flip and
executes — exactly-once EXECUTION (not merely exactly-once commit; the model is called once). What
is DEFERRED to a future ops pod is auto-recovery of a driver that crashes mid-drive: the CAS cannot
tell a LIVE RUNNING run from a DEAD one, so that needs a run-lease (owner token + expiry +
heartbeat) plus a stranded-RUNNING reaper. Until then a crashed RUNNING run is recovered by an
explicit ``resume(run_id)`` — the same gap pod 1.0 carried.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from cogworx.substrate.journal import Journal


class Sweeper:
    """Polls the journal for due timers and re-drives their runs through the engine."""

    def __init__(
        self,
        *,
        journal: Journal,
        fire: Callable[[str], Awaitable[Any]],
        clock: Callable[[], datetime],
    ) -> None:
        self._journal = journal
        self._fire = fire
        self._clock = clock

    async def tick(self, now: datetime, *, lease_ttl: timedelta) -> int:
        """Lease the timers due at ``now`` and re-drive each one's run; return the count fired.

        The lease is atomic per timer (single-delivery, even under concurrent sweepers), so each due
        timer is handed to exactly one ``fire`` call. ``fire`` (the engine's ``fire_timer``) replays
        the committed ``Wait`` as an advance — no model re-call (S6).
        """
        claimed = await self._journal.claim_due_timers(now, lease_ttl=lease_ttl)
        for timer in claimed:
            await self._fire(timer.run_id)
        return len(claimed)

    async def run_forever(self, *, interval: timedelta, lease_ttl: timedelta) -> None:
        """Poll ``tick`` every ``interval`` until cancelled (``CancelledError`` propagates)."""
        delay = interval.total_seconds()
        while True:
            await self.tick(self._clock(), lease_ttl=lease_ttl)
            await asyncio.sleep(delay)


__all__ = [
    "Sweeper",
]
