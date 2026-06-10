"""The latent-tier sweeper (CANON S1, S6, Pod 2.2).

The ``LatentTierSweeper`` re-assigns hot/cold tier membership periodically by delegating to
``store.sweep_tiers()``. Like the ``Sweeper`` (durable timers), it owns NO model and does NO
model-class work (S1). Unlike the ``Sweeper``, it requires NO lease and NO cursor: the tier sweep
is a pure function of the current table snapshot — a crash rolls back the implicit transaction and
the table is byte-untouched; re-run converges to the same result (S6).

Degradation (S8): with the sweeper switched off, new records stay ``cold`` (insert default), the
hot set decays stale or empties out, but default search (global exact top-k) is byte-identical —
only tier-scoped callers see the difference. The system degrades, it does not fail.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import datetime, timedelta

from cogworx.substrate.latent import LatentStore, TierSweepResult


class LatentTierSweeper:
    """Polls the latent store and re-assigns hot/cold tiers at each tick."""

    def __init__(
        self,
        *,
        store: LatentStore,
        hot_capacity: int,
        clock: Callable[[], datetime],
    ) -> None:
        self._store = store
        self._hot_capacity = hot_capacity
        self._clock = clock

    async def tick(self) -> TierSweepResult:
        """Re-assign tiers for the current moment; return promotion/demotion counts."""
        return await self._store.sweep_tiers(
            now=self._clock(),
            hot_capacity=self._hot_capacity,
        )

    async def run_forever(self, *, interval: timedelta) -> None:
        """Run tick() every ``interval`` until cancelled (CancelledError propagates)."""
        delay = interval.total_seconds()
        while True:
            await self.tick()
            await asyncio.sleep(delay)


__all__ = ["LatentTierSweeper"]
