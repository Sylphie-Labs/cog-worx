"""Fire-and-forget run handle (CANON S2, S6).

``RunHandle`` is a thin wrapper over the ``asyncio.Task`` that drives a background
``Engine.run()``.  It is **IN-PROCESS ONLY** and carries **NO durability invariant**: the
run itself is fully journaled and resumable via ``Engine.resume`` or the ``Sweeper``
regardless of whether any ``RunHandle`` still exists.  A lost handle means only the
*caller* lost its reference to the background task; the run continues or is recovered
normally through the journal (S6 — the journal is the authority, not the handle).
"""

from __future__ import annotations

import asyncio

from cogworx.substrate.journal import RunState


class RunHandle:
    """In-process reference to a background-running ``Engine.run`` task.

    This handle owns **no durability**: the run is fully journaled (S6).  If the handle
    is discarded, the run is not lost — recover it via ``Engine.resume`` or the sweeper.

    Attributes
    ----------
    run_id:
        The unique run identifier passed to ``Engine.start``.
    """

    def __init__(self, run_id: str, task: asyncio.Task[RunState]) -> None:
        self.run_id: str = run_id
        self._task: asyncio.Task[RunState] = task

    @property
    def done(self) -> bool:
        """``True`` iff the backing asyncio.Task has finished (completed, failed, or cancelled)."""
        return self._task.done()

    async def result(self) -> RunState:
        """Await the backing task and return the ``RunState`` at its first park-or-terminal.

        Propagates any exception the task raised (e.g. ``SimulatedCrash``, ``ResumeError``).
        If the backing task was CANCELLED, this raises ``asyncio.CancelledError`` — which is
        neither a ``RunState`` nor an exception the run raised; inside a ``TaskGroup`` or
        timeout scope the caller's framework may interpret it as the CALLER's cancellation.
        """
        return await self._task


__all__ = ["RunHandle"]
