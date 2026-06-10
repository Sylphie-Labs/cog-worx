"""The episode projector — committed journal steps to cogworx_episodes, off-write-path (S1/S6).

Sweeper-shaped: no model, no hot-path commit. Each tick polls the TimescaleDB journal for newly
committed steps, parses turn stamps via turns_of(), and batch-inserts episodes into cogworx_episodes
via EpisodeStore.project_episodes() — advancing the cursor IN THE SAME PG TXN as the inserts.

Architecture mirrors TrialProjector (runtime/projector.py): same fenced cursor read, same
cursor-in-destination-txn atomicity, same fail-loud on malformed stamps, same session cache.

CURSOR MODEL (COMMIT-ORDER, SINGLE FENCED READ):
  The cursor (ProjectionCursor) is advanced in the SAME project_episodes() PG txn as the episode
  INSERTs (S6). Each tick does ONE bounded, visibility-fenced journal read:
    * committed_steps_after(cursor, limit=batch_limit) — strictly past the cursor, oldest-first,
      capped at batch_limit, visibility-fenced on commit_xid (never consumes a row an in-flight txn
      could still commit beneath). The cursor always advances to the LAST SCANNED ROW, even if no
      turns were stamped in that batch (zero-episode control batches advance the cursor: P0-2).

TURN STAMPING (S9 — structure over self-report):
  A stage stamps output.data["turns"] via stamp_turns(). The projector reads this stamp via
  turns_of() — authoritative. A step with no "turns" key is silently skipped (ordinary control
  flow). A step whose "turns" key is present but malformed RAISES ValueError — the projector stops
  dead (fail-loud, exactly as resolve_outcome raises on bad stamps in TrialProjector) so a
  programmer error is never silently swallowed into a missing episode.

CORRECTNESS MODEL:
  * project_episodes() is ON CONFLICT (episode_id) DO NOTHING — idempotent first-write-wins (S6):
    a re-projection after a crash produces the same episode_ids and is a harmless no-op.
  * The cursor advances IN THE SAME PG TXN as the inserts, so a crash cannot leave the cursor
    ahead of unwritten episodes (exactly-once into cogworx_episodes, mirroring TrialProjector).
  * occurred_at = step.committed_at — the only replay-stable timestamp; never wall-clock from Turn
    (S6).
  * episode_id = f"{run_id}:{step_index}:{turn_index}" (0-based turn_index, deterministic).
"""

from __future__ import annotations

from collections.abc import Sequence

from cogworx.knowledge.episodes import turns_of
from cogworx.substrate.episodes import Episode, EpisodeStore
from cogworx.substrate.journal import Journal, ProjectedStep, ProjectionCursor

__all__ = ["DEFAULT_EPISODE_CONSUMER", "EpisodeProjector"]

DEFAULT_EPISODE_CONSUMER = "episodic/episode-projector"
"""The default cursor consumer key.

Each projection consumer owns its own cursor row in ``cogworx_projection_cursors`` so the journal
table stays pure (S3) and a future projector reads from its own watermark without interfering.
"""


class EpisodeProjector:
    """Polls the journal for committed steps and projects each turn-stamped step into episodes.

    Sweeper-shaped (S1): no model, no hot-path commit. Each :meth:`tick` does ONE bounded,
    visibility-fenced journal read, parses turn stamps via
    :func:`~cogworx.knowledge.episodes.turns_of`, and calls
    :meth:`~cogworx.substrate.episodes.EpisodeStore.project_episodes` once — advancing the cursor
    IN THE SAME PG TXN as the episode INSERTs (S6, exactly-once).

    A step with no ``"turns"`` key is silently skipped. A step with a present but malformed
    ``"turns"`` key raises :class:`ValueError` and stalls the projector (fail-loud).
    """

    def __init__(
        self,
        *,
        journal: Journal,
        episode_store: EpisodeStore,
        consumer: str = DEFAULT_EPISODE_CONSUMER,
        batch_limit: int = 256,
    ) -> None:
        self._journal = journal
        self._episode_store = episode_store
        self._consumer = consumer
        self._batch_limit = batch_limit
        # Resolve run -> session_id once per tick; a run's session_id is immutable for its life.
        self._session_cache: dict[str, str | None] = {}

    async def tick(self) -> int:
        """Project one batch; return the episode rows inserted.

        ONE bounded journal read per tick: ``committed_steps_after(cursor, limit=batch_limit)`` —
        the STRICT keyset read STRICTLY past the cursor, capped at ``batch_limit``,
        visibility-fenced on ``commit_xid``. Then in ONE ``project_episodes`` PG txn: insert the
        staged episodes and advance the cursor to the LAST SCANNED ROW (always — even zero-episode
        control batches advance the cursor to avoid rescanning them). An idle tick (no rows at all)
        returns 0 and leaves the cursor untouched.

        Raises:
            ValueError: if any scanned step has a present but malformed ``"turns"`` stamp. The
                cursor is NOT advanced — the projector stalls at that row until the stamp is fixed
                (fail-loud, exactly as TrialProjector's resolve_outcome raises on bad stamps).
        """
        self._session_cache.clear()
        cursor = await self._episode_store.read_cursor(self._consumer)

        scanned = tuple(await self._journal.committed_steps_after(cursor, limit=self._batch_limit))
        if not scanned:
            return 0

        episodes = await self._stage(scanned)

        new_cursor = _cursor_of(scanned[-1])
        await self._episode_store.project_episodes(self._consumer, episodes, new_cursor)
        return len(episodes)

    async def _stage(self, scanned: Sequence[ProjectedStep]) -> list[Episode]:
        episodes: list[Episode] = []
        for projected in scanned:
            step = projected.record
            session_id = await self._session_for(step.run_id)
            # turns_of raises ValueError on malformed stamp (fail-loud); propagate.
            turns = turns_of(step)
            for i, turn in enumerate(turns):
                episodes.append(
                    Episode(
                        episode_id=f"{step.run_id}:{step.step_index}:{i}",
                        run_id=step.run_id,
                        step_index=step.step_index,
                        turn_index=i,
                        session_id=session_id or "",
                        role=turn.role,
                        content=turn.content,
                        kind=turn.kind,
                        occurred_at=step.committed_at,
                    )
                )
        return episodes

    async def _session_for(self, run_id: str) -> str | None:
        if run_id not in self._session_cache:
            run = await self._journal.load_run(run_id)
            self._session_cache[run_id] = run.session_id if run is not None else None
        return self._session_cache[run_id]


def _cursor_of(projected: ProjectedStep) -> ProjectionCursor:
    """The cursor ``(commit_ordinal, run_id, step_index)`` of a scanned step."""
    return ProjectionCursor(
        commit_ordinal=projected.commit_ordinal,
        run_id=projected.record.run_id,
        step_index=projected.record.step_index,
    )
