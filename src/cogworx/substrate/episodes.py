"""Episode store seam — Postgres (CANON S3, Pod 2.3).

The durable, time-ordered episode store. Episodes are projections of committed journal steps
(a stage stamps output.data["turns"]; the EpisodeProjector materializes them here off-path).
Rows are immutable — the cursor is the progress state, not per-row flags (S3: table stays pure).

## Contract

- ``project_episodes(consumer, episodes, progress)`` — insert episodes AND advance cursor in ONE
  atomic transaction. ON CONFLICT (episode_id) DO NOTHING — idempotent first-write-wins (S6).
- ``read_cursor(consumer)`` — return the last-scanned cursor for this consumer, or None on cold
  start.
- ``episodes_for_session(session_id)`` — return episodes for a session, ordered by
  (step_index, turn_index). The time-ordered view for context assembly and recall.
- ``get_episode(episode_id)`` — return an episode by id, or None.

## Why rows are immutable and the cursor is the progress state

Episodes are projections of committed journal steps — they inherit the journal's exactly-once
guarantee. A re-projection of the same step would produce the same episode_id and be silently
dropped by the ON CONFLICT DO NOTHING. Per-row status columns (a "dirty" bit, a "processed" flag)
would be write state on an immutable projection record — a category error. The cursor IS the
progress state.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cogworx.substrate.journal import ProjectionCursor


class Episode(BaseModel):
    """An immutable projection of one conversational turn from a committed journal step."""

    model_config = ConfigDict(frozen=True)

    episode_id: str
    """Framework-assigned: ``f"{run_id}:{step_index}:{turn_index}"``."""
    run_id: str
    step_index: int
    turn_index: int
    session_id: str
    role: str
    """``"user"`` | ``"assistant"`` | ``"system"`` | ``"tool"``"""
    content: str
    kind: str
    """An :data:`~cogworx.knowledge.episodes.EpisodeKind` value."""
    occurred_at: datetime
    """The step's ``committed_at`` — the only replay-stable timestamp (S6)."""


@runtime_checkable
class EpisodeStore(Protocol):
    """The Postgres episode-store seam (CANON S3, Pod 2.3)."""

    async def project_episodes(
        self,
        consumer: str,
        episodes: Sequence[Episode],
        progress: ProjectionCursor,
    ) -> None:
        """Insert episodes + advance cursor in ONE atomic transaction.

        ON CONFLICT (episode_id) DO NOTHING — idempotent first-write-wins (S6). An empty
        ``episodes`` sequence still advances the cursor (zero-turn control batch).
        """
        ...

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        """Return the last-scanned cursor for this consumer, or None on cold start."""
        ...

    async def episodes_for_session(
        self,
        session_id: str,
        *,
        limit: int = 100,
    ) -> Sequence[Episode]:
        """Return episodes for a session, ordered by (step_index, turn_index) ASC."""
        ...

    async def get_episode(self, episode_id: str) -> Episode | None:
        """Return an episode by id, or None."""
        ...

    async def recent_episodes(
        self,
        session_id: str,
        *,
        limit: int = 20,
        before: datetime | None = None,
    ) -> Sequence[Episode]:
        """Return the most-recent episodes for a session, newest-first.

        Ordered by (step_index, turn_index) DESC.
        before: restricts to episodes with occurred_at < before (exclusive).
        """
        ...


__all__ = ["Episode", "EpisodeStore"]
