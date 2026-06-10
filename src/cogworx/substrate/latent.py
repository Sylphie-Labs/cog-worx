"""The latent-space seam — pgvector (CANON S3).

The hot/cold latent space. This is the engine-shaped seam for dense recall — not a generic store
that discards pgvector's nearest-neighbour search.

## Contract (Pod 2.2)

- ``put(record)``     — insert-or-replace *content* (embedding + payload). On conflict, never
                        touches use_count / last_used_at / tier / created_at. Idempotent (S6 spirit:
                        a replayed put must not corrupt usage history).
- ``record_use(ids)`` — atomic in-place: use_count += 1, last_used_at = clock(). Batched because
                        the typical caller is context-assembly (a set of matched records enters
                        context together). Returns the count of ids actually touched
                        (typo-detector).
- ``search``          — global exact top-k by cosine, tier-agnostic by default. ``tier=`` scopes
                        the query to one tier. Score is always pure geometry (1 - cosine distance)
                        and is NEVER adjusted for tier — that would be a quality bug (S9: structure,
                        not prompting a priority ranking onto the search scores). For a hot-first
                        composite (hot results first, cold fill), see the documented two-call
                        pattern at Pod 2.6.
- ``sweep_tiers``     — one atomic statement that re-assigns every row to ``hot`` or ``cold``
                        based on the ACT-R activation ranking. Pure function of
                        (snapshot, now, capacity) — idempotent and crash-convergent by
                        construction (S6).

## Why use_count is NOT on LatentRecord (the write model)

``LatentRecord`` is the write/content model. A field the write path ignores is a contract lie, and
a field the write path *uses* forces callers to do a read-modify-write (a race under concurrency).
Usage is store-observed state; content is caller-supplied. They are different operations with
different idempotency semantics (``put`` is idempotent; ``record_use`` is not) and must be separate
methods. Use count / last_used_at appear on ``LatentMatch`` (the read model) where they are honest.

## Search semantics and tier-agnostic default

Hot-first-with-cold-fallback is NOT the default search, because without a similarity threshold +
runner-up margin (sylphie's gate, deferred to Pod 2.6) it would silently rank a 0.30-similarity hot
row above a 0.90-similarity cold row — a latent retrieval quality bug. The default returns the
globally best k results by cosine; tier membership affects WHERE you may SCOPE a search, never WHAT
IS BEST. The hot-first composite is a two-call pattern for callers who also apply a confidence gate.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

Tier = Literal["hot", "cold"]


class LatentRecord(BaseModel):
    """Write/content model — embedding and payload only."""

    model_config = ConfigDict(frozen=True)

    id: str
    embedding: tuple[float, ...]
    payload: dict[str, Any] = Field(default_factory=dict)


class LatentMatch(BaseModel):
    """Read model — content + usage metadata + tier."""

    model_config = ConfigDict(frozen=True)

    record: LatentRecord
    score: float
    tier: Tier
    use_count: int
    last_used_at: datetime


class TierSweepResult(BaseModel):
    """Summary returned by sweep_tiers."""

    model_config = ConfigDict(frozen=True)

    promoted: int
    demoted: int
    hot_size: int


@runtime_checkable
class LatentStore(Protocol):
    """The pgvector latent-space seam (CANON S3)."""

    async def put(self, record: LatentRecord) -> None:
        """Insert or replace content (embedding + payload); never touches usage/tier fields."""
        ...

    async def record_use(self, ids: Sequence[str]) -> int:
        """Atomically increment use_count and advance last_used_at for each id in ids.

        Returns the number of rows actually updated (0 for unknown ids — detects typos).
        """
        ...

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        tier: Tier | None = None,
    ) -> Sequence[LatentMatch]:
        """Return up to k nearest records by cosine similarity, descending.

        ``tier`` scopes the query: ``"hot"`` → hot rows only, ``"cold"`` → cold rows only,
        ``None`` → all rows (default). Score is pure geometry (1 - cosine distance); zero-vector
        queries return an empty sequence.
        """
        ...

    async def sweep_tiers(
        self,
        *,
        now: datetime,
        hot_capacity: int,
    ) -> TierSweepResult:
        """Re-assign every row to hot or cold based on ACT-R activation ranking.

        hot = top-``hot_capacity`` rows by (activation DESC, use_count DESC, last_used_at DESC,
        id ASC). One atomic statement — crash-convergent and idempotent under a fixed ``now``.
        Returns promoted/demoted counts and the resulting hot_size.
        """
        ...


__all__ = [
    "LatentMatch",
    "LatentRecord",
    "LatentStore",
    "Tier",
    "TierSweepResult",
]
