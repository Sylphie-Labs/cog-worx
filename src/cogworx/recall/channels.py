"""Recall channel Protocol + concrete channel implementations (Pod 2.5, CANON S1/S3/S5/S8/S9).

Each channel wraps exactly one substrate engine method (S3). No model calls (S1). The Protocol
drives the recall stack; the concrete classes are the five Pod 2.5 channels.

can_serve() returning False means the stack records 'skipped' and never calls search(). A channel
that can serve but finds nothing returns [] — these are distinct observable states (S8 Lesion Test).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from cogworx.recall.query import RecallQuery
from cogworx.recall.rendering import render_claim, render_episode, render_latent
from cogworx.recall.results import ChannelHit, RecallResult
from cogworx.substrate.entity_kg import EntityKG, ScoredClaim
from cogworx.substrate.episodes import Episode, EpisodeStore
from cogworx.substrate.latent import LatentMatch, LatentStore

__all__ = [
    "ClaimDenseChannel",
    "ClaimGraphChannel",
    "ClaimTextChannel",
    "EpisodeRecencyChannel",
    "LatentDenseChannel",
    "RecallChannel",
]


@runtime_checkable
class RecallChannel(Protocol):
    name: str

    def can_serve(self, query: RecallQuery) -> bool:
        """True if this channel has the required query fields to execute.

        False → stack records status 'skipped' and does NOT call search().
        A channel that can serve but finds nothing returns [] and is 'ok, count=0'.
        These are distinct observable states (important for the S8 lesion test).
        """
        ...

    async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
        """Execute the channel query. Returns results ranked 1-based in ChannelHit.rank."""
        ...


def _validity_filter(
    results: Sequence[ScoredClaim],
    cutoff: datetime,
) -> list[ScoredClaim]:
    """Drop claims expired relative to cutoff (S9: remove, never reweight)."""
    return [r for r in results if r.claim.valid_to is None or r.claim.valid_to > cutoff]


class ClaimDenseChannel:
    """Dense vector recall over the entity KG (S3: Neo4j native vector index)."""

    name: str = "dense.claims"

    def __init__(
        self,
        entity_kg: EntityKG,
        *,
        min_score: float = 0.70,
        include_invalidated: bool = False,
        renderer: Callable[[ScoredClaim], str] = render_claim,
    ) -> None:
        self._kg = entity_kg
        self._min_score = min_score
        self._include_invalidated = include_invalidated
        self._renderer = renderer

    def can_serve(self, query: RecallQuery) -> bool:
        return query.embedding is not None

    async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
        assert query.embedding is not None  # guarded by can_serve
        raw = await self._kg.claims_by_similarity(
            query.embedding,
            k=query.k,
            min_score=self._min_score,
            scope=query.scope,
        )
        if not self._include_invalidated:
            cutoff = query.as_of or datetime.now(UTC)
            filtered: Sequence[ScoredClaim] = _validity_filter(raw, cutoff)
        else:
            filtered = raw
        return tuple(
            RecallResult(
                key=f"claim:{r.claim.id}",
                kind="claim",
                item=r,
                text=self._renderer(r),
                hit=ChannelHit(channel=self.name, rank=i + 1, raw_score=r.similarity),
            )
            for i, r in enumerate(filtered)
        )


class ClaimTextChannel:
    """BM25 full-text recall over the entity KG (S3: Neo4j full-text index)."""

    name: str = "bm25.claims"

    def __init__(
        self,
        entity_kg: EntityKG,
        *,
        include_invalidated: bool = False,
        renderer: Callable[[ScoredClaim], str] = render_claim,
    ) -> None:
        self._kg = entity_kg
        self._include_invalidated = include_invalidated
        self._renderer = renderer

    def can_serve(self, query: RecallQuery) -> bool:
        return query.text is not None and bool(query.text.strip())

    async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
        assert query.text is not None  # guarded by can_serve
        raw = await self._kg.claims_full_text(
            query.text,
            k=query.k,
            scope=query.scope,
            as_of=query.as_of,
        )
        if not self._include_invalidated:
            cutoff = query.as_of or datetime.now(UTC)
            filtered: Sequence[ScoredClaim] = _validity_filter(raw, cutoff)
        else:
            filtered = raw
        return tuple(
            RecallResult(
                key=f"claim:{r.claim.id}",
                kind="claim",
                item=r,
                text=self._renderer(r),
                hit=ChannelHit(channel=self.name, rank=i + 1, raw_score=r.text_score),
            )
            for i, r in enumerate(filtered)
        )


class ClaimGraphChannel:
    """Graph-expansion recall over the entity KG (S3: Neo4j [:HAS_CLAIM]/[:REFERS_TO])."""

    name: str = "graph.claims"

    def __init__(
        self,
        entity_kg: EntityKG,
        *,
        include_invalidated: bool = False,
        renderer: Callable[[ScoredClaim], str] = render_claim,
    ) -> None:
        self._kg = entity_kg
        self._include_invalidated = include_invalidated
        self._renderer = renderer

    def can_serve(self, query: RecallQuery) -> bool:
        return len(query.anchor_entities) > 0

    async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
        # Fan-out: one call per anchor entity, concurrent (S3: graph traversal, not vector).
        per_entity_lists = await asyncio.gather(
            *(
                self._kg.claims_about(
                    entity,
                    limit=query.k,
                    as_of=query.as_of,
                    scope=query.scope,
                )
                for entity in query.anchor_entities
            )
        )
        # Merge with dedup: keep first occurrence by claim id.
        seen: set[str] = set()
        merged: list[ScoredClaim] = []
        for entity_results in per_entity_lists:
            for sc in entity_results:
                if sc.claim.id not in seen:
                    seen.add(sc.claim.id)
                    merged.append(sc)

        if not self._include_invalidated:
            cutoff = query.as_of or datetime.now(UTC)
            merged = _validity_filter(merged, cutoff)

        # Sort: newest ingest_time first, then strongest lineage_min_confidence.
        merged.sort(
            key=lambda sc: (sc.claim.ingest_time, sc.lineage_min_confidence),
            reverse=True,
        )

        return tuple(
            RecallResult(
                key=f"claim:{r.claim.id}",
                kind="claim",
                item=r,
                text=self._renderer(r),
                hit=ChannelHit(channel=self.name, rank=i + 1, raw_score=None),
            )
            for i, r in enumerate(merged)
        )


class EpisodeRecencyChannel:
    """Recency-ordered episode recall (S3: Postgres episodes table)."""

    name: str = "temporal.episodes"

    def __init__(
        self,
        episode_store: EpisodeStore,
        *,
        renderer: Callable[[Episode], str] = render_episode,
    ) -> None:
        self._store = episode_store
        self._renderer = renderer

    def can_serve(self, query: RecallQuery) -> bool:
        return query.session_id is not None

    async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
        assert query.session_id is not None  # guarded by can_serve
        episodes = await self._store.recent_episodes(
            query.session_id,
            limit=query.k,
            before=query.as_of,
        )
        return tuple(
            RecallResult(
                key=f"episode:{ep.episode_id}",
                kind="episode",
                item=ep,
                text=self._renderer(ep),
                hit=ChannelHit(channel=self.name, rank=i + 1, raw_score=None),
            )
            for i, ep in enumerate(episodes)
        )


class LatentDenseChannel:
    """Dense vector recall over the pgvector latent space (S3: pgvector)."""

    name: str = "dense.latent"

    def __init__(
        self,
        latent_store: LatentStore,
        *,
        renderer: Callable[[LatentMatch], str] = render_latent,
    ) -> None:
        self._store = latent_store
        self._renderer = renderer

    def can_serve(self, query: RecallQuery) -> bool:
        return query.embedding is not None

    async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
        assert query.embedding is not None  # guarded by can_serve
        # Tier-agnostic default: no tier= argument (Pod 2.2 contract).
        # record_use is NOT called here — that is Pod 2.6's responsibility at injection time.
        matches = await self._store.search(query.embedding, k=query.k)
        return tuple(
            RecallResult(
                key=f"latent:{m.record.id}",
                kind="latent",
                item=m,
                text=self._renderer(m),
                hit=ChannelHit(channel=self.name, rank=i + 1, raw_score=m.score),
            )
            for i, m in enumerate(matches)
        )
