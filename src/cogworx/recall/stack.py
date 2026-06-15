"""RecallStack — fan-out over recall channels, RRF fusion, reranker seam (CANON S1/S5/S8/S9).

No model calls here (S1). The Reranker may be model-bearing; that posture is declared structurally
via ``reranker.model_bearing``. A channel that raises degrades to ``failed`` status with zero
results — the other channels still contribute (S8). Every FusedResult in RecallOutcome.results
has >=1 ChannelHit (S5), verified by assert_rerank_subset + the upstream fuse() guarantee.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from cogworx.recall.channels import (
    ClaimDenseChannel,
    ClaimGraphChannel,
    ClaimTextChannel,
    EpisodeRecencyChannel,
    LatentDenseChannel,
    RecallChannel,
)
from cogworx.recall.fusion import RRF_K_DEFAULT, assert_rerank_subset, fuse
from cogworx.recall.query import RecallQuery
from cogworx.recall.rerank import NoopReranker, Reranker
from cogworx.recall.results import FusedResult, RecallResult
from cogworx.substrate.entity_kg import EntityKG
from cogworx.substrate.episodes import EpisodeStore
from cogworx.substrate.latent import LatentStore

__all__ = [
    "ChannelStatus",
    "RecallOutcome",
    "RecallStack",
    "default_recall_stack",
]


class ChannelStatus(BaseModel):
    """Status summary for one channel after a recall() call."""

    model_config = ConfigDict(frozen=True)

    channel: str
    state: Literal["ok", "skipped", "failed"]
    count: int
    error: str | None = None  # repr-only, for diagnostics; never a control signal


class RecallOutcome(BaseModel):
    """The complete output of RecallStack.recall()."""

    model_config = ConfigDict(frozen=True)

    results: tuple[FusedResult, ...]
    channel_status: tuple[ChannelStatus, ...]


class RecallStack:
    """Fan-out recall over N channels, fuse with RRF, rerank via an injected seam.

    Construction validates that all channel names are unique — provenance labels must be
    unambiguous per S5. The reranker is optional; ``NoopReranker`` is used when not supplied.
    """

    def __init__(
        self,
        channels: Sequence[RecallChannel],
        *,
        reranker: Reranker | None = None,
        k_rrf: int = RRF_K_DEFAULT,
    ) -> None:
        # Reject duplicate channel names at construction — provenance labels must be unique (S5).
        names = [ch.name for ch in channels]
        if len(names) != len(set(names)):
            dupes = [n for n in names if names.count(n) > 1]
            raise ValueError(f"Duplicate channel names: {sorted(set(dupes))}")
        self._channels = list(channels)
        self._reranker: Reranker = reranker if reranker is not None else NoopReranker()
        self._k_rrf = k_rrf

    async def recall(self, query: RecallQuery) -> RecallOutcome:
        """Execute all channels concurrently, fuse, rerank, and return a RecallOutcome.

        Per S8: a channel that raises is recorded as ``failed`` — it contributes zero results
        but never crashes the stack. The ``error`` field on ChannelStatus carries the repr of
        the exception for diagnostics; it is never used as a control signal.

        Per S9: the reranker output is validated by assert_rerank_subset before it is returned —
        a reranker that injects foreign keys or mutates items raises ValueError immediately.

        Per S5: every FusedResult in the outcome has at least one ChannelHit (enforced by fuse()).
        """

        async def _run_channel(
            ch: RecallChannel,
        ) -> tuple[str, list[RecallResult], ChannelStatus]:
            if not ch.can_serve(query):
                return (
                    ch.name,
                    [],
                    ChannelStatus(channel=ch.name, state="skipped", count=0),
                )
            try:
                results = list(await ch.search(query))
                return (
                    ch.name,
                    results,
                    ChannelStatus(channel=ch.name, state="ok", count=len(results)),
                )
            except Exception as exc:
                return (
                    ch.name,
                    [],
                    ChannelStatus(channel=ch.name, state="failed", count=0, error=repr(exc)),
                )

        outcomes = await asyncio.gather(*[_run_channel(ch) for ch in self._channels])

        channel_results: dict[str, list[RecallResult]] = {
            name: results for name, results, _ in outcomes
        }
        statuses: list[ChannelStatus] = [status for _, _, status in outcomes]

        fused = fuse(channel_results, k_rrf=self._k_rrf)

        reranked_seq = await self._reranker.rerank(query, fused)
        reranked = list(reranked_seq)
        assert_rerank_subset(fused, reranked)  # raises ValueError on invalid output (S9)

        return RecallOutcome(
            results=tuple(reranked),
            channel_status=tuple(statuses),
        )


_FACTORY_K_RRF: Final[int] = RRF_K_DEFAULT


def default_recall_stack(
    *,
    entity_kg: EntityKG,
    episode_store: EpisodeStore | None = None,
    latent_store: LatentStore | None = None,
    reranker: Reranker | None = None,
    k_rrf: int = _FACTORY_K_RRF,
    latent_hot_first: bool = False,
    latent_min_similarity: float = 0.80,
) -> RecallStack:
    """Build the standard 5-channel recall stack; omit channels whose store is None (S8 lesion).

    The 3 entity-KG channels (dense, BM25, graph) are always included — they require only
    the entity_kg argument.  The episode and latent channels are included only when their
    respective stores are supplied, so the stack degrades gracefully when those substrates
    are not available (S8).

    Args:
        entity_kg: The entity-KG substrate (always required).
        episode_store: When supplied, adds
            :class:`~cogworx.recall.channels.EpisodeRecencyChannel`.
        latent_store: When supplied, adds
            :class:`~cogworx.recall.channels.LatentDenseChannel`.
        reranker: Optional reranker; defaults to :class:`~cogworx.recall.rerank.NoopReranker`.
        k_rrf: RRF constant; defaults to :data:`~cogworx.recall.fusion.RRF_K_DEFAULT`.
        latent_hot_first: Passed to :class:`~cogworx.recall.channels.LatentDenseChannel` as
            ``hot_first``.  When ``True``, the channel applies the similarity-gated hot-first
            composite (Pod 2.6).  Default ``False`` (tier-agnostic, S8/S12).
        latent_min_similarity: Passed to :class:`~cogworx.recall.channels.LatentDenseChannel`
            as ``min_similarity`` (τ threshold for the hot-first gate).  Default ``0.80``.

    Returns:
        A fully configured :class:`RecallStack`.
    """
    channels: list[RecallChannel] = [
        ClaimDenseChannel(entity_kg),
        ClaimTextChannel(entity_kg),
        ClaimGraphChannel(entity_kg),
    ]
    if episode_store is not None:
        channels.append(EpisodeRecencyChannel(episode_store))
    if latent_store is not None:
        channels.append(
            LatentDenseChannel(
                latent_store,
                hot_first=latent_hot_first,
                min_similarity=latent_min_similarity,
            )
        )
    return RecallStack(channels, reranker=reranker, k_rrf=k_rrf)
