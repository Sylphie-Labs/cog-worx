from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode
from cogworx.substrate.latent import LatentMatch

__all__ = [
    "AssembledContext",
    "ChannelHit",
    "ContextChunk",
    "FusedResult",
    "RecallResult",
]


class ChannelHit(BaseModel):
    model_config = ConfigDict(frozen=True)

    channel: str  # S5 provenance: which channel surfaced this
    rank: int  # 1-based within-channel rank
    raw_score: float | None  # cosine or BM25 score; None for rank-only channels (graph, temporal)


class RecallResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str  # namespaced: "claim:{id}" | "episode:{id}" | "latent:{id}"
    kind: Literal["claim", "episode", "latent"]
    item: ScoredClaim | Episode | LatentMatch
    text: str  # deterministic per-kind text rendering
    hit: ChannelHit


class FusedResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    kind: Literal["claim", "episode", "latent"]
    item: ScoredClaim | Episode | LatentMatch
    text: str
    hits: tuple[ChannelHit, ...]  # ALL channels that surfaced this result (S5 plural provenance)
    fused_score: float
    fused_rank: int  # 1-based post-fusion rank


class ContextChunk(BaseModel):
    model_config = ConfigDict(frozen=True)

    text: str
    key: str
    kind: Literal["claim", "episode", "latent"]
    hits: tuple[ChannelHit, ...]
    fused_score: float
    relevance_rank: int  # pre-fold rank; preserved for Pod 2.6


class AssembledContext(BaseModel):
    model_config = ConfigDict(frozen=True)

    chunks: tuple[ContextChunk, ...]
    token_count: int
    budget: int
    dropped: int  # items skipped because they individually exceeded budget
