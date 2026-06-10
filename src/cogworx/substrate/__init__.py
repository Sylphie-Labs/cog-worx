"""Polyglot substrate seams (CANON S3): distinct engine-shaped Protocols, not a generic store.

Neo4j (``GraphStore``, ``EntityKG``), pgvector (``LatentStore``), TimescaleDB (``Journal``) — each
kept as its own typed seam. The only abstraction is this thin internal one for tests/mocks.
"""

from __future__ import annotations

from cogworx.substrate.entity_kg import ClaimProjection, EntityKG, ScoredClaim
from cogworx.substrate.episodes import Episode, EpisodeStore
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import (
    Journal,
    ProjectedStep,
    ProjectionCursor,
    RunState,
    StepRecord,
    Timer,
)
from cogworx.substrate.latent import LatentMatch, LatentRecord, LatentStore
from cogworx.substrate.procedural_kg import (
    CursorAdvance,
    Outcome,
    ProblemType,
    ProceduralKG,
    Procedure,
    ScoredProcedure,
    Trial,
    TrialWrite,
)

__all__ = [
    "ClaimProjection",
    "CursorAdvance",
    "EntityKG",
    "Episode",
    "EpisodeStore",
    "GraphStore",
    "Journal",
    "LatentMatch",
    "LatentRecord",
    "LatentStore",
    "Outcome",
    "ProblemType",
    "ProceduralKG",
    "Procedure",
    "ProjectedStep",
    "ProjectionCursor",
    "RunState",
    "ScoredClaim",
    "ScoredProcedure",
    "StepRecord",
    "Timer",
    "Trial",
    "TrialWrite",
]
