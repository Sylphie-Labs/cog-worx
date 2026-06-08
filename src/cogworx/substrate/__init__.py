"""Polyglot substrate seams (CANON S3): distinct engine-shaped Protocols, not a generic store.

Neo4j (``GraphStore``), pgvector (``LatentStore``), TimescaleDB (``Journal``) — each kept as its own
typed seam. The only abstraction is this thin internal one for tests/mocks.
"""

from __future__ import annotations

from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal, RunState, StepRecord, Timer
from cogworx.substrate.latent import LatentMatch, LatentRecord, LatentStore

__all__ = [
    "GraphStore",
    "Journal",
    "LatentMatch",
    "LatentRecord",
    "LatentStore",
    "RunState",
    "StepRecord",
    "Timer",
]
