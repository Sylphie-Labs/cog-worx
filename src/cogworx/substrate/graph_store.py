"""The graph knowledge seam — Neo4j (CANON S3).

The graph knowledge layer (procedural/entity KGs, world model, user model). This is the
engine-shaped seam for graph recall — not a generic store that discards Neo4j's graph power.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cogworx.claims.provenance import Claim


@runtime_checkable
class GraphStore(Protocol):
    """The Neo4j graph-knowledge seam."""

    async def upsert_claim(self, claim: Claim) -> str: ...

    async def get_claim(self, claim_id: str) -> Claim | None: ...

    async def neighbors(self, claim_id: str, *, limit: int = 20) -> Sequence[Claim]: ...


__all__ = [
    "GraphStore",
]
