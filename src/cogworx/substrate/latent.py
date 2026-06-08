"""The latent-space seam — pgvector (CANON S3).

The hot/cold latent space. This is the engine-shaped seam for dense recall — not a generic store
that discards pgvector's nearest-neighbour search.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class LatentRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    embedding: tuple[float, ...]
    payload: dict[str, Any] = Field(default_factory=dict)
    use_count: int = 0


class LatentMatch(BaseModel):
    model_config = ConfigDict(frozen=True)

    record: LatentRecord
    score: float


@runtime_checkable
class LatentStore(Protocol):
    """The pgvector latent-space seam."""

    async def upsert(self, record: LatentRecord) -> None: ...

    async def search(self, embedding: Sequence[float], *, k: int = 10) -> Sequence[LatentMatch]: ...


__all__ = [
    "LatentMatch",
    "LatentRecord",
    "LatentStore",
]
