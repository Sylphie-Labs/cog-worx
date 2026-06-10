from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cogworx.recall.query import RecallQuery
from cogworx.recall.results import FusedResult

__all__ = ["NoopReranker", "Reranker"]


@runtime_checkable
class Reranker(Protocol):
    model_bearing: bool  # True if this reranker calls a model (S1: posture declared structurally)

    async def rerank(
        self, query: RecallQuery, results: Sequence[FusedResult]
    ) -> Sequence[FusedResult]: ...


class NoopReranker:
    model_bearing: bool = False

    async def rerank(
        self, query: RecallQuery, results: Sequence[FusedResult]
    ) -> Sequence[FusedResult]:
        return list(results)
