"""The pathway registry — code-authored, named, versioned pathways (CANON S2, S6).

This is how "config over code for the loop" is realized without handing loop control to the model or
a managed runtime: a pathway is a code-authored ``StageGraph`` (Stage classes wired by the
developer) registered under a named, versioned id. A run stores ONLY the ``pathway_id`` +
``pathway_version`` pointer in its journal record (see ``RunState``), so a cold cross-process resume
rehydrates the graph from this registry rather than carrying an in-process graph object (S6 durable
resume, S2 own-the-loop — no agent SDK, no managed durable-execution dependency). Cold resume thus
works as long as the SAME pathways are registered at process startup; an unregistered pathway is an
honest, surfaced error.
"""

from __future__ import annotations

from dataclasses import dataclass

from cogworx.loop.graph import StageGraph


class PathwayError(Exception):
    """Raised on a duplicate registration or a lookup miss (cold resume needs the pathway)."""


@dataclass(frozen=True)
class Pathway:
    id: str
    version: int
    graph: StageGraph


class PathwayRegistry:
    """A registry of named, versioned ``StageGraph`` pathways keyed by ``(pathway_id, version)``."""

    def __init__(self) -> None:
        self._pathways: dict[tuple[str, int], StageGraph] = {}

    def register(self, pathway_id: str, graph: StageGraph, *, version: int = 1) -> None:
        key = (pathway_id, version)
        if key in self._pathways:
            raise PathwayError(f"pathway {pathway_id!r} version {version} is already registered")
        self._pathways[key] = graph

    def get(self, pathway_id: str, version: int = 1) -> StageGraph:
        graph = self._pathways.get((pathway_id, version))
        if graph is None:
            raise PathwayError(
                f"pathway {pathway_id!r} version {version} is not registered; cold resume needs "
                "the pathway registered in this process (register the same pathways at startup)"
            )
        return graph

    def has(self, pathway_id: str, version: int = 1) -> bool:
        return (pathway_id, version) in self._pathways


__all__ = [
    "Pathway",
    "PathwayError",
    "PathwayRegistry",
]
