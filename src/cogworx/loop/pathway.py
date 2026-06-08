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

import hashlib
from dataclasses import dataclass

from cogworx.loop.graph import StageGraph

_FINGERPRINT_HEX_LEN = 16


def pathway_fingerprint(graph: StageGraph) -> str:
    """A deterministic hash of a ``StageGraph``'s STRUCTURE (CANON S6 — resume integrity).

    Canonicalises the graph as ``entry`` plus, for every stage sorted by name, the stage's name and
    its transitions (themselves sorted), then ``sha256``-es that canonical string and returns the
    leading hex digits. Deterministic across processes — no clock, no ``random``, no object identity
    — so a cold resume in a fresh process can compare the rehydrated graph's fingerprint against the
    one stored at ``start_run`` time.

    What it CATCHES: STRUCTURAL divergence under a resumed run — a stage added, removed, renamed, or
    rewired (transitions changed) — even when the ``(pathway_id, version)`` pointer is unchanged
    (an in-place edit that forgot to bump the version).

    What it does NOT catch: PURELY BEHAVIOURAL changes — same stages, same transitions, but a
    stage's ``run()`` logic was edited. The structure is byte-identical, so the fingerprint matches.
    Those are the responsibility of the version-bump convention (bump ``pathway_version`` when stage
    behaviour changes), not the fingerprint. The fingerprint is a structural backstop, not a
    behaviour oracle.
    """
    parts: list[str] = [f"entry={graph.entry}"]
    for name in sorted(graph.names()):
        transitions = ",".join(sorted(graph.transitions_from(name)))
        parts.append(f"{name}->[{transitions}]")
    canonical = "|".join(parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:_FINGERPRINT_HEX_LEN]


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
    "pathway_fingerprint",
]
