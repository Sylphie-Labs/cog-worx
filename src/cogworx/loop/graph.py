"""The Spine graph + Loop runtime contract (CANON S2, S6).

The Spine is a graph of stages whose structure is validated at construction: unique names, no
dangling edges, full reachability from the entry, and termination by construction (at least one
reachable terminal stage). The ``Loop`` Protocol is the runtime contract the engine implements; on
``resume`` it reads journaled outputs and never re-calls the model (S6).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from cogworx.claims.provenance import Artifact
from cogworx.loop.retry import RetryPolicy
from cogworx.loop.stage import Stage

if TYPE_CHECKING:
    # Annotation-only (``Loop`` Protocol signatures). A runtime import here is the edge of an
    # import cycle: substrate.journal -> loop.result -> loop/__init__ -> graph -> substrate.journal
    # (same shape as the stage.py edge — see the note there).
    from cogworx.substrate.journal import RunState


class StageGraphError(Exception):
    """Raised when a ``StageGraph`` would violate a structural invariant."""


def _edges(stage: Stage) -> tuple[str, ...]:
    """Every static edge out of a stage: its declared ``transitions`` PLUS the implicit
    retry-exhaustion edge ``retry_policy.exhausted_to`` (S8 degraded-onward routes there). Treating
    exhaustion as a real edge keeps the dangling-edge + reachability + termination checks honest for
    a stage whose only path to its fallback is retry exhaustion.

    This is the SINGLE source of truth for a stage's structural edge set, consumed by both graph
    validation (here) and the resume fingerprint (``StageGraph.edges_from`` ->
    ``pathway_fingerprint``) so an ``exhausted_to`` edit can never be a structural blind spot (S6).
    """
    policy: RetryPolicy | None = getattr(stage, "retry_policy", None)
    if policy is not None and policy.exhausted_to is not None:
        return (*stage.transitions, policy.exhausted_to)
    return stage.transitions


class StageGraph:
    """A validated graph of stages: structure and termination hold by construction."""

    def __init__(self, stages: Sequence[Stage], *, entry: str) -> None:
        if not stages:
            raise StageGraphError("a stage graph needs at least one stage")

        by_name: dict[str, Stage] = {}
        for stage in stages:
            if stage.name in by_name:
                raise StageGraphError(f"duplicate stage name {stage.name!r}")
            by_name[stage.name] = stage

        if entry not in by_name:
            raise StageGraphError(f"entry {entry!r} is not a known stage")

        for stage in stages:
            for target in _edges(stage):
                if target not in by_name:
                    raise StageGraphError(
                        f"stage {stage.name!r} transitions to unknown stage {target!r}"
                    )

        reachable = self._reachable_from(entry, by_name)
        unreachable = set(by_name) - reachable
        if unreachable:
            raise StageGraphError(f"unreachable stages from {entry!r}: {sorted(unreachable)}")

        if not any(by_name[name].transitions == () for name in reachable):
            raise StageGraphError(
                f"no terminal stage reachable from {entry!r}: the graph can never end"
            )

        self._stages = by_name
        self._entry = entry

    @staticmethod
    def _reachable_from(entry: str, by_name: dict[str, Stage]) -> set[str]:
        seen: set[str] = {entry}
        queue: deque[str] = deque([entry])
        while queue:
            name = queue.popleft()
            for target in _edges(by_name[name]):
                if target not in seen:
                    seen.add(target)
                    queue.append(target)
        return seen

    @property
    def entry(self) -> str:
        return self._entry

    def names(self) -> tuple[str, ...]:
        return tuple(self._stages)

    def get(self, name: str) -> Stage:
        if name not in self._stages:
            raise StageGraphError(f"unknown stage {name!r}")
        return self._stages[name]

    def transitions_from(self, name: str) -> tuple[str, ...]:
        """The stage's DECLARED transitions only (the ``Done``/``Transition`` routing edges)."""
        return self.get(name).transitions

    def edges_from(self, name: str) -> tuple[str, ...]:
        """The stage's FULL structural edge set: declared transitions PLUS the retry-exhaustion edge
        (``retry_policy.exhausted_to``). This is the same edge set graph validation/reachability
        trusts (``_edges``) and is what the resume fingerprint canonicalises over — so an in-place
        ``exhausted_to`` edit changes the structural fingerprint (S6), unlike ``transitions_from``.
        """
        return _edges(self.get(name))

    def is_terminal(self, name: str) -> bool:
        return self.get(name).transitions == ()


@runtime_checkable
class Loop(Protocol):
    """The runtime contract the engine implements to drive a ``StageGraph``.

    ``resume`` reads journaled step outputs and never re-calls the model (S6): a step that committed
    before a crash is replayed from the journal, not recomputed.
    """

    async def run(
        self,
        *,
        run_id: str,
        session_id: str,
        pathway_id: str,
        initial: Artifact,
        pathway_version: int = 1,
    ) -> RunState: ...

    async def resume(self, run_id: str) -> RunState: ...


__all__ = [
    "Loop",
    "StageGraph",
    "StageGraphError",
]
