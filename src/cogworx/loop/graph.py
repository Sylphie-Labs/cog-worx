"""The Spine graph + Loop runtime contract (CANON S2, S6).

The Spine is a graph of stages whose structure is validated at construction: unique names, no
dangling edges, full reachability from the entry, and termination by construction (at least one
reachable terminal stage). The ``Loop`` Protocol is the runtime contract the engine implements; on
``resume`` it reads journaled outputs and never re-calls the model (S6).
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from cogworx.claims.provenance import Artifact
from cogworx.loop.stage import Stage
from cogworx.substrate.journal import RunState


class StageGraphError(Exception):
    """Raised when a ``StageGraph`` would violate a structural invariant."""


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
            for target in stage.transitions:
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
            for target in by_name[name].transitions:
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
        return self.get(name).transitions

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
