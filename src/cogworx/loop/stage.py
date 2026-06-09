"""Stage composition primitives (CANON S2, S8).

The Spine is a graph of stages. A ``Stage`` runs against a ``StageContext`` and returns a
discriminated ``StageResult`` — ``transition`` / ``done`` / ``await-human`` / ``degraded`` (defined
in :mod:`cogworx.loop.result`) — making HITL and graceful degradation first-class loop transitions.
Ported from biz-firm's composition primitives (Stage · Capability · Context · Loop).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import datetime
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from cogworx.claims.provenance import Artifact
from cogworx.coordination.events import Event
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.result import StageResult
from cogworx.model.base import Model

if TYPE_CHECKING:
    # Annotation-only (Protocol property return types). A runtime import here is the edge of an
    # import cycle: substrate.journal -> loop.result -> loop/__init__ -> stage -> substrate.journal,
    # which detonates when cogworx.substrate (or cogworx.runtime) is the first package imported.
    from cogworx.substrate.graph_store import GraphStore
    from cogworx.substrate.journal import Journal
    from cogworx.substrate.latent import LatentStore


@runtime_checkable
class StageContext(Protocol):
    """Everything a stage is handed by the runner. Substrate seams stay distinct (S3)."""

    run_id: str
    session_id: str

    @property
    def budget(self) -> BudgetGuard: ...

    @property
    def model(self) -> Model: ...

    @property
    def journal(self) -> Journal: ...

    @property
    def graph(self) -> GraphStore: ...

    @property
    def latent(self) -> LatentStore: ...

    @property
    def clock(self) -> Callable[[], datetime]: ...

    def emit(self, event: Event) -> None: ...

    async def dispatch(self, capability: str, args: Mapping[str, Any]) -> Any: ...

    async def read_human_input(self, step_index: int) -> Artifact | None:
        """Return the HITL answer committed at ``step_index`` for this run, or ``None`` if absent.

        PULL-based: stages pull answers from the journal. The engine never pushes a
        ``ctx.human_input`` attribute, so cold resume works without re-injection (S6/S9).
        """
        ...


@runtime_checkable
class Stage(Protocol):
    """One node in the Spine's graph of stages."""

    name: str
    transitions: tuple[str, ...]

    async def run(self, ctx: StageContext) -> StageResult: ...


__all__ = [
    "Stage",
    "StageContext",
]
