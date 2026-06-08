"""Concrete ``StageContext`` handed to every stage (CANON S3, S7, S8).

``RunContext`` carries the distinct substrate seams (S3 — no flattening), enforces the event
boundary on every ``emit`` (S7), and routes capability dispatch through the registry where ``get``
raises on a disabled capability (the S8 lesion path).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

from cogworx.capability.registry import Registry
from cogworx.coordination.events import Event, validate_event_boundary
from cogworx.cost.budget import BudgetGuard
from cogworx.model.base import Model
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal
from cogworx.substrate.latent import LatentStore


class DispatchError(Exception):
    """Raised when a capability is dispatched with no registry wired into the context."""


class RunContext:
    """Everything a stage is handed by the engine for one run."""

    def __init__(
        self,
        *,
        run_id: str,
        session_id: str,
        model: Model,
        journal: Journal,
        graph_store: GraphStore,
        latent: LatentStore,
        budget: BudgetGuard,
        registry: Registry | None = None,
        event_sink: Callable[[Event], None] | None = None,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self._model = model
        self._journal = journal
        self._graph = graph_store
        self._latent = latent
        self._budget = budget
        self._registry = registry
        self._event_sink = event_sink
        self._events: list[Event] = []

    @property
    def budget(self) -> BudgetGuard:
        return self._budget

    @property
    def model(self) -> Model:
        return self._model

    @property
    def journal(self) -> Journal:
        return self._journal

    @property
    def graph(self) -> GraphStore:
        return self._graph

    @property
    def latent(self) -> LatentStore:
        return self._latent

    def emit(self, event: Event) -> None:
        validate_event_boundary(event)
        self._events.append(event)
        if self._event_sink is not None:
            self._event_sink(event)

    async def dispatch(self, capability: str, args: Mapping[str, Any]) -> Any:
        if self._registry is None:
            raise DispatchError(
                f"cannot dispatch {capability!r}: no capability registry wired into this run"
            )
        cap = self._registry.get(capability)
        return await cap.invoke(args)

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)


__all__ = [
    "DispatchError",
    "RunContext",
]
