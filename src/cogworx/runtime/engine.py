"""The walking-skeleton loop driver (CANON S1, S2, S6).

The ``Engine`` owns the loop (S2): it drives a ``StageGraph``, committing each stage's
``StageResult`` to the journal BEFORE advancing (S6 exactly-once), and replays a committed step from
the journal WITHOUT re-running the stage or re-calling the model (S6). The write path is a pure
persist — no model touches it (S1). All timestamps come from an injectable ``Clock`` so resume is
replay-safe and tests are reproducible.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from cogworx.capability.registry import Registry
from cogworx.claims.provenance import Artifact
from cogworx.coordination.events import Event, EventType, Subsystem, validate_event_boundary
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.state import RunStatus
from cogworx.model.base import Model
from cogworx.runtime.context import RunContext
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal, RunState, StepRecord
from cogworx.substrate.latent import LatentStore

Clock = Callable[[], datetime]


class ResumeError(Exception):
    """Raised when a run cannot be resumed (unknown run, or no in-process graph held for it)."""


class Engine:
    """The runtime that drives a ``StageGraph`` to a terminal ``RunState``."""

    def __init__(
        self,
        *,
        model: Model,
        journal: Journal,
        graph_store: GraphStore,
        latent: LatentStore,
        budget: BudgetGuard | None = None,
        registry: Registry | None = None,
        event_sink: Callable[[Event], None] | None = None,
        clock: Clock = lambda: datetime.now(UTC),
    ) -> None:
        self._model = model
        self._journal = journal
        self._graph_store = graph_store
        self._latent = latent
        self._budget = budget if budget is not None else BudgetGuard()
        self._registry = registry
        self._event_sink = event_sink
        self._clock = clock
        self._graphs: dict[str, StageGraph] = {}
        self._event_seq = 0

    def _build_context(self, *, run_id: str, session_id: str) -> RunContext:
        return RunContext(
            run_id=run_id,
            session_id=session_id,
            model=self._model,
            journal=self._journal,
            graph_store=self._graph_store,
            latent=self._latent,
            budget=self._budget,
            registry=self._registry,
            event_sink=self._event_sink,
        )

    def _emit(
        self,
        ctx: RunContext,
        event_type: EventType,
        *,
        run_id: str,
        session_id: str,
    ) -> None:
        seq = self._event_seq
        self._event_seq += 1
        event = Event(
            id=f"{run_id}:{event_type.value}:{seq}",
            type=event_type,
            timestamp=self._clock(),
            subsystem=Subsystem.SPINE,
            session_id=session_id,
            run_id=run_id,
        )
        validate_event_boundary(event)
        ctx.emit(event)

    async def run(
        self,
        *,
        run_id: str,
        session_id: str,
        graph: StageGraph,
        initial: Artifact,
    ) -> RunState:
        await self._journal.start_run(run_id, session_id)
        self._graphs[run_id] = graph
        ctx = self._build_context(run_id=run_id, session_id=session_id)
        self._emit(ctx, EventType.RUN_STARTED, run_id=run_id, session_id=session_id)
        return await self._drive(ctx, graph, current=graph.entry)

    async def resume(self, run_id: str) -> RunState:
        state = await self._journal.load_run(run_id)
        if state is None:
            raise ResumeError(f"cannot resume unknown run {run_id!r}")
        if state.status in (
            RunStatus.COMPLETED,
            RunStatus.AWAITING_HUMAN,
            RunStatus.DEGRADED,
        ):
            return state
        graph = self._graphs.get(run_id)
        if graph is None:
            raise ResumeError(
                f"cannot resume run {run_id!r}: no in-process graph held "
                "(cross-process resume is Phase 1)"
            )
        next_stage = self._next_stage(state)
        if next_stage is None:
            return state
        ctx = self._build_context(run_id=run_id, session_id=state.session_id)
        return await self._drive(ctx, graph, current=next_stage)

    @staticmethod
    def _next_stage(state: RunState) -> str | None:
        if not state.steps:
            return None
        result = state.steps[-1].result
        match result.kind:
            case "transition":
                return result.to
            case "degraded":
                return result.to
            case "done" | "await-human":
                return None

    async def _drive(self, ctx: RunContext, graph: StageGraph, *, current: str) -> RunState:
        run_id = ctx.run_id
        session_id = ctx.session_id
        while True:
            step_id = current
            existing = await self._journal.read_step(run_id, step_id)
            if existing is not None:
                result = existing.result
            else:
                self._emit(ctx, EventType.STAGE_ENTERED, run_id=run_id, session_id=session_id)
                stage = graph.get(current)
                result = await stage.run(ctx)
                record = StepRecord(
                    run_id=run_id,
                    step_id=step_id,
                    stage_name=current,
                    result=result,
                    idempotency_key=f"{run_id}:{step_id}",
                    committed_at=self._clock(),
                )
                await self._journal.commit_step(record)
                self._emit(ctx, EventType.STEP_COMMITTED, run_id=run_id, session_id=session_id)

            match result.kind:
                case "done":
                    self._emit(ctx, EventType.RUN_COMPLETED, run_id=run_id, session_id=session_id)
                    break
                case "transition":
                    self._emit(ctx, EventType.STAGE_COMPLETED, run_id=run_id, session_id=session_id)
                    current = result.to
                case "degraded":
                    self._emit(ctx, EventType.STAGE_DEGRADED, run_id=run_id, session_id=session_id)
                    if result.to is None:
                        break
                    current = result.to
                case "await-human":
                    self._emit(
                        ctx,
                        EventType.STAGE_AWAITING_HUMAN,
                        run_id=run_id,
                        session_id=session_id,
                    )
                    break

        final = await self._journal.load_run(run_id)
        if final is None:
            raise ResumeError(f"journal lost run {run_id!r} mid-drive")
        return final


__all__ = [
    "Clock",
    "Engine",
    "ResumeError",
]
