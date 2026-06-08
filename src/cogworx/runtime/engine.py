"""The walking-skeleton loop driver (CANON S1, S2, S6).

The ``Engine`` owns the loop (S2): it drives a ``StageGraph``, committing each stage's
``StageResult`` to the journal BEFORE advancing (S6 exactly-once), and replays a committed step from
the journal WITHOUT re-running the stage or re-calling the model (S6). The write path is a pure
persist — no model touches it (S1). All timestamps come from an injectable ``Clock`` so resume is
replay-safe and tests are reproducible.

Steps are keyed POSITIONALLY (``step_index``): each executed or replayed position consumes one
index, so a cyclic pathway that revisits a stage commits a distinct step per visit. A structural
step ceiling (``max_steps``) FAILS a run that would loop forever — it pairs with ``BudgetGuard`` so
cycles are bounded by construction (S9/S11). Cold cross-process resume rehydrates the graph from the
``PathwayRegistry`` using the run's stored ``pathway_id`` pointer (S2/S6) — no in-process graph.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime

from cogworx.capability.registry import Registry
from cogworx.claims.provenance import Artifact
from cogworx.coordination.events import Event, EventType, Subsystem, validate_event_boundary
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.state import RunStatus
from cogworx.model.base import Model
from cogworx.runtime.context import RunContext
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal, RunState, StepRecord
from cogworx.substrate.latent import LatentStore

Clock = Callable[[], datetime]


class ResumeError(Exception):
    """Raised when a run cannot be resumed (unknown run, or journal/graph divergence)."""


class Engine:
    """The runtime that drives a ``StageGraph`` to a terminal ``RunState``."""

    def __init__(
        self,
        *,
        model: Model,
        journal: Journal,
        graph_store: GraphStore,
        latent: LatentStore,
        pathways: PathwayRegistry,
        budget: BudgetGuard | None = None,
        registry: Registry | None = None,
        event_sink: Callable[[Event], None] | None = None,
        clock: Clock = lambda: datetime.now(UTC),
        max_steps: int = 1000,
    ) -> None:
        self._model = model
        self._journal = journal
        self._graph_store = graph_store
        self._latent = latent
        self._pathways = pathways
        self._budget = budget if budget is not None else BudgetGuard()
        self._registry = registry
        self._event_sink = event_sink
        self._clock = clock
        self._max_steps = max_steps
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
        pathway_id: str,
        initial: Artifact,
        pathway_version: int = 1,
    ) -> RunState:
        graph = self._pathways.get(pathway_id, pathway_version)
        await self._journal.start_run(
            run_id, session_id, pathway_id=pathway_id, pathway_version=pathway_version
        )
        ctx = self._build_context(run_id=run_id, session_id=session_id)
        self._emit(ctx, EventType.RUN_STARTED, run_id=run_id, session_id=session_id)
        return await self._drive(ctx, graph, current=graph.entry)

    async def resume(self, run_id: str) -> RunState:
        """Re-drive the run, replaying committed steps from the journal (S6 — no model re-call).

        Cold cross-process resume works: the graph is rehydrated from the ``PathwayRegistry`` using
        the run's stored ``pathway_id`` pointer — NOT in-process state — so a brand-new ``Engine``
        in a fresh process resumes as long as the SAME pathways are registered at startup. If
        the registry lacks the run's pathway, ``PathwayError`` propagates (the honest requirement).
        ``_drive`` walks from the entry; every already-committed position is replayed from the
        journal WITHOUT re-running the stage or re-calling the model. Only uncommitted stages run.
        """
        state = await self._journal.load_run(run_id)
        if state is None:
            raise ResumeError(f"cannot resume unknown run {run_id!r}")
        if state.status in (
            RunStatus.COMPLETED,
            RunStatus.AWAITING_HUMAN,
            RunStatus.DEGRADED,
            RunStatus.FAILED,
        ):
            return state
        graph = self._pathways.get(state.pathway_id, state.pathway_version)
        ctx = self._build_context(run_id=run_id, session_id=state.session_id)
        return await self._drive(ctx, graph, current=graph.entry)

    async def _drive(self, ctx: RunContext, graph: StageGraph, *, current: str) -> RunState:
        run_id = ctx.run_id
        session_id = ctx.session_id
        seq = 0
        while True:
            if seq >= self._max_steps:
                await self._journal.set_run_status(run_id, RunStatus.FAILED)
                self._emit(ctx, EventType.RUN_FAILED, run_id=run_id, session_id=session_id)
                break

            existing = await self._journal.read_step(run_id, seq)
            if existing is not None:
                if existing.stage_name != current:
                    raise ResumeError(
                        f"journal/graph divergence at step {seq} of run {run_id!r}: journaled "
                        f"stage {existing.stage_name!r} but the graph reached {current!r} "
                        "(the pathway changed under a resumed run)"
                    )
                result = existing.result  # REPLAY — no model, no commit (S1/S6).
            else:
                self._emit(ctx, EventType.STAGE_ENTERED, run_id=run_id, session_id=session_id)
                stage = graph.get(current)
                result = await stage.run(ctx)
                record = StepRecord(
                    run_id=run_id,
                    step_index=seq,
                    stage_name=current,
                    result=result,
                    committed_at=self._clock(),
                )
                await self._journal.commit_step(record)
                self._emit(ctx, EventType.STEP_COMMITTED, run_id=run_id, session_id=session_id)

            # On resume, replayed (committed) steps re-emit these transition/terminal events. Events
            # are observational broadcasts; idempotency/dedup is a Phase 1 concern (no dedup here).
            match result.kind:
                case "done":
                    await self._journal.set_run_status(run_id, RunStatus.COMPLETED)
                    self._emit(ctx, EventType.RUN_COMPLETED, run_id=run_id, session_id=session_id)
                    break
                case "transition":
                    self._emit(ctx, EventType.STAGE_COMPLETED, run_id=run_id, session_id=session_id)
                    current = result.to
                case "degraded":
                    self._emit(ctx, EventType.STAGE_DEGRADED, run_id=run_id, session_id=session_id)
                    if result.to is None:
                        await self._journal.set_run_status(run_id, RunStatus.DEGRADED)
                        break
                    current = result.to
                case "await-human":
                    self._emit(
                        ctx,
                        EventType.STAGE_AWAITING_HUMAN,
                        run_id=run_id,
                        session_id=session_id,
                    )
                    await self._journal.set_run_status(run_id, RunStatus.AWAITING_HUMAN)
                    break

            seq += 1

        final = await self._journal.load_run(run_id)
        if final is None:
            raise ResumeError(f"journal lost run {run_id!r} mid-drive")
        return final


__all__ = [
    "Clock",
    "Engine",
    "ResumeError",
]
