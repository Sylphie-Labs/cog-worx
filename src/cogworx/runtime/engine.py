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

Durable sleep + pause/resume ride this same journal (S6). A ``Wait`` step parks the run: on its
FRESH commit the engine arms one durable timer (id ``f"{run_id}:{step_index}"``), sets the run
WAITING, and breaks; later the ``Sweeper`` leases the due timer and calls ``fire_timer``, which
re-drives from the entry. The committed ``Wait`` then REPLAYS as a plain advance to ``to`` — it
cancels its (now-superfluous) timer and never re-arms ``set_timer`` — so advancing past a ``Wait``
is exactly-once even though the timer may fire at-least-once (a crashed lease goes stale and
re-fires; the advance is idempotent on ``(run_id, step_index)``). A plain crash-``resume`` treats
WAITING and PAUSED as non-advancing terminal-for-resume states: only the sweeper (``fire_timer``) or
``unpause`` advance a parked run. Pause is COOPERATIVE: each ``_drive`` iteration reads the run's
status (``get_run_status``, not a cached ``RunState``) at the top and breaks if PAUSED, so a pause
from another process is honored at the next step boundary.

Timer cleanup on stop is uniform across the parked/terminal paths: ``done``, terminal ``degraded``,
the ``max_steps`` ceiling FAILED, AND ``await-human`` all ``cancel_timers_for_run`` so no orphaned
timer can re-fire a finished or parked-for-human run. The two stops that must NOT cancel are the
fresh ``Wait`` (WAITING — the timer is the wake mechanism) and the cooperative PAUSED break (the
timer must survive so an ``unpause`` of a still-WAITING run keeps its wake). Re-driving a run is
serialized by ``compare_and_set_run_status`` (the run-level mutual-exclusion primitive): a driver
must win the CAS into RUNNING before executing uncommitted stages — exactly-once EXECUTION for the
CAS-guarded entrypoints (``fire_timer``, ``unpause``, ``provide_human_input``). A bare ``resume``
of a still-RUNNING run does NOT take that CAS, so two concurrent crash-resumes of the same RUNNING
run can double-execute an uncommitted stage's SIDE EFFECT (the journal commit stays exactly-once on
``(run_id, step_index)``; the side effect is not protected). Single-flight crash-resume is the
operator's responsibility until the run-lease/reaper lands (deferred ops pod).
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any

from cogworx.capability.registry import Registry
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.coordination.events import Event, EventType, Subsystem, validate_event_boundary
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph, StageGraphError
from cogworx.loop.pathway import PathwayRegistry, pathway_fingerprint
from cogworx.loop.result import Degraded
from cogworx.loop.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from cogworx.loop.state import RunStatus
from cogworx.model.base import Model
from cogworx.runtime.context import RunContext
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal, RunState, StepRecord, Timer
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
            clock=self._clock,
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

    def _exhaustion_degraded(
        self, exc: BaseException, attempt: int, to: str | None
    ) -> Degraded:
        """Build the first-class exhaustion ``Degraded`` (S8) committed at ``seq`` on retry-exhaust.

        The output artifact carries provenance with ``source="system"`` — a deterministic,
        zero-model engine/control-plane event, NOT model cognition (S1 reserves
        "reflection"/"inference" for model-heavy work; S5: every write provenanced AND the source
        honest). ``to`` is the policy ``exhausted_to`` (``None`` => the run terminates DEGRADED).
        """
        artifact = Artifact(
            kind="retry-exhausted",
            produced_by="engine",
            provenance=Provenance(
                source="system", confidence=1.0, recorded_at=self._clock()
            ),
            data={"failure_class": type(exc).__name__, "attempts": attempt},
        )
        reason = f"retry exhausted after {attempt} attempts: {type(exc).__name__}"
        return Degraded(reason=reason, output=artifact, to=to)

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
            run_id,
            session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint(graph),
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

        Before driving, the rehydrated graph's STRUCTURAL fingerprint is compared against the one
        stored at ``start_run``; a mismatch means the pathway was edited in place under the same
        ``(pathway_id, version)`` (rewired transitions, added/removed/renamed stages) and resume is
        refused with ``ResumeError`` rather than re-driving against the wrong graph. The per-step
        ``stage_name`` guard in ``_drive`` remains a deeper backstop. ``_drive`` walks from the
        entry; every already-committed position is replayed from the journal WITHOUT re-running the
        stage or re-calling the model. Only uncommitted stages run.
        """
        state = await self._journal.load_run(run_id)
        if state is None:
            raise ResumeError(f"cannot resume unknown run {run_id!r}")
        if state.status in (
            RunStatus.COMPLETED,
            RunStatus.AWAITING_HUMAN,
            RunStatus.DEGRADED,
            RunStatus.FAILED,
            RunStatus.WAITING,
            RunStatus.RETRYING,
            RunStatus.PAUSED,
        ):
            # WAITING/RETRYING/PAUSED are parked, not terminal: a plain crash-resume returns them
            # as-is. Only the sweeper (fire_timer) or unpause advance a parked run (S6).
            return state
        graph = self._pathways.get(state.pathway_id, state.pathway_version)
        rehydrated_fingerprint = pathway_fingerprint(graph)
        if rehydrated_fingerprint != state.pathway_fingerprint:
            raise ResumeError(
                f"pathway structure changed under resumed run {run_id!r}: the rehydrated "
                f"{state.pathway_id!r} v{state.pathway_version} graph fingerprints "
                f"{rehydrated_fingerprint!r} but the run was started against "
                f"{state.pathway_fingerprint!r}. An in-place pathway edit (rewired transitions, "
                "added/removed/renamed stages) resumes against the wrong graph — bump the pathway "
                "version instead of editing in place."
            )
        ctx = self._build_context(run_id=run_id, session_id=state.session_id)
        return await self._drive(ctx, graph, current=graph.entry)

    async def _rehydrate(self, run_id: str) -> tuple[RunState, StageGraph, RunContext]:
        """Load a run + rehydrate its graph from the registry, guarding the structural fingerprint.

        Shared by ``fire_timer``/``unpause`` (the parked-run re-drive entrypoints) and mirrors
        ``resume``'s cold-resume contract: the graph comes from the ``PathwayRegistry`` via the
        run's stored pathway pointer (no in-process graph), and an in-place structural edit under
        the same ``(pathway_id, version)`` is refused with ``ResumeError`` rather than re-driving
        the wrong graph.
        """
        state = await self._journal.load_run(run_id)
        if state is None:
            raise ResumeError(f"cannot drive unknown run {run_id!r}")
        graph = self._pathways.get(state.pathway_id, state.pathway_version)
        rehydrated_fingerprint = pathway_fingerprint(graph)
        if rehydrated_fingerprint != state.pathway_fingerprint:
            raise ResumeError(
                f"pathway structure changed under driven run {run_id!r}: the rehydrated "
                f"{state.pathway_id!r} v{state.pathway_version} graph fingerprints "
                f"{rehydrated_fingerprint!r} but the run was started against "
                f"{state.pathway_fingerprint!r}. Bump the pathway version instead of editing "
                "in place."
            )
        ctx = self._build_context(run_id=run_id, session_id=state.session_id)
        return state, graph, ctx

    async def fire_timer(self, run_id: str) -> RunState:
        """Re-drive a parked (WAITING **or** RETRYING) run after the sweeper leased its due timer
        (S6) — no commits of its own (the ``Sweeper`` calls this once a timer is due).

        The pre-drive action is a CAS ``<parked status> -> RUNNING`` — the run-level
        mutual-exclusion primitive that makes this RUN (not its commits alone) exactly-once: a
        stale-lease
        re-delivery, a fire on an already-terminal run, or a fire racing a pause/another sweeper
        LOSES the CAS and is a no-op (the loser must NOT execute the uncommitted, model-bearing
        stage — the F1 / R7 double-execution fix). The CAS reads the run's CURRENT parked status
        first (a WAITING wake and a RETRYING re-attempt share this path), so the expect matches the
        run rather than hardcoding WAITING.

        On winning, the run's timer(s) are cancelled (``cancel_timers_for_run``): the timer has
        served its purpose, and dropping it avoids a stale retry timer re-firing the same ``seq``.
        Then ``_drive`` walks from the entry: a WAITING re-drive replays the committed ``Wait`` and
        advances past it (its own ``cancel_timer`` is now a redundant no-op — behaviour-equivalent
        to 1.1); a RETRYING re-drive finds ``read_step(seq) is None`` and RE-ATTEMPTS the stage at
        the same frozen ``seq`` with the journaled attempt count. No model re-call on a committed
        prefix (at-least-once fire / exactly-once advance).
        """
        state, graph, ctx = await self._rehydrate(run_id)
        parked = state.status
        if parked not in (RunStatus.WAITING, RunStatus.RETRYING):
            return state
        if not await self._journal.compare_and_set_run_status(
            run_id, expect=parked, new=RunStatus.RUNNING
        ):
            return state
        await self._journal.cancel_timers_for_run(run_id)
        return await self._drive(ctx, graph, current=graph.entry)

    async def pause(self, run_id: str) -> bool:
        """Cooperatively pause a run — flips to PAUSED only from RUNNING/WAITING (else a no-op).

        Returns ``True`` iff this call transitioned the run to PAUSED (one of the conditional CAS
        attempts won); ``False`` if the pause was a no-op because the run was unknown, terminal, or
        in-transition (no CAS matched). Pause is COOPERATIVE, so a ``False`` is a DROPPED pause the
        caller can detect and re-issue — there is no implicit retry here.

        The flip is a CAS (RUNNING->PAUSED, falling back to WAITING->PAUSED), so a pause racing a
        COMPLETED/terminal run can never clobber the terminal status (the F2 fix): both CASes lose,
        ``False`` is returned, and pause is a no-op. ``_drive`` honors PAUSED at the next step
        boundary (it reads the authoritative status at the top of each iteration), so an in-flight
        drive stops cleanly after the current step commits.
        """
        state = await self._journal.load_run(run_id)
        if state is None:
            return False
        flipped = await self._journal.compare_and_set_run_status(
            run_id, expect=RunStatus.RUNNING, new=RunStatus.PAUSED
        ) or await self._journal.compare_and_set_run_status(
            run_id, expect=RunStatus.WAITING, new=RunStatus.PAUSED
        )
        if not flipped:
            return False
        ctx = self._build_context(run_id=run_id, session_id=state.session_id)
        self._emit(ctx, EventType.RUN_PAUSED, run_id=run_id, session_id=state.session_id)
        return True

    async def unpause(self, run_id: str) -> RunState:
        """Resume a PAUSED run: CAS PAUSED->RUNNING, re-drive from entry, return the terminal state.

        unpause resumes the run IMMEDIATELY; if it was waiting on a timer, the pending wait is
        discarded and the run advances past it now — do not pause a waiting run if the remaining
        sleep must be preserved. A run PAUSED while WAITING-on-a-timer has a COMMITTED ``Wait``
        step, which replays as a plain advance to its ``to`` (cancelling the now-superfluous timer);
        the remaining sleep is NOT honored on unpause, the run proceeds at once.

        The CAS is the run-level mutual exclusion: a fire/unpause race or a double unpause means
        only one caller flips PAUSED->RUNNING and drives; a loser returns the current state without
        driving (it must NOT execute uncommitted, model-bearing stages). ``_drive`` replays the
        committed prefix (no model re-call, S6) and runs only the stages that had not yet committed
        when the pause landed.
        """
        state, graph, ctx = await self._rehydrate(run_id)
        if not await self._journal.compare_and_set_run_status(
            run_id, expect=RunStatus.PAUSED, new=RunStatus.RUNNING
        ):
            return state
        self._emit(
            ctx, EventType.RUN_RESUMED, run_id=run_id, session_id=ctx.session_id
        )
        return await self._drive(ctx, graph, current=graph.entry)

    async def provide_human_input(
        self,
        run_id: str,
        *,
        payload: dict[str, Any],
        kind: str = "human-input",
        confidence: float = 1.0,
    ) -> RunState:
        """Record a human answer and re-drive the run from the entry (S6, S5, H3).

        If the run is not in ``AWAITING_HUMAN`` status this is a no-op and returns the current
        state unchanged (idempotent on a non-parked or already-completed run).

        The ordering is HARD (H3): ``record_human_input`` persists the answer BEFORE the CAS
        flips ``AWAITING_HUMAN -> RUNNING``. If the process dies between the two calls, a
        re-issued ``provide_human_input`` finds the answer already committed and completes without
        overwriting it (first-answer-wins idempotency in the adapter).

        The answer ``Artifact`` carries ``provenance.source="human"`` — the honest epistemic label
        for a human-supplied input (S5). ``kind`` and ``confidence`` are caller-supplied (defaults
        match the S5 human-input convention).
        """
        state, graph, ctx = await self._rehydrate(run_id)
        if state.status is not RunStatus.AWAITING_HUMAN:
            return state

        # The awaiting step is the last committed step (status guarantees it is an await-human).
        last_step = state.steps[-1]
        seq = last_step.step_index

        answer = Artifact(
            kind=kind,
            produced_by="human",
            provenance=Provenance(
                source="human",
                confidence=confidence,
                recorded_at=self._clock(),
            ),
            data=payload,
        )

        # H3 hard ordering: persist BEFORE the CAS. A crash here leaves the answer committed but
        # the run still AWAITING_HUMAN — a re-issued call finds the answer (first-answer-wins) and
        # the CAS succeeds on the next attempt.
        await self._journal.record_human_input(run_id, seq, answer)

        self._emit(
            ctx,
            EventType.HUMAN_INPUT_RECEIVED,
            run_id=run_id,
            session_id=ctx.session_id,
        )

        if not await self._journal.compare_and_set_run_status(
            run_id, expect=RunStatus.AWAITING_HUMAN, new=RunStatus.RUNNING
        ):
            # Another caller won the CAS — load and return the current state.
            reloaded = await self._journal.load_run(run_id)
            if reloaded is None:
                raise ResumeError(
                    f"journal lost run {run_id!r} after CAS loss in provide_human_input"
                )
            return reloaded

        self._emit(ctx, EventType.RUN_RESUMED, run_id=run_id, session_id=ctx.session_id)
        return await self._drive(ctx, graph, current=graph.entry)

    async def _drive(self, ctx: RunContext, graph: StageGraph, *, current: str) -> RunState:
        run_id = ctx.run_id
        session_id = ctx.session_id
        seq = 0
        while True:
            # Cooperative pause: read the AUTHORITATIVE status (not a cached RunState) so a pause
            # from another process is honored at this step boundary before the next stage runs.
            if await self._journal.get_run_status(run_id) is RunStatus.PAUSED:
                break

            if seq >= self._max_steps:
                await self._journal.set_run_status(run_id, RunStatus.FAILED)
                await self._journal.cancel_timers_for_run(run_id)
                self._emit(ctx, EventType.RUN_FAILED, run_id=run_id, session_id=session_id)
                break

            existing = await self._journal.read_step(run_id, seq)
            replaying = existing is not None
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
                policy: RetryPolicy = getattr(stage, "retry_policy", DEFAULT_RETRY_POLICY)
                # Classification is STRUCTURAL (S9): only a timeout or a type in the dev-authored
                # ``retryable`` allowlist enters the retry machine; anything else is a BUG and
                # propagates loud (fail-loud) — uncaught here, no increment, no timer, no FAILED.
                retryable: tuple[type[BaseException], ...] = (TimeoutError, *policy.retryable)
                try:
                    if policy.timeout is not None:
                        result = await asyncio.wait_for(
                            stage.run(ctx), policy.timeout.total_seconds()
                        )
                    else:
                        result = await stage.run(ctx)
                except retryable as exc:
                    # A FAILED attempt commits NOTHING — only the durable attempt counter climbs at
                    # a frozen ``seq`` (the (run_id, step_index) exactly-once invariant stays
                    # pristine). The increment is the authority for exhaustion across a crash (S6).
                    attempt = await self._journal.increment_attempt(run_id, seq)
                    if attempt >= policy.max_attempts:
                        if policy.on_exhausted == "fail":
                            await self._journal.set_run_status(run_id, RunStatus.FAILED)
                            await self._journal.cancel_timers_for_run(run_id)
                            self._emit(
                                ctx, EventType.RUN_FAILED, run_id=run_id, session_id=session_id
                            )
                            break
                        # Degraded-onward (S8 default): synthesize a provenance-bearing exhaustion
                        # artifact (source="system" — a deterministic engine/control event, NOT a
                        # model inference) and commit it at ``seq``; the existing ``degraded`` match
                        # arm then routes onward / terminates.
                        result = self._exhaustion_degraded(exc, attempt, policy.exhausted_to)
                    else:
                        await self._journal.set_timer(
                            Timer(
                                run_id=run_id,
                                timer_id=f"{run_id}:{seq}:retry:{attempt}",
                                wake_at=self._clock() + policy.backoff(attempt),
                                payload={"action": "retry", "step_index": seq},
                            )
                        )
                        await self._journal.set_run_status(run_id, RunStatus.RETRYING)
                        self._emit(
                            ctx, EventType.STAGE_RETRYING, run_id=run_id, session_id=session_id
                        )
                        self._emit(
                            ctx, EventType.RUN_RETRYING, run_id=run_id, session_id=session_id
                        )
                        break
                # Declared-route guard (S9/S6): a fresh result that routes to an undeclared stage
                # is rejected BEFORE commit — a bad ``to`` must never reach the journal, or it
                # re-detonates on every future resume. Guards ALL to-bearing results: Transition,
                # Wait, AwaitHuman, and Degraded when to is not None. Done and Degraded(to=None)
                # are terminal and have no ``to`` to check. Replay is intentionally excluded: a
                # committed ``to`` was valid at write time; re-validating against a version-bumped
                # graph could falsely reject a legitimately parked run (S6 trusts the journal).
                # Uses edges_from (not transitions_from) so the engine's own exhaustion-degraded
                # route (policy.exhausted_to, present in edges_from but not transitions_from)
                # passes the guard correctly.
                declared_to: str | None = getattr(result, "to", None)
                if declared_to is not None:
                    allowed = graph.edges_from(current)
                    if declared_to not in allowed:
                        raise StageGraphError(
                            f"stage {current!r} returned a result routing to {declared_to!r}, "
                            f"which is not in its declared edge set {sorted(allowed)!r}; "
                            "add the target to the stage's transitions (or exhausted_to for a "
                            "retry-exhaustion route) before running"
                        )
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
                    await self._journal.cancel_timers_for_run(run_id)
                    self._emit(ctx, EventType.RUN_COMPLETED, run_id=run_id, session_id=session_id)
                    break
                case "transition":
                    self._emit(ctx, EventType.STAGE_COMPLETED, run_id=run_id, session_id=session_id)
                    current = result.to
                case "wait":
                    # A committed Wait REPLAYS as a plain advance (cancel its timer, go to ``to``)
                    # — never re-park, never re-arm set_timer (C6). A FRESH Wait arms one durable
                    # timer and parks the run WAITING for the sweeper.
                    if replaying:
                        await self._journal.cancel_timer(f"{run_id}:{seq}")
                        self._emit(
                            ctx, EventType.STAGE_COMPLETED, run_id=run_id, session_id=session_id
                        )
                        current = result.to
                    else:
                        await self._journal.set_timer(
                            Timer(
                                run_id=run_id,
                                timer_id=f"{run_id}:{seq}",
                                wake_at=result.wake_at,
                            )
                        )
                        await self._journal.set_run_status(run_id, RunStatus.WAITING)
                        self._emit(
                            ctx, EventType.STAGE_WAITING, run_id=run_id, session_id=session_id
                        )
                        self._emit(ctx, EventType.RUN_WAITING, run_id=run_id, session_id=session_id)
                        break
                case "degraded":
                    self._emit(ctx, EventType.STAGE_DEGRADED, run_id=run_id, session_id=session_id)
                    if result.to is None:
                        await self._journal.set_run_status(run_id, RunStatus.DEGRADED)
                        await self._journal.cancel_timers_for_run(run_id)
                        break
                    current = result.to
                case "await-human":
                    # A committed AwaitHuman REPLAYS as a plain advance to ``to`` — never re-park,
                    # never re-await. FRESH: park AWAITING_HUMAN, cancel timers (same cleanup rule
                    # as Wait/done/degraded), and break. REPLAY: advance to ``result.to`` exactly
                    # like a Transition (emit STAGE_COMPLETED, set current, continue).
                    if replaying:
                        self._emit(
                            ctx,
                            EventType.STAGE_COMPLETED,
                            run_id=run_id,
                            session_id=session_id,
                        )
                        current = result.to
                    else:
                        self._emit(
                            ctx,
                            EventType.STAGE_AWAITING_HUMAN,
                            run_id=run_id,
                            session_id=session_id,
                        )
                        await self._journal.set_run_status(run_id, RunStatus.AWAITING_HUMAN)
                        await self._journal.cancel_timers_for_run(run_id)
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
