"""Concrete ``StageContext`` handed to every stage (CANON S3, S7, S8).

``RunContext`` carries the distinct substrate seams (S3 — no flattening), enforces the event
boundary on every ``emit`` (S7), and routes capability dispatch through the registry — surfacing any
unavailable capability (no registry, unknown, or disabled/lesioned) as a single
``CapabilityUnavailable`` so a stage degrades uniformly (the S8 lesion path).

Contract changelog:
  - 2026-06-12 (Pod 3.1e): added ``assembler`` param + ``assemble_context`` +
    ``bind_context_policy`` to RunContext.  Unwired assembler degrades to task-only context
    (S8 lesion behaviour).
  - 2026-06-12 (Pod 3.2c): added ``gate`` param + ``registry`` property + ``bind_tool_policy``
    to RunContext.  ``dispatch`` now goes through ``dispatch_one`` (the shared security chokepoint)
    when a gate is wired (tier-check + arg-validation + taint pipeline, S9/S10).  Concrete-only
    (no StageContext Protocol change — stages do not call bind_tool_policy; the engine does).
  - 2026-06-12 (Pod 3.5b): dispatch_approved concrete method — additive (new method, NOT added
    to the StageContext Protocol; only RunContext exposes it). Stages that need it must assert
    isinstance(ctx, RunContext) first — the type-narrowing makes mypy --strict happy.
  - 2026-06-15 (Pod 4.0 F5, BREAKING via /update-canon): added ``last_output`` — implements the
    new StageContext Protocol member. Reads the run's committed steps from the journal
    (``load_run``) and returns the latest matching stage's output. Mirrors ``read_human_input``
    (pull-based, crash-correct).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from typing import Any

from cogworx.capability.base import CapabilityUnavailable
from cogworx.capability.policy import StageToolPolicy, ToolGate
from cogworx.capability.registry import Registry
from cogworx.capability.router import dispatch_one
from cogworx.claims.provenance import Artifact
from cogworx.context.assembler import ContextAssembler
from cogworx.context.errors import ContextBudgetError
from cogworx.context.types import (
    DEFAULT_CONTEXT_POLICY,
    AssembledCallContext,
    ContextPolicy,
    ContextRequest,
    SlotReport,
)
from cogworx.coordination.events import Event, validate_event_boundary
from cogworx.cost.budget import BudgetGuard
from cogworx.injection.injector import MemoryInjector, resolve_token_counter
from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, InjectedMemory, MemoryPolicy
from cogworx.model.base import ChatMessage, Model
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import (  # noqa: F401 — ContextChunk available for stage authors
    AssembledContext,
    ContextChunk,
)
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal
from cogworx.substrate.latent import LatentStore


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
        gate: ToolGate | None = None,
        event_sink: Callable[[Event], None] | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now(UTC),
        injector: MemoryInjector | None = None,
        assembler: ContextAssembler | None = None,
    ) -> None:
        self.run_id = run_id
        self.session_id = session_id
        self._model = model
        self._journal = journal
        self._graph = graph_store
        self._latent = latent
        self._budget = budget
        self._registry = registry
        self._gate = gate
        self._event_sink = event_sink
        self._clock = clock
        self._injector = injector
        self._assembler = assembler
        self._bound_policy: MemoryPolicy | None = None
        self._bound_context_policy: ContextPolicy | None = None
        self._events: list[Event] = []

    @property
    def budget(self) -> BudgetGuard:
        return self._budget

    @property
    def registry(self) -> Registry | None:
        """The capability registry for this run.

        Exposed as a public read-only property so the router (``run_tool_loop``) can pass it
        alongside the gate without going through a private attribute.  ``None`` when no registry
        was wired at construction time (unwired path → ``dispatch`` raises
        ``CapabilityUnavailable``).
        """
        return self._registry

    @property
    def clock(self) -> Callable[[], datetime]:
        """The engine's injected clock. A Wait-bearing stage computes ``wake_at = ctx.clock() +
        delay`` from this, never wall-clock, so ``wake_at`` is deterministic + replay-safe (S6)."""
        return self._clock

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

    def bind_tool_policy(self, policy: StageToolPolicy | None) -> None:
        """Bind a per-stage StageToolPolicy (read from stage.tool_policy by the engine).

        Calls ``gate.bind_policy`` on the drive-level gate so the tier set and taint-drops-
        external logic update atomically for the next stage.  Calling with ``None`` clears to
        ``DEFAULT_TOOL_POLICY`` (mirrors the ``bind_memory_policy`` clear-first D15 invariant).

        Concrete-only on ``RunContext`` — NOT added to the ``StageContext`` Protocol (stages do
        not need to call it; only the engine calls it between stages).
        """
        if self._gate is not None:
            self._gate.bind_policy(policy)

    async def dispatch(self, capability: str, args: Mapping[str, Any]) -> Any:
        """Dispatch a capability call through the security gate (the single chokepoint, S10).

        Uses ``dispatch_one`` from ``capability.router`` so the tier-check, arg-validation,
        invoke, and taint-update pipeline is identical to the batch ``route_tool_calls`` path —
        no duplicated security logic (mythos Decision 4 #3).

        When no gate is wired the call falls back to the legacy name-lookup-only path so
        existing callers that construct ``RunContext`` without a gate continue to work (S8
        graceful degrade — the gate is opt-in at engine construction time).

        Raises
        ------
        ``CapabilityUnavailable``
            Registry not wired, or the name is unknown / disabled / lesioned.
        ``TierViolation``
            Gate is wired and the capability's tier is outside the effective allowed set (S10).
        ``jsonschema.ValidationError``
            Gate is wired and the args fail framework-side schema validation (S9).
        ``TimeoutError``
            Gate is wired and invoke exceeded ``gate.policy.tool_timeout_s``.
        Any exception from the capability's ``invoke``
            Propagated unmodified (unlike ``route_tool_calls``, which maps errors to
            ``ToolResult`` — this path is for stage code that wants raw exceptions).
        """
        if self._registry is None:
            raise CapabilityUnavailable(
                f"cannot dispatch {capability!r}: no capability registry wired into this run"
            )
        if self._gate is None:
            # Legacy path: no gate wired — name-lookup only (no tier/arg validation).
            # This preserves backwards-compatibility for RunContext instances built without
            # a gate (e.g. test helpers that predate Pod 3.2c).
            from cogworx.capability.registry import RegistryError

            try:
                cap = self._registry.get(capability)
            except RegistryError as exc:
                raise CapabilityUnavailable(
                    f"capability {capability!r} is unavailable: {exc}"
                ) from exc
            return await cap.invoke(args)

        # Gated path: full tier + arg-validation + taint pipeline via the shared helper.
        return await dispatch_one(self._gate, self._registry, capability, dict(args))

    async def dispatch_approved(self, capability: str, args: Mapping[str, Any]) -> Any:
        """Dispatch WITH a human-approval grant (S10 trifecta approval path).

        Callable only from deterministic stage code, after the stage has read the journaled
        HITL answer via ctx.read_human_input and routed structurally on it (S9). Delegates to
        dispatch_one with approved=True, bypassing the ApprovalRequired check for this call.
        All other security checks (tier, arg validation, taint latch, timeout) remain in effect.

        This method is NOT on the StageContext Protocol. A stage that needs it must type-narrow:
            assert isinstance(ctx, RunContext)
            await ctx.dispatch_approved(capability, args)
        This is intentional (S9/S10): model-driven code cannot discover or invoke this path.

        Raises
        ------
        CapabilityUnavailable
            Registry not wired, or the name is unknown / disabled / lesioned.
        TierViolation
            The capability's tier is outside the effective allowed set.
        jsonschema.ValidationError
            Args fail framework-side schema validation.
        TimeoutError
            Invoke exceeded gate.policy.tool_timeout_s.
        """
        if self._registry is None:
            raise CapabilityUnavailable(
                f"cannot dispatch_approved {capability!r}: no capability registry wired"
            )
        if self._gate is None:
            raise CapabilityUnavailable(
                f"cannot dispatch_approved {capability!r}: no tool gate wired"
            )
        return await dispatch_one(self._gate, self._registry, capability, dict(args), approved=True)

    async def read_human_input(self, step_index: int) -> Artifact | None:
        """Return the HITL answer committed at ``step_index`` for this run, or ``None`` if absent.

        PULL-based: stages pull the answer from the journal; the engine never pushes a
        ``ctx.human_input`` attribute. This means cold resume works correctly — the answer is
        in the journal and any stage can read it without the engine re-injecting it (S6/S9).
        """
        return await self._journal.read_human_input(self.run_id, step_index)

    async def last_output(self, stage_name: str) -> Artifact | None:
        """Most recent committed output of ``stage_name`` for this run, or ``None``.

        PULL-based + journal-backed (mirrors :meth:`read_human_input`): reads the run's committed
        steps via ``load_run`` and returns the latest matching stage's output Artifact.
        Crash-correct — a fresh ``RunContext`` after a cold resume reads the same durable answer
        (S6); a stage holds no in-process step history.
        """
        run = await self._journal.load_run(self.run_id)
        if run is None:
            return None
        for step in reversed(run.steps):
            if step.stage_name == stage_name:
                return step.result.output
        return None

    def bind_memory_policy(self, policy: MemoryPolicy | None) -> None:
        """Bind a per-stage MemoryPolicy (read from stage.memory_policy by the engine)."""
        self._bound_policy = policy

    async def recall(
        self,
        query: RecallQuery,
        *,
        policy: MemoryPolicy | None = None,
    ) -> InjectedMemory:
        if self._injector is None:
            # Unwired: return empty InjectedMemory with status="unwired"
            empty = AssembledContext(chunks=(), token_count=0, budget=0, dropped=0)
            return InjectedMemory(context=empty, status="unwired")
        # Policy resolution: explicit arg > bound stage policy > DEFAULT
        resolved = (
            policy
            if policy is not None
            else (self._bound_policy if self._bound_policy is not None else DEFAULT_MEMORY_POLICY)
        )
        return await self._injector.inject(query, policy=resolved)

    def bind_context_policy(self, policy: ContextPolicy | None) -> None:
        """Bind a per-stage ContextPolicy (read from stage.context_policy by the engine).

        Calling with None clears the binding so no policy leaks across stages (mirrors
        bind_memory_policy / D15 invariant).
        """
        self._bound_context_policy = policy

    async def assemble_context(
        self,
        request: ContextRequest,
        *,
        policy: ContextPolicy | None = None,
    ) -> AssembledCallContext:
        """Assemble a fully-budgeted call context for this stage.

        Policy resolution order: explicit ``policy`` arg > stage-level ``_bound_context_policy``
        (set by the engine from ``stage.context_policy``) > ``DEFAULT_CONTEXT_POLICY``.

        Unwired (assembler=None) S8 lesion degrade: returns a minimal task-only
        ``AssembledCallContext`` that still lets the run complete.  The ``status`` field
        on the returned object carries ``"unwired"`` in the first slot report so callers
        can detect the degrade (mirrors the ``recall()`` unwired path).
        """
        resolved_policy = (
            policy
            if policy is not None
            else (
                self._bound_context_policy
                if self._bound_context_policy is not None
                else DEFAULT_CONTEXT_POLICY
            )
        )

        if self._assembler is None:
            # S8 lesion degrade: no assembler wired — return task-only context.
            # S11 still enforces the hard ceiling on this path: resolve the SAME counter the
            # wired path uses (resolve_token_counter delegates to model.count_tokens or
            # approx_tokens), never a raw len()//4 estimate.
            count_fn = resolve_token_counter(self._model)
            messages: list[ChatMessage] = []
            if request.instructions:
                messages.append(ChatMessage(role="system", content=request.instructions))
            messages.append(ChatMessage(role="user", content=request.task))
            token_count = sum(count_fn(m.content) for m in messages)
            if token_count > resolved_policy.total_budget:
                slot_names = ("instructions", "task") if request.instructions else ("task",)
                raise ContextBudgetError(
                    required_tokens=token_count,
                    budget=resolved_policy.total_budget,
                    slot_names=slot_names,
                )
            unwired_report = SlotReport(
                name="task",
                status="unwired",
                tokens=count_fn(request.task),
                chunks_admitted=1,
                chunks_dropped=0,
                evicted=False,
            )
            return AssembledCallContext(
                messages=tuple(messages),
                tools=(),
                token_count=token_count,
                budget=resolved_policy.total_budget,
                slots=(unwired_report,),
                memory=None,
            )

        # Build a new request with the resolved policy attached (policy resolution lives here).
        request_with_policy = request.model_copy(update={"policy": resolved_policy})
        return await self._assembler.assemble(request_with_policy)

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(self._events)


__all__ = [
    "ContextAssembler",
    "RunContext",
]
