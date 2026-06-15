"""Stage composition primitives (CANON S2, S8).

The Spine is a graph of stages. A ``Stage`` runs against a ``StageContext`` and returns a
discriminated ``StageResult`` — ``transition`` / ``done`` / ``await-human`` / ``degraded`` (defined
in :mod:`cogworx.loop.result`) — making HITL and graceful degradation first-class loop transitions.
Ported from biz-firm's composition primitives (Stage · Capability · Context · Loop).

Contract changelog:
  - 2026-06-12 (Pod 3.1e, BREAKING, Jim-approved via /update-canon): added ``assemble_context`` +
    ``bind_context_policy`` to StageContext Protocol.
  - 2026-06-15 (Pod 4.0 F5, BREAKING, Jim-approved via /update-canon): added ``last_output`` to the
    StageContext Protocol — the journal-backed pull accessor for an upstream stage's most recent
    committed output (the Phase 4 EvaluateStage routes on the antithesis Verdict). New Protocol
    member → C3-breaking (§6.1).
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
    from cogworx.context.types import AssembledCallContext, ContextPolicy, ContextRequest
    from cogworx.injection.policy import InjectedMemory, MemoryPolicy
    from cogworx.recall.query import RecallQuery
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

    async def last_output(self, stage_name: str) -> Artifact | None:
        """Return the most recent committed output ``Artifact`` of ``stage_name`` for this run, or
        ``None`` if that stage has committed no output yet.

        PULL-based, journal-backed — the same idiom as :meth:`read_human_input` (the engine
        threads no artifact between stages). A stage that routes on an upstream stage's output
        (e.g. the dialectic ``EvaluateStage`` reading the antithesis ``Verdict``) pulls it here.
        "Most recent" is cycle-correct for a refine loop without coupling to any stage name: a
        consumer that runs after its producer each cycle sees THIS cycle's output (the latest
        commit). Crash-correct — it reads the durable committed steps, so a cold resume returns
        the same answer (S6/S9).
        """
        ...

    async def recall(
        self,
        query: RecallQuery,
        *,
        policy: MemoryPolicy | None = None,
    ) -> InjectedMemory:
        """Execute recall and return an InjectedMemory for this stage.

        policy overrides any stage-level memory_policy bound by the engine.
        Returns status='unwired' with empty context when no RecallStack is wired.
        """
        ...

    async def assemble_context(
        self,
        request: ContextRequest,
        *,
        policy: ContextPolicy | None = None,
    ) -> AssembledCallContext:
        """Assemble a fully-budgeted call context for this stage.

        policy resolution order: explicit arg > stage-level context_policy bound by the engine >
        DEFAULT_CONTEXT_POLICY.
        Returns a task-only AssembledCallContext with status="unwired" when no ContextAssembler
        is wired (S8 graceful degradation).
        """
        ...

    def bind_context_policy(self, policy: ContextPolicy | None) -> None:
        """Bind a per-stage ContextPolicy (read from stage.context_policy by the engine).

        Calling with None clears the binding so no policy leaks across stages.
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
