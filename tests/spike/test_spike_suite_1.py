"""Spike Suite 1 — the falsifiable GATE that unblocks Phase 1 (CANON S12).

Three criteria, proven ON THE REAL POLYGLOT SUBSTRATE (Neo4j + Postgres/Timescale/pgvector), each
of which falsifies a load-bearing bet if it fails:

- **Criterion 1 (S3 — the polyglot bet):** all THREE real engines do real, independent work in ONE
  test. Timescale round-trips a run journal; Neo4j round-trips a provenance-bearing claim and walks
  its ``DERIVED_FROM`` lineage; pgvector returns nearest-first dense recall. Failure of any one
  engine fails the polyglot-capability criterion (they are asserted in the same test body).
- **Criterion 2 (S6 — THE durability gate):** the reusable kill-mid-run chaos harness
  (``assert_resume_never_recalls_model``) wired to a REAL ``TimescaleJournal``. A run crashes after
  the model-bearing ``respond`` stage durably commits; a fresh engine resumes over the same Postgres
  rows with a zero-response model and makes ZERO model calls. A post-assert queries Postgres
  directly to prove exactly ONE row carries the respond idempotency_key — exactly-once survived a
  real commit + crash + resume against durable rows, not memory.
- **Criterion 3 (S8 — first Lesion pass):** disabling a real component (a ``latent_recall``
  capability backed by ``PgLatentStore``) makes the loop DEGRADE, not crash, and emits the lesion
  events (``LESION_ENABLED`` + ``STAGE_DEGRADED``) that pass the event-boundary contract.

These hit the live substrate; under ``-m spike`` (also ``-m integration``). They ERROR rather than
silently pass when the substrate is unreachable, since the gate is only meaningful when it is up.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from datetime import UTC, datetime

import psycopg
import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_graph import Neo4jGraphStore
from cogworx.adapters.pg_latent import PgLatentStore
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.capability.base import CapabilityUnavailable
from cogworx.capability.registry import Registry, function_capability
from cogworx.claims.provenance import Artifact, Claim, Provenance
from cogworx.coordination.events import (
    Event,
    EventType,
    Subsystem,
    validate_event_boundary,
)
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.graph import StageGraph
from cogworx.loop.result import Degraded, Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.runtime.context import RunContext
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal, StepRecord
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import assert_resume_never_recalls_model
from cogworx.testing.reference_agent import IntakeStage

pytestmark = [pytest.mark.spike, pytest.mark.integration]


# The 3-stage chaos graph (intake[no model] -> respond[the model call] -> close[terminal Done]),
# redefined here verbatim from ``tests/unit/test_durability_chaos.py``: ``tests`` is not an
# importable package (no rootdir __init__), so the deterministic module cannot be imported
# cross-tier. The spike wires the SAME stages with the REAL ``TimescaleJournal`` to prove the
# durability claim against durable Postgres rows rather than the in-memory journal.


class _RespondToCloseStage:
    """Model-bearing stage that transitions to ``close`` (the single model call in the graph)."""

    name: str = "respond"
    transitions: tuple[str, ...] = ("close",)

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Respond to the intake.")]
        )
        ctx.budget.record(response.usage)
        artifact = Artifact(
            kind="response",
            produced_by="respond",
            provenance=Provenance(
                source="inference", confidence=1.0, recorded_at=datetime.now(UTC)
            ),
            data={"text": response.text or ""},
        )
        return Transition(to="close", output=artifact)


class _CloseStage:
    """Terminal, no-model stage: echoes the response text the committed respond step produced."""

    name: str = "close"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        prior = await ctx.journal.read_step(ctx.run_id, "respond")
        text = ""
        if prior is not None:
            output = getattr(prior.result, "output", None)
            if output is not None:
                text = str(output.data.get("text", ""))
        artifact = Artifact(
            kind="closed",
            produced_by="close",
            provenance=Provenance(
                source="inference", confidence=1.0, recorded_at=datetime.now(UTC)
            ),
            data={"text": text},
        )
        return Done(output=artifact)


def build_chaos_graph() -> StageGraph:
    return StageGraph([IntakeStage(), _RespondToCloseStage(), _CloseStage()], entry="intake")


def _chaos_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=datetime.now(UTC)),
        data={"text": "hello"},
    )


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # Async psycopg 3 cannot run on Windows' default ProactorEventLoop; it needs a selector loop.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


_NOW = datetime(2026, 6, 8, 12, 0, 0, tzinfo=UTC)
_LATER = datetime(2026, 6, 8, 12, 0, 1, tzinfo=UTC)


def _artifact(kind: str = "t") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by="spike",
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_NOW),
    )


def _spike_claim(claim_id: str, *, payload: str, evidence: tuple[str, ...] = ()) -> Claim:
    return Claim(
        id=claim_id,
        subject="cogworx",
        predicate="proves",
        payload=payload,
        epistemic_type="inference",
        provenance=Provenance(
            source="extraction",
            source_ref="spike-suite-1",
            confidence=0.9,
            evidence=evidence,
            recorded_at=_NOW,
        ),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="stage:spike",
    )


# ==================================================================================================
# Criterion 1 — polyglot substrate capability (the S3 bet): all THREE engines do real work.
# ==================================================================================================


async def test_criterion_1_polyglot_substrate_capability(settings: SubstrateSettings) -> None:
    """All three real engines independently do real work in ONE test (S3).

    Any one engine failing fails the polyglot-capability criterion — the asserts share a body.
    """
    journal = TimescaleJournal(settings=settings)
    graph = Neo4jGraphStore(settings=settings)
    latent = PgLatentStore(dim=4, settings=settings)
    try:
        # --- TimescaleDB: a run journal round-trips with exactly-once, ordered, status-derived. ---
        await journal.ensure_schema()
        await journal.reset()
        run_id = f"spike1-{uuid.uuid4().hex}"
        await journal.start_run(run_id, "spike1-sess")
        transition_step = StepRecord(
            run_id=run_id,
            step_id="intake",
            stage_name="intake",
            result=Transition(to="respond", output=_artifact("intake")),
            idempotency_key=f"{run_id}:intake",
            committed_at=_NOW,
        )
        done_step = StepRecord(
            run_id=run_id,
            step_id="respond",
            stage_name="respond",
            result=Done(output=_artifact("response")),
            idempotency_key=f"{run_id}:respond",
            committed_at=_LATER,
        )
        await journal.commit_step(transition_step)
        await journal.commit_step(done_step)

        read_back = await journal.read_step(run_id, "respond")
        assert read_back is not None
        assert read_back.result == done_step.result  # StageResult round-trips through jsonb.
        run_state = await journal.load_run(run_id)
        assert run_state is not None
        assert run_state.status is RunStatus.COMPLETED
        assert tuple(step.step_id for step in run_state.steps) == ("intake", "respond")

        # --- Neo4j: a provenance-bearing claim round-trips and its DERIVED_FROM lineage walks. ---
        # Upsert BOTH the evidence and the derived claim as FULL claims, then link via the derived
        # claim's ``evidence`` tuple. (``neighbors`` rebuilds a full ``Claim`` per hop, so the
        # neighbor must itself be a fully-propertied claim — not a bare MERGE stub.)
        await graph.ensure_schema()
        await graph.reset()
        evidence_id = "spike1:evidence"
        evidence_claim = _spike_claim(evidence_id, payload="raw observation")
        derived_claim = _spike_claim(
            "spike1:claim", payload="polyglot substrate is up", evidence=(evidence_id,)
        )
        await graph.upsert_claim(evidence_claim)
        await graph.upsert_claim(derived_claim)

        fetched = await graph.get_claim("spike1:claim")
        assert fetched is not None
        assert fetched.payload == "polyglot substrate is up"
        # S5 provenance survives the round-trip.
        assert fetched.provenance.evidence == (evidence_id,)
        neighbors = await graph.neighbors("spike1:claim")
        assert evidence_id in {n.id for n in neighbors}  # DERIVED_FROM edge is a real traversal.

        # --- pgvector: dense recall returns the nearest record first with a bounded score. ---
        await latent.ensure_schema()
        await latent.reset()
        await latent.upsert(LatentRecord(id="near", embedding=(1.0, 0.0, 0.0, 0.0)))
        await latent.upsert(LatentRecord(id="far", embedding=(0.0, 1.0, 0.0, 0.0)))
        await latent.upsert(LatentRecord(id="mid", embedding=(1.0, 1.0, 0.0, 0.0)))
        matches = await latent.search((1.0, 0.0, 0.0, 0.0), k=3)
        assert tuple(m.record.id for m in matches) == ("near", "mid", "far")
        assert matches[0].score >= matches[1].score >= matches[2].score
        assert all(-1.0 <= m.score <= 1.0 for m in matches)
    finally:
        await journal.aclose()
        await graph.aclose()
        await latent.aclose()


# ==================================================================================================
# Criterion 2 — exactly-once resume on the REAL Timescale journal (S6): THE durability gate.
# ==================================================================================================


def _build_engine(journal: Journal, model: ReplayModel) -> Engine:
    # The S6 claim is about the JOURNAL, so keep graph/latent as in-memory doubles to isolate the
    # proof to durable Postgres rows. The chaos graph touches neither.
    return Engine(
        model=model,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
    )


def _register_chaos_graph(engine: Engine, run_id: str) -> None:
    # Seed the resuming engine's in-process graph map (in-proc resume is Phase-0 behaviour).
    engine._graphs[run_id] = build_chaos_graph()


async def _count_rows_for_key(dsn: str, idempotency_key: str) -> int:
    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        cursor = await conn.execute(
            "SELECT count(*) FROM cogworx_journal_steps WHERE idempotency_key = %s",
            (idempotency_key,),
        )
        row = await cursor.fetchone()
        assert row is not None
        return int(row[0])
    finally:
        await conn.close()


async def test_criterion_2_exactly_once_resume_on_real_timescale(
    settings: SubstrateSettings,
) -> None:
    """Kill mid-run after ``respond`` commits; resume over the REAL journal re-calls no model (S6).

    Proves exactly-once against durable Postgres rows: the harness asserts engine B made 0 model
    calls and the replayed StepRecords equal what was committed before the crash; the post-assert
    queries Postgres directly for exactly ONE row carrying the respond idempotency_key.
    """
    journal = TimescaleJournal(settings=settings)
    await journal.ensure_schema()
    await journal.reset()
    run_id = f"spike2-{uuid.uuid4().hex}"
    respond_key = f"{run_id}:respond"
    try:
        scripted = ReplayModel(
            [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
        )
        final = await assert_resume_never_recalls_model(
            build_engine=_build_engine,
            register_graph=_register_chaos_graph,
            graph_factory=build_chaos_graph,
            initial=_chaos_initial(),
            crash_after_stage="respond",
            shared_journal=journal,
            scripted_model=scripted,
            run_id=run_id,
            session_id="spike2-sess",
        )

        # The harness already proved: 0 model calls on resume + replayed steps byte-identical. Pin
        # the terminal outcome and that the model ran exactly once (during engine A, before crash).
        assert final.status is RunStatus.COMPLETED
        assert tuple(step.stage_name for step in final.steps) == ("intake", "respond", "close")
        assert scripted.call_count == 1

        # The close stage carried the SAME text respond produced before the crash — durable replay,
        # not recomputation.
        close_result = final.steps[-1].result
        assert isinstance(close_result, Done)
        assert close_result.output.data["text"] == "durable answer"

        # Exactly-once survived a REAL commit + crash + resume: one durable row for the respond key.
        rows = await _count_rows_for_key(settings.pg_dsn, respond_key)
        assert rows == 1
    finally:
        await journal.aclose()


# ==================================================================================================
# Criterion 3 — first Lesion pass (S8): a disabled component DEGRADES the loop, it does not crash.
# ==================================================================================================


class LesionableRecallStage:
    """A stage whose recall is a lesionable capability: degrade (not crash) when it is disabled.

    Dispatches ``latent_recall`` through the registry. With the capability ENABLED the dispatch
    succeeds and the stage completes (terminal ``Done``). With it DISABLED the dispatch raises a
    single ``CapabilityUnavailable`` and the stage catches that, returning a terminal ``Degraded``
    so the loop degrades gracefully (S8) rather than crashing — uniform regardless of WHY the
    capability is unavailable (no registry, unknown, or lesioned). This is the corrected S8 dispatch
    contract: a degradation-aware stage catches exactly one error type.
    """

    name: str = "recall"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        try:
            await ctx.dispatch("latent_recall", {"query": (1.0, 0.0, 0.0, 0.0)})
        except CapabilityUnavailable:
            return Degraded(
                reason="latent_recall lesioned",
                output=_artifact("recall-degraded"),
            )
        return Done(output=_artifact("recall-done"))


def _build_recall_graph() -> StageGraph:
    return StageGraph([LesionableRecallStage()], entry="recall")


async def _make_recall_engine(
    settings: SubstrateSettings,
    latent: PgLatentStore,
    journal: TimescaleJournal,
    events: list[Event],
) -> tuple[Engine, Registry]:
    async def latent_recall(query: tuple[float, ...]) -> tuple[str, ...]:
        # A genuine component lesion: this capability really hits pgvector when enabled.
        matches = await latent.search(query, k=3)
        return tuple(m.record.id for m in matches)

    registry = Registry()
    registry.register(
        function_capability(latent_recall, name="latent_recall", tier="read"),
        tags=("memory",),
    )
    engine = Engine(
        model=ReplayModel([]),  # the recall graph is model-free; any model call would exhaust.
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=latent,
        registry=registry,
        event_sink=events.append,
    )
    return engine, registry


async def test_criterion_3a_capability_enabled_completes(settings: SubstrateSettings) -> None:
    """With the recall capability ENABLED, the loop reaches a terminal NON-degraded status (S8)."""
    latent = PgLatentStore(dim=4, settings=settings)
    journal = TimescaleJournal(settings=settings)
    events: list[Event] = []
    engine, _registry = await _make_recall_engine(settings, latent, journal, events)
    try:
        await journal.ensure_schema()
        await journal.reset()
        await latent.ensure_schema()
        await latent.reset()
        await latent.upsert(LatentRecord(id="hit", embedding=(1.0, 0.0, 0.0, 0.0)))

        run_id = f"spike3a-{uuid.uuid4().hex}"
        state = await engine.run(
            run_id=run_id,
            session_id="spike3a-sess",
            graph=_build_recall_graph(),
            initial=_artifact("input"),
        )
        assert state.status is RunStatus.COMPLETED
        assert state.steps[-1].result.kind == "done"
    finally:
        await journal.aclose()
        await latent.aclose()


async def test_criterion_3b_lesioned_capability_degrades(settings: SubstrateSettings) -> None:
    """Disabling the recall capability DEGRADES the loop (no crash) and emits lesion events (S8).

    Asserts: (1) a ``LESION_ENABLED`` event (``OPERATIONS``) passes the boundary contract;
    (2) the run reaches ``RunStatus.DEGRADED`` — the loop degraded, it did NOT raise; (3) a
    ``STAGE_DEGRADED`` event was emitted through the engine's event sink.
    """
    latent = PgLatentStore(dim=4, settings=settings)
    journal = TimescaleJournal(settings=settings)
    events: list[Event] = []
    engine, registry = await _make_recall_engine(settings, latent, journal, events)
    try:
        await journal.ensure_schema()
        await journal.reset()
        await latent.ensure_schema()
        await latent.reset()

        # The S8 lesion switch: disable the component.
        registry.disable("latent_recall")

        # Emit a LESION_ENABLED event through a RunContext and prove it passes the boundary contract
        # (LESION_ENABLED is owned by OPERATIONS). The RunContext.emit also runs the boundary check.
        run_id = f"spike3b-{uuid.uuid4().hex}"
        lesion_event = Event(
            id=f"{run_id}:lesion:0",
            type=EventType.LESION_ENABLED,
            timestamp=_NOW,
            subsystem=Subsystem.OPERATIONS,
            session_id="spike3b-sess",
            run_id=run_id,
            attributes={"component": "latent_recall"},
        )
        validate_event_boundary(lesion_event)  # constructive proof: boundary contract holds.
        lesion_ctx = RunContext(
            run_id=run_id,
            session_id="spike3b-sess",
            model=ReplayModel([]),
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=latent,
            budget=BudgetGuard(),
            registry=registry,
            event_sink=events.append,
        )
        lesion_ctx.emit(lesion_event)

        # Run the graph: the disabled capability must DEGRADE the loop, not raise.
        state = await engine.run(
            run_id=run_id,
            session_id="spike3b-sess",
            graph=_build_recall_graph(),
            initial=_artifact("input"),
        )
        assert state.status is RunStatus.DEGRADED
        degraded_result = state.steps[-1].result
        assert isinstance(degraded_result, Degraded)
        assert degraded_result.to is None  # terminal degraded.

        # The lesion is observable on the event stream: both the operator's LESION_ENABLED and the
        # loop's STAGE_DEGRADED were emitted through the sink.
        emitted_types = {event.type for event in events}
        assert EventType.LESION_ENABLED in emitted_types
        assert EventType.STAGE_DEGRADED in emitted_types
    finally:
        await journal.aclose()
        await latent.aclose()
