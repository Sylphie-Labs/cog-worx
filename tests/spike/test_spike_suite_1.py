"""Spike Suite 1 — the falsifiable GATE that unblocks Phase 1 (CANON S12).

Three criteria, proven ON THE REAL POLYGLOT SUBSTRATE (Neo4j + Postgres/Timescale/pgvector), each
of which falsifies a load-bearing bet if it fails:

- **Criterion 1 (S3 — the polyglot bet):** all THREE real engines do real, independent work in ONE
  test. Timescale round-trips a run journal; Neo4j round-trips a provenance-bearing claim and walks
  its ``DERIVED_FROM`` lineage; pgvector returns nearest-first dense recall. Failure of any one
  engine fails the polyglot-capability criterion (they are asserted in the same test body).
- **Criterion 2 (S6 — THE durability gate):** the reusable kill-mid-run chaos harness
  (``assert_resume_never_recalls_model``) wired to a REAL ``TimescaleJournal``. A run is killed
  mid-flight after the model-bearing ``respond`` stage durably commits; a fresh engine SAME-PROCESS
  resume re-reads the run's committed step rows from Postgres and replays them, making ZERO model
  calls. A post-assert queries Postgres directly to prove exactly ONE row carries the respond
  idempotency_key — exactly-once and no-model-recall hold against durable rows, not memory. What
  this proves is in-process resume over durable Postgres rows: the graph object is still held
  in-process (``engine._graphs``). Durable CROSS-PROCESS cold resume (rebuilding the graph after a
  real process death) is Phase 1.
- **Criterion 3 (S8 — first Lesion pass):** disabling a real component (a ``latent_recall``
  capability backed by ``PgLatentStore``) makes the loop DEGRADE, not crash, and the ENGINE emits a
  ``STAGE_DEGRADED`` event through its sink — proof the loop witnessed the lesion, not the test.

These hit the live substrate; under ``-m spike`` (also ``-m integration``). They ERROR rather than
silently pass when the substrate is unreachable, since the gate is only meaningful when it is up.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Callable
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
from cogworx.coordination.events import Event, EventType
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Degraded, Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal, StepRecord
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import assert_resume_never_recalls_model
from cogworx.testing.reference_agent import IntakeStage

pytestmark = [pytest.mark.spike, pytest.mark.integration]

_SPIKE_CHAOS_PATHWAY_ID = "spike-chaos"
_SPIKE_RECALL_PATHWAY_ID = "spike-recall"


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
        # respond is the second stage in the chaos graph, committed at positional step_index 1.
        prior = await ctx.journal.read_step(ctx.run_id, 1)
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
        await journal.start_run(run_id, "spike1-sess", pathway_id="reference", pathway_version=1)
        transition_step = StepRecord(
            run_id=run_id,
            step_index=0,
            stage_name="intake",
            result=Transition(to="respond", output=_artifact("intake")),
            committed_at=_NOW,
        )
        done_step = StepRecord(
            run_id=run_id,
            step_index=1,
            stage_name="respond",
            result=Done(output=_artifact("response")),
            committed_at=_LATER,
        )
        await journal.commit_step(transition_step)
        await journal.commit_step(done_step)
        await journal.set_run_status(run_id, RunStatus.COMPLETED)

        read_back = await journal.read_step(run_id, 1)
        assert read_back is not None
        assert read_back.result == done_step.result  # StageResult round-trips through jsonb.
        run_state = await journal.load_run(run_id)
        assert run_state is not None
        assert run_state.status is RunStatus.COMPLETED  # the PERSISTED status is the authority
        assert tuple(step.step_index for step in run_state.steps) == (0, 1)
        assert tuple(step.stage_name for step in run_state.steps) == ("intake", "respond")

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


def _make_build_engine(pathways: PathwayRegistry) -> Callable[[Journal, ReplayModel], Engine]:
    # The S6 claim is about the JOURNAL, so keep graph/latent as in-memory doubles to isolate the
    # proof to durable Postgres rows. The chaos graph touches neither. Engine A and the FRESH engine
    # B share this factory's registry, so engine B rehydrates the graph from the run's stored
    # pathway pointer — true cold resume, no in-process graph carried.
    def build(journal: Journal, model: ReplayModel) -> Engine:
        return Engine(
            model=model,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
        )

    return build


def _chaos_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_SPIKE_CHAOS_PATHWAY_ID, build_chaos_graph())
    return registry


async def _count_rows_for_position(dsn: str, run_id: str, step_index: int) -> int:
    conn = await psycopg.AsyncConnection.connect(dsn, autocommit=True)
    try:
        cursor = await conn.execute(
            "SELECT count(*) FROM cogworx_journal_steps WHERE run_id = %s AND step_index = %s",
            (run_id, step_index),
        )
        row = await cursor.fetchone()
        assert row is not None
        return int(row[0])
    finally:
        await conn.close()


async def test_criterion_2_exactly_once_resume_on_real_timescale(
    settings: SubstrateSettings,
) -> None:
    """Kill mid-run after ``respond`` commits; a FRESH engine cold-resumes, re-calls no model (S6).

    Proves exactly-once against durable Postgres rows: the harness asserts engine B made 0 model
    calls and the replayed StepRecords equal what was committed before the crash; the post-assert
    queries Postgres directly for exactly ONE row at the respond position. Engine B is a FRESH
    engine that rehydrates the graph from its ``PathwayRegistry`` via the run's stored pathway
    pointer — durable cold resume off the durable Postgres rows.
    """
    journal = TimescaleJournal(settings=settings)
    await journal.ensure_schema()
    await journal.reset()
    run_id = f"spike2-{uuid.uuid4().hex}"
    try:
        scripted = ReplayModel(
            [ModelResponse(text="durable answer", model_id="replay", finish_reason="stop")]
        )
        final = await assert_resume_never_recalls_model(
            build_engine=_make_build_engine(_chaos_pathways()),
            initial=_chaos_initial(),
            pathway_id=_SPIKE_CHAOS_PATHWAY_ID,
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

        # Exactly-once survived a REAL commit + crash + resume: one durable row at the respond
        # position (step_index 1 in the chaos graph).
        rows = await _count_rows_for_position(settings.pg_dsn, run_id, 1)
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
    pathways = PathwayRegistry()
    pathways.register(_SPIKE_RECALL_PATHWAY_ID, _build_recall_graph())
    engine = Engine(
        model=ReplayModel([]),  # the recall graph is model-free; any model call would exhaust.
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=latent,
        pathways=pathways,
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
            pathway_id=_SPIKE_RECALL_PATHWAY_ID,
            initial=_artifact("input"),
        )
        assert state.status is RunStatus.COMPLETED
        assert state.steps[-1].result.kind == "done"
    finally:
        await journal.aclose()
        await latent.aclose()


async def test_criterion_3b_lesioned_capability_degrades(settings: SubstrateSettings) -> None:
    """Disabling the recall capability DEGRADES the loop (no crash) and the ENGINE emits the lesion.

    Asserts only things the ENGINE produces BECAUSE the capability was lesioned: (1) the run reaches
    ``RunStatus.DEGRADED`` — the loop degraded, it did NOT raise; (2) the loop emitted
    ``STAGE_DEGRADED`` through its event sink. (An earlier revision also asserted a
    ``LESION_ENABLED`` event that the TEST itself constructed and emitted — that proved nothing
    about engine behaviour, so it was removed.)
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

        # The S8 lesion switch: disable the component so its dispatch raises CapabilityUnavailable.
        registry.disable("latent_recall")

        # Run the graph: the disabled capability must DEGRADE the loop, not raise.
        run_id = f"spike3b-{uuid.uuid4().hex}"
        state = await engine.run(
            run_id=run_id,
            session_id="spike3b-sess",
            pathway_id=_SPIKE_RECALL_PATHWAY_ID,
            initial=_artifact("input"),
        )
        assert state.status is RunStatus.DEGRADED
        degraded_result = state.steps[-1].result
        assert isinstance(degraded_result, Degraded)
        assert degraded_result.to is None  # terminal degraded.

        # The lesion is observable on the event stream: the LOOP emitted STAGE_DEGRADED through the
        # sink (the engine produced this, not the test).
        emitted_types = {event.type for event in events}
        assert EventType.STAGE_DEGRADED in emitted_types
    finally:
        await journal.aclose()
        await latent.aclose()
