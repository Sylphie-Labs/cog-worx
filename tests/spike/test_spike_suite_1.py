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
  idempotency_key — exactly-once and no-model-recall hold against durable rows, not memory. Cold
  resume rehydrates the graph from the ``PathwayRegistry`` via the run's stored ``pathway_id`` (no
  in-process graph is carried); a structural ``pathway_fingerprint`` stored at ``start_run`` guards
  the rehydrated graph against pathway drift before resume re-drives.
- **Criterion 3 (S8 — first Lesion pass):** disabling a real component (a ``latent_recall``
  capability backed by ``PgLatentStore``) makes the loop DEGRADE, not crash, and the ENGINE emits a
  ``STAGE_DEGRADED`` event through its sink — proof the loop witnessed the lesion, not the test.
- **Criterion 4 (S6 — cyclic per-visit durability + COLD resume on durable rows):** the
  genuinely-new Phase-1 durability claim. A CYCLIC ``intake -> work -> gate`` pathway (``gate``
  loops back to ``work`` a fixed number of times, then ``Done``) is driven on a REAL
  ``TimescaleJournal``, killed AFTER the SECOND ``work`` visit durably commits, then COLD-resumed by
  a brand-new ``Engine`` + the SAME ``PathwayRegistry`` over the SAME Postgres rows. Proves: the
  model-bearing ``work`` stage was visited multiple times producing MULTIPLE durable rows at
  DISTINCT ``step_index`` with the SAME ``stage_name="work"`` (per-visit positional keying on
  durable rows, not a stage-name key); resume re-calls NO model (every committed ``work`` visit is
  replayed positionally from Postgres, never recomputed); and the replayed prefix is byte-identical
  to what engine A committed before the crash. This is the durable counterpart of the deterministic
  ``test_durability_chaos`` cyclic cold-resume — the cycle's per-visit exactly-once now holds
  against real Timescale rows across a fresh engine.

These hit the live substrate; under ``-m spike`` (also ``-m integration``). They ERROR rather than
silently pass when the substrate is unreachable, since the gate is only meaningful when it is up.
"""

from __future__ import annotations

import asyncio
import sys
import uuid
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

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
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import (
    Journal,
    ProjectedStep,
    ProjectionCursor,
    RunState,
    StepRecord,
    Timer,
)
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    SimulatedCrash,
    assert_resume_never_recalls_model,
)
from cogworx.testing.reference_agent import IntakeStage

pytestmark = [pytest.mark.spike, pytest.mark.integration]

_SPIKE_CHAOS_PATHWAY_ID = "spike-chaos"
_SPIKE_RECALL_PATHWAY_ID = "spike-recall"
_SPIKE_CYCLE_PATHWAY_ID = "spike-cycle"


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
        await journal.start_run(
            run_id,
            "spike1-sess",
            pathway_id="reference",
            pathway_version=1,
            pathway_fingerprint="fp",
        )
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
        await latent.put(LatentRecord(id="near", embedding=(1.0, 0.0, 0.0, 0.0)))
        await latent.put(LatentRecord(id="far", embedding=(0.0, 1.0, 0.0, 0.0)))
        await latent.put(LatentRecord(id="mid", embedding=(1.0, 1.0, 0.0, 0.0)))
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
        registry = ModelRegistry()
        registry.register("default", model)
        return Engine(
            models=registry,
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
            # A list, not a tuple: the args are validated as JSON (S9), and JSON has no tuple.
            await ctx.dispatch("latent_recall", {"query": [1.0, 0.0, 0.0, 0.0]})
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
    _recall_model = ReplayModel([])  # the recall graph is model-free; any model call would exhaust.
    _recall_reg = ModelRegistry()
    _recall_reg.register("default", _recall_model)
    engine = Engine(
        models=_recall_reg,
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
        await latent.put(LatentRecord(id="hit", embedding=(1.0, 0.0, 0.0, 0.0)))

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


# ==================================================================================================
# Criterion 4 — cyclic per-visit durability + COLD resume on durable rows (S6): the genuinely-new
# Phase-1 claim. A cyclic refine-back pathway is killed mid-cycle and cold-resumed by a FRESH
# engine.
# ==================================================================================================

# How many times ``work`` must run before ``gate`` returns ``Done``. The crash fires AFTER the 2nd
# (final) ``work`` visit durably commits — so the committed prefix already holds EVERY model-bearing
# visit. Resume runs only the uncommitted ``gate`` (no model), which sees both ``work`` rows and
# terminates: cold resume completes with ZERO model calls, every ``work`` visit replayed from
# Postgres. The two ``work`` rows sit at DISTINCT, non-first ``step_index`` (1 and 3) — the
# per-visit durable-keying surface. (A LATER crash, e.g. after a 2nd-of-3 visit, would force a
# 3rd ``work`` and re-call the model; that is a different, weaker claim, so the crash is pinned to
# the final visit to isolate the no-recall proof.)
_WORK_TARGET_VISITS = 2


def _cycle_artifact(kind: str, *, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by="spike-cycle",
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=datetime.now(UTC)),
        data={"text": text} if text else {},
    )


def _work_text(step: StepRecord) -> str:
    # A committed ``work`` step is always a ``Transition`` (work -> gate); narrow the StageResult
    # union so the typed ``output.data`` access is sound.
    result = step.result
    assert isinstance(result, Transition)
    return str(result.output.data["text"])


async def _committed_work_count(ctx: StageContext) -> int:
    # Count the ``work`` steps DURABLY committed so far by reading the run's journal rows. This is
    # the only state ``gate`` decides on (CANON S9: control flow is a function of committed
    # StageResults + graph, never of the model's words). It is replay-safe: on resume ``gate``'s own
    # committed result is read back, never recomputed, but were it recomputed it would see the same
    # rows.
    state = await ctx.journal.load_run(ctx.run_id)
    if state is None:
        return 0
    return sum(1 for step in state.steps if step.stage_name == "work")


class _CycleIntakeStage:
    """No-model entry stage that hands control INTO the cycle (transitions to ``work``)."""

    name: str = "intake"
    transitions: tuple[str, ...] = ("work",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="work", output=_cycle_artifact("intake"))


class _WorkStage:
    """The model-bearing stage of the cycle: exactly one ``ctx.model.complete`` per visit.

    Revisited each time ``gate`` loops back, so it commits a DISTINCT positional ``step_index`` per
    visit while keeping ``stage_name == "work"`` — the per-visit-keying surface under test.
    """

    name: str = "work"
    transitions: tuple[str, ...] = ("gate",)

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Do one unit of work.")]
        )
        ctx.budget.record(response.usage)
        return Transition(to="gate", output=_cycle_artifact("work", text=response.text or ""))


class _GateStage:
    """The controller: loop to ``work`` until it has run ``_WORK_TARGET_VISITS`` times, then Done.

    Decides DETERMINISTICALLY by counting prior ``work`` rows in the journal — no model call, no
    self-report. Declares edges to BOTH ``work`` (the refine-back cycle) and ``end`` (so the graph
    passes the by-construction termination check). Returns ``Done`` itself when the target is met.
    """

    name: str = "gate"
    transitions: tuple[str, ...] = ("work", "end")

    async def run(self, ctx: StageContext) -> StageResult:
        work_visits = await _committed_work_count(ctx)
        if work_visits >= _WORK_TARGET_VISITS:
            return Done(output=_cycle_artifact("gate-done", text=f"work ran {work_visits}x"))
        return Transition(to="work", output=_cycle_artifact("gate-loop"))


class _CycleEndStage:
    """A terminal stage for the reachability check; ``gate`` returns Done before reaching it."""

    name: str = "end"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_cycle_artifact("end"))


def _build_cycle_graph() -> StageGraph:
    # intake -> work -> gate, where gate loops back to work until the target visit count, then Done;
    # ``end`` is a reachable terminal so the graph passes the by-construction termination check.
    return StageGraph(
        [_CycleIntakeStage(), _WorkStage(), _GateStage(), _CycleEndStage()], entry="intake"
    )


def _cycle_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_SPIKE_CYCLE_PATHWAY_ID, _build_cycle_graph())
    return registry


def _cycle_initial() -> Artifact:
    return _cycle_artifact("user-input", text="please refine")


class _CrashAfterStageVisitJournal:
    """Like ``CrashAfterStepJournal`` but crashes after the Nth VISIT of a stage (S6, cyclic).

    ``CrashAfterStepJournal`` fires on the FIRST commit whose ``stage_name`` matches — it cannot
    express "crash after the 2nd ``work`` visit" in a cyclic pathway. This wrapper durably commits
    every step to the inner journal (so the prefix is exactly what a real crash would leave), counts
    commits of ``crash_after_stage``, and raises ``SimulatedCrash`` only once that count reaches
    ``crash_after_visit`` — modelling the process dying after a LATER cyclic visit committed.
    """

    def __init__(self, *, inner: Journal, crash_after_stage: str, crash_after_visit: int) -> None:
        self._inner = inner
        self._crash_after_stage = crash_after_stage
        self._crash_after_visit = crash_after_visit
        self._visits = 0

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        await self._inner.start_run(
            run_id,
            session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint,
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._inner.set_run_status(run_id, status)

    async def commit_step(self, record: StepRecord) -> None:
        await self._inner.commit_step(record)
        if record.stage_name == self._crash_after_stage:
            self._visits += 1
            if self._visits == self._crash_after_visit:
                raise SimulatedCrash(
                    f"simulated crash after visit {self._visits} of stage "
                    f"{record.stage_name!r} (step_index {record.step_index})"
                )

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

    async def committed_steps_after(
        self, cursor: ProjectionCursor | None, *, limit: int
    ) -> Sequence[ProjectedStep]:
        return await self._inner.committed_steps_after(cursor, limit=limit)

    async def load_run(self, run_id: str) -> RunState | None:
        return await self._inner.load_run(run_id)

    async def set_timer(self, timer: Timer) -> None:
        await self._inner.set_timer(timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return await self._inner.due_timers(now)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        return await self._inner.claim_due_timers(now, lease_ttl=lease_ttl)

    async def cancel_timer(self, timer_id: str) -> None:
        await self._inner.cancel_timer(timer_id)

    async def cancel_timers_for_run(self, run_id: str) -> None:
        await self._inner.cancel_timers_for_run(run_id)

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return await self._inner.get_run_status(run_id)

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.increment_attempt(run_id, step_index)

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.read_attempt(run_id, step_index)

    async def record_human_input(self, run_id: str, step_index: int, answer: Artifact) -> None:
        await self._inner.record_human_input(run_id, step_index, answer)

    async def read_human_input(self, run_id: str, step_index: int) -> Artifact | None:
        return await self._inner.read_human_input(run_id, step_index)

    async def set_run_tainted(self, run_id: str) -> None:
        await self._inner.set_run_tainted(run_id)

    async def append_design_look(
        self,
        *,
        planning_variance_config_hash: str,
        design_lineage_chain: Sequence[str],
        fingerprint: str,
    ) -> None:
        await self._inner.append_design_look(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,
            fingerprint=fingerprint,
        )

    async def read_design_lineage_budget(
        self,
        *,
        planning_variance_config_hash: str,
        design_lineage_chain: Sequence[str],
    ) -> int:
        return await self._inner.read_design_lineage_budget(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,
        )


async def test_criterion_4_cyclic_per_visit_durability_cold_resume(
    settings: SubstrateSettings,
) -> None:
    """A cyclic pathway killed after the 2nd ``work`` visit cold-resumes exactly-once, no recall.

    (S6.) Engine A drives ``intake -> work -> gate`` (gate loops back to work) on a REAL
    ``TimescaleJournal`` until ``work`` has committed twice, then a simulated crash fires. A
    BRAND-NEW engine B with the SAME ``PathwayRegistry`` and a ZERO-response model cold-resumes off
    the durable Postgres rows.

    Asserts (each is the deterministic falsifier of a distinct property):
    1. resume reaches ``RunStatus.COMPLETED`` with engine B ``call_count == 0`` — every committed
       ``work`` visit replayed positionally from Postgres, never recomputed (kill the replay branch
       and the zero-response model raises).
    2. the journal holds >= 2 ``work`` rows at DISTINCT ``step_index`` — per-visit positional keying
       on durable rows (break the key and the visits collide).
    3. the replayed prefix is byte-identical to what engine A committed before the crash.
    """
    journal = TimescaleJournal(settings=settings)
    await journal.ensure_schema()
    await journal.reset()
    run_id = f"spike4-{uuid.uuid4().hex}"
    pathways = _cycle_pathways()  # the SAME registry threaded into BOTH engines (cold resume).
    try:
        # --- Engine A: drive the cycle on REAL Timescale; crash after the 2nd ``work`` commits. ---
        scripted = ReplayModel(
            [
                ModelResponse(text="work-1", model_id="replay", finish_reason="stop"),
                ModelResponse(text="work-2", model_id="replay", finish_reason="stop"),
                # A 3rd response is present so a BROKEN replay (engine B re-running work) would not
                # be masked by exhaustion on engine A — engine A consumes only the first two.
                ModelResponse(text="work-3", model_id="replay", finish_reason="stop"),
            ]
        )
        crash_journal = _CrashAfterStageVisitJournal(
            inner=journal, crash_after_stage="work", crash_after_visit=2
        )
        _reg_a = ModelRegistry()
        _reg_a.register("default", scripted)
        engine_a = Engine(
            models=_reg_a,
            journal=crash_journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
        )
        crashed = False
        try:
            await engine_a.run(
                run_id=run_id,
                session_id="spike4-sess",
                pathway_id=_SPIKE_CYCLE_PATHWAY_ID,
                initial=_cycle_initial(),
            )
        except SimulatedCrash:
            crashed = True
        assert crashed, "engine A did not crash; the cycle completed before the 2nd work visit"
        # The model ran exactly twice on engine A (the two committed work visits before the crash).
        assert scripted.call_count == 2

        # The durable prefix at crash: intake, work(1), gate, work(2). work sits at two DISTINCT,
        # non-zero, non-first positional indices with the SAME stage_name — the per-visit proof.
        pre_resume = await journal.load_run(run_id)
        assert pre_resume is not None
        assert pre_resume.status is RunStatus.RUNNING  # crashed mid-flight, not terminal.
        committed_before = pre_resume.steps
        pre_work = [s for s in committed_before if s.stage_name == "work"]
        assert len(pre_work) == 2
        pre_work_indices = [s.step_index for s in pre_work]
        assert len(set(pre_work_indices)) == 2  # distinct positions, not a single colliding row.
        assert all(i > 0 for i in pre_work_indices)  # neither work visit landed at step 0.

        # --- Engine B: a FRESH engine, SAME registry + SAME real journal, ZERO-response model. ---
        zero_model = ReplayModel([])  # any model re-call on resume raises ReplayExhaustedError.
        _reg_b = ModelRegistry()
        _reg_b.register("default", zero_model)
        engine_b = Engine(
            models=_reg_b,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
        )
        assert engine_b is not engine_a
        final = await engine_b.resume(run_id)

        # (1) Cold resume completed making ZERO model calls — every committed work visit replayed.
        assert final.status is RunStatus.COMPLETED
        assert zero_model.call_count == 0

        # (2) Durable rows hold MULTIPLE work steps at DISTINCT step_index (per-visit keying proof).
        final_work = [s for s in final.steps if s.stage_name == "work"]
        assert len(final_work) >= 2
        work_indices = [s.step_index for s in final_work]
        assert len(set(work_indices)) == len(work_indices)  # all distinct positions.
        assert work_indices == sorted(work_indices)  # monotonic positional order on durable rows.
        # The full cycle ran the model-bearing stage exactly _WORK_TARGET_VISITS times.
        assert len(final_work) == _WORK_TARGET_VISITS

        # (3) The replayed prefix is byte-identical to what engine A committed before the crash.
        replayed_prefix = final.steps[: len(committed_before)]
        assert tuple(replayed_prefix) == tuple(committed_before)

        # The two pre-crash work rows carry the SAME text engine A's model produced — durable
        # replay, not recomputation (a re-run zero-response model could not reproduce these).
        assert [_work_text(s) for s in pre_work] == ["work-1", "work-2"]
        # And the resumed run kept those exact rows at those exact indices.
        resumed_pre_work = [
            s for s in final.steps[: len(committed_before)] if s.stage_name == "work"
        ]
        assert [s.step_index for s in resumed_pre_work] == pre_work_indices
        assert [_work_text(s) for s in resumed_pre_work] == ["work-1", "work-2"]
    finally:
        await journal.aclose()


# ==================================================================================================
# Criterion 4b — the step ceiling FAILS an unbounded cycle on the LIVE Timescale journal
# (S6/S9/S11). A live-substrate echo of deterministic ``test_step_ceiling`` (which owns the logic).
# ==================================================================================================


class _LivePingStage:
    name: str = "ping"
    transitions: tuple[str, ...] = ("pong", "stop")

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="pong", output=_cycle_artifact("ping"))


class _LivePongStage:
    name: str = "pong"
    transitions: tuple[str, ...] = ("ping",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="ping", output=_cycle_artifact("pong"))


class _LiveStopStage:
    name: str = "stop"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_cycle_artifact("stop"))


_SPIKE_RUNAWAY_PATHWAY_ID = "spike-runaway"


async def test_criterion_4b_step_ceiling_fails_runaway_on_real_timescale(
    settings: SubstrateSettings,
) -> None:
    """An unbounded cycle at a small ``max_steps`` persists ``FAILED`` in Postgres with N rows."""
    journal = TimescaleJournal(settings=settings)
    await journal.ensure_schema()
    await journal.reset()
    run_id = f"spike4b-{uuid.uuid4().hex}"
    max_steps = 6
    pathways = PathwayRegistry()
    pathways.register(
        _SPIKE_RUNAWAY_PATHWAY_ID,
        StageGraph([_LivePingStage(), _LivePongStage(), _LiveStopStage()], entry="ping"),
    )
    _runaway_model = ReplayModel([])  # the cycle is model-free; any model call would exhaust.
    _runaway_reg = ModelRegistry()
    _runaway_reg.register("default", _runaway_model)
    engine = Engine(
        models=_runaway_reg,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        max_steps=max_steps,
    )
    try:
        state = await engine.run(
            run_id=run_id,
            session_id="spike4b-sess",
            pathway_id=_SPIKE_RUNAWAY_PATHWAY_ID,
            initial=_cycle_initial(),
        )
        # The PERSISTED status in Postgres is FAILED — the ceiling stopped the runaway, no hang.
        reloaded = await journal.load_run(run_id)
        assert reloaded is not None
        assert reloaded.status is RunStatus.FAILED
        assert state.status is RunStatus.FAILED
        # Exactly ``max_steps`` distinct positional rows despite revisiting two stage names.
        assert len(reloaded.steps) == max_steps
        indices = [s.step_index for s in reloaded.steps]
        assert indices == list(range(max_steps))  # distinct, monotonic positions on durable rows.
    finally:
        await journal.aclose()
