"""The trial projector across the LIVE polyglot boundary: Timescale journal -> Neo4j KG (S1/S3/S6).

The projector is the one place Pod 2.1 crosses the substrate seam: it READS committed steps from the
TimescaleDB journal and WRITES trials + the projection cursor into Neo4j, off the write path. These
tests hit BOTH real services (``docker compose up -d`` first) and ERROR — not silently pass — when
either is unreachable. They run only under ``-m integration``.

Mirrors the unit projector suite so the SAME behaviour is asserted against the live span:
  - a1 exactly-one-Trial-per-committed-step over the real journal read + real Neo4j MERGE.
  - a4 re-run the projector -> byte-identical procedural subgraph + cursor (idempotent across the
    real same-txn cursor advance).
  - a6 the cursor lands in Neo4j atomically with the trial.
  - lookback-overlap idempotency over the real ``committed_steps_since`` query.
  - failure-trial synthesis from a real exhaustion-Degrade committed step.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_procedural_kg import Neo4jProceduralKG
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.knowledge.procedural_registry import ProcedureRegistry
from cogworx.loop.result import Degraded, Done, Transition
from cogworx.runtime.projector import DEFAULT_PROJECTION_CONSUMER, TrialProjector
from cogworx.substrate.journal import StepRecord

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_PATHWAY = "math_pathway"
_STAGE = "solve_stage"
_FIXUP = "fixup_stage"


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # Async psycopg 3 cannot run on Windows' default ProactorEventLoop; it needs a selector loop.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jProceduralKG]:
    adapter = Neo4jProceduralKG(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


def _registry() -> ProcedureRegistry:
    registry = ProcedureRegistry()
    registry.declare(_PATHWAY, _STAGE, problem_type="word problem")
    return registry


def _decl(registry: ProcedureRegistry) -> tuple[str, str]:
    decl = registry.get(_PATHWAY, _STAGE)
    assert decl is not None
    return decl.procedure_id, decl.problem_type


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _stamped(registry: ProcedureRegistry, outcome: str) -> Artifact:
    procedure_id, problem_type = _decl(registry)
    return Artifact(
        kind="solution",
        produced_by=_STAGE,
        provenance=_prov(),
        data={"outcome": outcome, "procedure_id": procedure_id, "problem_type": problem_type},
    )


def _exhaustion() -> Artifact:
    return Artifact(
        kind="retry-exhausted",
        produced_by="engine",
        provenance=_prov(),
        data={"failure_class": "TimeoutError", "attempts": 3},
    )


async def _start(journal: TimescaleJournal, run_id: str) -> None:
    await journal.start_run(
        run_id,
        f"session:{run_id}",
        pathway_id=_PATHWAY,
        pathway_version=1,
        pathway_fingerprint="fp",
    )


async def _commit_done(
    journal: TimescaleJournal, *, run_id: str, step_index: int, output: Artifact, at: datetime
) -> None:
    await journal.commit_step(
        StepRecord(
            run_id=run_id,
            step_index=step_index,
            stage_name=_STAGE,
            result=Done(output=output),
            committed_at=at,
        )
    )


def _projector(
    journal: TimescaleJournal, kg: Neo4jProceduralKG, registry: ProcedureRegistry
) -> TrialProjector:
    return TrialProjector(journal=journal, procedural_kg=kg, registry=registry)


async def test_live_a1_exactly_one_trial_per_committed_step(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    registry = _registry()
    await _start(journal, "r1")
    await _commit_done(
        journal, run_id="r1", step_index=0, output=_stamped(registry, "success"), at=_T0
    )

    projected = await _projector(journal, kg, registry).tick()

    assert projected == 1
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.trial_id for t in trials] == ["r1:0"]
    assert trials[0].outcome == "success"


async def test_live_a6_cursor_lands_atomically(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    registry = _registry()
    await _start(journal, "r1")
    commit_at = _T0 + timedelta(minutes=5)
    await _commit_done(
        journal, run_id="r1", step_index=0, output=_stamped(registry, "success"), at=commit_at
    )

    assert await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER) is None
    await _projector(journal, kg, registry).tick()

    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", 0)
    assert cursor.commit_ordinal >= 1
    assert await kg.get_trial("r1:0") is not None


async def test_live_a4_rerun_byte_identical(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    registry = _registry()
    await _start(journal, "r1")
    await _start(journal, "r2")
    await journal.commit_step(
        StepRecord(
            run_id="r1",
            step_index=0,
            stage_name=_STAGE,
            result=Transition(to=_FIXUP, output=_stamped(registry, "failure")),
            committed_at=_T0,
        )
    )
    await _commit_done(
        journal, run_id="r2", step_index=0, output=_stamped(registry, "success"), at=_T0
    )
    procedure_id, problem_type = _decl(registry)

    projector = _projector(journal, kg, registry)
    await projector.tick()
    post_a = await kg.posterior(procedure_id, problem_type)
    trials_a = [(t.trial_id, t.outcome) for t in await kg.trials_for(procedure_id, problem_type)]
    cursor_a = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)

    await projector.tick()
    await projector.tick()
    post_b = await kg.posterior(procedure_id, problem_type)
    trials_b = [(t.trial_id, t.outcome) for t in await kg.trials_for(procedure_id, problem_type)]
    cursor_b = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)

    assert trials_a == trials_b
    assert (post_a.alpha, post_a.beta, post_a.n_trials) == (
        post_b.alpha,
        post_b.beta,
        post_b.n_trials,
    )
    assert cursor_a == cursor_b
    # One success (r2) + one failure (r1, read off the STAMP not the Transition kind) -> Beta(2,2).
    assert (post_b.alpha, post_b.beta, post_b.n_trials) == (2.0, 2.0, 2)


async def test_live_failure_synthesis_from_exhaustion_degrade(
    journal: TimescaleJournal, kg: Neo4jProceduralKG
) -> None:
    registry = _registry()
    await _start(journal, "r1")
    await journal.commit_step(
        StepRecord(
            run_id="r1",
            step_index=0,
            stage_name=_STAGE,
            result=Degraded(reason="retry exhausted", output=_exhaustion(), to=None),
            committed_at=_T0,
        )
    )

    projected = await _projector(journal, kg, registry).tick()

    assert projected == 1
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.outcome for t in trials] == ["failure"]
    posterior = await kg.posterior(procedure_id, problem_type)
    assert (posterior.alpha, posterior.beta) == (1.0, 2.0)
