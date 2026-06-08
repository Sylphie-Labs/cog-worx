"""Integration tests for the real Postgres substrate adapters (CANON S3, S6).

These hit a live Postgres cluster hosting BOTH ``timescaledb`` and ``vector``. They are marked
``integration`` (deselected unless ``-m integration``) and prove the real adapters are behaviourally
equivalent to the in-memory doubles — same exactly-once idempotency, same status derivation, same
nearest-first search ordering. When the substrate is unreachable these ERROR rather than silently
pass, since they only run when the stack is confirmed up.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.pg_latent import PgLatentStore
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.state import RunStatus
from cogworx.substrate.journal import StepRecord
from cogworx.substrate.latent import LatentRecord

_PATHWAY_ID = "reference"

pytestmark = pytest.mark.integration


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    # Async psycopg 3 cannot run on Windows' default ProactorEventLoop; it needs a selector loop.
    # This pins the integration tier's loop policy without touching the shared conftest.
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_LATER = datetime(2026, 1, 1, 0, 0, 1, tzinfo=UTC)


def _artifact() -> Artifact:
    return Artifact(
        kind="t",
        produced_by="t",
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_NOW),
    )


def _step(
    step_index: int, result: StageResult, *, committed_at: datetime, stage_name: str | None = None
) -> StepRecord:
    return StepRecord(
        run_id="r1",
        step_index=step_index,
        stage_name=stage_name if stage_name is not None else f"stage-{step_index}",
        result=result,
        committed_at=committed_at,
    )


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
async def latent(settings: SubstrateSettings) -> AsyncIterator[PgLatentStore]:
    adapter = PgLatentStore(dim=4, settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


async def test_commit_step_exactly_once(journal: TimescaleJournal) -> None:
    await journal.start_run("r1", "s1", pathway_id=_PATHWAY_ID, pathway_version=1)
    record = _step(0, Transition(to="b", output=_artifact()), committed_at=_NOW, stage_name="a")
    await journal.commit_step(record)
    await journal.commit_step(record)

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1
    assert state.steps[0].step_index == 0


async def test_commit_step_idempotent_on_position_with_differing_fields(
    journal: TimescaleJournal,
) -> None:
    """A SECOND commit at the SAME position is a silent no-op, not a UNIQUE error (S6).

    This targets the ``ON CONFLICT (run_id, step_index) DO NOTHING`` clause specifically: the two
    records share ``(run_id, step_index)`` but differ in EVERY other column (stage_name, result,
    committed_at). The second commit must (a) not raise and (b) leave exactly ONE row at that
    position (the FIRST write wins) — exactly-once is on the POSITION, not on the stage name.
    """
    await journal.start_run("r1", "s1", pathway_id=_PATHWAY_ID, pathway_version=1)
    first = _step(0, Transition(to="b", output=_artifact()), committed_at=_NOW, stage_name="a")
    second = _step(0, Done(output=_artifact()), committed_at=_LATER, stage_name="z")

    await journal.commit_step(first)
    await journal.commit_step(second)  # same position, all other fields differ.

    conn = await journal._connection()
    cursor = await conn.execute(
        "SELECT count(*), min(stage_name) FROM cogworx_journal_steps "
        "WHERE run_id = %s AND step_index = %s",
        ("r1", 0),
    )
    row = await cursor.fetchone()
    assert row is not None
    assert row[0] == 1  # exactly one durable row at the position
    assert row[1] == "a"  # the FIRST commit won; the conflicting second was dropped

    # And the run-level view shows a single step, not two.
    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1


async def test_read_step_round_trips_stage_result(journal: TimescaleJournal) -> None:
    await journal.start_run("r1", "s1", pathway_id=_PATHWAY_ID, pathway_version=1)
    original = _step(0, Transition(to="b", output=_artifact()), committed_at=_NOW, stage_name="a")
    await journal.commit_step(original)

    loaded = await journal.read_step("r1", 0)
    assert loaded is not None
    assert loaded == original


async def test_load_run_persisted_status_with_ordered_steps(
    journal: TimescaleJournal,
) -> None:
    await journal.start_run("r1", "s1", pathway_id=_PATHWAY_ID, pathway_version=1)
    await journal.commit_step(
        _step(
            0, Transition(to="respond", output=_artifact()), committed_at=_NOW, stage_name="intake"
        )
    )
    await journal.commit_step(
        _step(1, Done(output=_artifact()), committed_at=_LATER, stage_name="respond")
    )
    await journal.set_run_status("r1", RunStatus.COMPLETED)

    state = await journal.load_run("r1")
    assert state is not None
    assert state.status is RunStatus.COMPLETED  # the PERSISTED status is the authority
    assert state.current_stage == "respond"
    assert tuple(step.step_index for step in state.steps) == (0, 1)
    assert tuple(step.stage_name for step in state.steps) == ("intake", "respond")


async def test_load_run_unknown_is_none(journal: TimescaleJournal) -> None:
    assert await journal.load_run("missing") is None


async def test_latent_search_returns_nearest_first(latent: PgLatentStore) -> None:
    await latent.upsert(LatentRecord(id="near", embedding=(1.0, 0.0, 0.0, 0.0)))
    await latent.upsert(LatentRecord(id="far", embedding=(0.0, 1.0, 0.0, 0.0)))
    await latent.upsert(LatentRecord(id="mid", embedding=(1.0, 1.0, 0.0, 0.0)))

    matches = await latent.search((1.0, 0.0, 0.0, 0.0), k=3)
    assert tuple(m.record.id for m in matches) == ("near", "mid", "far")
    assert matches[0].score >= matches[1].score >= matches[2].score
    assert all(-1.0 <= m.score <= 1.0 for m in matches)


async def test_latent_upsert_replaces_by_id(latent: PgLatentStore) -> None:
    await latent.upsert(LatentRecord(id="x", embedding=(1.0, 0.0, 0.0, 0.0)))
    await latent.upsert(LatentRecord(id="x", embedding=(0.0, 1.0, 0.0, 0.0)))

    matches = await latent.search((0.0, 1.0, 0.0, 0.0), k=5)
    assert len(matches) == 1
    assert matches[0].record.id == "x"
