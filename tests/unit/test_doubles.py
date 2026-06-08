"""Contract tests for the in-memory substrate doubles (S3, S6 reference behaviour)."""

from __future__ import annotations

from datetime import UTC, datetime

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.state import RunStatus
from cogworx.substrate.journal import StepRecord
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import InMemoryJournal, InMemoryLatentStore

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _artifact() -> Artifact:
    return Artifact(
        kind="t",
        produced_by="t",
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_NOW),
    )


def _step(step_index: int, result: StageResult, *, stage_name: str | None = None) -> StepRecord:
    return StepRecord(
        run_id="r1",
        step_index=step_index,
        stage_name=stage_name if stage_name is not None else f"stage-{step_index}",
        result=result,
        committed_at=_NOW,
    )


async def _start(journal: InMemoryJournal) -> None:
    await journal.start_run("r1", "s1", pathway_id="p", pathway_version=1)


async def test_commit_step_is_idempotent_on_position() -> None:
    """A second commit at the SAME ``(run_id, step_index)`` is a no-op (S6 exactly-once)."""
    journal = InMemoryJournal()
    await _start(journal)
    record = _step(0, Transition(to="b", output=_artifact()), stage_name="a")
    await journal.commit_step(record)
    await journal.commit_step(record)

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1
    assert state.steps[0].step_index == 0


async def test_commit_step_idempotent_on_position_with_differing_fields() -> None:
    """A second commit at the SAME position is a no-op even if other fields differ (S6).

    Two records share ``(run_id, step_index)`` but differ in every other field; the second must be
    dropped (first-write-wins), leaving exactly one step — exactly-once is on the POSITION, not on
    the stage name.
    """
    journal = InMemoryJournal()
    await _start(journal)
    first = _step(0, Transition(to="b", output=_artifact()), stage_name="a")
    second = _step(0, Done(output=_artifact()), stage_name="z")

    await journal.commit_step(first)
    await journal.commit_step(second)  # same position, all other fields differ.

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1
    assert state.steps[0].stage_name == "a"  # the FIRST commit won; the conflicting second dropped


async def test_same_stage_name_at_distinct_positions_are_distinct_steps() -> None:
    """Revisiting the SAME stage_name commits a DISTINCT step per position (per-visit keying).

    A stage name is not a key: a cyclic pathway that revisits ``loop`` at positions 0 and 1 yields
    two journaled steps, so exactly-once + replay work per-visit (S6).
    """
    journal = InMemoryJournal()
    await _start(journal)
    await journal.commit_step(
        _step(0, Transition(to="loop", output=_artifact()), stage_name="loop")
    )
    await journal.commit_step(_step(1, Done(output=_artifact()), stage_name="loop"))

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 2
    assert tuple(step.step_index for step in state.steps) == (0, 1)
    assert tuple(step.stage_name for step in state.steps) == ("loop", "loop")


async def test_start_run_does_not_wipe_existing_steps() -> None:
    journal = InMemoryJournal()
    await _start(journal)
    await journal.commit_step(_step(0, Transition(to="b", output=_artifact()), stage_name="a"))
    await _start(journal)

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1


async def test_load_run_unknown_is_none() -> None:
    journal = InMemoryJournal()
    assert await journal.load_run("missing") is None


async def test_run_status_is_persisted_authority_not_derived() -> None:
    """The run's status is whatever ``set_run_status`` last persisted, not derived from steps."""
    journal = InMemoryJournal()
    await _start(journal)
    await journal.commit_step(_step(0, Transition(to="b", output=_artifact()), stage_name="a"))

    running = await journal.load_run("r1")
    assert running is not None
    assert running.status is RunStatus.RUNNING  # start_run set RUNNING; no terminal status yet
    assert running.current_stage == "a"

    await journal.set_run_status("r1", RunStatus.FAILED)
    failed = await journal.load_run("r1")
    assert failed is not None
    assert failed.status is RunStatus.FAILED  # the persisted record is the authority


async def test_run_carries_pathway_pointer_for_cold_resume() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1", pathway_id="checkout", pathway_version=3)
    state = await journal.load_run("r1")
    assert state is not None
    assert state.pathway_id == "checkout"
    assert state.pathway_version == 3


async def test_latent_search_returns_nearest_first() -> None:
    store = InMemoryLatentStore()
    await store.upsert(LatentRecord(id="near", embedding=(1.0, 0.0)))
    await store.upsert(LatentRecord(id="far", embedding=(0.0, 1.0)))
    await store.upsert(LatentRecord(id="mid", embedding=(1.0, 1.0)))

    matches = await store.search((1.0, 0.0), k=3)
    assert tuple(m.record.id for m in matches) == ("near", "mid", "far")
    assert matches[0].score >= matches[1].score >= matches[2].score


async def test_latent_search_empty_and_zero_vector() -> None:
    store = InMemoryLatentStore()
    assert await store.search((1.0, 0.0)) == ()

    await store.upsert(LatentRecord(id="x", embedding=(1.0, 0.0)))
    assert await store.search((0.0, 0.0)) == ()


async def test_latent_upsert_replaces_by_id() -> None:
    store = InMemoryLatentStore()
    await store.upsert(LatentRecord(id="x", embedding=(1.0, 0.0)))
    await store.upsert(LatentRecord(id="x", embedding=(0.0, 1.0)))

    matches = await store.search((0.0, 1.0), k=5)
    assert len(matches) == 1
    assert matches[0].record.id == "x"
