"""Contract tests for the in-memory substrate doubles (S3, S6 reference behaviour)."""

from __future__ import annotations

from datetime import UTC, datetime

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.result import AwaitHuman, Degraded, Done, StageResult, Transition
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


def _step(step_id: str, result: StageResult, *, key: str | None = None) -> StepRecord:
    return StepRecord(
        run_id="r1",
        step_id=step_id,
        stage_name=step_id,
        result=result,
        idempotency_key=key if key is not None else f"r1:{step_id}",
        committed_at=_NOW,
    )


async def test_commit_step_is_idempotent_on_key() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    record = _step("a", Transition(to="b", output=_artifact()))
    await journal.commit_step(record)
    await journal.commit_step(record)

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1
    assert state.steps[0].step_id == "a"


async def test_start_run_does_not_wipe_existing_steps() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    await journal.commit_step(_step("a", Transition(to="b", output=_artifact())))
    await journal.start_run("r1", "s1")

    state = await journal.load_run("r1")
    assert state is not None
    assert len(state.steps) == 1


async def test_load_run_unknown_is_none() -> None:
    journal = InMemoryJournal()
    assert await journal.load_run("missing") is None


async def test_status_derivation_done() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    await journal.commit_step(_step("a", Transition(to="b", output=_artifact())))
    await journal.commit_step(_step("b", Done(output=_artifact())))

    state = await journal.load_run("r1")
    assert state is not None
    assert state.status is RunStatus.COMPLETED
    assert state.current_stage == "b"


async def test_status_derivation_await_human() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    await journal.commit_step(_step("a", AwaitHuman(question="?")))

    state = await journal.load_run("r1")
    assert state is not None
    assert state.status is RunStatus.AWAITING_HUMAN


async def test_status_derivation_degraded_terminal() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    await journal.commit_step(_step("a", Degraded(reason="x", output=_artifact(), to=None)))

    state = await journal.load_run("r1")
    assert state is not None
    assert state.status is RunStatus.DEGRADED


async def test_status_derivation_degraded_with_target_is_running() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    await journal.commit_step(_step("a", Degraded(reason="x", output=_artifact(), to="b")))

    state = await journal.load_run("r1")
    assert state is not None
    assert state.status is RunStatus.RUNNING


async def test_status_derivation_running_and_pending() -> None:
    journal = InMemoryJournal()
    await journal.start_run("r1", "s1")
    empty = await journal.load_run("r1")
    assert empty is not None
    assert empty.status is RunStatus.PENDING

    await journal.commit_step(_step("a", Transition(to="b", output=_artifact())))
    state = await journal.load_run("r1")
    assert state is not None
    assert state.status is RunStatus.RUNNING


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
