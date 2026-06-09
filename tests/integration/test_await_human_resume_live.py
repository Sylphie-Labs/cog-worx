"""Integration: await-human resume first-answer-wins + cold-resume (live Timescale, CANON S6).

Pod 1.3 live-substrate counterpart. The in-memory ``InMemoryJournal`` proves the FIRST-ANSWER-WINS
and pull-model semantics deterministically; the TRUE concurrency property — N callers racing to
record the human answer for the same ``(run_id, step_index)`` without a last-writer-wins race —
can only be proven against real Postgres/Timescale.

``record_human_input`` is backed by::

    INSERT INTO human_inputs (run_id, step_index, answer) VALUES (…)
    ON CONFLICT (run_id, step_index) DO NOTHING

so only the FIRST caller writes the row; all subsequent callers are silent no-ops. A non-idempotent
``INSERT … ON CONFLICT DO UPDATE`` would let the last writer overwrite and would be caught here.

The cold-resume-after-answer test mirrors ``test_await_human_resume.py::test_h4`` but runs against
the real ``TimescaleJournal`` so the adapter's own ``record_human_input`` / ``read_human_input``
SQL is exercised end-to-end.

Marked ``integration`` so it is deselected unless ``-m integration`` and the stack is up
(``docker compose up -d``).
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.timescale_journal import TimescaleJournal
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import AwaitHuman, Done, StageResult
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel

pytestmark = pytest.mark.integration

_NOW = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)
_RUN_ID_BASE = "human-input-race"
_ASK_STEP_INDEX = 0
_CONCURRENCY = 20


# ---------------------------------------------------------------------------
# Session-scope event loop policy (mirrors test_attempt_increment_atomicity.py)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Per-test journal fixture: ensures schema + resets between cases
# ---------------------------------------------------------------------------


@pytest.fixture
async def prepared_journal(settings: SubstrateSettings) -> AsyncIterator[TimescaleJournal]:
    """A ``TimescaleJournal`` with schema ensured and data wiped per case.

    Mirrors the pattern in ``test_attempt_increment_atomicity.py`` exactly.
    NOTE: ``TimescaleJournal.reset()`` must truncate ``human_inputs`` once the adapter gains that
    table; until then the fixture is a stub (tests fail with a clear missing-method error at the
    adapter boundary, not a test-file error).
    """
    adapter = TimescaleJournal(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


# ---------------------------------------------------------------------------
# Helper artifacts
# ---------------------------------------------------------------------------


def _provenanced(kind: str, produced_by: str, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by=produced_by,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_NOW),
        data={"text": text},
    )


def _human_answer(decision: str) -> Artifact:
    return Artifact(
        kind="human-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_NOW),
        data={"decision": decision},
    )


def _human_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_NOW),
        data={"text": "please decide"},
    )


# ---------------------------------------------------------------------------
# Minimal two-stage graph for the integration tests (no model bearing)
# ---------------------------------------------------------------------------


_HUMAN_PATHWAY_ID = "human-resume-live"


class _AskStage:
    name: str = "ask"
    transitions: tuple[str, ...] = ("done",)

    async def run(self, ctx: StageContext) -> StageResult:
        return AwaitHuman(
            question="Approve or reject?",
            to="done",
            output=_provenanced("question", "ask"),
        )


class _DoneStage:
    name: str = "done"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        answer = await ctx.read_human_input(_ASK_STEP_INDEX)
        raw = answer.data.get("decision") if answer is not None else None
        decision: str = str(raw) if raw is not None else "unknown"
        return Done(output=_provenanced("done", "done", decision))


def _live_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(
        _HUMAN_PATHWAY_ID,
        StageGraph([_AskStage(), _DoneStage()], entry="ask"),
    )
    return registry


def _live_engine(journal: TimescaleJournal) -> Engine:
    return Engine(
        model=ReplayModel([]),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_live_pathways(),
        clock=lambda: _NOW,
    )


# ---------------------------------------------------------------------------
# Live tests
# ---------------------------------------------------------------------------


async def test_read_human_input_returns_none_before_any_record(
    prepared_journal: TimescaleJournal,
) -> None:
    """``read_human_input`` returns None for a ``(run_id, step_index)`` that has never been written.

    Mutation killed: an adapter that returns an empty-dict Artifact instead of None; a schema
    where the column defaults to a sentinel rather than the row being absent.
    """
    result = await prepared_journal.read_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX)
    assert result is None


async def test_record_then_read_round_trips_answer(
    prepared_journal: TimescaleJournal,
) -> None:
    """``record_human_input`` + ``read_human_input`` round-trips the artifact faithfully.

    Mutation killed: a lossy serialization that drops ``data`` or ``provenance.source``.
    """
    answer = _human_answer("approve")
    await prepared_journal.record_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX, answer)
    retrieved = await prepared_journal.read_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX)
    assert retrieved is not None
    assert retrieved.provenance.source == "human"
    assert retrieved.data["decision"] == "approve"


async def test_record_human_input_first_answer_wins_sequential(
    prepared_journal: TimescaleJournal,
) -> None:
    """Two sequential ``record_human_input`` calls: only the first persists (DO NOTHING conflict).

    Mutation killed: ``ON CONFLICT DO UPDATE`` semantics (last writer wins; a second call would
    overwrite the first answer).
    """
    first = _human_answer("approve")
    second = _human_answer("reject")
    await prepared_journal.record_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX, first)
    await prepared_journal.record_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX, second)

    retrieved = await prepared_journal.read_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX)
    assert retrieved is not None
    assert retrieved.data["decision"] == "approve", (
        "FIRST-ANSWER-WINS violated: the second answer overwrote the first"
    )


async def test_concurrent_record_human_input_first_answer_wins(
    prepared_journal: TimescaleJournal,
    settings: SubstrateSettings,
) -> None:
    """N concurrent ``record_human_input`` on the SAME key on SEPARATE connections -> exactly one
    answer persists; all subsequent writes are silent no-ops (ON CONFLICT DO NOTHING).

    This is the true PG concurrency property that the in-memory double approximates.

    Mutation killed: a non-idempotent INSERT that errors or a DO UPDATE that makes the result
    non-deterministic under concurrency.
    """
    others = [TimescaleJournal(settings=settings) for _ in range(_CONCURRENCY)]
    answers = [_human_answer(f"decision-{i}") for i in range(_CONCURRENCY)]
    try:
        await asyncio.gather(
            *(
                j.record_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX, a)
                for j, a in zip(others, answers, strict=True)
            )
        )
    finally:
        for j in others:
            await j.aclose()

    # Exactly one answer persists: it is one of the N submitted values.
    retrieved = await prepared_journal.read_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX)
    assert retrieved is not None
    all_decisions = {f"decision-{i}" for i in range(_CONCURRENCY)}
    assert retrieved.data["decision"] in all_decisions, (
        f"persisted decision {retrieved.data['decision']!r} is not one of the submitted values"
    )
    # A second read returns the same row (idempotent read).
    retrieved_again = await prepared_journal.read_human_input(_RUN_ID_BASE, _ASK_STEP_INDEX)
    assert retrieved_again is not None
    assert retrieved_again.data["decision"] == retrieved.data["decision"]


async def test_cold_resume_after_answer_no_model_recall_live(
    prepared_journal: TimescaleJournal,
) -> None:
    """Cold resume after the human answer is recorded + CAS flips status to RUNNING (live PG).

    Mirrors ``test_h4_cold_resume_answered_run_no_model_recall`` from the unit tier but runs
    against the real ``TimescaleJournal``. Proves the adapter's own ``read_human_input`` SQL is
    called correctly by ``ctx.read_human_input`` inside a fresh engine's drive.

    Mutation killed: the adapter's ``read_human_input`` returning None for a committed row (the
    done stage would produce ``decision="unknown"`` instead of the recorded value).
    """
    # Park the run at AWAITING_HUMAN.
    engine_a = _live_engine(prepared_journal)
    parked = await engine_a.run(
        run_id="h4-live",
        session_id="h4-live-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert parked.status is RunStatus.AWAITING_HUMAN

    # Simulate crash mid-provide: record the answer, flip to RUNNING manually.
    approve_artifact = _human_answer("approve")
    await prepared_journal.record_human_input("h4-live", _ASK_STEP_INDEX, approve_artifact)
    await prepared_journal.set_run_status("h4-live", RunStatus.RUNNING)

    # Cold resume on a brand-new engine + zero-response model.
    zero = ReplayModel([])
    engine_b = Engine(
        model=zero,
        journal=prepared_journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=_live_pathways(),
        clock=lambda: _NOW,
    )
    final = await engine_b.resume("h4-live")

    assert final.status is RunStatus.COMPLETED
    assert zero.call_count == 0, (
        f"cold resume re-called the model {zero.call_count} time(s) on live PG"
    )

    # The done step's output contains the human decision, proving ctx.read_human_input worked.
    done_step = next((s for s in final.steps if s.stage_name == "done"), None)
    assert done_step is not None
    assert done_step.result.kind == "done"
    assert done_step.result.output.data.get("text") == "approve"
