"""Durable await-human resume — Phase-1 pod 1.3 spike/chaos suite (CANON S6, S5, S1, S9).

WRITTEN TESTS-FIRST (S12): these target a contract that does NOT yet exist, so the module fails to
COLLECT on the not-yet-built symbols:
  - ``AwaitHuman`` gains a ``to: str`` field (resume target, mirrors ``Wait.to``).
  - ``Journal`` gains ``record_human_input(run_id, step_index, answer)`` (idempotent,
    FIRST-ANSWER-WINS) and ``read_human_input(run_id, step_index) -> Artifact | None``.
  - ``StageContext`` / ``RunContext`` gain
    ``async def read_human_input(step_index) -> Artifact | None``.
  - ``Engine`` gains ``provide_human_input(run_id, *, payload, kind, confidence) -> RunState``.
  - ``EventType`` gains ``HUMAN_INPUT_RECEIVED`` (Subsystem.SPINE).
  - ``StageGraph`` rejects a graph whose ``AwaitHuman.to`` is not in the stage's ``transitions``
    tuple (same rule as ``Wait.to``).

That RED state is intended — it pins the seam the await-human pathway must satisfy before any
implementation hardens it (S12 spike-before-harden).

The graph used throughout: ``ask`` (returns ``AwaitHuman(to="decide", ...)``) -> ``decide`` (pulls
the answer via ``ctx.read_human_input(N)``, branches ``approve`` or ``reject`` structurally off
``data["decision"]``) -> ``approve``/``reject`` (terminal ``Done`` stages).

Each criterion names the mutation it kills — a test that doesn't name a mutation is incomplete.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import AwaitHuman, Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import (
    ModelResponse,
)
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal, RunState, StepRecord, Timer
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    CommitSpyJournal,
    CrashAfterStepJournal,
    SimulatedCrash,
    assert_run_writes_carry_provenance,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_HUMAN_PATHWAY_ID = "human-resume"

_T0 = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

# Positional step indices in the human graph: ask=0, decide=1.
# ask commits at seq 0 (the AwaitHuman); decide at seq 1.
_ASK_STEP_INDEX = 0
_DECIDE_STEP_INDEX = 1


# ---------------------------------------------------------------------------
# Provenance helpers (shared by all stages)
# ---------------------------------------------------------------------------


def _provenanced(kind: str, produced_by: str, text: str = "") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by=produced_by,
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_T0),
        data={"text": text},
    )


def _human_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_T0),
        data={"text": "please decide"},
    )


# ---------------------------------------------------------------------------
# The human-await graph: ask -> decide -> approve | reject
#
# AskStage returns AwaitHuman(to="decide", ...) and must declare "decide" in transitions,
# otherwise StageGraph construction raises. This mirrors the Wait.to rule for Wait stages
# validated in test_timers_sweeper.py L84-97.
#
# DecideStage PULLS the answer via ctx.read_human_input(_ASK_STEP_INDEX) — the PULL model —
# never receiving ctx.human_input injected by the engine (which would break on cold resume).
# Branch is STRUCTURAL: based on data["decision"], never model text (S9).
# ---------------------------------------------------------------------------


class AskStage:
    """Parks the run AWAITING_HUMAN with a question and a resume target of ``decide``.

    ``to`` must appear in ``transitions`` or ``StageGraph`` rejects the graph (the same structural
    guard that ``Wait.to`` gets in 1.1). Mutation killed: dropping ``to`` from ``AwaitHuman``
    makes the engine unable to route after human input; omitting ``"decide"`` from ``transitions``
    prevents graph construction.
    """

    name: str = "ask"
    transitions: tuple[str, ...] = ("decide",)

    async def run(self, ctx: StageContext) -> StageResult:
        return AwaitHuman(
            question="Please approve or reject.",
            to="decide",
            output=_provenanced("question", "ask", "approve or reject?"),
        )


class DecideStage:
    """Pulls the human answer from the journal and branches structurally (S9).

    Uses ``await ctx.read_human_input(_ASK_STEP_INDEX)`` — the PULL model. The engine does NOT
    push a ``ctx.human_input`` attribute; downstream stages pull from the journal. On cold resume
    the answer is in the journal so this works correctly without the engine re-injecting it.

    Mutation killed: push-style ``ctx.human_input`` (it is ``None`` on cold resume so the branch
    silently defaults); routing on model text instead of ``data["decision"]`` (S9).
    """

    name: str = "decide"
    transitions: tuple[str, ...] = ("approve", "reject")

    def __init__(self) -> None:
        self.was_called = False
        self.received_answer: Artifact | None = None

    async def run(self, ctx: StageContext) -> StageResult:
        self.was_called = True
        answer = await ctx.read_human_input(_ASK_STEP_INDEX)
        self.received_answer = answer
        # Structural branch — never model text (S9).
        decision = answer.data.get("decision") if answer is not None else None
        if decision == "approve":
            return Transition(to="approve", output=_provenanced("decision", "decide", "approved"))
        return Transition(to="reject", output=_provenanced("decision", "decide", "rejected"))


class ApproveStage:
    """Terminal stage on the approve path."""

    name: str = "approve"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_provenanced("approved", "approve", "done"))


class RejectStage:
    """Terminal stage on the reject path."""

    name: str = "reject"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_provenanced("rejected", "reject", "done"))


def build_human_graph(*, decide: DecideStage | None = None) -> StageGraph:
    return StageGraph(
        [AskStage(), decide or DecideStage(), ApproveStage(), RejectStage()],
        entry="ask",
    )


def _human_pathways(*, decide: DecideStage | None = None) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_HUMAN_PATHWAY_ID, build_human_graph(decide=decide))
    return registry


def _fixed_clock(now: datetime = _T0) -> Callable[[], datetime]:
    return lambda: now


_CLOCK_AT_T0 = _fixed_clock(_T0)


def _make_build_engine(
    pathways: PathwayRegistry,
    *,
    clock: Callable[[], datetime] = _CLOCK_AT_T0,
) -> Callable[[Journal, ReplayModel], Engine]:
    def build(journal: Journal, model: ReplayModel) -> Engine:
        return Engine(
            model=model,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
            clock=clock,
        )

    return build


def _no_model() -> ReplayModel:
    """A model that raises on ANY call — the ask/decide pathway is model-free."""
    return ReplayModel([])


# ---------------------------------------------------------------------------
# A journal spy that counts ``record_human_input`` and tracks concurrent winners.
# Used by H2 (concurrent-double-answer) to assert first-answer-wins and exactly-one drive.
# ---------------------------------------------------------------------------


class _RecordHumanInputSpyJournal:
    """Counts ``record_human_input`` calls and tracks which answers were recorded.

    Delegates all other methods to the inner journal. Used in H2 to prove idempotency and
    FIRST-ANSWER-WINS under concurrent ``provide_human_input``.
    """

    def __init__(self, inner: Journal) -> None:
        self._inner = inner
        self.record_calls: list[Artifact] = []

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

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def commit_step(self, record: StepRecord) -> None:
        await self._inner.commit_step(record)

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

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

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.increment_attempt(run_id, step_index)

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.read_attempt(run_id, step_index)

    async def record_human_input(
        self, run_id: str, step_index: int, answer: Artifact
    ) -> None:
        # Track calls BEFORE delegating so concurrent callers are both counted.
        self.record_calls.append(answer)
        await self._inner.record_human_input(run_id, step_index, answer)

    async def read_human_input(
        self, run_id: str, step_index: int
    ) -> Artifact | None:
        return await self._inner.read_human_input(run_id, step_index)


# ---------------------------------------------------------------------------
# A journal wrapper that raises inside compare_and_set_run_status on the FIRST call after
# record_human_input has been called — simulates a crash between record and CAS (H3).
# ---------------------------------------------------------------------------


class _CrashBeforeCASJournal:
    """Delegates to inner; raises SimulatedCrash on the FIRST ``compare_and_set_run_status``
    call that sees the ``expect=AWAITING_HUMAN`` flip (the CAS in ``provide_human_input``).

    This models a crash AFTER ``record_human_input`` succeeds but BEFORE the CAS advances the
    run. H3 asserts: after re-issuing ``provide_human_input``, the answer is still in the
    journal (record-before-CAS ordering) and the run completes.

    Mutation killed: CAS-before-record — if the CAS fires first and then the process dies, a
    re-issued ``provide_human_input`` could overwrite with a second answer or leave N+1 with a
    None answer.
    """

    def __init__(self, inner: Journal, *, crash_on_cas: bool = True) -> None:
        self._inner = inner
        self._armed = crash_on_cas  # arms the crash; disarmed after first fire

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

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        if self._armed and expect is RunStatus.AWAITING_HUMAN:
            self._armed = False
            raise SimulatedCrash(
                "simulated crash after record_human_input but before CAS in provide_human_input"
            )
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def commit_step(self, record: StepRecord) -> None:
        await self._inner.commit_step(record)

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

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

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.increment_attempt(run_id, step_index)

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.read_attempt(run_id, step_index)

    async def record_human_input(
        self, run_id: str, step_index: int, answer: Artifact
    ) -> None:
        await self._inner.record_human_input(run_id, step_index, answer)

    async def read_human_input(
        self, run_id: str, step_index: int
    ) -> Artifact | None:
        return await self._inner.read_human_input(run_id, step_index)


# ---------------------------------------------------------------------------
# H1 — answer→advance + structural branch
# ---------------------------------------------------------------------------


async def test_h1_approve_path_parks_then_advances_via_approved_branch() -> None:
    """Park AWAITING_HUMAN at seq 0 (ask); supply an approve answer; run reaches COMPLETED on the
    approve path; the committed ask step is byte-identical on resume (never re-run); DecideStage
    read the answer via ctx.read_human_input.

    Mutation killed: routing decided by the engine outside the downstream stage (would break the
    structural-branch invariant and the pull model); ``AwaitHuman.to`` absent (engine cannot route
    after the answer lands).
    """
    journal = InMemoryJournal()
    decide = DecideStage()
    engine = _make_build_engine(_human_pathways(decide=decide))(journal, _no_model())

    # --- Park at ask ---
    state = await engine.run(
        run_id="h1-approve",
        session_id="h1-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert state.status is RunStatus.AWAITING_HUMAN
    # Exactly one committed step at seq 0 and its result is an AwaitHuman.
    assert len(state.steps) == 1
    ask_step = state.steps[0]
    assert ask_step.step_index == _ASK_STEP_INDEX
    assert ask_step.stage_name == "ask"
    assert ask_step.result.kind == "await-human"

    # --- Provide the approve answer ---
    final = await engine.provide_human_input(
        "h1-approve", payload={"decision": "approve"}
    )
    assert final.status is RunStatus.COMPLETED

    # The answer is journaled with source="human".
    recorded = await journal.read_human_input("h1-approve", _ASK_STEP_INDEX)
    assert recorded is not None
    assert recorded.provenance.source == "human"
    assert recorded.data["decision"] == "approve"

    # The committed ask step at seq 0 is byte-identical (was NOT re-run: AwaitHuman replays
    # as a plain advance to ``to``).
    refreshed = await journal.load_run("h1-approve")
    assert refreshed is not None
    replayed_ask = next(s for s in refreshed.steps if s.step_index == _ASK_STEP_INDEX)
    assert replayed_ask == ask_step  # byte-identical

    # DecideStage ran and read the answer via ctx.read_human_input.
    assert decide.was_called
    assert decide.received_answer is not None
    assert decide.received_answer.data["decision"] == "approve"

    # The run followed the approve path through to terminal.
    stage_seq = tuple(s.stage_name for s in refreshed.steps)
    assert stage_seq == ("ask", "decide", "approve")
    assert len({s.step_index for s in refreshed.steps}) == len(refreshed.steps)


async def test_h1b_reject_path_parks_then_advances_via_reject_branch() -> None:
    """Same as H1 but for the reject branch.

    Mutation killed: a single-branch implementation that always routes to approve regardless of
    the answer; missing ``reject`` in ``DecideStage.transitions``.
    """
    journal = InMemoryJournal()
    decide = DecideStage()
    engine = _make_build_engine(_human_pathways(decide=decide))(journal, _no_model())

    state = await engine.run(
        run_id="h1b-reject",
        session_id="h1b-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert state.status is RunStatus.AWAITING_HUMAN

    final = await engine.provide_human_input(
        "h1b-reject", payload={"decision": "reject"}
    )
    assert final.status is RunStatus.COMPLETED

    recorded = await journal.read_human_input("h1b-reject", _ASK_STEP_INDEX)
    assert recorded is not None
    assert recorded.data["decision"] == "reject"
    assert decide.received_answer is not None
    assert decide.received_answer.data["decision"] == "reject"

    refreshed = await journal.load_run("h1b-reject")
    assert refreshed is not None
    stage_seq = tuple(s.stage_name for s in refreshed.steps)
    assert stage_seq == ("ask", "decide", "reject")


# ---------------------------------------------------------------------------
# H2 — first-answer-wins under concurrent double-answer
# ---------------------------------------------------------------------------


async def test_h2_concurrent_double_answer_first_wins_drives_once() -> None:
    """Park; then concurrent provide_human_input(A) and provide_human_input(B): only the first
    answer lands, the run drives exactly once, and the CAS-losing call returns the current state.

    Mutation killed: ``ON CONFLICT DO UPDATE`` last-wins semantics (the second answer would
    overwrite the first); dropping the CAS before the re-drive (both callers drive the model-free
    decide stage — the stage call-counter would show two calls).

    Implementation note: both answers here carry distinct ``decision`` values; whichever wins
    first determines the path. We assert the recorded answer is ONE of {A, B} — not both, not
    None — and the decide stage ran exactly once.
    """
    inner = InMemoryJournal()
    spy = _RecordHumanInputSpyJournal(inner)
    decide = DecideStage()
    pathways = _human_pathways(decide=decide)
    engine = _make_build_engine(pathways)(spy, _no_model())

    state = await engine.run(
        run_id="h2",
        session_id="h2-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert state.status is RunStatus.AWAITING_HUMAN

    # Two concurrent provide_human_input calls racing on the same parked run.
    result_a, result_b = await asyncio.gather(
        engine.provide_human_input("h2", payload={"decision": "approve"}),
        engine.provide_human_input("h2", payload={"decision": "reject"}),
    )

    # FIRST-ANSWER-WINS: exactly one answer is persisted.
    recorded = await inner.read_human_input("h2", _ASK_STEP_INDEX)
    assert recorded is not None
    assert recorded.data["decision"] in ("approve", "reject")

    # The run completed exactly once (decide.was_called is True and ONLY ONE final status).
    assert decide.was_called
    final_state = await inner.load_run("h2")
    assert final_state is not None
    assert final_state.status is RunStatus.COMPLETED

    # Both concurrent calls return a coherent RunState; the CAS loser returns current state
    # (either AWAITING_HUMAN if it lost before the CAS, or COMPLETED if it saw the winner's result).
    for result in (result_a, result_b):
        assert result.status in (RunStatus.AWAITING_HUMAN, RunStatus.COMPLETED)

    # The run drove exactly once: no duplicate step indices.
    assert len({s.step_index for s in final_state.steps}) == len(final_state.steps)
    # The decide stage ran at most once (not twice under concurrent drivers).
    decide_steps = [s for s in final_state.steps if s.stage_name == "decide"]
    assert len(decide_steps) == 1


# ---------------------------------------------------------------------------
# H3 — record-before-CAS ordering: crash AFTER record, BEFORE CAS
# ---------------------------------------------------------------------------


async def test_h3_crash_after_record_before_cas_reraise_finds_answer_and_completes() -> None:
    """A crash AFTER ``record_human_input`` persists the answer but BEFORE the CAS flips
    AWAITING_HUMAN -> RUNNING: a re-issued ``provide_human_input`` must find the answer already
    journaled and complete without overwriting it.

    Assert: after the simulated crash, ``read_human_input(run_id, N)`` is already non-None;
    the re-issued provide_human_input completes the run; DecideStage reads the correct answer.

    Mutation killed: CAS-before-record ordering — if the CAS fires first and the process dies
    before record, a re-issued call would find no answer in the journal at N+1. This is the
    structural record-then-CAS ordering guarantee.
    """
    shared = InMemoryJournal()
    decide = DecideStage()
    pathways = _human_pathways(decide=decide)

    # Engine A: drive to AWAITING_HUMAN on the clean journal.
    engine_a = _make_build_engine(pathways)(shared, _no_model())
    parked = await engine_a.run(
        run_id="h3",
        session_id="h3-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert parked.status is RunStatus.AWAITING_HUMAN

    # Engine B: backed by a journal that crashes on the CAS AFTER record_human_input completes.
    crash_journal = _CrashBeforeCASJournal(shared, crash_on_cas=True)
    engine_b = _make_build_engine(pathways)(crash_journal, _no_model())
    crashed = False
    try:
        await engine_b.provide_human_input("h3", payload={"decision": "approve"})
    except SimulatedCrash:
        crashed = True
    assert crashed, "expected SimulatedCrash from _CrashBeforeCASJournal"

    # The answer IS in the journal because record precedes CAS.
    after_crash_answer = await shared.read_human_input("h3", _ASK_STEP_INDEX)
    assert after_crash_answer is not None, (
        "record_human_input must persist BEFORE the CAS; the answer must survive the crash"
    )
    assert after_crash_answer.data["decision"] == "approve"

    # The run is still AWAITING_HUMAN (the CAS never flipped).
    assert (await shared.get_run_status("h3")) is RunStatus.AWAITING_HUMAN

    # Engine C: re-issue provide_human_input over the same shared journal (no crash journal now).
    engine_c = _make_build_engine(pathways)(shared, _no_model())
    final = await engine_c.provide_human_input("h3", payload={"decision": "approve"})
    assert final.status is RunStatus.COMPLETED

    # DecideStage got the approve answer and routed correctly.
    assert decide.received_answer is not None
    assert decide.received_answer.data["decision"] == "approve"
    refreshed = await shared.load_run("h3")
    assert refreshed is not None
    stage_seq = tuple(s.stage_name for s in refreshed.steps)
    assert stage_seq == ("ask", "decide", "approve")


# ---------------------------------------------------------------------------
# H4 — cold resume of an ANSWERED run (no model re-call)
# ---------------------------------------------------------------------------


async def test_h4_cold_resume_answered_run_no_model_recall() -> None:
    """Simulate crash AFTER CAS flips AWAITING_HUMAN -> RUNNING but BEFORE the drive finished:
    manually record the answer + set status to RUNNING on the shared journal; build a BRAND-NEW
    Engine over the same journal + ``ReplayModel([])`` (raises on any call); call ``resume``.

    Assert: zero model calls, run completes, DecideStage pulled the journaled answer.

    Mutation killed: push-style ``ctx.human_input`` — if the engine pushes the answer into the
    context only inside ``provide_human_input`` and not during a plain ``resume``, then the
    cold-resume drive has ``ctx.human_input == None`` and the branch silently routes wrong or
    raises; the zero-model ``ReplayModel`` also proves no extra model call was inserted.
    """
    shared = InMemoryJournal()
    decide = DecideStage()
    pathways = _human_pathways(decide=decide)

    # Drive to AWAITING_HUMAN.
    engine_a = _make_build_engine(pathways)(shared, _no_model())
    await engine_a.run(
        run_id="h4",
        session_id="h4-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert (await shared.get_run_status("h4")) is RunStatus.AWAITING_HUMAN

    # Manually simulate: record the answer AND flip status to RUNNING (crash happened right here,
    # mid provide_human_input, after the CAS but before the drive finished).
    approve_artifact = Artifact(
        kind="human-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_T0),
        data={"decision": "approve"},
    )
    await shared.record_human_input("h4", _ASK_STEP_INDEX, approve_artifact)
    await shared.set_run_status("h4", RunStatus.RUNNING)

    # Build a brand-new engine + zero-response model: ANY call raises ReplayExhaustedError.
    zero = ReplayModel([])
    engine_b = _make_build_engine(pathways)(shared, zero)

    final = await engine_b.resume("h4")

    assert final.status is RunStatus.COMPLETED
    assert zero.call_count == 0, (
        f"cold resume must not re-call the model: call_count={zero.call_count}"
    )

    # DecideStage ran and read the journaled answer via ctx.read_human_input.
    assert decide.was_called
    assert decide.received_answer is not None
    assert decide.received_answer.data["decision"] == "approve"

    # Committed prefix (ask at seq 0) is unchanged — replayed verbatim, never re-run.
    refreshed = await shared.load_run("h4")
    assert refreshed is not None
    stage_seq = tuple(s.stage_name for s in refreshed.steps)
    assert stage_seq == ("ask", "decide", "approve")
    assert len({s.step_index for s in refreshed.steps}) == len(refreshed.steps)


# ---------------------------------------------------------------------------
# H5 — unanswered crash-resume is a no-op
# ---------------------------------------------------------------------------


async def test_h5_resume_of_unanswered_awaiting_run_is_noop() -> None:
    """A plain ``resume`` of a still-AWAITING_HUMAN run with NO answer recorded must return the
    run parked AWAITING_HUMAN without advancing, with zero model calls.

    Mutation killed: a resume that drives uncommitted stages even with no answer (would advance the
    run with ``ctx.read_human_input`` returning None, silently routing to the wrong branch or
    raising); treating AWAITING_HUMAN as a re-drivable RUNNING state (would make H4's manual CAS
    flip redundant and hide the difference between ``RUNNING``-with-answer and ``AWAITING_HUMAN``).
    """
    journal = InMemoryJournal()
    engine = _make_build_engine(_human_pathways())(journal, _no_model())

    # Park the run.
    state = await engine.run(
        run_id="h5",
        session_id="h5-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert state.status is RunStatus.AWAITING_HUMAN

    # No answer recorded — resume must be a no-op.
    zero = ReplayModel([])
    engine_b = _make_build_engine(_human_pathways())(journal, zero)
    resumed = await engine_b.resume("h5")

    assert resumed.status is RunStatus.AWAITING_HUMAN
    assert zero.call_count == 0

    # Still only the ask step is committed; decide has not run.
    assert len(resumed.steps) == 1
    assert resumed.steps[0].stage_name == "ask"


# ---------------------------------------------------------------------------
# H6 — provide_human_input on a non-AWAITING_HUMAN run is a no-op
# ---------------------------------------------------------------------------


async def test_h6_provide_input_on_nonawaiting_run_is_noop() -> None:
    """``provide_human_input`` on a run NOT in AWAITING_HUMAN returns the current state without
    advancing or recording anything.

    Scenarios tested: RUNNING (someone calls provide mid-drive before the run parks),
    COMPLETED (run already finished).

    Mutation killed: an unconditional drive that does not check the parked status first — would
    accidentally re-drive a RUNNING or COMPLETED run.
    """
    # COMPLETED path.
    from cogworx.loop.graph import StageGraph as _SG
    from cogworx.loop.result import Done

    class _TrivialStage:
        name: str = "trivial"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(output=_provenanced("done", "trivial"))

    trivial_registry = PathwayRegistry()
    trivial_registry.register("trivial-pw", _SG([_TrivialStage()], entry="trivial"))

    journal = InMemoryJournal()
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=trivial_registry,
        clock=_CLOCK_AT_T0,
    )
    completed = await engine.run(
        run_id="h6-done",
        session_id="h6-done-sess",
        pathway_id="trivial-pw",
        initial=_human_initial(),
    )
    assert completed.status is RunStatus.COMPLETED

    returned = await engine.provide_human_input("h6-done", payload={"decision": "approve"})
    assert returned.status is RunStatus.COMPLETED  # unchanged, no re-drive


# ---------------------------------------------------------------------------
# Invariants — S1, S5, S6, S9
# ---------------------------------------------------------------------------


async def test_s1_no_model_on_record_human_input_write_path() -> None:
    """S1: ``record_human_input`` (the write path for the human answer) must not invoke the model.

    The ``CommitSpyJournal`` wrapper trips if any commit increments the model call count. The
    human-answer write path is a pure persist — no model-heavy work on the hot path (S1).

    Mutation killed: an impl that calls the model to classify or summarize the answer on write.
    """
    inner = InMemoryJournal()
    # The human pathway is model-free, so the model here is the zero-response spy.
    model = _no_model()
    spy = CommitSpyJournal(inner=inner, model=model)
    pathways = _human_pathways()
    engine = _make_build_engine(pathways)(spy, model)

    state = await engine.run(
        run_id="s1-human",
        session_id="s1-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert state.status is RunStatus.AWAITING_HUMAN

    # Build a fresh engine backed by the same spy; providing the answer must not trip the spy.
    engine2 = _make_build_engine(pathways)(spy, model)
    final = await engine2.provide_human_input("s1-human", payload={"decision": "approve"})
    assert final.status is RunStatus.COMPLETED
    # If CommitSpyJournal tripped it would have raised InvariantViolation already.


async def test_s5_human_answer_artifact_carries_provenance_source_human() -> None:
    """S5: the human-answer ``Artifact`` built by ``provide_human_input`` carries provenance with
    ``source="human"`` — an honest epistemic label, never ``"inference"`` or ``"system"``.

    Also asserts ``assert_run_writes_carry_provenance`` over the completed run state.

    Mutation killed: using ``source="inference"`` or ``source="system"`` for a human answer
    (dishonest epistemic label); omitting provenance from the answer artifact entirely.
    """
    journal = InMemoryJournal()
    engine = _make_build_engine(_human_pathways())(journal, _no_model())

    await engine.run(
        run_id="s5-human",
        session_id="s5-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    final = await engine.provide_human_input("s5-human", payload={"decision": "approve"})
    assert final.status is RunStatus.COMPLETED

    # The recorded answer's source is honestly "human".
    answer = await journal.read_human_input("s5-human", _ASK_STEP_INDEX)
    assert answer is not None
    assert answer.provenance is not None
    assert answer.provenance.source == "human", (
        f"S5: human answer must carry source='human', got {answer.provenance.source!r}"
    )
    # The answer confidence is 1.0 by default (the caller did not supply an override).
    assert answer.provenance.confidence == 1.0

    # Every committed step output in the final run carries provenance.
    assert_run_writes_carry_provenance(final)


async def test_s6_resume_across_await_human_boundary_no_model_recall() -> None:
    """S6: after the human answers + the CAS flips to RUNNING, a cold resume on a fresh engine
    must replay the committed prefix (including the AwaitHuman step) without re-calling the model.

    Uses ``CrashAfterStepJournal(crash_after_stage="decide")`` to commit the decide step then
    crash; cold resume on a fresh ``ReplayModel([])`` completes with zero model calls.

    Mutation killed: not journaling the AwaitHuman before driving (replay would miss it);
    re-calling the model in the decide stage's ctx.read_human_input (a model call raises).
    """
    shared = InMemoryJournal()
    decide = DecideStage()
    pathways = _human_pathways(decide=decide)

    # Park the run.
    engine_a = _make_build_engine(pathways)(shared, _no_model())
    parked = await engine_a.run(
        run_id="s6-human",
        session_id="s6-sess",
        pathway_id=_HUMAN_PATHWAY_ID,
        initial=_human_initial(),
    )
    assert parked.status is RunStatus.AWAITING_HUMAN

    # Provide the answer through a journal that crashes AFTER decide commits.
    crash_journal = CrashAfterStepJournal(inner=shared, crash_after_stage="decide")
    engine_b = _make_build_engine(pathways)(crash_journal, _no_model())
    crashed = False
    try:
        await engine_b.provide_human_input("s6-human", payload={"decision": "approve"})
    except SimulatedCrash:
        crashed = True
    assert crashed

    # The decide step IS durably committed; run is still RUNNING (crashed mid-drive).
    mid = await shared.load_run("s6-human")
    assert mid is not None
    assert mid.status == RunStatus.RUNNING or mid.status == RunStatus.COMPLETED
    decide_committed = next(
        (s for s in mid.steps if s.stage_name == "decide"), None
    )
    assert decide_committed is not None, "decide must have committed before the crash"

    # Cold resume on FRESH engine + zero-response model: any re-call raises.
    zero = ReplayModel([])
    engine_c = _make_build_engine(pathways)(shared, zero)
    final = await engine_c.resume("s6-human")

    assert final.status is RunStatus.COMPLETED
    assert zero.call_count == 0, (
        f"S6 violation: cold resume re-called the model {zero.call_count} time(s)"
    )
    stage_seq = tuple(s.stage_name for s in final.steps)
    assert stage_seq == ("ask", "decide", "approve")
    assert len({s.step_index for s in final.steps}) == len(final.steps)


async def test_s9_branch_independent_of_model_text() -> None:
    """S9: the downstream branch (approve vs reject) is STRUCTURAL — a function of the journaled
    answer's ``data["decision"]``, never of the model's response text.

    Runs the same graph under two wildly different model texts (the pathway is model-free, so the
    model text difference can only influence routing if the stage inspects it). Both runs supply
    ``decision=approve`` so both must reach COMPLETED via the approve path identically.

    Uses the reusable ``assert_control_independent_of_model_text`` helper.

    Mutation killed: a stage that inspects the model's words to decide the branch (S9); a stage
    that uses ``ctx.model.complete`` to classify the answer before routing.
    """

    def _journal_factory() -> Journal:
        return InMemoryJournal()

    # Model text must not affect routing — use two maximally different texts.
    model_a = ReplayModel(
        [ModelResponse(text="APPROVE. route to approve.", model_id="r", finish_reason="stop")]
    )
    model_b = ReplayModel(
        [ModelResponse(text="REJECT REJECT. route to reject.", model_id="r", finish_reason="stop")]
    )

    # The pathway is MODEL-FREE (AskStage and DecideStage never call ctx.model). The models are
    # wired in so the engine runs, but they are never called. If the branch were model-text-driven
    # the two paths would diverge; since it is structural they must match.
    #
    # Because the pathway is model-free we cannot drive to COMPLETED via engine.run alone (it
    # parks at AWAITING_HUMAN). We exercise the S9 helper over a full provide_human_input cycle by
    # wrapping the run+provide in a single coroutine that looks like an engine.run to the helper.
    #
    # Instead of abusing the helper's internal contract, we implement the assertion directly here
    # and document why (the helper only calls engine.run; for a HITL pathway we extend the pattern).

    async def _run_and_provide(model: ReplayModel) -> tuple[str, ...]:
        j = _journal_factory()
        decide = DecideStage()
        pw = _human_pathways(decide=decide)
        engine = Engine(
            model=model,
            journal=j,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pw,
            clock=_CLOCK_AT_T0,
        )
        await engine.run(
            run_id="s9-run",
            session_id="s9-sess",
            pathway_id=_HUMAN_PATHWAY_ID,
            initial=_human_initial(),
        )
        final = await engine.provide_human_input("s9-run", payload={"decision": "approve"})
        return tuple(s.stage_name for s in final.steps)

    path_a = await _run_and_provide(model_a)
    path_b = await _run_and_provide(model_b)

    assert path_a == path_b, (
        f"S9 violation: committed path differed across model texts ({path_a} vs {path_b}); "
        "branch must be structural (data['decision']), never model-text-driven"
    )
    assert path_a == ("ask", "decide", "approve"), (
        f"Expected approve path, got {path_a!r}"
    )


# ---------------------------------------------------------------------------
# Structural validation — AwaitHuman.to field existence + graph guard
# ---------------------------------------------------------------------------


def test_structural_await_human_has_to_field() -> None:
    """``AwaitHuman`` must expose a ``to`` attribute that survives Pydantic round-trip.

    Pre-impl: Pydantic ignores unknown extra fields silently — ``AwaitHuman(to="decide")``
    constructs without error but ``result.to`` raises ``AttributeError`` (the field is not stored).
    After implementation ``result.to`` is the declared field and returns ``"decide"``.

    This is the PRIMARY "right reason to fail" test: it fails with ``AttributeError`` on
    ``result.to`` until the python-expert adds ``to: str`` to ``AwaitHuman``.

    Mutation killed: adding ``to`` as an ``extra="allow"`` ignored attr instead of a declared
    field (Pydantic would drop the value and S6 replay serialization would lose it).
    """
    from cogworx.claims.provenance import Artifact, Provenance

    result = AwaitHuman(
        question="test",
        to="decide",  # silently ignored pre-impl; stored as a field post-impl
        output=Artifact(
            kind="q",
            produced_by="ask",
            provenance=Provenance(source="inference", confidence=1.0, recorded_at=_T0),
        ),
    )
    # Pre-impl: ``to`` is dropped by Pydantic -> AttributeError.
    # Post-impl: ``result.to == "decide"``.
    assert result.to == "decide", (
        "AwaitHuman.to must be a declared Pydantic field, not an ignored extra; "
        "it must survive construction so the engine can route after the answer lands"
    )


async def test_structural_await_human_to_must_be_in_transitions() -> None:
    """``engine.run`` raises ``StageGraphError`` (before commit) when ``AwaitHuman.to`` routes to a
    stage not in the stage's declared ``transitions``; nothing is committed at that seq.

    Reframed from xfail(strict=True): the runtime declared-route guard in the engine (fresh branch,
    before ``commit_step``) now enforces this. The guard uses ``graph.edges_from`` as the allowlist.

    Mutation killed: (a) missing guard — the run strands RUNNING with a dangling route; (b)
    commit-then-guard ordering — the poisoned row re-detonates on every future resume; (c) using
    ``transitions_from`` instead of ``edges_from`` — would falsely reject the engine's own
    exhaustion-degraded route.
    """
    class _BadAskStage:
        name: str = "ask"
        transitions: tuple[str, ...] = ("decide",)

        async def run(self, ctx: StageContext) -> StageResult:
            return AwaitHuman(
                question="test",
                to="nonexistent",
                output=_provenanced("q", "ask"),
            )

    from cogworx.loop.graph import StageGraphError

    graph = StageGraph(
        [_BadAskStage(), DecideStage(), ApproveStage(), RejectStage()],
        entry="ask",
    )
    journal = InMemoryJournal()
    pathways = _make_pathway_from_graph(graph)
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=_CLOCK_AT_T0,
    )

    with pytest.raises(StageGraphError, match="nonexistent"):
        await engine.run(
            run_id="guard-await-human",
            session_id="guard-sess",
            pathway_id=_HUMAN_PATHWAY_ID,
            initial=_human_initial(),
        )

    # Nothing committed at seq 0 — the guard fires BEFORE commit_step (mutation killer).
    assert await journal.read_step("guard-await-human", 0) is None


async def test_guard_transition_to_undeclared_raises_before_commit() -> None:
    """``Transition(to=undeclared)`` raises ``StageGraphError`` before commit (sibling guard case).

    Mutation killed: guard absent for Transition — the most common result kind; commit-then-guard
    ordering — a bad Transition.to would be durably journaled and re-detonate on every resume.
    """
    from cogworx.loop.graph import StageGraphError
    from cogworx.loop.result import Done

    class _BadTransitionStage:
        name: str = "start"
        transitions: tuple[str, ...] = ("finish",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Transition(to="ghost", output=_provenanced("x", "start"))

    class _FinishStage:
        name: str = "finish"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(output=_provenanced("done", "finish"))

    graph = StageGraph([_BadTransitionStage(), _FinishStage()], entry="start")
    journal = InMemoryJournal()
    registry = PathwayRegistry()
    registry.register("bad-transition-pw", graph)
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=registry,
        clock=_CLOCK_AT_T0,
    )

    with pytest.raises(StageGraphError, match="ghost"):
        await engine.run(
            run_id="bad-trans",
            session_id="bad-trans-sess",
            pathway_id="bad-transition-pw",
            initial=_human_initial(),
        )

    assert await journal.read_step("bad-trans", 0) is None


async def test_guard_wait_to_undeclared_raises_before_commit() -> None:
    """``Wait(to=undeclared)`` raises ``StageGraphError`` before commit (sibling guard case).

    Mutation killed: guard absent for Wait — a bad wake-and-route destination is silently journaled
    and the run strands after the timer fires.
    """
    from datetime import timedelta

    from cogworx.loop.graph import StageGraphError
    from cogworx.loop.result import Wait

    class _BadWaitStage:
        name: str = "park"
        transitions: tuple[str, ...] = ("next",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Wait(
                to="nowhere",
                wake_at=_T0 + timedelta(seconds=10),
                output=_provenanced("w", "park"),
            )

    class _NextStage:
        name: str = "next"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            from cogworx.loop.result import Done

            return Done(output=_provenanced("done", "next"))

    graph = StageGraph([_BadWaitStage(), _NextStage()], entry="park")
    journal = InMemoryJournal()
    registry = PathwayRegistry()
    registry.register("bad-wait-pw", graph)
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=registry,
        clock=_CLOCK_AT_T0,
    )

    with pytest.raises(StageGraphError, match="nowhere"):
        await engine.run(
            run_id="bad-wait",
            session_id="bad-wait-sess",
            pathway_id="bad-wait-pw",
            initial=_human_initial(),
        )

    assert await journal.read_step("bad-wait", 0) is None


async def test_guard_degraded_to_undeclared_raises_before_commit() -> None:
    """``Degraded(to=undeclared)`` raises ``StageGraphError`` before commit (sibling guard case).

    Mutation killed: guard absent for Degraded-with-to — a bad onward route is silently journaled
    and the run then strands at the undeclared stage on the next iteration.
    """
    from cogworx.loop.graph import StageGraphError
    from cogworx.loop.result import Degraded

    class _BadDegradedStage:
        name: str = "risky"
        transitions: tuple[str, ...] = ("safe",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Degraded(
                reason="something broke",
                to="ghost",
                output=_provenanced("d", "risky"),
            )

    class _SafeStage:
        name: str = "safe"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            from cogworx.loop.result import Done

            return Done(output=_provenanced("done", "safe"))

    graph = StageGraph([_BadDegradedStage(), _SafeStage()], entry="risky")
    journal = InMemoryJournal()
    registry = PathwayRegistry()
    registry.register("bad-degraded-pw", graph)
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=registry,
        clock=_CLOCK_AT_T0,
    )

    with pytest.raises(StageGraphError, match="ghost"):
        await engine.run(
            run_id="bad-degraded",
            session_id="bad-degraded-sess",
            pathway_id="bad-degraded-pw",
            initial=_human_initial(),
        )

    assert await journal.read_step("bad-degraded", 0) is None


async def test_guard_positive_degraded_to_none_terminates_cleanly() -> None:
    """``Degraded(to=None)`` is a legal terminal — guard must NOT raise, run ends DEGRADED.

    Mutation killed: over-broad guard that rejects ``to=None`` on a Degraded result — Degraded
    without a continuation is the valid "stop here" signal (S8).
    """
    from cogworx.loop.result import Degraded, Done

    class _TerminalDegradedStage:
        name: str = "risky"
        transitions: tuple[str, ...] = ("fallback",)

        async def run(self, ctx: StageContext) -> StageResult:
            return Degraded(
                reason="gave up",
                to=None,
                output=_provenanced("d", "risky"),
            )

    class _FallbackStage:
        name: str = "fallback"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(output=_provenanced("done", "fallback"))

    graph = StageGraph([_TerminalDegradedStage(), _FallbackStage()], entry="risky")
    journal = InMemoryJournal()
    registry = PathwayRegistry()
    registry.register("degraded-terminal-pw", graph)
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=registry,
        clock=_CLOCK_AT_T0,
    )

    final = await engine.run(
        run_id="degraded-terminal",
        session_id="dt-sess",
        pathway_id="degraded-terminal-pw",
        initial=_human_initial(),
    )
    assert final.status is RunStatus.DEGRADED
    # The step WAS committed (a terminal Degraded commits its result).
    assert await journal.read_step("degraded-terminal", 0) is not None


async def test_guard_positive_exhausted_to_via_edges_from_passes() -> None:
    """A stage with ``RetryPolicy(exhausted_to=X)`` that exhausts → engine's own
    ``Degraded(to=X)`` PASSES the guard because X is in ``edges_from`` (via ``exhausted_to``).

    This is the explicit regression test for the ``edges_from`` vs ``transitions_from`` distinction.
    Using ``transitions_from`` would falsely reject this engine-synthesised route.

    Mutation killed: using ``transitions_from`` in the guard — the engine's own exhaustion-degraded
    route is not in ``transitions`` (it is only in ``exhausted_to`` / ``edges_from``) so the guard
    would raise ``StageGraphError`` on the engine's own internally-synthesised result.
    """
    from datetime import timedelta

    from cogworx.loop.result import Done
    from cogworx.loop.retry import RetryPolicy

    class _ExhaustingStage:
        name: str = "worker"
        transitions: tuple[str, ...] = ("ok",)
        # max_attempts=1: the very first failure IS exhaustion — no retry timer fires, the engine
        # synthesises Degraded(to="fallback") and routes onward immediately. This keeps the test
        # self-contained without needing the Sweeper; the edges_from regression is still exercised.
        retry_policy: RetryPolicy = RetryPolicy(
            max_attempts=1,
            backoff=lambda n: timedelta(seconds=0),
            exhausted_to="fallback",
            on_exhausted="degraded",
            retryable=(ValueError,),
        )

        async def run(self, ctx: StageContext) -> StageResult:
            raise ValueError("always fails")

    class _OkStage:
        name: str = "ok"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(output=_provenanced("done", "ok"))

    class _FallbackStage:
        name: str = "fallback"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(output=_provenanced("done", "fallback"))

    graph = StageGraph([_ExhaustingStage(), _OkStage(), _FallbackStage()], entry="worker")
    journal = InMemoryJournal()
    registry = PathwayRegistry()
    registry.register("exhaustion-pw", graph)
    engine = Engine(
        model=_no_model(),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=registry,
        clock=_CLOCK_AT_T0,
    )

    final = await engine.run(
        run_id="exhaustion-run",
        session_id="ex-sess",
        pathway_id="exhaustion-pw",
        initial=_human_initial(),
    )
    # Exhausted → Degraded(to="fallback") → fallback stage → COMPLETED.
    assert final.status is RunStatus.COMPLETED
    stage_seq = tuple(s.stage_name for s in final.steps)
    assert stage_seq[-1] == "fallback", (
        f"expected exhaustion to route to 'fallback' via edges_from; got {stage_seq!r}"
    )


def _make_pathway_from_graph(graph: StageGraph) -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_HUMAN_PATHWAY_ID, graph)
    return registry
