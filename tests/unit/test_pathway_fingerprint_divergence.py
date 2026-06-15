"""Resume integrity: the structural pathway fingerprint + the per-step name guard (CANON S6).

A run stores only a ``(pathway_id, version)`` pointer, so a cold resume rehydrates the graph from
the registry. Two guards keep that honest:

- ``pathway_fingerprint`` (in ``Engine.resume``, BEFORE ``_drive``): a structural hash stored at
  ``start_run``. If the SAME ``(pathway_id, version)`` is edited in place — transitions rewired, a
  stage added/removed/renamed — the rehydrated graph fingerprints differently and resume is refused
  with ``ResumeError`` rather than re-driving against the wrong graph.
- the per-step ``stage_name != current`` guard in ``Engine._drive``: a deeper backstop that trips if
  a committed step's ``stage_name`` does not match the stage the rehydrated graph reaches at that
  position.

Both tests are mutation-resistant by construction: each is the deterministic falsifier of exactly
one guard, so deleting that guard makes the corresponding test fail (the red-team's finding was that
the per-step guard was a dead check — disabling it left every test green).
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry, pathway_fingerprint
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.retry import RetryPolicy
from cogworx.loop.stage import StageContext
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine, ResumeError
from cogworx.substrate.journal import Journal, StepRecord
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import CrashAfterStepJournal, SimulatedCrash

_PATHWAY_ID = "divergent"
_NOW = datetime(2026, 6, 8, tzinfo=UTC)


def _artifact(kind: str = "t") -> Artifact:
    return Artifact(
        kind=kind,
        produced_by="test",
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=_NOW),
    )


# --------------------------------------------------------------------------------------------------
# A 3-stage model-free pathway. The crash fires after ``middle`` commits, leaving an uncommitted
# tail so resume must actually re-drive (not short-circuit on a terminal status).
# --------------------------------------------------------------------------------------------------


class _IntakeStage:
    name: str = "intake"
    transitions: tuple[str, ...] = ("middle",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="middle", output=_artifact("intake"))


class _MiddleStage:
    name: str = "middle"
    transitions: tuple[str, ...] = ("close",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="close", output=_artifact("middle"))


class _CloseStage:
    name: str = "close"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_artifact("close"))


def _original_graph() -> StageGraph:
    return StageGraph([_IntakeStage(), _MiddleStage(), _CloseStage()], entry="intake")


# A STRUCTURALLY DIFFERENT graph that keeps the SAME stage NAMES and the SAME set of reachable,
# terminal stages, but REWIRES a transition: ``middle`` now also loops back to ``intake`` (its edge
# set is ``("close", "intake")`` instead of ``("close",)``). Same ``(pathway_id, version)``, same
# names, still reachable + terminal by construction — only the structure differs. This is the
# NAME-ONLY blind spot the red-team flagged: the per-step name guard sees the SAME prefix names, so
# only the fingerprint catches the rewire.


class _RewiredMiddleStage:
    name: str = "middle"
    transitions: tuple[str, ...] = ("close", "intake")

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="close", output=_artifact("middle"))


def _rewired_graph() -> StageGraph:
    return StageGraph([_IntakeStage(), _RewiredMiddleStage(), _CloseStage()], entry="intake")


# A graph identical to the original in stage names AND declared transitions, differing ONLY in
# ``middle``'s ``retry_policy.exhausted_to`` (``None`` in the original, ``"close"`` here). The
# exhaustion route is a REAL structural edge the engine validates + routes on (``_edges`` /
# ``StageGraph.edges_from``), so an in-place edit of it under the same ``(pathway_id, version)`` is
# a structural divergence the fingerprint must catch — the blind spot the red-team flagged when the
# fingerprint canonicalised over ``transitions_from`` (which omits ``exhausted_to``) instead of
# ``edges_from``.


def _retry_policy(exhausted_to: str | None) -> RetryPolicy:
    return RetryPolicy(
        max_attempts=1,
        backoff=lambda n: timedelta(seconds=n),
        retryable=(),
        exhausted_to=exhausted_to,
    )


class _ExhaustingMiddleStage:
    name: str = "middle"
    transitions: tuple[str, ...] = ("close",)

    def __init__(self, *, exhausted_to: str | None) -> None:
        self.retry_policy = _retry_policy(exhausted_to)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="close", output=_artifact("middle"))


def _exhausting_graph(*, exhausted_to: str | None) -> StageGraph:
    return StageGraph(
        [_IntakeStage(), _ExhaustingMiddleStage(exhausted_to=exhausted_to), _CloseStage()],
        entry="intake",
    )


def _build_engine(journal: Journal, model: ReplayModel, pathways: PathwayRegistry) -> Engine:
    _reg = ModelRegistry()
    _reg.register("default", model)
    return Engine(
        models=_reg,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
    )


def _initial() -> Artifact:
    return _artifact("user-input")


def test_fingerprint_distinguishes_rewired_same_name_graph() -> None:
    """The fingerprint differs for a same-names, rewired-transitions graph (structural sensitivity).

    Same stage NAMES, different transitions — the name-only blind spot. The fingerprint must split
    them; this is what catches an in-place rewire under the same ``(pathway_id, version)``.
    """
    assert pathway_fingerprint(_original_graph()) != pathway_fingerprint(_rewired_graph())


def test_fingerprint_distinguishes_exhausted_to_edit() -> None:
    """The fingerprint differs when only ``retry_policy.exhausted_to`` changes (FINDING 1).

    Same stage names, same DECLARED transitions — the two graphs differ ONLY in ``middle``'s
    exhaustion route (``None`` vs ``"close"``). Because the fingerprint canonicalises over
    ``edges_from`` (which includes ``exhausted_to``), not ``transitions_from`` (which omits it), it
    must SPLIT them. Mutation evidence: canonicalising over ``transitions_from`` makes these two
    fingerprints identical and this test fails — the exact red-team blind spot.
    """
    assert pathway_fingerprint(_exhausting_graph(exhausted_to=None)) != pathway_fingerprint(
        _exhausting_graph(exhausted_to="close")
    )


def test_fingerprint_is_deterministic_across_rebuilds() -> None:
    """Rebuilding the SAME structure yields the SAME fingerprint (no identity/clock leakage)."""
    assert pathway_fingerprint(_original_graph()) == pathway_fingerprint(_original_graph())


async def test_resume_refuses_structurally_diverged_pathway() -> None:
    """Resume against a STRUCTURALLY different graph at the same ``(id, version)`` raises (S6).

    Engine A drives the original graph and crashes after ``middle`` durably commits. Engine B
    resumes with a registry whose SAME ``(pathway_id, version)`` now maps to the REWIRED graph (same
    stage names, different transitions). The stored fingerprint no longer matches the rehydrated
    graph's, so ``resume`` refuses with ``ResumeError`` BEFORE re-driving.

    Mutation evidence: delete the fingerprint comparison in ``Engine.resume`` and this test fails —
    resume proceeds silently against the wrong graph.
    """
    shared_journal = InMemoryJournal()
    original_pathways = PathwayRegistry()
    original_pathways.register(_PATHWAY_ID, _original_graph())

    crash_journal = CrashAfterStepJournal(inner=shared_journal, crash_after_stage="middle")
    engine_a = _build_engine(crash_journal, ReplayModel([]), original_pathways)
    crashed = False
    try:
        await engine_a.run(
            run_id="div-1",
            session_id="div-sess",
            pathway_id=_PATHWAY_ID,
            initial=_initial(),
        )
    except SimulatedCrash:
        crashed = True
    assert crashed

    mid = await shared_journal.load_run("div-1")
    assert mid is not None
    assert mid.pathway_fingerprint == pathway_fingerprint(_original_graph())

    # Engine B: SAME (pathway_id, version), but the registry now holds the REWIRED graph.
    rewired_pathways = PathwayRegistry()
    rewired_pathways.register(_PATHWAY_ID, _rewired_graph())
    engine_b = _build_engine(shared_journal, ReplayModel([]), rewired_pathways)

    with pytest.raises(ResumeError, match="pathway structure changed"):
        await engine_b.resume("div-1")


async def test_resume_refuses_in_place_exhausted_to_edit() -> None:
    """Resume against an in-place ``exhausted_to`` edit at the same ``(id, version)`` raises (S6).

    Engine A drives the ``exhausted_to=None`` graph and crashes after ``middle`` durably commits.
    Engine B resumes with a registry whose SAME ``(pathway_id, version)`` now maps to the graph
    whose ``middle.retry_policy.exhausted_to`` is ``"close"`` — same stage names, same declared
    transitions, only the exhaustion route differs. The stored fingerprint no longer matches the
    rehydrated graph's, so ``resume`` refuses with ``ResumeError`` BEFORE re-driving — an exhaustion
    route can no longer slip past the resume divergence guard (FINDING 1, the pod-1.0D guard).

    Mutation evidence: canonicalise the fingerprint over ``transitions_from`` (dropping
    ``exhausted_to``) and this test fails — the two graphs fingerprint identically and resume
    proceeds silently against the wrong exhaustion route.
    """
    shared_journal = InMemoryJournal()
    original_pathways = PathwayRegistry()
    original_pathways.register(_PATHWAY_ID, _exhausting_graph(exhausted_to=None))

    crash_journal = CrashAfterStepJournal(inner=shared_journal, crash_after_stage="middle")
    engine_a = _build_engine(crash_journal, ReplayModel([]), original_pathways)
    crashed = False
    try:
        await engine_a.run(
            run_id="exh-1",
            session_id="exh-sess",
            pathway_id=_PATHWAY_ID,
            initial=_initial(),
        )
    except SimulatedCrash:
        crashed = True
    assert crashed

    mid = await shared_journal.load_run("exh-1")
    assert mid is not None
    assert mid.pathway_fingerprint == pathway_fingerprint(_exhausting_graph(exhausted_to=None))

    edited_pathways = PathwayRegistry()
    edited_pathways.register(_PATHWAY_ID, _exhausting_graph(exhausted_to="close"))
    engine_b = _build_engine(shared_journal, ReplayModel([]), edited_pathways)

    with pytest.raises(ResumeError, match="pathway structure changed"):
        await engine_b.resume("exh-1")


async def test_resume_succeeds_when_fingerprint_matches() -> None:
    """The same structural pathway resumes normally — the guard does not over-fire (control)."""
    shared_journal = InMemoryJournal()
    pathways = PathwayRegistry()
    pathways.register(_PATHWAY_ID, _original_graph())

    crash_journal = CrashAfterStepJournal(inner=shared_journal, crash_after_stage="middle")
    engine_a = _build_engine(crash_journal, ReplayModel([]), pathways)
    with contextlib.suppress(SimulatedCrash):
        await engine_a.run(
            run_id="ok-1", session_id="ok-sess", pathway_id=_PATHWAY_ID, initial=_initial()
        )

    # A FRESH registry with the SAME structure (rebuilt graph) — fingerprint matches, resume runs.
    fresh_pathways = PathwayRegistry()
    fresh_pathways.register(_PATHWAY_ID, _original_graph())
    engine_b = _build_engine(shared_journal, ReplayModel([]), fresh_pathways)
    final = await engine_b.resume("ok-1")
    assert tuple(step.stage_name for step in final.steps) == ("intake", "middle", "close")


async def test_drive_per_step_name_guard_trips_on_mismatched_prefix() -> None:
    """The ``_drive`` ``stage_name != current`` backstop trips on a mismatched committed prefix.

    (S6.) A hand-seeded ``InMemoryJournal`` holds a committed step at position 0 whose
    ``stage_name`` is ``wrong`` — a stage the graph never reaches at position 0 (it reaches
    ``intake``).
    The fingerprint stored at ``start_run`` is set to MATCH the rehydrated graph, so the
    fingerprint guard passes and control reaches ``_drive``, isolating the per-step backstop. Resume
    must raise ``ResumeError`` from the per-step guard.

    Mutation evidence: delete the ``existing.stage_name != current`` check in ``Engine._drive`` and
    this test fails — the loop replays the wrong committed result without complaint.
    """
    journal = InMemoryJournal()
    pathways = PathwayRegistry()
    graph = _original_graph()
    pathways.register(_PATHWAY_ID, graph)

    # Seed a run whose stored fingerprint MATCHES the rehydrated graph (so the fingerprint guard is
    # satisfied), then commit a position-0 step whose stage_name the graph does not reach at 0.
    await journal.start_run(
        "guard-1",
        "guard-sess",
        pathway_id=_PATHWAY_ID,
        pathway_version=1,
        pathway_fingerprint=pathway_fingerprint(graph),
    )
    await journal.commit_step(
        StepRecord(
            run_id="guard-1",
            step_index=0,
            stage_name="wrong",
            result=Transition(to="close", output=_artifact("seed")),
            committed_at=_NOW,
        )
    )

    engine = _build_engine(journal, ReplayModel([]), pathways)
    with pytest.raises(ResumeError, match="journal/graph divergence"):
        await engine.resume("guard-1")
