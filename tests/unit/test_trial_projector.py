"""The trial projector — committed steps to procedural-KG trials, off the write path (CANON S1/S6).

The load-bearing S1/S6 integration: it turns committed journal ``StepRecord``s into ``(:Trial)``
nodes via a sweeper-shaped poll, with a Neo4j-resident cursor advanced in the trial's own txn. Runs
on the in-memory journal + in-memory procedural-KG doubles (no live stores), so it is fully
deterministic. The live Timescale + Neo4j span (incl. the commit_xid visibility-fence race) is
exercised in tests/spike and tests/integration.

Spike claim (a) invariants, each paired with a NEGATIVE CONTROL where feasible (the standing
red-team rule: inject the bug, confirm the assertion would fail):
  a1 exactly-one-Trial-per-committed-step
  a2 zero-Trials-for-never-committed
  a3 the flip schedule (failed attempt commits nothing; success at the next index -> ONE success)
  a4 re-run projector twice -> byte-identical subgraph + cursor (idempotent)
  a6 cursor advances IN the trial write (no partial state)
  + commit-order forward progress (FP), failure-trial synthesis from exhaustion-Degrade, the
    stamped-vs-result.kind discipline (S9), and the FAILED-at-seq known-limitation marker.

The cursor is now the commit-order tuple ``(commit_ordinal, run_id, step_index)``
(:class:`ProjectionCursor`); the in-memory journal assigns ``commit_ordinal`` in commit order (the
mirror of Postgres' ``commit_xid``). The old wall-clock ``committed_at`` keyset + lookback band is
gone, so the FP-3 (skew/lookback) and band-overflow / union-starvation controls — whose failure mode
no longer exists — are deleted (see the session doc).
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.knowledge.procedural_registry import ProcedureRegistry
from cogworx.loop.result import Degraded, Done, Transition
from cogworx.loop.state import RunStatus
from cogworx.runtime.projector import (
    DEFAULT_PROJECTION_CONSUMER,
    TrialProjector,
    resolve_outcome,
)
from cogworx.substrate.journal import ProjectionCursor, StepRecord
from cogworx.substrate.procedural_kg import Outcome, ordinal_ge
from cogworx.testing.doubles import InMemoryJournal, InMemoryProceduralKG

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_PATHWAY = "math_pathway"
_STAGE = "solve_stage"
_FIXUP = "fixup_stage"


def _registry() -> ProcedureRegistry:
    registry = ProcedureRegistry()
    registry.declare(_PATHWAY, _STAGE, problem_type="word problem")
    registry.declare(_PATHWAY, _FIXUP, problem_type="word problem")
    return registry


def _decl(registry: ProcedureRegistry, stage: str = _STAGE) -> tuple[str, str]:
    decl = registry.get(_PATHWAY, stage)
    assert decl is not None
    return decl.procedure_id, decl.problem_type


def _system_prov(at: datetime = _T0) -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=at)


def _stamped_artifact(
    registry: ProcedureRegistry, *, outcome: Outcome, stage: str = _STAGE
) -> Artifact:
    procedure_id, problem_type = _decl(registry, stage)
    return Artifact(
        kind="solution",
        produced_by=stage,
        provenance=_system_prov(),
        data={"outcome": outcome, "procedure_id": procedure_id, "problem_type": problem_type},
    )


def _exhaustion_artifact() -> Artifact:
    # The shape the engine commits on retry-exhaustion -> Degrade: NO procedure_id, carries
    # failure_class/attempts. The projector recovers the ids from the registry.
    return Artifact(
        kind="retry-exhausted",
        produced_by="engine",
        provenance=_system_prov(),
        data={"failure_class": "TimeoutError", "attempts": 3},
    )


async def _start(journal: InMemoryJournal, run_id: str) -> None:
    await journal.start_run(
        run_id,
        f"session:{run_id}",
        pathway_id=_PATHWAY,
        pathway_version=1,
        pathway_fingerprint="fp",
    )


async def _commit(
    journal: InMemoryJournal,
    *,
    run_id: str,
    step_index: int,
    stage: str,
    output: Artifact,
    committed_at: datetime,
    transition_to: str | None = None,
) -> None:
    result: Transition | Done | Degraded
    if output.kind == "retry-exhausted":
        result = Degraded(reason="retry exhausted", output=output, to=transition_to)
    elif transition_to is not None:
        result = Transition(to=transition_to, output=output)
    else:
        result = Done(output=output)
    await journal.commit_step(
        StepRecord(
            run_id=run_id,
            step_index=step_index,
            stage_name=stage,
            result=result,
            committed_at=committed_at,
        )
    )


def _projector(
    journal: InMemoryJournal,
    kg: InMemoryProceduralKG,
    registry: ProcedureRegistry,
    *,
    batch_limit: int = 256,
) -> TrialProjector:
    return TrialProjector(
        journal=journal,
        procedural_kg=kg,
        registry=registry,
        batch_limit=batch_limit,
    )


async def _commit_control(
    journal: InMemoryJournal, *, run_id: str, step_index: int, committed_at: datetime
) -> None:
    """Commit a NON-procedure control step (no stamp): the projector scans but skips it."""
    plain = Artifact(kind="note", produced_by="router", provenance=_system_prov(), data={})
    await journal.commit_step(
        StepRecord(
            run_id=run_id,
            step_index=step_index,
            stage_name="router_stage",
            result=Done(output=plain),
            committed_at=committed_at,
        )
    )


async def _subgraph_snapshot(
    kg: InMemoryProceduralKG, registry: ProcedureRegistry
) -> Mapping[str, Any]:
    """A comparable snapshot of the procedural subgraph + cursor for the byte-identity check."""
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    posterior = await kg.posterior(procedure_id, problem_type)
    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    return {
        "trial_ids": sorted(t.trial_id for t in trials),
        "outcomes": sorted((t.trial_id, t.outcome) for t in trials),
        "alpha": posterior.alpha,
        "beta": posterior.beta,
        "n_trials": posterior.n_trials,
        "cursor": cursor,
    }


# ---------------------------------------------------------------------------
# resolve_outcome (S9 — stamped authority, never result.kind)
# ---------------------------------------------------------------------------


def test_resolve_reads_stamped_outcome_not_kind() -> None:
    # The S9 trap: a FAILING procedure routing to a fixup stage commits a Transition (kind reads as
    # "success"), but its stamped outcome is "failure". resolve_outcome MUST honour the stamp.
    registry = _registry()
    failing = _stamped_artifact(registry, outcome="failure")
    step = StepRecord(
        run_id="r",
        step_index=0,
        stage_name=_STAGE,
        result=Transition(to=_FIXUP, output=failing),
        committed_at=_T0,
    )
    resolved = resolve_outcome(step, pathway_id=_PATHWAY, registry=registry)
    assert resolved is not None
    assert resolved.outcome == "failure"


def test_resolve_skips_unstamped_non_procedure_step() -> None:
    registry = _registry()
    plain = Artifact(kind="note", produced_by="router", provenance=_system_prov(), data={})
    step = StepRecord(
        run_id="r",
        step_index=0,
        stage_name="router_stage",
        result=Transition(to=_STAGE, output=plain),
        committed_at=_T0,
    )
    assert resolve_outcome(step, pathway_id=_PATHWAY, registry=registry) is None


def test_resolve_rejects_stamped_outcome_missing_ids() -> None:
    registry = _registry()
    bad = Artifact(
        kind="solution",
        produced_by=_STAGE,
        provenance=_system_prov(),
        data={"outcome": "success"},  # no procedure_id / problem_type
    )
    step = StepRecord(
        run_id="r", step_index=0, stage_name=_STAGE, result=Done(output=bad), committed_at=_T0
    )
    with pytest.raises(ValueError, match="missing a string procedure_id/problem_type"):
        resolve_outcome(step, pathway_id=_PATHWAY, registry=registry)


def test_resolve_rejects_noncanonical_stamped_outcome() -> None:
    registry = _registry()
    procedure_id, problem_type = _decl(registry)
    bad = Artifact(
        kind="solution",
        produced_by=_STAGE,
        provenance=_system_prov(),
        data={"outcome": "SUCCESS", "procedure_id": procedure_id, "problem_type": problem_type},
    )
    step = StepRecord(
        run_id="r", step_index=0, stage_name=_STAGE, result=Done(output=bad), committed_at=_T0
    )
    with pytest.raises(ValueError, match="non-canonical outcome"):
        resolve_outcome(step, pathway_id=_PATHWAY, registry=registry)


# ---------------------------------------------------------------------------
# Projection invariants (spike claim a)
# ---------------------------------------------------------------------------


async def test_a1_exactly_one_trial_per_committed_step() -> None:
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    await _commit(
        journal,
        run_id="r1",
        step_index=0,
        stage=_STAGE,
        output=_stamped_artifact(registry, outcome="success"),
        committed_at=_T0,
    )
    projected = await _projector(journal, kg, registry).tick()

    assert projected == 1
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.trial_id for t in trials] == ["r1:0"]
    assert trials[0].outcome == "success"
    assert trials[0].provenance.source == "system"


async def test_a2_zero_trials_for_never_committed() -> None:
    # A committed step that is NOT a procedure produces no trial; an empty journal produces none.
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    plain = Artifact(kind="note", produced_by="router", provenance=_system_prov(), data={})
    await journal.commit_step(
        StepRecord(
            run_id="r1",
            step_index=0,
            stage_name="router_stage",
            result=Done(output=plain),
            committed_at=_T0,
        )
    )
    projected = await _projector(journal, kg, registry).tick()

    assert projected == 0
    procedure_id, problem_type = _decl(registry)
    assert await kg.trials_for(procedure_id, problem_type) == ()


async def test_a3_flip_schedule_one_success() -> None:
    # The flip: attempt N (a retryable failure) commits NOTHING — only the durable attempt counter
    # climbs. Attempt N+1 succeeds and commits at the NEXT step_index. So the journal holds exactly
    # one committed step for this procedure -> ONE success trial -> alpha=PRIOR+1, beta=PRIOR.
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    # Model the failed attempt the way the engine does: increment the counter, commit no step.
    await journal.increment_attempt("r1", 0)
    # The successful (re-)attempt commits a success at step_index 1 (positional keying: new seq).
    await _commit(
        journal,
        run_id="r1",
        step_index=1,
        stage=_STAGE,
        output=_stamped_artifact(registry, outcome="success"),
        committed_at=_T0,
    )
    await _projector(journal, kg, registry).tick()

    procedure_id, problem_type = _decl(registry)
    posterior = await kg.posterior(procedure_id, problem_type)
    # Beta(1,1) prior + one success: alpha=2, beta=1, one deduped contribution.
    assert (posterior.alpha, posterior.beta, posterior.n_trials) == (2.0, 1.0, 1)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.trial_id for t in trials] == ["r1:1"]


async def test_a3_negative_control_double_count_would_break_posterior() -> None:
    # NEGATIVE CONTROL for a3: if a failed attempt HAD committed a step (the bug the journal design
    # forecloses), the projector would see TWO steps and the posterior would be success+failure, not
    # a clean single success. Prove the assertion in a3 is sensitive to that by simulating it.
    registry = _registry()
    failure_step = StepRecord(
        run_id="r1",
        step_index=0,
        stage_name=_STAGE,
        result=Transition(to=_FIXUP, output=_stamped_artifact(registry, outcome="failure")),
        committed_at=_T0,
    )
    success_step = StepRecord(
        run_id="r1",
        step_index=1,
        stage_name=_STAGE,
        result=Done(output=_stamped_artifact(registry, outcome="success")),
        committed_at=_T0,
    )
    journal, kg = InMemoryJournal(), InMemoryProceduralKG()
    await _start(journal, "r1")
    await journal.commit_step(failure_step)
    await journal.commit_step(success_step)
    await _projector(journal, kg, registry).tick()

    procedure_id, problem_type = _decl(registry)
    posterior = await kg.posterior(procedure_id, problem_type)
    # With the bug, the same-run dedup keeps n_trials at 1 but polarity is split -> NOT the clean
    # alpha=2,beta=1 of a3. This is exactly what the journal write-once design prevents.
    assert (posterior.alpha, posterior.beta) != (2.0, 1.0)


async def test_a4_rerun_is_byte_identical() -> None:
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    for i, outcome in enumerate(["success", "success", "failure"]):
        await _commit(
            journal,
            run_id="r1",
            step_index=i,
            stage=_STAGE,
            output=_stamped_artifact(registry, outcome=outcome),  # type: ignore[arg-type]
            committed_at=_T0 + timedelta(seconds=i),
        )
    projector = _projector(journal, kg, registry)
    await projector.tick()
    first = await _subgraph_snapshot(kg, registry)
    await projector.tick()
    await projector.tick()
    second = await _subgraph_snapshot(kg, registry)

    assert first == second


async def test_a6_cursor_advances_in_trial_write() -> None:
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    commit_time = _T0 + timedelta(minutes=5)
    await _commit(
        journal,
        run_id="r1",
        step_index=0,
        stage=_STAGE,
        output=_stamped_artifact(registry, outcome="success"),
        committed_at=commit_time,
    )
    assert await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER) is None
    await _projector(journal, kg, registry).tick()

    # The cursor moved to the projected step (commit-order), written atomically with the trial.
    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", 0)
    assert cursor.commit_ordinal >= 1
    assert await kg.get_trial("r1:0") is not None


async def test_control_step_advances_cursor_without_trial() -> None:
    # The P0-2 FIX: a non-procedure committed step writes NO trial but the cursor STILL advances to
    # that scanned step's commit-order, so a burst of control steps can never strand the watermark
    # at cold start.
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    plain = Artifact(kind="note", produced_by="router", provenance=_system_prov(), data={})
    await journal.commit_step(
        StepRecord(
            run_id="r1",
            step_index=0,
            stage_name="router_stage",
            result=Done(output=plain),
            committed_at=_T0,
        )
    )
    projected = await _projector(journal, kg, registry).tick()
    assert projected == 0
    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", 0)
    # No orphan trial was written for the control step.
    assert await kg.get_trial("r1:0") is None


async def test_empty_journal_leaves_cursor_unchanged() -> None:
    # A 0-row tick returns 0 and never advances the cursor (no last-scanned row to advance to).
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    projected = await _projector(journal, kg, registry).tick()
    assert projected == 0
    assert await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER) is None


async def test_reprojection_is_idempotent() -> None:
    # Re-reads already-projected steps every tick (cursor at the end -> 0-row reads); the MERGE
    # makes that a no-op. Drive several ticks over a static journal: the trial count never grows.
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    await _commit(
        journal,
        run_id="r1",
        step_index=0,
        stage=_STAGE,
        output=_stamped_artifact(registry, outcome="success"),
        committed_at=_T0,
    )
    projector = _projector(journal, kg, registry)
    for _ in range(5):
        await projector.tick()

    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.trial_id for t in trials] == ["r1:0"]


async def test_failure_trial_synthesis_from_exhaustion_degrade() -> None:
    # The survivorship fix: an engine-synthesized retry-exhaustion Degrade carries NO procedure_id,
    # but the projector recovers (procedure_id, problem_type) from the registry and records a
    # FAILURE trial — so hard failures still count against the posterior.
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    await _commit(
        journal,
        run_id="r1",
        step_index=0,
        stage=_STAGE,
        output=_exhaustion_artifact(),
        committed_at=_T0,
        transition_to=None,
    )
    projected = await _projector(journal, kg, registry).tick()

    assert projected == 1
    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.outcome for t in trials] == ["failure"]
    posterior = await kg.posterior(procedure_id, problem_type)
    assert (posterior.alpha, posterior.beta) == (1.0, 2.0)


async def test_exhaustion_degrade_on_unregistered_stage_is_skipped() -> None:
    # NEGATIVE CONTROL for synthesis: an exhaustion-Degrade on a stage that is NOT a declared
    # procedure has no (procedure_id, problem_type) to recover, so it is skipped (no orphan trial).
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    await _commit(
        journal,
        run_id="r1",
        step_index=0,
        stage="unregistered_stage",
        output=_exhaustion_artifact(),
        committed_at=_T0,
    )
    projected = await _projector(journal, kg, registry).tick()
    assert projected == 0


async def test_multi_run_dedup_grain() -> None:
    # Two DISTINCT runs each succeed once -> two deduped contributions (n_trials == 2). Confirms the
    # projector keys trials per (run_id, step_index) and the posterior dedups at run_id.
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    for run_id in ("r1", "r2"):
        await _start(journal, run_id)
        await _commit(
            journal,
            run_id=run_id,
            step_index=0,
            stage=_STAGE,
            output=_stamped_artifact(registry, outcome="success"),
            committed_at=_T0,
        )
    await _projector(journal, kg, registry).tick()

    procedure_id, problem_type = _decl(registry)
    posterior = await kg.posterior(procedure_id, problem_type)
    assert posterior.n_trials == 2
    assert (posterior.alpha, posterior.beta) == (3.0, 1.0)


@pytest.mark.xfail(
    reason=(
        "KNOWN LIMITATION (flagged for Jim): a FAILED-at-seq run (on_exhausted='fail') commits "
        "NOTHING at the failing seq, so the journal-poll projector cannot observe it. Observing "
        "never-committed terminal failures needs a run-status projection seam (engine/pathway "
        "territory), deferred. The exhaustion-Degrade failure path IS projected fully."
    ),
    strict=True,
)
async def test_failed_at_seq_not_projected_known_limitation() -> None:
    # Model on_exhausted='fail': the attempt counter climbs, the run flips to FAILED, and NO step is
    # committed at the failing seq. The projector sees no row -> no failure trial. This xfail PINS
    # the gap: when the run-status projection seam lands, this test should start PASSING (strict
    # xfail makes an unexpected pass a failure, so the limitation cannot be silently closed).
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    await journal.increment_attempt("r1", 0)
    await journal.set_run_status("r1", RunStatus.FAILED)
    await _projector(journal, kg, registry).tick()

    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert [t.outcome for t in trials] == ["failure"]


# ---------------------------------------------------------------------------
# Forward-progress (FP) — the commit-order keyset, mutation-resistant with negative controls
# ---------------------------------------------------------------------------


async def _buggy_scalar_tick(
    journal: InMemoryJournal,
    kg: InMemoryProceduralKG,
    registry: ProcedureRegistry,
    *,
    batch_limit: int,
    _state: dict[str, int],
) -> int:
    """A BUGGY tick that advances a SCALAR commit_ordinal max of PROCESSED trials only, for the FP
    negative controls.

    Reads strictly past a scalar ``commit_ordinal`` floor (NOT the full keyset, NOT advancing on
    control batches): it stores only the max ordinal of a PROJECTED trial, pinning run_id/step_index
    to the minima. When the first batch is all control steps it never advances (re-livelocks P0-2 at
    cold start); when many rows share progress it cannot resume strictly past them. The real tick
    uses the full-keyset strict bound and advances on the LAST SCANNED row, killing both.
    """
    floor = _state.get("ordinal", 0)
    cursor = ProjectionCursor(commit_ordinal=floor, run_id="", step_index=0) if floor > 0 else None
    scanned = await journal.committed_steps_after(cursor, limit=batch_limit)
    projected = 0
    for ps in scanned:
        step = ps.record
        run = await journal.load_run(step.run_id)
        if run is None:
            continue
        resolved = resolve_outcome(step, pathway_id=run.pathway_id, registry=registry)
        if resolved is None:
            continue
        await kg.record_trial(
            trial_id=f"{step.run_id}:{step.step_index}",
            procedure_id=resolved.procedure_id,
            problem_type=resolved.problem_type,
            outcome=resolved.outcome,
            occurred_at=step.committed_at,
            provenance=_system_prov(step.committed_at),
        )
        # The bug: advance the scalar floor only to a PROJECTED trial's ordinal.
        _state["ordinal"] = max(_state.get("ordinal", 0), ps.commit_ordinal)
        projected += 1
    return projected


async def test_fp_completeness_under_overflow_and_control_burst() -> None:
    # FP (kills the P0 livelock): N steps committed, N > batch_limit, ordered so the first
    # batch_limit rows are NON-procedure control and the procedure rows sit BEHIND them in commit
    # order. The fixed projector must drain every procedure step into exactly one Trial in a
    # bounded number of ticks, and the stored cursor must end at the last committed step.
    batch_limit = 4
    controls = 4  # exactly fills the first batch with control steps
    procedures = 5
    n = controls + procedures
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    # Commit controls first, then procedures: the commit-order ordinal puts controls ahead.
    for i in range(controls):
        await _commit_control(journal, run_id="r1", step_index=i, committed_at=_T0)
    for j in range(procedures):
        await _commit(
            journal,
            run_id="r1",
            step_index=controls + j,
            stage=_STAGE,
            output=_stamped_artifact(registry, outcome="success"),
            committed_at=_T0,
        )

    projector = _projector(journal, kg, registry, batch_limit=batch_limit)
    max_ticks = math.ceil(n / batch_limit) + 1
    for _ in range(max_ticks):
        await projector.tick()

    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    # a1 holds: exactly one Trial per committed procedure step.
    assert sorted(t.trial_id for t in trials) == [f"r1:{controls + j}" for j in range(procedures)]
    assert len(trials) == procedures
    # The stored cursor is at the last committed step (run r1, step n-1).
    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", n - 1)


async def test_fp_negative_control_scalar_advance_livelocks_on_control_burst() -> None:
    # NEGATIVE CONTROL for FP: the buggy scalar-ordinal/advance-on-trial logic LIVELOCKS on the
    # control burst — the first batch is all control steps (zero trials), so the floor never moves
    # off cold start and every subsequent tick re-reads the SAME first batch forever. No procedure
    # trial is ever projected.
    batch_limit = 4
    controls = 4
    procedures = 5
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    for i in range(controls):
        await _commit_control(journal, run_id="r1", step_index=i, committed_at=_T0)
    for j in range(procedures):
        await _commit(
            journal,
            run_id="r1",
            step_index=controls + j,
            stage=_STAGE,
            output=_stamped_artifact(registry, outcome="success"),
            committed_at=_T0,
        )

    state: dict[str, int] = {}
    for _ in range(20):  # far more than ceil(N/B)+1 — still stuck
        await _buggy_scalar_tick(journal, kg, registry, batch_limit=batch_limit, _state=state)

    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert trials == ()  # livelocked: zero progress


async def test_fp_strict_monotone_advance() -> None:
    # FP: across ticks returning >=1 row the stored cursor strictly increases lexicographically; a
    # 0-row tick leaves it unchanged. With one procedure step per batch and rows still unscanned, no
    # tick re-stores the same cursor.
    steps = 4
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    for i in range(steps):
        await _commit(
            journal,
            run_id="r1",
            step_index=i,
            stage=_STAGE,
            output=_stamped_artifact(registry, outcome="success"),
            committed_at=_T0 + timedelta(seconds=i),
        )

    projector = _projector(journal, kg, registry, batch_limit=1)
    seen: list[ProjectionCursor] = []
    for _ in range(steps):
        await projector.tick()
        cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
        assert cursor is not None
        if seen:
            assert not ordinal_ge(seen[-1], cursor)  # strictly greater: prev is NOT >= cursor
        seen.append(cursor)

    before = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    projected = await projector.tick()  # now drained -> 0-row tick
    after = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    assert projected == 0
    assert before == after


async def test_fp_advances_past_rows_sharing_commit_order_tiebreak() -> None:
    # FP: many rows committed in one batch_limit-exceeding burst at the SAME wall-clock
    # committed_at. commit_ordinal is assigned per-commit (strictly increasing), so even with
    # identical committed_at the full-keyset strict bound resumes strictly past the last scanned row
    # and drains every step — the old scalar-committed_at livelock is structurally impossible now.
    batch_limit = 2
    journal, kg, registry = InMemoryJournal(), InMemoryProceduralKG(), _registry()
    await _start(journal, "r1")
    for i in range(5):  # 5 > batch_limit, all at the SAME committed_at
        await _commit(
            journal,
            run_id="r1",
            step_index=i,
            stage=_STAGE,
            output=_stamped_artifact(registry, outcome="success"),
            committed_at=_T0,
        )

    projector = _projector(journal, kg, registry, batch_limit=batch_limit)
    for _ in range(math.ceil(5 / batch_limit) + 1):
        await projector.tick()

    procedure_id, problem_type = _decl(registry)
    trials = await kg.trials_for(procedure_id, problem_type)
    assert sorted(t.trial_id for t in trials) == [f"r1:{i}" for i in range(5)]
    cursor = await kg.read_cursor(DEFAULT_PROJECTION_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", 4)
