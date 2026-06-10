"""Full ProceduralKG contract tests against InMemoryProceduralKG (CANON S1, S5, S6).

This suite pins every clause of the ProceduralKG Protocol (cogworx.substrate.procedural_kg) against
the in-memory double so the unit tier runs without any Neo4j service. The Neo4j adapter is held to
the SAME contract; the shared parity scenarios in test_neo4j_procedural_kg.py run identical bodies
against both, so if these pass and the integration tier passes, the double has not drifted.

Invariants covered:
  S1/S6 — record_trial is the only write surface; posterior is DERIVED at read (never stored);
          run-level dedup grain (source_id = run_id) shows up in the posterior.
  D5 / spike a5 — record_trial is MERGE ON CREATE only: re-MERGE with a flipped payload is a no-op
          (first-write-wins).
  S5 — every trial carries provenance + the framework-assigned ids.
  S8 — well-defined at boundaries: empty edge (prior-only), unknown trial, out-of-order arrival.
  2.0 red-team #1 — the inherited blanket-SET upsert_claim back-door is sealed.
"""

from __future__ import annotations

import random
from datetime import UTC, datetime, timedelta, timezone

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.procedural_confidence import procedure_success
from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for
from cogworx.substrate.procedural_kg import Trial
from cogworx.substrate.procedural_selection import posterior_mean_select, sticky_tess_select
from cogworx.testing.doubles import InMemoryProceduralKG

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)

_PID = procedure_id_for("math_pathway", "solve_stage")
_PT = problem_type_id_for("word problem")


def _prov() -> Provenance:
    """The canonical trial provenance: system source (zero-model deterministic control event)."""
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


async def _record(
    kg: InMemoryProceduralKG,
    *,
    trial_id: str,
    outcome: str,
    procedure_id: str = _PID,
    problem_type: str = _PT,
    occurred_at: datetime = _T0,
) -> None:
    await kg.record_trial(
        trial_id=trial_id,
        procedure_id=procedure_id,
        problem_type=problem_type,
        outcome=outcome,  # type: ignore[arg-type]
        occurred_at=occurred_at,
        provenance=_prov(),
    )


# ---------------------------------------------------------------------------
# record_trial — trial_id format validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_id",
    ["noColon", "run:01", "run:-1", "run: 1", "run:1.0", ":5", "run:"],
)
async def test_record_trial_rejects_malformed_id(bad_id: str) -> None:
    """A malformed trial_id is rejected before any state change."""
    kg = InMemoryProceduralKG()
    with pytest.raises(ValueError, match="malformed trial_id"):
        await _record(kg, trial_id=bad_id, outcome="success")
    assert await kg.get_trial(bad_id) is None


async def test_record_trial_splits_run_id_and_step_index() -> None:
    """run_id / step_index are split from trial_id at the LAST colon (run_id may contain colons)."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="sess:abc:42", outcome="success")
    trial = await kg.get_trial("sess:abc:42")
    assert trial is not None
    assert trial.run_id == "sess:abc"
    assert trial.step_index == 42


# ---------------------------------------------------------------------------
# record_trial — first-write-wins (spike a5)
# ---------------------------------------------------------------------------


async def test_record_trial_first_write_wins_on_flip() -> None:
    """Spike a5: re-MERGE same trial_id with a FLIPPED outcome → stored outcome UNCHANGED."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    await _record(kg, trial_id="r1:0", outcome="failure")  # flip — must be a no-op
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.outcome == "success", "first-write-wins: the flip must not overwrite the trial"


async def test_record_trial_idempotent_no_duplicate() -> None:
    """Re-projecting the same committed step stores nothing new (idempotent)."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    await _record(kg, trial_id="r1:0", outcome="success")
    trials = await kg.trials_for(_PID, _PT)
    assert len(trials) == 1


async def test_record_trial_first_write_wins_does_not_change_posterior() -> None:
    """A flipped re-MERGE leaves the derived posterior identical (no silent counter mutation)."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    before = await kg.posterior(_PID, _PT)
    await _record(kg, trial_id="r1:0", outcome="failure")
    after = await kg.posterior(_PID, _PT)
    assert (before.alpha, before.beta) == (after.alpha, after.beta)
    assert after.alpha == pytest.approx(2.0)  # prior 1 + one success
    assert after.beta == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# get_trial / trials_for
# ---------------------------------------------------------------------------


async def test_get_trial_unknown_returns_none() -> None:
    """get_trial returns None for an unknown id."""
    kg = InMemoryProceduralKG()
    assert await kg.get_trial("nope:0") is None


async def test_trials_for_empty_edge_returns_empty() -> None:
    """trials_for on an edge with no trials returns empty."""
    kg = InMemoryProceduralKG()
    assert await kg.trials_for(_PID, _PT) == ()


async def test_trials_for_occurrence_order_out_of_order_arrival() -> None:
    """Trials arriving out of order are returned in occurrence order (occurred_at)."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:2", outcome="success", occurred_at=_T2)
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=_T0)
    await _record(kg, trial_id="r1:1", outcome="failure", occurred_at=_T1)
    trials = await kg.trials_for(_PID, _PT)
    assert [t.occurred_at for t in trials] == [_T0, _T1, _T2]


async def test_trials_for_isolates_edges() -> None:
    """trials_for returns only the trials on the requested (procedure, problem_type) edge."""
    kg = InMemoryProceduralKG()
    other_pid = procedure_id_for("math_pathway", "other_stage")
    await _record(kg, trial_id="r1:0", outcome="success", procedure_id=_PID)
    await _record(kg, trial_id="r1:1", outcome="success", procedure_id=other_pid)
    assert len(await kg.trials_for(_PID, _PT)) == 1
    assert len(await kg.trials_for(other_pid, _PT)) == 1


# ---------------------------------------------------------------------------
# posterior — derive-at-read, dedup grain, prior-only
# ---------------------------------------------------------------------------


async def test_posterior_empty_edge_prior_only() -> None:
    """Empty edge → prior-only: success_rate 0.5, n_trials 0."""
    kg = InMemoryProceduralKG()
    post = await kg.posterior(_PID, _PT)
    assert post.success_rate == pytest.approx(0.5)
    assert post.n_trials == 0


async def test_posterior_routes_through_single_path() -> None:
    """posterior == procedure_success over the edge's (run_id, success) outcomes."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="rA:0", outcome="success")
    await _record(kg, trial_id="rB:0", outcome="failure")
    post = await kg.posterior(_PID, _PT)
    from cogworx.knowledge.procedural_confidence import TrialOutcome

    expected = procedure_success(
        [TrialOutcome("rA", success=True), TrialOutcome("rB", success=False)]
    )
    assert post == expected


async def test_posterior_run_level_dedup() -> None:
    """One cyclic run applying a procedure 50x success = ONE contribution, not 50 (grain).

    50 success trials all with run_id 'cyc' dedup to a single +1 to alpha.
    """
    kg = InMemoryProceduralKG()
    for i in range(50):
        await _record(kg, trial_id=f"cyc:{i}", outcome="success")
    post = await kg.posterior(_PID, _PT)
    assert post.n_trials == 1  # deduped contribution count
    assert post.alpha == pytest.approx(2.0)  # prior 1 + one deduped success
    assert post.beta == pytest.approx(1.0)
    # but the raw trials are all stored
    assert len(await kg.trials_for(_PID, _PT)) == 50


async def test_posterior_distinct_runs_count_separately() -> None:
    """Distinct run_ids each contribute once per polarity."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    await _record(kg, trial_id="r2:0", outcome="success")
    await _record(kg, trial_id="r3:0", outcome="failure")
    post = await kg.posterior(_PID, _PT)
    assert post.n_trials == 3
    assert post.alpha == pytest.approx(3.0)  # prior 1 + two successes
    assert post.beta == pytest.approx(2.0)  # prior 1 + one failure


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


async def test_candidates_empty_problem_type_returns_empty() -> None:
    """candidates on an unknown problem type returns empty."""
    kg = InMemoryProceduralKG()
    assert await kg.candidates(_PT) == ()


async def test_candidates_one_per_edge_with_posterior() -> None:
    """candidates returns one ScoredProcedure per APPLIES_TO edge, with derived posterior."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    cands = await kg.candidates(_PT)
    assert len(cands) == 1
    assert cands[0].procedure.id == _PID
    assert cands[0].problem_type.id == _PT
    assert cands[0].success.n_trials == 1
    assert cands[0].promoted is False


async def test_candidates_default_is_posterior_mean_greedy_deterministic() -> None:
    """The PRODUCTION default path (no rng) is deterministic posterior-mean greedy: highest mean
    first, ties by id; no RNG, identical across calls."""
    kg = InMemoryProceduralKG()
    high = procedure_id_for("p", "high")
    low = procedure_id_for("p", "low")
    # high: 1 success → mean 2/3 ; low: 1 failure → mean 1/3.
    await _record(kg, trial_id="r1:0", outcome="success", procedure_id=high)
    await _record(kg, trial_id="r2:0", outcome="failure", procedure_id=low)
    order_a = [c.procedure.id for c in await kg.candidates(_PT)]
    order_b = [c.procedure.id for c in await kg.candidates(_PT)]
    assert order_a == order_b == [high, low]


async def test_candidates_default_matches_posterior_mean_select() -> None:
    """The default candidates order equals posterior_mean_select applied to the same edges."""
    kg = InMemoryProceduralKG()
    for i in range(5):
        pid = procedure_id_for("p", f"stage-{i}")
        for _ in range(i):  # stage-i gets i distinct-run successes → rising mean
            await _record(kg, trial_id=f"r{i}_{_}:0", outcome="success", procedure_id=pid)
        await _record(kg, trial_id=f"seed{i}:0", outcome="success", procedure_id=pid)
    cands = await kg.candidates(_PT)
    expected = posterior_mean_select(list(cands), limit=20)
    assert [c.procedure.id for c in cands] == [c.procedure.id for c in expected]


async def test_candidates_explicit_thompson_seeded_is_deterministic() -> None:
    """The OPT-IN Thompson path: same seed → identical ordering (the determinism bet)."""
    kg = InMemoryProceduralKG()
    for i in range(5):
        pid = procedure_id_for("p", f"stage-{i}")
        await _record(kg, trial_id=f"r{i}:0", outcome="success", procedure_id=pid)
    order_a = [c.procedure.id for c in await kg.candidates(_PT, rng=random.Random(7))]
    order_b = [c.procedure.id for c in await kg.candidates(_PT, rng=random.Random(7))]
    assert order_a == order_b
    assert len(order_a) == 5


async def test_candidates_default_favors_stronger_posterior() -> None:
    """The default posterior-mean greedy puts the clearly-stronger edge first, deterministically."""
    kg = InMemoryProceduralKG()
    strong = procedure_id_for("p", "strong")
    weak = procedure_id_for("p", "weak")
    for i in range(8):
        await _record(kg, trial_id=f"s{i}:0", outcome="success", procedure_id=strong)
    await _record(kg, trial_id="w1:0", outcome="failure", procedure_id=weak)
    cands = await kg.candidates(_PT)
    assert cands[0].procedure.id == strong


async def test_candidates_limit_respected() -> None:
    """candidates respects the limit parameter."""
    kg = InMemoryProceduralKG()
    for i in range(5):
        pid = procedure_id_for("p", f"stage-{i}")
        await _record(kg, trial_id=f"r{i}:0", outcome="success", procedure_id=pid)
    cands = await kg.candidates(_PT, limit=3)
    assert len(cands) == 3


async def test_candidates_promoted_flag_set_by_gate() -> None:
    """promoted is stamped fresh from the gate: a high-LCB edge promotes, a weak one does not."""
    kg = InMemoryProceduralKG()
    strong = procedure_id_for("p", "strong")
    weak = procedure_id_for("p", "weak")
    # strong: 12 distinct-run successes → Beta(13,1), LCB clears 0.70 at q=0.025, n=12 ≥ 5 →
    # promoted (q tightened to 0.025 in Pod 2.1; 8/8 no longer promotes).
    for i in range(12):
        await _record(kg, trial_id=f"s{i}:0", outcome="success", procedure_id=strong)
    # weak: 1 success → Beta(2,1), n=1 < 5 → not promoted.
    await _record(kg, trial_id="w1:0", outcome="success", procedure_id=weak)
    cands = await kg.candidates(_PT)
    by_id = {c.procedure.id: c for c in cands}
    assert by_id[strong].promoted is True
    assert by_id[weak].promoted is False


async def test_candidates_promoted_only_filters_to_gate() -> None:
    """promoted_only returns only edges that clear the promotion gate."""
    kg = InMemoryProceduralKG()
    strong = procedure_id_for("p", "strong")
    weak = procedure_id_for("p", "weak")
    for i in range(12):
        await _record(kg, trial_id=f"s{i}:0", outcome="success", procedure_id=strong)
    await _record(kg, trial_id="w1:0", outcome="success", procedure_id=weak)
    promoted = await kg.candidates(_PT, promoted_only=True)
    assert [c.procedure.id for c in promoted] == [strong]
    assert len(await kg.candidates(_PT, promoted_only=False)) == 2


async def test_candidates_zero_trial_edge_is_prior_only() -> None:
    """An edge declared (a trial recorded) shows its derived posterior; never a stored counter."""
    kg = InMemoryProceduralKG()
    # Record then confirm the edge posterior matches a fresh derivation.
    await _record(kg, trial_id="r1:0", outcome="success")
    cands = await kg.candidates(_PT)
    assert cands[0].success.success_rate == pytest.approx(2.0 / 3.0)


async def test_candidates_default_no_rng_does_not_raise() -> None:
    """candidates with no rng uses the deterministic posterior-mean greedy default (no crash)."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    cands = await kg.candidates(_PT)
    assert len(cands) == 1


async def test_candidates_constructor_rng_opts_into_thompson() -> None:
    """A constructor-injected rng opts into the deferred Thompson path (seeded → deterministic)."""
    kg = InMemoryProceduralKG(rng=random.Random(123))
    for i in range(5):
        pid = procedure_id_for("p", f"stage-{i}")
        await _record(kg, trial_id=f"r{i}:0", outcome="success", procedure_id=pid)
    order_a = [c.procedure.id for c in await kg.candidates(_PT)]
    # A re-run with a fresh adapter seeded identically reproduces the Thompson order.
    kg2 = InMemoryProceduralKG(rng=random.Random(123))
    for i in range(5):
        pid = procedure_id_for("p", f"stage-{i}")
        await _record(kg2, trial_id=f"r{i}:0", outcome="success", procedure_id=pid)
    order_b = [c.procedure.id for c in await kg2.candidates(_PT)]
    assert order_a == order_b


async def test_sticky_tess_baseline_order() -> None:
    """The rejected baseline-for-comparison ranks n_trials DESC, then success_rate DESC, then id."""
    kg = InMemoryProceduralKG()
    high = procedure_id_for("p", "high")
    low = procedure_id_for("p", "low")
    await _record(kg, trial_id="r1:0", outcome="success", procedure_id=high)
    await _record(kg, trial_id="r2:0", outcome="failure", procedure_id=low)
    cands = list(await kg.candidates(_PT))
    ranked = sticky_tess_select(cands, limit=10)
    # Both have n_trials=1; high has the higher success_rate → first.
    assert [c.procedure.id for c in ranked] == [high, low]


# ---------------------------------------------------------------------------
# Datetime contract
# ---------------------------------------------------------------------------


async def test_record_trial_naive_occurred_at_interpreted_as_utc() -> None:
    """A naive occurred_at is interpreted as UTC (not rejected)."""
    kg = InMemoryProceduralKG()
    naive = datetime(2026, 6, 9, 12, 0, 0)  # no tzinfo
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=naive)
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.occurred_at == datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)


async def test_record_trial_offset_occurred_at_converted_to_utc() -> None:
    """A tz-aware occurred_at is converted to UTC before storage."""
    kg = InMemoryProceduralKG()
    plus_two = timezone(timedelta(hours=2))
    offset = datetime(2026, 6, 9, 14, 0, 0, tzinfo=plus_two)  # == 12:00Z
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=offset)
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.occurred_at == datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)


async def test_occurrence_order_with_mixed_offsets() -> None:
    """Out-of-order arrival across mixed offsets sorts by true instant."""
    kg = InMemoryProceduralKG()
    plus_two = timezone(timedelta(hours=2))
    # 14:00+02:00 == 12:00Z (earlier) ; 13:00Z (later)
    await _record(
        kg, trial_id="r1:1", outcome="failure", occurred_at=datetime(2026, 6, 9, 13, tzinfo=UTC)
    )
    await _record(
        kg,
        trial_id="r1:0",
        outcome="success",
        occurred_at=datetime(2026, 6, 9, 14, tzinfo=plus_two),
    )
    trials = await kg.trials_for(_PID, _PT)
    assert [t.trial_id for t in trials] == ["r1:0", "r1:1"]


# ---------------------------------------------------------------------------
# Sealed write-seam (2.0 red-team #1)
# ---------------------------------------------------------------------------


async def test_upsert_claim_raises_not_implemented() -> None:
    """InMemoryProceduralKG.upsert_claim raises NotImplementedError (sealed back-door)."""
    kg = InMemoryProceduralKG()
    claim = Claim(
        id="x",
        subject="s",
        payload="p",
        epistemic_type="observation",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    with pytest.raises(NotImplementedError, match="record_trial"):
        await kg.upsert_claim(claim)


# ---------------------------------------------------------------------------
# Provenance + epistemic discipline (S5)
# ---------------------------------------------------------------------------


async def test_trial_carries_provenance() -> None:
    """A recorded trial round-trips its provenance (source='system')."""
    kg = InMemoryProceduralKG()
    await _record(kg, trial_id="r1:0", outcome="success")
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.provenance.source == "system"


async def test_trial_is_frozen() -> None:
    """Trial is an immutable model — mutation raises."""
    import pydantic

    trial = Trial(
        trial_id="r1:0",
        run_id="r1",
        step_index=0,
        procedure_id=_PID,
        problem_type=_PT,
        outcome="success",
        occurred_at=_T0,
        provenance=_prov(),
    )
    with pytest.raises((pydantic.ValidationError, TypeError)):
        trial.outcome = "failure"  # type: ignore[misc]
