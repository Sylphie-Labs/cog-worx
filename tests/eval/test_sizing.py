"""Fast smoke tests for the Pod 4.4c-0 pre-nightly sizing sim (plan §6.0).

These exercise the EXACT landed code path (``synth_cells`` + ``nested_bootstrap_delta`` +
``power_lcb_from_studies``) at tiny ``n_outer``/``N_studies`` so the path is covered under pytest
without the heavy wall-clock. The full heavy sweep is the DIRECT script invocation
(``python -m cogworx.eval.sizing``), kept out of pytest per the test-hang discipline. No model
calls anywhere (S1/S6).
"""

from __future__ import annotations

from cogworx.eval import sizing
from cogworx.eval.sizing import MASTER_SEED, run_sizing, run_study_loop


def test_fast_sizing_runs_the_landed_code_path() -> None:
    """The fast smoke produces a populated, reproducible artifact via the landed bootstrap."""
    artifact = run_sizing(fast=True)
    assert artifact.master_seed == MASTER_SEED
    assert artifact.grid_corrected, "fast sweep produced no corrected-quantile results"
    assert artifact.grid_nominal, "fast sweep produced no nominal-quantile results"
    assert artifact.decision in {"E1", "E2", "ABORT"} or artifact.decision.startswith("escalate-n=")
    for r in artifact.grid_corrected:
        assert 0 <= r.clears <= r.n_studies
        assert 0.0 <= r.power_lcb <= 1.0


def test_fast_sizing_is_deterministic() -> None:
    """Same MASTER_SEED -> identical artifact (reproducibility is the F2-audit precondition)."""
    a = run_sizing(fast=True)
    b = run_sizing(fast=True)
    assert a.to_json() == b.to_json()


def test_study_seed_sequence_is_shared_across_n_arms() -> None:
    """CRN build constraint: the 56 and 80 arms must draw the SAME study-seed sequence so the
    paired gap resolves. We pin that ``run_study_loop`` is seed-driven, not n-driven, by feeding
    one seed list to two n and confirming both consume it without divergence in length."""
    seeds = sizing._study_seeds(MASTER_SEED, 8)
    r56 = run_study_loop(
        n=56,
        sb_sens=0.020,
        sb_spec=0.010,
        dsens=0.15,
        dspec=0.15,
        quantile=0.0025,
        n_studies=8,
        n_outer=120,
        study_seeds=seeds,
    )
    r80 = run_study_loop(
        n=80,
        sb_sens=0.020,
        sb_spec=0.010,
        dsens=0.15,
        dspec=0.15,
        quantile=0.0025,
        n_studies=8,
        n_outer=120,
        study_seeds=seeds,
    )
    assert r56.n_studies == r80.n_studies == 8
    assert sizing._study_seeds(MASTER_SEED, 8) == seeds, "study-seed sequence is not reproducible"


def test_true_null_companion_does_not_certify() -> None:
    """A true-null draw (dsens=dspec=0) should NOT spuriously clear at the tiny smoke scale: the
    bootstrap CI must straddle zero often enough that power_lcb stays low (the test can fail)."""
    seeds = sizing._study_seeds(MASTER_SEED, 12)
    null = run_study_loop(
        n=80,
        sb_sens=0.020,
        sb_spec=0.010,
        dsens=0.0,
        dspec=0.0,
        quantile=0.0025,
        n_studies=12,
        n_outer=200,
        study_seeds=seeds,
    )
    assert null.power_lcb < 0.80, f"true-null power_lcb={null.power_lcb:.3f} above bar"


def test_decide_aborts_when_no_n_clears() -> None:
    """The terminal branch returns ABORT when nothing in the ladder clears the bar at the
    pessimistic corner -- the sim must NOT silently ship the largest-n-that-looked-close."""
    from cogworx.eval.sizing import StudyLoopResult, _decide

    grid = [
        StudyLoopResult(
            n=n,
            sb_sens=0.045,
            sb_spec=0.028,
            quantile=0.0025,
            n_studies=500,
            n_outer=20000,
            clears=10,
            power_lcb=0.05,
        )
        for n in (56, 80, 100, 120, 150, 200)
    ]
    assert _decide(grid) == "ABORT"


def test_decide_picks_e1_when_both_clear() -> None:
    from cogworx.eval.sizing import StudyLoopResult, _decide

    grid = [
        StudyLoopResult(
            n=n,
            sb_sens=0.045,
            sb_spec=0.028,
            quantile=0.0025,
            n_studies=500,
            n_outer=20000,
            clears=460,
            power_lcb=0.90,
        )
        for n in (56, 80)
    ]
    assert _decide(grid) == "E1"
