"""Unit tests for cogworx.knowledge.procedural_confidence (CANON S1, S6).

Pins the read-side Beta-posterior success derivation for procedural-KG edges:
  - Empty trials → prior-only (Beta(1,1), success_rate 0.5, n_trials 0).
  - One success → alpha=2.0, beta=1.0, success_rate≈0.667 (unit trial weight).
  - One failure → alpha=1.0, beta=2.0, success_rate≈0.333.
  - RUN-LEVEL DEDUP GRAIN: 50 trials from ONE run = ONE contribution (n_trials 1), not 50.
  - DEDUPED-COUNT-NOT-RAW-COUNT floor property: n_trials counts distinct (run_id, polarity), so a
    promotion floor reads the deduped count and a single noisy run cannot inflate past the floor.
  - A run that both succeeds and fails contributes to BOTH polarities (dedup key is per polarity).

Hypothesis property tests:
  - success_rate always strictly in (0, 1); variance > 0.
  - n_trials == number of distinct (run_id, polarity) pairs (never the raw trial count).
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cogworx.knowledge.procedural_confidence import (
    ProcedureSuccess,
    TrialOutcome,
    procedure_success,
)

# ---------------------------------------------------------------------------
# Deterministic cases
# ---------------------------------------------------------------------------


def test_no_trials_prior_only() -> None:
    result = procedure_success([])
    assert result.alpha == pytest.approx(1.0)
    assert result.beta == pytest.approx(1.0)
    assert result.success_rate == pytest.approx(0.5)
    assert result.n_trials == 0
    assert result.variance > 0


def test_one_success() -> None:
    """One success from one run: unit weight → alpha=2.0, beta=1.0, mean=2/3."""
    result = procedure_success([TrialOutcome(run_id="run-1", success=True)])
    assert result.alpha == pytest.approx(2.0)
    assert result.beta == pytest.approx(1.0)
    assert result.success_rate == pytest.approx(2.0 / 3.0)
    assert result.n_trials == 1


def test_one_failure() -> None:
    """One failure from one run: unit weight → alpha=1.0, beta=2.0, mean=1/3."""
    result = procedure_success([TrialOutcome(run_id="run-1", success=False)])
    assert result.alpha == pytest.approx(1.0)
    assert result.beta == pytest.approx(2.0)
    assert result.success_rate == pytest.approx(1.0 / 3.0)
    assert result.n_trials == 1


def test_distinct_runs_each_count() -> None:
    """Two successes from DIFFERENT runs both contribute → alpha=3.0, n_trials=2."""
    result = procedure_success(
        [TrialOutcome(run_id="run-1", success=True), TrialOutcome(run_id="run-2", success=True)]
    )
    assert result.alpha == pytest.approx(3.0)
    assert result.beta == pytest.approx(1.0)
    assert result.n_trials == 2


def test_run_level_dedup_50_same_run_trials_is_one_contribution() -> None:
    """50 successful trials from the SAME run = ONE contribution (pseudo-replication guard).

    A cyclic run applying the procedure 50 times must not inflate the posterior or the floor.
    Deduped at source_id = run_id: alpha = prior 1.0 + ONE unit = 2.0, n_trials = 1 (not 50).
    """
    trials = [TrialOutcome(run_id="run-1", success=True) for _ in range(50)]
    result = procedure_success(trials)
    assert result.n_trials == 1
    assert result.alpha == pytest.approx(2.0)
    assert result.beta == pytest.approx(1.0)


def test_same_run_both_polarities_counts_twice() -> None:
    """A run with a success AND a failure contributes to both alpha and beta (n_trials=2).

    The dedup key is (run_id, polarity) — two distinct pairs from one run.
    """
    result = procedure_success(
        [TrialOutcome(run_id="run-1", success=True), TrialOutcome(run_id="run-1", success=False)]
    )
    assert result.alpha == pytest.approx(2.0)
    assert result.beta == pytest.approx(2.0)
    assert result.n_trials == 2
    assert result.success_rate == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# The deduped-count-not-raw-count promotion-floor property
# ---------------------------------------------------------------------------


def test_floor_uses_deduped_count_not_raw_trials() -> None:
    """A promotion floor of n>=5 must NOT be cleared by 50 trials all from the same run.

    This is the load-bearing invariant: n_trials is the deduped contribution count, so a single run
    (however many trials it logged) counts as one and cannot pass the floor on its own.
    """
    floor = 5
    one_run_many_trials = [TrialOutcome(run_id="run-1", success=True) for _ in range(50)]
    assert procedure_success(one_run_many_trials).n_trials < floor

    # Five DISTINCT runs DO clear the floor.
    five_runs = [TrialOutcome(run_id=f"run-{i}", success=True) for i in range(5)]
    assert procedure_success(five_runs).n_trials >= floor


def test_n_trials_is_never_raw_trial_count() -> None:
    """Across a mixed bag, n_trials equals distinct (run_id, polarity), never len(trials)."""
    trials = [
        TrialOutcome(run_id="run-1", success=True),
        TrialOutcome(run_id="run-1", success=True),  # dup pair, dropped
        TrialOutcome(run_id="run-1", success=False),  # distinct polarity
        TrialOutcome(run_id="run-2", success=True),
    ]
    result = procedure_success(trials)
    assert len(trials) == 4
    assert result.n_trials == 3  # (run-1,+), (run-1,-), (run-2,+)


# ---------------------------------------------------------------------------
# ProcedureSuccess is frozen
# ---------------------------------------------------------------------------


def test_procedure_success_is_frozen() -> None:
    import pydantic

    result = procedure_success([])
    with pytest.raises((pydantic.ValidationError, TypeError)):
        result.success_rate = 0.99  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------


@st.composite
def _trials_st(draw: st.DrawFn) -> list[TrialOutcome]:
    n = draw(st.integers(min_value=0, max_value=20))
    return [
        TrialOutcome(
            run_id=draw(st.text(alphabet="abc", min_size=1, max_size=3)),
            success=draw(st.booleans()),
        )
        for _ in range(n)
    ]


@given(_trials_st())
@settings(max_examples=200)
def test_success_rate_always_in_open_unit_interval(trials: list[TrialOutcome]) -> None:
    result = procedure_success(trials)
    assert 0.0 < result.success_rate < 1.0


@given(_trials_st())
@settings(max_examples=200)
def test_variance_always_positive(trials: list[TrialOutcome]) -> None:
    assert procedure_success(trials).variance > 0.0


@given(_trials_st())
@settings(max_examples=200)
def test_n_trials_equals_distinct_run_polarity_pairs(trials: list[TrialOutcome]) -> None:
    """n_trials == distinct (run_id, polarity) pairs — the deduped count, never the raw count."""
    distinct = {(t.run_id, t.success) for t in trials}
    result: ProcedureSuccess = procedure_success(trials)
    assert result.n_trials == len(distinct)
    assert result.n_trials <= len(trials)
