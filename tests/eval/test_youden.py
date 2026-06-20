"""Deterministic unit tests for the Phase-4 GATE statistics core (Pod 4.4a).

Mutation-resistant per plan sec 8: assert robust signs/orderings, not fragile magnitudes -- EXCEPT
the MC-vs-``lcb`` cross-check, whose tolerance is pinned to the MC standard error. Every test is
seeded and fast (small ``n_outer``); none calls a model or the substrate.
"""

from __future__ import annotations

from random import Random

import pytest

from cogworx.eval import youden
from cogworx.eval.youden import (
    Cell,
    mc_proportion_lcb,
    nested_bootstrap_delta,
    power_lcb_from_studies,
    realized_variance_diagnostic,
    synth_cells,
)
from cogworx.knowledge import beta

# Planning inputs (plan sec 13.6): mean sens 0.60 / spec 0.85, sigma2_b,sens=0.020,
# sigma2_b,spec=0.010, rho_b~=0.5, rho_w=0.3; Deltamin split dsens=0.10, dspec=0.05.
_SENS_C = 0.60
_SPEC_C = 0.85
_SB_SENS = 0.020
_SB_SPEC = 0.010
_DSENS = 0.10
_DSPEC = 0.05
_RHO_W = 0.30
_R = 7


def _make_cells(
    rng: Random,
    *,
    m_K: int = 80,
    m_clean: int = 80,
    dsens: float = _DSENS,
    dspec: float = _DSPEC,
) -> list[Cell]:
    return synth_cells(
        rng,
        m_K,
        m_clean,
        _R,
        sens_C=_SENS_C,
        dsens=dsens,
        sb_sens=_SB_SENS,
        spec_C=_SPEC_C,
        dspec=dspec,
        sb_spec=_SB_SPEC,
        rho_w=_RHO_W,
    )


# ---------------------------------------------------------------------------
# MC-vs-lcb cross-check (the only magnitude-pinned test)
# ---------------------------------------------------------------------------


def test_mc_proportion_lcb_matches_analytic_lcb_within_mc_se() -> None:
    """The MC sampler's 5th percentile agrees with the analytic Beta LCB within MC SE."""
    alpha, beta_param = 18.0, 12.0
    samples = 200_000
    mc = mc_proportion_lcb(alpha, beta_param, samples=samples, seed=4_4_2026)
    analytic = beta.lcb(alpha, beta_param, quantile=0.05)
    assert abs(mc - analytic) < 0.01, f"MC {mc:.4f} drifted from analytic lcb {analytic:.4f}"


def test_mc_proportion_lcb_rises_with_evidence() -> None:
    """More successes (higher alpha, fixed total) push the lower bound up (sign, not magnitude)."""
    weak = mc_proportion_lcb(6.0, 4.0, samples=50_000, seed=1)
    strong = mc_proportion_lcb(60.0, 40.0, samples=50_000, seed=1)
    assert strong > weak


# ---------------------------------------------------------------------------
# Shared-schema pin
# ---------------------------------------------------------------------------


def test_synth_cells_feeds_bootstrap_without_adaptation() -> None:
    """synth_cells output is consumed by nested_bootstrap_delta directly (sec 13.6 invariant)."""
    cells = _make_cells(Random(0), m_K=20, m_clean=20)
    assert all(isinstance(c, Cell) for c in cells)
    mean_delta, lo, hi = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=200, seed=7
    )
    assert lo <= mean_delta <= hi


def test_cell_schema_is_exactly_the_frozen_record() -> None:
    """The frozen-artifact schema is pinned: drift here breaks the sizing/gate shared code path."""
    assert set(Cell.model_fields) == {
        "item_id",
        "stratum",
        "arm",
        "trial",
        "seed",
        "flagged",
        "route",
        "regime",
    }


# ---------------------------------------------------------------------------
# R5 quantile-kwarg pins (4.4c-0): additive, default-preserving
# ---------------------------------------------------------------------------


def test_quantile_default_reproduces_shipped_ci_byte_for_byte() -> None:
    """The default ``quantile=0.025`` is the shipped 2.5/97.5 CI: passing it explicitly is identical
    to the pre-kwarg default. Pins R5's backward-compatibility contract on a fixed seed."""
    cells = _make_cells(Random(0), m_K=30, m_clean=30)
    shipped = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=300, seed=7
    )
    explicit = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=300, seed=7, quantile=0.025
    )
    assert shipped == explicit
    # The endpoints land on the 2.5th/97.5th order statistics for this n_outer.
    assert shipped[1] == explicit[1]
    assert shipped[2] == explicit[2]


def test_quantile_selects_requested_order_statistic() -> None:
    """``quantile=0.0025`` selects the ``int(0.0025 * n_outer)`` order statistic for ``lo`` and the
    matching upper endpoint -- a strictly wider CI than the 0.025 default on the same draws."""
    cells = _make_cells(Random(1), m_K=30, m_clean=30)
    n_outer = 4000
    _, lo_default, hi_default = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=n_outer, seed=7, quantile=0.025
    )
    _, lo_corrected, hi_corrected = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=n_outer, seed=7, quantile=0.0025
    )
    # int(0.0025*4000)=10 < int(0.025*4000)=100: the deeper-tail order statistic is <= shallower.
    assert lo_corrected <= lo_default
    assert hi_corrected >= hi_default


# ---------------------------------------------------------------------------
# Bootstrap urn-size pins
# ---------------------------------------------------------------------------


def _spy_randrange_moduli(
    monkeypatch: pytest.MonkeyPatch, cells: list[Cell], *, n_outer: int, seed: int
) -> list[int]:
    """Run the bootstrap with the module-level ``Random`` swapped for a spy that records every
    ``randrange`` modulus, so a test can pin the outer (item-count) vs inner (R) urn sizes."""
    calls: list[int] = []

    class _Spy(Random):
        def randrange(self, *args: int, **kwargs: int) -> int:  # type: ignore[override]
            calls.append(args[0])
            return super().randrange(*args, **kwargs)

    monkeypatch.setattr(youden, "Random", _Spy)
    youden.nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=n_outer, seed=seed
    )
    return calls


def test_outer_urn_is_item_count_not_item_times_trials(monkeypatch: pytest.MonkeyPatch) -> None:
    """Outer resample draws over ITEMS (clean count == m_clean), not itemxR cells.

    The spy records ``randrange`` moduli; with m_clean=20, m_K=20 each outer iteration must draw 40
    item-level ids (modulus 20), never 280 (= 40x7), which would mean cell-level resampling.
    """
    cells = _make_cells(Random(1), m_K=20, m_clean=20)
    calls = _spy_randrange_moduli(monkeypatch, cells, n_outer=1, seed=3)

    clean_draws = sum(1 for n in calls if n == 20)
    assert clean_draws == 40, f"expected 40 item-level outer draws, got {clean_draws}"
    assert all(n in (20, 7) for n in calls), "unexpected urn modulus -- outer/inner urns drifted"


def test_clean_pool_resampled_once_per_outer_iteration(monkeypatch: pytest.MonkeyPatch) -> None:
    """The shared clean stratum is resampled exactly once per outer iteration (counter==n_outer)."""
    cells = _make_cells(Random(2), m_K=15, m_clean=15)
    n_outer = 13
    calls = _spy_randrange_moduli(monkeypatch, cells, n_outer=n_outer, seed=9)

    item_draws = sum(1 for n in calls if n == 15)
    # 15 clean + 15 K per outer iteration, once each -> 30/iter, never per-arm (which would double
    # to 60). Resampling clean twice (once per arm) would yield 45/iter.
    assert item_draws == 30 * n_outer, f"clean pool not resampled once/iter: {item_draws}"


# ---------------------------------------------------------------------------
# Degenerate pins (the gate-relevant regressions)
# ---------------------------------------------------------------------------


def _yes_machine_cells(rng: Random, m_K: int, m_clean: int) -> list[Cell]:
    """A yes-machine: flag everything (sens=1, but spec collapses to 0)."""
    cells: list[Cell] = []
    for stratum, base, m in (("K", 0, m_K), ("clean", 10_000, m_clean)):
        for i in range(m):
            for trial in range(_R):
                for arm in ("C", "D"):
                    cells.append(
                        Cell(
                            item_id=base + i,
                            stratum=stratum,
                            arm=arm,
                            trial=trial,
                            seed=(base + i) << 8 ^ trial,
                            flagged=1,
                            route="flag",
                        )
                    )
    return cells


def test_yes_machine_collapses_specificity_and_j() -> None:
    """Flag everything -> spec=0, sens=1 -> J=0 (J is not gameable by flag-spam)."""
    from cogworx.eval.youden import _index_cells, _sens_spec

    cells = _yes_machine_cells(Random(0), 30, 30)
    by_item, strata = _index_cells(cells)
    sens, spec = _sens_spec(by_item, strata["K"], "D", strata["clean"])
    assert sens == 1.0
    assert spec == 0.0
    assert abs(sens + spec - 1.0) < 1e-12


def _coin_judge_cells(rng: Random, m_K: int, m_clean: int) -> list[Cell]:
    """A coin judge: Bernoulli(0.5) flag per (item, trial), CRN-shared across arms. J~=0; the inner
    trial-nest must reinflate the variance so the delta CI contains 0."""
    cells: list[Cell] = []
    for stratum, base, m in (("K", 0, m_K), ("clean", 10_000, m_clean)):
        for i in range(m):
            for trial in range(_R):
                seed = (base + i) << 8 ^ trial
                shared = Random(seed).random()
                flagged = 1 if shared < 0.5 else 0
                for arm in ("C", "D"):
                    cells.append(
                        Cell(
                            item_id=base + i,
                            stratum=stratum,
                            arm=arm,
                            trial=trial,
                            seed=seed,
                            flagged=flagged,
                            route="flag" if flagged else "pass",
                        )
                    )
    return cells


def test_coin_judge_delta_ci_contains_zero() -> None:
    """A coin judge (Bernoulli 0.5, identical across arms) -> delta CI must straddle 0.

    Removing the inner trial-resample would make this spuriously tight and break the test -- the pin
    that the inner nest reinflates within-item flip variance.
    """
    cells = _coin_judge_cells(Random(0), 80, 80)
    _, lo, hi = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=500, seed=11
    )
    assert lo <= 0.0 <= hi, f"coin-judge delta CI [{lo:.4f}, {hi:.4f}] should contain 0"


def _flake_cells(catch_K: int, catch_total: int, m_K: int, m_clean: int) -> list[Cell]:
    """Arm D catches ``catch_K``/``catch_total`` trials on every K item (a flake); arm C catches
    all of them (perfect). Both arms perfect spec on clean. The point delta in J is large but the
    within-item flip variance is huge -> the CI should not clear 0.15."""
    cells: list[Cell] = []
    for i in range(m_K):
        for trial in range(catch_total):
            d_flag = 1 if trial < catch_K else 0
            cells.append(
                Cell(item_id=i, stratum="K", arm="D", trial=trial, seed=i << 8 ^ trial,
                     flagged=d_flag, route="flag" if d_flag else "pass")
            )
            cells.append(
                Cell(item_id=i, stratum="K", arm="C", trial=trial, seed=i << 8 ^ trial,
                     flagged=1, route="flag")
            )
    for i in range(m_clean):
        for trial in range(catch_total):
            for arm in ("C", "D"):
                cells.append(
                    Cell(item_id=10_000 + i, stratum="clean", arm=arm, trial=trial,
                         seed=(10_000 + i) << 8 ^ trial, flagged=0, route="pass")
                )
    return cells


def test_flake_delta_ci_does_not_clear_planning_effect() -> None:
    """A 3/7-vs-7/7 flake: a large point delta in J yet the CI does not clear 0.15.

    (D under-catches relative to C here, so delta(D-C) is negative; the load-bearing pin is that a
    noisy arm does not produce a CI lower bound above the planning Deltamin of 0.15.)
    """
    cells = _flake_cells(catch_K=3, catch_total=7, m_K=80, m_clean=80)
    _, lo, _ = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=500, seed=13
    )
    assert lo < 0.15, f"flake delta CI lo={lo:.4f} unexpectedly cleared the 0.15 planning effect"


# ---------------------------------------------------------------------------
# Pairing-reduction pin
# ---------------------------------------------------------------------------


def test_pairing_reduces_variance_rho_arm_positive() -> None:
    """CRN pairing buys variance reduction: rho_arm > 0 (plan sec 4.3 -- assert it)."""
    cells = _make_cells(Random(5), m_K=80, m_clean=80)
    diag = realized_variance_diagnostic(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=400, seed=17
    )
    assert diag.rho_arm > 0.0, f"rho_arm={diag.rho_arm:.4f} <= 0 -- CRN bought nothing"


def test_unpaired_delta_is_strictly_wider() -> None:
    """An UNPAIRED delta (J(D), J(C) on different draws) yields a strictly wider CI than paired.

    The unpaired width is reconstructed from the diagnostic's per-arm variances: paired Var(delta) <
    Var(J_D)+Var(J_C) iff rho_arm>0, so the paired CI is narrower than the independent-draw CI.
    """
    cells = _make_cells(Random(6), m_K=80, m_clean=80)
    diag = realized_variance_diagnostic(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=400, seed=19
    )
    unpaired_var = diag.var_j_a + diag.var_j_b
    assert diag.var_delta_paired < unpaired_var, (
        f"paired Var(delta)={diag.var_delta_paired:.5f} not < unpaired {unpaired_var:.5f}"
    )


# ---------------------------------------------------------------------------
# power_lcb_from_studies
# ---------------------------------------------------------------------------


def test_power_lcb_monotone_in_clears() -> None:
    """More clearing studies -> higher power LCB; matches the analytic Beta LCB exactly."""
    low = power_lcb_from_studies(200, 300)
    high = power_lcb_from_studies(290, 300)
    assert high > low
    assert low == beta.lcb(201, 101, quantile=0.05)
