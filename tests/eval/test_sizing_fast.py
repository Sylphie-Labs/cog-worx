"""Equivalence + mutation suite for the numpy fast kernel (Pod 4.4c-0).

These pin that :func:`cogworx.eval._sizing_fast.nested_bootstrap_delta_fast` is the SAME estimator
as the stdlib :func:`cogworx.eval.youden.nested_bootstrap_delta`, and -- the key result -- that the
:mod:`cogworx.eval.equiv_check` harness is POWERED to FAIL on a different estimator: four
deliberately-broken kernels (M1, M2, M4, M5) are each caught by the gate the spec assigns them, with
a clean negative control proving the gates don't fire on the true kernel.

Per the eval-stats designated-gate contract (each mutant dies on its NAMED gate; the proof never
rests on an incidental catch that scale could erase). TWO mutations were RETRACTED from the
kill-set: the cross-n M3 leg (eval-stats corrected-scope: a phantom -- the fast kernel is
scorer-only with no cross-n bootstrap CRN to preserve, so what that leg measured was ``synth_cells``
study-difficulty, shared stdlib in both paths) and M6 (the boundary-localized lo shift --
ANTI-RESOLVED: its trigger band tau = 3 * local-gap -> 0 as n_outer densifies the order statistics,
so it is unkillable by ``evaluate_boundary`` at every scale and adds no coverage beyond M4's
unconditional endpoint-index pin; Jim-approved, see the §5 retraction note). Both retractions keep
the powered kill-set:

* M1 drop inner trial-resample -> gate B (run-spread): the inner nest is gone, fast SD collapses.
* M2 break CRN across arms      -> inner-share draw pin: the within-item CRN is ONE inner pick array
  drawn per iter and shared across arms; M2 draws it per-arm (the RNG draw-schedule spy catches it).
* M4 off-by-one endpoint index  -> deterministic endpoint-index pin (paired same-RNG: MUST differ).
* M5 clean resampled per-arm    -> urn-shape pin (RNG draw-schedule spy: clean drawn ONCE per iter).

These run at a small ``n_outer``/``S`` smoke so they stay fast and pytest-safe; the full
``n_outer=50_000 x S=40`` validation is the standalone ``python -m cogworx.eval.equiv_check`` script
(NEVER under the spike tier -- test-hang discipline). No model calls anywhere (S1/S6).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from random import Random

import pytest

# numpy ships in the optional `sizing` extra, which the `test` nox session deliberately does not
# install (CANON S2: numpy is the adopter's burden, and `import cogworx.eval` must work without
# it). Without this guard the missing import is a collection ERROR, aborting the whole pytest run
# rather than skipping this module.
#
# This module therefore does NOT run under `nox -s test`. It runs under `nox -s sizing`, which
# installs the extra, asserts numpy is importable so this guard cannot silently skip everything,
# and type-checks the numpy-present environment too.
pytest.importorskip("numpy", reason="requires the optional `sizing` extra")

import numpy as np
from numpy.random import PCG64, SeedSequence
from numpy.random import Generator as NPGenerator

import cogworx.eval._sizing_fast as fast_mod
from cogworx.eval import equiv_check
from cogworx.eval._sizing_fast import (
    ColumnarArtifact,
    _gather_counts,
    cells_to_columnar,
    columnar_to_cells,
    nested_bootstrap_delta_fast,
)
from cogworx.eval.equiv_check import (
    DEFAULT_CONFIGS,
    EquivConfig,
    _seed_pairs,
    evaluate_boundary,
    evaluate_config,
)
from cogworx.eval.youden import Cell, synth_cells

_R = 7
_PESSIMISTIC = EquivConfig("pessimistic-corner", 0.15, 0.15, 0.045, 0.028, 80)

# Smoke scale: small enough for pytest, large enough that the designated gates have power.
_SMOKE_N_OUTER = 4000
_SMOKE_S = 12

# The run-spread gate is a STATISTICAL log-SD-ratio test scaling as 1/sqrt(S-1) (equiv_check
# redefinition), so its power is coupled to S. M1 (drop-inner) deflates the SD by ~1.55x; at
# k_spread=2.5 the band is +-2.5/sqrt(S-1), and a 1.55x effect only clears it at the VALIDATION S,
# not at S=12 (where the band is +-0.755 and M1's worst is ~1.4 sigma). So M1's designated-gate test
# and its negative control run at S=40 -- the S where the gate has power -- keeping n_outer at the
# pytest-feasible smoke scale (the log-ratio is n_outer-invariant in expectation). Structural-pin
# mutants (M2/M4/M5) are S-invariant draw-schedule/index assertions and stay at smoke S.
_SPREAD_S = 40


# ---------------------------------------------------------------------------
# §1 Cell <-> columnar lossless round-trip (the shared-schema pin)
# ---------------------------------------------------------------------------


def _make_cells(seed: int, cfg: EquivConfig = _PESSIMISTIC) -> list[Cell]:
    return synth_cells(
        Random(seed),
        cfg.n,
        cfg.n,
        _R,
        sens_C=0.60,
        dsens=cfg.dsens,
        sb_sens=cfg.sb_sens,
        spec_C=0.85,
        dspec=cfg.dspec,
        sb_spec=cfg.sb_spec,
        rho_w=0.30,
    )


def test_columnar_round_trip_is_lossless() -> None:
    """arrays -> back to Cell rows == original (as a set of (item,arm,trial)->fields records).

    The columnar view reorders rows (it emits per-item arm_a/arm_b pairs); the stdlib indexer is
    order-insensitive, so set-equality on the keyed records is the meaningful losslessness. Pins the
    fast kernel reads the EXACT frozen Cell schema (shared-schema invariant)."""
    cells = _make_cells(0)
    # Stamp a per-item regime so the round-trip is exercised on a NON-default value (synth cells all
    # carry regime="" -- without this the pin would not prove regime survives cells->arrays->cells).
    _REGIMES = ("logic-wrong", "edge-case-miss", "spec-misread", "silent-degradation", "")
    cells = [c.model_copy(update={"regime": _REGIMES[c.item_id % len(_REGIMES)]}) for c in cells]
    col = cells_to_columnar(cells, arm_a="D", arm_b="C")
    back = columnar_to_cells(col)

    def keyed(cs: Sequence[Cell]) -> dict[tuple[int, str, int], tuple[int, int, str, str, str]]:
        return {
            (c.item_id, c.arm, c.trial): (c.flagged, c.seed, c.route, c.stratum, c.regime)
            for c in cs
        }

    assert keyed(back) == keyed(cells)
    assert len(back) == len(cells)


def test_columnar_rejects_inconsistent_artifact() -> None:
    """A boundary validation: the columnar adapter rejects an item missing an arm (external input
    to the fast path; we validate at the seam, not inside the loop)."""
    cells = [c for c in _make_cells(0) if not (c.item_id == 0 and c.arm == "C")]
    with pytest.raises(ValueError, match="arms"):
        cells_to_columnar(cells, arm_a="D", arm_b="C")


# ---------------------------------------------------------------------------
# §2 Structural pins run THROUGH the fast kernel
# ---------------------------------------------------------------------------


def _yes_machine_cells(m_K: int, m_clean: int) -> list[Cell]:
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


def test_fast_yes_machine_mean_is_exactly_zero() -> None:
    """Flag-everything -> sens=1, spec=0 on both arms -> J=0 each -> delta exactly 0 every draw."""
    mean, lo, hi = nested_bootstrap_delta_fast(
        _yes_machine_cells(30, 30),
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=300,
        seed=5,
    )
    assert mean == 0.0
    assert lo == 0.0 == hi


def _coin_judge_cells(m_K: int, m_clean: int) -> list[Cell]:
    cells: list[Cell] = []
    for stratum, base, m in (("K", 0, m_K), ("clean", 10_000, m_clean)):
        for i in range(m):
            for trial in range(_R):
                seed = (base + i) << 8 ^ trial
                flagged = 1 if Random(seed).random() < 0.5 else 0
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


def test_fast_coin_judge_delta_ci_contains_zero() -> None:
    """THE inner-resample pin through the fast kernel: a coin judge (identical across arms) must
    yield a delta CI straddling 0. If the inner trial-resample were optimized away the variance
    would still be 0 here (identical arms) -- but the wider M1 catch lives in the equivalence gate;
    this pins the fast kernel reproduces the stdlib's straddle."""
    _, lo, hi = nested_bootstrap_delta_fast(
        _coin_judge_cells(80, 80),
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=500,
        seed=11,
    )
    assert lo <= 0.0 <= hi, f"fast coin-judge CI [{lo:.4f}, {hi:.4f}] should contain 0"


def _flake_cells(catch_K: int, catch_total: int, m_K: int, m_clean: int) -> list[Cell]:
    cells: list[Cell] = []
    for i in range(m_K):
        for trial in range(catch_total):
            d_flag = 1 if trial < catch_K else 0
            cells.append(
                Cell(
                    item_id=i,
                    stratum="K",
                    arm="D",
                    trial=trial,
                    seed=i << 8 ^ trial,
                    flagged=d_flag,
                    route="flag" if d_flag else "pass",
                )
            )
            cells.append(
                Cell(
                    item_id=i,
                    stratum="K",
                    arm="C",
                    trial=trial,
                    seed=i << 8 ^ trial,
                    flagged=1,
                    route="flag",
                )
            )
    for i in range(m_clean):
        for trial in range(catch_total):
            for arm in ("C", "D"):
                cells.append(
                    Cell(
                        item_id=10_000 + i,
                        stratum="clean",
                        arm=arm,
                        trial=trial,
                        seed=(10_000 + i) << 8 ^ trial,
                        flagged=0,
                        route="pass",
                    )
                )
    return cells


def test_fast_flake_delta_ci_does_not_clear_planning_effect() -> None:
    """A 3/7-vs-7/7 flake -> the CI lower bound does not clear the 0.15 planning effect (fast)."""
    _, lo, _ = nested_bootstrap_delta_fast(
        _flake_cells(3, 7, 80, 80),
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=500,
        seed=13,
    )
    assert lo < 0.15, f"fast flake CI lo={lo:.4f} unexpectedly cleared 0.15"


def test_fast_quantile_monotonicity_R5() -> None:
    """R5: lo(q=0.0025) <= lo(q=0.025) and hi(q=0.0025) >= hi(q=0.025) on one fixed artifact."""
    cells = _make_cells(1)
    _, lo_n, hi_n = nested_bootstrap_delta_fast(
        cells,
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=4000,
        seed=7,
        quantile=0.025,
    )
    _, lo_c, hi_c = nested_bootstrap_delta_fast(
        cells,
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=4000,
        seed=7,
        quantile=0.0025,
    )
    assert lo_c <= lo_n
    assert hi_c >= hi_n


def _spy_integers_sizes(
    monkeypatch: pytest.MonkeyPatch,
    cells: list[Cell],
    *,
    n_outer: int,
    seed: int,
    kernel: equiv_check.BootstrapFn = nested_bootstrap_delta_fast,
) -> list[object]:
    """Record every numpy ``integers`` draw size while ``kernel`` runs, so a test can pin the urn
    shapes (outer item urn, inner trial urn, clean-drawn-once). Monkeypatches the ``Generator``
    factory the fast kernel constructs (the stdlib ``_spy_randrange_moduli`` analogue). The kernel
    is injectable so a mutant can be spied identically."""
    sizes: list[object] = []

    # Subclassing is real when numpy is installed; without it `NPGenerator` is `Any` and
    # --strict refuses to subclass Any. `unused-ignore` keeps both environments clean.
    class _SpyGen(NPGenerator):  # type: ignore[misc, unused-ignore]
        def integers(self, low, high=None, size=None, *a, **k):  # type: ignore[no-untyped-def]
            sizes.append(tuple(size) if isinstance(size, tuple) else size)
            return super().integers(low, high, size, *a, **k)

    monkeypatch.setattr(fast_mod, "Generator", lambda bitgen: _SpyGen(bitgen))
    kernel(cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=n_outer, seed=seed)
    return sizes


def test_fast_urn_shapes_clean_drawn_once(monkeypatch: pytest.MonkeyPatch) -> None:
    """Urn-shape pin (ported to the fast kernel): per outer iter the inner urn is (m, n_items, R),
    the error urn (m, n_K), and the clean urn (m, n_clean) drawn EXACTLY once (not per arm)."""
    cells = _make_cells(2, EquivConfig("c", 0.15, 0.15, 0.045, 0.028, 20))
    sizes = _spy_integers_sizes(monkeypatch, cells, n_outer=1, seed=3)
    n_clean_draws = sum(1 for s in sizes if s == (1, 20))
    inner_draws = [s for s in sizes if isinstance(s, tuple) and len(s) == 3]
    assert inner_draws == [(1, 40, 7)], f"inner urn shape drifted: {inner_draws}"
    # Two (1,20) draws: ONE error (n_K=20) + ONE clean (n_clean=20). Clean drawn per-arm -> 3.
    assert n_clean_draws == 2, f"expected error+clean once each (2 draws), got {n_clean_draws}"


# ---------------------------------------------------------------------------
# §3 Small-n_outer equivalence smoke (the true kernel passes its own harness)
# ---------------------------------------------------------------------------


def test_equivalence_smoke_true_kernel_passes_functionals() -> None:
    """At the smoke scale the true kernel passes the gate-A equivalence on all three functionals on
    every config (the full spread/cross-n gates need the heavy run; here we pin gate A)."""
    seed_pairs = _seed_pairs(equiv_check.EQUIV_MASTER_SEED, _SMOKE_S)
    artifact_seeds = Random(equiv_check.EQUIV_MASTER_SEED ^ 0xA17FAC)
    for cfg in DEFAULT_CONFIGS:
        result = evaluate_config(
            cfg,
            n_outer=_SMOKE_N_OUTER,
            s_pairs=_SMOKE_S,
            artifact_seed=artifact_seeds.randrange(2**31),
            seed_pairs=seed_pairs,
        )
        for f in result.functionals:
            assert f.passed, (
                f"{cfg.name}/{f.functional}: |Dbar|={abs(f.dbar):.6f} > bound={f.bound:.6f} "
                f"-- true kernel failed its own equivalence gate"
            )


def test_analytic_mean_se_agrees_with_method_c() -> None:
    """Cross-check: the analytic mean SE (sqrt(Var(delta)/n_outer)) agrees with the method-C per-run
    SD within ~20% on the planting-effect config (a sampler-correctness anchor)."""
    seed_pairs = _seed_pairs(equiv_check.EQUIV_MASTER_SEED, _SMOKE_S)
    result = evaluate_config(
        DEFAULT_CONFIGS[0],
        n_outer=_SMOKE_N_OUTER,
        s_pairs=_SMOKE_S,
        artifact_seed=12345,
        seed_pairs=seed_pairs,
    )
    mean_fn = next(f for f in result.functionals if f.functional == "mean")
    assert mean_fn.analytic_mean_se is not None
    assert mean_fn.se_perrun > 0
    rel = abs(mean_fn.analytic_mean_se - mean_fn.se_perrun) / mean_fn.se_perrun
    assert rel < 0.5, f"analytic {mean_fn.analytic_mean_se:.5f} vs methodC {mean_fn.se_perrun:.5f}"


# ---------------------------------------------------------------------------
# §4 Mutation suite -- the proof of power (each mutant dies on its NAMED gate)
# ---------------------------------------------------------------------------


def _mutant_kernel(
    *,
    drop_inner: bool = False,
    break_crn_arms: bool = False,
    off_by_one: bool = False,
    clean_per_arm: bool = False,
    m6_boundary: bool = False,
) -> equiv_check.BootstrapFn:
    """A deliberately-broken fast kernel selecting ONE mutation, CHUNK-FAITHFUL to the real kernel.

    It mirrors :func:`nested_bootstrap_delta_fast`'s exact vectorized draw schedule (chunk size,
    same (m, n_items, R) / (m, gsize) / (m, n_clean) draw order) so that an unbroken mutation draws
    the IDENTICAL PCG64 stream as the real kernel on the same seed. This is what makes M4's
    deterministic endpoint-index pin valid: the off-by-one mutant shares the real kernel's deltas
    byte-for-byte and differs ONLY in the final index. The other defects perturb the math at the
    point they're injected (M1/M2: inner picks, M5: clean draw).

    M6 (``m6_boundary``) is a BOUNDARY-LOCALIZED off-by-one: it shifts the lo index +1 ONLY when the
    lo order statistic ``abs(deltas[idx]) < tau``, with ``tau`` a few times the local inter-order-
    statistic gap near the lo endpoint -- so it perturbs only near-boundary studies (lo ~= 0) and is
    invisible far from the boundary. Unlike M4 (unconditional) it hides from a far-from-boundary
    artifact and only surfaces where a verdict flip is plausible."""

    def kern(
        cells: Sequence[Cell] | ColumnarArtifact,
        *,
        arm_a: str,
        arm_b: str,
        error_strata: Sequence[str],
        n_outer: int,
        seed: int,
        quantile: float = 0.025,
    ) -> tuple[float, float, float]:
        col = (
            cells
            if isinstance(cells, ColumnarArtifact)
            else cells_to_columnar(cells, arm_a=arm_a, arm_b=arm_b)
        )
        # Construct via the module's Generator attr so the draw-schedule spy can patch it; PCG64 /
        # SeedSequence come straight from numpy.random (the mutant pins the same bit generator).
        rng = fast_mod.Generator(PCG64(SeedSequence(seed)))  # type: ignore[attr-defined]
        clean_rows = col.stratum_rows.get("clean", np.empty(0, dtype=np.int64))
        egroups = [col.stratum_rows.get(s, np.empty(0, dtype=np.int64)) for s in error_strata]
        esizes = [int(g.size) for g in egroups]
        r = col.r
        n_items = int(col.item_ids.size)
        n_clean = int(clean_rows.size)
        fa, fb = col.flags_a, col.flags_b
        chunk = max(1, min(n_outer, fast_mod._MAX_CHUNK_CELLS // max(1, n_items * r)))
        deltas = np.empty(n_outer, dtype=np.float64)
        done = 0
        while done < n_outer:
            m = min(chunk, n_outer - done)
            if drop_inner:
                fc_a = np.broadcast_to(fa.sum(axis=1), (m, n_items)).astype(np.float64)
                fc_b = np.broadcast_to(fb.sum(axis=1), (m, n_items)).astype(np.float64)
            elif break_crn_arms:
                pa = rng.integers(0, r, size=(m, n_items, r))
                pb = rng.integers(0, r, size=(m, n_items, r))
                fc_a = _gather_counts(fa, pa)
                fc_b = _gather_counts(fb, pb)
            else:
                picks = rng.integers(0, r, size=(m, n_items, r))
                fc_a = _gather_counts(fa, picks)
                fc_b = _gather_counts(fb, picks)
            sna = np.zeros(m)
            snb = np.zeros(m)
            sden = 0
            for g, gs in zip(egroups, esizes, strict=True):
                if gs == 0:
                    continue
                rows = g[rng.integers(0, gs, size=(m, gs))]
                sna += np.take_along_axis(fc_a, rows, axis=1).sum(axis=1)
                snb += np.take_along_axis(fc_b, rows, axis=1).sum(axis=1)
                sden += gs * r
            if n_clean:
                if clean_per_arm:
                    ca = clean_rows[rng.integers(0, n_clean, size=(m, n_clean))]
                    cb = clean_rows[rng.integers(0, n_clean, size=(m, n_clean))]
                    spa = (r - np.take_along_axis(fc_a, ca, axis=1)).sum(axis=1)
                    spb = (r - np.take_along_axis(fc_b, cb, axis=1)).sum(axis=1)
                else:
                    cc = clean_rows[rng.integers(0, n_clean, size=(m, n_clean))]
                    spa = (r - np.take_along_axis(fc_a, cc, axis=1)).sum(axis=1)
                    spb = (r - np.take_along_axis(fc_b, cc, axis=1)).sum(axis=1)
                spden = n_clean * r
            else:
                spa = np.zeros(m)
                spb = np.zeros(m)
                spden = 0
            z = np.zeros(m)
            ja = (sna / sden if sden else z) + (spa / spden if spden else z) - 1.0
            jb = (snb / sden if sden else z) + (spb / spden if spden else z) - 1.0
            deltas[done : done + m] = ja - jb
            done += m
        deltas.sort()
        off = 1 if off_by_one else 0
        lo_idx = int(quantile * n_outer)
        if m6_boundary and lo_idx + 1 < n_outer:
            # Local inter-order-statistic gap near the lo endpoint (median spacing over a small
            # window, robust to a single tied pair); tau = a few * that gap. Shift +1 ONLY when the
            # lo order statistic is itself within tau of zero -- a near-boundary study.
            window = deltas[lo_idx : min(lo_idx + 11, n_outer)]
            gaps = np.diff(window)
            local_gap = float(np.median(gaps)) if gaps.size else 0.0
            tau_m6 = 3.0 * local_gap
            if abs(float(deltas[lo_idx])) < tau_m6:
                off = 1
        lo = float(deltas[lo_idx + off])
        hi = float(
            deltas[min(int((1.0 - quantile) * n_outer) + (1 if off_by_one else 0), n_outer - 1)]
        )
        return float(deltas.mean()), lo, hi

    return kern


def _config_against_mutant(
    mutant: equiv_check.BootstrapFn,
    *,
    s_pairs: int = _SMOKE_S,
    cfg: EquivConfig = _PESSIMISTIC,
    artifact_seed: int = 99,
) -> equiv_check.ConfigResult:
    seed_pairs = _seed_pairs(equiv_check.EQUIV_MASTER_SEED, s_pairs)
    return evaluate_config(
        cfg,
        n_outer=_SMOKE_N_OUTER,
        s_pairs=s_pairs,
        artifact_seed=artifact_seed,
        seed_pairs=seed_pairs,
        fast_kernel=mutant,
    )


def test_M1_drop_inner_resample_fails_run_spread() -> None:
    """M1 -> gate B (run-spread): dropping the inner trial-resample collapses within-item flip
    variance, so the fast SD deflates ~1.55x vs the stdlib reference on the variance-bearing
    functionals. The redefined gate is the statistical log-SD-ratio test |log(SD_fast/SD_std)| <=
    k_spread/sqrt(S-1) (k_spread=2.5); a 1.55x effect only clears it at the VALIDATION S=40 (band
    +-0.40 in log-space), so this designated-gate test runs at S=40 -- where the gate has power --
    keeping n_outer at smoke scale. Measured: M1 worst sigma=3.20 at this exact (n_outer=4000, S=40,
    pinned seed); the bare boolean is backed by a pinned-sigma floor so MC/seed drift is visible."""
    result = _config_against_mutant(_mutant_kernel(drop_inner=True), s_pairs=_SPREAD_S)
    se_logratio = 1.0 / math.sqrt(_SPREAD_S - 1)
    worst_sigma = max(
        abs(math.log(f.sd_fast / f.sd_stdlib)) / se_logratio
        for f in result.functionals
        if f.sd_stdlib > 0 and f.sd_fast > 0
    )
    assert not result.passed, "M1 (drop inner resample) was NOT caught"
    assert any(not f.spread_ok for f in result.functionals), "M1 should trip the run-spread gate"
    assert worst_sigma > 2.5, (
        f"M1 run-spread sigma={worst_sigma:.2f} at S={_SPREAD_S} (measured 3.20); margin eroding"
    )


def test_M2_break_crn_arms_fails_inner_share_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """M2 -> inner-pick-shared draw-schedule pin: the within-item CRN IS the single inner
    trial-resample drawn ONCE per outer iteration and applied to BOTH arms. Drawing the inner picks
    independently per arm breaks the CRN. The RNG draw-schedule spy catches the duplicate inner draw
    (true: one (1,n_items,R) inner array/iter; M2: two).

    FINDING: M2's OUTPUT-distribution signature (lo/hi SD inflation) is WEAK at any feasible scale
    because the inner within-item CRN contributes only a minority of the paired-delta variance --
    the between-item resample (CRN-preserved by synth_cells' shared base rate, rho_b~0.5) dominates,
    so rho_arm and the lo-tail SD barely move when the inner CRN breaks. The structural draw-
    schedule pin is therefore M2's designated discriminator, not gate B. (Reported to architect.)
    """
    cells = _make_cells(2, EquivConfig("c", 0.15, 0.15, 0.045, 0.028, 20))
    sizes = _spy_integers_sizes(
        monkeypatch, cells, n_outer=1, seed=3, kernel=_mutant_kernel(break_crn_arms=True)
    )

    # n_items = 20 K + 20 clean = 40; the inner pick array is (1, n_items, R) per draw.
    inner_draws = [s for s in sizes if isinstance(s, tuple) and len(s) == 3]
    assert inner_draws == [(1, 40, _R), (1, 40, _R)], (
        f"M2 should draw the inner picks per-arm (two (1,40,7) draws), got {inner_draws}"
    )


def test_M4_off_by_one_fails_endpoint_index_pin() -> None:
    """M4 -> deterministic endpoint-index pin: the true kernel and the off-by-one kernel share the
    SAME RNG draws (same seed), so their delta arrays are IDENTICAL -- only the endpoint index
    differs. lo_M4 = sorted[idx+1] >= lo_true = sorted[idx], strictly greater whenever the adjacent
    order statistics differ. A zero-variance discriminator (no MC scaling needed)."""
    cells = _make_cells(99)
    mutant = _mutant_kernel(off_by_one=True)
    differed = False
    for seed in range(6):
        true = nested_bootstrap_delta_fast(
            cells,
            arm_a="D",
            arm_b="C",
            error_strata=("K",),
            n_outer=4000,
            seed=seed,
            quantile=0.0025,
        )
        mut = mutant(
            cells,
            arm_a="D",
            arm_b="C",
            error_strata=("K",),
            n_outer=4000,
            seed=seed,
            quantile=0.0025,
        )
        assert mut[1] >= true[1], "off-by-one lo must shift to the next-higher order statistic"
        assert mut[2] >= true[2]
        if (mut[1], mut[2]) != (true[1], true[2]):
            differed = True
    assert differed, "M4 (off-by-one index) produced no endpoint shift -- pin has no power"


def test_M5_clean_per_arm_fails_urn_shape_pin(monkeypatch: pytest.MonkeyPatch) -> None:
    """M5 -> urn-shape pin: resampling the clean stratum per-arm draws clean indices TWICE per outer
    iteration (2*n_clean) instead of once, injecting artificial independence into spec. The RNG
    draw-schedule spy catches the extra clean draw."""
    cells = _make_cells(2, EquivConfig("c", 0.15, 0.15, 0.045, 0.028, 20))
    sizes = _spy_integers_sizes(
        monkeypatch, cells, n_outer=1, seed=3, kernel=_mutant_kernel(clean_per_arm=True)
    )

    # Chunk-faithful mutant at n_outer=1 -> chunk m=1, outer urns drawn as (1, 20) tuples.
    size20_draws = sum(1 for s in sizes if s == (1, 20))
    # M5: error(1,20) + clean-a(1,20) + clean-b(1,20) = 3 (1,20) draws (clean drawn per-arm).
    assert size20_draws == 3, f"M5 should draw clean per-arm (3 (1,20) draws), got {size20_draws}"


def test_M5_clean_negative_control_passes_all_gates(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: the TRUE kernel passes the urn-shape pin and gate-A equivalence -- gates
    that catch M1/M2/M4/M5 do not fire on the correct kernel (no incidental false positive). Runs at
    S=40 (the run-spread gate's power scale) so the gate is proven SILENT on the true kernel at the
    same S where it catches M1 -- measured true-kernel worst run-spread sigma=1.18, well inside
    +-2.5."""
    result = _config_against_mutant(nested_bootstrap_delta_fast, s_pairs=_SPREAD_S)
    for f in result.functionals:
        assert f.passed, f"true kernel failed gate-A {f.functional}"
        assert f.spread_ok, f"true kernel tripped the run-spread gate on {f.functional}"
    # urn-shape control: true kernel draws clean exactly once (2 size-20 draws: error + clean) and
    # the inner picks exactly once per iter (one (1,20,7) array, shared across arms = the CRN).
    cells = _make_cells(2, EquivConfig("c", 0.15, 0.15, 0.045, 0.028, 20))
    sizes = _spy_integers_sizes(monkeypatch, cells, n_outer=1, seed=3)
    assert sum(1 for s in sizes if s == (1, 20)) == 2
    assert [s for s in sizes if isinstance(s, tuple) and len(s) == 3] == [(1, 40, _R)]


# ---------------------------------------------------------------------------
# §5 M6 boundary-localized lo-shift mutation -- RETRACTED from the kill-set (Jim-approved scope
#    call, mirroring the cross-n M3 retraction above). M6 shifts the lo index +1 ONLY when the lo
#    order statistic is within tau = 3 * (local inter-order-statistic gap) of zero (eval-stats' "a
#    few * the gap" spec). That makes it ANTI-RESOLVED against evaluate_boundary: as n_outer grows
#    the bootstrap delta order statistics densify, so the local gap -> 0, so tau -> 0, so M6 fires
#    on fewer studies AND its +1 shift (~tau ~ 1e-4) never crosses the lo>0 verdict. Probe (isolated
#    M6, knee dsens=dspec=0.068, n=56, s_cross=60): n_outer=4000 -> fired 3/60, 0 zero-crossings,
#    mean tau=0.00179; n_outer=12000 -> fired 1/60, 0 zero-crossings, mean tau=0.00000. So M6 gets
#    STRICTLY HARDER to catch as the gate scales up; at validation scale it produces 0 verdict flips
#    / 0 clears drift -- a sub-gap lo wobble below the verdict's resolution, indistinguishable from
#    legitimate stdlib-vs-PCG64 substrate spacing.
#
#    Coverage is intact: M6 was the boundary-localized special case of M4 (the UNCONDITIONAL
#    endpoint-index off-by-one, killed with zero-variance power by test_M4_off_by_one...). The
#    endpoint-index family is covered by M4; M6 added no coverage the gate can exercise. The powered
#    kill-set -- M1 (run-spread), M2/M5 (RNG draw-schedule pins), M4 (endpoint-index) -- is all
#    green with a clean negative control. The _mutant_kernel(m6_boundary=True) machinery is RETAINED
#    as the documented anti-resolved mutant so the retraction probe re-runs. (CF-4.4-M6.)
# ---------------------------------------------------------------------------


_M6_BOUNDARY_N_OUTER = 1500
_M6_BOUNDARY_S_CROSS = 80
_M6_KNEE = EquivConfig("m6-knee", 0.068, 0.068, 0.045, 0.028, 56)


def test_M6_clean_negative_control_passes_boundary() -> None:
    """M6-clean negative control (LIVE pin): the TRUE kernel passes evaluate_boundary at the knee --
    the two legitimate arms (stdlib vs fast) do not false-trip the boundary gate, which is exactly
    what protects the gate from the substrate clears noise it must tolerate. (M6's own catch test is
    retracted -- see the §5 header: M6 is anti-resolved and adds no coverage beyond M4.)"""
    boundary = evaluate_boundary(
        s_cross=_M6_BOUNDARY_S_CROSS,
        n_outer=_M6_BOUNDARY_N_OUTER,
        cfg=_M6_KNEE,
    )
    assert boundary.passed, (
        f"true kernel tripped the boundary gate: n_boundary={boundary.n_boundary} "
        f"n_flip={boundary.n_flip} clears_std={boundary.clears_stdlib} "
        f"clears_fast={boundary.clears_fast}"
    )
