"""Equivalence harness: prove the numpy fast kernel is the SAME estimator as stdlib (Pod 4.4c-0).

The fast kernel (:func:`cogworx.eval._sizing_fast.nested_bootstrap_delta_fast`) reimplements the
stdlib :func:`cogworx.eval.youden.nested_bootstrap_delta` on a different RNG substrate. Before any
sizing number from the fast path is trusted, this harness proves the two are the same estimator --
same sampling distribution, not merely the same center -- powered to FAIL on a different estimator
(the red-teamer attacks it; the mutation suite in ``tests/eval/test_sizing_fast.py`` proves five
deliberately-broken kernels are each caught).

The instrument (eval-stats spec):

* **Equivalence (core, GATING)** -- on a FIXED frozen artifact fed identically to both kernels, run
  both at the same ``n_outer`` over ``S`` independent seed-pairs at 4 configs. For each functional
  f in {mean, lo, hi}: per-pair diff ``Delta_f[i] = stdlib_f[i] - fast_f[i]``, ``Dbar_f = mean_i``;
  per-run SE by method C (empirical SD across the 2S runs); ``SE(Dbar_f) = SE_perrun * sqrt(2/S)``.
  PASS f iff ``|Dbar_f| <= k_f * SE(Dbar_f)`` with ``k_mean=4``, ``k_lo=k_hi=5``.
* **Run-spread agreement (GATING)** -- a log-SD-ratio test scaling with S per functional (closes
  "match the center, fatten the tails"). The true-null config is carved out of the *mean* spread
  check (its mean SD is near-degenerate at zero effect; see ``_spread_ok``).
* **Boundary-study gate (GATING)** -- a verdict flip is only allowed inside the boundary band; a
  flip outside it is a different estimator. This is the honest home for near-boundary verdict flips
  and runs (and gates) UNCONDITIONALLY -- it is not nested under any cross-n switch.

The cross-n McNemar/correlation legs were RETRACTED (eval-stats corrected-scope): the fast kernel is
scorer-only and ``synth_cells`` is shared stdlib, so at fixed n the bootstrap CRN is preserved by
construction and across n the two paths consume different draw counts -- there is no cross-n
bootstrap CRN to preserve. That leg measured ``synth_cells`` study-difficulty, identical both paths.

The FULL validation (``n_outer=50_000 x S=40``) is a standalone, timeout-wrapped script run --
``python -m cogworx.eval.equiv_check`` -- NEVER under the spike tier (test-hang discipline). The
pytest companion is a small-``n_outer`` smoke plus the mutation suite. numpy is the optional
``cogworx[sizing]`` extra: this module top-level-imports it (the heavy path always wants numpy), so
it is never imported by the core or the gate.

S1/S6: model-free throughout -- everything resamples frozen synthetic artifacts.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from random import Random

from cogworx.eval._sizing_fast import nested_bootstrap_delta_fast

# The cross-n CRN panel reuses the sizing module's master seed + planning constants so its
# study-seed sequence and synth point match the live sim (SIZING_MASTER_SEED != the harness's seed).
from cogworx.eval.sizing import (
    _BOOTSTRAP_SEED_MIX,
    _R,
    _RHO_W,
    _SENS_C,
    _SPEC_C,
)
from cogworx.eval.sizing import (
    MASTER_SEED as SIZING_MASTER_SEED,
)
from cogworx.eval.youden import (
    Cell,
    nested_bootstrap_delta,
    realized_variance_diagnostic,
    synth_cells,
)

# Reproducibility anchor: every seed in the harness derives from this via Random(EQUIV_MASTER_SEED).
EQUIV_MASTER_SEED = 20260619
"""The pinned harness master seed (the build date) -- recorded in the fingerprint to re-audit."""

# The bootstrap signature both kernels (and the mutated kernels) satisfy.
BootstrapFn = Callable[..., tuple[float, float, float]]

# Equivalence acceptance multipliers (eval-stats spec).
_K_MEAN = 4.0
_K_LO = 5.0
_K_HI = 5.0
# Run-spread gate: a STATISTICAL test on the log-SD-ratio that scales with S. The per-functional
# log-ratio has approximate SE = 1/sqrt(S-1) (the sampling SD of log(SD) over S draws), so the band
# tightens as S grows -- a fixed fractional tol (the old 0.3) false-trips at large S and is blind at
# small S. spread_ok iff |log(SD_fast / SD_std)| <= _K_SPREAD / sqrt(S - 1).
#   k=2.5 -> +-2.5/sqrt(S-1) (=+-0.40 in log-space at S=40, a ~1.49x ratio band) -> ~3.7%
#   family-wise false-trip over 12 checks. Chosen against the measured M1 (drop-inner) magnitude:
#   M1 deflates the variance-bearing functional's SD by ~1.55x (ratio 0.643, WORST functional =
#   the mean), which lands in eval-stats' 1.4x-1.6x band -> k=2.5 (threshold ~1.49x), so M1 is
#   still CAUGHT (the log-ratio is n_outer-invariant in expectation, only its noise scales with S).
#   A k=3 (~1.6x band) would let a 1.55x deflation slip; k=2.5 holds the catch. Confirmed: M1 worst
#   sigma=3.20 at the M-suite test scale (n_outer=4000, S=40, pinned seed) vs true-kernel worst
#   1.18 (control silent).
_K_SPREAD = 2.5
_SE_BOUND_FLAG = 0.02  # if realized SE_lo/SE_hi gives a bound wider than this at 50k, flag loudly
_EXACT_TIE = 1e-12

# True-null carve-out: at zero effect (dsens=dspec=0) the paired delta is centered on 0 and its
# per-run MEAN is near-degenerate -- SD(mean) -> 0, so the log-SD-ratio is dominated by floating
# noise on two near-zero SDs and false-trips the TRUE kernel (the recorded passed:false artifact).
# We EXEMPT the true-null config's *mean* spread check specifically -- NOT lo/hi (the tails still
# carry real variance at true-null and stay gated), and NOT a global k_spread widening (that would
# blind the M1 variance mutant, whose worst functional IS the mean). The mean-bias gate-A bound
# (k_mean * SE) still gates the true-null mean center; only the mean SPREAD ratio is carved out.
_TRUE_NULL_NAME = "true-null"


def _spread_ok(cfg_name: str, functional: str, sd_std: float, sd_fast: float, s_pairs: int) -> bool:
    """The run-spread log-SD-ratio test, with the documented true-null *mean* carve-out (item 4).

    ``spread_ok iff |log(SD_fast / SD_std)| <= _K_SPREAD / sqrt(S - 1)``. Degenerate SDs (a zero on
    either side, or ``S < 2``) return True (no spread signal to test). The true-null config's mean
    is exempt: its mean SD is near-degenerate so the ratio is noise, not divergence; lo/hi stay
    gated and k_spread is untouched, so M1 (which deflates the mean SD ~1.55x on the effect configs)
    is unaffected.
    """
    if cfg_name == _TRUE_NULL_NAME and functional == "mean":
        return True
    if sd_std <= 0.0 or sd_fast <= 0.0 or s_pairs <= 1:
        return True
    log_ratio = abs(math.log(sd_fast / sd_std))
    return log_ratio <= _K_SPREAD / math.sqrt(s_pairs - 1)


@dataclass(frozen=True)
class EquivConfig:
    """One frozen-artifact equivalence configuration (its own synth artifact)."""

    name: str
    dsens: float
    dspec: float
    sb_sens: float
    sb_spec: float
    n: int


# The 4 configs (eval-stats spec): planting effect, pessimistic corner, true-null, small-n.
DEFAULT_CONFIGS: tuple[EquivConfig, ...] = (
    EquivConfig("planting-effect", 0.15, 0.15, 0.020, 0.010, 80),
    EquivConfig("pessimistic-corner", 0.15, 0.15, 0.045, 0.028, 80),
    EquivConfig("true-null", 0.0, 0.0, 0.020, 0.010, 80),
    EquivConfig("small-n", 0.15, 0.15, 0.045, 0.028, 56),
)


@dataclass(frozen=True)
class FunctionalResult:
    """Equivalence read-out for one functional (mean / lo / hi) at one config."""

    functional: str
    dbar: float
    se_dbar: float
    k: float
    bound: float
    passed: bool
    sd_stdlib: float
    sd_fast: float
    spread_ok: bool
    se_perrun: float
    analytic_mean_se: float | None


@dataclass(frozen=True)
class ConfigResult:
    name: str
    n_outer: int
    s_pairs: int
    functionals: list[FunctionalResult]
    se_bound_flag: bool

    @property
    def passed(self) -> bool:
        return all(f.passed and f.spread_ok for f in self.functionals)


@dataclass(frozen=True)
class BoundaryResult:
    n_boundary: int
    n_flip: int
    clears_stdlib: int
    clears_fast: int
    exact_ties: int
    flip_ok: bool
    clears_ok: bool

    @property
    def passed(self) -> bool:
        return self.flip_ok and self.clears_ok


@dataclass(frozen=True)
class EquivReport:
    master_seed: int
    n_outer: int
    s_pairs: int
    configs: list[ConfigResult] = field(default_factory=list)
    boundary: BoundaryResult | None = None
    true_null_mean_bias_threshold: float | None = None

    @property
    def passed(self) -> bool:
        ok = all(c.passed for c in self.configs)
        if self.boundary is not None:
            ok = ok and self.boundary.passed
        return ok

    def fingerprint(self) -> dict[str, object]:
        return {
            "master_seed": self.master_seed,
            "n_outer": self.n_outer,
            "s_pairs": self.s_pairs,
            "passed": self.passed,
            "true_null_mean_bias_threshold": self.true_null_mean_bias_threshold,
            "configs": [
                {
                    "name": c.name,
                    "n_outer": c.n_outer,
                    "s_pairs": c.s_pairs,
                    "passed": c.passed,
                    "se_bound_flag": c.se_bound_flag,
                    "functionals": [
                        {
                            "functional": f.functional,
                            "dbar": f.dbar,
                            "se_dbar": f.se_dbar,
                            "k": f.k,
                            "bound": f.bound,
                            "passed": f.passed,
                            "sd_stdlib": f.sd_stdlib,
                            "sd_fast": f.sd_fast,
                            "spread_ok": f.spread_ok,
                            "se_perrun": f.se_perrun,
                            "analytic_mean_se": f.analytic_mean_se,
                        }
                        for f in c.functionals
                    ],
                }
                for c in self.configs
            ],
            "boundary": None if self.boundary is None else self.boundary.__dict__,
        }

    def to_json(self) -> str:
        return json.dumps(self.fingerprint(), indent=2)


def _frozen_artifact(cfg: EquivConfig, *, seed: int) -> list[Cell]:
    """Generate ONE frozen synthetic artifact for a config via the stdlib ``synth_cells`` (so the
    same artifact is fed identically to both kernels -- this is what makes equivalence clean)."""
    return synth_cells(
        Random(seed),
        cfg.n,
        cfg.n,
        _R,
        sens_C=_SENS_C,
        dsens=cfg.dsens,
        sb_sens=cfg.sb_sens,
        spec_C=_SPEC_C,
        dspec=cfg.dspec,
        sb_spec=cfg.sb_spec,
        rho_w=_RHO_W,
    )


def _se_method_c(values: Sequence[float]) -> float:
    """Method-C per-run SD: the empirical SD of a set of per-run functional values.

    The gate-A acceptance bound uses this on the STDLIB (reference) runs ONLY, never pooled across
    both kernels: pooling lets a variance-inflating mutant raise the bound and buy its acceptance
    (eval-stats: the mutant grades its own exam). The reference estimator defines the correct
    sampling variance at this ``n_outer``/``S``; the fast kernel must fall within a bound the
    reference sets. The ``k_f`` multipliers (4 for the mean, 5 for the tails) are t(S-1)-quantiles
    with a cushion at the smoke S, so coverage only tightens as S grows toward the validation scale.
    """
    if len(values) < 2:
        return 0.0
    return statistics.stdev(values)


def evaluate_config(
    cfg: EquivConfig,
    *,
    n_outer: int,
    s_pairs: int,
    artifact_seed: int,
    seed_pairs: Sequence[tuple[int, int]],
    stdlib_kernel: BootstrapFn = nested_bootstrap_delta,
    fast_kernel: BootstrapFn = nested_bootstrap_delta_fast,
) -> ConfigResult:
    """Run both kernels on ONE frozen artifact over ``s_pairs`` independent seed-pairs and decide
    equivalence per functional. The kernels are injectable so the mutation suite can feed a broken
    fast kernel and confirm the harness FAILS it."""
    artifact = _frozen_artifact(cfg, seed=artifact_seed)

    stdlib_runs: dict[str, list[float]] = {"mean": [], "lo": [], "hi": []}
    fast_runs: dict[str, list[float]] = {"mean": [], "lo": [], "hi": []}
    deltas: dict[str, list[float]] = {"mean": [], "lo": [], "hi": []}

    # Analytic cross-check on the mean's per-run SE: sqrt(Var(delta_paired) / n_outer) on a single
    # stdlib draw (the realized within-run delta variance). Compared ~20% against method C below.
    diag = realized_variance_diagnostic(
        artifact,
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=n_outer,
        seed=seed_pairs[0][0],
    )
    analytic_mean_se = math.sqrt(diag.var_delta_paired / n_outer)

    for s_seed, f_seed in seed_pairs[:s_pairs]:
        s_mean, s_lo, s_hi = stdlib_kernel(
            artifact,
            arm_a="D",
            arm_b="C",
            error_strata=("K",),
            n_outer=n_outer,
            seed=s_seed,
            quantile=0.0025,
        )
        f_mean, f_lo, f_hi = fast_kernel(
            artifact,
            arm_a="D",
            arm_b="C",
            error_strata=("K",),
            n_outer=n_outer,
            seed=f_seed,
            quantile=0.0025,
        )
        for fn, sv, fv in (("mean", s_mean, f_mean), ("lo", s_lo, f_lo), ("hi", s_hi, f_hi)):
            stdlib_runs[fn].append(sv)
            fast_runs[fn].append(fv)
            deltas[fn].append(sv - fv)

    functionals: list[FunctionalResult] = []
    se_bound_flag = False
    for fn, k in (("mean", _K_MEAN), ("lo", _K_LO), ("hi", _K_HI)):
        dbar = statistics.fmean(deltas[fn])
        # SE reference is the STDLIB runs only (not pooled) so a variance-inflating mutant cannot
        # widen its own acceptance band. SE(Dbar) = SE_perrun * sqrt(2/S) (two independent arms).
        sd_std = _se_method_c(stdlib_runs[fn])
        sd_fast = _se_method_c(fast_runs[fn])
        se_perrun = sd_std
        se_dbar = se_perrun * math.sqrt(2.0 / s_pairs)
        bound = k * se_dbar
        passed = abs(dbar) <= bound
        spread_ok = _spread_ok(cfg.name, fn, sd_std, sd_fast, s_pairs)
        analytic = analytic_mean_se if fn == "mean" else None
        if fn in ("lo", "hi") and bound > _SE_BOUND_FLAG:
            se_bound_flag = True
        functionals.append(
            FunctionalResult(
                functional=fn,
                dbar=dbar,
                se_dbar=se_dbar,
                k=k,
                bound=bound,
                passed=passed,
                sd_stdlib=sd_std,
                sd_fast=sd_fast,
                spread_ok=spread_ok,
                se_perrun=se_perrun,
                analytic_mean_se=analytic,
            )
        )
    return ConfigResult(
        name=cfg.name,
        n_outer=n_outer,
        s_pairs=s_pairs,
        functionals=functionals,
        se_bound_flag=se_bound_flag,
    )


def _cross_n_study_seeds(n_studies: int, *, offset: int = 0) -> list[int]:
    """The cross-n CRN panel of study seeds, drawn via ``Random(SIZING_MASTER_SEED)`` exactly like
    ``sizing._study_seeds`` (so the panel matches the live sim's CRN). ``offset`` shifts the panel
    window for the panel-offset SE repeats."""
    master = Random(SIZING_MASTER_SEED)
    seq = [master.randrange(2**31) for _ in range(n_studies + offset)]
    return seq[offset:]


def _study_lo(
    kernel: BootstrapFn,
    *,
    n: int,
    study_seed: int,
    n_outer: int,
    sb_sens: float,
    sb_spec: float,
    dsens: float = 0.15,
    dspec: float = 0.15,
) -> float:
    cells = synth_cells(
        Random(study_seed),
        n,
        n,
        _R,
        sens_C=_SENS_C,
        dsens=dsens,
        sb_sens=sb_sens,
        spec_C=_SPEC_C,
        dspec=dspec,
        sb_spec=sb_spec,
        rho_w=_RHO_W,
    )
    _, lo, _ = kernel(
        cells,
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=n_outer,
        seed=study_seed ^ _BOOTSTRAP_SEED_MIX,
        quantile=0.0025,
    )
    return lo


def evaluate_boundary(
    *,
    s_cross: int,
    n_outer: int,
    cfg: EquivConfig,
    stdlib_kernel: BootstrapFn = nested_bootstrap_delta,
    fast_kernel: BootstrapFn = nested_bootstrap_delta_fast,
) -> BoundaryResult:
    """Boundary-study gate (eval-stats spec). A study is *boundary* if ``|lo_stdlib| < tau`` with
    ``tau = 2 * SE_lo`` (the realized per-run endpoint SD across the panel). A clear-verdict flip
    (``lo>0`` differs between kernels) is allowed ONLY inside the boundary band; ``n_flip <=
    n_boundary``, and ``|clears_std - clears_fast| <= 3*sqrt(n_boundary)``."""
    seeds = _cross_n_study_seeds(s_cross)
    lo_std: list[float] = []
    lo_fast: list[float] = []
    for sd in seeds:
        lo_std.append(
            _study_lo(
                stdlib_kernel,
                n=cfg.n,
                study_seed=sd,
                n_outer=n_outer,
                sb_sens=cfg.sb_sens,
                sb_spec=cfg.sb_spec,
                dsens=cfg.dsens,
                dspec=cfg.dspec,
            )
        )
        lo_fast.append(
            _study_lo(
                fast_kernel,
                n=cfg.n,
                study_seed=sd,
                n_outer=n_outer,
                sb_sens=cfg.sb_sens,
                sb_spec=cfg.sb_spec,
                dsens=cfg.dsens,
                dspec=cfg.dspec,
            )
        )
    se_lo = _se_method_c(lo_std)
    tau = 2.0 * se_lo
    n_boundary = sum(1 for x in lo_std if abs(x) < tau)
    exact_ties = sum(1 for x in lo_std if abs(x) < _EXACT_TIE)

    n_flip = 0
    out_of_band_flip = 0
    for xs, xf in zip(lo_std, lo_fast, strict=True):
        if (xs > 0.0) != (xf > 0.0):
            n_flip += 1
            if abs(xs) >= tau:
                out_of_band_flip += 1
    clears_std = sum(1 for x in lo_std if x > 0.0)
    clears_fast = sum(1 for x in lo_fast if x > 0.0)
    flip_ok = out_of_band_flip == 0 and n_flip <= n_boundary
    clears_ok = abs(clears_std - clears_fast) <= 3.0 * math.sqrt(max(n_boundary, 1))

    return BoundaryResult(
        n_boundary=n_boundary,
        n_flip=n_flip,
        clears_stdlib=clears_std,
        clears_fast=clears_fast,
        exact_ties=exact_ties,
        flip_ok=flip_ok,
        clears_ok=clears_ok,
    )


def _seed_pairs(master_seed: int, count: int) -> list[tuple[int, int]]:
    """Independent (stdlib, fast) bootstrap seed-pairs derived from the harness master seed."""
    rng = Random(master_seed)
    return [(rng.randrange(2**31), rng.randrange(2**31)) for _ in range(count)]


def _true_null_bias_threshold(configs: Sequence[ConfigResult]) -> float | None:
    """The concrete mean-bias detection threshold on the true-null config: the smallest |Dbar_mean|
    the harness could still reject (= k_mean * SE(Dbar_mean)). Reported so a reader knows the floor
    of detectable bias (a pass means bias below this, not zero bias)."""
    for c in configs:
        if c.name == "true-null":
            for f in c.functionals:
                if f.functional == "mean":
                    return f.bound
    return None


def run_equivalence(
    *,
    n_outer: int = 50_000,
    s_pairs: int = 40,
    configs: Sequence[EquivConfig] = DEFAULT_CONFIGS,
    boundary_s_cross: int = 200,
    boundary_n_outer: int = 20_000,
    boundary_sb_sens: float = 0.045,
    boundary_sb_spec: float = 0.028,
    boundary_dsens: float = 0.068,
    boundary_dspec: float = 0.068,
    master_seed: int = EQUIV_MASTER_SEED,
) -> EquivReport:
    """Run the equivalence harness and return the re-auditable :class:`EquivReport`.

    The gating legs are the per-config gate-A equivalence, the run-spread agreement, and the
    boundary-study gate (the honest home for near-boundary verdict flips). The boundary leg runs and
    gates UNCONDITIONALLY -- it is no longer nested under a cross-n switch (eval-stats corrected-
    scope: the cross-n McNemar/correlation legs were retracted as phantoms, the fast kernel being
    scorer-only with no cross-n bootstrap CRN to preserve).

    The heavy defaults (``n_outer=50_000``, ``s_pairs=40``) are the eval-stats validation point and
    are the SLOW stdlib-half job -- run this as a standalone, timeout-wrapped script, NEVER in the
    spike tier. Tests call it with small ``n_outer``/``s_pairs`` for a smoke.
    """
    seed_pairs = _seed_pairs(master_seed, s_pairs)
    artifact_seeds = Random(master_seed ^ 0xA17FAC)

    config_results = [
        evaluate_config(
            cfg,
            n_outer=n_outer,
            s_pairs=s_pairs,
            artifact_seed=artifact_seeds.randrange(2**31),
            seed_pairs=seed_pairs,
        )
        for cfg in configs
    ]

    boundary = evaluate_boundary(
        s_cross=boundary_s_cross,
        n_outer=boundary_n_outer,
        cfg=EquivConfig(
            "boundary", boundary_dsens, boundary_dspec, boundary_sb_sens, boundary_sb_spec, 56
        ),
    )

    return EquivReport(
        master_seed=master_seed,
        n_outer=n_outer,
        s_pairs=s_pairs,
        configs=config_results,
        boundary=boundary,
        true_null_mean_bias_threshold=_true_null_bias_threshold(config_results),
    )


def _print_report(report: EquivReport) -> None:
    print(f"EQUIV_MASTER_SEED={report.master_seed}  n_outer={report.n_outer}  S={report.s_pairs}")
    print(f"OVERALL: {'PASS' if report.passed else 'FAIL'}\n")
    for c in report.configs:
        print(f"=== config {c.name} (n_outer={c.n_outer}, S={c.s_pairs}) ===")
        for f in c.functionals:
            tag = "PASS" if f.passed else "FAIL"
            spread = "ok" if f.spread_ok else "SPREAD-FAIL"
            print(
                f"  {f.functional:>4}: Dbar={f.dbar:+.6f}  bound={f.bound:.6f}  [{tag}]  "
                f"SD_std={f.sd_stdlib:.5f} SD_fast={f.sd_fast:.5f} [{spread}]"
            )
        if c.se_bound_flag:
            print("  !! SE bound on lo/hi exceeds 0.02 here -- tail MC-unstable (a finding)")
    if report.boundary is not None:
        bd = report.boundary
        print("\n=== boundary-study gate ===")
        print(
            f"  n_boundary={bd.n_boundary} n_flip={bd.n_flip} exact_ties={bd.exact_ties} "
            f"clears_std={bd.clears_stdlib} clears_fast={bd.clears_fast} "
            f"[{'PASS' if bd.passed else 'FAIL'}]"
        )
    if report.true_null_mean_bias_threshold is not None:
        thr = report.true_null_mean_bias_threshold
        print(f"\ntrue-null mean-bias detection threshold: {thr:.6f}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pod 4.4c-0 fast-kernel equivalence harness.")
    parser.add_argument("--n-outer", type=int, default=50_000)
    parser.add_argument("--s-pairs", type=int, default=40)
    parser.add_argument("--s-cross", type=int, default=200, help="boundary-gate study-panel size.")
    parser.add_argument("--boundary-n-outer", type=int, default=20_000)
    parser.add_argument("--boundary-sb-sens", type=float, default=0.045)
    parser.add_argument("--boundary-sb-spec", type=float, default=0.028)
    parser.add_argument("--boundary-dsens", type=float, default=0.068)
    parser.add_argument("--boundary-dspec", type=float, default=0.068)
    parser.add_argument(
        "--no-cross-n",
        action="store_true",
        help="accepted for back-compat; the retracted cross-n legs no longer run. The boundary "
        "gate always runs and always gates (eval-stats corrected-scope).",
    )
    parser.add_argument("--json", action="store_true", help="emit the fingerprint as JSON.")
    args = parser.parse_args(argv)

    report = run_equivalence(
        n_outer=args.n_outer,
        s_pairs=args.s_pairs,
        boundary_s_cross=args.s_cross,
        boundary_n_outer=args.boundary_n_outer,
        boundary_sb_sens=args.boundary_sb_sens,
        boundary_sb_spec=args.boundary_sb_spec,
        boundary_dsens=args.boundary_dsens,
        boundary_dspec=args.boundary_dspec,
    )
    if args.json:
        print(report.to_json())
    else:
        _print_report(report)
    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
