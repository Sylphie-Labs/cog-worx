"""Pre-nightly sizing simulation for the Phase-4 GATE corpus (Pod 4.4c-0).

This is the simulation CRITICAL-2 says was missing: every n-decision in the plan rests on a
``power_lcb >= 0.80`` claim that was **never computed**. This module computes it, on the EXACT
landed code path (:func:`cogworx.eval.youden.synth_cells` +
:func:`cogworx.eval.youden.nested_bootstrap_delta` + :func:`power_lcb_from_studies`) -- no parallel
reimplementation of any statistical function. It is the artifact that drives Jim's F2 ratification.

The design is plan §6.0:

* **Study-loop** -- for ``N_studies`` synthetic studies, draw a ``Cell`` artifact at the planning
  variances (effect = Δmin), run the nested cluster bootstrap, count how many clear (CI lo > 0),
  then take ``power_lcb_from_studies(clears, N_studies)``. Certify iff that LCB ``>= 0.80``.
* **Sized at the gate you actually run** -- the inner bootstrap CIs use the *corrected* quantile the
  live gate uses (q=0.0025, the look-corrected ``alpha/(2*K_max)``), via the R5 ``quantile`` kwarg;
  nominal q=0.025 is reported alongside for transparency. The WHOLE sim runs at ONE sim-wide
  ``n_outer >= 10000`` so the two quantile arms share estimator variance.
* **Companions** -- a true-null run (``dsens=dspec=0`` -> require ``power_lcb <= 0.10``, proving the
  test can fail) and an under-power tripwire (``power_lcb(30) < 0.80 <= power_lcb(80)``).
* **Joint 3x3 variance-sensitivity sweep** -- the ``sb_*`` planning variances are author-chosen with
  no gate, so n is ratified ONLY against the joint pessimistic corner ``(0.045, 0.028)``.
* **n-decision ladder + terminal ABORT** -- the smallest n in the ascending ladder that clears 0.80
  at the pessimistic corner; if none clears the sim ABORTS (escalate to architect + Jim) rather than
  silently shipping the largest-n-that-looked-close.
* **CRN across the n=56 and n=80 arms** -- the same ``study_seed`` sequence for both, so the paired
  56-vs-80 gap resolves at N=500 (a build constraint, not optional).

S1/S6: the bootstrap is model-free -- it resamples the frozen synthetic artifact, never calls a
model. Reproducible: one ``MASTER_SEED`` recorded in :class:`SizingArtifact`.

Run the full heavy sweep directly as a script (NOT under pytest -- the spike tier can wedge the
machine; see the test-hang discipline)::

    python -m cogworx.eval.sizing            # full sweep, n_outer=20000
    python -m cogworx.eval.sizing --fast     # tiny smoke (what the pytest smoke exercises)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from random import Random
from typing import Literal

from cogworx.eval.youden import (
    nested_bootstrap_delta,
    power_lcb_from_studies,
    realized_variance_diagnostic,
    synth_cells,
)

Kernel = Literal["stdlib", "fast"]
"""Which bootstrap substrate runs the inner CI. ``"stdlib"`` is the live-gate authority (pure
stdlib, always available); ``"fast"`` is the numpy-vectorized SAME estimator for the heavy sweep and
requires the optional ``cogworx[sizing]`` extra. ``synth_cells`` stays stdlib in BOTH branches, so a
given ``study_seed`` yields byte-identical cells -- only the bootstrap RNG differs."""

# The bootstrap signature both kernels satisfy: (cells, *, arm_a, arm_b, error_strata, n_outer,
# seed, quantile) -> (mean, lo, hi). The fast kernel is lazy-imported so numpy stays optional.
_BootstrapFn = Callable[..., tuple[float, float, float]]


def _resolve_bootstrap(kernel: Kernel) -> _BootstrapFn:
    """Return the bootstrap callable for ``kernel``. The fast path lazy-imports numpy so that
    ``import cogworx.eval`` and the stdlib gate never touch numpy (CANON S2: numpy is the optional
    ``sizing`` extra). A missing numpy on the fast path is a system-boundary error surfaced here."""
    if kernel == "stdlib":
        return nested_bootstrap_delta
    from cogworx.eval._sizing_fast import nested_bootstrap_delta_fast

    return nested_bootstrap_delta_fast


MASTER_SEED = 20260616
"""The pinned master seed (the plan date as a decimal literal) -- recorded in the artifact so the
sizing run is reproducible and re-auditable."""

_BOOTSTRAP_SEED_MIX = 0xB007
"""XOR mix that derives the bootstrap seed from the synth seed, so synth and bootstrap never share
one RNG (synth consumes a variable number of draws -> a shared stream would be non-reproducible)."""

# Planning point (plan §6.0 / §13.6): effect = Δmin, mean sens/spec, within-item correlation.
_SENS_C = 0.60
_SPEC_C = 0.85
_DSENS = 0.15
_DSPEC = 0.15
_RHO_W = 0.30
_R = 7

# Planning between-item variances and the joint 3x3 stress grid (plan sec 6.0 variance sweep).
_SB_SENS_PLANNING = 0.020
_SB_SPEC_PLANNING = 0.010
_SB_SENS_GRID = (0.020, 0.030, 0.045)
_SB_SPEC_GRID = (0.010, 0.018, 0.028)
_PESSIMISTIC_CORNER = (0.045, 0.028)

# The gate's corrected (look-counter) quantile and the nominal within-run quantile.
_Q_CORRECTED = 0.0025
_Q_NOMINAL = 0.025

# The certification bar and the companion bars.
_POWER_BAR = 0.80
_TRUE_NULL_BAR = 0.10

# The n-decision ladder (ascending) and the E1/E2 split point.
_N_LADDER = (80, 100, 120, 150, 200)
_N_SPLIT = 56


@dataclass(frozen=True)
class StudyLoopResult:
    """One ``(n, sb_sens, sb_spec, quantile)`` study-loop outcome on the landed code path."""

    n: int
    sb_sens: float
    sb_spec: float
    quantile: float
    n_studies: int
    n_outer: int
    clears: int
    power_lcb: float


@dataclass(frozen=True)
class SizingArtifact:
    """The reproducible sizing artifact -- the F2-ratification input.

    ``master_seed`` is recorded so the run re-audits; ``decision`` is one of ``E1`` / ``E2`` /
    ``escalate-n=<n>`` / ``ABORT``.
    """

    master_seed: int
    n_studies: int
    n_outer: int
    corrected_quantile: float
    nominal_quantile: float
    pessimistic_corner: tuple[float, float]
    grid_corrected: list[StudyLoopResult] = field(default_factory=list)
    grid_nominal: list[StudyLoopResult] = field(default_factory=list)
    true_null: StudyLoopResult | None = None
    tripwire_30: StudyLoopResult | None = None
    tripwire_80: StudyLoopResult | None = None
    paired_56: StudyLoopResult | None = None
    paired_80: StudyLoopResult | None = None
    decision: str = ""
    kernel: Kernel = "stdlib"
    boundary_studies: int = 0

    def to_json(self) -> str:
        return json.dumps(_artifact_to_dict(self), indent=2)


def _study_seeds(master_seed: int, n_studies: int) -> list[int]:
    """The pinned per-study seed sequence from one master RNG. Sharing this sequence across two n
    arms is the CRN pairing (plan §6.0): a study that is easy is easy for both n."""
    master = Random(master_seed)
    return [master.randrange(2**31) for _ in range(n_studies)]


def run_study_loop(
    *,
    n: int,
    sb_sens: float,
    sb_spec: float,
    dsens: float,
    dspec: float,
    quantile: float,
    n_studies: int,
    n_outer: int,
    study_seeds: list[int],
    kernel: Kernel = "stdlib",
) -> StudyLoopResult:
    """Run ``n_studies`` synthetic studies and return the certification LCB on the gate's power.

    Per study: a fresh ``Random(study_seed)`` drives :func:`synth_cells`; the bootstrap runs under
    ``seed = study_seed ^ _BOOTSTRAP_SEED_MIX`` so the two RNGs never share a stream. A study
    *clears* when the paired delta CI lower bound is strictly above zero (the gate verdict).

    ``kernel`` selects the bootstrap substrate (CANON S2): ``"stdlib"`` is the live-gate authority,
    ``"fast"`` is the numpy-vectorized SAME estimator. ``synth_cells`` is stdlib regardless, so the
    cells are byte-identical across kernels for a given ``study_seed`` -- only the bootstrap RNG
    differs (stdlib ``Random`` vs numpy ``PCG64``), which is exactly what the equivalence harness
    proves does not change the estimator.
    """
    bootstrap = _resolve_bootstrap(kernel)
    clears = 0
    for study_seed in study_seeds:
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
        _, lo, _ = bootstrap(
            cells,
            arm_a="D",
            arm_b="C",
            error_strata=("K",),
            n_outer=n_outer,
            seed=study_seed ^ _BOOTSTRAP_SEED_MIX,
            quantile=quantile,
        )
        if lo > 0.0:
            clears += 1
    return StudyLoopResult(
        n=n,
        sb_sens=sb_sens,
        sb_spec=sb_spec,
        quantile=quantile,
        n_studies=n_studies,
        n_outer=n_outer,
        clears=clears,
        power_lcb=power_lcb_from_studies(clears, n_studies, quantile=0.05),
    )


def _decide(grid_corrected: list[StudyLoopResult]) -> str:
    """The n-decision ladder + terminal ABORT (plan §6.0), against the pessimistic corner only.

    E2 iff power_lcb(56) < 0.80 <= power_lcb(80); E1 iff both 56 and 80 clear; else the smallest
    ladder n clearing 0.80; if none in the ladder clears -> ABORT (escalate to architect + Jim --
    NOT silently ship the largest-n-that-looked-close).
    """
    corner = {
        r.n: r.power_lcb for r in grid_corrected if (r.sb_sens, r.sb_spec) == _PESSIMISTIC_CORNER
    }
    p56 = corner.get(_N_SPLIT)
    p80 = corner.get(80)
    if p56 is not None and p80 is not None and p56 >= _POWER_BAR and p80 >= _POWER_BAR:
        return "E1"
    if p56 is not None and p80 is not None and p56 < _POWER_BAR <= p80:
        return "E2"
    for n in _N_LADDER:
        power = corner.get(n)
        if power is not None and power >= _POWER_BAR:
            return f"escalate-n={n}"
    return "ABORT"


def run_sizing(
    *,
    n_studies: int = 500,
    n_outer: int = 20_000,
    fast: bool = False,
    kernel: Kernel = "stdlib",
) -> SizingArtifact:
    """Run the full §6.0 sizing sim and return the reproducible :class:`SizingArtifact`.

    ``fast`` collapses the sweep to a tiny smoke (one grid corner, n_outer=200, n_studies=10) so the
    code path is exercised under pytest without the heavy wall-clock; the full sweep is for the
    direct script invocation. Both ride the IDENTICAL landed code path.

    POST-RUN-1 RE-SIZE CONTRACT (plan §6.0, CRITICAL-2(c)) -- NOT a live trigger here: after 4.4d
    run-1, the realized variance is read via
    :func:`realized_variance_diagnostic`'s ``var_delta_paired`` and converted to
    ``(sb_sens_real, sb_spec_real)``; if either exceeds the value n was *ratified against* (the
    pessimistic corner) OR ``rho_arm <= 0`` (CRN bought no reduction -> the rho planning assumptions
    were wrong and n was over-credited), this sim is re-run with the realized variances and the
    corpus EXPANDS to the new smallest n clearing 0.80 before the verdict is trusted. The trigger
    fires only after run-1; here we land the contract surface, not a live trigger. The diagnostic
    is imported (``realized_variance_diagnostic``) so the 4.4d re-size hook has the seam in hand.
    """
    _ = realized_variance_diagnostic  # POST-RUN-1 re-size hook seam (see docstring); not live here.

    grid: tuple[tuple[float, float], ...]
    ladder: tuple[int, ...]
    if fast:
        n_studies, n_outer = 10, 200
        grid = ((_SB_SENS_PLANNING, _SB_SPEC_PLANNING), _PESSIMISTIC_CORNER)
        ladder = (56, 80)
    else:
        grid = tuple((s, p) for s in _SB_SENS_GRID for p in _SB_SPEC_GRID)
        ladder = (30, _N_SPLIT, *_N_LADDER)

    seeds = _study_seeds(MASTER_SEED, n_studies)

    grid_corrected: list[StudyLoopResult] = []
    grid_nominal: list[StudyLoopResult] = []
    for n in ladder:
        for sb_sens, sb_spec in grid:
            grid_corrected.append(
                run_study_loop(
                    n=n,
                    sb_sens=sb_sens,
                    sb_spec=sb_spec,
                    dsens=_DSENS,
                    dspec=_DSPEC,
                    quantile=_Q_CORRECTED,
                    n_studies=n_studies,
                    n_outer=n_outer,
                    study_seeds=seeds,
                    kernel=kernel,
                )
            )
            grid_nominal.append(
                run_study_loop(
                    n=n,
                    sb_sens=sb_sens,
                    sb_spec=sb_spec,
                    dsens=_DSENS,
                    dspec=_DSPEC,
                    quantile=_Q_NOMINAL,
                    n_studies=n_studies,
                    n_outer=n_outer,
                    study_seeds=seeds,
                    kernel=kernel,
                )
            )

    # TRUE-NULL companion: dsens=dspec=0 at the planning variances -> require power_lcb <= 0.10.
    true_null = run_study_loop(
        n=80,
        sb_sens=_SB_SENS_PLANNING,
        sb_spec=_SB_SPEC_PLANNING,
        dsens=0.0,
        dspec=0.0,
        quantile=_Q_CORRECTED,
        n_studies=n_studies,
        n_outer=n_outer,
        study_seeds=seeds,
        kernel=kernel,
    )

    # UNDER-POWER tripwire: power_lcb(30) < 0.80 <= power_lcb(80) at the planning variances.
    tripwire_30 = _find(grid_corrected, 30, _SB_SENS_PLANNING, _SB_SPEC_PLANNING) or run_study_loop(
        n=30,
        sb_sens=_SB_SENS_PLANNING,
        sb_spec=_SB_SPEC_PLANNING,
        dsens=_DSENS,
        dspec=_DSPEC,
        quantile=_Q_CORRECTED,
        n_studies=n_studies,
        n_outer=n_outer,
        study_seeds=seeds,
        kernel=kernel,
    )
    tripwire_80 = _find(grid_corrected, 80, _SB_SENS_PLANNING, _SB_SPEC_PLANNING)

    paired_56 = _find(grid_corrected, _N_SPLIT, *_PESSIMISTIC_CORNER)
    paired_80 = _find(grid_corrected, 80, *_PESSIMISTIC_CORNER)

    decision = _decide(grid_corrected)

    return SizingArtifact(
        master_seed=MASTER_SEED,
        n_studies=n_studies,
        n_outer=n_outer,
        corrected_quantile=_Q_CORRECTED,
        nominal_quantile=_Q_NOMINAL,
        pessimistic_corner=_PESSIMISTIC_CORNER,
        grid_corrected=grid_corrected,
        grid_nominal=grid_nominal,
        true_null=true_null,
        tripwire_30=tripwire_30,
        tripwire_80=tripwire_80,
        paired_56=paired_56,
        paired_80=paired_80,
        decision=decision,
        kernel=kernel,
    )


def _find(
    results: list[StudyLoopResult], n: int, sb_sens: float, sb_spec: float
) -> StudyLoopResult | None:
    for r in results:
        if r.n == n and r.sb_sens == sb_sens and r.sb_spec == sb_spec:
            return r
    return None


def _result_to_dict(r: StudyLoopResult) -> dict[str, float | int]:
    return {
        "n": r.n,
        "sb_sens": r.sb_sens,
        "sb_spec": r.sb_spec,
        "quantile": r.quantile,
        "n_studies": r.n_studies,
        "n_outer": r.n_outer,
        "clears": r.clears,
        "power_lcb": r.power_lcb,
    }


def _artifact_to_dict(a: SizingArtifact) -> dict[str, object]:
    return {
        "master_seed": a.master_seed,
        "n_studies": a.n_studies,
        "n_outer": a.n_outer,
        "corrected_quantile": a.corrected_quantile,
        "nominal_quantile": a.nominal_quantile,
        "pessimistic_corner": list(a.pessimistic_corner),
        "grid_corrected": [_result_to_dict(r) for r in a.grid_corrected],
        "grid_nominal": [_result_to_dict(r) for r in a.grid_nominal],
        "true_null": _result_to_dict(a.true_null) if a.true_null else None,
        "tripwire_30": _result_to_dict(a.tripwire_30) if a.tripwire_30 else None,
        "tripwire_80": _result_to_dict(a.tripwire_80) if a.tripwire_80 else None,
        "paired_56": _result_to_dict(a.paired_56) if a.paired_56 else None,
        "paired_80": _result_to_dict(a.paired_80) if a.paired_80 else None,
        "decision": a.decision,
        "kernel": a.kernel,
        "boundary_studies": a.boundary_studies,
    }


def _print_report(a: SizingArtifact) -> None:
    print(f"MASTER_SEED={a.master_seed}  N_studies={a.n_studies}  n_outer={a.n_outer}")
    print(f"corrected q={a.corrected_quantile}  nominal q={a.nominal_quantile}")
    pc = a.pessimistic_corner
    print(f"pessimistic corner (sb_sens, sb_spec) = {pc}\n")

    ns = sorted({r.n for r in a.grid_corrected})
    print("=== power_lcb (CORRECTED q) -- full 3x3 sweep per n; * = pessimistic corner ===")
    header = "  sb_sens  sb_spec | " + "  ".join(f"n={n:>3}" for n in ns)
    print(header)
    for sb_sens in sorted({r.sb_sens for r in a.grid_corrected}):
        for sb_spec in sorted({r.sb_spec for r in a.grid_corrected}):
            cells = []
            for n in ns:
                r = _find(a.grid_corrected, n, sb_sens, sb_spec)
                cells.append(f"{r.power_lcb:6.3f}" if r else "   ---")
            star = " *" if (sb_sens, sb_spec) == pc else "  "
            print(f"{star}{sb_sens:7.3f}  {sb_spec:7.3f} | " + "  ".join(cells))

    print("\n=== companions ===")
    if a.true_null:
        ok = "PASS" if a.true_null.power_lcb <= _TRUE_NULL_BAR else "FAIL"
        print(f"true-null  power_lcb={a.true_null.power_lcb:.3f}  (<= {_TRUE_NULL_BAR}) [{ok}]")
    if a.tripwire_30 and a.tripwire_80:
        ok = "PASS" if a.tripwire_30.power_lcb < _POWER_BAR <= a.tripwire_80.power_lcb else "FAIL"
        print(
            f"tripwire   power_lcb(30)={a.tripwire_30.power_lcb:.3f} < "
            f"{_POWER_BAR} <= power_lcb(80)={a.tripwire_80.power_lcb:.3f} [{ok}]"
        )
    if a.paired_56 and a.paired_80:
        gap = a.paired_80.power_lcb - a.paired_56.power_lcb
        print(
            f"CRN 56-vs-80 (pessimistic corner)  power_lcb(56)={a.paired_56.power_lcb:.3f}  "
            f"power_lcb(80)={a.paired_80.power_lcb:.3f}  paired gap={gap:+.3f}"
        )

    print(f"\n=== n-DECISION: {a.decision} ===")
    if a.decision == "ABORT":
        print("ABORT: no n in the ladder cleared 0.80 at the pessimistic corner.")
        print("Escalate to architect + Jim. Recovery levers: re-derive planning variances (with a")
        print("defensible source) OR raise Dmin (n ~ 1/Dmin^2; 0.15->0.20 cuts required n ~0.56).")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Pod 4.4c-0 pre-nightly sizing simulation.")
    parser.add_argument("--fast", action="store_true", help="tiny smoke sweep (CI / sanity).")
    parser.add_argument("--n-studies", type=int, default=500)
    parser.add_argument("--n-outer", type=int, default=20_000)
    parser.add_argument(
        "--kernel",
        choices=("stdlib", "fast"),
        default="stdlib",
        help="bootstrap substrate: 'stdlib' (live-gate authority) or 'fast' (numpy, needs the "
        "cogworx[sizing] extra). Same estimator either way.",
    )
    parser.add_argument("--json", action="store_true", help="emit the artifact as JSON.")
    args = parser.parse_args(argv)

    artifact = run_sizing(
        n_studies=args.n_studies, n_outer=args.n_outer, fast=args.fast, kernel=args.kernel
    )
    if args.json:
        print(artifact.to_json())
    else:
        _print_report(artifact)
    return 0


if __name__ == "__main__":
    sys.exit(main())
