"""Pod 4.4c-0 F2 decision run: the decision-critical sizing cells on the validated fast kernel.

The F2 n-decision (`_decide`) is made ONLY against the pessimistic corner (sb=0.045,0.028) at the
corrected gate quantile (q=0.0025); the off-corner 3x3 cells are transparency-only and do not enter
the decision. So this computes the pessimistic-corner ladder + the true-null and under-power
companions via the REAL `run_study_loop`/`_decide` code path (kernel='fast', validated equivalent to
stdlib), persists a JSON artifact, and prints the E1/E2/escalate/ABORT verdict.
"""

import json
import time

from cogworx.eval.sizing import (
    _DSENS,
    _DSPEC,
    _PESSIMISTIC_CORNER,
    _Q_CORRECTED,
    _SB_SENS_PLANNING,
    _SB_SPEC_PLANNING,
    _decide,
    _study_seeds,
    run_study_loop,
    MASTER_SEED,
)

N_STUDIES = 500
N_OUTER = 20_000
PESS = _PESSIMISTIC_CORNER          # (0.045, 0.028) -- the ratification corner
PLAN = (_SB_SENS_PLANNING, _SB_SPEC_PLANNING)  # (0.020, 0.010)
seeds = _study_seeds(MASTER_SEED, N_STUDIES)
t_start = time.time()


def loop(n, sb_sens, sb_spec, dsens, dspec, tag):
    t0 = time.time()
    r = run_study_loop(
        n=n, sb_sens=sb_sens, sb_spec=sb_spec, dsens=dsens, dspec=dspec,
        quantile=_Q_CORRECTED, n_studies=N_STUDIES, n_outer=N_OUTER,
        study_seeds=seeds, kernel="fast",
    )
    print(f"{tag:9s} n={n:>3} clears={r.clears}/{N_STUDIES} "
          f"power_lcb={r.power_lcb:.4f}  t={time.time()-t0:.0f}s  "
          f"[elapsed {time.time()-t_start:.0f}s]", flush=True)
    return r


# Decision: pessimistic corner at n=56 (E1/E2 split) and n=80 (>=0.80 already shown).
# n=30 included for the under-power discrimination proof (power_lcb(30) < 0.80).
grid_corrected = []
for n in (30, 56, 80):
    grid_corrected.append(loop(n, *PESS, _DSENS, _DSPEC, "PESS"))

# Under-power tripwire at the PLANNING corner: require power_lcb(30) < 0.80 <= power_lcb(80).
trip30 = loop(30, *PLAN, _DSENS, _DSPEC, "PLAN")
trip80 = loop(80, *PLAN, _DSENS, _DSPEC, "PLAN")

# True-null companion: dsens=dspec=0 -> require power_lcb <= 0.10 (proves the test can fail).
true_null = loop(80, *PLAN, 0.0, 0.0, "TRUENULL")

decision = _decide(grid_corrected)


def row(r):
    return {
        "n": r.n, "sb_sens": r.sb_sens, "sb_spec": r.sb_spec, "quantile": r.quantile,
        "n_studies": r.n_studies, "n_outer": r.n_outer, "clears": r.clears,
        "power_lcb": r.power_lcb,
    }


corner = {r.n: r.power_lcb for r in grid_corrected if (r.sb_sens, r.sb_spec) == PESS}
artifact = {
    "master_seed": MASTER_SEED, "n_studies": N_STUDIES, "n_outer": N_OUTER,
    "kernel": "fast", "corrected_quantile": _Q_CORRECTED,
    "pessimistic_corner": list(PESS),
    "decision": decision,
    "pessimistic_corner_corrected": [row(r) for r in grid_corrected],
    "power_lcb_56": corner.get(56), "power_lcb_80": corner.get(80),
    "tripwire_planning_30": row(trip30), "tripwire_planning_80": row(trip80),
    "true_null": row(true_null),
    "wall_s": round(time.time() - t_start, 1),
}
with open("docs/sessions/2026-06-20-4.4c-0-sizing-F2.json", "w") as f:
    json.dump(artifact, f, indent=2)

print("\n=== F2 DECISION:", decision, "===")
print(f"power_lcb(56) pess={corner.get(56):.4f}  power_lcb(80) pess={corner.get(80):.4f}")
print(f"under-power tripwire: planning p30={trip30.power_lcb:.4f} < 0.80 <= p80={trip80.power_lcb:.4f}"
      f"  -> {'OK' if trip30.power_lcb < 0.80 <= trip80.power_lcb else 'CHECK'}")
print(f"true-null p={true_null.power_lcb:.4f} (<=0.10) -> {'OK' if true_null.power_lcb <= 0.10 else 'CHECK'}")
print("DONE_F2", flush=True)
