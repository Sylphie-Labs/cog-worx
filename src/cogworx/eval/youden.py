"""Nested cluster bootstrap for paired Youden's-J deltas -- the Phase-4 GATE statistics core.

The Phase-4 gate (Pod 4.4) asks whether the adversarial-antithesis dialectic (arm **D**) beats a
neutral second-look (arm **C**/**C'**) and the oracle-only floor (arm **A**) on **Youden's J**
(``J = sens + spec - 1``), with a *paired* bootstrap CI entirely above zero. A single catch-rate is
not honest -- a yes-machine scores 100 % catch-rate and J ~= 0 -- so J is the gated statistic.

The measurement model (plan sec 13.6 / eval-stats sec 4R) is a **two-level cluster bootstrap**:

* **outer** -- a *stratified* resample of *item* IDs with replacement, fixing the per-stratum n;
* **inner** -- a resample of the R *trials within each chosen item*, drawn once and shared across
  both arms under **common random numbers** (CRN), folded into the same outer draw.

J is recomputed end-to-end for both arms on the *same* draw, so (a) the inner trial-resample
re-inflates within-item flip variance -- a flaky judge earns a *wide* CI -- and (b) the negative
``Cov(sens, spec)`` a single global-bias judge induces lives inside J's spread, never assumed zero.
The shared **clean** stratum is resampled **once per outer iteration** and reused for every J that
needs spec; resampling it per-arm would inject artificial independence.

Pure stdlib + :mod:`cogworx.knowledge.beta`. No scipy, no numpy (CANON S2). All Monte-Carlo uses a
seeded :class:`random.Random` so every result is reproducible.

Contract changelog:
  - 2026-06-16 (Pod 4.4a): initial -- Cell (frozen value type), synth_cells, nested_bootstrap_delta,
    power_lcb_from_studies, realized_variance_diagnostic.  Additive new module; no existing callers.
  - 2026-06-19 (Pod 4.4c-0): nested_bootstrap_delta gains ``quantile: float = 0.025`` (the look-
    corrected gate quantile is plumbed via this kwarg).  Additive: the default reproduces the
    shipped 2.5/97.5 CI byte-for-byte, so existing callers are unaffected (pinned).
  - 2026-06-20 (Pod 4.4c-1): Cell gains ``regime: str = ""`` (reported-only error-regime taxonomy
    tag; never read by nested_bootstrap_delta / _index_cells, which partition on stratum).  Additive
    per CANON S6.1: default "" leaves every pre-existing Cell byte-identical; the three schema-pin
    sites + the _sizing_fast columnar round-trip moved in lockstep.
"""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Mapping, Sequence
from random import Random

from pydantic import BaseModel, ConfigDict

from cogworx.knowledge import beta

__all__ = [
    "Cell",
    "VarianceDiagnostic",
    "mc_proportion_lcb",
    "nested_bootstrap_delta",
    "power_lcb_from_studies",
    "realized_variance_diagnostic",
    "synth_cells",
]

# The three error/clean strata. "O" = oracle-adjudicable error, "K" = oracle-blind error,
# "clean" = independently re-verified correct (the specificity population, shared across J's).
Stratum = str


class Cell(BaseModel):
    """One ``(item, trial, arm)`` outcome -- the frozen-artifact record schema.

    ``seed`` is the CRN pairing unit: for a given ``(item_id, trial)`` every arm ran under the same
    seed, so a paired delta cancels the shared luck. ``flagged`` is the only outcome bit the scorer
    reads (1 = the arm raised a flag on this item); ``route`` is audit/diagnostic only.
    ``regime`` is the error-regime taxonomy tag (plan sec 2.A) -- reported-only corpus metadata,
    defaulting to ``""`` so the bootstrap, which partitions error-vs-clean purely from ``stratum``,
    never reads it. This is the EXACT schema both :func:`synth_cells` emits and
    :func:`nested_bootstrap_delta` consumes -- the shared-schema invariant the sizing simulation
    depends on (plan sec 13.6).
    """

    model_config = ConfigDict(frozen=True)

    item_id: int
    stratum: Stratum
    arm: str
    trial: int
    seed: int
    flagged: int
    route: str
    regime: str = ""


class VarianceDiagnostic(BaseModel):
    """Realized-variance read-out for the paired-J delta (plan sec 4.3).

    ``rho_arm = 1 - Var(delta_paired) / (Var(J_a) + Var(J_b))`` is the CRN pairing credit: pairing
    helped iff ``rho_arm > 0``. ``cov_sens_spec`` is estimated per arm as
    ``[Var(J) - Var(sens) - Var(spec)] / 2`` across the bootstrap resamples and is expected negative
    under a single global-bias judge.
    """

    model_config = ConfigDict(frozen=True)

    var_delta_paired: float
    var_j_a: float
    var_j_b: float
    rho_arm: float
    cov_sens_spec_a: float
    cov_sens_spec_b: float


# Internal artifact shape: item_id -> arm -> stratum -> list of `flagged` over the R trials.
# Built once from the flat Cell list; the bootstrap indexes it without re-scanning the artifact.
_CellsByItem = dict[int, dict[str, dict[Stratum, list[int]]]]


def _index_cells(artifact: Sequence[Cell]) -> tuple[_CellsByItem, dict[Stratum, list[int]]]:
    """Index a flat ``Cell`` list into ``item -> arm -> stratum -> flags`` and a stratum->items
    map. The trial list per ``(item, arm, stratum)`` is the inner-bootstrap urn."""
    by_item: _CellsByItem = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    items_in_stratum: dict[Stratum, set[int]] = defaultdict(set)
    for c in artifact:
        by_item[c.item_id][c.arm][c.stratum].append(c.flagged)
        items_in_stratum[c.stratum].add(c.item_id)
    return by_item, {s: sorted(ids) for s, ids in items_in_stratum.items()}


def _sens_spec(
    cells_by_item: _CellsByItem,
    error_item_ids: Sequence[int],
    arm: str,
    clean_ids: Sequence[int],
) -> tuple[float, float]:
    """Sensitivity and specificity for one arm over a given set of error and clean items.

    A flag on an error item is a true positive; an unflagged error item is a false negative. A flag
    on a clean item is a false positive; an unflagged clean item is a true negative. Each item may
    appear more than once in the resampled id lists (bootstrap with replacement) and contributes its
    trial outcomes once per appearance, so multiplicity scales the counts correctly.

    sens = TP / (TP + FN) over error items; spec = TN / (TN + FP) over clean items. Zero
    denominators (no error or no clean items) guard to 0.0 -- a degenerate draw, not a verdict.
    """
    tp = fn = 0
    for item_id in error_item_ids:
        for trials in cells_by_item[item_id][arm].values():
            for f in trials:
                if f:
                    tp += 1
                else:
                    fn += 1
    tn = fp = 0
    for item_id in clean_ids:
        for trials in cells_by_item[item_id][arm].values():
            for f in trials:
                if f:
                    fp += 1
                else:
                    tn += 1
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def _resample_trials(
    cells_by_item: _CellsByItem,
    item_id: int,
    arm: str,
    stratum: Stratum,
    trial_picks: Sequence[int],
) -> list[int]:
    """Apply a shared inner trial-resample (``trial_picks`` indices into the R trials) to one
    ``(item, arm, stratum)`` cell. Shared across arms under CRN so within-item flip variance is
    reinflated identically for every arm on the same draw."""
    trials = cells_by_item[item_id][arm][stratum]
    return [trials[i] for i in trial_picks]


def _bootstrap_sens_spec(
    resampled: Mapping[int, Mapping[str, list[int]]],
    error_item_ids: Sequence[int],
    arm: str,
    clean_ids: Sequence[int],
) -> tuple[float, float]:
    """sens/spec for one arm over an already-inner-resampled draw (``item -> arm -> flags``)."""
    tp = fn = 0
    for item_id in error_item_ids:
        for f in resampled[item_id][arm]:
            if f:
                tp += 1
            else:
                fn += 1
    tn = fp = 0
    for item_id in clean_ids:
        for f in resampled[item_id][arm]:
            if f:
                fp += 1
            else:
                tn += 1
    sens = tp / (tp + fn) if (tp + fn) else 0.0
    spec = tn / (tn + fp) if (tn + fp) else 0.0
    return sens, spec


def nested_bootstrap_delta(
    artifact: Sequence[Cell],
    *,
    arm_a: str,
    arm_b: str,
    error_strata: Sequence[Stratum],
    n_outer: int,
    seed: int,
    quantile: float = 0.025,
) -> tuple[float, float, float]:
    """Paired nested cluster bootstrap for ``delta = J(arm_a) - J(arm_b)``; percentile CI.

    Returns ``(mean_delta, ci_lo, ci_hi)``. ``error_strata`` names the error population for sens
    (e.g. ``("K",)``); spec always runs over the shared ``"clean"`` stratum. Per outer iteration:

    1. the error items are resampled *with replacement* (stratified -- per-stratum n fixed), and the
       clean items are resampled **once** and reused for both arms' spec;
    2. for every chosen item a single inner trial-resample is drawn and **shared across both arms**
       (CRN), so within-item flip variance reinflates identically;
    3. J is recomputed end-to-end for both arms on that one draw and ``delta = J_a - J_b`` recorded.

    ``quantile`` sets the (two-sided) percentile CI endpoints: ``lo`` at the ``quantile``-th order
    statistic, ``hi`` at the ``(1 - quantile)``-th. The default ``0.025`` is the shipped 2.5/97.5
    two-sided 95 % CI (the within-run Bonferroni half of alpha=0.05); existing callers that pass
    nothing get the byte-identical pre-kwarg result. The pre-nightly sizing simulation (plan
    sec 6.0 / sec 3.9-B) passes ``quantile=0.0025`` to size against the look-corrected gate
    ``alpha/(2*K_max)``. **Additive,
    backward-compatible (pre-1.0 minor bump): an optional param that never changes the default
    output.** The percentile CI mirrors ``test_pod_2_1_stats_spike.py:266-287``: sort the
    per-resample deltas, take the ``quantile``/``1 - quantile`` percentile indices.
    """
    cells_by_item, items_in_stratum = _index_cells(artifact)
    clean_pool = items_in_stratum.get("clean", [])
    error_pools = {s: items_in_stratum.get(s, []) for s in error_strata}
    rng = Random(seed)

    deltas: list[float] = []
    for _ in range(n_outer):
        clean_ids = [clean_pool[rng.randrange(len(clean_pool))] for _ in range(len(clean_pool))]
        error_ids: list[int] = []
        for stratum_items in error_pools.values():
            n = len(stratum_items)
            error_ids.extend(stratum_items[rng.randrange(n)] for _ in range(n))

        # One shared inner trial-resample per chosen item, applied to BOTH arms (CRN). Drawn against
        # the arm_a trial count; the artifact pins equal R across arms for a given (item, trial).
        resampled: dict[int, dict[str, list[int]]] = {}
        for item_id in (*error_ids, *clean_ids):
            if item_id in resampled:
                continue
            r = len(cells_by_item[item_id][arm_a][next(iter(cells_by_item[item_id][arm_a]))])
            trial_picks = [rng.randrange(r) for _ in range(r)]
            resampled[item_id] = {
                arm_a: _resample_trials(
                    cells_by_item,
                    item_id,
                    arm_a,
                    next(iter(cells_by_item[item_id][arm_a])),
                    trial_picks,
                ),
                arm_b: _resample_trials(
                    cells_by_item,
                    item_id,
                    arm_b,
                    next(iter(cells_by_item[item_id][arm_b])),
                    trial_picks,
                ),
            }

        sens_a, spec_a = _bootstrap_sens_spec(resampled, error_ids, arm_a, clean_ids)
        sens_b, spec_b = _bootstrap_sens_spec(resampled, error_ids, arm_b, clean_ids)
        j_a = sens_a + spec_a - 1.0
        j_b = sens_b + spec_b - 1.0
        deltas.append(j_a - j_b)

    deltas.sort()
    mean_delta = statistics.fmean(deltas)
    lo = deltas[int(quantile * n_outer)]
    hi = deltas[min(int((1.0 - quantile) * n_outer), n_outer - 1)]
    return mean_delta, lo, hi


def realized_variance_diagnostic(
    artifact: Sequence[Cell],
    *,
    arm_a: str,
    arm_b: str,
    error_strata: Sequence[Stratum],
    n_outer: int,
    seed: int,
) -> VarianceDiagnostic:
    """Recompute the realized-variance read-out (plan sec 4.3) on the same nested-bootstrap draws.

    Re-runs the sec 4.2 draw loop, collecting per-resample ``(J_a, J_b, sens_a, spec_a, sens_b,
    spec_b)`` so it can report ``Var(delta_paired)``, ``Var(J_a)``, ``Var(J_b)``, the CRN pairing
    credit ``rho_arm``, and the per-arm ``Cov(sens, spec)`` estimated as
    ``[Var(J) - Var(sens) - Var(spec)] / 2``. ``rho_arm > 0`` proves pairing bought variance
    reduction; a negative ``Cov`` is the expected global-bias signature.
    """
    cells_by_item, items_in_stratum = _index_cells(artifact)
    clean_pool = items_in_stratum.get("clean", [])
    error_pools = {s: items_in_stratum.get(s, []) for s in error_strata}
    rng = Random(seed)

    j_a_vals: list[float] = []
    j_b_vals: list[float] = []
    delta_vals: list[float] = []
    sens_a_vals: list[float] = []
    spec_a_vals: list[float] = []
    sens_b_vals: list[float] = []
    spec_b_vals: list[float] = []

    for _ in range(n_outer):
        clean_ids = [clean_pool[rng.randrange(len(clean_pool))] for _ in range(len(clean_pool))]
        error_ids: list[int] = []
        for stratum_items in error_pools.values():
            n = len(stratum_items)
            error_ids.extend(stratum_items[rng.randrange(n)] for _ in range(n))

        resampled: dict[int, dict[str, list[int]]] = {}
        for item_id in (*error_ids, *clean_ids):
            if item_id in resampled:
                continue
            key_a = next(iter(cells_by_item[item_id][arm_a]))
            r = len(cells_by_item[item_id][arm_a][key_a])
            trial_picks = [rng.randrange(r) for _ in range(r)]
            resampled[item_id] = {
                arm_a: _resample_trials(cells_by_item, item_id, arm_a, key_a, trial_picks),
                arm_b: _resample_trials(
                    cells_by_item,
                    item_id,
                    arm_b,
                    next(iter(cells_by_item[item_id][arm_b])),
                    trial_picks,
                ),
            }

        sens_a, spec_a = _bootstrap_sens_spec(resampled, error_ids, arm_a, clean_ids)
        sens_b, spec_b = _bootstrap_sens_spec(resampled, error_ids, arm_b, clean_ids)
        j_a = sens_a + spec_a - 1.0
        j_b = sens_b + spec_b - 1.0
        j_a_vals.append(j_a)
        j_b_vals.append(j_b)
        delta_vals.append(j_a - j_b)
        sens_a_vals.append(sens_a)
        spec_a_vals.append(spec_a)
        sens_b_vals.append(sens_b)
        spec_b_vals.append(spec_b)

    var_delta = statistics.pvariance(delta_vals)
    var_j_a = statistics.pvariance(j_a_vals)
    var_j_b = statistics.pvariance(j_b_vals)
    denom = var_j_a + var_j_b
    rho_arm = 1.0 - var_delta / denom if denom else 0.0
    cov_a = (var_j_a - statistics.pvariance(sens_a_vals) - statistics.pvariance(spec_a_vals)) / 2.0
    cov_b = (var_j_b - statistics.pvariance(sens_b_vals) - statistics.pvariance(spec_b_vals)) / 2.0
    return VarianceDiagnostic(
        var_delta_paired=var_delta,
        var_j_a=var_j_a,
        var_j_b=var_j_b,
        rho_arm=rho_arm,
        cov_sens_spec_a=cov_a,
        cov_sens_spec_b=cov_b,
    )


def synth_cells(
    rng: Random,
    m_K: int,
    m_clean: int,
    R: int,
    *,
    sens_C: float,
    dsens: float,
    sb_sens: float,
    spec_C: float,
    dspec: float,
    sb_spec: float,
    rho_w: float,
) -> list[Cell]:
    """Generate a synthetic ``Cell`` artifact at the planning variances (plan sec 4R.3 / sec 13.6).

    Two arms are emitted, ``"C"`` and ``"D"``, on the K (error) and clean strata. Per item a true
    rate is drawn with a between-item variance ``sb_*`` (the spread that survives bootstrapping
    items); D's rate is the C rate shifted by the effect ``d*`` and clamped to ``[0, 1]``, so the
    arms share a CRN-correlated base rate. Each cell then draws R Bernoulli trials via
    ``rng.random() < rate``, with a shared trial seed mixing in ``rho_w`` to couple within-item
    trials across arms (the within-item correlation). On K, arm A flags nothing
    (``sens_A == 0``); A is therefore not emitted -- its J on K is identically ``J_A = sens_A +
    spec_A - 1 = 0 + 1 - 1 = 0`` and the caller scores it analytically.

    For sens (K errors) ``flagged == catch`` (we WANT a flag on an error). For spec (clean items)
    the per-item flag *rate* is ``1 - spec`` (a flag on a clean item is a false positive), so a
    higher spec yields fewer flags.

    The emitted records use the SAME :class:`Cell` schema :func:`nested_bootstrap_delta` consumes --
    the shared-schema invariant the pre-nightly sizing verification rests on (asserted below and in
    the unit suite).
    """
    sens_D = min(1.0, max(0.0, sens_C + dsens))
    spec_D = min(1.0, max(0.0, spec_C + dspec))
    cells: list[Cell] = []

    def _beta_params(mean: float, var: float) -> tuple[float, float]:
        """Method-of-moments Beta(a, b) for a per-item rate with the given mean and between-item
        variance, clamped to a valid concentration."""
        mean = min(1.0 - 1e-6, max(1e-6, mean))
        max_var = mean * (1.0 - mean)
        var = min(max(var, 1e-9), max_var * 0.999)
        conc = mean * (1.0 - mean) / var - 1.0
        return mean * conc, (1.0 - mean) * conc

    def _emit(stratum: Stratum, m: int, rate_C: float, rate_D: float, id_base: int) -> None:
        a_C, b_C = _beta_params(rate_C, sb_sens if stratum == "K" else sb_spec)
        a_D, b_D = _beta_params(rate_D, sb_sens if stratum == "K" else sb_spec)
        for i in range(m):
            item_id = id_base + i
            # CRN base rate: a shared uniform draw maps through each arm's Beta inverse-CDF so the
            # two arms' per-item rates are positively correlated (rho_b ~ 0.5 by construction).
            u = rng.random()
            rate_c_item = beta.beta_inverse_cdf(min(1.0 - 1e-9, max(1e-9, u)), a_C, b_C)
            rate_d_item = beta.beta_inverse_cdf(min(1.0 - 1e-9, max(1e-9, u)), a_D, b_D)
            for trial in range(R):
                seed = (item_id << 16) ^ (trial << 1)
                # Shared within-item noise (couples arms across the trial); the per-arm draw mixes
                # in an arm-specific stream weighted by (1 - rho_w).
                shared = Random(seed).random()
                for arm, rate in (("C", rate_c_item), ("D", rate_d_item)):
                    arm_noise = Random(seed ^ (1 if arm == "C" else 2)).random()
                    draw = rho_w * shared + (1.0 - rho_w) * arm_noise
                    if stratum == "K":
                        flagged = 1 if draw < rate else 0  # rate == sensitivity (catch rate)
                    else:
                        flagged = 1 if draw < (1.0 - rate) else 0  # flag rate == 1 - spec
                    route = "flag" if flagged else "pass"
                    cells.append(
                        Cell(
                            item_id=item_id,
                            stratum=stratum,
                            arm=arm,
                            trial=trial,
                            seed=seed,
                            flagged=flagged,
                            route=route,
                        )
                    )

    _emit("K", m_K, sens_C, sens_D, id_base=0)
    _emit("clean", m_clean, spec_C, spec_D, id_base=10_000)

    # Shared-schema structural pin: synth_cells MUST emit the exact schema nested_bootstrap_delta
    # consumes. If this ever drifts the sizing simulation proves nothing about the real gate.
    assert all(isinstance(c, Cell) for c in cells)
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
    return cells


def mc_proportion_lcb(
    alpha: float,
    beta_param: float,
    *,
    samples: int,
    seed: int,
    quantile: float = 0.05,
) -> float:
    """Monte-Carlo ``quantile``-th percentile of ``Beta(alpha, beta_param)`` draws.

    The sampler-correctness cross-check (plan sec 4.5/sec 4R): the MC quantile must agree with the
    analytic :func:`cogworx.knowledge.beta.lcb` within the MC standard error. This bounds **one**
    proportion (a single Beta) -- it is NOT a joint-J bound. There is no shipped joint-J bound, and
    ``lcb(sens) + lcb(spec) - 1`` is **not** a valid 95 % J bound (it ignores ``Cov(sens, spec)``
    and is mis-calibrated). The nested-bootstrap percentile CI is the only gate authority on the
    paired delta-in-J; the two instruments are never substituted.
    """
    rng = Random(seed)
    draws = sorted(rng.betavariate(alpha, beta_param) for _ in range(samples))
    return draws[int(quantile * samples)]


def power_lcb_from_studies(clears: int, n_studies: int, *, quantile: float = 0.05) -> float:
    """One-sided lower confidence bound on the simulation-gate power (no scipy).

    ``clears`` = how many of ``n_studies`` synthetic studies cleared (CI lower bound > 0); the
    realized power is a proportion ``clears / n_studies``. Wraps
    ``beta.lcb(clears + 1, n_studies - clears + 1, quantile)`` -- the Jeffreys-style Beta posterior
    LCB on that proportion -- so the sizing gate can require ``power_lcb >= 0.80`` without numpy.
    """
    return beta.lcb(clears + 1, n_studies - clears + 1, quantile=quantile)
