"""numpy-vectorized fast kernel for the Pod 4.4c-0 sizing simulation (CANON S1/S2, S12).

This is the SAME estimator as :func:`cogworx.eval.youden.nested_bootstrap_delta` -- the same
two-level cluster bootstrap sampling distribution -- differing ONLY in its RNG substrate (numpy
``PCG64`` instead of stdlib ``random.Random``). It exists so the heavy sizing sweep
(``n_outer`` up to 50k x hundreds of studies) runs in tractable wall-clock. The stdlib function
stays the live gate's sole authority and this kernel's correctness anchor; the
:mod:`cogworx.eval.equiv_check` harness proves the two are the same estimator before any sizing
number from this path is trusted.

**numpy is the OPTIONAL ``cogworx[sizing]`` extra.** This module top-level-imports numpy and is
therefore NEVER imported by the core or the gate runtime -- :mod:`cogworx.eval` does not import it,
and :mod:`cogworx.eval.sizing` lazy-imports it only inside the ``kernel="fast"`` branch. ``import
cogworx.eval`` works with numpy absent.

The estimator, preserved verbatim from the stdlib (``youden.py:183-262``):

* **outer** -- stratified resample of *item* IDs with replacement, per-stratum n fixed; the clean
  pool is resampled ONCE per outer iteration and reused across both arms;
* **inner** -- ONE shared trial-resample per item applied to BOTH arms (the within-item CRN);
* **estimator** -- ``J = sens + spec - 1`` recomputed end-to-end per arm on the same draw,
  ``delta = J_a - J_b``, zero-denominator guards -> 0.0;
* **endpoints** -- ``lo = sorted[int(q*n_outer)]``,
  ``hi = sorted[min(int((1-q)*n_outer), n_outer-1)]``.

S1/S6: model-free -- it resamples the frozen synthetic artifact, never calls a model.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.random import PCG64, Generator, SeedSequence

from cogworx.eval.youden import Cell, Stratum

__all__ = [
    "ColumnarArtifact",
    "cells_to_columnar",
    "columnar_to_cells",
    "nested_bootstrap_delta_fast",
]

# Memory fence: cap the materialized (chunk, n_items, R) inner-pick tensor. n_outer is processed in
# chunks bounded by this so a 50k x hundreds-of-items x 7 run stays in RAM without changing draws.
_MAX_CHUNK_CELLS = 8_000_000


class ColumnarArtifact:
    """Lossless columnar view of a frozen :class:`Cell` list for the fast kernel.

    One row per item (every item belongs to exactly one stratum and carries exactly two arms with a
    constant trial count R -- the invariants :func:`cogworx.eval.youden.synth_cells` guarantees).
    ``flags_a`` / ``flags_b`` are ``(n_items, R)`` int arrays of the per-trial ``flagged`` bit; a
    row's arm-a / arm-b trials are paired so one shared inner trial-resample applies to both (the
    CRN). The
    ``cells_to_columnar`` / ``columnar_to_cells`` round-trip is pinned lossless in the test suite.
    """

    __slots__ = (
        "arm_a",
        "arm_b",
        "flags_a",
        "flags_b",
        "item_ids",
        "r",
        "regimes",
        "routes",
        "seeds",
        "strata",
        "stratum_rows",
        "trials",
    )

    def __init__(
        self,
        *,
        item_ids: np.ndarray,
        strata: list[Stratum],
        arm_a: str,
        arm_b: str,
        flags_a: np.ndarray,
        flags_b: np.ndarray,
        r: int,
        trials: np.ndarray,
        seeds: np.ndarray,
        routes: list[list[tuple[str, str]]],
        regimes: list[str],
    ) -> None:
        self.item_ids = item_ids
        self.strata = strata
        self.arm_a = arm_a
        self.arm_b = arm_b
        self.flags_a = flags_a
        self.flags_b = flags_b
        self.r = r
        self.trials = trials
        self.seeds = seeds
        self.routes = routes
        self.regimes = regimes
        self.stratum_rows: dict[Stratum, np.ndarray] = {
            s: np.flatnonzero(np.array([st == s for st in strata]))
            for s in dict.fromkeys(strata)
        }


def cells_to_columnar(artifact: Sequence[Cell], *, arm_a: str, arm_b: str) -> ColumnarArtifact:
    """Build the columnar view from a flat ``Cell`` list (the exact schema the stdlib consumes).

    Lossless: :func:`columnar_to_cells` reconstructs the original rows (pinned in the test suite). A
    boundary validation -- the artifact is external input to the fast path -- requires every item to
    carry exactly two arms (``arm_a``, ``arm_b``), one stratum, and a constant trial count R.
    """
    by_item: dict[int, dict[str, dict[int, tuple[Stratum, int, int, str, str]]]] = {}
    for c in artifact:
        by_item.setdefault(c.item_id, {}).setdefault(c.arm, {})[c.trial] = (
            c.stratum,
            c.seed,
            c.flagged,
            c.route,
            c.regime,
        )

    item_ids_sorted = sorted(by_item)
    if not item_ids_sorted:
        raise ValueError("empty artifact")

    r = len(by_item[item_ids_sorted[0]][arm_a])

    n = len(item_ids_sorted)
    item_ids = np.empty(n, dtype=np.int64)
    flags_a = np.empty((n, r), dtype=np.int64)
    flags_b = np.empty((n, r), dtype=np.int64)
    seeds = np.empty((n, r), dtype=np.int64)
    trials = np.empty((n, r), dtype=np.int64)
    strata: list[Stratum] = []
    regimes: list[str] = []
    routes: list[list[tuple[str, str]]] = []

    for row, item_id in enumerate(item_ids_sorted):
        arms = by_item[item_id]
        if set(arms) != {arm_a, arm_b}:
            raise ValueError(f"item {item_id} arms {set(arms)} != {{{arm_a!r}, {arm_b!r}}}")
        a_trials = arms[arm_a]
        b_trials = arms[arm_b]
        if len(a_trials) != r or len(b_trials) != r:
            raise ValueError(f"item {item_id} has inconsistent trial count (expected R={r})")
        item_strata = {v[0] for v in a_trials.values()} | {v[0] for v in b_trials.values()}
        if len(item_strata) != 1:
            raise ValueError(f"item {item_id} spans strata {item_strata}; expected exactly one")
        item_regimes = {v[4] for v in a_trials.values()} | {v[4] for v in b_trials.values()}
        if len(item_regimes) != 1:
            raise ValueError(f"item {item_id} spans regimes {item_regimes}; expected exactly one")
        item_ids[row] = item_id
        strata.append(item_strata.pop())
        regimes.append(item_regimes.pop())
        row_routes: list[tuple[str, str]] = []
        for trial in range(r):
            _stratum_a, seed_a, flag_a, route_a, _regime_a = a_trials[trial]
            _stratum_b, _seed_b, flag_b, route_b, _regime_b = b_trials[trial]
            flags_a[row, trial] = flag_a
            flags_b[row, trial] = flag_b
            seeds[row, trial] = seed_a
            trials[row, trial] = trial
            row_routes.append((route_a, route_b))
        routes.append(row_routes)

    return ColumnarArtifact(
        item_ids=item_ids,
        strata=strata,
        arm_a=arm_a,
        arm_b=arm_b,
        flags_a=flags_a,
        flags_b=flags_b,
        r=r,
        trials=trials,
        seeds=seeds,
        routes=routes,
        regimes=regimes,
    )


def columnar_to_cells(col: ColumnarArtifact) -> list[Cell]:
    """Reconstruct the flat ``Cell`` list from a columnar view (the lossless round-trip inverse).

    Emits rows in ``synth_cells`` order: per item, per trial, arm_a then arm_b. Paired with
    :func:`cells_to_columnar` this is the pinned shared-schema round-trip.
    """
    cells: list[Cell] = []
    for row, item_id in enumerate(col.item_ids.tolist()):
        stratum = col.strata[row]
        regime = col.regimes[row]
        for trial in range(col.r):
            seed = int(col.seeds[row, trial])
            route_a, route_b = col.routes[row][trial]
            cells.append(
                Cell(
                    item_id=int(item_id),
                    stratum=stratum,
                    arm=col.arm_a,
                    trial=trial,
                    seed=seed,
                    flagged=int(col.flags_a[row, trial]),
                    route=route_a,
                    regime=regime,
                )
            )
            cells.append(
                Cell(
                    item_id=int(item_id),
                    stratum=stratum,
                    arm=col.arm_b,
                    trial=trial,
                    seed=seed,
                    flagged=int(col.flags_b[row, trial]),
                    route=route_b,
                    regime=regime,
                )
            )
    return cells


def nested_bootstrap_delta_fast(
    cells: Sequence[Cell] | ColumnarArtifact,
    *,
    arm_a: str,
    arm_b: str,
    error_strata: Sequence[Stratum],
    n_outer: int,
    seed: int,
    quantile: float = 0.025,
) -> tuple[float, float, float]:
    """Vectorized paired nested cluster bootstrap -- the SAME estimator as the stdlib, faster RNG.

    Returns ``(mean_delta, ci_lo, ci_hi)``. ``error_strata`` names the error population for sens;
    spec runs over the shared ``"clean"`` stratum. Per outer iteration: the error items are
    resampled with replacement (stratified, per-stratum n fixed); the clean items resampled once and
    reused for both arms; one shared inner trial-resample per item applied to both arms (CRN); J
    recomputed end-to-end per arm and ``delta = J_a - J_b`` recorded.

    The RNG is ``Generator(PCG64(SeedSequence(seed)))`` -- the bit generator is pinned explicitly so
    a numpy version bump can't silently change draws. The endpoint index convention is verbatim from
    the stdlib (``int(q*n_outer)`` / ``min(int((1-q)*n_outer), n_outer-1)``).
    """
    col = cells if isinstance(cells, ColumnarArtifact) else cells_to_columnar(
        cells, arm_a=arm_a, arm_b=arm_b
    )
    rng = Generator(PCG64(SeedSequence(seed)))

    clean_rows = col.stratum_rows.get("clean", np.empty(0, dtype=np.int64))
    error_row_groups = [
        col.stratum_rows.get(s, np.empty(0, dtype=np.int64)) for s in error_strata
    ]

    n_clean = int(clean_rows.size)
    error_sizes = [int(g.size) for g in error_row_groups]
    r = col.r
    n_items = int(col.item_ids.size)

    flags_a = col.flags_a
    flags_b = col.flags_b

    chunk = max(1, min(n_outer, _MAX_CHUNK_CELLS // max(1, n_items * r)))
    deltas = np.empty(n_outer, dtype=np.float64)

    done = 0
    while done < n_outer:
        m = min(chunk, n_outer - done)

        # Inner trial-resample: one shared (m, n_items, R) pick array; applied to BOTH arms (CRN).
        # Gathering along the trial axis gives each item's resampled flag count per arm.
        picks = rng.integers(0, r, size=(m, n_items, r))
        # gather flags[item, picks] -> (m, n_items, R), sum over R -> resampled flag count per item.
        fc_a = _gather_counts(flags_a, picks)
        fc_b = _gather_counts(flags_b, picks)

        # Outer item resample. Error: stratified -- per stratum n fixed (sampled within the stratum
        # row block). Clean: resampled once, reused across both arms.
        sens_num_a = np.zeros(m, dtype=np.float64)
        sens_num_b = np.zeros(m, dtype=np.float64)
        sens_den = 0
        for group, gsize in zip(error_row_groups, error_sizes, strict=True):
            if gsize == 0:
                continue
            idx = rng.integers(0, gsize, size=(m, gsize))
            rows = group[idx]
            sens_num_a += np.take_along_axis(fc_a, rows, axis=1).sum(axis=1)
            sens_num_b += np.take_along_axis(fc_b, rows, axis=1).sum(axis=1)
            sens_den += gsize * r

        if n_clean:
            cidx = rng.integers(0, n_clean, size=(m, n_clean))
            crows = clean_rows[cidx]
            # spec numerator = true negatives = trials NOT flagged on clean items.
            spec_num_a = (r - np.take_along_axis(fc_a, crows, axis=1)).sum(axis=1)
            spec_num_b = (r - np.take_along_axis(fc_b, crows, axis=1)).sum(axis=1)
            spec_den = n_clean * r
        else:
            spec_num_a = np.zeros(m, dtype=np.float64)
            spec_num_b = np.zeros(m, dtype=np.float64)
            spec_den = 0

        sens_a = sens_num_a / sens_den if sens_den else np.zeros(m)
        sens_b = sens_num_b / sens_den if sens_den else np.zeros(m)
        spec_a = spec_num_a / spec_den if spec_den else np.zeros(m)
        spec_b = spec_num_b / spec_den if spec_den else np.zeros(m)

        j_a = sens_a + spec_a - 1.0
        j_b = sens_b + spec_b - 1.0
        deltas[done : done + m] = j_a - j_b
        done += m

    deltas.sort()
    mean_delta = float(deltas.mean())
    lo = float(deltas[int(quantile * n_outer)])
    hi = float(deltas[min(int((1.0 - quantile) * n_outer), n_outer - 1)])
    return mean_delta, lo, hi


def _gather_counts(flags: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """Resampled flag count per (outer iter, item): gather ``flags[item, picks]`` over the R trial
    axis and sum. ``flags`` is ``(n_items, R)``; ``picks`` is ``(m, n_items, R)`` of trial indices;
    returns ``(m, n_items)``."""
    gathered = np.take_along_axis(flags[np.newaxis, :, :], picks, axis=2)
    counts: np.ndarray = gathered.sum(axis=2).astype(np.float64)
    return counts
