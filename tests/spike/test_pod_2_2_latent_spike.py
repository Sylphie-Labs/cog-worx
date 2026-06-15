"""Pod 2.2 ACT-R activation quality spike (CANON S12).

Falsifiable question: does ``ln(1 + use_count) - 0.5 · ln(Δt_hours)`` select a hot-set that
captures future use better than the proven-prior-art baseline (sylphie's lexicographic ordering
``use_count DESC, last_used_at DESC``)?

CONCLUSION (recorded here in-code after running the suite):
  Four workload scenarios x 200 seeds x paired-bootstrap (10 k resamples) determine the outcome.

  PASS criteria (from the locked mythos design spec):
    W1 (stationary Zipf)   — composite CI not worse than lexicographic (both track freq well →
                             tie OK)
    W2 (regime shift)      — composite CI-better than frequency_only (ACT-R adapts; freq is
                             sticky)
    W3 (burst-and-die)     — composite CI-better than frequency_only (burst dominates
                             freq_only forever)
    W4 (cold-start trickle)— composite CI not worse than lexicographic

  A DEFERRED outcome (composite never CI-beats lexicographic) is a valid S12 result: ship
  lexicographic and defer ACT-R, exactly as Thompson sampling was deferred in Pod 2.1.
  The pre-committed fallback is recorded in each test that reaches it.

Pure Python — no live database, no adapters, no substrate imports.
"""

from __future__ import annotations

import random
import statistics
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.knowledge.latent_activation import ActivationParams, ActivationRow, select_hot_ids

pytestmark = pytest.mark.spike

# ---------------------------------------------------------------------------
# Shared simulation constants
# ---------------------------------------------------------------------------

_N_SEEDS = 200
_RESAMPLES = 10_000

# Epoch anchor — the "start" of simulated time.  Step n advances by _STEP_HOURS.
_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)
_STEP_HOURS = 1.0  # one simulated hour per step


def _ts(step: int) -> datetime:
    """Simulated wall-clock time at the given step index."""
    return _EPOCH + timedelta(hours=step * _STEP_HOURS)


# ---------------------------------------------------------------------------
# Selector implementations — pure Python, no DB
# ---------------------------------------------------------------------------


def _select_composite(
    rows: list[ActivationRow], now: datetime, capacity: int, d: float = 0.5
) -> frozenset[str]:
    """ACT-R composite selector (the spike candidate)."""
    params = ActivationParams(d=d, hot_capacity=capacity)
    return select_hot_ids(rows, now, params)


def _select_lexicographic(
    rows: list[ActivationRow], _now: datetime, capacity: int
) -> frozenset[str]:
    """Sylphie proven-prior-art baseline: (use_count DESC, last_used_at DESC, id ASC) top-N."""
    ranked = sorted(rows, key=lambda r: (-r.use_count, -r.last_used_at.timestamp(), r.id))
    return frozenset(r.id for r in ranked[:capacity])


def _select_frequency_only(
    rows: list[ActivationRow], _now: datetime, capacity: int
) -> frozenset[str]:
    """Frequency-only baseline: (use_count DESC, id ASC) top-N."""
    ranked = sorted(rows, key=lambda r: (-r.use_count, r.id))
    return frozenset(r.id for r in ranked[:capacity])


def _select_recency_only(
    rows: list[ActivationRow], _now: datetime, capacity: int
) -> frozenset[str]:
    """Recency-only baseline: (last_used_at DESC, id ASC) top-N."""
    ranked = sorted(rows, key=lambda r: (-r.last_used_at.timestamp(), r.id))
    return frozenset(r.id for r in ranked[:capacity])


# ---------------------------------------------------------------------------
# Bootstrap CI helper (same approach as Pod 2.1 stats spike)
# ---------------------------------------------------------------------------


def _bootstrap_ci(deltas: list[float], *, resamples: int, seed: int) -> tuple[float, float]:
    """95 % bootstrap CI for the mean of `deltas` via 10 k resamples."""
    rng = random.Random(seed)
    n = len(deltas)
    means: list[float] = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples)]
    return lo, hi


# ---------------------------------------------------------------------------
# Workload harness helpers
# ---------------------------------------------------------------------------


def _zipf_weights(m: int) -> list[float]:
    """Return unnormalised Zipf weights 1/rank for ranks 1..m."""
    return [1.0 / (i + 1) for i in range(m)]


def _weighted_choice(weights: list[float], rng: random.Random) -> int:
    """Single draw proportional to weights (items indexed 0..len-1)."""
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for i, w in enumerate(weights):
        acc += w
        if r <= acc:
            return i
    return len(weights) - 1  # numerical safety


def _hot_hit_rate(
    future_uses: list[str],
    hot_set: frozenset[str],
) -> float:
    """Fraction of future uses whose item id is in hot_set."""
    if not future_uses:
        return 0.0
    hits = sum(1 for item_id in future_uses if str(item_id) in hot_set)
    return hits / len(future_uses)


# ---------------------------------------------------------------------------
# W1 — Stationary Zipf
# ---------------------------------------------------------------------------
# M=50 items, Zipf demand, T=100 steps, sweep every W=10 steps, capacity=10.
# PASS: composite CI not worse than lexicographic (tie is acceptable; both track frequency well).

_W1_M = 50
_W1_T = 100
_W1_W = 10  # sweep window
_W1_CAP = 10


def _run_w1(seed: int) -> dict[str, float]:
    """Return final-window hit-rates for all selectors on one W1 seed."""
    rng = random.Random(seed)
    weights = _zipf_weights(_W1_M)
    use_counts = [0] * _W1_M
    last_used: list[datetime] = [_ts(0)] * _W1_M

    # Run T steps; evaluate on uses within the LAST window (steps T-W .. T).
    future_uses: list[int] = []  # item ids used in the evaluation window
    now = _ts(0)

    for step in range(1, _W1_T + 1):
        now = _ts(step)
        item = _weighted_choice(weights, rng)
        use_counts[item] += 1
        last_used[item] = now

        if step > _W1_T - _W1_W:
            future_uses.append(item)

        # Sweep at every W-th step — build hot-set from the state at sweep time.
        if step % _W1_W == 0:
            rows = [
                ActivationRow(id=str(i), use_count=use_counts[i], last_used_at=last_used[i])
                for i in range(_W1_M)
            ]
            hot_composite = _select_composite(rows, now, _W1_CAP)
            hot_lexico = _select_lexicographic(rows, now, _W1_CAP)
            hot_freq = _select_frequency_only(rows, now, _W1_CAP)
            hot_rec = _select_recency_only(rows, now, _W1_CAP)

    # Rate against the final hot-sets (the last sweep before T).
    future_ids = [str(i) for i in future_uses]
    return {
        "composite": _hot_hit_rate(future_ids, hot_composite),
        "lexicographic": _hot_hit_rate(future_ids, hot_lexico),
        "frequency_only": _hot_hit_rate(future_ids, hot_freq),
        "recency_only": _hot_hit_rate(future_ids, hot_rec),
    }


def test_spike_w1_stationary_zipf() -> None:
    """W1 — Stationary Zipf: composite should not be CI-worse than lexicographic.

    Both selectors track frequency well on a stable distribution; a tie is the expected outcome.
    PASS: lower bound of CI(composite - lexicographic) >= -0.02 (tolerance for numerical noise).
    """
    results = [_run_w1(seed) for seed in range(_N_SEEDS)]
    deltas_vs_lexico = [r["composite"] - r["lexicographic"] for r in results]
    mean_delta = statistics.fmean(deltas_vs_lexico)
    lo, hi = _bootstrap_ci(deltas_vs_lexico, resamples=_RESAMPLES, seed=20260_01)
    print(
        f"\n[W1 stationary Zipf] composite - lexicographic: mean={mean_delta:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}]"
    )
    print(
        f"[W1] composite hit-rate mean={statistics.fmean(r['composite'] for r in results):.4f} | "
        f"lexico={statistics.fmean(r['lexicographic'] for r in results):.4f} | "
        f"freq={statistics.fmean(r['frequency_only'] for r in results):.4f} | "
        f"rec={statistics.fmean(r['recency_only'] for r in results):.4f}"
    )

    _TOLERANCE = -0.02  # small grace: a tie is fine; a large loss is not
    if lo >= _TOLERANCE:
        print(
            f"[W1] PASS — composite not CI-worse than lexicographic (lo={lo:.4f} >= {_TOLERANCE})"
        )
    else:
        print(
            f"[W1] DEFERRED — composite CI-worse than lexicographic (lo={lo:.4f} < {_TOLERANCE}): "
            "ship lexicographic, defer ACT-R (valid S12 outcome)"
        )
    assert lo >= _TOLERANCE, (
        f"W1: composite is CI-worse than lexicographic by more than tolerance "
        f"(CI=[{lo:.4f},{hi:.4f}], tolerance={_TOLERANCE}): the formula regresses on stable demand"
    )


# ---------------------------------------------------------------------------
# W2 — Regime shift
# ---------------------------------------------------------------------------
# M=50 items. Steps 1-50: Zipf on items 0..24. Steps 51-100: Zipf shifts to items 25..49.
# Sweep every W=5 steps, capacity=10.
# PASS: composite CI-better than frequency_only (freq is sticky; ACT-R adapts).

_W2_M = 50
_W2_T = 100
_W2_SHIFT = 50
_W2_W = 5
_W2_CAP = 10


def _run_w2(seed: int) -> dict[str, float]:
    """Return second-half evaluation hit-rates on one W2 seed."""
    rng = random.Random(seed)
    half = _W2_M // 2
    weights_before = _zipf_weights(half) + [0.0] * half
    weights_after = [0.0] * half + _zipf_weights(half)

    use_counts = [0] * _W2_M
    last_used: list[datetime] = [_ts(0)] * _W2_M

    # Evaluation window: uses in the final W steps.
    future_uses: list[str] = []
    now = _ts(0)
    hot_composite: frozenset[str] = frozenset()
    hot_lexico: frozenset[str] = frozenset()
    hot_freq: frozenset[str] = frozenset()
    hot_rec: frozenset[str] = frozenset()

    for step in range(1, _W2_T + 1):
        now = _ts(step)
        weights = weights_before if step <= _W2_SHIFT else weights_after
        item = _weighted_choice(weights, rng)
        use_counts[item] += 1
        last_used[item] = now

        if step > _W2_T - _W2_W:
            future_uses.append(str(item))

        if step % _W2_W == 0:
            rows = [
                ActivationRow(id=str(i), use_count=use_counts[i], last_used_at=last_used[i])
                for i in range(_W2_M)
            ]
            hot_composite = _select_composite(rows, now, _W2_CAP)
            hot_lexico = _select_lexicographic(rows, now, _W2_CAP)
            hot_freq = _select_frequency_only(rows, now, _W2_CAP)
            hot_rec = _select_recency_only(rows, now, _W2_CAP)

    return {
        "composite": _hot_hit_rate(future_uses, hot_composite),
        "lexicographic": _hot_hit_rate(future_uses, hot_lexico),
        "frequency_only": _hot_hit_rate(future_uses, hot_freq),
        "recency_only": _hot_hit_rate(future_uses, hot_rec),
    }


def test_spike_w2_regime_shift() -> None:
    """W2 — Regime shift: composite should CI-beat frequency_only after the popularity reshuffle.

    frequency_only is sticky (can't forget the first-half leaders); ACT-R's recency term helps it
    adapt. PASS: CI(composite - frequency_only) lower bound > 0.
    """
    results = [_run_w2(seed) for seed in range(_N_SEEDS)]
    deltas_vs_freq = [r["composite"] - r["frequency_only"] for r in results]
    mean_delta = statistics.fmean(deltas_vs_freq)
    lo, hi = _bootstrap_ci(deltas_vs_freq, resamples=_RESAMPLES, seed=20260_02)
    print(
        f"\n[W2 regime shift] composite - frequency_only: mean={mean_delta:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}]"
    )
    print(
        f"[W2] composite={statistics.fmean(r['composite'] for r in results):.4f} | "
        f"lexico={statistics.fmean(r['lexicographic'] for r in results):.4f} | "
        f"freq={statistics.fmean(r['frequency_only'] for r in results):.4f} | "
        f"rec={statistics.fmean(r['recency_only'] for r in results):.4f}"
    )

    if lo > 0.0:
        print(f"[W2] PASS — composite CI-better than frequency_only (lo={lo:.4f} > 0)")
    else:
        print(
            f"[W2] DEFERRED — composite does NOT CI-beat frequency_only (lo={lo:.4f} <= 0): "
            "ship lexicographic, defer ACT-R (valid S12 outcome)"
        )
    assert lo > 0.0, (
        f"W2 DEFERRED: composite does not CI-beat frequency_only "
        f"(CI=[{lo:.4f},{hi:.4f}]) — ship lexicographic (pre-committed fallback)"
    )


# ---------------------------------------------------------------------------
# W3 — Burst-and-die
# ---------------------------------------------------------------------------
# M=50. Steps 1-20: item 0 ("burst") used 80 % of the time. Steps 21-100: item 0 never used;
# uniform over items 1-49. Sweep every W=10, capacity=10.
# PASS: composite CI-better than frequency_only (burst item dominates freq_only's ranking forever).

_W3_M = 50
_W3_T = 100
_W3_BURST_END = 20
_W3_BURST_ID = 0
_W3_BURST_PROB = 0.8
_W3_W = 10
_W3_CAP = 10


def _run_w3(seed: int) -> dict[str, float]:
    """Return evaluation-window hit-rates on one W3 seed."""
    rng = random.Random(seed)
    use_counts = [0] * _W3_M
    last_used: list[datetime] = [_ts(0)] * _W3_M

    future_uses: list[str] = []
    now = _ts(0)
    hot_composite: frozenset[str] = frozenset()
    hot_lexico: frozenset[str] = frozenset()
    hot_freq: frozenset[str] = frozenset()
    hot_rec: frozenset[str] = frozenset()

    for step in range(1, _W3_T + 1):
        now = _ts(step)
        if step <= _W3_BURST_END:
            item = _W3_BURST_ID if rng.random() < _W3_BURST_PROB else rng.randint(1, _W3_M - 1)
        else:
            item = rng.randint(1, _W3_M - 1)

        use_counts[item] += 1
        last_used[item] = now

        if step > _W3_T - _W3_W:
            future_uses.append(str(item))

        if step % _W3_W == 0:
            rows = [
                ActivationRow(id=str(i), use_count=use_counts[i], last_used_at=last_used[i])
                for i in range(_W3_M)
            ]
            hot_composite = _select_composite(rows, now, _W3_CAP)
            hot_lexico = _select_lexicographic(rows, now, _W3_CAP)
            hot_freq = _select_frequency_only(rows, now, _W3_CAP)
            hot_rec = _select_recency_only(rows, now, _W3_CAP)

    return {
        "composite": _hot_hit_rate(future_uses, hot_composite),
        "lexicographic": _hot_hit_rate(future_uses, hot_lexico),
        "frequency_only": _hot_hit_rate(future_uses, hot_freq),
        "recency_only": _hot_hit_rate(future_uses, hot_rec),
    }


def test_spike_w3_burst_and_die() -> None:
    """W3 — Burst-and-die: composite should CI-beat frequency_only post-burst.

    The burst item accumulates a high use_count but goes silent; frequency_only keeps it in the
    hot-set indefinitely while ACT-R's time-decay term eventually demotes it.
    PASS: CI(composite - frequency_only) lower bound > 0.
    """
    results = [_run_w3(seed) for seed in range(_N_SEEDS)]
    deltas_vs_freq = [r["composite"] - r["frequency_only"] for r in results]
    mean_delta = statistics.fmean(deltas_vs_freq)
    lo, hi = _bootstrap_ci(deltas_vs_freq, resamples=_RESAMPLES, seed=20260_03)
    print(
        f"\n[W3 burst-and-die] composite - frequency_only: mean={mean_delta:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}]"
    )
    print(
        f"[W3] composite={statistics.fmean(r['composite'] for r in results):.4f} | "
        f"lexico={statistics.fmean(r['lexicographic'] for r in results):.4f} | "
        f"freq={statistics.fmean(r['frequency_only'] for r in results):.4f} | "
        f"rec={statistics.fmean(r['recency_only'] for r in results):.4f}"
    )

    if lo > 0.0:
        print(f"[W3] PASS — composite CI-better than frequency_only (lo={lo:.4f} > 0)")
    else:
        print(
            f"[W3] DEFERRED — composite does NOT CI-beat frequency_only (lo={lo:.4f} <= 0): "
            "ship lexicographic, defer ACT-R (valid S12 outcome)"
        )
    assert lo > 0.0, (
        f"W3 DEFERRED: composite does not CI-beat frequency_only "
        f"(CI=[{lo:.4f},{hi:.4f}]) — ship lexicographic (pre-committed fallback)"
    )


# ---------------------------------------------------------------------------
# W4 — Cold-start trickle
# ---------------------------------------------------------------------------
# M=100 items added one-at-a-time every 2 steps over T=100 steps.
# Once added, items have Zipf demand. Capacity=20.
# PASS: composite CI not worse than lexicographic.

_W4_M = 100
_W4_T = 100
_W4_TRICKLE_EVERY = 2  # new item added every N steps
_W4_CAP = 20
_W4_W = 10


def _run_w4(seed: int) -> dict[str, float]:
    """Return evaluation-window hit-rates on one W4 seed."""
    rng = random.Random(seed)
    use_counts = [0] * _W4_M
    last_used: list[datetime] = [_ts(0)] * _W4_M
    n_active = 0  # items currently available

    future_uses: list[str] = []
    now = _ts(0)
    hot_composite: frozenset[str] = frozenset()
    hot_lexico: frozenset[str] = frozenset()
    hot_freq: frozenset[str] = frozenset()
    hot_rec: frozenset[str] = frozenset()

    for step in range(1, _W4_T + 1):
        now = _ts(step)

        # Trickle in a new item.
        if step % _W4_TRICKLE_EVERY == 0 and n_active < _W4_M:
            n_active += 1

        if n_active == 0:
            continue

        # Demand: Zipf over currently-active items.
        weights = _zipf_weights(n_active)
        item = _weighted_choice(weights, rng)
        use_counts[item] += 1
        last_used[item] = now

        if step > _W4_T - _W4_W:
            future_uses.append(str(item))

        if step % _W4_W == 0 and n_active > 0:
            rows = [
                ActivationRow(id=str(i), use_count=use_counts[i], last_used_at=last_used[i])
                for i in range(n_active)
            ]
            hot_composite = _select_composite(rows, now, _W4_CAP)
            hot_lexico = _select_lexicographic(rows, now, _W4_CAP)
            hot_freq = _select_frequency_only(rows, now, _W4_CAP)
            hot_rec = _select_recency_only(rows, now, _W4_CAP)

    return {
        "composite": _hot_hit_rate(future_uses, hot_composite),
        "lexicographic": _hot_hit_rate(future_uses, hot_lexico),
        "frequency_only": _hot_hit_rate(future_uses, hot_freq),
        "recency_only": _hot_hit_rate(future_uses, hot_rec),
    }


def test_spike_w4_cold_start_trickle() -> None:
    """W4 — Cold-start trickle: composite should not be CI-worse than lexicographic.

    As items trickle in, fresh rows briefly carry a high activation from the recency term; the
    question is whether this helps or hurts. PASS: lower bound of CI >= -0.02.
    """
    results = [_run_w4(seed) for seed in range(_N_SEEDS)]
    deltas_vs_lexico = [r["composite"] - r["lexicographic"] for r in results]
    mean_delta = statistics.fmean(deltas_vs_lexico)
    lo, hi = _bootstrap_ci(deltas_vs_lexico, resamples=_RESAMPLES, seed=20260_04)
    print(
        f"\n[W4 cold-start trickle] composite - lexicographic: mean={mean_delta:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}]"
    )
    print(
        f"[W4] composite={statistics.fmean(r['composite'] for r in results):.4f} | "
        f"lexico={statistics.fmean(r['lexicographic'] for r in results):.4f} | "
        f"freq={statistics.fmean(r['frequency_only'] for r in results):.4f} | "
        f"rec={statistics.fmean(r['recency_only'] for r in results):.4f}"
    )

    _TOLERANCE = -0.02
    if lo >= _TOLERANCE:
        print(
            f"[W4] PASS — composite not CI-worse than lexicographic (lo={lo:.4f} >= {_TOLERANCE})"
        )
    else:
        print(
            f"[W4] DEFERRED — composite CI-worse than lexicographic (lo={lo:.4f} < {_TOLERANCE}): "
            "ship lexicographic, defer ACT-R (valid S12 outcome)"
        )
    assert lo >= _TOLERANCE, (
        f"W4 DEFERRED: composite is CI-worse than lexicographic by more than tolerance "
        f"(CI=[{lo:.4f},{hi:.4f}], tolerance={_TOLERANCE}): cold-start trickle regression"
    )


# ---------------------------------------------------------------------------
# d-sensitivity sweep
# ---------------------------------------------------------------------------
# Run composite with d ∈ {0.25, 0.5, 1.0} on W2 (the differentiation scenario).
# Weak but falsifiable claim: d=0.5 is never the worst d in W2.

_D_VALUES = [0.25, 0.5, 1.0]


def _run_w2_d(seed: int, d: float) -> float:
    """Return composite hit-rate for W2 at a given d value."""
    rng = random.Random(seed)
    half = _W2_M // 2
    weights_before = _zipf_weights(half) + [0.0] * half
    weights_after = [0.0] * half + _zipf_weights(half)

    use_counts = [0] * _W2_M
    last_used: list[datetime] = [_ts(0)] * _W2_M

    future_uses: list[str] = []
    now = _ts(0)
    hot: frozenset[str] = frozenset()

    for step in range(1, _W2_T + 1):
        now = _ts(step)
        weights = weights_before if step <= _W2_SHIFT else weights_after
        item = _weighted_choice(weights, rng)
        use_counts[item] += 1
        last_used[item] = now

        if step > _W2_T - _W2_W:
            future_uses.append(str(item))

        if step % _W2_W == 0:
            rows = [
                ActivationRow(id=str(i), use_count=use_counts[i], last_used_at=last_used[i])
                for i in range(_W2_M)
            ]
            hot = _select_composite(rows, now, _W2_CAP, d=d)

    return _hot_hit_rate(future_uses, hot)


def test_spike_d_sensitivity() -> None:
    """d-sensitivity sweep on W2: d=0.5 should not be the worst-performing d value.

    Runs composite with d ∈ {0.25, 0.5, 1.0} on the regime-shift workload.  Prints CIs for each
    d; asserts the weak but falsifiable claim that the shipped d=0.5 is not the single worst d
    (i.e. at least one other d has a lower mean hit-rate than d=0.5).

    If d=0.5 IS the worst, that suggests the formula is knife-edge or the wrong d is shipped —
    report to Jim and re-open the d-selection question before hardening.
    """
    per_d_means: dict[float, float] = {}
    per_d_ci: dict[float, tuple[float, float]] = {}

    for d in _D_VALUES:
        hit_rates = [_run_w2_d(seed, d) for seed in range(_N_SEEDS)]
        mean_hr = statistics.fmean(hit_rates)
        per_d_means[d] = mean_hr
        # Deltas vs lexicographic to get a comparable CI (same seeds → valid comparison).
        lexico_rates = [_run_w2(seed)["lexicographic"] for seed in range(_N_SEEDS)]
        deltas = [h - lx for h, lx in zip(hit_rates, lexico_rates, strict=True)]
        lo, hi = _bootstrap_ci(deltas, resamples=_RESAMPLES, seed=int(20260_05 + d * 100))
        per_d_ci[d] = (lo, hi)
        print(
            f"\n[d-sensitivity W2] d={d:.2f}: mean hit-rate={mean_hr:.4f} | "
            f"composite-lexico CI=[{lo:.4f}, {hi:.4f}]"
        )

    worst_d = min(per_d_means, key=per_d_means.__getitem__)
    best_d = max(per_d_means, key=per_d_means.__getitem__)
    print(
        f"\n[d-sensitivity summary] best d={best_d:.2f} ({per_d_means[best_d]:.4f}), "
        f"worst d={worst_d:.2f} ({per_d_means[worst_d]:.4f}), "
        f"shipped d=0.5 ({per_d_means[0.5]:.4f})"
    )

    if worst_d != 0.5:
        print("[d-sensitivity] PASS — d=0.5 is not the worst d in W2")
    else:
        print(
            "[d-sensitivity] NOTE — d=0.5 is the worst d in W2 (knife-edge concern); "
            "investigate d before hardening the formula"
        )

    assert worst_d != 0.5, (
        f"d-sensitivity: d=0.5 is the WORST performing d in W2 "
        f"(means: {per_d_means}) — the shipped d may be knife-edge; "
        "report to Jim and re-examine the d-selection before hardening"
    )
