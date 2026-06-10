"""Pod 2.1 read-surface statistics spike (CANON S12) - the falsifiable bets for the promotion gate
and the read-surface selector. PURE Monte-Carlo (no substrate), seeded -> deterministic and CI-fast.

CONCLUSION (Jim's two decisions, recorded here in-code):
  - Claim (c) - GATE RETUNED. The ported tess one-sided 5 % constant (``lcb_quantile=0.05``) is
    ANTI-conservative under the exact discrete gate: at the binding null theta=0.70 it breaches the
    5 % bound at several small n, WORST at n=13 (P(promote)=0.0637, +1.4pp; n=8 also breaches at
    0.0576), recurring up to ~n=40. Tightening to ``lcb_quantile=0.025`` restores true <=5 %
    one-sided coverage EVERYWHERE (worst ~3.0 % at n=30). This was an S12 win - the breach was in a
    ported constant and surfaced only by checking exact discrete coverage. The shipped default is
    0.025; the historical 0.05 breach is PINNED below so the fix stays documented.

  - Claim (b) - THOMPSON DEFERRED, POSTERIOR-MEAN GREEDY SHIPS. Against the PATHOLOGICALLY STICKY
    tess baseline (n_trials DESC), Thompson "wins" - but that baseline is a strawman. Against a FAIR
    posterior-mean-greedy baseline (the selector we actually ship), Thompson does NOT robustly beat
    it in the hard near-tie regime: it ties or LOSES at T in {25,50,100}. Per Jim, Thompson is
    deferred (kept in-tree, callable, pending a future spike against a fair baseline at longer
    horizons / higher K); posterior-mean greedy is the production read surface.

Both claims use the same rigor: exact binomial ground truth for the gate; paired-bootstrap
(N>=200 seeds, CRN, per-seed arm permutation, random tie-breaks) for the bandit.
"""

from __future__ import annotations

import math
import random
import statistics

import pytest

from cogworx.knowledge.beta import lcb
from cogworx.knowledge.procedural_promotion import PromotionPolicy

pytestmark = pytest.mark.spike


# ---------------------------------------------------------------------------
# Claim (c) - promotion-gate calibration
# ---------------------------------------------------------------------------

# Beta(1,1) prior, matching ProcedureSuccess / the shipped gate.
_PRIOR_ALPHA = 1.0
_PRIOR_BETA = 1.0

# The shipped default gate quantile (Pod 2.1: tightened from the tess 0.05 to 0.025) and the
# historical tess value the gate was ported with (kept ONLY to pin the breach the spike found).
_SHIPPED_QUANTILE = 0.025
_LEGACY_QUANTILE = 0.05


def _gate_promotes(successes: int, n: int, policy: PromotionPolicy) -> bool:
    """The shipped gate applied to a deduped trial record of n trials with `successes` successes.

    Mirrors PromotionPolicy.should_promote against a Beta(1,1)+data posterior - the same alpha/beta
    a ProcedureSuccess would carry for n distinct-run trials.
    """
    if n < policy.min_trials:
        return False
    alpha = _PRIOR_ALPHA + successes
    beta = _PRIOR_BETA + (n - successes)
    return lcb(alpha, beta, quantile=policy.lcb_quantile) >= policy.lcb_threshold


def _false_promotion_rate(
    *, theta: float, n: int, sims: int, rng: random.Random, policy: PromotionPolicy
) -> float:
    promoted = 0
    for _ in range(sims):
        successes = sum(1 for _ in range(n) if rng.random() < theta)
        if _gate_promotes(successes, n, policy):
            promoted += 1
    return promoted / sims


def _exact_false_promotion_prob(*, theta: float, n: int, policy: PromotionPolicy) -> float:
    """Exact (no-MC) P(promote | null theta) for the discrete gate: sum the binomial mass over the
    success counts that clear the LCB. Flake-free ground truth the MC cells are checked against."""
    from math import comb

    return sum(
        comb(n, s) * theta**s * (1.0 - theta) ** (n - s)
        for s in range(n + 1)
        if _gate_promotes(s, n, policy)
    )


def _worst_cell(*, theta: float, quantile: float, n_lo: int, n_hi: int) -> tuple[int, float]:
    """The (n, P(promote)) cell with the largest exact false-promotion prob over [n_lo, n_hi]."""
    policy = PromotionPolicy(lcb_quantile=quantile)
    worst_n, worst_p = n_lo, -1.0
    for n in range(n_lo, n_hi + 1):
        p = _exact_false_promotion_prob(theta=theta, n=n, policy=policy)
        if p > worst_p:
            worst_n, worst_p = n, p
    return worst_n, worst_p


# The full small-n sweep the gate must respect at the SHIPPED quantile. No cherry-picking: EVERY
# cell in this range must clear the one-sided 5 % bound (the old harness only checked cells where it
# happened to pass).
_CALIBRATION_SWEEP = list(range(5, 41))


@pytest.mark.parametrize("n", _CALIBRATION_SWEEP)
def test_shipped_gate_respects_5pct_bound_everywhere(n: int) -> None:
    """Claim (c), SHIPPED gate (q=0.025): exact false-promotion at theta=0.70 is <= 5 % at ALL n.

    Exact binomial - no MC, no flake. The whole point of the 0.025 retune is that the discrete
    sawtooth never poke above 5 % anywhere in the operating range, so this asserts the full sweep,
    not a hand-picked subset.
    """
    policy = PromotionPolicy(lcb_quantile=_SHIPPED_QUANTILE)
    p_exact = _exact_false_promotion_prob(theta=0.70, n=n, policy=policy)
    print(f"[calibration q=0.025] theta=0.70 n={n}: exact P(promote)={p_exact:.5f} (bound 0.05)")
    assert p_exact <= 0.05, (
        f"shipped gate (q=0.025) breaches the 5% one-sided bound at theta=0.70, n={n}: "
        f"P(promote)={p_exact:.5f}"
    )


def test_shipped_gate_worst_cell_is_under_bound() -> None:
    """The worst cell of the SHIPPED gate over n=5..40 stays under 5 % (~3.0 % at n=30)."""
    worst_n, worst_p = _worst_cell(theta=0.70, quantile=_SHIPPED_QUANTILE, n_lo=5, n_hi=40)
    print(f"[calibration q=0.025] WORST cell over n=5..40: n={worst_n} P(promote)={worst_p:.5f}")
    assert worst_p <= 0.05
    assert worst_p > 0.0  # the gate does fire on the null sometimes - it is not vacuously safe


def test_legacy_gate_breaches_and_worst_cell_is_n13() -> None:
    """PINNED HISTORY (the S12 win): the LEGACY q=0.05 gate breaches the 5 % bound, WORST at n=13.

    The previous harness mislabelled the worst cell as n=8 (0.0576) and only checked cells where the
    gate happened to pass. The TRUE worst cell over n=5..40 is n=13 at 0.0637 (n=8 also breaches).
    This documents BOTH the breach and that 0.025 fixes it; it flips to a failure only if the legacy
    math is changed.
    """
    legacy = PromotionPolicy(lcb_quantile=_LEGACY_QUANTILE)

    p_n8 = _exact_false_promotion_prob(theta=0.70, n=8, policy=legacy)
    p_n13 = _exact_false_promotion_prob(theta=0.70, n=13, policy=legacy)
    worst_n, worst_p = _worst_cell(theta=0.70, quantile=_LEGACY_QUANTILE, n_lo=5, n_hi=40)
    print(
        f"[calibration LEGACY q=0.05] n=8 P={p_n8:.5f} (8/8=0.70^8); n=13 P={p_n13:.5f}; "
        f"WORST n={worst_n} P={worst_p:.5f} - ALL breach 0.05"
    )

    # n=8 breaches and equals 0.70**8 (promotion at n=8 requires 8/8 successes).
    assert math.isclose(p_n8, 0.70**8, rel_tol=1e-9)
    assert p_n8 > 0.05
    # The TRUE worst cell is n=13, NOT n=8, and it breaches harder.
    assert worst_n == 13
    assert worst_p > p_n8
    assert math.isclose(worst_p, p_n13, rel_tol=1e-12)
    assert p_n13 > 0.05
    # And the shipped 0.025 gate does NOT breach at the legacy worst cell - the fix is real.
    shipped = PromotionPolicy(lcb_quantile=_SHIPPED_QUANTILE)
    assert _exact_false_promotion_prob(theta=0.70, n=13, policy=shipped) <= 0.05


def test_shipped_gate_mc_matches_exact_at_worst_cell() -> None:
    """An MC cross-check that the live gate code agrees with the exact-binomial ground truth.

    Run at the shipped gate's worst cell so the MC estimate is meaningful (a non-trivial promotion
    rate to estimate), pinning that the Python gate and the binomial sum compute the same thing.
    """
    worst_n, p_exact = _worst_cell(theta=0.70, quantile=_SHIPPED_QUANTILE, n_lo=5, n_hi=40)
    policy = PromotionPolicy(lcb_quantile=_SHIPPED_QUANTILE)
    sims = 40_000
    rng = random.Random(2026_06_09 + worst_n)
    p_hat = _false_promotion_rate(theta=0.70, n=worst_n, sims=sims, rng=rng, policy=policy)
    se = math.sqrt(max(p_hat * (1.0 - p_hat), 1.0 / sims) / sims)
    print(
        f"[calibration MC q=0.025] n={worst_n} sims={sims}: phat={p_hat:.5f} exact={p_exact:.5f} "
        f"SE={se:.5f}"
    )
    assert math.isclose(p_hat, p_exact, abs_tol=4.0 * se), "live gate MC drifted from exact"


def test_promotion_gate_monotone_in_theta() -> None:
    """Sanity: false-promotion rate is non-decreasing as the null theta rises toward the
    threshold."""
    policy = PromotionPolicy(lcb_quantile=_SHIPPED_QUANTILE)
    rng = random.Random(7)
    rates = [
        _false_promotion_rate(theta=theta, n=13, sims=5_000, rng=rng, policy=policy)
        for theta in (0.60, 0.65, 0.70)
    ]
    assert rates[0] <= rates[1] + 0.01
    assert rates[1] <= rates[2] + 0.01


# ---------------------------------------------------------------------------
# Claim (b) - Thompson vs baselines synthetic bandit
# ---------------------------------------------------------------------------

# The three online policies. "sticky" is the rejected tess strawman; "pmgreedy" is the FAIR baseline
# (the posterior-mean greedy we SHIP); "ts" is the deferred Thompson candidate.
_Policy = str


def _play_bandit(
    thetas: list[float],
    *,
    horizon: int,
    policy: _Policy,
    seed: int,
) -> float:
    """Play `horizon` pulls of a K-arm Bernoulli bandit; return cumulative regret.

    All policies run off the SAME per-(arm, t) Bernoulli reward table (paired common-random-numbers)
    so they see identical luck and the per-seed delta isolates the policy. The arm->theta mapping is
    PERMUTED per seed so arm POSITION carries no information; every tie is broken from a seeded
    stream so no policy gets free index-order exploration.

    Policies:
      - ``sticky``   - the tess exploitation-only ranking ``n_trials (pulls) DESC, posterior-mean
        DESC``. Pathologically sticky: an early lucky pull can lock it onto one arm. This is the
        rejected strawman comparator (NOT what we ship).
      - ``pmgreedy`` - posterior-mean greedy: rank by Beta posterior MEAN. The Beta(1,1) prior
        gives free early exploration (a never-pulled arm sits at 0.5). THIS is the production read
        surface (``posterior_mean_select``) and the FAIR baseline the spec names.
      - ``ts``       - Thompson: draw ``theta ~ Beta(alpha, beta)`` per arm via the same
        ``rng.betavariate`` the read surface uses. The deferred candidate.
    Posterior is Beta(1,1) + data throughout.
    """
    k = len(thetas)
    best = max(thetas)
    perm_rng = random.Random((seed << 8) ^ 0xA11CE)
    perm = list(range(k))
    perm_rng.shuffle(perm)
    arm_theta = [thetas[perm[a]] for a in range(k)]

    alpha = [1.0] * k
    beta = [1.0] * k
    pulls = [0] * k
    reward_rng = random.Random((seed << 8) ^ 0xBEEF)
    rewards = [
        [1 if reward_rng.random() < arm_theta[a] else 0 for _ in range(horizon)] for a in range(k)
    ]
    select_rng = random.Random((seed << 8) ^ 0x1234)
    tie_rng = random.Random((seed << 8) ^ 0x5A5A)

    regret = 0.0
    for t in range(horizon):
        if policy == "ts":
            keys: list[tuple[float, ...]] = [
                (select_rng.betavariate(alpha[a], beta[a]),) for a in range(k)
            ]
        elif policy == "pmgreedy":
            keys = [(alpha[a] / (alpha[a] + beta[a]),) for a in range(k)]
        elif policy == "sticky":
            keys = [(float(pulls[a]), alpha[a] / (alpha[a] + beta[a])) for a in range(k)]
        else:  # pragma: no cover - guarded by the call sites
            raise ValueError(f"unknown policy {policy!r}")
        arm = max(range(k), key=lambda a: (keys[a], tie_rng.random()))
        if rewards[arm][t]:
            alpha[arm] += 1.0
        else:
            beta[arm] += 1.0
        pulls[arm] += 1
        regret += best - arm_theta[arm]
    return regret


def _bootstrap_ci(deltas: list[float], *, resamples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(deltas)
    means = []
    for _ in range(resamples):
        sample = [deltas[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    lo = means[int(0.025 * resamples)]
    hi = means[int(0.975 * resamples)]
    return lo, hi


def _paired_regret_deltas(
    thetas: list[float], *, horizon: int, seeds: int, baseline: _Policy
) -> list[float]:
    """Per-seed delta = regret(baseline) - regret(ts). Positive => TS has LOWER regret (TS wins)."""
    return [
        _play_bandit(thetas, horizon=horizon, policy=baseline, seed=s)
        - _play_bandit(thetas, horizon=horizon, policy="ts", seed=s)
        for s in range(seeds)
    ]


_NEAR_TIE = [0.72, 0.70, 0.55, 0.50, 0.45]
_CLEAR_WINNER = [0.90, 0.55, 0.50, 0.45, 0.40]
_SEEDS = 200
_RESAMPLES = 10_000


@pytest.mark.parametrize("horizon", [25, 50, 100])
def test_thompson_beats_sticky_tess_strawman(horizon: int) -> None:
    """RECORDED STRAWMAN: TS beats the pathologically sticky tess baseline (CI entirely > 0).

    This is the comparison the original spike used to (wrongly) "validate" TS. We KEEP it - it is a
    true statement about TS vs a bad baseline - but it does NOT justify shipping TS. See
    test_thompson_does_not_robustly_beat_fair_baseline for the honest comparison.
    """
    deltas = _paired_regret_deltas(_NEAR_TIE, horizon=horizon, seeds=_SEEDS, baseline="sticky")
    mean_delta = statistics.fmean(deltas)
    lo, hi = _bootstrap_ci(deltas, resamples=_RESAMPLES, seed=horizon)
    print(
        f"[TS vs STICKY strawman] near-tie T={horizon}: mean delta(sticky-TS)={mean_delta:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}] - TS beats the strawman iff lo>0"
    )
    assert lo > 0.0, "TS should beat the sticky strawman (this is the comparison being deprecated)"


@pytest.mark.parametrize("horizon", [25, 50, 100])
def test_thompson_does_not_robustly_beat_fair_baseline(horizon: int) -> None:
    """HONEST OUTCOME (the recorded reason to DEFER TS): against the FAIR posterior-mean-greedy
    baseline - the selector we actually ship - TS does NOT robustly win in the hard near-tie regime.

    The paired-bootstrap CI of delta(pmgreedy - TS) does NOT sit entirely above zero at T in
    {25,50,100}: TS ties or loses. (Empirically at T=25 the fair baseline is BETTER - the CI is
    entirely below zero.) Per Jim, this defers TS and ships posterior-mean greedy. We assert the
    NON-win directly so the suite never again declares "TS validated" off the strawman: TS must FAIL
    to clear the bar (lo <= 0) here.
    """
    deltas = _paired_regret_deltas(_NEAR_TIE, horizon=horizon, seeds=_SEEDS, baseline="pmgreedy")
    mean_delta = statistics.fmean(deltas)
    lo, hi = _bootstrap_ci(deltas, resamples=_RESAMPLES, seed=5000 + horizon)
    print(
        f"[TS vs FAIR pmgreedy] near-tie T={horizon}: mean delta(pmgreedy-TS)={mean_delta:.4f} "
        f"95% CI=[{lo:.4f}, {hi:.4f}] - TS would 'win' iff lo>0 (it does NOT: TS deferred)"
    )
    assert lo <= 0.0, (
        f"TS unexpectedly beats the FAIR baseline at T={horizon} (CI=[{lo:.4f},{hi:.4f}]): "
        "the deferral rationale would no longer hold - re-open the TS spike and report to Jim."
    )


def test_clear_winner_tradeoff_is_honest_and_horizon_dependent() -> None:
    """HONEST clear-winner characterization (NOT a one-sided "pmgreedy is free" claim).

    The fair comparison pmgreedy vs TS in the clear-winner regime is a genuine, horizon-dependent
    TRADE-OFF, recorded here so the spike does not oversell either selector:
      - SHORT T (25): TS has lower regret (pmgreedy's optimistic prior wastes a few early pulls
        exploring an obvious winner) → delta(pmgreedy-TS) < 0.
      - LONG  T (100): pmgreedy has lower regret (TS keeps paying an exploration tax it cannot stop)
        → delta(pmgreedy-TS) > 0.
    Neither dominates everywhere. This is why TS was deferred on the NEAR-TIE + determinism grounds
    (Jim's call), not on a false "pmgreedy never loses" basis. We assert the robust SIGNS (each CI
    clears zero in opposite directions) — mutation-resistant — not a fragile magnitude bound.
    """
    short = _paired_regret_deltas(_CLEAR_WINNER, horizon=25, seeds=_SEEDS, baseline="pmgreedy")
    far = _paired_regret_deltas(_CLEAR_WINNER, horizon=100, seeds=_SEEDS, baseline="pmgreedy")
    short_lo, short_hi = _bootstrap_ci(short, resamples=_RESAMPLES, seed=9025)
    far_lo, far_hi = _bootstrap_ci(far, resamples=_RESAMPLES, seed=9100)
    print(
        f"[clear-winner trade-off] T=25 mean(pmgreedy-TS)={statistics.fmean(short):.4f} "
        f"CI=[{short_lo:.4f}, {short_hi:.4f}] (TS better) | "
        f"T=100 mean={statistics.fmean(far):.4f} CI=[{far_lo:.4f}, {far_hi:.4f}] (pmgreedy better)"
    )
    assert short_hi < 0.0, f"expected TS to win clear-winner at T=25, CI hi={short_hi:.4f}"
    assert far_lo > 0.0, f"expected pmgreedy to win clear-winner at T=100, CI lo={far_lo:.4f}"
