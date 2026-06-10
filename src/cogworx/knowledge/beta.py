"""Pure-stdlib Beta-distribution math for the procedural-KG promotion gate (CANON S2, S12).

The promotion gate needs the inverse CDF of a Beta posterior (the lower confidence bound). The only
analytic route is the regularized incomplete beta function and its inverse — neither is in the
stdlib ``math`` module. Rather than take a ~80 MB scipy dependency for one call per promotion check
(S2: OSS is reference, not dependency; every dep is the adopter's burden), this module computes both
in pure Python: the regularized incomplete beta via a continued-fraction expansion (Numerical
Recipes §6.4) and its inverse via bisection over that CDF.

Ported from ``tess/tess/stats.py`` (the ``betacf`` / ``regularized_incomplete_beta`` /
``beta_inverse_cdf`` / ``lcb_5pct`` lineage), generalised so the LCB quantile is a parameter rather
than a hardcoded 5 %. Parity with ``scipy.stats.beta`` is pinned by precomputed reference values in
``tests/unit/test_beta.py`` — that test IS the audit point; recompute it when this math changes.

This module is pure computation: no model calls, no substrate I/O, no other cogworx imports.
"""

from __future__ import annotations

import math

__all__ = [
    "beta_inverse_cdf",
    "lcb",
    "regularized_incomplete_beta",
]


def _betacf(a: float, b: float, x: float, max_iter: int = 200, eps: float = 3e-12) -> float:
    """Continued-fraction expansion for the incomplete beta function (Numerical Recipes §6.4).

    Converges fast for ``0 <= x <= (a+1)/(a+b+2)``; outside that range the caller uses the symmetry
    relation ``I_x(a, b) = 1 - I_{1-x}(b, a)`` instead.
    """
    qab = a + b
    qap = a + 1.0
    qam = a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30:
        d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30:
            d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30:
            c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            return h
    raise ValueError(f"betacf failed to converge after {max_iter} iterations (a={a}, b={b}, x={x})")


def regularized_incomplete_beta(a: float, b: float, x: float) -> float:
    """``I_x(a, b)`` — the regularized incomplete beta function, the CDF of Beta(a, b) at ``x``.

    Returns a value in ``[0, 1]``. Raises ``ValueError`` for ``x`` outside ``[0, 1]`` or
    non-positive shape parameters (a system-boundary validation: callers pass derived alpha/beta).
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError(f"x must be in [0, 1], got {x}")
    if a <= 0 or b <= 0:
        raise ValueError(f"a and b must be > 0, got a={a}, b={b}")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_bt = (
        math.lgamma(a + b)
        - math.lgamma(a)
        - math.lgamma(b)
        + a * math.log(x)
        + b * math.log(1.0 - x)
    )
    bt = math.exp(log_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def beta_inverse_cdf(p: float, a: float, b: float, eps: float = 1e-8) -> float:
    """Inverse CDF of Beta(a, b) at probability ``p``, via bisection on the regularized incomplete
    beta. Returns ``x`` such that ``I_x(a, b) ~= p``.

    Beta densities are unimodal and their CDF is strictly increasing on ``(0, 1)``, so bisection
    converges without surprises. Raises ``ValueError`` for ``p`` at or outside the open interval.
    """
    if not (0.0 < p < 1.0):
        raise ValueError(f"p must be in (0, 1), got {p}")
    lo, hi = 0.0, 1.0
    while hi - lo > eps:
        mid = (lo + hi) / 2.0
        if regularized_incomplete_beta(a, b, mid) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def lcb(alpha: float, beta: float, *, quantile: float) -> float:
    """One-sided lower confidence bound: the ``quantile``-th percentile of Beta(alpha, beta).

    With ``quantile=0.05`` this is "we are 95 % confident the true success rate is at least this
    value". A Beta(1, 1) (flat prior, no evidence) returns ``quantile`` exactly (the percentile of
    the uniform distribution); accumulating successes pushes alpha up and the bound rises.
    """
    return beta_inverse_cdf(quantile, alpha, beta)
