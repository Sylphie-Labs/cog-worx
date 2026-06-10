"""Beta-distribution math parity + boundary tests (CANON S12).

The reference values are ``scipy.stats.beta.cdf`` / ``scipy.stats.beta.ppf`` outputs precomputed and
pinned here so scipy is NOT a runtime or test dependency (S2: pure-stdlib math, small dep surface).
Ported verbatim from ``tess/tests/test_stats.py``. Recomputing these when
:mod:`cogworx.knowledge.beta`
changes is the deliberate audit point - if a parity value drifts, the math changed and must be
checked
against scipy by hand.
"""

from __future__ import annotations

import math

import pytest

from cogworx.knowledge.beta import (
    beta_inverse_cdf,
    lcb,
    regularized_incomplete_beta,
)


# Reference values from scipy.stats.beta.cdf(x, a, b), precomputed. Beta(5, 2) at 0.3 also verified
# by hand against the closed-form binomial expansion of the regularized incomplete beta for integer
# a, b: sum_{i=a}^{a+b-1} C(a+b-1, i) x^i (1-x)^(a+b-1-i) = 0.010935.
@pytest.mark.parametrize(
    ("a", "b", "x", "expected"),
    [
        (2.0, 2.0, 0.5, 0.5),
        (5.0, 2.0, 0.3, 0.010935),
        (1.0, 1.0, 0.9, 0.9),
        (10.0, 1.0, 0.1, 1e-10),
        (1.0, 1.0, 0.0, 0.0),
        (1.0, 1.0, 1.0, 1.0),
    ],
)
def test_regularized_incomplete_beta_matches_scipy_reference(
    a: float, b: float, x: float, expected: float
) -> None:
    actual = regularized_incomplete_beta(a, b, x)
    assert math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)


def test_regularized_incomplete_beta_rejects_x_out_of_range() -> None:
    with pytest.raises(ValueError, match=r"x must be in"):
        regularized_incomplete_beta(2.0, 2.0, 1.5)
    with pytest.raises(ValueError, match=r"x must be in"):
        regularized_incomplete_beta(2.0, 2.0, -0.1)


def test_regularized_incomplete_beta_rejects_non_positive_params() -> None:
    with pytest.raises(ValueError, match=r"a and b must be > 0"):
        regularized_incomplete_beta(0.0, 2.0, 0.5)
    with pytest.raises(ValueError, match=r"a and b must be > 0"):
        regularized_incomplete_beta(2.0, -1.0, 0.5)


# Reference values from scipy.stats.beta.ppf(p, a, b):
#   scipy.stats.beta.ppf(0.05, 1, 1)   ~= 0.05
#   scipy.stats.beta.ppf(0.05, 2, 1)   ~= 0.22360679774997896
#   scipy.stats.beta.ppf(0.05, 6, 1)   ~= 0.6069673676298591
#   scipy.stats.beta.ppf(0.05, 10, 1)  ~= 0.7411344491069014
#   scipy.stats.beta.ppf(0.5, 5, 5)    ~= 0.5
@pytest.mark.parametrize(
    ("p", "a", "b", "expected"),
    [
        (0.05, 1.0, 1.0, 0.05),
        (0.05, 2.0, 1.0, 0.22360679774997896),
        (0.05, 6.0, 1.0, 0.6069673676298591),
        (0.05, 10.0, 1.0, 0.7411344491069014),
        (0.5, 5.0, 5.0, 0.5),
    ],
)
def test_beta_inverse_cdf_matches_scipy_reference(
    p: float, a: float, b: float, expected: float
) -> None:
    actual = beta_inverse_cdf(p, a, b)
    assert math.isclose(actual, expected, rel_tol=1e-5, abs_tol=1e-7)


def test_beta_inverse_cdf_rejects_p_at_boundary() -> None:
    with pytest.raises(ValueError, match=r"p must be in"):
        beta_inverse_cdf(0.0, 2.0, 2.0)
    with pytest.raises(ValueError, match=r"p must be in"):
        beta_inverse_cdf(1.0, 2.0, 2.0)


def test_lcb_flat_prior_returns_quantile() -> None:
    """Beta(1, 1) is uniform on [0, 1]; its q-th percentile is q. Anchor test - must never drift."""
    assert math.isclose(lcb(1.0, 1.0, quantile=0.05), 0.05, rel_tol=1e-3)


def test_lcb_grows_with_alpha() -> None:
    """More observed successes (higher alpha) -> higher confidence the procedure is durable ->
    higher
    LCB."""
    weak = lcb(2.0, 1.0, quantile=0.05)
    moderate = lcb(6.0, 1.0, quantile=0.05)
    strong = lcb(15.0, 1.0, quantile=0.05)
    assert weak < moderate < strong


def test_lcb_is_inverse_cdf_at_quantile() -> None:
    """lcb(a, b, quantile=q) == beta_inverse_cdf(q, a, b) (it is a thin named wrapper)."""
    assert lcb(6.0, 1.0, quantile=0.05) == beta_inverse_cdf(0.05, 6.0, 1.0)
