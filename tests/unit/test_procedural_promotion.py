"""PromotionPolicy gate tests (CANON S1, S9, S12).

Ports tess's ``should_promote`` behaviour onto the dev-configurable :class:`PromotionPolicy` over a
:class:`ProcedureSuccess`. The gate is the conjunction: ``n_trials >= min_trials`` AND ``LCB >=
lcb_threshold``. The trial floor reads the DEDUPED count (``ProcedureSuccess.n_trials``) by design -
that is the only count the success summary surfaces.
"""

from __future__ import annotations

import pytest

from cogworx.knowledge.procedural_confidence import ProcedureSuccess
from cogworx.knowledge.procedural_promotion import PromotionPolicy


def _success(*, alpha: float, beta: float, n_trials: int) -> ProcedureSuccess:
    """Build a ProcedureSuccess directly (the gate only reads alpha/beta/n_trials)."""
    s = alpha + beta
    return ProcedureSuccess(
        alpha=alpha,
        beta=beta,
        success_rate=alpha / s,
        variance=(alpha * beta) / (s * s * (s + 1.0)),
        n_trials=n_trials,
    )


def test_defaults_threshold_and_floor_match_tess() -> None:
    """Threshold and floor are the tess values (>= 0.70, n >= 5)."""
    policy = PromotionPolicy()
    assert policy.lcb_threshold == 0.70
    assert policy.min_trials == 5


def test_default_quantile_is_tightened_to_0_025_not_tess_0_05() -> None:
    """The shipped quantile DIVERGES from the tess 0.05 (Pod 2.1): tightened to 0.025 because the
    discrete gate is anti-conservative at 0.05 (breaches the 5% bound, worst n=13 → 0.0637)."""
    assert PromotionPolicy().lcb_quantile == 0.025


def test_blocks_below_min_trials() -> None:
    """Even with extreme alpha, n_trials < 5 blocks promotion (the 'enough evidence' floor)."""
    policy = PromotionPolicy()
    # Beta(100, 1) has a very high LCB, but n_trials=4 is below the floor.
    assert not policy.should_promote(_success(alpha=100.0, beta=1.0, n_trials=4))


def test_blocks_below_lcb_threshold() -> None:
    """n_trials high enough but LCB too low -> no promotion."""
    policy = PromotionPolicy()
    # Beta(5, 5) has LCB ~= 0.21, far below 0.70.
    assert not policy.should_promote(_success(alpha=5.0, beta=5.0, n_trials=8))


def test_passes_when_both_satisfied() -> None:
    """Beta(15, 1) gives LCB ~= 0.82 and n_trials=14 satisfies both gates."""
    policy = PromotionPolicy()
    assert policy.should_promote(_success(alpha=15.0, beta=1.0, n_trials=14))


def test_strict_at_min_trials_boundary() -> None:
    """The floor is ``>= min_trials``, not ``>``: exactly 5 deduped trials can promote."""
    policy = PromotionPolicy()
    # Beta(11, 1) is 10 successes from flat prior -> LCB ~= 0.762, passes.
    assert policy.should_promote(_success(alpha=11.0, beta=1.0, n_trials=5))
    assert not policy.should_promote(_success(alpha=11.0, beta=1.0, n_trials=4))


def test_floor_reads_deduped_n_trials() -> None:
    """The floor reads ProcedureSuccess.n_trials (deduped) - NOT raw trial count.

    A single cyclic run with one deduped contribution must NOT promote, no matter the alpha implied
    by a high raw count, because n_trials is the deduped value carried on the summary.
    """
    policy = PromotionPolicy()
    # alpha is high (as if many raw successes) but the deduped contribution count is 1 < 5.
    assert not policy.should_promote(_success(alpha=50.0, beta=1.0, n_trials=1))


def test_is_configurable() -> None:
    """A dev can tune the bar: lowering the threshold/floor promotes an edge the default rejects."""
    success = _success(alpha=4.0, beta=1.0, n_trials=3)  # Beta(4,1): LCB@0.025 ~= 0.398, n=3
    assert not PromotionPolicy().should_promote(success)  # default floor 5 > 3 → rejected
    lenient = PromotionPolicy(lcb_threshold=0.35, min_trials=3)
    assert lenient.should_promote(success)


def test_is_frozen() -> None:
    """PromotionPolicy is an immutable value object."""
    import pydantic

    policy = PromotionPolicy()
    with pytest.raises(pydantic.ValidationError):
        policy.lcb_threshold = 0.5  # type: ignore[misc]


@pytest.mark.parametrize("bad", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_threshold(bad: float) -> None:
    """lcb_threshold and lcb_quantile must be in (0, 1) - a system-boundary validation."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        PromotionPolicy(lcb_threshold=bad)
    with pytest.raises(pydantic.ValidationError):
        PromotionPolicy(lcb_quantile=bad)
