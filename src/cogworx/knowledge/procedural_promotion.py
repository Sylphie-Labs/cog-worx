"""The procedural-KG promotion gate: a per-edge lower-confidence-bound gate (CANON S1, S9, S12).

A ``(procedure, problem_type)`` edge is PROMOTED when we are confident its true success rate clears
a bar — not merely that its observed mean is high. The gate is the conjunction of two conditions:

  1. a floor on the number of DEDUPED contributions (``success.n_trials`` — never the raw
     ``(:Trial)`` count; that distinction is enforced upstream, see
     :mod:`cogworx.knowledge.procedural_confidence`);
  2. the one-sided lower confidence bound on the Beta posterior clearing a threshold.

The bound penalises small samples: a single success (Beta(2, 1)) has a high mean (0.67) but a low
LCB, so it cannot promote. The Beta(1, 1) prior makes the gate conservative IN EXPECTATION, but the
gate is DISCRETE (it can only fire at integer success counts), so its true one-sided coverage
sawtooths with n and can poke ABOVE the nominal bound at small n.

WHY ``lcb_quantile`` defaults to 0.025, not the tess 0.05 (Pod 2.1 spike, calibration claim (c)):
the ported tess constant (one-sided 5 %) is anti-conservative under our exact discrete gate. At
``quantile=0.05`` the true false-promotion rate at the binding null (θ=0.70) breaches the 5 % bound
at several small n — worst at n=13 (P(promote)=0.0637, +1.4pp over nominal; n=8 also breaches at
0.0576), recurring up to ~n=40. Tightening to ``quantile=0.025`` restores true ≤5 % one-sided
coverage EVERYWHERE (worst cell ~3.0 % at n=30, no breach anywhere n=5..40), at the cost of a few
extra successes to promote — the correct conservative posture for a gate. This was an S12 win: the
breach was latent in a ported constant and surfaced only because the spike checked exact discrete
coverage rather than trusting the source.

The threshold and trial-floor defaults stay the tess values (``LCB_PROMOTION_THRESHOLD = 0.70``,
``MIN_TRIALS = 5``). All three live on a dev-configurable frozen pydantic model rather than module
constants so an adopter can tune the bar per deployment without forking the math (S2: own the loop,
configurable not hardcoded). The math itself is pure stdlib (:mod:`cogworx.knowledge.beta`), no
scipy.

DERIVE-AT-READ / no-demotion (S1, S6): there is NO stored ``promoted`` flag. The gate is evaluated
FRESH at every read from the edge's current posterior. "Monotonic / no demotion" is therefore a
statement about the doctrine, not an invariant this code enforces: mathematically, more trials can
move the LCB either way (a run of failures lowers it). What the design guarantees is that promotion
is never a latched piece of state that could go stale — every reader sees the gate applied to the
live posterior, so there is nothing to demote.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from cogworx.knowledge.beta import lcb
from cogworx.knowledge.procedural_confidence import ProcedureSuccess

__all__ = ["PromotionPolicy"]


class PromotionPolicy(BaseModel):
    """Dev-configurable promotion gate for a ``(procedure, problem_type)`` edge.

    Frozen so a policy is a value, not mutable shared state. The tess defaults reproduce the
    original promotion behaviour; override any field at construction to tune the bar.
    """

    model_config = ConfigDict(frozen=True)

    lcb_threshold: float = Field(default=0.70, gt=0.0, lt=1.0)
    """The lower confidence bound must reach at least this value to promote (tess default 0.70)."""
    min_trials: int = Field(default=5, ge=0)
    """Floor on DEDUPED contributions (``ProcedureSuccess.n_trials``) before promotion is even
    considered (tess default 5). This counts distinct-run contributions, NOT raw trials."""
    lcb_quantile: float = Field(default=0.025, gt=0.0, lt=1.0)
    """The one-sided lower-tail quantile for the confidence bound. Defaults to 0.025 (NOT the tess
    0.05): the discrete gate is anti-conservative at 0.05 — its exact false-promotion rate breaches
    the 5 % bound at small n (worst n=13 → 0.0637). 0.025 restores true ≤5 % one-sided coverage
    everywhere (worst ~3.0 %). See the module docstring and the Pod 2.1 calibration spike."""

    def should_promote(self, success: ProcedureSuccess) -> bool:
        """Return whether this edge's posterior clears the gate.

        ``True`` iff ``success.n_trials >= min_trials`` AND the ``lcb_quantile``-th percentile of
        Beta(``success.alpha``, ``success.beta``) is at least ``lcb_threshold``. The trial floor is
        checked first so a below-floor edge never pays for the (cheap, but non-trivial) LCB
        bisection.
        """
        if success.n_trials < self.min_trials:
            return False
        bound = lcb(success.alpha, success.beta, quantile=self.lcb_quantile)
        return bound >= self.lcb_threshold
