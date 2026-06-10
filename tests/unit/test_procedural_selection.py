"""Selection unit tests (CANON S1, S12).

These pin the PURE selection functions directly (no substrate). Production is
:func:`posterior_mean_select` (deterministic); :func:`thompson_select` is DEFERRED (kept callable,
deterministic in its seed); :func:`sticky_tess_select` is the rejected baseline-for-comparison. The
determinism bet — same input -> identical order regardless of iteration order — is the load-bearing
property for both the parity suite and any replay harness.
"""

from __future__ import annotations

import random

from cogworx.knowledge.procedural_confidence import ProcedureSuccess
from cogworx.knowledge.procedural_promotion import PromotionPolicy
from cogworx.substrate.procedural_kg import ProblemType, Procedure, ScoredProcedure
from cogworx.substrate.procedural_selection import (
    posterior_mean_select,
    sticky_tess_select,
    thompson_select,
)

_PT = ProblemType(id="pt", label="pt")


def _candidate(pid: str, *, alpha: float, beta: float, n_trials: int) -> ScoredProcedure:
    s = alpha + beta
    return ScoredProcedure(
        procedure=Procedure(id=pid, label=pid),
        problem_type=_PT,
        success=ProcedureSuccess(
            alpha=alpha,
            beta=beta,
            success_rate=alpha / s,
            variance=(alpha * beta) / (s * s * (s + 1.0)),
            n_trials=n_trials,
        ),
    )


def test_thompson_same_seed_same_order() -> None:
    cands = [_candidate(f"p{i}", alpha=2.0 + i, beta=1.0, n_trials=i + 1) for i in range(5)]
    a = thompson_select(cands, rng=random.Random(3), limit=5)
    b = thompson_select(cands, rng=random.Random(3), limit=5)
    assert [c.procedure.id for c in a] == [c.procedure.id for c in b]


def test_thompson_order_independent_of_input_iteration_order() -> None:
    """Shuffling the input must NOT change the seeded order (canonical draw assignment)."""
    cands = [_candidate(f"p{i}", alpha=2.0 + i, beta=1.0, n_trials=i + 1) for i in range(6)]
    forward = thompson_select(cands, rng=random.Random(11), limit=6)
    shuffled = list(cands)
    random.Random(99).shuffle(shuffled)
    reverse = thompson_select(shuffled, rng=random.Random(11), limit=6)
    assert [c.procedure.id for c in forward] == [c.procedure.id for c in reverse]


def test_thompson_respects_limit() -> None:
    cands = [_candidate(f"p{i}", alpha=2.0, beta=1.0, n_trials=1) for i in range(10)]
    assert len(thompson_select(cands, rng=random.Random(0), limit=3)) == 3


# A strong edge that clears the q=0.025 gate: Beta(13,1) (12 distinct-run successes), LCB ≈ 0.75 ≥
# 0.70, n=12 ≥ 5. (8/8 = Beta(9,1) no longer promotes since the gate tightened to 0.025 in Pod 2.1.)
def _strong_promotable() -> ScoredProcedure:
    return _candidate("strong", alpha=13.0, beta=1.0, n_trials=12)


def test_thompson_stamps_promotion_fresh() -> None:
    """Incoming promoted is ignored; the gate re-derives it from the posterior."""
    strong = _strong_promotable()  # LCB clears 0.70 at q=0.025, n>=5 -> promoted
    weak = _candidate("weak", alpha=2.0, beta=1.0, n_trials=1)  # n<5 -> not promoted
    # Lie about the incoming flags to prove they're overwritten.
    strong = strong.model_copy(update={"promoted": False})
    weak = weak.model_copy(update={"promoted": True})
    out = {
        c.procedure.id: c for c in thompson_select([strong, weak], rng=random.Random(0), limit=5)
    }
    assert out["strong"].promoted is True
    assert out["weak"].promoted is False


def test_thompson_promoted_only_filters() -> None:
    strong = _strong_promotable()
    weak = _candidate("weak", alpha=2.0, beta=1.0, n_trials=1)
    out = thompson_select([strong, weak], rng=random.Random(0), limit=5, promoted_only=True)
    assert [c.procedure.id for c in out] == ["strong"]


def test_thompson_empty_returns_empty() -> None:
    assert thompson_select([], rng=random.Random(0), limit=5) == ()


def test_thompson_strong_arm_wins_top_slot_on_average() -> None:
    strong = _candidate("strong", alpha=9.0, beta=1.0, n_trials=8)
    weak = _candidate("weak", alpha=1.0, beta=3.0, n_trials=2)
    first_strong = sum(
        thompson_select([strong, weak], rng=random.Random(s), limit=2)[0].procedure.id == "strong"
        for s in range(200)
    )
    assert first_strong > 180


def test_posterior_mean_ranks_by_mean_then_id() -> None:
    """Production selector: posterior mean DESC, then id ASC — NOT n_trials (the sticky pathology).

    The higher-mean edge wins the top slot even though it has FEWER trials; a sticky selector would
    have ranked the more-tried edge first.
    """
    many = _candidate("many", alpha=6.0, beta=2.0, n_trials=7)  # n=7, mean 0.75
    fewer_higher = _candidate("fewer", alpha=4.0, beta=1.0, n_trials=4)  # n=4, mean 0.80
    out = posterior_mean_select([many, fewer_higher], limit=5)
    assert [c.procedure.id for c in out] == ["fewer", "many"]


def test_posterior_mean_tie_breaks_by_id_not_n_trials() -> None:
    """Equal-mean edges (e.g. fresh Beta(1,1)) break by id ASC, not by trial count."""
    a = _candidate("b", alpha=1.0, beta=1.0, n_trials=0)  # mean 0.5
    b = _candidate("a", alpha=3.0, beta=3.0, n_trials=4)  # mean 0.5, more trials
    out = posterior_mean_select([a, b], limit=5)
    assert [c.procedure.id for c in out] == ["a", "b"]


def test_posterior_mean_is_deterministic_no_rng() -> None:
    cands = [_candidate(f"p{i}", alpha=2.0 + i, beta=1.0, n_trials=3) for i in range(5)]
    assert posterior_mean_select(cands, limit=5) == posterior_mean_select(
        list(reversed(cands)), limit=5
    )


def test_posterior_mean_stamps_promotion_fresh() -> None:
    """Incoming promoted is ignored; the gate re-derives it (parity with thompson_select)."""
    strong = _strong_promotable().model_copy(update={"promoted": False})
    weak = _candidate("weak", alpha=2.0, beta=1.0, n_trials=1).model_copy(update={"promoted": True})
    out = {c.procedure.id: c for c in posterior_mean_select([strong, weak], limit=5)}
    assert out["strong"].promoted is True
    assert out["weak"].promoted is False


def test_posterior_mean_promoted_only_filters() -> None:
    strong = _strong_promotable()
    weak = _candidate("weak", alpha=2.0, beta=1.0, n_trials=1)
    out = posterior_mean_select([strong, weak], limit=5, promoted_only=True)
    assert [c.procedure.id for c in out] == ["strong"]


def test_sticky_tess_baseline_ranks_by_n_trials_then_rate() -> None:
    """The rejected baseline-for-comparison: n_trials DESC, then success_rate DESC, then id ASC.

    Kept only so the spike's strawman finding is reproducible. The more-tried edge wins even with a
    LOWER mean — the sticky pathology posterior-mean greedy avoids.
    """
    many = _candidate("many", alpha=6.0, beta=2.0, n_trials=7)  # n=7, rate 0.75
    fewer_higher = _candidate("fewer", alpha=4.0, beta=1.0, n_trials=4)  # n=4, rate 0.80
    out = sticky_tess_select([fewer_higher, many], limit=5)
    assert [c.procedure.id for c in out] == ["many", "fewer"]


def test_sticky_tess_is_deterministic_no_rng() -> None:
    cands = [_candidate(f"p{i}", alpha=2.0, beta=1.0, n_trials=3) for i in range(5)]
    assert sticky_tess_select(cands, limit=5) == sticky_tess_select(list(reversed(cands)), limit=5)


def test_custom_policy_changes_promotion() -> None:
    """A lenient custom policy promotes an edge the DEFAULT gate rejects (n=3 < default floor 5)."""
    success = _candidate("p", alpha=4.0, beta=1.0, n_trials=3)  # Beta(4,1), LCB@0.025 ≈ 0.398
    # Default gate rejects (min_trials=5 > 3); a lenient floor + lower threshold promotes.
    assert thompson_select([success], rng=random.Random(0), limit=1)[0].promoted is False
    lenient = PromotionPolicy(lcb_threshold=0.35, min_trials=3)
    out = thompson_select([success], rng=random.Random(0), limit=1, policy=lenient)
    assert out[0].promoted is True
