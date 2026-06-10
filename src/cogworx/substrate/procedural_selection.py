"""Procedural-KG read-surface selection over scored procedures (CANON S1, S4, S12).

This is the procedural-KG candidate ranking. Three selectors live here, with a clear production
default after the Pod 2.1 spike:

- :func:`posterior_mean_select` — THE PRODUCTION DEFAULT. Ranks by the Beta posterior MEAN
  (``success_rate``) descending. The Beta(1, 1) prior bakes in optimism (an unseen edge sits at
  0.5), so the first few selections over a fresh problem type get free exploration before
  exploitation takes over — no RNG required. Fully deterministic; a stable tie-break by
  ``procedure.id`` makes the order byte-identical across the adapter and the double.

- :func:`thompson_select` — DEFERRED / UNPROVEN. A pure function with an INJECTED, seeded RNG. Fully
  working and callable, but NO LONGER the default: in the Pod 2.1 spike it did not beat a FAIR
  posterior-mean-greedy baseline in the near-tie regime (it tied or lost; it only "won" against the
  pathologically sticky tess baseline, a strawman). It stays in-tree for a future spike against a
  fair baseline at longer horizons / higher K. A caller opts in explicitly by passing an ``rng``.

- :func:`sticky_tess_select` — the ported tess exploitation-only ranking (``n_trials DESC,
  posterior-mean DESC``). NOT a production selector: it is the pathologically sticky baseline the
  spike used as a (rejected) comparator. Kept only as a documented baseline-for-comparison so the
  spike's strawman finding is reproducible; do not wire it into a read path.

DETERMINISM (CANON Test Kit): the adapter and the in-memory double both call the SAME selector, so
given the same posteriors they return byte-identical orderings — that is what lets the parity suite
hold the two implementations to the same behaviour. The default path needs no RNG; the deferred
Thompson path is deterministic in its injected seed.

LAYERING (S1 purity): this module lives in ``substrate`` because it operates on the substrate domain
type :class:`ScoredProcedure`. The ``knowledge`` layer is kept strictly model-free and
substrate-free (an import-purity guard asserts importing ``cogworx.knowledge`` never pulls in
``cogworx.model``); the pure gate math (:class:`PromotionPolicy`, the Beta functions) lives there,
the selection that touches substrate types lives here. Dependency is one-directional:
substrate -> knowledge.
"""

from __future__ import annotations

import random
from collections.abc import Iterable

from cogworx.knowledge.procedural_promotion import PromotionPolicy
from cogworx.substrate.procedural_kg import ScoredProcedure

__all__ = [
    "DEFAULT_PROMOTION_POLICY",
    "posterior_mean_select",
    "sticky_tess_select",
    "thompson_select",
]

DEFAULT_PROMOTION_POLICY = PromotionPolicy()
"""The default gate (LCB 2.5 % >= 0.70 AND n_trials >= 5). Shared default so the adapter, the
double, and the spike all gate identically unless a caller overrides it."""


def _stamp_promotion(
    candidates: Iterable[ScoredProcedure],
    *,
    policy: PromotionPolicy,
    promoted_only: bool,
) -> list[ScoredProcedure]:
    """Re-stamp each candidate's ``promoted`` from the live posterior, optionally filtering to it.

    Promotion is DERIVED here (S1): the incoming ``ScoredProcedure.promoted`` is ignored and
    replaced by ``policy.should_promote(c.success)`` so the gate is fresh against the current
    posterior.
    """
    stamped = [
        c.model_copy(update={"promoted": policy.should_promote(c.success)}) for c in candidates
    ]
    if promoted_only:
        return [c for c in stamped if c.promoted]
    return stamped


def posterior_mean_select(
    candidates: Iterable[ScoredProcedure],
    *,
    limit: int,
    promoted_only: bool = False,
    policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
) -> tuple[ScoredProcedure, ...]:
    """PRODUCTION read surface: rank by Beta posterior MEAN DESC; return the top ``limit``.

    No RNG — fully deterministic. The score is ``success.success_rate`` (the Beta posterior mean,
    ``alpha / (alpha + beta)``). The Beta(1, 1) prior gives an unseen edge a mean of 0.5, so a fresh
    problem type's first few selections explore optimistically before the data pulls the means apart
    — "free" first-K exploration without sampling. Ties (equal means, e.g. all-fresh edges) break by
    ``procedure.id`` ASC for a total, byte-stable order, NOT by ``n_trials`` (the sticky pathology
    that locks onto an early-pulled arm). Each returned candidate has ``promoted`` stamped fresh by
    ``policy``; ``promoted_only`` keeps only promoted edges.
    """
    stamped = _stamp_promotion(candidates, policy=policy, promoted_only=promoted_only)
    stamped.sort(key=lambda c: (-c.success.success_rate, c.procedure.id))
    return tuple(stamped[:limit])


def thompson_select(
    candidates: Iterable[ScoredProcedure],
    *,
    rng: random.Random,
    limit: int,
    promoted_only: bool = False,
    policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
) -> tuple[ScoredProcedure, ...]:
    """DEFERRED / UNPROVEN selector: rank by a Thompson draw from each edge's Beta posterior.

    Fully working and deterministic in ``rng`` (same posteriors + same seeded ``rng`` -> identical
    order, regardless of input iteration order), but NOT the production default. In the Pod 2.1
    spike (calibration claim (b)) Thompson did NOT robustly beat a FAIR posterior-mean-greedy
    baseline in the hard near-tie regime — it tied or lost (it only beat the pathologically sticky
    tess baseline, which is a strawman). Per Jim's decision it is DEFERRED: kept in-tree, callable
    when a caller explicitly passes an ``rng``, pending a future spike against a fair baseline at
    longer horizons / higher K. :func:`posterior_mean_select` is what ships.

    The algorithm: draws are assigned in a canonical order (procedure id ASC) BEFORE sampling, so
    the i-th RNG draw always lands on the same edge whether candidates arrive from the Neo4j adapter
    (ORDER BY p.id) or the double (dict/set iteration) — that is what makes the two implementations
    byte-identical under one seed. For each candidate draw ``theta ~ Beta(alpha, beta)``, then rank
    DESC by the sample; degenerate continuous ties break by procedure id ASC. Each returned
    candidate has ``promoted`` stamped fresh by ``policy``; ``promoted_only`` keeps only promoted
    edges. Draws are taken over all candidates before truncation so consumption is ``limit``-stable.
    """
    stamped = _stamp_promotion(candidates, policy=policy, promoted_only=promoted_only)
    stamped.sort(key=lambda c: c.procedure.id)
    drawn = [(rng.betavariate(c.success.alpha, c.success.beta), c) for c in stamped]
    drawn.sort(key=lambda pair: (-pair[0], pair[1].procedure.id))
    return tuple(c for _, c in drawn[:limit])


def sticky_tess_select(
    candidates: Iterable[ScoredProcedure],
    *,
    limit: int,
    promoted_only: bool = False,
    policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
) -> tuple[ScoredProcedure, ...]:
    """Ported tess exploitation-only ranking (``n_trials DESC, posterior-mean DESC``) — NOT for
    production.

    No RNG — fully deterministic. This ranking is pathologically STICKY: it prefers the most-tried
    edge, so an early run of trials can lock it in with no further exploration. That is the strawman
    baseline the Pod 2.1 spike used (and rejected) — Thompson only "won" against THIS, not against a
    fair posterior-mean greedy. Kept ONLY as a documented baseline-for-comparison so the spike's
    finding is reproducible; the production read surface is :func:`posterior_mean_select`.
    ``promoted`` is stamped fresh exactly as in the other selectors.
    """
    stamped = _stamp_promotion(candidates, policy=policy, promoted_only=promoted_only)
    stamped.sort(key=lambda c: (-c.success.n_trials, -c.success.success_rate, c.procedure.id))
    return tuple(stamped[:limit])
