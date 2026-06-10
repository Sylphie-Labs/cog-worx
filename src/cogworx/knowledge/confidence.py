"""Beta-posterior confidence derivation for entity-KG claims (CANON S1, S5).

Confidence is NEVER stored in the substrate — it is always re-derived at read time from the
immutable evidence list attached to each claim. This module is the single computation path so
every reader sees the same posteriors. Ported from tess.stats.claim_confidence.

Algorithm (§2.2 of the entity-KG redesign):
  1. Dedup by (source_id, polarity) — keep MAX weight per pair. Prevents the same source from
     inflating confidence by appearing multiple times across runs.
  2. alpha = CLAIM_PRIOR_ALPHA + Σ deduped "+" weights
     beta  = CLAIM_PRIOR_BETA  + Σ deduped "-" weights
  3. confidence = alpha / (alpha + beta)   (posterior mean — "how true")
     variance   = alpha*beta / ((alpha+beta)²·(alpha+beta+1))  ("how sure of the mean")
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict

from cogworx.knowledge.evidence import EvidenceEvent

__all__ = [
    "CLAIM_PRIOR_ALPHA",
    "CLAIM_PRIOR_BETA",
    "ClaimConfidence",
    "claim_confidence",
]

# Skeptical uniform prior: both start at 1.0 so a new claim with no evidence lands exactly at the
# Beta(1,1) = Uniform(0,1) prior mean of 0.5 with maximum variance. Raising CLAIM_PRIOR_BETA (not
# CLAIM_PRIOR_ALPHA) makes the prior more skeptical. See calibration commentary in tess.stats.
CLAIM_PRIOR_ALPHA: float = 1.0
CLAIM_PRIOR_BETA: float = 1.0


class ClaimConfidence(BaseModel):
    """Derived Beta-posterior summary for one claim.

    Never stored directly — always re-derived from the claim's immutable evidence list via
    :func:`claim_confidence`.

    Variance interpretation:
      - alpha=1, beta=1 → confidence≈0.5, HIGH variance → "we don't know"
      - alpha=5, beta=5 → confidence≈0.5, LOW  variance → "genuinely contested"
    """

    model_config = ConfigDict(frozen=True)

    alpha: float
    beta: float
    confidence: float
    """Posterior mean: alpha / (alpha + beta) — 'how true'."""
    variance: float
    """alpha·beta / ((alpha+beta)²·(alpha+beta+1)) — 'how certain of the mean'."""
    n_evidence: int
    """Count of deduped (source_id, polarity) contributions used to compute this posterior."""


def claim_confidence(events: Iterable[EvidenceEvent]) -> ClaimConfidence:
    """Derive the Beta-posterior confidence for a claim from its evidence events.

    Deduplication by (source_id, polarity) keeps only the MAX effective weight per pair, where
    weight(e) = e.base_weight * e.source_authority. A source CAN contribute to BOTH polarities
    (dedup is per (source_id, polarity), not global).

    Empty events → prior-only: ClaimConfidence at (CLAIM_PRIOR_ALPHA, CLAIM_PRIOR_BETA).
    """
    items = list(events)

    # Dedup by (source_id, polarity) — keep max effective weight per pair.
    # Effective weight = base_weight * source_authority (both stamped at event creation).
    best: dict[tuple[str, str], float] = {}
    for ev in items:
        key = (ev.source_id, ev.polarity)
        w = ev.base_weight * ev.source_authority
        if key not in best or w > best[key]:
            best[key] = w

    # Accumulate deduped weights into alpha / beta.
    alpha = CLAIM_PRIOR_ALPHA
    beta = CLAIM_PRIOR_BETA
    for (_, polarity), w in best.items():
        if polarity == "+":
            alpha += w
        else:
            beta += w

    s = alpha + beta
    confidence = alpha / s
    variance = (alpha * beta) / (s * s * (s + 1.0))

    return ClaimConfidence(
        alpha=alpha,
        beta=beta,
        confidence=confidence,
        variance=variance,
        n_evidence=len(best),
    )
