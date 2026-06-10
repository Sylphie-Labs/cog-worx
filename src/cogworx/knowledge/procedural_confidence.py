"""Beta-posterior success derivation for procedural-KG edges (CANON S1, S6).

The success rate of a ``(procedure, problem_type)`` edge is NEVER stored — it is re-derived at read
from the immutable ``(:Trial)`` event source, exactly as entity-KG confidence is derived from
evidence (:mod:`cogworx.knowledge.confidence`). This is the single computation path so every reader
(promotion gate, Thompson sampler) sees the same posterior.

GRAIN (eval-stats sign-off, 2026-06-09): trials dedup at ``source_id = run_id`` per polarity. One
cyclic run that applies a procedure 50 times is ONE contribution, not 50 (pseudo-replication /
correlated trials inflate the posterior and the promotion floor). This module reuses
:func:`cogworx.knowledge.confidence.claim_confidence` VERBATIM by mapping each trial outcome to a
unit-weight :class:`~cogworx.knowledge.evidence.EvidenceEvent` whose ``source_id`` is the run id and
whose polarity is the stamped outcome — so the existing ``(source_id, polarity)`` max-weight dedup
IS the run-level dedup, and ``n_evidence`` IS the deduped contribution count.

CRITICAL (promotion floor): the ``n >= floor`` promotion gate MUST count DEDUPED contributions
(``len(best)``), NOT raw ``(:Trial)`` nodes. This module makes that impossible to get wrong: it
exposes only :attr:`ProcedureSuccess.n_trials`, which is the deduped count carried straight from
``ClaimConfidence.n_evidence``. There is no raw-count surface here for a caller to reach for.

This module is pure math — it depends ONLY on :mod:`cogworx.knowledge`, never on the substrate. The
read surface maps each :class:`~cogworx.substrate.procedural_kg.Trial` onto a ``TrialOutcome``
``(run_id, success)`` pair, keeping the dependency one-directional (substrate -> knowledge).
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict

from cogworx.knowledge.confidence import ClaimConfidence, claim_confidence
from cogworx.knowledge.evidence import EvidenceEvent

__all__ = [
    "TRIAL_BASE_WEIGHT",
    "ProcedureSuccess",
    "TrialOutcome",
    "procedure_success",
]

# Each trial is one binary success/failure observation: unit weight, full authority. A trial is a
# stamped structural fact (we ran the procedure and read the committed outcome), so it carries no
# graded trust the way model-derived evidence does — every distinct run contributes exactly 1.0 to
# alpha (success) or beta (failure). Beta(1,1) prior is inherited from confidence.py, so a never-run
# edge sits at success rate 0.5 with maximum variance ("we don't know").
TRIAL_BASE_WEIGHT: float = 1.0

# recorded_at is irrelevant to the Beta math (it drives neither dedup nor weight); the read surface
# carries the real Trial.occurred_at on the (:Trial) node. A fixed sentinel keeps the mapping pure.
_RECORDED_AT_SENTINEL = datetime(1970, 1, 1, tzinfo=UTC)


class TrialOutcome(NamedTuple):
    """The minimal projection of a Trial the success derivation consumes: ``(run_id, success)``.

    The read surface maps each ``(:Trial)`` node onto one of these. Only the run id (the dedup
    grain) and the boolean outcome reach the math — the timestamp, procedure, and problem-type live
    on the node and are filtered by the caller before derivation.
    """

    run_id: str
    success: bool


class ProcedureSuccess(BaseModel):
    """Derived Beta-posterior success summary for one ``(procedure, problem_type)`` edge.

    Never stored — always re-derived from the edge's immutable trials via :func:`procedure_success`.
    A thin, intentional rename of :class:`ClaimConfidence` into procedural-KG vocabulary so the
    promotion gate and Thompson sampler read ``success_rate``/``n_trials``, not claim wording.
    """

    model_config = ConfigDict(frozen=True)

    alpha: float
    """Beta alpha = prior + Σ deduped success weights — the Thompson-sampling parameter."""
    beta: float
    """Beta beta = prior + Σ deduped failure weights — the Thompson-sampling parameter."""
    success_rate: float
    """Posterior mean: alpha / (alpha + beta) — the exploitation score."""
    variance: float
    """alpha·beta / ((alpha+beta)²·(alpha+beta+1)) — drives the LCB promotion gate."""
    n_trials: int
    """DEDUPED contribution count (distinct run ids per polarity). This is the value the promotion
    floor (``n >= floor``) MUST use — NOT the raw ``(:Trial)`` count (see module docstring)."""


def procedure_success(trials: Iterable[TrialOutcome]) -> ProcedureSuccess:
    """Derive the Beta-posterior success rate for an edge from its trial outcomes.

    Each trial maps to a unit-weight evidence event keyed by ``source_id = run_id`` and polarity =
    ``"+"`` on success / ``"-"`` on failure, then :func:`claim_confidence` is reused verbatim.
    Run-level dedup (one run = one contribution per polarity) therefore falls out of the existing
    ``(source_id, polarity)`` max-weight dedup at no extra cost.

    Empty trials → prior-only: success_rate 0.5, n_trials 0.
    """
    events = [
        EvidenceEvent(
            id=f"{t.run_id}:{'+' if t.success else '-'}",
            type="tool_proof",
            polarity="+" if t.success else "-",
            source_id=t.run_id,
            source_authority=1.0,
            base_weight=TRIAL_BASE_WEIGHT,
            recorded_at=_RECORDED_AT_SENTINEL,
        )
        for t in trials
    ]
    derived: ClaimConfidence = claim_confidence(events)
    return ProcedureSuccess(
        alpha=derived.alpha,
        beta=derived.beta,
        success_rate=derived.confidence,
        variance=derived.variance,
        n_trials=derived.n_evidence,
    )
