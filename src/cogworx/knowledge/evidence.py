"""Immutable evidence events for entity-KG claims (CANON S1, S5).

Evidence events ACCUMULATE on a claim (CREATE, never MERGE). Confidence is DERIVED at read via
the Beta posterior in :mod:`cogworx.knowledge.confidence` — it is NEVER stored. ``base_weight`` is
stamped at event creation so future recalibrations of EVIDENCE_BASE_WEIGHTS never rewrite history.

Polarity semantics: "+" supports the claim (contributes to Beta alpha); "-" is FIRST-HAND
disconfirmation only (contributes to Beta beta). Mere absence of evidence is not "-".

Ported from tess.stats.EVIDENCE_BASE_WEIGHTS and tess.kg (EvidenceItem stamping convention).
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "EVIDENCE_BASE_WEIGHTS",
    "EvidenceEvent",
    "EvidenceType",
    "Polarity",
    "make_evidence",
]

EvidenceType = Literal[
    "tool_proof", "refutation", "antithesis_survival", "corroboration", "recall", "extraction"
]
# "+" contributes to alpha (supports the claim); "-" is first-hand disconfirmation only (S1 —
# only our own tool execution or direct contradiction qualifies; model self-assessment does not).
Polarity = Literal["+", "-"]

# Base weight by evidence type. Stamped at write time; recalibration bumps these values but leaves
# historical events untouched (they carry their original base_weight). Calibration rationale:
#   tool_proof     — we ran the tool ourselves; first-hand observation. Highest trust.
#   refutation     — first-hand disconfirmation; equally decisive (same weight as tool_proof).
#   antithesis_survival — adversarial team tried to break it and failed; strong but indirect.
#   corroboration  — independent source agrees; less decisive than first-hand.
#   recall         — recalled from memory/prior context; weakest signal, easy to over-count.
#   extraction     — model-extracted inference from a first-hand user statement; second-hand.
# Source: tess.stats.EVIDENCE_BASE_WEIGHTS (calibrated 2026-05 against reliability-curve eval).
EVIDENCE_BASE_WEIGHTS: dict[EvidenceType, float] = {
    "tool_proof": 3.0,
    "refutation": 3.0,
    "antithesis_survival": 1.5,
    "corroboration": 1.0,
    "recall": 0.5,
    "extraction": 1.0,
}


class EvidenceEvent(BaseModel):
    """An immutable evidence event attached to a (:Claim) node.

    Events are CREATED once and never updated. The same source writing the same evidence twice
    produces two events — dedup by (source_id, polarity) is the READ path's responsibility (see
    :func:`cogworx.knowledge.confidence.claim_confidence`).

    ``base_weight`` is stamped at creation from EVIDENCE_BASE_WEIGHTS so old records are immune to
    future weight-table changes.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    type: EvidenceType
    polarity: Polarity
    source_id: str
    """Stable identity for the evidence source. Used for independence dedup at READ — the same
    source_id within the same polarity contributes only its maximum weight.

    CONTRACT (S9): source_id MUST be assigned by framework code (run id, tool id, document URL,
    user id) and NEVER taken from model output. A model-chosen source_id is an S9 violation —
    self-reported identity gating confidence allows a model to mint arbitrary sockpuppet sources
    and inflate its own confidence. Structural source-registry enforcement is deferred to the
    ops/coherence pod; this contract is the interim discipline.
    """
    source_authority: float = Field(ge=0.0, le=1.0)
    """[0, 1]. Scaled by the write path from the caller's authority estimate; not re-derived."""
    base_weight: float = Field(gt=0.0, allow_inf_nan=False)
    """Stamped from EVIDENCE_BASE_WEIGHTS at creation — never re-derived so historical records are
    immune to weight-table recalibrations.

    Must be strictly positive and finite: confidence is a control signal (Beta posterior mean =
    alpha/(alpha+beta)); a non-positive or infinite weight would push the posterior outside (0, 1)
    or produce NaN, which breaks every downstream consumer. The type enforces this invariant at
    construction time so invalid weights can never be persisted or replayed.
    """
    run_id: str | None = None
    stage: str | None = None
    recorded_at: datetime


def make_evidence(
    *,
    type: EvidenceType,
    polarity: Polarity,
    source_id: str,
    source_authority: float,
    recorded_at: datetime,
    run_id: str | None = None,
    stage: str | None = None,
    event_id: str | None = None,
) -> EvidenceEvent:
    """Mint a new :class:`EvidenceEvent`, stamping ``base_weight`` from EVIDENCE_BASE_WEIGHTS.

    ``event_id`` defaults to ``uuid4().hex``; pass an explicit value only in tests that need
    deterministic ids.

    ``source_id`` CONTRACT (S9): must be assigned by framework code (run id, tool id, document URL,
    user id) — NEVER taken from model output. A model-chosen source_id is an S9 violation. See
    :attr:`EvidenceEvent.source_id` for the full contract. Structural enforcement is deferred to
    the ops/coherence pod.
    """
    return EvidenceEvent(
        id=event_id if event_id is not None else uuid.uuid4().hex,
        type=type,
        polarity=polarity,
        source_id=source_id,
        source_authority=source_authority,
        base_weight=EVIDENCE_BASE_WEIGHTS[type],
        run_id=run_id,
        stage=stage,
        recorded_at=recorded_at,
    )
