"""Unit tests for cogworx.knowledge.confidence (CANON S1, S5).

Pins the Beta-posterior derivation rules:
  - Empty events → prior-only (Beta(1,1), confidence=0.5).
  - One "+" tool_proof authority 1.0 → alpha=4.0, beta=1.0, confidence=0.8.
  - Dedup by (source_id, polarity): same source_id/polarity keeps MAX weight.
  - Different source_ids both count; same source_id with DIFFERENT polarity counts twice.
  - "-" refutation feeds beta; mixed evidence yields expected posterior.
  - base_weight is STAMPED at event creation (from EVIDENCE_BASE_WEIGHTS); a hand-set
    nonstandard base_weight on the event object is honoured exactly (history is immune to
    recalibration of the weight table).
  - make_evidence stamps base_weight from EVIDENCE_BASE_WEIGHTS and defaults id to uuid hex.

Hypothesis property tests:
  - confidence always in (0, 1).
  - variance > 0.
  - Adding a "+" event with a fresh source_id never decreases alpha.
  - n_evidence == number of distinct (source_id, polarity) pairs.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from typing import cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cogworx.knowledge.confidence import (
    CLAIM_PRIOR_ALPHA,
    CLAIM_PRIOR_BETA,
    claim_confidence,
)
from cogworx.knowledge.evidence import (
    EVIDENCE_BASE_WEIGHTS,
    EvidenceEvent,
    EvidenceType,
    Polarity,
    make_evidence,
)

_NOW = datetime(2026, 6, 9, tzinfo=UTC)


def _ev(
    *,
    source_id: str = "src-a",
    polarity: Polarity = "+",
    ev_type: EvidenceType = "corroboration",
    source_authority: float = 1.0,
    base_weight: float | None = None,
    event_id: str | None = None,
) -> EvidenceEvent:
    """Construct an EvidenceEvent, optionally overriding base_weight directly."""
    bw = base_weight if base_weight is not None else EVIDENCE_BASE_WEIGHTS[ev_type]
    return EvidenceEvent(
        id=event_id or uuid.uuid4().hex,
        type=ev_type,
        polarity=polarity,
        source_id=source_id,
        source_authority=source_authority,
        base_weight=bw,
        recorded_at=_NOW,
    )


# ---------------------------------------------------------------------------
# FIX 3: base_weight validation (must be > 0, finite, not nan)
# ---------------------------------------------------------------------------


def test_evidence_event_rejects_zero_base_weight() -> None:
    """base_weight=0 raises ValidationError — weight must be strictly positive."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvidenceEvent(
            id="x",
            type="corroboration",
            polarity="+",
            source_id="s",
            source_authority=1.0,
            base_weight=0.0,
            recorded_at=_NOW,
        )


def test_evidence_event_rejects_negative_base_weight() -> None:
    """base_weight=-1.0 raises ValidationError."""
    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvidenceEvent(
            id="x",
            type="corroboration",
            polarity="+",
            source_id="s",
            source_authority=1.0,
            base_weight=-1.0,
            recorded_at=_NOW,
        )


def test_evidence_event_rejects_inf_base_weight() -> None:
    """base_weight=inf raises ValidationError."""
    import math

    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvidenceEvent(
            id="x",
            type="corroboration",
            polarity="+",
            source_id="s",
            source_authority=1.0,
            base_weight=math.inf,
            recorded_at=_NOW,
        )


def test_evidence_event_rejects_nan_base_weight() -> None:
    """base_weight=nan raises ValidationError."""
    import math

    import pydantic

    with pytest.raises(pydantic.ValidationError):
        EvidenceEvent(
            id="x",
            type="corroboration",
            polarity="+",
            source_id="s",
            source_authority=1.0,
            base_weight=math.nan,
            recorded_at=_NOW,
        )


# ---------------------------------------------------------------------------
# Deterministic cases
# ---------------------------------------------------------------------------


def test_no_events_prior_only() -> None:
    """Empty evidence list → Beta(1,1) prior — confidence 0.5, n_evidence 0."""
    result = claim_confidence([])
    assert result.alpha == pytest.approx(CLAIM_PRIOR_ALPHA)
    assert result.beta == pytest.approx(CLAIM_PRIOR_BETA)
    assert result.confidence == pytest.approx(0.5)
    assert result.n_evidence == 0
    assert result.variance > 0


def test_one_tool_proof_authority_1() -> None:
    """One '+' tool_proof with authority 1.0 → alpha=4.0, beta=1.0, confidence=0.8.

    weight = base_weight * source_authority = 3.0 * 1.0 = 3.0
    alpha  = 1.0 + 3.0 = 4.0
    beta   = 1.0
    mean   = 4.0 / 5.0 = 0.8
    """
    ev = _ev(source_id="tool-run-1", polarity="+", ev_type="tool_proof", source_authority=1.0)
    result = claim_confidence([ev])
    assert result.alpha == pytest.approx(4.0)
    assert result.beta == pytest.approx(1.0)
    assert result.confidence == pytest.approx(0.8)
    assert result.n_evidence == 1


def test_dedup_same_source_same_polarity_keeps_max_weight() -> None:
    """Two corroborations from the SAME source_id + polarity: only MAX weight counts.

    n_evidence stays 1; only the higher-weight event contributes.
    ev1: weight = 1.0 * 1.0 = 1.0
    ev2: weight = 1.0 * 0.5 = 0.5   ← weaker, dropped
    alpha = 1.0 + 1.0 = 2.0
    """
    ev1 = _ev(source_id="src-a", polarity="+", ev_type="corroboration", source_authority=1.0)
    ev2 = _ev(source_id="src-a", polarity="+", ev_type="corroboration", source_authority=0.5)
    result = claim_confidence([ev1, ev2])
    assert result.n_evidence == 1
    assert result.alpha == pytest.approx(2.0)  # prior 1.0 + max_weight 1.0
    assert result.beta == pytest.approx(1.0)


def test_different_source_ids_both_count() -> None:
    """Two corroborations from DIFFERENT source_ids: both contribute (n_evidence=2).

    ev1 (src-a): weight = 1.0 * 1.0 = 1.0
    ev2 (src-b): weight = 1.0 * 1.0 = 1.0
    alpha = 1.0 + 1.0 + 1.0 = 3.0
    """
    ev1 = _ev(source_id="src-a", polarity="+", ev_type="corroboration", source_authority=1.0)
    ev2 = _ev(source_id="src-b", polarity="+", ev_type="corroboration", source_authority=1.0)
    result = claim_confidence([ev1, ev2])
    assert result.n_evidence == 2
    assert result.alpha == pytest.approx(3.0)
    assert result.beta == pytest.approx(1.0)


def test_same_source_different_polarity_both_count() -> None:
    """Same source_id but DIFFERENT polarity: contributes to BOTH alpha and beta (n_evidence=2).

    Dedup key is (source_id, polarity) — two distinct pairs.
    ev+ : weight = 3.0 * 1.0 = 3.0  → alpha += 3.0
    ev-  : weight = 3.0 * 1.0 = 3.0  → beta  += 3.0
    (tool_proof / refutation both have base_weight 3.0)
    alpha = 1.0 + 3.0 = 4.0
    beta  = 1.0 + 3.0 = 4.0
    confidence = 4.0 / 8.0 = 0.5
    """
    ev_pos = _ev(source_id="src-x", polarity="+", ev_type="tool_proof", source_authority=1.0)
    ev_neg = _ev(source_id="src-x", polarity="-", ev_type="refutation", source_authority=1.0)
    result = claim_confidence([ev_pos, ev_neg])
    assert result.n_evidence == 2
    assert result.alpha == pytest.approx(4.0)
    assert result.beta == pytest.approx(4.0)
    assert result.confidence == pytest.approx(0.5)


def test_refutation_feeds_beta_only() -> None:
    """A single '-' refutation event feeds beta, not alpha.

    ev- : weight = 3.0 * 1.0 = 3.0
    alpha = 1.0 (prior only)
    beta  = 1.0 + 3.0 = 4.0
    confidence = 1.0 / 5.0 = 0.2
    """
    ev = _ev(source_id="src-a", polarity="-", ev_type="refutation", source_authority=1.0)
    result = claim_confidence([ev])
    assert result.alpha == pytest.approx(1.0)
    assert result.beta == pytest.approx(4.0)
    assert result.confidence == pytest.approx(0.2)
    assert result.n_evidence == 1


def test_mixed_evidence_hand_computed() -> None:
    """Mixed evidence: two sources pro, one con; verify alpha/beta/mean/variance by hand.

    ev1 src-a '+' corroboration auth=1.0 → weight=1.0*1.0=1.0
    ev2 src-b '+' antithesis_survival auth=1.0 → weight=1.5*1.0=1.5
    ev3 src-c '-' refutation auth=0.5 → weight=3.0*0.5=1.5
    alpha = 1.0 + 1.0 + 1.5 = 3.5
    beta  = 1.0 + 1.5 = 2.5
    s = 6.0
    confidence = 3.5 / 6.0 ≈ 0.58333...
    variance   = 3.5*2.5 / (6.0^2 * 7.0) = 8.75 / 252 ≈ 0.034722...
    """
    ev1 = _ev(source_id="src-a", polarity="+", ev_type="corroboration", source_authority=1.0)
    ev2 = _ev(source_id="src-b", polarity="+", ev_type="antithesis_survival", source_authority=1.0)
    ev3 = _ev(source_id="src-c", polarity="-", ev_type="refutation", source_authority=0.5)
    result = claim_confidence([ev1, ev2, ev3])
    assert result.n_evidence == 3
    assert result.alpha == pytest.approx(3.5)
    assert result.beta == pytest.approx(2.5)
    assert result.confidence == pytest.approx(3.5 / 6.0)
    # variance = alpha*beta / (s^2 * (s+1))
    expected_var = (3.5 * 2.5) / (36.0 * 7.0)
    assert result.variance == pytest.approx(expected_var, rel=1e-6)


def test_base_weight_from_event_not_table() -> None:
    """Event's own base_weight is used, not re-derived from EVIDENCE_BASE_WEIGHTS.

    An event stamped with a nonstandard base_weight (e.g. hand-set to 7.5 for a special
    calibration) must feed that exact weight into the posterior. Future recalibrations of
    EVIDENCE_BASE_WEIGHTS never rewrite historical event records.

    weight = 7.5 * 1.0 = 7.5
    alpha  = 1.0 + 7.5 = 8.5
    """
    ev = _ev(
        source_id="src-a",
        polarity="+",
        ev_type="corroboration",
        source_authority=1.0,
        base_weight=7.5,
    )
    result = claim_confidence([ev])
    assert result.alpha == pytest.approx(8.5)
    assert result.n_evidence == 1


def test_make_evidence_stamps_base_weight_from_table() -> None:
    """make_evidence stamps base_weight from EVIDENCE_BASE_WEIGHTS at creation time."""
    for ev_type, expected_bw in EVIDENCE_BASE_WEIGHTS.items():
        ev = make_evidence(
            type=ev_type,
            polarity="+",
            source_id="src",
            source_authority=1.0,
            recorded_at=_NOW,
        )
        assert ev.base_weight == pytest.approx(expected_bw), (
            f"make_evidence did not stamp correct base_weight for type={ev_type!r}"
        )


def test_make_evidence_defaults_id_to_uuid_hex() -> None:
    """make_evidence defaults id to a uuid4 hex string (32 lowercase hex chars)."""
    ev = make_evidence(
        type="corroboration",
        polarity="+",
        source_id="src",
        source_authority=1.0,
        recorded_at=_NOW,
    )
    assert len(ev.id) == 32
    assert all(c in "0123456789abcdef" for c in ev.id)


def test_make_evidence_explicit_id_accepted() -> None:
    """make_evidence accepts an explicit event_id (test-deterministic use case)."""
    ev = make_evidence(
        type="corroboration",
        polarity="+",
        source_id="src",
        source_authority=1.0,
        recorded_at=_NOW,
        event_id="fixed-id-001",
    )
    assert ev.id == "fixed-id-001"


# ---------------------------------------------------------------------------
# Hypothesis property tests
# ---------------------------------------------------------------------------

_ev_type_st = st.sampled_from(
    ["tool_proof", "refutation", "antithesis_survival", "corroboration", "recall"]
)
_polarity_st = st.sampled_from(["+", "-"])
_authority_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)
# FIX 3: arbitrary positive finite base_weights ensure the 0 < confidence < 1 property is
# guaranteed by construction (not just for the table values). The type now enforces gt=0 and
# allow_inf_nan=False so this range matches the validation constraint.
_base_weight_st = st.floats(min_value=1e-6, max_value=1e6, allow_nan=False, allow_infinity=False)


@st.composite
def _evidence_list_st(draw: st.DrawFn) -> list[EvidenceEvent]:
    """A strategy for a list of 0-8 EvidenceEvent objects with random fields.

    base_weight is drawn from arbitrary positive finite floats (1e-6..1e6) rather than only the
    table values — the 0 < confidence < 1 property must be guaranteed by construction (FIX 3).
    """
    n = draw(st.integers(min_value=0, max_value=8))
    return [
        _ev(
            source_id=draw(st.text(alphabet="abcdef", min_size=1, max_size=4)),
            polarity=cast(Polarity, draw(_polarity_st)),
            ev_type=cast(EvidenceType, draw(_ev_type_st)),
            source_authority=draw(_authority_st),
            base_weight=draw(_base_weight_st),
        )
        for _ in range(n)
    ]


@given(_evidence_list_st())
@settings(max_examples=200)
def test_confidence_always_in_open_unit_interval(events: list[EvidenceEvent]) -> None:
    """confidence is always strictly in (0, 1) — the posterior mean of a Beta."""
    result = claim_confidence(events)
    assert 0.0 < result.confidence < 1.0


@given(_evidence_list_st())
@settings(max_examples=200)
def test_variance_always_positive(events: list[EvidenceEvent]) -> None:
    """variance is always > 0 (Beta distribution has positive variance when alpha, beta > 0)."""
    result = claim_confidence(events)
    assert result.variance > 0.0


@given(_evidence_list_st(), st.text(alphabet="ghijklmn", min_size=2, max_size=6))
@settings(max_examples=200)
def test_adding_positive_event_never_decreases_alpha(
    events: list[EvidenceEvent], fresh_source: str
) -> None:
    """Adding a '+' event with a FRESH source_id (never seen before) never decreases alpha."""
    before = claim_confidence(events)
    # Fresh source_id guaranteed not to collide with any existing (source_id, polarity).
    new_ev = _ev(
        source_id=f"fresh-{fresh_source}",
        polarity="+",
        ev_type="corroboration",
        source_authority=1.0,
    )
    after = claim_confidence([*events, new_ev])
    assert after.alpha >= before.alpha - 1e-9  # allow floating-point epsilon


@given(_evidence_list_st())
@settings(max_examples=200)
def test_n_evidence_equals_distinct_source_polarity_pairs(events: list[EvidenceEvent]) -> None:
    """n_evidence == number of distinct (source_id, polarity) pairs after dedup by max weight."""
    # The set of distinct (source_id, polarity) pairs — same as what claim_confidence deduplicates.
    distinct_pairs = {(ev.source_id, ev.polarity) for ev in events}
    result = claim_confidence(events)
    assert result.n_evidence == len(distinct_pairs)


# ---------------------------------------------------------------------------
# ClaimConfidence is frozen
# ---------------------------------------------------------------------------


def test_claim_confidence_is_frozen() -> None:
    """ClaimConfidence is an immutable model — mutation raises ValidationError."""
    import pydantic

    result = claim_confidence([])
    with pytest.raises((pydantic.ValidationError, TypeError)):
        result.confidence = 0.99  # type: ignore[misc]
