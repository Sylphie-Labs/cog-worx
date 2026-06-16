"""Pod 4.3 Build step 1 -- verdict_from_antithesis adapter (outcome.py).

Covers:
- Three AntithesisDispositions (BROKE / COULD_NOT_BREAK / ABSTAINED) x oracle_backed True/False.
- source == "tool" iff oracle_backed; source == "inference" otherwise.
- Integration: record_for(verdict_from_antithesis(av), role="antithesis") produces the
  expected VerificationRecord or None for each disposition (the contract that matters for the
  evidence projector).
"""

from __future__ import annotations

import pytest

from cogworx.verification.contracts import Verdict
from cogworx.verification.honest_failure import AntithesisDisposition, AntithesisVerdict
from cogworx.verification.outcome import VerificationRecord, record_for, verdict_from_antithesis

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _av(
    disposition: AntithesisDisposition,
    *,
    breakage: str | None = None,
    oracle_backed: bool = False,
    confidence: float = 0.5,
) -> AntithesisVerdict:
    return AntithesisVerdict(
        disposition=disposition,
        breakage=breakage,
        oracle_backed=oracle_backed,
        confidence=confidence,
    )


def _broke(*, oracle_backed: bool = False) -> AntithesisVerdict:
    return _av(AntithesisDisposition.BROKE, breakage="Found a flaw.", oracle_backed=oracle_backed)


def _could_not_break(*, oracle_backed: bool = False) -> AntithesisVerdict:
    return _av(AntithesisDisposition.COULD_NOT_BREAK, oracle_backed=oracle_backed)


def _abstained(*, oracle_backed: bool = False) -> AntithesisVerdict:
    return _av(AntithesisDisposition.ABSTAINED, oracle_backed=oracle_backed)


# ---------------------------------------------------------------------------
# source mapping — the OB-PROV invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "av, expected_source",
    [
        (_broke(oracle_backed=False), "inference"),
        (_broke(oracle_backed=True), "tool"),
        (_could_not_break(oracle_backed=False), "inference"),
        (_could_not_break(oracle_backed=True), "tool"),
        (_abstained(oracle_backed=False), "inference"),
        (_abstained(oracle_backed=True), "tool"),
    ],
)
def test_source_is_tool_iff_oracle_backed(av: AntithesisVerdict, expected_source: str) -> None:
    """source == "tool" iff oracle_backed; "inference" otherwise (OB-PROV invariant)."""
    result = verdict_from_antithesis(av)
    assert result.source == expected_source


# ---------------------------------------------------------------------------
# ABSTAINED disposition → invalid check (valid_check=False, holds=False)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle_backed", [False, True])
def test_abstained_yields_invalid_check(oracle_backed: bool) -> None:
    result = verdict_from_antithesis(_abstained(oracle_backed=oracle_backed))
    assert isinstance(result, Verdict)
    assert result.holds is False
    assert result.valid_check is False


# ---------------------------------------------------------------------------
# COULD_NOT_BREAK disposition → holds=True, valid_check=True
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle_backed", [False, True])
def test_could_not_break_yields_holds_true_valid_check_true(oracle_backed: bool) -> None:
    result = verdict_from_antithesis(_could_not_break(oracle_backed=oracle_backed))
    assert result.holds is True
    assert result.valid_check is True


# ---------------------------------------------------------------------------
# BROKE disposition → holds=False, valid_check=True; reasoning carries breakage text
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("oracle_backed", [False, True])
def test_broke_yields_holds_false_valid_check_true(oracle_backed: bool) -> None:
    result = verdict_from_antithesis(_broke(oracle_backed=oracle_backed))
    assert result.holds is False
    assert result.valid_check is True


def test_broke_reasoning_carries_breakage_text() -> None:
    av = _av(AntithesisDisposition.BROKE, breakage="Specific flaw detail.")
    result = verdict_from_antithesis(av)
    assert result.reasoning == "Specific flaw detail."


def test_could_not_break_reasoning_is_sensible_default() -> None:
    """When there is no breakage text (COULD_NOT_BREAK), reasoning is a non-empty fallback."""
    result = verdict_from_antithesis(_could_not_break())
    assert result.reasoning  # non-empty string


# ---------------------------------------------------------------------------
# Integration: record_for(verdict_from_antithesis(av), role="antithesis")
# ---------------------------------------------------------------------------


def test_integration_could_not_break_inference_yields_antithesis_survival() -> None:
    """COULD_NOT_BREAK + oracle_backed=False → antithesis_survival "+", epistemic inference."""
    av = _could_not_break(oracle_backed=False)
    rec = record_for(verdict_from_antithesis(av), role="antithesis")
    assert rec == VerificationRecord("antithesis_survival", "+", "inference", None)


def test_integration_could_not_break_oracle_backed_yields_confirmed() -> None:
    """COULD_NOT_BREAK + oracle_backed=True → antithesis_survival "+", epistemic confirmed."""
    av = _could_not_break(oracle_backed=True)
    rec = record_for(verdict_from_antithesis(av), role="antithesis")
    assert rec == VerificationRecord("antithesis_survival", "+", "confirmed", None)


def test_integration_broke_yields_none() -> None:
    """BROKE (model-claimed break) → record_for returns None — routing-only, not truth evidence."""
    for oracle_backed in (False, True):
        av = _broke(oracle_backed=oracle_backed)
        assert record_for(verdict_from_antithesis(av), role="antithesis") is None


def test_integration_abstained_yields_none() -> None:
    """ABSTAINED → valid_check=False → record_for returns None."""
    for oracle_backed in (False, True):
        av = _abstained(oracle_backed=oracle_backed)
        assert record_for(verdict_from_antithesis(av), role="antithesis") is None


def test_integration_no_procedural_beta_on_antithesis_role() -> None:
    """The antithesis role NEVER stamps the procedural Beta (procedural_outcome is always None)."""
    av = _could_not_break(oracle_backed=True)  # most "privileged" case — still no Beta
    rec = record_for(verdict_from_antithesis(av), role="antithesis")
    assert rec is not None
    assert rec.procedural_outcome is None
