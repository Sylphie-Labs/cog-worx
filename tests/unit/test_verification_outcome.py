"""Pod 4.0 F1/F9 — honest off-path recording of a verification verdict.

Deterministic, model-free: the procedural-Beta stamp gate and the verdict -> evidence mapping,
including the S9 executable-vs-model-judge boundary (F2).
"""

from __future__ import annotations

from cogworx.claims.provenance import ProvenanceSource
from cogworx.verification import (
    Verdict,
    VerificationRecord,
    record_for,
    stamps_procedural_beta,
)


def _verdict(
    *, holds: bool, valid_check: bool = True, source: ProvenanceSource = "tool"
) -> Verdict:
    return Verdict(holds=holds, valid_check=valid_check, reasoning="", source=source)


def test_beta_stamped_only_on_valid_executable_verdict() -> None:
    assert stamps_procedural_beta(_verdict(holds=True, source="tool"))
    assert stamps_procedural_beta(_verdict(holds=False, source="system"))
    assert not stamps_procedural_beta(_verdict(holds=True, source="inference"))  # model-judge
    assert not stamps_procedural_beta(_verdict(holds=True, valid_check=False))  # noise


def test_invalid_check_records_nothing() -> None:
    assert record_for(_verdict(holds=True, valid_check=False), role="oracle") is None
    assert record_for(_verdict(holds=True, valid_check=False), role="antithesis") is None


def test_executable_oracle_holds_is_tool_proof_success() -> None:
    rec = record_for(_verdict(holds=True, source="tool"), role="oracle")
    assert rec == VerificationRecord("tool_proof", "+", "confirmed", "success")


def test_executable_oracle_breaks_is_refutation_failure() -> None:
    rec = record_for(_verdict(holds=False, source="system"), role="oracle")
    assert rec == VerificationRecord("refutation", "-", "confirmed", "failure")


def test_model_judge_oracle_records_nothing() -> None:
    # F2: a model-judge oracle is a prioritizer, not a verifier -> no truth evidence, no Beta.
    assert record_for(_verdict(holds=True, source="inference"), role="oracle") is None
    assert record_for(_verdict(holds=False, source="inference"), role="oracle") is None


def test_antithesis_survival_is_indirect_positive_with_no_beta() -> None:
    model = record_for(_verdict(holds=True, source="inference"), role="antithesis")
    assert model == VerificationRecord("antithesis_survival", "+", "inference", None)
    # An executable-backed survival is first-hand -> confirmed, but STILL no Beta (the oracle role
    # owns the procedural posterior; the antithesis never stamps it).
    executable = record_for(_verdict(holds=True, source="tool"), role="antithesis")
    assert executable == VerificationRecord("antithesis_survival", "+", "confirmed", None)


def test_antithesis_break_is_routing_only_not_truth_evidence() -> None:
    # S9: a model-claimed break is not a first-hand refutation; it only drives refinement.
    assert record_for(_verdict(holds=False, source="inference"), role="antithesis") is None
    assert record_for(_verdict(holds=False, source="tool"), role="antithesis") is None
