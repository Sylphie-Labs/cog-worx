"""Verification & truth — the Phase 4 oracle layer + dialectic (CANON S9, the capstone).

The external, structural ground-truth signal S9 demands ("never trust the model's self-report;
require external/structural signals"). Public surface:
- the :class:`Oracle` seam, its typed :class:`Verdict` / :class:`OracleFrame` / :class:`Thesis`
  contracts, and the :class:`OracleRegistry` (resolve by ``(completion_criterion, problem_type)``,
  always-on fallback, never raises);
- :class:`LLMJudgeOracle`, the always-on model-judge fallback (verdict ``source="inference"``);
- the honest off-path recording map (:func:`record_for`, :func:`stamps_procedural_beta`);
- the model-output quarantine channel (:func:`quarantine` / :func:`unwrap`);
- honest-failure primitives (:func:`classify_error`, :func:`route_failure`,
  :class:`AntithesisVerdict`, the error taxonomy, and the routing decision).
"""

from __future__ import annotations

from cogworx.verification.contracts import (
    EXECUTABLE_SOURCES,
    OracleFrame,
    Thesis,
    Verdict,
)
from cogworx.verification.honest_failure import (
    RETRYABLE_CLASSES,
    TRANSIENT_EXCEPTION_TYPES,
    AntithesisDisposition,
    AntithesisVerdict,
    ErrorClass,
    FailureOutcome,
    RoutingDecision,
    classify_error,
    route_failure,
)
from cogworx.verification.oracle import Oracle, OracleRegistry
from cogworx.verification.oracles import LLMJudgeOracle
from cogworx.verification.outcome import (
    VerdictRole,
    VerificationRecord,
    record_for,
    stamps_procedural_beta,
)
from cogworx.verification.quarantine import new_nonce, quarantine, unwrap

__all__ = [
    "EXECUTABLE_SOURCES",
    "RETRYABLE_CLASSES",
    "TRANSIENT_EXCEPTION_TYPES",
    "AntithesisDisposition",
    "AntithesisVerdict",
    "ErrorClass",
    "FailureOutcome",
    "LLMJudgeOracle",
    "Oracle",
    "OracleFrame",
    "OracleRegistry",
    "RoutingDecision",
    "Thesis",
    "Verdict",
    "VerdictRole",
    "VerificationRecord",
    "classify_error",
    "new_nonce",
    "quarantine",
    "record_for",
    "route_failure",
    "stamps_procedural_beta",
    "unwrap",
]
