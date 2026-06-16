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
from cogworx.verification.dialectic import (
    AntithesisStage,
    ConcludeStage,
    EvaluateStage,
    ExperimentStage,
    HumanResolution,
    ThesisStage,
    VerificationStatus,
)
from cogworx.verification.dialectic_state import (
    MAX_CYCLES,
    REFINE,
    STUCK_JACCARD,
    DialecticAccumulator,
    DialecticRoute,
    cycle_verdicts,
    derive_accumulator,
    jaccard_stuck,
    route_dialectic,
)
from cogworx.verification.evidence_projector import (
    EVIDENCE_PROJECTOR_CONSUMER,
    VerificationEvidenceProjector,
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
from cogworx.verification.oracles import CodeOracle, LLMJudgeOracle
from cogworx.verification.outcome import (
    VerdictRole,
    VerificationRecord,
    record_for,
    stamps_procedural_beta,
    verdict_from_antithesis,
)
from cogworx.verification.quarantine import new_nonce, quarantine, unwrap

__all__ = [
    "EVIDENCE_PROJECTOR_CONSUMER",
    "EXECUTABLE_SOURCES",
    "MAX_CYCLES",
    "REFINE",
    "RETRYABLE_CLASSES",
    "STUCK_JACCARD",
    "TRANSIENT_EXCEPTION_TYPES",
    "AntithesisDisposition",
    "AntithesisStage",
    "AntithesisVerdict",
    "CodeOracle",
    "ConcludeStage",
    "DialecticAccumulator",
    "DialecticRoute",
    "ErrorClass",
    "EvaluateStage",
    "ExperimentStage",
    "FailureOutcome",
    "HumanResolution",
    "LLMJudgeOracle",
    "Oracle",
    "OracleFrame",
    "OracleRegistry",
    "RoutingDecision",
    "Thesis",
    "ThesisStage",
    "Verdict",
    "VerdictRole",
    "VerificationEvidenceProjector",
    "VerificationRecord",
    "VerificationStatus",
    "classify_error",
    "cycle_verdicts",
    "derive_accumulator",
    "jaccard_stuck",
    "new_nonce",
    "quarantine",
    "record_for",
    "route_dialectic",
    "route_failure",
    "stamps_procedural_beta",
    "unwrap",
    "verdict_from_antithesis",
]
