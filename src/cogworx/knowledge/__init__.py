"""Entity knowledge-graph domain layer — confidence math + claim identity (CANON S3, S5).

Pure computation: no model calls, no substrate I/O. The substrate adapter
(:mod:`cogworx.adapters.neo4j_entity_kg`) consumes these types and functions.
"""

from __future__ import annotations

from cogworx.knowledge.beta import beta_inverse_cdf, lcb, regularized_incomplete_beta
from cogworx.knowledge.confidence import (
    CLAIM_PRIOR_ALPHA,
    CLAIM_PRIOR_BETA,
    ClaimConfidence,
    claim_confidence,
)
from cogworx.knowledge.episodes import EpisodeKind, Turn, stamp_turns, turns_of
from cogworx.knowledge.evidence import (
    EVIDENCE_BASE_WEIGHTS,
    EvidenceEvent,
    EvidenceType,
    Polarity,
    make_evidence,
)
from cogworx.knowledge.extraction import (
    ExtractionResult,
    RawClaimItem,
    mint_extraction_claims,
    render_transcript,
    validate_raw_model_json,
)
from cogworx.knowledge.identity import DEFAULT_SCOPE, claim_id_for, normalize_topic_part
from cogworx.knowledge.procedural_confidence import (
    TRIAL_BASE_WEIGHT,
    ProcedureSuccess,
    TrialOutcome,
    procedure_success,
)
from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for
from cogworx.knowledge.procedural_promotion import PromotionPolicy
from cogworx.knowledge.procedural_registry import ProcedureDeclaration, ProcedureRegistry
from cogworx.knowledge.scopes import Scope, ScopeKind, ScopeRegistry, ScopeWriteToken
from cogworx.knowledge.source_registry import SourceDeclaration, SourceKind, SourceRegistry

__all__ = [
    "CLAIM_PRIOR_ALPHA",
    "CLAIM_PRIOR_BETA",
    "DEFAULT_SCOPE",
    "EVIDENCE_BASE_WEIGHTS",
    "TRIAL_BASE_WEIGHT",
    "ClaimConfidence",
    "EpisodeKind",
    "EvidenceEvent",
    "EvidenceType",
    "ExtractionResult",
    "Polarity",
    "ProcedureDeclaration",
    "ProcedureRegistry",
    "ProcedureSuccess",
    "PromotionPolicy",
    "RawClaimItem",
    "Scope",
    "ScopeKind",
    "ScopeRegistry",
    "ScopeWriteToken",
    "SourceDeclaration",
    "SourceKind",
    "SourceRegistry",
    "TrialOutcome",
    "Turn",
    "beta_inverse_cdf",
    "claim_confidence",
    "claim_id_for",
    "lcb",
    "make_evidence",
    "mint_extraction_claims",
    "normalize_topic_part",
    "problem_type_id_for",
    "procedure_id_for",
    "procedure_success",
    "regularized_incomplete_beta",
    "render_transcript",
    "stamp_turns",
    "turns_of",
    "validate_raw_model_json",
]
