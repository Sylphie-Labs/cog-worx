"""Coherence reconciler — async/batched contradiction detection and resolution (Pod 2.7).

This package contains the off-write-path coherence reconciler that detects and resolves
contradictions between entity-KG claims (CANON S1, S5).
"""

from cogworx.coherence.config import CoherenceConfig
from cogworx.coherence.entrenchment import (
    EPISTEMIC_RANK,
    Entrenchment,
    Resolution,
    decide_resolution,
    entrenchment_of,
)
from cogworx.coherence.mus import BudgetExceeded, CallBudget, MusResult, find_mus
from cogworx.coherence.oracle import (
    ConsistencyOracle,
    ModelConsistencyOracle,
    OracleAnswer,
    OracleProtocolError,
)
from cogworx.coherence.pairs import CandidatePair, CandidateReport, candidate_pairs
from cogworx.coherence.promotion import PromotionRule, ScopePromoter
from cogworx.coherence.reconciler import CoherenceReconciler, ReconcilerStats
from cogworx.coherence.upgrade import (
    UPGRADE_ELIGIBLE_EVIDENCE,
    UpgradeResult,
    epistemic_upgrade,
)

__all__ = [
    "EPISTEMIC_RANK",
    "UPGRADE_ELIGIBLE_EVIDENCE",
    "BudgetExceeded",
    "CallBudget",
    "CandidatePair",
    "CandidateReport",
    "CoherenceConfig",
    "CoherenceReconciler",
    "ConsistencyOracle",
    "Entrenchment",
    "ModelConsistencyOracle",
    "MusResult",
    "OracleAnswer",
    "OracleProtocolError",
    "PromotionRule",
    "ReconcilerStats",
    "Resolution",
    "ScopePromoter",
    "UpgradeResult",
    "candidate_pairs",
    "decide_resolution",
    "entrenchment_of",
    "epistemic_upgrade",
    "find_mus",
]
