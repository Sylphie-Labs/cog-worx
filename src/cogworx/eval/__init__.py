"""Pure-stdlib evaluation statistics for the cog-worx spike gates (CANON S2, S12).

This package holds the eval-time statistics the Phase-4 GATE spike consumes — the nested
(cluster) bootstrap for paired Youden's-J deltas, the synthetic-data generator that powers the
pre-nightly sizing simulation, and the variance diagnostics. It is pure computation: no model
calls, no substrate I/O, and — per S2 (OSS is reference, not dependency; every dep is the
adopter's burden) — **no scipy and no numpy**. The only sanctioned numeric primitive is the shipped
:mod:`cogworx.knowledge.beta` (regularized-incomplete-beta math) plus :class:`random.Random`.

Semver (pre-1.0): 4.4c-0 adds the additive, backward-compatible ``quantile: float = 0.025`` kwarg
to :func:`nested_bootstrap_delta` (the look-corrected-gate sizing knob, plan §0 R5). The default
reproduces the shipped 2.5/97.5 CI byte-for-byte -> an additive minor bump, no breaking change.
"""

from __future__ import annotations

from cogworx.eval.corpus import (
    Adjudication,
    CorpusItem,
    CorpusLoadError,
    DeterministicPlanterStamp,
    DifficultyMarker,
    HumanLabelProvenance,
    LabelProvenance,
    LLMPlanterStamp,
    OracleLabelProvenance,
    PlanterStamp,
    load_corpus,
)
from cogworx.eval.youden import (
    Cell,
    VarianceDiagnostic,
    mc_proportion_lcb,
    nested_bootstrap_delta,
    power_lcb_from_studies,
    realized_variance_diagnostic,
    synth_cells,
)

__all__ = [
    "Adjudication",
    "Cell",
    "CorpusItem",
    "CorpusLoadError",
    "DeterministicPlanterStamp",
    "DifficultyMarker",
    "HumanLabelProvenance",
    "LLMPlanterStamp",
    "LabelProvenance",
    "OracleLabelProvenance",
    "PlanterStamp",
    "VarianceDiagnostic",
    "load_corpus",
    "mc_proportion_lcb",
    "nested_bootstrap_delta",
    "power_lcb_from_studies",
    "realized_variance_diagnostic",
    "synth_cells",
]
