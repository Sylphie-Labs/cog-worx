"""Verification & truth — the Phase 4 oracle layer + dialectic (CANON S9, the capstone).

The external, structural ground-truth signal S9 demands ("never trust the model's self-report;
require external/structural signals"). Public surface (Pod 4.0): the :class:`Oracle` seam, its typed
:class:`Verdict` / :class:`OracleFrame` / :class:`Thesis` contracts, and the
:class:`OracleRegistry` (resolve by ``(completion_criterion, problem_type)``, always-on fallback,
never raises).
"""

from __future__ import annotations

from cogworx.verification.contracts import (
    EXECUTABLE_SOURCES,
    OracleFrame,
    Thesis,
    Verdict,
)
from cogworx.verification.oracle import Oracle, OracleRegistry

__all__ = [
    "EXECUTABLE_SOURCES",
    "Oracle",
    "OracleFrame",
    "OracleRegistry",
    "Thesis",
    "Verdict",
]
