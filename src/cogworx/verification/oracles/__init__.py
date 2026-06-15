"""Concrete oracle implementations -- the always-on LLM-judge fallback and executable oracles.

Pod 4.1 deliverable: :class:`~cogworx.verification.oracles.judge.LLMJudgeOracle` -- the universal
fallback for non-executable criteria wired into
:class:`~cogworx.verification.oracle.OracleRegistry` at construction time. Executable oracles
(pytest runner, math, chem, physics) land as sub-pods.

The module structure mirrors the ``cogworx.coherence`` package: a seam module defines the protocol,
this package provides concrete implementations.
"""

from __future__ import annotations

from cogworx.verification.oracles.judge import LLMJudgeOracle

__all__ = [
    "LLMJudgeOracle",
]
