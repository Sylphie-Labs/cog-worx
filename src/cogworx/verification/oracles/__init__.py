"""Concrete oracle implementations -- the always-on LLM-judge fallback and executable oracles.

Pod 4.1 deliverables: :class:`~cogworx.verification.oracles.judge.LLMJudgeOracle` -- the universal
fallback for non-executable criteria wired into
:class:`~cogworx.verification.oracle.OracleRegistry` at construction time -- and
:class:`~cogworx.verification.oracles.code.CodeOracle`, the deterministic code-execution oracle
(pytest under a taint-bound local/docker tier). Math + chem/physics oracles land as later sub-pods.

The module structure mirrors the ``cogworx.coherence`` package: a seam module defines the protocol,
this package provides concrete implementations.
"""

from __future__ import annotations

from cogworx.verification.oracles.code import CodeOracle
from cogworx.verification.oracles.judge import LLMJudgeOracle

__all__ = [
    "CodeOracle",
    "LLMJudgeOracle",
]
