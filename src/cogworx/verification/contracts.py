"""Verification-seam value contracts — the oracle's typed inputs and its verdict (CANON S5, S9).

These are the frozen data contracts the Phase 4 verification oracle speaks in. They are kept
distinct from :mod:`cogworx.coherence.oracle` (the *consistency reconciler's* oracle — a different
concept) and from the generic :class:`~cogworx.claims.provenance.Artifact` (the dialectic stages
serialize these into ``Artifact.data`` for journaling, but the oracle SEAM itself stays typed).
Ported from tess ``artifacts.py`` (``FrameArtifact`` / ``ThesisArtifact`` / ``ExperimentTrial``).

:class:`Verdict` follows the S9 control-bit discipline of ``coherence.oracle.OracleAnswer``:
``holds`` and ``valid_check`` are the ONLY fields routing may read; ``reasoning`` is audit/log only
and MUST NOT be parsed for control. ``source`` is the honest-provenance bit (F1) — the epistemic
class of the verdict, set by the oracle from its OWN structural nature and NEVER from model output,
so a model-judge verdict can never be laundered into a confirmed tool-proof.

Contract changelog (CANON §6.1):
  - 2026-06-16 (Pod 4.4b, ADDITIVE): ``Verdict.test_provenance``
    (``Literal["thesis","frozen","n/a"]``, default ``"n/a"``) — audit-only test-provenance bit for
    INV-3a; control-inert (``is_executable`` still derives from ``source`` only).
"""

from __future__ import annotations

from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import ProvenanceSource

# Executable / deterministic oracle sources. A verdict from one of these is a first-hand observation
# of an external check (we ran the tool / it is a zero-model control event), so it MAY back a
# CONFIRMED correctness claim and a procedural-KG Beta update. A model-judge verdict is "inference"
# and may NOT — it is a heuristic prioritizer, not a verifier (F1/F2).
EXECUTABLE_SOURCES: Final[tuple[ProvenanceSource, ...]] = ("tool", "system")


class OracleFrame(BaseModel):
    """What the oracle verifies against: the problem and its completion criterion.

    ``(completion_criterion, problem_type)`` is the :class:`OracleRegistry` key
    (exact -> wildcard -> fallback).
    """

    model_config = ConfigDict(frozen=True)

    completion_criterion: str
    problem_type: str
    problem_statement: str


class Thesis(BaseModel):
    """A proposed solution plus the experiment that would test it.

    ``verifiable_claim is None`` is the honest abstention (Pod 4.2) — "I cannot establish this" —
    structurally distinct from a confident proposal; the dialectic values an honest abstention
    over a confident fabrication (it is not penalized as a refuted thesis).
    """

    model_config = ConfigDict(frozen=True)

    proposed_solution: str
    experiment_design: str
    verifiable_claim: str | None = None


class Verdict(BaseModel):
    """The oracle's typed answer — port of tess ``ExperimentTrial``'s three load-bearing fields
    (``success`` / ``experiment_was_valid_test`` / ``reasoning``), plus the honest ``source`` (F1).

    Control discipline (S9): ``holds`` and ``valid_check`` are the ONLY fields routing may read;
    ``reasoning`` is audit/log ONLY and MUST NOT be parsed for control flow anywhere (mirrors
    ``coherence.oracle.OracleAnswer.raw_text``).
    """

    model_config = ConfigDict(frozen=True)

    holds: bool
    """THE verdict bit — did the thesis hold under the check. Routing may read this."""

    valid_check: bool
    """The honest "this actually tested something" bit — ``False`` on timeout / no tests collected /
    tool threw. It GATES the downstream Beta update so a noise verdict cannot bias the posterior
    (port tess ``experiment_was_valid_test``). Routing may read this."""

    reasoning: str
    """Audit / log ONLY — NEVER parsed for control flow (S9)."""

    source: ProvenanceSource
    """The epistemic class of THIS verdict (F1 honest provenance).

    CONTRACT (S9): ``source`` MUST be set by the oracle implementation from its OWN structural
    nature — an executable / deterministic oracle sets ``"tool"`` (we ran the check) or
    ``"system"`` (a zero-model control event); the LLM-judge fallback sets ``"inference"``. It
    MUST NEVER be derived from model output. A model-chosen ``source`` is an S9 violation of the
    same class as a model-chosen evidence ``source_id`` (:mod:`cogworx.knowledge.evidence`): it
    would let a model launder its own self-report into a confirmed tool-proof. The
    verification-evidence projector (Pod 4.3) reads this to choose the evidence type + provenance
    confidence, and stamps the procedural-KG Beta ONLY when ``valid_check`` and
    :attr:`is_executable`. Structural enforcement is the F2 routing invariant + code review; this
    contract is the interim discipline.
    """

    test_provenance: Literal["thesis", "frozen", "n/a"] = "n/a"
    """Audit-only record of which test corpus was used in the code-execution oracle (INV-3a).

    Set by :class:`~cogworx.verification.oracles.code.CodeOracle` from its OWN ``test_source``
    selector (``"thesis"`` or ``"frozen"``); every other oracle and every direct ``Verdict``
    construction keeps the default ``"n/a"``.

    CONTRACT (S9 / audit-only discipline — mirrors ``source`` and ``reasoning``):
      (a) NEVER read for routing, gating, or control flow anywhere in the system.
      (b) :attr:`is_executable` continues to derive ONLY from ``source``; ``test_provenance``
          plays no role in it.
      (c) MUST be set by the oracle from its own ``test_source`` configuration, NEVER from model
          output.  A model-supplied value here is an S9 violation of the same class as a
          model-supplied ``source``.
    Wiring this field into confirmed-minting or epistemic decisions is deferred as
    CF-4.4-CODEORACLE-SELFTEST (Jim-gated carry-forward).
    """

    @property
    def is_executable(self) -> bool:
        """``True`` when this verdict came from an executable / deterministic oracle
        (``source in {"tool", "system"}``) — the only verdicts that may back a CONFIRMED correctness
        claim or a procedural-KG Beta update (F1/F2). A model-judge (``inference``) verdict is a
        heuristic prioritizer, not a verifier."""
        return self.source in EXECUTABLE_SOURCES


__all__ = [
    "EXECUTABLE_SOURCES",
    "OracleFrame",
    "Thesis",
    "Verdict",
]
