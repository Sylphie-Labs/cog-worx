"""Honest off-path recording of a verification verdict (CANON S1, S5, S9 — F1/F2/F9).

How a verification :class:`~cogworx.verification.contracts.Verdict` becomes entity-KG evidence and
(sometimes) a procedural-KG Beta update — a PURE decision, no I/O. The Pod 4.3 verification-evidence
projector (sweeper-shaped, off-path, S1) consumes this; Pod 4.0 only fixes the mapping.

The honest-provenance core (F1) keeps two questions separate:
  * PROVENANCE source (on any resulting claim) is a function of ``verdict.source`` — ``tool``/
    ``system`` for an executable oracle, ``inference`` for a model-judge verdict. The verdict
    carries it structurally (never model output); the projector reads ``verdict.source`` directly.
  * EVIDENCE TYPE / truth-weight (on the entity-KG ``EvidenceEvent``) is chosen here from the
    calibrated set already shipped for this phase: ``tool_proof`` / ``refutation`` (weight 3.0,
    FIRST-HAND only) and ``antithesis_survival`` (weight 1.5 — the adversary tried and failed;
    indirect). No ``'oracle'`` ``ProvenanceSource`` member is added (F9: a frozen-Literal
    addition is breaking, not additive).

The S9 boundary (F2): only an EXECUTABLE/deterministic verdict (``verdict.is_executable``) is
first-hand truth. A model-judge verdict is a heuristic prioritizer, not a verifier — it produces NO
truth-posterior evidence and NEVER stamps the procedural Beta; it only drives routing (Pod 4.3). The
procedural Beta is stamped ONLY on ``valid_check`` AND an executable source (F1) — an invalid or
model-judge experiment stamps nothing, which keeps the Pod 2.1 posterior honest.

F1/F2 calibration ratified — architect, 2026-06-15: the antithesis-role calibration (survival =
indirect positive; a model-claimed break is routing-only, not a first-hand refutation) was reviewed
as part of the H4 OB-PROV ruling and confirmed conservative + faithful. No longer owed.

Contract changelog (CANON §6.1):
  - 2026-06-15 (Pod 4.3, ADDITIVE): added :func:`verdict_from_antithesis`, the pure
    ``AntithesisVerdict`` -> ``Verdict`` adapter the verification-evidence projector uses to reuse
    :func:`record_for` for the antithesis role. A new function — no existing caller or implementer
    of this module becomes non-conformant, so it is additive (no pre-approval required).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from cogworx.claims.provenance import EpistemicType
from cogworx.knowledge.evidence import EvidenceType, Polarity
from cogworx.verification.contracts import Verdict
from cogworx.verification.honest_failure import AntithesisDisposition, AntithesisVerdict

if TYPE_CHECKING:
    # Annotation-only — avoids importing the substrate.procedural_kg chain at runtime.
    from cogworx.substrate.procedural_kg import Outcome

VerdictRole = Literal["oracle", "antithesis"]
"""Which dialectic stage produced the verdict: the experiment ORACLE (the verifier) or the
ANTITHESIS (the adversarial challenger). The S9 weight of an outcome depends on its role."""


@dataclass(frozen=True)
class VerificationRecord:
    """The off-path consequences of one verdict: which entity-KG evidence to write, and whether to
    stamp the procedural-KG Beta. A PURE decision — the projector (Pod 4.3) does the I/O."""

    evidence_type: EvidenceType
    polarity: Polarity
    epistemic_type: EpistemicType
    procedural_outcome: Outcome | None
    """``"success"`` / ``"failure"`` to stamp the procedural-KG Beta, or ``None`` to stamp nothing.
    Non-``None`` ONLY for an executable ORACLE verdict (F1) — never for a model-judge or the
    antithesis role, so the posterior stays on first-hand executable checks."""


def stamps_procedural_beta(verdict: Verdict) -> bool:
    """The Pod 2.1 Beta is updated ONLY on a valid, executable verdict (F1): ``valid_check`` AND
    ``is_executable``. A model-judge or invalid experiment stamps nothing."""
    return verdict.valid_check and verdict.is_executable


def record_for(verdict: Verdict, *, role: VerdictRole) -> VerificationRecord | None:
    """Map a verdict to its honest off-path record, or ``None`` when it yields no truth evidence.

    ``None`` (routing-only, no truth-posterior write) when: the check was invalid
    (``valid_check=False``); the verdict came from a model-judge ORACLE (a prioritizer, not a
    verifier — F2); or it is a model-claimed antithesis break (a flaw the model asserts but no
    first-hand check confirmed — S9). A first-hand executable break is recorded via the ``oracle``
    role (as a ``refutation``), never via the adversary.
    """
    if not verdict.valid_check:
        return None

    if role == "oracle":
        if not verdict.is_executable:
            return None  # F2: a model-judge oracle prioritizes; it does not verify.
        if verdict.holds:
            return VerificationRecord("tool_proof", "+", "confirmed", "success")
        return VerificationRecord("refutation", "-", "confirmed", "failure")

    # role == "antithesis": the adversary's only first-hand truth signal is the thesis SURVIVING.
    if verdict.holds:  # could_not_break — the thesis withstood the challenge
        epistemic: EpistemicType = "confirmed" if verdict.is_executable else "inference"
        return VerificationRecord("antithesis_survival", "+", epistemic, None)
    return None  # a model-claimed break drives refinement (routing), not truth evidence.


def verdict_from_antithesis(av: AntithesisVerdict) -> Verdict:
    """Adapt an :class:`~cogworx.verification.honest_failure.AntithesisVerdict` to a
    :class:`~cogworx.verification.contracts.Verdict` so the evidence projector can pass it to
    :func:`record_for` uniformly.

    Faithful to ``record_for``'s antithesis semantics (``outcome.py:86-90``):

    * ``COULD_NOT_BREAK`` → ``holds=True, valid_check=True`` → ``antithesis_survival "+"``
      (epistemic ``confirmed`` iff executable, else ``inference``).
    * ``BROKE`` → ``holds=False, valid_check=True`` → ``record_for`` returns ``None`` (a
      model-claimed break drives refinement, not truth evidence — S9).
    * ``ABSTAINED`` → ``valid_check=False`` → ``record_for`` returns ``None``.

    The ``source`` mapping is the ONLY place the ``oracle_backed`` bit crosses into the evidence
    stream.  This mapping is safe ONLY because of **OB-PROV** (§2.3 of the Pod 4.3 plan): the
    producing stage sets ``oracle_backed`` from its own oracle call, never from parsed model output.
    In v1 (model adversary, no executable oracle) ``oracle_backed`` is always ``False``, so this
    adapter always yields ``source="inference"`` — antithesis-survival evidence is epistemic
    ``inference``, never ``confirmed``.
    """
    # OB-PROV: oracle_backed was set by the stage's own oracle call, never from model output.
    source = "tool" if av.oracle_backed else "inference"
    if av.disposition is AntithesisDisposition.ABSTAINED:
        return Verdict(
            holds=False,
            valid_check=False,
            reasoning="Antithesis abstained — no meaningful challenge could be formed.",
            source=source,
        )
    holds = av.disposition is AntithesisDisposition.COULD_NOT_BREAK
    return Verdict(
        holds=holds,
        valid_check=True,
        reasoning=av.breakage or "Antithesis could not find a concrete flaw.",
        source=source,
    )


__all__ = [
    "VerdictRole",
    "VerificationRecord",
    "record_for",
    "stamps_procedural_beta",
    "verdict_from_antithesis",
]
