"""The static procedure registry — declaration-time ``(pathway, stage) -> procedure`` (CANON S9).

A ``ProcedureRegistry`` is how the framework DECLARES, at code-authoring time, that a given
``(pathway, stage)`` is a learnable procedure scored against a given problem type. It is the
structural enforcement of S9's identity discipline: the procedure id and problem type are MINTED
deterministically from the declared labels (:mod:`cogworx.knowledge.procedural_identity`), NEVER
taken from model output. A model cannot split or merge procedures to inflate its own promotion
because it never names them — the developer does, once, in code.

This registry is what lets the off-write-path trial projector do two things it otherwise could not:

  1. RECOGNISE a committed journal step as a procedure trial — a step whose ``(pathway, stage)`` is
     registered is a trial; one whose stage is not registered is ordinary control flow the projector
     skips.
  2. SYNTHESISE a failure trial for a step that failed WITHOUT a stamped outcome (the survivorship
     fix). An engine-synthesised retry-exhaustion ``Degraded`` carries
     ``failure_class``/``attempts`` but NO ``procedure_id`` (the engine does not know the procedure
     mapping); the projector recovers the ``(procedure_id, problem_type)`` from this registry via
     the failed step's ``(pathway, stage)`` so the failure still contributes to the posterior.
     Without this, hard failures would never be counted and every posterior would be biased upward.

PURE / SUBSTRATE-FREE: this module depends only on :mod:`cogworx.knowledge`, never on the substrate
or the model — it is declaration data + deterministic minting. The projector (in ``runtime``)
consumes it.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for

__all__ = [
    "ProcedureDeclaration",
    "ProcedureRegistry",
]


@dataclass(frozen=True)
class ProcedureDeclaration:
    """The resolved identity of a declared procedure: the deterministic ids the projector stamps.

    Minted once at declaration time from the human-supplied labels so that every trial the projector
    writes for this ``(pathway, stage)`` collides on ONE ``(:Procedure)`` node and ONE
    ``(:ProblemType)`` node (trials accumulate against them). ``procedure_id`` and ``problem_type``
    are exactly the ids the substrate ``record_trial`` seam expects (S9 — framework-assigned).
    """

    pathway: str
    stage: str
    procedure_id: str
    problem_type: str


class ProcedureRegistry:
    """A declaration-time map ``(pathway, stage) -> ProcedureDeclaration``, framework-assigned.

    Build it once at startup beside the pathway registry: for each stage that is a learnable
    procedure, ``declare(pathway, stage, problem_type=...)``. The projector then looks up a
    committed step's ``(pathway_id, stage_name)`` to decide whether it is a trial and, if so, with
    what ``(procedure_id, problem_type)``.

    Re-declaring the SAME ``(pathway, stage)`` with the SAME ``problem_type`` is an idempotent
    no-op (the minted declaration is identical); re-declaring it with a DIFFERENT ``problem_type``
    raises — a single ``(pathway, stage)`` scoring against two problem types is an ambiguity the
    projector cannot resolve from a step record, so it is rejected at declaration (fail-loud, S9).
    """

    def __init__(self) -> None:
        self._declarations: dict[tuple[str, str], ProcedureDeclaration] = {}

    def declare(self, pathway: str, stage: str, *, problem_type: str) -> ProcedureDeclaration:
        """Register ``(pathway, stage)`` as a procedure scored against ``problem_type``.

        ``problem_type`` is the human-supplied LABEL; it is minted to a deterministic id via
        :func:`problem_type_id_for`. The procedure id is minted from ``(pathway, stage)`` via
        :func:`procedure_id_for`. Returns the resolved :class:`ProcedureDeclaration`.
        """
        declaration = ProcedureDeclaration(
            pathway=pathway,
            stage=stage,
            procedure_id=procedure_id_for(pathway, stage),
            problem_type=problem_type_id_for(problem_type),
        )
        key = (pathway, stage)
        existing = self._declarations.get(key)
        if existing is not None and existing != declaration:
            raise ValueError(
                f"procedure ({pathway!r}, {stage!r}) is already declared against a different "
                f"problem type (have {existing.problem_type!r}, got {declaration.problem_type!r}); "
                "a (pathway, stage) scores against exactly one problem type"
            )
        self._declarations[key] = declaration
        return declaration

    def get(self, pathway: str, stage: str) -> ProcedureDeclaration | None:
        """Return the declaration for ``(pathway, stage)``, or ``None`` if the stage is not a
        declared procedure (ordinary control flow the projector skips)."""
        return self._declarations.get((pathway, stage))

    def __contains__(self, key: tuple[str, str]) -> bool:
        return key in self._declarations

    def __iter__(self) -> Iterator[ProcedureDeclaration]:
        return iter(self._declarations.values())

    def __len__(self) -> int:
        return len(self._declarations)
