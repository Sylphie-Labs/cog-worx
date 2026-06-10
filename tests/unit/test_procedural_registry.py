"""The static procedure registry — declaration-time identity discipline (CANON S9).

Pins the ProcedureRegistry contract: deterministic id minting from (pathway, stage) + problem-type
label, idempotent same-declaration, and the fail-loud rejection of a (pathway, stage) re-declared
against a different problem type (the ambiguity the projector cannot resolve from a step record).
"""

from __future__ import annotations

import pytest

from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for
from cogworx.knowledge.procedural_registry import ProcedureRegistry


def test_declare_mints_deterministic_ids() -> None:
    registry = ProcedureRegistry()
    decl = registry.declare("math_pathway", "solve_stage", problem_type="word problem")

    assert decl.procedure_id == procedure_id_for("math_pathway", "solve_stage")
    assert decl.problem_type == problem_type_id_for("word problem")
    assert decl.pathway == "math_pathway"
    assert decl.stage == "solve_stage"


def test_get_returns_declaration_or_none() -> None:
    registry = ProcedureRegistry()
    registry.declare("math_pathway", "solve_stage", problem_type="word problem")

    assert registry.get("math_pathway", "solve_stage") is not None
    assert registry.get("math_pathway", "unknown_stage") is None
    assert ("math_pathway", "solve_stage") in registry
    assert ("math_pathway", "unknown_stage") not in registry


def test_redeclare_same_problem_type_is_idempotent() -> None:
    registry = ProcedureRegistry()
    a = registry.declare("math_pathway", "solve_stage", problem_type="word problem")
    b = registry.declare("math_pathway", "solve_stage", problem_type="word problem")

    assert a == b
    assert len(registry) == 1


def test_redeclare_different_problem_type_raises() -> None:
    registry = ProcedureRegistry()
    registry.declare("math_pathway", "solve_stage", problem_type="word problem")

    with pytest.raises(ValueError, match="already declared against a different problem type"):
        registry.declare("math_pathway", "solve_stage", problem_type="geometry")


def test_problem_type_label_canonicalises() -> None:
    # NEGATIVE CONTROL on the identity discipline: two labels that canonicalise to the same id are
    # the SAME problem type, so re-declaring a case/space variant is idempotent, not a conflict.
    registry = ProcedureRegistry()
    registry.declare("math_pathway", "solve_stage", problem_type="Word Problem")
    registry.declare("math_pathway", "solve_stage", problem_type="word problem")

    assert len(registry) == 1


def test_distinct_stages_mint_distinct_procedures() -> None:
    registry = ProcedureRegistry()
    a = registry.declare("math_pathway", "solve_stage", problem_type="word problem")
    b = registry.declare("math_pathway", "verify_stage", problem_type="word problem")

    assert a.procedure_id != b.procedure_id
    assert a.problem_type == b.problem_type
    assert {d.stage for d in registry} == {"solve_stage", "verify_stage"}
