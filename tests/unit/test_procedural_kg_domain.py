"""Unit tests for the procedural-KG substrate domain types (CANON S3, S5, S6, S9).

Pins the frozen/validated domain model the Protocol carries — NOT the adapter behaviour (the Neo4j
adapter + in-memory double + parity tests are a later task):
  - Procedure / ProblemType / Trial / ScoredProcedure are frozen and round-trip.
  - Trial validates outcome to the success/failure literal and rejects junk.
  - The ProceduralKG Protocol is runtime_checkable and exposes the locked record_trial signature.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogworx.claims.provenance import Provenance
from cogworx.knowledge.procedural_confidence import procedure_success
from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for
from cogworx.substrate.procedural_kg import (
    ProblemType,
    ProceduralKG,
    Procedure,
    ScoredProcedure,
    Trial,
)

_NOW = datetime(2026, 6, 9, tzinfo=UTC)


def _provenance() -> Provenance:
    """A trial's provenance: deterministic control event (S9 / S5)."""
    return Provenance(source="system", confidence=1.0, recorded_at=_NOW)


def _trial(*, run_id: str = "run-1", step_index: int = 0, outcome: str = "success") -> Trial:
    return Trial(
        trial_id=f"{run_id}:{step_index}",
        run_id=run_id,
        step_index=step_index,
        procedure_id=procedure_id_for("solve_math", "decompose"),
        problem_type=problem_type_id_for("math word problem"),
        outcome=outcome,
        occurred_at=_NOW,
        provenance=_provenance(),
    )


# ---------------------------------------------------------------------------
# Trial — frozen, validated, round-trips
# ---------------------------------------------------------------------------


def test_trial_round_trips() -> None:
    trial = _trial()
    assert Trial.model_validate(trial.model_dump()) == trial


def test_trial_is_frozen() -> None:
    trial = _trial()
    with pytest.raises((ValidationError, TypeError)):
        trial.outcome = "failure"  # type: ignore[misc]


def test_trial_accepts_both_outcomes() -> None:
    assert _trial(outcome="success").outcome == "success"
    assert _trial(outcome="failure").outcome == "failure"


def test_trial_rejects_unknown_outcome() -> None:
    """Outcome is a stamped structural fact — only success/failure are legal (S9)."""
    with pytest.raises(ValidationError):
        _trial(outcome="maybe")


def test_trial_id_is_journal_step_pk() -> None:
    """trial_id is f'{run_id}:{step_index}' — the journal step PK (architect P0.2)."""
    trial = _trial(run_id="abc", step_index=7)
    assert trial.trial_id == "abc:7"


# ---------------------------------------------------------------------------
# Procedure / ProblemType — frozen, embedding nullable
# ---------------------------------------------------------------------------


def test_procedure_is_frozen() -> None:
    proc = Procedure(id=procedure_id_for("p", "s"), label="decompose")
    with pytest.raises((ValidationError, TypeError)):
        proc.label = "other"  # type: ignore[misc]


def test_problem_type_embedding_defaults_none() -> None:
    """Recall channels are deferred (Pod 2.5) — embedding is nullable now, default None."""
    pt = ProblemType(id=problem_type_id_for("math"), label="math")
    assert pt.embedding is None


def test_problem_type_accepts_embedding() -> None:
    pt = ProblemType(id=problem_type_id_for("math"), label="math", embedding=(0.1, 0.2, 0.3))
    assert pt.embedding == (0.1, 0.2, 0.3)


def test_problem_type_is_frozen() -> None:
    pt = ProblemType(id=problem_type_id_for("math"), label="math")
    with pytest.raises((ValidationError, TypeError)):
        pt.label = "logic"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# ScoredProcedure — carries the derived posterior + deduped count
# ---------------------------------------------------------------------------


def test_scored_procedure_carries_posterior_and_is_frozen() -> None:
    proc = Procedure(id=procedure_id_for("p", "s"), label="decompose")
    pt = ProblemType(id=problem_type_id_for("math"), label="math")
    success = procedure_success([])
    scored = ScoredProcedure(procedure=proc, problem_type=pt, success=success)
    assert scored.success.n_trials == 0
    assert scored.promoted is False
    with pytest.raises((ValidationError, TypeError)):
        scored.promoted = True  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Protocol shape
# ---------------------------------------------------------------------------


def test_proceduralkg_is_runtime_checkable_protocol() -> None:
    """A non-implementer is NOT an instance (the seam is structurally typed)."""
    assert not isinstance(object(), ProceduralKG)


def test_record_trial_signature_is_keyword_only() -> None:
    """The locked record_trial seam takes keyword-only trial_id/procedure_id/problem_type/
    outcome/occurred_at/provenance plus the optional same-txn cursor advance — the shape the
    projector calls."""
    import inspect

    sig = inspect.signature(ProceduralKG.record_trial)
    params = list(sig.parameters)
    assert params[0] == "self"
    assert set(params[1:]) == {
        "trial_id",
        "procedure_id",
        "problem_type",
        "outcome",
        "occurred_at",
        "provenance",
        "cursor",
    }
    for name in params[1:]:
        assert sig.parameters[name].kind is inspect.Parameter.KEYWORD_ONLY
