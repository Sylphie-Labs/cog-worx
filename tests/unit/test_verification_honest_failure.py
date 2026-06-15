"""Pod 4.2 — honest-failure primitives: taxonomy, abstention validators, routing decision.

Deterministic and model-free. Three primitive suites:

1. ``classify_error`` — each error class is correct; PERMANENT errors are NOT retryable;
   ``BudgetExceededError`` is permanent (never retried, S11).
2. ``AntithesisVerdict`` validators — incoherent verdicts (broke-without-breakage,
   non-broke-with-breakage, full-confidence-without-oracle) are REJECTED at construction.
3. ``route_failure`` — each ``FailureOutcome`` maps to the correct ``StageResult`` kind;
   ``OVER_BUDGET`` / ``STUCK`` / ``ABSTAIN`` → ``"await-human"``; ``UNVERIFIABLE`` →
   ``"degraded"``; ``COULD_NOT_BREAK_AND_ORACLE_PASS`` → ``"done"``.

``asyncio_mode = "auto"`` (pyproject.toml) — no ``@pytest.mark.asyncio`` needed.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cogworx.cost.budget import BudgetExceededError
from cogworx.verification.honest_failure import (
    RETRYABLE_CLASSES,
    TRANSIENT_EXCEPTION_TYPES,
    AntithesisDisposition,
    AntithesisVerdict,
    ErrorClass,
    FailureOutcome,
    RoutingDecision,
    classify_error,
    route_failure,
)

# ---------------------------------------------------------------------------
# 1. Error taxonomy — classify_error
# ---------------------------------------------------------------------------


class _StubTransientError(ConnectionError):
    """A transient subclass (ConnectionError is in _TRANSIENT_TYPES)."""


class _StubIOError(OSError):
    """OSError covers socket/IO — transient."""


class _StubPermamentValidation(ValueError):
    """Simulates a model-output parsing failure (permanent)."""


class _StubHallucinatedKey(KeyError):
    """Simulates a hallucinated tool name / missing field (permanent)."""


class _StubUnexpected(RuntimeError):
    """Unknown exception class — should classify as 'degraded'."""


def test_timeout_is_transient() -> None:
    assert classify_error(TimeoutError()) == "transient"


def test_connection_error_is_transient() -> None:
    assert classify_error(ConnectionError()) == "transient"


def test_oserror_is_transient() -> None:
    assert classify_error(OSError()) == "transient"


def test_subclass_of_transient_is_transient() -> None:
    assert classify_error(_StubTransientError()) == "transient"


def test_valueerror_is_permanent() -> None:
    assert classify_error(ValueError("bad output")) == "permanent"


def test_keyerror_is_permanent() -> None:
    assert classify_error(KeyError("missing_tool")) == "permanent"


def test_attributeerror_is_permanent() -> None:
    assert classify_error(AttributeError("no such attr")) == "permanent"


def test_typeerror_is_permanent() -> None:
    assert classify_error(TypeError("wrong type")) == "permanent"


def test_budget_exceeded_is_permanent() -> None:
    # BudgetExceededError — retrying immediately trips the same guard again (S11).
    assert classify_error(BudgetExceededError("ceiling")) == "permanent"


def test_unknown_exception_is_degraded() -> None:
    assert classify_error(_StubUnexpected()) == "degraded"


def test_runtime_error_is_degraded() -> None:
    assert classify_error(RuntimeError("surprise")) == "degraded"


def test_only_transient_class_is_retryable() -> None:
    """PERMANENT and DEGRADED errors must NOT enter the retry machine (S11 / biz-firm F4)."""
    all_classes: list[ErrorClass] = ["transient", "permanent", "degraded"]
    for cls in all_classes:
        if cls == "transient":
            assert cls in RETRYABLE_CLASSES
        else:
            assert cls not in RETRYABLE_CLASSES


def test_transient_exception_types_are_all_transient() -> None:
    """Every type in TRANSIENT_EXCEPTION_TYPES classifies as transient (constant is coherent)."""
    for exc_type in TRANSIENT_EXCEPTION_TYPES:
        try:
            exc = exc_type()
        except TypeError:
            # Some OSError subclasses require args; use a generic message.
            exc = exc_type("msg")
        assert classify_error(exc) == "transient", f"{exc_type} should be transient"


def test_permanent_errors_are_not_in_retryable_types() -> None:
    """PERMANENT exception types must NOT appear in TRANSIENT_EXCEPTION_TYPES."""
    permanent_types = (AttributeError, KeyError, TypeError, ValueError, BudgetExceededError)
    for pt in permanent_types:
        assert pt not in TRANSIENT_EXCEPTION_TYPES, (
            f"{pt.__name__} is permanent but appears in TRANSIENT_EXCEPTION_TYPES"
        )


# ---------------------------------------------------------------------------
# 2. AntithesisVerdict — typed abstention + cross-field validators
# ---------------------------------------------------------------------------


def test_broke_with_breakage_is_valid() -> None:
    v = AntithesisVerdict(
        disposition=AntithesisDisposition.BROKE,
        breakage="Step 3 contradicts premise A.",
        confidence=0.8,
    )
    assert v.disposition is AntithesisDisposition.BROKE
    assert v.breakage == "Step 3 contradicts premise A."


def test_could_not_break_with_no_breakage_is_valid() -> None:
    v = AntithesisVerdict(disposition=AntithesisDisposition.COULD_NOT_BREAK)
    assert v.disposition is AntithesisDisposition.COULD_NOT_BREAK
    assert v.breakage is None


def test_abstained_with_no_breakage_is_valid() -> None:
    v = AntithesisVerdict(disposition=AntithesisDisposition.ABSTAINED)
    assert v.disposition is AntithesisDisposition.ABSTAINED
    assert v.breakage is None


def test_broke_without_breakage_is_rejected() -> None:
    """The core tess invariant: a 'broke' verdict without evidence is incoherent (S9)."""
    with pytest.raises(ValidationError, match="BROKE requires non-empty breakage"):
        AntithesisVerdict(disposition=AntithesisDisposition.BROKE)


def test_broke_with_empty_string_breakage_is_rejected() -> None:
    """Empty string breakage is structurally equivalent to no breakage."""
    with pytest.raises(ValidationError, match="BROKE requires non-empty breakage"):
        AntithesisVerdict(disposition=AntithesisDisposition.BROKE, breakage="")


def test_could_not_break_with_breakage_is_rejected() -> None:
    """Non-broke disposition + breakage text is incoherent: contradictory signals."""
    with pytest.raises(ValidationError, match="must not carry breakage"):
        AntithesisVerdict(
            disposition=AntithesisDisposition.COULD_NOT_BREAK,
            breakage="This shouldn't be here",
        )


def test_abstained_with_breakage_is_rejected() -> None:
    """ABSTAINED + breakage is the same contradiction as COULD_NOT_BREAK + breakage."""
    with pytest.raises(ValidationError, match="must not carry breakage"):
        AntithesisVerdict(
            disposition=AntithesisDisposition.ABSTAINED,
            breakage="Contradiction.",
        )


def test_full_confidence_without_oracle_is_rejected() -> None:
    """confidence >= 1.0 without oracle_backed is an S9 / F2 violation."""
    with pytest.raises(ValidationError, match="oracle_backed=True"):
        AntithesisVerdict(
            disposition=AntithesisDisposition.COULD_NOT_BREAK,
            confidence=1.0,
            oracle_backed=False,
        )


def test_full_confidence_with_oracle_backed_is_valid() -> None:
    """confidence=1.0 is permitted when an executable oracle confirmed the outcome (F2)."""
    v = AntithesisVerdict(
        disposition=AntithesisDisposition.COULD_NOT_BREAK,
        confidence=1.0,
        oracle_backed=True,
    )
    assert v.confidence == 1.0
    assert v.oracle_backed is True


def test_confidence_just_below_1_without_oracle_is_valid() -> None:
    """The threshold is >= 1.0; 0.99 is allowed without oracle_backed."""
    v = AntithesisVerdict(
        disposition=AntithesisDisposition.COULD_NOT_BREAK,
        confidence=0.99,
        oracle_backed=False,
    )
    assert v.confidence == 0.99


def test_could_not_break_is_equally_valued_not_penalized() -> None:
    """COULD_NOT_BREAK must construct without error — it is a first-class positive signal."""
    v = AntithesisVerdict(
        disposition=AntithesisDisposition.COULD_NOT_BREAK,
        oracle_backed=False,
        confidence=0.7,
    )
    assert v.disposition is AntithesisDisposition.COULD_NOT_BREAK


def test_antithesis_verdict_is_frozen() -> None:
    v = AntithesisVerdict(disposition=AntithesisDisposition.ABSTAINED)
    with pytest.raises(ValidationError):
        v.disposition = AntithesisDisposition.BROKE  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. route_failure — honest routing decision (pure function)
# ---------------------------------------------------------------------------


def test_abstain_routes_to_await_human() -> None:
    decision = route_failure(FailureOutcome.ABSTAIN)
    assert decision.disposition == "await-human"


def test_stuck_routes_to_await_human() -> None:
    decision = route_failure(FailureOutcome.STUCK)
    assert decision.disposition == "await-human"


def test_over_budget_routes_to_await_human() -> None:
    """S11 / CANON S11: the dialectic must NOT self-terminate on cost."""
    decision = route_failure(FailureOutcome.OVER_BUDGET)
    assert decision.disposition == "await-human"


def test_unverifiable_routes_to_degraded() -> None:
    decision = route_failure(FailureOutcome.UNVERIFIABLE)
    assert decision.disposition == "degraded"


def test_could_not_break_and_oracle_pass_routes_to_done() -> None:
    """The dialectic success condition: antithesis could_not_break ∧ oracle holds."""
    decision = route_failure(FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS)
    assert decision.disposition == "done"


def test_routing_decision_is_frozen() -> None:
    decision = route_failure(FailureOutcome.ABSTAIN)
    with pytest.raises(ValidationError):
        decision.disposition = "done"  # type: ignore[misc]


def test_routing_decision_reason_is_non_empty() -> None:
    """Each outcome carries a non-empty audit reason (log only — never parsed for control)."""
    for outcome in FailureOutcome:
        decision = route_failure(outcome)
        assert decision.reason, f"FailureOutcome.{outcome.name} produced an empty reason"


def test_routing_covers_all_outcomes() -> None:
    """route_failure must handle every FailureOutcome — exhaustive by the routing table."""
    for outcome in FailureOutcome:
        decision = route_failure(outcome)
        assert isinstance(decision, RoutingDecision)


def test_await_human_outcomes_are_exactly_three() -> None:
    """abstain + stuck + over_budget → await-human; exactly these three, no more, no less."""
    await_human = {o for o in FailureOutcome if route_failure(o).disposition == "await-human"}
    assert await_human == {
        FailureOutcome.ABSTAIN,
        FailureOutcome.STUCK,
        FailureOutcome.OVER_BUDGET,
    }


def test_no_new_stage_result_kind_needed() -> None:
    """Disposition values are a subset of the five existing StageResult kinds (plan-confirmed)."""
    valid_kinds = {"transition", "done", "await-human", "degraded", "wait"}
    for outcome in FailureOutcome:
        assert route_failure(outcome).disposition in valid_kinds
