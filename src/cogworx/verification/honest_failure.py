"""Honest-failure primitives for the Phase 4 dialectic (CANON S8, S9, S11).

Three pure, model-free primitives composable by the Pod 4.3 ``EvaluateStage``:

1. **Error taxonomy** — classify a failure as ``transient | permanent | degraded`` before retrying
   (port of biz-firm F4). A PERMANENT error (hallucinated tool name, schema/validation mismatch,
   malformed output) MUST NOT be retried; a TRANSIENT one (timeout, rate-limit, transient I/O) MAY.
   Feeds the Pod 1.2 :class:`~cogworx.loop.retry.RetryPolicy` via its exception-type ``retryable``
   allowlist — no breaking change required.

2. **Typed abstention + antithesis cross-field validators** — a thesis may honestly abstain
   (``Thesis.verifiable_claim is None``); the antithesis's ``could_not_break`` is equally valued,
   NOT penalized. Pydantic validators on frozen models structurally forbid incoherent verdicts:
   a "broke"/refuted disposition with NO breakage, or a confident "verified" with no executable
   backing.

3. **Honest-failure routing decision** — a pure function mapping an outcome
   (abstain / stuck / over-budget / unverifiable / ``could_not_break ∧ oracle-pass``) to the
   *name* of the existing :class:`~cogworx.loop.result.StageResult` kind that is honest. No new
   ``StageResult`` kind is introduced (confirmed in the plan). The cost-ceiling path routes a
   budget-guard trip to ``"await-human"`` — the dialectic MUST NOT self-terminate on cost (S11);
   the :class:`~cogworx.cost.budget.BudgetGuard` is the structural trigger.

None of these primitives do I/O; they carry no substrate reference. The Pod 4.3 ``EvaluateStage``
imports them and turns the routing decision into the actual ``StageResult``.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from cogworx.cost.budget import BudgetExceededError

# ---------------------------------------------------------------------------
# 1. Error taxonomy
# ---------------------------------------------------------------------------

ErrorClass = Literal["transient", "permanent", "degraded"]
"""Three-way error classification (biz-firm F4).

- ``"transient"``  — the failure is likely circumstantial; a retry after backoff may succeed
  (timeout, rate-limit, transient I/O error, network blip).
- ``"permanent"``  — the failure is structural; retrying the SAME call will produce the same
  result (hallucinated tool name, schema/validation mismatch, malformed model output, missing
  capability). Retrying wastes calls (S11) and can commit incorrect state (S6).
- ``"degraded"``   — the failure is at the boundary between transient and permanent (e.g. an
  unexpected / unknown error class). Treated as exhaustion by the engine: route to
  ``on_exhausted`` (the S8 default is ``"degraded"``), not re-queued for retry.
"""

# Exception types that are structurally transient: the engine-level causes the RetryPolicy
# already handles (TimeoutError) plus the explicit runtime variants below.
# This list intentionally matches the class of exceptions a developer would put in
# ``RetryPolicy.retryable`` — the classifier produces the class; the policy uses the type.
_TRANSIENT_TYPES: Final[tuple[type[BaseException], ...]] = (
    TimeoutError,
    ConnectionError,
    OSError,  # includes socket/IO errors on most platforms
)

# Exception types that signal a structural / permanent failure.
# ValidationError (Pydantic) covers schema mismatch, malformed output.
# AttributeError / KeyError arise from hallucinated tool-name / missing-field access.
# ValueError from model-output parsing failures (wrong shape, type coercion).
_PERMANENT_TYPES: Final[tuple[type[BaseException], ...]] = (
    AttributeError,
    KeyError,
    TypeError,
    ValueError,
)


def classify_error(exc: BaseException) -> ErrorClass:
    """Return the :data:`ErrorClass` for *exc* WITHOUT inspecting its message text (S9).

    Classification is STRUCTURAL: exception *type* only, never string-parsing the model's words
    (mirrors :mod:`~cogworx.loop.retry` — "a type NOT in the allowlist is a BUG and propagates
    loud"). A :class:`~cogworx.cost.budget.BudgetExceededError` is ``"permanent"`` — retrying
    after the budget guard trips will immediately trip it again, so the only honest route is
    escalation to ``AwaitHuman`` (S11).

    Callers (e.g. the Pod 4.3 ``EvaluateStage``) use this to gate ``RetryPolicy.retryable``:
    only ``"transient"`` classes belong in that tuple.
    """
    # Budget-guard trip: escalate, never retry.
    if isinstance(exc, BudgetExceededError):
        return "permanent"

    if isinstance(exc, _TRANSIENT_TYPES):
        return "transient"

    if isinstance(exc, _PERMANENT_TYPES):
        return "permanent"

    # Unknown / unexpected exception: treat as degraded (S8 — degrade, do not fail hard).
    return "degraded"


# The set of error classes the RetryPolicy should allow to re-queue. Permanent and degraded
# must NOT enter the retry machine. Expose as a constant so callers can build the allowlist
# without re-spelling the rule.
RETRYABLE_CLASSES: Final[frozenset[ErrorClass]] = frozenset({"transient"})

# The exception types that classify_error routes to "transient" — ready to pass verbatim to
# RetryPolicy(retryable=TRANSIENT_EXCEPTION_TYPES) without a breaking change to RetryPolicy.
TRANSIENT_EXCEPTION_TYPES: Final[tuple[type[BaseException], ...]] = _TRANSIENT_TYPES


# ---------------------------------------------------------------------------
# 2. Typed abstention + antithesis verdict model with cross-field validators
# ---------------------------------------------------------------------------


class AntithesisDisposition(StrEnum):
    """The adversarial challenger's verdict on the thesis.

    ``COULD_NOT_BREAK`` is structurally distinct from ``BROKE`` and EQUALLY valued (not penalized)
    — an honest "I tried and the thesis survived" is a positive signal (``antithesis_survival``
    evidence). An honest abstention (``ABSTAINED``) is valued over a fabricated break.
    """

    BROKE = "broke"
    """The antithesis found a concrete flaw and has the breakage to back it up."""

    COULD_NOT_BREAK = "could_not_break"
    """The antithesis tried in earnest and the thesis survived. Equally valued, not penalized."""

    ABSTAINED = "abstained"
    """The antithesis could not form a meaningful challenge (e.g. no verifiable claim to attack).
    Honest abstention; does not drive refinement."""


class AntithesisVerdict(BaseModel):
    """The antithesis stage's typed output — cross-field validators enforce coherence (S9).

    Two structural invariants, both enforced at construction:

    1. ``BROKE`` without ``breakage`` is incoherent: asserting a flaw without evidence is a
       model self-report masquerading as a structural signal (S9 violation).
    2. A ``confidence >= 1.0`` "verified" disposition without ``oracle_backed`` is incoherent:
       full confidence is only warranted when an executable oracle (not the model) backed the
       claim (F2 — mirrors the ``Verdict.is_executable`` boundary on the oracle side).

    ``breakage`` is audit/evidence text, never a control signal by itself.
    ``oracle_backed`` is set TRUE only when an executable oracle confirmed the break — never
    from model self-report (S9; same discipline as ``Verdict.source``).
    """

    model_config = ConfigDict(frozen=True)

    disposition: AntithesisDisposition
    breakage: str | None = None
    """Concrete description of the flaw found. REQUIRED when ``disposition == BROKE``; MUST be
    ``None`` when the antithesis did not break anything (cross-field validator enforces this)."""

    confidence: float = 0.5
    """How confident the antithesis is in its assessment (0.0-1.0). MUST NOT reach 1.0 without
    ``oracle_backed=True`` — full confidence is only warranted by an external structural check
    (F2 / S9). Control: routing reads ``disposition``, NOT ``confidence``; ``confidence`` is a
    heuristic weight, never the decision gate."""

    oracle_backed: bool = False
    """Set TRUE only when an EXECUTABLE oracle confirmed this outcome — never from model
    output (S9). Required for ``confidence >= 1.0`` (cross-field validator enforces)."""

    @model_validator(mode="after")
    def _enforce_broke_requires_breakage(self) -> AntithesisVerdict:
        """A ``BROKE`` disposition MUST supply ``breakage`` text.

        Without it the verdict is a structural contradiction: "I broke it" + no evidence.
        This is the tess "cross-field validator that structurally forbids a broke verdict with
        no breakage" pattern.
        """
        if self.disposition is AntithesisDisposition.BROKE and not self.breakage:
            raise ValueError(
                "AntithesisVerdict: disposition=BROKE requires non-empty breakage; "
                "a 'broke' assertion without evidence is an S9 violation."
            )
        return self

    @model_validator(mode="after")
    def _enforce_non_broke_has_no_breakage(self) -> AntithesisVerdict:
        """A ``COULD_NOT_BREAK`` or ``ABSTAINED`` verdict MUST NOT supply ``breakage``.

        A non-broke disposition with breakage text is incoherent: the antithesis is asserting
        both "I didn't break it" and presenting a flaw. The model should produce one or the
        other; accepting both silently hides the contradiction.
        """
        if self.disposition is not AntithesisDisposition.BROKE and self.breakage is not None:
            raise ValueError(
                f"AntithesisVerdict: disposition={self.disposition.value} must not carry "
                "breakage; only BROKE verdicts supply evidence of a flaw."
            )
        return self

    @model_validator(mode="after")
    def _enforce_full_confidence_requires_oracle(self) -> AntithesisVerdict:
        """``confidence >= 1.0`` requires ``oracle_backed=True`` (F2 / S9).

        Full confidence is only warranted when an executable structural check (not the model's
        self-assessment) confirmed the outcome. A model self-reporting ``confidence=1.0``
        without oracle backing is the exact S9 violation F2 identifies.
        """
        if self.confidence >= 1.0 and not self.oracle_backed:
            raise ValueError(
                "AntithesisVerdict: confidence >= 1.0 requires oracle_backed=True; "
                "full confidence without an executable oracle backing is an S9 violation (F2)."
            )
        return self


# ---------------------------------------------------------------------------
# 3. Honest-failure routing decision
# ---------------------------------------------------------------------------


class FailureOutcome(StrEnum):
    """The honest failure modes that require a routing decision (Pod 4.2 / Pod 4.3).

    These are the failure-side inputs to the routing function. The success-side (``DONE``) is
    included so callers can pass any final outcome to :func:`route_failure` and get a consistent
    decision value back.
    """

    ABSTAIN = "abstain"
    """The thesis honestly abstained (``Thesis.verifiable_claim is None``) — "I cannot establish
    this." Honest; should escalate to a human rather than fabricate."""

    STUCK = "stuck"
    """The refine loop has cycled without progress (Jaccard stuck-detector / hard cycle ceiling).
    Must NOT self-continue (S11); must escalate to human."""

    OVER_BUDGET = "over_budget"
    """The ``BudgetGuard`` tripped (``BudgetExceededError``). The dialectic MUST NOT
    self-terminate on cost (S11 / CANON S11 note: "model cannot self-terminate"). Route to
    ``AwaitHuman`` — a human decides whether to continue under a fresh budget."""

    UNVERIFIABLE = "unverifiable"
    """No executable oracle was available or the oracle returned ``valid_check=False``. The
    answer may be correct but the framework cannot structurally confirm it. Honest route:
    ``Degraded`` — the loop continues but the result is explicitly incomplete."""

    COULD_NOT_BREAK_AND_ORACLE_PASS = "could_not_break_and_oracle_pass"
    """The antithesis ``COULD_NOT_BREAK`` AND the oracle returned ``holds=True`` with
    ``valid_check=True``. This is the dialectic's success condition — route to ``Done``."""


class RoutingDecision(BaseModel):
    """The output of :func:`route_failure`: a named disposition + a human-readable reason.

    ``disposition`` is one of the five :class:`~cogworx.loop.result.StageResult` ``kind`` values
    (``"transition"`` / ``"done"`` / ``"await-human"`` / ``"degraded"`` / ``"wait"``). The Pod 4.3
    ``EvaluateStage`` reads ``disposition`` to construct the actual ``StageResult``; it does NOT
    parse ``reason`` for control flow (S9: reason is audit/log only).
    """

    model_config = ConfigDict(frozen=True)

    disposition: Literal["done", "await-human", "degraded"]
    """The StageResult kind the Pod 4.3 EvaluateStage should construct.

    Mapping (from the plan):
    - abstain | stuck | over_budget  → ``"await-human"``   (durable HITL, S8/S11)
    - unverifiable                   → ``"degraded"``       (honestly incomplete)
    - could_not_break ∧ oracle-pass  → ``"done"``           (dialectic success)
    """

    reason: str
    """Human-readable explanation of the routing decision. AUDIT / LOG ONLY — never parsed for
    control flow (S9)."""


# The canonical routing table — a pure mapping from FailureOutcome to disposition.
# Expressed as a constant so callers can inspect it in tests without calling route_failure.
_ROUTING_TABLE: Final[dict[FailureOutcome, Literal["done", "await-human", "degraded"]]] = {
    FailureOutcome.ABSTAIN: "await-human",
    FailureOutcome.STUCK: "await-human",
    FailureOutcome.OVER_BUDGET: "await-human",
    FailureOutcome.UNVERIFIABLE: "degraded",
    FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS: "done",
}

_ROUTING_REASONS: Final[dict[FailureOutcome, str]] = {
    FailureOutcome.ABSTAIN: (
        "Thesis abstained (verifiable_claim is None); escalating to human rather than fabricating."
    ),
    FailureOutcome.STUCK: (
        "Refine loop is stuck (no progress across cycles or hard ceiling reached); "
        "escalating to human (S11 — must not self-terminate)."
    ),
    FailureOutcome.OVER_BUDGET: (
        "BudgetGuard tripped; the dialectic cannot self-terminate on cost (CANON S11). "
        "Escalating to human to decide whether to continue under a fresh budget."
    ),
    FailureOutcome.UNVERIFIABLE: (
        "No executable oracle available or oracle returned valid_check=False; "
        "answer may be correct but cannot be structurally confirmed (F2/S9). "
        "Routing to Degraded — honestly incomplete."
    ),
    FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS: (
        "Antithesis could_not_break AND oracle holds=True with valid_check=True. "
        "Dialectic success condition met — routing to Done."
    ),
}


def route_failure(outcome: FailureOutcome) -> RoutingDecision:
    """Map a :class:`FailureOutcome` to the honest :class:`RoutingDecision` (pure function).

    This is the single decision point for Phase 4 failure routing. It consumes only structural
    signals (the enum value) — never model text, never a confidence float (S9). The Pod 4.3
    ``EvaluateStage`` calls this and constructs the actual ``StageResult`` from
    ``decision.disposition``.

    The full mapping (confirmed in the plan — no new StageResult kind needed):

    ===================================== ==============
    Outcome                               Disposition
    ===================================== ==============
    ``ABSTAIN``                           ``await-human``
    ``STUCK``                             ``await-human``
    ``OVER_BUDGET``                       ``await-human``
    ``UNVERIFIABLE``                      ``degraded``
    ``COULD_NOT_BREAK_AND_ORACLE_PASS``   ``done``
    ===================================== ==============
    """
    return RoutingDecision(
        disposition=_ROUTING_TABLE[outcome],
        reason=_ROUTING_REASONS[outcome],
    )


__all__ = [
    "RETRYABLE_CLASSES",
    "TRANSIENT_EXCEPTION_TYPES",
    "AntithesisDisposition",
    "AntithesisVerdict",
    "ErrorClass",
    "FailureOutcome",
    "RoutingDecision",
    "classify_error",
    "route_failure",
]
