"""Pure F2 routing heart + F6 loop accumulator for the Pod 4.3 thesis/antithesis dialectic.

This module is entirely **pure** — no I/O, no model calls, no substrate writes. It holds:

* :class:`DialecticAccumulator` — the loop's epistemic state derived at read from committed steps.
* :func:`derive_accumulator` — reconstruct the accumulator from a :class:`RunState` (crash-safe).
* :class:`DialecticRoute` — discriminated union: one of the five :class:`FailureOutcome` values OR
  the sentinel :data:`REFINE` (a sixth, dialectic-local outcome that maps to ``Transition``).
* :func:`route_dialectic` — the pure, priority-ordered routing heart (the F2 boundary).

---

Caller mapping (for the EvaluateStage author)
---------------------------------------------
``route_dialectic`` returns a ``DialecticRoute``.  The ``EvaluateStage`` maps it as follows:

.. code-block:: python

    route = route_dialectic(...)
    if route is REFINE:
        return Transition(to="thesis", output=audit_artifact)
    # route is a FailureOutcome — but NEVER call route_failure on the success outcome here:
    if route is FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS:
        # LANDMINE: route_failure maps this to disposition="done" → Done.
        # Done at EvaluateStage COMPLETES the run before ConcludeStage runs (engine.py:855-859),
        # bypassing the H5 single-synthesis-point + conclude re-gate entirely.
        # CORRECT mapping: wrap as Transition(to="conclude"), kind="transition".
        return Transition(to="conclude", output=audit_artifact)
    decision = route_failure(route)
    if decision.disposition == "await-human":
        return AwaitHuman(question=..., to="conclude", output=audit_artifact)
    # disposition == "degraded":
    return Degraded(reason=decision.reason, to="conclude", output=audit_artifact)

The success outcome MUST become ``Transition(to="conclude")``, **not** ``Done``.
ConcludeStage then discriminates by the committed routing step's ``result.kind`` and
``result.to == "conclude"`` (``kind=="transition"`` → exec-success predicate re-check → ``Done``).

---

S9 invariant (mutation-resistance — the spike asserts this)
-----------------------------------------------------------
``route_dialectic`` reads ONLY structural bits:

* ``oracle.holds``, ``oracle.valid_check``, ``oracle.is_executable``
* ``antithesis.disposition``
* ``acc.cycle_index``
* ``acc.thesis_texts`` (for the Jaccard stuck-detector — token-set overlap, never semantics)
* ``thesis_abstained: bool``, ``budget_exhausted: bool``

It NEVER reads:

* ``antithesis.confidence`` (a float heuristic — model self-report)
* ``antithesis.breakage`` or ``oracle.reasoning`` (audit/log text — S9)

Mutating ``confidence``, ``breakage``, or ``reasoning`` arbitrarily produces an **identical**
routing decision.  The spike mutation-tests this invariant exhaustively.

---

Honest documentation (plan §4, red-teamer Attack 1)
----------------------------------------------------
Routing is self-report-free ONLY when an executable oracle (``oracle.is_executable``) is in play.
Behind the LLM-judge fallback (``source=="inference"``, ``is_executable==False``), routing reads
the STRUCTURAL ``is_executable`` bit to **downgrade** a judge "pass" to honestly-incomplete
(``UNVERIFIABLE → Degraded``).  The module does NOT claim general self-report-freedom —
it claims S9-clean routing from structural bits given the inputs it receives.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

from pydantic import BaseModel, ConfigDict

from cogworx.verification.contracts import Thesis, Verdict
from cogworx.verification.honest_failure import (
    AntithesisDisposition,
    AntithesisVerdict,
    FailureOutcome,
)

if TYPE_CHECKING:
    from cogworx.substrate.journal import RunState

__all__ = [
    "MAX_CYCLES",
    "REFINE",
    "STUCK_JACCARD",
    "DialecticAccumulator",
    "DialecticRoute",
    "cycle_verdicts",
    "derive_accumulator",
    "jaccard_stuck",
    "route_dialectic",
]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CYCLES: Final[int] = 5
"""Hard cycle ceiling (mirrors tess MAX_ITERATIONS).  When ``cycle_index >= MAX_CYCLES``,
``route_dialectic`` returns ``FailureOutcome.STUCK → AwaitHuman`` — escalation, NOT
self-termination (S11)."""

STUCK_JACCARD: Final[float] = 0.92
"""Jaccard similarity threshold for the stuck-detector.  When the latest two thesis texts share
≥ 0.92 token overlap the refine loop is not making progress → ``FailureOutcome.STUCK``."""


# ---------------------------------------------------------------------------
# REFINE sentinel — the sixth, dialectic-local routing outcome
# ---------------------------------------------------------------------------


class _Refine:
    """Singleton sentinel for the refine outcome (``Transition(to="thesis")``).

    This is the only non-failure outcome ``route_dialectic`` returns besides the five
    :class:`FailureOutcome` values.  It is NOT a :class:`FailureOutcome` and MUST NOT be
    passed to :func:`~cogworx.verification.honest_failure.route_failure`.

    The ``EvaluateStage`` maps ``REFINE`` to ``Transition(to="thesis", output=audit_artifact)``.
    """

    _instance: _Refine | None = None

    def __new__(cls) -> _Refine:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:
        return "REFINE"


REFINE: Final[_Refine] = _Refine()
"""The singleton refine sentinel.  Use ``route is REFINE`` to identify the refine outcome."""

DialecticRoute = FailureOutcome | _Refine
"""Discriminated union: one of the five :class:`FailureOutcome` enum values OR :data:`REFINE`.

Caller pattern::

    route = route_dialectic(...)
    if route is REFINE:
        ...  # Transition(to="thesis")
    elif route is FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS:
        ...  # Transition(to="conclude")  — NOT Done; see module docstring LANDMINE
    else:
        decision = route_failure(route)
        ...  # AwaitHuman or Degraded, both to="conclude"
"""


# ---------------------------------------------------------------------------
# F6 Accumulator
# ---------------------------------------------------------------------------


class DialecticAccumulator(BaseModel):
    """The refine loop's epistemic state, derived at read from committed journal steps (F6).

    Crash-correctness (architect R5): this is a PURE function of the durable committed steps.
    A cold resume on a fresh ``EvaluateStage`` recomputes the IDENTICAL accumulator with no
    model re-call — the headline S6 spike assertion for Pod 4.3.

    ``breakage_history`` includes the ``breakage`` string from each committed antithesis step
    where ``disposition == BROKE`` (i.e. ``breakage is not None``).  Non-broke / abstained
    steps contribute ``None`` omitted from the tuple — only actual breakage strings are
    collected.  This keeps the Jaccard detector's ``thesis_texts`` parallel to the cycles, and
    the tuple length equals the number of cycles where the antithesis actually broke something.
    The ``EvaluateStage`` uses ``breakage_history`` to build the refinement prompt for
    ``ThesisStage``; it does NOT use it as a routing signal (S9 — breakage is audit/text).

    ``thesis_texts`` is parallel to committed thesis steps (one entry per cycle), ordered
    commit-first.  The stuck-detector compares the latest two entries; < 2 entries → not stuck.
    """

    model_config = ConfigDict(frozen=True)

    cycle_index: int
    """Count of committed antithesis steps.  Because experiment→antithesis commit BEFORE
    evaluate runs, ``cycle_index`` counts the CURRENT cycle's antithesis (already committed)."""

    breakage_history: tuple[str, ...]
    """Breakage strings from committed antithesis steps where ``disposition == BROKE``.
    Each entry is the ``breakage`` text (never ``None`` — only BROKE verdicts contribute)."""

    thesis_texts: tuple[str, ...]
    """``proposed_solution`` from each committed thesis step, in commit order.
    Used by the Jaccard stuck-detector (structural token overlap, never semantic)."""


def derive_accumulator(run_state: RunState) -> DialecticAccumulator:
    """Derive the :class:`DialecticAccumulator` from committed steps in ``run_state`` (pure).

    Walk ``run_state.steps`` in commit order (the tuple is already ordered by ``step_index``
    which is monotonically assigned by the engine):

    * ``cycle_index`` — count of committed ``"antithesis"`` steps.
    * ``breakage_history`` — ``breakage`` text from each antithesis step where the reconstructed
      :class:`~cogworx.verification.honest_failure.AntithesisVerdict` has
      ``disposition == BROKE`` (only BROKE verdicts carry non-``None`` breakage, enforced by the
      cross-field validator at construction).
    * ``thesis_texts`` — ``proposed_solution`` from each committed ``"thesis"`` step.

    Reconstruction idiom mirrors ``runtime/projector.py``:
    ``output.data`` is the plain ``dict`` serialised by ``model_dump(mode="json")``; reconstruct
    with ``model_validate``.  An ``output`` attribute that is ``None`` (``AwaitHuman.output``) or
    whose ``kind`` does not match the expected contract is silently skipped — defensive only at
    the substrate boundary (the journal ``data`` is external / persisted state).

    Args:
        run_state: The :class:`~cogworx.substrate.journal.RunState` from
            ``await ctx.journal.load_run(ctx.run_id)``.

    Returns:
        A frozen :class:`DialecticAccumulator` reflecting ALL committed thesis/antithesis steps.
    """
    cycle_index = 0
    breakage_list: list[str] = []
    thesis_texts_list: list[str] = []

    for step in run_state.steps:
        if step.stage_name == "antithesis":
            output: Any = getattr(step.result, "output", None)
            if output is None:
                continue
            if output.kind != "antithesis-verdict":
                continue
            try:
                av = AntithesisVerdict.model_validate(output.data)
            except Exception:
                # Corrupt journal row at the substrate boundary — skip rather than crash.
                continue
            cycle_index += 1
            if av.disposition is AntithesisDisposition.BROKE and av.breakage is not None:
                breakage_list.append(av.breakage)

        elif step.stage_name == "thesis":
            output = getattr(step.result, "output", None)
            if output is None:
                continue
            if output.kind != "thesis":
                continue
            try:
                thesis = Thesis.model_validate(output.data)
            except Exception:
                continue
            thesis_texts_list.append(thesis.proposed_solution)

    return DialecticAccumulator(
        cycle_index=cycle_index,
        breakage_history=tuple(breakage_list),
        thesis_texts=tuple(thesis_texts_list),
    )


def cycle_verdicts(
    run_state: RunState, before_index: int
) -> tuple[Verdict | None, AntithesisVerdict | None]:
    """Extract the cycle-coupled oracle and antithesis verdicts for the cycle ending at
    ``before_index``.

    Window definition: scan ``reversed(run_state.steps)`` considering only steps with
    ``step_index < before_index``; stop at the first ``"thesis"`` step encountered going backward
    (that closes the cycle's boundary). Within that window, take:

    * the most-recent ``"oracle-verdict"`` step's :class:`~cogworx.verification.contracts.Verdict`
    * the most-recent ``"antithesis-verdict"`` step's
      :class:`~cogworx.verification.honest_failure.AntithesisVerdict`

    Both are reconstructed via ``model_validate(output.data)`` exactly as ``derive_accumulator``
    and the projector do — defensive at the substrate boundary, silently skipping corrupt rows.
    ``verifiable_claim`` is stripped before validation (it is stored inline in artifact data but is
    not part of the Verdict/AntithesisVerdict model schemas).

    Returns ``(oracle, antithesis)`` — ``None`` for either if absent in the window.  A ``None``
    in either position means the routing step's cycle did not commit both verdicts, which is a
    corrupt or partially-committed journal state; callers treat it as a failed predicate
    (fail-closed, S8).

    This function is pure — no I/O, no model calls.  A cold resume re-derives the identical
    window and verdict pair from the same committed steps with no model re-call (S6).

    The ``before_index`` anchor is the routing step's ``step_index`` (already durable in the
    journal), making this a pure function of durable committed state.

    Args:
        run_state: The :class:`~cogworx.substrate.journal.RunState` from
            ``await ctx.journal.load_run(ctx.run_id)``.
        before_index: The ``step_index`` of the routing step (the evaluate step that produced
            ``result.to == "conclude"``).  Only steps with ``step_index < before_index`` are
            considered.

    Returns:
        A ``(Verdict | None, AntithesisVerdict | None)`` pair for the cycle immediately preceding
        the routing step.
    """
    oracle_verdict: Verdict | None = None
    antithesis_verdict: AntithesisVerdict | None = None

    for step in reversed(run_state.steps):
        if step.step_index >= before_index:
            continue

        if step.stage_name == "thesis":
            # The first "thesis" step we hit going backward closes the cycle window.
            break

        output: Any = getattr(step.result, "output", None)
        if output is None:
            continue

        if oracle_verdict is None and output.kind == "oracle-verdict":
            try:
                data = {k: v for k, v in output.data.items() if k != "verifiable_claim"}
                oracle_verdict = Verdict.model_validate(data)
            except Exception:
                pass

        if antithesis_verdict is None and output.kind == "antithesis-verdict":
            try:
                data = {k: v for k, v in output.data.items() if k != "verifiable_claim"}
                antithesis_verdict = AntithesisVerdict.model_validate(data)
            except Exception:
                pass

        if oracle_verdict is not None and antithesis_verdict is not None:
            break

    return oracle_verdict, antithesis_verdict


# ---------------------------------------------------------------------------
# Jaccard stuck-detector (pure, structural)
# ---------------------------------------------------------------------------


def jaccard_stuck(thesis_texts: tuple[str, ...]) -> bool:
    """Return ``True`` when the latest two thesis texts are ≥ ``STUCK_JACCARD`` similar.

    Similarity is computed as standard Jaccard over whitespace-tokenised token SETS
    (duplicates collapsed, case-preserved).  If fewer than 2 texts are present the loop
    cannot have converged yet → always returns ``False``.

    This is a structural signal (set overlap), never a semantic one — the router reads token
    overlap, not meaning (S9: no model call).

    Args:
        thesis_texts: Committed thesis ``proposed_solution`` strings, commit-order.

    Returns:
        ``True`` iff the two most-recent thesis texts are ≥ :data:`STUCK_JACCARD` similar.
    """
    if len(thesis_texts) < 2:
        return False
    a_tokens = set(thesis_texts[-2].split())
    b_tokens = set(thesis_texts[-1].split())
    union = a_tokens | b_tokens
    if not union:
        # Both texts are empty — treat as maximally similar (stuck).
        return True
    intersection = a_tokens & b_tokens
    similarity = len(intersection) / len(union)
    return similarity >= STUCK_JACCARD


# ---------------------------------------------------------------------------
# F2 routing heart — route_dialectic (pure, priority-ordered)
# ---------------------------------------------------------------------------


def route_dialectic(
    *,
    oracle: Verdict,
    antithesis: AntithesisVerdict,
    acc: DialecticAccumulator,
    thesis_abstained: bool,
    budget_exhausted: bool,
) -> DialecticRoute:
    """The pure, priority-ordered routing heart for the Pod 4.3 dialectic (F2 boundary).

    Reads ONLY structural bits — never ``antithesis.confidence``, never ``antithesis.breakage``,
    never ``oracle.reasoning`` (those are audit/log fields; S9).  Mutating any of those fields
    arbitrarily produces an **identical** routing decision (spike-asserted, mutation-resistant).

    Priority order (first match wins):

    1. ``thesis_abstained`` → :attr:`FailureOutcome.ABSTAIN` → AwaitHuman.
       The thesis could not form a verifiable claim; escalate to a human rather than fabricate.

    2. ``budget_exhausted`` → :attr:`FailureOutcome.OVER_BUDGET` → AwaitHuman.
       S11: the dialectic MUST NOT self-terminate on cost; the ``BudgetGuard`` is the structural
       trigger; a human decides whether to continue under a fresh budget.

    3. ``acc.cycle_index >= MAX_CYCLES`` OR Jaccard stuck-detector trips →
       :attr:`FailureOutcome.STUCK` → AwaitHuman.
       Hard ceiling (S11 — not the global ``max_steps`` guillotine) or no-progress detector.

    4. **Success (the ONLY Done path):**
       ``oracle.holds AND oracle.valid_check AND oracle.is_executable AND
       antithesis.disposition == COULD_NOT_BREAK`` →
       :attr:`FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS`.

       **LANDMINE — MUST be mapped to ``Transition(to="conclude")``, NEVER ``Done``.**
       See the module docstring and :data:`DialecticRoute` for the correct caller mapping.

    5. ``oracle.valid_check is False`` → :attr:`FailureOutcome.UNVERIFIABLE` → Degraded.
       No executable oracle ran / the oracle declared the check invalid.

    6. **F2 boundary:** ``oracle.holds AND oracle.valid_check AND NOT oracle.is_executable``
       (an LLM-judge "pass") → :attr:`FailureOutcome.UNVERIFIABLE` → Degraded.
       An LLM-judge is a heuristic PRIORITIZER, not a verifier — it MUST NOT by itself emit a
       confident "verified."  Routing is self-report-free here only insofar as the structural
       ``is_executable`` bit (set by the oracle from its OWN nature, never model output) gates
       the outcome.

    7. **Refine:** else (antithesis ``BROKE``, or executable oracle ``¬holds`` — a real
       refutation) → :data:`REFINE` → ``Transition(to="thesis")``.

    Args:
        oracle: The :class:`~cogworx.verification.contracts.Verdict` from ``ExperimentStage``
            (reconstructed from the committed ``"experiment"`` step's artifact data).
        antithesis: The :class:`~cogworx.verification.honest_failure.AntithesisVerdict` from
            ``AntithesisStage`` (reconstructed from the committed ``"antithesis"`` step).
        acc: The :class:`DialecticAccumulator` derived by :func:`derive_accumulator`.
        thesis_abstained: ``True`` when the committed thesis step carries
            ``Thesis.verifiable_claim is None`` (honest abstention, Pod 4.2).
        budget_exhausted: ``True`` when ``ctx.budget.remaining_usd()`` is insufficient for
            another cycle.  Derived by ``EvaluateStage`` from the structural ``BudgetGuard``;
            NOT a model self-report.

    Returns:
        A :data:`DialecticRoute`: one of the five :class:`FailureOutcome` values or
        :data:`REFINE`.
    """
    # Rule 1 — honest abstention.
    if thesis_abstained:
        return FailureOutcome.ABSTAIN

    # Rule 2 — budget ceiling (S11: structural guard, not self-report).
    if budget_exhausted:
        return FailureOutcome.OVER_BUDGET

    # Rule 3 — hard cycle ceiling OR Jaccard no-progress detector.
    if acc.cycle_index >= MAX_CYCLES or jaccard_stuck(acc.thesis_texts):
        return FailureOutcome.STUCK

    # Rule 4 — SUCCESS (the only Done path).
    # Requires an EXECUTABLE oracle (not a model judge) — the F2 gate.
    if (
        oracle.holds
        and oracle.valid_check
        and oracle.is_executable
        and antithesis.disposition is AntithesisDisposition.COULD_NOT_BREAK
    ):
        return FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS

    # Rule 5 — no valid check (oracle declared the experiment invalid).
    if not oracle.valid_check:
        return FailureOutcome.UNVERIFIABLE

    # Rule 6 — F2 boundary: model-judge "pass" is not verifiable.
    # oracle.holds AND oracle.valid_check AND NOT oracle.is_executable
    if oracle.holds and oracle.valid_check and not oracle.is_executable:
        return FailureOutcome.UNVERIFIABLE

    # Rule 7 — refute/refine: antithesis BROKE, or executable oracle ¬holds.
    return REFINE
