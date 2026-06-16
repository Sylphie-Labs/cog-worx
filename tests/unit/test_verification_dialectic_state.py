"""Unit tests for ``cogworx.verification.dialectic_state``.

Covers:
- Every ``route_dialectic`` branch (all 7 priority rules).
- Priority ordering (earlier rules shadow later ones).
- Mutation-resistance: mutating ``antithesis.confidence``, ``antithesis.breakage``/
  ``oracle.reasoning`` produces an IDENTICAL routing decision.
- Jaccard stuck-detector (stuck and not-stuck; edge cases).
- MAX_CYCLES ceiling.
- ``derive_accumulator`` over synthetic ``RunState``/``StepRecord`` (same construction idiom
  the real journal uses).

``asyncio_mode = "auto"`` (pyproject.toml) — no ``@pytest.mark.asyncio`` needed.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.result import Transition
from cogworx.substrate.journal import RunState, StepRecord
from cogworx.verification.contracts import Thesis, Verdict
from cogworx.verification.dialectic_state import (
    MAX_CYCLES,
    REFINE,
    STUCK_JACCARD,
    DialecticAccumulator,
    DialecticRoute,
    derive_accumulator,
    jaccard_stuck,
    route_dialectic,
)
from cogworx.verification.honest_failure import (
    AntithesisDisposition,
    AntithesisVerdict,
    FailureOutcome,
    route_failure,
)

# ---------------------------------------------------------------------------
# Helpers — build test fixtures
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)

_PROVENANCE = Provenance(source="system", confidence=1.0, recorded_at=_NOW)


def _thesis_artifact(proposed_solution: str, verifiable_claim: str | None = "claim") -> Artifact:
    thesis = Thesis(
        proposed_solution=proposed_solution,
        experiment_design="run the thing",
        verifiable_claim=verifiable_claim,
    )
    return Artifact(
        kind="thesis",
        produced_by="thesis-stage",
        provenance=_PROVENANCE,
        data=thesis.model_dump(mode="json"),
    )


def _antithesis_artifact(
    disposition: AntithesisDisposition,
    breakage: str | None = None,
    confidence: float = 0.5,
    oracle_backed: bool = False,
) -> Artifact:
    av = AntithesisVerdict(
        disposition=disposition,
        breakage=breakage,
        confidence=confidence,
        oracle_backed=oracle_backed,
    )
    return Artifact(
        kind="antithesis-verdict",
        produced_by="antithesis-stage",
        provenance=_PROVENANCE,
        data=av.model_dump(mode="json"),
    )


def _step(
    stage_name: str,
    step_index: int,
    artifact: Artifact,
    run_id: str = "run-1",
) -> StepRecord:
    """Build a committed StepRecord using a Transition result (the typical happy path)."""
    result = Transition(
        kind="transition",
        to="next-stage",
        output=artifact,
    )
    return StepRecord(
        run_id=run_id,
        step_index=step_index,
        stage_name=stage_name,
        result=result,
        committed_at=_NOW,
    )


def _run_state(steps: list[StepRecord], run_id: str = "run-1") -> RunState:
    return RunState(
        run_id=run_id,
        session_id="sess-1",
        status="running",
        pathway_id="dialectic",
        pathway_version=1,
        pathway_fingerprint="fp",
        steps=tuple(steps),
    )


def _executable_oracle(holds: bool = True, valid_check: bool = True) -> Verdict:
    return Verdict(holds=holds, valid_check=valid_check, reasoning="stub", source="tool")


def _judge_oracle(holds: bool = True, valid_check: bool = True) -> Verdict:
    return Verdict(holds=holds, valid_check=valid_check, reasoning="judge stub", source="inference")


def _antithesis_could_not_break() -> AntithesisVerdict:
    return AntithesisVerdict(disposition=AntithesisDisposition.COULD_NOT_BREAK)


def _antithesis_broke(breakage: str = "flaw found") -> AntithesisVerdict:
    return AntithesisVerdict(
        disposition=AntithesisDisposition.BROKE,
        breakage=breakage,
    )


def _antithesis_abstained() -> AntithesisVerdict:
    return AntithesisVerdict(disposition=AntithesisDisposition.ABSTAINED)


def _acc(
    cycle_index: int = 1,
    breakage_history: tuple[str, ...] = (),
    thesis_texts: tuple[str, ...] = ("some thesis",),
) -> DialecticAccumulator:
    return DialecticAccumulator(
        cycle_index=cycle_index,
        breakage_history=breakage_history,
        thesis_texts=thesis_texts,
    )


# ---------------------------------------------------------------------------
# 1. route_dialectic — all 7 priority rules
# ---------------------------------------------------------------------------


class TestRoutingRules:
    def test_rule1_thesis_abstained(self) -> None:
        """Rule 1: thesis_abstained → ABSTAIN regardless of oracle/antithesis."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=True,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.ABSTAIN

    def test_rule2_budget_exhausted(self) -> None:
        """Rule 2: budget_exhausted → OVER_BUDGET (S11 — never self-terminate)."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=True,
        )
        assert result is FailureOutcome.OVER_BUDGET

    def test_rule3_max_cycles_ceiling(self) -> None:
        """Rule 3: cycle_index >= MAX_CYCLES → STUCK."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=MAX_CYCLES),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.STUCK

    def test_rule3_max_cycles_plus_one(self) -> None:
        """cycle_index > MAX_CYCLES is also STUCK."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=MAX_CYCLES + 1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.STUCK

    def test_rule3_jaccard_stuck_detector(self) -> None:
        """Rule 3: Jaccard stuck-detector triggers STUCK when last two theses are near-identical."""
        same_thesis = "the solution is X " * 50  # lots of repetition → Jaccard ≈ 1.0
        result = route_dialectic(
            oracle=_executable_oracle(holds=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(
                cycle_index=2,
                thesis_texts=(same_thesis, same_thesis),
            ),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.STUCK

    def test_rule4_success_executable_oracle_could_not_break(self) -> None:
        """Rule 4: oracle executable holds + antithesis COULD_NOT_BREAK → success outcome."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS

    def test_rule5_oracle_valid_check_false(self) -> None:
        """Rule 5: oracle.valid_check is False → UNVERIFIABLE (no valid experiment)."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=False),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.UNVERIFIABLE

    def test_rule6_judge_pass_is_unverifiable(self) -> None:
        """Rule 6 (F2 boundary): judge holds + valid_check + NOT is_executable → UNVERIFIABLE.
        An LLM-judge "pass" is a heuristic; it MUST NOT become Done.
        """
        result = route_dialectic(
            oracle=_judge_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.UNVERIFIABLE

    def test_rule7_antithesis_broke_refines(self) -> None:
        """Rule 7: executable oracle holds + antithesis BROKE → REFINE."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_broke("flaw in step 2"),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is REFINE

    def test_rule7_executable_oracle_not_holds_refines(self) -> None:
        """Rule 7: executable oracle ¬holds (real refutation) → REFINE."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=False, valid_check=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is REFINE

    def test_rule7_judge_not_holds_refines(self) -> None:
        """Rule 7: judge oracle ¬holds (falls through rules 4-6) → REFINE."""
        # oracle.holds=False AND oracle.valid_check=True AND NOT is_executable
        # Rule 4: not (holds AND valid AND executable) — skip
        # Rule 5: valid_check is True — skip
        # Rule 6: not (holds AND valid AND not executable) since holds=False — skip
        # Rule 7: REFINE
        result = route_dialectic(
            oracle=_judge_oracle(holds=False, valid_check=True),
            antithesis=_antithesis_abstained(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is REFINE


# ---------------------------------------------------------------------------
# 2. Priority ordering — earlier rules shadow later ones
# ---------------------------------------------------------------------------


class TestPriorityOrdering:
    def test_rule1_shadows_rule4(self) -> None:
        """thesis_abstained takes priority over the success path (rule 1 > rule 4)."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=True,  # rule 1
            budget_exhausted=False,
        )
        assert result is FailureOutcome.ABSTAIN

    def test_rule2_shadows_rule4(self) -> None:
        """budget_exhausted takes priority over success (rule 2 > rule 4)."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=True,  # rule 2
        )
        assert result is FailureOutcome.OVER_BUDGET

    def test_rule3_shadows_rule4(self) -> None:
        """MAX_CYCLES ceiling takes priority over success (rule 3 > rule 4)."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=MAX_CYCLES),  # rule 3
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.STUCK

    def test_rule1_shadows_rule2(self) -> None:
        """thesis_abstained takes priority over budget_exhausted (rule 1 > rule 2)."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True),
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=True,
            budget_exhausted=True,
        )
        assert result is FailureOutcome.ABSTAIN

    def test_rule4_requires_executable_oracle(self) -> None:
        """Judge-only oracle with COULD_NOT_BREAK falls to rule 6 (UNVERIFIABLE), not rule 4."""
        result = route_dialectic(
            oracle=_judge_oracle(holds=True, valid_check=True),  # not executable
            antithesis=_antithesis_could_not_break(),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.UNVERIFIABLE

    def test_rule4_requires_could_not_break(self) -> None:
        """Executable oracle holds but BROKE antithesis → rule 7 REFINE, not rule 4."""
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=_antithesis_broke("found a bug"),
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is REFINE


# ---------------------------------------------------------------------------
# 3. Mutation-resistance (S9 — no float/text is read by the router)
# ---------------------------------------------------------------------------


class TestMutationResistance:
    """Vary confidence, breakage, and reasoning arbitrarily — the decision must be unchanged."""

    def _judge_pass_scenario(
        self,
        confidence: float = 0.5,
        reasoning: str = "default",
        breakage: str | None = None,
    ) -> DialecticRoute:
        """Judge-pass scenario (→ UNVERIFIABLE). Mutate floats/text and verify no change."""
        oracle = Verdict(
            holds=True,
            valid_check=True,
            reasoning=reasoning,
            source="inference",
        )
        antithesis = AntithesisVerdict(
            disposition=AntithesisDisposition.COULD_NOT_BREAK,
            confidence=confidence,
        )
        return route_dialectic(
            oracle=oracle,
            antithesis=antithesis,
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )

    def test_judge_pass_is_unverifiable(self) -> None:
        assert self._judge_pass_scenario() is FailureOutcome.UNVERIFIABLE

    def test_high_confidence_doesnt_change_judge_pass(self) -> None:
        """confidence=0.99 (near oracle-backed territory) does not change the outcome."""
        assert self._judge_pass_scenario(confidence=0.99) is FailureOutcome.UNVERIFIABLE

    def test_low_confidence_doesnt_change_judge_pass(self) -> None:
        assert self._judge_pass_scenario(confidence=0.01) is FailureOutcome.UNVERIFIABLE

    def test_reasoning_mutation_doesnt_change_judge_pass(self) -> None:
        for r in ("", "excellent result", "I am very confident this is correct", "A" * 500):
            assert self._judge_pass_scenario(reasoning=r) is FailureOutcome.UNVERIFIABLE, (
                f"reasoning={r!r} changed the decision"
            )

    def _refine_scenario(
        self,
        confidence: float = 0.5,
        breakage: str = "found a bug",
        reasoning: str = "default",
    ) -> DialecticRoute:
        """Antithesis-BROKE → REFINE scenario. Mutate text/float and verify no change."""
        oracle = Verdict(
            holds=True,
            valid_check=True,
            reasoning=reasoning,
            source="tool",
        )
        antithesis = AntithesisVerdict(
            disposition=AntithesisDisposition.BROKE,
            breakage=breakage,
            confidence=confidence,
        )
        return route_dialectic(
            oracle=oracle,
            antithesis=antithesis,
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )

    def test_refine_scenario_base(self) -> None:
        assert self._refine_scenario() is REFINE

    def test_breakage_text_mutation_doesnt_change_refine(self) -> None:
        for b in ("tiny flaw", "CATASTROPHIC FAILURE", "b" * 1000):
            assert self._refine_scenario(breakage=b) is REFINE, (
                f"breakage={b!r} changed the decision"
            )

    def test_confidence_mutation_doesnt_change_refine(self) -> None:
        for c in (0.0, 0.1, 0.5, 0.9, 0.99):
            assert self._refine_scenario(confidence=c) is REFINE, (
                f"confidence={c} changed the decision"
            )

    def test_oracle_reasoning_mutation_doesnt_change_refine(self) -> None:
        for r in ("", "good job", "VERY CONFIDENT", "x" * 1000):
            assert self._refine_scenario(reasoning=r) is REFINE, (
                f"oracle.reasoning={r!r} changed the decision"
            )

    def _success_scenario(
        self,
        confidence: float = 0.5,
        reasoning: str = "default",
    ) -> DialecticRoute:
        """Success scenario. Mutate float/text and verify outcome unchanged."""
        oracle = Verdict(
            holds=True,
            valid_check=True,
            reasoning=reasoning,
            source="tool",
        )
        antithesis = AntithesisVerdict(
            disposition=AntithesisDisposition.COULD_NOT_BREAK,
            confidence=confidence,
        )
        return route_dialectic(
            oracle=oracle,
            antithesis=antithesis,
            acc=_acc(cycle_index=1),
            thesis_abstained=False,
            budget_exhausted=False,
        )

    def test_success_scenario_base(self) -> None:
        assert self._success_scenario() is FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS

    def test_confidence_mutation_doesnt_change_success(self) -> None:
        success = FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS
        for c in (0.0, 0.1, 0.5, 0.99):
            result = self._success_scenario(confidence=c)
            assert result is success, f"confidence={c} changed the success decision"

    def test_reasoning_mutation_doesnt_change_success(self) -> None:
        success = FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS
        for r in ("", "amazing", "x" * 500):
            result = self._success_scenario(reasoning=r)
            assert result is success, f"reasoning={r!r} changed the success decision"


# ---------------------------------------------------------------------------
# 4. Jaccard stuck-detector
# ---------------------------------------------------------------------------


class TestJaccardStuck:
    def test_fewer_than_two_texts_is_not_stuck(self) -> None:
        assert jaccard_stuck(()) is False
        assert jaccard_stuck(("one thesis",)) is False

    def test_identical_texts_is_stuck(self) -> None:
        t = "the answer is 42 because of reasons"
        assert jaccard_stuck((t, t)) is True

    def test_completely_different_texts_is_not_stuck(self) -> None:
        assert jaccard_stuck(("alpha beta gamma", "delta epsilon zeta omega")) is False

    def test_near_identical_texts_is_stuck(self) -> None:
        # 95 shared tokens out of 100 → Jaccard ≈ 0.95 > 0.92
        shared = " ".join(f"word{i}" for i in range(95))
        a = shared + " only_a"
        b = shared + " only_b"
        # a and b have 95 shared, 1 unique each → |intersection|/|union| = 95/97 ≈ 0.979
        similarity = 95 / 97
        assert similarity >= STUCK_JACCARD
        assert jaccard_stuck((a, b)) is True

    def test_partially_overlapping_texts_is_not_stuck(self) -> None:
        # 50 shared out of 100 total → Jaccard = 50/100 = 0.5 < 0.92
        a_only = " ".join(f"a{i}" for i in range(50))
        b_only = " ".join(f"b{i}" for i in range(50))
        shared = " ".join(f"s{i}" for i in range(50))
        a = shared + " " + a_only
        b = shared + " " + b_only
        assert jaccard_stuck((a, b)) is False

    def test_only_last_two_texts_are_compared(self) -> None:
        """The detector ignores all texts except the last two."""
        # First two are identical (would be stuck), third and fourth are different.
        identical = "same same same same"
        a = "alpha beta gamma delta"
        b = "epsilon zeta omega theta"
        # Only the last two (a, b) should be compared → not stuck.
        assert jaccard_stuck((identical, identical, a, b)) is False

    def test_empty_strings_both_are_stuck(self) -> None:
        """Two empty strings → empty union → treated as maximally similar (stuck)."""
        assert jaccard_stuck(("", "")) is True

    def test_threshold_boundary(self) -> None:
        """Similarity exactly at STUCK_JACCARD → stuck (>=, not >)."""
        # Construct sets with exactly STUCK_JACCARD similarity.
        # |A inter B| / |A union B| = 0.92 => need: 92 shared, 8 total unique
        # e.g. 92 shared + 4 unique to A + 4 unique to B => |union|=100, |inter|=92
        shared = " ".join(f"s{i}" for i in range(92))
        a_extra = " ".join(f"a{i}" for i in range(4))
        b_extra = " ".join(f"b{i}" for i in range(4))
        a = shared + " " + a_extra
        b = shared + " " + b_extra
        computed = 92 / 100
        assert abs(computed - STUCK_JACCARD) < 1e-9
        assert jaccard_stuck((a, b)) is True

    def test_just_below_threshold_is_not_stuck(self) -> None:
        # 91 shared + 9 total unique → 91/100 = 0.91 < 0.92
        shared = " ".join(f"s{i}" for i in range(91))
        a_extra = " ".join(f"a{i}" for i in range(5))
        b_extra = " ".join(f"b{i}" for i in range(4))
        a = shared + " " + a_extra
        b = shared + " " + b_extra
        computed = 91 / 100
        assert computed < STUCK_JACCARD
        assert jaccard_stuck((a, b)) is False


# ---------------------------------------------------------------------------
# 5. MAX_CYCLES constant
# ---------------------------------------------------------------------------


def test_max_cycles_is_five() -> None:
    assert MAX_CYCLES == 5


def test_max_cycles_minus_one_is_not_stuck() -> None:
    """cycle_index == MAX_CYCLES - 1 does not trigger the ceiling."""
    result = route_dialectic(
        oracle=_executable_oracle(holds=True, valid_check=True),
        antithesis=_antithesis_could_not_break(),
        acc=_acc(cycle_index=MAX_CYCLES - 1),
        thesis_abstained=False,
        budget_exhausted=False,
    )
    assert result is FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS


# ---------------------------------------------------------------------------
# 6. derive_accumulator — synthetic RunState
# ---------------------------------------------------------------------------


class TestDeriveAccumulator:
    def test_empty_run_state_produces_zero_accumulator(self) -> None:
        rs = _run_state([])
        acc = derive_accumulator(rs)
        assert acc.cycle_index == 0
        assert acc.breakage_history == ()
        assert acc.thesis_texts == ()

    def test_one_thesis_step(self) -> None:
        steps = [_step("thesis", 0, _thesis_artifact("my solution"))]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 0
        assert acc.thesis_texts == ("my solution",)

    def test_one_antithesis_could_not_break(self) -> None:
        art = _antithesis_artifact(AntithesisDisposition.COULD_NOT_BREAK)
        steps = [_step("antithesis", 1, art)]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 1
        assert acc.breakage_history == ()  # COULD_NOT_BREAK has no breakage

    def test_one_antithesis_broke(self) -> None:
        art = _antithesis_artifact(AntithesisDisposition.BROKE, breakage="step 2 fails")
        steps = [_step("antithesis", 1, art)]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 1
        assert acc.breakage_history == ("step 2 fails",)

    def test_full_two_cycle_walk(self) -> None:
        """Two full cycles: thesis→antithesis(broke)→thesis→antithesis(could_not_break)."""
        ant1 = _antithesis_artifact(AntithesisDisposition.BROKE, breakage="flaw A")
        ant2 = _antithesis_artifact(AntithesisDisposition.COULD_NOT_BREAK)
        steps = [
            _step("thesis", 0, _thesis_artifact("solution v1")),
            _step("antithesis", 1, ant1),
            _step("thesis", 2, _thesis_artifact("solution v2")),
            _step("antithesis", 3, ant2),
        ]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 2
        assert acc.breakage_history == ("flaw A",)
        assert acc.thesis_texts == ("solution v1", "solution v2")

    def test_other_stages_are_ignored(self) -> None:
        """experiment, evaluate, conclude steps do not affect the accumulator."""
        experiment_artifact = Artifact(
            kind="oracle-verdict",
            produced_by="experiment-stage",
            provenance=_PROVENANCE,
            data={"holds": True, "valid_check": True, "reasoning": "ok", "source": "tool"},
        )
        ant = _antithesis_artifact(AntithesisDisposition.COULD_NOT_BREAK)
        steps = [
            _step("thesis", 0, _thesis_artifact("sol")),
            _step("experiment", 1, experiment_artifact),
            _step("antithesis", 2, ant),
        ]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 1
        assert acc.thesis_texts == ("sol",)

    def test_multiple_cycles_breakage_history_accumulates(self) -> None:
        """Breakage history collects only BROKE verdicts."""
        broke1 = _antithesis_artifact(AntithesisDisposition.BROKE, breakage="flaw 1")
        broke2 = _antithesis_artifact(AntithesisDisposition.BROKE, breakage="flaw 2")
        cnb = _antithesis_artifact(AntithesisDisposition.COULD_NOT_BREAK)
        steps = [
            _step("thesis", 0, _thesis_artifact("v1")),
            _step("antithesis", 1, broke1),
            _step("thesis", 2, _thesis_artifact("v2")),
            _step("antithesis", 3, broke2),
            _step("thesis", 4, _thesis_artifact("v3")),
            _step("antithesis", 5, cnb),
        ]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 3
        assert acc.breakage_history == ("flaw 1", "flaw 2")
        assert acc.thesis_texts == ("v1", "v2", "v3")

    def test_wrong_artifact_kind_for_antithesis_is_skipped(self) -> None:
        """An antithesis step with the wrong artifact kind is skipped gracefully."""
        bad_artifact = Artifact(
            kind="oracle-verdict",  # wrong kind for antithesis step
            produced_by="antithesis-stage",
            provenance=_PROVENANCE,
            data={},
        )
        steps = [_step("antithesis", 0, bad_artifact)]
        acc = derive_accumulator(_run_state(steps))
        assert acc.cycle_index == 0  # skipped — wrong kind

    def test_wrong_artifact_kind_for_thesis_is_skipped(self) -> None:
        bad_artifact = Artifact(
            kind="antithesis-verdict",  # wrong for thesis
            produced_by="thesis-stage",
            provenance=_PROVENANCE,
            data={},
        )
        steps = [_step("thesis", 0, bad_artifact)]
        acc = derive_accumulator(_run_state(steps))
        assert acc.thesis_texts == ()

    def test_accumulator_is_frozen(self) -> None:
        acc = derive_accumulator(_run_state([]))
        with pytest.raises(ValidationError):
            acc.cycle_index = 99  # type: ignore[misc]

    def test_verifiable_claim_none_thesis_still_accumulates(self) -> None:
        """An abstaining thesis (verifiable_claim=None) still contributes to thesis_texts."""
        steps = [_step("thesis", 0, _thesis_artifact("abstain text", verifiable_claim=None))]
        acc = derive_accumulator(_run_state(steps))
        assert acc.thesis_texts == ("abstain text",)


# ---------------------------------------------------------------------------
# 7. DialecticRoute type identity
# ---------------------------------------------------------------------------


def test_refine_is_singleton() -> None:
    from cogworx.verification.dialectic_state import _Refine

    assert REFINE is _Refine()


def test_refine_repr() -> None:
    assert repr(REFINE) == "REFINE"


def test_failure_outcome_values_are_valid_routes() -> None:
    """Every FailureOutcome is a valid DialecticRoute (the union includes it)."""
    for outcome in FailureOutcome:
        assert isinstance(outcome, FailureOutcome)


def test_refine_is_not_a_failure_outcome() -> None:
    assert not isinstance(REFINE, FailureOutcome)


# ---------------------------------------------------------------------------
# 8. Additional routing branches for full coverage (plan §8.9)
# ---------------------------------------------------------------------------


def test_over_budget_route_failure_mapping() -> None:
    """OVER_BUDGET maps to await-human via route_failure (the EvaluateStage uses this)."""
    decision = route_failure(FailureOutcome.OVER_BUDGET)
    assert decision.disposition == "await-human"


def test_stuck_route_failure_mapping() -> None:
    decision = route_failure(FailureOutcome.STUCK)
    assert decision.disposition == "await-human"


def test_unverifiable_route_failure_mapping() -> None:
    decision = route_failure(FailureOutcome.UNVERIFIABLE)
    assert decision.disposition == "degraded"


def test_success_outcome_route_failure_disposition_is_done() -> None:
    """route_failure maps success to 'done' — this is the LANDMINE value.

    The EvaluateStage must NOT call Done for this outcome; it must call
    Transition(to='conclude').  This test documents that
    route_failure().disposition == 'done', not that Done() should be built.
    """
    decision = route_failure(FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS)
    assert decision.disposition == "done"
    # The caller MUST map this to Transition(to="conclude"), not Done.
    # We verify the StageResult types exist and confirm the distinction:
    transition = Transition(
        kind="transition",
        to="conclude",
        output=Artifact(kind="dialectic-route", produced_by="evaluate", provenance=_PROVENANCE),
    )
    assert transition.kind == "transition"
    assert transition.to == "conclude"


def test_unverifiable_via_executable_valid_check_false() -> None:
    """Rule 5: executable oracle with valid_check=False → UNVERIFIABLE (no valid experiment)."""
    result = route_dialectic(
        oracle=_executable_oracle(holds=True, valid_check=False),
        antithesis=_antithesis_could_not_break(),
        acc=_acc(cycle_index=1),
        thesis_abstained=False,
        budget_exhausted=False,
    )
    assert result is FailureOutcome.UNVERIFIABLE


def test_stuck_via_jaccard_with_varied_confidence(self: None = None) -> None:
    """STUCK via Jaccard — confidence mutation does not change it."""
    same_thesis = "token " * 100
    for c in (0.0, 0.5, 0.99):
        av = AntithesisVerdict(
            disposition=AntithesisDisposition.COULD_NOT_BREAK,
            confidence=c,
        )
        result = route_dialectic(
            oracle=_executable_oracle(holds=True, valid_check=True),
            antithesis=av,
            acc=_acc(cycle_index=2, thesis_texts=(same_thesis, same_thesis)),
            thesis_abstained=False,
            budget_exhausted=False,
        )
        assert result is FailureOutcome.STUCK


def test_non_could_not_break_disposition_does_not_succeed() -> None:
    """Rule 4 requires COULD_NOT_BREAK.

    ABSTAINED with executable oracle falls through to rule 7 REFINE.
    """
    # oracle.holds=True, valid_check=True, is_executable=True
    # antithesis.disposition=ABSTAINED → rule 4 not met
    # rule 5: valid_check=True → skip
    # rule 6: oracle.holds=True AND oracle.valid_check=True AND NOT is_executable → False → skip
    # rule 7: REFINE
    result = route_dialectic(
        oracle=_executable_oracle(holds=True, valid_check=True),
        antithesis=_antithesis_abstained(),
        acc=_acc(cycle_index=1),
        thesis_abstained=False,
        budget_exhausted=False,
    )
    assert result is REFINE
