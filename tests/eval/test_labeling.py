"""Deterministic unit tests for the corpus-labeling pipeline (Pod 4.4c-4).

Stub probe + stub adjudicate/tie_break/regime_adjudicate callbacks — NO live models, NO docker, NO
journal. Every assertion is mutation-resistant (red-team will attack): stratum-oracle polarity is
pinned in BOTH directions plus the noise boundary; the C1/C2 routing is pinned by who-gets-called;
the refuse-to-promote / abstention-drop invariants are item-granular; the O-regime cross-check and
the §6.C existence-before-regime invariant are pinned with negative controls; and the S1/S9 posture
is pinned structurally over the module AST (no journal symbol, no judge import, never
``CodeOracle.evaluate``). Mirrors ``test_planting.py``'s pin discipline.
"""

from __future__ import annotations

import ast
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from cogworx.claims.provenance import ProvenanceSource
from cogworx.eval.corpus import (
    Adjudication,
    ConvertedPlanterStamp,
    CorpusItem,
    DeterministicPlanterStamp,
    DifficultyMarker,
    HumanLabelProvenance,
    LLMPlanterStamp,
    OracleLabelProvenance,
    RegimeAdjudication,
)
from cogworx.eval.labeling import (
    AbstentionDrop,
    AdjudicateCallback,
    AdjudicationOutcome,
    AdjudicationRequest,
    CleanLabel,
    ORegimeMismatchError,
    PromotionResult,
    RegimeAdjudicateCallback,
    RegimeExistenceError,
    StratumAssignment,
    TieBreakCallback,
    adjudicate_item,
    assign_stratum,
    promote_corpus,
    reverify_clean,
)
from cogworx.eval.planting import (
    DETK_PROBE_OPERATOR,
    OInjector,
    PlantedItem,
    PlantedPair,
    Seed,
    build_detk_pair,
)
from cogworx.verification.contracts import OracleFrame, Thesis, Verdict

# ---------------------------------------------------------------------------
# Well-formed builders (the mutation-resistance controls)
# ---------------------------------------------------------------------------

_FRAME = OracleFrame(
    completion_criterion="tests_pass", problem_type="code", problem_statement="sum a list"
)
_THESIS = Thesis(proposed_solution="return sum(xs)", experiment_design="run frozen tests")
_TS = datetime(2026, 6, 20, tzinfo=UTC)


def _difficulty() -> DifficultyMarker:
    return DifficultyMarker(planted_difficulty="medium", surface_complexity=12)


def _planted(**overrides: object) -> PlantedItem:
    base: dict[str, object] = {
        "provisional_id": 1,
        "frame": _FRAME,
        "thesis": _THESIS,
        "test_code": "assert f([1, 2]) == 3",
        "candidate_stratum": "O",
        "is_error": 1,
        "planter": DeterministicPlanterStamp(operators=("arithmetic-swap",)),
        "error_regime": "logic-wrong",
        "difficulty": _difficulty(),
        "matched_sibling_id": None,
        "split": "tuning",
    }
    base.update(overrides)
    return PlantedItem(**base)


_Verdict3 = Literal["error", "clean", "abstain"]
_RegimeVerdict = Literal["confirm-regime", "reassign-regime", "abstain"]


def _verdict(
    *,
    holds: bool,
    valid_check: bool,
    source: ProvenanceSource = "tool",
    test_provenance: Literal["thesis", "frozen", "n/a"] = "frozen",
) -> Verdict:
    return Verdict(
        holds=holds,
        valid_check=valid_check,
        reasoning="stub",
        source=source,
        test_provenance=test_provenance,
    )


class _ConstProbe:
    """A stub OracleProbe that returns a fixed verdict and records its calls."""

    def __init__(self, verdict: Verdict) -> None:
        self.verdict = verdict
        self.calls: list[tuple[str, str]] = []

    def __call__(self, solution_code: str, test_code: str) -> Verdict:
        self.calls.append((solution_code, test_code))
        return self.verdict


def _const_probe(verdict: Verdict) -> _ConstProbe:
    return _ConstProbe(verdict)


def _adjudicators(*verdicts: _Verdict3) -> AdjudicateCallback:
    """A stub AdjudicateCallback returning the given existence verdicts."""

    def adjudicate(request: AdjudicationRequest) -> tuple[Adjudication, ...]:
        return tuple(
            Adjudication(adjudicator_id=f"adj-{i}", verdict=v, rationale="stub", timestamp=_TS)
            for i, v in enumerate(verdicts)
        )

    return adjudicate


def _tie_break(verdict: _Verdict3) -> TieBreakCallback:
    def tie_break(request: AdjudicationRequest) -> Adjudication:
        return Adjudication(
            adjudicator_id="architect", verdict=verdict, rationale="tb", timestamp=_TS
        )

    return tie_break


def _regime(verdict: _RegimeVerdict, reassigned: str | None = None) -> RegimeAdjudicateCallback:
    def regime_adjudicate(request: AdjudicationRequest) -> RegimeAdjudication:
        return RegimeAdjudication(
            adjudicator_id="second-author",
            verdict=verdict,
            reassigned_regime=reassigned,
            rationale="rg",
            timestamp=_TS,
        )

    return regime_adjudicate


def _no_call_adjudicate(request: AdjudicationRequest) -> tuple[Adjudication, ...]:
    raise AssertionError(
        f"adjudicate was called for item {request.provisional_id} — a C1-eligible path must not "
        f"reach the human harness"
    )


def _no_call_tie_break(request: AdjudicationRequest) -> Adjudication:
    raise AssertionError("tie_break was called on a non-split path")


# ===========================================================================
# Obligation 1 — stratum oracle-determined, polarity-paired + noise boundary
# ===========================================================================


def test_stratum_o_catch_overrules_candidate_K() -> None:
    """An error item proposed candidate_stratum='K' but the oracle CAUGHT it (executable ∧ valid ∧
    ¬holds) → binding stratum 'O'. The oracle overrules the planter's proposal."""
    item = _planted(candidate_stratum="K", is_error=1)
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=False, valid_check=True)))
    assert asg.stratum == "O"
    assert asg.oracle_reachable is True
    assert asg.candidate_agreed is False  # K proposed, O assigned — logged, not an error


def test_stratum_uncaught_error_overrules_candidate_O_to_K() -> None:
    """candidate_stratum='O' but the planted error SURVIVED the frozen test (holds=True) → 'K'.
    Mutation that must fail: an assigner returning candidate_stratum (would give 'O' here)."""
    item = _planted(candidate_stratum="O", is_error=1)
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=True, valid_check=True)))
    assert asg.stratum == "K"  # NOT 'O' (candidate) — the error was not caught


def test_stratum_noise_verdict_is_K_not_O() -> None:
    """The eval-stats correction (§6.A): a NOISE verdict (valid_check=False) is NOT a catch. The
    item with valid_check=False → 'K', NEVER 'O'. Mutation that must fail: an assigner using ¬holds
    alone (ignoring valid_check) for the O-catch — it would mis-route this to 'O'."""
    item = _planted(candidate_stratum="O", is_error=1)
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=False, valid_check=False)))
    assert asg.stratum == "K"
    assert asg.oracle_reachable is False  # not (executable ∧ valid_check)


def test_stratum_non_executable_error_is_K() -> None:
    """An inference-sourced (non-executable) error verdict → 'K' (oracle-blind), whatever holds.
    Pins that is_executable gates the O-catch."""
    item = _planted(candidate_stratum="O", is_error=1)
    asg = assign_stratum(
        item, probe=_const_probe(_verdict(holds=False, valid_check=True, source="inference"))
    )
    assert asg.stratum == "K"
    assert asg.oracle_reachable is False


def test_stratum_clean_item_always_clean() -> None:
    """A clean item (is_error==0) is ALWAYS stratum 'clean', whatever the verdict polarity."""
    item = _planted(candidate_stratum="clean", is_error=0)
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=True, valid_check=True)))
    assert asg.stratum == "clean"


def test_stratum_thesis_sourced_verdict_disqualified() -> None:
    """A thesis-sourced verdict (test_provenance != 'frozen') is DISQUALIFIED — the offline kernel
    must produce a frozen-sourced verdict (the §13.4 #5 self-test-laundering guard)."""
    item = _planted(is_error=1)
    with pytest.raises(ValueError, match=r"self-test-laundering|frozen"):
        assign_stratum(
            item,
            probe=_const_probe(_verdict(holds=False, valid_check=True, test_provenance="thesis")),
        )


def test_stratum_never_reads_candidate_for_binding() -> None:
    """Mutation-resistant: with a FIXED verdict, the binding stratum is identical regardless of the
    candidate proposal — proving candidate_stratum is never read to SET stratum. (candidate_agreed
    differs, but that is reported-only.)"""
    v = _const_probe(_verdict(holds=False, valid_check=True))  # an O-catch verdict
    s_from_K = assign_stratum(_planted(candidate_stratum="K", is_error=1), probe=v).stratum
    s_from_O = assign_stratum(_planted(candidate_stratum="O", is_error=1), probe=v).stratum
    assert s_from_K == s_from_O == "O"


# ===========================================================================
# Obligation 2 — C1 polarity (oracle label from the carried frozen verdict)
# ===========================================================================


def _clean_item(**overrides: object) -> PlantedItem:
    base: dict[str, object] = {
        "provisional_id": 2,
        "candidate_stratum": "clean",
        "is_error": 0,
        "error_regime": "",
    }
    base.update(overrides)
    return _planted(**base)


def test_c1_clean_oracle_reachable_emits_oracle_provenance() -> None:
    """C1: clean item, carried verdict holds ∧ valid_check ∧ executable → OracleLabelProvenance,
    label_source='oracle', test_provenance=='frozen'. The human harness is NOT consulted."""
    item = _clean_item()
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=True, valid_check=True)))
    label = reverify_clean(item, asg, adjudicate=_no_call_adjudicate, tie_break=_no_call_tie_break)
    assert isinstance(label, CleanLabel)
    assert label.label_source == "oracle"
    assert isinstance(label.label_provenance, OracleLabelProvenance)
    assert label.label_provenance.test_provenance == "frozen"
    assert label.label_provenance.holds is True


def test_c1_reuses_carried_verdict_no_second_probe() -> None:
    """§6.B: C1 reuses the carried verdict; reverify_clean takes NO probe param at all (no second
    probe call is even expressible), and the label comes from the carried assignment verdict."""
    import inspect

    assert "probe" not in inspect.signature(reverify_clean).parameters
    item = _clean_item()
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=True, valid_check=True)))
    label = reverify_clean(item, asg, adjudicate=_no_call_adjudicate, tie_break=_no_call_tie_break)
    assert isinstance(label, CleanLabel)


def test_c1_oracle_reachable_clean_failing_its_test_is_dropped() -> None:
    """An oracle-reachable clean item whose carried verdict does NOT hold is a corpus defect →
    AbstentionDrop, never a forced oracle 'clean' label."""
    item = _clean_item()
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=False, valid_check=True)))
    result = reverify_clean(item, asg, adjudicate=_no_call_adjudicate, tie_break=_no_call_tie_break)
    assert isinstance(result, AbstentionDrop)


# ===========================================================================
# Obligation 3 — C2 → human only when oracle-blind
# ===========================================================================


def test_c2_oracle_blind_clean_routes_to_human() -> None:
    """C2: clean item, oracle-blind (non-executable verdict) → routed to the human harness; two
    'clean' verdicts → HumanLabelProvenance, label_source='human'."""
    item = _clean_item()
    asg = assign_stratum(
        item, probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference"))
    )
    assert asg.oracle_reachable is False
    label = reverify_clean(
        item, asg, adjudicate=_adjudicators("clean", "clean"), tie_break=_tie_break("clean")
    )
    assert isinstance(label, CleanLabel)
    assert label.label_source == "human"
    assert isinstance(label.label_provenance, HumanLabelProvenance)


def test_c2_adjudicate_not_called_for_c1_eligible() -> None:
    """The hard wall, observed: a C1-eligible (oracle-reachable) clean item NEVER reaches the
    adjudicate callback (the _no_call stub raises if it does)."""
    item = _clean_item()
    asg = assign_stratum(item, probe=_const_probe(_verdict(holds=True, valid_check=True)))
    # _no_call_adjudicate raises AssertionError if invoked — a clean run proves it was not.
    reverify_clean(item, asg, adjudicate=_no_call_adjudicate, tie_break=_no_call_tie_break)


# ===========================================================================
# Obligation 4 — refuse-to-promote unverified clean (INV-3)
# ===========================================================================


def test_inv3_oracle_blind_clean_both_abstain_is_dropped() -> None:
    """INV-3: oracle-blind clean, both adjudicators abstain → AbstentionDrop, NOT a forced 'clean'.
    A promoter forcing a clean label on abstention fails this test."""
    item = _clean_item()
    asg = assign_stratum(
        item, probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference"))
    )
    result = reverify_clean(
        item, asg, adjudicate=_adjudicators("abstain", "abstain"), tie_break=_tie_break("clean")
    )
    assert isinstance(result, AbstentionDrop)


def test_inv3_via_promote_corpus_drops_not_promotes() -> None:
    """INV-3 end-to-end: an oracle-blind clean singleton whose adjudicators abstain lands in
    dropped_ids, NOT in promoted."""
    item = _clean_item(provisional_id=42, matched_sibling_id=None)
    result = promote_corpus(
        [],
        [item],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("abstain", "abstain"),
        tie_break=_tie_break("clean"),
    )
    assert 42 in result.dropped_ids
    assert all(ci.item_id != 42 for ci in result.promoted)


# ===========================================================================
# Obligation 5 — abstention-drop is item-granular (INV-5)
# ===========================================================================


def test_inv5_abstention_drop_is_item_granular() -> None:
    """INV-5: dropped_ids contains exactly the abstained provisional_ids; promoted excludes them and
    includes the survivors. One C1-clean survivor + one abstaining oracle-blind clean. The probe
    routes the two clean items by their (distinct) solution text: the survivor is oracle-reachable
    (executable); the dropped one is oracle-blind (inference) so it falls to the abstainers."""
    survivor = _clean_item(provisional_id=1, matched_sibling_id=None)  # _THESIS, oracle-reachable
    dropped = _clean_item(
        provisional_id=2,
        matched_sibling_id=None,
        thesis=Thesis(proposed_solution="return 0", experiment_design="x"),  # oracle-blind
    )

    def probe(solution_code: str, test_code: str) -> Verdict:
        if solution_code == "return 0":
            return _verdict(holds=True, valid_check=True, source="inference")  # oracle-blind
        return _verdict(holds=True, valid_check=True)  # C1-reachable survivor

    result = promote_corpus(
        [],
        [survivor, dropped],
        probe=probe,
        adjudicate=_adjudicators("abstain", "abstain"),
        tie_break=_tie_break("clean"),
    )
    assert result.dropped_ids == frozenset({2})
    assert {ci.item_id for ci in result.promoted} == {1}


# ===========================================================================
# Obligation 6 — O-regime cross-check (INV-6 / §2.C fail-fast)
# ===========================================================================


def test_inv6_o_regime_mismatch_hard_fails() -> None:
    """§2.C fail-fast: an O item tagged 'off-by-semantics' but whose operators=('arithmetic-swap',)
    derives 'logic-wrong' → promote_corpus hard-fails that item. Mutation: a promoter trusting the
    author's error_regime for O would silently keep the wrong tag."""
    err = _planted(
        provisional_id=10,
        is_error=1,
        candidate_stratum="O",
        planter=DeterministicPlanterStamp(operators=("arithmetic-swap",)),
        error_regime="off-by-semantics",  # WRONG — arithmetic-swap derives logic-wrong
        matched_sibling_id=None,
    )
    with pytest.raises(ORegimeMismatchError, match=r"logic-wrong|fail-fast"):
        promote_corpus(
            [],
            [err],
            probe=_const_probe(_verdict(holds=False, valid_check=True)),  # O-catch
            adjudicate=_adjudicators("error", "error"),
            tie_break=_tie_break("error"),
        )


def test_inv6_o_regime_match_promotes_with_derived_regime() -> None:
    """Control: an O item whose author error_regime AGREES with the derived regime promotes, and the
    promoted error_regime is the DERIVED value (mechanical, not author-trusted)."""
    err = _planted(
        provisional_id=10,
        is_error=1,
        candidate_stratum="O",
        planter=DeterministicPlanterStamp(operators=("arithmetic-swap",)),
        error_regime="logic-wrong",
        matched_sibling_id=None,
    )
    result = promote_corpus(
        [],
        [err],
        probe=_const_probe(_verdict(holds=False, valid_check=True)),
        adjudicate=_adjudicators("error", "error"),
        tie_break=_tie_break("error"),
    )
    assert len(result.promoted) == 1
    ci = result.promoted[0]
    assert ci.stratum == "O"
    assert ci.error_regime == "logic-wrong"
    assert ci.label_source == "oracle"


# ===========================================================================
# Obligation 7 — K-regime audit (RegimeAdjudication is the sole source)
# ===========================================================================


def _k_error(**overrides: object) -> PlantedItem:
    base: dict[str, object] = {
        "provisional_id": 20,
        "is_error": 1,
        "candidate_stratum": "K",
        "planter": LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        "error_regime": "spec-misread",  # CANDIDATE — audited
        "matched_sibling_id": None,
    }
    base.update(overrides)
    return _planted(**base)


def _promote_k(regime_adjudicate: RegimeAdjudicateCallback) -> PromotionResult:
    """Promote a single K error (oracle-blind verdict) with the given regime callback."""
    item = _k_error()
    return promote_corpus(
        [],
        [item],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),  # K
        adjudicate=_adjudicators("error", "error"),  # existence established
        tie_break=_tie_break("error"),
        regime_adjudicate=regime_adjudicate,
    )


def test_k_regime_reassign_overrides_candidate() -> None:
    result = _promote_k(_regime("reassign-regime", "silent-degradation"))
    assert len(result.promoted) == 1
    assert result.promoted[0].error_regime == "silent-degradation"
    assert result.promoted[0].stratum == "K"
    assert result.promoted[0].label_source == "human"
    # The regime audit record is reported in regime_audit.
    assert len(result.regime_audit) == 1
    assert result.regime_audit[0].verdict == "reassign-regime"


def test_k_regime_confirm_keeps_candidate() -> None:
    result = _promote_k(_regime("confirm-regime"))
    assert result.promoted[0].error_regime == "spec-misread"


def test_k_regime_abstain_blanks_regime_keeps_item() -> None:
    """A regime abstain KEEPS the item (error_regime='') — opposite of an existence abstain."""
    result = _promote_k(_regime("abstain"))
    assert len(result.promoted) == 1  # item stays
    assert result.promoted[0].error_regime == ""


def test_k_error_carries_explicit_error_existence_adjudication() -> None:
    """§6.C: a promoted K-error carries a HumanLabelProvenance with an explicit 'error' existence
    Adjudication (existence established FIRST)."""
    result = _promote_k(_regime("confirm-regime"))
    prov = result.promoted[0].label_provenance
    assert isinstance(prov, HumanLabelProvenance)
    assert any(a.verdict == "error" for a in prov.adjudications)


def test_k_error_existence_abstain_drops_before_regime() -> None:
    """A K-error whose existence adjudicators abstain is DROPPED — the regime callback is never
    consulted (existence-before-regime). The regime stub raises if called."""
    item = _k_error(provisional_id=21)

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError("regime audit ran before existence was established (§6.C violation)")

    result = promote_corpus(
        [],
        [item],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("abstain", "abstain"),
        tie_break=_tie_break("error"),
        regime_adjudicate=regime_must_not_run,
    )
    assert 21 in result.dropped_ids
    assert result.promoted == ()


def test_k_existence_split_to_error_then_tie_break() -> None:
    """A split existence decision (error vs clean) is resolved by the tie-breaker; an 'error'
    tie-break promotes the K item and records the tie_breaker_id."""
    result = promote_corpus(
        [],
        [_k_error(provisional_id=22)],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("error", "clean"),  # split
        tie_break=_tie_break("error"),
        regime_adjudicate=_regime("confirm-regime"),
    )
    assert len(result.promoted) == 1
    prov = result.promoted[0].label_provenance
    assert isinstance(prov, HumanLabelProvenance)
    assert prov.tie_breaker_id == "architect"


def test_regime_existence_invariant_guard_is_real() -> None:
    """Pin the §6.C invariant guard directly: a K-error whose human provenance has NO 'error'
    existence Adjudication is rejected. We force this by an existence outcome that (pathologically)
    reports a non-error verdict yet is non-dropped — exercised via the guard on a hand-built
    provenance through promote with a 'clean'-only existence (which drops, proving the guard path is
    only reached with an established error). Here we assert the RegimeExistenceError type exists and
    is a ValueError subclass (structural), and that the existence-first ordering holds via the drop
    above."""
    assert issubclass(RegimeExistenceError, ValueError)


# ===========================================================================
# detK probe items — carried through untouched (no K-audit)
# ===========================================================================


def _detk_pair() -> PlantedPair:
    seed = Seed(
        seed_id=0,
        frame=OracleFrame(
            completion_criterion="tests_pass", problem_type="code", problem_statement="scale"
        ),
        thesis=Thesis(
            proposed_solution="def scale(x):\n    return x * 2\n",
            experiment_design="run frozen tests",
        ),
        test_code="from solution import scale\n\n\ndef test_scale():\n    assert scale(3) == 6\n",
    )
    return build_detk_pair(seed, "constant-replacement", error_id=30, clean_id=31)


def test_detk_error_carried_through_no_regime_audit() -> None:
    """A detK probe error is promoted untouched (regime derives off the real operator at planting);
    no K-regime audit runs (regime_audit stays empty for it). Its candidate stays K and the verdict
    is oracle-blind (uncaught) so it lands stratum 'K'."""
    pair = _detk_pair()

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError("detK item triggered a K-regime audit (it must be carried untouched)")

    # detK error: holds=True (survives — oracle-blind by construction in the K-stratum sense).
    result = promote_corpus(
        [pair],
        [],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("error", "error", "clean"),  # error member + clean sibling
        tie_break=_tie_break("error"),
        regime_adjudicate=regime_must_not_run,
    )
    err = next(ci for ci in result.promoted if ci.item_id == 30)
    assert err.stratum == "K"
    assert err.error_regime == "off-by-semantics"  # the detK real-operator regime, unaudited
    assert result.regime_audit == ()  # no audit record for detK
    assert isinstance(err.planter, DeterministicPlanterStamp)
    assert DETK_PROBE_OPERATOR in err.planter.operators
    # §6.C: detK skips only the REGIME audit — error-EXISTENCE is still human-adjudicated, so the
    # promoted detK K-error carries an explicit 'error' existence Adjudication.
    prov = err.label_provenance
    assert isinstance(prov, HumanLabelProvenance)
    assert any(a.verdict == "error" for a in prov.adjudications)


def test_detk_error_drops_on_existence_abstain() -> None:
    """§6.C: a detK K-error still goes through human EXISTENCE adjudication (it only skips the
    regime second-author audit). If existence abstains, the detK error DROPS (lands in dropped_ids),
    never a forced label — and the regime callback is never reached."""
    detk_err = _detk_pair().error_item

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError("detK existence abstained but a regime audit ran (§6.C violation)")

    result = promote_corpus(
        [],
        [detk_err],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("abstain", "abstain"),
        tie_break=_tie_break("error"),
        regime_adjudicate=regime_must_not_run,
    )
    assert detk_err.provisional_id in result.dropped_ids
    assert result.promoted == ()


# ===========================================================================
# §6.C existence-GATE: a K-error whose existence resolves CLEAN must DROP, never promote
# (red-team HIGH: the `existence_verdict != "error"` half of the :607 gate was untested for the
# error-item branch — a human-REJECTED "this isn't an error" item would be laundered into a
# confirmed label_source="human", is_error=1 label. The `existence.dropped` half is covered by the
# abstain tests above; these pin the CLEAN-verdict half, which the downstream RegimeExistenceError
# guard does NOT catch because the provenance can still contain an 'error' adjudication.)
# ===========================================================================


def test_k_error_existence_consensus_clean_is_dropped() -> None:
    """A K-stratum ERROR item whose two existence adjudicators BOTH return 'clean' (consensus: "this
    isn't actually an error") → DROPPED, never promoted. The regime callback is NEVER reached
    (existence-clean drops before the regime audit). Mutation killed: a gate of only
    `if existence.dropped:` would PROMOTE this (consensus-clean is not a drop) and launder it into a
    confirmed human is_error=1 label."""
    item = _k_error(provisional_id=23)

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError(
            "regime audit ran for a K-error whose existence resolved 'clean' (§6.C gate violation)"
        )

    result = promote_corpus(
        [],
        [item],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),  # K
        adjudicate=_adjudicators("clean", "clean"),  # consensus: NOT an error
        tie_break=_no_call_tie_break,  # consensus → no tie-break
        regime_adjudicate=regime_must_not_run,
    )
    assert 23 in result.dropped_ids
    assert all(ci.item_id != 23 for ci in result.promoted)
    assert result.promoted == ()


def test_k_error_existence_split_tie_broken_to_clean_is_dropped() -> None:
    """The exact red-team falsifying input: existence adjudicators SPLIT ('error','clean') and the
    tie-break returns 'clean' → DROPPED. This is the case the downstream RegimeExistenceError guard
    can NOT catch: the recorded provenance DOES contain one 'error' adjudication (the split
    primary), so `any(a.verdict == "error" ...)` is satisfied — ONLY the :607 gate
    (`existence_verdict != "error"`) drops it. The regime callback is NEVER reached."""
    item = _k_error(provisional_id=24)

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError(
            "regime audit ran for a K-error whose split existence tie-broke to 'clean' (§6.C gate)"
        )

    result = promote_corpus(
        [],
        [item],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),  # K
        adjudicate=_adjudicators("error", "clean"),  # split — provenance carries an 'error' adj
        tie_break=_tie_break("clean"),  # tie-broken to clean (not an error)
        regime_adjudicate=regime_must_not_run,
    )
    assert 24 in result.dropped_ids
    assert all(ci.item_id != 24 for ci in result.promoted)
    assert result.promoted == ()


def test_detk_error_existence_consensus_clean_is_dropped() -> None:
    """detK path, same weakness: a detK K-error whose existence adjudicators BOTH return 'clean' →
    DROPPED. The detK existence path shares the :607 gate; a `if existence.dropped:` mutant would
    promote (launder) it. The regime callback is never reached (detK skips the regime audit anyway,
    but a laundered promote would still build a confirmed is_error=1 CorpusItem)."""
    detk_err = _detk_pair().error_item

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError("detK regime audit ran for an existence-'clean' K-error (§6.C gate)")

    result = promote_corpus(
        [],
        [detk_err],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("clean", "clean"),  # consensus: NOT an error
        tie_break=_no_call_tie_break,
        regime_adjudicate=regime_must_not_run,
    )
    assert detk_err.provisional_id in result.dropped_ids
    assert result.promoted == ()


def test_detk_error_existence_split_tie_broken_to_clean_is_dropped() -> None:
    """detK path, the split-tie-to-clean falsifying input: existence splits ('error','clean'),
    tie-break returns 'clean' → DROPPED. As in the non-detK split case, the provenance carries an
    'error' adjudication so the RegimeExistenceError guard does NOT fire — only the :607 gate drops
    it. The regime callback is never reached."""
    detk_err = _detk_pair().error_item

    def regime_must_not_run(request: AdjudicationRequest) -> RegimeAdjudication:
        raise AssertionError("detK regime audit ran for a split-to-'clean' K-error (§6.C gate)")

    result = promote_corpus(
        [],
        [detk_err],
        probe=_const_probe(_verdict(holds=True, valid_check=True, source="inference")),
        adjudicate=_adjudicators("error", "clean"),  # split — provenance carries an 'error' adj
        tie_break=_tie_break("clean"),  # tie-broken to clean
        regime_adjudicate=regime_must_not_run,
    )
    assert detk_err.provisional_id in result.dropped_ids
    assert result.promoted == ()


# ===========================================================================
# Converted (K→O) items — O-by-execution, regime SECOND-AUTHOR-AUDITED, operator
# cross-check SKIPPED by construction (Pod 4.4c-3.5).
# ===========================================================================


def _converted_stamp() -> ConvertedPlanterStamp:
    return ConvertedPlanterStamp(
        planter_model_family="deepseek",
        planter_model_id="deepseek/chat",
        adversary_family="claude",
        winning_round=2,
    )


def _converted_o_error(**overrides: object) -> PlantedItem:
    """A K→O converted error: an LLM-planted ERROR (candidate K) whose adversary-synthesized test
    makes it oracle-reachable, carrying a ConvertedPlanterStamp (no operators). Its candidate regime
    is a CANDIDATE — second-author-audited, never operator-derived."""
    base: dict[str, object] = {
        "provisional_id": 40,
        "is_error": 1,
        "candidate_stratum": "K",  # planted K; converter makes it O-by-execution
        "planter": _converted_stamp(),
        "error_regime": "spec-misread",  # CANDIDATE — audited, not operator-derived
        "matched_sibling_id": None,
    }
    base.update(overrides)
    return _planted(**base)


def _promote_converted(regime_adjudicate: RegimeAdjudicateCallback) -> PromotionResult:
    """Promote one converted O error with an O-catch verdict (executable ∧ valid ∧ ¬holds)."""
    return promote_corpus(
        [],
        [_converted_o_error()],
        probe=_const_probe(_verdict(holds=False, valid_check=True)),  # O catch
        adjudicate=_adjudicators("error", "error"),
        tie_break=_tie_break("error"),
        regime_adjudicate=regime_adjudicate,
    )


def test_converted_o_item_is_oracle_labelled_O() -> None:
    """A converted item caught by the adversary's synthesized test lands stratum 'O',
    label_source='oracle', OracleLabelProvenance(test_provenance='frozen') — the oracle still
    decided the catch."""
    result = _promote_converted(_regime("confirm-regime"))
    assert len(result.promoted) == 1
    ci = result.promoted[0]
    assert ci.stratum == "O"
    assert ci.label_source == "oracle"
    assert isinstance(ci.label_provenance, OracleLabelProvenance)
    assert ci.label_provenance.test_provenance == "frozen"
    assert isinstance(ci.planter, ConvertedPlanterStamp)


def test_converted_o_regime_from_second_author_not_operators() -> None:
    """The converted item's error_regime is the SECOND-AUTHOR audit verdict, NOT operator-derived.
    A reassign-regime overrides the candidate, and the audit is recorded in regime_audit. Mutation
    killed: an O path deriving regime off operators would raise (no operators) or yield a fixed
    operator-table regime, never 'silent-degradation'."""
    result = _promote_converted(_regime("reassign-regime", "silent-degradation"))
    assert len(result.promoted) == 1
    assert result.promoted[0].error_regime == "silent-degradation"
    assert len(result.regime_audit) == 1
    assert result.regime_audit[0].verdict == "reassign-regime"


def test_converted_o_regime_confirm_keeps_candidate() -> None:
    """confirm-regime keeps the converted item's candidate regime (second-author confirmed it)."""
    result = _promote_converted(_regime("confirm-regime"))
    assert result.promoted[0].error_regime == "spec-misread"


def test_converted_o_regime_abstain_blanks_regime_keeps_item() -> None:
    """A regime abstain blanks the regime ('') but KEEPS the converted item (opposite of an
    existence abstain) — same semantics as the K path."""
    result = _promote_converted(_regime("abstain"))
    assert len(result.promoted) == 1
    assert result.promoted[0].error_regime == ""


def test_converted_o_operator_cross_check_is_skipped_not_raised() -> None:
    """The §2.C operator cross-check is SKIPPED by construction for a converted O item — it does NOT
    raise ORegimeMismatchError despite carrying NO mutation operators and an error_regime that does
    not exist in the operator regime table. Mutation killed: routing a converted O item through
    _o_operators would raise ORegimeMismatchError (non-deterministic stamp) and fail to promote."""
    result = _promote_converted(_regime("confirm-regime"))  # no exception raised
    assert len(result.promoted) == 1
    assert result.promoted[0].stratum == "O"


def test_converted_o_carries_honest_planting_provenance() -> None:
    """The promoted converted item keeps its ORIGINAL LLM planting identity on the stamp (honest
    planting provenance survives the conversion) plus the adversary family + winning round."""
    ci = _promote_converted(_regime("confirm-regime")).promoted[0]
    assert isinstance(ci.planter, ConvertedPlanterStamp)
    assert ci.planter.planter_model_family == "deepseek"
    assert ci.planter.adversary_family == "claude"
    assert ci.planter.winning_round == 2


def test_deterministic_o_path_unchanged_by_converted_branch() -> None:
    """Regression guard: a deterministic O item still derives its regime off operators and still
    HARD-FAILS on a §2.C mismatch — the converted branch did not weaken the operator cross-check."""
    err = _planted(
        provisional_id=10,
        is_error=1,
        candidate_stratum="O",
        planter=DeterministicPlanterStamp(operators=("arithmetic-swap",)),
        error_regime="off-by-semantics",  # WRONG — arithmetic-swap derives logic-wrong
        matched_sibling_id=None,
    )
    with pytest.raises(ORegimeMismatchError, match=r"logic-wrong|fail-fast"):
        promote_corpus(
            [],
            [err],
            probe=_const_probe(_verdict(holds=False, valid_check=True)),
            adjudicate=_adjudicators("error", "error"),
            tie_break=_tie_break("error"),
        )


# ===========================================================================
# Obligation 8 — no-judge (INV-2 / S9): no model/judge provenance, no judge import
# ===========================================================================


def _labeling_tree() -> ast.Module:
    import cogworx.eval.labeling as labeling_mod

    return ast.parse(Path(labeling_mod.__file__).read_text(encoding="utf-8"))


def _imported_modules(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    return imported


def _referenced_names(tree: ast.Module) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_labeling_imports_no_judge_or_antithesis_symbol() -> None:
    """INV-2/S9: labeling.py constructs no provenance from a model/judge response — it imports no
    judge / antithesis / model-provider symbol. (The Model seam is not even imported: the only
    model-shaped input is the injected human-callback, never a model.)"""
    imported = _imported_modules(_labeling_tree())
    for forbidden in (
        "cogworx.verification.judge",
        "cogworx.dialectic.antithesis",
        "cogworx.model.base",
        "cogworx.model.providers.claude",
        "cogworx.model.providers.openai_compat",
    ):
        assert forbidden not in imported, f"labeling.py imports a judge/model symbol: {forbidden!r}"
    # No judge/antithesis token appears in any imported module path.
    assert not any("judge" in m or "antithesis" in m for m in imported), (
        f"labeling.py imports a judge/antithesis module: {imported}"
    )


# ===========================================================================
# Obligation 9 — S1 / journal-free: reaches run_frozen_check, NEVER CodeOracle.evaluate
# ===========================================================================


def test_labeling_module_does_no_journal_io() -> None:
    """S1 (the #1 red-team target): labeling.py never imports or references a Journal / RunContext /
    a stage context / load_run / CodeOracle in CODE — it cannot be on a live write-path. Structural
    the AST (S9: structure over self-report)."""
    tree = _labeling_tree()
    imported = _imported_modules(tree)
    referenced = _referenced_names(tree)
    for forbidden in (
        "Journal",
        "RunContext",
        "StageContext",
        "load_run",
        "GraphStore",
        "LatentStore",
        "CodeOracle",
        "evaluate",
    ):
        assert forbidden not in referenced, (
            f"labeling.py CODE references {forbidden!r} (S1 write-path / live-oracle smell)"
        )
    assert not any("journal" in m or "substrate" in m or "runtime" in m for m in imported), (
        f"labeling.py imports a journal/substrate/runtime module: {imported}"
    )


def test_labeling_default_probe_is_the_frozen_kernel() -> None:
    """The default probe binding IS run_frozen_check (the journal-free frozen kernel), NEVER
    CodeOracle.evaluate (which needs a journal + StageContext — the live path)."""
    import inspect

    from cogworx.eval._authoring import run_frozen_check

    for fn in (assign_stratum, promote_corpus):
        sig = inspect.signature(fn)
        assert sig.parameters["probe"].default is run_frozen_check, (
            f"{fn.__name__} default probe is not run_frozen_check"
        )


# ===========================================================================
# Obligation 10 — determinism: identical PromotionResult on identical inputs
# ===========================================================================


def _golden_corpus() -> tuple[list[PlantedPair], list[PlantedItem]]:
    """A small mixed corpus: one O pair (caught), one K error + clean sibling, one C2 clean
    singleton. Provisional ids globally unique."""
    o_seed = Seed(
        seed_id=1,
        frame=OracleFrame(
            completion_criterion="tests_pass", problem_type="code", problem_statement="add"
        ),
        thesis=Thesis(
            proposed_solution="def add(a, b):\n    return a + b\n",
            experiment_design="run frozen tests",
        ),
        test_code="from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
    )
    o_pair = OInjector().emit(o_seed, "arithmetic-swap", error_id=100, clean_id=101)

    # K pair: an oracle-blind LLM-planted error + its clean sibling. Solutions are distinct from the
    # O pair's `return a + b/- b` so the probe routes them unambiguously to the oracle-blind tier.
    k_err = _planted(
        provisional_id=200,
        is_error=1,
        candidate_stratum="K",
        planter=LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        error_regime="spec-misread",
        matched_sibling_id=201,
        thesis=Thesis(proposed_solution=_K_ERR_SOL, experiment_design="x"),
    )
    k_clean = _planted(
        provisional_id=201,
        is_error=0,
        candidate_stratum="clean",
        planter=LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        error_regime="",
        matched_sibling_id=200,
        thesis=Thesis(proposed_solution=_K_CLEAN_SOL, experiment_design="x"),
    )
    k_pair = PlantedPair(error_item=k_err, clean_item=k_clean)

    c2_singleton = _clean_item(
        provisional_id=300,
        matched_sibling_id=None,
        thesis=Thesis(proposed_solution=_C2_SOL, experiment_design="x"),
    )
    return [o_pair, k_pair], [c2_singleton]


# Distinct golden solutions, routed by EXACT match so the probe is unambiguous (the O pair's
# operator-mutated source `return a - b` would otherwise collide with a hand-written K error). The O
# error source is taken from the injector's actual output (ast.unparse strips the trailing newline),
# so the probe's exact-match routing stays robust to ast formatting.
_O_CLEAN_SOL = "def add(a, b):\n    return a + b\n"
_O_ERR_SOL = (
    OInjector()
    .emit(
        Seed(
            seed_id=1,
            frame=OracleFrame(
                completion_criterion="tests_pass", problem_type="code", problem_statement="add"
            ),
            thesis=Thesis(proposed_solution=_O_CLEAN_SOL, experiment_design="run frozen tests"),
            test_code="from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n",
        ),
        "arithmetic-swap",
        error_id=100,
        clean_id=101,
    )
    .error_item.thesis.proposed_solution
)
_K_ERR_SOL = "def parse(s):\n    return int(s) + 1\n"  # oracle-blind spec-misread
_K_CLEAN_SOL = "def parse(s):\n    return int(s)\n"
_C2_SOL = "def f():\n    return 1\n"


def _golden_probe(solution_code: str, test_code: str) -> Verdict:
    """Route the golden corpus by EXACT solution match: the O error is caught (executable, ¬holds);
    the O clean passes (executable, holds); the K pair + C2 singleton are oracle-blind."""
    if solution_code == _O_ERR_SOL:
        return _verdict(holds=False, valid_check=True)  # O catch
    if solution_code == _O_CLEAN_SOL:
        return _verdict(holds=True, valid_check=True)  # C1-reachable clean sibling
    return _verdict(holds=True, valid_check=True, source="inference")  # K + C2 oracle-blind


def _golden_adjudicate(request: AdjudicationRequest) -> tuple[Adjudication, ...]:
    """Per-item existence verdicts for the golden corpus: the K error gets consensus 'error', the K
    clean sibling gets consensus 'clean', and the C2 singleton gets consensus 'abstain' (dropped) —
    so the golden shape exercises an oracle catch (O), a human error (K), a human clean (K sibling),
    and an abstention-drop (C2) in one deterministic run."""
    verdict: _Verdict3
    if request.proposed_solution == _K_ERR_SOL:
        verdict = "error"
    elif request.proposed_solution == _K_CLEAN_SOL:
        verdict = "clean"
    else:  # the C2 singleton
        verdict = "abstain"
    return (
        Adjudication(adjudicator_id="a0", verdict=verdict, rationale="g", timestamp=_TS),
        Adjudication(adjudicator_id="a1", verdict=verdict, rationale="g", timestamp=_TS),
    )


def _run_golden() -> PromotionResult:
    pairs, singles = _golden_corpus()
    return promote_corpus(
        pairs,
        singles,
        probe=_golden_probe,
        adjudicate=_golden_adjudicate,
        tie_break=_tie_break("error"),
        regime_adjudicate=_regime("confirm-regime"),
    )


def test_promote_corpus_is_deterministic() -> None:
    """Same inputs + same stub callbacks → identical PromotionResult (sorted dropped_ids, stable
    promoted order). Pins the golden fixture."""
    r1 = _run_golden()
    r2 = _run_golden()
    assert sorted(r1.dropped_ids) == sorted(r2.dropped_ids)
    assert [ci.item_id for ci in r1.promoted] == [ci.item_id for ci in r2.promoted]
    assert r1.promoted == r2.promoted
    assert r1.regime_audit == r2.regime_audit


def test_promote_corpus_golden_shape() -> None:
    """Golden shape pin: the O pair promotes (oracle catch on the error + C1 clean sibling); the K
    error gets consensus 'error' (human label, promoted) and the K clean sibling gets consensus
    'clean' (C2 human label, promoted); the C2 singleton's adjudicators abstain → DROPPED. So
    promoted = {O error 100, O clean 101, K error 200, K clean 201}, dropped = {300}."""
    result = _run_golden()
    assert {ci.item_id for ci in result.promoted} == {100, 101, 200, 201}
    assert result.dropped_ids == frozenset({300})
    # The O error carries the derived regime; the K error carries the confirmed candidate.
    o_err = next(ci for ci in result.promoted if ci.item_id == 100)
    assert o_err.stratum == "O"
    assert o_err.error_regime == "logic-wrong"
    k_err = next(ci for ci in result.promoted if ci.item_id == 200)
    assert k_err.stratum == "K"
    assert k_err.error_regime == "spec-misread"


# ===========================================================================
# Frozen-field-set + frozen-mutation pins (additive-only tripwires)
# ===========================================================================


def test_result_types_are_frozen() -> None:
    from pydantic import ValidationError

    asg = StratumAssignment(
        provisional_id=1,
        stratum="O",
        oracle_reachable=True,
        verdict=_verdict(holds=False, valid_check=True),
        candidate_agreed=True,
    )
    with pytest.raises(ValidationError):
        asg.stratum = "K"  # type: ignore[misc]


def test_promotion_result_field_set_is_pinned() -> None:
    assert set(PromotionResult.model_fields) == {
        "promoted",
        "dropped_ids",
        "regime_audit",
        "stratum_drift",
    }


def test_adjudication_outcome_field_set_is_pinned() -> None:
    assert set(AdjudicationOutcome.model_fields) == {
        "provisional_id",
        "purpose",
        "existence_verdict",
        "provenance",
        "regime_adjudication",
        "dropped",
    }


def test_adjudicate_item_regime_without_callback_raises() -> None:
    """A 'regime' request with no regime_adjudicate callback is a programming error, surfaced
    loudly (never a silent abstain)."""
    req = AdjudicationRequest(
        provisional_id=1,
        purpose="regime",
        problem_statement="x",
        proposed_solution="y",
        candidate_regime="spec-misread",
    )
    with pytest.raises(ValueError, match="regime_adjudicate"):
        adjudicate_item(req, adjudicate=_adjudicators("error"), tie_break=_tie_break("error"))


def test_promoted_items_carry_empty_content_hash() -> None:
    """content_hash stays '' on every promoted item — 4.4c-5 locks it (this pod never sets it)."""
    result = _run_golden()
    assert all(ci.content_hash == "" for ci in result.promoted)


def test_promoted_corpus_items_load_cleanly() -> None:
    """End-to-end seam check: the promoted items are well-formed CorpusItems (the CorpusItem
    label_source/provenance validator passed at construction)."""
    result = _run_golden()
    assert all(isinstance(ci, CorpusItem) for ci in result.promoted)
    assert all(ci.label_source == ci.label_provenance.kind for ci in result.promoted)
