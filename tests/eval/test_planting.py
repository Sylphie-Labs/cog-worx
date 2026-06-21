"""Deterministic unit tests for the pre-label planting schema (Pod 4.4c-3).

Schema pins only — the injectors, operator->regime table, split draw, and strip-field list are
test-qa-expert's work in the same module. Mirrors ``test_corpus.py``'s pin discipline: a frozen
field-set tripwire per model (additive-only), the PlantedPair structural-invariant pins (each paired
with a passing well-formed control so the assertions are mutation-resistant), frozen-mutation pins,
and the ``error_regime`` default pin. No model calls, no substrate.
"""

from __future__ import annotations

import ast
from collections.abc import Mapping
from pathlib import Path

import pytest
from pydantic import ValidationError

from cogworx.eval._authoring import run_frozen_check
from cogworx.eval.corpus import DeterministicPlanterStamp, DifficultyMarker, LLMPlanterStamp
from cogworx.eval.planting import (
    ANSWER_BEARING_FIELDS,
    DETK_MIN_POOL,
    DETK_PROBE_OPERATOR,
    K_ERROR_KINDS,
    K_REGIME_TO_SPLIT_BUCKET,
    OPERATOR_REGIME_TABLE,
    InfeasibleSplitError,
    KInjector,
    MutationFailed,
    OInjector,
    PlantedItem,
    PlantedPair,
    SameFamilyFallback,
    Seed,
    _derive_o_regime,
    build_detk_pair,
    canonical_split_regime,
    draw_splits,
    is_detk,
    mutate_source,
)
from cogworx.model.base import ModelResponse, Usage
from cogworx.testing.fake_model import ReplayModel
from cogworx.verification.contracts import OracleFrame, Thesis

# ---------------------------------------------------------------------------
# Well-formed builders (the mutation-resistance controls)
# ---------------------------------------------------------------------------

_FRAME = OracleFrame(
    completion_criterion="tests_pass",
    problem_type="code",
    problem_statement="sum a list",
)
_THESIS = Thesis(proposed_solution="return sum(xs)", experiment_design="run frozen tests")


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
        "planter": DeterministicPlanterStamp(operators=("swap-op",)),
        "difficulty": _difficulty(),
        "matched_sibling_id": None,
        "split": "tuning",
    }
    base.update(overrides)
    return PlantedItem(**base)


def _pair(*, split: str = "tuning") -> PlantedPair:
    err = _planted(
        provisional_id=10,
        is_error=1,
        candidate_stratum="O",
        matched_sibling_id=11,
        split=split,
    )
    clean = _planted(
        provisional_id=11,
        is_error=0,
        candidate_stratum="clean",
        matched_sibling_id=10,
        split=split,
    )
    return PlantedPair(error_item=err, clean_item=clean)


def test_wellformed_controls_construct() -> None:
    assert _planted().provisional_id == 1
    assert _pair().error_item.provisional_id == 10


# ---------------------------------------------------------------------------
# 1. Frozen-field-set assertions (the additive-only tripwire)
# ---------------------------------------------------------------------------


def test_planted_item_field_set_is_pinned() -> None:
    assert set(PlantedItem.model_fields) == {
        "provisional_id",
        "frame",
        "thesis",
        "test_code",
        "candidate_stratum",
        "is_error",
        "planter",
        "error_regime",
        "difficulty",
        "matched_sibling_id",
        "split",
    }


def test_planted_pair_field_set_is_pinned() -> None:
    assert set(PlantedPair.model_fields) == {"error_item", "clean_item"}


# ---------------------------------------------------------------------------
# 2. PlantedPair structural-invariant pins (each with a passing control)
# ---------------------------------------------------------------------------


def test_pair_wellformed_constructs() -> None:
    assert _pair().clean_item.candidate_stratum == "clean"


def test_pair_error_member_wrong_is_error_raises() -> None:
    err = _planted(provisional_id=10, is_error=0, matched_sibling_id=11)
    clean = _planted(
        provisional_id=11, is_error=0, candidate_stratum="clean", matched_sibling_id=10
    )
    with pytest.raises(ValidationError):
        PlantedPair(error_item=err, clean_item=clean)


def test_pair_clean_member_wrong_is_error_raises() -> None:
    err = _planted(provisional_id=10, is_error=1, matched_sibling_id=11)
    clean = _planted(
        provisional_id=11, is_error=1, candidate_stratum="clean", matched_sibling_id=10
    )
    with pytest.raises(ValidationError):
        PlantedPair(error_item=err, clean_item=clean)


def test_pair_error_member_non_mutual_sibling_raises() -> None:
    err = _planted(provisional_id=10, is_error=1, matched_sibling_id=999)
    clean = _planted(
        provisional_id=11, is_error=0, candidate_stratum="clean", matched_sibling_id=10
    )
    with pytest.raises(ValidationError):
        PlantedPair(error_item=err, clean_item=clean)


def test_pair_clean_member_non_mutual_sibling_raises() -> None:
    err = _planted(provisional_id=10, is_error=1, matched_sibling_id=11)
    clean = _planted(
        provisional_id=11, is_error=0, candidate_stratum="clean", matched_sibling_id=999
    )
    with pytest.raises(ValidationError):
        PlantedPair(error_item=err, clean_item=clean)


def test_pair_mismatched_split_raises() -> None:
    err = _planted(
        provisional_id=10, is_error=1, matched_sibling_id=11, split="tuning"
    )
    clean = _planted(
        provisional_id=11,
        is_error=0,
        candidate_stratum="clean",
        matched_sibling_id=10,
        split="measurement",
    )
    with pytest.raises(ValidationError):
        PlantedPair(error_item=err, clean_item=clean)


def test_pair_same_split_both_directions_construct() -> None:
    assert _pair(split="tuning").error_item.split == "tuning"
    assert _pair(split="measurement").clean_item.split == "measurement"


def test_pair_clean_member_non_clean_candidate_stratum_raises() -> None:
    err = _planted(provisional_id=10, is_error=1, matched_sibling_id=11)
    clean = _planted(
        provisional_id=11, is_error=0, candidate_stratum="K", matched_sibling_id=10
    )
    with pytest.raises(ValidationError):
        PlantedPair(error_item=err, clean_item=clean)


# ---------------------------------------------------------------------------
# 3. Frozen pins (mutation raises)
# ---------------------------------------------------------------------------


def test_planted_item_is_frozen() -> None:
    item = _planted()
    with pytest.raises(ValidationError):
        item.provisional_id = 2  # type: ignore[misc]


def test_planted_pair_is_frozen() -> None:
    pair = _pair()
    with pytest.raises(ValidationError):
        pair.error_item = pair.clean_item  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 4. error_regime default pin
# ---------------------------------------------------------------------------


def test_error_regime_defaults_empty() -> None:
    assert _planted().error_regime == ""


def test_error_regime_accepts_a_tag() -> None:
    assert _planted(error_regime="off-by-one").error_regime == "off-by-one"


# ===========================================================================
# Pipeline builders (test-qa-expert) — pieces 1-5 + the split draw
# ===========================================================================

_ADD_SOLUTION = "def add(a, b):\n    return a + b\n"
_ADD_TEST = (
    "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
    "    assert add(-1, 1) == 0\n"
)
_SCALE_SOLUTION = "def scale(x):\n    return x * 2\n"
_SCALE_TEST = "from solution import scale\n\n\ndef test_scale():\n    assert scale(3) == 6\n"
_CMP_SOLUTION = "def maximum(a, b):\n    if a > b:\n        return a\n    return b\n"
_CMP_TEST = (
    "from solution import maximum\n\n\ndef test_maximum():\n    assert maximum(3, 7) == 7\n"
    "    assert maximum(9, 2) == 9\n"
)


def _seed(
    *, sol: str = _ADD_SOLUTION, test: str = _ADD_TEST, statement: str = "add", seed_id: int = 0
) -> Seed:
    return Seed(
        seed_id=seed_id,
        frame=OracleFrame(
            completion_criterion="tests_pass", problem_type="code", problem_statement=statement
        ),
        thesis=Thesis(proposed_solution=sol, experiment_design="run the frozen tests"),
        test_code=test,
    )


# ---------------------------------------------------------------------------
# Pin 1 — OPERATOR_REGIME_TABLE totality + _derive_o_regime (the un-gameable core)
# ---------------------------------------------------------------------------


def test_operator_regime_table_is_exactly_the_seven_operators() -> None:
    assert OPERATOR_REGIME_TABLE == {
        "arithmetic-swap": "logic-wrong",
        "relational-swap": "logic-wrong",
        "boundary": "edge-case-miss",
        "statement-deletion": "edge-case-miss",
        "constant-replacement": "off-by-semantics",
        "sign-flip": "off-by-semantics",
        "unit": "off-by-semantics",
    }


@pytest.mark.parametrize(
    ("operator", "regime"),
    [
        ("arithmetic-swap", "logic-wrong"),
        ("relational-swap", "logic-wrong"),
        ("boundary", "edge-case-miss"),
        ("statement-deletion", "edge-case-miss"),
        ("constant-replacement", "off-by-semantics"),
        ("sign-flip", "off-by-semantics"),
        ("unit", "off-by-semantics"),
    ],
)
def test_derive_o_regime_maps_each_operator(operator: str, regime: str) -> None:
    """Passing control: every table operator derives to its regime (mutation-resistant — the
    parametrize covers the full vocabulary, so a dropped/renamed key fails here)."""
    assert _derive_o_regime((operator,)) == regime


def test_derive_o_regime_unknown_operator_raises() -> None:
    """The §2.C un-gameable guarantee: an operator not in the table RAISES — no silent default."""
    with pytest.raises(KeyError, match="not in OPERATOR_REGIME_TABLE"):
        _derive_o_regime(("not-an-operator",))


def test_derive_o_regime_multi_regime_class_within_one_item_raises() -> None:
    """One regime-class per item (architect ruling): two operators spanning two regime classes is
    rejected so the derivation stays unambiguous."""
    with pytest.raises(ValueError, match="regime class"):
        _derive_o_regime(("arithmetic-swap", "boundary"))  # logic-wrong + edge-case-miss


def test_derive_o_regime_two_operators_same_class_is_allowed() -> None:
    """Control: two operators in the SAME regime class derive cleanly (it is class count, not
    operator count, that must be 1)."""
    assert _derive_o_regime(("constant-replacement", "sign-flip")) == "off-by-semantics"


def test_derive_o_regime_strips_detk_sentinel() -> None:
    """The detK control sentinel is stripped before the table read; the regime derives off the real
    operator (off-by-semantics)."""
    assert _derive_o_regime(("sign-flip", DETK_PROBE_OPERATOR)) == "off-by-semantics"


def test_derive_o_regime_bare_sentinel_raises() -> None:
    """A deterministic item carrying ONLY the control sentinel (no genuine error) is a corpus
    defect — RAISES rather than returning a regime."""
    with pytest.raises(ValueError, match="no genuine mutation operator"):
        _derive_o_regime((DETK_PROBE_OPERATOR,))


def test_no_table_key_uses_the_control_namespace() -> None:
    """The eval-stats firewall invariant: a real operator can never collide with a control token."""
    assert not any(op.startswith("__") for op in OPERATOR_REGIME_TABLE)


# ---------------------------------------------------------------------------
# Pin 2 — O injector: real mutation, regime stamp, mutate-then-revert pair
# ---------------------------------------------------------------------------


def test_o_injector_emits_pair_with_operator_derived_regime() -> None:
    pair = OInjector().emit(_seed(), "arithmetic-swap", error_id=100, clean_id=101)
    assert isinstance(pair, PlantedPair)
    assert pair.error_item.error_regime == _derive_o_regime(("arithmetic-swap",)) == "logic-wrong"
    assert pair.error_item.candidate_stratum == "O"
    assert pair.clean_item.candidate_stratum == "clean"
    # mutate-then-revert: the clean sibling carries the ORIGINAL un-mutated solution.
    assert pair.clean_item.thesis.proposed_solution == _ADD_SOLUTION
    assert pair.error_item.thesis.proposed_solution != _ADD_SOLUTION


# A seed whose source carries an applicable site for each operator (FIX#4.3 binding test).
_OP_SEED_SOURCE: dict[str, str] = {
    "arithmetic-swap": _ADD_SOLUTION,  # a + b
    "relational-swap": _CMP_SOLUTION,  # if a > b
    "boundary": _CMP_SOLUTION,  # if a > b (relational op boundary)
    "statement-deletion": _CMP_SOLUTION,  # 2-statement function body
    "constant-replacement": _SCALE_SOLUTION,  # x * 2
    "sign-flip": _SCALE_SOLUTION,  # x * 2
    "unit": _SCALE_SOLUTION,  # x * 2
}


@pytest.mark.parametrize("operator", list(OPERATOR_REGIME_TABLE))
def test_o_injector_stamp_mutation_regime_share_one_operator_binding(operator: str) -> None:
    """FIX#4.3 (S9): the un-gameable claim's v1 footing made green. ``OInjector.emit`` derives the
    STAMP, the MUTATION, and the regime from ONE operator binding — so they cannot diverge there.
    This pins: (1) the stamp records exactly the operator emit was asked to apply; (2) the source IS
    mutated (not a silent no-op); (3) ``error_regime == _derive_o_regime(planter.operators)``. If a
    refactor splits the binding (stamp operator A, mutate operator B), this breaks FIRST."""
    seed = _seed(sol=_OP_SEED_SOURCE[operator])
    pair = OInjector().emit(seed, operator, error_id=1, clean_id=2)  # type: ignore[arg-type]
    stamp = pair.error_item.planter
    assert isinstance(stamp, DeterministicPlanterStamp)
    # (1) the stamp records exactly the requested operator
    assert stamp.operators == (operator,)
    # (2) the mutation is real (source changed) — what mutate_source was asked to do, it did
    assert pair.error_item.thesis.proposed_solution != _OP_SEED_SOURCE[operator]
    assert pair.error_item.thesis.proposed_solution == mutate_source(
        _OP_SEED_SOURCE[operator], operator  # type: ignore[arg-type]
    )
    # (3) the regime derives off the SAME stamped operators (one binding)
    assert pair.error_item.error_regime == _derive_o_regime(stamp.operators)


@pytest.mark.parametrize("operator", ["constant-replacement", "sign-flip", "unit"])
def test_detk_stamp_mutation_regime_share_one_operator_binding(operator: str) -> None:
    """FIX#4.3 (S9): the same single-binding pin for ``build_detk_pair`` (off-by-semantics only).
    The stamp carries the real operator (+ the detK sentinel), the source is the operator's
    mutation, and ``error_regime == _derive_o_regime(planter.operators)`` (sentinel stripped)."""
    seed = _seed(sol=_OP_SEED_SOURCE[operator])
    pair = build_detk_pair(seed, operator, error_id=1, clean_id=2)  # type: ignore[arg-type]
    stamp = pair.error_item.planter
    assert isinstance(stamp, DeterministicPlanterStamp)
    assert stamp.operators == (operator, DETK_PROBE_OPERATOR)
    assert pair.error_item.thesis.proposed_solution == mutate_source(
        _OP_SEED_SOURCE[operator], operator  # type: ignore[arg-type]
    )
    assert pair.error_item.error_regime == _derive_o_regime(stamp.operators)


def test_o_injector_stamps_the_single_operator() -> None:
    pair = OInjector().emit(
        _seed(sol=_CMP_SOLUTION, test=_CMP_TEST), "relational-swap", error_id=1, clean_id=2
    )
    stamp = pair.error_item.planter
    assert isinstance(stamp, DeterministicPlanterStamp)
    assert stamp.operators == ("relational-swap",)  # ONE regime-bearing operator per item


def test_o_injector_mutation_is_real_failure_clean_passes() -> None:
    """The mutation actually FAILS the frozen test (a real refutation, returncode 1) while the clean
    sibling passes — the mutation is not a no-op. Uses the journal-free oracle kernel (S1-clean)."""
    pair = OInjector().emit(_seed(), "arithmetic-swap", error_id=1, clean_id=2)
    v_clean = run_frozen_check(pair.clean_item.thesis.proposed_solution, _ADD_TEST, timeout_s=30.0)
    v_mut = run_frozen_check(pair.error_item.thesis.proposed_solution, _ADD_TEST, timeout_s=30.0)
    assert (v_clean.holds, v_clean.valid_check) == (True, True)
    assert v_mut.holds is False
    assert v_mut.valid_check is True  # tests ran and failed — a genuine error, not a crash


def test_mutate_source_no_applicable_site_raises_not_silent_noop() -> None:
    """A sign-flip on a constant-free source has no site — MutationFailed, never a silent no-op that
    would emit a fake error identical to its clean sibling (the §3.6 realism poison)."""
    with pytest.raises(MutationFailed):
        mutate_source(_ADD_SOLUTION, "sign-flip")


def test_mutate_source_arithmetic_swap_changes_operator() -> None:
    out = mutate_source(_SCALE_SOLUTION, "arithmetic-swap")
    assert "*" not in out.replace("**", "")  # the Mult became something else
    assert out != _SCALE_SOLUTION


# ---------------------------------------------------------------------------
# Pin 3 — K injector (ReplayModel stub): well-formed item, LLM stamp, loud fallback
# ---------------------------------------------------------------------------


def _replay(text: str) -> ReplayModel:
    return ReplayModel(
        [ModelResponse(text=text, model_id="other-fam/x", finish_reason="stop", usage=Usage())]
    )


def _k_injector(model: ReplayModel) -> KInjector:
    return KInjector(
        model,
        model_family="other-fam",
        model_id="other-fam/x",
        thesis_family="claude",
        antithesis_family="deepseek",
    )


async def test_k_injector_parses_scripted_response_into_wellformed_item() -> None:
    inj = _k_injector(_replay("def add(a, b):\n    return a - b\n"))
    pair, usage = await inj.plant(_seed(), "spec-misread", error_id=1, clean_id=2)
    assert isinstance(pair, PlantedPair)
    assert pair.error_item.candidate_stratum == "K"
    assert pair.error_item.is_error == 1
    # error_regime is a CANDIDATE (the error_kind), not operator-derived.
    assert pair.error_item.error_regime == "spec-misread"
    assert pair.clean_item.thesis.proposed_solution == _ADD_SOLUTION
    assert isinstance(usage, Usage)


async def test_k_injector_stamps_llm_planter_from_injected_model() -> None:
    inj = _k_injector(_replay("def add(a, b):\n    return a * b\n"))
    pair, _ = await inj.plant(_seed(), "silent-degradation", error_id=1, clean_id=2)
    stamp = pair.error_item.planter
    assert isinstance(stamp, LLMPlanterStamp)
    assert stamp.model_family == "other-fam"
    assert stamp.model_id == "other-fam/x"


def test_k_injector_same_family_fallback_raises_and_logs(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """CF-4.4c-PLANTER: a planting family colliding with an arm family is a LOUD, logged failure at
    build time — never a silent same-family stamp."""
    with caplog.at_level("ERROR"), pytest.raises(SameFamilyFallback):
        KInjector(
            _replay("x"),
            model_family="claude",  # collides with thesis_family
            model_id="claude/y",
            thesis_family="claude",
            antithesis_family="deepseek",
        )
    assert any("same_family_fallback" in r.message for r in caplog.records)


def test_k_injector_antithesis_family_collision_also_raises() -> None:
    """Negative control: collision with the ANTITHESIS family (not just thesis) also raises."""
    with pytest.raises(SameFamilyFallback):
        KInjector(
            _replay("x"),
            model_family="deepseek",
            model_id="deepseek/z",
            thesis_family="claude",
            antithesis_family="deepseek",
        )


def test_k_injector_cross_family_constructs_clean() -> None:
    """Passing control: a genuinely cross-family planter constructs without raising."""
    inj = _k_injector(_replay("x"))
    assert isinstance(inj.stamp, LLMPlanterStamp)


# ---------------------------------------------------------------------------
# Pin 4 — split draw: stratum marginal, Hamilton regime cells, co-location, feasibility
# ---------------------------------------------------------------------------


def _o_pair(error_id: int, clean_id: int, regime: str) -> PlantedPair:
    err = _planted(
        provisional_id=error_id,
        is_error=1,
        candidate_stratum="O",
        error_regime=regime,
        matched_sibling_id=clean_id,
        split="tuning",
    )
    clean = _planted(
        provisional_id=clean_id,
        is_error=0,
        candidate_stratum="clean",
        error_regime="",
        matched_sibling_id=error_id,
        split="tuning",
    )
    return PlantedPair(error_item=err, clean_item=clean)


def _o_corpus_80_pairs() -> list[PlantedPair]:
    """80 O pairs spread across the 3 regime cells (27/27/26) — the plan's 0.7*80=56 headline."""
    regimes = ["logic-wrong"] * 27 + ["edge-case-miss"] * 27 + ["off-by-semantics"] * 26
    pairs = []
    pid = 0
    for r in regimes:
        pairs.append(_o_pair(pid, pid + 1, r))
        pid += 2
    return pairs


def test_split_draw_stratum_marginal_hits_planned_n() -> None:
    pairs = _o_corpus_80_pairs()
    asg = draw_splits(pairs, [])
    meas = sum(1 for p in pairs if asg[p.error_item.provisional_id] == "measurement")
    assert meas == round(0.70 * len(pairs)) == 56


def test_split_draw_regime_cells_sum_to_stratum_via_hamilton() -> None:
    pairs = _o_corpus_80_pairs()
    asg = draw_splits(pairs, [])
    per_cell = {}
    for regime in ("logic-wrong", "edge-case-miss", "off-by-semantics"):
        cell = [p for p in pairs if p.error_item.error_regime == regime]
        per_cell[regime] = sum(
            1 for p in cell if asg[p.error_item.provisional_id] == "measurement"
        )
    # Hamilton over {27,27,26}: 19+19+18 = 56, summing EXACTLY to the stratum target.
    assert per_cell == {"logic-wrong": 19, "edge-case-miss": 19, "off-by-semantics": 18}
    assert sum(per_cell.values()) == 56


def test_split_draw_hamilton_five_equal_cells_sum_exact() -> None:
    """The plan's apportionment headline: 0.7*16=11.2 over 5 equal regime cells -> {12,11,11,11,11}
    = 56, never 5*11=55. Exercised via _hamilton_apportion through a 5-named-cell stratum proxy."""
    from cogworx.eval.planting import _hamilton_apportion

    alloc = _hamilton_apportion(56, [11.2] * 5, ["a", "b", "c", "d", "e"])
    assert sorted(alloc, reverse=True) == [12, 11, 11, 11, 11]
    assert sum(alloc) == 56


def test_split_draw_pairs_are_co_located() -> None:
    pairs = _o_corpus_80_pairs()
    asg = draw_splits(pairs, [])
    for p in pairs:
        assert asg[p.error_item.provisional_id] == asg[p.clean_item.provisional_id]


def test_split_draw_is_deterministic() -> None:
    pairs = _o_corpus_80_pairs()
    assert draw_splits(pairs, []) == draw_splits(pairs, [])


def test_split_draw_singleton_topup_feasible() -> None:
    """Control: a feasible singleton top-up assigns the right measurement count for a clean
    singleton pool."""
    singles = [
        _planted(
            provisional_id=500 + i,
            is_error=0,
            candidate_stratum="clean",
            error_regime="",
            matched_sibling_id=None,
        )
        for i in range(10)
    ]
    asg = draw_splits([], singles)
    meas = sum(1 for s in singles if asg[s.provisional_id] == "measurement")
    assert meas == round(0.70 * 10) == 7


# --- FIX#2: draw_splits clean-stratum clamp — eval-stats mutation-resistant T1-T7 -------------
# These assert on REALIZED marginals + the no-raise property, NOT internal call counts. They pin the
# clean-stratum false-refuse fix: on an HONEST corpus (no clean singletons) the per-stratum-rounded
# clean inheritance can leave a +/-1 deficit the OLD single-round refused as InfeasibleSplitError.


def _k_pair(error_id: int, clean_id: int, regime: str) -> PlantedPair:
    """A K pair whose clean sibling co-locates in the clean stratum (the clean union is K-siblings +
    O-siblings). The error member is candidate_stratum='K' carrying a real regime."""
    err = _planted(
        provisional_id=error_id,
        is_error=1,
        candidate_stratum="K",
        error_regime=regime,
        planter=LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        matched_sibling_id=clean_id,
        split="tuning",
    )
    clean = _planted(
        provisional_id=clean_id,
        is_error=0,
        candidate_stratum="clean",
        error_regime="",
        planter=LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        matched_sibling_id=error_id,
        split="tuning",
    )
    return PlantedPair(error_item=err, clean_item=clean)


def _meas_count(asg: Mapping[int, str], items: list[PlantedItem]) -> int:
    return sum(1 for it in items if asg[it.provisional_id] == "measurement")


def _clean_members(pairs: list[PlantedPair]) -> list[PlantedItem]:
    return [
        m
        for p in pairs
        for m in (p.error_item, p.clean_item)
        if m.candidate_stratum == "clean"
    ]


_REGIMES_3 = ("logic-wrong", "edge-case-miss", "off-by-semantics")


def _balanced_pairs(nK: int, nO: int) -> list[PlantedPair]:
    """nK K-pairs spread over the 3 regimes + nO O-pairs spread over the 3 regimes, all with their
    clean sibling in the clean stratum. Provisional ids are globally unique."""
    pairs: list[PlantedPair] = []
    pid = 0
    for i in range(nK):
        pairs.append(_k_pair(pid, pid + 1, _REGIMES_3[i % 3]))
        pid += 2
    for i in range(nO):
        pairs.append(_o_pair(pid, pid + 1, _REGIMES_3[i % 3]))
        pid += 2
    return pairs


def test_split_draw_T1_honest_corpus_no_clean_singletons_does_not_raise() -> None:
    """T1 (the falsifier): 1 K-pair + 1 O-pair, 0 clean singletons -> no raise (the OLD single-round
    clean deficit was -1 here and falsely refused). clean meas count == 2 (both error siblings go to
    measurement at their regime cells of size 1, and clean siblings follow); both clean siblings
    match their error sibling's split."""
    pairs = _balanced_pairs(1, 1)
    asg = draw_splits(pairs, [])  # MUST NOT raise
    cleans = _clean_members(pairs)
    assert _meas_count(asg, cleans) == 2
    for p in pairs:
        assert asg[p.error_item.provisional_id] == asg[p.clean_item.provisional_id]


def test_split_draw_T2_planned_n_exact() -> None:
    """T2 (planned n exact): 80 K-pairs + 80 O-pairs, 0 singletons → K meas==56, O meas==56,
    clean meas==112, clean frac==0.70 exactly."""
    pairs = _balanced_pairs(80, 80)
    asg = draw_splits(pairs, [])
    k_err = [p.error_item for p in pairs if p.error_item.candidate_stratum == "K"]
    o_err = [p.error_item for p in pairs if p.error_item.candidate_stratum == "O"]
    cleans = _clean_members(pairs)
    assert _meas_count(asg, k_err) == 56
    assert _meas_count(asg, o_err) == 56
    assert _meas_count(asg, cleans) == 112
    assert _meas_count(asg, cleans) / len(cleans) == 0.70


def test_split_draw_T3_symmetric_under_placement_does_not_raise() -> None:
    """T3 (symmetric under-placement): nK=2, nO=2, 0 clean singletons -> raw clean deficit = +1 >
    S=0; a ONE-SIDED clamp (clamping only the low end) would let this exceed |singletons| or
    mis-handle — the both-directions clamp keeps it at 0 extra. No raise; clean meas == 2."""
    pairs = _balanced_pairs(2, 2)
    asg = draw_splits(pairs, [])  # MUST NOT raise
    cleans = _clean_members(pairs)
    # 0.7*2=1.4->1 meas per K and per O stratum (step 1); clean follows its sibling -> 2 clean meas.
    assert _meas_count(asg, cleans) == 2


@pytest.mark.parametrize(("nK", "nO"), [(69, 72), (71, 88)])
def test_split_draw_T4_odd_counts_clean_frac_within_one_item(nK: int, nO: int) -> None:
    """T4 (odd counts): nK/nO odd, 0 clean singletons -> no raise; the clean frac is within
    1/|clean| of 0.70 (the pairing-induced rounding band)."""
    pairs = _balanced_pairs(nK, nO)
    asg = draw_splits(pairs, [])  # MUST NOT raise
    cleans = _clean_members(pairs)
    frac = _meas_count(asg, cleans) / len(cleans)
    assert abs(frac - 0.70) <= 1 / len(cleans)


def test_split_draw_T5_singletons_nudge_frac_within_band() -> None:
    """T5 (singletons nudge): nK=80, nO=80, +16 clean singletons → clean frac within ±1/176 of 0.70;
    no raise."""
    pairs = _balanced_pairs(80, 80)
    singles = [
        _planted(
            provisional_id=10_000 + i,
            is_error=0,
            candidate_stratum="clean",
            error_regime="",
            matched_sibling_id=None,
        )
        for i in range(16)
    ]
    asg = draw_splits(pairs, singles)  # MUST NOT raise
    cleans = _clean_members(pairs) + singles
    assert len(cleans) == 176
    frac = _meas_count(asg, cleans) / len(cleans)
    assert abs(frac - 0.70) <= 1 / 176


def test_split_draw_T6_check_deficit_still_raises_on_genuine_infeasibility() -> None:
    """T6 (guard still real — the delete-survives discipline): a genuinely-infeasible triple
    (placement > stratum_size, so deficit far outside [0, n_singletons]) RAISES. Deleting the
    ``_check_deficit`` raise must make THIS fail while T1-T5,T7 still pass — i.e. the guard is a
    real tripwire on internal inconsistency, not a refuse-to-lock on a valid-but-skewed corpus."""
    from cogworx.eval.planting import _check_deficit

    with pytest.raises(InfeasibleSplitError, match="outside"):
        _check_deficit("K", deficit=-92, n_singletons=0)  # negative deficit (over-placed)
    with pytest.raises(InfeasibleSplitError, match="outside"):
        _check_deficit("O", deficit=5, n_singletons=2)  # placement deficit exceeds singletons


@pytest.mark.parametrize("nK", [30, 50, 70, 90])
@pytest.mark.parametrize("nO", [30, 50, 70, 90])
@pytest.mark.parametrize("nS", [0, 8, 16, 24])
def test_split_draw_T7_property_no_refuse_every_stratum_near_target(
    nK: int, nO: int, nS: int
) -> None:
    """T7 (property/no-refuse): over nK,nO ∈ {30..90}, S ∈ {0,8,16,24}, draw_splits NEVER raises and
    every stratum frac is within 1/total of 0.70."""
    pairs = _balanced_pairs(nK, nO)
    singles = [
        _planted(
            provisional_id=20_000 + i,
            is_error=0,
            candidate_stratum="clean",
            error_regime="",
            matched_sibling_id=None,
        )
        for i in range(nS)
    ]
    asg = draw_splits(pairs, singles)  # MUST NOT raise

    k_items = [p.error_item for p in pairs if p.error_item.candidate_stratum == "K"]
    o_items = [p.error_item for p in pairs if p.error_item.candidate_stratum == "O"]
    clean_items = _clean_members(pairs) + singles
    for items in (k_items, o_items, clean_items):
        total = len(items)
        if total == 0:
            continue
        frac = _meas_count(asg, items) / total
        assert abs(frac - 0.70) <= 1 / total


# ---------------------------------------------------------------------------
# Pin 4b — regime->canon split projection (4.4c-6b fix): raw K kinds draw a feasible split,
# the round-robin keeps all 3 buckets reachable, and the candidate tag is NEVER laundered.
# ---------------------------------------------------------------------------


def _raw_k_pair(error_id: int, clean_id: int, kind: str) -> PlantedPair:
    """A K pair whose error member carries a RAW candidate K kind (spec-misread /
    silent-degradation) — exactly what ``KInjector.plant`` stamps. This is the crash scenario: the
    raw kind is NOT a canonical split bucket, so pre-fix ``draw_splits`` skipped/monocultured it."""
    err = _planted(
        provisional_id=error_id,
        is_error=1,
        candidate_stratum="K",
        error_regime=kind,
        planter=LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        matched_sibling_id=clean_id,
        split="tuning",
    )
    clean = _planted(
        provisional_id=clean_id,
        is_error=0,
        candidate_stratum="clean",
        error_regime="",
        planter=LLMPlanterStamp(model_family="other-fam", model_id="other-fam/x"),
        matched_sibling_id=error_id,
        split="tuning",
    )
    return PlantedPair(error_item=err, clean_item=clean)


def _raw_k_corpus(n_per_kind: int) -> list[PlantedPair]:
    """``n_per_kind`` K-pairs for EACH raw K kind (both kinds present), globally-unique ids."""
    pairs: list[PlantedPair] = []
    pid = 0
    for kind in K_ERROR_KINDS:
        for _ in range(n_per_kind):
            pairs.append(_raw_k_pair(pid, pid + 1, kind))
            pid += 2
    return pairs


def test_k_regime_projection_domain_is_exactly_k_error_kinds() -> None:
    """The projection's domain is EXACTLY the 2 oracle-blind K kinds (a drift in either vocabulary
    trips here, not silently in the draw)."""
    assert set(K_REGIME_TO_SPLIT_BUCKET) == set(K_ERROR_KINDS)


def test_k_regime_projection_image_covers_all_three_buckets() -> None:
    """eval-stats A1: the union image must cover all 3 canonical buckets — a 2->1 scalar map would
    permanently zero one bucket (a silent 2-regime K monoculture)."""
    image = {b for buckets in K_REGIME_TO_SPLIT_BUCKET.values() for b in buckets}
    assert image == {"logic-wrong", "edge-case-miss", "off-by-semantics"}
    # each kind maps across exactly two buckets (degeneracy-avoidance)
    assert all(len(buckets) == 2 for buckets in K_REGIME_TO_SPLIT_BUCKET.values())
    assert K_REGIME_TO_SPLIT_BUCKET["spec-misread"] == ("logic-wrong", "edge-case-miss")
    assert K_REGIME_TO_SPLIT_BUCKET["silent-degradation"] == ("off-by-semantics", "edge-case-miss")


def test_canonical_split_regime_passes_through_o_regimes_unchanged() -> None:
    """Behavior-preserving: a deterministic-O / detK item already carries a canonical regime —
    ``canonical_split_regime`` returns it UNCHANGED (the already-passing O/detK paths)."""
    for regime in ("logic-wrong", "edge-case-miss", "off-by-semantics"):
        item = _planted(candidate_stratum="O", error_regime=regime)
        assert canonical_split_regime(item) == regime


def test_canonical_split_regime_projects_raw_k_kind_into_a_bucket() -> None:
    """A raw K candidate tag projects into ONE of its kind's two split buckets — never left raw."""
    for kind in K_ERROR_KINDS:
        item = _planted(candidate_stratum="K", error_regime=kind)
        bucket = canonical_split_regime(item)
        assert bucket in K_REGIME_TO_SPLIT_BUCKET[kind]
        assert bucket in ("logic-wrong", "edge-case-miss", "off-by-semantics")


def test_canonical_split_regime_unknown_regime_raises_loudly() -> None:
    """Mirrors ``_derive_o_regime``: a regime that is neither canonical nor a known K kind RAISES —
    no silent default."""
    item = _planted(candidate_stratum="K", error_regime="not-a-regime")
    with pytest.raises(ValueError, match="no silent default"):
        canonical_split_regime(item)


def test_canonical_split_regime_is_pure_does_not_mutate_item() -> None:
    """The laundering wall (unit level): calling the helper does NOT change ``item.error_regime`` —
    the raw candidate tag is preserved. (PlantedItem is frozen, so a write would also raise.)"""
    item = _planted(candidate_stratum="K", error_regime="spec-misread")
    _ = canonical_split_regime(item)
    assert item.error_regime == "spec-misread"


def test_split_draw_raw_k_corpus_does_not_raise() -> None:
    """THE 4.4c-6b CRASH SCENARIO: a corpus of RAW K-kind tags (spec-misread / silent-degradation)
    now draws a FEASIBLE split. Pre-fix, step-1 read the raw candidate tag (not a canonical bucket)
    and either skipped every K pair (monoculture) or mis-counted -> InfeasibleSplitError."""
    pairs = _raw_k_corpus(20)  # 20 spec-misread + 20 silent-degradation K-pairs
    asg = draw_splits(pairs, [])  # MUST NOT raise
    k_err = [p.error_item for p in pairs]
    # the K stratum hits its 0.70 marginal (not monocultured / under-placed)
    assert _meas_count(asg, k_err) == round(0.70 * len(k_err)) == 28


def test_split_draw_raw_k_round_robin_keeps_all_buckets_reachable() -> None:
    """Round-robin reachability: with BOTH K kinds present, every one of the 3 K-split buckets
    receives at least one pair (no permanently-empty bucket — the monoculture the 2->1 map caused).
    Asserted on the ALLOCATION buckets via the same projection draw_splits uses."""
    pairs = _raw_k_corpus(20)
    buckets_hit = {canonical_split_regime(p.error_item) for p in pairs}
    assert buckets_hit == {"logic-wrong", "edge-case-miss", "off-by-semantics"}


def test_split_draw_does_not_launder_raw_k_candidate_tag() -> None:
    """THE RED-TEAM-CRITICAL INVARIANT (laundering wall): after ``draw_splits``, every K item's
    ``error_regime`` is STILL its original raw candidate (spec-misread / silent-degradation). The
    split bucket is allocation-only — it must NEVER be written back to ``error_regime`` (which feeds
    Cell.regime / the §2.B contribution bound + the §2.C second-author audit)."""
    pairs = _raw_k_corpus(20)
    raw_before = {p.error_item.provisional_id: p.error_item.error_regime for p in pairs}
    _ = draw_splits(pairs, [])
    for p in pairs:
        # the tag is unchanged AND is still a raw K kind, never a canonical split bucket
        assert p.error_item.error_regime == raw_before[p.error_item.provisional_id]
        assert p.error_item.error_regime in K_ERROR_KINDS
        assert p.error_item.error_regime not in (
            "logic-wrong",
            "edge-case-miss",
            "off-by-semantics",
        )


def test_split_draw_raw_k_is_reproducible_under_split_seed() -> None:
    """Reproducibility: same SPLIT_SEED (frozen) -> identical bucket assignment + identical split
    draw across two independent calls on the raw-K corpus."""
    pairs = _raw_k_corpus(20)
    assert draw_splits(pairs, []) == draw_splits(pairs, [])
    # the round-robin bucket choice is itself reproducible (folds into SPLIT_SEED)
    first = [canonical_split_regime(p.error_item) for p in pairs]
    second = [canonical_split_regime(p.error_item) for p in pairs]
    assert first == second


def test_split_draw_o_path_byte_identical_after_fix() -> None:
    """Behavior-preserving guard: the O draw is UNCHANGED by the projection (O regimes pass through
    canonical_split_regime untouched) — same assignment as the canonical 80-pair O headline."""
    pairs = _o_corpus_80_pairs()
    asg = draw_splits(pairs, [])
    per_cell = {}
    for regime in ("logic-wrong", "edge-case-miss", "off-by-semantics"):
        cell = [p for p in pairs if p.error_item.error_regime == regime]
        per_cell[regime] = sum(
            1 for p in cell if asg[p.error_item.provisional_id] == "measurement"
        )
    assert per_cell == {"logic-wrong": 19, "edge-case-miss": 19, "off-by-semantics": 18}


# --- The mutation test: reverting draw_splits to read error_regime directly re-breaks ---------


def test_mutation_reading_raw_error_regime_directly_reproduces_the_crash() -> None:
    """MUTATION-RESISTANCE (the spec's required mutation test): a faithful re-creation of the
    PRE-FIX step-1 placement — keying the cell on the RAW ``err.error_regime`` with the old
    ``in cell_pairs`` skip-guard — monocultures the K stratum (every raw-K pair is skipped because
    its tag is not a canonical bucket), so the K measurement marginal collapses to 0. This pins that
    the fix (``canonical_split_regime``) is load-bearing: revert it and this regime-projection guard
    fails. (The original symptom was crash-OR-monoculture; we assert the monoculture branch, which
    is deterministic and does not depend on singleton mis-counts.)"""
    from cogworx.eval.planting import _REGIMES

    pairs = _raw_k_corpus(20)
    # Re-create the buggy step-1 cell assignment verbatim (raw read + skip-guard).
    buggy_cells: dict[str, list[PlantedPair]] = {r: [] for r in _REGIMES}
    placed = 0
    for pair in pairs:
        err = pair.error_item
        if err.candidate_stratum == "K" and err.error_regime in buggy_cells:  # the pre-fix guard
            buggy_cells[err.error_regime].append(pair)
            placed += 1
    # Pre-fix: NO raw-K pair lands in any canonical cell -> a total monoculture (all skipped).
    assert placed == 0
    # The fix routes all of them through canonical_split_regime -> they DO land.
    fixed_placed = sum(
        1
        for pair in pairs
        if canonical_split_regime(pair.error_item) in _REGIMES
    )
    assert fixed_placed == len(pairs) == 40


# ---------------------------------------------------------------------------
# Pin 5 — detK: additional pool, >=30, measurement, reported-only/excludable
# ---------------------------------------------------------------------------


def test_detk_pool_floor_is_thirty() -> None:
    assert DETK_MIN_POOL >= 30


def test_detk_pair_is_measurement_split_and_tagged() -> None:
    pair = build_detk_pair(_seed(sol=_SCALE_SOLUTION, test=_SCALE_TEST), "constant-replacement",
                           error_id=1, clean_id=2)
    assert pair.error_item.split == "measurement"
    assert pair.clean_item.split == "measurement"
    assert is_detk(pair.error_item) is True
    # The genuine regime still derives off the real operator (off-by-semantics).
    assert pair.error_item.error_regime == "off-by-semantics"
    # candidate_stratum is K (oracle-blind, K-shaped) — audit-only proposal.
    assert pair.error_item.candidate_stratum == "K"


def test_detk_stamp_carries_real_operator_plus_sentinel() -> None:
    pair = build_detk_pair(_seed(sol=_SCALE_SOLUTION, test=_SCALE_TEST), "sign-flip",
                           error_id=1, clean_id=2)
    stamp = pair.error_item.planter
    assert isinstance(stamp, DeterministicPlanterStamp)
    assert "sign-flip" in stamp.operators
    assert DETK_PROBE_OPERATOR in stamp.operators


def test_detk_is_not_a_plain_o_item() -> None:
    """The exclusion predicate distinguishes detK from a binding O item carrying the same operator —
    a plain O item is NOT detK (so it stays in the binding deltas), a detK item IS (so 4.4d excludes
    it). This is the firewall that keeps detK out of error_pools['K']."""
    o_pair = OInjector().emit(_seed(sol=_SCALE_SOLUTION, test=_SCALE_TEST), "constant-replacement",
                              error_id=1, clean_id=2)
    dk_pair = build_detk_pair(_seed(sol=_SCALE_SOLUTION, test=_SCALE_TEST), "constant-replacement",
                              error_id=3, clean_id=4)
    assert is_detk(o_pair.error_item) is False  # plain O -> stays binding
    assert is_detk(dk_pair.error_item) is True  # detK -> excluded


def test_detk_requires_off_by_semantics_operator() -> None:
    """Negative control: detK is off-by-semantics by construction; a logic-wrong operator is
    rejected (the collusion probe's positive signature depends on the regime)."""
    with pytest.raises(ValueError, match="off-by-semantics"):
        build_detk_pair(_seed(sol=_SCALE_SOLUTION, test=_SCALE_TEST), "arithmetic-swap",
                        error_id=1, clean_id=2)


# ---------------------------------------------------------------------------
# Pin 6 — seed self-check (red-team bait #5)
# ---------------------------------------------------------------------------

# The full seed sweep spawns one pytest subprocess per seed — opt-in (slow) per the hang discipline.
# The default loop runs a 3-seed subset + the negative control, which still pins the mechanism.

_SEED_SUBSET_IDS = (0, 11, 20)  # one arithmetic, one boundary/compare, one unit seed


def _load_seeds() -> tuple[Seed, ...]:
    from fixtures.eval.seeds import SEEDS

    return SEEDS


def test_seed_corpus_is_floor_sized() -> None:
    seeds = _load_seeds()
    assert len(seeds) >= 25


@pytest.mark.parametrize("seed_id", _SEED_SUBSET_IDS)
def test_seed_subset_passes_frozen_self_check(seed_id: int) -> None:
    """A representative subset of seeds passes CodeOracle's frozen kernel (holds ∧ valid_check)."""
    seed = _load_seeds()[seed_id]
    v = run_frozen_check(seed.thesis.proposed_solution, seed.test_code, timeout_s=30.0)
    assert v.holds is True
    assert v.valid_check is True


def test_deliberately_wrong_seed_is_rejected() -> None:
    """Negative control (red-team bait #5): a seed whose 'correct' solution fails its own frozen
    test does NOT pass the self-check — the gate that keeps a poisoned seed out of the set."""
    bad = _seed(sol="def add(a, b):\n    return a - b\n", test=_ADD_TEST)  # wrong solution
    v = run_frozen_check(bad.thesis.proposed_solution, bad.test_code, timeout_s=30.0)
    assert v.holds is False


@pytest.mark.parametrize(
    ("solution", "test", "expect_holds"),
    [
        (_ADD_SOLUTION, _ADD_TEST, True),  # passing case
        ("def add(a, b):\n    return a - b\n", _ADD_TEST, False),  # failing case
    ],
)
async def test_run_frozen_check_equals_code_oracle_frozen_path(
    solution: str, test: str, expect_holds: bool
) -> None:
    """FIX#3 (S9): pin the docstring equivalence claim. ``run_frozen_check(sol, test)`` must return
    the same Verdict (on the fields that matter) as ``CodeOracle(test_source="frozen",
    frozen_test_code=test).evaluate(...)`` on an UNTAINTED drive with ``always_sandbox=False``. The
    docstring asserted this but no test constructed both — this converts the prose claim into a
    mutation-resistant pin (one passing, one failing case)."""
    from cogworx.cost.budget import BudgetGuard
    from cogworx.runtime.context import RunContext
    from cogworx.testing.doubles import (
        InMemoryGraphStore,
        InMemoryJournal,
        InMemoryLatentStore,
    )
    from cogworx.testing.fake_model import echo_model
    from cogworx.verification.oracles.code import CodeOracle

    journal = InMemoryJournal()
    await journal.start_run(
        "run-frozen-eq", "sess", pathway_id="p", pathway_version=1, pathway_fingerprint="fp"
    )
    ctx = RunContext(
        run_id="run-frozen-eq",
        session_id="sess",
        model=echo_model("unused"),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
    )
    # The solution is taken from thesis.proposed_solution; the FROZEN test overrides the thesis
    # experiment_design, so a wrong experiment_design proves the frozen test (not thesis) is used.
    thesis = Thesis(proposed_solution=solution, experiment_design="this is not a valid test")
    oracle = CodeOracle(test_source="frozen", frozen_test_code=test, always_sandbox=False)
    v_oracle = await oracle.evaluate(frame=_FRAME, thesis=thesis, ctx=ctx)

    v_helper = run_frozen_check(solution, test, timeout_s=30.0)

    # Equal on the fields that matter (reasoning carries a quarantine nonce + raw pytest output, so
    # it is intentionally NOT compared; behavior is pinned by the structural fields below).
    assert v_helper.holds is v_oracle.holds is expect_holds
    assert v_helper.valid_check is v_oracle.valid_check
    assert v_helper.is_executable is v_oracle.is_executable
    assert v_helper.source == v_oracle.source == "tool"
    assert v_helper.test_provenance == v_oracle.test_provenance == "frozen"


@pytest.mark.slow
def test_all_seeds_pass_frozen_self_check() -> None:
    """Full sweep (opt-in) — EVERY fixture seed passes its own frozen test before entering the set.
    Spawns one pytest subprocess PER SEED, so per the test-hang discipline it is doubly gated off
    the default loop: it carries ``@pytest.mark.slow`` AND skips unless ``COGWORX_RUN_SEED_SWEEP``
    is set (the marker alone is opt-OUT-only without ``-m 'not slow'`` in addopts, so the env gate
    is the belt). The 3-seed subset + negative control above pin the mechanism in the tight loop."""
    import os

    if not os.environ.get("COGWORX_RUN_SEED_SWEEP"):
        pytest.skip("seed self-check sweep is opt-in: set COGWORX_RUN_SEED_SWEEP=1 (hang risk)")
    failures: list[int] = []
    for seed in _load_seeds():
        v = run_frozen_check(seed.thesis.proposed_solution, seed.test_code, timeout_s=30.0)
        if not (v.holds and v.valid_check):
            failures.append(seed.seed_id)
    assert not failures, f"seeds failing their own frozen test: {failures}"


# ---------------------------------------------------------------------------
# Pin 7 — S1: planting.py performs no journal I/O / no live-write-path coupling
# ---------------------------------------------------------------------------


def _planting_tree() -> ast.Module:
    import cogworx.eval.planting as planting_mod

    return ast.parse(Path(planting_mod.__file__).read_text(encoding="utf-8"))


def _imported_modules(tree: ast.Module) -> set[str]:
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
        elif isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
    return imported


def _referenced_names(tree: ast.Module) -> set[str]:
    """Names and attributes referenced in CODE (Name ids + Attribute attrs), excluding docstrings —
    so an S1-clean module that merely *describes* its posture in prose is not flagged."""
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Attribute):
            names.add(node.attr)
    return names


def test_planting_module_does_no_journal_io() -> None:
    """S1 (the #1 red-team target): the authoring module never imports or references a Journal /
    RunContext / StageContext / load_run in CODE, so it cannot be on a live write-path or a replay
    path. Structural over the AST (S9: structure over self-report) — the docstring may DESCRIBE the
    S1 posture in prose, but no journal symbol is imported or called."""
    tree = _planting_tree()
    imported = _imported_modules(tree)
    referenced = _referenced_names(tree)
    for forbidden in ("Journal", "RunContext", "StageContext", "load_run", "GraphStore",
                      "LatentStore"):
        assert forbidden not in referenced, (
            f"planting.py CODE references {forbidden!r} (S1 write-path smell)"
        )
    assert not any("journal" in m or "substrate" in m for m in imported), (
        f"planting.py imports a journal/substrate module: {imported}"
    )


def test_planting_module_imports_no_substrate() -> None:
    """No substrate seam is imported (no graph_store / latent / journal / runtime context) — pure
    authoring tooling. Only seams touched: eval.corpus, model.base, verification.contracts."""
    imported = _imported_modules(_planting_tree())
    assert not any("substrate" in m for m in imported), f"substrate import: {imported}"
    assert "cogworx.runtime.context" not in imported
    # The model seam is allowed (the K injector talks to a Model, S4) — but only the protocol/types,
    # never a concrete provider (S4: provider is the call-site's choice).
    for concrete in (
        "cogworx.model.providers.claude",
        "cogworx.model.providers.openai_compat",
    ):
        assert concrete not in imported, f"planting.py imports concrete provider {concrete!r} (S4)"


def test_run_frozen_check_not_reachable_from_non_authoring_code() -> None:
    """canon's load-bearing requirement (S10, widened): the journal-free, F7-bypassing
    run_frozen_check must remain physically unreachable from any NON-authoring code. The live
    journal/taint path is ``runtime/engine.py`` + ``verification/`` (NOT just ``loop/``), and the
    helper is de-exported, so the only legitimate caller lives under ``src/cogworx/eval/``. Scan ALL
    of ``src/cogworx/**/*.py`` and assert no module OUTSIDE ``eval/`` mentions ``run_frozen_check``
    — this now covers ``runtime/engine.py``, ``verification/``, ``capability/``, ``adapters/``.
    Structural, not docstring-only (S9). The authoring module ``eval/_authoring.py`` (its def) and
    the rest of ``eval/`` are the sanctioned callers and are excluded."""
    src_root = Path(__file__).resolve().parents[2] / "src" / "cogworx"
    eval_dir = src_root / "eval"
    offenders: list[str] = []
    for py in src_root.rglob("*.py"):
        # The eval/ package (incl. _authoring.py's own def + planting.py's docstring ref) is the
        # legitimate authoring caller — exclude it; everything else must not reach the helper.
        if eval_dir in py.parents or py == eval_dir:
            continue
        if "run_frozen_check" in py.read_text(encoding="utf-8"):
            offenders.append(str(py))
    assert not offenders, (
        f"non-authoring modules reference run_frozen_check (F7 bypass leak): {offenders}"
    )


# ---------------------------------------------------------------------------
# Pin 8 — ANSWER_BEARING_FIELDS strip list
# ---------------------------------------------------------------------------


def test_answer_bearing_fields_is_exactly_experiment_design() -> None:
    assert ANSWER_BEARING_FIELDS == ("thesis.experiment_design",)


def test_answer_bearing_fields_excludes_test_code() -> None:
    """test_code is held on the CorpusItem but never in the arm-input projection (it names the
    planted error) — so it is NOT in the strippable answer-bearing list."""
    assert "test_code" not in ANSWER_BEARING_FIELDS
    assert not any("test_code" in f for f in ANSWER_BEARING_FIELDS)
