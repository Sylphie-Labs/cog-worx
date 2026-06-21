"""Deterministic unit tests for the Phase-4 GATE corpus schema (Pod 4.4c-2).

Schema pins only — the loader (refuse-to-load-unlocked + lock-time content_hash) is
test-qa-expert's 4.4c-3 work. Mirrors the youden.py pin discipline (``test_youden.py:96-106``): a
frozen
field-set tripwire per model (additive-only), discriminator routing pins, the S5 cross-field
validator pin, frozen-mutation pins, the ``test_provenance`` narrowing pin, and default pins — each
paired with a passing well-formed control so the assertions are mutation-resistant. No model calls,
no substrate.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogworx.eval.corpus import (
    Adjudication,
    CorpusItem,
    CorpusLoadError,
    DeterministicPlanterStamp,
    DifficultyMarker,
    HumanLabelProvenance,
    LLMPlanterStamp,
    OracleLabelProvenance,
    RegimeAdjudication,
    load_corpus,
)
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
_TS = datetime(2026, 6, 20, 12, 0, 0, tzinfo=UTC)


def _oracle_prov() -> OracleLabelProvenance:
    return OracleLabelProvenance(
        returncode=1,
        test_provenance="frozen",
        holds=False,
        valid_check=True,
        oracle_id="code-oracle",
    )


def _human_prov() -> HumanLabelProvenance:
    return HumanLabelProvenance(
        adjudications=(
            Adjudication(
                adjudicator_id="a1",
                verdict="error",
                rationale="off-by-one",
                timestamp=_TS,
            ),
        ),
    )


def _difficulty() -> DifficultyMarker:
    return DifficultyMarker(planted_difficulty="medium", surface_complexity=12)


def _item(**overrides: object) -> CorpusItem:
    base: dict[str, object] = {
        "item_id": 1,
        "frame": _FRAME,
        "thesis": _THESIS,
        "test_code": "assert f([1, 2]) == 3",
        "is_error": 1,
        "label_source": "oracle",
        "label_provenance": _oracle_prov(),
        "stratum": "O",
        "oracle_reachable": True,
        "difficulty": _difficulty(),
        "matched_sibling_id": None,
        "split": "measurement",
        "planter": DeterministicPlanterStamp(operators=("swap-op",)),
    }
    base.update(overrides)
    return CorpusItem(**base)


def test_wellformed_controls_construct() -> None:
    """Every builder yields a valid model — the controls the mutation pins are measured against."""
    assert _item().item_id == 1
    assert _item(label_source="human", label_provenance=_human_prov()).label_source == "human"


# ---------------------------------------------------------------------------
# 1. Frozen-field-set assertions (the additive-only tripwire)
# ---------------------------------------------------------------------------


def test_corpus_item_field_set_is_pinned() -> None:
    assert set(CorpusItem.model_fields) == {
        "item_id",
        "frame",
        "thesis",
        "test_code",
        "is_error",
        "label_source",
        "label_provenance",
        "stratum",
        "oracle_reachable",
        "error_regime",
        "difficulty",
        "matched_sibling_id",
        "split",
        "planter",
        "content_hash",
    }


def test_difficulty_marker_field_set_is_pinned() -> None:
    assert set(DifficultyMarker.model_fields) == {
        "planted_difficulty",
        "surface_complexity",
        "is_matched_sibling",
    }


def test_adjudication_field_set_is_pinned() -> None:
    assert set(Adjudication.model_fields) == {
        "adjudicator_id",
        "verdict",
        "rationale",
        "timestamp",
    }


def test_oracle_label_provenance_field_set_is_pinned() -> None:
    assert set(OracleLabelProvenance.model_fields) == {
        "kind",
        "returncode",
        "test_provenance",
        "holds",
        "valid_check",
        "oracle_id",
    }


def test_human_label_provenance_field_set_is_pinned() -> None:
    assert set(HumanLabelProvenance.model_fields) == {
        "kind",
        "adjudications",
        "tie_breaker_id",
    }


def test_deterministic_planter_field_set_is_pinned() -> None:
    assert set(DeterministicPlanterStamp.model_fields) == {"injector_kind", "operators"}


def test_llm_planter_field_set_is_pinned() -> None:
    assert set(LLMPlanterStamp.model_fields) == {"injector_kind", "model_family", "model_id"}


# ---------------------------------------------------------------------------
# 2. Discriminator routing pins
# ---------------------------------------------------------------------------


def test_label_provenance_discriminates_oracle() -> None:
    assert isinstance(_item().label_provenance, OracleLabelProvenance)


def test_label_provenance_discriminates_human() -> None:
    item = _item(label_source="human", label_provenance=_human_prov())
    assert isinstance(item.label_provenance, HumanLabelProvenance)


def test_label_provenance_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        _item(label_provenance={"kind": "judge", "oracle_id": "x"})


def test_label_provenance_rejects_missing_kind() -> None:
    with pytest.raises(ValidationError):
        _item(label_provenance={"returncode": 0})


def test_planter_discriminates_deterministic() -> None:
    assert isinstance(_item().planter, DeterministicPlanterStamp)


def test_planter_discriminates_llm() -> None:
    item = _item(planter=LLMPlanterStamp(model_family="claude", model_id="opus"))
    assert isinstance(item.planter, LLMPlanterStamp)


def test_planter_rejects_unknown_injector_kind() -> None:
    with pytest.raises(ValidationError):
        _item(planter={"injector_kind": "manual", "operators": ()})


def test_planter_rejects_missing_injector_kind() -> None:
    with pytest.raises(ValidationError):
        _item(planter={"operators": ("swap-op",)})


# ---------------------------------------------------------------------------
# 3. Cross-field validator pin (S5 guard)
# ---------------------------------------------------------------------------


def test_label_source_must_match_provenance_kind() -> None:
    with pytest.raises(ValidationError):
        _item(label_source="human", label_provenance=_oracle_prov())
    with pytest.raises(ValidationError):
        _item(label_source="oracle", label_provenance=_human_prov())


def test_label_source_matching_provenance_passes() -> None:
    assert _item(label_source="human", label_provenance=_human_prov()).label_source == "human"


# ---------------------------------------------------------------------------
# 4. Frozen pins (mutation raises)
# ---------------------------------------------------------------------------


def test_corpus_item_is_frozen() -> None:
    item = _item()
    with pytest.raises(ValidationError):
        item.item_id = 2  # type: ignore[misc]


def test_difficulty_marker_is_frozen() -> None:
    marker = _difficulty()
    with pytest.raises(ValidationError):
        marker.surface_complexity = 99  # type: ignore[misc]


def test_oracle_provenance_is_frozen() -> None:
    prov = _oracle_prov()
    with pytest.raises(ValidationError):
        prov.holds = True  # type: ignore[misc]


def test_adjudication_is_frozen() -> None:
    adj = _human_prov().adjudications[0]
    with pytest.raises(ValidationError):
        adj.verdict = "clean"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# RegimeAdjudication — K-stratum error-regime second-author audit (eval-stats §6.D).
# Separate verdict vocabulary; reassign-regime ⟺ reassigned_regime is not None.
# ---------------------------------------------------------------------------


def test_regime_adjudication_field_set_is_pinned() -> None:
    assert set(RegimeAdjudication.model_fields) == {
        "adjudicator_id",
        "verdict",
        "reassigned_regime",
        "rationale",
        "timestamp",
    }


def test_regime_adjudication_confirm_regime_valid() -> None:
    adj = RegimeAdjudication(
        adjudicator_id="r1",
        verdict="confirm-regime",
        rationale="planter tag stands",
        timestamp=_TS,
    )
    assert adj.verdict == "confirm-regime"
    assert adj.reassigned_regime is None


def test_regime_adjudication_reassign_regime_valid() -> None:
    adj = RegimeAdjudication(
        adjudicator_id="r1",
        verdict="reassign-regime",
        reassigned_regime="wrong-operator",
        rationale="mistagged by planter",
        timestamp=_TS,
    )
    assert adj.verdict == "reassign-regime"
    assert adj.reassigned_regime == "wrong-operator"


def test_regime_adjudication_abstain_valid() -> None:
    adj = RegimeAdjudication(
        adjudicator_id="r1",
        verdict="abstain",
        rationale="cannot establish a regime",
        timestamp=_TS,
    )
    assert adj.verdict == "abstain"
    assert adj.reassigned_regime is None


def test_regime_adjudication_reassign_without_regime_raises() -> None:
    with pytest.raises(ValidationError):
        RegimeAdjudication(
            adjudicator_id="r1",
            verdict="reassign-regime",
            reassigned_regime=None,
            rationale="x",
            timestamp=_TS,
        )


def test_regime_adjudication_confirm_with_regime_raises() -> None:
    with pytest.raises(ValidationError):
        RegimeAdjudication(
            adjudicator_id="r1",
            verdict="confirm-regime",
            reassigned_regime="wrong-operator",
            rationale="x",
            timestamp=_TS,
        )


def test_regime_adjudication_abstain_with_regime_raises() -> None:
    with pytest.raises(ValidationError):
        RegimeAdjudication(
            adjudicator_id="r1",
            verdict="abstain",
            reassigned_regime="wrong-operator",
            rationale="x",
            timestamp=_TS,
        )


def test_regime_adjudication_rejects_unknown_verdict() -> None:
    with pytest.raises(ValidationError):
        RegimeAdjudication(
            adjudicator_id="r1",
            verdict="error",  # type: ignore[arg-type]
            rationale="x",
            timestamp=_TS,
        )


def test_regime_adjudication_is_frozen() -> None:
    adj = RegimeAdjudication(
        adjudicator_id="r1",
        verdict="confirm-regime",
        rationale="planter tag stands",
        timestamp=_TS,
    )
    with pytest.raises(ValidationError):
        adj.verdict = "abstain"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 5. test_provenance narrowing pin (§13.4 #5 self-test-laundering guard)
# ---------------------------------------------------------------------------


def test_oracle_provenance_accepts_frozen() -> None:
    assert _oracle_prov().test_provenance == "frozen"


def test_oracle_provenance_rejects_thesis_provenance() -> None:
    with pytest.raises(ValidationError):
        OracleLabelProvenance(
            returncode=0,
            test_provenance="thesis",
            holds=True,
            valid_check=True,
            oracle_id="x",
        )


def test_oracle_provenance_rejects_na_provenance() -> None:
    with pytest.raises(ValidationError):
        OracleLabelProvenance(
            returncode=0,
            test_provenance="n/a",
            holds=True,
            valid_check=True,
            oracle_id="x",
        )


# ---------------------------------------------------------------------------
# 6. Default pins
# ---------------------------------------------------------------------------


def test_error_regime_defaults_empty() -> None:
    assert _item().error_regime == ""


def test_error_regime_accepts_a_tag() -> None:
    assert _item(error_regime="off-by-one").error_regime == "off-by-one"


def test_content_hash_defaults_empty_unlocked() -> None:
    assert _item().content_hash == ""


def test_content_hash_accepts_a_value() -> None:
    assert _item(content_hash="abc123").content_hash == "abc123"


def test_difficulty_is_matched_sibling_defaults_false() -> None:
    assert _difficulty().is_matched_sibling is False


def test_human_tie_breaker_defaults_none() -> None:
    assert _human_prov().tie_breaker_id is None


# ===========================================================================
# load_corpus — structural-only loader (Pod 4.4c-2). Deterministic, in-memory,
# zero model/oracle/journal. Every guard pin is paired with a passing negative
# control so the assertions are mutation-resistant (redteam-before-validated).
# ===========================================================================

# ---------------------------------------------------------------------------
# Loader fixtures: error item / clean item / matched pair factories.
# Loader-scoped item ids start at 100+ to avoid colliding with schema-pin ids.
# ---------------------------------------------------------------------------


def _error_item(item_id: int, *, split: str = "tuning", **overrides: object) -> CorpusItem:
    """A well-formed error-stratum item (oracle-labelled). Default split=tuning so it loads."""
    base: dict[str, object] = {
        "item_id": item_id,
        "is_error": 1,
        "label_source": "oracle",
        "label_provenance": _oracle_prov(),
        "stratum": "O",
        "split": split,
    }
    base.update(overrides)
    return _item(**base)


def _clean_item(item_id: int, *, split: str = "tuning", **overrides: object) -> CorpusItem:
    """A well-formed clean item carrying a valid oracle (C1) stamp by default."""
    base: dict[str, object] = {
        "item_id": item_id,
        "is_error": 0,
        "label_source": "oracle",
        "label_provenance": _oracle_prov(),
        "stratum": "clean",
        "split": split,
    }
    base.update(overrides)
    return _item(**base)


def _clean_human_adj(n: int) -> HumanLabelProvenance:
    """A human (C2) stamp carrying exactly ``n`` adjudications (n may be 0 for the G1 mutant)."""
    return HumanLabelProvenance(
        adjudications=tuple(
            Adjudication(
                adjudicator_id=f"a{i}",
                verdict="clean",
                rationale="independently re-verified",
                timestamp=_TS,
            )
            for i in range(n)
        )
    )


def _matched_pair(
    err_id: int, clean_id: int, *, split: str = "tuning"
) -> tuple[CorpusItem, CorpusItem]:
    """A symmetric, opposite-label mutate-then-revert pair: one error-stratum + one clean."""
    err = _error_item(
        err_id,
        split=split,
        matched_sibling_id=clean_id,
        difficulty=DifficultyMarker(
            planted_difficulty="medium", surface_complexity=12, is_matched_sibling=True
        ),
    )
    clean = _clean_item(
        clean_id,
        split=split,
        matched_sibling_id=err_id,
        difficulty=DifficultyMarker(
            planted_difficulty="medium", surface_complexity=12, is_matched_sibling=True
        ),
    )
    return err, clean


# ---------------------------------------------------------------------------
# Master negative control — a full well-formed mixed corpus loads clean.
# ---------------------------------------------------------------------------


def test_load_wellformed_mixed_corpus_loads_clean() -> None:
    err, clean = _matched_pair(100, 101)
    corpus = [
        err,
        clean,
        _error_item(102),
        _clean_item(103, label_source="human", label_provenance=_clean_human_adj(2)),
        _error_item(200, split="measurement"),  # filtered out by default
    ]
    loaded = load_corpus(corpus)
    assert {i.item_id for i in loaded} == {100, 101, 102, 103}


# ---------------------------------------------------------------------------
# G1 — clean item must carry a well-shaped label-provenance stamp (shape only).
# ---------------------------------------------------------------------------


def test_g1_clean_with_oracle_stamp_loads() -> None:
    loaded = load_corpus([_clean_item(100)])
    assert loaded[0].item_id == 100


def test_g1_clean_with_human_stamp_loads() -> None:
    item = _clean_item(100, label_source="human", label_provenance=_clean_human_adj(1))
    assert load_corpus([item])[0].item_id == 100


def test_g1_clean_human_stamp_zero_adjudications_raises() -> None:
    item = _clean_item(100, label_source="human", label_provenance=_clean_human_adj(0))
    with pytest.raises(CorpusLoadError, match=r"G1.*100"):
        load_corpus([item])


def test_g1_applies_to_is_error_zero_even_if_stratum_not_clean() -> None:
    # is_error==0 marks a clean item too (§12 redundancy); G1 still demands a stamp shape.
    # An oracle stamp is well-shaped, so this loads — the control for the redundancy path.
    item = _clean_item(100, stratum="O", is_error=0)
    assert load_corpus([item])[0].is_error == 0


# ---------------------------------------------------------------------------
# G2 — non-judge label source matching the provenance kind (defense-in-depth).
# The frozen model blocks a judge source AND the mismatch at construction, so
# the loader guard is unreachable via a constructed item: we pin the model
# block and note the loader carries the same assertion belt-and-suspenders.
# ---------------------------------------------------------------------------


def test_g2_model_blocks_judge_source_at_construction() -> None:
    # label_source is Literal["oracle","human"]; a judge source cannot be constructed.
    with pytest.raises(ValidationError):
        _item(label_source="judge")


def test_g2_model_blocks_source_provenance_mismatch_at_construction() -> None:
    # The loader's G2 match check is defense-in-depth behind this model validator.
    with pytest.raises(ValidationError):
        _item(label_source="human", label_provenance=_oracle_prov())


def test_g2_wellformed_oracle_and_human_sources_load() -> None:
    corpus = [
        _error_item(100),
        _clean_item(101, label_source="human", label_provenance=_clean_human_adj(1)),
    ]
    assert {i.item_id for i in load_corpus(corpus)} == {100, 101}


# ---------------------------------------------------------------------------
# G3 — split firewall (highest-value guard).
# ---------------------------------------------------------------------------


def test_g3_default_returns_tuning_only() -> None:
    corpus = [
        _error_item(100, split="tuning"),
        _clean_item(101, split="tuning"),
        _error_item(200, split="measurement"),
        _clean_item(201, split="measurement"),
    ]
    loaded = load_corpus(corpus)
    assert {i.item_id for i in loaded} == {100, 101}
    assert all(i.split == "tuning" for i in loaded)


def test_g3_measurement_run_returns_measurement_only() -> None:
    corpus = [
        _error_item(100, split="tuning"),
        _clean_item(101, split="tuning"),
        _error_item(200, split="measurement", content_hash="locked-a"),
        _clean_item(201, split="measurement", content_hash="locked-b"),
    ]
    loaded = load_corpus(corpus, measurement_run=True)
    assert {i.item_id for i in loaded} == {200, 201}
    assert all(i.split == "measurement" for i in loaded)


def test_g3_counts_are_exact_both_directions() -> None:
    corpus = [_error_item(100 + i, split="tuning") for i in range(3)] + [
        _error_item(200 + i, split="measurement", content_hash="h") for i in range(5)
    ]
    assert len(load_corpus(corpus)) == 3
    assert len(load_corpus(corpus, measurement_run=True)) == 5


# ---------------------------------------------------------------------------
# G4 — frozen-artifact well-formedness (STATIC reference integrity only).
# ---------------------------------------------------------------------------


def test_g4_duplicate_item_id_raises() -> None:
    with pytest.raises(CorpusLoadError, match=r"G4.*duplicate item_id 100"):
        load_corpus([_error_item(100), _clean_item(100)])


def test_g4_unique_ids_load() -> None:
    assert len(load_corpus([_error_item(100), _clean_item(101)])) == 2


def test_g4_dangling_sibling_ref_raises() -> None:
    # A names B, B absent from loaded set.
    item = _error_item(100, matched_sibling_id=999)
    with pytest.raises(CorpusLoadError, match=r"G4.*100.*999.*absent"):
        load_corpus([item])


def test_g4_asymmetric_sibling_raises() -> None:
    # A->B, B->C (B does not name A back).
    a = _error_item(100, matched_sibling_id=101)
    b = _clean_item(101, matched_sibling_id=102)
    c = _clean_item(102, matched_sibling_id=101)
    with pytest.raises(CorpusLoadError, match="G4: asymmetric"):
        load_corpus([a, b, c])


def test_g4_same_label_pair_both_clean_raises() -> None:
    a = _clean_item(100, matched_sibling_id=101)
    b = _clean_item(101, matched_sibling_id=100)
    with pytest.raises(CorpusLoadError, match="G4: same-label"):
        load_corpus([a, b])


def test_g4_same_label_pair_both_error_raises() -> None:
    a = _error_item(100, matched_sibling_id=101)
    b = _error_item(101, matched_sibling_id=100)
    with pytest.raises(CorpusLoadError, match="G4: same-label"):
        load_corpus([a, b])


def test_g4_wellformed_opposite_label_symmetric_pair_loads() -> None:
    err, clean = _matched_pair(100, 101)
    loaded = load_corpus([err, clean])
    assert {i.item_id for i in loaded} == {100, 101}


def test_g4_sibling_filtered_by_split_is_dangling() -> None:
    # G3 filters first: an error item in tuning naming a clean sibling that is
    # measurement-only becomes a dangling ref in the default load. Static integrity
    # is over the LOADED subset, exactly as the docstring states.
    err = _error_item(100, split="tuning", matched_sibling_id=201)
    clean = _clean_item(201, split="measurement", matched_sibling_id=100)
    with pytest.raises(CorpusLoadError, match=r"G4.*absent"):
        load_corpus([err, clean])


def test_g4_code_item_missing_test_code_raises() -> None:
    item = _error_item(100, test_code=None)  # _FRAME.problem_type == "code"
    with pytest.raises(CorpusLoadError, match=r"G4.*100.*missing test_code"):
        load_corpus([item])


def test_g4_noncode_item_with_test_code_raises() -> None:
    proof_frame = OracleFrame(
        completion_criterion="proof_valid",
        problem_type="proof",
        problem_statement="show sqrt(2) irrational",
    )
    item = _error_item(100, frame=proof_frame, test_code="assert True")
    with pytest.raises(CorpusLoadError, match=r"G4.*100.*carries test_code"):
        load_corpus([item])


def test_g4_noncode_item_without_test_code_loads() -> None:
    proof_frame = OracleFrame(
        completion_criterion="proof_valid",
        problem_type="proof",
        problem_statement="show sqrt(2) irrational",
    )
    item = _error_item(100, frame=proof_frame, test_code=None)
    assert load_corpus([item])[0].item_id == 100


# ---------------------------------------------------------------------------
# Optional content_hash never-locked tripwire (measurement_run only).
# ---------------------------------------------------------------------------


def test_content_hash_tripwire_raises_on_unlocked_measurement_item() -> None:
    item = _error_item(200, split="measurement", content_hash="")
    with pytest.raises(CorpusLoadError, match=r"G4.*200.*content_hash"):
        load_corpus([item], measurement_run=True)


def test_content_hash_tripwire_silent_on_default_tuning_load() -> None:
    # Tuning items are never lock-checked — empty content_hash is fine off the measurement path.
    item = _error_item(100, split="tuning", content_hash="")
    assert load_corpus([item])[0].content_hash == ""
