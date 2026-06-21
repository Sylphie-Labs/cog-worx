"""Deterministic unit tests for the 5-arm GATE runner skeleton (Pod 4.4d-0).

Pure, offline, deterministic — NO model, NO docker, NO journal, NO substrate. Every assertion is
mutation-resistant (red-team will attack):

  - CRN seeding is deterministic + ARM-INDEPENDENT (same (item,trial) -> same seed across arms;
    different (item,trial) -> different seed) and reproducible across runs.
  - PER-ITEM WELDING (load-bearing — the §5 shuffle depends on it): every cell of one item_id
    carries IDENTICAL stratum/regime/converted_o across every arm/trial. The mutation control: an
    emitter that read stratum off the arm would fail (the arm cannot supply one — structural).
  - the arm NEVER sets ground truth: ArmOutcome has only flagged/route (structural, not a runtime
    check) — pinned by the model schema + an AST/symbol pin that no journal/StageContext leaks in.
  - arm-A floor: a corpus whose arm-A cells flag NOTHING on K and barely FP on clean satisfies
    ``assert_arm_a_floor`` (INV-A0 = sens_A==0 on K — see the module docstring; arm A is the ORACLE,
    so on a correctly-stratified oracle-BLIND K item it flags nothing).
  - round-trip: emitted Cells (stub C/D arms) run through ``nested_bootstrap_delta`` without error.
  - R default 7, R param honored (R=3 fast fixture).
  - the real ``arm_a_executor`` O-catch polarity, pinned on tiny inline pass/fail tests.

The arm-A O-catch polarity (the resolved INV-A0 question): arm A = the deterministic oracle; it
flags iff ``is_executable ∧ valid_check ∧ ¬holds`` (an O-catch). K is DEFINED as the complement of
the O-catch, so a correctly-stratified K item makes A flag NOTHING -> sens_A==0 -> INV-A0 holds. No
contradiction: INV-A0 is a stratification-consistency check, not a claim that arm A is inert.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from cogworx.eval.corpus import (
    ConvertedPlanterStamp,
    CorpusItem,
    DifficultyMarker,
    LLMPlanterStamp,
    OracleLabelProvenance,
)
from cogworx.eval.lock import MASTER_SEED, assert_arm_a_floor
from cogworx.eval.runner import (
    ArmInput,
    ArmOutcome,
    arm_a_executor,
    crn_seed,
    run_arms,
    scripted_executor,
)
from cogworx.eval.youden import is_converted_o, nested_bootstrap_delta
from cogworx.verification.contracts import OracleFrame, Thesis

# ---------------------------------------------------------------------------
# Builders (the mutation-resistance controls)
# ---------------------------------------------------------------------------

_TS_PROV = OracleLabelProvenance(
    returncode=1, test_provenance="frozen", holds=False, valid_check=True, oracle_id="x"
)
_CLEAN_PROV = OracleLabelProvenance(
    returncode=0, test_provenance="frozen", holds=True, valid_check=True, oracle_id="x"
)


def _frame() -> OracleFrame:
    return OracleFrame(
        completion_criterion="tests_pass", problem_type="code", problem_statement="sum a list"
    )


def _difficulty() -> DifficultyMarker:
    return DifficultyMarker(planted_difficulty="medium", surface_complexity=12)


def _item(
    item_id: int,
    stratum: str,
    *,
    is_error: int,
    error_regime: str = "",
    converted: bool = False,
    test_code: str | None = "assert f([1, 2]) == 3",
    solution: str = "return sum(xs)",
) -> CorpusItem:
    """One frozen corpus item with the given ground truth. ``converted`` swaps the planter to a
    :class:`ConvertedPlanterStamp` so the runner reads ``converted_o=True`` off it."""
    if converted:
        planter = ConvertedPlanterStamp(
            planter_model_family="planterfam",
            planter_model_id="p1",
            adversary_family="advfam",
            winning_round=2,
        )
    else:
        planter = LLMPlanterStamp(model_family="planterfam", model_id="p1")
    label_source = "oracle"
    provenance = _CLEAN_PROV if is_error == 0 else _TS_PROV
    return CorpusItem(
        item_id=item_id,
        frame=_frame(),
        thesis=Thesis(proposed_solution=solution, experiment_design="run frozen tests"),
        test_code=test_code,
        is_error=is_error,
        label_source=label_source,
        label_provenance=provenance,
        stratum=stratum,  # type: ignore[arg-type]
        oracle_reachable=(stratum != "K"),
        error_regime=error_regime,
        difficulty=_difficulty(),
        matched_sibling_id=None,
        split="measurement",
        planter=planter,
    )


def _flag_all(arm_input: ArmInput, seed: int) -> ArmOutcome:
    """A stub arm that flags everything (every item, every trial), ignoring the seed."""
    _ = seed
    return ArmOutcome(flagged=1, route="flag")


def _flag_none(arm_input: ArmInput, seed: int) -> ArmOutcome:
    """A stub arm that flags nothing."""
    _ = seed
    return ArmOutcome(flagged=0, route="pass")


# ===========================================================================
# CRN seeding — deterministic, arm-independent, reproducible
# ===========================================================================


def test_crn_seed_deterministic() -> None:
    """Same (master_seed, item, trial) -> byte-identical seed (reproducible across calls/runs)."""
    assert crn_seed(MASTER_SEED, 7, 3) == crn_seed(MASTER_SEED, 7, 3)


def test_crn_seed_is_arm_independent_by_construction() -> None:
    """The CRN seed is a function of (master_seed, item, trial) ONLY — arm never enters the sig
    (the pairing contract nested_bootstrap_delta cancels on). Two arms in the SAME run get the same
    seed for the same (item, trial)."""
    cells = run_arms(
        [_item(1, "K", is_error=1), _item(2, "clean", is_error=0)],
        arm_executors={"C": _flag_all, "D": _flag_none},
        R=4,
    )
    by_key: dict[tuple[int, int], set[int]] = {}
    for c in cells:
        by_key.setdefault((c.item_id, c.trial), set()).add(c.seed)
    # Every (item, trial) maps to exactly ONE seed across both arms.
    assert all(len(seeds) == 1 for seeds in by_key.values())


def test_crn_seed_varies_by_item_and_trial() -> None:
    """Different (item, trial) -> different seed (a constant seed would defeat trial resampling)."""
    seeds = {
        crn_seed(MASTER_SEED, item, trial) for item in (1, 2, 3) for trial in range(7)
    }
    assert len(seeds) == 21  # all distinct


def test_crn_seed_nonnegative_int() -> None:
    """The seed is a plain non-negative int (a valid RNG seed; recorded on Cell.seed: int)."""
    s = crn_seed(MASTER_SEED, 99, 6)
    assert isinstance(s, int)
    assert s >= 0


def test_crn_seed_master_seed_separates() -> None:
    """A different master_seed yields a different (item, trial) seed -> a different artifact."""
    assert crn_seed(MASTER_SEED, 5, 2) != crn_seed(MASTER_SEED + 1, 5, 2)


# ===========================================================================
# Per-item welding (LOAD-BEARING) — ground truth from the corpus, never the arm
# ===========================================================================


def test_per_item_welding_stratum_regime_converted_o() -> None:
    """ALL cells of one item carry IDENTICAL stratum/regime/converted_o across every arm/trial —
    the §5-shuffle precondition. Ground truth comes from the corpus item, never the arm."""
    corpus = [
        _item(1, "K", is_error=1, error_regime="off-by-one"),
        _item(2, "O", is_error=1, error_regime="wrong-op", converted=True),
        _item(3, "clean", is_error=0),
    ]
    cells = run_arms(
        corpus,
        arm_executors={"A": _flag_all, "C": _flag_none, "D": _flag_all},
        R=5,
    )
    truth = {
        1: ("K", "off-by-one", False),
        2: ("O", "wrong-op", True),
        3: ("clean", "", False),
    }
    for item_id, (stratum, regime, conv) in truth.items():
        item_cells = [c for c in cells if c.item_id == item_id]
        # 3 arms x 5 trials = 15 cells per item, ALL identical on the welded fields.
        assert len(item_cells) == 15
        assert {(c.stratum, c.regime, c.converted_o) for c in item_cells} == {
            (stratum, regime, conv)
        }


def test_welding_independent_of_arm_flag_behavior() -> None:
    """MUTATION CONTROL: stratum/regime/converted_o are welded from the corpus REGARDLESS of what
    the arm returns. An arm that flags everything and an arm that flags nothing emit identical
    ground-truth fields for the same item — so an emitter reading stratum off the arm's behavior
    would diverge here and FAIL."""
    corpus = [_item(1, "O", is_error=1, error_regime="r", converted=True)]
    cells = run_arms(corpus, arm_executors={"flagger": _flag_all, "passer": _flag_none}, R=3)
    flagger = [c for c in cells if c.arm == "flagger"]
    passer = [c for c in cells if c.arm == "passer"]
    # Ground truth identical across the two arms; only `flagged`/`route` differ.
    assert {(c.stratum, c.regime, c.converted_o) for c in flagger} == {("O", "r", True)}
    assert {(c.stratum, c.regime, c.converted_o) for c in passer} == {("O", "r", True)}
    assert {c.flagged for c in flagger} == {1}
    assert {c.flagged for c in passer} == {0}


def test_converted_o_marker_round_trips_to_youden_predicate() -> None:
    """The welded ``converted_o`` flag is what ``youden.is_converted_o`` reads to keep converted-O
    of the binding delta — pin the seam end-to-end."""
    corpus = [
        _item(1, "O", is_error=1, converted=True),
        _item(2, "O", is_error=1, converted=False),
    ]
    cells = run_arms(corpus, arm_executors={"A": _flag_all}, R=2)
    assert all(is_converted_o(c) for c in cells if c.item_id == 1)
    assert not any(is_converted_o(c) for c in cells if c.item_id == 2)


# ===========================================================================
# The arm cannot set ground truth — STRUCTURAL
# ===========================================================================


def test_arm_outcome_has_no_ground_truth_fields() -> None:
    """ArmOutcome carries ONLY flagged/route — an arm is STRUCTURALLY unable to return a
    stratum/regime/converted_o (the S9 wall). A stub arm 'trying' to set ground truth cannot even
    name the field."""
    assert set(ArmOutcome.model_fields) == {"flagged", "route"}


def test_arm_input_carries_no_ground_truth() -> None:
    """ArmInput hands the arm the PROBLEM only — never the item's true label (an arm that could read
    its own ground truth would be grading itself, S9)."""
    fields = set(ArmInput.model_fields)
    assert "stratum" not in fields
    assert "is_error" not in fields
    assert "error_regime" not in fields
    assert "label_source" not in fields


# ===========================================================================
# Arm-A floor — INV-A0 (sens_A==0 on K) / INV-A1 (spec_A ~ 1 on clean)
# ===========================================================================

_N_CLEAN_PLANNED = 80
_R = 7


def _oracle_floor_arm(arm_input: ArmInput, seed: int) -> ArmOutcome:
    """A STUB standing in for arm A's oracle-floor BEHAVIOR (so the floor test needs no real pytest
    run): flag iff the item is an O-stratum error. We encode the O-catch via the solution string
    marker ``__O_CATCH__`` (the test corpus sets it on O items) — the runner never sees the stratum,
    so this stub mimics what the real ``run_frozen_check`` O-catch yields without executing code.
    On K (oracle-blind) and clean it flags nothing — exactly the arm-A floor."""
    _ = seed
    flagged = 1 if "__O_CATCH__" in arm_input.proposed_solution else 0
    return ArmOutcome(flagged=flagged, route="flag" if flagged else "pass")


def _arm_a_floor_corpus(
    *, k_flag: bool = False, clean_fp_item: int | None = None
) -> list[CorpusItem]:
    """A corpus where the oracle-floor stub flags O items, passes K + clean.

    ``k_flag=True`` plants the O-catch marker on a K item (INV-A0 violation — A flags an
    oracle-blind item, i.e. it was mis-stratified). ``clean_fp_item`` plants the marker on a clean
    item (INV-A1 false positive)."""
    corpus: list[CorpusItem] = []
    # 40 K items (oracle-blind: arm A flags nothing) + a token to flag if k_flag.
    for i in range(40):
        sol = "return sum(xs) __O_CATCH__" if (k_flag and i == 0) else "return sum(xs)"
        corpus.append(_item(100 + i, "K", is_error=1, solution=sol))
    # n_clean clean items (arm A barely false-positives).
    for i in range(_N_CLEAN_PLANNED):
        sol = "return sum(xs) __O_CATCH__" if (clean_fp_item == i) else "return sum(xs)"
        corpus.append(_item(200 + i, "clean", is_error=0, solution=sol))
    return corpus


def test_arm_a_cells_satisfy_floor() -> None:
    """The emitted arm-A cells (oracle-floor behavior) satisfy ``assert_arm_a_floor``: A flags
    NOTHING on K (INV-A0 sens_A==0) and zero clean FPs (INV-A1 spec_A==1). A token 'D' arm is
    emitted so the artifact is realistic (the floor reads arm 'A' only)."""
    corpus = _arm_a_floor_corpus()
    cells = run_arms(
        corpus, arm_executors={"A": _oracle_floor_arm, "D": _flag_all}, R=_R
    )
    # Direct INV-A0 check on the emitted cells: arm A flags nothing on K.
    k_a = [c for c in cells if c.arm == "A" and c.stratum == "K"]
    assert sum(c.flagged for c in k_a) == 0
    # The lock's floor assertion passes (returns None) on the emitted artifact.
    assert assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R) is None


def test_arm_a_floor_refuses_when_a_flags_k() -> None:
    """NEGATIVE CONTROL: if arm A flags a K item (mis-stratified — the oracle reached it), the
    emitted cells VIOLATE INV-A0 and ``assert_arm_a_floor`` refuses. Proves the floor assertion has
    teeth on the runner's real output, not just hand-built cells."""
    corpus = _arm_a_floor_corpus(k_flag=True)
    cells = run_arms(corpus, arm_executors={"A": _oracle_floor_arm, "D": _flag_all}, R=_R)
    with pytest.raises(Exception, match="INV-A0"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


# ===========================================================================
# Real arm_a_executor O-catch polarity (tiny inline tests; fast, no docker)
# ===========================================================================


def test_arm_a_executor_flags_o_catch() -> None:
    """Arm A flags a solution whose FROZEN test FAILS (an O-catch) — the oracle caught the error.
    Tiny inline solution+test so the frozen kernel runs in well under a second."""
    ai = ArmInput(
        item_id=1,
        problem_statement="p",
        completion_criterion="tests_pass",
        problem_type="code",
        proposed_solution="def f(x):\n    return x + 1\n",  # wrong: should be x*2
        test_code="def test_it():\n    from solution import f\n    assert f(3) == 6\n",
    )
    outcome = arm_a_executor(ai, seed=0)
    assert outcome.flagged == 1
    assert outcome.route == "flag"


def test_arm_a_executor_passes_clean() -> None:
    """Arm A flags NOTHING when the frozen test PASSES (no O-catch) — the clean / oracle-blind
    floor."""
    ai = ArmInput(
        item_id=2,
        problem_statement="p",
        completion_criterion="tests_pass",
        problem_type="code",
        proposed_solution="def f(x):\n    return x * 2\n",
        test_code="def test_it():\n    from solution import f\n    assert f(3) == 6\n",
    )
    outcome = arm_a_executor(ai, seed=0)
    assert outcome.flagged == 0
    assert outcome.route == "pass"


def test_arm_a_executor_deterministic_across_seeds() -> None:
    """Arm A is deterministic — the CRN seed is a no-op (no trial variance). Two different seeds
    yield the identical outcome."""
    ai = ArmInput(
        item_id=3,
        problem_statement="p",
        completion_criterion="tests_pass",
        problem_type="code",
        proposed_solution="def f(x):\n    return x * 2\n",
        test_code="def test_it():\n    from solution import f\n    assert f(3) == 6\n",
    )
    assert arm_a_executor(ai, seed=1) == arm_a_executor(ai, seed=999)


def test_arm_a_executor_none_test_passes() -> None:
    """A non-code / pure-K item (``test_code is None``) gives a non-executable verdict -> not an
    O-catch -> flagged 0 (the oracle is blind to it, exactly as K demands)."""
    ai = ArmInput(
        item_id=4,
        problem_statement="p",
        completion_criterion="judged",
        problem_type="prose",
        proposed_solution="some prose answer",
        test_code=None,
    )
    outcome = arm_a_executor(ai, seed=0)
    assert outcome.flagged == 0


# ===========================================================================
# Positive control — the scripted executor carries signal end-to-end
# ===========================================================================


def test_scripted_executor_flags_named_items_only() -> None:
    """The scripted positive control flags EXACTLY the named item_ids (every trial), passes the rest
    — the runner -> scorer plumbing carries signal (plan §13.4 #7)."""
    corpus = [
        _item(1, "K", is_error=1),
        _item(2, "clean", is_error=0),
        _item(3, "K", is_error=1),
    ]
    exec_ = scripted_executor(frozenset({1, 3}))
    cells = run_arms(corpus, arm_executors={"pos": exec_}, R=3)
    flagged_ids = {c.item_id for c in cells if c.flagged == 1}
    passed_ids = {c.item_id for c in cells if c.flagged == 0}
    assert flagged_ids == {1, 3}
    assert passed_ids == {2}


def test_scripted_executor_is_deterministic() -> None:
    """The scripted executor ignores the seed (deterministic): same item -> same outcome."""
    exec_ = scripted_executor(frozenset({5}))
    ai = ArmInput(
        item_id=5,
        problem_statement="p",
        completion_criterion="c",
        problem_type="code",
        proposed_solution="s",
        test_code=None,
    )
    assert exec_(ai, seed=1) == exec_(ai, seed=2) == ArmOutcome(flagged=1, route="flag")


# ===========================================================================
# Round-trip — emitted Cells feed nested_bootstrap_delta without error
# ===========================================================================


def test_round_trip_through_nested_bootstrap_delta() -> None:
    """Emitted Cells (stub C/D arms over K + clean) run through ``nested_bootstrap_delta`` cleanly —
    the emission schema IS the bootstrap input schema. D flags errors more than C, so delta > 0."""
    # D flags K errors; C flags less. Both leave clean alone (spec ~ 1).
    def _arm_d(arm_input: ArmInput, seed: int) -> ArmOutcome:
        _ = seed
        # flag K items (error stratum proxy via solution marker), pass clean.
        flagged = 1 if "__ERR__" in arm_input.proposed_solution else 0
        return ArmOutcome(flagged=flagged, route="flag" if flagged else "pass")

    def _arm_c(arm_input: ArmInput, seed: int) -> ArmOutcome:
        _ = seed
        return ArmOutcome(flagged=0, route="pass")

    corpus: list[CorpusItem] = []
    for i in range(8):
        corpus.append(_item(i, "K", is_error=1, solution="return sum(xs) __ERR__"))
    for i in range(8):
        corpus.append(_item(100 + i, "clean", is_error=0, solution="return sum(xs)"))
    cells = run_arms(corpus, arm_executors={"C": _arm_c, "D": _arm_d}, R=_R)

    mean_delta, lo, hi = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=50, seed=123
    )
    # D catches errors C misses with identical (perfect) spec -> J_D - J_C > 0.
    assert mean_delta > 0.0
    assert lo <= mean_delta <= hi


# ===========================================================================
# R default + R as a param
# ===========================================================================


def test_R_default_is_7() -> None:
    """``run_arms`` defaults to R=7 trials per (item, arm)."""
    cells = run_arms([_item(1, "K", is_error=1)], arm_executors={"A": _flag_none})
    trials = {c.trial for c in cells}
    assert trials == set(range(7))


def test_R_param_honored() -> None:
    """A smaller R runs fewer trials (the fast-fixture path)."""
    cells = run_arms(
        [_item(1, "K", is_error=1)], arm_executors={"A": _flag_none, "D": _flag_all}, R=3
    )
    # 1 item x 2 arms x 3 trials = 6 cells.
    assert len(cells) == 6
    assert {c.trial for c in cells} == {0, 1, 2}


def test_emission_cardinality() -> None:
    """The artifact is exactly item x arm x trial cells (no dropped/duplicated cells)."""
    corpus = [_item(i, "K", is_error=1) for i in range(4)]
    cells = run_arms(corpus, arm_executors={"A": _flag_none, "C": _flag_all, "D": _flag_all}, R=5)
    assert len(cells) == 4 * 3 * 5
    # Every (item, arm, trial) is unique.
    keys = {(c.item_id, c.arm, c.trial) for c in cells}
    assert len(keys) == len(cells)


def test_run_is_reproducible() -> None:
    """Re-running with the same corpus + executors + master_seed reproduces a byte-identical
    artifact (CRN determinism)."""
    corpus = [_item(1, "K", is_error=1), _item(2, "clean", is_error=0)]
    a = run_arms(corpus, arm_executors={"A": _flag_none, "D": _flag_all}, R=4)
    b = run_arms(corpus, arm_executors={"A": _flag_none, "D": _flag_all}, R=4)
    assert a == b


# ===========================================================================
# S1 / S9 posture — pinned structurally over the module AST
# ===========================================================================


def _runner_source() -> str:
    import cogworx.eval.runner as mod

    return Path(mod.__file__).read_text(encoding="utf-8")


def test_runner_imports_no_live_path_symbols() -> None:
    """S1: the runner reaches the journal-free ``run_frozen_check`` ONLY — never
    ``CodeOracle``/``StageContext``/``Journal``, and no concrete model provider import."""
    src = _runner_source()
    tree = ast.parse(src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    forbidden = {"CodeOracle", "StageContext", "Journal"}
    assert not (forbidden & imported), f"runner imports a live-path symbol: {forbidden & imported}"
    # No concrete provider import (S4): the model lives behind the DI'd ArmExecutor only.
    assert "run_frozen_check" in imported  # the one sanctioned offline kernel
    for token in ("anthropic", "openai", "ollama", "deepseek"):
        assert token not in src.lower(), f"runner names a concrete provider: {token}"


def test_runner_does_not_call_code_oracle_evaluate() -> None:
    """S1: ``CodeOracle.evaluate`` (the live, journal-bound path) is never CALLED in the runner —
    checked over the parsed AST (the docstring legitimately NAMES it in the 'never call this' prose,
    so a raw substring scan would false-fire; we scan executable code, not comments/strings)."""
    tree = ast.parse(_runner_source())
    for node in ast.walk(tree):
        # No attribute access named `.evaluate` and no reference to a `CodeOracle` name in code.
        if isinstance(node, ast.Attribute):
            assert node.attr != "evaluate", "runner references .evaluate (the live oracle path)"
        if isinstance(node, ast.Name):
            assert node.id != "CodeOracle", "runner references the CodeOracle live-path symbol"
