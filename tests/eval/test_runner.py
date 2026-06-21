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
from collections.abc import Sequence
from pathlib import Path

import pytest

from cogworx.eval.corpus import (
    ConvertedPlanterStamp,
    CorpusItem,
    DifficultyMarker,
    LLMPlanterStamp,
    OracleLabelProvenance,
)
from cogworx.eval.lock import (
    MASTER_SEED,
    MeasurementFingerprint,
    assert_arm_a_floor,
    lock_corpus,
)
from cogworx.eval.runner import (
    ArmExecutor,
    ArmInput,
    ArmOutcome,
    arm_a_executor,
    crn_resize_diagnostic,
    crn_seed,
    run_and_stamp,
    run_arms,
    scripted_executor,
)
from cogworx.eval.youden import Cell, is_converted_o, nested_bootstrap_delta
from cogworx.testing.doubles import InMemoryJournal
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
    planter: ConvertedPlanterStamp | LLMPlanterStamp
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
        stratum=stratum,
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
    # The lock's floor assertion passes (does not raise) on the emitted artifact.
    assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


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
    """S1: the runner reaches the journal-free ``run_frozen_check`` and the ``Journal`` PROTOCOL
    seam type ONLY — never ``CodeOracle``/``StageContext`` (the live verification path) and never a
    CONCRETE journal/store/provider.

    ``Journal`` is a DI seam TYPE (a Protocol), used solely as the ``journal: Journal`` annotation
    on ``run_and_stamp`` (4.4d-2, S6 — the post-run design-look append). Importing the protocol is
    the S1-correct way to type a dependency-injected seam (the same posture as the ``ArmExecutor``
    Protocol). The wall is "no concrete store / no live verification path / model off the
    write-path", not "no type annotation" — so the concrete journal impls (``TimescaleJournal``,
    ``InMemoryJournal``) stay forbidden while the Protocol is allowed."""
    src = _runner_source()
    tree = ast.parse(src)
    imported: set[str] = set()
    journal_symbols: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.name)
            if node.module:
                imported.add(node.module)
            if node.module == "cogworx.substrate.journal":
                journal_symbols.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
    # The live verification path stays wholly forbidden (S1).
    forbidden = {"CodeOracle", "StageContext"}
    assert not (forbidden & imported), f"runner imports a live-path symbol: {forbidden & imported}"
    # The ONLY symbol pulled from substrate.journal is the `Journal` Protocol — never a concrete
    # store (S1: the DI seam type is allowed, the concrete journal is not).
    assert journal_symbols <= {"Journal"}, f"runner imports non-protocol journal: {journal_symbols}"
    concrete_stores = {"TimescaleJournal", "InMemoryJournal"}
    assert not (concrete_stores & imported), "runner imports a concrete journal store (S1 wall)"
    assert "run_frozen_check" in imported  # the one sanctioned offline kernel
    # No concrete provider import (S4): the model lives behind the DI'd ArmExecutor only.
    for token in ("anthropic", "openai", "ollama", "deepseek"):
        assert token not in src.lower(), f"runner names a concrete provider: {token}"


def test_runner_emission_path_is_journal_free() -> None:
    """S1 / S6 (the emission-path lock): the model-free Cell-emitter functions are SYNCHRONOUS and
    carry ZERO awaits — so they do ZERO journal I/O (every journal write is async/awaited, so
    "no await in the emitter" structurally IS "no journal touch"). The ONLY async function in the
    module is ``run_and_stamp``, and the ONLY ``append_design_look`` call lives there, EXACTLY ONCE
    (the S6 one-look invariant, pinned over the AST to complement the behavioral count test).

    Assumption (mark for the next author): the "zero awaits" heuristic for "journal-free" holds
    because TODAY the only async dependency the emitter could acquire is the journal (the model
    lives behind the SYNCHRONOUS ``ArmExecutor.__call__``). If a future pod gives the emitter a
    different legitimately-async dependency, narrow this from "no awaits" to "no awaits on a
    Journal-typed callee"."""
    tree = ast.parse(_runner_source())
    funcs: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            funcs[node.name] = node

    emission_path = (
        "run_arms",
        "crn_seed",
        "_arm_input",
        "arm_a_executor",
        "scripted_executor",
        "crn_resize_diagnostic",
    )
    for name in emission_path:
        fn = funcs[name]
        assert isinstance(fn, ast.FunctionDef), f"{name} must be synchronous (no journal I/O)"
        awaits = [n for n in ast.walk(fn) if isinstance(n, ast.Await)]
        assert not awaits, f"emission-path fn {name} contains an await (journal I/O leak, S1)"

    # The ONLY async function in the module is run_and_stamp.
    async_fns = {n for n, fn in funcs.items() if isinstance(fn, ast.AsyncFunctionDef)}
    assert async_fns == {"run_and_stamp"}, f"unexpected async fn(s) in the runner: {async_fns}"

    # EXACTLY ONE append_design_look call, and it lives inside run_and_stamp (S6 one-look).
    all_appends = [
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Attribute) and n.attr == "append_design_look"
    ]
    assert len(all_appends) == 1, "the runner must record EXACTLY ONE design look (S6)"
    stamp_appends = [
        n
        for n in ast.walk(funcs["run_and_stamp"])
        if isinstance(n, ast.Attribute) and n.attr == "append_design_look"
    ]
    assert len(stamp_appends) == 1, "the single design-look append must live in run_and_stamp"


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


# ===========================================================================
# Pod 4.4d-2 — run_and_stamp: the fingerprint stamp + the SINGLE design-look append (S6)
# ===========================================================================

_CONFIG = "pv-config-hash-A"
_LINEAGE = ("sha-aaa", "sha-bbb")
_GIT_SHA = "deadbeefcafef00d"


def _locked_corpus() -> list[CorpusItem]:
    """A small, LOCKED corpus (content_hash stamped) the stamp path can fingerprint."""
    raw = [
        _item(1, "K", is_error=1, error_regime="off-by-one"),
        _item(2, "clean", is_error=0),
        _item(3, "O", is_error=1, error_regime="wrong-op", converted=True),
    ]
    return lock_corpus(raw)


def _stamp_executors() -> dict[str, ArmExecutor]:
    return {"A": _flag_none, "C": _flag_none, "D": _flag_all}


async def _stamp(
    j: InMemoryJournal,
    corpus: Sequence[CorpusItem],
    *,
    git_sha: str = _GIT_SHA,
    master_seed: int = MASTER_SEED,
    config: str = _CONFIG,
    lineage: tuple[str, ...] = _LINEAGE,
) -> tuple[list[Cell], MeasurementFingerprint]:
    """Drive ``run_and_stamp`` with the stamp fixtures (explicit kwargs — no dict splat, so mypy
    keeps every argument typed)."""
    return await run_and_stamp(
        corpus,
        arm_executors=_stamp_executors(),
        journal=j,
        planning_variance_config_hash=config,
        design_lineage_chain=lineage,
        git_sha=git_sha,
        master_seed=master_seed,
        R=3,
    )


async def test_run_and_stamp_appends_exactly_one_look() -> None:
    """ONE call -> ONE design-look append: the (config, lineage) budget reads exactly 1 (S6, the
    one-look invariant). NOT per-arm / per-item — 3 arms x 3 items would read 9 if it leaked."""
    j = InMemoryJournal()
    cells, fp = await _stamp(j, _locked_corpus())
    budget = await j.read_design_lineage_budget(
        planning_variance_config_hash=_CONFIG, design_lineage_chain=_LINEAGE
    )
    assert budget == 1
    # The emitted artifact is the real run output (3 items x 3 arms x 3 trials).
    assert len(cells) == 3 * 3 * 3
    # The appended fingerprint IS fp.digest (the stamp), not some other hash.
    assert isinstance(fp.digest, str) and fp.digest


async def test_run_and_stamp_idempotent_same_config_corpus() -> None:
    """Re-running the SAME (config, corpus) appends the SAME digest -> the distinct-look count is
    UNCHANGED (idempotent on (key, fingerprint), S6). A non-idempotent append would read 2."""
    j = InMemoryJournal()
    corpus = _locked_corpus()
    _, fp1 = await _stamp(j, corpus)
    _, fp2 = await _stamp(j, corpus)
    # The fingerprint is a total function of the locked corpus + folded inputs -> identical digest.
    assert fp1.digest == fp2.digest
    budget = await j.read_design_lineage_budget(
        planning_variance_config_hash=_CONFIG, design_lineage_chain=_LINEAGE
    )
    assert budget == 1


async def test_run_and_stamp_keyed_by_config_not_corpus_identity() -> None:
    """The ledger is keyed by (config_hash, lineage), NOT corpus identity: a corpus re-roll under
    the SAME config keeps ACCUMULATING looks (plan §3.9-B). A re-rolled corpus folds a different
    fingerprint, so the distinct-look count climbs to 2 under one key."""
    j = InMemoryJournal()
    corpus_a = _locked_corpus()
    # "Re-roll": a different corpus (an extra item) under the SAME (config, lineage) key.
    corpus_b = lock_corpus(
        [
            _item(1, "K", is_error=1, error_regime="off-by-one"),
            _item(2, "clean", is_error=0),
            _item(3, "O", is_error=1, error_regime="wrong-op", converted=True),
            _item(4, "clean", is_error=0),
        ]
    )
    _, fp_a = await _stamp(j, corpus_a)
    _, fp_b = await _stamp(j, corpus_b)
    assert fp_a.digest != fp_b.digest  # distinct corpora -> distinct fingerprints
    budget = await j.read_design_lineage_budget(
        planning_variance_config_hash=_CONFIG, design_lineage_chain=_LINEAGE
    )
    assert budget == 2  # accumulated under one key — NEVER reset by the re-roll


async def test_run_and_stamp_fingerprint_determinism_and_sensitivity() -> None:
    """The stamped digest is deterministic (same locked corpus + inputs -> same digest) and
    SENSITIVE (flip a folded input -> a different digest). Pins the fingerprint is load-bearing,
    not a constant."""
    j = InMemoryJournal()
    corpus = _locked_corpus()
    _, fp1 = await _stamp(j, corpus)
    _, fp2 = await _stamp(j, corpus)
    assert fp1.digest == fp2.digest  # determinism
    # Flip a folded input (the git SHA) -> a different digest (sensitivity).
    _, fp3 = await _stamp(j, corpus, git_sha="0000000000000000")
    assert fp3.digest != fp1.digest
    # Flip the master_seed (also folded) -> a different digest.
    _, fp4 = await _stamp(j, corpus, master_seed=MASTER_SEED + 1)
    assert fp4.digest != fp1.digest


async def test_run_and_stamp_refuses_never_locked_corpus() -> None:
    """The fingerprint guard fires FIRST: a never-locked corpus (content_hash == "") is refused
    (loud ValueError naming the item_id) BEFORE any arm work — run_and_stamp does NOT lock for the
    caller (lock is the audited pipeline's job; minting a fingerprint on an un-audited corpus would
    be a stale clean bill, S9/S12). And NO look is appended on the refusal."""
    j = InMemoryJournal()
    unlocked = [_item(1, "K", is_error=1), _item(2, "clean", is_error=0)]  # content_hash == ""
    with pytest.raises(ValueError, match="never-locked"):
        await _stamp(j, unlocked)
    budget = await j.read_design_lineage_budget(
        planning_variance_config_hash=_CONFIG, design_lineage_chain=_LINEAGE
    )
    assert budget == 0  # the refusal appended nothing


async def test_run_and_stamp_does_not_read_the_budget() -> None:
    """S12: run_and_stamp RECORDS the look — budget ENFORCEMENT is 4.4e. A prior look under the key
    does NOT change the run's behavior: it still appends and the count climbs, proving the function
    never gates on a pre-existing count."""
    j = InMemoryJournal()
    # Pre-seed the ledger with an unrelated look under the same key.
    await j.append_design_look(
        planning_variance_config_hash=_CONFIG,
        design_lineage_chain=_LINEAGE,
        fingerprint="pre-existing-look",
    )
    await _stamp(j, _locked_corpus())
    budget = await j.read_design_lineage_budget(
        planning_variance_config_hash=_CONFIG, design_lineage_chain=_LINEAGE
    )
    assert budget == 2  # the pre-existing look + the run's look — it appended regardless


# ===========================================================================
# Pod 4.4d-2 — the scoped CRN rho_arm re-size diagnostic (D-A exemption)
# ===========================================================================


def _cell(item_id: int, stratum: str, arm: str, trial: int, flagged: int) -> Cell:
    return Cell(
        item_id=item_id,
        stratum=stratum,
        arm=arm,
        trial=trial,
        seed=0,
        flagged=flagged,
        route="flag" if flagged else "pass",
    )


def _anti_correlated_cells(*, arm_a: str, arm_b: str, n_clean: int = 12, R: int = 4) -> list[Cell]:
    """A synthetic artifact engineered so the (arm_a, arm_b) pair has rho_arm <= 0: the two arms are
    ANTI-correlated across the clean items (where one flags, the other does not), so the paired
    delta variance EXCEEDS Var(J_a)+Var(J_b) -> CRN bought nothing (rho_arm <= 0). One K error item
    keeps sens well-defined; the spread that drives rho lives in the anti-correlated clean pool."""
    cells: list[Cell] = []
    # One K error item both arms flag (sens fixed, contributes no spec spread).
    for arm in (arm_a, arm_b):
        for t in range(R):
            cells.append(_cell(1, "K", arm, t, 1))
    # Clean pool: arm_a flags the EVEN items, arm_b flags the ODD items (perfect anti-correlation).
    for i in range(n_clean):
        cid = 100 + i
        a_flag = 1 if i % 2 == 0 else 0
        b_flag = 1 - a_flag
        for t in range(R):
            cells.append(_cell(cid, "clean", arm_a, t, a_flag))
            cells.append(_cell(cid, "clean", arm_b, t, b_flag))
    return cells


def test_crn_diagnostic_fires_on_stochastic_pair_with_nonpositive_rho() -> None:
    """The trigger FIRES on a synthetic D-C' pair engineered with rho_arm <= 0 (anti-correlated arms
    -> CRN bought no variance reduction). The pair is NOT exempt (neither arm is the deterministic
    constant A), so it contributes to ``triggered``."""
    cells = _anti_correlated_cells(arm_a="D", arm_b="C'")
    diag = crn_resize_diagnostic(
        cells, pairs=[("D>C'", "D", "C'")], error_strata=("K",), n_outer=200, seed=7
    )
    (pair,) = diag.per_pair
    assert pair.label == "D>C'"
    assert pair.exempt is False
    assert pair.rho_arm <= 0.0  # the engineered anti-correlation
    assert diag.triggered is True


def test_crn_diagnostic_exempts_d_a_pair_by_construction() -> None:
    """The D-A pair is EXEMPT: even with the IDENTICAL anti-correlated (rho<=0) flag pattern, a pair
    containing arm A does NOT trigger (arm A is a deterministic constant on K -> Var(J_A)==0 by
    construction, so rho~=0 is correct, not a CRN failure). This is the by-construction exemption
    that stops D-A false-alarming every run."""
    # Same engineered rho<=0 pattern, but the pair names arm A.
    cells = _anti_correlated_cells(arm_a="D", arm_b="A")
    diag = crn_resize_diagnostic(
        cells, pairs=[("D>A", "D", "A")], error_strata=("K",), n_outer=200, seed=7
    )
    (pair,) = diag.per_pair
    assert pair.label == "D>A"
    assert pair.exempt is True  # contains the deterministic-constant arm A
    # The pair is REPORTED (rho computed and carried) but NEVER contributes to triggered.
    assert diag.triggered is False


def test_crn_diagnostic_exemption_does_not_silence_a_stochastic_pair() -> None:
    """MUTATION CONTROL: the D-A exemption must NOT bleed into the stochastic pairs. A run carrying
    BOTH an exempt D-A (rho<=0) and a stochastic D-C' (rho<=0) STILL triggers — the exemption is
    scoped to the pair that names A, not a blanket disarm."""
    cells = _anti_correlated_cells(arm_a="D", arm_b="C'")
    # Add an arm-A clean+K view (R=4 trials per cell, matching the D/C' arms) so a D-A pair is also
    # computable on the same artifact.
    extra: list[Cell] = []
    for t in range(4):
        extra.append(_cell(1, "K", "A", t, 1))
    for i in range(12):
        for t in range(4):
            extra.append(_cell(100 + i, "clean", "A", t, 1 if i % 2 == 0 else 0))
    cells = cells + extra
    diag = crn_resize_diagnostic(
        cells,
        pairs=[("D>A", "D", "A"), ("D>C'", "D", "C'")],
        error_strata=("K",),
        n_outer=200,
        seed=7,
    )
    by_label = {p.label: p for p in diag.per_pair}
    assert by_label["D>A"].exempt is True
    assert by_label["D>C'"].exempt is False
    # The stochastic pair's non-positive rho STILL trips the trigger despite the exempt D-A present.
    assert by_label["D>C'"].rho_arm <= 0.0
    assert diag.triggered is True


def test_crn_diagnostic_does_not_trigger_when_pairing_helps() -> None:
    """NEGATIVE CONTROL: when the arms are POSITIVELY correlated (CRN helps, rho_arm > 0), the
    trigger does NOT fire — the diagnostic distinguishes a real CRN failure from a healthy run."""
    # Both arms flag the SAME even clean items (positively correlated) -> paired delta variance is
    # small relative to the marginal variances -> rho_arm > 0.
    cells: list[Cell] = []
    for arm in ("D", "C'"):
        for t in range(4):
            cells.append(_cell(1, "K", arm, t, 1))
    for i in range(12):
        flag = 1 if i % 2 == 0 else 0
        for t in range(4):
            cells.append(_cell(100 + i, "clean", "D", t, flag))
            cells.append(_cell(100 + i, "clean", "C'", t, flag))
    diag = crn_resize_diagnostic(
        cells, pairs=[("D>C'", "D", "C'")], error_strata=("K",), n_outer=200, seed=7
    )
    (pair,) = diag.per_pair
    assert pair.rho_arm > 0.0
    assert diag.triggered is False
