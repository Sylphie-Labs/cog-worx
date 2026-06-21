"""The Pod 4.4e-0 GATE-verdict-assembly spike: the verdict-CLASS suite over synthetic Cells.

These tests ARE the falsifiable spike for :func:`cogworx.eval.scorer.score_gate` (S12). They are
built to be MUTATION-RESISTANT: the load-bearing negative controls (a true-null artifact must FAIL,
a leaky artifact must be INSTRUMENT_INVALID, the converted-O/detK exclusion must keep a null from
PASSing) each carry a paired mutation test that flips the assertion when the guard is removed.

THE SHUFFLE-NULL BUDGET (eval-stats, Pod 4.4e-0): the §5 coverage ceiling 0.08 is calibrated at
``n_shuffles=200``, so the live shuffle null runs at 200 (NOT lowered) with ``n_outer=150`` (k stays
<=7, comfortable under the k=10 fail boundary; a prior n_outer>=2000 run caused a Windows access
violation under the ~2.5M-iteration load — MEMORY: test-hang-discipline). The live shuffle costs
~4.5s, so only the TWO shuffle-WIRING tests pay it; every other tier test injects a pre-built
``ShuffleNullResult`` via ``score_gate(..., shuffle_result=...)`` so it never re-runs the bootstrap.
The shuffle null's own statistics are pinned in test_lock; here we pin the scorer WIRING.

THE FIXTURE (eval-stats recipe): a 4-arm (A/D/C/B) artifact over 16 matched K<->clean pairs at R=8.
C' (Cell label "C") is a REAL stratum-associated reviewer (catches ~6/8 K errors, ~1/8 clean FPs);
D catches ~8/8 K errors with ~0 clean FPs. So D and C' carry the SAME K<->clean confound (D simply
catches MORE), which is what lets the §5 membership shuffle center at 0 while D>C' still has a real
positive δ and D>A clears the 0.8 positive-control floor.
"""

from __future__ import annotations

from random import Random

import pytest
from pydantic import ValidationError

from cogworx.eval.corpus import (
    CorpusItem,
    DeterministicPlanterStamp,
    DifficultyMarker,
    HumanLabelProvenance,
    OracleLabelProvenance,
)
from cogworx.eval.lock import (
    CorpusLockError,
    ExecEnvIdentity,
    MeasurementFingerprint,
    ShuffleNullResult,
    _shuffle_stratum_membership_LEAKY,
    assert_shuffle_null,
    shuffle_null_centering,
)
from cogworx.eval.scorer import (
    _ADAPTIVE_LADDER_CAVEAT,
    GateVerdict,
    VerdictStatus,
    _binding_k_pool,
    score_gate,
    score_gate_live,
)
from cogworx.eval.youden import Cell, nested_bootstrap_delta
from cogworx.testing.doubles import InMemoryJournal
from cogworx.verification.contracts import OracleFrame, Thesis

# Spec-ceiling / arm-A-floor PLANNED constants (a TOTAL function of planned n + R, never realized n;
# the realized clean pool here is 16, but the ceiling is sized at the ratified planning n=80). The
# planning between-item spec variance is set to the fixture's regime (C' clean flag rate jitters in
# {1/8, 2/8}, pvariance ~0.0034) so the 0.5x floor (0.0025) is comfortably cleared by the honest
# fixture and a hollow clean pool (variance 0) still fails — both binding directions exercised.
_SIGMA_SQ_B_SPEC = 0.005
_R = 8
_N_CLEAN_PLANNED = 80

# The shuffle budget (eval-stats): 200 shuffles (the calibrated ceiling's n) at n_outer=150.
_N_SHUFFLES = 200
_N_OUTER_SHUFFLE = 150
# Direct controls / binding deltas: a precise n_outer (cheap — ~0.3s for both).
_N_OUTER = 2000
_SEED = 42

_REGIMES = ("off-by-one", "spec-misread", "silent-degradation", "type-confusion", "boundary")

# 24 matched K<->clean sibling pairs — eval-stats' recipe at 16, widened to 24 so the small-n
# bootstrap CI on the positive control (D>A) is tight enough that a true-null (where D and C' both
# carry a clean-FP spec) still clears the control floor while a genuinely-weak D fails it.
_N_PAIRS = 24
_K_BASE = 100  # K item_ids 100..123
_CLEAN_BASE = 200  # clean item_ids 200..223
# The positive-control floor for THIS suite. The scorer default is 0.8 (the architect's sign-gate
# recommendation, PROVISIONAL pending the eval-stats J_pos read); the suite pins the control LOGIC
# (a real D-effect clears it, a weak D fails it) at a fixture-appropriate 0.65 — D>A is ~0.89 on a
# clean-spec D, ~0.73 on a clean-FP-carrying null D, ~0.26 on a weak D — not the production value.
_J_POS_MIN = 0.65


# ---------------------------------------------------------------------------
# Synthetic-Cell builder — the eval-stats 4-arm (A/D/C/B) matched-pair recipe. Per-arm K-catch and
# clean-FP counts are controllable so one builder emits every verdict-class fixture.
# ---------------------------------------------------------------------------


def _emit(
    cells: list[Cell],
    item_id: int,
    stratum: str,
    arm: str,
    n_flag: int,
    R: int,
    regime: str,
    *,
    converted: bool = False,
) -> None:
    """Emit R cells for ``(item, stratum, arm)`` with ``n_flag`` of R trials flagged (clamped)."""
    n = max(0, min(R, n_flag))
    for t in range(R):
        f = 1 if t < n else 0
        cells.append(
            Cell(
                item_id=item_id,
                stratum=stratum,
                arm=arm,
                trial=t,
                seed=item_id * 1000 + t,
                flagged=f,
                route="flag" if f else "pass",
                regime=regime if stratum == "K" else "",
                converted_o=converted,
            )
        )


def _artifact(
    *,
    d_k: int = 8,
    c_k: int = 6,
    b_k: int = 8,
    pc_k: int = 8,
    d_clean_fp: int = 0,
    c_clean_fp: int = 1,
    b_clean_fp: int = 7,
    pc_clean_fp: int = 0,
    a_k_flagged: int = 0,
    a_clean_fp: int = 0,
    n_pairs: int = _N_PAIRS,
    R: int = _R,
    regime_loaded: bool = False,
    hollow_clean: bool = False,
    converted_o_ids: frozenset[int] = frozenset(),
    emit_pc: bool = True,
    seed: int = 7,
) -> list[Cell]:
    """The eval-stats matched-pair fixture. Arm D catches ``d_k``/R K errors with ``d_clean_fp``
    clean FPs; C'/B catch ``c_k``/``b_k`` with ``c_clean_fp``/``b_clean_fp``; arm A is the floor
    (catches ``a_k_flagged`` K errors — INV-A0 axis — and ``a_clean_fp`` clean cells — INV-A1 axis).
    ``PC`` is the DEDICATED positive-control arm (Finding 2 — the scripted known-flag stub): it
    catches ``pc_k``/R K errors with ``pc_clean_fp`` clean FPs (a strong known effect by default).
    ``emit_pc=False`` OMITS the PC arm entirely — the missing-control fixture (Finding 1/2).

    Per item the K-catch counts share a per-item jitter in {-1,0,1} (applied identically to D/C'/B
    so the D-C' gap is constant but the LEVEL wobbles — wide-enough per-shuffle CIs), and clean FPs
    a per-item jitter in {0,1}. ``regime_loaded`` dumps every K item into one regime.
    ``converted_o_ids`` marks those K item_ids ``converted_o=True``."""
    rng = Random(seed)
    cells: list[Cell] = []
    for i in range(n_pairs):
        k_id = _K_BASE + i
        clean_id = _CLEAN_BASE + i
        regime = _REGIMES[0] if regime_loaded else _REGIMES[i % len(_REGIMES)]
        k_jit = rng.randint(-1, 1)  # shared across D/C'/B (constant gap, wobbling level)
        fp_jit = 0 if hollow_clean else rng.randint(0, 1)
        converted = k_id in converted_o_ids
        # K stratum (converted_o is a per-item provenance flag — welded across ALL arms).
        _emit(cells, k_id, "K", "A", a_k_flagged if i < a_k_flagged else 0, R, regime,
              converted=converted)
        _emit(cells, k_id, "K", "D", d_k + k_jit, R, regime, converted=converted)
        _emit(cells, k_id, "K", "C", c_k + k_jit, R, regime, converted=converted)
        _emit(cells, k_id, "K", "B", b_k + k_jit, R, regime, converted=converted)
        if emit_pc:
            _emit(cells, k_id, "K", "PC", pc_k, R, regime, converted=converted)
        # clean stratum.
        _emit(cells, clean_id, "clean", "A", 1 if i < a_clean_fp else 0, R, "")
        _emit(cells, clean_id, "clean", "D", d_clean_fp, R, "")
        _emit(cells, clean_id, "clean", "C", c_clean_fp + fp_jit, R, "")
        _emit(cells, clean_id, "clean", "B", b_clean_fp + fp_jit, R, "")
        if emit_pc:
            _emit(cells, clean_id, "clean", "PC", pc_clean_fp, R, "")
    return cells


def _null_artifact(
    *,
    n_pairs: int = _N_PAIRS,
    R: int = _R,
    pc_k: int = 8,
    seed: int = 7,
) -> list[Cell]:
    """A BYTE-IDENTICAL true-null artifact: on every genuine K and clean item, arm C' is a LITERAL
    copy of arm D's flag vector (same K catch, same clean FP, per item) — so D>C' is
    deterministically 0 (``lo == 0``), never a small real effect that only reaches lo≈0 by seed
    coincidence (the red-team's test-sturdiness flag). D still beats the floor A and the PC control
    still clears, so the instrument is valid and the FAIL is the genuine-null binding comparison,
    not an instrument.

    D's per-item K catch wobbles in {7,8} and clean FP in {0,1} (so the §5 CIs stay wide and the
    spec pool keeps variance); whatever D draws, C' is set IDENTICALLY. B is the strawman (flags
    everything); A is the floor; PC catches ``pc_k``/R K errors as the dedicated known effect."""
    rng = Random(seed)
    cells: list[Cell] = []
    for i in range(n_pairs):
        k_id = _K_BASE + i
        clean_id = _CLEAN_BASE + i
        regime = _REGIMES[i % len(_REGIMES)]
        d_k_catch = 7 + rng.randint(0, 1)  # D's K catch this item (7 or 8)
        d_clean_fp = rng.randint(0, 1)  # D's clean FP this item (0 or 1)
        _emit(cells, k_id, "K", "A", 0, R, regime)
        _emit(cells, k_id, "K", "D", d_k_catch, R, regime)
        _emit(cells, k_id, "K", "C", d_k_catch, R, regime)  # C' == D byte-for-byte on the genuine K
        _emit(cells, k_id, "K", "B", R, R, regime)  # strawman flags everything
        _emit(cells, k_id, "K", "PC", pc_k, R, regime)
        _emit(cells, clean_id, "clean", "A", 0, R, "")
        _emit(cells, clean_id, "clean", "D", d_clean_fp, R, "")
        _emit(cells, clean_id, "clean", "C", d_clean_fp, R, "")  # C' == D on the genuine clean too
        _emit(cells, clean_id, "clean", "B", R, R, "")
        _emit(cells, clean_id, "clean", "PC", 0, R, "")
    return cells


def _pairs() -> list[tuple[int, int]]:
    """The 16 matched (K-item, clean-item) sibling pairs — the §5 within-pair sign-flip input."""
    return [(_K_BASE + i, _CLEAN_BASE + i) for i in range(_N_PAIRS)]


def _corpus(cells: list[Cell], *, detk_ids: frozenset[int] = frozenset()) -> list[CorpusItem]:
    """A locked CorpusItem truth table matching the artifact's item_ids: one item per ``item_id``,
    stratum/regime carried off the cells. ``detk_ids`` stamp ``detk=True`` (the locked-truth-table
    marker the scorer JOINs on for the detK exclusion)."""
    frame = OracleFrame(completion_criterion="c", problem_type="code", problem_statement="p")
    thesis = Thesis(proposed_solution="s", experiment_design="e")
    by_item: dict[int, Cell] = {}
    for c in cells:
        by_item.setdefault(c.item_id, c)
    items: list[CorpusItem] = []
    for item_id, c in sorted(by_item.items()):
        is_clean = c.stratum == "clean"
        if is_clean:
            prov: OracleLabelProvenance | HumanLabelProvenance = OracleLabelProvenance(
                returncode=0, test_provenance="frozen", holds=True, valid_check=True, oracle_id="o"
            )
            source = "oracle"
        else:
            prov = HumanLabelProvenance(adjudications=())
            source = "human"
        items.append(
            CorpusItem(
                item_id=item_id,
                frame=frame,
                thesis=thesis,
                test_code="t",
                is_error=0 if is_clean else 1,
                label_source=source,
                label_provenance=prov,
                stratum=c.stratum,
                oracle_reachable=False,
                error_regime=c.regime,
                difficulty=DifficultyMarker(planted_difficulty="medium", surface_complexity=1),
                matched_sibling_id=None,
                split="measurement",
                planter=DeterministicPlanterStamp(operators=("op",)),
                detk=item_id in detk_ids,
                content_hash=f"h{item_id}",
            )
        )
    return items


def _fingerprint(*, audited: bool = True) -> MeasurementFingerprint:
    env = ExecEnvIdentity(
        python_version="3.13.0",
        python_implementation="CPython",
        package_versions=(("pydantic", "2.0"),),
        locale_lc_ctype="C",
    )
    return MeasurementFingerprint(
        content_hash_aggregate="agg",
        git_sha="deadbeef",
        planter_families=("deterministic-mutation",),
        exec_env=env,
        residual_epsilon=0.0 if audited else -1.0,
    )


def _clean_shuffle_result() -> ShuffleNullResult:
    """A hand-built CLEAN ShuffleNullResult (centering at 0, no leakage-direction exclusions) — the
    cheap injection that lets a non-shuffle-wiring tier test pass the §5 gate without re-running the
    ~4.5s bootstrap. Mirrors the test_lock hand-built-result pattern."""
    n = _N_SHUFFLES
    return ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>C'": tuple(0.0 for _ in range(n))},
        paired_ci_bounds={"D>C'": tuple((-0.5, 0.5) for _ in range(n))},
        global_ci_bounds={"D>C'": tuple((-0.5, 0.5) for _ in range(n))},
    )


def _score(cells: list[Cell], corpus: list[CorpusItem], **kw: object) -> GateVerdict:
    """Drive score_gate with the suite defaults. By default the §5 null is INJECTED (clean) so the
    test is cheap; pass ``shuffle_result=None`` + the pairs to run it live."""
    params: dict[str, object] = {
        "expected_fingerprint": None,
        "lineage_look_count": 0,
        "n_clean_planned": _N_CLEAN_PLANNED,
        "R": _R,
        "gate_threshold": 0.0,
        "n_outer": _N_OUTER,
        "look_budget_max": 10,
        "sigma_sq_b_spec_planning": _SIGMA_SQ_B_SPEC,
        "shuffle_seed": _SEED,
        "n_shuffles": _N_SHUFFLES,
        "shuffle_paired_ids": _pairs(),
        "shuffle_result": _clean_shuffle_result(),
        "j_pos_min": _J_POS_MIN,
    }
    params.update(kw)
    return score_gate(cells, corpus, _fingerprint(), **params)  # type: ignore[arg-type]


# ===========================================================================
# Tier 4 — the binding comparison: PASS and the load-bearing true-NULL FAIL
# ===========================================================================


def test_clean_d_win_artifact_passes() -> None:
    """A clean D-win artifact (D catches 8/8 K errors, C' ~6/8, A=0; spec non-trivial) clears BOTH
    binding deltas + every instrument + control -> PASS, ``passed is True``. The §5 null is injected
    clean here (its live wiring is pinned separately in the shuffle-wiring test)."""
    cells = _artifact()
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "PASS"
    assert verdict.passed is True
    assert verdict.binding_deltas["D>A"].lo > 0.0
    assert verdict.binding_deltas["D>C'"].lo > 0.0
    assert verdict.controls is not None and verdict.controls.passed


def test_true_null_artifact_fails_not_passes() -> None:
    """THE LOAD-BEARING NEGATIVE CONTROL: a BYTE-IDENTICAL true-null antithesis (C' is a literal
    copy of D's flag vector on every genuine K and clean item) MUST FAIL, never PASS. Because D==C'
    byte for byte, EVERY bootstrap resample yields δ=0 deterministically -> ``lo == 0`` (not a
    seed-coincident lo≈0). D>A still wins (D beats the flags-nothing floor) and the dedicated PC
    control clears, so the instrument is valid and the FAIL is the genuine null."""
    cells = _null_artifact()
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "FAIL"
    assert verdict.passed is False
    assert verdict.binding_deltas["D>A"].lo > 0.0  # the instrument is still valid (D beats floor)
    assert verdict.binding_deltas["D>C'"].lo == 0.0  # byte-identical null -> deterministic lo == 0


# ===========================================================================
# Tier 1 — the §5 shuffle null, LIVE-WIRED (the honest-pass + the leaky-fail mutation)
# ===========================================================================


def test_live_shuffle_null_honest_artifact_passes_wiring() -> None:
    """SHUFFLE-WIRING (live, ~4.5s): the honest D-win artifact, with the §5 null run LIVE over the
    matched pairs at the calibrated n_shuffles=200, passes the shuffle gate -> the scorer reaches a
    real verdict (PASS). Proves the scorer wires shuffle_null_centering -> assert_shuffle_null."""
    cells = _artifact()
    verdict = _score(
        cells,
        _corpus(cells),
        shuffle_result=None,  # run the §5 null LIVE
        n_outer=_N_OUTER_SHUFFLE,
    )
    shuffle_gate = next(g for g in verdict.instrument_gates if g.name == "shuffle-null")
    assert shuffle_gate.passed is True
    assert verdict.verdict_status == "PASS"


def test_live_shuffle_leaky_mutation_is_instrument_invalid() -> None:
    """SHUFFLE-WIRING MUTATION (the shuffle gate is load-bearing): the §5 null is the honest one,
    but we prove its teeth by contrast — a hand-run of the leaky shuffle on this artifact FIRES the
    centering assertion, while the honest shuffle does NOT. So the gate distinguishes leak from
    honest; an assertion that cannot fire tests nothing."""
    cells = _artifact()
    pairs = _pairs()
    unpaired: list[int] = []
    deltas = {"D>C'": ("D", "C")}
    honest = shuffle_null_centering(
        cells,
        paired_ids=pairs,
        unpaired_ids=unpaired,
        deltas=deltas,
        n_shuffles=_N_SHUFFLES,
        seed=_SEED,
        n_outer=_N_OUTER_SHUFFLE,
    )
    assert_shuffle_null(honest)  # honest -> no raise (the gate does NOT false-fire)
    leaky = shuffle_null_centering(
        cells,
        paired_ids=pairs,
        unpaired_ids=unpaired,
        deltas=deltas,
        n_shuffles=_N_SHUFFLES,
        seed=_SEED,
        n_outer=_N_OUTER_SHUFFLE,
        _shuffle=_shuffle_stratum_membership_LEAKY,
    )
    with pytest.raises(CorpusLockError, match="centering"):
        assert_shuffle_null(leaky)
    # And the scorer maps a refused shuffle to INSTRUMENT_INVALID (inject the leaky result).
    verdict = _score(cells, _corpus(cells), shuffle_result=leaky)
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "shuffle-null")
    assert gate.passed is False


# ===========================================================================
# Tier 1 — the other instrument-validity gates -> INSTRUMENT_INVALID
# ===========================================================================


def test_arm_a_flags_k_is_instrument_invalid() -> None:
    """INV-A0: arm A flags a K item -> the item is oracle-reachable, mis-stratified -> the arm-A
    floor refuses -> INSTRUMENT_INVALID (the scorer reports the gate, never crashes)."""
    cells = _artifact(a_k_flagged=2)
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "arm-a-floor")
    assert gate.passed is False and "INV-A0" in gate.detail


def test_hollow_clean_is_instrument_invalid_via_spec_ceiling() -> None:
    """A hollow clean pool (no clean item ever false-positives on C' -> spec at the planned-n
    ceiling) -> the spec-ceiling tripwire refuses -> INSTRUMENT_INVALID."""
    # C' never false-positives (hollow_clean zeroes the FP jitter too) -> spec_C' = 1.0, AT or above
    # the ceiling 1 - 2/(R*n_clean_planned).
    cells = _artifact(c_clean_fp=0, b_clean_fp=0, d_clean_fp=0, hollow_clean=True)
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "spec-ceiling")
    assert gate.passed is False


def test_regime_loaded_is_instrument_invalid_via_contribution() -> None:
    """A regime-loaded artifact (ALL K sensitivity dumped into one regime) -> the §2.B per-regime
    contribution bound refuses -> INSTRUMENT_INVALID."""
    cells = _artifact(regime_loaded=True)
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "regime-contribution")
    assert gate.passed is False


# ===========================================================================
# Tier 2 — controls -> INSTRUMENT_INVALID
# ===========================================================================


def test_positive_control_fails_is_instrument_invalid() -> None:
    """The positive control (PC>A J-LCB > j_pos_min) fails when the DEDICATED control arm barely
    beats the floor: PC's K catch is so low that PC>A's lo does not clear j_pos_min ->
    INSTRUMENT_INVALID (the instrument cannot detect a known effect). All Tier-1 gates pass first
    (spec + regime + arm-A floor still hold). D stays a clean win — the control failure is the PC
    arm, INDEPENDENT of the arm-under-test (Finding 2: the control is no longer read off D)."""
    # PC catches only ~3/8 K errors -> PC>A lo well under the control floor; D/C'/spec stay honest
    # so Tier-1 clears and the failure is the control, not an instrument or the binding comparison.
    cells = _artifact(pc_k=3)
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    assert verdict.controls is not None
    assert verdict.controls.positive_control_cleared is False
    assert all(g.passed for g in verdict.instrument_gates)


# ===========================================================================
# Tier 0 — fingerprint integrity -> INSTRUMENT_INVALID
# ===========================================================================


def test_unaudited_fingerprint_is_instrument_invalid() -> None:
    """An unaudited fingerprint (residual-ε sentinel) -> refused at Tier 0 BEFORE any statistic
    runs -> INSTRUMENT_INVALID with the fingerprint-audited gate failed and NO binding deltas."""
    cells = _artifact()
    verdict = score_gate(
        cells,
        _corpus(cells),
        _fingerprint(audited=False),
        n_outer=_N_OUTER,
        sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
        shuffle_seed=_SEED,
        shuffle_result=_clean_shuffle_result(),
    )
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "fingerprint-audited")
    assert gate.passed is False
    assert verdict.binding_deltas == {}


def test_fingerprint_digest_mismatch_is_instrument_invalid() -> None:
    """A digest mismatch against an ``expected_fingerprint`` -> Tier 0 refuse (tampered/stale
    instrument)."""
    cells = _artifact()
    expected = _fingerprint().model_copy(update={"git_sha": "different"})
    verdict = _score(cells, _corpus(cells), expected_fingerprint=expected)
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "fingerprint-digest")
    assert gate.passed is False


# ===========================================================================
# Tier 3 — lineage budget -> BUDGET_EXHAUSTED
# ===========================================================================


def test_lineage_budget_exhausted_short_circuits() -> None:
    """``lineage_look_count >= look_budget_max`` -> BUDGET_EXHAUSTED. The artifact is a clean PASS
    in every other respect, so the budget is the SOLE cause (proves the short-circuit), and no
    binding deltas are computed."""
    cells = _artifact()
    verdict = _score(cells, _corpus(cells), lineage_look_count=10, look_budget_max=10)
    assert verdict.verdict_status == "BUDGET_EXHAUSTED"
    assert verdict.passed is False
    assert verdict.lineage.exhausted is True
    assert verdict.binding_deltas == {}


def test_lineage_budget_under_ceiling_does_not_short_circuit() -> None:
    """The negative control: a look count just under the ceiling does NOT exhaust -> the gate runs
    to a real verdict (PASS here)."""
    cells = _artifact()
    verdict = _score(cells, _corpus(cells), lineage_look_count=9, look_budget_max=10)
    assert verdict.verdict_status == "PASS"
    assert verdict.lineage.exhausted is False


# ===========================================================================
# The converted-O / detK exclusion is LOAD-BEARING (+ the mutation test)
# ===========================================================================


def test_converted_o_and_detk_excluded_from_binding_pool() -> None:
    """The firewall: converted-O cells (per-cell flag) AND detK rows (corpus JOIN on
    ``CorpusItem.detk``) are removed from the binding K pool; clean + non-K cells pass through. The
    reported-only bundle surfaces the counts so the bite is auditable."""
    converted = frozenset({100, 101})
    detk = frozenset({110, 111})
    cells = _artifact(converted_o_ids=converted)
    corpus = _corpus(cells, detk_ids=detk)
    pool, n_conv, n_detk = _binding_k_pool(cells, corpus)
    pool_k_ids = {c.item_id for c in pool if c.stratum == "K"}
    assert pool_k_ids.isdisjoint(converted | detk)
    assert n_conv == len(converted) * 5 * _R  # 5 K arms (A/D/C/B/PC) x R per excluded converted-O
    assert n_detk == len(detk) * 5 * _R
    assert any(c.stratum == "clean" for c in pool)  # clean cells survive (spec arm keeps its pool)


def test_detk_exclusion_mutation_easy_rows_flip_a_null_to_pass() -> None:
    """MUTATION TEST (the exclusion is load-bearing): seed the K pool with EASY detK rows that D
    catches but C' does not, on an otherwise-NULL artifact (D == C' on the honest K items). WITH the
    exclusion the binding D>C' stays null -> lo <= 0. WITHOUT it (mutation: do not JOIN detk) the
    easy detK rows inflate D>C' -> lo > 0. The contrast proves the exclusion flips the verdict."""
    rng = Random(7)
    cells = _artifact(d_k=8, c_k=8, d_clean_fp=1, c_clean_fp=1)  # honest K items: D == C' (null)
    detk_ids: set[int] = set()
    for i in range(16):
        item_id = 900 + i
        detk_ids.add(item_id)
        regime = _REGIMES[i % len(_REGIMES)]
        _emit(cells, item_id, "K", "A", 0, _R, regime)
        _emit(cells, item_id, "K", "D", _R, _R, regime)  # D catches the easy detK row every trial
        _emit(cells, item_id, "K", "C", 0, _R, regime)  # C' misses it -> inflates D>C'
        _emit(cells, item_id, "K", "B", rng.randint(0, _R), _R, regime)
    corpus = _corpus(cells, detk_ids=frozenset(detk_ids))

    pool_excl, _, n_detk = _binding_k_pool(cells, corpus)
    assert n_detk == len(detk_ids) * 4 * _R
    _, lo_excl, _ = nested_bootstrap_delta(
        pool_excl,
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=_N_OUTER,
        seed=1,
        quantile=0.025,
    )
    # WITHOUT the exclusion (the mutation: detk NOT joined): the easy detK rows inflate D>C'.
    _, lo_incl, _ = nested_bootstrap_delta(
        cells,
        arm_a="D",
        arm_b="C",
        error_strata=("K",),
        n_outer=_N_OUTER,
        seed=1,
        quantile=0.025,
    )
    assert lo_excl <= 0.0 < lo_incl, (
        f"the detK exclusion must keep the null from clearing the gate: "
        f"excluded lo={lo_excl:.4f} (<=0), included lo={lo_incl:.4f} (>0)"
    )


# ===========================================================================
# Finding 1/2 — a MISSING arm -> INSTRUMENT_INVALID (the scorer never crashes)
# ===========================================================================


def test_missing_positive_control_arm_is_instrument_invalid_not_crash() -> None:
    """FINDING 1/2: ``positive_control_arm="ZZZ"`` (an arm absent from the artifact) -> the scorer
    REPORTS INSTRUMENT_INVALID naming the missing arm; it does NOT raise StopIteration/KeyError. The
    arm-present gate fails; no binding deltas are computed (Tier 2 short-circuits)."""
    cells = _artifact()
    verdict = _score(cells, _corpus(cells), positive_control_arm="ZZZ")
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "arm-present")
    assert gate.passed is False and "ZZZ" in gate.detail
    assert verdict.binding_deltas == {}


def test_missing_pc_arm_from_pipeline_is_instrument_invalid() -> None:
    """FINDING 2: the live run MUST emit a dedicated ``PC`` arm. An artifact that did NOT run it
    (``emit_pc=False``) -> the default ``positive_control_arm="PC"`` is absent -> INSTRUMENT_INVALID
    (the pipeline skipped the required control) rather than silently reading the control off D."""
    cells = _artifact(emit_pc=False)
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "arm-present")
    assert gate.passed is False and "PC" in gate.detail


def test_pc_is_the_default_positive_control_arm() -> None:
    """FINDING 2: the scorer's positive control reads the DEDICATED ``PC`` arm, not the arm-under-
    test D. Driving a weak PC (but a strong D) FAILS the control -> INSTRUMENT_INVALID; if the
    control still read off D, the strong D would falsely clear it. Pins the de-circularization."""
    cells = _artifact(pc_k=2)  # weak control, strong D-win on D/C'
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    assert verdict.controls is not None and verdict.controls.positive_control_cleared is False


def test_missing_binding_arm_is_instrument_invalid_not_crash() -> None:
    """FINDING 1: a missing BINDING arm (drop every D cell — so the D>A / D>C' binding reads name an
    absent arm) -> INSTRUMENT_INVALID naming "D", not a crash. Tier 0/1 pass (C'/B keep the spec +
    regime variance; the §5 null is injected) and the controls clear (PC>A + B>A do not need D), so
    the missing arm surfaces at the Tier-4 binding read with the controls already reported."""
    cells = [c for c in _artifact() if c.arm != "D"]
    verdict = _score(cells, _corpus(cells))
    assert verdict.verdict_status == "INSTRUMENT_INVALID"
    gate = next(g for g in verdict.instrument_gates if g.name == "arm-present")
    assert gate.passed is False and "D" in gate.detail
    assert verdict.controls is not None and verdict.controls.passed  # controls ran before Tier 4
    assert verdict.binding_deltas == {}


# ===========================================================================
# Finding 3 — the converted-O/detK firewall is load-bearing AT THE VERDICT level
# ===========================================================================


def test_firewall_e2e_easy_rows_do_not_flip_null_to_pass() -> None:
    """FINDING 3 (end-to-end, MUTATION PIN): a BYTE-IDENTICAL true-null (D==C' on the genuine K)
    PADDED with easy converted-O + detK rows where D flags every trial and C' flags none. The
    FIREWALLED verdict is FAIL (the binding D>C' stays null, lo==0) because the easy rows are
    excluded from the binding K pool. If Tier 4 is mutated to read raw ``cells`` instead of the
    firewalled ``pool``, the easy rows inflate δ -> lo>0 -> the verdict flips to PASS -> THIS TEST
    FAILS, proving the firewall WIRING into the verdict is load-bearing (not just the helper)."""
    cells = _null_artifact()
    converted_ids: set[int] = set()
    detk_ids: set[int] = set()
    for i in range(12):
        # easy converted-O rows: D catches every trial, C' none -> inflates D>C' if not firewalled.
        conv_id = 800 + i
        converted_ids.add(conv_id)
        regime = _REGIMES[i % len(_REGIMES)]
        _emit(cells, conv_id, "K", "A", 0, _R, regime, converted=True)
        _emit(cells, conv_id, "K", "D", _R, _R, regime, converted=True)
        _emit(cells, conv_id, "K", "C", 0, _R, regime, converted=True)
        _emit(cells, conv_id, "K", "B", _R, _R, regime, converted=True)
        _emit(cells, conv_id, "K", "PC", _R, _R, regime, converted=True)
        # easy detK rows: same shape, excluded via the corpus JOIN on detk.
        dk_id = 900 + i
        detk_ids.add(dk_id)
        _emit(cells, dk_id, "K", "A", 0, _R, regime)
        _emit(cells, dk_id, "K", "D", _R, _R, regime)
        _emit(cells, dk_id, "K", "C", 0, _R, regime)
        _emit(cells, dk_id, "K", "B", _R, _R, regime)
        _emit(cells, dk_id, "K", "PC", _R, _R, regime)
    corpus = _corpus(cells, detk_ids=frozenset(detk_ids))
    verdict = _score(cells, corpus)
    assert verdict.verdict_status == "FAIL"  # MUTATION PIN: raw-cells Tier-4 -> PASS -> this fails
    assert verdict.passed is False
    assert verdict.binding_deltas["D>C'"].lo == 0.0  # the genuine null survives the padding
    assert verdict.reported_only.excluded_converted_o == len(converted_ids) * 5 * _R
    assert verdict.reported_only.excluded_detk == len(detk_ids) * 5 * _R


# ===========================================================================
# Finding 4 — passed <-> verdict_status coupled in the TYPE
# ===========================================================================


def test_passed_status_validator_rejects_contradiction() -> None:
    """FINDING 4 (S9): the ``passed <-> verdict_status`` invariant lives in the GateVerdict TYPE. A
    directly-constructed verdict whose boolean disagrees with its status RAISES — neither a buggy
    factory nor a tampered deserialization can mint passed=True on a non-PASS (or the inverse)."""
    base = _score(_artifact(), _corpus(_artifact()))
    bad_pairs: tuple[tuple[bool, VerdictStatus], ...] = (
        (True, "FAIL"),
        (False, "PASS"),
        (True, "INSTRUMENT_INVALID"),
    )
    for bad_passed, bad_status in bad_pairs:
        with pytest.raises(ValidationError, match="contradicts verdict_status"):
            GateVerdict(
                passed=bad_passed,
                verdict_status=bad_status,
                lineage=base.lineage,
                reported_only=base.reported_only,
            )
    # The honest pairing validates.
    ok = GateVerdict(
        passed=True,
        verdict_status="PASS",
        lineage=base.lineage,
        reported_only=base.reported_only,
    )
    assert ok.passed is True


# ===========================================================================
# Verdict shape / content-addressability
# ===========================================================================


def test_verdict_is_content_addressable_and_frozen() -> None:
    """The verdict is frozen + carries a deterministic content digest (identical inputs -> identical
    digest); ``passed`` is True ONLY for PASS."""
    cells = _artifact()
    corpus = _corpus(cells)
    v1 = _score(cells, corpus)
    v2 = _score(cells, corpus)
    assert v1.digest == v2.digest
    with pytest.raises(ValidationError):
        v1.passed = False  # type: ignore[misc]


# ===========================================================================
# Pod 4.4e-1 — score_gate_live: the journal-budget-READ wrapper over the pure scorer
# ===========================================================================

_PV_CONFIG = "pv-config-hash-live"
_LINEAGE: tuple[str, ...] = ("sha-aaa", "sha-bbb")


class _CountingJournal(InMemoryJournal):
    """An ``InMemoryJournal`` that COUNTS its budget-ledger calls — so a test can pin that
    ``score_gate_live`` READs the budget exactly once and NEVER appends (the S6 no-double-append
    invariant: ``run_and_stamp`` owns the append at measurement time, not the scorer)."""

    def __init__(self) -> None:
        super().__init__()
        self.read_calls = 0
        self.append_calls = 0

    async def read_design_lineage_budget(
        self, *, planning_variance_config_hash: str, design_lineage_chain: object
    ) -> int:
        self.read_calls += 1
        return await super().read_design_lineage_budget(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,  # type: ignore[arg-type]
        )

    async def append_design_look(
        self, *, planning_variance_config_hash: str, design_lineage_chain: object, fingerprint: str
    ) -> None:
        self.append_calls += 1
        await super().append_design_look(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,  # type: ignore[arg-type]
            fingerprint=fingerprint,
        )


async def _score_live(
    cells: list[Cell],
    corpus: list[CorpusItem],
    journal: _CountingJournal,
    **kw: object,
) -> GateVerdict:
    """Drive ``score_gate_live`` with the suite defaults + the injected clean §5 null (cheap)."""
    params: dict[str, object] = {
        "journal": journal,
        "planning_variance_config_hash": _PV_CONFIG,
        "design_lineage_chain": _LINEAGE,
        "expected_fingerprint": None,
        "n_clean_planned": _N_CLEAN_PLANNED,
        "R": _R,
        "gate_threshold": 0.0,
        "n_outer": _N_OUTER,
        "look_budget_max": 10,
        "sigma_sq_b_spec_planning": _SIGMA_SQ_B_SPEC,
        "shuffle_seed": _SEED,
        "n_shuffles": _N_SHUFFLES,
        "shuffle_paired_ids": _pairs(),
        "shuffle_result": _clean_shuffle_result(),
        "j_pos_min": _J_POS_MIN,
    }
    params.update(kw)
    return await score_gate_live(cells, corpus, _fingerprint(), **params)  # type: ignore[arg-type]


async def _seed_looks(journal: _CountingJournal, n: int) -> None:
    """Append ``n`` distinct design looks under the suite key so the READ returns ``n``."""
    for i in range(n):
        await journal.append_design_look(
            planning_variance_config_hash=_PV_CONFIG,
            design_lineage_chain=_LINEAGE,
            fingerprint=f"fp-{i}",
        )
    journal.append_calls = 0  # reset: the seeding appends are the harness, not the wrapper's


async def test_score_gate_live_under_budget_scores_normally() -> None:
    """``look_count < 10`` on the journal -> the wrapper reaches a real verdict (Tier 4 PASS): the
    READ feeds the pure scorer, which runs the binding comparison."""
    journal = _CountingJournal()
    await _seed_looks(journal, 9)
    cells = _artifact()
    verdict = await _score_live(cells, _corpus(cells), journal)
    assert verdict.verdict_status == "PASS"
    assert verdict.lineage.look_count == 9
    assert verdict.lineage.exhausted is False


async def test_score_gate_live_at_budget_is_exhausted() -> None:
    """``look_count >= 10`` on the journal -> BUDGET_EXHAUSTED, short-circuited before the binding
    comparison (no binding deltas), even though the artifact is an otherwise-clean PASS."""
    journal = _CountingJournal()
    await _seed_looks(journal, 10)
    cells = _artifact()
    verdict = await _score_live(cells, _corpus(cells), journal)
    assert verdict.verdict_status == "BUDGET_EXHAUSTED"
    assert verdict.passed is False
    assert verdict.lineage.look_count == 10
    assert verdict.lineage.exhausted is True
    assert verdict.binding_deltas == {}


async def test_score_gate_live_reads_once_appends_never() -> None:
    """THE NO-DOUBLE-APPEND PIN (load-bearing S6): ``score_gate_live`` calls
    ``read_design_lineage_budget`` EXACTLY ONCE and ``append_design_look`` ZERO times — the append
    is ``run_and_stamp``'s job at measurement time; a second append here would double-count."""
    journal = _CountingJournal()
    await _seed_looks(journal, 3)
    cells = _artifact()
    await _score_live(cells, _corpus(cells), journal)
    assert journal.read_calls == 1
    assert journal.append_calls == 0


def _assert_caveat(lineage: object) -> None:
    """The lineage status carries the verbatim CF-4.4c-ADAPTIVE-LADDER caveat + the within-config
    scope line (so the look-count is observable WITH its loose-bound caveat on every verdict)."""
    assert isinstance(lineage, object)
    caveat = lineage.caveat  # type: ignore[attr-defined]
    assert caveat == _ADAPTIVE_LADDER_CAVEAT
    assert "Bonferroni approximation" in caveat
    assert "monotone only WITHIN a planning_variance_config_hash" in caveat
    assert lineage.path == "ceiling-v1"  # type: ignore[attr-defined]


async def test_lineage_caveat_on_pass_fail_and_exhausted() -> None:
    """The caveat + within-config scope line ride on EVERY verdict class — PASS, FAIL, AND
    BUDGET_EXHAUSTED — so the spent-vs-ceiling distance is always observable with its honest caveat
    (eval-stats S9-honesty: 'approaching K_max' must be visible before it trips)."""
    # PASS — honest D-win, under budget.
    j_pass = _CountingJournal()
    await _seed_looks(j_pass, 0)
    pass_cells = _artifact()
    v_pass = await _score_live(pass_cells, _corpus(pass_cells), j_pass)
    assert v_pass.verdict_status == "PASS"
    _assert_caveat(v_pass.lineage)

    # FAIL — BYTE-IDENTICAL true-null antithesis (C' == D), under budget.
    j_fail = _CountingJournal()
    await _seed_looks(j_fail, 0)
    fail_cells = _null_artifact()
    v_fail = await _score_live(fail_cells, _corpus(fail_cells), j_fail)
    assert v_fail.verdict_status == "FAIL"
    _assert_caveat(v_fail.lineage)

    # BUDGET_EXHAUSTED — clean artifact, but the budget is spent.
    j_exh = _CountingJournal()
    await _seed_looks(j_exh, 10)
    exh_cells = _artifact()
    v_exh = await _score_live(exh_cells, _corpus(exh_cells), j_exh)
    assert v_exh.verdict_status == "BUDGET_EXHAUSTED"
    _assert_caveat(v_exh.lineage)
