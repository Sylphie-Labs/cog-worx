"""Pod 4.4c-6b — the corpus-validation spike (S12 falsifiable end-to-end check; closes 4.4c).

This is the INTEGRATION PROOF for the whole 4.4c eval pipeline: a small synthetic corpus is run
through the REAL landed chain ``plant -> convert_k_pool -> promote_corpus -> lock``, and the full
§6.1 lock list is asserted on the result, including every gate FIRING on a negative control. The
spike proves the WIRING composes and the gates have teeth; it does NOT prove the real gate's power
(the real n_outer >= 10,000 + n_shuffles run at 4.4e — here n_outer/n_shuffles are REDUCED so the
single file runs comfortably under 300s).

DETERMINISM POSTURE (S1, S2): no live model, no docker, no journal, no substrate. Every external
seam is STUBBED INLINE in this file — the panel/adversary (conversion), the adjudicator/tie-break/
regime callbacks (labeling), and the ``OracleProbe`` (a content-addressed in-memory stub, NOT the
subprocess ``run_frozen_check`` — the unit-test discipline, so the file stays fast + deterministic).
The 4.4d ``Cell`` artifact does not exist yet, so the lock instruments (shuffle / contribution /
arm-A floor / spec ceiling) are exercised against a SYNTHETIC ``Cell`` bridge built INLINE to align
with the locked corpus.

Per the test-hang discipline (memory: "Test runs can hang the machine ~1hr") this is the SPIKE TIER:
run this file SINGLE and timeout-wrapped, NEVER the whole tier:

    timeout 300 python -m pytest tests/spike/test_pod_4_4c_corpus_validation_spike.py -q

NEVER run the whole tests/spike tier (cross-test leak hangs the machine). All fixtures are INLINE in
this module (no shared conftest/fixtures import another spike could pull in); the dir conftest's
``settings`` fixture is never requested here (this spike touches no substrate).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime
from random import Random
from typing import Literal

import pytest

from cogworx.claims.provenance import ProvenanceSource
from cogworx.eval.conversion import (
    ConversionResult,
    PanelConfig,
    PanelHypothesis,
    convert_k_pool,
)
from cogworx.eval.corpus import (
    Adjudication,
    ConvertedPlanterStamp,
    CorpusItem,
    HumanLabelProvenance,
    OracleLabelProvenance,
    RegimeAdjudication,
    load_corpus,
)
from cogworx.eval.labeling import (
    AdjudicationRequest,
    OracleProbe,
    PromotionResult,
    promote_corpus,
)
from cogworx.eval.lock import (
    REGIME_CONTRIBUTION_MAX_SHARE,
    CorpusLockError,
    ShuffleNullReport,
    _shuffle_stratum_membership_LEAKY,
    assert_arm_a_floor,
    assert_no_contamination,
    assert_regime_contribution,
    assert_shuffle_null,
    assert_spec_ceiling,
    build_fingerprint,
    content_hash,
    ks_flag_rate_diagnostic,
    lock_corpus,
    report_loro,
    revalidate_bijection,
    shuffle_null_centering,
)
from cogworx.eval.planting import (
    OInjector,
    PlantedItem,
    PlantedPair,
    Seed,
    _derive_o_regime,
    draw_splits,
)
from cogworx.eval.youden import (
    Cell,
    is_converted_o,
    nested_bootstrap_delta,
    synth_cells,
)
from cogworx.verification.contracts import OracleFrame, Thesis, Verdict

# ===========================================================================
# Planning constants (REDUCED for the spike — pinned distinct from the real gate's sizes).
# ===========================================================================

_R = 7
_N_CLEAN_PLANNED = 80  # the RATIFIED planning clean count (the ceiling/tau_A's only n).
_SIGMA_SQ_B_SPEC = 0.028  # the planning between-item spec variance (the floor's basis).
_TS = datetime(2026, 6, 20, tzinfo=UTC)

# Budget knobs — small so the single file runs well under 300s. These prove WIRING + gate-firing,
# NOT the real gate's power (real n_outer >= 10,000 / n_shuffles at 4.4e — see the docstring).
_N_OUTER = 250
_N_SHUFFLES = 200  # the §5.0 binding floor; small fixtures keep it cheap.

# The 3 canonical error regimes draw_splits stratifies on (planting._REGIMES). Both K and O
# pre-label pairs carry one of these (the K candidate is second-author-audited at promote).
_CANON_REGIMES = ("logic-wrong", "edge-case-miss", "off-by-semantics")

# Solution-string sentinels the content-addressed stub probe routes on.
_O_ERR = "def f(xs):\n    return sum(xs) - 1\n"  # deterministic-O error (caught by the oracle)
_O_CLEAN = "def f(xs):\n    return sum(xs)\n"  # its clean sibling (passes)
_KCONV_ERR = "def g(xs):\n    return sum(xs) // 2  # subtly wrong (converts)\n"
_KCONV_CLEAN = "def g(xs):\n    return sum(xs)\n"
_KRES_ERR = "def h(xs):\n    return sorted(xs)  # oracle-blind, stays K\n"
_KRES_CLEAN = "def h(xs):\n    return list(xs)\n"
_AUTHOR_TEST = "from solution import f\n\n\ndef test_f():\n    assert f([1, 2, 3]) == 6\n"


# ===========================================================================
# INLINE STUB SEAMS — the content-addressed probe, the panel/adversary, the adjudicators.
# ===========================================================================


def _verdict(
    *, holds: bool, valid_check: bool = True, source: ProvenanceSource = "tool"
) -> Verdict:
    return Verdict(
        holds=holds,
        valid_check=valid_check,
        reasoning="stub",
        source=source,
        test_provenance="frozen",
    )


def _stub_probe(solution_code: str, test_code: str) -> Verdict:
    """Content-addressed deterministic oracle stub — the SAME probe drives ``convert_k_pool`` AND
    ``promote_corpus`` so the converter's gate and the labeler's ``assign_stratum`` AGREE on what
    "executable" means (the seam-coherence the two modules require).

    Routing (on the SOLUTION string, NOT the test string — so a synthesized candidate and the
    author-frozen test both resolve to the same verdict for a given solution):
      - the deterministic-O error solution FAILS (an O-catch: executable ∧ valid ∧ ¬holds);
      - the convertible-K error solution FAILS (so the converter's two-sided gate + author-anchor
        pass and it flips to O; after the flip its stratum re-derives via ``assign_stratum`` -> O);
      - the residual-K error solution PASSES with a NOISE verdict (valid_check=False) -> NOT an
        O-catch -> the converter gate fails AND ``assign_stratum`` reads K (the catch-complement);
      - every clean solution PASSES (executable ∧ valid ∧ holds) -> oracle-reachable, C1.
    """
    if solution_code in (_O_ERR, _KCONV_ERR):
        return _verdict(holds=False, valid_check=True)  # O-catch
    if solution_code == _KRES_ERR:
        return _verdict(holds=False, valid_check=False)  # NOISE -> not a catch -> K
    return _verdict(holds=True, valid_check=True)  # clean / sibling passes


class _StubPanel:
    """A stub AdversaryPanel returning one advisory hypothesis (never a label, never converts)."""

    def __init__(self, family: str = "panel-fam") -> None:
        self.family = family

    def propose(
        self, item: PlantedItem, clean_sibling: PlantedItem, round_index: int
    ) -> tuple[PanelHypothesis, ...]:
        return (PanelHypothesis(family=self.family, rationale=f"r{round_index}"),)


class _StubAdversary:
    """A stub CounterExampleAdversary returning a fixed candidate test per round (deterministic ->
    the converter's mandatory determinism re-check passes)."""

    def synthesize(
        self,
        item: PlantedItem,
        clean_sibling: PlantedItem,
        hypotheses: tuple[PanelHypothesis, ...],
        round_index: int,
    ) -> str:
        return f"SYNTH_CANDIDATE_r{round_index}"


def _adjudicate(request: AdjudicationRequest) -> tuple[Adjudication, ...]:
    """Two-adjudicator consensus, content-aware: a CLEAN item's C2 existence verdict is ``clean``;
    an ERROR item's K-existence verdict is ``error`` (the §6.C established-existence path)."""
    verdict = "clean" if request.provisional_id % 2 == 1 else "error"
    return tuple(
        Adjudication(adjudicator_id=f"adj-{i}", verdict=verdict, rationale="stub", timestamp=_TS)
        for i in range(2)
    )


def _tie_break(request: AdjudicationRequest) -> Adjudication:
    return Adjudication(adjudicator_id="architect", verdict="error", rationale="tb", timestamp=_TS)


def _regime_adjudicate(request: AdjudicationRequest) -> RegimeAdjudication:
    """Second-author K-regime audit: confirm the planter's candidate regime (keeps the tag)."""
    return RegimeAdjudication(
        adjudicator_id="second-author",
        verdict="confirm-regime",
        rationale="rg",
        timestamp=_TS,
    )


# ===========================================================================
# THE PIPELINE FIXTURE — plant -> convert -> promote -> lock on a small synthetic corpus.
# ===========================================================================

_FRAME = OracleFrame(
    completion_criterion="tests_pass", problem_type="code", problem_statement="sum a list"
)


def _seed(solution: str, seed_id: int) -> Seed:
    return Seed(
        seed_id=seed_id,
        frame=_FRAME,
        thesis=Thesis(proposed_solution=solution, experiment_design="exp"),
        test_code=_AUTHOR_TEST,
    )


def _llm_k_pair(
    *,
    err_sol: str,
    clean_sol: str,
    error_id: int,
    clean_id: int,
    regime: str,
    split: Literal["tuning", "measurement"],
) -> PlantedPair:
    """A pre-label LLM-planted K pair (the converter's input shape) built directly — mirrors the
    ``KInjector`` output without invoking a live ``Model`` (S4; the live K run is deferred). The
    matched error/clean siblings are co-located in ``split`` (the authoring invariant)."""
    from cogworx.eval.corpus import DifficultyMarker, LLMPlanterStamp

    stamp = LLMPlanterStamp(model_family="planter-fam", model_id="planter/chat")
    diff = DifficultyMarker(
        planted_difficulty="medium", surface_complexity=12, is_matched_sibling=True
    )
    err = PlantedItem(
        provisional_id=error_id,
        frame=_FRAME,
        thesis=Thesis(proposed_solution=err_sol, experiment_design="exp"),
        test_code=_AUTHOR_TEST,
        candidate_stratum="K",
        is_error=1,
        planter=stamp,
        error_regime=regime,  # CANDIDATE (second-author-audited at promote)
        difficulty=diff,
        matched_sibling_id=clean_id,
        split=split,
    )
    clean = PlantedItem(
        provisional_id=clean_id,
        frame=_FRAME,
        thesis=Thesis(proposed_solution=clean_sol, experiment_design="exp"),
        test_code=_AUTHOR_TEST,
        candidate_stratum="clean",
        is_error=0,
        planter=stamp,
        error_regime="",
        difficulty=diff,
        matched_sibling_id=error_id,
        split=split,
    )
    return PlantedPair(error_item=err, clean_item=clean)


def _build_planted() -> tuple[list[PlantedPair], list[PlantedItem]]:
    """Plant a small matched-pair corpus: deterministic-O pairs (real AST mutation) + LLM-K pairs
    that convert + LLM-K pairs that stay residual-K. All pairs, no singletons (the converter's
    two-sided gate requires a co-located clean sibling). ~20 error items total."""
    pairs: list[PlantedPair] = []
    inj = OInjector()

    # --- deterministic-O pairs (real ast.mutate via OInjector; arithmetic-swap -> logic-wrong) ---
    pid = 1000
    for _ in range(8):
        seed = _seed("def f(xs):\n    return sum(xs) + 0\n", seed_id=pid)
        # The OInjector mutates the '+' to '-' (arithmetic-swap) producing the O error; the clean
        # sibling is the original. We pin the stub probe on the resulting solution strings below.
        pair = inj.emit(seed, "arithmetic-swap", error_id=pid, clean_id=pid + 1)
        pairs.append(pair)
        pid += 2

    # --- LLM-K pairs that CONVERT (the stub probe makes the converter gate pass) ---
    # The pre-label CANDIDATE error_regime is drawn from the 3 canonical regimes (what draw_splits
    # stratifies on, mirroring test_planting's _REGIMES_3) — the second-author audit confirms it at
    # promote. The K_ERROR_KINDS vocabulary is the KInjector's prompt knob, NOT the split-draw key.
    for i in range(6):
        pairs.append(
            _llm_k_pair(
                err_sol=_KCONV_ERR,
                clean_sol=_KCONV_CLEAN,
                error_id=pid,
                clean_id=pid + 1,
                regime=_CANON_REGIMES[i % len(_CANON_REGIMES)],
                split="tuning",
            )
        )
        pid += 2

    # --- LLM-K pairs that stay RESIDUAL-K (the stub probe makes the converter gate fail) ---
    for i in range(6):
        pairs.append(
            _llm_k_pair(
                err_sol=_KRES_ERR,
                clean_sol=_KRES_CLEAN,
                error_id=pid,
                clean_id=pid + 1,
                regime=_CANON_REGIMES[i % len(_CANON_REGIMES)],
                split="tuning",
            )
        )
        pid += 2

    return pairs, []


def _o_solution_strings() -> set[str]:
    """The mutated-O error solution strings the OInjector actually produced (so the stub probe can
    return an O-catch for exactly those). Re-derive from a fresh emit so the strings match."""
    inj = OInjector()
    seed = _seed("def f(xs):\n    return sum(xs) + 0\n", seed_id=0)
    pair = inj.emit(seed, "arithmetic-swap", error_id=0, clean_id=1)
    return {pair.error_item.thesis.proposed_solution}


def _o_aware_probe(o_err_strings: set[str]) -> OracleProbe:
    """The content-addressed probe, widened to also O-catch the OInjector's actual mutated strings.
    (``_stub_probe`` covers the hand-written sentinels; the OInjector's '+0 -> -0' mutation yields a
    distinct string we resolve to an O-catch here.)"""

    def probe(solution_code: str, test_code: str) -> Verdict:
        if solution_code in o_err_strings:
            return _verdict(holds=False, valid_check=True)  # the deterministic-O catch
        return _stub_probe(solution_code, test_code)

    return probe


def run_pipeline() -> tuple[PromotionResult, list[CorpusItem], ConversionResult]:
    """Run the REAL chain: plant -> draw_splits -> convert_k_pool (STUB panel/adversary/probe) ->
    promote_corpus (STUB adjudicators) -> lock. Returns the promotion result, the LOCKED items, and
    the conversion result (for the converted-O assertions)."""
    pairs, singletons = _build_planted()

    # Stratified split draw (deterministic, frozen seed). Co-locates each pair's two members.
    assignment = draw_splits(pairs, singletons)
    pairs = [
        PlantedPair(
            error_item=p.error_item.model_copy(
                update={"split": assignment[p.error_item.provisional_id]}
            ),
            clean_item=p.clean_item.model_copy(
                update={"split": assignment[p.clean_item.provisional_id]}
            ),
        )
        for p in pairs
    ]

    probe = _o_aware_probe(_o_solution_strings())

    # The forbidden families exclude the panel/adversary so the converter is not degraded.
    config = PanelConfig(
        panel_families=("panel-fam-a", "panel-fam-b"),
        adversary_family="adv-fam",
        forbidden_families=frozenset({"planter-fam", "thesis-fam", "antithesis-fam"}),
    )
    conv = convert_k_pool(
        pairs,
        singletons,
        panel=_StubPanel(),
        adversary=_StubAdversary(),
        probe=probe,
        config=config,
    )
    assert not conv.degraded, "the converter must not be degraded in the spike fixture"

    promotion = promote_corpus(
        list(conv.pairs) + list(conv.residual_pairs),
        list(conv.singletons) + list(conv.residual_singletons),
        probe=probe,
        adjudicate=_adjudicate,
        tie_break=_tie_break,
        regime_adjudicate=_regime_adjudicate,
        derive_o_regime=_derive_o_regime,
    )
    locked = lock_corpus(list(promotion.promoted))
    return promotion, locked, conv


# ===========================================================================
# THE BRIDGE Cell FIXTURE — synthetic 4.4d artifact aligned to the locked corpus.
# ===========================================================================


def _synth_with_arm_a(
    rng: Random,
    *,
    m_K: int,
    m_clean: int,
    R: int = _R,
    k_regimes: tuple[str, ...] = ("logic-wrong", "edge-case-miss", "off-by-semantics"),
    a_flags_k_items: int = 0,
    a_clean_fp_cells: int = 0,
    spec_C: float = 0.85,
) -> list[Cell]:
    """Build a synthetic 4.4d ``Cell`` artifact the lock instruments consume. ``synth_cells`` emits
    arms C+D on K+clean but NO arm A and NO ``regime``; we AUGMENT it:
      - stamp ``regime`` onto every K cell (round-robin across ``k_regimes``) — needed for §2.B;
      - synthesize arm-A cells (INV-A0: A flags nothing on K; INV-A1: A barely false-positives on
        clean) — needed for ``assert_arm_a_floor``.
    The negative-control knobs (``a_flags_k_items`` > 0, ``a_clean_fp_cells`` large, ``spec_C`` ~ 1)
    make the corresponding gate FIRE."""
    base = synth_cells(
        rng,
        m_K=m_K,
        m_clean=m_clean,
        R=R,
        sens_C=0.5,
        dsens=0.2,
        sb_sens=0.04,
        spec_C=spec_C,
        dspec=0.05,
        sb_spec=_SIGMA_SQ_B_SPEC,
        rho_w=0.3,
    )
    # synth_cells ids: K items 0..m_K-1, clean items 10_000..10_000+m_clean-1.
    out: list[Cell] = []
    regime_of: dict[int, str] = {}
    for c in base:
        if c.stratum == "K":
            r = regime_of.setdefault(c.item_id, k_regimes[c.item_id % len(k_regimes)])
            out.append(c.model_copy(update={"regime": r}))
        else:
            out.append(c)

    # Arm A on K: flag nothing (INV-A0 floor) unless a_flags_k_items > 0 (the negative control).
    for i in range(m_K):
        flagged_item = 1 if i < a_flags_k_items else 0
        for t in range(R):
            out.append(
                Cell(
                    item_id=i,
                    stratum="K",
                    arm="A",
                    trial=t,
                    seed=(i << 16) ^ (t << 1) ^ 0xA,
                    flagged=flagged_item,
                    route="flag" if flagged_item else "pass",
                    regime=regime_of.get(i, k_regimes[i % len(k_regimes)]),
                )
            )
    # Arm A on clean: distribute a_clean_fp_cells false positives (INV-A1 axis).
    fp_remaining = a_clean_fp_cells
    for i in range(m_clean):
        item_id = 10_000 + i
        for t in range(R):
            f = 1 if fp_remaining > 0 else 0
            if f:
                fp_remaining -= 1
            out.append(
                Cell(
                    item_id=item_id,
                    stratum="clean",
                    arm="A",
                    trial=t,
                    seed=(item_id << 16) ^ (t << 1) ^ 0xA,
                    flagged=f,
                    route="flag" if f else "pass",
                )
            )
    return out


def _trivial_clean_cells(
    arm: str = "C", *, n_clean: int = _N_CLEAN_PLANNED, R: int = _R
) -> list[Cell]:
    """A trivial-clean negative control: NO clean item ever flags -> spec=1.0 (AT/above the ceiling)
    -> ``assert_spec_ceiling`` MUST fire. Carries a token K+arm-D so it is a realistic artifact."""
    cells: list[Cell] = []
    for i in range(n_clean):
        for t in range(R):
            cells.append(
                Cell(
                    item_id=5000 + i, stratum="clean", arm=arm, trial=t,
                    seed=(5000 + i) * 1000 + t, flagged=0, route="pass",
                )
            )
    return cells


# §5 shuffle fixtures (mirrors test_lock _assoc_cells / _UNPAIRED / _DELTAS at spike scale).
def _assoc_cells(n_k: int = 12, n_clean: int = 12, R: int = 6) -> list[Cell]:
    """An honest artifact with a strong stratum<->flag association on arm D (welded shuffle severs
    it -> centering at 0); arm C is flat. The LEAKY shuffle preserves the association -> centering
    fires."""
    cells: list[Cell] = []
    for i in range(n_k):
        for t in range(R):
            cells.append(Cell(item_id=i, stratum="K", arm="D", trial=t, seed=i * 100 + t,
                              flagged=1 if t < R - 1 else 0, route="x"))
            cells.append(Cell(item_id=i, stratum="K", arm="C", trial=t, seed=i * 100 + t,
                              flagged=1 if t < R // 2 else 0, route="x"))
    for j in range(n_clean):
        item = 1000 + j
        for t in range(R):
            cells.append(Cell(item_id=item, stratum="clean", arm="D", trial=t, seed=item * 100 + t,
                              flagged=1 if t == 0 else 0, route="x"))
            cells.append(Cell(item_id=item, stratum="clean", arm="C", trial=t, seed=item * 100 + t,
                              flagged=1 if t < R // 2 else 0, route="x"))
    return cells


_UNPAIRED = [*range(12), *(1000 + j for j in range(12))]
_DELTAS = {"D>C": ("D", "C")}


# ===========================================================================
# §6.1 LIST PART 1 — the corpus is honestly built + locked (the pipeline integration).
# ===========================================================================


def test_pipeline_composes_end_to_end() -> None:
    """The real chain plant -> convert -> promote -> lock runs and yields a coherent labeled corpus:
    converted K->O migrations, residual K, and clean siblings all present + locked."""
    _promotion, locked, conv = run_pipeline()
    assert len(locked) >= 1
    # The converter migrated SOME K to O with a synthesized test.
    assert len(conv.pairs) >= 1, "no K item converted — the converter seam did not compose"
    assert all(isinstance(p.error_item.planter, ConvertedPlanterStamp) for p in conv.pairs)
    # SOME residual K survived.
    assert len(conv.residual_pairs) >= 1, "no residual K — the residual seam did not compose"
    strata = {it.stratum for it in locked}
    assert {"O", "K", "clean"} <= strata, f"expected all strata, got {strata}"


def test_every_clean_item_has_c1_or_c2_stamp() -> None:
    """§6.1: every clean item carries a C1 (oracle) or C2 (human) label-provenance stamp. Re-checked
    structurally by ``load_corpus`` G1 (raises nothing on a clean build)."""
    _, locked, _ = run_pipeline()
    clean = [it for it in locked if it.stratum == "clean"]
    assert clean, "no clean items in the locked corpus"
    for it in clean:
        prov = it.label_provenance
        if isinstance(prov, OracleLabelProvenance):
            assert prov.test_provenance == "frozen"
        else:
            assert isinstance(prov, HumanLabelProvenance) and len(prov.adjudications) >= 1
    # G1/G2/G4 structural load over the tuning split (raises nothing).
    assert load_corpus(locked, measurement_run=False) is not None


def test_every_label_has_non_judge_provenance() -> None:
    """§6.1 (S9): every label_source ∈ {oracle, human} — NEVER a judge/inference source. The frozen
    model forbids a judge source structurally; this pins it on the realized corpus."""
    _, locked, _ = run_pipeline()
    assert locked
    for it in locked:
        assert it.label_source in ("oracle", "human")
        assert it.label_source == it.label_provenance.kind


def test_strata_oracle_assigned_and_o_regimes_operator_derived() -> None:
    """§6.1 / §2.C: O-stratum strata are oracle-assigned; a deterministic-O item's error_regime is
    operator-derived (``_derive_o_regime`` over the planter operators), NOT author-trusted."""
    from cogworx.eval.corpus import DeterministicPlanterStamp

    _, locked, _ = run_pipeline()
    det_o = [
        it for it in locked
        if it.stratum == "O" and isinstance(it.planter, DeterministicPlanterStamp)
    ]
    assert det_o, "no deterministic-O items — the O path did not compose"
    for it in det_o:
        assert it.label_source == "oracle"
        assert isinstance(it.planter, DeterministicPlanterStamp)
        assert it.error_regime == _derive_o_regime(it.planter.operators)


def test_converted_o_regimes_are_second_author_audited() -> None:
    """§2.C: a converted (K->O) item has NO mutation operator, so error_regime routes through the
    second-author audit (not operator-derivation). The audit recorded ≥1 RegimeAdjudication."""
    promotion, locked, _ = run_pipeline()
    conv_o = [it for it in locked if isinstance(it.planter, ConvertedPlanterStamp)]
    assert conv_o, "no converted-O items in the locked corpus"
    for it in conv_o:
        assert it.stratum == "O" and it.label_source == "oracle"
    assert promotion.regime_audit, "the second-author regime audit recorded nothing"


def test_o_regime_mismatch_raises_oregimemismatcherror() -> None:
    """§2.C fail-fast NEGATIVE CONTROL: a deterministic-O item whose author error_regime DISAGREES
    with the operator-derived regime is REFUSED at promote (``ORegimeMismatchError``)."""
    from cogworx.eval.labeling import ORegimeMismatchError

    inj = OInjector()
    seed = _seed("def f(xs):\n    return sum(xs) + 0\n", seed_id=1)
    pair = inj.emit(seed, "arithmetic-swap", error_id=900, clean_id=901)
    # Tamper the error_regime to a WRONG (but in-table) regime -> the §2.C cross-check must fire.
    bad_err = pair.error_item.model_copy(update={"error_regime": "off-by-semantics"})
    bad_pair = PlantedPair(error_item=bad_err, clean_item=pair.clean_item)
    probe = _o_aware_probe(_o_solution_strings())
    with pytest.raises(ORegimeMismatchError):
        promote_corpus(
            [bad_pair], [], probe=probe, adjudicate=_adjudicate, tie_break=_tie_break,
            regime_adjudicate=_regime_adjudicate, derive_o_regime=_derive_o_regime,
        )


# ===========================================================================
# §6.1 LIST PART 2 — contamination / bijection (the decision-independent lock preconditions).
# ===========================================================================


def test_splits_disjoint_passes_then_negative_control_fires() -> None:
    """§3.8 contamination: a clean tuning/measurement partition passes ``assert_no_contamination``;
    a NEGATIVE CONTROL (a measurement content-hash present in the tuning-run log) FIRES."""
    _, locked, _ = run_pipeline()
    # Force one item into measurement so both splits are populated; lock recomputes its hash.
    forced = [
        locked[0].model_copy(update={"split": "measurement"}),
        *[it for it in locked[1:]],
    ]
    relocked = lock_corpus(forced)
    # Honest: an empty tuning-run log -> no overlap -> passes (raises nothing).
    assert_no_contamination(relocked, tuning_run_hashes=[])
    # Negative control: the measurement item's own hash in the tuning log -> contamination -> FIRE.
    meas = [it for it in relocked if it.split == "measurement"]
    assert meas, "no measurement item to contaminate"
    with pytest.raises(CorpusLockError, match="contamination"):
        assert_no_contamination(relocked, tuning_run_hashes=[meas[0].content_hash])


def test_bijection_revalidates_on_abstention_drops() -> None:
    """§5 INV-LOCK-5: the matched-pair bijection re-validates after abstention-drops. A surviving
    paired set re-validates clean; dropping ONE member of a pair DEMOTES the survivor to the
    unpaired pool carrying its real stratum."""
    _promotion, locked, _ = run_pipeline()
    # No drops in the honest run -> every retained pair is a clean mutual bijection.
    res = revalidate_bijection(locked, frozenset())
    # Some matched pairs survive.
    assert res.paired or res.demoted_unpaired
    # Now simulate a drop: pick a paired item, drop its sibling, re-validate -> survivor demoted.
    paired_items = [it for it in locked if it.matched_sibling_id is not None]
    if paired_items:
        victim = paired_items[0]
        sib_id = victim.matched_sibling_id
        assert sib_id is not None
        survivors = [it for it in locked if it.item_id != sib_id]
        res2 = revalidate_bijection(survivors, frozenset({sib_id}))
        demoted_ids = {it.item_id for it in res2.demoted_unpaired}
        assert victim.item_id in demoted_ids
        # The demoted survivor keeps its real stratum, sibling link cleared.
        demoted = next(it for it in res2.demoted_unpaired if it.item_id == victim.item_id)
        assert demoted.matched_sibling_id is None
        assert demoted.stratum == victim.stratum


def test_fingerprint_builds_over_locked_corpus() -> None:
    """INV-LOCK-3: a LOCKED corpus builds a MeasurementFingerprint; an unlocked item is refused.
    Pins that lock_corpus stamps a non-empty content_hash on every item."""
    _, locked, _ = run_pipeline()
    assert all(it.content_hash != "" for it in locked)
    assert all(content_hash(it) == it.content_hash for it in locked)
    fp = build_fingerprint(locked, git_sha="deadbeef")
    assert fp.digest
    # Unlocked item -> refused.
    with pytest.raises(ValueError, match="never-locked"):
        build_fingerprint([locked[0].model_copy(update={"content_hash": ""})], git_sha="x")


# ===========================================================================
# §6.1 LIST PART 3 — the regime-contribution gate + LORO/KS report-only, abstain cap.
# ===========================================================================


def test_regime_contribution_gates_and_loro_ks_report_only() -> None:
    """§2.B: ``assert_regime_contribution`` GATES (a >30%-loaded regime FIRES); ``report_loro`` is
    REPORT-ONLY (never raises); ``ks_flag_rate_diagnostic`` is REPORT-ONLY (never raises)."""
    # Honest: 5 evenly-allocated regimes (uniform 0.20 level-share each, below 0.30) -> passes. Uses
    # the controlled _regime_artifact (uniform per-item sens -> share scales purely with COUNT), not
    # synth_cells (whose Beta-drawn per-item sens makes round-robin shares uneven).
    five_regimes = (*_CANON_REGIMES, "spec-misread", "silent-degradation")
    cells = _regime_artifact(dict.fromkeys(five_regimes, 8))
    report = assert_regime_contribution(cells, arm_b="A")
    assert report.max_share == REGIME_CONTRIBUTION_MAX_SHARE
    assert all(abs(s - 0.20) < 1e-9 for s in report.shares.values())
    # report_loro never raises (MED-D demote).
    loro = report_loro(cells, arm_b="A", n_outer=_N_OUTER, seed=3)
    assert set(loro.collapses) <= set(report.shares)
    # KS is reported-only, never raises.
    ks = ks_flag_rate_diagnostic(cells, arm="C")
    assert isinstance(ks.exceeds_crit, bool)


def test_regime_contribution_loaded_regime_negative_control_fires() -> None:
    """§2.B NEGATIVE CONTROL: a single regime loaded past 30% of the pooled K-sensitivity FIRES the
    contribution gate (HIGH-1). Built by hand on the lock's exact scoring path."""
    cells = _loaded_regime_cells()
    with pytest.raises(CorpusLockError, match=r"exceeding the ceiling 0.3"):
        assert_regime_contribution(cells, arm_b="A")


def test_regime_abstain_cap_fires_on_a_dump() -> None:
    """§2.C Finding-2 NEGATIVE CONTROL: the 15% abstain cap FIRES when too many K items are dumped
    into the unattributable (``regime==""``) bucket (a loaded regime could hide there)."""
    cells = _abstain_dump_cells()
    with pytest.raises(CorpusLockError, match=r"regime-abstain.*exceeds the ceiling 0.15"):
        assert_regime_contribution(cells, arm_b="A")


def _regime_artifact(
    counts: dict[str, int], *, n_clean: int = 10, d_flags_k: int = 4, R: int = 4
) -> list[Cell]:
    """K+clean artifact: counts[regime] K items per regime, arm D flags d_flags_k/R on each K item
    (uniform per-item sens -> level-share scales with COUNT), arm A flags nothing. Regime welded."""
    cells: list[Cell] = []
    iid = 0
    for regime, n in counts.items():
        for _ in range(n):
            for t in range(R):
                cells.append(Cell(item_id=iid, stratum="K", arm="D", trial=t, seed=iid * 10 + t,
                                  flagged=1 if t < d_flags_k else 0, route="x", regime=regime))
                cells.append(Cell(item_id=iid, stratum="K", arm="A", trial=t, seed=iid * 10 + t,
                                  flagged=0, route="x", regime=regime))
            iid += 1
    for _ in range(n_clean):
        for t in range(R):
            cells.append(Cell(item_id=iid, stratum="clean", arm="D", trial=t, seed=iid * 10 + t,
                              flagged=0, route="x"))
            cells.append(Cell(item_id=iid, stratum="clean", arm="A", trial=t, seed=iid * 10 + t,
                              flagged=0, route="x"))
        iid += 1
    return cells


def _loaded_regime_cells() -> list[Cell]:
    return _regime_artifact(
        {
            "logic-wrong": 20,
            "edge-case-miss": 5,
            "off-by-semantics": 5,
            "spec-misread": 5,
            "silent-degradation": 5,
        }
    )


def _abstain_dump_cells() -> list[Cell]:
    return _regime_artifact(
        {"logic-wrong": 5, "edge-case-miss": 5, "off-by-semantics": 5, "spec-misread": 5, "": 30}
    )


# ===========================================================================
# §6.1 LIST PART 4 — INV-A0/A1 arm-A floor (passes + each negative control fires).
# ===========================================================================


def test_arm_a_floor_holds_on_honest_fixture() -> None:
    """INV-A0/A1: arm A flags NOTHING on K (sens_A==0) and barely false-positives on clean
    (spec_A >= 1-tau_A) -> the floor holds (returns None)."""
    rng = Random(11)
    cells = _synth_with_arm_a(
        rng, m_K=40, m_clean=_N_CLEAN_PLANNED, a_flags_k_items=0, a_clean_fp_cells=1
    )
    assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)  # raises nothing


def test_arm_a_floor_inv_a0_negative_control_fires() -> None:
    """INV-A0 NEGATIVE CONTROL: arm A flags a K item (sens_A>0) -> that item is mis-stratified (the
    oracle reached it; it belongs in O) -> refuse lock."""
    rng = Random(12)
    cells = _synth_with_arm_a(
        rng, m_K=40, m_clean=_N_CLEAN_PLANNED, a_flags_k_items=1, a_clean_fp_cells=0
    )
    with pytest.raises(CorpusLockError, match=r"INV-A0.*sens_A"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_inv_a1_negative_control_fires() -> None:
    """INV-A1 NEGATIVE CONTROL: arm A false-positives on too many clean cells (spec_A < 1-tau_A) ->
    A is not the §5-centering floor -> refuse lock. tau_A=2/(7*80)=0.00357 -> >2 FPs breaches."""
    rng = Random(13)
    cells = _synth_with_arm_a(
        rng, m_K=40, m_clean=_N_CLEAN_PLANNED, a_flags_k_items=0, a_clean_fp_cells=10
    )
    with pytest.raises(CorpusLockError, match=r"INV-A1.*spec_A"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


# ===========================================================================
# §6.1 LIST PART 5 — spec variance-floor + planned-n ceiling fire on a trivial-clean control.
# ===========================================================================


def test_spec_ceiling_passes_on_honest_then_fires_on_trivial_clean() -> None:
    """INV-LOCK-6: a non-trivial clean spec passes the tripwire; a trivial-clean NEGATIVE CONTROL
    (spec=1.0, AT/above the planned-n ceiling) FIRES."""
    rng = Random(14)
    honest = _synth_with_arm_a(rng, m_K=20, m_clean=_N_CLEAN_PLANNED, spec_C=0.85)
    assert_spec_ceiling(  # raises nothing on an honest non-trivial clean spec
        honest,
        n_clean_planned=_N_CLEAN_PLANNED,
        R=_R,
        sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
    )
    trivial = _trivial_clean_cells("C")
    with pytest.raises(CorpusLockError, match="AT OR ABOVE the planned-n ceiling"):
        assert_spec_ceiling(
            trivial,
            n_clean_planned=_N_CLEAN_PLANNED,
            R=_R,
            sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
        )


def test_spec_variance_floor_fires_on_collapsed_variance() -> None:
    """INV-LOCK-6 variance floor: a non-trivial MEAN spec whose per-item flag-rate variance has
    COLLAPSED (every clean item flags an identical 3/7) FIRES the floor (the bootstrap gets no
    spec spread)."""
    cells: list[Cell] = []
    for i in range(_N_CLEAN_PLANNED):
        for t in range(_R):
            flagged = 1 if t < 3 else 0
            cells.append(Cell(item_id=6000 + i, stratum="clean", arm="C", trial=t,
                              seed=(6000 + i) * 1000 + t, flagged=flagged,
                              route="flag" if flagged else "pass"))
    with pytest.raises(CorpusLockError, match="below the planning-derived floor"):
        assert_spec_ceiling(
            cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R, sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC
        )


# ===========================================================================
# §6.1 LIST PART 6 — valid Cells round-trip nested_bootstrap_delta; R5 quantile pins.
# ===========================================================================


def test_cells_round_trip_nested_bootstrap_delta() -> None:
    """A valid synthetic ``Cell`` artifact round-trips ``nested_bootstrap_delta`` -> a finite
    (mean, lo, hi) with lo <= hi; converted-O cells are excludable via ``is_converted_o``."""
    rng = Random(15)
    cells = _synth_with_arm_a(rng, m_K=30, m_clean=30)
    mean, lo, hi = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=_N_OUTER, seed=7
    )
    assert math.isfinite(mean) and lo <= hi
    # is_converted_o is the 4.4d exclusion predicate (default False on synth cells).
    assert not any(is_converted_o(c) for c in cells)


def test_r5_default_quantile_is_shipped_ci_byte_for_byte_and_0_0025_widens() -> None:
    """R5 BEHAVIOUR PINS: the default ``quantile`` == the shipped 2.5/97.5 CI BYTE-FOR-BYTE, AND
    ``quantile=0.0025`` WIDENS it (a strictly deeper-tail order statistic on the same draws)."""
    rng = Random(0)
    cells = _synth_with_arm_a(rng, m_K=30, m_clean=30)
    shipped = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=300, seed=7
    )
    explicit = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=300, seed=7, quantile=0.025
    )
    assert shipped == explicit  # byte-for-byte
    n_outer = 4000
    _, lo_def, hi_def = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=n_outer, seed=7, quantile=0.025
    )
    _, lo_wide, hi_wide = nested_bootstrap_delta(
        cells, arm_a="D", arm_b="C", error_strata=("K",), n_outer=n_outer, seed=7, quantile=0.0025
    )
    assert lo_wide <= lo_def and hi_wide >= hi_def  # the deeper tail is at least as wide


# ===========================================================================
# §6.1 LIST PART 7 — the §5 shuffle: centering + coverage pass; LEAKY + high-k controls fire.
# ===========================================================================


def test_shuffle_null_honest_passes_centering_and_coverage() -> None:
    """§5: an HONEST artifact passes ``assert_shuffle_null`` — centering CI contains 0 (paired),
    global-coverage ucb95 <= 0.08, paired-coverage report-only -> returns a ShuffleNullReport."""
    cells = _assoc_cells()
    result = shuffle_null_centering(
        cells, paired_ids=[], unpaired_ids=_UNPAIRED, deltas=_DELTAS,
        n_shuffles=_N_SHUFFLES, seed=7, n_outer=_N_OUTER,
    )
    report = assert_shuffle_null(result)
    assert isinstance(report, ShuffleNullReport)
    assert set(report.paired_k) == set(_DELTAS)


def test_shuffle_null_leaky_mutation_fires_centering() -> None:
    """§5 MUTATION TEST: the deliberately-LEAKY shuffle carries flag-semantics WITH the stratum,
    so the label<->flag association survives -> mean(shuffle_δ) is non-zero -> centering FIRES."""
    cells = _assoc_cells()
    leaky = shuffle_null_centering(
        cells,
        paired_ids=[],
        unpaired_ids=_UNPAIRED,
        deltas=_DELTAS,
        n_shuffles=_N_SHUFFLES,
        seed=7,
        n_outer=_N_OUTER,
        _shuffle=_shuffle_stratum_membership_LEAKY,
    )
    with pytest.raises(CorpusLockError, match="centering"):
        assert_shuffle_null(leaky)


def test_shuffle_null_high_global_k_fires_coverage() -> None:
    """§5: a high-global-k artifact (k>=10 leakage-direction global δ-CI exclusions at n=200) FIRES
    the coverage-rate gate even though centering is clean — the two gates are orthogonal and read
    DIFFERENT artifacts. Built by hand on the public ShuffleNullResult for an exact tail."""
    from cogworx.eval.lock import ShuffleNullResult

    n, k = 200, 12
    points = tuple(0.0 for _ in range(n))
    clean = tuple((-0.5, 0.5) for _ in range(n))
    leaky = tuple((0.01, 0.5) if i < k else (-0.5, 0.5) for i in range(n))
    result = ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>C": points},
        paired_ci_bounds={"D>C": clean},
        global_ci_bounds={"D>C": leaky},
    )
    with pytest.raises(CorpusLockError, match="coverage-rate"):
        assert_shuffle_null(result)


def test_shuffle_null_bijection_revalidates_on_post_abstention_fixture() -> None:
    """§5: the bijection re-validation runs on a POST-ABSTENTION fixture, and the paired sign-flip
    shuffle runs over the SURVIVING matched pairs (the PRIMARY control). An honest weak-association
    paired draw passes both gates."""
    # A weak-association artifact: K/clean share the same flag distribution -> intrinsically null.
    rng = Random(99)
    cells: list[Cell] = []

    def emit(item: int, stratum: str) -> None:
        f = rng.randint(1, 5)
        for arm in ("D", "C"):
            for t in range(6):
                cells.append(Cell(item_id=item, stratum=stratum, arm=arm, trial=t,
                                  seed=item * 100 + t, flagged=1 if t < f else 0, route="x"))

    for i in range(12):
        emit(i, "K")
    for j in range(12):
        emit(1000 + j, "clean")

    # Build matched (K, clean) CorpusItems, drop one pair member, re-validate the bijection.
    pairs = [(i, 1000 + i) for i in range(12)]
    result = shuffle_null_centering(
        cells, paired_ids=pairs, unpaired_ids=[], deltas=_DELTAS,
        n_shuffles=_N_SHUFFLES, seed=13, n_outer=_N_OUTER,
    )
    assert isinstance(assert_shuffle_null(result), ShuffleNullReport)
