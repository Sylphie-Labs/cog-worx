"""Deterministic tests for the corpus-lock fingerprint spine (Pod 4.4c-5a).

Mutation-resistant pins for INV-LOCK-1 (content hash), INV-LOCK-2 (git-SHA pin), and INV-LOCK-3
(the MeasurementFingerprint). Each sensitivity assertion is paired with a determinism control so a
hash/fingerprint that silently ignored an input is CAUGHT. The exec-env-identity test closes
CF-4.4c-CONVERTER-ENV-PIN (two env identities -> two fingerprints). No model calls, no substrate, no
git shell-out (the SHA + env are injected).
"""

from __future__ import annotations

import hashlib
from random import Random

import pytest

from cogworx.eval.corpus import (
    ConvertedPlanterStamp,
    CorpusItem,
    DeterministicPlanterStamp,
    DifficultyMarker,
    LLMPlanterStamp,
    OracleLabelProvenance,
)
from cogworx.eval.lock import (
    MASTER_SEED,
    REGIME_CONTRIBUTION_MAX_SHARE,
    RESIDUAL_EPSILON_UNAUDITED,
    CorpusLockError,
    ExecEnvIdentity,
    MeasurementFingerprint,
    ShuffleNullReport,
    ShuffleNullResult,
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
    shuffle_stratum_membership,
)
from cogworx.eval.youden import Cell, synth_cells
from cogworx.verification.contracts import OracleFrame, Thesis

_FRAME = OracleFrame(
    completion_criterion="tests_pass",
    problem_type="code",
    problem_statement="sum a list",
)
_THESIS = Thesis(proposed_solution="return sum(xs)", experiment_design="run frozen tests")


def _prov() -> OracleLabelProvenance:
    return OracleLabelProvenance(
        returncode=1,
        test_provenance="frozen",
        holds=False,
        valid_check=True,
        oracle_id="code-oracle",
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
        "label_provenance": _prov(),
        "stratum": "O",
        "oracle_reachable": True,
        "error_regime": "wrong-op",
        "difficulty": _difficulty(),
        "matched_sibling_id": None,
        "split": "measurement",
        "planter": DeterministicPlanterStamp(operators=("swap-op",)),
    }
    base.update(overrides)
    return CorpusItem(**base)


def _env(**overrides: object) -> ExecEnvIdentity:
    base: dict[str, object] = {
        "python_version": "3.12.7",
        "python_implementation": "CPython",
        "package_versions": (("pydantic", "2.12.5"), ("pytest", "8.0.0")),
        "locale_lc_ctype": "C",
    }
    base.update(overrides)
    return ExecEnvIdentity(**base)


# ---------------------------------------------------------------------------
# INV-LOCK-1 — content hash: determinism + per-field sensitivity + the mutation test
# ---------------------------------------------------------------------------


def test_content_hash_is_deterministic() -> None:
    """Same item -> byte-identical hash (the control every sensitivity pin is measured against)."""
    assert content_hash(_item()) == content_hash(_item())


def test_content_hash_is_a_sha256_hexdigest() -> None:
    h = content_hash(_item())
    assert len(h) == 64
    assert all(c in "0123456789abcdef" for c in h)


def test_content_hash_sensitive_to_frame() -> None:
    other = OracleFrame(
        completion_criterion="tests_pass",
        problem_type="code",
        problem_statement="MUTATED statement",
    )
    assert content_hash(_item()) != content_hash(_item(frame=other))


def test_content_hash_sensitive_to_thesis() -> None:
    other = Thesis(proposed_solution="return 0", experiment_design="run frozen tests")
    assert content_hash(_item()) != content_hash(_item(thesis=other))


def test_content_hash_sensitive_to_test_code() -> None:
    assert content_hash(_item()) != content_hash(_item(test_code="assert f([1, 2]) == 4"))


def test_content_hash_sensitive_to_is_error() -> None:
    assert content_hash(_item(is_error=1)) != content_hash(_item(is_error=0, stratum="clean"))


def test_content_hash_sensitive_to_stratum() -> None:
    assert content_hash(_item(stratum="O")) != content_hash(_item(stratum="K"))


def test_content_hash_sensitive_to_error_regime() -> None:
    assert content_hash(_item(error_regime="wrong-op")) != content_hash(
        _item(error_regime="off-by-one")
    )


def test_content_hash_sensitive_to_split() -> None:
    assert content_hash(_item(split="measurement")) != content_hash(_item(split="tuning"))


def test_content_hash_ignores_non_identity_fields() -> None:
    """The hash is over the SEVEN identity fields ONLY: changing a non-identity field (item_id,
    oracle_reachable, difficulty, sibling, label provenance, planter, content_hash itself) MUST NOT
    move the hash. This pins the field set's UPPER bound — a hash that folded in extra fields would
    over-trip and re-lock spuriously."""
    base = content_hash(_item())
    assert content_hash(_item(item_id=999)) == base
    assert content_hash(_item(oracle_reachable=False)) == base
    assert (
        content_hash(
            _item(
                matched_sibling_id=None,
                difficulty=DifficultyMarker(planted_difficulty="hard", surface_complexity=99),
            )
        )
        == base
    )
    assert content_hash(_item(planter=LLMPlanterStamp(model_family="x", model_id="y"))) == base
    assert content_hash(_item(content_hash="already-set-somehow")) == base


def test_content_hash_mutation_a_hash_that_drops_a_field_is_caught() -> None:
    """Mutation test (INV-LOCK-1): a content hash that IGNORED ``error_regime`` (a hand-rolled
    serialization that drops the field) would collide two items differing only in regime. The real
    ``content_hash`` separates them; this asserts the broken variant would fail, so the field-drop
    mutation cannot pass silently."""

    def broken_hash(item: CorpusItem) -> str:
        payload = "|".join(
            [
                item.frame.model_dump_json(),
                item.thesis.model_dump_json(),
                str(item.test_code),
                str(item.is_error),
                item.stratum,
                # error_regime DROPPED — the mutation.
                item.split,
            ]
        )
        return hashlib.sha256(payload.encode()).hexdigest()

    a = _item(error_regime="wrong-op")
    b = _item(error_regime="off-by-one")
    assert broken_hash(a) == broken_hash(b)  # the mutant collides
    assert content_hash(a) != content_hash(b)  # the real hash does not


# ---------------------------------------------------------------------------
# lock_corpus — every item carries a non-empty content_hash after the lock
# ---------------------------------------------------------------------------


def test_lock_corpus_stamps_every_item() -> None:
    raw = [_item(item_id=1), _item(item_id=2, stratum="K")]
    assert all(it.content_hash == "" for it in raw)
    locked = lock_corpus(raw)
    assert all(it.content_hash != "" for it in locked)
    assert locked[0].content_hash == content_hash(raw[0])


def test_lock_corpus_is_pure() -> None:
    """The frozen inputs are untouched; the stamp lands on the returned copies only."""
    raw = [_item()]
    lock_corpus(raw)
    assert raw[0].content_hash == ""


# ---------------------------------------------------------------------------
# INV-LOCK-3 — MeasurementFingerprint: determinism + per-input sensitivity
# ---------------------------------------------------------------------------


def _locked(*items: CorpusItem) -> list[CorpusItem]:
    return lock_corpus(list(items))


def _fp(
    *,
    items: list[CorpusItem] | None = None,
    git_sha: str = "abc123",
    exec_env: ExecEnvIdentity | None = None,
    master_seed: int = MASTER_SEED,
    residual_epsilon: float = RESIDUAL_EPSILON_UNAUDITED,
) -> MeasurementFingerprint:
    return build_fingerprint(
        items if items is not None else _locked(_item(item_id=1), _item(item_id=2, stratum="K")),
        git_sha=git_sha,
        exec_env=exec_env if exec_env is not None else _env(),
        master_seed=master_seed,
        residual_epsilon=residual_epsilon,
    )


def test_fingerprint_is_deterministic() -> None:
    assert _fp().digest == _fp().digest


def test_fingerprint_refuses_never_locked_item() -> None:
    with pytest.raises(ValueError, match="never-locked"):
        build_fingerprint([_item()], git_sha="abc123", exec_env=_env())


def test_fingerprint_sensitive_to_content_hash_aggregate() -> None:
    a = _fp()
    b = _fp(items=_locked(_item(item_id=1), _item(item_id=2, stratum="clean", is_error=0)))
    assert a.content_hash_aggregate != b.content_hash_aggregate
    assert a.digest != b.digest


def test_fingerprint_aggregate_is_order_independent() -> None:
    """The corpus is a SET: re-ordering the items locks to the same aggregate (and digest)."""
    i1, i2 = _item(item_id=1), _item(item_id=2, stratum="K")
    forward = _fp(items=_locked(i1, i2))
    reversed_ = _fp(items=_locked(i2, i1))
    assert forward.digest == reversed_.digest


def test_fingerprint_sensitive_to_git_sha() -> None:
    assert _fp(git_sha="abc123").digest != _fp(git_sha="def456").digest


def test_fingerprint_sensitive_to_master_seed() -> None:
    assert _fp(master_seed=MASTER_SEED).digest != _fp(master_seed=MASTER_SEED + 1).digest


def test_fingerprint_sensitive_to_planter_families() -> None:
    """A converted-O family present vs absent flips the fingerprint (the env-fenced population)."""
    det_only = _fp(items=_locked(_item(item_id=1), _item(item_id=2, stratum="K")))
    converted = ConvertedPlanterStamp(
        planter_model_family="f",
        planter_model_id="m",
        adversary_family="adv",
        winning_round=2,
    )
    with_converted = _fp(
        items=_locked(_item(item_id=1), _item(item_id=2, stratum="K", planter=converted))
    )
    assert det_only.planter_families != with_converted.planter_families
    assert det_only.digest != with_converted.digest


def test_fingerprint_sensitive_to_exec_env() -> None:
    """CF-4.4c-CONVERTER-ENV-PIN: two different execution-environment identities -> two
    fingerprints. A converted-O test minted under one env must not silently re-evaluate under
    another."""
    env_a = _env(python_version="3.12.7", locale_lc_ctype="C")
    env_b = _env(python_version="3.13.0", locale_lc_ctype="en_US.UTF-8")
    assert _fp(exec_env=env_a).digest != _fp(exec_env=env_b).digest


def test_fingerprint_sensitive_to_exec_env_package_versions() -> None:
    env_a = _env(package_versions=(("pydantic", "2.12.5"),))
    env_b = _env(package_versions=(("pydantic", "2.13.0"),))
    assert _fp(exec_env=env_a).digest != _fp(exec_env=env_b).digest


def test_fingerprint_sensitive_to_residual_epsilon() -> None:
    """The ε-slot is folded: a post-audit fill (4.4c-5b) flips the fingerprint, so a stale clean
    bill cannot survive a re-lock."""
    unaudited = _fp()
    audited = _fp(residual_epsilon=0.012)
    assert unaudited.digest != audited.digest


# ---------------------------------------------------------------------------
# The residual-ε STUB slot — defaults to the not-yet-audited sentinel
# ---------------------------------------------------------------------------


def test_residual_epsilon_defaults_to_unaudited_sentinel() -> None:
    fp = _fp()
    assert fp.residual_epsilon == RESIDUAL_EPSILON_UNAUDITED
    assert fp.epsilon_audited is False


def test_residual_epsilon_audited_flag_flips_when_filled() -> None:
    assert _fp(residual_epsilon=0.0).epsilon_audited is True
    assert _fp(residual_epsilon=0.012).epsilon_audited is True


def test_master_seed_defaults_to_build_date_seed() -> None:
    assert _fp().master_seed == MASTER_SEED
    assert MASTER_SEED == 20260616


# ---------------------------------------------------------------------------
# PIECE D — contamination audit (INV-LOCK-4, §3.8): item_id disjointness + hash-log overlap
# ---------------------------------------------------------------------------


def _measurement(item_id: int, **overrides: object) -> CorpusItem:
    """A measurement-split item carrying a real (locked) content_hash."""
    base: dict[str, object] = {"item_id": item_id, "split": "measurement"}
    base.update(overrides)
    raw = _item(**base)
    return raw.model_copy(update={"content_hash": content_hash(raw)})


def _tuning(item_id: int, **overrides: object) -> CorpusItem:
    base: dict[str, object] = {"item_id": item_id, "split": "tuning"}
    base.update(overrides)
    raw = _item(**base)
    return raw.model_copy(update={"content_hash": content_hash(raw)})


def test_contamination_clean_partition_passes() -> None:
    """Disjoint item_ids + no measurement hash in the tuning log -> the lock precondition holds
    (returns None, raises nothing)."""
    items = [_tuning(1), _tuning(2), _measurement(10), _measurement(11)]
    assert_no_contamination(items, tuning_run_hashes=["unrelated-hash"])  # must not raise


def test_contamination_refuses_item_id_in_both_splits() -> None:
    """A planted item_id present in BOTH the tuning and measurement split refuses the lock — the
    held-out partition is not disjoint."""
    items = [_tuning(7), _measurement(7, stratum="clean", is_error=0)]
    with pytest.raises(CorpusLockError, match="appear in BOTH"):
        assert_no_contamination(items, tuning_run_hashes=[])


def test_contamination_refuses_measurement_hash_in_tuning_log() -> None:
    """A measurement item whose content_hash appears in the tuning-run log refuses the lock — the
    tuning loop already saw this item."""
    leaked = _measurement(10)
    items = [_tuning(1), leaked]
    with pytest.raises(CorpusLockError, match="tuning-run log"):
        assert_no_contamination(items, tuning_run_hashes=[leaked.content_hash, "other"])


def test_contamination_ignores_tuning_hash_overlap() -> None:
    """Only MEASUREMENT content-hashes are audited against the tuning log: a tuning item's own hash
    appearing in the tuning log is expected, not a violation (negative control — the check is
    measurement-scoped, not corpus-wide)."""
    t = _tuning(1)
    items = [t, _measurement(10)]
    # t.content_hash is in the log (a tuning item DID feed tuning) — must NOT refuse.
    assert_no_contamination(items, tuning_run_hashes=[t.content_hash])  # must not raise


def test_contamination_refuses_never_locked_measurement_item() -> None:
    """A measurement item with an empty content_hash (never locked) has no auditable identity ->
    refuse (mirrors build_fingerprint's never-locked tripwire)."""
    unlocked = _item(item_id=10, split="measurement")  # content_hash == ""
    with pytest.raises(CorpusLockError, match="never-locked"):
        assert_no_contamination([unlocked], tuning_run_hashes=[])


def test_contamination_tuning_run_hashes_single_pass_iterator_ok() -> None:
    """The tuning-run log is modeled as an Iterable[str] and consumed once — a single-pass
    generator (the realistic downstream shape) is materialized safely, audits >1 item correctly."""
    leaked = _measurement(10)
    items = [leaked, _measurement(11)]
    gen = (h for h in [leaked.content_hash])
    with pytest.raises(CorpusLockError, match="tuning-run log"):
        assert_no_contamination(items, tuning_run_hashes=gen)


def test_contamination_mutation_audit_skipping_hash_check_is_caught() -> None:
    """Mutation test (INV-LOCK-4): an audit that checks ONLY item_id disjointness and SKIPS the
    content-hash log overlap would PASS a contaminated corpus (disjoint ids, but a measurement
    item's content was already seen in tuning). The real audit refuses; the mutant does not — so the
    hash-overlap check cannot be silently dropped."""
    leaked = _measurement(10)
    items = [_tuning(1), leaked]
    log = {leaked.content_hash}

    def mutant_audit_id_only(its: list[CorpusItem], *, tuning_run_hashes: set[str]) -> None:
        tuning_ids = {i.item_id for i in its if i.split == "tuning"}
        measurement_ids = {i.item_id for i in its if i.split == "measurement"}
        if tuning_ids & measurement_ids:
            raise CorpusLockError("disjointness")
        # hash-overlap check DROPPED — the mutation.

    # The mutant passes the contaminated corpus (ids are disjoint).
    mutant_audit_id_only(items, tuning_run_hashes=log)  # must not raise
    # The real audit refuses it.
    with pytest.raises(CorpusLockError, match="tuning-run log"):
        assert_no_contamination(items, tuning_run_hashes=log)


# ---------------------------------------------------------------------------
# PIECE E — matched-pair bijection re-validation (INV-LOCK-5 = INV-7, §5)
# ---------------------------------------------------------------------------


def _err(item_id: int, sibling: int | None) -> CorpusItem:
    """An error-stratum (K) member of a matched pair."""
    return _item(item_id=item_id, stratum="K", is_error=1, matched_sibling_id=sibling)


def _clean(item_id: int, sibling: int | None) -> CorpusItem:
    """A clean member of a matched pair (the corrected sibling)."""
    return _item(
        item_id=item_id,
        stratum="clean",
        is_error=0,
        matched_sibling_id=sibling,
        label_source="oracle",
        label_provenance=_prov(),
    )


def test_bijection_both_survive_pair_intact() -> None:
    """Both members survive the (empty) drop set -> the pair stays PAIRED, nothing demoted."""
    p, q = _err(1, sibling=2), _clean(2, sibling=1)
    result = revalidate_bijection([p, q], frozenset())
    assert {it.item_id for it in result.paired} == {1, 2}
    assert result.demoted_unpaired == ()
    # The paired items keep their matched_sibling_id (NOT cleared).
    assert all(it.matched_sibling_id is not None for it in result.paired)


def test_bijection_one_dropped_pair_dissolved_survivor_demoted() -> None:
    """Drop ONE member -> the pair DISSOLVES; the survivor is demoted to the unpaired pool, carrying
    its REAL, UNCHANGED stratum with matched_sibling_id cleared to None."""
    # Pair (1=K, 2=clean); item 2 was abstention-dropped -> only item 1 (K) survives in `items`.
    survivor = _err(1, sibling=2)
    result = revalidate_bijection([survivor], frozenset({2}))
    assert result.paired == ()
    assert len(result.demoted_unpaired) == 1
    demoted = result.demoted_unpaired[0]
    assert demoted.item_id == 1
    assert demoted.matched_sibling_id is None  # link cleared
    assert demoted.stratum == "K"  # real stratum unchanged
    assert demoted.is_error == 1


def test_bijection_demotion_preserves_real_clean_stratum() -> None:
    """A surviving CLEAN orphan is demoted carrying its real `clean` stratum (the §5
    marginal-preservation discipline: it contributes +1 to the marginal it truly belongs to)."""
    survivor = _clean(2, sibling=1)
    result = revalidate_bijection([survivor], frozenset({1}))
    assert result.demoted_unpaired[0].stratum == "clean"
    assert result.demoted_unpaired[0].matched_sibling_id is None


def test_bijection_unpaired_item_stays_global_not_demoted() -> None:
    """A never-paired item (matched_sibling_id is None) is neither retained-paired nor a dissolution
    survivor — it is not in `demoted_unpaired` (which is ONLY one-survivor-pair orphans)."""
    p, q = _err(1, sibling=2), _clean(2, sibling=1)
    solo = _item(item_id=9, matched_sibling_id=None)
    result = revalidate_bijection([p, q, solo], frozenset())
    assert {it.item_id for it in result.paired} == {1, 2}
    assert result.demoted_unpaired == ()  # solo is global, not a dissolution survivor


def test_bijection_surviving_pairs_are_a_clean_mutual_bijection() -> None:
    """The marginal-preservation property: after drops, every retained paired item's sibling is
    present in the paired set AND the relation is mutual (a clean bijection). Two pairs, one
    dissolved by a drop; the other re-validates intact."""
    a1, a2 = _err(1, sibling=2), _clean(2, sibling=1)  # intact pair
    b1 = _err(3, sibling=4)  # pair (3,4); 4 dropped -> dissolves
    result = revalidate_bijection([a1, a2, b1], frozenset({4}))

    paired_ids = {it.item_id for it in result.paired}
    assert paired_ids == {1, 2}
    by_id = {it.item_id: it for it in result.paired}
    # Clean mutual bijection: each paired item's sibling is present and names it back.
    for it in result.paired:
        assert it.matched_sibling_id in by_id
        assert by_id[it.matched_sibling_id].matched_sibling_id == it.item_id
    # The dissolved survivor demoted with its real stratum + cleared link.
    assert {it.item_id for it in result.demoted_unpaired} == {3}
    assert result.demoted_unpaired[0].matched_sibling_id is None


def test_bijection_refuses_present_yet_dropped_item() -> None:
    """Caller contract: `items` is the PROMOTED survivor set, disjoint from dropped_ids. An id that
    is both present and dropped is inconsistent -> refuse."""
    p, q = _err(1, sibling=2), _clean(2, sibling=1)
    with pytest.raises(CorpusLockError, match="inconsistent"):
        revalidate_bijection([p, q], frozenset({1}))


def test_bijection_refuses_dangling_sibling_reference() -> None:
    """A matched_sibling_id naming an id that is NEITHER present NOR dropped is a dangling reference
    (corpus defect) -> refuse."""
    orphan = _err(1, sibling=99)  # 99 is neither present nor in dropped_ids
    with pytest.raises(CorpusLockError, match="dangling"):
        revalidate_bijection([orphan], frozenset())


def test_bijection_mutation_uncleared_orphan_is_caught() -> None:
    """Mutation test (INV-LOCK-5): a re-validation that LEAVES an orphaned paired item — both
    members present but the link is asymmetric (one names a non-mutual sibling, the cleared-link
    discipline skipped) — must be CAUGHT, not silently retained as paired.

    Construct an asymmetric pair: item 1 names 2, but item 2 names 5 (a non-mutual link). The real
    re-validation refuses it; a mutant that only checked presence (not mutuality) would wrongly
    accept item 1 as paired."""
    a = _err(1, sibling=2)
    b = _clean(2, sibling=5)  # asymmetric: 2 does NOT name 1 back
    # item 5 present so 2's link is not 'dangling' — the defect is pure asymmetry.
    c = _err(5, sibling=2)

    def mutant_presence_only(items: list[CorpusItem], dropped: frozenset[int]) -> set[int]:
        present = {i.item_id for i in items}
        paired = set()
        for i in items:
            if i.matched_sibling_id is not None and i.matched_sibling_id in present:
                paired.add(i.item_id)  # presence-only, NO mutuality check — the mutation.
        return paired

    # The mutant wrongly accepts item 1 (sibling 2 is present) despite the asymmetry.
    assert 1 in mutant_presence_only([a, b, c], frozenset())
    # The real re-validation refuses the asymmetric pair.
    with pytest.raises(CorpusLockError, match="asymmetric"):
        revalidate_bijection([a, b, c], frozenset())


# ---------------------------------------------------------------------------
# PIECE G1 -- clean-spec tripwire (INV-LOCK-6, §3.3): the PLANNED-n design ceiling
#            + the variance floor. STRUCTURAL STUB -- runs on hand-built / synth Cells.
# ---------------------------------------------------------------------------

# A planning between-item spec variance for the floor (the gate is sized against this; the planned-n
# ceiling is computed from R + n_clean_planned, never from the realized artifact).
_SIGMA_SQ_B_SPEC = 0.028
_R = 7
_N_CLEAN_PLANNED = 80


def _cell(item_id: int, stratum: str, arm: str, *, flagged: int, n_trials: int) -> list[Cell]:
    """Emit ``n_trials`` Cells for one ``(item, stratum, arm)`` with a fixed ``flagged`` value."""
    return [
        Cell(
            item_id=item_id,
            stratum=stratum,
            arm=arm,
            trial=t,
            seed=item_id * 1000 + t,
            flagged=flagged,
            route="flag" if flagged else "pass",
        )
        for t in range(n_trials)
    ]


def _clean_cells_with_spec(
    arm: str, *, n_clean: int, n_flagged_items: int, R: int = _R, id_base: int = 5000
) -> list[Cell]:
    """A clean-stratum artifact for ``arm`` where ``n_flagged_items`` of ``n_clean`` items flag on
    EVERY trial (a per-item flag rate of 1.0) and the rest never flag (rate 0.0). Spec over the pool
    is ``1 - n_flagged_items/n_clean`` (a flag on a clean item is a false positive). The split into
    all-flag / no-flag items gives the per-item flag-rate variance a non-trivial value, so it drives
    the CEILING independently of the variance floor."""
    cells: list[Cell] = []
    for i in range(n_clean):
        flagged = 1 if i < n_flagged_items else 0
        cells.extend(_cell(id_base + i, "clean", arm, flagged=flagged, n_trials=R))
    return cells


def test_spec_ceiling_nontrivial_spec_passes() -> None:
    """A clean pool with a genuinely non-trivial spec (well below the planned-n ceiling) AND ample
    per-item flag-rate variance passes the tripwire (returns None, raises nothing)."""
    # 8 of 80 clean items flag every trial -> spec = 0.90, var of {1.0x8, 0.0x72} ~ 0.0825 (>>
    # 0.014 floor). Both parts satisfied.
    cells = _clean_cells_with_spec("C", n_clean=_N_CLEAN_PLANNED, n_flagged_items=8)
    # Must not raise.
    assert_spec_ceiling(
        cells,
        n_clean_planned=_N_CLEAN_PLANNED,
        R=_R,
        sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
    )


def test_spec_ceiling_refuses_hollow_spec_at_ceiling() -> None:
    """A perfectly-trivial clean pool (NO clean item ever flags -> spec = 1.0, AT/above the
    ceiling 1 - 2/(R*n_clean_planned)) refuses the lock -- the spec arm of J is hollow."""
    cells = _clean_cells_with_spec("C", n_clean=_N_CLEAN_PLANNED, n_flagged_items=0)
    with pytest.raises(CorpusLockError, match="AT OR ABOVE the planned-n ceiling"):
        assert_spec_ceiling(
            cells,
            n_clean_planned=_N_CLEAN_PLANNED,
            R=_R,
            sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
        )


def test_spec_ceiling_refuses_collapsed_variance_with_nontrivial_mean() -> None:
    """The spec_C'=0.96-but-hollow case the mean-based <0.98 tripwire could never see: a non-trivial
    MEAN spec whose per-item flag-rate variance has collapsed to near-zero. Build EVERY clean item
    with an identical fractional flag rate (3 of 7 trials flag -> rate 3/7 on every item): mean spec
    = 1 - 3/7 ~ 0.571 (well below the ceiling, so part 1 passes) but pvariance(rates) = 0 (the floor
    fires)."""
    # Every clean item flags exactly 3 of its 7 trials -> identical 3/7 per-item rate -> mean
    # = 1 - 3/7 ~ 0.571 (below the ceiling), but pvariance(rates) = 0 (the floor fires).
    cells: list[Cell] = []
    for i in range(_N_CLEAN_PLANNED):
        for t in range(_R):
            flagged = 1 if t < 3 else 0
            cells.append(
                Cell(
                    item_id=6000 + i,
                    stratum="clean",
                    arm="C",
                    trial=t,
                    seed=(6000 + i) * 1000 + t,
                    flagged=flagged,
                    route="flag" if flagged else "pass",
                )
            )
    with pytest.raises(CorpusLockError, match="below the planning-derived floor"):
        assert_spec_ceiling(
            cells,
            n_clean_planned=_N_CLEAN_PLANNED,
            R=_R,
            sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
        )


def test_spec_ceiling_uses_planned_n_not_realized_backwards_tripwire() -> None:
    """The §3.3 NEW-MED-2 backwards-tripwire pin: a SHRUNK realized n_clean must NOT relax the
    ceiling. The construction uses a realized clean pool of 10 items (abstention shrank it from the
    planned 80) whose arm-C spec = 0.98571 -- a value strictly BELOW the planned-n ceiling
    (1 - 2/(7*80) = 0.99643, so it PASSES) but ABOVE the realized-n ceiling (1 - 2/(7*10) = 0.97143,
    so a realized-n ceiling would REFUSE a still-hollow-ish pool inconsistently). The point: the
    ceiling must be a TOTAL function of PLANNED n + R, never of the realized clean count.

    Load-bearing inequality: planned n yields a HIGHER (tighter) ceiling than a shrunk realized n --
    the backwards bug would LOWER it, loosening the tripwire when scrutiny should increase."""
    ceiling_planned = 1.0 - 2.0 / (_R * _N_CLEAN_PLANNED)  # 0.996428
    ceiling_realized = 1.0 - 2.0 / (_R * 10)  # 0.971428
    assert ceiling_planned > ceiling_realized  # the anti-loosening law

    # Behavioral pin: a perfectly-hollow shrunk pool (10 items, spec = 1.0) is refused by the
    # planned-n ceiling -- abstention down to 10 items cannot relax it.
    hollow = _clean_cells_with_spec("C", n_clean=10, n_flagged_items=0, id_base=7000)
    with pytest.raises(CorpusLockError, match="AT OR ABOVE the planned-n ceiling"):
        assert_spec_ceiling(
            hollow,
            n_clean_planned=_N_CLEAN_PLANNED,
            R=_R,
            sigma_sq_b_spec_planning=0.0,  # floor disabled -- isolate the ceiling.
        )


def test_spec_ceiling_refuses_no_clean_cells() -> None:
    """An artifact with no clean-stratum cells cannot be cleared -- a hollow corpus refuses."""
    cells = _cell(1, "K", "C", flagged=1, n_trials=_R)
    with pytest.raises(CorpusLockError, match="no clean-stratum cells"):
        assert_spec_ceiling(
            cells,
            n_clean_planned=_N_CLEAN_PLANNED,
            R=_R,
            sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
        )


def test_spec_ceiling_refuses_malformed_planning_constants() -> None:
    cells = _clean_cells_with_spec("C", n_clean=_N_CLEAN_PLANNED, n_flagged_items=8)
    with pytest.raises(CorpusLockError, match="must both be"):
        assert_spec_ceiling(
            cells, n_clean_planned=0, R=_R, sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC
        )
    with pytest.raises(CorpusLockError, match="must both be"):
        assert_spec_ceiling(
            cells, n_clean_planned=_N_CLEAN_PLANNED, R=0, sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC
        )


def test_spec_ceiling_mutation_realized_n_in_ceiling_is_caught() -> None:
    """Mutation test (INV-LOCK-6 / NEW-MED-2): a ceiling that read the REALIZED clean count instead
    of PLANNED n_clean would compute a LOOSER ceiling on a shrunk pool, passing a hollow-but-shrunk
    spec the real (planned-n) check refuses.

    The artifact: a 10-item clean pool (shrunk by abstention) with 1 false-positive cell -> arm-C
    spec = 1 - 1/70 = 0.98571. This spec is strictly BELOW the planned-n ceiling 0.99643 (the real
    check PASSES it on the ceiling axis) but ABOVE the realized-n ceiling 0.97143. A direct hollow
    case (spec = 1.0) then pins the behavioral refusal under planned n that a realized-n mutant on a
    tiny pool would loosen toward."""
    n_realized = 10
    ceiling_realized = 1.0 - 2.0 / (_R * n_realized)  # 0.971428
    ceiling_planned = 1.0 - 2.0 / (_R * _N_CLEAN_PLANNED)  # 0.996428

    realized_spec = 1.0 - 1.0 / (n_realized * _R)  # 0.985714
    assert ceiling_realized < realized_spec < ceiling_planned  # the separating band exists

    # Load-bearing law: planned n => the TIGHTER (higher) ceiling; a realized-n mutant on a shrunk
    # pool computes a LOWER ceiling and so admits hollow specs the planned-n check refuses.
    assert ceiling_planned > ceiling_realized

    # Behavioral pin: an all-clean shrunk pool (spec=1.0) is refused by the real planned-n ceiling.
    hollow = _clean_cells_with_spec("C", n_clean=n_realized, n_flagged_items=0, id_base=8500)
    with pytest.raises(CorpusLockError, match="AT OR ABOVE the planned-n ceiling"):
        assert_spec_ceiling(
            hollow, n_clean_planned=_N_CLEAN_PLANNED, R=_R, sigma_sq_b_spec_planning=0.0
        )


def test_spec_ceiling_on_synth_cells_passes() -> None:
    """Integration-ish smoke: a synth_cells artifact at a realistic non-trivial spec_C passes the
    tripwire on arm 'C' (the gate's spec arm). synth_cells emits C/D on K + clean."""
    cells = synth_cells(
        Random(42),
        m_K=40,
        m_clean=_N_CLEAN_PLANNED,
        R=_R,
        sens_C=0.5,
        dsens=0.2,
        sb_sens=0.04,
        spec_C=0.85,  # well below the ceiling, real item-to-item variance
        dspec=0.05,
        sb_spec=_SIGMA_SQ_B_SPEC,
        rho_w=0.3,
    )
    # Must not raise.
    assert_spec_ceiling(
        cells,
        n_clean_planned=_N_CLEAN_PLANNED,
        R=_R,
        sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC,
    )


# ---------------------------------------------------------------------------
# PIECE G2 -- INV-A0 / INV-A1 arm-A floor (§5). STRUCTURAL STUB -- hand-built arm-A Cells
#            (synth_cells emits no arm A; A's floor is scored analytically there).
# ---------------------------------------------------------------------------


def _arm_a_artifact(
    *,
    k_flagged_items: int = 0,
    n_k: int = 40,
    clean_flagged_cells: int = 0,
    n_clean: int = _N_CLEAN_PLANNED,
    R: int = _R,
) -> list[Cell]:
    """Build a frozen artifact carrying arm 'A' (plus a token 'D' so the corpus is realistic) on K +
    clean. ``k_flagged_items`` arm-A K items flag on every trial (INV-A0 violation when > 0);
    ``clean_flagged_cells`` arm-A clean CELLS are false positives (drive spec_A, INV-A1's axis)."""
    cells: list[Cell] = []
    # K stratum, arm A: flag the first k_flagged_items items on every trial.
    for i in range(n_k):
        flagged = 1 if i < k_flagged_items else 0
        cells.extend(_cell(100 + i, "K", "A", flagged=flagged, n_trials=R))
        cells.extend(_cell(100 + i, "K", "D", flagged=1, n_trials=R))  # D flags (a real antithesis)
    # clean stratum, arm A: distribute clean_flagged_cells false positives across cells.
    fp_remaining = clean_flagged_cells
    for i in range(n_clean):
        for t in range(R):
            f = 1 if fp_remaining > 0 else 0
            if f:
                fp_remaining -= 1
            cells.append(
                Cell(
                    item_id=200 + i,
                    stratum="clean",
                    arm="A",
                    trial=t,
                    seed=(200 + i) * 1000 + t,
                    flagged=f,
                    route="flag" if f else "pass",
                )
            )
        cells.extend(_cell(200 + i, "clean", "D", flagged=0, n_trials=R))
    return cells


def test_arm_a_floor_satisfied_passes() -> None:
    """Arm A flags NOTHING on K (sens_A == 0) and false-positives on <=~2 clean cells (spec_A in
    1 - tau_A) -> the floor holds (returns None)."""
    # tau_A = 2/(7*80) = 0.003571 -> 1-tau_A = 0.996428. Over 80x7=560 clean cells, <=2 FPs keeps
    # spec_A = 1 - 2/560 = 0.99643 >= floor. Use exactly 1 FP -> spec_A = 0.99821 (> floor).
    cells = _arm_a_artifact(k_flagged_items=0, clean_flagged_cells=1)
    assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)  # must not raise


def test_arm_a_floor_zero_clean_fps_passes() -> None:
    """A perfectly clean arm A (spec_A = 1.0) trivially clears INV-A1."""
    cells = _arm_a_artifact(k_flagged_items=0, clean_flagged_cells=0)
    assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)  # must not raise


def test_arm_a_floor_refuses_inv_a0_violation() -> None:
    """INV-A0 negative control: arm A flags ANY K item -> sens_A > 0 -> the item is mis-stratified
    (oracle-blind item the oracle actually reached) -> refuse lock."""
    cells = _arm_a_artifact(k_flagged_items=1, clean_flagged_cells=0)
    with pytest.raises(CorpusLockError, match=r"INV-A0.*sens_A"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_refuses_inv_a1_violation() -> None:
    """INV-A1 negative control: arm A false-positives on too many clean cells -> spec_A below
    1 - tau_A -> A is not the floor the §5 centering assumes -> refuse lock.

    tau_A = 2/(7*80) = 0.003571; 1-tau_A = 0.996428. To breach: spec_A < 0.996428 over 560 clean
    means > 2 FPs. Use 10 FPs -> spec_A = 1 - 10/560 = 0.98214 < floor."""
    cells = _arm_a_artifact(k_flagged_items=0, clean_flagged_cells=10)
    with pytest.raises(CorpusLockError, match=r"INV-A1.*spec_A"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_uses_planned_n_not_realized() -> None:
    """NEW-MED-2 applied to tau_A: the clean-FP budget is fixed at PLANNED n_clean, so
    abstention (a shrunk realized clean pool) cannot relax it.

    Load-bearing law: PLANNED n => a SMALLER tau_A => a HIGHER (tighter) floor 1-tau_A; a realized-n
    bug on a shrunk pool computes a LARGER tau_A and LOOSENS the floor."""
    tau_planned = 2.0 / (_R * _N_CLEAN_PLANNED)  # 0.003571
    tau_realized = 2.0 / (_R * 10)  # 0.028571
    assert tau_planned < tau_realized

    # Behavioral pin: a shrunk 10-item clean pool with 1 FP -> spec_A = 1 - 1/70 = 0.98571, which is
    # below the planned floor 0.99643 (REFUSE, correct) but above the realized-n floor 0.97143 (a
    # realized-n bug would PASS). The real check refuses it.
    cells = _arm_a_artifact(k_flagged_items=0, clean_flagged_cells=1, n_clean=10)
    with pytest.raises(CorpusLockError, match=r"INV-A1.*spec_A"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_mutation_realized_n_in_tau_is_caught() -> None:
    """Mutation test (INV-A1 / NEW-MED-2): a floor that computed tau_A from the REALIZED clean count
    instead of PLANNED n_clean would LOOSEN the clean-FP budget on a shrunk pool, passing an arm A
    that false-positives more than the planned ~2-cell budget.

    The artifact: 10 clean items (shrunk), 1 FP -> spec_A = 0.98571. The real check uses PLANNED
    n_clean=80 (floor 0.996428) and REFUSES. The mutant would use realized n=10 (floor 0.971428) and
    PASS. We assert the real check refuses, and that the mutant's looser floor would have admitted
    it."""
    cells = _arm_a_artifact(k_flagged_items=0, clean_flagged_cells=1, n_clean=10)
    realized_spec_a = 1.0 - 1.0 / (10 * _R)  # 0.98571
    floor_planned = 1.0 - 2.0 / (_R * _N_CLEAN_PLANNED)  # 0.996428
    floor_realized = 1.0 - 2.0 / (_R * 10)  # 0.971428
    # The mutant (realized-n floor) WOULD have passed this artifact...
    assert realized_spec_a >= floor_realized
    # ...but the real (planned-n) floor refuses it.
    assert realized_spec_a < floor_planned
    with pytest.raises(CorpusLockError, match=r"INV-A1.*spec_A"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_refuses_missing_arm_a_on_k() -> None:
    """An artifact with no arm-A cells on K cannot have its floor asserted -> refuse (a missing arm
    is not a silent pass)."""
    # Only D on K, A only on clean.
    cells: list[Cell] = []
    for i in range(40):
        cells.extend(_cell(100 + i, "K", "D", flagged=1, n_trials=_R))
    for i in range(_N_CLEAN_PLANNED):
        cells.extend(_cell(200 + i, "clean", "A", flagged=0, n_trials=_R))
    with pytest.raises(CorpusLockError, match="arm 'A' is absent from the 'K'"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_refuses_no_k_cells() -> None:
    """No K error-stratum cells at all -> nothing to audit INV-A0 against -> refuse."""
    cells: list[Cell] = []
    for i in range(_N_CLEAN_PLANNED):
        cells.extend(_cell(200 + i, "clean", "A", flagged=0, n_trials=_R))
    with pytest.raises(CorpusLockError, match="no 'K' error-stratum cells"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_refuses_no_clean_cells() -> None:
    """No clean cells -> INV-A1 has nothing to audit -> refuse (INV-A0 on K passes first)."""
    cells: list[Cell] = []
    for i in range(40):
        cells.extend(_cell(100 + i, "K", "A", flagged=0, n_trials=_R))
        cells.extend(_cell(100 + i, "K", "D", flagged=1, n_trials=_R))
    with pytest.raises(CorpusLockError, match="no clean-stratum cells"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=_R)


def test_arm_a_floor_refuses_malformed_planning_constants() -> None:
    cells = _arm_a_artifact(k_flagged_items=0, clean_flagged_cells=0)
    with pytest.raises(CorpusLockError, match="must both be"):
        assert_arm_a_floor(cells, n_clean_planned=0, R=_R)
    with pytest.raises(CorpusLockError, match="must both be"):
        assert_arm_a_floor(cells, n_clean_planned=_N_CLEAN_PLANNED, R=0)


# ===========================================================================
# Pod 4.4c-6a -- INSTRUMENT 1: §5 stratum-membership shuffle null
# ===========================================================================
#
# Mutation-resistant: an HONEST artifact (no stratum->flag leakage after welding) passes both the
# centering (Q-A) and coverage-rate (Q-B) assertions; the deliberately-LEAKY shuffle (flag-semantics
# carried WITH the stratum move) makes mean(shuffle_delta) non-zero and the centering assertion
# FIRES; a hand-built high-k artifact fires the coverage gate; the per-item welding + determinism
# are pinned. n_outer is kept SMALL (200) + n_shuffles at the binding floor (200) so the suite stays
# fast (~1.5s/run). The §5 J_shuf identity (J = mean_S f - mean_{S^c} f -> E=0) makes ANY welded
# artifact center at 0 -- a leaky variant that breaks the welding is the only way to non-center.


def _assoc_cells(n_k: int = 12, n_clean: int = 12, R: int = 6) -> list[Cell]:
    """An artifact with a STRONG stratum<->flag association on arm D: K items flag almost every
    trial, clean items flag almost none; arm C is FLAT (the same flag pattern on every item of both
    strata) so ``δ = J_D - J_C`` is driven entirely by D. After an HONEST welded shuffle the
    association is severed -> ``mean(shuffle_δ)`` centers at 0; a LEAKY shuffle preserves it ->
    ``mean(shuffle_δ)`` stays large. Items are id'd 0..n_k-1 (K) and 1000.. (clean)."""
    cells: list[Cell] = []
    for i in range(n_k):
        for t in range(R):
            cells.append(
                Cell(
                    item_id=i,
                    stratum="K",
                    arm="D",
                    trial=t,
                    seed=i * 100 + t,
                    flagged=1 if t < R - 1 else 0,
                    route="x",
                )
            )
            cells.append(
                Cell(
                    item_id=i,
                    stratum="K",
                    arm="C",
                    trial=t,
                    seed=i * 100 + t,
                    flagged=1 if t < R // 2 else 0,
                    route="x",
                )
            )
    for j in range(n_clean):
        item = 1000 + j
        for t in range(R):
            cells.append(
                Cell(
                    item_id=item,
                    stratum="clean",
                    arm="D",
                    trial=t,
                    seed=item * 100 + t,
                    flagged=1 if t == 0 else 0,
                    route="x",
                )
            )
            cells.append(
                Cell(
                    item_id=item,
                    stratum="clean",
                    arm="C",
                    trial=t,
                    seed=item * 100 + t,
                    flagged=1 if t < R // 2 else 0,
                    route="x",
                )
            )
    return cells


def _weak_assoc_cells(n_k: int = 12, n_clean: int = 12, R: int = 6, seed: int = 99) -> list[Cell]:
    """An HONEST artifact with NO stratum<->flag association: K and clean items draw their per-item
    flag count from the SAME overlapping distribution (1..R-1 flags), and arms D and C share the
    pattern per item, so the binding ``δ = J_D - J_C`` is intrinsically null with WIDE per-shuffle
    CIs. The within-pair sign-flip null on this artifact centers cleanly at 0 with few ``lo>0``
    exclusions -- the correct fixture for "the paired control runs + an honest corpus passes BOTH
    assertions" (the strong-association ``_assoc_cells`` makes the paired null pathologically tight,
    which is the honest ~3% false-refuse regime, not what this PASS pin wants to land in)."""
    rng = Random(seed)
    cells: list[Cell] = []

    def emit(item: int, stratum: str) -> None:
        f = rng.randint(1, R - 1)  # same flag-count distribution for K and clean (no confound)
        for arm in ("D", "C"):
            for t in range(R):
                cells.append(
                    Cell(
                        item_id=item,
                        stratum=stratum,
                        arm=arm,
                        trial=t,
                        seed=item * 100 + t,
                        flagged=1 if t < f else 0,
                        route="x",
                    )
                )

    for i in range(n_k):
        emit(i, "K")
    for j in range(n_clean):
        emit(1000 + j, "clean")
    return cells


_UNPAIRED = [*range(12), *(1000 + j for j in range(12))]
_DELTAS = {"D>C": ("D", "C")}


def test_shuffle_membership_is_deterministic() -> None:
    """A seeded :class:`~random.Random` -> byte-identical relabeled artifact (the control every
    sensitivity pin is measured against)."""
    cells = _assoc_cells()
    a = shuffle_stratum_membership(cells, paired_ids=[], unpaired_ids=_UNPAIRED, rng=Random(3))
    b = shuffle_stratum_membership(cells, paired_ids=[], unpaired_ids=_UNPAIRED, rng=Random(3))
    assert [c.stratum for c in a] == [c.stratum for c in b]


def test_shuffle_membership_welds_stratum_per_item() -> None:
    """The §5 implementation pin: the SAME new stratum lands on EVERY cell of an item (every arm,
    every trial). A split item would corrupt ``_index_cells``'s urn sizes."""
    cells = _assoc_cells()
    relabeled = shuffle_stratum_membership(
        cells,
        paired_ids=[(1, 1001)],
        unpaired_ids=[i for i in _UNPAIRED if i not in (1, 1001)],
        rng=Random(11),
    )
    for item_id in {c.item_id for c in cells}:
        strata = {c.stratum for c in relabeled if c.item_id == item_id}
        assert len(strata) == 1, f"item {item_id} split across strata {strata}"


def test_shuffle_membership_is_pure() -> None:
    """The frozen input cells are untouched; only the returned copies carry the new stratum."""
    cells = _assoc_cells()
    before = [c.stratum for c in cells]
    shuffle_stratum_membership(cells, paired_ids=[], unpaired_ids=_UNPAIRED, rng=Random(5))
    assert [c.stratum for c in cells] == before


def test_shuffle_membership_holds_marginals_fixed() -> None:
    """Fixed-marginal: the global permutation preserves the K/clean counts over the unpaired set."""
    cells = _assoc_cells()
    relabeled = shuffle_stratum_membership(
        cells, paired_ids=[], unpaired_ids=_UNPAIRED, rng=Random(9)
    )
    item_stratum = {c.item_id: c.stratum for c in relabeled}
    n_k = sum(1 for s in item_stratum.values() if s == "K")
    n_clean = sum(1 for s in item_stratum.values() if s == "clean")
    assert n_k == 12 and n_clean == 12


def test_shuffle_null_honest_corpus_passes() -> None:
    """An honest artifact (association severed by the welded shuffle) -> centering CI contains 0
    AND global-coverage ucb95 <= 0.08 -> :func:`assert_shuffle_null` returns a
    :class:`ShuffleNullReport` (raises nothing) carrying the report-only paired-coverage tail."""
    cells = _assoc_cells()
    result = shuffle_null_centering(
        cells,
        paired_ids=[],
        unpaired_ids=_UNPAIRED,
        deltas=_DELTAS,
        n_shuffles=200,
        seed=7,
        n_outer=200,
    )
    report = assert_shuffle_null(result)
    assert isinstance(report, ShuffleNullReport)
    assert set(report.paired_k) == set(_DELTAS)
    assert set(report.paired_coverage_ucb95) == set(_DELTAS)


def test_shuffle_null_leaky_mutation_fires_centering() -> None:
    """MUTATION TEST (§5, mandatory instrument-validity): the deliberately-LEAKY shuffle carries
    flag-semantics WITH the stratum move (it does NOT weld flags to items), so the label<->flag
    association survives -> ``mean(shuffle_δ)`` is non-zero -> :func:`assert_shuffle_null` MUST fire
    on the centering assertion. An assertion that cannot be made to fail tests nothing."""
    cells = _assoc_cells()
    leaky = shuffle_null_centering(
        cells,
        paired_ids=[],
        unpaired_ids=_UNPAIRED,
        deltas=_DELTAS,
        n_shuffles=200,
        seed=7,
        n_outer=200,
        _shuffle=_shuffle_stratum_membership_LEAKY,
    )
    with pytest.raises(CorpusLockError, match="centering"):
        assert_shuffle_null(leaky)


def test_shuffle_null_high_k_artifact_fires_coverage() -> None:
    """A high-k GLOBAL artifact (k>=10 leakage-direction δ-CI exclusions, ``lo>0``, at n=200) trips
    the coverage-rate gate (ucb95 > 0.08) even though the point estimates center perfectly at 0 --
    the centering and coverage-rate assertions are statistically ORTHOGONAL and gate off DIFFERENT
    artifacts. Built by hand on the public :class:`ShuffleNullResult` so the leakage tail is exact +
    deterministic. The coverage gate reads ``global_ci_bounds`` (Finding-1)."""
    n, k = 200, 12
    points = tuple(0.0 for _ in range(n))  # centering is clean...
    clean = tuple((-0.5, 0.5) for _ in range(n))
    leaky = tuple((0.01, 0.5) if i < k else (-0.5, 0.5) for i in range(n))  # k=12 have lo>0
    result = ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>A": points},
        paired_ci_bounds={"D>A": clean},  # paired tail is clean -> not gated anyway
        global_ci_bounds={"D>A": leaky},  # the GLOBAL draw carries the leaky tail
    )
    with pytest.raises(CorpusLockError, match="coverage-rate"):
        assert_shuffle_null(result)


def test_shuffle_null_honest_k_rate_passes_coverage() -> None:
    """The honest nominal one-sided-high rate is ~2.5%: k=5 at n=200 -> ucb95 ~ 0.0516 <= 0.08 ->
    coverage passes (the negative control for the high-k gate). Reads ``global_ci_bounds``."""
    n, k = 200, 5
    points = tuple(0.0 for _ in range(n))
    clean = tuple((-0.5, 0.5) for _ in range(n))
    near = tuple((0.01, 0.5) if i < k else (-0.5, 0.5) for i in range(n))
    result = ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>A": points},
        paired_ci_bounds={"D>A": clean},
        global_ci_bounds={"D>A": near},
    )
    assert isinstance(assert_shuffle_null(result), ShuffleNullReport)


def test_shuffle_null_paired_high_k_is_report_only_not_gated() -> None:
    """Finding-1 regression: a high-k tail on the PAIRED artifact (k=14 lo>0 at n=200, ucb95~0.107)
    must NOT raise -- the paired-coverage tail is REPORT-ONLY. With the GLOBAL draw clean and
    centering clean, the call returns a :class:`ShuffleNullReport` whose paired_k surfaces the tail
    for the reviewer. (This is the un-calibrated paired tail the 0.08 ceiling must not gate.)"""
    n, k = 200, 14
    points = tuple(0.0 for _ in range(n))
    paired_leaky = tuple((0.01, 0.5) if i < k else (-0.5, 0.5) for i in range(n))
    clean = tuple((-0.5, 0.5) for _ in range(n))
    result = ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>A": points},
        paired_ci_bounds={"D>A": paired_leaky},  # k=14 lo>0 -> ucb95 ~ 0.107 > 0.08
        global_ci_bounds={"D>A": clean},  # the GATED draw is clean
    )
    report = assert_shuffle_null(result)  # must NOT raise (paired-coverage is report-only)
    assert report.paired_k["D>A"] == k
    assert report.paired_coverage_ucb95["D>A"] > 0.08  # surfaced, but never gated


def test_shuffle_null_hi_lt_zero_not_counted_as_leakage() -> None:
    """Only ``lo>0`` (the leakage direction) counts toward ``k``; ``hi<0`` exclusions are NOT a
    leakage signature and must NOT trip the coverage gate (else an anti-leakage corpus
    false-refuses). 20 GLOBAL shuffles with ``hi<0`` and a clean centering -> passes."""
    n = 200
    points = tuple(0.0 for _ in range(n))
    clean = tuple((-0.5, 0.5) for _ in range(n))
    hi_neg = tuple((-0.5, -0.01) if i < 20 else (-0.5, 0.5) for i in range(n))  # hi<0, never lo>0
    result = ShuffleNullResult(
        n_shuffles=n,
        paired_point_estimates={"D>A": points},
        paired_ci_bounds={"D>A": clean},
        global_ci_bounds={"D>A": hi_neg},
    )
    assert isinstance(assert_shuffle_null(result), ShuffleNullReport)


def test_shuffle_null_paired_sign_flip_runs_over_pairs() -> None:
    """The PRIMARY within-pair control: with matched (K, clean) pairs supplied, the shuffle swaps
    members' strata per pair (prob 0.5) and still passes on an honest artifact. Pins that the paired
    path is exercised (not only the global fallback). Uses the WEAK-association fixture (the strong
    one makes the paired null pathologically tight -- the honest ~3% false-refuse regime)."""
    cells = _weak_assoc_cells()
    pairs = [(i, 1000 + i) for i in range(12)]  # bijective matched pairs
    result = shuffle_null_centering(
        cells,
        paired_ids=pairs,
        unpaired_ids=[],
        deltas=_DELTAS,
        n_shuffles=200,
        seed=13,
        n_outer=200,
    )
    assert isinstance(assert_shuffle_null(result), ShuffleNullReport)


def test_shuffle_null_finding1_strong_confound_paired_passes() -> None:
    """Finding-1 regression (the corpus the old fused suite FALSE-REFUSED): an HONEST corpus with a
    STRONG difficulty confound, run through the matched-pair sign-flip, used to drive k=14/200 lo>0
    on the PAIRED tail (ucb95=0.107 > 0.08) and falsely refuse. With the fix, centering passes (the
    sign-flip is confound-robust term-by-term), the paired tail is REPORT-ONLY, and the GLOBAL-draw
    coverage passes -> :func:`assert_shuffle_null` returns a report, no raise.

    Uses the STRONG-association fixture deliberately (the one the old suite dodged with the weak
    fixture): on matched pairs the strong association makes the paired null pathologically tight,
    which is exactly the regime that drove the old false-refuse."""
    cells = _assoc_cells()
    pairs = [(i, 1000 + i) for i in range(12)]  # difficulty-matched (K, clean) pairs
    result = shuffle_null_centering(
        cells,
        paired_ids=pairs,
        unpaired_ids=[],
        deltas=_DELTAS,
        n_shuffles=200,
        seed=29,
        n_outer=200,
    )
    report = assert_shuffle_null(result)  # MUST NOT raise (the old suite false-refused here)
    assert isinstance(report, ShuffleNullReport)
    # Centering is the load-bearing gate and is confound-robust: the paired-δ mean centers at 0.
    pe = result.paired_point_estimates["D>C"]
    assert abs(sum(pe) / len(pe)) < 0.05


def test_shuffle_null_is_reproducible() -> None:
    """Seeded end-to-end: same seed -> identical per-shuffle paired/global point estimates + CIs."""
    cells = _assoc_cells()
    a = shuffle_null_centering(
        cells,
        paired_ids=[],
        unpaired_ids=_UNPAIRED,
        deltas=_DELTAS,
        n_shuffles=50,
        seed=21,
        n_outer=150,
    )
    b = shuffle_null_centering(
        cells,
        paired_ids=[],
        unpaired_ids=_UNPAIRED,
        deltas=_DELTAS,
        n_shuffles=50,
        seed=21,
        n_outer=150,
    )
    assert a.paired_point_estimates == b.paired_point_estimates
    assert a.paired_ci_bounds == b.paired_ci_bounds
    assert a.global_ci_bounds == b.global_ci_bounds


def test_shuffle_null_paired_and_global_draws_are_independent() -> None:
    """Finding-1: the paired (A) and global (B) draws are INDEPENDENT relabelings of one null --
    the B-pass master is salted, so the two artifacts' CI series are not byte-identical (else
    coverage and centering would collapse back onto one artifact, defeating the fix)."""
    cells = _assoc_cells()
    r = shuffle_null_centering(
        cells,
        paired_ids=[],
        unpaired_ids=_UNPAIRED,
        deltas=_DELTAS,
        n_shuffles=50,
        seed=21,
        n_outer=150,
    )
    assert r.paired_ci_bounds["D>C"] != r.global_ci_bounds["D>C"]


# ===========================================================================
# Pod 4.4c-6a -- INSTRUMENT 2: §2.B per-regime contribution bound + LORO
# ===========================================================================
#
# Mutation-resistant: even regimes pass; a regime loaded past 30% share -> CorpusLockError; the ~25%
# knee passes; regime=="" (abstain) is EXCLUDED from shares but COUNTED in the pooled δ; the
# sens-SHARE basis (sens_r * n_r, level-share) is pinned against a pure-per-regime-sens basis (which
# would ignore n_r). LORO never raises even when a regime drop collapses the lo bound.

_REGIMES5 = (
    "logic-wrong",
    "edge-case-miss",
    "spec-misread",
    "silent-degradation",
    "off-by-semantics",
)


def _regime_artifact(
    counts: dict[str, int], *, n_clean: int = 10, d_flags_k: int = 4, R: int = 4
) -> list[Cell]:
    """Build a K+clean artifact with ``counts[regime]`` K items per regime. Arm D flags
    ``d_flags_k`` of R trials on every K item (uniform per-item sens, so a regime's weight scales
    with its item COUNT -- the level-share pin); arm A flags nothing on K (the floor). The clean
    pool is flag-free (a clean spec arm). ``regime`` is welded per item."""
    cells: list[Cell] = []
    iid = 0
    for regime, n in counts.items():
        for _ in range(n):
            for t in range(R):
                cells.append(
                    Cell(
                        item_id=iid,
                        stratum="K",
                        arm="D",
                        trial=t,
                        seed=iid * 10 + t,
                        flagged=1 if t < d_flags_k else 0,
                        route="x",
                        regime=regime,
                    )
                )
                cells.append(
                    Cell(
                        item_id=iid,
                        stratum="K",
                        arm="A",
                        trial=t,
                        seed=iid * 10 + t,
                        flagged=0,
                        route="x",
                        regime=regime,
                    )
                )
            iid += 1
    for _ in range(n_clean):
        for t in range(R):
            cells.append(
                Cell(
                    item_id=iid,
                    stratum="clean",
                    arm="D",
                    trial=t,
                    seed=iid * 10 + t,
                    flagged=0,
                    route="x",
                )
            )
            cells.append(
                Cell(
                    item_id=iid,
                    stratum="clean",
                    arm="A",
                    trial=t,
                    seed=iid * 10 + t,
                    flagged=0,
                    route="x",
                )
            )
        iid += 1
    return cells


def test_regime_contribution_even_regimes_pass() -> None:
    """Five evenly-allocated regimes (uniform 0.20 share each, below 0.30) -> the bound holds; the
    per-regime shares are returned for the artifact log."""
    cells = _regime_artifact(dict.fromkeys(_REGIMES5, 8))
    report = assert_regime_contribution(cells, arm_b="A")
    assert report.max_share == REGIME_CONTRIBUTION_MAX_SHARE
    assert all(abs(s - 0.20) < 1e-9 for s in report.shares.values())
    assert report.abstain_excluded == 0


def test_regime_contribution_loaded_regime_refused() -> None:
    """A regime loaded past 30% of the pooled sensitivity refuses the lock (HIGH-1: a single regime
    carrying more than its fair share loads the pooled J). 20 of 40 items in one regime -> 0.5
    share."""
    counts = {
        "logic-wrong": 20,
        "edge-case-miss": 5,
        "spec-misread": 5,
        "silent-degradation": 5,
        "off-by-semantics": 5,
    }
    cells = _regime_artifact(counts)
    with pytest.raises(CorpusLockError, match=r"exceeding the ceiling 0.3"):
        assert_regime_contribution(cells, arm_b="A")


def test_regime_contribution_knee_at_25pct_passes() -> None:
    """The ~25% knee passes (30% delivers ~2% honest false-refuse; 25% would over-trip if the
    ceiling were tighter). Max regime share is 0.25, below the 0.30 ceiling."""
    counts = {
        "logic-wrong": 10,
        "edge-case-miss": 10,
        "spec-misread": 10,
        "silent-degradation": 5,
        "off-by-semantics": 5,
    }
    cells = _regime_artifact(counts)
    report = assert_regime_contribution(cells, arm_b="A")
    assert abs(max(report.shares.values()) - 0.25) < 1e-9


def test_regime_contribution_abstain_excluded_from_shares_counted_in_pool() -> None:
    """``regime==""`` (K-abstain, §2.C) items are EXCLUDED from the per-regime shares
    (unattributable) but COUNTED in the pooled δ (they remain K error cells). Four attributed
    regimes at 8 each (0.25 share) + 4 abstain items (4/36 ~ 11% < the 0.15 cap) -> shares omit "",
    abstain_excluded == 4."""
    counts = {
        "logic-wrong": 8,
        "edge-case-miss": 8,
        "spec-misread": 8,
        "silent-degradation": 8,
        "": 4,
    }
    cells = _regime_artifact(counts)
    report = assert_regime_contribution(cells, arm_b="A")
    assert "" not in report.shares
    assert set(report.shares) == set(_REGIMES5[:4])
    assert report.abstain_excluded == 4


def test_regime_contribution_uses_level_share_not_per_regime_sens() -> None:
    """MUTATION TEST: the basis is sens-SHARE (``sens_r * n_r`` / pooled, eval-stats Q-C), NOT the
    per-regime sensitivity alone. With UNIFORM per-item sens across regimes, a regime with double
    the items must carry double the share -- a pure-per-regime-sens basis (ignoring n_r) would
    assign every regime an EQUAL share and miss the count-loading. One regime with 16 vs three of 8
    -> its share = 16/40 = 0.40 (> 0.30) -> refused; a per-regime-sens-only basis would see equal
    sens and pass. Pins level-share."""
    counts = {"logic-wrong": 16, "edge-case-miss": 8, "spec-misread": 8, "silent-degradation": 8}
    cells = _regime_artifact(counts)
    # Sanity: per-item sens is uniform across regimes (same d_flags_k), so a per-regime-sens-only
    # basis would assign equal shares (0.25 each) and PASS -- the level-share basis refuses.
    with pytest.raises(CorpusLockError, match=r"exceeding the ceiling 0.3"):
        assert_regime_contribution(cells, arm_b="A")


def test_regime_contribution_abstain_dump_refused() -> None:
    """Finding-2: the abstain-fraction cap. A 60%-dump evasion -- 30 of 50 K items mis-tagged
    ``regime==""`` (loaded into the unattributable bucket to evade the per-regime contribution
    bound) -- exceeds the 0.15 ceiling -> refuse lock. The 20 attributed items spread under the
    share ceiling, so WITHOUT the cap this corpus would pass the contribution bound while hiding a
    loaded regime in abstain."""
    counts = {
        "logic-wrong": 5,
        "edge-case-miss": 5,
        "spec-misread": 5,
        "silent-degradation": 5,
        "": 30,
    }
    cells = _regime_artifact(counts)  # n_error = 50, abstain fraction = 30/50 = 0.60
    with pytest.raises(CorpusLockError, match=r"regime-abstain.*exceeds the ceiling 0.15"):
        assert_regime_contribution(cells, arm_b="A")


def test_regime_contribution_honest_small_abstain_passes() -> None:
    """Finding-2 negative control: an honest small abstain fraction (<=15%) PASSES. 5 attributed
    regimes at 8 each (40 attributed) + 6 abstain -> 6/46 ~ 13% < 0.15 -> the cap does not fire."""
    counts = {
        "logic-wrong": 8,
        "edge-case-miss": 8,
        "spec-misread": 8,
        "silent-degradation": 8,
        "off-by-semantics": 8,
        "": 6,
    }
    cells = _regime_artifact(counts)  # n_error = 46, abstain fraction = 6/46 ~ 0.130
    report = assert_regime_contribution(cells, arm_b="A")
    assert report.abstain_excluded == 6


def test_regime_contribution_abstain_cap_mutation_removing_it_lets_dump_pass() -> None:
    """Finding-2 MUTATION: removing the cap (modeled by relaxing ``abstain_max_fraction`` to 1.0)
    lets the 60%-dump evasion PASS -- proving the cap is what refuses it (an assertion that can't be
    made to NOT fire tests nothing). With the cap defeated, the 20 attributed items (each at 0.25
    share, below 0.30) clear the contribution bound and the loaded abstain bucket slips through."""
    counts = {
        "logic-wrong": 5,
        "edge-case-miss": 5,
        "spec-misread": 5,
        "silent-degradation": 5,
        "": 30,
    }
    cells = _regime_artifact(counts)
    report = assert_regime_contribution(cells, arm_b="A", abstain_max_fraction=1.0)
    assert report.abstain_excluded == 30  # the dump slips through once the cap is defeated


def test_regime_contribution_rejects_non_default_arm_b() -> None:
    """Finding-3: a non-default ``arm_b`` is REFUSED -- the sens-share basis is pinned on the D>A
    reduction (J_A ≡ 0 -> δ = sens_D); the C' paired-δ basis is NOT wired (S12/YAGNI). ``arm_b="C"``
    -> raise, naming the ruling. The default ``"A"`` works."""
    cells = _regime_artifact(dict.fromkeys(_REGIMES5, 8))
    with pytest.raises(CorpusLockError, match=r"arm_b='C'.*NOT wired.*Flag to eval-stats"):
        assert_regime_contribution(cells, arm_b="C")
    # the default basis still works
    report = assert_regime_contribution(cells)  # arm_b defaults to "A"
    assert report.max_share == REGIME_CONTRIBUTION_MAX_SHARE


def test_regime_contribution_refuses_no_error_cells() -> None:
    """An artifact with no error-stratum cells has nothing to audit -> refuse."""
    cells = [
        Cell(
            item_id=9000 + i,
            stratum="clean",
            arm="D",
            trial=t,
            seed=i * 10 + t,
            flagged=0,
            route="x",
        )
        for i in range(5)
        for t in range(4)
    ]
    with pytest.raises(CorpusLockError, match="no 'K' error-stratum cells"):
        assert_regime_contribution(cells, arm_b="A")


def test_regime_contribution_refuses_degenerate_all_miss_arm() -> None:
    """If the error arm flags NOTHING on the attributed regimes, the pooled sensitivity weight is 0
    and no contribution share is defined -> refuse (a degenerate all-miss arm cannot be cleared)."""
    cells = _regime_artifact(dict.fromkeys(_REGIMES5[:2], 8), d_flags_k=0)
    with pytest.raises(CorpusLockError, match="pooled sensitivity weight"):
        assert_regime_contribution(cells, arm_b="A")


def test_loro_returns_report_never_raises() -> None:
    """LORO is REPORTED-ONLY: it returns a :class:`LoroReport` (full_lo + per-regime collapse flags)
    and NEVER raises -- even when a regime drop collapses the lo bound (the ~30% honest false-refuse
    is why LORO does not gate; the contribution bound is the actual single-regime defense)."""
    cells = _regime_artifact(dict.fromkeys(_REGIMES5, 8))
    report = report_loro(cells, arm_b="A", n_outer=200, seed=1)
    assert set(report.collapses) == set(_REGIMES5)
    assert isinstance(report.full_lo, float)


def test_loro_does_not_raise_even_when_lo_collapses() -> None:
    """Explicit collapse case: a corpus whose binding lo sits just above 0 will tip <=0 when a
    regime is dropped -- LORO REPORTS the collapse (collapses[r] True) but still does NOT raise."""
    # A loaded corpus where dropping the dominant regime collapses the remaining effect's lo.
    counts = {"logic-wrong": 16, "edge-case-miss": 4, "spec-misread": 4, "silent-degradation": 4}
    cells = _regime_artifact(counts)
    report = report_loro(cells, arm_b="A", n_outer=300, seed=2)  # must not raise
    assert isinstance(report.collapses, dict)


# ===========================================================================
# Pod 4.4c-6a -- INSTRUMENT 3: §3.3 KS flag-rate diagnostic (reported-only)
# ===========================================================================
#
# Lock-INDEPENDENT: ks_flag_rate_diagnostic returns a KSReport and NEVER refuses, regardless of the
# KS value (the matched-sibling construction is the difficulty control; KS on flag rate is circular
# as a gate -- MED-1). Pins: it returns a report; a large KS gap does NOT raise.


def test_ks_returns_report_never_blocks() -> None:
    """KS is reported-only: it returns a :class:`KSReport` with the one-sided statistic + D_crit and
    NEVER refuses, regardless of the KS value."""
    cells = _assoc_cells()
    report = ks_flag_rate_diagnostic(cells, arm="D")
    assert report.n_error == 12 and report.n_clean == 12
    assert report.d_crit > 0.0
    # The arm-D K-vs-clean flag-rate gap is LARGE here (K flags ~5/6, clean ~1/6) -> exceeds_crit,
    # yet the call returns normally (it is a logged diagnostic, never a refusal).
    assert report.exceeds_crit is True


def test_ks_large_gap_does_not_raise() -> None:
    """A maximal KS gap (K always flags, clean never) does NOT raise -- the diagnostic is
    lock-independent by construction."""
    cells: list[Cell] = []
    for i in range(8):
        cells += [
            Cell(item_id=i, stratum="K", arm="C", trial=t, seed=i * 10 + t, flagged=1, route="x")
            for t in range(4)
        ]
    for i in range(8):
        cells += [
            Cell(
                item_id=100 + i,
                stratum="clean",
                arm="C",
                trial=t,
                seed=i * 10 + t,
                flagged=0,
                route="x",
            )
            for t in range(4)
        ]
    report = ks_flag_rate_diagnostic(cells, arm="C")  # must not raise
    assert report.ks_statistic == 1.0  # the maximal one-sided gap


def test_ks_degenerate_pool_returns_finite_report() -> None:
    """An empty stratum -> a zero statistic + an inf D_crit, returned (not raised) -- the degenerate
    nothing-to-compare case is logged, never a refusal."""
    cells = [
        Cell(item_id=i, stratum="K", arm="C", trial=t, seed=i * 10 + t, flagged=1, route="x")
        for i in range(5)
        for t in range(4)
    ]  # no clean cells
    report = ks_flag_rate_diagnostic(cells, arm="C")
    assert report.n_clean == 0
    assert report.ks_statistic == 0.0
