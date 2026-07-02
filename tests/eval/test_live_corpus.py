"""Deterministic tests for the 1-key bring-up corpus (Pod 4.4-live L5).

Pure, offline, deterministic — no model, no docker, no network, no journal. The only external touch
is the REAL frozen-oracle kernel (:func:`~cogworx.eval._authoring.run_frozen_check`, a local
``pytest`` subprocess) that :func:`build_bring_up_corpus` runs by default — so this file is slower
than a pure-stdlib unit test (subprocess-per-item) but still fully deterministic and offline. Keep
the per-test pair counts small; this is NOT the spike tier, but it is not free either.

Covers:
  1. the built corpus passes ``load_corpus`` G1-G4 (measurement split) and locks (``lock_corpus`` +
     ``build_fingerprint`` succeed, non-empty digest).
  2. ``residual_epsilon`` is the ``RESIDUAL_EPSILON_UNAUDITED`` sentinel by design.
  3. the binding-K pool (the scorer's own Tier-4 firewall: K minus detK minus converted-O) is
     provably EMPTY — checked both directly on the locked truth table AND by driving the real
     ``cogworx.eval.scorer._binding_k_pool`` helper over a minimal synthetic ``Cell`` bridge.
  4. determinism: same seed -> byte-identical item ids/hashes/digest; a different seed -> a
     different ``content_hash`` (and therefore fingerprint digest).
  5. the corpus carries non-empty clean + deterministic-O + detK strata.
"""

from __future__ import annotations

import pytest

from cogworx.eval._live.corpus import build_bring_up_corpus
from cogworx.eval.corpus import CorpusItem, DeterministicPlanterStamp, load_corpus
from cogworx.eval.lock import RESIDUAL_EPSILON_UNAUDITED, MeasurementFingerprint, content_hash
from cogworx.eval.scorer import _binding_k_pool
from cogworx.eval.youden import Cell

_GIT_SHA = "test-fixed-sha"
_DEFAULT_SEED = 20260616

# Small pair counts — enough for every stratum to be non-trivial while keeping the per-item
# subprocess pytest cost (~0.2-0.6s each) well under a sane single-file test budget.
_N_O = 4
_N_DETK = 4


def _build(seed: int) -> tuple[list[CorpusItem], MeasurementFingerprint]:
    return build_bring_up_corpus(seed=seed, n_o_pairs=_N_O, n_detk_pairs=_N_DETK, git_sha=_GIT_SHA)


@pytest.fixture(scope="module")
def default_corpus() -> tuple[list[CorpusItem], MeasurementFingerprint]:
    """Built ONCE per module at the default seed — every test below that does not itself probe
    seed-sensitivity shares this build (each build is ~2*(n_o_pairs+n_detk_pairs) real subprocess
    ``pytest`` runs; the determinism tests build fresh on their own, deliberately never sharing this
    fixture, so they still prove purity rather than a cache)."""
    return _build(_DEFAULT_SEED)


def _synthetic_cells(items: list[CorpusItem]) -> list[Cell]:
    """The minimal synthetic ``Cell`` bridge for ``_binding_k_pool`` — one arm-D cell per locked
    item, stratum carried through verbatim. No trial/flag semantics are exercised here; this only
    drives the pool-partition logic (K-stratum minus detK minus converted-O)."""
    return [
        Cell(
            item_id=it.item_id,
            stratum=it.stratum,
            arm="D",
            trial=0,
            seed=it.item_id,
            flagged=0,
            route="pass",
        )
        for it in items
    ]


# ===========================================================================
# 1. load_corpus G1-G4 + lock validity
# ===========================================================================


def test_corpus_passes_load_corpus_measurement_guards(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    locked, fingerprint = default_corpus
    assert locked
    loaded = load_corpus(locked, measurement_run=True)
    assert len(loaded) == len(locked), "every bring-up item is split='measurement' by design"
    assert fingerprint.digest


def test_corpus_locks_with_stable_content_hashes(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    locked, _ = default_corpus
    assert all(it.content_hash != "" for it in locked)
    assert all(content_hash(it) == it.content_hash for it in locked)


def test_tuning_only_load_is_empty(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    """The bring-up corpus is entirely measurement-split (a single-family smoke corpus has no
    tuning/calibration purpose) — the default ``load_corpus`` (tuning-only) load is empty."""
    locked, _ = default_corpus
    assert load_corpus(locked, measurement_run=False) == []


# ===========================================================================
# 2. epsilon sentinel
# ===========================================================================


def test_fingerprint_carries_unaudited_epsilon_sentinel(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    _, fingerprint = default_corpus
    assert fingerprint.residual_epsilon == RESIDUAL_EPSILON_UNAUDITED
    assert fingerprint.epsilon_audited is False


# ===========================================================================
# 3. binding-K pool is provably EMPTY
# ===========================================================================


def test_every_k_stratum_item_is_a_detk_probe(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    """Direct corpus-truth-table proof: no cross-family LLM K pool was ever planted, so every
    K-stratum item in the locked corpus MUST be a detK collusion probe."""
    locked, _ = default_corpus
    k_items = [it for it in locked if it.stratum == "K"]
    assert k_items, "the bring-up corpus must carry SOME K-stratum material (the detK pairs)"
    assert all(it.detk for it in k_items), "a non-detK K item would be a binding-K leak"


def test_binding_k_pool_is_empty_via_the_real_scorer_helper(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    """Drive the scorer's OWN Tier-4 pool-partition helper (``_binding_k_pool``) over a minimal
    synthetic ``Cell`` bridge (no live 4.4d run exists yet for this corpus) — proving the binding K
    pool a real ``score_gate`` run would build is empty of K-stratum cells, so the gate would
    honestly refuse with ``INSTRUMENT_INVALID`` rather than certify a hollow K population."""
    locked, _ = default_corpus
    cells = _synthetic_cells(locked)
    n_k_cells_in = sum(1 for c in cells if c.stratum == "K")
    assert n_k_cells_in > 0, "need K-stratum cells in the input to prove the exclusion has teeth"

    pool, n_conv, n_detk = _binding_k_pool(cells, locked)

    assert n_conv == 0, "the bring-up corpus never runs the K->O converter"
    assert n_detk == n_k_cells_in, "every K-stratum cell must be excluded as a detK probe"
    assert not any(c.stratum == "K" for c in pool), "the binding K pool must be empty"


# ===========================================================================
# 4. determinism
# ===========================================================================


def test_same_seed_is_byte_identical() -> None:
    locked_a, fp_a = _build(seed=777)
    locked_b, fp_b = _build(seed=777)
    assert [it.item_id for it in locked_a] == [it.item_id for it in locked_b]
    assert [it.content_hash for it in locked_a] == [it.content_hash for it in locked_b]
    assert fp_a.digest == fp_b.digest


def test_different_seed_changes_content_hash() -> None:
    locked_a, fp_a = _build(seed=777)
    locked_b, fp_b = _build(seed=778)
    assert [it.content_hash for it in locked_a] != [it.content_hash for it in locked_b]
    assert fp_a.digest != fp_b.digest


# ===========================================================================
# 5. non-empty clean + detO + detK strata
# ===========================================================================


def test_corpus_has_material_in_every_stratum(
    default_corpus: tuple[list[CorpusItem], MeasurementFingerprint],
) -> None:
    locked, _ = default_corpus
    by_stratum: dict[str, list[CorpusItem]] = {"O": [], "K": [], "clean": []}
    for it in locked:
        by_stratum[it.stratum].append(it)

    assert by_stratum["clean"], "no clean items — arm-A / spec-ceiling instruments have no material"
    assert by_stratum["O"], "no deterministic-O items — the arm-A K-floor has no O material"
    assert by_stratum["K"], "no K-stratum (detK) items — the regime instruments have no K material"

    o_operators = {
        op
        for it in by_stratum["O"]
        if isinstance(it.planter, DeterministicPlanterStamp)
        for op in it.planter.operators
    }
    assert o_operators == {"arithmetic-swap", "boundary"}

    k_regimes = {it.error_regime for it in by_stratum["K"]}
    assert k_regimes == {"off-by-semantics"}
