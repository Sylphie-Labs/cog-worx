"""Deterministic unit tests for the live-run idempotency cache (Pod 4.4-live L3).

Pure, offline, deterministic — no model, no docker, no journal, no network. Covers:

  - a cache HIT returns the stored outcome and makes ZERO calls to the wrapped executor.
  - kill-mid-run resume: a fresh wrapper reloaded over a partially-populated cache file skips
    already-answered cells and only calls the inner executor for the missing ones.
  - a fingerprint-digest mismatch is a cold cache (no hits on stale-fingerprint entries), because
    the digest is PART of the cache key, not a load-time filter.
  - a malformed / torn trailing JSONL line (a kill mid-write) is tolerated on reload, not fatal.
  - round-tripping through ``run_arms`` with a cached executor emits the same Cells as uncached.
"""

from __future__ import annotations

from pathlib import Path

from cogworx.eval._live.cache import cached_executor
from cogworx.eval.corpus import (
    CorpusItem,
    DifficultyMarker,
    LLMPlanterStamp,
    OracleLabelProvenance,
)
from cogworx.eval.runner import ArmExecutor, ArmInput, ArmOutcome, run_arms
from cogworx.verification.contracts import OracleFrame, Thesis

_FP_A = "fpdigest-aaaa"
_FP_B = "fpdigest-bbbb"


def _arm_input(item_id: int) -> ArmInput:
    return ArmInput(
        item_id=item_id,
        problem_statement="sum a list",
        completion_criterion="tests_pass",
        problem_type="code",
        proposed_solution="return sum(xs)",
        test_code="assert f([1, 2]) == 3",
    )


class _CountingExecutor:
    """A stub :class:`ArmExecutor` that counts its calls and flags a fixed item_id set."""

    def __init__(self, flag_item_ids: frozenset[int] = frozenset()) -> None:
        self.calls: list[tuple[int, int]] = []
        self._flag_item_ids = flag_item_ids

    def __call__(self, arm_input: ArmInput, seed: int) -> ArmOutcome:
        self.calls.append((arm_input.item_id, seed))
        flagged = 1 if arm_input.item_id in self._flag_item_ids else 0
        return ArmOutcome(flagged=flagged, route="flag" if flagged else "pass")


# ===========================================================================
# Cache HIT -> zero inner calls
# ===========================================================================


def test_cache_hit_returns_stored_outcome_without_calling_inner(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner = _CountingExecutor(flag_item_ids=frozenset({1}))
    wrapped = cached_executor(inner, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path)

    first = wrapped(_arm_input(1), 42)
    assert inner.calls == [(1, 42)]
    assert first.flagged == 1

    second = wrapped(_arm_input(1), 42)
    assert inner.calls == [(1, 42)]  # unchanged — the second call was a cache HIT
    assert second.flagged == first.flagged
    assert second.route == first.route


def test_cache_hit_on_reload_makes_zero_inner_calls(tmp_path: Path) -> None:
    """A FRESH wrapper reloaded over an already-populated cache never touches a fresh inner."""
    cache_path = tmp_path / "cache.jsonl"
    warm_inner = _CountingExecutor(flag_item_ids=frozenset({5}))
    warm = cached_executor(warm_inner, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path)
    warm(_arm_input(5), 7)
    assert warm_inner.calls == [(5, 7)]

    cold_inner = _CountingExecutor(flag_item_ids=frozenset())  # would flag nothing if called
    reloaded = cached_executor(cold_inner, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path)
    outcome = reloaded(_arm_input(5), 7)

    assert cold_inner.calls == []  # ZERO calls — served entirely from the reloaded cache
    assert outcome.flagged == 1  # the stored (warm) outcome, not what a live cold_inner would say


# ===========================================================================
# Kill-mid-run resume
# ===========================================================================


def test_kill_mid_run_resume_skips_completed_cells_calls_missing_ones(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner_a = _CountingExecutor()
    wrapped_a = cached_executor(inner_a, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path)

    # Simulate a run that completed (item_id=1, seed=10) and (item_id=2, seed=20) before being
    # killed; (item_id=3, seed=30) was never reached.
    wrapped_a(_arm_input(1), 10)
    wrapped_a(_arm_input(2), 20)
    assert inner_a.calls == [(1, 10), (2, 20)]

    # Resume: a fresh wrapper + a fresh inner over the SAME cache file.
    inner_b = _CountingExecutor()
    wrapped_b = cached_executor(inner_b, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path)

    wrapped_b(_arm_input(1), 10)  # completed — must be a HIT
    wrapped_b(_arm_input(2), 20)  # completed — must be a HIT
    wrapped_b(_arm_input(3), 30)  # missing — must call inner_b

    assert inner_b.calls == [(3, 30)]  # only the genuinely-missing cell touched the inner executor


# ===========================================================================
# Fingerprint-digest mismatch -> cold cache
# ===========================================================================


def test_fingerprint_mismatch_is_a_cold_cache(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    inner_old = _CountingExecutor(flag_item_ids=frozenset({9}))
    wrapped_old = cached_executor(
        inner_old, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path
    )
    wrapped_old(_arm_input(9), 3)
    assert inner_old.calls == [(9, 3)]

    # A re-lock stamps a NEW fingerprint digest; same (item_id, arm, seed), same file.
    inner_new = _CountingExecutor(flag_item_ids=frozenset())  # would flag nothing if called
    wrapped_new = cached_executor(
        inner_new, arm="D", fingerprint_digest=_FP_B, cache_path=cache_path
    )
    outcome = wrapped_new(_arm_input(9), 3)

    assert inner_new.calls == [(9, 3)]  # a real call was made — the stale-fingerprint entry MISSED
    assert outcome.flagged == 0  # inner_new's own answer, not the stale fingerprint_A entry's


# ===========================================================================
# Torn trailing JSONL line tolerated on reload
# ===========================================================================


def test_torn_trailing_line_is_tolerated_not_fatal(tmp_path: Path) -> None:
    cache_path = tmp_path / "cache.jsonl"
    good = (
        f'{{"item_id": 1, "arm": "D", "seed": 11, "fingerprint_digest": "{_FP_A}", '
        f'"flagged": 1, "route": "flag"}}\n'
    )
    torn = '{"item_id": 2, "arm": "D", "seed": 22, "fingerp'  # a kill mid-write, no trailing \n
    cache_path.write_text(good + torn, encoding="utf-8")

    inner = _CountingExecutor(flag_item_ids=frozenset())
    wrapped = cached_executor(inner, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path)

    hit = wrapped(_arm_input(1), 11)
    assert inner.calls == []  # the well-formed line loaded fine
    assert hit.flagged == 1

    miss = wrapped(_arm_input(2), 22)
    assert inner.calls == [(2, 22)]  # the torn line was skipped, not honored as a stale hit
    assert miss.flagged == 0


# ===========================================================================
# Round-trip through run_arms
# ===========================================================================


def _corpus_item(item_id: int, *, is_error: int) -> CorpusItem:
    provenance = OracleLabelProvenance(
        returncode=1 if is_error else 0,
        test_provenance="frozen",
        holds=not is_error,
        valid_check=True,
        oracle_id="x",
    )
    return CorpusItem(
        item_id=item_id,
        frame=OracleFrame(
            completion_criterion="tests_pass", problem_type="code", problem_statement="sum a list"
        ),
        thesis=Thesis(proposed_solution="return sum(xs)", experiment_design="run frozen tests"),
        test_code="assert f([1, 2]) == 3",
        is_error=is_error,
        label_source="oracle",
        label_provenance=provenance,
        stratum="O" if is_error else "clean",
        oracle_reachable=True,
        error_regime="wrong-op" if is_error else "",
        difficulty=DifficultyMarker(planted_difficulty="medium", surface_complexity=12),
        matched_sibling_id=None,
        split="measurement",
        planter=LLMPlanterStamp(model_family="planterfam", model_id="p1"),
    )


def test_run_arms_round_trip_cached_matches_uncached(tmp_path: Path) -> None:
    corpus = [_corpus_item(1, is_error=1), _corpus_item(2, is_error=0)]
    uncached_inner = _CountingExecutor(flag_item_ids=frozenset({1}))
    uncached_cells = run_arms(corpus, arm_executors={"D": uncached_inner}, R=3)

    cache_path = tmp_path / "cache.jsonl"
    cached_inner: ArmExecutor = _CountingExecutor(flag_item_ids=frozenset({1}))
    cached = cached_executor(
        cached_inner, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path
    )
    cached_cells = run_arms(corpus, arm_executors={"D": cached}, R=3)

    assert cached_cells == uncached_cells

    # A SECOND pass over the same cache file must be served entirely from cache.
    replay_inner = _CountingExecutor(flag_item_ids=frozenset())  # would disagree if called live
    replay = cached_executor(
        replay_inner, arm="D", fingerprint_digest=_FP_A, cache_path=cache_path
    )
    replay_cells = run_arms(corpus, arm_executors={"D": replay}, R=3)

    assert replay_inner.calls == []
    assert replay_cells == uncached_cells
