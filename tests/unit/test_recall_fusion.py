"""Unit tests for cogworx.recall.fusion — RRF and subset-permutation guard (Pod 2.5 Stream D).

All tests are pure — no substrate, no async, no model calls (S1, S9).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.fusion import RRF_K_DEFAULT, assert_rerank_subset, fuse
from cogworx.recall.results import ChannelHit, FusedResult, RecallResult
from cogworx.substrate.entity_kg import ScoredClaim

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _scored_claim(key: str) -> ScoredClaim:
    """Build a minimal ScoredClaim whose claim uses key as subject."""
    cid = claim_id_for(key, "is", "fixture")
    claim = Claim(
        id=cid,
        subject=key,
        predicate="is",
        payload="fixture",
        epistemic_type="inference",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    return ScoredClaim(
        claim=claim,
        confidence=claim_confidence([]),
        lineage_min_confidence=1.0,
    )


def make_result(
    key: str,
    channel: str,
    rank: int,
    score: float | None = None,
) -> RecallResult:
    """Build a RecallResult for a claim-kind item.

    Args:
        key: Namespaced key, e.g. ``"claim:abc"``.
        channel: Channel name, e.g. ``"dense"``.
        rank: 1-based within-channel rank.
        score: Raw score for the ChannelHit; None for rank-only channels.
    """
    return RecallResult(
        key=key,
        kind="claim",
        item=_scored_claim(key),
        text=f"text for {key}",
        hit=ChannelHit(channel=channel, rank=rank, raw_score=score),
    )


def _expected_score(rank: int, k: int = RRF_K_DEFAULT) -> float:
    return 1.0 / (k + rank)


# ---------------------------------------------------------------------------
# 1. Single channel, single result
# ---------------------------------------------------------------------------


def test_single_channel_single_result() -> None:
    r1 = make_result("claim:aaa", "dense", 1, score=0.9)
    fused = fuse({"dense": [r1]})

    assert len(fused) == 1
    f = fused[0]
    assert f.key == "claim:aaa"
    assert f.fused_rank == 1
    assert f.fused_score == pytest.approx(_expected_score(1))
    assert len(f.hits) == 1
    assert f.hits[0].channel == "dense"
    assert f.hits[0].rank == 1


# ---------------------------------------------------------------------------
# 2. Single channel, N results — scores decrease monotonically; ranks 1-based
# ---------------------------------------------------------------------------


def test_single_channel_n_results_monotonic() -> None:
    results = [make_result(f"claim:r{i}", "dense", i + 1, score=0.9) for i in range(5)]
    fused = fuse({"dense": results})

    assert len(fused) == 5
    # Ranks are 1-based and contiguous
    assert [f.fused_rank for f in fused] == list(range(1, 6))
    # Scores decrease: rank 1 > rank 2 > ...
    scores = [f.fused_score for f in fused]
    assert scores == sorted(scores, reverse=True)
    # Each hit has exactly one entry
    for f in fused:
        assert len(f.hits) == 1


# ---------------------------------------------------------------------------
# 3. Two channels, no overlap — all independent, correct individual scores
# ---------------------------------------------------------------------------


def test_two_channels_no_overlap() -> None:
    r_dense = make_result("claim:dense", "dense", 1, score=0.9)
    r_bm25 = make_result("claim:bm25", "bm25", 1, score=5.0)
    fused = fuse({"dense": [r_dense], "bm25": [r_bm25]})

    assert len(fused) == 2
    by_key = {f.key: f for f in fused}

    assert by_key["claim:dense"].fused_score == pytest.approx(_expected_score(1))
    assert by_key["claim:bm25"].fused_score == pytest.approx(_expected_score(1))
    assert len(by_key["claim:dense"].hits) == 1
    assert len(by_key["claim:bm25"].hits) == 1


# ---------------------------------------------------------------------------
# 4. Two channels, one overlap — ChannelHit records merged; score summed
# ---------------------------------------------------------------------------


def test_two_channels_one_overlap() -> None:
    # Place the shared item at position 1 (rank 2) in dense and position 2 (rank 3) in bm25.
    # fuse() derives rank from list position, so we pad with non-overlapping items before it.
    r_dense_pad = make_result("claim:dense-pad", "dense", 1, score=0.9)
    r_dense = make_result("claim:shared", "dense", 2, score=0.8)
    r_bm25_pad1 = make_result("claim:bm25-pad1", "bm25", 1, score=5.0)
    r_bm25_pad2 = make_result("claim:bm25-pad2", "bm25", 2, score=4.5)
    r_bm25 = make_result("claim:shared", "bm25", 3, score=4.0)

    fused = fuse(
        {
            "dense": [r_dense_pad, r_dense],       # shared at list pos 1 → rank 2
            "bm25": [r_bm25_pad1, r_bm25_pad2, r_bm25],  # shared at list pos 2 → rank 3
        }
    )

    by_key = {f.key: f for f in fused}
    assert "claim:shared" in by_key
    f = by_key["claim:shared"]
    assert len(f.hits) == 2
    hit_channels = {h.channel for h in f.hits}
    assert hit_channels == {"dense", "bm25"}
    expected = _expected_score(2) + _expected_score(3)
    assert f.fused_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 5. Three channels overlap — score is sum of three contributions
# ---------------------------------------------------------------------------


def test_three_channels_overlap_score_sum() -> None:
    # Place the shared item at:
    #   dense:  pos 0 → rank 1
    #   bm25:   pos 3 → rank 4  (pad with 3 non-overlapping items)
    #   graph:  pos 1 → rank 2  (pad with 1 non-overlapping item)
    r_dense = make_result("claim:shared", "dense", 1, score=0.9)

    r_bm25_pads = [make_result(f"claim:bm25-p{i}", "bm25", i + 1) for i in range(3)]
    r_bm25 = make_result("claim:shared", "bm25", 4, score=3.0)

    r_graph_pad = make_result("claim:graph-p0", "graph", 1, score=None)
    r_graph = make_result("claim:shared", "graph", 2, score=None)

    fused = fuse(
        {
            "dense": [r_dense],
            "bm25": [*r_bm25_pads, r_bm25],
            "graph": [r_graph_pad, r_graph],
        }
    )

    by_key = {f.key: f for f in fused}
    assert "claim:shared" in by_key
    f = by_key["claim:shared"]
    assert len(f.hits) == 3
    expected = _expected_score(1) + _expected_score(4) + _expected_score(2)
    assert f.fused_score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# 6. Deterministic under channel list order permutation
# ---------------------------------------------------------------------------


def test_deterministic_under_channel_order() -> None:
    r_a = make_result("claim:alpha", "chan_a", 1, score=0.9)
    r_b = make_result("claim:beta", "chan_b", 1, score=0.9)

    fused_ab = fuse({"chan_a": [r_a], "chan_b": [r_b]})
    fused_ba = fuse({"chan_b": [r_b], "chan_a": [r_a]})

    assert len(fused_ab) == len(fused_ba)
    for fa, fb in zip(fused_ab, fused_ba, strict=True):
        assert fa.key == fb.key
        assert fa.fused_score == pytest.approx(fb.fused_score)
        assert fa.fused_rank == fb.fused_rank


# ---------------------------------------------------------------------------
# 7. Tie-break by key (key ASC)
# ---------------------------------------------------------------------------


def test_tiebreak_by_key_asc() -> None:
    # Both at rank 1 in their own channel → equal score; key ASC wins
    r_z = make_result("claim:zzz", "chan_a", 1, score=None)
    r_a = make_result("claim:aaa", "chan_b", 1, score=None)
    fused = fuse({"chan_a": [r_z], "chan_b": [r_a]})

    assert len(fused) == 2
    assert fused[0].key == "claim:aaa"
    assert fused[1].key == "claim:zzz"


# ---------------------------------------------------------------------------
# 8. fused_rank is contiguous 1-based; no gaps
# ---------------------------------------------------------------------------


def test_fused_rank_contiguous_one_based() -> None:
    results = [make_result(f"claim:r{i}", "dense", i + 1) for i in range(7)]
    fused = fuse({"dense": results})

    ranks = [f.fused_rank for f in fused]
    assert ranks == list(range(1, len(fused) + 1))


# ---------------------------------------------------------------------------
# 9. S5 — every result has >= 1 hit
# ---------------------------------------------------------------------------


def test_s5_every_result_has_at_least_one_hit() -> None:
    results = [make_result(f"claim:x{i}", "dense", i + 1) for i in range(4)]
    fused = fuse({"dense": results})

    assert all(len(f.hits) >= 1 for f in fused)


# ---------------------------------------------------------------------------
# 10. Dedup text mismatch raises ValueError
# ---------------------------------------------------------------------------


def test_text_mismatch_raises_value_error() -> None:
    key = "claim:ambiguous"
    r_dense = RecallResult(
        key=key,
        kind="claim",
        item=_scored_claim(key),
        text="version A",
        hit=ChannelHit(channel="dense", rank=1, raw_score=0.9),
    )
    r_bm25 = RecallResult(
        key=key,
        kind="claim",
        item=_scored_claim(key),
        text="version B",  # Different text — rendering bug
        hit=ChannelHit(channel="bm25", rank=1, raw_score=3.0),
    )
    with pytest.raises(ValueError, match="rendering-layer bug"):
        fuse({"dense": [r_dense], "bm25": [r_bm25]})


# ---------------------------------------------------------------------------
# 11. Empty input
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    assert fuse({}) == ()


# ---------------------------------------------------------------------------
# 12. Empty channel list in input
# ---------------------------------------------------------------------------


def test_empty_channel_list() -> None:
    assert fuse({"dense": []}) == ()


# ---------------------------------------------------------------------------
# Guard tests — assert_rerank_subset
# ---------------------------------------------------------------------------


def _make_fused(key: str, rank: int = 1, score: float = 0.9) -> FusedResult:
    return FusedResult(
        key=key,
        kind="claim",
        item=_scored_claim(key),
        text=f"text for {key}",
        hits=(ChannelHit(channel="dense", rank=rank, raw_score=score),),
        fused_score=score,
        fused_rank=rank,
    )


@pytest.fixture()
def three_fused() -> tuple[FusedResult, FusedResult, FusedResult]:
    return (
        _make_fused("claim:a", rank=1, score=0.9),
        _make_fused("claim:b", rank=2, score=0.5),
        _make_fused("claim:c", rank=3, score=0.3),
    )


# 13. Valid passthrough — identity is valid
def test_guard_valid_passthrough(
    three_fused: tuple[FusedResult, FusedResult, FusedResult],
) -> None:
    fused = list(three_fused)
    assert_rerank_subset(fused, fused)  # no exception


# 14. Valid subset
def test_guard_valid_subset(
    three_fused: tuple[FusedResult, FusedResult, FusedResult],
) -> None:
    fused = list(three_fused)
    assert_rerank_subset(fused, fused[:2])  # no exception


# 15. Valid reorder
def test_guard_valid_reorder(
    three_fused: tuple[FusedResult, FusedResult, FusedResult],
) -> None:
    fused = list(three_fused)
    assert_rerank_subset(fused, [fused[1], fused[0]])  # no exception


# 16. Injected key raises ValueError
def test_guard_injected_key_raises(
    three_fused: tuple[FusedResult, FusedResult, FusedResult],
) -> None:
    fused = list(three_fused)
    foreign = _make_fused("claim:injected", rank=1, score=0.99)
    with pytest.raises(ValueError, match="not present in original"):
        assert_rerank_subset(fused, [fused[0], foreign])


# 17. Duplicate key raises ValueError
def test_guard_duplicate_key_raises(
    three_fused: tuple[FusedResult, FusedResult, FusedResult],
) -> None:
    fused = list(three_fused)
    with pytest.raises(ValueError, match="duplicate keys"):
        assert_rerank_subset(fused, [fused[0], fused[0]])


# 18. Mutated item raises ValueError
def test_guard_mutated_item_raises(
    three_fused: tuple[FusedResult, FusedResult, FusedResult],
) -> None:
    fused = list(three_fused)
    original = fused[0]
    # Build a FusedResult with the same key but different fused_score (a mutation).
    mutated = FusedResult(
        key=original.key,
        kind=original.kind,
        item=original.item,
        text=original.text,
        hits=original.hits,
        fused_score=original.fused_score + 99.0,  # mutated
        fused_rank=original.fused_rank,
    )
    with pytest.raises(ValueError, match="must not mutate"):
        assert_rerank_subset(fused, [mutated, fused[1]])
