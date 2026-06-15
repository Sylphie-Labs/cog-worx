"""Unit tests for assemble() min_per_kind assembly-floor feature (Pod 2.6).

All tests are pure — no substrate, no async, no model calls (S1).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.assembly import assemble
from cogworx.recall.results import (
    ChannelHit,
    FusedResult,
)
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode
from cogworx.substrate.latent import LatentMatch, LatentRecord

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _scored_claim(subject: str) -> ScoredClaim:
    cid = claim_id_for(subject, "is", "thing")
    claim = Claim(
        id=cid,
        subject=subject,
        predicate="is",
        payload="thing",
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


def _latent_match(record_id: str) -> LatentMatch:
    return LatentMatch(
        record=LatentRecord(id=record_id, embedding=(1.0, 0.0, 0.0)),
        score=0.9,
        tier="hot",
        use_count=0,
        last_used_at=_T0,
    )


def _episode_item(episode_id: str) -> Episode:
    return Episode(
        episode_id=episode_id,
        run_id="run-01",
        step_index=0,
        turn_index=0,
        session_id="sess-1",
        role="user",
        content="hello",
        kind="conversational",
        occurred_at=_T0,
    )


def _hit(channel: str = "dense", rank: int = 1) -> ChannelHit:
    return ChannelHit(channel=channel, rank=rank, raw_score=0.9)


def _result(
    key: str,
    text: str,
    *,
    kind: Literal["claim", "episode", "latent"] = "claim",
    fused_rank: int = 1,
    fused_score: float | None = None,
    hits: tuple[ChannelHit, ...] | None = None,
) -> FusedResult:
    if hits is None:
        hits = (_hit(),)
    if fused_score is None:
        fused_score = 1.0 / fused_rank
    if kind == "claim":
        item: ScoredClaim | Episode | LatentMatch = _scored_claim(key)
    elif kind == "episode":
        item = _episode_item(key)
    else:
        item = _latent_match(key)
    return FusedResult(
        key=key,
        kind=kind,
        item=item,
        text=text,
        hits=hits,
        fused_score=fused_score,
        fused_rank=fused_rank,
    )


# ---------------------------------------------------------------------------
# Fixture: mixed-kind sequence (episodes dominate in count, one latent minority)
# ---------------------------------------------------------------------------
# Results in relevance order:
#   rank 1: episode  "ep-1"   (score 1.00)
#   rank 2: episode  "ep-2"   (score 0.50)
#   rank 3: episode  "ep-3"   (score 0.33)
#   rank 4: latent   "lat-1"  (score 0.25) — minority kind
#   rank 5: episode  "ep-4"   (score 0.20)
#
# With a modest budget (say 4 tokens each → 4 tokens total → only 1 fits at full budget)
# and min_per_kind=0, the greedy loop admits only ep-1.
# With min_per_kind=1, ep-1 and lat-1 are floor-admitted (if budget allows both).


def _mixed_fixture() -> list[FusedResult]:
    """5 items: 4 episodes + 1 latent, all single-token texts."""
    return [
        _result("ep-1", "a" * 4, kind="episode", fused_rank=1),
        _result("ep-2", "b" * 4, kind="episode", fused_rank=2),
        _result("ep-3", "c" * 4, kind="episode", fused_rank=3),
        _result("lat-1", "d" * 4, kind="latent", fused_rank=4),
        _result("ep-4", "e" * 4, kind="episode", fused_rank=5),
    ]


# ---------------------------------------------------------------------------
# 1. Golden test: min_per_kind=0 is byte-identical to default (no floor)
# ---------------------------------------------------------------------------


def test_golden_min_per_kind_zero_identical_to_default() -> None:
    """min_per_kind=0 must produce byte-identical AssembledContext to the call without it."""
    results = _mixed_fixture()
    ctx_default = assemble(results, budget=200)
    ctx_zero = assemble(results, budget=200, min_per_kind=0)
    # Compare the full pydantic model dumps (fields, values, ordering) — byte-identical semantics.
    assert ctx_default.model_dump() == ctx_zero.model_dump()


def test_golden_min_per_kind_zero_identical_large_budget() -> None:
    """Repeat the golden test with an all-fits budget to confirm no side-effects."""
    results = _mixed_fixture()
    ctx_default = assemble(results, budget=10_000)
    ctx_zero = assemble(results, budget=10_000, min_per_kind=0)
    assert ctx_default.model_dump() == ctx_zero.model_dump()


# ---------------------------------------------------------------------------
# 2. Floor admits 1 of each kind when episodes would dominate otherwise
# ---------------------------------------------------------------------------


def test_floor_admits_minority_kind() -> None:
    """With tight budget and min_per_kind=1, the latent minority is admitted alongside ep-1."""
    # approx_tokens("a"*4) = max(1, 4//4) = 1; same for all items.
    # Budget=3 fits 3 items; greedy-only would take ep-1, ep-2, ep-3 (all episodes).
    # With min_per_kind=1: floor = [ep-1, ep-2, ep-3, ep-4 as episodes... wait — floor picks
    # first 1 of each kind: ep-1 (episode), lat-1 (latent).  Then remaining = ep-2, ep-3, ep-4.
    # Floor admitted: ep-1 (1 tok), lat-1 (1 tok) = 2 tokens used; remaining budget = 1.
    # Greedy on remaining: ep-2 fits (1 tok) → admitted.
    results = _mixed_fixture()
    ctx = assemble(results, budget=3, min_per_kind=1)
    admitted_keys = {c.key for c in ctx.chunks}
    assert "ep-1" in admitted_keys, "Best episode should be admitted"
    assert "lat-1" in admitted_keys, "Latent minority must be floor-admitted"
    assert len(ctx.chunks) == 3  # ep-1 (floor) + lat-1 (floor) + ep-2 (greedy)


def test_floor_both_kinds_with_budget_for_all() -> None:
    """With generous budget, floor + greedy admits all 5 items."""
    results = _mixed_fixture()
    ctx = assemble(results, budget=100, min_per_kind=1)
    assert len(ctx.chunks) == 5
    admitted_keys = {c.key for c in ctx.chunks}
    assert "lat-1" in admitted_keys


# ---------------------------------------------------------------------------
# 3. Mutation control: min_per_kind=0 does NOT admit minority kind when budget is tight
# ---------------------------------------------------------------------------


def test_mutation_control_min_per_kind_zero_no_minority() -> None:
    """min_per_kind=0 on the same fixture does NOT force-admit lat-1 when episodes fill budget."""
    results = _mixed_fixture()
    # Budget=3, min_per_kind=0: greedy admits ep-1, ep-2, ep-3 (all episodes, each 1 token).
    ctx = assemble(results, budget=3, min_per_kind=0)
    admitted_keys = {c.key for c in ctx.chunks}
    assert "lat-1" not in admitted_keys, (
        "Without floor, the minority latent should be crowded out by episodes"
    )
    assert len(ctx.chunks) == 3
    for key in ("ep-1", "ep-2", "ep-3"):
        assert key in admitted_keys


# ---------------------------------------------------------------------------
# 4. Floor item that individually exceeds budget is skipped (no crash)
# ---------------------------------------------------------------------------


def test_floor_item_exceeds_budget_is_skipped_not_crash() -> None:
    """A floor item whose token cost > remaining budget is silently skipped."""
    # ep-big: 400 chars → approx_tokens = 100 tokens — too big for budget=5.
    # lat-small: 4 chars → 1 token — fits.
    ep_big = _result("ep-big", "x" * 400, kind="episode", fused_rank=1)
    lat_small = _result("lat-small", "y" * 4, kind="latent", fused_rank=2)
    results = [ep_big, lat_small]

    # Floor for min_per_kind=1: floor_items = [ep-big, lat-small] (first 1 of each kind).
    # ep-big: 100 tokens > budget=5 → skipped (dropped), not exception.
    # lat-small: 1 token ≤ 5 → admitted.
    ctx = assemble(results, budget=5, min_per_kind=1)
    admitted_keys = {c.key for c in ctx.chunks}
    assert "lat-small" in admitted_keys
    assert "ep-big" not in admitted_keys
    assert ctx.dropped == 1


def test_floor_item_exceeds_budget_no_exception_all_oversized() -> None:
    """All floor items exceed budget → empty result, no exception."""
    big_episode = _result("ep-x", "x" * 4000, kind="episode", fused_rank=1)
    big_latent = _result("lat-x", "y" * 4000, kind="latent", fused_rank=2)
    ctx = assemble([big_episode, big_latent], budget=1, min_per_kind=1)
    assert ctx.chunks == ()
    assert ctx.dropped == 2


# ---------------------------------------------------------------------------
# 5. U-fold ordering is preserved after floor admission
# ---------------------------------------------------------------------------


def test_ufold_preserved_with_floor_two_items() -> None:
    """U-fold: admitted order rank-1 → position 0, rank-2 → position 1 (m=2 case)."""
    ep_item = _result("ep-a", "a" * 4, kind="episode", fused_rank=1)
    lat_item = _result("lat-b", "b" * 4, kind="latent", fused_rank=2)
    results = [ep_item, lat_item]

    ctx = assemble(results, budget=100, min_per_kind=1)
    assert len(ctx.chunks) == 2
    # U-fold for m=2: rank-1 → pos 0, rank-2 → pos 1.
    by_key = {c.key: c for c in ctx.chunks}
    assert ctx.chunks[0].key == "ep-a"
    assert ctx.chunks[1].key == "lat-b"
    assert by_key["ep-a"].relevance_rank == 1
    assert by_key["lat-b"].relevance_rank == 2


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6])
def test_ufold_invariant_with_floor(m: int) -> None:
    """U-fold rank→position invariant holds across all admitted sizes when min_per_kind=1."""
    # Alternate episodes and latents so floor picks one of each (when m >= 2).
    results = [
        _result(
            f"k{i}",
            "a" * 4,
            kind="episode" if i % 2 == 0 else "latent",
            fused_rank=i + 1,
        )
        for i in range(m)
    ]
    ctx = assemble(results, budget=m * 10, min_per_kind=1)
    assert len(ctx.chunks) == m

    rank_to_pos = {c.relevance_rank: pos for pos, c in enumerate(ctx.chunks)}
    front_ptr = 0
    back_ptr = m - 1
    for i in range(m):
        rank = i + 1
        if i % 2 == 0:
            assert rank_to_pos[rank] == front_ptr, (
                f"rank {rank} expected pos {front_ptr}, got {rank_to_pos[rank]}"
            )
            front_ptr += 1
        else:
            assert rank_to_pos[rank] == back_ptr, (
                f"rank {rank} expected pos {back_ptr}, got {rank_to_pos[rank]}"
            )
            back_ptr -= 1


# ---------------------------------------------------------------------------
# 6. min_per_kind > 1: admit up to N per kind
# ---------------------------------------------------------------------------


def test_floor_admits_two_per_kind() -> None:
    """min_per_kind=2 admits the best 2 of each kind before greedy."""
    # 3 episodes and 2 latents, each 1 token, budget=4.
    results = [
        _result("ep-1", "a" * 4, kind="episode", fused_rank=1),
        _result("ep-2", "b" * 4, kind="episode", fused_rank=2),
        _result("lat-1", "c" * 4, kind="latent", fused_rank=3),
        _result("ep-3", "d" * 4, kind="episode", fused_rank=4),
        _result("lat-2", "e" * 4, kind="latent", fused_rank=5),
    ]
    # Floor with min_per_kind=2: ep-1, ep-2 (episodes, first 2), lat-1, lat-2 (latents, first 2).
    # Floor ordering (by original relevance): ep-1, ep-2, lat-1, ep-3 goes to remaining,
    # lat-2 is 4th item overall — wait, let's trace:
    # i=0 ep-1: episode count=0 < 2 → floor
    # i=1 ep-2: episode count=1 < 2 → floor
    # i=2 lat-1: latent count=0 < 2 → floor
    # i=3 ep-3: episode count=2 >= 2 → remaining
    # i=4 lat-2: latent count=1 < 2 → floor
    # floor = [ep-1, ep-2, lat-1, lat-2], remaining = [ep-3]
    # Budget=4: floor admits ep-1(1), ep-2(1), lat-1(1), lat-2(1) = 4 tokens exactly.
    ctx = assemble(results, budget=4, min_per_kind=2)
    admitted_keys = {c.key for c in ctx.chunks}
    assert "ep-1" in admitted_keys
    assert "ep-2" in admitted_keys
    assert "lat-1" in admitted_keys
    assert "lat-2" in admitted_keys
    assert "ep-3" not in admitted_keys  # no budget left after floor
    assert ctx.dropped == 1
