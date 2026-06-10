"""Unit tests for cogworx.recall.assembly (Pod 2.5 Stream E).

All tests are pure — no substrate, no async, no model calls (S1).
"""

from __future__ import annotations

import sys
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.assembly import assemble
from cogworx.recall.results import (
    AssembledContext,
    ChannelHit,
    FusedResult,
)
from cogworx.substrate.entity_kg import ScoredClaim

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


def _hit(channel: str = "dense", rank: int = 1) -> ChannelHit:
    return ChannelHit(channel=channel, rank=rank, raw_score=0.9)


def _result(
    key: str,
    text: str,
    *,
    fused_rank: int = 1,
    hits: tuple[ChannelHit, ...] | None = None,
) -> FusedResult:
    if hits is None:
        hits = (_hit(),)
    return FusedResult(
        key=key,
        kind="claim",
        item=_scored_claim(key),
        text=text,
        hits=hits,
        fused_score=1.0 / fused_rank,
        fused_rank=fused_rank,
    )


# ---------------------------------------------------------------------------
# 1. Empty input
# ---------------------------------------------------------------------------


def test_empty_input() -> None:
    ctx = assemble([], budget=100)
    assert isinstance(ctx, AssembledContext)
    assert ctx.chunks == ()
    assert ctx.token_count == 0
    assert ctx.dropped == 0
    assert ctx.budget == 100


# ---------------------------------------------------------------------------
# 2. Budget=0 — all items skipped
# ---------------------------------------------------------------------------


def test_budget_zero_skips_all() -> None:
    results = [_result(f"k{i}", "hello", fused_rank=i + 1) for i in range(4)]
    ctx = assemble(results, budget=0)
    assert ctx.chunks == ()
    assert ctx.dropped == len(results)


# ---------------------------------------------------------------------------
# 3. All items fit
# ---------------------------------------------------------------------------


def test_all_items_fit() -> None:
    texts = ["a", "bb", "ccc"]
    results = [_result(f"k{i}", t, fused_rank=i + 1) for i, t in enumerate(texts)]
    # approx_tokens("a") = max(1, 0) = 1; all are 1 each — total 3
    ctx = assemble(results, budget=100)
    assert ctx.token_count <= 100
    assert len(ctx.chunks) == len(results)
    assert ctx.dropped == 0


# ---------------------------------------------------------------------------
# 4. Greedy-skip (not greedy-stop): small item after large is admitted
# ---------------------------------------------------------------------------


def test_greedy_skip_not_stop() -> None:
    # "x" * 800 → approx_tokens = 200 (> budget of 100)
    large = _result("large", "x" * 800, fused_rank=1)
    # "y" * 4 → approx_tokens = 1
    small = _result("small", "y" * 4, fused_rank=2)
    ctx = assemble([large, small], budget=100)
    assert len(ctx.chunks) == 1
    assert ctx.chunks[0].key == "small"
    assert ctx.dropped == 1


# ---------------------------------------------------------------------------
# 5. dropped count
# ---------------------------------------------------------------------------


def test_dropped_count_exact() -> None:
    # budget = 10 tokens; each "x"*40 = 10 tokens exactly
    # first two fit (0+10=10, 10+10=20 > 10 so only first fits); rest are skipped
    results = [_result(f"k{i}", "x" * 40, fused_rank=i + 1) for i in range(5)]
    ctx = assemble(results, budget=10)
    admitted = len(ctx.chunks)
    assert ctx.dropped == len(results) - admitted


# ---------------------------------------------------------------------------
# 6. relevance_rank is 1-based in admission order (not original position)
# ---------------------------------------------------------------------------


def test_relevance_rank_admission_order() -> None:
    # k1 fits, k2 is too large (skipped), k3 fits — ranks should be 1 and 2
    k1 = _result("k1", "a" * 4, fused_rank=1)    # approx_tokens = 1
    k2 = _result("k2", "b" * 800, fused_rank=2)  # approx_tokens = 200 (skipped)
    k3 = _result("k3", "c" * 4, fused_rank=3)    # approx_tokens = 1
    ctx = assemble([k1, k2, k3], budget=10)

    by_key = {c.key: c for c in ctx.chunks}
    assert "k1" in by_key
    assert "k3" in by_key
    assert "k2" not in by_key

    assert by_key["k1"].relevance_rank == 1
    assert by_key["k3"].relevance_rank == 2


# ---------------------------------------------------------------------------
# 7. U-fold invariant (property): ranks 1,2,3,4... map to positions 0,m-1,1,m-2,...
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [1, 2, 3, 4, 5, 6, 7, 8])
def test_ufold_invariant(m: int) -> None:
    # Build m items that each cost 1 token; budget is generous.
    results = [_result(f"k{i}", "a" * 4, fused_rank=i + 1) for i in range(m)]
    ctx = assemble(results, budget=m * 10)
    assert len(ctx.chunks) == m

    # Build mapping: relevance_rank → position in output
    rank_to_pos = {c.relevance_rank: pos for pos, c in enumerate(ctx.chunks)}

    # Verify: rank 1 → pos 0, rank 2 → pos m-1, rank 3 → pos 1, rank 4 → pos m-2, ...
    front_ptr = 0
    back_ptr = m - 1
    for i in range(m):
        rank = i + 1  # 1-based
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
# 8. U-fold single item
# ---------------------------------------------------------------------------


def test_ufold_single_item() -> None:
    r = _result("solo", "hello", fused_rank=1)
    ctx = assemble([r], budget=100)
    assert len(ctx.chunks) == 1
    assert ctx.chunks[0].key == "solo"
    assert ctx.chunks[0].relevance_rank == 1


# ---------------------------------------------------------------------------
# 9. U-fold two items: [r0, r1] → output [r0, r1]
# ---------------------------------------------------------------------------


def test_ufold_two_items() -> None:
    r0 = _result("r0", "a" * 4, fused_rank=1)
    r1 = _result("r1", "b" * 4, fused_rank=2)
    ctx = assemble([r0, r1], budget=100)
    assert len(ctx.chunks) == 2
    # r0 (rank 1, even i=0) → front pos 0; r1 (rank 2, odd i=1) → back pos 1
    assert ctx.chunks[0].key == "r0"
    assert ctx.chunks[1].key == "r1"


# ---------------------------------------------------------------------------
# 10. S5 provenance passthrough: chunk.hits equals fused_result.hits
# ---------------------------------------------------------------------------


def test_s5_provenance_passthrough() -> None:
    h1 = _hit("dense", 1)
    h2 = _hit("bm25", 2)
    r = _result("prov-test", "some text", hits=(h1, h2))
    ctx = assemble([r], budget=100)
    assert len(ctx.chunks) == 1
    assert ctx.chunks[0].hits == (h1, h2)
    # Identical tuple — same object references inside
    assert ctx.chunks[0].hits[0] is h1
    assert ctx.chunks[0].hits[1] is h2


# ---------------------------------------------------------------------------
# 11. S1 purity: importing assembly does not trigger model or substrate imports
# ---------------------------------------------------------------------------


def test_s1_no_substrate_import_on_module_load() -> None:
    """Verify that a fresh import of cogworx.recall.assembly does not pull in live drivers.

    Uses a subprocess to get a clean sys.modules — the main pytest process may have
    already loaded drivers via sibling test modules, making in-process checks unreliable.
    """
    import subprocess

    script = (
        "import sys; "
        "import cogworx.recall.assembly; "
        "forbidden = ['neo4j', 'asyncpg', 'psycopg', "
        "'cogworx.adapters.neo4j_entity_kg', 'cogworx.adapters.pg_episodes']; "
        "bad = [m for m in forbidden if m in sys.modules]; "
        "print('BAD:' + ','.join(bad) if bad else 'OK')"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    output = result.stdout.strip()
    assert output == "OK", (
        f"S1 violation: fresh import of cogworx.recall.assembly pulled in live drivers: {output}"
    )


# ---------------------------------------------------------------------------
# 12. Custom token counter (char-level budget)
# ---------------------------------------------------------------------------


def test_custom_token_counter_char_level() -> None:
    # Use char-level counter: len(s)
    char_counter = lambda s: len(s)  # noqa: E731

    r1 = _result("k1", "hello", fused_rank=1)     # 5 chars
    r2 = _result("k2", "world!", fused_rank=2)     # 6 chars
    r3 = _result("k3", "x", fused_rank=3)          # 1 char

    # budget=6: "hello"(5) fits, "world!"(6) exceeds (5+6=11>6), "x"(1) fits (5+1=6)
    ctx = assemble([r1, r2, r3], budget=6, count_tokens=char_counter)

    admitted_keys = {c.key for c in ctx.chunks}
    assert "k1" in admitted_keys
    assert "k3" in admitted_keys
    assert "k2" not in admitted_keys
    assert ctx.dropped == 1
    assert ctx.token_count == 6  # 5 + 1
