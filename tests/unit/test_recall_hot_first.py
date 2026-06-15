"""Unit tests for LatentDenseChannel hot-first gate (Pod 2.6).

All tests are pure-Python: no database, no model, no network.
asyncio_mode = "auto" (pyproject.toml), so no @pytest.mark.asyncio needed.

Key invariants verified:
  hot_first=False  → single tier-agnostic call (original S3 behaviour, unchanged).
  hot_first=True   → two-call composite; sub-τ hot rows fall back into geometry pool (D10).
  CF-1 bug dead    → hot 0.30 is NOT ranked above cold 0.90 when hot_first=True.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from cogworx.recall.channels import LatentDenseChannel
from cogworx.recall.query import RecallQuery
from cogworx.recall.stack import default_recall_stack
from cogworx.substrate.latent import LatentMatch, LatentRecord, Tier, TierSweepResult
from cogworx.testing.doubles import InMemoryEntityKG

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_EMBED_A: tuple[float, ...] = (1.0, 0.0, 0.0)
_EMBED_B: tuple[float, ...] = (0.0, 1.0, 0.0)


# ---------------------------------------------------------------------------
# LatentStore spy double
# ---------------------------------------------------------------------------


class SpyLatentStore:
    """Minimal LatentStore double that records all search() calls.

    ``search_results`` maps ``tier`` value (or ``None``) to the sequence of
    ``LatentMatch`` objects to return for that tier.  A missing key returns [].
    """

    def __init__(
        self,
        search_results: dict[Tier | None, Sequence[LatentMatch]] | None = None,
    ) -> None:
        self._results: dict[Tier | None, Sequence[LatentMatch]] = search_results or {}
        self.search_calls: list[dict[str, Any]] = []

    async def put(self, record: LatentRecord) -> None:
        pass

    async def record_use(self, ids: Sequence[str]) -> int:
        return 0

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        tier: Tier | None = None,
    ) -> Sequence[LatentMatch]:
        self.search_calls.append({"embedding": tuple(embedding), "k": k, "tier": tier})
        return list(self._results.get(tier, []))

    async def sweep_tiers(self, *, now: datetime, hot_capacity: int) -> TierSweepResult:
        return TierSweepResult(promoted=0, demoted=0, hot_size=0)


def _match(record_id: str, score: float, tier: Tier = "hot") -> LatentMatch:
    return LatentMatch(
        record=LatentRecord(id=record_id, embedding=_EMBED_A),
        score=score,
        tier=tier,
        use_count=0,
        last_used_at=_T0,
    )


# ---------------------------------------------------------------------------
# 1. hot_first=False: exactly one tier-agnostic search call
# ---------------------------------------------------------------------------


async def test_hot_first_false_single_search_call() -> None:
    """Default hot_first=False issues exactly one search with tier=None."""
    spy = SpyLatentStore(search_results={None: [_match("lat-1", 0.9, "hot")]})
    ch = LatentDenseChannel(spy)  # hot_first=False by default
    q = RecallQuery(embedding=_EMBED_A, k=5)
    results = await ch.search(q)

    assert len(spy.search_calls) == 1
    assert spy.search_calls[0]["tier"] is None
    assert len(results) == 1
    assert results[0].key == "latent:lat-1"


async def test_hot_first_false_explicit_same_as_default() -> None:
    """hot_first=False explicit is identical to default."""
    spy = SpyLatentStore(search_results={None: [_match("lat-2", 0.7, "cold")]})
    ch = LatentDenseChannel(spy, hot_first=False)
    await ch.search(RecallQuery(embedding=_EMBED_A, k=3))

    assert len(spy.search_calls) == 1
    assert spy.search_calls[0]["tier"] is None


# ---------------------------------------------------------------------------
# 2. CF-1 inversion fixture: hot 0.30, cold 0.90 → cold ranked above hot
# ---------------------------------------------------------------------------


async def test_inversion_fixture_cold_ranked_above_sub_tau_hot() -> None:
    """CF-1 bug dead: sub-τ hot row (0.30) must NOT appear above cold row (0.90).

    With hot_first=True and min_similarity=0.80:
    - hot search returns hot_low (0.30) — below τ.
    - cold search returns cold_high (0.90).
    - Pool = [hot_low(0.30)] + [cold_high(0.90)], sorted desc by score.
    - Final: accepted(empty) + pool[:k] = [cold_high(0.90), hot_low(0.30)].
    """
    hot_low = _match("hot-low", 0.30, "hot")
    cold_high = _match("cold-high", 0.90, "cold")
    spy = SpyLatentStore(
        search_results={
            "hot": [hot_low],
            "cold": [cold_high],
        }
    )
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=5))

    assert len(results) == 2
    # cold_high (0.90) must come first — D10 correctness.
    assert results[0].key == "latent:cold-high", (
        "CF-1 inversion bug: sub-τ hot row ranked above higher-score cold row"
    )
    assert results[1].key == "latent:hot-low"
    # Two search calls: one hot, one cold.
    tiers_called = [c["tier"] for c in spy.search_calls]
    assert "hot" in tiers_called
    assert "cold" in tiers_called


# ---------------------------------------------------------------------------
# 3. hot 0.85 (≥ τ) ranks above cold 0.90 — intended bounded bias
# ---------------------------------------------------------------------------


async def test_above_tau_hot_ranks_first_then_cold() -> None:
    """A hot row ≥ τ is accepted first; cold rows fill the remaining budget."""
    hot_accepted = _match("hot-good", 0.85, "hot")
    cold_higher = _match("cold-higher", 0.90, "cold")
    spy = SpyLatentStore(
        search_results={
            "hot": [hot_accepted],
            "cold": [cold_higher],
        }
    )
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=5))

    assert len(results) == 2
    # hot_accepted (0.85) is accepted and placed FIRST (before the pool).
    assert results[0].key == "latent:hot-good"
    # cold_higher (0.90) fills the remainder.
    assert results[1].key == "latent:cold-higher"


# ---------------------------------------------------------------------------
# 4. All-hot-sub-τ → ids+order matches tier-agnostic global search (fallback equivalence)
# ---------------------------------------------------------------------------


async def test_all_hot_sub_tau_falls_back_to_global_geometry() -> None:
    """When all hot rows are sub-τ, result is equivalent to the tier-agnostic geometry pool.

    hot: [h1=0.40, h2=0.30] — both sub-τ (τ=0.80).
    cold: [c1=0.95, c2=0.70].
    Expected final: accepted=[] + pool sorted desc = [c1=0.95, h1=0.40, c2=0.35... wait—
    We must sort rejected_hot + cold_results by score descending.
    rejected_hot = [h1(0.40), h2(0.30)], cold = [c1(0.95), c2(0.70)].
    pool = sorted([h1(0.40), h2(0.30), c1(0.95), c2(0.70)], desc) = [c1, c2, h1, h2].
    Final (k=4): [c1(0.95), c2(0.70), h1(0.40), h2(0.30)].
    """
    h1 = _match("h1", 0.40, "hot")
    h2 = _match("h2", 0.30, "hot")
    c1 = _match("c1", 0.95, "cold")
    c2 = _match("c2", 0.70, "cold")
    spy = SpyLatentStore(
        search_results={
            "hot": [h1, h2],
            "cold": [c1, c2],
        }
    )
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=4))

    assert len(results) == 4
    result_keys = [r.key for r in results]
    assert result_keys == ["latent:c1", "latent:c2", "latent:h1", "latent:h2"]


# ---------------------------------------------------------------------------
# 5. Hot fills k at ≥ τ → cold search was never called
# ---------------------------------------------------------------------------


async def test_hot_fills_k_cold_never_called() -> None:
    """When k hot rows ≥ τ exist, cold search must never be called."""
    hot_rows = [_match(f"hot-{i}", 0.85 + i * 0.01, "hot") for i in range(3)]
    spy = SpyLatentStore(search_results={"hot": hot_rows, "cold": []})
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=3))

    assert len(results) == 3
    tiers_called = [c["tier"] for c in spy.search_calls]
    assert "cold" not in tiers_called, (
        "Cold search must NOT be called when hot tier fills k at or above τ"
    )
    assert "hot" in tiers_called


async def test_hot_fills_k_cold_never_called_exact_k() -> None:
    """Verify the >= boundary: exactly k hot rows ≥ τ → cold not called."""
    hot_rows = [_match(f"h{i}", 0.90, "hot") for i in range(5)]
    spy = SpyLatentStore(search_results={"hot": hot_rows})
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=5))

    assert len(results) == 5
    tiers_called = [c["tier"] for c in spy.search_calls]
    assert "cold" not in tiers_called


async def test_hot_fills_more_than_k_truncated_to_k() -> None:
    """Hot returns more than k rows ≥ τ; result must be capped at k."""
    hot_rows = [_match(f"h{i}", 0.90 - i * 0.001, "hot") for i in range(10)]
    spy = SpyLatentStore(search_results={"hot": hot_rows})
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=4))

    assert len(results) == 4


# ---------------------------------------------------------------------------
# 6. default_recall_stack passthrough: latent_hot_first kwarg reaches the channel
# ---------------------------------------------------------------------------


async def test_default_recall_stack_latent_hot_first_passthrough() -> None:
    """latent_hot_first=True reaches LatentDenseChannel and triggers the hot-first path."""
    # Build a spy store: hot returns one above-τ row, cold is never needed.
    hot_row = _match("lat-stack", 0.90, "hot")
    spy = SpyLatentStore(search_results={"hot": [hot_row], None: [hot_row]})

    kg = InMemoryEntityKG()
    stack = default_recall_stack(
        entity_kg=kg,
        latent_store=spy,
        latent_hot_first=True,
        latent_min_similarity=0.80,
    )

    q = RecallQuery(embedding=_EMBED_A, k=1)
    await stack.recall(q)

    # Verify that the latent channel's search call used tier="hot" (hot_first path).
    tiers_called = [c["tier"] for c in spy.search_calls]
    assert "hot" in tiers_called, (
        "latent_hot_first=True did not reach LatentDenseChannel — hot tier was never queried"
    )


async def test_default_recall_stack_latent_hot_first_false_tier_agnostic() -> None:
    """latent_hot_first=False (default) uses tier=None (tier-agnostic) search."""
    spy = SpyLatentStore(search_results={None: [_match("lat-default", 0.7, "cold")]})
    kg = InMemoryEntityKG()
    stack = default_recall_stack(entity_kg=kg, latent_store=spy, latent_hot_first=False)

    q = RecallQuery(embedding=_EMBED_A, k=1)
    await stack.recall(q)

    tiers_called = [c["tier"] for c in spy.search_calls]
    assert all(t is None for t in tiers_called), (
        "latent_hot_first=False must use only tier=None (tier-agnostic) search"
    )


# ---------------------------------------------------------------------------
# 7. Rank invariant: results are always 1-based ranked in score order
# ---------------------------------------------------------------------------


async def test_results_ranked_1_based() -> None:
    """All returned RecallResults have 1-based contiguous ranks."""
    h1 = _match("h1", 0.91, "hot")
    c1 = _match("c1", 0.60, "cold")
    spy = SpyLatentStore(search_results={"hot": [h1], "cold": [c1]})
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=5))

    for i, r in enumerate(results):
        assert r.hit.rank == i + 1


async def test_hot_first_false_results_ranked_1_based() -> None:
    """hot_first=False path also produces 1-based contiguous ranks."""
    matches = [_match(f"m{i}", 0.9 - i * 0.1, "hot") for i in range(4)]
    spy = SpyLatentStore(search_results={None: matches})
    ch = LatentDenseChannel(spy, hot_first=False)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10))

    for i, r in enumerate(results):
        assert r.hit.rank == i + 1


# ---------------------------------------------------------------------------
# 8. min_similarity kwarg propagation
# ---------------------------------------------------------------------------


async def test_min_similarity_custom_threshold() -> None:
    """Custom τ=0.50 accepts hot rows ≥ 0.50."""
    h_above = _match("h-above", 0.55, "hot")  # ≥ 0.50 → accepted
    h_below = _match("h-below", 0.40, "hot")  # < 0.50 → rejected, goes to pool
    spy = SpyLatentStore(search_results={"hot": [h_above, h_below], "cold": []})
    ch = LatentDenseChannel(spy, hot_first=True, min_similarity=0.50)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=5))

    result_keys = [r.key for r in results]
    # h_above accepted first; h_below re-enters pool but nothing else → appended after
    assert result_keys[0] == "latent:h-above"
    assert "latent:h-below" in result_keys
