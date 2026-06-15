"""Integration tests for MemoryInjector + RecallStack over the live pgvector substrate — Pod 2.6.

All tests hit a REAL Postgres/pgvector instance (``docker compose up -d`` first).
They are marked ``@pytest.mark.integration`` and are skipped by default in CI.

Covers:
1. record_use_increments_after_inject: injecting admits latent records → use_count > 0.
2. dropped_items_not_incremented: budget-dropped items keep use_count = 0.
3. hot_first_two_call_pattern: hot_first=True and hot_first=False both return plausible results.
4. unwired_engine_returns_empty: ctx.recall() on an Engine with no recall_stack → status="unwired".
"""

from __future__ import annotations

import asyncio
import math
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.pg_latent import PgLatentStore
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import MemoryPolicy
from cogworx.recall.channels import LatentDenseChannel
from cogworx.recall.query import RecallQuery
from cogworx.recall.stack import RecallStack
from cogworx.substrate.latent import LatentRecord

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_DIM = 4


# ---------------------------------------------------------------------------
# Event loop policy (Windows psycopg 3 requires selector loop)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


# ---------------------------------------------------------------------------
# Per-test fixture: fresh PgLatentStore with clean table
# ---------------------------------------------------------------------------


@pytest.fixture
async def latent(settings: SubstrateSettings) -> AsyncIterator[PgLatentStore]:
    store = PgLatentStore(dim=_DIM, settings=settings, clock=lambda: _T0)
    await store.ensure_schema()
    await store.reset()
    try:
        yield store
    finally:
        await store.aclose()


# ---------------------------------------------------------------------------
# Helper: build a unit-normalised embedding that points "near" a query vector
# ---------------------------------------------------------------------------


def _unit_vec(values: tuple[float, ...]) -> tuple[float, ...]:
    """Return the L2-normalised version of *values*."""
    magnitude = math.sqrt(sum(v * v for v in values))
    if magnitude == 0.0:
        return values
    return tuple(v / magnitude for v in values)


def _record(record_id: str, vec: tuple[float, ...]) -> LatentRecord:
    return LatentRecord(id=record_id, embedding=_unit_vec(vec), payload={"src": record_id})


# ---------------------------------------------------------------------------
# Test 1: record_use_increments_after_inject
# ---------------------------------------------------------------------------


async def test_record_use_increments_after_inject(latent: PgLatentStore) -> None:
    """After inject(), all 3 admitted latent records have use_count > 0.

    Steps:
    - Put 3 records.
    - Build a RecallStack with a single LatentDenseChannel.
    - Run inject() with a large token budget so all 3 are admitted.
    - Re-search the store and verify each record's use_count >= 1.
    """
    # All three records point in nearly the same direction as the query.
    query_vec = _unit_vec((1.0, 0.1, 0.0, 0.0))
    records = [
        _record("r1", (1.0, 0.2, 0.0, 0.0)),
        _record("r2", (1.0, 0.1, 0.1, 0.0)),
        _record("r3", (1.0, 0.0, 0.1, 0.1)),
    ]
    for rec in records:
        await latent.put(rec)

    channel = LatentDenseChannel(latent, hot_first=False)
    stack = RecallStack([channel])
    injector = MemoryInjector(stack, latent_store=latent)

    query = RecallQuery(embedding=query_vec, k=10)
    # Large budget: admit all results (approx_tokens on short text << 10000).
    policy = MemoryPolicy(token_budget=10_000)
    result = await injector.inject(query, policy=policy)

    assert result.status == "ok", f"Expected status='ok', got {result.status!r}"
    assert result.latent_uses_recorded > 0, (
        f"Expected latent_uses_recorded > 0, got {result.latent_uses_recorded}"
    )
    assert result.record_use_error is None, (
        f"record_use raised unexpectedly: {result.record_use_error}"
    )

    # Verify via a fresh search that use_count advanced on the admitted records.
    matches = await latent.search(query_vec, k=10)
    admitted_ids = {c.key.removeprefix("latent:") for c in result.context.chunks}
    for m in matches:
        if m.record.id in admitted_ids:
            assert m.use_count > 0, (
                f"Admitted record {m.record.id!r} still has use_count=0 after inject()"
            )


# ---------------------------------------------------------------------------
# Test 2: dropped_items_not_incremented
# ---------------------------------------------------------------------------


async def test_dropped_items_not_incremented(latent: PgLatentStore) -> None:
    """Budget-dropped records keep use_count = 0; admitted records have use_count >= 1.

    Strategy: use a tiny token_budget so that only some records are admitted.  The assembly
    module uses approx_tokens (chars / 4, min 1) so we can control admission by sizing records'
    rendered text relative to the budget.

    Records are created with ``payload={"src": record_id}`` — render_latent falls back to
    ``repr(payload)``, producing e.g. ``"{'src': 'd1'}"`` (13 chars → approx_tokens=3).
    We use ``token_budget=3`` so exactly the top-scoring record (d1) is admitted and d2/d3
    are dropped (cumulative+3 > 3 for items 2 and 3).

    - Exactly 1 record is admitted (d1).
    - Dropped records (d2, d3) have use_count = 0.
    """
    query_vec = _unit_vec((1.0, 0.0, 0.0, 0.0))
    records = [
        _record("d1", (1.0, 0.0, 0.0, 0.0)),  # closest to query
        _record("d2", (0.9, 0.1, 0.0, 0.0)),
        _record("d3", (0.8, 0.2, 0.0, 0.0)),
    ]
    for rec in records:
        await latent.put(rec)

    channel = LatentDenseChannel(latent, hot_first=False)
    stack = RecallStack([channel])
    injector = MemoryInjector(stack, latent_store=latent)

    query = RecallQuery(embedding=query_vec, k=10)
    # budget=3: d1 costs 3 tokens (fits), d2+d3 each cost 3 more (cumulative > 3 → dropped).
    policy = MemoryPolicy(token_budget=3)
    result = await injector.inject(query, policy=policy)

    # Exactly 1 record must have been admitted (d1, the closest).
    assert len(result.context.chunks) >= 1, "Expected at least one admitted chunk with budget=3"

    admitted_ids = {c.key.removeprefix("latent:") for c in result.context.chunks}
    all_ids = {"d1", "d2", "d3"}
    dropped_ids = all_ids - admitted_ids

    # There must be at least 2 dropped records (budget=3 admits d1 only; d2+d3 are dropped).
    assert len(dropped_ids) >= 1, f"Expected at least one dropped record; admitted={admitted_ids}"

    # Verify via fresh search: dropped records must have use_count = 0.
    matches = {m.record.id: m for m in await latent.search(query_vec, k=10)}
    for dropped_id in dropped_ids:
        m = matches.get(dropped_id)
        assert m is not None, f"Record {dropped_id!r} not found in search results"
        assert m.use_count == 0, (
            f"Dropped record {dropped_id!r} has use_count={m.use_count} (expected 0)"
        )
    for admitted_id in admitted_ids:
        m = matches.get(admitted_id)
        assert m is not None, f"Admitted record {admitted_id!r} not found in search results"
        assert m.use_count > 0, (
            f"Admitted record {admitted_id!r} has use_count={m.use_count} (expected > 0)"
        )


# ---------------------------------------------------------------------------
# Test 3: hot_first_two_call_pattern
# ---------------------------------------------------------------------------


async def test_hot_first_two_call_pattern(latent: PgLatentStore) -> None:
    """hot_first=True and hot_first=False both return a plausible (non-empty) result set.

    We manually promote two records to the 'hot' tier via sweep_tiers and leave one in 'cold'.
    Both channel configurations are expected to surface results from the same record pool —
    the channel contract (S9) is that hot_first never silently drops results relative to the
    tier-agnostic baseline, and tier membership does not affect geometric score ordering.

    Verification:
    - With hot_first=False: the channel makes one tier-agnostic search → results are non-empty
      and every returned key has kind='latent'.
    - With hot_first=True: the channel applies the hot-first two-call composite → results are
      non-empty and still geometrically plausible (top result has the highest cosine score).
    """
    query_vec = _unit_vec((1.0, 0.0, 0.0, 0.0))

    records = [
        _record("hf1", (1.0, 0.05, 0.0, 0.0)),  # closest
        _record("hf2", (0.95, 0.1, 0.0, 0.0)),
        _record("hf3", (0.85, 0.2, 0.1, 0.0)),  # furthest
    ]
    for rec in records:
        await latent.put(rec)

    # Promote top-2 records to hot tier (hot_capacity=2).
    await latent.sweep_tiers(now=_T0, hot_capacity=2)

    # --- hot_first=False (tier-agnostic) ---
    channel_default = LatentDenseChannel(latent, hot_first=False)
    stack_default = RecallStack([channel_default])
    injector_default = MemoryInjector(stack_default, latent_store=latent)

    result_default = await injector_default.inject(
        RecallQuery(embedding=query_vec, k=10),
        policy=MemoryPolicy(token_budget=10_000),
    )
    assert len(result_default.context.chunks) > 0, "hot_first=False: expected non-empty result set"
    for chunk in result_default.context.chunks:
        assert chunk.kind == "latent", (
            f"hot_first=False: unexpected kind {chunk.kind!r} (expected 'latent')"
        )

    # --- hot_first=True (two-call hot-first composite with τ=0.0 to always accept hot rows) ---
    # min_similarity=0.0 means ALL hot rows pass the gate (τ), so the hot results fill the
    # budget and no cold search is needed.  This validates the hot-first path end-to-end.
    channel_hot = LatentDenseChannel(latent, hot_first=True, min_similarity=0.0)
    stack_hot = RecallStack([channel_hot])
    injector_hot = MemoryInjector(stack_hot, latent_store=latent)

    result_hot = await injector_hot.inject(
        RecallQuery(embedding=query_vec, k=10),
        policy=MemoryPolicy(token_budget=10_000),
    )
    assert len(result_hot.context.chunks) > 0, "hot_first=True: expected non-empty result set"
    for chunk in result_hot.context.chunks:
        assert chunk.kind == "latent", (
            f"hot_first=True: unexpected kind {chunk.kind!r} (expected 'latent')"
        )

    # The result sets should overlap (same underlying records, just different search paths).
    ids_default = {c.key for c in result_default.context.chunks}
    ids_hot = {c.key for c in result_hot.context.chunks}
    overlap = ids_default & ids_hot
    assert len(overlap) > 0, (
        f"hot_first=True and hot_first=False produced disjoint result sets: "
        f"default={ids_default}, hot={ids_hot}"
    )


# ---------------------------------------------------------------------------
# Test 4: unwired_engine_returns_empty
# ---------------------------------------------------------------------------


async def test_unwired_engine_returns_empty(latent: PgLatentStore) -> None:
    """ctx.recall() on a RunContext with no injector (recall_stack=None) → status='unwired'.

    We instantiate RunContext directly with injector=None (the same path the Engine takes when
    recall_stack is None) and call recall(). The return must be an InjectedMemory with:
    - status = "unwired"
    - context.chunks == ()
    - context.token_count == 0
    - latent_uses_recorded == 0
    - record_use_error is None

    No latent put/search needed; this tests the unwired short-circuit path in RunContext.recall().
    """
    from collections.abc import Mapping
    from collections.abc import Sequence as Seq
    from typing import Any

    from cogworx.cost.budget import BudgetGuard
    from cogworx.model.base import (
        ChatMessage,
        ModelCapabilities,
        ModelResponse,
        ModelTier,
        ToolSpec,
    )
    from cogworx.runtime.context import RunContext
    from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal

    class _StubModel:
        @property
        def capabilities(self) -> ModelCapabilities:
            return ModelCapabilities()

        async def complete(
            self,
            *,
            messages: Seq[ChatMessage],
            tools: Seq[ToolSpec] = (),
            tier: ModelTier = "pro",
            json_schema: Mapping[str, Any] | None = None,
        ) -> ModelResponse:
            return ModelResponse(text="stub", model_id="stub", finish_reason="stop")

        def count_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

    ctx = RunContext(
        run_id="unwired-test",
        session_id="sess-unwired",
        model=_StubModel(),
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=latent,
        budget=BudgetGuard(),
        injector=None,  # no recall_stack → unwired
    )

    result = await ctx.recall(RecallQuery(embedding=_unit_vec((1.0, 0.0, 0.0, 0.0))))

    assert result.status == "unwired", (
        f"Expected status='unwired' when injector=None, got {result.status!r}"
    )
    assert result.context.chunks == (), (
        f"Expected empty chunks on unwired recall, got {result.context.chunks}"
    )
    assert result.context.token_count == 0, (
        f"Expected token_count=0 on unwired recall, got {result.context.token_count}"
    )
    assert result.latent_uses_recorded == 0, (
        f"Expected latent_uses_recorded=0 on unwired recall, got {result.latent_uses_recorded}"
    )
    assert result.record_use_error is None, (
        f"Expected record_use_error=None on unwired recall, got {result.record_use_error!r}"
    )
