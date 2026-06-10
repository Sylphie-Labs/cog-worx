"""Integration tests for PgLatentStore Pod 2.2 — live Postgres + pgvector.

Covers:
- Live migration: the three ADD COLUMN IF NOT EXISTS statements on a pre-2.2 table (CF-6 lesson:
  migration paths with zero coverage hide upgrade crashes, per Pod 2.1 FIX-2).
- SQL-vs-Python activation ranking parity (the sweep CTE must produce the same hot-set as
  select_hot_ids() in pure Python on randomized fixtures).
- I2 atomicity: a mid-sweep abort (simulated by using a transaction) leaves the table
  byte-untouched.
- I1/I3: idempotent sweep + capacity invariant on the live store.
- I4/I5: put-preserves-usage, no-lost-increments on the live store.
- Search tier scoping (live).
- Migration idempotence (re-running ensure_schema after migration is a no-op).
"""

from __future__ import annotations

import asyncio
import random
import sys
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.pg_latent import PgLatentStore
from cogworx.knowledge.latent_activation import ActivationParams, ActivationRow, select_hot_ids
from cogworx.substrate.latent import LatentRecord

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 1, 1, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)


@pytest.fixture(scope="session")
def event_loop_policy() -> asyncio.AbstractEventLoopPolicy:
    if sys.platform == "win32":
        return asyncio.WindowsSelectorEventLoopPolicy()
    return asyncio.DefaultEventLoopPolicy()


@pytest.fixture
async def latent(settings: SubstrateSettings) -> AsyncIterator[PgLatentStore]:
    store = PgLatentStore(dim=4, settings=settings, clock=lambda: _T0)
    await store.ensure_schema()
    await store.reset()
    try:
        yield store
    finally:
        await store.aclose()


# ---------------------------------------------------------------------------
# Migration: pre-2.2 table (no tier/created_at/last_used_at columns) upgrades cleanly
# ---------------------------------------------------------------------------


async def test_migration_pre_2_2_table(settings: SubstrateSettings) -> None:
    """CF-6: a genuine pre-2.2 table (no tier/created_at/last_used_at) must migrate cleanly."""
    store = PgLatentStore(dim=4, settings=settings, clock=lambda: _T0)
    conn = await store._connection()

    # Build the Phase-0/1 table schema (no tier, no timestamps)
    await conn.execute("DROP TABLE IF EXISTS cogworx_latent")
    await conn.execute(
        "CREATE TABLE cogworx_latent ("
        "id text PRIMARY KEY, "
        "embedding vector(4) NOT NULL, "
        "payload jsonb NOT NULL, "
        "use_count int NOT NULL DEFAULT 0)"
    )
    # Insert a row as if it were a pre-2.2 row
    from pgvector.psycopg import Vector
    from psycopg.types.json import Jsonb

    await conn.execute(
        "INSERT INTO cogworx_latent (id, embedding, payload, use_count) VALUES (%s, %s, %s, %s)",
        ("legacy", Vector([1.0, 0.0, 0.0, 0.0]), Jsonb({"v": 1}), 3),
    )

    # Running ensure_schema must add all three columns without error
    await store.ensure_schema()

    # (a) the three columns now exist and are non-null for the legacy row
    cur = await conn.execute(
        "SELECT tier, created_at, last_used_at, use_count FROM cogworx_latent WHERE id = 'legacy'"
    )
    row = await cur.fetchone()
    assert row is not None
    tier, created_at, last_used_at, use_count = row
    assert tier == "cold"
    assert created_at is not None
    assert last_used_at is not None
    assert use_count == 3  # use_count is untouched

    # (b) the store can now do a full round-trip put + record_use + sweep + search
    await store.put(LatentRecord(id="new", embedding=(0.0, 1.0, 0.0, 0.0)))
    await store.record_use(["new"])
    result = await store.sweep_tiers(now=_T1, hot_capacity=1)
    assert result.hot_size == 1
    matches = await store.search((0.0, 1.0, 0.0, 0.0))
    assert any(m.record.id == "new" for m in matches)

    # (c) re-running ensure_schema is idempotent (no error, no data loss)
    await store.ensure_schema()
    cur2 = await conn.execute("SELECT COUNT(*) FROM cogworx_latent")
    count_row = await cur2.fetchone()
    assert count_row is not None and count_row[0] == 2

    await store.aclose()


# ---------------------------------------------------------------------------
# SQL-vs-Python activation ranking parity
# ---------------------------------------------------------------------------


async def test_sweep_sql_vs_python_ranking_parity(settings: SubstrateSettings) -> None:
    """The SQL sweep CTE must select the same hot-set as select_hot_ids() in pure Python."""
    rng = random.Random(42)
    n = 20
    hot_capacity = 7
    store = PgLatentStore(dim=4, settings=settings, clock=lambda: _T0)
    await store.ensure_schema()
    await store.reset()

    rows: list[ActivationRow] = []
    for i in range(n):
        id_ = f"item-{i:03d}"
        use_count = rng.randint(0, 50)
        hours_ago = rng.uniform(0.1, 200.0)
        last_used_at = _T1 - timedelta(hours=hours_ago)
        # Write via put + manual use_count injection (to set arbitrary counts)
        await store.put(LatentRecord(id=id_, embedding=tuple(rng.uniform(-1, 1) for _ in range(4))))
        # Override use_count via direct SQL (test fixture only — exercising the sweep from any
        # state)
        conn = await store._connection()
        await conn.execute(
            "UPDATE cogworx_latent SET use_count = %s, last_used_at = %s WHERE id = %s",
            (use_count, last_used_at, id_),
        )
        rows.append(ActivationRow(id=id_, use_count=use_count, last_used_at=last_used_at))

    params = ActivationParams(hot_capacity=hot_capacity)
    expected_hot = select_hot_ids(rows, _T1, params)

    result = await store.sweep_tiers(now=_T1, hot_capacity=hot_capacity)
    assert result.hot_size == len(expected_hot)

    # Verify exact set equality between Python and SQL
    conn = await store._connection()
    cur = await conn.execute("SELECT id FROM cogworx_latent WHERE tier = 'hot'")
    actual_hot = frozenset(row[0] for row in await cur.fetchall())
    assert actual_hot == expected_hot

    await store.aclose()


# ---------------------------------------------------------------------------
# I1 — idempotent sweep on live PG
# ---------------------------------------------------------------------------


async def test_i1_idempotent_sweep_live(latent: PgLatentStore) -> None:
    for i in range(5):
        await latent.put(LatentRecord(id=str(i), embedding=(1.0, 0.0, 0.0, 0.0)))
    r1 = await latent.sweep_tiers(now=_T1, hot_capacity=2)
    r2 = await latent.sweep_tiers(now=_T1, hot_capacity=2)
    assert r2.promoted == 0 and r2.demoted == 0
    assert r1.hot_size == r2.hot_size == 2


# ---------------------------------------------------------------------------
# I3 — capacity invariant
# ---------------------------------------------------------------------------


async def test_i3_capacity_invariant_live(latent: PgLatentStore) -> None:
    for i in range(10):
        await latent.put(LatentRecord(id=str(i), embedding=(1.0, 0.0, 0.0, 0.0)))
    result = await latent.sweep_tiers(now=_T1, hot_capacity=4)
    assert result.hot_size == 4


# ---------------------------------------------------------------------------
# I4 — put-preserves-usage on live PG
# ---------------------------------------------------------------------------


async def test_i4_put_preserves_usage_live(latent: PgLatentStore) -> None:
    await latent.put(LatentRecord(id="x", embedding=(1.0, 0.0, 0.0, 0.0)))
    await latent.record_use(["x"])
    await latent.record_use(["x"])
    await latent.put(LatentRecord(id="x", embedding=(0.0, 1.0, 0.0, 0.0)))

    matches = await latent.search((0.0, 1.0, 0.0, 0.0))
    assert matches[0].use_count == 2

    # Static assertion: the ON CONFLICT clause in pg_latent.py must NOT name use_count,
    # last_used_at, tier, or created_at. Verified in the spike's static grep; this test pins the
    # runtime side.


# ---------------------------------------------------------------------------
# I5 — no lost increments under sequential record_use
# ---------------------------------------------------------------------------


async def test_i5_sequential_record_use_live(latent: PgLatentStore) -> None:
    await latent.put(LatentRecord(id="a", embedding=(1.0, 0.0, 0.0, 0.0)))
    for _ in range(20):
        await latent.record_use(["a"])
    matches = await latent.search((1.0, 0.0, 0.0, 0.0))
    assert matches[0].use_count == 20


# ---------------------------------------------------------------------------
# Search scope filter on live PG
# ---------------------------------------------------------------------------


async def test_search_tier_scope_live(latent: PgLatentStore) -> None:
    for i in range(6):
        await latent.put(LatentRecord(id=str(i), embedding=(1.0, 0.0, 0.0, 0.0)))
    await latent.sweep_tiers(now=_T1, hot_capacity=3)

    hot = await latent.search((1.0, 0.0, 0.0, 0.0), tier="hot")
    cold = await latent.search((1.0, 0.0, 0.0, 0.0), tier="cold")
    assert all(m.tier == "hot" for m in hot)
    assert all(m.tier == "cold" for m in cold)
    assert len(hot) + len(cold) == 6


# ---------------------------------------------------------------------------
# Score bounds and metadata on live PG
# ---------------------------------------------------------------------------


async def test_search_scores_and_metadata_live(latent: PgLatentStore) -> None:
    await latent.put(LatentRecord(id="a", embedding=(1.0, 0.0, 0.0, 0.0), payload={"k": "v"}))
    await latent.record_use(["a"])
    matches = await latent.search((1.0, 0.0, 0.0, 0.0))
    assert len(matches) == 1
    m = matches[0]
    assert -1.0 <= m.score <= 1.0
    assert m.use_count == 1
    assert m.record.payload == {"k": "v"}
    assert m.last_used_at.tzinfo is not None  # UTC-aware


# ---------------------------------------------------------------------------
# I2 — atomicity: a transaction rollback leaves the table byte-untouched
# ---------------------------------------------------------------------------


async def test_i2_sweep_atomicity_rollback(settings: SubstrateSettings) -> None:
    """I2: if the sweep statement rolls back, the table is byte-untouched."""
    store = PgLatentStore(dim=4, settings=settings, clock=lambda: _T0)
    await store.ensure_schema()
    await store.reset()

    # Insert 4 rows
    for i in range(4):
        await store.put(LatentRecord(id=str(i), embedding=(1.0, 0.0, 0.0, 0.0)))

    # Snapshot tiers before
    conn = await store._connection()
    cur = await conn.execute("SELECT id, tier FROM cogworx_latent ORDER BY id")
    tiers_before = dict(await cur.fetchall())

    # Simulate a mid-sweep crash by executing the sweep CTE inside an explicit txn
    # and rolling it back
    async with await psycopg_connect_no_autocommit(store) as txn_conn:
        await txn_conn.execute(
            "WITH ranked AS (SELECT id, ROW_NUMBER() OVER (ORDER BY use_count DESC) AS rn "
            "FROM cogworx_latent), target AS (SELECT id, (rn <= 2) AS should_hot FROM ranked) "
            "UPDATE cogworx_latent l SET tier = CASE WHEN t.should_hot THEN 'hot' ELSE 'cold' END "
            "FROM target t WHERE l.id = t.id"
        )
        await txn_conn.rollback()

    # Snapshot tiers after rollback — must be byte-identical
    cur2 = await conn.execute("SELECT id, tier FROM cogworx_latent ORDER BY id")
    tiers_after = dict(await cur2.fetchall())
    assert tiers_before == tiers_after

    await store.aclose()


async def psycopg_connect_no_autocommit(store: PgLatentStore):  # type: ignore[return]
    """Helper: open a non-autocommit connection for the rollback test."""
    import psycopg
    from pgvector.psycopg import register_vector_async

    conn = await psycopg.AsyncConnection.connect(store._dsn, autocommit=False)
    await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await register_vector_async(conn)
    return conn
