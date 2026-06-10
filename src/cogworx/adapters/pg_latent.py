"""The pgvector latent-space adapter — hot/cold tiering (CANON S3, Pod 2.2).

Implements the full :class:`~cogworx.substrate.latent.LatentStore` seam: dense recall over the
hot/cold latent space using pgvector's native nearest-neighbour search (the ``<=>`` cosine-distance
operator). Behaviourally equivalent to ``InMemoryLatentStore``.

## Key invariants

- **I4 (put-preserves-usage):** the ``put`` ON CONFLICT clause names ONLY ``embedding`` and
  ``payload`` — never ``use_count``, ``last_used_at``, ``tier``, or ``created_at``. Pinned by a
  static grep in the spike suite (a5-style) and a runtime probe (I4 unit test).
- **I5 (no lost increments):** ``record_use`` is an atomic
  ``UPDATE … SET use_count = use_count + 1`` — no read-modify-write race under concurrent callers.
- **Sweep atomicity (I2):** ``sweep_tiers`` is a single UPDATE CTE statement; a crash rolls back
  the implicit txn leaving the table byte-untouched.
- **Idempotent sweep (I1):** with a frozen ``now``, applying the sweep twice leaves the table
  byte-identical (``IS DISTINCT FROM`` guard, zero rows touched on the second call).
- **Capacity invariant (I3):** after every sweep,
  ``count(tier='hot') == min(hot_capacity, count(*))``.

## Schema (after Pod 2.2 migration)

    cogworx_latent (
      id           text PRIMARY KEY,
      embedding    vector(dim) NOT NULL,
      payload      jsonb NOT NULL,
      use_count    int NOT NULL DEFAULT 0,
      tier         text NOT NULL DEFAULT 'cold' CHECK (tier IN ('hot','cold')),
      created_at   timestamptz NOT NULL,
      last_used_at timestamptz NOT NULL
    )
    INDEX cogworx_latent_tier ON cogworx_latent (tier)

The ``created_at`` / ``last_used_at`` DEFAULT in the migration backfill is the only place DB time
is used; all live writes bind an explicit clock-stamped value (injectable clock, house rule).
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from typing import Any

import psycopg
from pgvector.psycopg import Vector, register_vector_async
from psycopg.types.json import Jsonb

from cogworx.adapters.config import SubstrateSettings
from cogworx.knowledge.latent_activation import ActivationParams
from cogworx.substrate.latent import LatentMatch, LatentRecord, Tier, TierSweepResult

# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS cogworx_latent (
  id           text PRIMARY KEY,
  embedding    vector({dim}) NOT NULL,
  payload      jsonb NOT NULL,
  use_count    int NOT NULL DEFAULT 0,
  tier         text NOT NULL DEFAULT 'cold',
  created_at   timestamptz NOT NULL DEFAULT now(),
  last_used_at timestamptz NOT NULL DEFAULT now()
)"""

_ADD_TIER = """\
ALTER TABLE cogworx_latent
  ADD COLUMN IF NOT EXISTS tier text NOT NULL DEFAULT 'cold'"""

_ADD_CREATED_AT = """\
ALTER TABLE cogworx_latent
  ADD COLUMN IF NOT EXISTS created_at timestamptz NOT NULL DEFAULT now()"""

_ADD_LAST_USED_AT = """\
ALTER TABLE cogworx_latent
  ADD COLUMN IF NOT EXISTS last_used_at timestamptz NOT NULL DEFAULT now()"""

_ADD_TIER_CHECK = """\
DO $$ BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM information_schema.constraint_column_usage
    WHERE table_name='cogworx_latent' AND constraint_name='cogworx_latent_tier_check'
  ) THEN
    ALTER TABLE cogworx_latent
      ADD CONSTRAINT cogworx_latent_tier_check CHECK (tier IN ('hot','cold'));
  END IF;
END $$"""

_CREATE_TIER_INDEX = """\
CREATE INDEX IF NOT EXISTS cogworx_latent_tier ON cogworx_latent (tier)"""

# ---------------------------------------------------------------------------
# CTE for sweep_tiers — one atomic statement (I1, I2)
#
# The activation formula is:  ln(1 + use_count) - d·ln(max(Δt_hours, ε))
# The ORDER BY mirrors select_hot_ids() in latent_activation.py exactly (SQL-vs-Python parity).
# ---------------------------------------------------------------------------

_SWEEP_CTE = """\
WITH ranked AS (
  SELECT
    id,
    ln(1.0 + use_count)
      - %(d)s * ln(
          GREATEST(
            EXTRACT(EPOCH FROM (%(now)s - last_used_at)) / 3600.0,
            %(eps)s
          )
        ) AS act,
    use_count,
    last_used_at
  FROM cogworx_latent
),
target AS (
  SELECT
    id,
    (ROW_NUMBER() OVER (
       ORDER BY act DESC, use_count DESC, last_used_at DESC, id ASC
     ) <= %(hot_capacity)s) AS should_hot
  FROM ranked
)
UPDATE cogworx_latent l
SET    tier = CASE WHEN t.should_hot THEN 'hot' ELSE 'cold' END
FROM   target t
WHERE  l.id = t.id
AND    l.tier IS DISTINCT FROM CASE WHEN t.should_hot THEN 'hot' ELSE 'cold' END
RETURNING l.tier"""


class PgLatentStore:
    """A :class:`LatentStore` backed by pgvector on the Postgres cluster."""

    def __init__(
        self,
        *,
        dim: int,
        settings: SubstrateSettings | None = None,
        dsn: str | None = None,
        clock: Callable[[], datetime] | None = None,
        activation_params: ActivationParams | None = None,
    ) -> None:
        self._dim = dim
        self._dsn = dsn if dsn is not None else (settings or SubstrateSettings()).pg_dsn
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))
        self._params = activation_params or ActivationParams()
        self._conn: psycopg.AsyncConnection[Any] | None = None

    async def _connection(self) -> psycopg.AsyncConnection[Any]:
        if self._conn is None or self._conn.closed:
            conn = await psycopg.AsyncConnection.connect(self._dsn, autocommit=True)
            await conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
            await register_vector_async(conn)
            self._conn = conn
        return self._conn

    async def ensure_schema(self) -> None:
        conn = await self._connection()
        await conn.execute(_SCHEMA.format(dim=self._dim))
        # Idempotent migration for pre-2.2 tables (the 2.1 upgrade-crash lesson: index after ALTER)
        await conn.execute(_ADD_TIER)
        await conn.execute(_ADD_CREATED_AT)
        await conn.execute(_ADD_LAST_USED_AT)
        await conn.execute(_ADD_TIER_CHECK)
        await conn.execute(_CREATE_TIER_INDEX)

    async def put(self, record: LatentRecord) -> None:
        """Insert-or-replace content; ON CONFLICT never touches
        use_count/last_used_at/tier/created_at."""
        if len(record.embedding) != self._dim:
            raise ValueError(
                f"embedding dimension {len(record.embedding)} != store dimension {self._dim}"
            )
        now = self._clock()
        conn = await self._connection()
        # I4 invariant: the ON CONFLICT clause names ONLY embedding and payload.
        # Static grep + runtime probe pin this (spike I4 test).
        await conn.execute(
            "INSERT INTO cogworx_latent (id, embedding, payload, created_at, last_used_at) "
            "VALUES (%s, %s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, "
            "payload = EXCLUDED.payload",
            (
                record.id,
                Vector(record.embedding),
                Jsonb(record.payload),
                now,
                now,
            ),
        )

    async def record_use(self, ids: Sequence[str]) -> int:
        """Atomic in-place use_count += 1 for each id; returns rows touched."""
        if not ids:
            return 0
        now = self._clock()
        conn = await self._connection()
        cursor = await conn.execute(
            "UPDATE cogworx_latent "
            "SET use_count = use_count + 1, last_used_at = %s "
            "WHERE id = ANY(%s)",
            (now, list(ids)),
        )
        return cursor.rowcount if cursor.rowcount is not None else 0

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        tier: Tier | None = None,
    ) -> Sequence[LatentMatch]:
        """Global exact top-k by cosine; ``tier`` scopes to hot/cold only (None = all)."""
        query = list(embedding)
        if len(query) != self._dim:
            raise ValueError(f"query dimension {len(query)} != store dimension {self._dim}")
        if math.sqrt(sum(v * v for v in query)) == 0.0:
            return ()
        conn = await self._connection()
        tier_clause = "WHERE tier = %s " if tier is not None else ""
        params: list[Any] = []
        if tier is not None:
            params.append(tier)
        params.extend([Vector(query), k])
        cursor = await conn.execute(
            f"SELECT id, embedding, payload, use_count, tier, last_used_at, "
            f"embedding <=> %s AS distance "
            f"FROM cogworx_latent "
            f"{tier_clause}"
            f"ORDER BY distance ASC, id ASC "
            f"LIMIT %s",
            params,
        )
        rows = await cursor.fetchall()
        return tuple(
            LatentMatch(
                record=LatentRecord(
                    id=row[0],
                    embedding=tuple(float(v) for v in row[1]),
                    payload=row[2],
                ),
                score=1.0 - float(row[6]),
                use_count=row[3],
                tier=row[4],
                last_used_at=row[5] if row[5].tzinfo is not None else row[5].replace(tzinfo=UTC),
            )
            for row in rows
        )

    async def sweep_tiers(self, *, now: datetime, hot_capacity: int) -> TierSweepResult:
        """Re-assign hot/cold tiers via one atomic UPDATE CTE. Idempotent under fixed ``now``."""
        conn = await self._connection()
        cursor = await conn.execute(
            _SWEEP_CTE,
            {
                "d": self._params.d,
                "now": now,
                "eps": self._params.eps_hours,
                "hot_capacity": hot_capacity,
            },
        )
        rows = await cursor.fetchall()
        promoted = sum(1 for (t,) in rows if t == "hot")
        demoted = sum(1 for (t,) in rows if t == "cold")
        # Compute hot_size separately (the RETURNING only covers changed rows)
        count_cursor = await conn.execute("SELECT COUNT(*) FROM cogworx_latent WHERE tier = 'hot'")
        count_row = await count_cursor.fetchone()
        hot_size = int(count_row[0]) if count_row else 0
        return TierSweepResult(promoted=promoted, demoted=demoted, hot_size=hot_size)

    async def reset(self) -> None:
        conn = await self._connection()
        await conn.execute("DROP TABLE IF EXISTS cogworx_latent")
        await self.ensure_schema()

    async def aclose(self) -> None:
        if self._conn is not None and not self._conn.closed:
            await self._conn.close()
        self._conn = None


__all__ = ["PgLatentStore"]
