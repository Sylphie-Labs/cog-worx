"""The pgvector latent-space adapter (CANON S3).

The real :class:`~cogworx.substrate.latent.LatentStore` seam: dense recall over the hot/cold latent
space using pgvector's native nearest-neighbour search (the ``<=>`` cosine-distance operator), not a
generic store that flattens it. Behaviourally equivalent to ``InMemoryLatentStore`` — replace-by-id
upsert, cosine SIMILARITY ordering (``1 - distance``), empty/zero-query returns no matches.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import psycopg
from pgvector.psycopg import register_vector_async
from psycopg.types.json import Jsonb

from cogworx.adapters.config import SubstrateSettings
from cogworx.substrate.latent import LatentMatch, LatentRecord


class PgLatentStore:
    """A :class:`LatentStore` backed by a pgvector column on the Postgres cluster."""

    def __init__(
        self,
        *,
        dim: int,
        settings: SubstrateSettings | None = None,
        dsn: str | None = None,
    ) -> None:
        self._dim = dim
        self._dsn = dsn if dsn is not None else (settings or SubstrateSettings()).pg_dsn
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
        await conn.execute(
            f"CREATE TABLE IF NOT EXISTS cogworx_latent ("
            f"id text PRIMARY KEY, "
            f"embedding vector({self._dim}) NOT NULL, "
            f"payload jsonb NOT NULL, "
            f"use_count int NOT NULL DEFAULT 0)"
        )

    async def upsert(self, record: LatentRecord) -> None:
        if len(record.embedding) != self._dim:
            raise ValueError(
                f"embedding dimension {len(record.embedding)} != store dimension {self._dim}"
            )
        conn = await self._connection()
        await conn.execute(
            "INSERT INTO cogworx_latent (id, embedding, payload, use_count) "
            "VALUES (%s, %s, %s, %s) "
            "ON CONFLICT (id) DO UPDATE SET "
            "embedding = EXCLUDED.embedding, "
            "payload = EXCLUDED.payload, "
            "use_count = EXCLUDED.use_count",
            (
                record.id,
                list(record.embedding),
                Jsonb(record.payload),
                record.use_count,
            ),
        )

    async def search(self, embedding: Sequence[float], *, k: int = 10) -> Sequence[LatentMatch]:
        query = list(embedding)
        if len(query) != self._dim:
            raise ValueError(f"query dimension {len(query)} != store dimension {self._dim}")
        if math.sqrt(sum(value * value for value in query)) == 0.0:
            return ()
        conn = await self._connection()
        cursor = await conn.execute(
            "SELECT id, embedding, payload, use_count, embedding <=> %s AS distance "
            "FROM cogworx_latent ORDER BY distance ASC LIMIT %s",
            (query, k),
        )
        rows = await cursor.fetchall()
        return tuple(
            LatentMatch(
                record=LatentRecord(
                    id=row[0],
                    embedding=tuple(float(value) for value in row[1]),
                    payload=row[2],
                    use_count=row[3],
                ),
                score=1.0 - float(row[4]),
            )
            for row in rows
        )

    async def reset(self) -> None:
        conn = await self._connection()
        await conn.execute("DROP TABLE IF EXISTS cogworx_latent")
        await self.ensure_schema()

    async def aclose(self) -> None:
        if self._conn is not None and not self._conn.closed:
            await self._conn.close()
        self._conn = None


__all__ = ["PgLatentStore"]
