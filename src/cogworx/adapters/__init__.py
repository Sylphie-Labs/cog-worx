"""Real polyglot-substrate adapters (CANON S3).

Each engine for its strength: Neo4j for the graph knowledge layer, the single Postgres cluster for
pgvector (latent) and TimescaleDB (journal). These adapters implement the frozen substrate seams
against the real services; the in-memory doubles in :mod:`cogworx.testing` are the reference.
"""

from __future__ import annotations

from cogworx.adapters.neo4j_entity_kg import Neo4jEntityKG
from cogworx.adapters.neo4j_graph import Neo4jGraphStore
from cogworx.adapters.neo4j_procedural_kg import Neo4jProceduralKG
from cogworx.adapters.pg_episodes import PgEpisodeStore

__all__ = [
    "Neo4jEntityKG",
    "Neo4jGraphStore",
    "Neo4jProceduralKG",
    "PgEpisodeStore",
]
