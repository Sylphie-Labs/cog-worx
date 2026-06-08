"""Real polyglot-substrate adapters (CANON S3).

Each engine for its strength: Neo4j for the graph knowledge layer, the single Postgres cluster for
pgvector (latent) and TimescaleDB (journal). These adapters implement the frozen substrate seams
against the real services; the in-memory doubles in :mod:`cogworx.testing` are the reference.
"""

from __future__ import annotations

__all__: list[str] = []
