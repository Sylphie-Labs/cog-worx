"""Connection settings for the real polyglot substrate (CANON S3).

Adapters read their endpoints from the environment (prefix ``COGWORX_``) so the same code runs
against a local ``docker compose`` stack, CI services, or a deployed cluster without edits. Defaults
match ``docker-compose.yml`` for zero-config local development. This is the only place a substrate
endpoint is named.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class SubstrateSettings(BaseSettings):
    """Endpoints for Neo4j (graph) and the single Postgres cluster (pgvector + TimescaleDB)."""

    model_config = SettingsConfigDict(env_prefix="COGWORX_", env_file=".env", extra="ignore")

    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "cogworx-test"
    # One DSN for the single Postgres cluster that hosts BOTH pgvector and TimescaleDB (S3).
    pg_dsn: str = "postgresql://postgres:cogworx-test@localhost:5432/cogworx"


__all__ = ["SubstrateSettings"]
