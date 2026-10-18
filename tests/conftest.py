"""Shared pytest configuration for cog-worx.

Deterministic tiers (unit + invariant suites) run with no external services — they use the Test
Kit's ``ReplayModel`` and in-memory substrate doubles. Integration/spike tiers (``-m integration``,
``-m spike``) require the polyglot substrate and are skipped by default when it is unreachable.
"""

from __future__ import annotations

import os
import socket
from functools import cache
from urllib.parse import urlsplit

import pytest

from cogworx.adapters.config import SubstrateSettings

pytest_plugins = ["cogworx.testing.fixtures"]

# Set to 1 where the substrate is supposed to be up (CI's integration job): an unreachable service
# then fails its tests loudly instead of skipping them.
_REQUIRE_SUBSTRATE_ENV = "COGWORX_REQUIRE_SUBSTRATE"
_PROBE_TIMEOUT_SECONDS = 1.0
_DEFAULT_PORTS = {"bolt": 7687, "neo4j": 7687, "postgres": 5432, "postgresql": 5432}


def _endpoint(url: str) -> tuple[str, int] | None:
    """``(host, port)`` from a URI-form endpoint, or ``None`` if it is not one we can probe (for
    example a keyword-form libpq DSN) — the caller then assumes the service is reachable."""
    parts = urlsplit(url)
    if not parts.hostname:
        return None
    port = parts.port or _DEFAULT_PORTS.get(parts.scheme.split("+")[0])
    return (parts.hostname, port) if port else None


@cache
def _reachable(url: str) -> bool:
    endpoint = _endpoint(url)
    if endpoint is None:
        return True
    try:
        with socket.create_connection(endpoint, timeout=_PROBE_TIMEOUT_SECONDS):
            return True
    except OSError:
        return False


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip substrate tests when the substrate is unreachable, so a bare ``pytest`` (no nox, no
    ``docker compose up``) reports them as skipped instead of failing them.

    A test counts as a substrate test when it is marked ``integration`` and requests the
    ``settings`` fixture (live-provider tests don't, and gate themselves on their API keys). The
    ``neo4j`` / ``postgres`` markers narrow the check to one service; otherwise both must answer.
    The probe is one TCP connect per endpoint per session.
    """
    if os.environ.get(_REQUIRE_SUBSTRATE_ENV) == "1":
        return
    settings: SubstrateSettings | None = None
    for item in items:
        if item.get_closest_marker("integration") is None:
            continue
        if "settings" not in getattr(item, "fixturenames", ()):
            continue
        settings = settings or SubstrateSettings()
        wants_neo4j = item.get_closest_marker("neo4j") is not None
        wants_pg = item.get_closest_marker("postgres") is not None
        if not (wants_neo4j or wants_pg):
            wants_neo4j = wants_pg = True
        missing = [
            name
            for name, wanted, url in (
                ("Neo4j", wants_neo4j, settings.neo4j_uri),
                ("Postgres", wants_pg, settings.pg_dsn),
            )
            if wanted and not _reachable(url)
        ]
        if missing:
            reason = (
                f"{' and '.join(missing)} unreachable (run `docker compose up -d`, "
                f"or set {_REQUIRE_SUBSTRATE_ENV}=1 to fail instead)"
            )
            item.add_marker(pytest.mark.skip(reason=reason))
