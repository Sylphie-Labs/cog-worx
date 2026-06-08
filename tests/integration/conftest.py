"""Integration-tier fixtures: the REAL polyglot substrate, wiped per case.

These tests are marked ``integration`` and skipped by default when the substrate is unreachable
(``docker compose up -d`` first). Each test gets an ephemeral, isolated schema — adapters create
their tables idempotently and the fixtures drop/truncate per case (ported from tess's isolated
instances) so cases never bleed into each other.

Connection endpoints come from :class:`cogworx.adapters.config.SubstrateSettings` (env prefix
``COGWORX_``), defaulting to the local ``docker-compose.yml`` stack.
"""

from __future__ import annotations

import pytest

from cogworx.adapters.config import SubstrateSettings


@pytest.fixture(scope="session")
def settings() -> SubstrateSettings:
    return SubstrateSettings()
