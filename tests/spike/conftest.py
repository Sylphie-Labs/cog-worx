"""Spike-tier fixtures: the REAL polyglot substrate (CANON S12).

Mirrors ``tests/integration/conftest.py`` — the spike suite is its own package dir (so mypy does not
collide conftests), so it carries its own session-scoped ``settings`` fixture. Endpoints come from
:class:`cogworx.adapters.config.SubstrateSettings` (env prefix ``COGWORX_``, repo-root ``.env``).
"""

from __future__ import annotations

import pytest

from cogworx.adapters.config import SubstrateSettings


@pytest.fixture(scope="session")
def settings() -> SubstrateSettings:
    return SubstrateSettings()
