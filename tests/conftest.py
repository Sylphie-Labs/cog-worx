"""Shared pytest configuration for cog-worx.

Deterministic tiers (unit + invariant suites) run with no external services — they use the Test
Kit's ``ReplayModel`` and in-memory substrate doubles. Integration/spike tiers (``-m integration``,
``-m spike``) require the polyglot substrate and are skipped by default when it is unreachable.
"""

from __future__ import annotations

pytest_plugins = ["cogworx.testing.fixtures"]
