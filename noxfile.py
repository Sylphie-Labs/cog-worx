"""Local task runner for cog-worx — the same checks CI runs.

    uv run nox -s lint typecheck test     # deterministic tiers (no Docker)
    uv run nox -s integration             # needs the polyglot substrate up

The deterministic sessions (lint/typecheck/test) must stay green on every change — they are the bet
that makes cog-worx heavily testable (CANON §0.4).
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "test"]

PYTHON_VERSIONS = ["3.13"]


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.15.16,<0.16")
    session.run("ruff", "check", "src", "tests")
    session.run("ruff", "format", "--check", "src", "tests")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    session.install("-e", ".", "--group", "dev")
    session.run("mypy")


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Deterministic tiers only — unit + invariant suites on in-memory doubles."""
    session.install("-e", ".", "--group", "dev")
    session.run("pytest", "-m", "not integration and not spike", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def integration(session: nox.Session) -> None:
    """Integration + Spike Suite 1 — requires the substrate (docker compose up -d)."""
    session.install("-e", ".", "--group", "dev")
    session.run("pytest", "-m", "integration or spike", *session.posargs)
