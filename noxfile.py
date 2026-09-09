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
    # The `sizing` extra is installed here but NOT in `test`, and the asymmetry is deliberate.
    # Typechecking wants numpy's real types: without it `numpy.*` falls back to `Any` under the
    # ignore_missing_imports override, and mypy --strict then refuses `class _SpyGen(NPGenerator)`
    # as subclassing Any. The `test` session must keep numpy absent, because CANON S2 makes it an
    # opt-in adopter dependency and the suite has to prove it stays optional.
    session.install("-e", ".[sizing]", "--group", "dev")
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
