"""Local task runner for cog-worx — the same checks CI runs.

    uv run nox -s lint typecheck test sizing   # deterministic tiers (no Docker)
    uv run nox -s integration             # needs the polyglot substrate up

The deterministic sessions (lint/typecheck/test) must stay green on every change — they are the bet
that makes cog-worx heavily testable (CANON §0.4).
"""

from __future__ import annotations

import nox

nox.options.default_venv_backend = "uv"
nox.options.sessions = ["lint", "typecheck", "test", "sizing"]

PYTHON_VERSIONS = ["3.13"]


@nox.session(python=PYTHON_VERSIONS)
def lint(session: nox.Session) -> None:
    session.install("ruff>=0.15.16,<0.16")
    session.run("ruff", "check", "src", "tests")
    session.run("ruff", "format", "--check", "src", "tests")


@nox.session(python=PYTHON_VERSIONS)
def typecheck(session: nox.Session) -> None:
    # Deliberately WITHOUT the `sizing` extra, per the invariant in pyproject.toml: a checkout
    # without it must still type-check. The two numpy-dependent lines that mypy reads differently
    # with and without numpy carry `[..., unused-ignore]`, so both environments are clean.
    session.install("-e", ".", "--group", "dev")
    session.run("mypy")


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Deterministic tiers only — unit + invariant suites on in-memory doubles."""
    session.install("-e", ".", "--group", "dev")
    session.run("pytest", "-m", "not integration and not spike", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def sizing(session: nox.Session) -> None:
    """The numpy fast-kernel suite — the one tier that installs the optional `sizing` extra.

    `test` deliberately runs without numpy, so CANON S2's "numpy is the adopter's burden" stays
    proven; the consequence is that `tests/eval/test_sizing_fast.py` skips there. That module is
    the ONLY coverage of `cogworx.eval._sizing_fast` and `cogworx.eval.equiv_check`, including
    their mutation kill-set, so without this session a fast-kernel regression would pass every
    check. Runs the same deselection as `test`; only the environment differs.
    """
    session.install("-e", ".[sizing]", "--group", "dev")
    session.run("pytest", "-m", "not integration and not spike", *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def integration(session: nox.Session) -> None:
    """Integration + Spike Suite 1 — requires the substrate (docker compose up -d)."""
    session.install("-e", ".", "--group", "dev")
    session.run("pytest", "-m", "integration or spike", *session.posargs)
