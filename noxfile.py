"""Local task runner for cog-worx — the same checks CI runs.

    uv run nox -s lint typecheck test sizing   # deterministic tiers (no Docker)
    uv run nox -s integration             # needs the polyglot substrate up

The deterministic sessions (lint/typecheck/test/sizing) must stay green on every change — they are
the bet that makes cog-worx heavily testable (CANON §0.4).
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


def _run_pytest(session: nox.Session, *paths: str) -> None:
    """The deterministic deselection, shared so `test` and `sizing` cannot drift apart."""
    session.run("pytest", "-m", "not integration and not spike", *paths, *session.posargs)


@nox.session(python=PYTHON_VERSIONS)
def test(session: nox.Session) -> None:
    """Deterministic tiers only — unit + invariant suites on in-memory doubles."""
    session.install("-e", ".", "--group", "dev")
    _run_pytest(session)


@nox.session(python=PYTHON_VERSIONS)
def sizing(session: nox.Session) -> None:
    """The eval tier with numpy present — type-checked and tested.

    `test` and `typecheck` both run WITHOUT numpy, so CANON S2's "numpy is the adopter's burden"
    stays proven and pyproject's "a checkout without the extra must still type-check" holds. The
    consequence is that everything numpy-only goes unverified there: `tests/eval/test_sizing_fast.py`
    skips, and mypy sees `numpy.*` as `Any`, so the ~326 lines of `cogworx.eval._sizing_fast` and
    the 626 of `cogworx.eval.equiv_check` are checked against nothing. This session is where both
    actually happen. It is also the environment README.md tells contributors to create
    (`uv sync --all-extras`).

    Scoped to `tests/eval` rather than the whole suite: that is where every numpy-dependent test
    lives, and `cogworx.eval.lock` puts numpy in `_ENV_PACKAGES`, so the eval env fingerprint
    genuinely differs between the two tiers.

    Measured, so nobody is surprised: this session takes ~8m for 589 tests, against ~4m for the
    2197 in `test`. The cost is the numpy fast-kernel suite itself — an equivalence check plus a
    mutation kill-set — not the scoping, and it is the same work that was silently skipping
    before. Scoping saves little wall-clock; what it buys is a session whose name matches what it
    runs, so a failure here points at the eval tier rather than at 2000 unrelated tests.
    """
    session.install("-e", ".[sizing]", "--group", "dev")
    # Fail loudly if the extra did not actually deliver an importable numpy — a renamed or
    # emptied extra, or a wheel/ABI mismatch, would otherwise leave `importorskip` silently
    # skipping the only coverage of the fast kernel while this session still reported green.
    session.run("python", "-c", "import numpy, cogworx.eval.equiv_check, cogworx.eval._sizing_fast")
    session.run("mypy")
    _run_pytest(session, "tests/eval")


@nox.session(python=PYTHON_VERSIONS)
def integration(session: nox.Session) -> None:
    """Integration + Spike Suite 1 — requires the substrate (docker compose up -d)."""
    session.install("-e", ".", "--group", "dev")
    session.run("pytest", "-m", "integration or spike", *session.posargs)
