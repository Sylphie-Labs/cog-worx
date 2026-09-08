"""Every public package must be importable FIRST in a fresh interpreter.

Regression for RT-1.4 L10: ``import cogworx.runtime`` as the FIRST cogworx import detonated on an
import cycle (``substrate.journal`` -> ``loop.result`` -> ``loop/__init__`` -> ``stage``/``graph``
-> ``substrate.journal``, broken by deferring annotation-only imports under ``TYPE_CHECKING``).
The cycle only fires when the vulnerable edge is imported first, so in-suite imports can never
catch it — the whole package is already initialized by earlier test imports. Hence subprocess
isolation: one fresh interpreter per entry point.

CF-1 (Task 3.0a-1): the ``TYPE_CHECKING`` band-aid in ``stage``/``graph`` is replaced by a
PEP-562 lazy ``__getattr__`` in ``loop/__init__``.  The two new tests below verify the structural
guarantee: importing ``cogworx.loop`` or cherry-picking ``StageResult`` must NOT pull
``cogworx.substrate.journal`` into ``sys.modules``.
"""

from __future__ import annotations

import pkgutil
import subprocess
import sys

import pytest

PACKAGES = (
    "cogworx",
    "cogworx.adapters",
    "cogworx.capability",
    "cogworx.claims",
    "cogworx.context",
    "cogworx.coordination",
    "cogworx.cost",
    "cogworx.loop",
    "cogworx.model",
    "cogworx.runtime",
    "cogworx.substrate",
    "cogworx.telemetry",
    "cogworx.testing",
)


# Every module under ``cogworx.adapters``, discovered rather than listed.  ``adapters/__init__``
# re-exports four of the six adapters, so ``import cogworx.adapters`` never reaches ``pg_latent``
# or ``timescale_journal`` — a break in either is invisible to the deterministic tier.  That is
# exactly how ``from pgvector.psycopg import Vector`` stayed broken: pgvector 0.5.0 moved
# ``Vector`` to the top-level package, the adapter stopped importing at all, and the only tests
# that touch it are marked ``integration`` and run on ``main`` alone.  Enumerating the package
# rather than hand-listing it keeps the guarantee true for adapters not yet written.
def _adapter_modules() -> tuple[str, ...]:
    import cogworx.adapters

    return tuple(
        sorted(
            f"cogworx.adapters.{info.name}"
            for info in pkgutil.iter_modules(cogworx.adapters.__path__)
            if not info.name.startswith("_")
        )
    )


ADAPTER_MODULES = _adapter_modules()


@pytest.mark.parametrize("package", PACKAGES)
def test_package_imports_first_in_fresh_interpreter(package: str) -> None:
    proc = subprocess.run(
        [sys.executable, "-c", f"import {package}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`import {package}` failed in a fresh interpreter (import-cycle regression):\n"
        f"{proc.stderr}"
    )


def test_adapter_module_discovery_is_not_empty() -> None:
    """A discovery bug that returned nothing would make the test below vacuously pass."""
    assert len(ADAPTER_MODULES) >= 6, f"expected every adapters/ module, found {ADAPTER_MODULES}"
    assert "cogworx.adapters.pg_latent" in ADAPTER_MODULES


@pytest.mark.parametrize("module", ADAPTER_MODULES)
def test_adapter_module_imports_in_fresh_interpreter(module: str) -> None:
    """Every adapter module must import against the substrate drivers actually resolved.

    The drivers (``neo4j``, ``psycopg``, ``pgvector``) are hard dependencies with floors and no
    committed lockfile, so an upstream rename lands here silently.  Importing the module is enough
    to catch it and needs no running service.
    """
    proc = subprocess.run(
        [sys.executable, "-c", f"import {module}"],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        f"`import {module}` failed in a fresh interpreter -- most likely a substrate driver "
        f"changed its API under an uncapped floor:\n{proc.stderr}"
    )


# ── CF-1 regression: PEP-562 lazy loader must keep substrate.journal off the hot path ──

_NO_JOURNAL_SCRIPT = """\
import sys
import cogworx.loop
assert "cogworx.substrate.journal" not in sys.modules, (
    "import cogworx.loop eagerly pulled in cogworx.substrate.journal -- "
    "the import cycle is NOT broken; check loop/__init__.py lazy loader"
)
"""

_NO_JOURNAL_STAGE_RESULT_SCRIPT = """\
import sys
from cogworx.loop import StageResult
assert "cogworx.substrate.journal" not in sys.modules, (
    "from cogworx.loop import StageResult eagerly pulled in cogworx.substrate.journal -- "
    "the import cycle is NOT broken; check loop/__init__.py lazy loader"
)
# Verify StageResult is actually usable (not a broken import).
from cogworx.loop.result import Transition
assert StageResult is not None
"""


def _run_isolated(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )


def test_import_cogworx_loop_does_not_pull_substrate_journal() -> None:
    """CF-1: ``import cogworx.loop`` must NOT import ``cogworx.substrate.journal``.

    The PEP-562 lazy ``__getattr__`` in ``loop/__init__`` defers ``stage``/``graph`` (which are
    on the substrate.journal cycle edge).  Importing the package alone must never cross that edge.
    """
    proc = _run_isolated(_NO_JOURNAL_SCRIPT)
    assert proc.returncode == 0, (
        "CF-1 cycle regression: import cogworx.loop pulled substrate.journal or failed:\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


def test_stage_result_import_does_not_pull_substrate_journal() -> None:
    """CF-1: ``from cogworx.loop import StageResult`` must NOT import ``cogworx.substrate.journal``.

    ``StageResult`` lives in ``loop.result`` (a leaf module with no substrate dependency).  The
    lazy loader must resolve it from the eagerly-loaded ``result`` module, never touching
    ``stage`` or ``graph``.
    """
    proc = _run_isolated(_NO_JOURNAL_STAGE_RESULT_SCRIPT)
    assert proc.returncode == 0, (
        "CF-1 cycle regression: from cogworx.loop import StageResult pulled substrate.journal "
        f"or failed:\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
