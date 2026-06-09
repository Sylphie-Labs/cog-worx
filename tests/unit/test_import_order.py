"""Every public package must be importable FIRST in a fresh interpreter.

Regression for RT-1.4 L10: ``import cogworx.runtime`` as the FIRST cogworx import detonated on an
import cycle (``substrate.journal`` -> ``loop.result`` -> ``loop/__init__`` -> ``stage``/``graph``
-> ``substrate.journal``, broken by deferring annotation-only imports under ``TYPE_CHECKING``).
The cycle only fires when the vulnerable edge is imported first, so in-suite imports can never
catch it — the whole package is already initialized by earlier test imports. Hence subprocess
isolation: one fresh interpreter per entry point.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

PACKAGES = (
    "cogworx",
    "cogworx.adapters",
    "cogworx.capability",
    "cogworx.claims",
    "cogworx.coordination",
    "cogworx.cost",
    "cogworx.loop",
    "cogworx.model",
    "cogworx.runtime",
    "cogworx.substrate",
    "cogworx.telemetry",
    "cogworx.testing",
)


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
