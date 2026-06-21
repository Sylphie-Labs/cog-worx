"""Offline corpus-authoring helpers (PRIVATE — never on the live verification path).

This module holds the journal-free execution helper the Pod 4.4c-3 planting pipeline uses to
self-check known-correct seeds at authoring time. It is a PRIVATE module (``_``-prefixed) — the
privacy boundary that keeps the F7-bypassing kernel out of reach of any non-authoring code. It is
NOT exported from ``cogworx.eval`` and carries no public ``__all__``.

It does NOT duplicate the executable-oracle kernel: it imports the shared
:func:`~cogworx.verification.oracles.code._run_pytest_local` /
:func:`~cogworx.verification.oracles.code.verdict_from_result` from
:mod:`cogworx.verification.oracles.code` and composes them WITHOUT the ``StageContext`` /
taint-latch read. The live verification path stays in ``CodeOracle.evaluate`` (F7 taint latch +
docker tier).

Contract changelog (CANON §6.1):
  2026-06-20 (Pod 4.4c-3, ADDITIVE): run_frozen_check — journal-free local-tier frozen execution for
  offline corpus-authoring seed self-checks. Relocated here (private authoring module) from
  ``cogworx.verification.oracles.code`` and DE-EXPORTED so the F7-bypassing kernel is unreachable
  from non-authoring code (S10 widen-the-guard). No behavior change to the kernel it composes.
"""

from __future__ import annotations

from cogworx.verification.contracts import Verdict
from cogworx.verification.oracles.code import (
    DEFAULT_TIMEOUT_S,
    _run_pytest_local,
    verdict_from_result,
)


def run_frozen_check(
    solution_code: str, test_code: str, *, timeout_s: float = DEFAULT_TIMEOUT_S
) -> Verdict:
    """Run a known-correct seed solution against its frozen test, journal-free (offline authoring).

    The local-tier execution kernel of :meth:`CodeOracle.evaluate` with ``test_source="frozen"``
    on a clean (untainted) drive, WITHOUT the ``StageContext`` / taint-latch read — so offline
    corpus-authoring tooling (Pod 4.4c-3 seed self-check) can verify a seed with NO journal I/O
    (S1-clean authoring path). NOT for the live verification path: a live run MUST go through
    ``CodeOracle.evaluate`` so the F7 taint latch + docker tier selection apply (this helper never
    reads taint and never sandboxes). ``source="tool"``, ``test_provenance="frozen"``.

    Equivalence (pinned by the 4.4c-3 test suite): for any ``(solution_code, test_code)`` this
    returns the same ``Verdict`` as ``CodeOracle(test_source="frozen", frozen_test_code=test_code)
    .evaluate(...)`` on an untainted drive with ``always_sandbox=False``.
    """
    result = _run_pytest_local(solution_code, test_code, timeout_s=timeout_s)
    return verdict_from_result(result, timeout_s=timeout_s, test_provenance="frozen")
