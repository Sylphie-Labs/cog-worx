"""Pod 4.4b — CodeOracle.test_source selector + Verdict.test_provenance (CANON S9, audit-only).

Unit-level acceptance tests (no subprocess, no I/O): exercises construction invariants,
fail-fast validation, the ``verdict_from_result`` stamp path, and the control-inertness pin
(``is_executable`` derives ONLY from ``source``).  The evaluate()-level ``test_provenance``
stamping on a real fixture thesis is in tests/spike/test_pod_4_1a_code_oracle_spike.py.
"""

from __future__ import annotations

import pytest

from cogworx.verification.contracts import Verdict
from cogworx.verification.oracles.code import (
    CodeOracle,
    _RunResult,  # private — internal unit test only
    verdict_from_result,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _passing_result() -> _RunResult:
    return _RunResult(returncode=0, stdout="1 passed\n", stderr="", timed_out=False)


def _failing_result() -> _RunResult:
    return _RunResult(returncode=1, stdout="1 failed\n", stderr="", timed_out=False)


def _no_tests_result() -> _RunResult:
    return _RunResult(returncode=5, stdout="no tests ran\n", stderr="", timed_out=False)


def _timeout_result() -> _RunResult:
    return _RunResult(returncode=-1, stdout="", stderr="", timed_out=True)


# ---------------------------------------------------------------------------
# Task B — Verdict.test_provenance field contract
# ---------------------------------------------------------------------------


def test_verdict_default_test_provenance_is_na() -> None:
    """Non-CodeOracle Verdict constructions default to 'n/a' (CANON §6.1 additive field)."""
    v = Verdict(holds=True, valid_check=True, reasoning="", source="tool")
    assert v.test_provenance == "n/a"


def test_verdict_test_provenance_not_read_by_is_executable() -> None:
    """is_executable derives ONLY from source — test_provenance is control-inert (S9 pin)."""
    v_tool = Verdict(
        holds=True, valid_check=True, reasoning="", source="tool", test_provenance="frozen"
    )
    v_inference = Verdict(
        holds=False, valid_check=True, reasoning="", source="inference", test_provenance="thesis"
    )
    assert v_tool.is_executable is True
    assert v_inference.is_executable is False


def test_verdict_test_provenance_n_a_does_not_affect_is_executable() -> None:
    """Baseline: n/a does not change is_executable for any source value."""
    assert (
        Verdict(
            holds=True, valid_check=True, reasoning="", source="system", test_provenance="n/a"
        ).is_executable
        is True
    )
    assert (
        Verdict(
            holds=True, valid_check=True, reasoning="", source="inference", test_provenance="n/a"
        ).is_executable
        is False
    )


# ---------------------------------------------------------------------------
# Task A + B — verdict_from_result stamps test_provenance
# ---------------------------------------------------------------------------


def test_verdict_from_result_defaults_to_thesis() -> None:
    """verdict_from_result with no test_provenance kwarg stamps 'thesis' (additive default)."""
    v = verdict_from_result(_passing_result(), timeout_s=30.0)
    assert v.test_provenance == "thesis"
    assert v.source == "tool"


def test_verdict_from_result_thesis_stamp() -> None:
    v = verdict_from_result(_passing_result(), timeout_s=30.0, test_provenance="thesis")
    assert v.test_provenance == "thesis"
    assert v.holds is True
    assert v.valid_check is True


def test_verdict_from_result_frozen_stamp() -> None:
    v = verdict_from_result(_passing_result(), timeout_s=30.0, test_provenance="frozen")
    assert v.test_provenance == "frozen"
    assert v.holds is True
    assert v.valid_check is True


def test_verdict_from_result_failure_carries_test_provenance() -> None:
    v = verdict_from_result(_failing_result(), timeout_s=30.0, test_provenance="frozen")
    assert v.test_provenance == "frozen"
    assert v.holds is False
    assert v.valid_check is True


def test_verdict_from_result_no_tests_carries_test_provenance() -> None:
    v = verdict_from_result(_no_tests_result(), timeout_s=30.0, test_provenance="thesis")
    assert v.test_provenance == "thesis"
    assert v.holds is False
    assert v.valid_check is False


def test_verdict_from_result_timeout_carries_test_provenance() -> None:
    v = verdict_from_result(_timeout_result(), timeout_s=5.0, test_provenance="frozen")
    assert v.test_provenance == "frozen"
    assert v.holds is False
    assert v.valid_check is False


# ---------------------------------------------------------------------------
# Task A — CodeOracle construction invariants
# ---------------------------------------------------------------------------


def test_code_oracle_default_construction() -> None:
    """CodeOracle() is valid; test_source defaults to 'thesis'."""
    oracle = CodeOracle()
    assert oracle._test_source == "thesis"
    assert oracle._frozen_test_code is None


def test_code_oracle_explicit_thesis_source() -> None:
    oracle = CodeOracle(test_source="thesis")
    assert oracle._test_source == "thesis"


def test_code_oracle_frozen_source_with_code() -> None:
    frozen = "def test_ok():\n    assert True\n"
    oracle = CodeOracle(test_source="frozen", frozen_test_code=frozen)
    assert oracle._test_source == "frozen"
    assert oracle._frozen_test_code == frozen


def test_code_oracle_frozen_source_no_code_raises() -> None:
    """Fail-fast: test_source='frozen' with no frozen_test_code is misconfiguration (not S8)."""
    with pytest.raises(ValueError, match="frozen_test_code"):
        CodeOracle(test_source="frozen")


def test_code_oracle_frozen_source_explicit_none_raises() -> None:
    with pytest.raises(ValueError, match="frozen_test_code"):
        CodeOracle(test_source="frozen", frozen_test_code=None)


# ---------------------------------------------------------------------------
# Task C — default preservation: non-CodeOracle Verdict stays "n/a"
# ---------------------------------------------------------------------------


def test_non_code_oracle_verdict_test_provenance_is_na() -> None:
    """Verdict constructed outside CodeOracle always has test_provenance='n/a'."""
    for source in ("tool", "system", "inference", "extraction"):
        v = Verdict(holds=True, valid_check=True, reasoning="r", source=source)
        assert v.test_provenance == "n/a", f"expected n/a for source={source!r}"


def test_verdict_is_frozen_still_holds() -> None:
    """The new field does not break Verdict's frozen config."""
    import pydantic

    v: object = Verdict(holds=True, valid_check=True, reasoning="", source="tool")
    with pytest.raises(pydantic.ValidationError):
        v.test_provenance = "frozen"  # type: ignore[attr-defined]
