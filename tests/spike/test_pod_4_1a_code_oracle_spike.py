"""Pod 4.1a spike (S12) — the deterministic code-execution oracle (CANON S9, S10).

Falsifiable criteria: real pytest execution maps exit codes to the right (holds, valid_check) and
``source="tool"``; the F7 safety rule holds — a tainted drive with no usable sandbox REFUSES to
execute model code (zero execution, ``source="system"``); and the docker tier's hardening argv
carries the security-critical flags.

Per the test-hang discipline: run this file SINGLE and timeout-wrapped — never the whole tests/spike
tier (cross-test resource leak). It spawns real pytest subprocesses (the happy paths), but each is
tiny and the oracle's own hard timeout + process-group kill bounds them; the docker tier is NOT
exercised (no docker in CI) — its argv is verified structurally by ``build_docker_cmd``.
"""

from __future__ import annotations

import pytest

from cogworx.cost.budget import BudgetGuard
from cogworx.runtime.context import RunContext
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import echo_model
from cogworx.verification import Oracle, OracleFrame, Thesis
from cogworx.verification.oracles.code import CodeOracle, build_docker_cmd

pytestmark = pytest.mark.spike

_GOOD_SOLUTION = "def add(a, b):\n    return a + b\n"
_PASSING_TEST = "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"
_FAILING_TEST = "from solution import add\n\n\ndef test_add():\n    assert add(2, 3) == 6\n"
_NO_TESTS = "value = 1  # no test_ function — nothing for pytest to collect\n"
_SLOW_TEST = "import time\n\n\ndef test_slow():\n    time.sleep(30)\n"


def _frame() -> OracleFrame:
    return OracleFrame(
        completion_criterion="empirical", problem_type="python", problem_statement="add two ints"
    )


def _thesis(solution: str, test: str) -> Thesis:
    return Thesis(proposed_solution=solution, experiment_design=test)


async def _ctx(*, tainted: bool = False) -> RunContext:
    journal = InMemoryJournal()
    await journal.start_run(
        "run-code", "sess", pathway_id="p", pathway_version=1, pathway_fingerprint="fp"
    )
    if tainted:
        await journal.set_run_tainted("run-code")
    return RunContext(
        run_id="run-code",
        session_id="sess",
        model=echo_model("unused"),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
    )


def test_code_oracle_satisfies_oracle_protocol() -> None:
    assert isinstance(CodeOracle(), Oracle)


async def test_passing_solution_holds() -> None:
    verdict = await CodeOracle().evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _PASSING_TEST), ctx=await _ctx()
    )
    assert verdict.holds is True
    assert verdict.valid_check is True
    assert verdict.source == "tool"
    assert verdict.is_executable  # an executable verdict MAY back a CONFIRMED claim + Beta update


async def test_failing_solution_is_a_real_refutation() -> None:
    verdict = await CodeOracle().evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _FAILING_TEST), ctx=await _ctx()
    )
    assert verdict.holds is False
    assert verdict.valid_check is True  # tests ran and failed — a real refutation
    assert verdict.source == "tool"


async def test_no_tests_collected_is_invalid() -> None:
    verdict = await CodeOracle().evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _NO_TESTS), ctx=await _ctx()
    )
    assert verdict.holds is False
    assert verdict.valid_check is False  # pytest exit 5 — nothing was actually tested


async def test_timeout_is_invalid() -> None:
    verdict = await CodeOracle(timeout_s=1.0).evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _SLOW_TEST), ctx=await _ctx()
    )
    assert verdict.holds is False
    assert verdict.valid_check is False  # killed on timeout — the experiment did not run cleanly


async def test_refuses_to_execute_on_tainted_drive() -> None:
    # F7/S10: a tainted drive with no usable sandbox REFUSES — it executes nothing. The would-pass
    # trial is never run (holds stays False) and the verdict is a system-sourced refusal.
    oracle = CodeOracle(docker_path="definitely-not-a-real-docker-binary")
    verdict = await oracle.evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _PASSING_TEST), ctx=await _ctx(tainted=True)
    )
    assert verdict.holds is False
    assert verdict.valid_check is False
    assert verdict.source == "system"
    assert "refused" in verdict.reasoning.lower()


async def test_thesis_source_stamps_test_provenance_thesis() -> None:
    """Pod 4.4b: default (thesis) oracle stamps test_provenance='thesis' on evaluate()."""
    verdict = await CodeOracle().evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _PASSING_TEST), ctx=await _ctx()
    )
    assert verdict.test_provenance == "thesis"
    assert verdict.source == "tool"
    assert verdict.is_executable  # control-inertness: test_provenance does NOT touch is_executable


async def test_frozen_source_stamps_test_provenance_frozen_and_uses_frozen_test() -> None:
    """Pod 4.4b: frozen oracle uses the frozen test (not thesis test) and stamps 'frozen'."""
    # The frozen test PASSES the good solution; the thesis experiment_design is the FAILING test.
    # If the oracle used the thesis test instead, holds would be False — which would falsify this.
    frozen = _PASSING_TEST
    oracle = CodeOracle(test_source="frozen", frozen_test_code=frozen)
    verdict = await oracle.evaluate(
        frame=_frame(), thesis=_thesis(_GOOD_SOLUTION, _FAILING_TEST), ctx=await _ctx()
    )
    assert verdict.test_provenance == "frozen"
    assert verdict.holds is True  # frozen test passed
    assert verdict.valid_check is True
    assert verdict.source == "tool"
    assert verdict.is_executable  # test_provenance is audit-only; is_executable still source-only


def test_build_docker_cmd_carries_the_hardening_flags() -> None:
    cmd = build_docker_cmd(tmp_str="/tmp/x", container_name="c", image="img", docker_bin="docker")
    assert "--network" in cmd and cmd[cmd.index("--network") + 1] == "none"
    assert "--cap-drop=ALL" in cmd
    assert "--read-only" in cmd
    assert "--memory=512m" in cmd and "--memory-swap=512m" in cmd
    assert "--pids-limit=64" in cmd
    assert "no-new-privileges" in cmd
    assert "--user" in cmd and cmd[cmd.index("--user") + 1] == "1000:1000"
