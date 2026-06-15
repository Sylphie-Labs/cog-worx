"""Pod 4.0 — the oracle seam foundation: Verdict contract + OracleRegistry resolution.

Deterministic and model-free (no I/O, no Model): exercises the registry's exact -> wildcard ->
always-on-fallback precedence, its totality (never raises — S8), delegation through the Oracle
protocol, and the F1 ``is_executable`` epistemic gate. ``asyncio_mode = "auto"`` (pyproject.toml),
so async tests need no decorator.
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from pydantic import ValidationError

from cogworx.claims.provenance import ProvenanceSource
from cogworx.loop.stage import StageContext
from cogworx.verification import Oracle, OracleFrame, OracleRegistry, Thesis, Verdict

# The oracle stub never touches ctx; a cast keeps the test free of a real StageContext.
_CTX = cast(StageContext, object())
_THESIS = Thesis(proposed_solution="s", experiment_design="e")


class _StubOracle:
    """A deterministic oracle: returns a fixed verdict and records the frames it evaluated."""

    def __init__(
        self,
        *,
        holds: bool,
        source: ProvenanceSource = "tool",
        valid_check: bool = True,
    ) -> None:
        self._verdict = Verdict(
            holds=holds, valid_check=valid_check, reasoning="stub", source=source
        )
        self.seen: list[OracleFrame] = []

    async def evaluate(self, *, frame: OracleFrame, thesis: Thesis, ctx: StageContext) -> Verdict:
        self.seen.append(frame)
        return self._verdict


def _frame(criterion: str, problem_type: str) -> OracleFrame:
    return OracleFrame(
        completion_criterion=criterion, problem_type=problem_type, problem_statement="p"
    )


def _verdict(source: ProvenanceSource) -> Verdict:
    return Verdict(holds=True, valid_check=True, reasoning="", source=source)


def test_stub_satisfies_oracle_protocol() -> None:
    assert isinstance(_StubOracle(holds=True), Oracle)


def test_registry_satisfies_oracle_protocol() -> None:
    # The registry is drop-in wherever a bare Oracle is expected (the tess pattern).
    assert isinstance(OracleRegistry(fallback=_StubOracle(holds=True)), Oracle)


def test_resolve_exact_beats_wildcard_beats_fallback() -> None:
    fallback = _StubOracle(holds=False)
    wildcard = _StubOracle(holds=True)
    exact = _StubOracle(holds=True)
    reg = OracleRegistry(fallback=fallback)
    reg.register(completion_criterion="tests-pass", problem_type="*", oracle=wildcard)
    reg.register(completion_criterion="tests-pass", problem_type="python", oracle=exact)

    assert reg.resolve(completion_criterion="tests-pass", problem_type="python") is exact
    assert reg.resolve(completion_criterion="tests-pass", problem_type="rust") is wildcard
    assert reg.resolve(completion_criterion="proof", problem_type="math") is fallback


def test_resolve_is_total_never_raises() -> None:
    fallback = _StubOracle(holds=False)
    reg = OracleRegistry(fallback=fallback)
    # No registrations: every key resolves to the fallback, never an "unregistered" raise (S8).
    assert reg.resolve(completion_criterion="anything", problem_type="whatever") is fallback


async def test_registry_evaluate_delegates_to_resolved_oracle() -> None:
    fallback = _StubOracle(holds=False)
    exact = _StubOracle(holds=True)
    reg = OracleRegistry(fallback=fallback)
    reg.register(completion_criterion="c", problem_type="t", oracle=exact)

    verdict = await reg.evaluate(frame=_frame("c", "t"), thesis=_THESIS, ctx=_CTX)

    assert verdict.holds is True
    assert exact.seen == [_frame("c", "t")]
    assert fallback.seen == []


def test_verdict_is_executable_only_for_tool_and_system() -> None:
    assert _verdict("tool").is_executable
    assert _verdict("system").is_executable
    assert not _verdict("inference").is_executable
    assert not _verdict("extraction").is_executable


def test_thesis_abstention_default_is_none() -> None:
    # An honest "I cannot establish this" is the default; Pod 4.2 values it over a fabrication.
    assert _THESIS.verifiable_claim is None


def test_verdict_is_frozen() -> None:
    verdict: Any = _verdict("tool")
    with pytest.raises(ValidationError):
        verdict.holds = False
