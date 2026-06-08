"""S2 structure tests: ``StageGraph`` validates termination and structure by construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph, StageGraphError
from cogworx.loop.result import Done, StageResult
from cogworx.loop.stage import StageContext


def _artifact() -> Artifact:
    return Artifact(
        kind="summary",
        produced_by="stub",
        provenance=Provenance(source="reflection", confidence=0.5, recorded_at=datetime.now(UTC)),
    )


@dataclass
class _StubStage:
    name: str
    transitions: tuple[str, ...] = field(default=())

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_artifact())


def test_linear_graph_with_terminal_constructs() -> None:
    graph = StageGraph(
        [_StubStage("a", ("b",)), _StubStage("b", ("c",)), _StubStage("c")],
        entry="a",
    )
    assert graph.entry == "a"
    assert graph.names() == ("a", "b", "c")
    assert graph.transitions_from("a") == ("b",)
    assert graph.is_terminal("c")
    assert not graph.is_terminal("a")
    assert graph.get("b").name == "b"


def test_empty_graph_errors() -> None:
    with pytest.raises(StageGraphError):
        StageGraph([], entry="a")


def test_duplicate_names_error() -> None:
    with pytest.raises(StageGraphError):
        StageGraph([_StubStage("a", ("a",)), _StubStage("a")], entry="a")


def test_entry_not_found_errors() -> None:
    with pytest.raises(StageGraphError):
        StageGraph([_StubStage("a")], entry="missing")


def test_dangling_transition_target_errors() -> None:
    with pytest.raises(StageGraphError):
        StageGraph([_StubStage("a", ("ghost",))], entry="a")


def test_unreachable_stage_errors() -> None:
    with pytest.raises(StageGraphError):
        StageGraph([_StubStage("a"), _StubStage("orphan")], entry="a")


def test_no_reachable_terminal_errors() -> None:
    with pytest.raises(StageGraphError):
        StageGraph([_StubStage("a", ("b",)), _StubStage("b", ("a",))], entry="a")


def test_get_unknown_stage_errors() -> None:
    graph = StageGraph([_StubStage("a")], entry="a")
    with pytest.raises(StageGraphError):
        graph.get("missing")
