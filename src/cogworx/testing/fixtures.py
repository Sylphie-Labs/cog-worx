"""Pytest fixtures exposing the Test Kit as an importable plugin (CANON S1, S6, S12).

The deterministic tier wires the engine entirely from in-memory doubles + a ``ReplayModel`` and a
counter clock that returns fixed, strictly increasing timestamps — so every run is reproducible and
resume is replay-safe. Register via ``pytest_plugins = ["cogworx.testing.fixtures"]``.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.runtime.engine import Clock, Engine
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel, echo_model
from cogworx.testing.reference_agent import build_reference_graph, reference_pathways

_EPOCH = datetime(2026, 1, 1, tzinfo=UTC)


def _counter_clock() -> Clock:
    state = {"tick": 0}

    def clock() -> datetime:
        moment = _EPOCH + timedelta(seconds=state["tick"])
        state["tick"] += 1
        return moment

    return clock


@pytest.fixture
def replay_model() -> ReplayModel:
    return echo_model("hello from replay")


@pytest.fixture
def in_memory_journal() -> InMemoryJournal:
    return InMemoryJournal()


@pytest.fixture
def in_memory_graph() -> InMemoryGraphStore:
    return InMemoryGraphStore()


@pytest.fixture
def in_memory_latent() -> InMemoryLatentStore:
    return InMemoryLatentStore()


@pytest.fixture
def reference_graph() -> StageGraph:
    return build_reference_graph()


@pytest.fixture
def pathways() -> PathwayRegistry:
    return reference_pathways()


@pytest.fixture
def counter_clock() -> Clock:
    return _counter_clock()


@pytest.fixture
def engine(
    replay_model: ReplayModel,
    in_memory_journal: InMemoryJournal,
    in_memory_graph: InMemoryGraphStore,
    in_memory_latent: InMemoryLatentStore,
    pathways: PathwayRegistry,
    counter_clock: Callable[[], datetime],
) -> Engine:
    return Engine(
        model=replay_model,
        journal=in_memory_journal,
        graph_store=in_memory_graph,
        latent=in_memory_latent,
        pathways=pathways,
        clock=counter_clock,
    )


__all__ = [
    "counter_clock",
    "engine",
    "in_memory_graph",
    "in_memory_journal",
    "in_memory_latent",
    "pathways",
    "reference_graph",
    "replay_model",
]
