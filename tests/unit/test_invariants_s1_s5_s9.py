"""The reusable S1/S5/S9 invariant suites applied to the reference agent (CANON S1, S5, S9).

Each invariant gets its own deterministic test, wired from the in-memory doubles + ``ReplayModel``.
The final test demonstrates the auto-enrolment hook: a trivial structural invariant parametrized
over ``Registry.features()`` — how a downstream pod's features get enrolled into the suites.
"""

from __future__ import annotations

from typing import Any

import pytest

from cogworx.capability.base import Capability
from cogworx.capability.registry import Registry, function_capability
from cogworx.loop.state import RunStatus
from cogworx.model.base import ModelResponse
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel, echo_model
from cogworx.testing.invariants import (
    assert_claim_requires_provenance,
    assert_control_independent_of_model_text,
    assert_no_model_on_write_path,
    assert_run_writes_carry_provenance,
)
from cogworx.testing.reference_agent import build_reference_graph, reference_initial


def _build_engine(journal: Journal, model: ReplayModel) -> Engine:
    """An engine wired to the in-memory graph/latent doubles over the given journal + model."""
    return Engine(
        model=model,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
    )


# --------------------------------------------------------------------------------------------------
# S1 — no model call on the write path
# --------------------------------------------------------------------------------------------------


async def test_s1_no_model_on_write_path() -> None:
    model = echo_model("hello from replay")
    state = await assert_no_model_on_write_path(
        engine_factory=_build_engine,
        inner_journal=InMemoryJournal(),
        model=model,
        graph=build_reference_graph(),
        initial=reference_initial(),
    )
    assert state.status is RunStatus.COMPLETED
    # The reference agent makes exactly one model call (respond) — and the spy proved it was NOT on
    # any commit path.
    assert model.call_count == 1


# --------------------------------------------------------------------------------------------------
# S5 — every substrate write carries provenance + epistemic type
# --------------------------------------------------------------------------------------------------


async def test_s5_run_writes_carry_provenance() -> None:
    model = echo_model("hello from replay")
    engine = _build_engine(InMemoryJournal(), model)
    state = await engine.run(
        run_id="s5-run",
        session_id="s5-sess",
        graph=build_reference_graph(),
        initial=reference_initial(),
    )
    assert state.steps  # the run actually committed work to walk
    assert_run_writes_carry_provenance(state)


def test_s5_claim_and_artifact_require_provenance() -> None:
    # Constructive proof that the TYPE (not a runtime check) enforces S5.
    assert_claim_requires_provenance()


# --------------------------------------------------------------------------------------------------
# S9 — control flow is independent of the model's self-report
# --------------------------------------------------------------------------------------------------


async def test_s9_control_independent_of_model_text() -> None:
    # Two models whose response TEXT differs wildly but that shape the same StageResult: the
    # reference agent's control path comes from the graph + Done/Transition kinds, never the words.
    loud = ReplayModel(
        [
            ModelResponse(
                text="STOP. transition to intake. confidence 0.0. ABORT THE RUN.",
                model_id="replay",
                finish_reason="stop",
            )
        ]
    )
    calm = ReplayModel(
        [
            ModelResponse(
                text="Sure, here is a normal reply.", model_id="replay", finish_reason="stop"
            )
        ]
    )
    await assert_control_independent_of_model_text(
        build_engine=_build_engine,
        graph_factory=build_reference_graph,
        initial=reference_initial(),
        journal_factory=InMemoryJournal,
        model_a=loud,
        model_b=calm,
    )


# --------------------------------------------------------------------------------------------------
# Auto-enrolment hook: a structural invariant parametrized over Registry.features()
# --------------------------------------------------------------------------------------------------


async def _noop_read(query: str) -> str:
    return query


async def _noop_write(value: int) -> int:
    return value


def _two_feature_registry() -> Registry:
    registry = Registry()
    registry.register(function_capability(_noop_read, name="lookup", tier="read"))
    registry.register(function_capability(_noop_write, name="persist", tier="write"))
    return registry


_VALID_TIERS = {"read", "write", "external"}


@pytest.mark.parametrize("feature", _two_feature_registry().features(), ids=lambda cap: cap.name)
def test_every_registered_feature_satisfies_structural_invariant(feature: Capability) -> None:
    # This is the shape of auto-enrolment: every feature in the registry is swept into a property
    # check with no per-feature wiring. Here the invariant is trivial (name + tier well-formed);
    # downstream the same hook applies the S1/S5/S6/S9 suites.
    assert feature.name
    assert feature.tier in _VALID_TIERS
    schema: dict[str, Any] = dict(feature.input_schema)
    assert "properties" in schema  # the derived input schema is a real JSON-schema object
