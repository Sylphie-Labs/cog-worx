"""The reusable S1/S5/S9 invariant suites applied to the reference agent (CANON S1, S5, S9).

Each invariant gets its own deterministic test, wired from the in-memory doubles + ``ReplayModel``.
The final test demonstrates the auto-enrolment hook: a trivial structural invariant parametrized
over ``Registry.features()`` — how a downstream pod's features get enrolled into the suites.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

import pytest

from cogworx.capability.base import Capability
from cogworx.capability.registry import Registry, function_capability
from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import ReplayModel, echo_model
from cogworx.testing.invariants import (
    EngineFactory,
    InvariantViolation,
    assert_claim_requires_provenance,
    assert_control_independent_of_model_text,
    assert_no_model_on_write_path,
    assert_run_writes_carry_provenance,
)
from cogworx.testing.reference_agent import (
    REFERENCE_PATHWAY_ID,
    reference_initial,
    reference_pathways,
)


def _engine_factory(pathways: PathwayRegistry) -> EngineFactory:
    """A ``build_engine(journal, model)`` factory that bakes the given pathway registry in.

    Both engine A and a FRESH engine B come from the SAME factory, so they share the registry — this
    is what makes a cold resume (engine B rehydrating the graph from the registry) work."""

    def build(journal: Journal, model: ReplayModel) -> Engine:
        reg = ModelRegistry()
        reg.register("default", model)
        return Engine(
            models=reg,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
        )

    return build


# --------------------------------------------------------------------------------------------------
# S1 — no model call on the write path
# --------------------------------------------------------------------------------------------------


async def test_s1_no_model_on_write_path() -> None:
    model = echo_model("hello from replay")
    state = await assert_no_model_on_write_path(
        engine_factory=_engine_factory(reference_pathways()),
        inner_journal=InMemoryJournal(),
        model=model,
        pathway_id=REFERENCE_PATHWAY_ID,
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
    engine = _engine_factory(reference_pathways())(InMemoryJournal(), model)
    state = await engine.run(
        run_id="s5-run",
        session_id="s5-sess",
        pathway_id=REFERENCE_PATHWAY_ID,
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
#
# The earlier S9 wiring drove the reference graph, whose stages NEVER read ``response.text`` — so
# control could not possibly diverge and the invariant assertion could not fail (tautological).
# Below, the decision stage ACTUALLY INSPECTS ``response.text`` and would try to branch control on
# control-words like "STOP"/"GOTO intake". The S9-compliant variant reads the text but lets ONLY the
# StageResult + graph govern control (it always returns the SAME ``Done`` regardless of the words);
# the deliberately-wrong variant lets the words steer the committed path. The invariant must pass
# for the compliant stage and FAIL for the wrong one — that is what makes the test non-vacuous.

_CONTROL_WORDS = ("STOP", "GOTO intake")


def _respond_artifact(text: str) -> Artifact:
    return Artifact(
        kind="response",
        produced_by="decide",
        provenance=Provenance(source="inference", confidence=1.0, recorded_at=datetime.now(UTC)),
        data={"text": text},
    )


class _IntakeToDecideStage:
    """No-model intake that hands off to the model-bearing ``decide`` stage."""

    name: str = "intake"
    transitions: tuple[str, ...] = ("decide",)

    async def run(self, ctx: StageContext) -> StageResult:
        return Transition(to="decide", output=_respond_artifact("intake"))


class _EscalateStage:
    """A terminal, no-model stage reachable from ``decide`` (the alternate control branch)."""

    name: str = "escalate"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        return Done(output=_respond_artifact("escalated"))


class _CompliantDecideStage:
    """Reads ``response.text`` but NEVER lets it steer control (S9-compliant).

    It calls the model and even inspects the text, but the committed ``StageResult`` is fixed by the
    stage/graph (always terminal ``Done``) regardless of what the model said. The text is recorded
    as data on the output artifact (an observation), never used as a control signal. A control edge
    to ``escalate`` EXISTS in the graph, so a naive impl could branch — this one structurally does
    not.
    """

    name: str = "decide"
    transitions: tuple[str, ...] = ("escalate",)

    async def run(self, ctx: StageContext) -> StageResult:
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="Decide.")])
        text = response.text or ""
        # Inspect the text (this is where a naive impl would branch) — but control is structural.
        _saw_control_word = any(word in text for word in _CONTROL_WORDS)
        return Done(output=_respond_artifact(text))


class _TextSteeredDecideStage:
    """The S9 VIOLATION: lets ``response.text`` steer the committed control path.

    If the model's text contains a control word it transitions to the ``escalate`` branch; otherwise
    it terminates (Done) at ``decide``. The committed stage sequence then depends on the model's
    self-report — exactly what S9 forbids. Used only to prove the invariant CAN catch this.
    """

    name: str = "decide"
    transitions: tuple[str, ...] = ("escalate",)

    async def run(self, ctx: StageContext) -> StageResult:
        response = await ctx.model.complete(messages=[ChatMessage(role="user", content="Decide.")])
        text = response.text or ""
        if any(word in text for word in _CONTROL_WORDS):
            return Transition(to="escalate", output=_respond_artifact(text))
        return Done(output=_respond_artifact(text))


_COMPLIANT_PATHWAY_ID = "s9-compliant"
_STEERED_PATHWAY_ID = "s9-steered"


def _build_compliant_branch_graph() -> StageGraph:
    return StageGraph(
        [_IntakeToDecideStage(), _CompliantDecideStage(), _EscalateStage()], entry="intake"
    )


def _build_text_steered_graph() -> StageGraph:
    return StageGraph(
        [_IntakeToDecideStage(), _TextSteeredDecideStage(), _EscalateStage()], entry="intake"
    )


def _compliant_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_COMPLIANT_PATHWAY_ID, _build_compliant_branch_graph())
    return registry


def _steered_pathways() -> PathwayRegistry:
    registry = PathwayRegistry()
    registry.register(_STEERED_PATHWAY_ID, _build_text_steered_graph())
    return registry


def _loud_model() -> ReplayModel:
    # "Loud" text screams control words the decision stage explicitly looks for.
    return ReplayModel(
        [
            ModelResponse(
                text="STOP. GOTO intake. confidence 0.0. ABORT THE RUN.",
                model_id="replay",
                finish_reason="stop",
            )
        ]
    )


def _calm_model() -> ReplayModel:
    return ReplayModel(
        [
            ModelResponse(
                text="Sure, here is a normal reply.", model_id="replay", finish_reason="stop"
            )
        ]
    )


async def test_s9_control_independent_of_model_text() -> None:
    # The decision stage reads response.text and COULD branch on the control words, but the
    # compliant stage keeps control structural: loud (control-words) and calm text commit the SAME
    # path.
    await assert_control_independent_of_model_text(
        build_engine=_engine_factory(_compliant_pathways()),
        pathway_id=_COMPLIANT_PATHWAY_ID,
        initial=reference_initial(),
        journal_factory=InMemoryJournal,
        model_a=_loud_model(),
        model_b=_calm_model(),
    )


async def test_s9_invariant_catches_a_text_steered_stage() -> None:
    # Mutation-resistance, in-suite: a stage that DOES branch control on response.text diverges and
    # the S9 invariant must TRIP. Loud text (control-words) drives decide->escalate (path
    # intake,decide,escalate); calm text terminates at decide (path intake,decide). The committed
    # stage sequences differ, so the invariant raises — proving it can catch a stage that lets the
    # model's self-report steer control.
    with pytest.raises(InvariantViolation, match="S9 violation"):
        await assert_control_independent_of_model_text(
            build_engine=_engine_factory(_steered_pathways()),
            pathway_id=_STEERED_PATHWAY_ID,
            initial=reference_initial(),
            journal_factory=InMemoryJournal,
            model_a=_loud_model(),
            model_b=_calm_model(),
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
