"""The trivial reference agent for the walking skeleton (CANON S1, S2).

Two stages — ``intake`` (no model call) then ``respond`` (the single model call) — wired into a
validated ``StageGraph``. This is user code in the Test Kit, so ``datetime.now(UTC)`` is acceptable
here for artifact provenance timestamps; the *framework* never reads a clock outside the injected
``Engine`` clock.
"""

from __future__ import annotations

from datetime import UTC, datetime

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.loop.graph import StageGraph
from cogworx.loop.result import Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.model.base import ChatMessage


class IntakeStage:
    name: str = "intake"
    transitions: tuple[str, ...] = ("respond",)

    async def run(self, ctx: StageContext) -> StageResult:
        artifact = Artifact(
            kind="intake",
            produced_by="intake",
            provenance=Provenance(source="human", confidence=1.0, recorded_at=datetime.now(UTC)),
        )
        return Transition(to="respond", output=artifact)


class RespondStage:
    name: str = "respond"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        ctx.budget.check()
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Respond to the intake.")]
        )
        ctx.budget.record(response.usage)
        artifact = Artifact(
            kind="response",
            produced_by="respond",
            provenance=Provenance(
                source="inference", confidence=1.0, recorded_at=datetime.now(UTC)
            ),
            data={"text": response.text or ""},
        )
        return Done(output=artifact)


def build_reference_graph() -> StageGraph:
    return StageGraph([IntakeStage(), RespondStage()], entry="intake")


def reference_initial() -> Artifact:
    return Artifact(
        kind="user-input",
        produced_by="human",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=datetime.now(UTC)),
        data={"text": "hello"},
    )


__all__ = [
    "IntakeStage",
    "RespondStage",
    "build_reference_graph",
    "reference_initial",
]
