"""The discriminated ``StageResult`` (CANON S8).

A stage returns one of four results — ``transition`` / ``done`` / ``await-human`` / ``degraded`` —
making HITL and graceful degradation first-class loop transitions rather than error states. These
are the most-reused loop types, so they live in a leaf module that depends only on ``claims`` (the
journal commits a ``StageResult``, and the journal must not import the stage seam).
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from cogworx.claims.provenance import Artifact


class Transition(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["transition"] = "transition"
    to: str
    output: Artifact


class Done(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["done"] = "done"
    output: Artifact


class AwaitHuman(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["await-human"] = "await-human"
    question: str
    output: Artifact | None = None


class Degraded(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["degraded"] = "degraded"
    reason: str
    output: Artifact
    to: str | None = None


StageResult = Annotated[
    Transition | Done | AwaitHuman | Degraded,
    Field(discriminator="kind"),
]


__all__ = [
    "AwaitHuman",
    "Degraded",
    "Done",
    "StageResult",
    "Transition",
]
