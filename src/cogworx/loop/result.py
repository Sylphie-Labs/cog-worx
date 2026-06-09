"""The discriminated ``StageResult`` (CANON S6, S8).

A stage returns one of five results — ``transition`` / ``done`` / ``await-human`` / ``degraded`` /
``wait`` — making HITL, graceful degradation, AND durable sleep first-class loop transitions rather
than error states. ``wait`` is the S6 durable-timer transition: a stage parks the run on an absolute
``wake_at`` and the sweeper re-drives it later (a committed ``Wait`` replays as a plain advance to
``to``, never re-parking). These are the most-reused loop types, so they live in a leaf module that
depends only on ``claims`` (the journal commits a ``StageResult``, and the journal must not import
the stage seam).
"""

from __future__ import annotations

from datetime import datetime
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


class Wait(BaseModel):
    """Park the run on a durable timer (S6). The sweeper re-drives it once ``wake_at`` passes.

    ``wake_at`` is an ABSOLUTE instant taken from the engine's injected clock (never wall-clock at
    replay), so a committed ``Wait`` replays deterministically: the engine advances to ``to`` just
    like a ``transition``, cancelling the timer rather than re-arming it.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["wait"] = "wait"
    to: str
    wake_at: datetime
    output: Artifact


StageResult = Annotated[
    Transition | Done | AwaitHuman | Degraded | Wait,
    Field(discriminator="kind"),
]


__all__ = [
    "AwaitHuman",
    "Degraded",
    "Done",
    "StageResult",
    "Transition",
    "Wait",
]
