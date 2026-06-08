"""Stage composition primitives (CANON S2, S8).

The Spine is a graph of stages. A ``Stage`` runs against a ``StageContext`` and returns a
discriminated ``StageResult`` — ``transition`` / ``done`` / ``await-human`` / ``degraded`` — making
HITL and graceful degradation first-class loop transitions. Ported from biz-firm's composition
primitives (Stage · Capability · Context · Loop).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Annotated, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from cogworx.claims.provenance import Artifact
from cogworx.coordination.events import Event
from cogworx.model.base import Model
from cogworx.substrate.graph_store import GraphStore
from cogworx.substrate.journal import Journal
from cogworx.substrate.latent import LatentStore


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


@runtime_checkable
class StageContext(Protocol):
    """Everything a stage is handed by the runner. Substrate seams stay distinct (S3)."""

    run_id: str
    session_id: str

    # budget added in cost seam (Pass B)

    @property
    def model(self) -> Model: ...

    @property
    def journal(self) -> Journal: ...

    @property
    def graph(self) -> GraphStore: ...

    @property
    def latent(self) -> LatentStore: ...

    def emit(self, event: Event) -> None: ...

    async def dispatch(self, capability: str, args: Mapping[str, Any]) -> Any: ...


@runtime_checkable
class Stage(Protocol):
    """One node in the Spine's graph of stages."""

    name: str
    transitions: tuple[str, ...]

    async def run(self, ctx: StageContext) -> StageResult: ...


__all__ = [
    "AwaitHuman",
    "Degraded",
    "Done",
    "Stage",
    "StageContext",
    "StageResult",
    "Transition",
]
