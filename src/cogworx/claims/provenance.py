"""Provenance + epistemic typing on every claim (CANON S5).

Every substrate claim carries its epistemic level (observation | inference | confirmed) and the
provenance that produced it, so "why did this surface" is a graph traversal. These are frozen
contract types — claims and the artifacts that carry them are immutable.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

DEFAULT_SCOPE: Final[str] = "agent"

EpistemicType = Literal["observation", "inference", "confirmed"]
# "system" = engine/control-plane-originated, zero-model (exhaustion degradations, timeouts,
# ceiling-fails). Distinct from "reflection"/"inference"/"extraction", which S1 reserves for
# model-heavy cognition — a deterministic control event must not borrow that epistemic weight.
ProvenanceSource = Literal[
    "human", "sensor", "tool", "extraction", "reflection", "inference", "system"
]


class Provenance(BaseModel):
    """Where a claim came from and how much we trust it."""

    model_config = ConfigDict(frozen=True)

    source: ProvenanceSource
    source_ref: str | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: tuple[str, ...] = ()
    recorded_at: datetime


class Claim(BaseModel):
    """A bi-temporal, provenance-typed substrate claim."""

    model_config = ConfigDict(frozen=True)

    id: str
    subject: str
    predicate: str | None = None
    payload: str
    epistemic_type: EpistemicType
    provenance: Provenance
    valid_from: datetime
    valid_to: datetime | None = None
    ingest_time: datetime
    created_by: str
    embedding: tuple[float, ...] | None = None
    # Set when the claim's object is itself an entity — drives a [:REFERS_TO] edge in the entity
    # KG and is used as object_repr in claim_id_for (CANON S5, entity-KG identity discipline).
    # Default None: additive, backward-compatible with all Phase-0 callers.
    object_entity: str | None = None
    # Which governed view owns this claim: "agent" (unscoped default), "world", or "user:<id>".
    # Default "agent" is backward-compatible — all pre-2.4 claims are unscoped.
    scope: str = DEFAULT_SCOPE


class Artifact(BaseModel):
    """A stage output that carries provenance."""

    model_config = ConfigDict(frozen=True)

    kind: str
    produced_by: str
    provenance: Provenance
    data: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "DEFAULT_SCOPE",
    "Artifact",
    "Claim",
    "EpistemicType",
    "Provenance",
    "ProvenanceSource",
]
