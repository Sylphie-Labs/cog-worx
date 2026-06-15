"""Configuration for the coherence reconciler (Pod 2.7).

All values are tunable at construction time; the frozen model enforces that a live
reconciler's config is immutable after construction.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class CoherenceConfig(BaseModel):
    """Tunable parameters for the coherence reconciler sweep loop."""

    model_config = ConfigDict(frozen=True)

    cosine_threshold: float = 0.5
    margin: float = 0.15
    lcb_z: float = 1.0
    batch_limit: int = 16
    max_claims_per_subject: int = 128
    max_oracle_calls_per_subject: int = 32
    max_mus_per_subject: int = 4
    max_attempts: int = 5


__all__ = ["CoherenceConfig"]
