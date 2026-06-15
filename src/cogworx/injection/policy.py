"""Memory injection policy and assembled-context result types (Pod 2.6).

No model imports at runtime — this module sits below cogworx.runtime in the dependency graph
(runtime → injection → recall/substrate, CANON D3).

Contract changelog (CANON §6.1 — additive changes; no sign-off required):
  2026-06-12  Pod 3.1b CF-B  MemoryPolicy.include_defeated (bool, default False) — escape hatch
              to retain defeasibly-defeated claims in the injected context.
  2026-06-12  Pod 3.1b CF-B  InjectedMemory.defeated_excluded (int, default 0) — diagnostic count
              of claims suppressed by the defeated-claim exclusion filter.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from cogworx.recall.results import AssembledContext, ContextChunk  # noqa: F401 — re-exported
from cogworx.recall.stack import ChannelStatus

__all__ = [
    "DEFAULT_MEMORY_POLICY",
    "InjectedMemory",
    "MemoryPolicy",
    "MemoryStatus",
]

MemoryStatus = Literal["ok", "unwired", "below_floor"]


class MemoryPolicy(BaseModel):
    """Declarative policy governing how much memory to inject and which kinds are required."""

    model_config = ConfigDict(frozen=True)

    token_budget: int = 2048
    min_per_kind: int = 0
    required_kinds: tuple[Literal["claim", "episode", "latent"], ...] = ()
    include_defeated: bool = False
    """When ``True``, skip the defeasibly-defeated claim filter so defeated claims may appear
    in the assembled context.  Default ``False`` (defeated claims are excluded, Pod 3.1b CF-B).
    """


DEFAULT_MEMORY_POLICY: MemoryPolicy = MemoryPolicy()


class InjectedMemory(BaseModel):
    """The result of a memory injection pass — the assembled context plus diagnostic metadata."""

    model_config = ConfigDict(frozen=True)

    context: AssembledContext
    status: MemoryStatus
    missing_kinds: tuple[str, ...] = ()
    kinds_present: tuple[str, ...] = ()
    channel_status: tuple[ChannelStatus, ...] = ()
    latent_uses_recorded: int = 0
    record_use_error: str | None = None
    defeated_excluded: int = 0
    """Count of fused results suppressed because their claim status was ``"defeasibly-defeated"``.
    Zero when no defeated claims were surfaced or when ``MemoryPolicy.include_defeated=True``
    (Pod 3.1b CF-B).
    """
