"""cogworx.injection — memory injection layer (Pod 2.6).

Public API:

- :data:`DEFAULT_MEMORY_POLICY` — ready-to-use default policy (2048-token budget, no floor, no
  required kinds).
- :class:`InjectedMemory` — result of one injection pass: assembled context + diagnostics.
- :class:`MemoryInjector` — the injector; takes a wired :class:`~cogworx.recall.stack.RecallStack`
  at construction and exposes :meth:`~MemoryInjector.inject`.
- :class:`MemoryPolicy` — declarative policy (budget, floor, required kinds).
- :data:`MemoryStatus` — ``Literal["ok", "unwired", "below_floor"]``.
- :func:`resolve_token_counter` — duck-type helper; extracts ``count_tokens`` from a model object
  or falls back to :func:`~cogworx.recall.assembly.approx_tokens`.

Dependency contract (CANON D3): ``runtime → injection → recall/substrate``.
No runtime imports from ``cogworx.model`` or ``cogworx.runtime``.
"""

from __future__ import annotations

from cogworx.injection.injector import MemoryInjector, resolve_token_counter
from cogworx.injection.policy import (
    DEFAULT_MEMORY_POLICY,
    InjectedMemory,
    MemoryPolicy,
    MemoryStatus,
)

__all__ = [
    "DEFAULT_MEMORY_POLICY",
    "InjectedMemory",
    "MemoryInjector",
    "MemoryPolicy",
    "MemoryStatus",
    "resolve_token_counter",
]
