"""Spine / loop layer (CANON S2, S6, S8, S9, S11): FSM states and stage composition primitives.

Public API is identical to the pre-lazy form; the PEP-562 ``__getattr__`` replaces eager imports of
``stage`` and ``graph`` (which pull ``cogworx.model.base`` and, transitively,
``cogworx.substrate.journal`` at import time, forming the cycle documented in S9's framing):

    substrate.journal → loop.result → loop/__init__ → stage/graph → substrate.journal

The lazy loader maps each public name to its source module; the module is imported and the attribute
cached on FIRST ACCESS.  ``result``, ``retry``, and ``state`` are still imported eagerly — they are
true leaf modules with no substrate dependency (``result`` imports only ``claims.provenance``;
``retry`` and ``state`` have no cogworx imports beyond stdlib/pydantic), so eagerly pulling them
does NOT widen the cycle.  Only ``stage`` and ``graph`` are deferred.

CANON refs: S2 (own the loop), S6 (durable exactly-once), S8 (graceful degradation),
S9 (structure over prompting), S11 (cost bounded structurally).
"""

from __future__ import annotations

import importlib
import sys
from typing import Any

# ── Eagerly-safe leaf imports (no substrate dependency) ───────────────────────
from cogworx.loop.result import AwaitHuman, Degraded, Done, StageResult, Transition, Wait
from cogworx.loop.retry import DEFAULT_RETRY_POLICY, RetryPolicy
from cogworx.loop.state import RunStatus, StageStatus

# ── Lazy-load map: attribute name → (module_path, attr_in_module) ────────────
# Only ``stage`` and ``graph`` are deferred — they import cogworx.model.base and
# (via TYPE_CHECKING-band-aided edges) are on the substrate.journal cycle path.
_LAZY: dict[str, tuple[str, str]] = {
    "Stage": ("cogworx.loop.stage", "Stage"),
    "StageContext": ("cogworx.loop.stage", "StageContext"),
    "Loop": ("cogworx.loop.graph", "Loop"),
    "StageGraph": ("cogworx.loop.graph", "StageGraph"),
    "StageGraphError": ("cogworx.loop.graph", "StageGraphError"),
}

__all__ = [
    "DEFAULT_RETRY_POLICY",
    "AwaitHuman",
    "Degraded",
    "Done",
    "Loop",
    "RetryPolicy",
    "RunStatus",
    "Stage",
    "StageContext",
    "StageGraph",
    "StageGraphError",
    "StageResult",
    "StageStatus",
    "Transition",
    "Wait",
]


def __getattr__(name: str) -> Any:
    """PEP-562 lazy loader: import ``stage``/``graph`` only when first accessed."""
    if name in _LAZY:
        mod_path, attr = _LAZY[name]
        mod = importlib.import_module(mod_path)
        value = getattr(mod, attr)
        # Cache on this module so subsequent accesses skip __getattr__ entirely.
        setattr(sys.modules[__name__], name, value)
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
