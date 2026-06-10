"""The runtime — the loop driver that owns the loop (CANON S2, S1, S6).

``Engine`` drives a ``StageGraph`` to a terminal ``RunState``, committing each step's result before
advancing (S6) and never calling the model on the write path (S1) or on replay (S6).
``RunContext`` is the concrete ``StageContext`` handed to each stage. ``Sweeper`` is the
off-write-path poller that leases due durable timers and re-drives their parked runs via the engine
(S1, S6). ``RunHandle`` is the in-process reference returned by ``Engine.start`` (fire-and-forget).
"""

from __future__ import annotations

from cogworx.runtime.context import RunContext
from cogworx.runtime.engine import Clock, Engine, ResumeError, RunHandle
from cogworx.runtime.projector import (
    DEFAULT_PROJECTION_CONSUMER,
    ResolvedOutcome,
    TrialProjector,
    resolve_outcome,
)
from cogworx.runtime.sweeper import Sweeper

__all__ = [
    "DEFAULT_PROJECTION_CONSUMER",
    "Clock",
    "Engine",
    "ResolvedOutcome",
    "ResumeError",
    "RunContext",
    "RunHandle",
    "Sweeper",
    "TrialProjector",
    "resolve_outcome",
]
