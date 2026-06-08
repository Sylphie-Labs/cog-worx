"""The runtime — the loop driver that owns the loop (CANON S2, S1, S6).

``Engine`` drives a ``StageGraph`` to a terminal ``RunState``, committing each step's result before
advancing (S6) and never calling the model on the write path (S1) or on replay (S6).
``RunContext`` is the concrete ``StageContext`` handed to each stage.
"""

from __future__ import annotations

from cogworx.runtime.context import DispatchError, RunContext
from cogworx.runtime.engine import Clock, Engine, ResumeError

__all__ = [
    "Clock",
    "DispatchError",
    "Engine",
    "ResumeError",
    "RunContext",
]
