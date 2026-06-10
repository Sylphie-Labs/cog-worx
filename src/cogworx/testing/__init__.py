"""The Test Kit — scripted model + in-memory substrate doubles + reference agent (CANON S12).

Deterministic, service-free doubles and a spy model so the invariant suites and the walking skeleton
run with no external substrate. ``fixtures`` is a pytest plugin; import it via
``pytest_plugins = ["cogworx.testing.fixtures"]``.
"""

from __future__ import annotations

from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
    InMemoryProceduralKG,
)
from cogworx.testing.fake_model import (
    ReplayCall,
    ReplayExhaustedError,
    ReplayModel,
    echo_model,
)
from cogworx.testing.reference_agent import (
    IntakeStage,
    RespondStage,
    build_reference_graph,
    reference_initial,
)

__all__ = [
    "InMemoryEntityKG",
    "InMemoryGraphStore",
    "InMemoryJournal",
    "InMemoryLatentStore",
    "InMemoryProceduralKG",
    "IntakeStage",
    "ReplayCall",
    "ReplayExhaustedError",
    "ReplayModel",
    "RespondStage",
    "build_reference_graph",
    "echo_model",
    "reference_initial",
]
