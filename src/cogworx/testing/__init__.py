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
from cogworx.testing.reference_dialectic import (
    DIALECTIC_PATHWAY_ID,
    PLANTED_CORRECT_THESIS,
    PLANTED_FLAWED_THESIS,
    PLANTED_INJECTION_STRING,
    PLANTED_SECRET_TOKEN,
    StubJudgeOracle,
    StubOracle,
    build_dialectic_graph,
    build_dialectic_stages,
    dialectic_initial,
    dialectic_pathways,
    make_stub_oracle_registry,
)

__all__ = [
    "DIALECTIC_PATHWAY_ID",
    "PLANTED_CORRECT_THESIS",
    "PLANTED_FLAWED_THESIS",
    "PLANTED_INJECTION_STRING",
    "PLANTED_SECRET_TOKEN",
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
    "StubJudgeOracle",
    "StubOracle",
    "build_dialectic_graph",
    "build_dialectic_stages",
    "build_reference_graph",
    "dialectic_initial",
    "dialectic_pathways",
    "echo_model",
    "make_stub_oracle_registry",
    "reference_initial",
]
