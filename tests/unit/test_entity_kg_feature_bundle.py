"""Pod 2.0 Feature Test Bundle — CANON invariants S1, S5, S8 (ROADMAP Definition of Done).

Tier 2 of the DoD: wires the entity-KG write into the walking-skeleton Engine and checks every
invariant mechanically. No Neo4j — everything runs on InMemoryEntityKG + InMemoryJournal +
ReplayModel.

S5 — Provenance + epistemic typing on every claim:
  - Claim construction without provenance raises pydantic.ValidationError (type enforces S5).
  - EvidenceEvent construction without source_id or recorded_at raises pydantic.ValidationError.
  - assert_claim_requires_provenance() reusable check passes.

S1 — Model work OFF the write path (structural):
  - cogworx.knowledge and cogworx.testing.doubles import cleanly without pulling in cogworx.model.
  - A 2-stage pathway (model stage → KG-write stage) driven through CommitSpyJournal never
    increments model.call_count during the KG-write commit.

S8 — Graceful degradation (the Lesion Test):
  - Register the entity-KG write as a function_capability in a Registry.
  - Disable it; show the pathway still completes (Degraded or Done).
  - The run reaches a terminal state with NO claim written.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import UTC, datetime

import pydantic
import pytest

from cogworx.capability.base import CapabilityUnavailable
from cogworx.capability.registry import Registry, RegistryError, function_capability
from cogworx.claims.provenance import Artifact, Claim, Provenance, ProvenanceSource
from cogworx.knowledge.evidence import EvidenceEvent
from cogworx.knowledge.identity import claim_id_for
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Degraded, Done, StageResult, Transition
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.base import ChatMessage, ModelResponse
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel, echo_model
from cogworx.testing.invariants import (
    assert_claim_requires_provenance,
    assert_no_model_on_write_path,
    assert_run_writes_carry_provenance,
)

_NOW = datetime(2026, 6, 9, tzinfo=UTC)

_KG_WRITE_CAP = "entity_kg_write"


# ---------------------------------------------------------------------------
# Shared pathway helpers
# ---------------------------------------------------------------------------


def _artifact(text: str = "ok", *, source: ProvenanceSource = "inference") -> Artifact:
    return Artifact(
        kind="output",
        produced_by="stage",
        provenance=Provenance(
            source=source,
            confidence=1.0,
            recorded_at=_NOW,
        ),
        data={"text": text},
    )


def _make_kg_stage(
    kg: InMemoryEntityKG,
    *,
    registry: Registry | None = None,
) -> type:
    """Return a Stage class that writes one claim to ``kg`` (or skips/degrades on RegistryError)."""

    class _KGWriteStage:
        name: str = "kg_write"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            # Optionally dispatch through the registry — if disabled, RegistryError fires.
            if registry is not None:
                try:
                    await ctx.dispatch(
                        _KG_WRITE_CAP,
                        {"subject": "pluto", "predicate": "has_mass", "payload": "1.3e22 kg"},
                    )
                except (RegistryError, CapabilityUnavailable):
                    # S8: capability disabled → degrade gracefully, never hard-fail.
                    # ctx.dispatch wraps RegistryError in CapabilityUnavailable; catch both.
                    return Degraded(
                        reason="entity_kg_write disabled",
                        output=_artifact("degraded", source="system"),
                    )
            else:
                # Direct write (no registry guard).
                claim = Claim(
                    id=claim_id_for("pluto", "has_mass", "1.3e22 kg"),
                    subject="pluto",
                    predicate="has_mass",
                    payload="1.3e22 kg",
                    epistemic_type="inference",
                    provenance=Provenance(
                        source="extraction",
                        confidence=0.9,
                        recorded_at=_NOW,
                    ),
                    valid_from=_NOW,
                    ingest_time=_NOW,
                    created_by="stage:kg_write",
                )
                ev = EvidenceEvent(
                    id="ev-001",
                    type="corroboration",
                    polarity="+",
                    source_id="extraction-run",
                    source_authority=0.9,
                    base_weight=1.0,
                    recorded_at=_NOW,
                )
                await kg.write_claim(claim, evidence=ev)

            return Done(output=_artifact("wrote claim", source="extraction"))

    return _KGWriteStage


def _make_model_stage() -> type:
    """A stage that calls the model (extraction simulation)."""

    class _ModelStage:
        name: str = "model_call"
        transitions: tuple[str, ...] = ("kg_write",)

        async def run(self, ctx: StageContext) -> StageResult:
            resp = await ctx.model.complete(
                messages=[ChatMessage(role="user", content="Extract entity.")]
            )
            return Transition(
                to="kg_write",
                output=_artifact(resp.text or "extracted", source="extraction"),
            )

    return _ModelStage


def _build_extraction_pathway(
    kg: InMemoryEntityKG,
    *,
    registry: Registry | None = None,
) -> StageGraph:
    return StageGraph(
        [_make_model_stage()(), _make_kg_stage(kg, registry=registry)()],
        entry="model_call",
    )


_PATHWAY_ID = "extraction-with-kg-write"


def _engine_factory(
    pathways: PathwayRegistry,
) -> type:
    """Return a factory for Engine."""

    def build(journal: Journal, model: ReplayModel) -> Engine:
        _reg = ModelRegistry()
        _reg.register("default", model)
        return Engine(
            models=_reg,
            journal=journal,
            graph_store=InMemoryGraphStore(),
            latent=InMemoryLatentStore(),
            pathways=pathways,
        )

    return build  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# S5 — Provenance required by type
# ---------------------------------------------------------------------------


def test_s5_claim_requires_provenance() -> None:
    """assert_claim_requires_provenance() proves the TYPE enforces S5."""
    assert_claim_requires_provenance()  # raises InvariantViolation on failure


def test_s5_evidence_event_rejects_missing_source_id() -> None:
    """EvidenceEvent rejects construction when source_id is missing (pydantic ValidationError)."""
    with pytest.raises(pydantic.ValidationError):
        EvidenceEvent.model_validate(
            {
                # "source_id" deliberately omitted
                "id": "ev-1",
                "type": "corroboration",
                "polarity": "+",
                "source_authority": 1.0,
                "base_weight": 1.0,
                "recorded_at": _NOW,
            }
        )


def test_s5_evidence_event_rejects_missing_recorded_at() -> None:
    """EvidenceEvent rejects construction when recorded_at is missing (pydantic ValidationError)."""
    with pytest.raises(pydantic.ValidationError):
        EvidenceEvent.model_validate(
            {
                "id": "ev-1",
                "type": "corroboration",
                "polarity": "+",
                "source_id": "src-a",
                "source_authority": 1.0,
                "base_weight": 1.0,
                # "recorded_at" deliberately omitted
            }
        )


# ---------------------------------------------------------------------------
# S1 — knowledge layer and entity-KG adapter import without pulling in cogworx.model
# ---------------------------------------------------------------------------


def test_s1_knowledge_module_has_no_model_import() -> None:
    """src/cogworx/knowledge/ files must not textually import cogworx.model."""
    import pathlib

    knowledge_root = pathlib.Path(__file__).parent.parent.parent / "src" / "cogworx" / "knowledge"
    adapter_file = (
        pathlib.Path(__file__).parent.parent.parent
        / "src"
        / "cogworx"
        / "adapters"
        / "neo4j_entity_kg.py"
    )

    files_to_check = [*list(knowledge_root.glob("*.py")), adapter_file]
    for path in files_to_check:
        source = path.read_text(encoding="utf-8")
        assert "cogworx.model" not in source, (
            f"S1 violation: {path.name} contains 'cogworx.model' import. "
            "Knowledge layer and entity-KG adapter must not depend on the model layer."
        )


def test_s1_no_model_in_fresh_interpreter_after_knowledge_import() -> None:
    """Importing cogworx.knowledge modules alone must not load cogworx.model (S1).

    The knowledge layer (identity, confidence, evidence) is a pure computation module — it
    carries no model dependency. The testing doubles import fake_model (which depends on
    cogworx.model.base by design) so they are NOT included in this check; only the
    knowledge layer itself is tested here.

    Uses a fresh subprocess (same pattern as test_import_order.py) so earlier test imports
    in the same process cannot mask the check.
    """
    script = (
        "import sys; "
        "import cogworx.knowledge.identity; "
        "import cogworx.knowledge.confidence; "
        "import cogworx.knowledge.evidence; "
        "assert 'cogworx.model' not in sys.modules, "
        "f'cogworx.model was loaded: {list(sys.modules.keys())}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        "S1 violation: importing cogworx.knowledge modules pulled in "
        "cogworx.model in a fresh interpreter (knowledge layer must have no model dependency):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )


# ---------------------------------------------------------------------------
# S1 deterministic integration — no model call on the KG write path
# ---------------------------------------------------------------------------


async def test_s1_no_model_on_kg_write_path() -> None:
    """CommitSpyJournal confirms the model is not called during the KG-write stage commit (S1).

    Pathway: model_call → kg_write → Done. The spy watches every commit_step;
    it trips if model.call_count increased during the kg_write commit.
    """
    kg = InMemoryEntityKG()
    pathways = PathwayRegistry()
    pathways.register(_PATHWAY_ID, _build_extraction_pathway(kg))

    model = ReplayModel(
        [ModelResponse(text="pluto has mass 1.3e22 kg", model_id="replay", finish_reason="stop")]
    )

    state = await assert_no_model_on_write_path(
        engine_factory=_engine_factory(pathways),
        inner_journal=InMemoryJournal(),
        model=model,
        pathway_id=_PATHWAY_ID,
        initial=_artifact("start", source="human"),
        run_id="s1-kg-run",
        session_id="s1-kg-sess",
    )
    assert state.status is RunStatus.COMPLETED
    # Exactly one model call total (model_call stage).
    assert model.call_count == 1
    # The claim landed.
    results = await kg.claims_about("pluto")
    assert len(results) == 1


# ---------------------------------------------------------------------------
# S5 deterministic integration — run writes carry provenance
# ---------------------------------------------------------------------------


async def test_s5_kg_write_stage_output_carries_provenance() -> None:
    """Every step committed by the 2-stage pathway carries provenance (S5)."""
    kg = InMemoryEntityKG()
    pathways = PathwayRegistry()
    pathways.register(_PATHWAY_ID, _build_extraction_pathway(kg))

    model = echo_model("pluto has mass")
    _reg = ModelRegistry()
    _reg.register("default", model)
    engine = Engine(
        models=_reg,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
    )
    state = await engine.run(
        run_id="s5-kg-run",
        session_id="s5-kg-sess",
        pathway_id=_PATHWAY_ID,
        initial=_artifact("start", source="human"),
    )
    assert state.steps
    assert_run_writes_carry_provenance(state)

    # The claim also landed.
    results = await kg.claims_about("pluto")
    assert len(results) == 1
    # Evidence event on the claim.
    ev = await kg.evidence_for(results[0].claim.id)
    assert len(ev) == 1


# ---------------------------------------------------------------------------
# S8 — Lesion test: entity-KG write disabled → run still reaches terminal
# ---------------------------------------------------------------------------


async def test_s8_lesion_entity_kg_write_disabled() -> None:
    """S8: disabling the entity_kg_write capability → run completes or degrades, no claim written.

    The kg_write stage catches RegistryError and returns Degraded rather than propagating —
    this is the S8 fallback. The run MUST reach a terminal state; no claim must be written.
    """
    kg = InMemoryEntityKG()
    registry = Registry()

    # Register the KG write capability.
    async def _write_kg(subject: str, predicate: str, payload: str) -> None:
        claim = Claim(
            id=claim_id_for(subject, predicate, payload),
            subject=subject,
            predicate=predicate,
            payload=payload,
            epistemic_type="inference",
            provenance=Provenance(
                source="extraction",
                confidence=0.9,
                recorded_at=_NOW,
            ),
            valid_from=_NOW,
            ingest_time=_NOW,
            created_by="stage:kg_write",
        )
        ev = EvidenceEvent(
            id="ev-lesion-001",
            type="corroboration",
            polarity="+",
            source_id="lesion-src",
            source_authority=0.9,
            base_weight=1.0,
            recorded_at=_NOW,
        )
        await kg.write_claim(claim, evidence=ev)

    registry.register(function_capability(_write_kg, name=_KG_WRITE_CAP, tier="write"))
    # LESION: disable the capability.
    registry.disable(_KG_WRITE_CAP)

    pathways = PathwayRegistry()
    pathways.register(_PATHWAY_ID, _build_extraction_pathway(kg, registry=registry))

    model = echo_model("pluto has mass")
    _reg = ModelRegistry()
    _reg.register("default", model)
    engine = Engine(
        models=_reg,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        registry=registry,
    )
    state = await engine.run(
        run_id="s8-kg-run",
        session_id="s8-kg-sess",
        pathway_id=_PATHWAY_ID,
        initial=_artifact("start", source="human"),
    )

    # Run must reach a terminal status (COMPLETED or DEGRADED).
    assert state.status in (RunStatus.COMPLETED, RunStatus.DEGRADED), (
        f"Expected terminal status, got {state.status!r}"
    )

    # NO claim must have been written (the KG write was lesioned).
    results = await kg.claims_about("pluto")
    assert len(results) == 0, (
        f"S8 lesion test: entity_kg_write was disabled but {len(results)} claim(s) were written"
    )

    # At least one step committed.
    assert len(state.steps) >= 1
