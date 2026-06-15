"""Phase 2 gate spike — Part B: S5 invariant storm + S8 lesion scenarios.

Part B covers:
  - S5 write-surface storm: all five Phase 2 write surfaces driven through RecordingEntityKG
    and audited by assert_s5_substrate_invariants.
  - S8 lesion scenarios: memory-fully-lesioned (L-1), partial-substrate (L-2a), and
    dead-channel isolation (L-2b).

Pure Python — no Neo4j, no Postgres, no live model calls.
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.coherence.promotion import PromotionRule, ScopePromoter
from cogworx.coherence.upgrade import epistemic_upgrade
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.scoped_kg import world_model
from cogworx.knowledge.scopes import ScopeRegistry
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import Done, StageResult
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.registry import ModelRegistry
from cogworx.recall.query import RecallQuery
from cogworx.recall.stack import default_recall_stack
from cogworx.runtime.engine import Engine
from cogworx.substrate.entity_kg import ClaimProjection
from cogworx.substrate.journal import ProjectionCursor
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.invariants import (
    InvariantViolation,
    RecordingEntityKG,
    assert_s5_substrate_invariants,
)
from cogworx.testing.recall_fixtures import (
    basis,
    build_gate_corpus,
    vec_toward,
)

pytestmark = [pytest.mark.spike]

# ---------------------------------------------------------------------------
# Fixed timestamps — no datetime.now() in test logic
# ---------------------------------------------------------------------------

_T0: datetime = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_SOURCE_ID: str = "source:system:gate-s5"


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _ev(
    *,
    ev_type: str = "attestation",
    source_id: str = _SOURCE_ID,
    recorded_at: datetime = _T0,
    event_id: str | None = None,
) -> EvidenceEvent:
    return make_evidence(
        type=ev_type,  # type: ignore[arg-type]
        polarity="+",
        source_id=source_id,
        source_authority=0.9,
        recorded_at=recorded_at,
        event_id=event_id if event_id is not None else uuid.uuid4().hex,
    )


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    epistemic_type: str = "observation",
    scope: str = "agent",
    valid_from: datetime = _T0,
    valid_until: datetime | None = None,
    embedding: tuple[float, ...] | None = None,
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type=epistemic_type,
        provenance=Provenance(
            source="human",
            confidence=0.9,
            recorded_at=valid_from,
        ),
        valid_from=valid_from,
        valid_to=valid_until,
        ingest_time=valid_from,
        created_by="gate-s5",
        embedding=embedding,
        scope=scope,
    )


# ---------------------------------------------------------------------------
# PART B — S5 Write-Surface Storm
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Surface 1: write_claim parametrized matrix
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "epistemic_type,scope",
    [
        ("observation", "agent"),
        ("inference", "agent"),
        ("confirmed", "agent"),
        ("observation", "user:alice"),
        ("inference", "world"),
    ],
)
async def test_s5_write_claim_surface(epistemic_type: str, scope: str) -> None:
    """Each (epistemic_type, scope) combination survives the S5 substrate audit."""
    kg = RecordingEntityKG(InMemoryEntityKG())
    claim = _make_claim(
        "subject:test",
        "predicate:test",
        f"payload-{epistemic_type}-{scope}",
        epistemic_type=epistemic_type,
        scope=scope,
    )
    ev = _ev(source_id=_SOURCE_ID)
    await kg.write_claim(claim, evidence=ev)
    # No exception means S5 audit passed.
    await assert_s5_substrate_invariants(kg, kg.recorded_claim_ids)


@pytest.mark.asyncio
async def test_s5_write_claim_wrong_id_raises() -> None:
    """A claim whose id does not match its content-hash is rejected by write_claim.

    InMemoryEntityKG enforces identity discipline via _assert_identity_mem before any I/O.
    A claim constructed with a mismatched id raises ValueError before the claim is stored.
    """
    inner = InMemoryEntityKG()
    # Build a claim with a deliberately wrong id (not the content-hash of its fields).
    wrong_id = "not-a-real-content-hash-000000000000"
    claim_bad = Claim(
        id=wrong_id,
        subject="subject:test",
        predicate="predicate:test",
        payload="payload-bad",
        epistemic_type="observation",
        provenance=Provenance(source="human", confidence=0.9, recorded_at=_T0),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="gate-s5",
        scope="agent",
    )
    ev = _ev()
    with pytest.raises(ValueError, match="does not match expected"):
        await inner.write_claim(claim_bad, evidence=ev)


# ---------------------------------------------------------------------------
# Surface 2: project_claims batch
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s5_project_claims_surface() -> None:
    """project_claims with 3 ClaimProjections survives the S5 audit."""
    kg = RecordingEntityKG(InMemoryEntityKG())
    projections = [
        ClaimProjection(
            claim=_make_claim("entity:a", "attr", f"val{i}", embedding=vec_toward(i, 0.9, 7)),
            evidence=_ev(source_id=f"src-{i}"),
        )
        for i in range(3)
    ]
    cursor = ProjectionCursor(commit_ordinal=1, run_id="gate-run", step_index=0)
    await kg.project_claims("test", projections, cursor)
    assert len(kg.recorded_claim_ids) == 3
    await assert_s5_substrate_invariants(kg, kg.recorded_claim_ids)


@pytest.mark.asyncio
async def test_s5_project_claims_idempotent() -> None:
    """Projecting the same 3 claims twice leaves exactly 3 unique ids, each with >=1 evidence."""
    kg = RecordingEntityKG(InMemoryEntityKG())
    projections = [
        ClaimProjection(
            claim=_make_claim("entity:b", "attr", f"val{i}", embedding=vec_toward(i, 0.9, 7)),
            evidence=_ev(source_id=f"src-{i}-first"),
        )
        for i in range(3)
    ]
    cursor = ProjectionCursor(commit_ordinal=1, run_id="gate-run", step_index=0)
    await kg.project_claims("test", projections, cursor)

    # Second projection: same claims, different evidence events (new source ids).
    projections2 = [
        ClaimProjection(
            claim=_make_claim("entity:b", "attr", f"val{i}", embedding=vec_toward(i, 0.9, 7)),
            evidence=_ev(source_id=f"src-{i}-second"),
        )
        for i in range(3)
    ]
    cursor2 = ProjectionCursor(commit_ordinal=2, run_id="gate-run", step_index=1)
    await kg.project_claims("test", projections2, cursor2)

    # Still 3 unique ids (first-write-wins on the claim node).
    assert len(kg.recorded_claim_ids) == 3
    # Each claim must have >=1 evidence event.
    for cid in kg.recorded_claim_ids:
        evidence = await kg.evidence_for(cid)
        assert len(evidence) >= 1, f"claim {cid} has no evidence after idempotent projection"
    await assert_s5_substrate_invariants(kg, kg.recorded_claim_ids)


# ---------------------------------------------------------------------------
# Surface 3: add_evidence
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s5_add_evidence_surface() -> None:
    """Adding a second evidence event to an existing claim leaves >=2 events, all well-formed."""
    kg = RecordingEntityKG(InMemoryEntityKG())
    claim = _make_claim("entity:c", "attr", "val-ev")
    ev1 = _ev(source_id="src-ev-1")
    cid = await kg.write_claim(claim, evidence=ev1)

    ev2 = _ev(source_id="src-ev-2")
    await kg.add_evidence(cid, ev2)

    evidence = await kg.evidence_for(cid)
    assert len(evidence) >= 2, f"expected >=2 evidence events, got {len(evidence)}"
    await assert_s5_substrate_invariants(kg, {cid})


# ---------------------------------------------------------------------------
# Surface 4: first-write-wins pin (S5 documented boundary)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s5_first_write_wins_pin() -> None:
    """Epistemic level is first-write-wins; re-writing at a different level leaves level unchanged
    but accumulates evidence.

    This is the documented S5 boundary in entity_kg.py (write_claim semantics, lines 80-87):
    the stored level never silently changes — the first writer's level sticks until an explicit
    upgrade via CoherenceStore.apply_epistemic_upgrade.
    """
    inner = InMemoryEntityKG()
    kg = RecordingEntityKG(inner)
    # First write at "inference" level.
    claim_inf = _make_claim("entity:d", "attr", "val-pin", epistemic_type="inference")
    ev1 = _ev(source_id="src-first")
    cid = await kg.write_claim(claim_inf, evidence=ev1)

    # Second write of the SAME claim id (same content-hash) at "observation" level.
    # write_claim is first-write-wins on the claim node itself; the new evidence is accumulated.
    claim_obs = _make_claim("entity:d", "attr", "val-pin", epistemic_type="observation")
    assert claim_obs.id == cid, "same triple must produce the same id"
    ev2 = _ev(source_id="src-second")
    cid2 = await kg.write_claim(claim_obs, evidence=ev2)
    assert cid2 == cid

    # Level is still "inference" (first-write-wins).
    stored = await kg.get_claim(cid)
    assert stored is not None
    assert stored.epistemic_type == "inference", (
        f"first-write-wins violated: expected 'inference', got {stored.epistemic_type!r}"
    )

    # Both evidence events accumulated (second write's evidence was added).
    evidence = await kg.evidence_for(cid)
    assert len(evidence) >= 2, (
        f"second write's evidence not accumulated: got {len(evidence)} event(s)"
    )
    await assert_s5_substrate_invariants(kg, {cid})


# ---------------------------------------------------------------------------
# Surface 5: epistemic_upgrade
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s5_epistemic_upgrade_surface() -> None:
    """Upgrading inference → observation with tool_proof evidence applies and passes S5 audit."""
    inner = InMemoryEntityKG()
    kg = RecordingEntityKG(inner)
    claim = _make_claim("entity:e", "attr", "val-upgrade", epistemic_type="inference")
    ev1 = _ev(source_id="src-initial", ev_type="attestation")
    cid = await kg.write_claim(claim, evidence=ev1)

    tool_proof_ev = make_evidence(
        type="tool_proof",
        polarity="+",
        source_id="src-tool-upgrade",
        source_authority=1.0,
        recorded_at=_T0,
    )
    # epistemic_upgrade operates on the CoherenceStore interface; InMemoryEntityKG satisfies it.
    result = await epistemic_upgrade(
        inner,
        cid,
        new_level="observation",
        evidence=tool_proof_ev,
        actor="gate-s5",
    )
    assert result.applied is True, f"upgrade did not apply: {result!r}"

    # Read back via recording wrapper.
    stored = await kg.get_claim(cid)
    assert stored is not None
    assert stored.epistemic_type == "observation", (
        f"expected 'observation' after upgrade, got {stored.epistemic_type!r}"
    )
    evidence = await kg.evidence_for(cid)
    assert len(evidence) >= 1
    await assert_s5_substrate_invariants(kg, {cid})


@pytest.mark.asyncio
async def test_s5_upgrade_ineligible_evidence_raises() -> None:
    """extraction evidence cannot upgrade epistemic level (not in UPGRADE_ELIGIBLE_EVIDENCE)."""
    inner = InMemoryEntityKG()
    claim = _make_claim("entity:f", "attr", "val-ineligible", epistemic_type="inference")
    ev1 = _ev(source_id="src-initial")
    cid = await inner.write_claim(claim, evidence=ev1)

    ineligible_ev = make_evidence(
        type="extraction",
        polarity="+",
        source_id="src-extraction",
        source_authority=0.8,
        recorded_at=_T0,
    )
    with pytest.raises(ValueError, match="not eligible"):
        await epistemic_upgrade(
            inner,
            cid,
            new_level="observation",
            evidence=ineligible_ev,
            actor="gate-s5",
        )


@pytest.mark.asyncio
async def test_s5_upgrade_downgrade_raises() -> None:
    """A downgrade (confirmed → observation) must raise ValueError (monotonic ladder).

    Note: upgrade.py checks evidence eligibility BEFORE the rank guard, so the target level
    must be one that has eligible evidence types ("observation" or "confirmed") to reach the
    rank-direction check. Targeting "inference" instead raises a different error ("no eligible
    evidence types defined") that fires before the monotonic-up guard. We use confirmed →
    observation (both have eligible types; observation has a lower rank than confirmed) to hit
    the downgrade path.
    """
    inner = InMemoryEntityKG()
    claim = _make_claim("entity:g", "attr", "val-downgrade", epistemic_type="confirmed")
    ev1 = _ev(source_id="src-initial", ev_type="tool_proof")
    cid = await inner.write_claim(claim, evidence=ev1)

    downgrade_ev = make_evidence(
        type="tool_proof",
        polarity="+",
        source_id="src-downgrade",
        source_authority=1.0,
        recorded_at=_T0,
    )
    with pytest.raises(ValueError, match="downgrade"):
        await epistemic_upgrade(
            inner,
            cid,
            new_level="observation",
            evidence=downgrade_ev,
            actor="gate-s5",
        )


# ---------------------------------------------------------------------------
# Surface 6: ScopePromoter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s5_scope_promoter_surface() -> None:
    """Promoting a user:alice claim to world scope mints a new world claim with >=1 evidence."""
    inner = InMemoryEntityKG()
    kg = RecordingEntityKG(inner)

    # Write a user:alice scope claim with a human source (require_source_kind="human" is default).
    claim = _make_claim(
        "user:alice",
        "preferred_language",
        "python",
        epistemic_type="observation",
        scope="user:alice",
    )
    human_ev = make_evidence(
        type="attestation",
        polarity="+",
        source_id="source:human:alice-session-1",  # starts with "source:human:" prefix
        source_authority=0.9,
        recorded_at=_T0,
    )
    _cid = await kg.write_claim(claim, evidence=human_ev)

    # Build a ScopeRegistry and world ScopedKG as the promotion sink.
    scope_registry = ScopeRegistry()
    world_kg = world_model(inner, scope_registry, owner="gate-s5-promoter")

    # PromotionRule: route user:alice "preferred_language" claims to world scope.
    rule = PromotionRule(
        target="world",
        predicate_norm="preferred_language",
        min_distinct_sources=1,
        require_source_kind="human",  # gate on human-attested claims
    )
    sinks: dict[str, Any] = {"world": world_kg}
    promoter = ScopePromoter(rules=[rule], sinks=sinks, store=inner, source_kg=inner)

    # Fetch the scored claim for the promoter.
    scored_claims = list(await kg.claims_about("user:alice", scope="user:alice"))
    assert len(scored_claims) >= 1

    count = await promoter.promote_for_subject(scored_claims)
    assert count == 1, f"expected 1 promotion, got {count}"

    # The promoted claim should exist in world scope.
    world_claims = list(await world_kg.claims_about("user:alice"))
    assert len(world_claims) >= 1, "promoted claim not found in world scope"

    # The promoted claim id must have >=1 evidence.
    promoted_cid = world_claims[0].claim.id
    promoted_evidence = await inner.evidence_for(promoted_cid)
    assert len(promoted_evidence) >= 1, "promoted claim has no evidence"

    # S5 audit on the promoted claim.
    await assert_s5_substrate_invariants(inner, {promoted_cid})


# ---------------------------------------------------------------------------
# Broken-double negative control (mutation resistance)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_s5_auditor_detects_evidence_gap() -> None:
    """The S5 auditor must actually fail when evidence is missing (not vacuous).

    This mutation-resistance test proves the audit assertion has teeth: an EvidenceDropper
    double that returns [] from evidence_for must cause InvariantViolation on the audit sweep.
    """

    class EvidenceDropper(InMemoryEntityKG):
        async def evidence_for(self, claim_id: str) -> Sequence[EvidenceEvent]:
            return []

    dropper = EvidenceDropper()
    recording_kg = RecordingEntityKG(dropper)
    # build_gate_corpus writes into the recording_kg via write_claim.
    await build_gate_corpus(recording_kg)
    # The recorded_claim_ids are non-empty.
    assert len(recording_kg.recorded_claim_ids) > 0

    with pytest.raises(InvariantViolation):
        await assert_s5_substrate_invariants(recording_kg, recording_kg.recorded_claim_ids)


# ---------------------------------------------------------------------------
# PART C — S8 Lesion Scenarios
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# L-1 — Memory fully lesioned (no injector wired)
# ---------------------------------------------------------------------------


class _MemoryCheckStage:
    """A minimal stage that calls ctx.recall() and stores the result for assertion."""

    name: str = "check_memory"
    transitions: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.last_memory: Any = None

    async def run(self, ctx: StageContext) -> StageResult:
        from cogworx.claims.provenance import Artifact, Provenance

        query = RecallQuery(text="test query")
        memory = await ctx.recall(query)
        self.last_memory = memory
        artifact = Artifact(
            kind="done",
            produced_by="check_memory",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_T0),
        )
        return Done(output=artifact)


@pytest.mark.asyncio
async def test_l1_memory_lesion_no_injector() -> None:
    """Engine run with no recall_stack wired: ctx.recall() returns status='unwired', not a crash.

    S8 degradation must be visible (status='unwired') rather than silent (empty 'ok' result).
    The run still reaches COMPLETED — the lesion is graceful, not fatal.
    """
    stage = _MemoryCheckStage()
    graph = StageGraph([stage], entry="check_memory")
    pathways = PathwayRegistry()
    pathways.register("lesion-test", graph, version=1)

    model = ReplayModel([])  # no model calls expected
    journal = InMemoryJournal()
    graph_store = InMemoryGraphStore()
    latent_store = InMemoryLatentStore()

    _reg = ModelRegistry()
    _reg.register("default", model)
    engine = Engine(
        models=_reg,
        journal=journal,
        graph_store=graph_store,
        latent=latent_store,
        pathways=pathways,
        # recall_stack=None  <-- intentionally omitted (the lesion)
    )

    from cogworx.claims.provenance import Artifact, Provenance

    initial = Artifact(
        kind="initial",
        produced_by="test",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_T0),
    )

    run_id = f"l1-run-{uuid.uuid4().hex[:8]}"
    final = await engine.run(
        run_id=run_id,
        session_id="l1-sess",
        pathway_id="lesion-test",
        initial=initial,
    )

    assert final.status is RunStatus.COMPLETED, f"L-1: expected COMPLETED, got {final.status!r}"

    # The stage recorded the InjectedMemory; status must be 'unwired'.
    memory = stage.last_memory
    assert memory is not None, "stage never ran or did not record memory"
    assert memory.status == "unwired", (
        f"L-1 S8 gap: expected status='unwired' but got {memory.status!r}. "
        "Degradation must be visible, not silent."
    )
    # context.chunks must be empty when unwired.
    assert len(memory.context.chunks) == 0, (
        f"L-1: expected 0 context chunks when unwired, got {len(memory.context.chunks)}"
    )


# ---------------------------------------------------------------------------
# L-2a — Partial substrate lesion (missing stores)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2a_partial_substrate_3_channels() -> None:
    """default_recall_stack with only entity_kg (episode_store=None, latent_store=None) has 3
    channels and can still surface C1 (timezone/"zylkorin") from the gate corpus.

    S8 degradation contract: the entity-KG channels (dense, BM25, graph) are always present.
    The stack degrades to 3 channels when optional stores are omitted — never crashes.
    """
    inner = InMemoryEntityKG()
    recording_kg = RecordingEntityKG(inner)
    corpus = await build_gate_corpus(recording_kg)
    c1_id = corpus.claim_ids["C1"]

    stack = default_recall_stack(
        entity_kg=inner,
        episode_store=None,  # lesioned
        latent_store=None,  # lesioned
    )
    # Exactly 3 channels should be wired.
    assert len(stack._channels) == 3, (
        f"L-2a: expected 3 channels with episode/latent absent, got {len(stack._channels)}"
    )

    # Query axis 0 with the distinctive C1 text ("zylkorin") via BM25.
    query = RecallQuery(
        text="zylkorin",
        embedding=basis(0),
        anchor_entities=("user:alice",),
        k=10,
    )
    outcome = await stack.recall(query)

    # All 3 channels must be ok or skipped (none failed).
    for status in outcome.channel_status:
        assert status.state in ("ok", "skipped"), (
            f"L-2a: channel {status.channel!r} failed with error: {status.error!r}"
        )

    # C1 must appear in the top-3 results (gold surfaces from partial stack).
    top3_ids = {
        r.item.claim.id  # type: ignore[union-attr]
        for r in outcome.results[:3]
        if r.kind == "claim"
    }
    assert c1_id in top3_ids, (
        f"L-2a: C1 ({c1_id!r}) not in top-3 results from 3-channel stack. "
        f"Top-3 keys: {[r.key for r in outcome.results[:3]]!r}"
    )


# ---------------------------------------------------------------------------
# L-2b — Channel error isolation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_l2b_dead_channel_no_empty_answer() -> None:
    """A channel that raises RuntimeError degrades to 'failed' status; the other channels
    still contribute and the gold claim still surfaces.

    S8 contract: a channel failure is recorded as failed in channel_status, but never crashes
    the RecallStack. The channel_status field on RecallOutcome carries failure observability.
    """

    class _FailingEntityKG(InMemoryEntityKG):
        """EntityKG where claims_by_similarity raises (dense channel dead)."""

        async def claims_by_similarity(
            self,
            embedding: Sequence[float],
            *,
            k: int = 10,
            min_score: float = 0.70,
            scope: str | None = None,
        ) -> Any:
            raise RuntimeError("Simulated dense-channel failure for L-2b lesion test")

    inner = _FailingEntityKG()
    recording_kg = RecordingEntityKG(inner)
    corpus = await build_gate_corpus(recording_kg)
    c1_id = corpus.claim_ids["C1"]

    # Build stack: 3 entity-KG channels; dense will fail, BM25 and graph still contribute.
    stack = default_recall_stack(entity_kg=inner, episode_store=None, latent_store=None)

    query = RecallQuery(
        text="zylkorin",
        embedding=basis(0),
        anchor_entities=("user:alice",),
        k=10,
    )
    outcome = await stack.recall(query)

    # The dense channel must be recorded as failed.
    channel_names_to_status = {s.channel: s.state for s in outcome.channel_status}
    assert channel_names_to_status.get("dense.claims") == "failed", (
        f"L-2b: expected dense.claims to be 'failed', got: {channel_names_to_status!r}"
    )

    # At least one other channel must be ok.
    ok_channels = [s.channel for s in outcome.channel_status if s.state == "ok" and s.count > 0]
    assert len(ok_channels) >= 1, (
        f"L-2b: no other channels contributed results. channel_status: {outcome.channel_status!r}"
    )

    # Gold claim (C1: timezone/zylkorin) must still surface from BM25 or graph channel.
    result_ids = {
        r.item.claim.id  # type: ignore[union-attr]
        for r in outcome.results
        if r.kind == "claim"
    }
    assert c1_id in result_ids, (
        f"L-2b: C1 ({c1_id!r}) did not surface despite dead dense channel. "
        f"Result keys: {[r.key for r in outcome.results]!r}"
    )

    # CF: channel failure observability is provided via channel_status field on RecallOutcome.
    # The state='failed' and error=repr(exception) fields are present and populated.
    dense_status = next((s for s in outcome.channel_status if s.channel == "dense.claims"), None)
    assert dense_status is not None
    assert dense_status.error is not None, (
        "L-2b: channel_status.error should carry exception repr for diagnostics"
    )
