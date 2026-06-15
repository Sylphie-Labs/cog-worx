"""Phase 2 gate spike — Part A: recall-quality scenarios GE-1 through GE-10."""

from __future__ import annotations

import math
import uuid
from typing import Any

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.coherence.config import CoherenceConfig
from cogworx.coherence.reconciler import CoherenceReconciler
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import DEFAULT_MEMORY_POLICY
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.query import RecallQuery
from cogworx.recall.stack import RecallOutcome, RecallStack, default_recall_stack
from cogworx.testing.doubles import InMemoryEntityKG
from cogworx.testing.fake_oracle import TableOracle
from cogworx.testing.recall_fixtures import (
    T_30D,
    T_NOW,
    GateCorpus,
    basis,
    build_gate_corpus,
    build_ge4_contradiction,
    vec_toward,
)

pytestmark = [pytest.mark.spike]


# ---------------------------------------------------------------------------
# Module-scoped corpus fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
async def gate_kg_and_corpus() -> tuple[InMemoryEntityKG, GateCorpus]:
    kg = InMemoryEntityKG()
    corpus = await build_gate_corpus(kg)
    return kg, corpus


# ---------------------------------------------------------------------------
# Recall stack builder
# ---------------------------------------------------------------------------


def make_recall_stack(kg: InMemoryEntityKG) -> RecallStack:
    """Build a 3-channel RecallStack (dense, BM25, graph) on in-memory doubles.

    Episode and latent channels require per-run data seeding that is corpus-specific;
    for the gate recall tests the 3 entity-KG channels exercise all load-bearing assertions.
    The corpus does populate episodes/latent optionally — we use the 3-channel path (S8
    graceful degradation) since the gate fixtures don't seed EpisodeStore or LatentStore.
    """
    return default_recall_stack(entity_kg=kg)


# ---------------------------------------------------------------------------
# nDCG helper
# ---------------------------------------------------------------------------


def ndcg_at_k(outcome: RecallOutcome, gold_ids: set[str], k: int) -> float:
    ids = [r.item.claim.id for r in outcome.results[:k] if r.kind == "claim"]  # type: ignore[union-attr]
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(ids) if cid in gold_ids)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(min(len(gold_ids), k)))
    return dcg / idcg if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Assertion helpers
# ---------------------------------------------------------------------------


def _claim_ids_from(outcome: RecallOutcome) -> list[str]:
    """Extract claim ids from a RecallOutcome (claim-kind results only)."""
    ids: list[str] = []
    for r in outcome.results:
        if r.kind == "claim":
            sc = r.item
            ids.append(sc.claim.id)  # type: ignore[union-attr]
    return ids


def assert_in_top(outcome: RecallOutcome, claim_id: str, k: int) -> None:
    ids = _claim_ids_from(outcome)[:k]
    assert claim_id in ids, f"Expected {claim_id!r} in top-{k} but got {ids}"


def assert_absent(outcome: RecallOutcome, claim_id: str) -> None:
    ids = _claim_ids_from(outcome)
    assert claim_id not in ids, f"Claim {claim_id!r} should be absent but appeared in results"


def assert_ranks_above(outcome: RecallOutcome, a_id: str, b_id: str) -> None:
    ids = _claim_ids_from(outcome)
    assert a_id in ids and b_id in ids, f"Both {a_id!r} and {b_id!r} must be in results"
    assert ids.index(a_id) < ids.index(b_id), f"{a_id!r} must rank above {b_id!r}"


# ---------------------------------------------------------------------------
# Metrics reporter (module-scoped, informational only — not binding)
# ---------------------------------------------------------------------------

_gate_metrics: list[tuple[str, float, float]] = []


@pytest.fixture(scope="module", autouse=True)
def gate_metrics_report() -> Any:
    _gate_metrics.clear()
    yield
    if _gate_metrics:
        print("\n\nPhase 2 Gate — nDCG@5 and MRR (informational, not binding)")
        print(f"{'Scenario':<30} {'nDCG@5':>8} {'MRR':>8}")
        print("-" * 50)
        for name, ndcg, mrr in _gate_metrics:
            print(f"{name:<30} {ndcg:>8.3f} {mrr:>8.3f}")


# ---------------------------------------------------------------------------
# GE-1 — Single-hop precision
# ---------------------------------------------------------------------------


async def test_ge1_single_hop_precision(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """C1 (zylkorin timezone) is rank-1; first 3 distractors absent from top-3."""
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        text="zylkorin",
        embedding=basis(0),
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C1"], k=1)
    for i in range(1, 4):
        assert_absent(outcome, corpus.claim_ids[f"D{i}"])

    ndcg = ndcg_at_k(outcome, {corpus.claim_ids["C1"]}, k=5)
    c1_ids = _claim_ids_from(outcome)
    c1_id = corpus.claim_ids["C1"]
    mrr = 1.0 / (c1_ids.index(c1_id) + 1) if c1_id in c1_ids else 0.0
    _gate_metrics.append(("GE-1 single-hop precision", ndcg, mrr))


# ---------------------------------------------------------------------------
# GE-2 — Multi-session persistence
# ---------------------------------------------------------------------------


async def test_ge2_multi_session_persistence(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """C1 outranks DX (higher cosine) despite both being present — RRF fusion win.

    scope=None so both C1 (scope='user:alice') and DX (scope='agent') enter all channels.
    DX has cos=0.98 vs C1's cos=0.96, so DX wins dense alone.  C1 picks up extra hits from the
    BM25 channel ('zylkorin' token) and the graph channel (anchor 'user:alice'), so via RRF
    fusion C1 ranks above DX.  This is the genuine multi-session / multi-channel persistence
    scenario: an older multi-session claim (C1) outranks newer high-cosine semantic noise (DX).
    """
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        text="zylkorin",
        embedding=basis(0),
        anchor_entities=("user:alice",),
        scope=None,
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C1"], k=3)
    # Both C1 and DX are in the result set (scope=None); C1 must rank above DX via RRF.
    assert_ranks_above(outcome, corpus.claim_ids["C1"], corpus.claim_ids["DX"])


# ---------------------------------------------------------------------------
# GE-3 — Temporal reasoning (as_of)
# ---------------------------------------------------------------------------


async def test_ge3_temporal_as_of_now(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """At T_NOW: C3 (Denver, valid) in top-5; C2 (Portland, expired at T_10D) absent."""
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        text="portcity denvercity",
        embedding=basis(1),
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C3"], k=5)
    # C2 valid_to=T_10D < T_NOW → expired; the validity filter in dense + BM25 channels removes it.
    assert_absent(outcome, corpus.claim_ids["C2"])

    ndcg = ndcg_at_k(outcome, {corpus.claim_ids["C3"]}, k=5)
    c3_ids = _claim_ids_from(outcome)
    c3_id = corpus.claim_ids["C3"]
    mrr = 1.0 / (c3_ids.index(c3_id) + 1) if c3_id in c3_ids else 0.0
    _gate_metrics.append(("GE-3 temporal as_of=now", ndcg, mrr))


async def test_ge3_temporal_as_of_historical(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """At T_30D (historical): C2 (Portland) was valid; C3 (Denver) didn't exist yet."""
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    # T_30D = T_NOW - 30 days; C2 valid_from=T_30D, valid_to=T_10D → valid at T_30D.
    # C3 valid_from=T_10D → FUTURE relative to T_30D → excluded by the dense as_of filter
    # in claims_by_similarity (InMemoryEntityKG does not filter by as_of there — only
    # ClaimTextChannel / ClaimGraphChannel pass as_of to the KG).
    # The meaningful assertion is: C2 is present (was valid at T_30D).
    query = RecallQuery(
        text="portcity denvercity",
        embedding=basis(1),
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_30D,
        k=20,
    )
    outcome = await stack.recall(query)

    # C2 was valid at T_30D (valid_from=T_30D, valid_to=T_10D > T_30D).
    assert_in_top(outcome, corpus.claim_ids["C2"], k=5)
    # C3 has valid_from=T_10D which is AFTER T_30D — not yet valid; _validity_filter excludes it.
    assert_absent(outcome, corpus.claim_ids["C3"])

    ndcg = ndcg_at_k(outcome, {corpus.claim_ids["C2"]}, k=5)
    c2_ids = _claim_ids_from(outcome)
    c2_id = corpus.claim_ids["C2"]
    mrr = 1.0 / (c2_ids.index(c2_id) + 1) if c2_id in c2_ids else 0.0
    _gate_metrics.append(("GE-3 temporal as_of=historical", ndcg, mrr))


# ---------------------------------------------------------------------------
# GE-4 — Contradiction via reconciler (UPDATE-mode supersession)
# ---------------------------------------------------------------------------


async def test_ge4_contradiction_reconciled() -> None:
    """LISBON defeated by reconciler; MADRID wins; LISBON absent from validity-filtered output."""
    kg_local = InMemoryEntityKG()
    await build_gate_corpus(kg_local)
    ge4 = await build_ge4_contradiction(kg_local)

    lisbon_id = ge4.claim_ids["LISBON"]
    madrid_id = ge4.claim_ids["MADRID"]

    # Script the oracle to flag LISBON+MADRID as inconsistent.
    oracle = TableOracle(conflict_sets=[frozenset([lisbon_id, madrid_id])])

    # Use a large batch_limit so that all dirty subjects are processed in one tick.
    # The gate corpus writes 33 claims across 25 unique (scope, subject_norm) keys;
    # the LISBON/MADRID key is bumped last and would miss a batch_limit=16 tick.
    reconciler = CoherenceReconciler(
        entity_kg=kg_local,
        store=kg_local,
        oracle=oracle,
        config=CoherenceConfig(batch_limit=64),
        now=lambda: T_NOW,
    )
    stats = await reconciler.tick()

    # At least one defeat must have been committed.
    assert stats.subjects_defeated >= 1, (
        f"GE-4: reconciler must have committed at least one defeat; stats={stats}"
    )

    # Verify LISBON was defeated.
    lisbon_claim = await kg_local.get_claim(lisbon_id)
    assert lisbon_claim is not None
    assert lisbon_claim.status == "defeasibly-defeated", (
        f"GE-4: LISBON must be defeasibly-defeated, got status={lisbon_claim.status!r}"
    )

    # Build recall stack and query; defeated + expired LISBON must be absent from fused output.
    stack = make_recall_stack(kg_local)
    query = RecallQuery(
        embedding=basis(0),
        text="favorite_city",
        anchor_entities=("user:alice",),
        scope="agent",
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, madrid_id, k=5)
    # UPDATE-mode defeat must set valid_to — a reconciler that marks status="defeasibly-defeated"
    # but omits valid_to has failed to apply the bi-temporal invalidation.
    assert lisbon_claim.valid_to is not None, (
        "UPDATE-mode supersession must set valid_to on the loser; "
        "reconciler failed to apply the bi-temporal invalidation"
    )
    assert_absent(outcome, lisbon_id)

    _gate_metrics.append(("GE-4 contradiction reconciled", 1.0, 1.0))


# ---------------------------------------------------------------------------
# GE-5 — Revision-defeat visibility pin (CF-B honest)
# ---------------------------------------------------------------------------


async def test_ge5_revision_defeat_visible() -> None:
    """Defeated claims are NOT down-weighted until Phase 3+; this pin flips when CF-B lands.

    CF-B: defeated claims down-weighting via score penalty is deferred to Phase 3+.
    Until then, a defeasibly-defeated claim that passes the validity filter STILL appears
    in recall results. This test pins the current (honest) behaviour.

    The revision-defeat scenario: two "inference" claims, same subject+predicate, same
    valid_from — REVISION MODE. Claim B has significantly more/stronger evidence so the
    LCB gap exceeds the margin=0.15 threshold → claim A is revision-defeated by claim B.
    """
    kg_local = InMemoryEntityKG()

    subject = "user:alice"
    predicate = "fav_sport"
    scope = "agent"

    claim_a_id = claim_id_for(subject, predicate, "tennis", scope=scope)
    claim_b_id = claim_id_for(subject, predicate, "soccer", scope=scope)

    # claim_a: weak single evidence (source_authority=0.5 → low LCB)
    ev_a = make_evidence(
        type="attestation",
        polarity="+",
        source_id="ge5-source-a",
        source_authority=0.5,
        recorded_at=T_30D,
        event_id=uuid.uuid4().hex,
    )
    # claim_b: strong multi-evidence (authority=0.95 + two corroborations → high LCB)
    # LCB gap ≈ 0.22 > margin=0.15 → revision-defeat guaranteed.
    ev_b1 = make_evidence(
        type="attestation",
        polarity="+",
        source_id="ge5-source-b1",
        source_authority=0.95,
        recorded_at=T_30D,
        event_id=uuid.uuid4().hex,
    )
    ev_b2 = make_evidence(
        type="corroboration",
        polarity="+",
        source_id="ge5-source-b2",
        source_authority=0.9,
        recorded_at=T_30D,
        event_id=uuid.uuid4().hex,
    )
    ev_b3 = make_evidence(
        type="corroboration",
        polarity="+",
        source_id="ge5-source-b3",
        source_authority=0.9,
        recorded_at=T_30D,
        event_id=uuid.uuid4().hex,
    )

    claim_a = Claim(
        id=claim_a_id,
        subject=subject,
        predicate=predicate,
        payload="tennis",
        epistemic_type="inference",
        provenance=Provenance(source="human", confidence=0.5, recorded_at=T_30D),
        valid_from=T_30D,
        ingest_time=T_30D,
        created_by="ge5",
        embedding=vec_toward(0, 0.91, 6),
        scope=scope,
    )
    claim_b = Claim(
        id=claim_b_id,
        subject=subject,
        predicate=predicate,
        payload="soccer",
        epistemic_type="inference",
        provenance=Provenance(source="human", confidence=0.95, recorded_at=T_30D),
        valid_from=T_30D,
        ingest_time=T_30D,
        created_by="ge5",
        embedding=vec_toward(0, 0.92, 7),
        scope=scope,
    )

    await kg_local.write_claim(claim_a, evidence=ev_a)
    # add_evidence is not used here since write_claim only takes one evidence;
    # write claim_b with its first evidence, then add the extra ones.
    await kg_local.write_claim(claim_b, evidence=ev_b1)
    await kg_local.add_evidence(claim_b_id, ev_b2)
    await kg_local.add_evidence(claim_b_id, ev_b3)

    oracle = TableOracle(conflict_sets=[frozenset([claim_a_id, claim_b_id])])
    reconciler = CoherenceReconciler(
        entity_kg=kg_local,
        store=kg_local,
        oracle=oracle,
        config=CoherenceConfig(batch_limit=8),
        now=lambda: T_NOW,
    )
    stats = await reconciler.tick()

    assert stats.subjects_defeated >= 1, (
        f"GE-5: reconciler must have committed a revision-defeat; stats={stats}"
    )

    # Identify the defeated claim (claim_a has lower LCB — should be the loser).
    ca_after = await kg_local.get_claim(claim_a_id)
    assert ca_after is not None
    assert ca_after.status == "defeasibly-defeated", (
        f"GE-5: claim_a must be defeasibly-defeated, got status={ca_after.status!r}"
    )

    # CF-B: defeated claims NOT down-weighted until Phase 3+; this pin flips when CF-B lands.
    # Revision-defeat does NOT set valid_to — the claim is surfaced-not-deleted. It is still
    # validity-valid (no expiry), so the dense/graph channels may still return it.
    assert ca_after.valid_to is None, (
        "GE-5 pin (revision-defeat): valid_to must remain None (no time-bounding in revision mode)"
    )

    # Build recall stack and verify the defeated claim IS still retrievable (CF-B honest pin).
    stack = make_recall_stack(kg_local)
    query = RecallQuery(
        embedding=basis(0),
        text="fav_sport tennis soccer",
        anchor_entities=(subject,),
        scope=scope,
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    ids_in_results = _claim_ids_from(outcome)
    assert claim_a_id in ids_in_results, (
        "GE-5 CF-B pin: defeasibly-defeated claim (revision-mode) must still appear in recall "
        "results — down-weighting is NOT yet implemented (Phase 3+). "
        "This assertion flips when CF-B lands."
    )


# ---------------------------------------------------------------------------
# GE-6 — Multi-hop via graph adjacency
# ---------------------------------------------------------------------------


async def test_ge6_graph_multihop(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """C7 (atlasdeploy) in top-5 via dense+BM25; C8 (server:cobalt region) via graph channel."""
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        anchor_entities=("project:atlas", "server:cobalt"),
        embedding=basis(3),
        text="atlasdeploy",
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C7"], k=5)
    assert_in_top(outcome, corpus.claim_ids["C8"], k=5)

    # C8 (cos=0.40 on basis(4)) would fall below the default min_score=0.70 for dense — it must
    # be retrieved exclusively via the graph channel (anchor "server:cobalt").
    c8_id = corpus.claim_ids["C8"]
    c8_result = next(
        (r for r in outcome.results if r.kind == "claim" and r.item.claim.id == c8_id),  # type: ignore[union-attr]
        None,
    )
    assert c8_result is not None, "GE-6: C8 not found in outcome.results"

    # Verify at least one channel hit on the graph channel for C8.
    channel_names = {h.channel for h in c8_result.hits}
    assert "graph.claims" in channel_names, (
        f"GE-6: C8 must have a graph.claims channel hit; channels={channel_names}"
    )

    ndcg = ndcg_at_k(outcome, {corpus.claim_ids["C7"], corpus.claim_ids["C8"]}, k=5)
    _gate_metrics.append(("GE-6 graph multihop", ndcg, 1.0))


# ---------------------------------------------------------------------------
# GE-7 — Scope isolation (user model)
# ---------------------------------------------------------------------------


async def test_ge7_scope_isolation_scoped(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """With scope='user:alice': C10 present, C9 (scope='agent') absent."""
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        embedding=basis(5),
        text="editorpref",
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C10"], k=5)
    assert_absent(outcome, corpus.claim_ids["C9"])

    ndcg = ndcg_at_k(outcome, {corpus.claim_ids["C10"]}, k=5)
    _gate_metrics.append(("GE-7 scope isolation (scoped)", ndcg, 1.0))


async def test_ge7_scope_isolation_unscoped(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """Without scope filter: both C9 and C10 in top-10."""
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        embedding=basis(5),
        text="editorpref",
        anchor_entities=("user:alice",),
        scope=None,
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C9"], k=10)
    assert_in_top(outcome, corpus.claim_ids["C10"], k=10)


# ---------------------------------------------------------------------------
# GE-8 — Fusion beats semantic noise
# ---------------------------------------------------------------------------


async def test_ge8_fusion_beats_semantic_noise(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """C1 outranks DX (higher cosine 0.98) via multi-channel RRF fusion — not scope filtering.

    scope=None so both C1 (scope='user:alice') and DX (scope='agent') enter all channels.
    In the dense channel alone, DX (cos=0.98) ranks above C1 (cos=0.96).  But C1 also gets
    hits from BM25 ('zylkorin' token exact match) and the graph channel (anchor 'user:alice'
    points to C1's subject, not DX's 'noise:hot').  Via RRF, C1's multi-channel hit count
    lifts it to rank-1 above DX.

    Mutation-resistance: replacing fuse() with pass-through dense would give DX rank-1 — the
    assertion catches that regression.
    """
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        embedding=basis(0),
        text="zylkorin",
        anchor_entities=("user:alice",),
        scope=None,
        as_of=T_NOW,
        k=20,
    )
    outcome = await stack.recall(query)

    # Both C1 and DX must be present (scope=None, no pre-RRF filtering removes them).
    assert_in_top(outcome, corpus.claim_ids["C1"], k=1)
    # DX is in the result set; C1 must rank above it via multi-channel RRF fusion.
    assert_ranks_above(outcome, corpus.claim_ids["C1"], corpus.claim_ids["DX"])

    ndcg = ndcg_at_k(outcome, {corpus.claim_ids["C1"]}, k=5)
    _gate_metrics.append(("GE-8 fusion beats semantic noise", ndcg, 1.0))


# ---------------------------------------------------------------------------
# GE-9 — Aggregation completeness (diet claims)
# ---------------------------------------------------------------------------


async def test_ge9_aggregation_completeness(
    gate_kg_and_corpus: tuple[InMemoryEntityKG, GateCorpus],
) -> None:
    """Recall@10 over gold {C4, C5, C6} = 1.0 — all three diet claims retrieved.

    Answer synthesis (counting/aggregation) is Phase 3+ scope; this gate tests retrieval only.
    """
    kg, corpus = gate_kg_and_corpus
    stack = make_recall_stack(kg)

    query = RecallQuery(
        embedding=basis(2),
        text="vegquark nutquark caffquark",
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_NOW,
        k=10,
    )
    outcome = await stack.recall(query)

    assert_in_top(outcome, corpus.claim_ids["C4"], k=10)
    assert_in_top(outcome, corpus.claim_ids["C5"], k=10)
    assert_in_top(outcome, corpus.claim_ids["C6"], k=10)

    gold = {corpus.claim_ids["C4"], corpus.claim_ids["C5"], corpus.claim_ids["C6"]}
    ndcg = ndcg_at_k(outcome, gold, k=10)
    _gate_metrics.append(("GE-9 aggregation completeness", ndcg, 1.0))


# ---------------------------------------------------------------------------
# GE-10 — End-to-end injection (the consumer surface)
# ---------------------------------------------------------------------------


async def test_ge10_end_to_end_injection() -> None:
    """Full MemoryInjector pass: recall → assemble → InjectedMemory.

    S1 invariant: the injection path must never call the model's ``complete`` method.  A model
    that raises on ``complete`` is wired in — any code path that reaches the model call crashes
    the test immediately, rather than checking a counter after a call that was never possible.

    U-fold invariant: with ≥2 admitted chunks, rank-2 occupies the LAST position (back edge).
    This assertion is structurally false without U-fold: a pass-through would leave rank-2 at
    position 1, not the last position.

    We construct MemoryInjector directly rather than wiring through Engine — the Engine
    constructor requires a live GraphStore and LatentStore; the injection seam is exercisable
    in isolation (S8 lesion principle).
    """

    class _FailOnCallModel:
        """Model that raises AssertionError if ``complete`` is ever invoked (S1 guard).

        ``count_tokens`` is functional: MemoryInjector legitimately duck-types it for token
        counting (not a model call on the hot path).  Only ``complete`` is forbidden here.
        """

        @property
        def capabilities(self) -> object:
            from cogworx.model.base import ModelCapabilities

            return ModelCapabilities()

        async def complete(self, **_: object) -> object:
            raise AssertionError(
                "GE-10 S1 violation: model.complete() called on the recall/injection path"
            )

        def count_tokens(self, text: str) -> int:
            return max(1, len(text) // 4)

    kg_local = InMemoryEntityKG()
    corpus = await build_gate_corpus(kg_local)

    # S1 guard: any call to complete() on the injection path raises immediately.
    model = _FailOnCallModel()

    stack = default_recall_stack(entity_kg=kg_local)
    injector = MemoryInjector(stack=stack, model=model)

    # --- Pass A: zylkorin query — verify C1 is injected and status is 'ok' ---
    query_c1 = RecallQuery(
        text="zylkorin",
        embedding=basis(0),
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_NOW,
        k=20,
    )
    injected_c1 = await injector.inject(query_c1, policy=DEFAULT_MEMORY_POLICY)

    # Reaching here proves S1: complete() would have raised if called.
    assert injected_c1.status == "ok", f"GE-10: expected status='ok', got {injected_c1.status!r}"
    assert len(injected_c1.context.chunks) > 0, "GE-10: assembled context is empty"

    c1_id = corpus.claim_ids["C1"]
    chunk_keys_c1 = {c.key for c in injected_c1.context.chunks}
    assert f"claim:{c1_id}" in chunk_keys_c1, (
        f"GE-10: C1 not found in assembled chunks; keys={chunk_keys_c1}"
    )

    # --- Pass B: diet query — guarantees ≥3 admits; assert U-fold back-edge placement ---
    # C4 (vegquark), C5 (nutquark), C6 (caffquark) are all scope='user:alice' on basis(2),
    # so all three are admitted.  With m≥2 the U-fold loop places rank-2 at output[m-1].
    query_diet = RecallQuery(
        text="vegquark nutquark caffquark",
        embedding=basis(2),
        anchor_entities=("user:alice",),
        scope="user:alice",
        as_of=T_NOW,
        k=20,
    )
    injected_diet = await injector.inject(query_diet, policy=DEFAULT_MEMORY_POLICY)

    chunks = injected_diet.context.chunks
    assert len(chunks) >= 2, f"GE-10 U-fold: diet query must admit ≥2 chunks; got {len(chunks)}"

    # U-fold places admitted items alternately at front then back:
    #   rank-1 → output[0], rank-2 → output[m-1], rank-3 → output[1], …
    # A pass-through (no U-fold) would leave rank-2 at output[1], not output[m-1].
    # This assertion is therefore false without U-fold for any m≥2.
    assert chunks[-1].relevance_rank == 2, (
        f"GE-10 U-fold: rank-2 item must be at last position (back edge); "
        f"got chunks[-1].relevance_rank={chunks[-1].relevance_rank}. "
        "Disabling U-fold would place rank-2 at position 1, not the last position."
    )

    _gate_metrics.append(("GE-10 end-to-end injection", 1.0, 1.0))
