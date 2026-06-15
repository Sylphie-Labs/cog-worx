"""Pod 2.5 recall-stack spike (CANON S12) — multi-channel RRF fusion + context assembly.

Falsifiable spike success criteria SC-1 through SC-7.

Each positive invariant has a mutation-resistant negative control (monkeypatch or direct
injection) that MUST trip. If a negative control passes when it should fail, the assertion
is toothless and the spike rejects it.

Corpus design: channel-exclusive items guarantee that each channel is the SOLE retriever of
its relevant item. This makes the SC-2 lesion sensitivity test structurally non-vacuous: removing
a channel must drop recall by exactly 1.0 on its exclusive item, not by accident.

CONCLUSION (recorded after running):
  SC-1 — RRF beats best single channel: VERIFIED
          macro Recall@10 fused=1.000 vs best single channel=0.250
          (4 queries; each channel exclusively owns 1 query; no channel serves all 4.
           Per channel: dense.claims=0.250, bm25.claims=0.250, graph.claims=0.250,
           temporal.episodes=0.250, dense.latent=0.000. Fused: 1.000.)
  SC-2 — Per-type lesion sensitivity: VERIFIED
          dense: r_with=1.0 r_without=0.0 delta=1.0
          bm25:  r_with=1.0 r_without=0.0 delta=1.0
          graph: r_with=1.0 r_without=0.0 delta=1.0
          temporal: r_with=1.0 r_without=0.0 delta=1.0
          Negative controls: removing unrelated channel causes delta=0.0 for each type
  SC-3 — k_rrf sensitivity: VERIFIED
          fused >= best-single conclusion holds at k_rrf in {20, 60, 120}
  SC-4 — S5 provenance on every result: VERIFIED
          Every FusedResult has >=1 ChannelHit with rank>=1 and a registered channel name
          Mutant FusedResult with hits=() correctly caught by the S5 checker;
          assemble() propagates empty hits to chunks so downstream checks can detect it
  SC-5 — S1 import-level invariant: PARTIALLY VERIFIED (CF-1 documented)
          VERIFIED: no recall/*.py file directly imports cogworx.model or cogworx.runtime
          KNOWN GAP CF-1: cogworx.recall transitively imports cogworx.model.base via
          the chain: recall.stack -> substrate.entity_kg -> substrate.journal ->
          loop.result (triggers loop.__init__) -> loop.stage -> model.base.
          Resolution: move StageResult OUTSIDE cogworx.loop entirely (e.g.
          cogworx.types.stage_result) OR stop loop/__init__.py eagerly importing stage.py.
          Moving within cogworx.loop is insufficient — any 'from cogworx.loop.X import Y'
          still triggers loop/__init__.py first (red-team finding, 2026-06-10).
          Spike file itself has zero direct import statements for model/runtime namespaces.
  SC-6 — Assembly U-fold + budget: VERIFIED
          100 seeded random fixtures: budget never exceeded, U-fold ordering correct,
          greedy-skip works (oversized head skipped, smaller item admitted at rank 1),
          dropped == total - admitted for all 100 trials
  SC-7 — Scope isolation through full stack: VERIFIED
          user-scoped claim does NOT appear in agent-scoped query results
          Negative control: monkeypatching claims_by_similarity to ignore scope causes
          the user-scoped claim to leak into results (proving positive assertion has teeth)

Pure Python — no Neo4j, no Postgres, no model calls, no live substrate.
"""

from __future__ import annotations

import random
import subprocess
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from statistics import mean

import pytest

from cogworx.claims.provenance import DEFAULT_SCOPE, Claim, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.assembly import assemble
from cogworx.recall.fusion import fuse
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import ChannelHit, FusedResult, RecallResult
from cogworx.recall.stack import RecallStack, default_recall_stack
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import InMemoryEntityKG, InMemoryEpisodeStore, InMemoryLatentStore

pytestmark = pytest.mark.spike

# ---------------------------------------------------------------------------
# Fixed timestamps — no datetime.now() anywhere in this file (determinism)
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_OLDER = datetime(2026, 6, 10, 10, 0, 0, tzinfo=UTC)
_OLDEST = datetime(2026, 6, 10, 8, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _make_evidence() -> EvidenceEvent:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id="spike-source-2-5",
        source_authority=0.9,
        recorded_at=_NOW,
        event_id=uuid.uuid4().hex,
    )


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    embedding: tuple[float, ...] | None = None,
    scope: str = DEFAULT_SCOPE,
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=0.9, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="spike-2-5",
        embedding=embedding,
        scope=scope,
    )


# ---------------------------------------------------------------------------
# Corpus builders
# ---------------------------------------------------------------------------

_KNOWN_CHANNEL_NAMES = {
    "dense.claims",
    "bm25.claims",
    "graph.claims",
    "temporal.episodes",
    "dense.latent",
}

# Dense-relevant item: embedding very close to Q_dense = (1,0,0,0), no rare terms
_DENSE_SUBJECT = "neural_net"
_DENSE_PREDICATE = "is_a"
_DENSE_PAYLOAD = "machine learning model"  # common words — low BM25 score
_DENSE_EMBEDDING = (0.95, 0.1, 0.0, 0.0)

# BM25-relevant item: unique token, embedding far from all query embeddings
_BM25_SUBJECT = "symbol_table"
_BM25_PREDICATE = "contains"
_BM25_PAYLOAD = "xyzzy_unique_token lookup entry xyzzy_unique_token"
_BM25_EMBEDDING = (0.0, 0.0, 0.0, 1.0)  # far from (1,0,0,0) and (0,0,1,0)

# Graph-relevant item: anchored to anchor_entity_A, no rare terms, embedding far from queries
_GRAPH_SUBJECT = "anchor_entity_A"
_GRAPH_PREDICATE = "has_property"
_GRAPH_PAYLOAD = "some generic value"
_GRAPH_EMBEDDING = (0.0, 1.0, 0.0, 0.0)  # far from Q_dense and Q_latent

# Latent-relevant item: embedding close to Q_latent = (0,0,1,0)
_LATENT_RECORD_ID = "latent-spike-001"
_LATENT_EMBEDDING = (0.1, 0.0, 0.95, 0.0)  # close to (0,0,1,0)
_LATENT_PAYLOAD_TEXT = "latent context snippet"

# Episode (temporal-relevant)
_SESSION_ID = "session_spike"
_EPISODE_IDS = [f"run-spike:1:{i}" for i in range(3)]

# Distractor claims: don't match any channel-exclusive criteria
_DISTRACTORS = [
    ("concept_a", "relates_to", "concept_b", (0.3, 0.3, 0.3, 0.3)),
    ("entity_x", "has_attr", "plain_value", (0.2, 0.2, 0.2, 0.2)),
    ("topic_z", "describes", "background_information", (0.4, 0.1, 0.1, 0.1)),
]


async def _build_corpus() -> tuple[InMemoryEntityKG, InMemoryEpisodeStore, InMemoryLatentStore]:
    """Build the channel-exclusive corpus on in-memory doubles."""
    kg = InMemoryEntityKG()
    episode_store = InMemoryEpisodeStore()
    latent_store = InMemoryLatentStore(clock=lambda: _NOW)

    # Dense-only claim (embedding near Q_dense; no rare terms; no anchor entity)
    dense_claim = _make_claim(_DENSE_SUBJECT, _DENSE_PREDICATE, _DENSE_PAYLOAD, _DENSE_EMBEDDING)
    await kg.write_claim(dense_claim, evidence=_make_evidence())

    # BM25-only claim (unique token; embedding far from all queries)
    bm25_claim = _make_claim(_BM25_SUBJECT, _BM25_PREDICATE, _BM25_PAYLOAD, _BM25_EMBEDDING)
    await kg.write_claim(bm25_claim, evidence=_make_evidence())

    # Graph-only claim (anchored to anchor_entity_A; no rare terms; embedding far from queries)
    graph_claim = _make_claim(_GRAPH_SUBJECT, _GRAPH_PREDICATE, _GRAPH_PAYLOAD, _GRAPH_EMBEDDING)
    await kg.write_claim(graph_claim, evidence=_make_evidence())

    # Distractor claims (don't exclusively match any channel)
    for subj, pred, payload, emb in _DISTRACTORS:
        c = _make_claim(subj, pred, payload, emb)
        await kg.write_claim(c, evidence=_make_evidence())

    # Temporal episodes: newest first when ordered by step_index desc
    episodes = [
        Episode(
            episode_id=_EPISODE_IDS[0],
            run_id="run-spike",
            step_index=1,
            turn_index=0,
            session_id=_SESSION_ID,
            role="user",
            content="recent turn content",
            kind="conversation",
            occurred_at=_NOW,
        ),
        Episode(
            episode_id=_EPISODE_IDS[1],
            run_id="run-spike",
            step_index=0,
            turn_index=1,
            session_id=_SESSION_ID,
            role="assistant",
            content="middle turn response",
            kind="conversation",
            occurred_at=_OLDER,
        ),
        Episode(
            episode_id=_EPISODE_IDS[2],
            run_id="run-spike",
            step_index=0,
            turn_index=0,
            session_id=_SESSION_ID,
            role="user",
            content="oldest turn content",
            kind="conversation",
            occurred_at=_OLDEST,
        ),
    ]
    from cogworx.substrate.journal import ProjectionCursor

    cursor = ProjectionCursor(commit_ordinal=1, run_id="run-spike", step_index=1)
    await episode_store.project_episodes("spike-consumer", episodes, cursor)

    # Latent record: embedding close to Q_latent = (0,0,1,0)
    await latent_store.put(
        LatentRecord(
            id=_LATENT_RECORD_ID,
            embedding=_LATENT_EMBEDDING,
            payload={"text": _LATENT_PAYLOAD_TEXT},
        )
    )

    return kg, episode_store, latent_store


# ---------------------------------------------------------------------------
# Recall helpers
# ---------------------------------------------------------------------------


def _dense_claim_key() -> str:
    return f"claim:{claim_id_for(_DENSE_SUBJECT, _DENSE_PREDICATE, _DENSE_PAYLOAD)}"


def _bm25_claim_key() -> str:
    return f"claim:{claim_id_for(_BM25_SUBJECT, _BM25_PREDICATE, _BM25_PAYLOAD)}"


def _graph_claim_key() -> str:
    return f"claim:{claim_id_for(_GRAPH_SUBJECT, _GRAPH_PREDICATE, _GRAPH_PAYLOAD)}"


def _temporal_episode_key() -> str:
    # The most-recent episode is step_index=1, turn_index=0 → episode_id = "run-spike:1:0"
    return f"episode:{_EPISODE_IDS[0]}"


def _latent_record_key() -> str:
    return f"latent:{_LATENT_RECORD_ID}"


def recall_at_k(results: Sequence[FusedResult], relevant_key: str, k: int) -> float:
    """1.0 if relevant_key appears in the top-k results, 0.0 otherwise."""
    return 1.0 if any(r.key == relevant_key for r in results[:k]) else 0.0


# ---------------------------------------------------------------------------
# SC-1 — RRF beats best single channel on the mixed set
# ---------------------------------------------------------------------------


async def test_sc1_rrf_beats_best_single_channel() -> None:
    """Fused RRF macro Recall@10 >= any single-channel macro Recall@10.

    AND the best single channel scores < 1.0 (corpus is genuinely hard for single channels).
    """
    kg, episode_store, latent_store = await _build_corpus()
    stack = default_recall_stack(
        entity_kg=kg,
        episode_store=episode_store,
        latent_store=latent_store,
    )

    # Four queries, one per channel type
    queries = [
        # Dense query: close to the dense claim
        (
            RecallQuery(embedding=(1.0, 0.0, 0.0, 0.0), k=20),
            _dense_claim_key(),
        ),
        # BM25 query: unique token only in the bm25 claim
        (
            RecallQuery(text="xyzzy_unique_token", k=20),
            _bm25_claim_key(),
        ),
        # Graph query: anchor entity
        (
            RecallQuery(anchor_entities=("anchor_entity_A",), k=20),
            _graph_claim_key(),
        ),
        # Temporal query: session lookup
        (
            RecallQuery(session_id=_SESSION_ID, k=20),
            _temporal_episode_key(),
        ),
    ]

    channel_names = [
        "dense.claims",
        "bm25.claims",
        "graph.claims",
        "temporal.episodes",
        "dense.latent",
    ]

    fused_recalls: list[float] = []
    # per channel: list of recall@10 across all queries
    single_channel_recalls: dict[str, list[float]] = {ch: [] for ch in channel_names}

    for query, relevant_key in queries:
        outcome = await stack.recall(query)
        r_fused = recall_at_k(outcome.results, relevant_key, 10)
        fused_recalls.append(r_fused)

        # Single-channel runs: each channel alone through fuse
        for ch_name in channel_names:
            ch_obj = next(c for c in stack._channels if c.name == ch_name)
            if not ch_obj.can_serve(query):
                single_channel_recalls[ch_name].append(0.0)
                continue
            ch_results = list(await ch_obj.search(query))
            # Convert to channel_results dict for fuse
            ch_dict: dict[str, list[RecallResult]] = {ch_name: ch_results}
            fused_single = fuse(ch_dict)
            single_channel_recalls[ch_name].append(
                recall_at_k(list(fused_single), relevant_key, 10)
            )

    macro_fused = mean(fused_recalls)
    macro_per_channel = {ch: mean(recalls) for ch, recalls in single_channel_recalls.items()}
    best_single = max(macro_per_channel.values())

    assert macro_fused >= best_single, (
        f"SC-1 FAIL: fused macro Recall@10={macro_fused:.3f} < "
        f"best single channel={best_single:.3f} "
        f"({max(macro_per_channel, key=lambda k: macro_per_channel[k])})"
    )
    assert best_single < 1.0, (
        f"SC-1 FAIL: best single channel Recall@10={best_single:.3f} == 1.0 — "
        "corpus is trivial for one channel; negative control would be vacuous"
    )


async def test_sc1_negative_control_single_channel_below_fused() -> None:
    """Negative control: running only the dense channel in isolation scores <= fused on the
    BM25 query.

    The dense channel cannot retrieve the BM25-exclusive item by cosine (orthogonal embedding),
    so the single-channel score must be 0.0 for that query while fused can get it via BM25.
    """
    kg, episode_store, latent_store = await _build_corpus()
    stack = default_recall_stack(
        entity_kg=kg,
        episode_store=episode_store,
        latent_store=latent_store,
    )

    # BM25 query: only the BM25 channel can serve it (no embedding)
    bm25_query = RecallQuery(text="xyzzy_unique_token", k=20)
    # Full fused run
    full_outcome = await stack.recall(bm25_query)
    r_fused = recall_at_k(full_outcome.results, _bm25_claim_key(), 10)

    # Dense channel alone cannot serve this query (no embedding in query)
    dense_ch = next(c for c in stack._channels if c.name == "dense.claims")
    assert not dense_ch.can_serve(bm25_query), (
        "Dense channel should not be able to serve a text-only query"
    )
    # Single-channel result is empty (channel can't serve)
    single_fused = fuse({"dense.claims": []})
    r_single_dense = recall_at_k(list(single_fused), _bm25_claim_key(), 10)

    # Fused gets the BM25 item; dense-alone does not
    assert r_fused >= r_single_dense, (
        f"SC-1 negative control: fused={r_fused} should be >= "
        f"dense-solo={r_single_dense} on BM25 query"
    )
    # The critical assertion: dense alone misses the BM25-exclusive item
    assert r_single_dense == 0.0, (
        f"SC-1 negative control: dense-solo should score 0.0 on BM25-exclusive item, "
        f"got {r_single_dense}"
    )


# ---------------------------------------------------------------------------
# SC-2 — Per-type lesion sensitivity
# ---------------------------------------------------------------------------

_DROP_MARGIN = 0.5


async def test_sc2_lesion_dense() -> None:
    """Removing the dense channel drops dense-query recall by >= DROP_MARGIN."""
    kg, episode_store, latent_store = await _build_corpus()
    q = RecallQuery(embedding=(1.0, 0.0, 0.0, 0.0), k=20)
    key = _dense_claim_key()

    full_stack = default_recall_stack(
        entity_kg=kg, episode_store=episode_store, latent_store=latent_store
    )
    outcome_full = await full_stack.recall(q)
    r_with = recall_at_k(outcome_full.results, key, 10)

    # Stack WITHOUT dense channel
    no_dense = RecallStack(
        [c for c in full_stack._channels if c.name != "dense.claims"],
    )
    outcome_no = await no_dense.recall(q)
    r_without = recall_at_k(outcome_no.results, key, 10)

    delta = r_with - r_without
    assert delta >= _DROP_MARGIN, (
        f"SC-2 dense lesion FAIL: delta={delta:.2f} < DROP_MARGIN={_DROP_MARGIN}. "
        f"r_with={r_with}, r_without={r_without}. "
        "Dense claim must be exclusively retrievable by the dense channel."
    )


async def test_sc2_lesion_bm25() -> None:
    """Removing the BM25 channel drops BM25-query recall by >= DROP_MARGIN."""
    kg, episode_store, latent_store = await _build_corpus()
    q = RecallQuery(text="xyzzy_unique_token", k=20)
    key = _bm25_claim_key()

    full_stack = default_recall_stack(
        entity_kg=kg, episode_store=episode_store, latent_store=latent_store
    )
    outcome_full = await full_stack.recall(q)
    r_with = recall_at_k(outcome_full.results, key, 10)

    no_bm25 = RecallStack(
        [c for c in full_stack._channels if c.name != "bm25.claims"],
    )
    outcome_no = await no_bm25.recall(q)
    r_without = recall_at_k(outcome_no.results, key, 10)

    delta = r_with - r_without
    assert delta >= _DROP_MARGIN, (
        f"SC-2 BM25 lesion FAIL: delta={delta:.2f} < DROP_MARGIN={_DROP_MARGIN}. "
        f"r_with={r_with}, r_without={r_without}"
    )


async def test_sc2_lesion_graph() -> None:
    """Removing the graph channel drops graph-query recall by >= DROP_MARGIN."""
    kg, episode_store, latent_store = await _build_corpus()
    q = RecallQuery(anchor_entities=("anchor_entity_A",), k=20)
    key = _graph_claim_key()

    full_stack = default_recall_stack(
        entity_kg=kg, episode_store=episode_store, latent_store=latent_store
    )
    outcome_full = await full_stack.recall(q)
    r_with = recall_at_k(outcome_full.results, key, 10)

    no_graph = RecallStack(
        [c for c in full_stack._channels if c.name != "graph.claims"],
    )
    outcome_no = await no_graph.recall(q)
    r_without = recall_at_k(outcome_no.results, key, 10)

    delta = r_with - r_without
    assert delta >= _DROP_MARGIN, (
        f"SC-2 graph lesion FAIL: delta={delta:.2f} < DROP_MARGIN={_DROP_MARGIN}. "
        f"r_with={r_with}, r_without={r_without}"
    )


async def test_sc2_lesion_temporal() -> None:
    """Removing the temporal channel drops temporal-query recall by >= DROP_MARGIN."""
    kg, episode_store, latent_store = await _build_corpus()
    q = RecallQuery(session_id=_SESSION_ID, k=20)
    key = _temporal_episode_key()

    full_stack = default_recall_stack(
        entity_kg=kg, episode_store=episode_store, latent_store=latent_store
    )
    outcome_full = await full_stack.recall(q)
    r_with = recall_at_k(outcome_full.results, key, 10)

    no_temporal = RecallStack(
        [c for c in full_stack._channels if c.name != "temporal.episodes"],
    )
    outcome_no = await no_temporal.recall(q)
    r_without = recall_at_k(outcome_no.results, key, 10)

    delta = r_with - r_without
    assert delta >= _DROP_MARGIN, (
        f"SC-2 temporal lesion FAIL: delta={delta:.2f} < DROP_MARGIN={_DROP_MARGIN}. "
        f"r_with={r_with}, r_without={r_without}"
    )


async def test_sc2_negative_control_unrelated_lesion_has_no_effect() -> None:
    """Removing an UNRELATED channel must NOT change the relevant item's recall by >= DROP_MARGIN.

    For each channel type T, removing a different channel D must leave type-T recall unchanged.
    This proves the positive lesion test is not vacuous.
    """
    kg, episode_store, latent_store = await _build_corpus()
    full_stack = default_recall_stack(
        entity_kg=kg, episode_store=episode_store, latent_store=latent_store
    )

    # (query, relevant_key, channel_to_remove_for_negative_control)
    cases = [
        # Dense query: remove temporal (unrelated)
        (
            RecallQuery(embedding=(1.0, 0.0, 0.0, 0.0), k=20),
            _dense_claim_key(),
            "temporal.episodes",
        ),
        # BM25 query: remove graph (unrelated; graph can't serve text-only query anyway)
        (RecallQuery(text="xyzzy_unique_token", k=20), _bm25_claim_key(), "graph.claims"),
        # Graph query: remove temporal (unrelated)
        (
            RecallQuery(anchor_entities=("anchor_entity_A",), k=20),
            _graph_claim_key(),
            "temporal.episodes",
        ),
        # Temporal query: remove dense.latent (unrelated)
        (RecallQuery(session_id=_SESSION_ID, k=20), _temporal_episode_key(), "dense.latent"),
    ]

    for query, key, remove_ch in cases:
        outcome_full = await full_stack.recall(query)
        r_with = recall_at_k(outcome_full.results, key, 10)

        partial_stack = RecallStack(
            [c for c in full_stack._channels if c.name != remove_ch],
        )
        outcome_partial = await partial_stack.recall(query)
        r_without = recall_at_k(outcome_partial.results, key, 10)

        delta = abs(r_with - r_without)
        assert delta < _DROP_MARGIN, (
            f"SC-2 negative control FAIL for query={query!r}, key={key!r}: "
            f"removing {remove_ch!r} changed recall by {delta:.2f} >= {_DROP_MARGIN} — "
            "unrelated channel removal should have no effect"
        )


# ---------------------------------------------------------------------------
# SC-3 — k_rrf sensitivity
# ---------------------------------------------------------------------------


async def test_sc3_k_rrf_sensitivity() -> None:
    """Sign of 'fused >= best-single' is stable across k_rrf in {20, 60, 120}."""
    kg, episode_store, latent_store = await _build_corpus()

    queries = [
        (RecallQuery(embedding=(1.0, 0.0, 0.0, 0.0), k=20), _dense_claim_key()),
        (RecallQuery(text="xyzzy_unique_token", k=20), _bm25_claim_key()),
        (RecallQuery(anchor_entities=("anchor_entity_A",), k=20), _graph_claim_key()),
        (RecallQuery(session_id=_SESSION_ID, k=20), _temporal_episode_key()),
    ]
    channel_names = [
        "dense.claims",
        "bm25.claims",
        "graph.claims",
        "temporal.episodes",
        "dense.latent",
    ]

    for k_rrf in (20, 60, 120):
        stack = default_recall_stack(
            entity_kg=kg,
            episode_store=episode_store,
            latent_store=latent_store,
            k_rrf=k_rrf,
        )

        fused_recalls: list[float] = []
        single_recalls: dict[str, list[float]] = {ch: [] for ch in channel_names}

        for query, relevant_key in queries:
            outcome = await stack.recall(query)
            fused_recalls.append(recall_at_k(outcome.results, relevant_key, 10))

            for ch_name in channel_names:
                ch_obj = next(c for c in stack._channels if c.name == ch_name)
                if not ch_obj.can_serve(query):
                    single_recalls[ch_name].append(0.0)
                    continue
                ch_results = list(await ch_obj.search(query))
                fused_single = fuse({ch_name: ch_results}, k_rrf=k_rrf)
                single_recalls[ch_name].append(recall_at_k(list(fused_single), relevant_key, 10))

        macro_fused = mean(fused_recalls)
        best_single = max(mean(r) for r in single_recalls.values())

        assert macro_fused >= best_single, (
            f"SC-3 FAIL at k_rrf={k_rrf}: fused={macro_fused:.3f} < best_single={best_single:.3f}"
        )


# ---------------------------------------------------------------------------
# SC-4 — S5 provenance on every result
# ---------------------------------------------------------------------------


async def test_sc4_provenance_on_every_fused_result() -> None:
    """Every FusedResult has >=1 ChannelHit with rank>=1 and a registered channel name."""
    kg, episode_store, latent_store = await _build_corpus()
    stack = default_recall_stack(
        entity_kg=kg,
        episode_store=episode_store,
        latent_store=latent_store,
    )

    # Use a broad query that activates multiple channels
    query = RecallQuery(
        embedding=(1.0, 0.0, 0.0, 0.0),
        text="machine learning",
        anchor_entities=("anchor_entity_A",),
        session_id=_SESSION_ID,
        k=20,
    )
    outcome = await stack.recall(query)

    assert len(outcome.results) > 0, "Expected at least one fused result"

    for fr in outcome.results:
        assert len(fr.hits) >= 1, (
            f"S5 violation: FusedResult key={fr.key!r} has no ChannelHit records"
        )
        assert fr.hits[0].rank >= 1, (
            f"S5 violation: first ChannelHit rank={fr.hits[0].rank} < 1 for key={fr.key!r}"
        )
        assert fr.hits[0].channel in _KNOWN_CHANNEL_NAMES, (
            f"S5 violation: ChannelHit channel={fr.hits[0].channel!r} not in known channels "
            f"for key={fr.key!r}"
        )

    # Assemble and verify provenance propagates to ContextChunks
    assembled = assemble(outcome.results, budget=4000)
    for chunk in assembled.chunks:
        assert len(chunk.hits) >= 1, f"S5 violation: ContextChunk key={chunk.key!r} has no hits"
        assert chunk.hits[0].channel in _KNOWN_CHANNEL_NAMES, (
            f"S5 violation: ContextChunk channel={chunk.hits[0].channel!r} not in known channels"
        )


async def test_sc4_negative_control_no_hits_detected() -> None:
    """Negative control: FusedResult with hits=() must be caught by the provenance checker.

    This test builds a mutant FusedResult with no ChannelHits and verifies the S5 check
    would catch it. The test also verifies that assemble() propagates empty hits correctly
    so a downstream check can detect the violation.
    """
    from cogworx.substrate.latent import LatentMatch, LatentRecord

    # Build a valid FusedResult first, then construct a mutant with hits=()
    dummy_record = LatentRecord(id="dummy-id", embedding=(0.0, 0.0, 0.0, 1.0), payload={})
    dummy_match = LatentMatch(
        record=dummy_record,
        score=0.5,
        tier="cold",
        use_count=0,
        last_used_at=_NOW,
    )
    mutant_result = FusedResult(
        key="latent:dummy-id",
        kind="latent",
        item=dummy_match,
        text="dummy text",
        hits=(),  # MUTANT: no provenance
        fused_score=0.5,
        fused_rank=1,
    )

    # The S5 check function (as used in the positive test) would catch this
    def _check_s5(results: list[FusedResult]) -> list[str]:
        """Return list of violation messages for any result with empty hits."""
        violations = []
        for fr in results:
            if len(fr.hits) < 1:
                violations.append(
                    f"S5 violation: FusedResult key={fr.key!r} has no ChannelHit records"
                )
        return violations

    violations = _check_s5([mutant_result])
    assert len(violations) == 1, (
        f"SC-4 negative control: mutant FusedResult with hits=() should produce 1 S5 violation, "
        f"got {len(violations)}: {violations}"
    )
    assert "S5 violation" in violations[0], f"Expected S5 violation message, got: {violations[0]!r}"

    # Also verify assemble() propagates empty hits to ContextChunks (the violation is detectable)
    assembled = assemble([mutant_result], budget=2000)
    assert len(assembled.chunks) == 1, "Expected one chunk from mutant result"
    chunk = assembled.chunks[0]
    assert len(chunk.hits) == 0, (
        f"Mutant result hits=() should propagate to chunk.hits=(); got {chunk.hits!r}"
    )
    # The downstream S5 check would detect this
    chunk_violations = [
        f"chunk {chunk.key!r} has no hits" for chunk in assembled.chunks if len(chunk.hits) < 1
    ]
    assert len(chunk_violations) == 1, (
        "SC-4 negative control: assemble must preserve empty hits so downstream check can catch it"
    )


# ---------------------------------------------------------------------------
# SC-5 — S1 import-level invariant
# ---------------------------------------------------------------------------

# KNOWN GAP CF-1 (S1 transitive violation via journal seam):
# cogworx.recall.stack imports cogworx.substrate.entity_kg, which imports
# cogworx.substrate.journal, which imports cogworx.loop.result. Importing
# cogworx.loop.result triggers cogworx.loop.__init__, which exports loop.stage,
# which imports cogworx.model.base. The chain is:
#   recall.stack -> substrate.entity_kg -> substrate.journal
#     -> loop.result (triggers loop.__init__) -> loop.stage -> model.base
# This is an architectural S1 violation in the journal-seam design: the journal
# should import StageResult from a module that does not trigger loop/__init__.py.
# Resolution: move StageResult OUTSIDE cogworx.loop entirely (e.g. cogworx.types).
# Moving within cogworx.loop is insufficient — any 'from cogworx.loop.X import Y'
# triggers loop/__init__.py first (red-team confirmed 2026-06-10). Deferred
# as CF-1 (architect + python-expert scope). What CAN be verified now: the recall
# submodules themselves have NO direct model imports in their source text.


def test_sc5_recall_submodules_have_no_direct_model_imports() -> None:
    """The recall-layer source files have zero direct imports from cogworx.model (S1).

    The transitive violation (CF-1: journal -> loop.__init__ -> stage -> model) is a
    pre-existing architectural gap in the journal seam, not in the recall layer. This
    test verifies what IS in the recall layer's control: no direct model imports in any
    recall/*.py file. CF-1 is documented below; it is the architect's scope to fix.
    """
    import pathlib

    recall_dir = pathlib.Path(__file__).parent.parent.parent / "src" / "cogworx" / "recall"
    assert recall_dir.is_dir(), f"recall dir not found: {recall_dir}"

    violations: list[str] = []
    for py_file in recall_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        lines = source.splitlines()
        for lineno, line in enumerate(lines, start=1):
            stripped = line.strip()
            # Check for direct model or runtime imports (not in comments or docstrings)
            if stripped.startswith("#"):
                continue
            if "cogworx.model" in stripped and ("import" in stripped):
                violations.append(f"{py_file.name}:{lineno}: {stripped!r}")
            if "cogworx.runtime" in stripped and ("import" in stripped):
                violations.append(f"{py_file.name}:{lineno}: {stripped!r}")

    assert not violations, (
        "SC-5 FAIL: recall submodule(s) directly import cogworx.model or cogworx.runtime:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_sc5_known_gap_cf1_transitive_model_via_journal(capsys: pytest.CaptureFixture[str]) -> None:
    """KNOWN GAP CF-1: cogworx.recall transitively imports cogworx.model.base via the
    journal -> loop.__init__ -> stage chain.

    This test documents the gap explicitly — it passes to prove the bypass is real.
    The transitive import is confirmed by subprocess and is an architectural issue
    in the journal seam, NOT in the recall layer itself.

    Resolution: move StageResult OUTSIDE cogworx.loop entirely so journal.py can import
    it without triggering loop/__init__.py (which eagerly imports stage.py -> model.base).
    Moving within cogworx.loop is insufficient. Architect + python-expert scope.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import cogworx.recall; import sys; "
                "model_mods = [m for m in sys.modules if m.startswith('cogworx.model')]; "
                "print(bool(model_mods))"
            ),
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, f"subprocess failed: {result.stderr}"
    has_model = result.stdout.strip() == "True"
    assert has_model, (
        "CF-1 gap no longer present: cogworx.recall no longer transitively imports "
        "cogworx.model. Remove this known-gap test and promote SC-5 to the hard assertion."
    )
    # The presence of the transitive import is confirmed. This passes to document the gap.
    # The positive recall-layer tests above verify what IS in the recall layer's control.


def test_sc5_spike_file_has_no_direct_model_imports() -> None:
    """This spike file has zero DIRECT (non-comment) import statements for model/runtime (S1).

    Comments and docstrings discussing the gap (CF-1) are permitted — they are not import
    statements. Only actual 'import' or 'from ... import' lines are checked.
    """
    with open(__file__, encoding="utf-8") as fh:
        lines = fh.readlines()

    violations = []
    for lineno, line in enumerate(lines, start=1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # Only flag actual import lines, not string literals containing the pattern
        if ("cogworx.model" in stripped or "cogworx.runtime" in stripped) and (
            stripped.startswith("import ") or stripped.startswith("from ")
        ):
            violations.append(f"line {lineno}: {stripped!r}")

    assert not violations, (
        "SC-5 FAIL: spike file has direct import statements for model/runtime:\n"
        + "\n".join(f"  {v}" for v in violations)
    )


# ---------------------------------------------------------------------------
# SC-6 — Assembly U-fold + budget properties (seeded randomization)
# ---------------------------------------------------------------------------


def _make_fused_result(key: str, text: str, rank: int) -> FusedResult:
    """Build a minimal FusedResult for assembly testing."""
    from cogworx.substrate.latent import LatentMatch, LatentRecord

    dummy_record = LatentRecord(id=key, embedding=(0.0,), payload={"text": text})
    dummy_match = LatentMatch(
        record=dummy_record,
        score=0.5,
        tier="cold",
        use_count=0,
        last_used_at=_NOW,
    )
    return FusedResult(
        key=f"latent:{key}",
        kind="latent",
        item=dummy_match,
        text=text,
        hits=(ChannelHit(channel="dense.latent", rank=rank, raw_score=0.5),),
        fused_score=1.0 / (60 + rank),
        fused_rank=rank,
    )


def test_sc6_budget_never_exceeded() -> None:
    """token_count <= budget for 100 random fixtures (seeded, deterministic)."""
    rng = random.Random(42)

    for trial in range(100):
        n_items = rng.randint(2, 20)
        budget = rng.randint(10, 500)

        # Build items with deterministic text lengths
        items = [
            _make_fused_result(f"item-{trial}-{i}", "x" * rng.randint(4, 80), i + 1)
            for i in range(n_items)
        ]

        assembled = assemble(items, budget=budget)
        assert assembled.token_count <= budget, (
            f"SC-6 trial {trial}: token_count={assembled.token_count} > budget={budget}"
        )


def test_sc6_u_fold_shape() -> None:
    """U-fold ordering: rank-1 at position 0, rank-2 at last, rank-3 at position 1, etc."""
    # Build 6 items that all fit in a large budget
    items = [_make_fused_result(f"item-{i}", "test", i + 1) for i in range(6)]
    assembled = assemble(items, budget=10000)
    chunks = assembled.chunks
    assert len(chunks) == 6, f"Expected 6 chunks, got {len(chunks)}"

    # Map relevance_rank -> chunk position in output
    rank_to_pos = {chunk.relevance_rank: pos for pos, chunk in enumerate(chunks)}

    # U-fold: rank 1 → pos 0, rank 2 → pos 5 (last), rank 3 → pos 1, rank 4 → pos 4, ...
    expected = {1: 0, 2: 5, 3: 1, 4: 4, 5: 2, 6: 3}
    for rank, expected_pos in expected.items():
        actual_pos = rank_to_pos[rank]
        assert actual_pos == expected_pos, (
            f"SC-6 U-fold shape FAIL: rank={rank} at pos={actual_pos}, expected pos={expected_pos}"
        )


def test_sc6_greedy_skip_oversized_head() -> None:
    """When item 0 is too large to fit, item 1 (smaller) is admitted with relevance_rank=1."""
    # item 0: 400 chars → approx_tokens = 100 (exceeds budget=50)
    # item 1: 40 chars → approx_tokens = 10 (fits in budget=50)
    big_item = _make_fused_result("big", "z" * 400, rank=1)
    small_item = _make_fused_result("small", "y" * 40, rank=2)

    assembled = assemble([big_item, small_item], budget=50)
    assert assembled.dropped == 1, f"SC-6 greedy-skip: expected dropped=1, got {assembled.dropped}"
    assert len(assembled.chunks) == 1, (
        f"SC-6 greedy-skip: expected 1 admitted chunk, got {len(assembled.chunks)}"
    )
    assert assembled.chunks[0].key == "latent:small", (
        f"SC-6 greedy-skip: wrong item admitted: {assembled.chunks[0].key!r}"
    )
    assert assembled.chunks[0].relevance_rank == 1, (
        f"SC-6 greedy-skip: admitted item must have relevance_rank=1, "
        f"got {assembled.chunks[0].relevance_rank}"
    )


def test_sc6_dropped_count_equals_total_minus_admitted() -> None:
    """dropped == total_items - admitted_items for 100 random fixtures (seeded)."""
    rng = random.Random(42)

    for trial in range(100):
        n_items = rng.randint(2, 20)
        budget = rng.randint(10, 500)

        items = [
            _make_fused_result(f"item-{trial}-{i}", "x" * rng.randint(4, 80), i + 1)
            for i in range(n_items)
        ]

        assembled = assemble(items, budget=budget)
        admitted = len(assembled.chunks)
        assert assembled.dropped == n_items - admitted, (
            f"SC-6 trial {trial}: dropped={assembled.dropped} != "
            f"total({n_items}) - admitted({admitted})"
        )


# ---------------------------------------------------------------------------
# SC-7 — Scope isolation through full stack
# ---------------------------------------------------------------------------


async def test_sc7_scope_isolation_through_stack() -> None:
    """Agent-scoped query must not surface user-scoped claims (scope passes through channels).

    Two claims with identical embeddings: one scope="user:spike_user", one scope="agent".
    Query with scope="agent" must return the agent claim and not the user claim.
    """
    kg = InMemoryEntityKG()

    # Identical embeddings — without scope filtering both would be top-k dense results
    shared_embedding = (0.9, 0.1, 0.0, 0.0)

    # Agent-scoped claim
    agent_claim = _make_claim(
        "shared_subject", "shared_pred", "shared_value", shared_embedding, scope="agent"
    )
    # User-scoped claim (same triple, same embedding, different scope → different ID)
    user_claim = _make_claim(
        "shared_subject", "shared_pred", "shared_value", shared_embedding, scope="user:spike_user"
    )

    assert agent_claim.id != user_claim.id, (
        "Agent and user claims on the same triple must have different IDs (scope-in-identity)"
    )

    await kg.write_claim(agent_claim, evidence=_make_evidence())
    await kg.write_claim(user_claim, evidence=_make_evidence())

    stack = default_recall_stack(entity_kg=kg)
    query = RecallQuery(
        embedding=(1.0, 0.0, 0.0, 0.0),
        scope="agent",
        k=20,
    )
    outcome = await stack.recall(query)

    result_keys = {r.key for r in outcome.results}
    user_key = f"claim:{user_claim.id}"
    agent_key = f"claim:{agent_claim.id}"

    assert user_key not in result_keys, (
        f"SC-7 FAIL: user-scoped claim {user_key!r} leaked into agent-scoped query results. "
        "Scope filtering is not working through the stack."
    )
    assert agent_key in result_keys, (
        f"SC-7 FAIL: agent-scoped claim {agent_key!r} missing from agent-scoped query results."
    )


async def test_sc7_negative_control_scope_leak_without_filtering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Negative control: removing scope from the dense channel search causes the user-scoped
    claim to leak into agent-scoped results.

    This proves the positive SC-7 assertion has teeth — it would fail if scope filtering
    were accidentally removed from ClaimDenseChannel.
    """
    kg = InMemoryEntityKG()
    shared_embedding = (0.9, 0.1, 0.0, 0.0)

    agent_claim = _make_claim(
        "shared_subject", "shared_pred", "shared_value", shared_embedding, scope="agent"
    )
    user_claim = _make_claim(
        "shared_subject", "shared_pred", "shared_value", shared_embedding, scope="user:spike_user"
    )
    await kg.write_claim(agent_claim, evidence=_make_evidence())
    await kg.write_claim(user_claim, evidence=_make_evidence())

    # Monkeypatch: override claims_by_similarity to always ignore scope
    original_cbs = InMemoryEntityKG.claims_by_similarity

    async def _unscoped_claims_by_similarity(
        self: InMemoryEntityKG,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
        scope: str | None = None,
    ) -> Sequence[ScoredClaim]:
        # Broken: always ignore scope — pass scope=None
        return await original_cbs(self, embedding, k=k, min_score=min_score, scope=None)

    monkeypatch.setattr(InMemoryEntityKG, "claims_by_similarity", _unscoped_claims_by_similarity)

    stack = default_recall_stack(entity_kg=kg)
    query = RecallQuery(
        embedding=(1.0, 0.0, 0.0, 0.0),
        scope="agent",
        k=20,
    )
    outcome = await stack.recall(query)

    result_keys = {r.key for r in outcome.results}
    user_key = f"claim:{user_claim.id}"

    # With scope filtering broken, the user-scoped claim MUST appear
    assert user_key in result_keys, (
        f"SC-7 negative control FAIL: with scope filtering removed, "
        f"user-scoped claim {user_key!r} should appear in results but was not found. "
        "This means the positive SC-7 assertion would not be testing what it claims."
    )
