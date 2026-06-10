"""Unit tests for Pod 2.5 Stream C — the 5 concrete recall channel classes.

All tests are pure-Python: no database, no model, no network.
asyncio_mode = "auto" (pyproject.toml), so no @pytest.mark.asyncio needed.

Invariants:
  S1 — no model calls (channels are pure substrate reads).
  S3 — each channel wraps exactly one substrate engine method.
  S5 — every RecallResult carries a ChannelHit with channel=name, 1-based rank, raw_score.
  S8 — can_serve False → search never called; can_serve True + no results → [] (distinct states).
  S9 — validity post-filter only removes (never reweights) expired claims.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.channels import (
    ClaimDenseChannel,
    ClaimGraphChannel,
    ClaimTextChannel,
    EpisodeRecencyChannel,
    LatentDenseChannel,
)
from cogworx.recall.query import RecallQuery
from cogworx.substrate.episodes import Episode
from cogworx.substrate.journal import ProjectionCursor
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryEpisodeStore,
    InMemoryLatentStore,
)

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)
_T_PAST = _T0 - timedelta(hours=1)

# A simple normalized unit-vector embedding
_EMBED_A: tuple[float, ...] = (1.0, 0.0, 0.0)
_EMBED_B: tuple[float, ...] = (0.0, 1.0, 0.0)
_EMBED_C: tuple[float, ...] = (0.0, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    embedding: tuple[float, ...] | None = None,
    valid_to: datetime | None = None,
    ingest_time: datetime = _T0,
    scope: str = "agent",
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=_prov(),
        valid_from=_T0,
        valid_to=valid_to,
        ingest_time=ingest_time,
        created_by="test",
        embedding=embedding,
        scope=scope,
    )


def _ev(source_id: str = "src-a") -> EvidenceEvent:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source_id,
        source_authority=1.0,
        recorded_at=_T0,
    )


def _episode(
    session_id: str,
    step_index: int,
    turn_index: int = 0,
    occurred_at: datetime = _T0,
    role: str = "user",
    content: str = "hello",
) -> Episode:
    return Episode(
        episode_id=f"run-01:{step_index}:{turn_index}",
        run_id="run-01",
        step_index=step_index,
        turn_index=turn_index,
        session_id=session_id,
        role=role,
        content=content,
        kind="conversational",
        occurred_at=occurred_at,
    )


def _cursor(ordinal: int = 1) -> ProjectionCursor:
    return ProjectionCursor(commit_ordinal=ordinal, run_id="run-01", step_index=0)


# ---------------------------------------------------------------------------
# 1. can_serve logic per channel
# ---------------------------------------------------------------------------


def test_claim_dense_can_serve_false_when_no_embedding() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimDenseChannel(kg)
    assert not ch.can_serve(RecallQuery())


def test_claim_dense_can_serve_true_with_embedding() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimDenseChannel(kg)
    assert ch.can_serve(RecallQuery(embedding=_EMBED_A))


def test_claim_text_can_serve_false_when_no_text() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimTextChannel(kg)
    assert not ch.can_serve(RecallQuery())


def test_claim_text_can_serve_false_when_text_blank() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimTextChannel(kg)
    assert not ch.can_serve(RecallQuery(text="   "))


def test_claim_text_can_serve_true_with_nonempty_text() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimTextChannel(kg)
    assert ch.can_serve(RecallQuery(text="chocolate"))


def test_claim_graph_can_serve_false_when_no_anchors() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimGraphChannel(kg)
    assert not ch.can_serve(RecallQuery())


def test_claim_graph_can_serve_true_with_anchor() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimGraphChannel(kg)
    assert ch.can_serve(RecallQuery(anchor_entities=("Alice",)))


def test_episode_recency_can_serve_false_when_no_session() -> None:
    store = InMemoryEpisodeStore()
    ch = EpisodeRecencyChannel(store)
    assert not ch.can_serve(RecallQuery())


def test_episode_recency_can_serve_true_with_session() -> None:
    store = InMemoryEpisodeStore()
    ch = EpisodeRecencyChannel(store)
    assert ch.can_serve(RecallQuery(session_id="sess-1"))


def test_latent_dense_can_serve_false_when_no_embedding() -> None:
    store = InMemoryLatentStore()
    ch = LatentDenseChannel(store)
    assert not ch.can_serve(RecallQuery())


def test_latent_dense_can_serve_true_with_embedding() -> None:
    store = InMemoryLatentStore()
    ch = LatentDenseChannel(store)
    assert ch.can_serve(RecallQuery(embedding=_EMBED_A))


# ---------------------------------------------------------------------------
# 2. ClaimDenseChannel.search
# ---------------------------------------------------------------------------


async def test_claim_dense_search_returns_recall_results() -> None:
    kg = InMemoryEntityKG()
    c1 = _make_claim("alice", "likes", "chocolate", embedding=_EMBED_A)
    c2 = _make_claim("bob", "is", "human", embedding=_EMBED_B)
    await kg.write_claim(c1, evidence=_ev())
    await kg.write_claim(c2, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    # Query toward A — c1 should rank higher
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10))

    assert len(results) >= 1
    first = results[0]
    assert first.kind == "claim"
    assert first.key.startswith("claim:")
    assert first.hit.channel == "dense.claims"
    assert first.hit.rank == 1
    assert first.hit.raw_score is not None
    assert first.hit.raw_score > 0.0
    # Ranks are 1-based and contiguous
    for i, r in enumerate(results):
        assert r.hit.rank == i + 1


async def test_claim_dense_search_top_result_is_most_similar() -> None:
    kg = InMemoryEntityKG()
    c1 = _make_claim("alice", "likes", "chocolate", embedding=_EMBED_A)
    c2 = _make_claim("bob", "is", "human", embedding=_EMBED_B)
    await kg.write_claim(c1, evidence=_ev())
    await kg.write_claim(c2, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10))

    assert results[0].key == f"claim:{c1.id}"


# ---------------------------------------------------------------------------
# 3. ClaimTextChannel.search
# ---------------------------------------------------------------------------


async def test_claim_text_search_matching_claim_first() -> None:
    kg = InMemoryEntityKG()
    c1 = _make_claim("alice", "likes", "chocolate cake")
    c2 = _make_claim("bob", "is", "engineer")
    await kg.write_claim(c1, evidence=_ev())
    await kg.write_claim(c2, evidence=_ev())

    ch = ClaimTextChannel(kg)
    results = await ch.search(RecallQuery(text="chocolate", k=10))

    assert len(results) >= 1
    assert results[0].key == f"claim:{c1.id}"
    assert results[0].hit.channel == "bm25.claims"
    assert results[0].hit.raw_score is not None
    assert results[0].hit.raw_score > 0.0


async def test_claim_text_search_ranks_1_based() -> None:
    kg = InMemoryEntityKG()
    c1 = _make_claim("alice", "likes", "chocolate cake")
    c2 = _make_claim("alice", "also likes", "chocolate cookies")
    await kg.write_claim(c1, evidence=_ev())
    await kg.write_claim(c2, evidence=_ev())

    ch = ClaimTextChannel(kg)
    results = await ch.search(RecallQuery(text="chocolate", k=10))

    for i, r in enumerate(results):
        assert r.hit.rank == i + 1


# ---------------------------------------------------------------------------
# 4. ClaimGraphChannel.search
# ---------------------------------------------------------------------------


async def test_claim_graph_search_returns_claims_for_anchor() -> None:
    kg = InMemoryEntityKG()
    c1 = _make_claim("Alice", "knows", "Python")
    await kg.write_claim(c1, evidence=_ev())

    ch = ClaimGraphChannel(kg)
    results = await ch.search(RecallQuery(anchor_entities=("Alice",), k=10))

    assert len(results) == 1
    assert results[0].key == f"claim:{c1.id}"
    assert results[0].hit.channel == "graph.claims"
    assert results[0].hit.raw_score is None
    assert results[0].hit.rank == 1


async def test_claim_graph_search_no_anchor_match_returns_empty() -> None:
    kg = InMemoryEntityKG()
    c1 = _make_claim("Alice", "knows", "Python")
    await kg.write_claim(c1, evidence=_ev())

    ch = ClaimGraphChannel(kg)
    results = await ch.search(RecallQuery(anchor_entities=("Bob",), k=10))

    assert list(results) == []


# ---------------------------------------------------------------------------
# 5. ClaimGraphChannel multi-anchor dedup
# ---------------------------------------------------------------------------


async def test_claim_graph_multi_anchor_dedup() -> None:
    """Claim C3 about both A and B must appear exactly once when both anchors are queried."""
    kg = InMemoryEntityKG()
    # C1: only about A
    c1 = _make_claim("A", "has", "property-x")
    # C2: only about B
    c2 = _make_claim("B", "has", "property-y")
    # C3: about A (subject) and B (object_entity)
    c3_id = claim_id_for("A", "relates_to", "B")
    c3 = Claim(
        id=c3_id,
        subject="A",
        predicate="relates_to",
        payload="B",
        object_entity="B",
        epistemic_type="inference",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    await kg.write_claim(c1, evidence=_ev("src-c1"))
    await kg.write_claim(c2, evidence=_ev("src-c2"))
    await kg.write_claim(c3, evidence=_ev("src-c3"))

    ch = ClaimGraphChannel(kg)
    results = await ch.search(RecallQuery(anchor_entities=("A", "B"), k=20))

    keys = [r.key for r in results]
    # C3 must appear exactly once
    assert keys.count(f"claim:{c3.id}") == 1
    # C1 and C2 should each appear once too
    assert f"claim:{c1.id}" in keys
    assert f"claim:{c2.id}" in keys


# ---------------------------------------------------------------------------
# 6. EpisodeRecencyChannel.search
# ---------------------------------------------------------------------------


async def test_episode_recency_returns_newest_first() -> None:
    store = InMemoryEpisodeStore()
    ep0 = _episode("sess-1", step_index=0, occurred_at=_T0)
    ep1 = _episode("sess-1", step_index=1, occurred_at=_T1)
    await store.project_episodes("proj", [ep0, ep1], _cursor())

    ch = EpisodeRecencyChannel(store)
    results = await ch.search(RecallQuery(session_id="sess-1", k=10))

    assert len(results) == 2
    assert results[0].item == ep1
    assert results[1].item == ep0


async def test_episode_recency_hit_fields() -> None:
    store = InMemoryEpisodeStore()
    ep = _episode("sess-2", step_index=0)
    await store.project_episodes("proj", [ep], _cursor())

    ch = EpisodeRecencyChannel(store)
    results = await ch.search(RecallQuery(session_id="sess-2", k=10))

    assert len(results) == 1
    r = results[0]
    assert r.kind == "episode"
    assert r.hit.channel == "temporal.episodes"
    assert r.hit.raw_score is None
    assert r.hit.rank == 1


# ---------------------------------------------------------------------------
# 7. LatentDenseChannel.search
# ---------------------------------------------------------------------------


async def test_latent_dense_search_returns_results() -> None:
    store = InMemoryLatentStore()
    rec1 = LatentRecord(id="lat-1", embedding=_EMBED_A, payload={"text": "hello world"})
    rec2 = LatentRecord(id="lat-2", embedding=_EMBED_B, payload={"text": "goodbye"})
    await store.put(rec1)
    await store.put(rec2)

    ch = LatentDenseChannel(store)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10))

    assert len(results) == 2
    first = results[0]
    assert first.kind == "latent"
    assert first.key == "latent:lat-1"
    assert first.hit.channel == "dense.latent"
    assert first.hit.raw_score is not None
    assert first.hit.raw_score > 0.0


async def test_latent_dense_search_ranks_1_based() -> None:
    store = InMemoryLatentStore()
    await store.put(LatentRecord(id="lat-a", embedding=_EMBED_A))
    await store.put(LatentRecord(id="lat-b", embedding=_EMBED_B))

    ch = LatentDenseChannel(store)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10))

    for i, r in enumerate(results):
        assert r.hit.rank == i + 1


async def test_latent_dense_no_record_use() -> None:
    """LatentDenseChannel must NOT call record_use (that is Pod 2.6's job)."""
    store = InMemoryLatentStore()
    await store.put(LatentRecord(id="lat-x", embedding=_EMBED_A))

    ch = LatentDenseChannel(store)
    await ch.search(RecallQuery(embedding=_EMBED_A, k=5))

    # use_count should remain 0 after a search
    matches_after = await store.search(_EMBED_A, k=1)
    assert len(matches_after) == 1
    assert matches_after[0].use_count == 0


# ---------------------------------------------------------------------------
# 8. Validity post-filter
# ---------------------------------------------------------------------------


async def test_dense_validity_filter_removes_expired() -> None:
    """include_invalidated=False (default) must drop claims with valid_to in the past."""
    kg = InMemoryEntityKG()
    active = _make_claim("alice", "is", "active", embedding=_EMBED_A)
    expired = _make_claim("bob", "was", "active", embedding=_EMBED_B, valid_to=_T_PAST)
    await kg.write_claim(active, evidence=_ev())
    await kg.write_claim(expired, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    # Query at _T0 (past valid_to for expired claim)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10, as_of=_T0))

    keys = [r.key for r in results]
    assert f"claim:{active.id}" in keys
    assert f"claim:{expired.id}" not in keys


async def test_dense_validity_filter_include_invalidated() -> None:
    """include_invalidated=True must keep expired claims."""
    kg = InMemoryEntityKG()
    expired = _make_claim("bob", "was", "active", embedding=_EMBED_A, valid_to=_T_PAST)
    await kg.write_claim(expired, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0, include_invalidated=True)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10, as_of=_T0))

    keys = [r.key for r in results]
    assert f"claim:{expired.id}" in keys


async def test_text_validity_filter_removes_expired() -> None:
    kg = InMemoryEntityKG()
    active = _make_claim("alice", "prefers", "chocolate")
    expired = _make_claim("bob", "preferred", "chocolate", valid_to=_T_PAST)
    await kg.write_claim(active, evidence=_ev())
    await kg.write_claim(expired, evidence=_ev())

    ch = ClaimTextChannel(kg)
    results = await ch.search(RecallQuery(text="chocolate", k=10, as_of=_T0))

    keys = [r.key for r in results]
    assert f"claim:{active.id}" in keys
    assert f"claim:{expired.id}" not in keys


async def test_text_validity_filter_include_invalidated_keeps_future_valid_to() -> None:
    """include_invalidated=True skips the channel-level post-filter.

    For ClaimTextChannel the substrate (claims_full_text) already applies its own as_of filter,
    so include_invalidated only affects the channel's secondary pass. A claim with valid_to set
    to a future time (relative to as_of) should always be returned regardless of the flag.
    """
    kg = InMemoryEntityKG()
    # valid_to is AFTER _T0, so the substrate does NOT filter it out at as_of=_T0.
    future_expiry = _make_claim("bob", "preferred", "chocolate", valid_to=_T1)
    await kg.write_claim(future_expiry, evidence=_ev())

    ch = ClaimTextChannel(kg, include_invalidated=True)
    results = await ch.search(RecallQuery(text="chocolate", k=10, as_of=_T0))

    keys = [r.key for r in results]
    assert f"claim:{future_expiry.id}" in keys


# ---------------------------------------------------------------------------
# 9. Key namespacing
# ---------------------------------------------------------------------------


async def test_key_namespacing_claim() -> None:
    kg = InMemoryEntityKG()
    c = _make_claim("X", "is", "Y", embedding=_EMBED_A)
    await kg.write_claim(c, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=1))

    assert results[0].key == f"claim:{c.id}"


async def test_key_namespacing_episode() -> None:
    store = InMemoryEpisodeStore()
    ep = _episode("sess-ns", step_index=0)
    await store.project_episodes("proj", [ep], _cursor())

    ch = EpisodeRecencyChannel(store)
    results = await ch.search(RecallQuery(session_id="sess-ns", k=5))

    assert results[0].key == f"episode:{ep.episode_id}"


async def test_key_namespacing_latent() -> None:
    store = InMemoryLatentStore()
    rec = LatentRecord(id="latent-ns-1", embedding=_EMBED_A)
    await store.put(rec)

    ch = LatentDenseChannel(store)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=1))

    assert results[0].key == "latent:latent-ns-1"


# ---------------------------------------------------------------------------
# 10. Scope filter passthrough (ClaimDenseChannel)
# ---------------------------------------------------------------------------


async def test_scope_filter_passthrough() -> None:
    """ClaimDenseChannel with scope='agent' returns only the agent-scope claim."""
    kg = InMemoryEntityKG()
    agent_claim = _make_claim("Paris", "is", "beautiful", embedding=_EMBED_A, scope="agent")
    world_claim = _make_claim("Paris", "is", "beautiful", embedding=_EMBED_A, scope="world")
    await kg.write_claim(agent_claim, evidence=_ev())
    await kg.write_claim(world_claim, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10, scope="agent"))

    keys = [r.key for r in results]
    assert f"claim:{agent_claim.id}" in keys
    assert f"claim:{world_claim.id}" not in keys


async def test_scope_filter_world_returns_only_world() -> None:
    kg = InMemoryEntityKG()
    agent_claim = _make_claim("Paris", "is", "beautiful", embedding=_EMBED_A, scope="agent")
    world_claim = _make_claim("Paris", "is", "beautiful", embedding=_EMBED_A, scope="world")
    await kg.write_claim(agent_claim, evidence=_ev())
    await kg.write_claim(world_claim, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=10, scope="world"))

    keys = [r.key for r in results]
    assert f"claim:{world_claim.id}" in keys
    assert f"claim:{agent_claim.id}" not in keys


# ---------------------------------------------------------------------------
# 11. can_serve False means search is never called (S8 lesion observable state)
# ---------------------------------------------------------------------------


def test_can_serve_false_claim_dense_when_no_embedding() -> None:
    """When can_serve returns False the stack never calls search — the state is 'skipped'."""
    kg = InMemoryEntityKG()
    ch = ClaimDenseChannel(kg)
    q = RecallQuery()  # no embedding
    assert not ch.can_serve(q)


def test_can_serve_false_claim_text_when_no_text() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimTextChannel(kg)
    q = RecallQuery()  # no text
    assert not ch.can_serve(q)


def test_can_serve_false_claim_graph_when_no_anchors() -> None:
    kg = InMemoryEntityKG()
    ch = ClaimGraphChannel(kg)
    q = RecallQuery()  # no anchors
    assert not ch.can_serve(q)


def test_can_serve_false_episode_when_no_session() -> None:
    store = InMemoryEpisodeStore()
    ch = EpisodeRecencyChannel(store)
    q = RecallQuery()  # no session_id
    assert not ch.can_serve(q)


def test_can_serve_false_latent_when_no_embedding() -> None:
    store = InMemoryLatentStore()
    ch = LatentDenseChannel(store)
    q = RecallQuery()  # no embedding
    assert not ch.can_serve(q)


# ---------------------------------------------------------------------------
# 12. can_serve True + no results returns [] (distinct from 'skipped')
# ---------------------------------------------------------------------------


async def test_claim_dense_can_serve_but_no_results_returns_empty() -> None:
    kg = InMemoryEntityKG()  # empty — no claims
    ch = ClaimDenseChannel(kg, min_score=0.0)
    q = RecallQuery(embedding=_EMBED_A)
    assert ch.can_serve(q)
    results = await ch.search(q)
    assert list(results) == []


async def test_episode_recency_can_serve_but_no_results_returns_empty() -> None:
    store = InMemoryEpisodeStore()  # empty
    ch = EpisodeRecencyChannel(store)
    q = RecallQuery(session_id="no-such-session")
    assert ch.can_serve(q)
    results = await ch.search(q)
    assert list(results) == []


async def test_latent_dense_can_serve_but_no_results_returns_empty() -> None:
    store = InMemoryLatentStore()  # empty
    ch = LatentDenseChannel(store)
    q = RecallQuery(embedding=_EMBED_A)
    assert ch.can_serve(q)
    results = await ch.search(q)
    assert list(results) == []


# ---------------------------------------------------------------------------
# 13. Rendering round-trip — text field is non-empty for each channel kind
# ---------------------------------------------------------------------------


async def test_claim_dense_text_non_empty() -> None:
    kg = InMemoryEntityKG()
    c = _make_claim("alice", "likes", "chocolate", embedding=_EMBED_A)
    await kg.write_claim(c, evidence=_ev())

    ch = ClaimDenseChannel(kg, min_score=0.0)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=1))

    assert len(results) == 1
    assert results[0].text != ""
    assert "alice" in results[0].text


async def test_episode_recency_text_non_empty() -> None:
    store = InMemoryEpisodeStore()
    ep = _episode("sess-render", step_index=0, content="hello from user")
    await store.project_episodes("proj", [ep], _cursor())

    ch = EpisodeRecencyChannel(store)
    results = await ch.search(RecallQuery(session_id="sess-render", k=5))

    assert len(results) == 1
    assert "hello from user" in results[0].text


async def test_latent_dense_text_from_payload_text_field() -> None:
    store = InMemoryLatentStore()
    rec = LatentRecord(id="lat-render", embedding=_EMBED_A, payload={"text": "payload content"})
    await store.put(rec)

    ch = LatentDenseChannel(store)
    results = await ch.search(RecallQuery(embedding=_EMBED_A, k=1))

    assert len(results) == 1
    assert results[0].text == "payload content"
