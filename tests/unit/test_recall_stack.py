"""Unit tests for RecallStack (Pod 2.5 Stream F, CANON S1/S5/S8/S9).

All tests are pure-Python: no database, no model, no network.
asyncio_mode = "auto" (pyproject.toml), so no @pytest.mark.asyncio needed.

Invariants verified:
  S1  — RecallStack itself never calls a model (Reranker is the caller's choice).
  S5  — every FusedResult in RecallOutcome.results has len(hits) >= 1.
  S8  — a failing channel degrades to 'failed' status; the stack continues.
  S9  — assert_rerank_subset prevents injected/mutated results from propagating.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.fusion import fuse
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import FusedResult, RecallResult
from cogworx.recall.stack import RecallOutcome, RecallStack, default_recall_stack
from cogworx.substrate.episodes import Episode
from cogworx.substrate.journal import ProjectionCursor
from cogworx.substrate.latent import LatentRecord
from cogworx.testing.doubles import InMemoryEntityKG, InMemoryEpisodeStore, InMemoryLatentStore

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)

_EMBED_A: tuple[float, ...] = (1.0, 0.0, 0.0)
_EMBED_B: tuple[float, ...] = (0.0, 1.0, 0.0)
_EMBED_C: tuple[float, ...] = (0.0, 0.0, 1.0)


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _ev(source_id: str = "src-a") -> EvidenceEvent:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source_id,
        source_authority=1.0,
        recorded_at=_T0,
    )


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    embedding: tuple[float, ...] | None = None,
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
        valid_to=None,
        ingest_time=_T0,
        created_by="test",
        embedding=embedding,
        scope=scope,
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
# Seeded fixture factories
# ---------------------------------------------------------------------------


async def _seeded_entity_kg() -> InMemoryEntityKG:
    """3-5 claims with distinct subjects/predicates and varied embeddings."""
    kg = InMemoryEntityKG()
    claims_evs = [
        (_make_claim("alice", "likes", "chocolate", embedding=_EMBED_A), _ev("src-1")),
        (_make_claim("bob", "knows", "alice", embedding=_EMBED_B), _ev("src-2")),
        (_make_claim("carol", "works_at", "acme", embedding=_EMBED_C), _ev("src-3")),
        (_make_claim("alice", "works_at", "acme", embedding=_EMBED_A), _ev("src-4")),
        (_make_claim("bob", "likes", "jazz", embedding=_EMBED_B), _ev("src-5")),
    ]
    for claim, ev in claims_evs:
        await kg.write_claim(claim, evidence=ev)
    return kg


async def _seeded_episode_store() -> InMemoryEpisodeStore:
    """2-3 episodes in session 'sess-1'."""
    store = InMemoryEpisodeStore()
    eps = [
        _episode("sess-1", step_index=0, turn_index=0, content="Hello there"),
        _episode("sess-1", step_index=0, turn_index=1, role="assistant", content="Hi!"),
        _episode("sess-1", step_index=1, turn_index=0, content="What's next?"),
    ]
    await store.project_episodes("consumer", eps, _cursor(1))
    return store


async def _seeded_latent_store() -> InMemoryLatentStore:
    """2-3 latent records."""
    store = InMemoryLatentStore()
    for i, emb in enumerate([_EMBED_A, _EMBED_B, _EMBED_C]):
        await store.put(LatentRecord(id=f"lat-{i}", embedding=emb, payload={"text": f"text-{i}"}))
    return store


# ---------------------------------------------------------------------------
# 1. Duplicate channel names rejected at construction
# ---------------------------------------------------------------------------


def test_duplicate_channel_names_raise_at_construction() -> None:
    class _FixedNameChannel:
        name: str = "dense.claims"

        def can_serve(self, query: RecallQuery) -> bool:
            return False

        async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
            return []

    ch1 = _FixedNameChannel()
    ch2 = _FixedNameChannel()
    with pytest.raises(ValueError, match="Duplicate channel names"):
        RecallStack([ch1, ch2])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# 2. Full 5-channel recall
# ---------------------------------------------------------------------------


async def test_full_5_channel_recall() -> None:
    kg = await _seeded_entity_kg()
    ep_store = await _seeded_episode_store()
    lat_store = await _seeded_latent_store()

    stack = default_recall_stack(
        entity_kg=kg, episode_store=ep_store, latent_store=lat_store
    )
    query = RecallQuery(
        text="alice likes",
        embedding=_EMBED_A,
        anchor_entities=("alice",),
        session_id="sess-1",
    )
    outcome = await stack.recall(query)

    assert isinstance(outcome, RecallOutcome)
    assert len(outcome.results) > 0
    assert len(outcome.channel_status) == 5


# ---------------------------------------------------------------------------
# 3. All channel statuses are "ok", "skipped", or "failed"
# ---------------------------------------------------------------------------


async def test_channel_status_values_are_valid() -> None:
    kg = await _seeded_entity_kg()
    ep_store = await _seeded_episode_store()
    lat_store = await _seeded_latent_store()

    stack = default_recall_stack(
        entity_kg=kg, episode_store=ep_store, latent_store=lat_store
    )
    query = RecallQuery(
        text="alice", embedding=_EMBED_A, anchor_entities=("alice",), session_id="sess-1"
    )
    outcome = await stack.recall(query)

    valid_states = {"ok", "skipped", "failed"}
    for cs in outcome.channel_status:
        assert cs.state in valid_states, f"Unexpected state {cs.state!r} on {cs.channel!r}"


# ---------------------------------------------------------------------------
# 4. S5 invariant: every FusedResult has >= 1 ChannelHit
# ---------------------------------------------------------------------------


async def test_s5_every_fused_result_has_at_least_one_hit() -> None:
    kg = await _seeded_entity_kg()
    ep_store = await _seeded_episode_store()
    lat_store = await _seeded_latent_store()

    stack = default_recall_stack(
        entity_kg=kg, episode_store=ep_store, latent_store=lat_store
    )
    query = RecallQuery(
        text="alice", embedding=_EMBED_A, anchor_entities=("alice",), session_id="sess-1"
    )
    outcome = await stack.recall(query)

    for r in outcome.results:
        assert len(r.hits) >= 1, f"Result {r.key!r} has no ChannelHit (S5 violation)"


# ---------------------------------------------------------------------------
# 5. S8 lesion — missing episode store
# ---------------------------------------------------------------------------


async def test_s8_lesion_missing_episode_store() -> None:
    kg = await _seeded_entity_kg()
    lat_store = await _seeded_latent_store()

    stack = default_recall_stack(entity_kg=kg, episode_store=None, latent_store=lat_store)
    query = RecallQuery(text="alice", embedding=_EMBED_A, anchor_entities=("alice",))
    outcome = await stack.recall(query)

    channel_names = {cs.channel for cs in outcome.channel_status}
    assert "temporal.episodes" not in channel_names


# ---------------------------------------------------------------------------
# 6. S8 lesion — missing latent store
# ---------------------------------------------------------------------------


async def test_s8_lesion_missing_latent_store() -> None:
    kg = await _seeded_entity_kg()
    ep_store = await _seeded_episode_store()

    stack = default_recall_stack(entity_kg=kg, episode_store=ep_store, latent_store=None)
    query = RecallQuery(text="alice", embedding=_EMBED_A, session_id="sess-1")
    outcome = await stack.recall(query)

    channel_names = {cs.channel for cs in outcome.channel_status}
    assert "dense.latent" not in channel_names


# ---------------------------------------------------------------------------
# 7. S8 lesion — channel raises
# ---------------------------------------------------------------------------


async def test_s8_failing_channel_degrades_gracefully() -> None:
    """A channel that always raises must not crash the stack."""

    class _BrokenChannel:
        name: str = "broken.channel"

        def can_serve(self, query: RecallQuery) -> bool:
            return True

        async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
            raise RuntimeError("substrate unavailable")

    kg = await _seeded_entity_kg()
    broken = _BrokenChannel()

    from cogworx.recall.channels import ClaimDenseChannel, ClaimTextChannel

    # A stack with the broken channel plus the two text/dense channels (which will return results).
    stack = RecallStack(
        [
            broken,  # type: ignore[list-item]
            ClaimTextChannel(kg),
            ClaimDenseChannel(kg),
        ]
    )

    query = RecallQuery(text="alice", embedding=_EMBED_A)
    outcome = await stack.recall(query)

    # The broken channel recorded as failed.
    broken_statuses = [cs for cs in outcome.channel_status if cs.channel == "broken.channel"]
    assert len(broken_statuses) == 1
    assert broken_statuses[0].state == "failed"

    # Other channels still contributed results.
    assert len(outcome.results) > 0


# ---------------------------------------------------------------------------
# 8. Skipped channel when embedding=None
# ---------------------------------------------------------------------------


async def test_skipped_channel_when_no_embedding() -> None:
    """Dense channels skip; BM25 and graph still run when they have their fields."""
    kg = await _seeded_entity_kg()
    ep_store = await _seeded_episode_store()
    lat_store = await _seeded_latent_store()

    stack = default_recall_stack(
        entity_kg=kg, episode_store=ep_store, latent_store=lat_store
    )
    # No embedding → dense.claims and dense.latent should be skipped.
    query = RecallQuery(text="alice", anchor_entities=("alice",), session_id="sess-1")
    outcome = await stack.recall(query)

    dense_statuses = [
        cs for cs in outcome.channel_status if cs.channel in ("dense.claims", "dense.latent")
    ]
    for cs in dense_statuses:
        assert cs.state == "skipped", f"{cs.channel!r} should be skipped, got {cs.state!r}"

    # Non-dense channels that can_serve=True must have run.
    non_dense_runnable = [
        cs
        for cs in outcome.channel_status
        if cs.channel not in ("dense.claims", "dense.latent")
    ]
    for cs in non_dense_runnable:
        assert cs.state in {"ok", "failed"}, (
            f"{cs.channel!r} should be ok or failed, got {cs.state!r}"
        )


# ---------------------------------------------------------------------------
# 9. NoopReranker default — output order matches fuse() output
# ---------------------------------------------------------------------------


async def test_noop_reranker_output_matches_fuse_order() -> None:
    kg = await _seeded_entity_kg()

    stack = default_recall_stack(entity_kg=kg)
    query = RecallQuery(text="alice", embedding=_EMBED_A, anchor_entities=("alice",))
    outcome = await stack.recall(query)

    # Recompute fuse() independently for the same query by running channels individually.
    from cogworx.recall.channels import ClaimDenseChannel, ClaimGraphChannel, ClaimTextChannel

    ch_dense = ClaimDenseChannel(kg)
    ch_text = ClaimTextChannel(kg)
    ch_graph = ClaimGraphChannel(kg)

    dense_res = list(await ch_dense.search(query)) if ch_dense.can_serve(query) else []
    text_res = list(await ch_text.search(query)) if ch_text.can_serve(query) else []
    graph_res = list(await ch_graph.search(query)) if ch_graph.can_serve(query) else []

    expected = fuse(
        {"dense.claims": dense_res, "bm25.claims": text_res, "graph.claims": graph_res}
    )

    assert [r.key for r in outcome.results] == [r.key for r in expected]


# ---------------------------------------------------------------------------
# 10. Custom reranker (valid) — reverses output
# ---------------------------------------------------------------------------


async def test_custom_reranker_reversal_is_accepted() -> None:
    class _ReverseReranker:
        model_bearing: bool = False

        async def rerank(
            self, query: RecallQuery, results: Sequence[FusedResult]
        ) -> Sequence[FusedResult]:
            return list(reversed(results))

    kg = await _seeded_entity_kg()
    stack = default_recall_stack(entity_kg=kg, reranker=_ReverseReranker())  # type: ignore[arg-type]
    query = RecallQuery(text="alice", embedding=_EMBED_A, anchor_entities=("alice",))
    outcome = await stack.recall(query)

    # Reconstruct the expected reversed order.
    from cogworx.recall.channels import ClaimDenseChannel, ClaimGraphChannel, ClaimTextChannel

    ch_dense = ClaimDenseChannel(kg)
    ch_text = ClaimTextChannel(kg)
    ch_graph = ClaimGraphChannel(kg)

    dense_res = list(await ch_dense.search(query)) if ch_dense.can_serve(query) else []
    text_res = list(await ch_text.search(query)) if ch_text.can_serve(query) else []
    graph_res = list(await ch_graph.search(query)) if ch_graph.can_serve(query) else []

    fused = fuse(
        {"dense.claims": dense_res, "bm25.claims": text_res, "graph.claims": graph_res}
    )
    expected_keys = [r.key for r in reversed(fused)]

    assert [r.key for r in outcome.results] == expected_keys


# ---------------------------------------------------------------------------
# 11. Malicious reranker — injected key raises ValueError
# ---------------------------------------------------------------------------


async def test_malicious_reranker_injecting_key_raises() -> None:
    """A reranker that appends a foreign FusedResult must trigger assert_rerank_subset."""

    class _InjectionReranker:
        model_bearing: bool = False

        async def rerank(
            self, query: RecallQuery, results: Sequence[FusedResult]
        ) -> Sequence[FusedResult]:
            from cogworx.recall.results import ChannelHit, FusedResult

            fake = FusedResult(
                key="claim:totally-foreign-key-that-was-never-in-fuse",
                kind="claim",
                item=results[0].item,
                text="injected",
                hits=(ChannelHit(channel="evil", rank=1, raw_score=None),),
                fused_score=9999.0,
                fused_rank=1,
            )
            return [*list(results), fake]

    kg = await _seeded_entity_kg()
    stack = default_recall_stack(entity_kg=kg, reranker=_InjectionReranker())  # type: ignore[arg-type]
    query = RecallQuery(text="alice", embedding=_EMBED_A, anchor_entities=("alice",))

    with pytest.raises(ValueError, match="not present in original results"):
        await stack.recall(query)


# ---------------------------------------------------------------------------
# 12. default_recall_stack factory — entity_kg only (3 channels)
# ---------------------------------------------------------------------------


def test_default_stack_entity_kg_only_has_3_channels() -> None:
    kg = InMemoryEntityKG()
    stack = default_recall_stack(entity_kg=kg)

    # The stack's internal channel list should contain exactly the 3 claim channels.
    assert len(stack._channels) == 3
    channel_names = {ch.name for ch in stack._channels}
    assert channel_names == {"dense.claims", "bm25.claims", "graph.claims"}


# ---------------------------------------------------------------------------
# 13. error field on failed ChannelStatus is non-empty
# ---------------------------------------------------------------------------


async def test_failed_channel_status_error_is_nonempty() -> None:
    class _AlwaysFails:
        name: str = "exploding.channel"

        def can_serve(self, query: RecallQuery) -> bool:
            return True

        async def search(self, query: RecallQuery) -> Sequence[RecallResult]:
            raise RuntimeError("boom!")

    stack = RecallStack([_AlwaysFails()])  # type: ignore[list-item]
    outcome = await stack.recall(RecallQuery())

    assert len(outcome.channel_status) == 1
    cs = outcome.channel_status[0]
    assert cs.state == "failed"
    assert cs.error is not None and len(cs.error) > 0
    assert "RuntimeError" in cs.error
