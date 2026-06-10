"""Unit tests for cogworx.recall contracts (Pod 2.5 Stream A).

All tests are pure-Python: no database, no model, no network. Doubles only.
asyncio_mode = "auto" (pyproject.toml), so no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall import (
    ChannelHit,
    FusedResult,
    NoopReranker,
    RecallQuery,
)
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode
from cogworx.testing.doubles import InMemoryEntityKG, InMemoryEpisodeStore

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _make_claim(subject: str, predicate: str, payload: str) -> Claim:
    cid = claim_id_for(subject, predicate, payload)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )


def _ev() -> EvidenceEvent:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id="src-test",
        source_authority=1.0,
        recorded_at=_T0,
    )


def _episode(
    session_id: str,
    step_index: int,
    turn_index: int,
    occurred_at: datetime = _T0,
) -> Episode:
    return Episode(
        episode_id=f"run-01:{step_index}:{turn_index}",
        run_id="run-01",
        step_index=step_index,
        turn_index=turn_index,
        session_id=session_id,
        role="user",
        content="hello",
        kind="conversational",
        occurred_at=occurred_at,
    )


def _fused_result(key: str = "claim:abc") -> FusedResult:
    claim = _make_claim("alice", "likes", "chocolate")
    sc = ScoredClaim(
        claim=claim,
        confidence=__import__(
            "cogworx.knowledge.confidence", fromlist=["claim_confidence"]
        ).claim_confidence([]),
        lineage_min_confidence=0.5,
    )
    return FusedResult(
        key=key,
        kind="claim",
        item=sc,
        text="alice likes chocolate",
        hits=(ChannelHit(channel="bm25", rank=1, raw_score=1.5),),
        fused_score=0.9,
        fused_rank=1,
    )


# ---------------------------------------------------------------------------
# 1. RecallQuery is frozen
# ---------------------------------------------------------------------------


def test_recall_query_frozen() -> None:
    q = RecallQuery(text="hello")
    with pytest.raises((ValidationError, TypeError)):
        q.text = "world"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 2. NoopReranker.model_bearing is False
# ---------------------------------------------------------------------------


def test_noop_reranker_model_bearing_false() -> None:
    assert NoopReranker.model_bearing is False
    assert NoopReranker().model_bearing is False


# ---------------------------------------------------------------------------
# 3. NoopReranker.rerank returns same items in same order
# ---------------------------------------------------------------------------


async def test_noop_reranker_preserves_order() -> None:
    r1 = _fused_result("claim:aaa")
    r2 = _fused_result("claim:bbb")
    reranker = NoopReranker()
    q = RecallQuery(text="test")
    result = await reranker.rerank(q, [r1, r2])
    assert list(result) == [r1, r2]


# ---------------------------------------------------------------------------
# 4. ScoredClaim can be constructed without text_score (back-compat)
# ---------------------------------------------------------------------------


def test_scored_claim_text_score_optional() -> None:
    from cogworx.knowledge.confidence import claim_confidence

    claim = _make_claim("bob", "is", "human")
    sc = ScoredClaim(
        claim=claim,
        confidence=claim_confidence([]),
        lineage_min_confidence=0.5,
    )
    assert sc.text_score is None


# ---------------------------------------------------------------------------
# 5. ScoredClaim with explicit text_score round-trips
# ---------------------------------------------------------------------------


def test_scored_claim_text_score_roundtrip() -> None:
    from cogworx.knowledge.confidence import claim_confidence

    claim = _make_claim("bob", "is", "human")
    sc = ScoredClaim(
        claim=claim,
        confidence=claim_confidence([]),
        lineage_min_confidence=0.5,
        text_score=0.75,
    )
    assert sc.text_score == 0.75


# ---------------------------------------------------------------------------
# 6. InMemoryEntityKG.claims_full_text
# ---------------------------------------------------------------------------


async def test_claims_full_text_exact_term_beats_nonmatch() -> None:
    kg = InMemoryEntityKG()
    matching = _make_claim("alice", "likes", "chocolate cake")
    unrelated = _make_claim("bob", "is", "engineer")
    await kg.write_claim(matching, evidence=_ev())
    await kg.write_claim(unrelated, evidence=_ev())

    results = await kg.claims_full_text("chocolate")
    assert len(results) >= 1
    assert results[0].claim.id == matching.id
    assert results[0].text_score is not None
    assert results[0].text_score > 0.0


async def test_claims_full_text_empty_query_returns_empty() -> None:
    kg = InMemoryEntityKG()
    claim = _make_claim("alice", "likes", "chocolate")
    await kg.write_claim(claim, evidence=_ev())

    results = await kg.claims_full_text("")
    assert list(results) == []


async def test_claims_full_text_whitespace_query_returns_empty() -> None:
    kg = InMemoryEntityKG()
    claim = _make_claim("alice", "likes", "chocolate")
    await kg.write_claim(claim, evidence=_ev())

    results = await kg.claims_full_text("   ")
    assert list(results) == []


async def test_claims_full_text_result_has_text_score_set() -> None:
    kg = InMemoryEntityKG()
    claim = _make_claim("eve", "studies", "machine learning")
    await kg.write_claim(claim, evidence=_ev())

    results = await kg.claims_full_text("machine")
    assert len(results) == 1
    assert results[0].text_score is not None
    assert results[0].similarity is None


# ---------------------------------------------------------------------------
# 7. InMemoryEpisodeStore.recent_episodes
# ---------------------------------------------------------------------------


async def test_recent_episodes_newest_first() -> None:
    store = InMemoryEpisodeStore()
    ep0 = _episode("sess-1", step_index=0, turn_index=0, occurred_at=_T0)
    ep1 = _episode("sess-1", step_index=1, turn_index=0, occurred_at=_T1)
    ep2 = _episode("sess-1", step_index=2, turn_index=0, occurred_at=_T2)
    from cogworx.substrate.journal import ProjectionCursor

    cursor = ProjectionCursor(commit_ordinal=1, run_id="run-01", step_index=2)
    await store.project_episodes("proj", [ep0, ep1, ep2], cursor)

    results = await store.recent_episodes("sess-1")
    assert len(results) == 3
    # Newest-first: (step_index DESC, turn_index DESC)
    assert results[0].step_index == 2
    assert results[1].step_index == 1
    assert results[2].step_index == 0


async def test_recent_episodes_before_filter_exclusive() -> None:
    store = InMemoryEpisodeStore()
    ep0 = _episode("sess-2", step_index=0, turn_index=0, occurred_at=_T0)
    ep1 = _episode("sess-2", step_index=1, turn_index=0, occurred_at=_T1)
    ep2 = _episode("sess-2", step_index=2, turn_index=0, occurred_at=_T2)
    from cogworx.substrate.journal import ProjectionCursor

    cursor = ProjectionCursor(commit_ordinal=3, run_id="run-01", step_index=2)
    await store.project_episodes("proj", [ep0, ep1, ep2], cursor)

    # before=_T2 → excludes ep2 (occurred_at == _T2 is NOT < _T2)
    results = await store.recent_episodes("sess-2", before=_T2)
    assert len(results) == 2
    assert all(r.step_index < 2 for r in results)


async def test_recent_episodes_no_match_returns_empty() -> None:
    store = InMemoryEpisodeStore()
    ep0 = _episode("sess-3", step_index=0, turn_index=0)
    from cogworx.substrate.journal import ProjectionCursor

    cursor = ProjectionCursor(commit_ordinal=1, run_id="run-01", step_index=0)
    await store.project_episodes("proj", [ep0], cursor)

    results = await store.recent_episodes("no-such-session")
    assert list(results) == []
