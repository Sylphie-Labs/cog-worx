"""Pod 3.1b CF-B — defeated-claim exclusion in the memory injection layer.

Tests confirm:
  1. A defeasibly-defeated claim (status set, valid_to UNSET — the exact gap) is excluded
     from the assembled context after inject(); defeated_excluded == 1.
  2. With include_defeated=True the defeated claim IS present (discriminability /
     negative control — proves the filter, not some other mechanism, is responsible).
  3. A fixture with no defeated claims yields defeated_excluded == 0 and output unchanged
     (golden regression).
  4. Zero model calls in the filter path (S9 — pure structural status check).

asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from cogworx.claims.provenance import Claim, Provenance
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import MemoryPolicy
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import ChannelHit, FusedResult
from cogworx.recall.stack import ChannelStatus, RecallOutcome, RecallStack
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC)


def _fixed_clock() -> datetime:
    return _EPOCH


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_provenance() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_EPOCH)


def _make_active_claim(subject: str, predicate: str, payload: str) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope="agent")
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=_make_provenance(),
        valid_from=_EPOCH,
        valid_to=None,  # NOT invalidated via valid_to
        ingest_time=_EPOCH,
        created_by="test",
        scope="agent",
        status="active",  # normal active claim
    )


def _make_defeated_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    defeated_by: str = "winner-claim-id",
) -> Claim:
    """Return a defeasibly-defeated claim with valid_to UNSET (the exact gap CF-B targets).

    The Pod 2.7 reconciler sets status="defeasibly-defeated" but does NOT set valid_to —
    so the dense-channel's _validity_filter does NOT exclude it; only the injector filter
    (Pod 3.1b CF-B) catches it.
    """
    cid = claim_id_for(subject, predicate, payload, scope="agent")
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=_make_provenance(),
        valid_from=_EPOCH,
        valid_to=None,  # UNSET — the exact gap: defeated but not temporally invalidated
        ingest_time=_EPOCH,
        created_by="test",
        scope="agent",
        status="defeasibly-defeated",
        defeated_by=defeated_by,
    )


def _make_scored(claim: Claim) -> ScoredClaim:
    return ScoredClaim(
        claim=claim,
        confidence=claim_confidence([]),
        lineage_min_confidence=1.0,
    )


def _make_fused(claim: Claim, *, rank: int = 1) -> FusedResult:
    scored = _make_scored(claim)
    hit = ChannelHit(channel="dense.claims", rank=rank, raw_score=0.9)
    return FusedResult(
        key=f"claim:{claim.id}",
        kind="claim",
        item=scored,
        text=f"{claim.subject} {claim.predicate} {claim.payload}",
        hits=(hit,),
        fused_score=1.0 / (60 + rank),
        fused_rank=rank,
    )


def _make_injector(fused_results: tuple[FusedResult, ...]) -> MemoryInjector:
    """Build a MemoryInjector backed by a mock RecallStack returning the given results."""
    outcome = RecallOutcome(
        results=fused_results,
        channel_status=(
            ChannelStatus(channel="dense.claims", state="ok", count=len(fused_results)),
        ),
    )
    stack_mock = MagicMock(spec=RecallStack)
    stack_mock.recall = AsyncMock(return_value=outcome)
    return MemoryInjector(stack_mock, clock=_fixed_clock)


# ---------------------------------------------------------------------------
# 1. Defeasibly-defeated claim is EXCLUDED by default
# ---------------------------------------------------------------------------


async def test_defeated_claim_excluded_from_context() -> None:
    """C2 (defeasibly-defeated, valid_to=None) must be absent from assembled chunks."""
    active = _make_active_claim("alice", "likes", "chocolate")
    defeated = _make_defeated_claim("alice", "likes", "vanilla")

    injector = _make_injector((_make_fused(active, rank=1), _make_fused(defeated, rank=2)))
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy())

    chunk_keys = {c.key for c in result.context.chunks}
    assert f"claim:{active.id}" in chunk_keys, "Active claim C1 must be present"
    assert f"claim:{defeated.id}" not in chunk_keys, "Defeated claim C2 must be absent"


async def test_defeated_excluded_count_is_one() -> None:
    """defeated_excluded must equal 1 when exactly one defeated claim is filtered out."""
    active = _make_active_claim("alice", "likes", "chocolate")
    defeated = _make_defeated_claim("alice", "likes", "vanilla")

    injector = _make_injector((_make_fused(active, rank=1), _make_fused(defeated, rank=2)))
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy())

    assert result.defeated_excluded == 1


# ---------------------------------------------------------------------------
# 2. include_defeated=True — defeated claim IS present (discriminability test)
# ---------------------------------------------------------------------------


async def test_defeated_claim_present_when_include_defeated_true() -> None:
    """Escape hatch: with include_defeated=True, C2 must appear in assembled chunks."""
    active = _make_active_claim("alice", "likes", "chocolate")
    defeated = _make_defeated_claim("alice", "likes", "vanilla")

    policy = MemoryPolicy(include_defeated=True)
    injector = _make_injector((_make_fused(active, rank=1), _make_fused(defeated, rank=2)))
    result = await injector.inject(RecallQuery(), policy=policy)

    chunk_keys = {c.key for c in result.context.chunks}
    assert f"claim:{active.id}" in chunk_keys, "Active claim C1 must be present"
    assert f"claim:{defeated.id}" in chunk_keys, (
        "Defeated claim C2 must be present with include_defeated=True"
    )


async def test_defeated_excluded_zero_when_include_defeated_true() -> None:
    """defeated_excluded must be 0 when the filter is skipped via include_defeated=True."""
    active = _make_active_claim("alice", "likes", "chocolate")
    defeated = _make_defeated_claim("alice", "likes", "vanilla")

    policy = MemoryPolicy(include_defeated=True)
    injector = _make_injector((_make_fused(active, rank=1), _make_fused(defeated, rank=2)))
    result = await injector.inject(RecallQuery(), policy=policy)

    assert result.defeated_excluded == 0


# ---------------------------------------------------------------------------
# 3. No defeated claims → defeated_excluded == 0, output unchanged (golden regression)
# ---------------------------------------------------------------------------


async def test_no_defeated_claims_excluded_count_zero() -> None:
    """When no defeated claims exist defeated_excluded must be 0."""
    c1 = _make_active_claim("bob", "knows", "python")
    c2 = _make_active_claim("bob", "prefers", "dark-mode")

    injector = _make_injector((_make_fused(c1, rank=1), _make_fused(c2, rank=2)))
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy())

    assert result.defeated_excluded == 0


async def test_no_defeated_claims_all_chunks_present() -> None:
    """When no claims are defeated all surfaced claims must appear in the assembled context."""
    c1 = _make_active_claim("bob", "knows", "python")
    c2 = _make_active_claim("bob", "prefers", "dark-mode")

    injector = _make_injector((_make_fused(c1, rank=1), _make_fused(c2, rank=2)))
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy())

    chunk_keys = {c.key for c in result.context.chunks}
    assert f"claim:{c1.id}" in chunk_keys
    assert f"claim:{c2.id}" in chunk_keys


async def test_empty_results_defeated_excluded_zero() -> None:
    """An empty recall result (no claims at all) must yield defeated_excluded == 0."""
    injector = _make_injector(())
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy())

    assert result.defeated_excluded == 0
    assert result.context.chunks == ()


# ---------------------------------------------------------------------------
# 4. Zero model calls in filter path (S9)
# ---------------------------------------------------------------------------


async def test_filter_makes_zero_model_calls() -> None:
    """The defeated-claim filter is a pure structural check — no model is consulted (S9).

    We wire the injector with model=None (the default); if the filter somehow invoked
    model-bearing code an AttributeError would be raised.  Passing the test confirms the
    path is model-free.
    """
    active = _make_active_claim("carol", "uses", "cog-worx")
    defeated = _make_defeated_claim("carol", "uses", "legacy-lib")

    # Explicitly pass model=None — no count_tokens, no model at all.
    outcome = RecallOutcome(
        results=(_make_fused(active, rank=1), _make_fused(defeated, rank=2)),
        channel_status=(ChannelStatus(channel="dense.claims", state="ok", count=2),),
    )
    stack_mock = MagicMock(spec=RecallStack)
    stack_mock.recall = AsyncMock(return_value=outcome)
    injector = MemoryInjector(stack_mock, model=None, clock=_fixed_clock)

    result = await injector.inject(RecallQuery(), policy=MemoryPolicy())

    # Structural assertion: filter ran, no model was needed
    assert result.defeated_excluded == 1
    chunk_keys = {c.key for c in result.context.chunks}
    assert f"claim:{defeated.id}" not in chunk_keys


# ---------------------------------------------------------------------------
# 5. Fix 4a — isinstance-based defeated-claim filter (Pod 3.1)
# ---------------------------------------------------------------------------


def _make_fused_with_kind(
    claim: Claim,
    *,
    rank: int = 1,
    kind: str = "episode",
) -> FusedResult:
    """Build a FusedResult with an explicit kind (may differ from 'claim')."""
    scored = _make_scored(claim)
    hit = ChannelHit(channel="dense.claims", rank=rank, raw_score=0.9)
    return FusedResult(
        key=f"{kind}:{claim.id}",
        kind=kind,
        item=scored,
        text=f"{claim.subject} {claim.predicate} {claim.payload}",
        hits=(hit,),
        fused_score=1.0 / (60 + rank),
        fused_rank=rank,
    )


async def test_fix4a_episode_kind_with_scored_claim_item_is_excluded() -> None:
    """Fix 4a: FusedResult(kind='episode', item=ScoredClaim(defeated)) is excluded.

    The old filter required kind=='claim'; Fix 4a keys off isinstance(r.item, ScoredClaim)
    so the mis-kind vector (kind='episode', item=ScoredClaim) is closed.
    """
    active = _make_active_claim("dave", "prefers", "coffeescript")
    defeated = _make_defeated_claim("dave", "prefers", "typescript")

    # Deliberately wrap the defeated claim under kind="episode" — the mis-kind vector.
    active_result = _make_fused(active, rank=1)
    defeated_episode_result = _make_fused_with_kind(defeated, rank=2, kind="episode")

    injector = _make_injector((active_result, defeated_episode_result))
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy(include_defeated=False))

    # Fix 4a: the defeated claim is excluded even though kind="episode".
    assert result.defeated_excluded == 1, (
        f"Fix 4a: expected defeated_excluded=1 for kind='episode' mis-kind vector, "
        f"got {result.defeated_excluded}"
    )
    chunk_keys = {c.key for c in result.context.chunks}
    defeated_in_chunks = (
        f"claim:{defeated.id}" in chunk_keys or f"episode:{defeated.id}" in chunk_keys
    )
    assert not defeated_in_chunks, (
        "Fix 4a: defeated claim with kind='episode' must NOT appear in assembled context"
    )
    # Active claim is still present.
    assert f"claim:{active.id}" in chunk_keys, (
        "Fix 4a: active claim must remain in assembled context"
    )


# ---------------------------------------------------------------------------
# 6. CF-3.1-Episode-echo — PINNING test (current behavior, do NOT change)
# ---------------------------------------------------------------------------
# KNOWN GAP: an Episode whose content echoes a now-defeated fact passes the
# defeated-claim filter because its item is an Episode, not a ScoredClaim.
# The filter keys off isinstance(r.item, ScoredClaim), so Episode items always
# pass through unchanged — there is no claim.status to inspect.
# Whether to suppress or annotate such episodes is deferred (ticket CF-3.1-Episode-echo).
# This test PINS current behavior: it must pass today, and any change to the filter
# that alters this behavior MUST update this comment and the ticket reference.


async def test_cf3_1_episode_echo_passes_filter_current_behavior() -> None:
    """CF-3.1-Episode-echo: an Episode echoing a defeated fact passes the filter (PINNED).

    Current behavior: Episode items have no claim.status, so isinstance(r.item, ScoredClaim)
    is False and the result passes through the defeated-claim filter unchanged.

    DO NOT change the filter behavior here — this is a deferred design question
    (ticket CF-3.1-Episode-echo).  This pin records what the code does TODAY so
    any future change is explicit and reviewed.
    """
    defeated_text = "eve believes flat-earth"

    episode = Episode(
        episode_id="run-x:3:0",
        run_id="run-x",
        step_index=3,
        turn_index=0,
        session_id="sess-x",
        role="assistant",
        content=defeated_text,  # echoes the defeated fact
        kind="conversation",
        occurred_at=_EPOCH,
    )

    episode_result = FusedResult(
        key="episode:run-x:3:0",
        kind="episode",
        item=episode,
        text=defeated_text,
        hits=(ChannelHit(channel="temporal.episodes", rank=1, raw_score=None),),
        fused_score=0.7,
        fused_rank=1,
    )

    injector = _make_injector((episode_result,))
    result = await injector.inject(RecallQuery(), policy=MemoryPolicy(include_defeated=False))

    # PINNED: Episode passes through — defeated_excluded remains 0.
    assert result.defeated_excluded == 0, (
        "CF-3.1-Episode-echo pin: Episode items are not ScoredClaim — they pass the filter. "
        "If this fails, the filter behavior changed; review ticket CF-3.1-Episode-echo."
    )
    # The episode text is present in the assembled context.
    chunk_texts = {c.text for c in result.context.chunks}
    assert defeated_text in chunk_texts, (
        "CF-3.1-Episode-echo pin: Episode content must appear in assembled context "
        "(current behavior — deferred decision on suppress-vs-annotate)."
    )
