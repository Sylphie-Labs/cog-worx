"""Unit tests for MemoryInjector and resolve_token_counter (Pod 2.6).

All tests are fully deterministic: no wall-clock, no random.
Uses RecallStack + InMemoryEntityKG / InMemoryEpisodeStore / AsyncMock LatentStore.
asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

from cogworx.injection.injector import MemoryInjector, resolve_token_counter
from cogworx.injection.policy import InjectedMemory, MemoryPolicy
from cogworx.recall.assembly import approx_tokens
from cogworx.recall.query import RecallQuery
from cogworx.recall.stack import RecallStack, default_recall_stack
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel

# ---------------------------------------------------------------------------
# Constants / helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_FIXED_CLOCK = lambda: _EPOCH  # noqa: E731  # deterministic wall-clock replacement


def _fixed_clock() -> datetime:
    return _EPOCH


def _make_stack_empty() -> RecallStack:
    """Return a 3-channel stack with an empty entity KG (returns no results)."""
    kg = InMemoryEntityKG()
    return default_recall_stack(entity_kg=kg)


def _make_minimal_latent_store() -> InMemoryLatentStore:
    return InMemoryLatentStore(clock=_fixed_clock)


# ---------------------------------------------------------------------------
# 1. resolve_token_counter — falls back to approx_tokens when model=None
# ---------------------------------------------------------------------------


def test_resolve_token_counter_model_none_returns_approx_tokens() -> None:
    counter = resolve_token_counter(None)
    text = "hello world"
    assert counter(text) == approx_tokens(text)


def test_resolve_token_counter_model_none_is_callable() -> None:
    counter = resolve_token_counter(None)
    assert callable(counter)


# ---------------------------------------------------------------------------
# 2. resolve_token_counter — uses model.count_tokens when available
# ---------------------------------------------------------------------------


def test_resolve_token_counter_uses_model_count_tokens() -> None:
    """ReplayModel exposes count_tokens — the resolver must return that bound method."""
    model = ReplayModel(token_counter=lambda s: 7)
    counter = resolve_token_counter(model)
    assert counter("anything") == 7


def test_resolve_token_counter_count_tokens_called_on_model() -> None:
    called: list[str] = []

    class _ModelWithCounter:
        def count_tokens(self, text: str) -> int:
            called.append(text)
            return 42

    counter = resolve_token_counter(_ModelWithCounter())
    result = counter("test text")
    assert result == 42
    assert called == ["test text"]


# ---------------------------------------------------------------------------
# 3. resolve_token_counter — falls back to approx_tokens when model lacks count_tokens
# ---------------------------------------------------------------------------


def test_resolve_token_counter_no_count_tokens_attr_falls_back() -> None:
    class _ModelWithout:
        pass  # no count_tokens

    counter = resolve_token_counter(_ModelWithout())
    text = "fallback test"
    assert counter(text) == approx_tokens(text)


def test_resolve_token_counter_non_callable_count_tokens_falls_back() -> None:
    class _ModelBadAttr:
        count_tokens = "not a callable"  # non-callable attribute

    counter = resolve_token_counter(_ModelBadAttr())
    text = "fallback test"
    assert counter(text) == approx_tokens(text)


# ---------------------------------------------------------------------------
# 4. inject with no recall results → status="ok" (empty context, no missing_kinds)
# ---------------------------------------------------------------------------


async def test_inject_empty_stack_returns_ok_status() -> None:
    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_fixed_clock)
    query = RecallQuery(text="anything")
    result = await injector.inject(query)
    assert isinstance(result, InjectedMemory)
    assert result.status == "ok"


async def test_inject_empty_stack_returns_empty_context() -> None:
    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_fixed_clock)
    query = RecallQuery(text="anything")
    result = await injector.inject(query)
    assert result.context.chunks == ()
    assert result.context.token_count == 0


# ---------------------------------------------------------------------------
# 5. inject resolves DEFAULT_MEMORY_POLICY when no explicit policy supplied
# ---------------------------------------------------------------------------


async def test_inject_uses_default_policy_when_not_supplied() -> None:
    """No required_kinds in DEFAULT_MEMORY_POLICY → status never "below_floor"."""
    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_fixed_clock)
    result = await injector.inject(RecallQuery())
    # DEFAULT has no required_kinds, so status is "ok" even with empty results
    assert result.status == "ok"
    assert result.missing_kinds == ()


# ---------------------------------------------------------------------------
# 6. inject stamps as_of with the clock when query.as_of is None
# ---------------------------------------------------------------------------


async def test_inject_stamps_as_of_when_none() -> None:
    """When query.as_of is None the injector stamps it with clock().

    We verify the clock was consumed by passing a counting clock and confirming
    the query received by the stack has a non-None as_of (indirectly, via no error
    in the outcome — the stack accepts the stamped query fine).
    """
    clock_calls: list[datetime] = []
    stamp_time = datetime(2026, 1, 1, 12, 0, 0, tzinfo=UTC)

    def _counting_clock() -> datetime:
        clock_calls.append(stamp_time)
        return stamp_time

    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_counting_clock)
    query = RecallQuery(text="test")
    assert query.as_of is None  # pre-condition

    await injector.inject(query)

    # Clock was called exactly once for the stamp
    assert len(clock_calls) == 1
    assert clock_calls[0] == stamp_time


async def test_inject_does_not_stamp_as_of_when_already_set() -> None:
    """When query.as_of is already set the clock must NOT be called."""
    clock_calls: list[datetime] = []

    def _counting_clock() -> datetime:
        clock_calls.append(_EPOCH)
        return _EPOCH

    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_counting_clock)
    preset_time = datetime(2026, 3, 15, 9, 0, 0, tzinfo=UTC)
    query = RecallQuery(as_of=preset_time)

    await injector.inject(query)

    assert len(clock_calls) == 0


# ---------------------------------------------------------------------------
# 7. inject calls record_use for admitted latent chunks ONLY
# ---------------------------------------------------------------------------


async def _make_injector_with_latent_chunks() -> tuple[MemoryInjector, AsyncMock]:
    """Build an injector backed by a minimal stack that returns a latent result."""
    from cogworx.recall.results import ChannelHit, FusedResult
    from cogworx.recall.stack import RecallOutcome
    from cogworx.substrate.latent import LatentMatch, LatentRecord

    # Build a stack stub whose recall() returns one latent FusedResult.
    latent_record = LatentRecord(id="lat-001", embedding=(1.0, 0.0), payload={"text": "hello"})
    hit = ChannelHit(channel="dense.latent", rank=1, raw_score=0.95)
    from cogworx.recall.stack import ChannelStatus

    fused = FusedResult(
        key="latent:lat-001",
        kind="latent",
        item=LatentMatch(
            record=latent_record,
            score=0.95,
            tier="hot",
            use_count=3,
            last_used_at=_EPOCH,
        ),
        text="hello",
        hits=(hit,),
        fused_score=0.95,
        fused_rank=1,
    )
    outcome = RecallOutcome(
        results=(fused,),
        channel_status=(ChannelStatus(channel="dense.latent", state="ok", count=1),),
    )

    stack_mock = MagicMock(spec=RecallStack)
    stack_mock.recall = AsyncMock(return_value=outcome)

    latent_store_mock = AsyncMock()
    latent_store_mock.record_use = AsyncMock(return_value=1)

    injector = MemoryInjector(
        stack_mock,
        latent_store=latent_store_mock,
        clock=_fixed_clock,
    )
    return injector, latent_store_mock


async def test_inject_calls_record_use_for_admitted_latent_chunks() -> None:
    injector, latent_store_mock = await _make_injector_with_latent_chunks()
    await injector.inject(RecallQuery())
    # record_use must have been called with the latent id (without the "latent:" prefix)
    latent_store_mock.record_use.assert_called_once()
    call_args = latent_store_mock.record_use.call_args[0][0]
    assert "lat-001" in call_args


async def test_inject_latent_uses_recorded_count_matches_return() -> None:
    injector, _latent_store_mock = await _make_injector_with_latent_chunks()
    result = await injector.inject(RecallQuery())
    assert result.latent_uses_recorded == 1


# ---------------------------------------------------------------------------
# 8. inject does NOT call record_use for non-latent chunks
# ---------------------------------------------------------------------------


async def test_inject_does_not_call_record_use_for_claim_chunks() -> None:
    """A stack that returns only claim results must NOT trigger record_use."""
    from cogworx.claims.provenance import Claim, Provenance
    from cogworx.knowledge.confidence import claim_confidence
    from cogworx.knowledge.identity import claim_id_for
    from cogworx.recall.results import ChannelHit, FusedResult
    from cogworx.recall.stack import ChannelStatus, RecallOutcome
    from cogworx.substrate.entity_kg import ScoredClaim

    prov = Provenance(source="system", confidence=1.0, recorded_at=_EPOCH)
    cid = claim_id_for("alice", "likes", "chocolate", scope="agent")
    claim = Claim(
        id=cid,
        subject="alice",
        predicate="likes",
        payload="chocolate",
        epistemic_type="inference",
        provenance=prov,
        valid_from=_EPOCH,
        valid_to=None,
        ingest_time=_EPOCH,
        created_by="test",
        scope="agent",
    )
    scored = ScoredClaim(
        claim=claim,
        confidence=claim_confidence([]),
        lineage_min_confidence=1.0,
    )
    hit = ChannelHit(channel="bm25.claims", rank=1, raw_score=0.8)
    fused = FusedResult(
        key=f"claim:{cid}",
        kind="claim",
        item=scored,
        text="alice likes chocolate",
        hits=(hit,),
        fused_score=0.8,
        fused_rank=1,
    )
    outcome = RecallOutcome(
        results=(fused,),
        channel_status=(ChannelStatus(channel="bm25.claims", state="ok", count=1),),
    )

    stack_mock = MagicMock(spec=RecallStack)
    stack_mock.recall = AsyncMock(return_value=outcome)

    latent_store_mock = AsyncMock()
    latent_store_mock.record_use = AsyncMock(return_value=0)

    injector = MemoryInjector(
        stack_mock,
        latent_store=latent_store_mock,
        clock=_fixed_clock,
    )
    result = await injector.inject(RecallQuery())

    # record_use must NOT be called when there are no latent chunks in the admitted set
    latent_store_mock.record_use.assert_not_called()
    assert result.latent_uses_recorded == 0


# ---------------------------------------------------------------------------
# 9. inject swallows record_use errors into record_use_error (status stays "ok")
# ---------------------------------------------------------------------------


async def test_inject_swallows_record_use_error_status_remains_ok() -> None:
    from cogworx.recall.results import ChannelHit, FusedResult
    from cogworx.recall.stack import ChannelStatus, RecallOutcome
    from cogworx.substrate.latent import LatentMatch, LatentRecord

    latent_record = LatentRecord(id="lat-err", embedding=(1.0, 0.0), payload={"text": "boom"})
    hit = ChannelHit(channel="dense.latent", rank=1, raw_score=0.9)
    fused = FusedResult(
        key="latent:lat-err",
        kind="latent",
        item=LatentMatch(
            record=latent_record,
            score=0.9,
            tier="cold",
            use_count=0,
            last_used_at=_EPOCH,
        ),
        text="boom",
        hits=(hit,),
        fused_score=0.9,
        fused_rank=1,
    )
    outcome = RecallOutcome(
        results=(fused,),
        channel_status=(ChannelStatus(channel="dense.latent", state="ok", count=1),),
    )

    stack_mock = MagicMock(spec=RecallStack)
    stack_mock.recall = AsyncMock(return_value=outcome)

    latent_store_mock = AsyncMock()
    latent_store_mock.record_use = AsyncMock(side_effect=RuntimeError("db down"))

    injector = MemoryInjector(
        stack_mock,
        latent_store=latent_store_mock,
        clock=_fixed_clock,
    )
    result = await injector.inject(RecallQuery())

    assert result.status == "ok"
    assert result.record_use_error is not None
    assert "RuntimeError" in result.record_use_error
    assert result.latent_uses_recorded == 0


# ---------------------------------------------------------------------------
# 10. below_floor status when required_kinds not met
# ---------------------------------------------------------------------------


async def test_inject_below_floor_when_required_kind_absent() -> None:
    """An empty result set with a required kind → status="below_floor"."""
    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_fixed_clock)
    policy = MemoryPolicy(required_kinds=("episode",))
    result = await injector.inject(RecallQuery(), policy=policy)
    assert result.status == "below_floor"
    assert "episode" in result.missing_kinds


async def test_inject_ok_when_required_kinds_empty() -> None:
    stack = _make_stack_empty()
    injector = MemoryInjector(stack, clock=_fixed_clock)
    policy = MemoryPolicy(required_kinds=())
    result = await injector.inject(RecallQuery(), policy=policy)
    assert result.status == "ok"
    assert result.missing_kinds == ()


# ---------------------------------------------------------------------------
# 11. inject with no latent_store wired — record_use is never attempted
# ---------------------------------------------------------------------------


async def test_inject_no_latent_store_does_not_crash() -> None:
    """latent_store=None → record_use is skipped silently, latent_uses_recorded=0."""
    stack = _make_stack_empty()
    injector = MemoryInjector(stack, latent_store=None, clock=_fixed_clock)
    result = await injector.inject(RecallQuery())
    assert result.latent_uses_recorded == 0
    assert result.record_use_error is None
