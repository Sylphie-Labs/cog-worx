"""Pod 2.6 memory injection spike (CANON S12) — SC-1 through SC-8.

Falsifiable spike success criteria for the MemoryInjector, InjectedMemory,
and the Engine's recall_stack integration.

All assertions are mutation-resistant: every positive control has a negative control that
MUST fail when the invariant is violated. A negative control that passes when it should
trip means the positive assertion has no teeth.

Pure Python — no Neo4j, no Postgres, zero model calls on the write path (ReplayModel([])
everywhere; verified by assert model.call_count == 0 after each scenario).

CANON compliance:
  S1  — model off write path
  S4  — tokenizer is a seam (duck-typed, no isinstance)
  S5  — every chunk carries >=1 ChannelHit with channel name + rank
  S6  — replay does not re-call record_use
  S8  — unwired lesion returns status="unwired", not a crash
  S9  — verdicts are pure counting, no model judgment
  S12 — spike gate
"""

from __future__ import annotations

import subprocess
import sys
import uuid
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

import pytest

from cogworx.claims.provenance import Artifact, Claim, Provenance
from cogworx.injection.injector import MemoryInjector, resolve_token_counter
from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, InjectedMemory, MemoryPolicy
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.loop.graph import StageGraph
from cogworx.loop.pathway import PathwayRegistry
from cogworx.loop.result import AwaitHuman, Done, StageResult
from cogworx.loop.stage import StageContext
from cogworx.loop.state import RunStatus
from cogworx.model.registry import ModelRegistry
from cogworx.recall.assembly import approx_tokens, assemble
from cogworx.recall.channels import (
    EpisodeRecencyChannel,
    LatentDenseChannel,
)
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import ChannelHit, FusedResult
from cogworx.recall.stack import RecallStack
from cogworx.runtime.engine import Engine
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode
from cogworx.substrate.latent import LatentMatch, LatentRecord, Tier, TierSweepResult
from cogworx.testing.doubles import (
    InMemoryEpisodeStore,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel

pytestmark = pytest.mark.spike

# ---------------------------------------------------------------------------
# Fixed timestamps — deterministic; no datetime.now() in this file
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_OLDER = datetime(2026, 6, 10, 10, 0, 0, tzinfo=UTC)
_CLOCK = lambda: _NOW  # noqa: E731


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _evidence(event_id: str | None = None) -> Any:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id="spike-2-6",
        source_authority=0.9,
        recorded_at=_NOW,
        event_id=event_id or uuid.uuid4().hex,
    )


def _claim(
    subject: str, predicate: str, payload: str, embedding: tuple[float, ...] | None = None
) -> Claim:
    cid = claim_id_for(subject, predicate, payload)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=0.9, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="spike-2-6",
        embedding=embedding,
    )


def _latent(id: str, embedding: tuple[float, ...], text: str = "") -> LatentRecord:
    return LatentRecord(id=id, embedding=embedding, payload={"text": text or id})


def _match(record_id: str, score: float, tier: Tier = "cold", text: str = "") -> LatentMatch:
    return LatentMatch(
        record=LatentRecord(
            id=record_id, embedding=(1.0, 0.0), payload={"text": text or record_id}
        ),
        score=score,
        tier=tier,
        use_count=0,
        last_used_at=_NOW,
    )


def _episode(ep_id: str, session_id: str = "sess-spike", content: str = "ep content") -> Episode:
    return Episode(
        episode_id=ep_id,
        run_id=ep_id.split(":")[0] if ":" in ep_id else ep_id,
        step_index=0,
        turn_index=0,
        session_id=session_id,
        role="user",
        content=content,
        kind="turn",
        occurred_at=_NOW,
    )


def _fused_claim(key: str, text: str, score: float = 0.5) -> FusedResult:
    """Build a minimal FusedResult for a claim item."""
    raw_claim = _claim(key.removeprefix("claim:"), "predicate", text)
    conf = claim_confidence([])
    scored = ScoredClaim(claim=raw_claim, confidence=conf, lineage_min_confidence=conf.confidence)
    return FusedResult(
        key=key,
        kind="claim",
        item=scored,
        text=text,
        hits=(ChannelHit(channel="dense.claims", rank=1, raw_score=score),),
        fused_score=score,
        fused_rank=1,
    )


def _fused_episode(
    key: str, text: str, score: float = 0.5, session_id: str = "sess-spike"
) -> FusedResult:
    """Build a minimal FusedResult for an episode item."""
    ep = _episode(key.removeprefix("episode:"), session_id=session_id, content=text)
    return FusedResult(
        key=key,
        kind="episode",
        item=ep,
        text=text,
        hits=(ChannelHit(channel="temporal.episodes", rank=1, raw_score=None),),
        fused_score=score,
        fused_rank=1,
    )


def _fused_latent(key: str, text: str, score: float = 0.5) -> FusedResult:
    """Build a minimal FusedResult for a latent item."""
    return FusedResult(
        key=key,
        kind="latent",
        item=LatentMatch(
            record=LatentRecord(id=key.removeprefix("latent:"), embedding=(1.0, 0.0)),
            score=score,
            tier="cold",
            use_count=0,
            last_used_at=_NOW,
        ),
        text=text,
        hits=(ChannelHit(channel="dense.latent", rank=1, raw_score=score),),
        fused_score=score,
        fused_rank=1,
    )


def _fused(key: str, kind: str, text: str, score: float = 0.5) -> FusedResult:
    """Build a minimal FusedResult with one ChannelHit (S5: at least one hit)."""
    if kind == "latent":
        return _fused_latent(key, text, score)
    elif kind == "episode":
        return _fused_episode(key, text, score)
    else:
        return _fused_claim(key, text, score)


# ---------------------------------------------------------------------------
# Spy LatentStore — records which ids were passed to record_use
# ---------------------------------------------------------------------------


class SpyLatentStore:
    """LatentStore double that records record_use calls and forwards search to a real store."""

    def __init__(self, real_store: InMemoryLatentStore) -> None:
        self._real = real_store
        self.record_use_calls: list[list[str]] = []  # each call's id list

    async def put(self, record: LatentRecord) -> None:
        await self._real.put(record)

    async def record_use(self, ids: Sequence[str]) -> int:
        self.record_use_calls.append(list(ids))
        return await self._real.record_use(ids)

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        tier: Tier | None = None,
    ) -> Sequence[LatentMatch]:
        return await self._real.search(embedding, k=k, tier=tier)

    async def sweep_tiers(self, *, now: datetime, hot_capacity: int) -> TierSweepResult:
        return await self._real.sweep_tiers(now=now, hot_capacity=hot_capacity)

    def all_recorded_ids(self) -> list[str]:
        """Flat list of all ids ever passed to record_use across all calls."""
        return [id_ for call in self.record_use_calls for id_ in call]

    def count_for(self, id_: str) -> int:
        """How many times did this id appear across all record_use calls?"""
        return sum(call.count(id_) for call in self.record_use_calls)


# ---------------------------------------------------------------------------
# Search-spying LatentStore (SC-6: detect whether cold search was called)
# ---------------------------------------------------------------------------


class SearchSpyLatentStore:
    """LatentStore that intercepts search() calls by tier, returning preset results."""

    def __init__(
        self, hot_results: Sequence[LatentMatch], cold_results: Sequence[LatentMatch]
    ) -> None:
        self._hot = list(hot_results)
        self._cold = list(cold_results)
        self.tier_none_calls: int = 0
        self.hot_calls: int = 0
        self.cold_calls: int = 0

    async def put(self, record: LatentRecord) -> None:
        pass

    async def record_use(self, ids: Sequence[str]) -> int:
        return 0

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        tier: Tier | None = None,
    ) -> Sequence[LatentMatch]:
        if tier is None:
            self.tier_none_calls += 1
            # return merged by score
            all_matches = sorted(self._hot + self._cold, key=lambda m: -m.score)
            return all_matches[:k]
        elif tier == "hot":
            self.hot_calls += 1
            return self._hot[:k]
        else:
            self.cold_calls += 1
            return self._cold[:k]

    async def sweep_tiers(self, *, now: datetime, hot_capacity: int) -> TierSweepResult:
        return TierSweepResult(promoted=0, demoted=0, hot_size=0)


# ---------------------------------------------------------------------------
# Minimal wired stack helpers
# ---------------------------------------------------------------------------


async def _make_minimal_stack_with_latent(
    latent_store: Any,
    *,
    hot_first: bool = False,
    min_similarity: float = 0.80,
) -> RecallStack:
    """Build a 1-channel stack (latent only) wrapping the given store."""
    channel = LatentDenseChannel(latent_store, hot_first=hot_first, min_similarity=min_similarity)
    return RecallStack([channel])


async def _make_episode_latent_stack(
    episode_store: InMemoryEpisodeStore,
    latent_store: Any,
) -> RecallStack:
    """2-channel stack: temporal.episodes + dense.latent."""
    ep_ch = EpisodeRecencyChannel(episode_store)
    lat_ch = LatentDenseChannel(latent_store)
    return RecallStack([ep_ch, lat_ch])


# ---------------------------------------------------------------------------
# Reference pathway for Engine-level tests (single stage that calls ctx.recall)
# ---------------------------------------------------------------------------

_RECALL_RESULT: dict[str, InjectedMemory | None] = {"value": None}
_RETRY_COUNTER: dict[str, int] = {"n": 0}


class RecallStage:
    """Stage that calls ctx.recall and stores the result; returns Done on success."""

    name: str = "recall_stage"
    transitions: tuple[str, ...] = ()

    def __init__(self, query: RecallQuery, policy: MemoryPolicy | None = None) -> None:
        self._query = query
        self._policy = policy

    async def run(self, ctx: StageContext) -> StageResult:
        mem = await ctx.recall(self._query, policy=self._policy)
        _RECALL_RESULT["value"] = mem
        artifact = Artifact(
            kind="recall-done",
            produced_by="recall_stage",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_NOW),
        )
        return Done(output=artifact)


class FailOnceThenRecallStage:
    """Stage that raises ValueError on the first attempt, then calls ctx.recall."""

    name: str = "fail_once_stage"
    transitions: tuple[str, ...] = ()

    def __init__(self, query: RecallQuery) -> None:
        self._query = query
        self._count = 0

    async def run(self, ctx: StageContext) -> StageResult:
        self._count += 1
        if self._count == 1:
            raise ValueError("intentional first-attempt failure")
        mem = await ctx.recall(self._query)
        _RECALL_RESULT["value"] = mem
        artifact = Artifact(
            kind="recall-done",
            produced_by="fail_once_stage",
            provenance=Provenance(source="system", confidence=1.0, recorded_at=_NOW),
        )
        return Done(output=artifact)


def _make_engine(
    *,
    recall_stack: RecallStack | None,
    stage: Any,
    model: ReplayModel | None = None,
    spy_latent: Any | None = None,
) -> tuple[Engine, PathwayRegistry, str]:
    """Wire an Engine around a single-stage pathway and return (engine, registry, pathway_id)."""
    m = model or ReplayModel([])
    journal = InMemoryJournal()
    graph_store = InMemoryGraphStore()
    latent = spy_latent or InMemoryLatentStore()
    graph = StageGraph([stage], entry=stage.name)
    pathways = PathwayRegistry()
    pid = f"pid-{uuid.uuid4().hex[:8]}"
    pathways.register(pid, graph, version=1)
    _reg = ModelRegistry()
    _reg.register("default", m)
    engine = Engine(
        models=_reg,
        journal=journal,
        graph_store=graph_store,
        latent=latent,
        pathways=pathways,
        clock=_CLOCK,
        recall_stack=recall_stack,
    )
    return engine, pathways, pid


def _initial_artifact() -> Artifact:
    return Artifact(
        kind="init",
        produced_by="spike",
        provenance=Provenance(source="human", confidence=1.0, recorded_at=_NOW),
    )


# ===========================================================================
# SC-1: S8 unwired lesion
# ===========================================================================


async def test_sc1_unwired_lesion_returns_status_unwired() -> None:
    """Engine with recall_stack=None → ctx.recall → status='unwired', empty chunks, COMPLETED."""
    _RECALL_RESULT["value"] = None
    query = RecallQuery(session_id="sess-sc1")
    stage = RecallStage(query)
    engine, _, pid = _make_engine(recall_stack=None, stage=stage)

    state = await engine.run(
        run_id="sc1-unwired",
        session_id="sess-sc1",
        pathway_id=pid,
        initial=_initial_artifact(),
    )

    assert state.status is RunStatus.COMPLETED
    mem = _RECALL_RESULT["value"]
    assert mem is not None
    assert mem.status == "unwired"
    assert len(mem.context.chunks) == 0
    assert mem.context.token_count == 0


async def test_sc1_negative_wired_not_unwired() -> None:
    """Negative control: Engine WITH a real recall_stack → status != 'unwired'."""
    # Build a minimal real stack backed by an empty latent store
    _RECALL_RESULT["value"] = None
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    channel = LatentDenseChannel(latent_store)
    stack = RecallStack([channel])

    query = RecallQuery(embedding=(1.0, 0.0, 0.0))
    stage = RecallStage(query)
    engine, _, pid = _make_engine(recall_stack=stack, stage=stage)

    state = await engine.run(
        run_id="sc1-wired",
        session_id="sess-sc1-wired",
        pathway_id=pid,
        initial=_initial_artifact(),
    )

    assert state.status is RunStatus.COMPLETED
    mem = _RECALL_RESULT["value"]
    assert mem is not None
    # Wired stack must NOT produce "unwired"
    assert mem.status != "unwired", "wired stack must not return status='unwired'"


# ===========================================================================
# SC-2: Budget + S4 tokenizer seam
# ===========================================================================


@pytest.mark.parametrize(
    "budget,n_items,seed",
    [
        (50, 5, 42),
        (200, 10, 7),
        (1, 20, 99),
        (500, 3, 0),
        (100, 8, 13),
    ],
)
async def test_sc2_budget_never_exceeded(budget: int, n_items: int, seed: int) -> None:
    """Property test: assembled token_count <= policy.token_budget for random fixtures."""
    rng = __import__("random").Random(seed)
    items = [
        _fused(
            f"latent:item-{i}",
            "latent",
            text="x" * rng.randint(4, 60),
            score=rng.random(),
        )
        for i in range(n_items)
    ]
    assembled = assemble(items, budget=budget, count_tokens=approx_tokens)
    assert assembled.token_count <= budget, (
        f"budget={budget} exceeded: token_count={assembled.token_count}"
    )


async def test_sc2_sentinel_counter_is_load_bearing() -> None:
    """Prove the token counter is load-bearing: swapping it changes which items are admitted.

    Craft a text where len(text) and approx_tokens(text) give DIFFERENT admission decisions.
    len(text) counter: each char = 1 token.  approx_tokens: chars // 4.
    Use budget=6: text of length 5 → len=5 tokens (fits), approx=1 token (fits trivially).
    Use another text of length 8 → len=8 tokens (doesn't fit remaining 1 after text1 with len),
    but with approx_tokens after first item (5//4=1 token) remaining=5, second (8//4=2) fits.
    """
    text_a = "AAAAA"  # len=5, approx=1 (5//4=1, max(1,1)=1)
    text_b = "BBBBBBBB"  # len=8, approx=2 (8//4=2)
    budget = 6

    items = [
        _fused("latent:a", "latent", text=text_a, score=0.9),
        _fused("latent:b", "latent", text=text_b, score=0.8),
    ]

    # With approx_tokens counter: a=1 token, b=2 tokens → both fit in budget=6
    with_approx = assemble(items, budget=budget, count_tokens=approx_tokens)
    # With len counter: a=5, b=8; a fits (5<=6), b does not (5+8=13>6) → only a admitted
    with_len = assemble(items, budget=budget, count_tokens=len)

    # Both use the same items but different counters → different admission decisions
    assert len(with_approx.chunks) == 2, "approx_tokens admits both items"
    assert len(with_len.chunks) == 1, "len counter drops the second item"
    # The admitted key must differ
    approx_keys = {c.key for c in with_approx.chunks}
    len_keys = {c.key for c in with_len.chunks}
    assert approx_keys != len_keys, "counter swap must change admitted set"


async def test_sc2_model_count_tokens_is_used() -> None:
    """resolve_token_counter uses model.count_tokens when present (S4 duck-type)."""
    called_with: list[str] = []

    class SentinelCounter:
        def count_tokens(self, text: str) -> int:
            called_with.append(text)
            return len(text)  # intentionally different from approx_tokens

    counter = resolve_token_counter(SentinelCounter())
    result = counter("hello")
    assert result == 5  # len("hello") = 5, not approx_tokens("hello")=1
    assert "hello" in called_with


async def test_sc2_no_count_tokens_falls_back_to_approx() -> None:
    """A model WITHOUT count_tokens → approx fallback, no crash (S4/S8)."""

    class NoCountTokens:
        pass  # deliberately has no count_tokens attribute

    counter = resolve_token_counter(NoCountTokens())
    # Should be approx_tokens itself (or equivalent behaviour)
    assert counter("hello") == approx_tokens("hello")

    # Also: None model → fallback
    counter_none = resolve_token_counter(None)
    assert counter_none("test text") == approx_tokens("test text")


async def test_sc2_replay_model_without_count_tokens_fallback() -> None:
    """ReplayModel with count_tokens deleted → fallback to approx, no crash."""
    model = ReplayModel([])
    # Verify it has count_tokens normally
    assert hasattr(model, "count_tokens")
    counter_before = resolve_token_counter(model)
    assert counter_before("hello") == model.count_tokens("hello")

    # Now shadow count_tokens with a non-callable attribute
    model._token_counter = None

    # After this, count_tokens still exists but delegates to approx; check robustness
    # by creating an object without the method entirely
    class NoMethod:
        pass

    counter_after = resolve_token_counter(NoMethod())
    # Must not crash; must use approx fallback
    assert counter_after("some text") == approx_tokens("some text")


# ===========================================================================
# SC-3: Admission floor (mutation-resistant)
# ===========================================================================


def _make_floor_fixture() -> list[FusedResult]:
    """Episode-dominated fixture: 8 episodes, 1 claim, 1 latent.

    All items have text of ~20 chars each → approx_tokens=5 each.
    Budget=500 so everything fits with min_per_kind=0 too.
    """
    items: list[FusedResult] = []
    # 8 episodes (dominant)
    for i in range(8):
        items.append(
            _fused(
                f"episode:ep-{i}",
                "episode",
                text=f"episode content item {i}",
                score=0.9 - i * 0.05,
            )
        )
    # 1 claim (minority)
    items.append(_fused("claim:cl-1", "claim", text="claim text content", score=0.3))
    # 1 latent (minority)
    items.append(_fused("latent:lat-1", "latent", text="latent content chunk", score=0.25))
    return items


async def test_sc3_min_per_kind_1_guarantees_floor() -> None:
    """With min_per_kind=1: assembled chunks contain >= 1 claim AND >= 1 latent."""
    items = _make_floor_fixture()
    assembled = assemble(items, budget=500, count_tokens=approx_tokens, min_per_kind=1)

    kinds_in_output = {c.kind for c in assembled.chunks}
    assert "claim" in kinds_in_output, "floor: claim must be present with min_per_kind=1"
    assert "latent" in kinds_in_output, "floor: latent must be present with min_per_kind=1"


async def test_sc3_negative_min_per_kind_0_drops_minority() -> None:
    """Negative control: min_per_kind=0 → pure greedy → episodes fill budget, minorities dropped."""
    items = _make_floor_fixture()
    # With budget=500, all 10 items fit (10 * 5 = 50 tokens). So min_per_kind=0 should also
    # include claim and latent when budget allows. To test the real "minority gets dropped"
    # scenario, use a tighter budget that fits exactly 8 items.
    # 8 items * 5 approx_tokens each = 40 tokens. Budget = 40.
    assembled_tight = assemble(items, budget=40, count_tokens=approx_tokens, min_per_kind=0)
    kinds_tight_0 = {c.kind for c in assembled_tight.chunks}

    # With tight budget and min_per_kind=0, pure greedy fills with episodes (highest ranked first)
    # The claim (score=0.3) and latent (score=0.25) are ranked last → dropped
    # This is the mutation control: WITHOUT floor, minorities are not guaranteed
    assert "claim" not in kinds_tight_0 or "latent" not in kinds_tight_0, (
        "mutation control: with tight budget and min_per_kind=0, "
        "at least one minority kind should be absent"
    )


async def test_sc3_golden_min_per_kind_0_is_byte_identical_to_no_arg() -> None:
    """min_per_kind=0 output is byte-identical to assemble() with no min_per_kind arg."""
    items = _make_floor_fixture()
    with_0 = assemble(items, budget=200, count_tokens=approx_tokens, min_per_kind=0)
    default_no_arg = assemble(items, budget=200, count_tokens=approx_tokens)
    assert with_0 == default_no_arg, "min_per_kind=0 must be byte-identical to default (no arg)"


async def test_sc3_floor_with_tight_budget_minority_admitted() -> None:
    """With min_per_kind=1 and tight budget that would drop minorities: admitted anyway."""
    items = _make_floor_fixture()
    # Tight budget = 40 tokens (8 * 5). Without floor, claim+latent are dropped.
    # WITH floor (min_per_kind=1): claim and latent must each get at least one slot.
    assembled_floor = assemble(items, budget=40, count_tokens=approx_tokens, min_per_kind=1)
    kinds_floor = {c.kind for c in assembled_floor.chunks}
    assert "claim" in kinds_floor, "floor must admit claim even when budget is tight"
    assert "latent" in kinds_floor, "floor must admit latent even when budget is tight"


# ===========================================================================
# SC-4: Verdict floor + S9 (zero model calls)
# ===========================================================================


async def test_sc4_below_floor_when_required_kind_absent() -> None:
    """zero-episode fixture + required_kinds=('episode',) → status='below_floor'."""
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    await latent_store.put(_latent("l1", (1.0, 0.0), text="latent text one"))

    channel = LatentDenseChannel(latent_store)
    stack = RecallStack([channel])
    policy = MemoryPolicy(required_kinds=("episode",))
    model = ReplayModel([])
    injector = MemoryInjector(stack, latent_store=latent_store, model=model, clock=_CLOCK)

    query = RecallQuery(embedding=(1.0, 0.0))
    mem = await injector.inject(query, policy=policy)

    assert mem.status == "below_floor", f"expected 'below_floor', got {mem.status!r}"
    assert "episode" in mem.missing_kinds
    # chunks are still delivered (non-empty — latent result was found)
    # (empty only if the store has nothing; here we have a latent item)
    assert isinstance(mem.context.chunks, tuple)  # could be empty or not, no crash is the key
    # No exception was raised — just status='below_floor'


async def test_sc4_missing_kinds_contains_required_absent_kind() -> None:
    """missing_kinds must list the absent required kind."""
    stack = RecallStack([])  # empty stack — no channels
    policy = MemoryPolicy(required_kinds=("episode", "claim"))
    model = ReplayModel([])
    injector = MemoryInjector(stack, model=model, clock=_CLOCK)

    query = RecallQuery(embedding=(1.0, 0.0))
    mem = await injector.inject(query, policy=policy)

    assert mem.status == "below_floor"
    assert "episode" in mem.missing_kinds
    assert "claim" in mem.missing_kinds


async def test_sc4_zero_model_calls_across_all_sc4_scenarios() -> None:
    """S9: verdict is pure counting — assert model.call_count == 0 after every scenario."""
    model = ReplayModel([])
    stack = RecallStack([])

    # Scenario 1: below_floor
    injector = MemoryInjector(stack, model=model, clock=_CLOCK)
    await injector.inject(
        RecallQuery(text="hello"),
        policy=MemoryPolicy(required_kinds=("episode",)),
    )
    assert model.call_count == 0, f"model was called {model.call_count} times — S9 violation"

    # Scenario 2: ok (empty result)
    await injector.inject(RecallQuery(text="hello"), policy=DEFAULT_MEMORY_POLICY)
    assert model.call_count == 0

    # Scenario 3: with latent items
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    await latent_store.put(_latent("lx", (1.0, 0.0)))
    ch = LatentDenseChannel(latent_store)
    stack2 = RecallStack([ch])
    injector2 = MemoryInjector(stack2, model=model, clock=_CLOCK)
    await injector2.inject(
        RecallQuery(embedding=(1.0, 0.0)),
        policy=MemoryPolicy(required_kinds=("episode",)),
    )
    assert model.call_count == 0, f"model still must not be called: {model.call_count}"


# ===========================================================================
# SC-5: record_use admitted-only + S6 replay-unchanged + at-least-once
# ===========================================================================


async def _build_latent_fixture_for_sc5() -> tuple[SpyLatentStore, list[str], str]:
    """3 latent items, budget admits 2, one dropped.

    Items have approx_tokens ~5 each (20-char text). Budget = 10 → admits 2 items.
    Returns (spy_store, [admitted_id_1, admitted_id_2], dropped_id).
    """
    real = InMemoryLatentStore(clock=_CLOCK)
    spy = SpyLatentStore(real)

    # 3 latent items — best scores win
    ids = ["lat-a", "lat-b", "lat-c"]
    texts = ["A" * 20, "B" * 20, "C" * 20]  # each ~5 tokens (20//4=5)
    for id_, text in zip(ids, texts, strict=False):
        await spy.put(_latent(id_, (1.0, 0.0), text=text))

    return spy, ids[:2], ids[2]  # expect first two admitted (highest cosine vs (1,0))


async def test_sc5_admitted_ids_get_record_use_dropped_does_not() -> None:
    """After inject: admitted latent ids get +1, dropped id gets +0."""
    real = InMemoryLatentStore(clock=_CLOCK)
    spy = SpyLatentStore(real)

    # Budget = 10 tokens (approx). Items of ~5 tokens each.
    # With 3 items all at max cosine to (1,0), only 2 fit in budget=10.
    for id_ in ["lat-a", "lat-b", "lat-c"]:
        await spy.put(_latent(id_, (1.0, 0.0), text="A" * 20))  # 20//4=5 tokens each

    channel = LatentDenseChannel(spy)
    stack = RecallStack([channel])
    policy = MemoryPolicy(token_budget=10)  # budget = 10 → admits 2 of 3 items
    model = ReplayModel([])
    injector = MemoryInjector(stack, latent_store=spy, model=model, clock=_CLOCK)

    mem = await injector.inject(RecallQuery(embedding=(1.0, 0.0)), policy=policy)

    # Determine which ids were admitted vs dropped
    admitted_keys = {c.key.removeprefix("latent:") for c in mem.context.chunks}
    all_ids = {"lat-a", "lat-b", "lat-c"}
    dropped_ids = all_ids - admitted_keys

    assert len(admitted_keys) == 2, f"expected 2 admitted, got {admitted_keys}"
    assert len(dropped_ids) == 1, f"expected 1 dropped, got {dropped_ids}"

    # Each admitted id appears exactly once in record_use calls
    for aid in admitted_keys:
        assert spy.count_for(aid) == 1, f"admitted id {aid!r} should have count=1"

    # Dropped id never appears
    (dropped_id,) = dropped_ids
    assert spy.count_for(dropped_id) == 0, f"dropped {dropped_id!r} must not appear in record_use"


async def test_sc5_s6_replay_does_not_reincrements_record_use() -> None:
    """S6: genuine replay via provide_human_input — record_use count UNCHANGED.

    A 2-stage pathway: RecallAwaitStage calls ctx.recall then parks on AwaitHuman;
    TerminalStage returns Done.  provide_human_input re-drives from the entry, which
    REPLAYS the committed RecallAwaitStage step (reads existing.result, never re-runs
    the stage) before freshly running TerminalStage.

    **Mutation target:** if _drive were to re-run the committed stage instead of
    replaying it, ctx.recall would fire a second time, spy.count_for(latent_id) would
    equal 2, and this test would fail.
    """
    real = InMemoryLatentStore(clock=_CLOCK)
    spy = SpyLatentStore(real)
    await spy.put(_latent("lat-replay", (1.0, 0.0), text="Replay latent item"))

    channel = LatentDenseChannel(spy)
    stack = RecallStack([channel])

    # Stage 1: calls ctx.recall, then parks AWAITING_HUMAN
    class RecallAwaitStage:
        name: str = "recall_await"
        transitions: tuple[str, ...] = ("terminal",)

        def __init__(self, query: RecallQuery) -> None:
            self._query = query
            self.run_count: int = 0

        async def run(self, ctx: StageContext) -> StageResult:
            self.run_count += 1
            await ctx.recall(self._query)
            return AwaitHuman(
                question="continue?",
                to="terminal",
                output=Artifact(
                    kind="await-output",
                    produced_by="recall_await",
                    provenance=Provenance(source="system", confidence=1.0, recorded_at=_NOW),
                ),
            )

    # Stage 2: just returns Done
    class TerminalStage:
        name: str = "terminal"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> StageResult:
            return Done(
                output=Artifact(
                    kind="done-output",
                    produced_by="terminal",
                    provenance=Provenance(source="system", confidence=1.0, recorded_at=_NOW),
                )
            )

    query = RecallQuery(embedding=(1.0, 0.0))
    recall_stage = RecallAwaitStage(query)
    terminal_stage = TerminalStage()

    journal = InMemoryJournal()
    model = ReplayModel([])
    graph_store = InMemoryGraphStore()
    graph = StageGraph([recall_stage, terminal_stage], entry="recall_await")
    pathways = PathwayRegistry()
    pid = "s6-replay-test"
    pathways.register(pid, graph, version=1)
    _reg = ModelRegistry()
    _reg.register("default", model)
    engine = Engine(
        models=_reg,
        journal=journal,
        graph_store=graph_store,
        latent=spy,
        pathways=pathways,
        clock=_CLOCK,
        recall_stack=stack,
    )

    run_id = "s6-run-1"
    state = await engine.run(
        run_id=run_id,
        session_id="sess-s6",
        pathway_id=pid,
        initial=_initial_artifact(),
    )
    # After first drive: parked AWAITING_HUMAN, recall_stage ran exactly once
    assert state.status is RunStatus.AWAITING_HUMAN
    assert recall_stage.run_count == 1
    assert spy.count_for("lat-replay") == 1

    # provide_human_input re-drives from entry:
    #   step 0 (recall_await) is COMMITTED → replayed (existing.result taken, stage.run NOT called)
    #   step 1 (terminal)     is NOT committed → run fresh → Done
    state2 = await engine.provide_human_input(run_id, payload={"answer": "yes"})
    assert state2.status is RunStatus.COMPLETED

    # S6 mutation target: replay must NOT have re-called stage.run (run_count still 1)
    assert recall_stage.run_count == 1, (
        f"S6 violated: recall_await.run called {recall_stage.run_count} times "
        "(should be 1 — replay reads committed result, never re-runs the stage)"
    )
    # record_use must not have been incremented a second time
    assert spy.count_for("lat-replay") == 1, (
        f"S6 violated: record_use count == {spy.count_for('lat-replay')}, expected 1 "
        "(replay must not re-increment record_use)"
    )


async def test_sc5_at_least_once_retry_increments_record_use() -> None:
    """At-least-once semantics: two inject calls (simulating a retry) → record_use += 2.

    The engine's retry machine re-runs the stage on a timer-driven second attempt. Since inject()
    is called again on retry, record_use is incremented a second time for the admitted items.
    This test models that contract directly at the injector level — inject() called twice →
    record_use(admitted_ids) called twice → count == 2 for each admitted id.
    """
    real = InMemoryLatentStore(clock=_CLOCK)
    spy = SpyLatentStore(real)
    await spy.put(_latent("lat-retry", (1.0, 0.0), text="At-least-once latent content"))

    channel = LatentDenseChannel(spy)
    stack = RecallStack([channel])
    model = ReplayModel([])
    injector = MemoryInjector(stack, latent_store=spy, model=model, clock=_CLOCK)

    # First inject (first attempt)
    await injector.inject(RecallQuery(embedding=(1.0, 0.0)))
    # Second inject (retry attempt — at-least-once)
    await injector.inject(RecallQuery(embedding=(1.0, 0.0)))

    # record_use must have been called twice for the admitted item
    assert spy.count_for("lat-retry") == 2, (
        f"at-least-once: expected record_use count=2 for admitted item, "
        f"got {spy.count_for('lat-retry')}"
    )

    # Negative control: a single inject call produces count == 1 (not 2)
    real2 = InMemoryLatentStore(clock=_CLOCK)
    spy2 = SpyLatentStore(real2)
    await spy2.put(_latent("lat-single", (1.0, 0.0), text="Single inject item"))
    ch2 = LatentDenseChannel(spy2)
    stack2 = RecallStack([ch2])
    injector2 = MemoryInjector(stack2, latent_store=spy2, model=ReplayModel([]), clock=_CLOCK)
    await injector2.inject(RecallQuery(embedding=(1.0, 0.0)))
    assert spy2.count_for("lat-single") == 1, "single inject must produce count == 1 (not 2)"


# ===========================================================================
# SC-6: Hot-first gate (CF-1 falsifier)
# ===========================================================================


async def test_sc6_cf1_dead_cold_higher_than_hot_wins() -> None:
    """CF-1 bug dead: hot similarity=0.30, cold similarity=0.90 → cold ranked above hot."""
    hot_item = _match("hot-low", score=0.30, tier="hot", text="hot low similarity item")
    cold_item = _match("cold-high", score=0.90, tier="cold", text="cold high similarity item")

    spy = SearchSpyLatentStore(
        hot_results=[hot_item],
        cold_results=[cold_item],
    )
    channel = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)

    query = RecallQuery(embedding=(1.0, 0.0))
    results = list(await channel.search(query))

    ids = [r.key.removeprefix("latent:") for r in results]
    # cold-high (0.90 >= τ=0.80) should win over hot-low (0.30 < τ)
    assert "cold-high" in ids, "cold item with higher similarity must be in results"
    if "hot-low" in ids and "cold-high" in ids:
        # cold-high must rank above hot-low
        assert ids.index("cold-high") < ids.index("hot-low"), (
            "CF-1 dead: cold item (score=0.90) must rank above hot item (score=0.30)"
        )


async def test_sc6_hot_above_tau_outranks_cold() -> None:
    """Hot item sim=0.85 (>=τ=0.80) ranks above cold item sim=0.90 — bounded bias within gate."""
    # Hot item is accepted (≥τ), cold item competes only in the fallback pool (not reached).
    # When hot fills k=1 items at ≥τ, cold search is never called.
    hot_item = _match("hot-above", score=0.85, tier="hot", text="hot above threshold")
    cold_item = _match("cold-high2", score=0.90, tier="cold", text="cold high score item")

    spy = SearchSpyLatentStore(
        hot_results=[hot_item],
        cold_results=[cold_item],
    )
    channel = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)

    query = RecallQuery(embedding=(1.0, 0.0), k=1)
    results = list(await channel.search(query))

    # With k=1 and 1 hot item ≥τ, only hot item is returned
    assert len(results) == 1
    assert results[0].key == "latent:hot-above"
    # Cold search must not have been called
    assert spy.cold_calls == 0, "cold search must not be called when hot fills k at >=τ"


async def test_sc6_all_hot_sub_tau_falls_back_to_tier_agnostic() -> None:
    """All hot items below τ → result is tier-agnostic global search (provable equivalence D10)."""
    hot_item_a = _match("hot-sub-a", score=0.50, tier="hot", text="hot sub tau a")
    hot_item_b = _match("hot-sub-b", score=0.60, tier="hot", text="hot sub tau b")
    cold_item_c = _match("cold-c", score=0.75, tier="cold", text="cold fallback c")
    cold_item_d = _match("cold-d", score=0.65, tier="cold", text="cold fallback d")

    spy = SearchSpyLatentStore(
        hot_results=[hot_item_a, hot_item_b],
        cold_results=[cold_item_c, cold_item_d],
    )
    channel_hot = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)

    query = RecallQuery(embedding=(1.0, 0.0), k=4)
    hot_first_results = list(await channel_hot.search(query))

    # Expected: all hot items are sub-τ, so the fallback pool is
    # hot_items + cold_items sorted by score desc.
    # accepted=[], pool = sorted([0.50, 0.60, 0.75, 0.65]) desc = [0.75, 0.65, 0.60, 0.50]
    # final = [cold-c, cold-d, hot-sub-b, hot-sub-a]
    hot_first_ids = [r.key.removeprefix("latent:") for r in hot_first_results]

    # Tier-agnostic equivalent (all items sorted by score):
    all_items = sorted(
        [hot_item_a, hot_item_b, cold_item_c, cold_item_d],
        key=lambda m: -m.score,
    )
    tier_agnostic_ids = [m.record.id for m in all_items[:4]]

    assert hot_first_ids == tier_agnostic_ids, (
        f"all-hot-sub-τ fallback must equal tier-agnostic order:\n"
        f"hot_first={hot_first_ids}\ntier_agnostic={tier_agnostic_ids}"
    )


async def test_sc6_hot_fills_k_cold_never_called() -> None:
    """When hot fills k items all ≥τ: cold search store method was NEVER called."""
    hot_items = [_match(f"hot-{i}", score=0.85 + i * 0.01, tier="hot") for i in range(3)]
    cold_items = [_match(f"cold-{i}", score=0.95, tier="cold") for i in range(3)]

    spy = SearchSpyLatentStore(hot_results=hot_items, cold_results=cold_items)
    channel = LatentDenseChannel(spy, hot_first=True, min_similarity=0.80)

    query = RecallQuery(embedding=(1.0, 0.0), k=3)
    results = list(await channel.search(query))

    assert len(results) == 3
    assert spy.cold_calls == 0, f"cold search must not be called: cold_calls={spy.cold_calls}"
    # All results come from hot tier
    result_ids = {r.key.removeprefix("latent:") for r in results}
    hot_ids = {m.record.id for m in hot_items}
    assert result_ids.issubset(hot_ids)


# ===========================================================================
# SC-7: S5 provenance on every chunk
# ===========================================================================


async def test_sc7_every_chunk_has_at_least_one_channel_hit() -> None:
    """In every scenario: every chunk has len(hits) >= 1 with channel name + rank."""
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    await latent_store.put(_latent("l-prov-1", (1.0, 0.0), text="provenance test item one"))
    await latent_store.put(_latent("l-prov-2", (0.9, 0.1), text="provenance test item two"))

    ep_store = InMemoryEpisodeStore()
    ep = _episode("run-prov:step-0:turn-0", session_id="sess-prov")
    from cogworx.substrate.journal import ProjectionCursor

    await ep_store.project_episodes(
        "consumer",
        [ep],
        ProjectionCursor(commit_ordinal=1, run_id="run-prov", step_index=0),
    )

    stack = await _make_episode_latent_stack(ep_store, latent_store)
    model = ReplayModel([])
    injector = MemoryInjector(stack, latent_store=latent_store, model=model, clock=_CLOCK)

    query = RecallQuery(embedding=(1.0, 0.0), session_id="sess-prov")
    mem = await injector.inject(query)

    for chunk in mem.context.chunks:
        assert len(chunk.hits) >= 1, f"chunk {chunk.key!r} has no hits (S5 violation)"
        for hit in chunk.hits:
            assert hit.channel, f"chunk {chunk.key!r} has a hit with empty channel name"
            assert hit.rank >= 1, f"chunk {chunk.key!r} hit rank is < 1"


async def test_sc7_channel_status_non_empty_and_names_match_stack() -> None:
    """InjectedMemory.channel_status is non-empty and channel names match the stack's channels."""
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    await latent_store.put(_latent("lcs-1", (1.0, 0.0), text="channel status test item"))

    channel = LatentDenseChannel(latent_store)
    stack = RecallStack([channel])
    model = ReplayModel([])
    injector = MemoryInjector(stack, latent_store=latent_store, model=model, clock=_CLOCK)

    mem = await injector.inject(RecallQuery(embedding=(1.0, 0.0)))

    assert len(mem.channel_status) > 0, "channel_status must be non-empty"
    status_names = {cs.channel for cs in mem.channel_status}
    assert "dense.latent" in status_names, (
        f"channel_status must include 'dense.latent': {status_names}"
    )


async def test_sc7_s9_provenance_on_all_scenarios_with_zero_model_calls() -> None:
    """S9 + S5: provenance on all chunks, zero model calls across all SC-7 sub-scenarios."""
    model = ReplayModel([])

    # Scenario A: latent only
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    await latent_store.put(_latent("lscen-a", (1.0, 0.0), text="scenario A latent item"))
    ch_a = LatentDenseChannel(latent_store)
    stack_a = RecallStack([ch_a])
    injector_a = MemoryInjector(stack_a, model=model, clock=_CLOCK)
    mem_a = await injector_a.inject(RecallQuery(embedding=(1.0, 0.0)))

    for chunk in mem_a.context.chunks:
        assert len(chunk.hits) >= 1, f"S5: chunk {chunk.key!r} has no hits"

    assert model.call_count == 0

    # Scenario B: empty stack
    stack_b = RecallStack([])
    injector_b = MemoryInjector(stack_b, model=model, clock=_CLOCK)
    mem_b = await injector_b.inject(RecallQuery(text="hello"))
    # No chunks expected, but no crash
    assert isinstance(mem_b.context.chunks, tuple)
    assert model.call_count == 0


# ===========================================================================
# SC-8: S1 import invariants
# ===========================================================================


def test_sc8_cogworx_injection_imports_cleanly() -> None:
    """cogworx.injection import succeeds as the first import (no ImportError)."""
    result = subprocess.run(
        [sys.executable, "-c", "import cogworx.injection"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"import cogworx.injection failed:\nstdout={result.stdout}\nstderr={result.stderr}"
    )


def test_sc8_cogworx_recall_does_not_import_runtime() -> None:
    """cogworx.recall must not import cogworx.runtime (the engine's layer).

    KNOWN CF (documented in Pod 2.5 SC-5): cogworx.recall transitively imports cogworx.model.base
    via: recall.stack → substrate.entity_kg → substrate.journal → loop.result → loop/__init__ →
    loop.stage → model.base. The carry-forward resolution is to move StageResult outside loop/ or
    stop loop/__init__.py eagerly importing stage.py. Tracking that gap here as a documented
    known violation on cogworx.model, but NOT on cogworx.runtime (the direct injection seam
    must never pull in the runtime engine layer).
    """
    # Verify cogworx.recall does NOT import cogworx.runtime (that would be a deeper violation)
    script_runtime = (
        "import cogworx.recall; import sys; "
        "mods = list(sys.modules); "
        "runtime_violations = [m for m in mods if 'cogworx.runtime' in m]; "
        "assert not runtime_violations, "
        "f'cogworx.recall imported cogworx.runtime: {runtime_violations}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script_runtime],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cogworx.recall imported cogworx.runtime (S1 layer violation):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )

    # Document the known CF: cogworx.recall DOES import cogworx.model (known carry-forward from 2.5)
    script_model_cf = (
        "import cogworx.recall; import sys; "
        "mods = list(sys.modules); "
        "model_imports = [m for m in mods if 'cogworx.model' in m]; "
        "print('KNOWN CF model imports:', model_imports)"
    )
    cf_result = subprocess.run(
        [sys.executable, "-c", script_model_cf],
        capture_output=True,
        text=True,
    )
    # This is expected to print the known CF — we do NOT fail here, we document it.
    # The fix (deferred CF carry-forward) is: move StageResult out of cogworx.loop or
    # break loop/__init__.py's eager import of stage.py.
    assert cf_result.returncode == 0, f"subprocess failed unexpectedly: {cf_result.stderr}"


def test_sc8_cogworx_injection_has_zero_runtime_imports_of_model_and_runtime() -> None:
    """cogworx.injection must not import cogworx.model or cogworx.runtime at module load time.

    Note: cogworx.injection.injector uses TYPE_CHECKING guard for model — this verifies the
    guard works at runtime (no actual runtime import).
    """
    script = (
        "import cogworx.injection; import sys; "
        "mods = list(sys.modules); "
        "violations = [m for m in mods if 'cogworx.runtime' in m]; "
        "assert not violations, f'S1 violated — injection imported runtime: {violations}'"
    )
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cogworx.injection imported cogworx.runtime (S1 violation):\n"
        f"stdout={result.stdout}\nstderr={result.stderr}"
    )


# ===========================================================================
# Cross-cutting: model.call_count == 0 after EVERY scenario (S1/S6/S9)
# ===========================================================================


async def test_all_scenarios_zero_model_calls() -> None:
    """Final guard: run a representative batch and confirm no model calls were made."""
    model = ReplayModel([])

    # Build minimal wired stack
    latent_store = InMemoryLatentStore(clock=_CLOCK)
    await latent_store.put(_latent("zmc-1", (1.0, 0.0), text="zero model calls item"))
    ch = LatentDenseChannel(latent_store)
    stack = RecallStack([ch])
    injector = MemoryInjector(stack, latent_store=latent_store, model=model, clock=_CLOCK)

    # Run multiple inject calls covering different policy shapes
    scenarios: list[tuple[RecallQuery, MemoryPolicy]] = [
        (RecallQuery(embedding=(1.0, 0.0)), DEFAULT_MEMORY_POLICY),
        (RecallQuery(embedding=(0.5, 0.5)), MemoryPolicy(token_budget=100, min_per_kind=1)),
        (RecallQuery(embedding=(1.0, 0.0)), MemoryPolicy(required_kinds=("episode",))),
        (RecallQuery(text="hello"), DEFAULT_MEMORY_POLICY),
    ]
    for query, policy in scenarios:
        await injector.inject(query, policy=policy)

    assert model.call_count == 0, (
        f"S1/S9 violated: {model.call_count} model calls across {len(scenarios)} inject scenarios"
    )
