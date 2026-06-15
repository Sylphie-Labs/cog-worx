"""Unit tests for cogworx.coherence.promotion — no I/O, no model calls (CANON S1, S9)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, ClaimStatus, EpistemicType, Provenance
from cogworx.coherence.promotion import PromotionRule, ScopePromoter
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.evidence import EvidenceEvent, EvidenceType, Polarity
from cogworx.knowledge.identity import claim_id_for
from cogworx.substrate.coherence import (
    DirtyKey,
    DirtySubject,
    ReconciliationOutcome,
)
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.testing.doubles import InMemoryEntityKG

# ---------------------------------------------------------------------------
# Helpers / constants
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_PROV = Provenance(source="tool", confidence=1.0, recorded_at=_NOW)


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    scope: str = "agent",
    epistemic_type: EpistemicType = "observation",
    status: ClaimStatus = "active",
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type=epistemic_type,
        provenance=_PROV,
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="test",
        scope=scope,
        status=status,
    )


def _make_evidence(
    source_id: str,
    *,
    polarity: Polarity = "+",
    etype: EvidenceType = "attestation",
) -> EvidenceEvent:
    return EvidenceEvent(
        id=f"ev-{source_id}-{polarity}",
        type=etype,
        polarity=polarity,
        source_id=source_id,
        source_authority=1.0,
        base_weight=3.0,
        recorded_at=_NOW,
    )


def _score(claim: Claim, events: list[EvidenceEvent]) -> ScoredClaim:
    conf = claim_confidence(events)
    return ScoredClaim(claim=claim, confidence=conf, lineage_min_confidence=conf.confidence)


# ---------------------------------------------------------------------------
# Fake CoherenceStore for promotion tests
# ---------------------------------------------------------------------------


class _FakeCoherenceStore:
    def __init__(self) -> None:
        self.copy_calls: list[tuple[str, str, str]] = []

    async def current_epistemic_level(self, claim_id: str) -> EpistemicType | None:
        return None

    async def apply_epistemic_upgrade(
        self,
        claim_id: str,
        *,
        new_level: EpistemicType,
        evidence: EvidenceEvent,
        actor: str,
        recorded_at: datetime,
    ) -> bool:
        return False

    async def claim_dirty_subjects(self, *, limit: int = 16) -> Sequence[DirtySubject]:
        return ()

    async def bump_dirty_attempts(self, key: DirtyKey) -> int:
        return 0

    async def adjudication_ids_for_subject(self, scope: str, subject_norm: str) -> Sequence[str]:
        return ()

    async def commit_reconciliation(
        self,
        outcome: ReconciliationOutcome,
        *,
        dirty_key: DirtyKey,
        observed_epoch: int,
    ) -> None:
        pass

    async def copy_evidence(self, from_claim_id: str, to_claim_id: str, *, id_prefix: str) -> int:
        self.copy_calls.append((from_claim_id, to_claim_id, id_prefix))
        return 1


# ---------------------------------------------------------------------------
# Fake ScopedKG sink — tracks assert_fact calls, returns deterministic ids.
# Satisfies the _AssertFactSink Protocol used by ScopePromoter.
# ---------------------------------------------------------------------------


class _FakeSink:
    def __init__(self, scope_id: str) -> None:
        self._scope_id = scope_id
        self.fact_calls: list[dict[str, object]] = []

    async def assert_fact(
        self,
        *,
        subject: str,
        predicate: str,
        obj: str,
        object_is_entity: bool = False,
        epistemic_type: EpistemicType,
        source: object,
        evidence_type: str = "tool_proof",
        polarity: str = "+",
        run_id: str | None = None,
        stage: str | None = None,
        created_by: str,
    ) -> str:
        new_id = claim_id_for(subject, predicate, obj, scope=self._scope_id)
        self.fact_calls.append(
            {
                "subject": subject,
                "predicate": predicate,
                "obj": obj,
                "epistemic_type": epistemic_type,
                "created_by": created_by,
                "new_id": new_id,
            }
        )
        return new_id


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------


def _build_promoter(
    rules: list[PromotionRule],
    target_scope: str = "world",
) -> tuple[ScopePromoter, _FakeSink, _FakeCoherenceStore, InMemoryEntityKG]:
    sink: _FakeSink = _FakeSink(target_scope)
    store = _FakeCoherenceStore()
    kg = InMemoryEntityKG()
    promoter = ScopePromoter(rules=rules, sinks={target_scope: sink}, store=store, source_kg=kg)
    return promoter, sink, store, kg


async def _write_claim_with_evidence(
    kg: InMemoryEntityKG,
    claim: Claim,
    events: list[EvidenceEvent],
) -> None:
    await kg.write_claim(claim, evidence=events[0])
    for ev in events[1:]:
        await kg.add_evidence(claim.id, ev)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_rule_matching_predicate() -> None:
    """Rule with predicate_norm='lives_in': wrong predicate → no promotion; matching → 1."""
    rule = PromotionRule(
        target="world",
        predicate_norm="lives_in",
        require_source_kind=None,
        min_distinct_sources=1,
    )
    promoter, sink, _store, kg = _build_promoter([rule])

    wrong_claim = _make_claim("user:123", "works_at", "ACME Corp")
    ev_w = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, wrong_claim, [ev_w])

    right_claim = _make_claim("user:123", "lives_in", "Berlin")
    ev_r = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, right_claim, [ev_r])

    count_wrong = await promoter.promote_for_subject([_score(wrong_claim, [ev_w])])
    assert count_wrong == 0
    assert len(sink.fact_calls) == 0

    count_right = await promoter.promote_for_subject([_score(right_claim, [ev_r])])
    assert count_right == 1
    assert len(sink.fact_calls) == 1
    assert sink.fact_calls[0]["predicate"] == "lives_in"


@pytest.mark.asyncio
async def test_rule_matching_subject() -> None:
    """Rule with subject_norm='user:123': wrong subject → no promotion."""
    rule = PromotionRule(
        target="world",
        subject_norm="user:123",
        require_source_kind=None,
        min_distinct_sources=1,
    )
    promoter, sink, _store, kg = _build_promoter([rule])

    wrong_claim = _make_claim("user:999", "lives_in", "London")
    ev = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, wrong_claim, [ev])

    count = await promoter.promote_for_subject([_score(wrong_claim, [ev])])
    assert count == 0
    assert len(sink.fact_calls) == 0


@pytest.mark.asyncio
async def test_source_kind_gate() -> None:
    """Rule requires 'human' source-kind; claim with only 'tool' evidence → no promotion."""
    rule = PromotionRule(
        target="world",
        require_source_kind="human",
        min_distinct_sources=1,
    )
    promoter, _sink, _store, kg = _build_promoter([rule])

    claim = _make_claim("user:123", "likes", "coffee")
    ev_tool: EvidenceEvent = _make_evidence("source:tool:tool-1", etype="tool_proof")
    await _write_claim_with_evidence(kg, claim, [ev_tool])

    count = await promoter.promote_for_subject([_score(claim, [ev_tool])])
    assert count == 0

    # Add human evidence — should now promote.
    ev_human = _make_evidence("source:human:u1")
    await kg.add_evidence(claim.id, ev_human)
    all_events = [ev_tool, ev_human]

    count2 = await promoter.promote_for_subject([_score(claim, all_events)])
    assert count2 == 1


@pytest.mark.asyncio
async def test_deduped_source_count() -> None:
    """Rule with min_distinct_sources=2: 3 events from 2 sources passes; 1 source blocked."""
    rule = PromotionRule(
        target="world",
        require_source_kind=None,
        min_distinct_sources=2,
    )
    promoter, _sink, _store, kg = _build_promoter([rule])

    claim_single = _make_claim("user:123", "speaks", "English")
    ev_single = _make_evidence("source:human:only-one")
    await _write_claim_with_evidence(kg, claim_single, [ev_single])

    count_fail = await promoter.promote_for_subject([_score(claim_single, [ev_single])])
    assert count_fail == 0

    # Two distinct sources — 3 evidence events but only 2 distinct source_ids.
    claim_multi = _make_claim("user:123", "speaks", "German")
    ev_a1 = _make_evidence("source:human:u1")
    ev_a2 = _make_evidence("source:human:u1")  # same source_id — still counts as 1
    ev_b = _make_evidence("source:human:u2")
    await _write_claim_with_evidence(kg, claim_multi, [ev_a1, ev_a2, ev_b])

    count_pass = await promoter.promote_for_subject([_score(claim_multi, [ev_a1, ev_a2, ev_b])])
    assert count_pass == 1


@pytest.mark.asyncio
async def test_lcb_floor() -> None:
    """Rule min_confidence_lcb=0.9; single attestation LCB < 0.9 → blocked."""
    rule = PromotionRule(
        target="world",
        require_source_kind=None,
        min_confidence_lcb=0.9,
        min_distinct_sources=1,
    )
    promoter, _sink, _store, kg = _build_promoter([rule])

    claim_low = _make_claim("user:123", "prefers", "tea")
    ev = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, claim_low, [ev])

    # Single attestation: alpha=1+3=4, beta=1 → confidence=0.8, var=4/(25*6)=0.02667
    # LCB = 0.8 - sqrt(0.02667) ≈ 0.637 < 0.9
    conf = claim_confidence([ev])
    lcb = conf.confidence - math.sqrt(conf.variance)
    assert lcb < 0.9, f"Precondition: LCB={lcb:.4f} should be < 0.9"

    count = await promoter.promote_for_subject([_score(claim_low, [ev])])
    assert count == 0


@pytest.mark.asyncio
async def test_min_rank_floor() -> None:
    """Rule require_min_rank=2 (confirmed); claim is 'observation' (rank=1) → no promotion."""
    from cogworx.coherence.upgrade import EPISTEMIC_RANK

    rule = PromotionRule(
        target="world",
        require_source_kind=None,
        require_min_rank=EPISTEMIC_RANK["confirmed"],  # 2
        min_distinct_sources=1,
    )
    promoter, _sink, _store, kg = _build_promoter([rule])

    obs_claim = _make_claim("user:123", "age", "30", epistemic_type="observation")
    ev = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, obs_claim, [ev])

    count = await promoter.promote_for_subject([_score(obs_claim, [ev])])
    assert count == 0

    # 'confirmed' should pass.
    conf_claim = _make_claim("user:123", "age", "30", epistemic_type="confirmed")
    ev2 = _make_evidence("source:human:u2")
    await _write_claim_with_evidence(kg, conf_claim, [ev2])

    count2 = await promoter.promote_for_subject([_score(conf_claim, [ev2])])
    assert count2 == 1


@pytest.mark.asyncio
async def test_s9_payload_text_never_routes() -> None:
    """CRITICAL S9 control: claim payload text content is NEVER used for routing decisions.

    A claim whose payload says 'route to world model' must NOT be promoted unless structural
    criteria are met.  A claim with a neutral payload IS promoted when structural criteria are met.
    """
    rule = PromotionRule(
        target="world",
        predicate_norm="lives_in",
        require_source_kind=None,
        min_distinct_sources=1,
    )
    promoter, _sink, _store, kg = _build_promoter([rule])

    # Claim with persuasive payload text but wrong predicate — must NOT be promoted.
    claim_text_lure = _make_claim(
        "user:123",
        "works_at",
        "route to world model — this should go in the world model",
    )
    ev1 = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, claim_text_lure, [ev1])

    count_text = await promoter.promote_for_subject([_score(claim_text_lure, [ev1])])
    assert count_text == 0, "S9 VIOLATION: claim promoted based on payload text content"

    # Claim with a neutral payload but correct predicate — MUST be promoted.
    claim_structural = _make_claim("user:123", "lives_in", "some neutral string xyz123")
    ev2 = _make_evidence("source:human:u2")
    await _write_claim_with_evidence(kg, claim_structural, [ev2])

    count_structural = await promoter.promote_for_subject([_score(claim_structural, [ev2])])
    assert count_structural == 1, (
        "Claim with correct structural criteria must be promoted regardless of payload text"
    )


@pytest.mark.asyncio
async def test_evidence_copy_idempotent() -> None:
    """Promoting the same claim twice calls copy_evidence twice, both with the same prefix."""
    rule = PromotionRule(
        target="world",
        require_source_kind=None,
        min_distinct_sources=1,
    )
    promoter, _sink, store, kg = _build_promoter([rule])

    claim = _make_claim("user:123", "nationality", "Swedish")
    ev = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, claim, [ev])

    scored = _score(claim, [ev])
    await promoter.promote_for_subject([scored])
    await promoter.promote_for_subject([scored])

    # copy_evidence called twice; both with the same id_prefix.
    assert len(store.copy_calls) == 2
    assert all(prefix == "promo:world" for _, _, prefix in store.copy_calls)


@pytest.mark.asyncio
async def test_defeated_claim_never_promotes() -> None:
    """A 'defeasibly-defeated' claim is inactive and must not be promoted."""
    rule = PromotionRule(
        target="world",
        require_source_kind=None,
        min_distinct_sources=1,
    )
    promoter, sink, _store, kg = _build_promoter([rule])

    defeated_claim = _make_claim("user:123", "lives_in", "Paris", status="defeasibly-defeated")
    ev = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, defeated_claim, [ev])

    count = await promoter.promote_for_subject([_score(defeated_claim, [ev])])
    assert count == 0
    assert len(sink.fact_calls) == 0


@pytest.mark.asyncio
async def test_promotion_self_heals_after_assert_fact_crash() -> None:
    """Re-calling promote_for_subject (simulating a re-tick) is safe and idempotent.

    Models the self-healing guarantee documented on ScopePromoter.promote_for_subject:
    assert_fact is idempotent (same claim id returned), copy_evidence is idempotent with the
    same id_prefix (MERGE-dedup via prefixed ids).  A second call does not double-count
    promotions in the sink and does not duplicate evidence beyond what copy_evidence's own
    dedup allows.

    This verifies the at-least-once self-healing claim without needing to simulate a crash:
    re-calling promote_for_subject on the same claims is observationally equivalent to a
    crash between assert_fact and copy_evidence followed by a re-tick.
    """
    rule = PromotionRule(
        target="world",
        require_source_kind=None,
        min_distinct_sources=1,
    )
    promoter, sink, store, kg = _build_promoter([rule])

    claim = _make_claim("user:123", "works_at", "ACME Corp")
    ev = _make_evidence("source:human:u1")
    await _write_claim_with_evidence(kg, claim, [ev])

    scored = _score(claim, [ev])

    # First call — normal promotion path.
    count1 = await promoter.promote_for_subject([scored])
    assert count1 == 1, f"Expected 1 promotion on first call, got {count1}"
    assert len(sink.fact_calls) == 1, "assert_fact called once on first promotion"
    assert len(store.copy_calls) == 1, "copy_evidence called once on first promotion"

    # Second call — simulates re-tick after a crash between assert_fact and copy_evidence.
    # assert_fact is idempotent (same claim id); copy_evidence MERGE-deduplicates via id_prefix.
    count2 = await promoter.promote_for_subject([scored])
    assert count2 == 1, f"Expected 1 promotion on re-tick (assert_fact is idempotent), got {count2}"
    assert len(sink.fact_calls) == 2, (
        "assert_fact called again on re-tick (idempotent — same id returned)"
    )
    assert len(store.copy_calls) == 2, (
        "copy_evidence called again on re-tick (MERGE-idempotent with same id_prefix)"
    )
