"""Unit tests for CoherenceReconciler (Pod 2.7, U6).

All tests use InMemoryEntityKG + TableOracle — no live Neo4j, no live model calls.
Fully deterministic.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.claims.provenance import Claim, EpistemicType, Provenance, ProvenanceSource
from cogworx.coherence.config import CoherenceConfig
from cogworx.coherence.promotion import ScopePromoter
from cogworx.coherence.reconciler import CoherenceReconciler
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.testing.doubles import InMemoryEntityKG
from cogworx.testing.fake_oracle import TableOracle

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(*, source: ProvenanceSource = "system") -> Provenance:
    return Provenance(source=source, confidence=0.9, recorded_at=_T0)


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    scope: str = "agent",
    epistemic_type: EpistemicType = "observation",
    valid_from: datetime = _T0,
    valid_to: datetime | None = None,
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type=epistemic_type,
        provenance=_prov(),
        valid_from=valid_from,
        valid_to=valid_to,
        ingest_time=valid_from,
        created_by="test",
        scope=scope,
    )


def _ev(event_id: str = "ev1", *, source_id: str = "src1") -> EvidenceEvent:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source_id,
        source_authority=0.8,
        recorded_at=_T0,
        event_id=event_id,
    )


def _reconciler(
    kg: InMemoryEntityKG,
    oracle: TableOracle,
    *,
    max_attempts: int = 5,
    max_claims: int = 128,
    batch_limit: int = 16,
    max_mus: int = 4,
    promoter: ScopePromoter | None = None,
) -> CoherenceReconciler:
    config = CoherenceConfig(
        batch_limit=batch_limit,
        max_claims_per_subject=max_claims,
        max_attempts=max_attempts,
        max_mus_per_subject=max_mus,
    )
    return CoherenceReconciler(
        entity_kg=kg,
        store=kg,
        oracle=oracle,
        config=config,
        promoter=promoter,
        now=lambda: _T2,
    )


# ---------------------------------------------------------------------------
# TC-1: consistent subject — dirty cleared, oracle called for candidate pair
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_happy_path_consistent_clears_dirty() -> None:
    """Two claims, no conflict registered → tick clears dirty mark, subjects_cleared=1."""
    kg = InMemoryEntityKG()
    claim_a = _make_claim("Alice", "likes", "apples")
    claim_b = _make_claim("Alice", "likes", "oranges")
    await kg.write_claim(claim_a, evidence=_ev("ev-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b"))

    # TableOracle with no conflict sets → always consistent.
    oracle = TableOracle([])
    rec = _reconciler(kg, oracle)

    stats = await rec.tick()

    assert stats.subjects_processed == 1
    assert stats.subjects_cleared == 1
    assert stats.subjects_defeated == 0
    assert stats.subjects_failed == 0
    # oracle_calls=1 from the initial find_mus consistency check.
    assert stats.oracle_calls == 1
    # Dirty queue should be empty after the tick.
    dirty = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty) == 0


# ---------------------------------------------------------------------------
# TC-2: planted conflict → defeat committed, loser marked defeasibly-defeated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_planted_conflict_defeat_committed() -> None:
    """Two conflicting claims (oracle returns inconsistent) → defeat committed, loser defeated."""
    kg = InMemoryEntityKG()
    # Different payloads, same predicate, overlapping validity window.
    claim_a = _make_claim("Bob", "salary", "100k", valid_from=_T0)
    claim_b = _make_claim("Bob", "salary", "80k", valid_from=_T1)
    await kg.write_claim(claim_a, evidence=_ev("ev-a", source_id="src-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b", source_id="src-b"))

    oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
    rec = _reconciler(kg, oracle)

    stats = await rec.tick()

    assert stats.subjects_processed == 1
    assert stats.subjects_defeated == 1
    assert stats.subjects_failed == 0
    assert stats.oracle_calls > 0

    # claim_a (valid_from=_T0) is older; claim_b (valid_from=_T1) is newer winner in update mode.
    loser = await kg.get_claim(claim_a.id)
    assert loser is not None
    assert loser.status == "defeasibly-defeated", (
        f"Expected loser to be defeated, got {loser.status!r}"
    )

    winner = await kg.get_claim(claim_b.id)
    assert winner is not None


# ---------------------------------------------------------------------------
# TC-3: cache hit on second tick → oracle_calls=0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cache_hit_zero_oracle_calls() -> None:
    """Second tick on a re-marked subject with same conflict set uses cached adjudication."""
    kg = InMemoryEntityKG()
    claim_a = _make_claim("Carol", "role", "admin", valid_from=_T0)
    claim_b = _make_claim("Carol", "role", "guest", valid_from=_T1)
    await kg.write_claim(claim_a, evidence=_ev("ev-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b"))

    oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
    rec = _reconciler(kg, oracle)

    # First tick — resolves the conflict.
    stats1 = await rec.tick()
    assert stats1.subjects_processed == 1
    assert stats1.oracle_calls > 0

    # Re-mark the subject dirty (simulates a new write arriving).
    await kg.write_claim(claim_a, evidence=_ev("ev-a2"))

    # Second tick — the adjudication ids are cached, so candidate_pairs skips the pair.
    stats2 = await rec.tick()
    assert stats2.subjects_processed == 1
    assert stats2.subjects_cleared == 1, f"Expected subjects_cleared=1 on cache hit, got {stats2}"
    assert stats2.oracle_calls == 0, (
        f"Expected zero oracle calls on cache hit, got oracle_calls={stats2.oracle_calls}"
    )


# ---------------------------------------------------------------------------
# TC-4: escalated — both "confirmed" claims stay active
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_escalated_both_claims_stay_active() -> None:
    """Two confirmed claims with same predicate → confirmed-vs-confirmed escalation, both active."""
    kg = InMemoryEntityKG()
    claim_a = _make_claim("Dave", "city", "Paris", epistemic_type="confirmed", valid_from=_T0)
    claim_b = _make_claim("Dave", "city", "Lyon", epistemic_type="confirmed", valid_from=_T0)
    await kg.write_claim(claim_a, evidence=_ev("ev-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b"))

    oracle = TableOracle([frozenset({claim_a.id, claim_b.id})])
    rec = _reconciler(kg, oracle)

    stats = await rec.tick()

    assert stats.subjects_escalated == 1
    assert stats.subjects_defeated == 0

    # Both claims must remain active — escalation does not defeat either.
    a_after = await kg.get_claim(claim_a.id)
    b_after = await kg.get_claim(claim_b.id)
    assert a_after is not None and a_after.status == "active"
    assert b_after is not None and b_after.status == "active"


# ---------------------------------------------------------------------------
# TC-5: per-subject failure isolation
# ---------------------------------------------------------------------------


class _FailKGForKey(InMemoryEntityKG):
    """Subclass that raises on bump_dirty_attempts for a specific dirty key."""

    def __init__(self, fail_key: str) -> None:
        super().__init__()
        self._fail_key = fail_key

    async def bump_dirty_attempts(self, key: str) -> int:
        if key == self._fail_key:
            raise RuntimeError(f"Simulated failure for key {key!r}")
        return await super().bump_dirty_attempts(key)


@pytest.mark.asyncio
async def test_per_subject_failure_isolation() -> None:
    """Exception in subject A does not prevent subject B from being processed."""
    kg = _FailKGForKey.__new__(_FailKGForKey)
    InMemoryEntityKG.__init__(kg)

    claim_a = _make_claim("FailSubject", "x", "y")
    claim_b = _make_claim("GoodSubject", "p", "q")
    await kg.write_claim(claim_a, evidence=_ev("ev-a"))
    await kg.write_claim(claim_b, evidence=_ev("ev-b"))

    # Find the dirty key for claim_a's subject.
    dirty = await kg.claim_dirty_subjects(limit=10)
    subject_a_key = next(d.key for d in dirty if "failsubject" in d.subject_norm.lower())

    # Replace the kg with a version that fails for that key.
    fail_kg = _FailKGForKey(subject_a_key)
    # Share the internal state so both subjects are in the same store.
    fail_kg._claims = kg._claims
    fail_kg._evidence = kg._evidence
    fail_kg._contradictions = kg._contradictions
    fail_kg._cursors = kg._cursors
    fail_kg._dirty = kg._dirty
    fail_kg._adjudications = kg._adjudications
    fail_kg._supersedes = kg._supersedes

    oracle = TableOracle([])
    rec = _reconciler(fail_kg, oracle)

    stats = await rec.tick()

    assert stats.subjects_failed == 1
    assert stats.subjects_cleared == 1, f"Good subject should be cleared, got stats={stats}"
    assert stats.subjects_processed == 2


# ---------------------------------------------------------------------------
# TC-6: poison guard at max_attempts
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_poison_guard_at_max_attempts() -> None:
    """A dirty subject with attempts >= max_attempts → subjects_skipped=1, dirty cleared."""
    kg = InMemoryEntityKG()
    claim = _make_claim("PoisonSubject", "a", "b")
    await kg.write_claim(claim, evidence=_ev("ev1"))

    dirty = await kg.claim_dirty_subjects(limit=1)
    assert len(dirty) == 1
    key = dirty[0].key

    # Manually set attempts to max_attempts.
    max_attempts = 3
    for _ in range(max_attempts):
        await kg.bump_dirty_attempts(key)

    # Verify attempts == max_attempts.
    dirty_after = await kg.claim_dirty_subjects(limit=1)
    assert dirty_after[0].attempts == max_attempts

    oracle = TableOracle([])
    rec = _reconciler(kg, oracle, max_attempts=max_attempts)

    stats = await rec.tick()

    assert stats.subjects_skipped == 1
    assert stats.subjects_failed == 0

    # Adjudication with escalation_reason="max-attempts-exceeded" was written.
    adj_ids = await kg.adjudication_ids_for_subject("agent", dirty_after[0].subject_norm)
    assert len(adj_ids) == 1

    # Dirty mark should be cleared (epoch guard passed since no new write occurred).
    dirty_final = await kg.claim_dirty_subjects(limit=10)
    assert len(dirty_final) == 0


# ---------------------------------------------------------------------------
# TC-7: subject-too-large
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_subject_too_large() -> None:
    """Writing max_claims_per_subject + 1 claims → subjects_skipped=1 (subject-too-large)."""
    max_claims = 4
    kg = InMemoryEntityKG()

    # Write max_claims + 1 distinct claims for the same subject (different payloads).
    for i in range(max_claims + 1):
        claim = _make_claim("BigSubject", "has", f"item-{i}")
        await kg.write_claim(claim, evidence=_ev(f"ev-{i}"))

    oracle = TableOracle([])
    rec = _reconciler(kg, oracle, max_claims=max_claims)

    stats = await rec.tick()

    assert stats.subjects_skipped == 1
    assert stats.subjects_failed == 0


# ---------------------------------------------------------------------------
# TC-8: promoter invoked on active claims
# ---------------------------------------------------------------------------


class _CountingPromoter(ScopePromoter):
    """Minimal promoter stub that counts calls and returns a fixed promotion count."""

    def __init__(self, promotions_per_call: int = 1) -> None:
        # Bypass ScopePromoter.__init__ — this is a direct protocol stub for testing.
        self.calls: int = 0
        self._promotions = promotions_per_call

    async def promote_for_subject(self, claims: Sequence[object]) -> int:
        self.calls += 1
        return self._promotions


@pytest.mark.asyncio
async def test_promoter_invoked_on_active_claims() -> None:
    """Promoter is called once per subject processed (consistent or not)."""
    kg = InMemoryEntityKG()
    claim = _make_claim("Emma", "job", "engineer")
    await kg.write_claim(claim, evidence=_ev("ev1"))

    oracle = TableOracle([])
    promoter = _CountingPromoter(promotions_per_call=1)
    rec = _reconciler(kg, oracle, promoter=promoter)

    stats = await rec.tick()

    assert promoter.calls == 1
    assert stats.promotions == 1


# ---------------------------------------------------------------------------
# TC-9: no promoter → no error, stats.promotions stays 0
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_promoter_no_error() -> None:
    """CoherenceReconciler with promoter=None succeeds and stats.promotions=0."""
    kg = InMemoryEntityKG()
    claim = _make_claim("Frank", "hobby", "cycling")
    await kg.write_claim(claim, evidence=_ev("ev1"))

    oracle = TableOracle([])
    rec = _reconciler(kg, oracle, promoter=None)

    stats = await rec.tick()

    assert stats.subjects_processed == 1
    assert stats.promotions == 0
    assert stats.subjects_failed == 0
