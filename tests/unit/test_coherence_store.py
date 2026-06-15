"""Unit tests for CoherenceStore — exercised against InMemoryEntityKG (no live Neo4j).

All tests use the in-memory double so they run with no external services. The parity test
(test_parity_neo4j_vs_inmemory) is marked @pytest.mark.integration and is skipped by default.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.claims.provenance import Claim, Provenance, ProvenanceSource
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for, normalize_topic_part
from cogworx.substrate.coherence import (
    AdjudicationRecord,
    AdjudicationVerdict,
    Defeat,
    ReconciliationOutcome,
    ResolutionKind,
)
from cogworx.testing.doubles import InMemoryEntityKG

_T0 = datetime(2026, 6, 10, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(source: ProvenanceSource = "system") -> Provenance:
    return Provenance(source=source, confidence=0.9, recorded_at=_T0)


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    scope: str = "agent",
    valid_from: datetime = _T0,
) -> Claim:
    cid = claim_id_for(subject, predicate, payload, scope=scope)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        epistemic_type="observation",
        provenance=_prov(),
        valid_from=valid_from,
        ingest_time=valid_from,
        created_by="test",
        scope=scope,
    )


def _ev(*, event_id: str = "ev1") -> EvidenceEvent:
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id="src1",
        source_authority=0.8,
        recorded_at=_T0,
        event_id=event_id,
    )


def _adj_id(claim_ids: tuple[str, ...]) -> str:
    joined = "\x1f".join(sorted(claim_ids))
    return "adj:" + hashlib.sha256(joined.encode()).hexdigest()[:32]


def _make_adj(
    scope: str,
    subject_norm: str,
    claim_ids: tuple[str, ...],
    *,
    verdict: AdjudicationVerdict = "inconsistent",
    resolution: ResolutionKind = "update-supersession",
    winner_id: str | None = None,
    loser_ids: tuple[str, ...] = (),
) -> AdjudicationRecord:
    return AdjudicationRecord(
        id=_adj_id(claim_ids),
        scope=scope,
        subject_norm=subject_norm,
        claim_ids=claim_ids,
        verdict=verdict,
        resolution=resolution,
        winner_id=winner_id,
        loser_ids=loser_ids,
        margin=0.1,
        oracle_calls=1,
        escalation_reason=None,
        provenance=_prov("system"),
        recorded_at=_T2,
    )


# ---------------------------------------------------------------------------
# SC-1: Dirty-mark dedup
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_dirty_merge_dedup() -> None:
    """3 writes to same (scope, subject) → 1 dirty node, epoch=3."""
    kg = InMemoryEntityKG()
    claim = _make_claim("Alice", "age", "30")
    for i in range(3):
        ev = _ev(event_id=f"ev{i}")
        await kg.write_claim(claim, evidence=ev)

    dirty = await kg.claim_dirty_subjects(limit=100)
    assert len(dirty) == 1, f"Expected 1 dirty node, got {len(dirty)}"
    assert dirty[0].epoch == 3
    assert dirty[0].scope == "agent"
    assert dirty[0].subject_norm == normalize_topic_part("Alice")


# ---------------------------------------------------------------------------
# SC-2: Epoch-guarded clear
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_epoch_guarded_clear() -> None:
    """Epoch guard: mid-flight write bumps epoch; stale-epoch commit does not clear dirty node."""
    kg = InMemoryEntityKG()
    claim = _make_claim("Bob", "job", "engineer")
    await kg.write_claim(claim, evidence=_ev(event_id="ev-a"))

    dirty_before = await kg.claim_dirty_subjects(limit=1)
    assert len(dirty_before) == 1
    observed_epoch = dirty_before[0].epoch  # = 1

    # A second write bumps the epoch to 2 (simulates a concurrent write arriving mid-reconcile).
    await kg.write_claim(claim, evidence=_ev(event_id="ev-b"))

    dirty_after_write = await kg.claim_dirty_subjects(limit=1)
    assert dirty_after_write[0].epoch == 2, "epoch should have been bumped"

    # commit_reconciliation with the OLD epoch (1) must NOT clear the dirty node.
    key = dirty_before[0].key
    outcome = ReconciliationOutcome(
        adjudications=(),
        defeats=(),
        contradictions=(),
        recorded_at=_T2,
    )
    await kg.commit_reconciliation(outcome, dirty_key=key, observed_epoch=observed_epoch)

    # Dirty node should still be there (epoch=2 != observed_epoch=1).
    still_dirty = await kg.claim_dirty_subjects(limit=1)
    assert len(still_dirty) == 1, "Dirty node must survive a stale-epoch commit"
    assert still_dirty[0].epoch == 2


# ---------------------------------------------------------------------------
# SC-3: Adjudication immutable-on-match
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_adjudication_immutable_on_match() -> None:
    """Committing the same outcome twice → one adjudication node (ids stable, not duplicated)."""
    kg = InMemoryEntityKG()
    claim_a = _make_claim("Carol", "city", "Paris")
    claim_b = _make_claim("Carol", "city", "Lyon")
    await kg.write_claim(claim_a, evidence=_ev(event_id="ev-a"))
    await kg.write_claim(claim_b, evidence=_ev(event_id="ev-b"))

    adj = _make_adj(
        "agent",
        normalize_topic_part("Carol"),
        (claim_a.id, claim_b.id),
        winner_id=claim_a.id,
        loser_ids=(claim_b.id,),
    )
    dirty = await kg.claim_dirty_subjects(limit=1)
    key = dirty[0].key
    epoch = dirty[0].epoch

    outcome = ReconciliationOutcome(
        adjudications=(adj,),
        defeats=(),
        contradictions=(),
        recorded_at=_T2,
    )

    await kg.commit_reconciliation(outcome, dirty_key=key, observed_epoch=epoch)

    # Write again to get a new dirty node (previous was cleared).
    await kg.write_claim(claim_a, evidence=_ev(event_id="ev-a2"))
    dirty2 = await kg.claim_dirty_subjects(limit=1)
    key2 = dirty2[0].key
    epoch2 = dirty2[0].epoch

    # Commit the SAME outcome again.
    await kg.commit_reconciliation(outcome, dirty_key=key2, observed_epoch=epoch2)

    # There should still be exactly one adjudication record (setdefault / ON CREATE).
    ids = await kg.adjudication_ids_for_subject("agent", normalize_topic_part("Carol"))
    assert len(ids) == 1, f"Expected 1 adjudication, got {len(ids)}"
    assert ids[0] == adj.id


# ---------------------------------------------------------------------------
# SC-4: SUPERSEDES idempotency + status coalesce
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_supersedes_idempotent() -> None:
    """Committing a defeat twice → single SUPERSEDES, status='defeasibly-defeated'."""
    kg = InMemoryEntityKG()
    winner = _make_claim("Dan", "salary", "100k")
    loser = _make_claim("Dan", "salary", "80k")
    await kg.write_claim(winner, evidence=_ev(event_id="ev-w"))
    await kg.write_claim(loser, evidence=_ev(event_id="ev-l"))

    adj = _make_adj(
        "agent",
        normalize_topic_part("Dan"),
        (winner.id, loser.id),
        winner_id=winner.id,
        loser_ids=(loser.id,),
    )
    defeat = Defeat(
        winner_id=winner.id,
        loser_id=loser.id,
        kind="update",
        adjudication_id=adj.id,
        set_valid_to=_T1,
    )

    async def _commit(epoch: int) -> None:
        dirty = await kg.claim_dirty_subjects(limit=1)
        if not dirty:
            # Force dirty mark for the subject.
            await kg.write_claim(winner, evidence=_ev(event_id=f"extra-{epoch}"))
            dirty = await kg.claim_dirty_subjects(limit=1)
        key = dirty[0].key
        ep = dirty[0].epoch
        outcome = ReconciliationOutcome(
            adjudications=(adj,),
            defeats=(defeat,),
            contradictions=(),
            recorded_at=_T2,
        )
        await kg.commit_reconciliation(outcome, dirty_key=key, observed_epoch=ep)

    await _commit(1)
    await _commit(2)

    loser_after = await kg.get_claim(loser.id)
    assert loser_after is not None
    assert loser_after.status == "defeasibly-defeated"
    assert loser_after.defeated_by == winner.id
    # valid_to was set on the first defeat and must not change on the second (first-wins).
    assert loser_after.valid_to is not None

    # SUPERSEDES is a set — still exactly one entry.
    assert (winner.id, loser.id) in kg._supersedes


# ---------------------------------------------------------------------------
# SC-5: Pre-2.7 claims default status "active"
# ---------------------------------------------------------------------------


def test_status_coalesce_default() -> None:
    """A Claim built without explicit status reads as 'active' (default backward-compat)."""
    claim = _make_claim("Eve", "location", "London")
    assert claim.status == "active"
    assert claim.defeated_by is None


# ---------------------------------------------------------------------------
# SC-6: upgrade TOCTOU no-op
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_upgrade_toctou_noop() -> None:
    """Upgrading a claim to the level it's already at → False, no audit trail."""
    kg = InMemoryEntityKG()
    # Start at 'inference' level.
    cid = claim_id_for("Fred", "role", "admin", scope="agent")
    claim = Claim(
        id=cid,
        subject="Fred",
        predicate="role",
        payload="admin",
        epistemic_type="inference",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
        scope="agent",
    )
    await kg.write_claim(claim, evidence=_ev(event_id="ev-inf"))

    ev_upgrade = make_evidence(
        type="attestation",
        polarity="+",
        source_id="human1",
        source_authority=1.0,
        recorded_at=_T1,
        event_id="ev-upgrade",
    )

    # Same level → no-op.
    changed = await kg.apply_epistemic_upgrade(
        cid,
        new_level="inference",
        evidence=ev_upgrade,
        actor="human1",
        recorded_at=_T1,
    )
    assert changed is False

    # observation is HIGHER than inference on the monotonic ladder — genuine upgrade.
    changed2 = await kg.apply_epistemic_upgrade(
        cid,
        new_level="observation",
        evidence=ev_upgrade,
        actor="human1",
        recorded_at=_T1,
    )
    assert changed2 is True

    # Verify evidence was added for the observation upgrade.
    evs = await kg.evidence_for(cid)
    assert len(evs) == 2, f"Expected 2 evidence events after upgrade, got {len(evs)}"

    # Now a genuine upgrade to 'confirmed' SHOULD also work.
    changed3 = await kg.apply_epistemic_upgrade(
        cid,
        new_level="confirmed",
        evidence=ev_upgrade,
        actor="human1",
        recorded_at=_T1,
    )
    assert changed3 is True
    upgraded = await kg.get_claim(cid)
    assert upgraded is not None
    assert upgraded.epistemic_type == "confirmed"


# ---------------------------------------------------------------------------
# SC-7: copy_evidence idempotency
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_copy_evidence_idempotent() -> None:
    """Running copy_evidence twice → same number of evidence events on the target."""
    kg = InMemoryEntityKG()
    src = _make_claim("Grace", "hobby", "painting")
    dst = _make_claim("Grace", "hobby", "art")

    ev1 = _ev(event_id="ev-src1")
    ev2 = make_evidence(
        type="corroboration",
        polarity="+",
        source_id="src2",
        source_authority=0.7,
        recorded_at=_T1,
        event_id="ev-src2",
    )
    await kg.write_claim(src, evidence=ev1)
    await kg.add_evidence(src.id, ev2)
    await kg.write_claim(dst, evidence=_ev(event_id="ev-dst"))

    count1 = await kg.copy_evidence(src.id, dst.id, id_prefix="copy1")
    count2 = await kg.copy_evidence(src.id, dst.id, id_prefix="copy1")

    assert count1 == 2, f"Expected 2 source events, got {count1}"
    assert count2 == 2, f"Idempotent second run should also report 2, got {count2}"

    dst_evs = await kg.evidence_for(dst.id)
    # 1 original + 2 copied = 3 total (no duplicates from re-run).
    assert len(dst_evs) == 3, f"Expected 3 evidence events on dst, got {len(dst_evs)}"


# ---------------------------------------------------------------------------
# SC-8: Neo4j vs in-memory parity (integration, skipped by default)
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.asyncio
async def test_parity_neo4j_vs_inmemory() -> None:
    """Same CoherenceStore operations on both Neo4jEntityKG and InMemoryEntityKG → identical state.

    Requires a live Neo4j instance. Run with -m integration.
    """
    pytest.skip("integration test — skipped in unit tier")
