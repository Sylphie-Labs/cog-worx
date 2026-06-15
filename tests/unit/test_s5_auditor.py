"""Unit tests for the S5 substrate auditor and RecordingEntityKG (Phase 2 gate spike U1).

asyncio_mode = "auto" (pyproject.toml) — no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from cogworx.knowledge.evidence import EvidenceEvent
from cogworx.testing.doubles import InMemoryEntityKG
from cogworx.testing.invariants import (
    InvariantViolation,
    RecordingEntityKG,
    assert_s5_substrate_invariants,
)
from cogworx.testing.recall_fixtures import build_gate_corpus


async def test_s5_auditor_passes_good_kg() -> None:
    """S5 auditor passes when all claims have complete provenance."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    await build_gate_corpus(recording_kg)
    # Should not raise
    await assert_s5_substrate_invariants(recording_kg, recording_kg.recorded_claim_ids)


async def test_s5_auditor_raises_on_zero_evidence() -> None:
    """S5 auditor raises InvariantViolation when a claim has no evidence."""

    class EvidenceDropper(InMemoryEntityKG):
        async def evidence_for(self, claim_id: str) -> Sequence[EvidenceEvent]:
            return []

    recording_kg = RecordingEntityKG(EvidenceDropper())
    await build_gate_corpus(recording_kg)

    with pytest.raises(InvariantViolation, match="no evidence"):
        await assert_s5_substrate_invariants(recording_kg, recording_kg.recorded_claim_ids)


async def test_s5_auditor_raises_on_missing_claim() -> None:
    """S5 auditor raises InvariantViolation when a recorded claim id doesn't exist."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    recording_kg.recorded_claim_ids.add("claim:nonexistent")

    with pytest.raises(InvariantViolation, match="not found"):
        await assert_s5_substrate_invariants(recording_kg, recording_kg.recorded_claim_ids)


async def test_recording_kg_captures_write_claim() -> None:
    """RecordingEntityKG records ids from write_claim."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    assert len(recording_kg.recorded_claim_ids) == 0

    corpus = await build_gate_corpus(recording_kg)

    assert len(recording_kg.recorded_claim_ids) > 0
    for name, cid in corpus.claim_ids.items():
        assert cid in recording_kg.recorded_claim_ids, f"{name} ({cid!r}) not in recorded_claim_ids"


async def test_recording_kg_is_empty_before_writes() -> None:
    """RecordingEntityKG starts with no recorded ids."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    assert recording_kg.recorded_claim_ids == set()


async def test_recording_kg_delegates_get_claim() -> None:
    """RecordingEntityKG.get_claim delegates correctly to the inner KG."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    corpus = await build_gate_corpus(recording_kg)

    for cid in corpus.claim_ids.values():
        claim = await recording_kg.get_claim(cid)
        assert claim is not None


async def test_recording_kg_delegates_evidence_for() -> None:
    """RecordingEntityKG.evidence_for delegates correctly to the inner KG."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    corpus = await build_gate_corpus(recording_kg)

    for cid in corpus.claim_ids.values():
        ev = await recording_kg.evidence_for(cid)
        assert len(ev) >= 1


async def test_s5_auditor_records_all_corpus_ids() -> None:
    """All corpus claim ids from build_gate_corpus are recorded by RecordingEntityKG."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    corpus = await build_gate_corpus(recording_kg)

    expected_names = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "DX"] + [
        f"D{i}" for i in range(1, 21)
    ]
    for name in expected_names:
        cid = corpus.claim_ids[name]
        assert cid in recording_kg.recorded_claim_ids, f"{name} ({cid!r}) not recorded"


async def test_s5_auditor_raises_on_bad_epistemic_type() -> None:
    """S5 auditor raises InvariantViolation when a claim has an unrecognised epistemic_type."""
    from cogworx.claims.provenance import Claim, Provenance
    from cogworx.knowledge.evidence import make_evidence
    from cogworx.knowledge.identity import claim_id_for
    from cogworx.testing.recall_fixtures import T_NOW

    inner = InMemoryEntityKG()
    # Build a minimal valid claim and write it normally first so identity discipline passes.
    cid = claim_id_for("test:subj", "test:pred", "testpayload", scope="agent")
    claim = Claim(
        id=cid,
        subject="test:subj",
        predicate="test:pred",
        payload="testpayload",
        epistemic_type="inference",
        provenance=Provenance(source="human", confidence=0.9, recorded_at=T_NOW),
        valid_from=T_NOW,
        ingest_time=T_NOW,
        created_by="test-s5",
        scope="agent",
    )
    ev = make_evidence(
        type="attestation",
        polarity="+",
        source_id="source:test",
        source_authority=0.9,
        recorded_at=T_NOW,
    )
    await inner.write_claim(claim, evidence=ev)

    # Now subclass to report a bad epistemic_type on get_claim.
    class BadEpistemicKG(InMemoryEntityKG):
        async def get_claim(self, claim_id: str) -> Claim | None:
            c = await super().get_claim(claim_id)
            if c is None:
                return None
            # Construct a copy with an invalid epistemic_type using model_copy to bypass Literal
            return c.model_copy(update={"epistemic_type": "unknown_level"})

    bad_inner = BadEpistemicKG()
    await bad_inner.write_claim(claim, evidence=ev)
    recording_kg = RecordingEntityKG(bad_inner)
    recording_kg.recorded_claim_ids.add(cid)

    with pytest.raises(InvariantViolation, match="epistemic_type"):
        await assert_s5_substrate_invariants(recording_kg, [cid])


async def test_s5_auditor_empty_claim_ids_passes() -> None:
    """S5 auditor with an empty iterable does not raise."""
    recording_kg = RecordingEntityKG(InMemoryEntityKG())
    await assert_s5_substrate_invariants(recording_kg, [])
