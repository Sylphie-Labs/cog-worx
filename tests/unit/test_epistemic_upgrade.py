"""Unit tests for cogworx.coherence.upgrade — no I/O, no model calls."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import EpistemicType
from cogworx.coherence.upgrade import (
    epistemic_upgrade,
)
from cogworx.knowledge.evidence import EvidenceEvent, EvidenceType, Polarity
from cogworx.substrate.coherence import (
    DirtyKey,
    DirtySubject,
    ReconciliationOutcome,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=UTC)


def _evidence(
    *,
    etype: EvidenceType = "tool_proof",
    polarity: Polarity = "+",
    source_id: str = "source:tool:unit-test",
) -> EvidenceEvent:
    return EvidenceEvent(
        id="ev-test",
        type=etype,
        polarity=polarity,
        source_id=source_id,
        source_authority=1.0,
        base_weight=3.0,
        recorded_at=_NOW,
    )


# ---------------------------------------------------------------------------
# Minimal fake CoherenceStore
# ---------------------------------------------------------------------------


class _FakeCoherenceStore:
    """Minimal CoherenceStore stub for upgrade unit tests.

    ``current_level``: the epistemic level returned by current_epistemic_level.
    ``apply_return``: the bool returned by apply_epistemic_upgrade (default True).
    ``apply_calls``: list of kwargs passed to apply_epistemic_upgrade for assertion.
    """

    def __init__(
        self,
        *,
        current_level: EpistemicType | None,
        apply_return: bool = True,
    ) -> None:
        self._current_level: EpistemicType | None = current_level
        self._apply_return = apply_return
        self.apply_calls: list[dict[str, object]] = []

    async def current_epistemic_level(self, claim_id: str) -> EpistemicType | None:
        return self._current_level

    async def apply_epistemic_upgrade(
        self,
        claim_id: str,
        *,
        new_level: EpistemicType,
        evidence: EvidenceEvent,
        actor: str,
        recorded_at: datetime,
    ) -> bool:
        self.apply_calls.append(
            {
                "claim_id": claim_id,
                "new_level": new_level,
                "actor": actor,
                "recorded_at": recorded_at,
            }
        )
        return self._apply_return

    # Remaining CoherenceStore methods — not exercised by upgrade tests.

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
        return 0


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_monotonic_upgrade_applies() -> None:
    """inference → observation should apply and return applied=True."""
    store = _FakeCoherenceStore(current_level="inference")
    ev = _evidence()
    result = await epistemic_upgrade(
        store, "claim-1", new_level="observation", evidence=ev, actor="test"
    )
    assert result.applied is True
    assert result.claim_id == "claim-1"
    assert result.to_level == "observation"
    assert result.from_level == "inference"
    assert len(store.apply_calls) == 1


@pytest.mark.asyncio
async def test_same_level_is_noop() -> None:
    """observation → observation should return applied=False without calling the store."""
    store = _FakeCoherenceStore(current_level="observation")
    ev = _evidence()
    result = await epistemic_upgrade(
        store, "claim-2", new_level="observation", evidence=ev, actor="test"
    )
    assert result.applied is False
    assert result.from_level == "observation"
    assert result.to_level == "observation"


@pytest.mark.asyncio
async def test_explicit_downgrade_raises() -> None:
    """Requesting a lower level when current is observation → ValueError."""
    store = _FakeCoherenceStore(current_level="observation")
    # inference has no UPGRADE_ELIGIBLE_EVIDENCE entry — ValueError fires at eligibility check.
    ev = _evidence()
    with pytest.raises(ValueError):
        await epistemic_upgrade(store, "claim-3", new_level="inference", evidence=ev, actor="test")


@pytest.mark.asyncio
async def test_ineligible_evidence_raises() -> None:
    """evidence.type='extraction' is not in UPGRADE_ELIGIBLE_EVIDENCE['confirmed'] → ValueError."""
    store = _FakeCoherenceStore(current_level="inference")
    ev = _evidence(etype="extraction")
    with pytest.raises(ValueError, match="not eligible"):
        await epistemic_upgrade(store, "claim-4", new_level="confirmed", evidence=ev, actor="test")


@pytest.mark.asyncio
async def test_negative_polarity_raises() -> None:
    """Negative polarity evidence cannot trigger an upgrade → ValueError."""
    store = _FakeCoherenceStore(current_level="inference")
    ev = _evidence(polarity="-")
    with pytest.raises(ValueError, match="polarity"):
        await epistemic_upgrade(
            store, "claim-5", new_level="observation", evidence=ev, actor="test"
        )


@pytest.mark.asyncio
async def test_idempotent_rerun() -> None:
    """When apply_epistemic_upgrade returns False the UpgradeResult is applied=False."""
    # Simulates the case where a concurrent upgrade already landed at the same level.
    store = _FakeCoherenceStore(current_level="inference", apply_return=False)
    ev = _evidence()
    result = await epistemic_upgrade(
        store, "claim-6", new_level="observation", evidence=ev, actor="test"
    )
    assert result.applied is False
    assert result.to_level == "observation"


@pytest.mark.asyncio
async def test_store_not_called_on_noop() -> None:
    """apply_epistemic_upgrade is NOT called when the pre-check determines a no-op."""
    store = _FakeCoherenceStore(current_level="observation")
    ev = _evidence()
    await epistemic_upgrade(store, "claim-7", new_level="observation", evidence=ev, actor="test")
    assert store.apply_calls == [], "store should not be called on a same-level no-op"


@pytest.mark.asyncio
async def test_confirmed_upgrade_requires_first_hand() -> None:
    """evidence.type='extraction' for 'confirmed' → ValueError (extraction is second-hand)."""
    store = _FakeCoherenceStore(current_level="observation")
    ev = _evidence(etype="extraction")
    with pytest.raises(ValueError, match="not eligible"):
        await epistemic_upgrade(store, "claim-8", new_level="confirmed", evidence=ev, actor="test")


@pytest.mark.asyncio
async def test_confirmed_upgrade_from_observation_applies() -> None:
    """observation → confirmed with tool_proof → applied=True."""
    store = _FakeCoherenceStore(current_level="observation")
    ev = _evidence(etype="tool_proof")
    result = await epistemic_upgrade(
        store, "claim-9", new_level="confirmed", evidence=ev, actor="test"
    )
    assert result.applied is True
    assert result.from_level == "observation"
    assert result.to_level == "confirmed"


@pytest.mark.asyncio
async def test_attestation_eligible_for_observation() -> None:
    """attestation evidence is eligible for observation upgrade."""
    store = _FakeCoherenceStore(current_level="inference")
    ev = _evidence(etype="attestation")
    result = await epistemic_upgrade(
        store, "claim-10", new_level="observation", evidence=ev, actor="test"
    )
    assert result.applied is True


@pytest.mark.asyncio
async def test_observation_above_confirmed_raises() -> None:
    """confirmed → observation is a downgrade → ValueError."""
    store = _FakeCoherenceStore(current_level="confirmed")
    ev = _evidence()
    with pytest.raises(ValueError):
        await epistemic_upgrade(
            store, "claim-11", new_level="observation", evidence=ev, actor="test"
        )


@pytest.mark.asyncio
async def test_inference_to_observation_upgrade_succeeds() -> None:
    """Regression for inverted-rank bug: inference->observation must succeed."""
    store = _FakeCoherenceStore(current_level="inference")
    ev = _evidence(etype="tool_proof")
    result = await epistemic_upgrade(
        store, "claim-reg-1", new_level="observation", evidence=ev, actor="test"
    )
    assert result.applied is True, (
        "inference->observation upgrade returned applied=False; "
        "likely the rank dict is inverted (inference=1 > observation=0)"
    )
    assert result.from_level == "inference"
    assert result.to_level == "observation"
    assert len(store.apply_calls) == 1


@pytest.mark.asyncio
async def test_observation_to_confirmed_upgrade_succeeds() -> None:
    """observation->confirmed with tool_proof must succeed."""
    store = _FakeCoherenceStore(current_level="observation")
    ev = _evidence(etype="tool_proof")
    result = await epistemic_upgrade(
        store, "claim-reg-2", new_level="confirmed", evidence=ev, actor="test"
    )
    assert result.applied is True, (
        "observation->confirmed upgrade returned applied=False; "
        "check EPISTEMIC_RANK ordering in the store"
    )
    assert result.from_level == "observation"
    assert result.to_level == "confirmed"
    assert len(store.apply_calls) == 1
