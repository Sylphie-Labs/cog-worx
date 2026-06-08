"""S5 contract tests: provenance/epistemic typing round-trips, bounds, immutability."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from cogworx.claims.provenance import Claim, Provenance


def _provenance() -> Provenance:
    return Provenance(
        source="human",
        source_ref="turn-1",
        confidence=0.9,
        evidence=("claim-0",),
        recorded_at=datetime(2026, 6, 8, tzinfo=UTC),
    )


def test_claim_round_trips() -> None:
    claim = Claim(
        id="claim-1",
        subject="user",
        predicate="prefers",
        payload="dark mode",
        epistemic_type="observation",
        provenance=_provenance(),
        valid_from=datetime(2026, 6, 8, tzinfo=UTC),
        ingest_time=datetime(2026, 6, 8, tzinfo=UTC),
        created_by="agent-a",
    )
    assert Claim.model_validate(claim.model_dump()) == claim


def test_confidence_out_of_range_raises() -> None:
    with pytest.raises(ValidationError):
        Provenance(source="sensor", confidence=1.5, recorded_at=datetime(2026, 6, 8, tzinfo=UTC))


def test_frozen_rejects_mutation() -> None:
    prov = _provenance()
    with pytest.raises(ValidationError):
        prov.confidence = 0.1  # type: ignore[misc]
