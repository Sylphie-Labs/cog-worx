"""Unit tests for ModelConsistencyOracle, TableOracle, NoisyOracle (Pod 2.7 U3).

All tests are pure async — no subprocess, no substrate, no real model calls (S1).  Uses
ReplayModel from the Test Kit for the ModelConsistencyOracle tests.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.coherence.oracle import (
    ModelConsistencyOracle,
    OracleProtocolError,
)
from cogworx.model.base import ModelCapabilities, ModelResponse
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.fake_oracle import NoisyOracle, TableOracle

_T0 = datetime(2026, 6, 10, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _claim(cid: str) -> Claim:
    return Claim(
        id=cid,
        subject=cid,
        predicate="is",
        payload="value",
        epistemic_type="inference",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )


def _text_model(text: str) -> ReplayModel:
    """ReplayModel with structured_output=False so the plain token-match path is exercised."""
    return ReplayModel(
        [ModelResponse(text=text, model_id="replay-oracle", finish_reason="stop")],
        capabilities=ModelCapabilities(structured_output=False),
    )


# ---------------------------------------------------------------------------
# ModelConsistencyOracle — parse paths
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_parse_consistent() -> None:
    model = _text_model("CONSISTENT")
    oracle = ModelConsistencyOracle(model)
    answer = await oracle.check([_claim("c1"), _claim("c2")])
    assert answer.consistent is True
    assert answer.model_ref == "replay-oracle"


@pytest.mark.asyncio
async def test_parse_inconsistent() -> None:
    model = _text_model("INCONSISTENT")
    oracle = ModelConsistencyOracle(model)
    answer = await oracle.check([_claim("c1"), _claim("c2")])
    assert answer.consistent is False


@pytest.mark.asyncio
async def test_parse_strips_whitespace() -> None:
    model = _text_model("  consistent  ")
    oracle = ModelConsistencyOracle(model)
    answer = await oracle.check([_claim("c1")])
    assert answer.consistent is True


@pytest.mark.asyncio
async def test_garbled_raises_protocol_error() -> None:
    model = _text_model("maybe")
    oracle = ModelConsistencyOracle(model)
    with pytest.raises(OracleProtocolError):
        await oracle.check([_claim("c1")])


# ---------------------------------------------------------------------------
# TableOracle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_table_oracle_no_conflict() -> None:
    oracle = TableOracle([frozenset({"x", "y"})])
    answer = await oracle.check([_claim("a"), _claim("b")])
    assert answer.consistent is True


@pytest.mark.asyncio
async def test_table_oracle_conflict_triggered() -> None:
    c1, c2 = _claim("c1"), _claim("c2")
    oracle = TableOracle([frozenset({c1.id, c2.id})])
    answer = await oracle.check([c1, c2])
    assert answer.consistent is False


@pytest.mark.asyncio
async def test_table_oracle_call_count() -> None:
    oracle = TableOracle([])
    c = _claim("c1")
    await oracle.check([c])
    await oracle.check([c])
    await oracle.check([c])
    assert oracle.call_count == 3


# ---------------------------------------------------------------------------
# NoisyOracle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_noisy_oracle_deterministic() -> None:
    """Same seed and flip_rate must produce the identical sequence of answers."""
    claims = [_claim(f"c{i}") for i in range(6)]
    inner_a = TableOracle([])
    inner_b = TableOracle([])

    noisy_a = NoisyOracle(inner_a, flip_rate=0.5, seed=42)
    noisy_b = NoisyOracle(inner_b, flip_rate=0.5, seed=42)

    answers_a: list[bool] = []
    answers_b: list[bool] = []
    for c in claims:
        a = await noisy_a.check([c])
        b = await noisy_b.check([c])
        answers_a.append(a.consistent)
        answers_b.append(b.consistent)

    assert answers_a == answers_b
