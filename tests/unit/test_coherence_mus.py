"""Unit tests for find_mus / QuickXplain (Pod 2.7 U3).

All tests are pure async — no substrate, no real model calls (S1).  Uses TableOracle and
NoisyOracle from the Test Kit.  The budget bound ≤ 2k·⌈log₂(N)⌉ + 2 is enforced as an assertion
on oracle_calls (where k = MUS size, N = input set size).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, Provenance
from cogworx.coherence.mus import BudgetExceeded, CallBudget, find_mus
from cogworx.coherence.oracle import OracleAnswer
from cogworx.testing.fake_oracle import TableOracle

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


def _claims(n: int) -> list[Claim]:
    """Return n claims with deterministic ids c000 … c{n-1:03d}."""
    return [_claim(f"c{i:03d}") for i in range(n)]


def _budget_bound(k: int, n: int) -> int:
    """Upper-bound oracle calls for a k-element MUS in an N-element set."""
    return 2 * k * math.ceil(math.log2(max(n, 2))) + 2


def _plant_conflict(claims: list[Claim], conflict_ids: frozenset[str]) -> TableOracle:
    return TableOracle([conflict_ids])


# ---------------------------------------------------------------------------
# 1. Consistent set — one oracle call, empty MUS
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_consistent_set_one_call() -> None:
    claims = _claims(5)
    oracle = TableOracle([])  # no conflicts
    result = await find_mus(claims, oracle, budget=CallBudget(64))
    assert result.mus == ()
    assert result.oracle_calls == 1
    assert result.verified is True


# ---------------------------------------------------------------------------
# 2. k=2, N=8
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mus_k2_n8() -> None:
    claims = _claims(8)
    conflict = frozenset({claims[2].id, claims[5].id})
    oracle = _plant_conflict(claims, conflict)
    result = await find_mus(claims, oracle, budget=CallBudget(256))

    assert set(result.mus) == conflict
    assert result.verified is True
    bound = _budget_bound(2, 8)  # 2*2*3+2 = 14
    assert result.oracle_calls <= bound, f"oracle_calls={result.oracle_calls} exceeds bound={bound}"


# ---------------------------------------------------------------------------
# 3. k=3, N=16
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mus_k3_n16() -> None:
    claims = _claims(16)
    conflict = frozenset({claims[1].id, claims[7].id, claims[14].id})
    oracle = _plant_conflict(claims, conflict)
    result = await find_mus(claims, oracle, budget=CallBudget(256))

    assert set(result.mus) == conflict
    assert result.verified is True
    bound = _budget_bound(3, 16)  # 2*3*4+2 = 26
    assert result.oracle_calls <= bound, f"oracle_calls={result.oracle_calls} exceeds bound={bound}"


# ---------------------------------------------------------------------------
# 4. k=4, N=32
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mus_k4_n32() -> None:
    claims = _claims(32)
    conflict = frozenset({claims[0].id, claims[9].id, claims[18].id, claims[30].id})
    oracle = _plant_conflict(claims, conflict)
    result = await find_mus(claims, oracle, budget=CallBudget(256))

    assert set(result.mus) == conflict
    assert result.verified is True
    bound = _budget_bound(4, 32)  # 2*4*5+2 = 42
    assert result.oracle_calls <= bound, f"oracle_calls={result.oracle_calls} exceeds bound={bound}"


# ---------------------------------------------------------------------------
# 5. verified=False when the verification call is flipped
# ---------------------------------------------------------------------------


class _FlipLastOracle:
    """Wraps TableOracle and flips the answer on a specified call number (1-based)."""

    def __init__(self, inner: TableOracle, flip_call: int) -> None:
        self._inner = inner
        self._flip_call = flip_call

    async def check(self, claims: Sequence[Claim]) -> OracleAnswer:
        answer = await self._inner.check(claims)
        if self._inner.call_count == self._flip_call:
            return OracleAnswer(consistent=not answer.consistent, model_ref=answer.model_ref)
        return answer


@pytest.mark.asyncio
async def test_verification_false_on_noisy_verify() -> None:
    claims = _claims(4)
    conflict = frozenset({claims[0].id, claims[1].id})
    inner = TableOracle([conflict])

    # We need to flip the verify call (the last call find_mus makes).
    # We don't know the exact call number ahead of time — run once with the plain oracle to find
    # total calls, then re-run with the flip on that call number.
    plain_result = await find_mus(claims, inner, budget=CallBudget(256))
    total_calls = plain_result.oracle_calls

    # Reset and re-run flipping the final (verification) call.
    inner2 = TableOracle([conflict])
    flipping = _FlipLastOracle(inner2, flip_call=total_calls)
    result = await find_mus(claims, flipping, budget=CallBudget(256))

    assert result.verified is False


# ---------------------------------------------------------------------------
# 6. BudgetExceeded raised pre-call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_budget_exceeded_pre_call() -> None:
    claims = _claims(4)
    conflict = frozenset({claims[0].id, claims[1].id})
    oracle = TableOracle([conflict])

    with pytest.raises(BudgetExceeded):
        await find_mus(claims, oracle, budget=CallBudget(1))


# ---------------------------------------------------------------------------
# 7. Determinism — identical input yields identical MusResult
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_determinism() -> None:
    claims = _claims(8)
    conflict = frozenset({claims[3].id, claims[6].id})

    oracle_a = TableOracle([conflict])
    result_a = await find_mus(claims, oracle_a, budget=CallBudget(256))

    oracle_b = TableOracle([conflict])
    result_b = await find_mus(claims, oracle_b, budget=CallBudget(256))

    assert result_a == result_b
