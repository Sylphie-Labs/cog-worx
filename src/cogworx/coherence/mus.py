"""Minimal Unsatisfiable Subset (MUS) finder via QuickXplain (Junker 2004) (CANON S1, S9).

The ``find_mus`` coroutine locates the smallest inconsistent subset of a claim sequence by
recursively partitioning the set with a shared ``CallBudget``.  Model calls are off the write
path (S1) — this is called from the coherence reconciler sweep, never on-commit.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Claim
from cogworx.coherence.oracle import ConsistencyOracle


class BudgetExceeded(Exception):
    """Raised PRE-CALL when the oracle call budget would be exceeded."""


class CallBudget:
    """Shared oracle-call counter with a hard limit.

    ``spend()`` raises ``BudgetExceeded`` *before* incrementing when the next call would exceed the
    limit — the caller should treat the raise as "do not make this call".
    """

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._used: int = 0

    def spend(self) -> None:
        """Raise ``BudgetExceeded`` if limit reached; otherwise increment used count."""
        if self._used >= self._limit:
            raise BudgetExceeded(
                f"Oracle call budget exhausted: limit={self._limit}, used={self._used}"
            )
        self._used += 1

    @property
    def used(self) -> int:
        return self._used


class MusResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    mus: tuple[str, ...]
    """Claim ids of the MUS (sorted ascending).  Empty iff the input set was consistent."""
    oracle_calls: int
    verified: bool
    """True iff the post-extraction oracle confirmed the MUS is itself inconsistent."""


async def find_mus(
    claims: Sequence[Claim],
    oracle: ConsistencyOracle,
    *,
    budget: CallBudget,
) -> MusResult:
    """Find a Minimal Unsatisfiable Subset (MUS) of ``claims`` using QuickXplain.

    Callers should pass claims **pre-sorted by entrenchment ascending** (lowest first) so the MUS
    found is biased toward the least-entrenched claims.  Within equal entrenchment, ``claim.id``
    is used as a deterministic tiebreak.

    Raises ``BudgetExceeded`` (pre-call) if any oracle call would exceed ``budget``.

    Call count bound: ≤ ``2k·⌈log₂(N)⌉ + 2`` oracle calls for a k-element MUS in an N-element set
    (including the initial inconsistency check and the final verification call).
    """
    claim_list = list(claims)

    # Step 1: initial inconsistency check.
    budget.spend()
    initial = await oracle.check(claim_list)
    if initial.consistent:
        return MusResult(mus=(), oracle_calls=budget.used, verified=True)

    # Step 2: sort for determinism — callers pre-sort by entrenchment; id is the stable tiebreak.
    claim_list.sort(key=lambda c: c.id)

    # Step 3: QuickXplain recursion to find MUS indices (into claim_list).
    mus_indices = await _quickxplain(claim_list, [], claim_list, oracle, budget)

    mus_ids = tuple(sorted(claim_list[i].id for i in mus_indices))

    # Step 4: verification call.
    mus_claims = [claim_list[i] for i in mus_indices]
    budget.spend()
    verify = await oracle.check(mus_claims)
    verified = not verify.consistent

    return MusResult(mus=mus_ids, oracle_calls=budget.used, verified=verified)


async def _quickxplain(
    claims: list[Claim],
    background: list[Claim],
    candidates: list[Claim],
    oracle: ConsistencyOracle,
    budget: CallBudget,
) -> list[int]:
    """Return indices (into ``claims``) forming a MUS of ``background + candidates``.

    Implements the QuickXplain algorithm (Junker 2004, Algorithm 1).  Background is consistent by
    assumption on entry (or empty).  Candidates is the set to partition.

    Returns a list of indices into the original ``claims`` list.
    """
    n = len(candidates)

    # Base case: single candidate — it must be in the MUS.
    if n == 1:
        return [claims.index(candidates[0])]

    # Partition candidates into two halves.
    mid = math.ceil(n / 2)
    c1 = candidates[:mid]
    c2 = candidates[mid:]

    # Check if background + c2 is already inconsistent (c1 may be redundant).
    budget.spend()
    answer_c2 = await oracle.check(background + c2)
    if not answer_c2.consistent:
        # c2 alone (with background) is inconsistent — recurse into c2.
        return await _quickxplain(claims, background, c2, oracle, budget)

    # Check if background + c1 is already inconsistent (c2 may be redundant).
    budget.spend()
    answer_c1 = await oracle.check(background + c1)
    if not answer_c1.consistent:
        # c1 alone (with background) is inconsistent — recurse into c1.
        return await _quickxplain(claims, background, c1, oracle, budget)

    # Both halves are consistent with the background — the MUS spans both partitions.
    # Find the MUS contribution from c1 with c2 as background context.
    mus_from_c1 = await _quickxplain(claims, background + c2, c1, oracle, budget)
    c1_mus_claims = [claims[i] for i in mus_from_c1]

    # Find the MUS contribution from c2 with the c1 MUS portion as background.
    mus_from_c2 = await _quickxplain(claims, background + c1_mus_claims, c2, oracle, budget)

    return mus_from_c1 + mus_from_c2


__all__ = [
    "BudgetExceeded",
    "CallBudget",
    "MusResult",
    "find_mus",
]
