"""Deterministic oracle test doubles for the Test Kit (CANON S4, S9).

``TableOracle`` and ``NoisyOracle`` satisfy the ``ConsistencyOracle`` structural protocol and are
designed for deterministic unit tests — no model calls, no I/O.
"""

from __future__ import annotations

import random
from collections.abc import Sequence

from cogworx.claims.provenance import Claim
from cogworx.coherence.oracle import ConsistencyOracle, OracleAnswer


class TableOracle:
    """Deterministic test oracle driven by a table of conflict sets.

    Returns ``consistent=False`` iff at least one ``conflict_set`` is a subset of the input claim
    id set.  Returns ``consistent=True`` when no conflict is triggered.
    """

    def __init__(self, conflict_sets: Sequence[frozenset[str]]) -> None:
        self._conflict_sets: tuple[frozenset[str], ...] = tuple(conflict_sets)
        self._call_count: int = 0

    @property
    def call_count(self) -> int:
        return self._call_count

    async def check(self, claims: Sequence[Claim]) -> OracleAnswer:
        self._call_count += 1
        input_ids = frozenset(c.id for c in claims)
        for conflict in self._conflict_sets:
            if conflict <= input_ids:
                return OracleAnswer(consistent=False, model_ref="table-oracle")
        return OracleAnswer(consistent=True, model_ref="table-oracle")


class NoisyOracle:
    """Wraps any oracle and flips answers with Bernoulli(flip_rate) using a seeded RNG.

    Uses ``random.Random(seed)`` for reproducibility.  ``raw_text`` is always None — the noise is
    structural, not textual (S9).
    """

    def __init__(
        self,
        inner: ConsistencyOracle,
        flip_rate: float,
        seed: int,
    ) -> None:
        self._inner = inner
        self._flip_rate = flip_rate
        self._rng = random.Random(seed)

    async def check(self, claims: Sequence[Claim]) -> OracleAnswer:
        answer = await self._inner.check(claims)
        if self._rng.random() < self._flip_rate:
            return OracleAnswer(consistent=not answer.consistent, model_ref=answer.model_ref)
        return OracleAnswer(
            consistent=answer.consistent,
            model_ref=answer.model_ref,
            raw_text=None,
        )


__all__ = [
    "NoisyOracle",
    "TableOracle",
]
