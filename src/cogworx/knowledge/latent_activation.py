"""ACT-R-style activation for hot/cold latent-space tiering (CANON S1, S9).

The activation formula is a pure function of stored fields — no model, no embeddings. This keeps
the tier sweeper deterministic and testable without touching the database (S9: structure, not
prompting). The formula is the S12 spike target for Pod 2.2.

Formula (ACT-R base-level):
    Δt_hours = max((now - last_used_at).total_seconds() / 3600, ε)
    activation = ln(1 + use_count) - d · ln(Δt_hours)

Properties:
- Monotone increasing in use_count.
- Monotone decreasing in Δt (older = less active).
- NaN-free for all use_count ≥ 0, any last_used_at ≤ now (ε floor bounds the log argument).
- Fresh row (use_count=0, Δt≈ε): ~+2.05 — briefly competitive, then decays to cold.
- Tie-break chain (use_count DESC, last_used_at DESC, id ASC) gives a total deterministic order.
"""

from __future__ import annotations

import math
from datetime import datetime
from typing import NamedTuple

from pydantic import BaseModel, ConfigDict, Field


class ActivationParams(BaseModel):
    model_config = ConfigDict(frozen=True)

    d: float = Field(default=0.5, gt=0.0)
    eps_hours: float = Field(default=1.0 / 60.0, gt=0.0)  # one-minute floor
    hot_capacity: int = Field(default=4096, gt=0)


class ActivationRow(NamedTuple):
    """Input data for a single activation computation, as read from storage."""

    id: str
    use_count: int
    last_used_at: datetime


_DEFAULT_PARAMS = ActivationParams()


def activation(
    use_count: int,
    last_used_at: datetime,
    now: datetime,
    params: ActivationParams = _DEFAULT_PARAMS,
) -> float:
    """Compute the ACT-R base-level activation for one record.

    The result is a float suitable for ORDER BY DESC. Higher is hotter.
    """
    delta_seconds = (now - last_used_at).total_seconds()
    delta_hours = max(delta_seconds / 3600.0, params.eps_hours)
    return math.log1p(use_count) - params.d * math.log(delta_hours)


def tier_order_key(
    row: ActivationRow,
    now: datetime,
    params: ActivationParams = _DEFAULT_PARAMS,
) -> tuple[float, int, float, str]:
    """Sort key for hot-set ranking: (activation DESC, use_count DESC, last_used_at DESC, id ASC).

    Negate the first three for use with Python's ascending sort.
    """
    act = activation(row.use_count, row.last_used_at, now, params)
    return (-act, -row.use_count, -row.last_used_at.timestamp(), row.id)


def select_hot_ids(
    rows: list[ActivationRow],
    now: datetime,
    params: ActivationParams = _DEFAULT_PARAMS,
) -> frozenset[str]:
    """Return the set of ids that belong in the hot tier given the current snapshot.

    This is the pure-Python reference implementation; the SQL sweep CTE must produce the same
    ranking. Tested by the SQL-vs-Python ranking parity suite.
    """
    if not rows:
        return frozenset()
    ranked = sorted(rows, key=lambda r: tier_order_key(r, now, params))
    return frozenset(r.id for r in ranked[: params.hot_capacity])


__all__ = [
    "ActivationParams",
    "ActivationRow",
    "activation",
    "select_hot_ids",
    "tier_order_key",
]
