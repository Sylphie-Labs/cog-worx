"""Epistemic upgrade surface — monotonic claim promotion (CANON S5, Pod 2.7 CF-②).

Validates preconditions (rank guard, evidence eligibility, polarity) then delegates the
atomic Neo4j transaction to ``CoherenceStore.apply_epistemic_upgrade``.  This module is the
validation layer; the store is the write layer.

Monotonic ladder: inference → observation → confirmed.
Downgrades are defeats (handled by the reconciler), not upgrades.
"""

from __future__ import annotations

import datetime
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ConfigDict

from cogworx.knowledge.evidence import EvidenceEvent

if TYPE_CHECKING:
    from cogworx.substrate.coherence import CoherenceStore

__all__ = [
    "EPISTEMIC_RANK",
    "UPGRADE_ELIGIBLE_EVIDENCE",
    "UpgradeResult",
    "epistemic_upgrade",
]

# Rank ordering — higher rank = stronger epistemic status.
EPISTEMIC_RANK: Final[dict[str, int]] = {
    "inference": 0,
    "observation": 1,
    "confirmed": 2,
}

# Evidence types eligible to trigger an upgrade to each level.
# Only first-hand evidence (tool_proof, attestation) qualifies.
# "inference" has no upgraders — you do not upgrade *to* inference.
UPGRADE_ELIGIBLE_EVIDENCE: Final[dict[str, frozenset[str]]] = {
    "observation": frozenset({"tool_proof", "attestation"}),
    "confirmed": frozenset({"tool_proof", "attestation"}),
}


class UpgradeResult(BaseModel):
    """Outcome of one :func:`epistemic_upgrade` call."""

    model_config = ConfigDict(frozen=True)

    applied: bool
    claim_id: str
    from_level: str | None
    """Previous epistemic level; ``None`` if the claim was not found."""
    to_level: str
    """The requested new level (even when ``applied=False``)."""


async def epistemic_upgrade(
    store: CoherenceStore,
    claim_id: str,
    *,
    new_level: str,
    evidence: EvidenceEvent,
    actor: str,
) -> UpgradeResult:
    """Promote a claim's epistemic level on the monotonic ladder.

    Validates preconditions (evidence eligibility, polarity, rank direction) then calls
    ``store.apply_epistemic_upgrade`` which performs the atomic Neo4j transaction (TOCTOU-
    guarded, idempotent).  The store is NOT called when the pre-check determines a no-op.

    Invariants:
    - Ladder is monotonic upward only (inference → observation → confirmed).  Explicit downgrade
      raises ``ValueError`` ("a downgrade is a defeat, use the reconciler").
    - Same-or-lower target is a no-op: returns ``UpgradeResult(applied=False)``.  Idempotent.
      The store is not called on the no-op path.
    - ``evidence.type`` must be in ``UPGRADE_ELIGIBLE_EVIDENCE[new_level]`` (first-hand only).
    - ``evidence.polarity`` must be ``"+"`` (positive evidence only).

    Args:
        store: Coherence store — provides ``current_epistemic_level`` and
               ``apply_epistemic_upgrade``.
        claim_id: The claim to upgrade.
        new_level: Target ``EpistemicType`` string.
        evidence: Justifying evidence event (first-hand, positive polarity).
        actor: Framework-assigned identity of the requesting actor (S9).

    Returns:
        :class:`UpgradeResult` with ``applied=True`` iff the level actually changed.

    Raises:
        ValueError: On explicit downgrade attempt, ineligible evidence type, or negative polarity.
        KeyError: If ``new_level`` is not a recognised epistemic level.
    """
    new_rank = EPISTEMIC_RANK[new_level]  # KeyError propagates for unknown levels

    # Polarity must be positive — negative evidence is disconfirmation, not an upgrade driver.
    if evidence.polarity != "+":
        raise ValueError(
            f"Epistemic upgrades require positive polarity evidence ('+'); "
            f"got {evidence.polarity!r}. Negative evidence is disconfirmation, not an upgrade."
        )

    # Evidence eligibility — only first-hand evidence may trigger an upgrade.
    eligible = UPGRADE_ELIGIBLE_EVIDENCE.get(new_level)
    if eligible is None:
        raise ValueError(
            f"Cannot upgrade to {new_level!r}: no upgrade-eligible evidence types defined. "
            "Upgrades to 'inference' are not permitted — inference is the base level."
        )
    if evidence.type not in eligible:
        raise ValueError(
            f"Evidence type {evidence.type!r} is not eligible to trigger an upgrade to "
            f"{new_level!r}. Eligible types: {sorted(eligible)!r}."
        )

    # Read the current level to enforce the monotonic-up constraint before touching the store.
    # store.current_epistemic_level returns None if the claim is unknown.
    current_level = await store.current_epistemic_level(claim_id)
    current_rank = EPISTEMIC_RANK.get(current_level or "inference", 0)

    # Explicit downgrade: current rank strictly greater than new rank.
    if current_level is not None and current_rank > new_rank:
        raise ValueError(
            f"Claim {claim_id!r} is at {current_level!r} (rank {current_rank}); "
            f"cannot downgrade to {new_level!r} (rank {new_rank}). "
            "A downgrade is a defeat — use the coherence reconciler."
        )

    # Same rank (or current is None meaning claim not found): no-op.  Do NOT call the store.
    if current_level is not None and current_rank >= new_rank:
        return UpgradeResult(
            applied=False,
            claim_id=claim_id,
            from_level=current_level,
            to_level=new_level,
        )

    # Pre-conditions met — delegate to the store's atomic transaction.
    recorded_at = datetime.datetime.now(datetime.UTC)
    changed = await store.apply_epistemic_upgrade(
        claim_id,
        new_level=new_level,  # type: ignore[arg-type]
        evidence=evidence,
        actor=actor,
        recorded_at=recorded_at,
    )

    return UpgradeResult(
        applied=changed,
        claim_id=claim_id,
        from_level=current_level,
        to_level=new_level,
    )
