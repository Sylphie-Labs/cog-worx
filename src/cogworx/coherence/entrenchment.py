"""Entrenchment ordering and resolution decisions for the coherence reconciler (Pod 2.7, U4).

Pure, deterministic functions only — no I/O, no async.  Entrenchment is NEVER stored;
it is always recomputed at read time from the immutable Claim + ClaimConfidence pair
(same discipline as confidence in knowledge.confidence — CANON S5).
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Claim
from cogworx.knowledge.confidence import ClaimConfidence

__all__ = [
    "EPISTEMIC_RANK",
    "Entrenchment",
    "Resolution",
    "ResolutionKind",
    "decide_resolution",
    "entrenchment_of",
]

# Lexicographic dominance rank for the three epistemic levels.
# inference < observation < confirmed — a confirmed fact outranks an inference in every
# resolution path (see decide_resolution).
EPISTEMIC_RANK: Final[dict[str, int]] = {
    "inference": 0,
    "observation": 1,
    "confirmed": 2,
}

ResolutionKind = Literal["none", "update-supersession", "revision-defeat", "escalated"]

_ESCALATION_REASONS = Literal[
    "mus-gt-2",
    "confirmed-vs-confirmed",
    "margin-tie",
    "rank-blocks-update",
    "budget-exhausted",
    "subject-too-large",
]


class Entrenchment(BaseModel):
    """Derived ordering key for one claim.  NEVER stored — always recomputed."""

    model_config = ConfigDict(frozen=True)

    rank: int
    """From EPISTEMIC_RANK — lexicographically dominant over LCB."""
    lcb: float
    """Beta posterior mean - z*sqrt(variance); lower-confidence bound."""
    claim_id: str
    """Deterministic final tie-break key."""


class Resolution(BaseModel):
    """Outcome of :func:`decide_resolution` for one MUS."""

    model_config = ConfigDict(frozen=True)

    kind: ResolutionKind
    winner_id: str | None
    loser_ids: tuple[str, ...]
    margin: float | None
    escalation_reason: str | None
    set_valid_to: datetime | None
    """For update-supersession: the newer claim's valid_from (closes the older claim's window).
    None for revision-defeat and all escalated paths."""


def entrenchment_of(
    claim: Claim,
    confidence: ClaimConfidence,
    *,
    z: float = 1.0,
) -> Entrenchment:
    """Derive entrenchment from a claim and its pre-computed confidence.

    NEVER stored — always recomputed at read time (CANON S5).

    Args:
        claim: The immutable claim.
        confidence: Beta-posterior summary derived from the claim's evidence events.
        z: Standard-deviation multiplier for the lower-confidence bound (default 1.0).

    Returns:
        An :class:`Entrenchment` with (rank, lcb, claim_id) for total ordering.
    """
    rank = EPISTEMIC_RANK.get(claim.epistemic_type, 0)
    lcb = confidence.confidence - z * math.sqrt(confidence.variance)
    return Entrenchment(rank=rank, lcb=lcb, claim_id=claim.id)


def _predicate_norm(claim: Claim) -> str | None:
    if claim.predicate is None:
        return None
    return claim.predicate.strip().lower()


def _subject_norm(claim: Claim) -> str:
    return claim.subject.strip().lower()


def _escalate(reason: str) -> Resolution:
    return Resolution(
        kind="escalated",
        winner_id=None,
        loser_ids=(),
        margin=None,
        escalation_reason=reason,
        set_valid_to=None,
    )


def decide_resolution(
    mus_claims: Sequence[Claim],
    entrenchments: Mapping[str, Entrenchment],
    *,
    margin: float = 0.15,
) -> Resolution:
    """Pure, deterministic resolution for a MUS (minimal unsatisfiable set).

    Order-insensitive: ``decide([A, B]) == decide([B, A])``.  Achieved by sorting
    claims by ``claim.id`` before all comparisons.

    Decision table:

    - ``len(mus) == 0`` → ``kind="none"``, no winner, no losers.
    - ``len(mus) == 1`` → ``kind="none"``, ``winner_id=mus[0].id``, no losers (degenerate).
    - ``len(mus) > 2``  → escalate("mus-gt-2").
    - ``len(mus) == 2`` → see UPDATE MODE / REVISION MODE below.

    UPDATE MODE (same normalised subject + predicate, strictly different valid_from):
        Let *newer* = claim with strictly later valid_from, *older* = the other.
        - rank(newer) >= rank(older) → ``update-supersession``, winner=newer,
          set_valid_to=newer.valid_from.
        - rank(newer) < rank(older) → escalate("rank-blocks-update").

    REVISION MODE (everything else):
        - Both "confirmed" → escalate("confirmed-vs-confirmed").
        - rank(A) != rank(B) → ``revision-defeat``, winner=higher rank, margin=None.
        - rank(A) == rank(B):
            - gap = abs(lcb(A) - lcb(B))
            - gap >= margin → ``revision-defeat``, winner=higher lcb.
            - gap < margin → escalate("margin-tie").

    Args:
        mus_claims: The claims forming the MUS (usually 2).
        entrenchments: Pre-computed entrenchment keyed by claim id.
        margin: Minimum lcb gap for a revision-defeat when ranks are equal.

    Returns:
        A :class:`Resolution` describing the outcome.
    """
    # Deterministic ordering — sort by claim id for order-insensitivity.
    claims = sorted(mus_claims, key=lambda c: c.id)

    if len(claims) == 0:
        return Resolution(
            kind="none",
            winner_id=None,
            loser_ids=(),
            margin=None,
            escalation_reason=None,
            set_valid_to=None,
        )

    if len(claims) == 1:
        return Resolution(
            kind="none",
            winner_id=claims[0].id,
            loser_ids=(),
            margin=None,
            escalation_reason=None,
            set_valid_to=None,
        )

    if len(claims) > 2:
        return _escalate("mus-gt-2")

    # Exactly 2 claims.
    a, b = claims[0], claims[1]
    ent_a = entrenchments[a.id]
    ent_b = entrenchments[b.id]

    # UPDATE if same (subject_norm, predicate_norm) AND strictly different valid_from.
    subj_a, subj_b = _subject_norm(a), _subject_norm(b)
    pred_a, pred_b = _predicate_norm(a), _predicate_norm(b)
    same_subject = subj_a == subj_b
    same_predicate = pred_a is not None and pred_a == pred_b

    if same_subject and same_predicate and a.valid_from != b.valid_from:
        # UPDATE MODE
        newer, older = (b, a) if b.valid_from > a.valid_from else (a, b)
        ent_newer = entrenchments[newer.id]
        ent_older = entrenchments[older.id]

        # confirmed-vs-confirmed always escalates, even in update mode.
        # A temporal change in a confirmed fact needs human review, not auto-supersession.
        if (
            ent_newer.rank == EPISTEMIC_RANK["confirmed"]
            and ent_older.rank == EPISTEMIC_RANK["confirmed"]
        ):
            return _escalate("confirmed-vs-confirmed")

        if ent_newer.rank >= ent_older.rank:
            return Resolution(
                kind="update-supersession",
                winner_id=newer.id,
                loser_ids=(older.id,),
                margin=None,
                escalation_reason=None,
                set_valid_to=newer.valid_from,
            )
        # newer has lower rank than older — inference cannot update-out a confirmed fact
        return _escalate("rank-blocks-update")

    # REVISION MODE
    if a.epistemic_type == "confirmed" and b.epistemic_type == "confirmed":
        return _escalate("confirmed-vs-confirmed")

    if ent_a.rank != ent_b.rank:
        winner, loser = (a, b) if ent_a.rank > ent_b.rank else (b, a)
        return Resolution(
            kind="revision-defeat",
            winner_id=winner.id,
            loser_ids=(loser.id,),
            margin=None,
            escalation_reason=None,
            set_valid_to=None,
        )

    # Equal rank — use LCB gap
    gap = abs(ent_a.lcb - ent_b.lcb)
    if gap >= margin:
        winner, loser = (a, b) if ent_a.lcb > ent_b.lcb else (b, a)
        return Resolution(
            kind="revision-defeat",
            winner_id=winner.id,
            loser_ids=(loser.id,),
            margin=gap,
            escalation_reason=None,
            set_valid_to=None,
        )

    return _escalate("margin-tie")
