"""Candidate-pair extraction for the coherence reconciler (Pod 2.7, U2).

Pure functions only — no I/O, no async.  The reconciler calls this before dispatching
to the oracle/mu pipeline.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence, Set
from math import sqrt
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Claim


def _cosine(e1: tuple[float, ...], e2: tuple[float, ...]) -> float:
    """Return the cosine similarity of two equal-length vectors.

    Returns 0.0 for any zero-magnitude vector rather than raising.
    """
    dot = sum(a * b for a, b in zip(e1, e2, strict=False))
    mag1 = sqrt(sum(x * x for x in e1))
    mag2 = sqrt(sum(x * x for x in e2))
    if mag1 == 0.0 or mag2 == 0.0:
        return 0.0
    return dot / (mag1 * mag2)


def _adjudication_id(id_a: str, id_b: str) -> str:
    """Deterministic adjudication id for the (a, b) pair — order-independent."""
    key = "\x1f".join(sorted([id_a, id_b]))
    return "adj:" + hashlib.sha256(key.encode()).hexdigest()[:32]


def _windows_overlap(a: Claim, b: Claim) -> bool:
    """Return True if claim A's and claim B's validity windows overlap.

    A claim with ``valid_to=None`` is valid from ``valid_from`` onward.
    Two closed windows [s1, e1] and [s2, e2] overlap iff s1 <= e2 and s2 <= e1.
    A half-open window [s, ∞) overlaps [s2, e2] iff s <= e2.
    Two half-open windows always overlap (both extend to ∞).
    """
    a_start, a_end = a.valid_from, a.valid_to
    b_start, b_end = b.valid_from, b.valid_to

    if a_end is None and b_end is None:
        return True
    if a_end is None:
        # a is [a_start, ∞); b is [b_start, b_end]
        return a_start <= b_end  # type: ignore[operator]
    if b_end is None:
        # b is [b_start, ∞); a is [a_start, a_end]
        return b_start <= a_end
    return a_start <= b_end and b_start <= a_end


def _predicate_norm(claim: Claim) -> str | None:
    """Return the normalised predicate key, or None if predicate is absent."""
    if claim.predicate is None:
        return None
    return claim.predicate.strip().lower()


class CandidatePair(BaseModel):
    """An unordered pair of claims that may be in conflict."""

    model_config = ConfigDict(frozen=True)

    a_id: str
    b_id: str
    reason: Literal["same-predicate", "embedding"]
    cosine: float | None


class CandidateReport(BaseModel):
    """Full output of :func:`candidate_pairs`."""

    model_config = ConfigDict(frozen=True)

    pairs: tuple[CandidatePair, ...]
    candidate_claim_ids: tuple[str, ...]
    dropped_disjoint_window: int
    dropped_no_embedding: int
    dropped_cached: int


def candidate_pairs(
    claims: Sequence[Claim],
    *,
    cosine_threshold: float = 0.5,
    skip_adjudication_ids: Set[str] = frozenset(),
) -> CandidateReport:
    """Extract candidate conflicting pairs from a set of active claims.

    Eligibility rules (applied in order for every unordered pair of active claims):

    1. Disjoint validity windows → drop (not a conflict; true at different times).
    2. Identical predicate_norm + identical payload → drop (re-derivation, not conflict).
    3. No-good cache: adjudication id in *skip_adjudication_ids* → drop.
    4. Same predicate_norm → candidate unconditionally (``reason="same-predicate"``).
       Embedding is irrelevant for same-predicate pairs.
    5. Different predicate: both claims must have a non-empty embedding and raw cosine
       ≥ *cosine_threshold* to become a candidate (``reason="embedding"``).  If either
       embedding is absent → drop, increment ``dropped_no_embedding``.

    Args:
        claims: The pool of claims to examine.  Non-active claims are ignored.
        cosine_threshold: Minimum cosine similarity to admit a cross-predicate pair.
        skip_adjudication_ids: Pre-adjudicated pair ids; matching pairs are skipped.

    Returns:
        A :class:`CandidateReport` with the accepted pairs and drop-reason counters.
    """
    active: list[Claim] = [c for c in claims if getattr(c, "status", "active") == "active"]

    pairs_out: list[CandidatePair] = []
    dropped_disjoint = 0
    dropped_no_emb = 0
    dropped_cached = 0

    n = len(active)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = active[i], active[j]

            # Rule 1 — disjoint validity windows
            if not _windows_overlap(a, b):
                dropped_disjoint += 1
                continue

            norm_a = _predicate_norm(a)
            norm_b = _predicate_norm(b)
            same_predicate = norm_a is not None and norm_a == norm_b

            # Rule 2 — identical re-derivation (same predicate + same payload)
            if same_predicate and a.payload == b.payload:
                continue

            # Rule 3 — no-good cache
            adj_id = _adjudication_id(a.id, b.id)
            if adj_id in skip_adjudication_ids:
                dropped_cached += 1
                continue

            # Rule 4 — same predicate → candidate unconditionally
            if same_predicate:
                pairs_out.append(
                    CandidatePair(
                        a_id=a.id,
                        b_id=b.id,
                        reason="same-predicate",
                        cosine=None,
                    )
                )
                continue

            # Rule 5 — cross-predicate: require embeddings and cosine gate
            emb_a = a.embedding
            emb_b = b.embedding
            if not emb_a or not emb_b:
                dropped_no_emb += 1
                continue

            cosine = _cosine(emb_a, emb_b)
            if cosine < cosine_threshold:
                continue

            pairs_out.append(
                CandidatePair(
                    a_id=a.id,
                    b_id=b.id,
                    reason="embedding",
                    cosine=cosine,
                )
            )

    # candidate_claim_ids: sorted union of all claim ids appearing in any pair
    id_set: set[str] = set()
    for p in pairs_out:
        id_set.add(p.a_id)
        id_set.add(p.b_id)

    return CandidateReport(
        pairs=tuple(pairs_out),
        candidate_claim_ids=tuple(sorted(id_set)),
        dropped_disjoint_window=dropped_disjoint,
        dropped_no_embedding=dropped_no_emb,
        dropped_cached=dropped_cached,
    )


__all__ = [
    "CandidatePair",
    "CandidateReport",
    "candidate_pairs",
]
