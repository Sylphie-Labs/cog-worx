"""Token-budget context assembly with lost-in-the-middle (U-fold) ordering (CANON S1, S5).

Pure function — no I/O, no model calls, no clock access. Independently testable without
any substrate (S1). Provenance tuple ``FusedResult.hits`` propagates to ``ContextChunk.hits``
unchanged (S5).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Final

from cogworx.recall.results import AssembledContext, ContextChunk, FusedResult

__all__ = ["TokenCounter", "approx_tokens", "assemble"]

TokenCounter = Callable[[str], int]

_CHARS_PER_TOKEN: Final[int] = 4


def approx_tokens(s: str) -> int:
    """Approximate token count: chars / 4, minimum 1."""
    return max(1, len(s) // _CHARS_PER_TOKEN)


def assemble(
    results: Sequence[FusedResult],
    *,
    budget: int,
    count_tokens: TokenCounter = approx_tokens,
) -> AssembledContext:
    """Assemble a token-budget-bounded context with U-fold (lost-in-the-middle) ordering.

    **Step 1 — Greedy-skip admission:**
    Walk ``results`` in given order (assumed relevance-ranked, best first). Admit each item
    that fits within the remaining budget; skip (but do not stop on) items that do not fit,
    so a shorter item later may still be admitted.

    **Step 2 — U-fold ordering:**
    Place admitted items alternately at the front and back of the output array so that the
    most-relevant content occupies both edges of the context window, guarding against the
    "lost in the middle" transformer attention degradation.

    Args:
        results: Relevance-ranked ``FusedResult`` sequence (best first).
        budget: Maximum total token count for the assembled context.
        count_tokens: Token-counting function; defaults to :func:`approx_tokens`.

    Returns:
        An :class:`~cogworx.recall.results.AssembledContext` whose ``chunks`` are in U-fold
        order, with ``token_count`` summed over admitted items, ``budget`` echoed back, and
        ``dropped`` counting items that were skipped.
    """
    # --- Step 1: greedy-skip admission ---
    admitted: list[FusedResult] = []
    cumulative = 0
    dropped = 0

    for item in results:
        item_tokens = count_tokens(item.text)
        if cumulative + item_tokens <= budget:
            admitted.append(item)
            cumulative += item_tokens
        else:
            dropped += 1

    m = len(admitted)
    if m == 0:
        return AssembledContext(
            chunks=(),
            token_count=0,
            budget=budget,
            dropped=dropped,
        )

    # --- Step 2: U-fold ordering ---
    # Place items alternately at front/back; most-relevant content ends up at both edges.
    # Use a pre-sized list of sentinel FusedResult slots, replaced in-place.
    output: list[FusedResult] = list(admitted)  # same length; slots filled below
    front_ptr = 0
    back_ptr = m - 1
    for i, item in enumerate(admitted):
        if i % 2 == 0:
            output[front_ptr] = item
            front_ptr += 1
        else:
            output[back_ptr] = item
            back_ptr -= 1

    # Build a relevance-rank lookup: original 1-based position in the admitted list.
    admitted_rank: dict[str, int] = {
        item.key: rank for rank, item in enumerate(admitted, start=1)
    }

    chunks = tuple(
        ContextChunk(
            text=fr.text,
            key=fr.key,
            kind=fr.kind,
            hits=fr.hits,  # S5: propagate full provenance tuple unchanged
            fused_score=fr.fused_score,
            relevance_rank=admitted_rank[fr.key],
        )
        for fr in output
    )

    return AssembledContext(
        chunks=chunks,
        token_count=sum(count_tokens(c.text) for c in chunks),
        budget=budget,
        dropped=dropped,
    )
