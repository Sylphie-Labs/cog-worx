"""Reciprocal Rank Fusion and subset-permutation guard (CANON S1, S5, S9).

Pure functions — no I/O, no model calls, no clock access (S1).
Every FusedResult.hits contains ALL ChannelHit records for that key (S5 plural provenance).
Scores are computed solely from rank arithmetic; no channel is weighted by content (S9).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Final

from cogworx.recall.results import ChannelHit, FusedResult, RecallResult

__all__ = [
    "RRF_K_DEFAULT",
    "assert_rerank_subset",
    "fuse",
]

RRF_K_DEFAULT: Final[int] = 60


def fuse(
    channel_results: Mapping[str, Sequence[RecallResult]],
    *,
    k_rrf: int = RRF_K_DEFAULT,
) -> tuple[FusedResult, ...]:
    """Fuse per-channel ranked results via Reciprocal Rank Fusion.

    Args:
        channel_results: Maps channel name to that channel's results in rank order
            (position 0 = rank 1).  An empty mapping or all-empty channels returns
            an empty tuple.
        k_rrf: The RRF constant.  Defaults to :data:`RRF_K_DEFAULT` (60).

    Returns:
        A tuple of :class:`~cogworx.recall.results.FusedResult` sorted by
        ``(fused_score DESC, key ASC)``, with 1-based ``fused_rank`` assigned.
        The ``hits`` tuple on every result contains one :class:`~cogworx.recall.results.ChannelHit`
        per channel that surfaced that key (S5 plural provenance).

    Raises:
        ValueError: When multiple channels surface the same key with differing ``text``
            values, indicating a rendering-layer bug.
    """
    # key -> accumulated state
    scores: dict[str, float] = {}
    hits_acc: dict[str, list[ChannelHit]] = {}
    first_item: dict[str, RecallResult] = {}

    for channel_name, results in channel_results.items():
        for i, result in enumerate(results):
            rank = i + 1
            key = result.key
            channel_hit = ChannelHit(
                channel=channel_name,
                rank=rank,
                raw_score=result.hit.raw_score,
            )

            if key not in scores:
                scores[key] = 0.0
                hits_acc[key] = []
                first_item[key] = result
            else:
                # Dedup text-consistency guard: different channels must agree on rendered text.
                if result.text != first_item[key].text:
                    raise ValueError(
                        f"Key {key!r} surfaced by channel {channel_name!r} with text "
                        f"{result.text!r} but earlier channel "
                        f"{hits_acc[key][0].channel!r} gave text "
                        f"{first_item[key].text!r}. This is a rendering-layer bug."
                    )

            scores[key] += 1.0 / (k_rrf + rank)
            hits_acc[key].append(channel_hit)

    if not scores:
        return ()

    # Sort: fused_score DESC, key ASC (deterministic tie-break independent of dict/channel order).
    sorted_keys = sorted(scores.keys(), key=lambda k: (-scores[k], k))

    return tuple(
        FusedResult(
            key=key,
            kind=first_item[key].kind,
            item=first_item[key].item,
            text=first_item[key].text,
            hits=tuple(hits_acc[key]),
            fused_score=scores[key],
            fused_rank=rank,
        )
        for rank, key in enumerate(sorted_keys, start=1)
    )


def assert_rerank_subset(
    original: Sequence[FusedResult],
    reranked: Sequence[FusedResult],
) -> None:
    """Raise ValueError if reranked is not a valid permutation-of-a-subset of original.

    Valid means:
    - No duplicate keys in reranked.
    - Every key in reranked appears in original.
    - Each reranked item is identical to the original item with that key (frozen model equality).

    This is the structural guard that RecallStack calls after every reranker invocation
    (S9: never trust a reranker's output shape).

    Args:
        original: The fused results before reranking.
        reranked: The results returned by a reranker.

    Raises:
        ValueError: On any structural violation (duplicates, injected keys, mutated items).
    """
    reranked_keys = [r.key for r in reranked]

    # 1. No duplicate keys in reranked.
    if len(set(reranked_keys)) != len(reranked_keys):
        seen: set[str] = set()
        dupes = [k for k in reranked_keys if k in seen or seen.add(k)]  # type: ignore[func-returns-value]
        raise ValueError(
            f"Reranked results contain duplicate keys: {dupes!r}"
        )

    # Build a lookup from the original list.
    original_by_key: dict[str, FusedResult] = {r.key: r for r in original}

    for item in reranked:
        # 2. Every key in reranked must appear in original.
        if item.key not in original_by_key:
            raise ValueError(
                f"Reranked result with key {item.key!r} was not present in original results. "
                "A reranker may only permute or drop items — it must not inject new ones."
            )

        # 3. The item must be value-equal to the original (frozen Pydantic model equality).
        original_item = original_by_key[item.key]
        if item != original_item:
            raise ValueError(
                f"Reranked result with key {item.key!r} differs from the original. "
                f"Rerankers must not mutate items. "
                f"original={original_item!r}, reranked={item!r}"
            )
