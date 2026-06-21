"""InMemoryJournal design-lineage look-budget ledger — SC-DL1 through SC-DL7 (CANON S6).

Tests that ``InMemoryJournal`` correctly implements the §3.9-B anti-overfitting / garden-of-
forking-paths ledger (Pod 4.4c-5a): ``append_design_look`` (append-only, idempotent, monotone) and
``read_design_lineage_budget`` (distinct-fingerprint count, key-isolated, re-roll-safe). No real DB
adapter; no spike marker. Each assertion names the mutation it kills.
"""

from __future__ import annotations

from cogworx.testing.doubles import InMemoryJournal

CONFIG = "pv-config-hash-A"
LINEAGE = ("sha-aaa", "sha-bbb")


async def _budget(j: InMemoryJournal, config: str, chain: tuple[str, ...]) -> int:
    return await j.read_design_lineage_budget(
        planning_variance_config_hash=config, design_lineage_chain=chain
    )


async def _append(j: InMemoryJournal, config: str, chain: tuple[str, ...], fp: str) -> None:
    await j.append_design_look(
        planning_variance_config_hash=config, design_lineage_chain=chain, fingerprint=fp
    )


# ---------------------------------------------------------------------------
# SC-DL1 — unseen key reads 0
# ---------------------------------------------------------------------------


async def test_unseen_key_budget_is_zero() -> None:
    """A key never appended to reads 0.

    Mutation killed: a read that returns a non-zero default (e.g. 1) for an unseen key.
    """
    j = InMemoryJournal()
    assert await _budget(j, CONFIG, LINEAGE) == 0


# ---------------------------------------------------------------------------
# SC-DL2 — one append → count 1
# ---------------------------------------------------------------------------


async def test_single_append_counts_one() -> None:
    """A single appended fingerprint reads as 1 distinct look.

    Mutation killed: an append that no-ops / writes nowhere, leaving the budget at 0.
    """
    j = InMemoryJournal()
    await _append(j, CONFIG, LINEAGE, "fp-1")
    assert await _budget(j, CONFIG, LINEAGE) == 1


# ---------------------------------------------------------------------------
# SC-DL3 — idempotent: same fingerprint twice → count unchanged (DISTINCT)
# ---------------------------------------------------------------------------


async def test_idempotent_same_fingerprint() -> None:
    """Appending the SAME fingerprint twice leaves the distinct count at 1.

    Mutation killed: a list/append-with-duplicates impl that counts occurrences, not distinct
    fingerprints (count would become 2).
    """
    j = InMemoryJournal()
    await _append(j, CONFIG, LINEAGE, "fp-1")
    await _append(j, CONFIG, LINEAGE, "fp-1")
    assert await _budget(j, CONFIG, LINEAGE) == 1


# ---------------------------------------------------------------------------
# SC-DL4 — distinct fingerprints accumulate
# ---------------------------------------------------------------------------


async def test_distinct_fingerprints_accumulate() -> None:
    """Two distinct fingerprints under one key read as 2.

    Mutation killed: an append that overwrites (set→single slot) rather than accumulating, leaving
    the count at 1.
    """
    j = InMemoryJournal()
    await _append(j, CONFIG, LINEAGE, "fp-1")
    await _append(j, CONFIG, LINEAGE, "fp-2")
    assert await _budget(j, CONFIG, LINEAGE) == 2


# ---------------------------------------------------------------------------
# SC-DL5 — monotone: count never decreases across an interleaved re-append
# ---------------------------------------------------------------------------


async def test_monotone_never_decreases() -> None:
    """The count never decreases as looks are appended (incl. a duplicate that is a no-op).

    Mutation killed: any impl where a repeat append or a read mutates/shrinks the ledger.
    """
    j = InMemoryJournal()
    counts: list[int] = []
    for fp in ("fp-1", "fp-1", "fp-2", "fp-2", "fp-3"):
        await _append(j, CONFIG, LINEAGE, fp)
        counts.append(await _budget(j, CONFIG, LINEAGE))
    assert counts == sorted(counts)
    assert counts == [1, 1, 2, 2, 3]


# ---------------------------------------------------------------------------
# SC-DL6 — key isolation: a different config or lineage is a distinct ledger
# ---------------------------------------------------------------------------


async def test_key_isolation_config_hash() -> None:
    """A different planning_variance_config_hash is a separate ledger (no bleed).

    Mutation killed: an impl that keys only on the lineage chain (or only on the config hash),
    collapsing distinct keys into one bucket.
    """
    j = InMemoryJournal()
    await _append(j, CONFIG, LINEAGE, "fp-1")
    await _append(j, "pv-config-hash-B", LINEAGE, "fp-1")
    assert await _budget(j, CONFIG, LINEAGE) == 1
    assert await _budget(j, "pv-config-hash-B", LINEAGE) == 1


async def test_key_isolation_lineage_chain() -> None:
    """A different design_lineage_chain (incl. a different ORDER) is a separate ledger.

    Mutation killed: an impl that keys on an order-insensitive or partial view of the chain, so a
    reordered/extended lineage aliases onto the original key.
    """
    j = InMemoryJournal()
    await _append(j, CONFIG, ("sha-aaa", "sha-bbb"), "fp-1")
    await _append(j, CONFIG, ("sha-bbb", "sha-aaa"), "fp-1")
    await _append(j, CONFIG, ("sha-aaa", "sha-bbb", "sha-ccc"), "fp-1")
    assert await _budget(j, CONFIG, ("sha-aaa", "sha-bbb")) == 1
    assert await _budget(j, CONFIG, ("sha-bbb", "sha-aaa")) == 1
    assert await _budget(j, CONFIG, ("sha-aaa", "sha-bbb", "sha-ccc")) == 1


# ---------------------------------------------------------------------------
# SC-DL7 — a corpus re-roll within a config does NOT reset the ledger
# ---------------------------------------------------------------------------


async def test_corpus_reroll_does_not_reset() -> None:
    """Re-rolling a corpus within a planning-variance config never resets the look budget.

    The ledger is keyed by (config_hash, lineage_chain) and carries NO corpus identity, so there
    is no operation that re-rolling a corpus could invoke to clear it. We model the re-roll as new
    looks accruing under the SAME key: the count keeps climbing, never resetting to the re-roll's
    own count.

    Mutation killed: any (hypothetical) reset/clear keyed off corpus identity, or an append that
    starts a fresh bucket per call instead of accumulating.
    """
    j = InMemoryJournal()
    await _append(j, CONFIG, LINEAGE, "fp-roll1-a")
    await _append(j, CONFIG, LINEAGE, "fp-roll1-b")
    assert await _budget(j, CONFIG, LINEAGE) == 2

    # "Re-roll" the corpus: new looks under the same planning-variance config + lineage.
    await _append(j, CONFIG, LINEAGE, "fp-roll2-a")
    assert await _budget(j, CONFIG, LINEAGE) == 3
