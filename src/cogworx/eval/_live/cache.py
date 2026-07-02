"""Append-only JSONL idempotency cache — crash-resume for a live :class:`ArmExecutor` (Pod 4.4-live
L3).

This module wraps any :class:`~cogworx.eval.runner.ArmExecutor` in a caching executor with the
IDENTICAL ``(arm_input, seed) -> ArmOutcome`` signature, so it drops into
:func:`~cogworx.eval.runner.run_arms` / :func:`~cogworx.eval.runner.run_and_stamp` unchanged. Its
one job: a live model-backed arm costs money and time per call; a kill-mid-run (or a deliberate
resume after adding items) must never re-call the model for a cell already answered.

CACHE KEY — ``(item_id, arm, seed, fingerprint_digest)``
---------------------------------------------------------
``item_id`` comes from ``arm_input.item_id`` at call time; ``arm`` and ``fingerprint_digest`` are
bound once at construction (the wrapper only ever answers for ITS arm, under ITS corpus
fingerprint); ``seed`` is the per-call argument.

The architect's spec frames the key as ``(item_id, arm, trial, seed)``. This wrapper is called as
``executor(arm_input, seed)`` ONLY (:mod:`cogworx.eval.runner`'s :class:`ArmExecutor` Protocol) — it
never sees ``trial`` directly. But under a fixed ``master_seed``,
``seed = crn_seed(master_seed, item_id, trial)`` is a deterministic function of ``trial`` alone (for
a fixed ``item_id``) and distinct trials hash to distinct seeds (pinned by
``test_crn_seed_varies_by_item_and_trial`` in ``tests/eval/test_runner.py``) — so ``seed`` already
carries everything ``trial`` would have contributed to the key. ``seed`` SUBSUMES ``trial``; adding
``trial`` back in would be a redundant key component the wrapper cannot even observe without
threading a new parameter through the ``ArmExecutor`` Protocol.

FINGERPRINT INVALIDATION
-------------------------
``fingerprint_digest`` is PART of the key (not a separate load-time filter): a lookup only ever asks
for entries under THIS wrapper's own digest, so an entry appended under a prior corpus fingerprint
(a re-lock, a re-plant) simply never matches — it sits inert in the loaded map. A corpus re-lock
therefore yields a cold cache automatically, with no special-cased invalidation branch to keep in
sync.

CANON T1 POSTURE — an idempotency cache, NOT a shadow journal
-----------------------------------------------------------------
This is eval-path machinery living under the private, non-exported ``_live`` composition root
(:mod:`cogworx.eval._live` — see its package docstring). The eval path is journal-free BY DESIGN
(:mod:`cogworx.eval.runner`'s S1 posture: no ``StageContext``, no substrate, no durable journal on
Cell emission). This cache does not change that: it durably persists ``ArmOutcome`` records to a
plain JSONL file so a killed process can resume without re-spending a model call, honoring S6's
*no-model-re-call-on-replay* INTENT for the live-run driver — nothing more. It is not
exactly-once-append-to-a-journal machinery, carries no Journal Protocol member, and must never be
mistaken for one.

Pure stdlib (``json``, ``pathlib``, ``os.fsync``) — no new dependency, no substrate import (CANON
S2/S3).

Contract changelog (CANON §6.1):
  - 2026-07-02 (Pod 4.4-live L3): initial — ``cached_executor``. New module; no existing callers.
    Additive new public surface only.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from cogworx.eval.runner import ArmExecutor, ArmInput, ArmOutcome

__all__ = ["cached_executor"]

_CacheKey = tuple[int, str, int, str]


def _load_cache(cache_path: Path) -> dict[_CacheKey, ArmOutcome]:
    """Load every well-formed record at *cache_path* into a key -> :class:`ArmOutcome` map.

    A line that fails to parse as JSON, or parses but is missing/mistypes a required field, is
    SKIPPED rather than raising — the one line a kill-mid-append can leave torn is always the last
    one written, and tolerating it (rather than refusing the whole cache) is the point of an
    append-only, fsync-per-record log: a crash leaves a durable PREFIX, not a corrupt file. Reading
    an on-disk artifact is a system boundary (CANON standing rule); this is that boundary's guard.
    """
    cache: dict[_CacheKey, ArmOutcome] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
                key: _CacheKey = (
                    int(record["item_id"]),
                    str(record["arm"]),
                    int(record["seed"]),
                    str(record["fingerprint_digest"]),
                )
                outcome = ArmOutcome(flagged=record["flagged"], route=record["route"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
            cache[key] = outcome
    return cache


class _CachedExecutor:
    """The :class:`~cogworx.eval.runner.ArmExecutor` adapter — see the module docstring."""

    def __init__(
        self, inner: ArmExecutor, *, arm: str, fingerprint_digest: str, cache_path: Path
    ) -> None:
        self._inner = inner
        self._arm = arm
        self._fingerprint_digest = fingerprint_digest
        self._cache_path = cache_path
        self._cache = _load_cache(cache_path)

    def __call__(self, arm_input: ArmInput, seed: int) -> ArmOutcome:
        key: _CacheKey = (arm_input.item_id, self._arm, seed, self._fingerprint_digest)
        hit = self._cache.get(key)
        if hit is not None:
            return hit
        outcome = self._inner(arm_input, seed)
        self._append(key, outcome)
        self._cache[key] = outcome
        return outcome

    def _append(self, key: _CacheKey, outcome: ArmOutcome) -> None:
        item_id, arm, seed, fingerprint_digest = key
        record = {
            "item_id": item_id,
            "arm": arm,
            "seed": seed,
            "fingerprint_digest": fingerprint_digest,
            "flagged": outcome.flagged,
            "route": outcome.route,
        }
        with self._cache_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")
            f.flush()
            os.fsync(f.fileno())


def cached_executor(
    inner: ArmExecutor, *, arm: str, fingerprint_digest: str, cache_path: Path
) -> ArmExecutor:
    """Wrap *inner* in an append-only JSONL idempotency cache keyed on
    ``(item_id, arm, seed, fingerprint_digest)`` (see the module docstring for the key rationale and
    the CANON T1 posture).

    On construction, loads every already-answered cell from *cache_path* (if it exists) into memory.
    On call: a cache HIT returns the stored :class:`~cogworx.eval.runner.ArmOutcome` WITHOUT calling
    *inner* (zero inner calls — the point, for a costed model-backed arm); a MISS calls *inner*,
    appends one JSON line to *cache_path* (flushed + ``fsync``\\ ed so a kill leaves a durable
    prefix), and returns the outcome.

    *arm* and *fingerprint_digest* are bound here, once, for the lifetime of the returned executor —
    a caller wires one :func:`cached_executor` per ``(arm, fingerprint)`` it wants durable-resumed.
    *cache_path*'s parent directory must already exist (this function does not create directories).
    """
    return _CachedExecutor(
        inner, arm=arm, fingerprint_digest=fingerprint_digest, cache_path=cache_path
    )
