"""The Phase-4 GATE live-run driver (PRIVATE — never on the live verification path).

This package is the ONLY composition root in :mod:`cogworx.eval` allowed to import a concrete
:mod:`cogworx.model.providers` adapter (CANON S4/T5) — the 1-key DeepSeek V4 Pro bring-up and,
eventually, the full multi-family nightly run. It is a PRIVATE package (``_``-prefixed, the
:mod:`cogworx.eval._authoring` precedent): never exported from ``cogworx.eval.__init__`` and carries
no semver surface.

Contract changelog (CANON §6.1):
  - 2026-07-02 (L0/L1): initial — ``settings.py`` (price authoring + the S11 zero-price refusal,
    ``GateRunSettings`` TOML loader). No other module in this package yet.
  - 2026-07-02 (L2): ``roster.py`` — the roster preflight guard (``RosterReport`` /
    ``preflight_roster`` / ``RosterUnsound``), the last gate before ``GateRunSettings`` reaches any
    network call.
  - 2026-07-02 (L3): ``cache.py`` — ``cached_executor``, the append-only JSONL idempotency cache
    that adapts any ``ArmExecutor`` into a crash-resumable one keyed on
    ``(item_id, arm, seed, fingerprint_digest)``.
"""

from __future__ import annotations
