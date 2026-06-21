"""Corpus-lock fingerprint spine for the Phase-4 GATE eval corpus (Pod 4.4c-5a; plan §3.8/§3.9).

This module is the **decision-independent** lock spine — the part of the 4.4c-5 lock that does NOT
read any 4.4d Cell, does NOT run the contamination audit, and does NOT touch the bijection
re-validation. It computes:

  - **INV-LOCK-1 — the per-item content hash.** A pure function over EXACTLY the seven frozen
    fields a corpus item is identified by (frame, thesis, test_code, is_error, stratum, regime,
    split). Deterministic (same item -> byte-identical hash) and sensitive (flip any of the seven ->
    a different hash). :func:`lock_corpus` stamps every item with its computed ``content_hash``.
  - **INV-LOCK-2 — the git-SHA pin.** The repository commit the corpus is locked at, read once via
    ``git rev-parse HEAD`` (injected for testability; :func:`read_git_sha` is the thin default).
  - **INV-LOCK-3 — the :class:`MeasurementFingerprint`.** A frozen, total function of the corpus
    content-hash aggregate, the git SHA, ``MASTER_SEED``, the planter families present, the
    execution-environment identity (Python + key package versions + locale — the env fence
    converted-O re-evaluation needs, closing CF-4.4c-CONVERTER-ENV-PIN), and a residual-ε STUB slot
    defaulted to the not-yet-audited sentinel (4.4c-5b fills it).

It mirrors :meth:`cogworx.eval.equiv_check.EquivReport.fingerprint`'s shape (a flat dict of folded
inputs) but produces a frozen pydantic model with a content-addressed digest, not a report dict.

Scope (load-bearing — read before extending): this is ONLY the content-hash + git-SHA +
MeasurementFingerprint spine. The contamination audit, the §5 sibling-bijection re-validation, the
design-lineage Journal ledger write (a gated breaking S6 change), and INV-A0/A1 are SEPARATE
4.4c-5a/5b pieces handled elsewhere. The residual-ε slot exists NOW so a post-audit edit cannot
silently re-lock with a stale clean bill — but FILLING it is 4.4c-5b's job, not this module's.

Pure stdlib + pydantic. No model calls, no substrate I/O, no numpy/scipy (CANON S1, S2). The only
external touch is the offline ``git rev-parse`` subprocess in :func:`read_git_sha` — injected, so a
test never shells out.

Contract changelog (CANON §6.1):
  - 2026-06-21 (Pod 4.4c-5a): initial — content_hash computation + git-SHA reader + the
    MeasurementFingerprint (content-hash aggregate, git SHA, MASTER_SEED, planter families,
    exec-env identity, residual-ε STUB). New module; no existing callers. ``lock_corpus`` is the
    only writer of ``CorpusItem.content_hash`` (carried opaque by 4.4c-2). Additive.
"""

from __future__ import annotations

import hashlib
import json
import locale
import platform
import subprocess
from collections.abc import Sequence
from importlib import metadata

from pydantic import BaseModel, ConfigDict, Field

from cogworx.eval.corpus import CorpusItem

MASTER_SEED = 20260616
"""The corpus build-date master seed (eval-stats), folded into every fingerprint so a re-lock under
a different seed yields a different fingerprint. Pinned; the lock only EXPOSES it."""

RESIDUAL_EPSILON_UNAUDITED = -1.0
"""The not-yet-audited sentinel for :attr:`MeasurementFingerprint.residual_epsilon`. A real residual
ε is a non-negative rate in ``[0, 1]``; this out-of-domain ``-1.0`` means 4.4c-5b's contamination
audit has NOT yet written a clean bill. The slot exists at THIS lock stage so a post-audit edit
cannot silently re-lock with a stale clean bill — the fingerprint folds it, so a fill flips the
fingerprint (INV-LOCK-3 sensitivity)."""

# The env-identity package set: the libraries whose version can change `test_code` re-evaluation
# (the execution kernel + the schema/validation layer + the optional numeric extra). A version
# change in any of these is what CF-4.4c-CONVERTER-ENV-PIN fences a converted-O test against.
_ENV_PACKAGES: tuple[str, ...] = ("pydantic", "pytest", "numpy")

# The seven frozen fields a CorpusItem is identified by (plan §3.9; corpus.py:250-254). The
# content hash is a total function of EXACTLY these — no run/behavior fields, no opaque content_hash
# self-reference, no provenance/difficulty/sibling metadata (those are not item identity).
_CONTENT_FIELDS: tuple[str, ...] = (
    "frame",
    "thesis",
    "test_code",
    "is_error",
    "stratum",
    "error_regime",
    "split",
)


def _canonical(payload: object) -> str:
    """Canonical, byte-stable JSON serialization for hashing: sorted keys, no insignificant
    whitespace, ``ensure_ascii`` so the digest is platform/locale-independent at the encoding layer
    (the env identity is folded SEPARATELY, deliberately, so the hash itself is portable)."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(item: CorpusItem) -> str:
    """Compute a :class:`~cogworx.eval.corpus.CorpusItem`'s content hash over EXACTLY the seven
    frozen identity fields (INV-LOCK-1): ``frame``, ``thesis``, ``test_code``, ``is_error``,
    ``stratum``, ``error_regime``, ``split``.

    Deterministic — the same item yields a byte-identical hex digest. Sensitive — flipping any one
    of the seven fields changes the digest (the canonical JSON serialization names each field, so a
    hash that ignored a field would be a mutation the test suite catches). ``content_hash`` itself
    and every run/behavior/provenance field are EXCLUDED: they are not item identity.

    The pydantic sub-models (``frame``/``thesis``) are serialized via ``model_dump(mode="json")`` so
    nested frozen models reduce to plain JSON-able values before the canonical encoding.
    """
    frame = item.frame.model_dump(mode="json")
    thesis = item.thesis.model_dump(mode="json")
    payload = {
        "frame": frame,
        "thesis": thesis,
        "test_code": item.test_code,
        "is_error": item.is_error,
        "stratum": item.stratum,
        "error_regime": item.error_regime,
        "split": item.split,
    }
    return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def read_git_sha() -> str:
    """Read the current commit SHA via ``git rev-parse HEAD`` (INV-LOCK-2). The thin default reader
    for :func:`build_fingerprint` — injected there so a test never shells out. Offline, no network,
    no substrate."""
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout.strip()


def _package_version(name: str) -> str:
    """The installed version of a package, or ``"absent"`` if not installed (the optional
    ``cogworx[sizing]`` numpy extra may be absent — its absence is part of the env identity)."""
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return "absent"


class ExecEnvIdentity(BaseModel):
    """The execution-environment fingerprint a converted-O ``test_code`` is fenced to
    (CF-4.4c-CONVERTER-ENV-PIN). A converted item's test was proven IN-PROCESS-deterministic at
    conversion time (``conversion.py``'s re-check), NOT environment-independent — so re-evaluating
    it is only sound under the env it was minted in. This record IS that env: the Python build, key
    package versions, and the locale (a test sensitive to float-repr / locale could flip across
    environments). Folded into the :class:`MeasurementFingerprint`, so a different env yields a
    different fingerprint."""

    model_config = ConfigDict(frozen=True)

    python_version: str
    python_implementation: str
    package_versions: tuple[tuple[str, str], ...]
    locale_lc_ctype: str


def read_exec_env() -> ExecEnvIdentity:
    """Capture the current :class:`ExecEnvIdentity` (INV-LOCK-3 env fence). Injected into
    :func:`build_fingerprint` so a test can feed two synthetic identities and prove the fingerprint
    separates them. ``package_versions`` is sorted for byte-stability; the ``LC_CTYPE`` locale is
    the float-repr / text-encoding-relevant category."""
    versions = tuple(sorted((name, _package_version(name)) for name in _ENV_PACKAGES))
    lc_ctype = locale.setlocale(locale.LC_CTYPE)
    return ExecEnvIdentity(
        python_version=platform.python_version(),
        python_implementation=platform.python_implementation(),
        package_versions=versions,
        locale_lc_ctype=lc_ctype,
    )


def _content_hash_aggregate(items: Sequence[CorpusItem]) -> str:
    """The corpus-level content-hash aggregate: a SHA-256 over the SORTED per-item content hashes.

    Sorted so the aggregate is order-independent (the corpus is a SET of items, not a list — a
    re-ordered load must lock to the same aggregate), while still sensitive to any item's hash
    changing or to an item being added/removed. Items MUST already carry a computed ``content_hash``
    (``lock_corpus`` guarantees this); a never-locked ``""`` would silently poison the aggregate, so
    the empty-hash case is refused at the call site (:func:`build_fingerprint`)."""
    hashes = sorted(item.content_hash for item in items)
    return hashlib.sha256(_canonical(hashes).encode("utf-8")).hexdigest()


def _planter_families(items: Sequence[CorpusItem]) -> tuple[str, ...]:
    """The SORTED, de-duplicated set of planter ``injector_kind`` discriminators present in the
    corpus (``"deterministic-mutation"`` / ``"llm"`` / ``"converted"``). Folded into the fingerprint
    so a corpus with a different planter-family composition locks to a different fingerprint — the
    converted-O family in particular is the env-fenced population CF-4.4c-CONVERTER-ENV-PIN cares
    about."""
    return tuple(sorted({item.planter.injector_kind for item in items}))


class MeasurementFingerprint(BaseModel):
    """The decision-independent corpus-lock fingerprint (INV-LOCK-3). A frozen, total function of
    every lock-time input that must fence re-evaluation:

      - ``content_hash_aggregate`` — the order-independent SHA over all items' content hashes.
      - ``git_sha`` — the commit the corpus is locked at (INV-LOCK-2).
      - ``master_seed`` — the corpus build-date seed (``MASTER_SEED``).
      - ``planter_families`` — the planter ``injector_kind``s present.
      - ``exec_env`` — the execution-environment fence (CF-4.4c-CONVERTER-ENV-PIN).
      - ``residual_epsilon`` — the residual-contamination ε. A STUB at THIS lock stage, defaulted to
        :data:`RESIDUAL_EPSILON_UNAUDITED` (``-1.0``, out-of-domain). 4.4c-5b's contamination audit
        fills it; because it is folded into :attr:`digest`, a post-audit fill FLIPS the fingerprint
        — a stale clean bill cannot silently survive a re-lock.

    :attr:`digest` is the content-addressed identity: a SHA-256 over the canonical serialization of
    all six folded inputs. Determinism: identical inputs -> identical digest. Sensitivity: each
    folded input flips the digest (the canonical encoding names every field).
    """

    model_config = ConfigDict(frozen=True)

    content_hash_aggregate: str
    git_sha: str
    master_seed: int = MASTER_SEED
    planter_families: tuple[str, ...]
    exec_env: ExecEnvIdentity
    residual_epsilon: float = Field(default=RESIDUAL_EPSILON_UNAUDITED)

    @property
    def epsilon_audited(self) -> bool:
        """``False`` while ``residual_epsilon`` is the :data:`RESIDUAL_EPSILON_UNAUDITED` sentinel —
        i.e. 4.4c-5b has NOT yet written a clean bill. A reader MUST NOT treat an unaudited
        fingerprint as contamination-cleared."""
        return self.residual_epsilon != RESIDUAL_EPSILON_UNAUDITED

    @property
    def digest(self) -> str:
        """The content-addressed SHA-256 over the canonical serialization of all six folded inputs.
        Determinism + sensitivity (INV-LOCK-3): same inputs -> same digest; any folded input flips
        it. The env identity and planter families serialize through their own canonical shapes."""
        payload = {
            "content_hash_aggregate": self.content_hash_aggregate,
            "git_sha": self.git_sha,
            "master_seed": self.master_seed,
            "planter_families": list(self.planter_families),
            "exec_env": self.exec_env.model_dump(mode="json"),
            "residual_epsilon": self.residual_epsilon,
        }
        return hashlib.sha256(_canonical(payload).encode("utf-8")).hexdigest()


def lock_corpus(items: Sequence[CorpusItem]) -> list[CorpusItem]:
    """Stamp every item with its computed ``content_hash`` (INV-LOCK-1), returning a new list of
    locked items. PURE — the input items are frozen, so each is reduced via ``model_copy`` with the
    computed hash; the originals are untouched.

    This is the only writer of :attr:`CorpusItem.content_hash` (4.4c-2 carried it opaque ``""``).
    After ``lock_corpus`` every item has ``content_hash != ""``, so a measurement-run
    :func:`~cogworx.eval.corpus.load_corpus` passes the never-locked tripwire (``corpus.py:397``).
    """
    return [item.model_copy(update={"content_hash": content_hash(item)}) for item in items]


def build_fingerprint(
    items: Sequence[CorpusItem],
    *,
    git_sha: str,
    exec_env: ExecEnvIdentity | None = None,
    master_seed: int = MASTER_SEED,
    residual_epsilon: float = RESIDUAL_EPSILON_UNAUDITED,
) -> MeasurementFingerprint:
    """Build the :class:`MeasurementFingerprint` over a LOCKED corpus (INV-LOCK-3). ``git_sha`` is
    injected (use :func:`read_git_sha` for the live value); ``exec_env`` defaults to
    :func:`read_exec_env` so a test can feed a synthetic identity.

    Refuses a never-locked item: every item MUST already carry a computed ``content_hash`` (call
    :func:`lock_corpus` first) — an empty hash would silently poison the aggregate, so it is a
    LOUD :exc:`ValueError`, naming the offending ``item_id``.

    ``residual_epsilon`` defaults to the :data:`RESIDUAL_EPSILON_UNAUDITED` sentinel — 4.4c-5b's
    contamination audit re-builds the fingerprint with the real ε, which (folded into the digest)
    flips the fingerprint so a stale clean bill cannot survive a re-lock.
    """
    for item in items:
        if item.content_hash == "":
            raise ValueError(
                f"build_fingerprint: item_id {item.item_id} has an empty content_hash "
                f"(never-locked) — call lock_corpus(...) before building the fingerprint"
            )
    env = exec_env if exec_env is not None else read_exec_env()
    return MeasurementFingerprint(
        content_hash_aggregate=_content_hash_aggregate(items),
        git_sha=git_sha,
        master_seed=master_seed,
        planter_families=_planter_families(items),
        exec_env=env,
        residual_epsilon=residual_epsilon,
    )


__all__ = [
    "MASTER_SEED",
    "RESIDUAL_EPSILON_UNAUDITED",
    "ExecEnvIdentity",
    "MeasurementFingerprint",
    "build_fingerprint",
    "content_hash",
    "lock_corpus",
    "read_exec_env",
    "read_git_sha",
]
