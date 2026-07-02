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

Scope (load-bearing — read before extending): the fingerprint spine is ONLY the content-hash +
git-SHA + MeasurementFingerprint. This module ALSO carries the decision-independent lock
PRECONDITIONS that run on the 4.4d frozen Cell artifact: the contamination audit (INV-LOCK-4), the
§5 sibling-bijection re-validation (INV-LOCK-5), the clean-spec tripwire (INV-LOCK-6,
:func:`assert_spec_ceiling`, §3.3), and the arm-A floor (INV-A0/INV-A1, :func:`assert_arm_a_floor`,
§5). The last two are STRUCTURAL STUBS: their assertion code lands now, bound to
``youden._sens_spec`` (the exact gate-scoring path), and is exercised against synthetic Cell
fixtures — they fire for real on 4.4d's Cells (which do not exist yet) at corpus-lock. The
design-lineage Journal ledger write (a gated breaking S6 change) and the residual-ε FILL remain
SEPARATE 4.4c-5b pieces handled elsewhere. The residual-ε slot exists NOW so a post-audit edit
cannot silently re-lock with a stale clean bill — but FILLING it is 4.4c-5b's job, not this
module's.

Pure stdlib + pydantic. No model calls, no substrate I/O, no numpy/scipy (CANON S1, S2). The only
external touch is the offline ``git rev-parse`` subprocess in :func:`read_git_sha` — injected, so a
test never shells out.

Contract changelog (CANON §6.1):
  - 2026-06-21 (Pod 4.4c-5a): initial — content_hash computation + git-SHA reader + the
    MeasurementFingerprint (content-hash aggregate, git SHA, MASTER_SEED, planter families,
    exec-env identity, residual-ε STUB). New module; no existing callers. ``lock_corpus`` is the
    only writer of ``CorpusItem.content_hash`` (carried opaque by 4.4c-2). Additive.
  - 2026-06-21 (Pod 4.4c-5a): the clean-spec tripwire (INV-LOCK-6, :func:`assert_spec_ceiling`,
    §3.3) and the arm-A floor (INV-A0/INV-A1, :func:`assert_arm_a_floor`, §5) land as STRUCTURAL
    STUBS — pure decision-independent lock preconditions over the 4.4d ``Cell`` artifact, bound to
    ``youden._sens_spec``/``_index_cells``. Additive: two new public functions, no existing caller
    touched. They fire for real on 4.4d's Cells + the 4.4c-6 spike.
  - 2026-07-02 (wiring fix): :func:`assert_no_verifiable_claim` — a new lock-time authoring assert
    (the displaced ``arms.project_arm_input`` fidelity guard's corpus-wide counterpart; see that
    function's docstring). Wired into :func:`lock_corpus` itself, first, before any item is stamped.
    Additive new public function + one new call inside ``lock_corpus``; every existing corpus
    fixture leaves ``thesis.verifiable_claim`` unset (the guard's own precondition), so this is a
    behavior-preserving no-op for every landed caller/test.
"""

from __future__ import annotations

import hashlib
import json
import locale
import math
import platform
import statistics
import subprocess
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata
from random import Random
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from cogworx.eval.corpus import CorpusItem
from cogworx.eval.youden import Cell, Stratum, _index_cells, _sens_spec, nested_bootstrap_delta
from cogworx.knowledge import beta

# The indexed-cell map shape `_index_cells` returns: item_id -> arm -> stratum -> flags-over-trials.
# Mirrors youden._CellsByItem (private there); re-stated here as a read-only Mapping for the
# flag-rate helper's signature under mypy --strict.
_CellsByItemView = Mapping[int, Mapping[str, Mapping[Stratum, Sequence[int]]]]

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


class CorpusLockError(ValueError):
    """A decision-independent lock precondition refused corpus-lock. Raised by the contamination
    audit (INV-LOCK-4, §3.8) and the matched-pair bijection re-validation (INV-LOCK-5 / the plan's
    INV-7, §5) when the corpus cannot be honestly locked. Mirrors
    :class:`cogworx.eval.corpus.CorpusLoadError`'s refuse-to-load convention — a loud, named
    refusal, never a silent skip. The message names the offending ``item_id`` (or hash) + which
    precondition failed. This is a LOCK failure, distinct from a structural load failure
    (``CorpusLoadError``) and from the never-locked fingerprint refusal
    (:func:`build_fingerprint`)."""


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


def assert_no_verifiable_claim(items: Sequence[CorpusItem]) -> None:
    """Lock-time authoring assert (CANON §6.1 2026-07-02): no corpus thesis carries a
    ``verifiable_claim``. REFUSES corpus-lock (loud :exc:`CorpusLockError`, naming the first
    offending ``item_id``) rather than silently locking a corpus the eval harness cannot honestly
    score.

    Why this lives here, not only in ``arms.py``: :func:`cogworx.eval.arms.project_arm_input` raises
    the SAME check, but it only runs when a caller drives a ``make_*_executor`` factory through
    THAT function — the ``CorpusItem`` path. A caller that instead drives a factory straight
    through :func:`~cogworx.eval.runner.run_arms` (which builds its own generic ``ArmInput`` off
    the item and never calls ``project_arm_input``) never reaches that guard. The live antithesis
    ``task`` (``dialectic.py:482-490``) appends a THIRD ``verifiable_claim``/abstention section
    that the eval arms' reproduction (``arms._dialectic_task``) does not reproduce; that
    divergence is harmless ONLY while every corpus thesis leaves ``verifiable_claim`` unset. This
    assert makes that
    corpus-wide invariant a STRUCTURAL lock precondition — the same "authoring invariant, checked
    once at lock time" pattern as the other ``assert_*`` functions in this module — so it holds no
    matter which path an arm is driven through, not only the one path that happens to call
    ``project_arm_input``.
    """
    for item in items:
        if item.thesis.verifiable_claim is not None:
            raise CorpusLockError(
                f"assert_no_verifiable_claim: item_id {item.item_id} has "
                f"thesis.verifiable_claim={item.thesis.verifiable_claim!r} set, but the model-arm "
                "dialectic diet (arms._dialectic_task) drops the live antithesis's "
                "verifiable_claim/abstention section (dialectic.py:482-490) -- a corpus thesis "
                "with a verifiable_claim would silently diverge arm D from the live antithesis. "
                "Refusing to lock rather than certifying a corpus the eval harness cannot "
                "honestly score."
            )


def lock_corpus(items: Sequence[CorpusItem]) -> list[CorpusItem]:
    """Stamp every item with its computed ``content_hash`` (INV-LOCK-1), returning a new list of
    locked items. PURE — the input items are frozen, so each is reduced via ``model_copy`` with the
    computed hash; the originals are untouched.

    Runs :func:`assert_no_verifiable_claim` FIRST (CANON §6.1 2026-07-02) — a corpus-wide authoring
    invariant, refused loudly before any item is stamped, rather than only checked per-call by
    :func:`cogworx.eval.arms.project_arm_input` on whichever path happens to call it.

    This is the only writer of :attr:`CorpusItem.content_hash` (4.4c-2 carried it opaque ``""``).
    After ``lock_corpus`` every item has ``content_hash != ""``, so a measurement-run
    :func:`~cogworx.eval.corpus.load_corpus` passes the never-locked tripwire (``corpus.py:397``).
    """
    assert_no_verifiable_claim(items)
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


# ===========================================================================
# PIECE D — contamination audit (INV-LOCK-4, plan §3.8)
# ===========================================================================


def assert_no_contamination(
    items: Sequence[CorpusItem], *, tuning_run_hashes: Iterable[str]
) -> None:
    """Lock precondition INV-LOCK-4 (plan §3.8): assert the tuning/measurement partition is clean.

    Two disjointness checks, both of which REFUSE corpus-lock (loud :exc:`CorpusLockError`) on
    violation — there is no soft return, a contaminated corpus must never lock:

      1. **item_id disjointness.** The ``item_id`` sets of ``split=="tuning"`` and
         ``split=="measurement"`` are DISJOINT. An ``item_id`` in BOTH splits means the same item
         was used to tune AND to measure — the held-out discipline is void. (Two *distinct*
         items, one per split, sharing an id is independently a corpus defect, so the same check
         catches both shapes.)
      2. **content-hash log overlap.** No ``measurement`` item's ``content_hash`` appears in the
         tuning-run log — modeled as the injected ``tuning_run_hashes`` (an ``Iterable[str]`` of
         content-hashes that fed a tuning run; DI'd so a test passes a stub, the real wiring is
         downstream). A measurement item whose content the tuning loop already saw is contaminated.

    The tuning-run log is consumed ONCE (materialized to a ``set`` up front) so a single-pass
    iterator is safe. Empty-``content_hash`` measurement items are treated as a never-locked defect
    and refused — an unlocked item has no auditable identity, so it cannot be cleared (call
    :func:`lock_corpus` first; this mirrors :func:`build_fingerprint`'s never-locked refusal).

    PURE, offline, deterministic: no model calls, no substrate, no journal (CANON S1/S2). Decision-
    independent — reads no 4.4d Cell, runs no oracle.

    :raises CorpusLockError: on any cross-split ``item_id`` overlap, any measurement content-hash
        present in ``tuning_run_hashes``, or a never-locked measurement item.
    """
    tuning_ids = {it.item_id for it in items if it.split == "tuning"}
    measurement_items = [it for it in items if it.split == "measurement"]
    measurement_ids = {it.item_id for it in measurement_items}

    # Check 1 — cross-split item_id disjointness.
    both = sorted(tuning_ids & measurement_ids)
    if both:
        raise CorpusLockError(
            f"INV-LOCK-4 contamination: item_id(s) {both} appear in BOTH the tuning and "
            f"measurement splits — the held-out partition is not disjoint"
        )

    # Check 2 — no measurement content-hash in the tuning-run log. Materialize the injected log to a
    # set ONCE (the seam is a single-pass Iterable by contract).
    tuning_log = set(tuning_run_hashes)
    for item in measurement_items:
        if item.content_hash == "":
            raise CorpusLockError(
                f"INV-LOCK-4 contamination: measurement item_id {item.item_id} has an empty "
                f"content_hash (never-locked) — call lock_corpus(...) before the audit"
            )
        if item.content_hash in tuning_log:
            raise CorpusLockError(
                f"INV-LOCK-4 contamination: measurement item_id {item.item_id} has a content_hash "
                f"present in the tuning-run log — the tuning loop already saw this item"
            )


# ===========================================================================
# PIECE E — matched-pair bijection re-validation (INV-LOCK-5 = plan's INV-7, §5)
# ===========================================================================


class BijectionResult(BaseModel):
    """The result of the post-abstention matched-pair bijection re-validation (INV-LOCK-5, plan §5 /
    the plan's INV-7). Runs AT LOCK, BEFORE any stratum shuffle, AFTER the labeling pipeline's
    abstention-drops are known (:attr:`cogworx.eval.labeling.PromotionResult.dropped_ids`).

      - ``paired`` — the surviving matched-sibling items that re-validate as a CLEAN, MUTUAL
        bijection: both members of every retained pair survive the drop set, each names the other,
        and the relation is symmetric. The §5 within-pair sign-flip control runs ONLY over these.
      - ``demoted_unpaired`` — the survivors of DISSOLVED pairs (exactly one member dropped). Each
        carries its REAL, UNCHANGED ``stratum`` and its ``matched_sibling_id`` cleared to ``None`` —
        demoted to the unpaired global-shuffle pool (the §5 marginal-preservation discipline: an
        orphan enters the global pool with its real label, contributing +1 to the marginal it truly
        belongs to). It loses only its difficulty-matched partner, never its label.

    Frozen + content-stable: ``paired`` and ``demoted_unpaired`` are sorted by ``item_id`` so the
    result is deterministic regardless of input ordering."""

    model_config = ConfigDict(frozen=True)

    paired: tuple[CorpusItem, ...]
    demoted_unpaired: tuple[CorpusItem, ...]


def revalidate_bijection(
    items: Sequence[CorpusItem], dropped_ids: frozenset[int]
) -> BijectionResult:
    """Re-validate the matched-sibling pair set as a clean bijection after abstention-drops
    (INV-LOCK-5, plan §5 / INV-7). Runs AT LOCK, BEFORE any stratum shuffle.

    Consumes the surviving promoted :class:`~cogworx.eval.corpus.CorpusItem`s (``items``) plus the
    abstention ``dropped_ids`` (the
    :attr:`cogworx.eval.labeling.PromotionResult.dropped_ids` frozenset). The drop set is item-
    granular (§3.4): a dropped ``item_id`` removes every cell of that item.

    Rules:
      - a matched pair ``(p, q)`` (linked by ``matched_sibling_id``) survives as PAIRED iff BOTH
        members survive the drop set;
      - a pair with exactly ONE survivor is DISSOLVED — the survivor is demoted to the unpaired
        global-shuffle pool, carrying its REAL, UNCHANGED ``stratum`` with ``matched_sibling_id``
        cleared to ``None`` (``model_copy`` — the frozen input is untouched);
      - an unpaired item (``matched_sibling_id is None``) is neither paired nor demoted-from-a-pair;
        it stays in the global pool as-is (it is NOT a dissolution survivor).

    After processing every drop, the surviving PAIRED set MUST re-validate as a clean MUTUAL
    bijection — every paired item's sibling is present in the paired set AND the relation is
    symmetric. A malformed result (an orphaned paired item whose sibling is gone but
    ``matched_sibling_id`` was not cleared, or an asymmetric link) is a corpus defect that REFUSES
    the lock.

    Note: ``items`` MUST already exclude the dropped items (it is the promoted survivor set); a
    ``dropped_id`` that still appears in ``items`` is a caller contract violation and is refused —
    the survivor set and the drop set must be consistent. A ``matched_sibling_id`` naming an id that
    is neither present nor in ``dropped_ids`` is a dangling reference (a corpus defect) — refused.

    PURE, offline, deterministic: no model calls, no substrate, no journal (CANON S1/S2).

    :raises CorpusLockError: on a present-yet-dropped item, a dangling sibling reference, an
        asymmetric retained pair, or an uncleared orphan (the malformed-bijection guards).
    """
    present_ids = {it.item_id for it in items}

    # Contract guard: the survivor set and the drop set must be disjoint — `items` is the PROMOTED
    # set, which already excludes drops. A dropped id still present is an inconsistent caller.
    leaked = sorted(present_ids & dropped_ids)
    if leaked:
        raise CorpusLockError(
            f"INV-LOCK-5 bijection: item_id(s) {leaked} are in dropped_ids yet still present in "
            f"the survivor set — the promoted items and the drop set are inconsistent"
        )

    by_id = {it.item_id: it for it in items}

    paired: list[CorpusItem] = []
    demoted: list[CorpusItem] = []

    for item in items:
        sib_id = item.matched_sibling_id
        if sib_id is None:
            # Never-paired item — belongs to the global pool, neither retained-paired nor demoted.
            continue
        if sib_id in dropped_ids:
            # One-survivor pair: DISSOLVE — demote this survivor, clear its sibling link, keep its
            # real stratum.
            demoted.append(item.model_copy(update={"matched_sibling_id": None}))
            continue
        if sib_id not in present_ids:
            # Sibling neither present nor explicitly dropped — a dangling reference (corpus defect).
            raise CorpusLockError(
                f"INV-LOCK-5 bijection: item_id {item.item_id} names matched_sibling_id {sib_id} "
                f"which is neither present in the survivor set nor in dropped_ids (dangling "
                f"reference)"
            )
        # Both present → candidate retained pair. Re-validate mutual symmetry.
        sib = by_id[sib_id]
        if sib.matched_sibling_id != item.item_id:
            raise CorpusLockError(
                f"INV-LOCK-5 bijection: asymmetric retained pair — item_id {item.item_id} names "
                f"{sib_id} but item_id {sib_id} names {sib.matched_sibling_id!r} (an orphaned "
                f"paired item whose link was not cleared)"
            )
        paired.append(item)

    return BijectionResult(
        paired=tuple(sorted(paired, key=lambda it: it.item_id)),
        demoted_unpaired=tuple(sorted(demoted, key=lambda it: it.item_id)),
    )


# ===========================================================================
# PIECE G1 — clean-spec tripwire: the PLANNED-n design ceiling + the variance
#            floor (plan §3.3 item 3, INV-LOCK-6)
# ===========================================================================
#
# STRUCTURAL STUB. This lock precondition consumes the 4.4d frozen Cell artifact (`flagged`/`route`
# are produced by the runner, plan §1) — which does NOT exist yet. The assertion CODE lands now,
# bound to `youden._index_cells`/`_sens_spec` (the exact scoring path the gate's spec arm uses), and
# is exercised in tests against synthetic Cell fixtures (`synth_cells` / hand-built Cells) that
# SATISFY and VIOLATE the tripwire. It fires for real on 4.4d's Cells at corpus-lock.


def _clean_flag_rates(
    cells_by_item: _CellsByItemView, clean_ids: Sequence[int], arm: str
) -> list[float]:
    """The per-item flag rate ``f_i = (#flagged over its R trials)/R`` for one arm over the clean
    items (plan §3.3 / §5 ``f_i``). The bootstrap's spec spread is exactly the item-to-item spread
    of these rates; a near-zero variance is the hollow-spec signature the variance floor catches.

    Reads the SAME indexed structure :func:`~cogworx.eval.youden._sens_spec` reads, so the rate is
    computed off the identical artifact the gate scores. An item with no clean cells for ``arm``
    contributes nothing (it is absent from the rate list -- it has no auditable spec)."""
    rates: list[float] = []
    for item_id in clean_ids:
        flags: list[int] = []
        for trials in cells_by_item[item_id][arm].values():
            flags.extend(trials)
        if flags:
            rates.append(sum(flags) / len(flags))
    return rates


def assert_spec_ceiling(
    cells: Sequence[Cell],
    *,
    n_clean_planned: int,
    R: int,
    sigma_sq_b_spec_planning: float,
    arm: str = "C",
) -> None:
    """Lock precondition INV-LOCK-6 (plan §3.3 item 3): the clean-spec tripwire. Refuses corpus-lock
    (loud :exc:`CorpusLockError`) when the clean pool is *trivial* -- a spec arm pinned at ~1.0
    silently degenerates ``delta = J_D - J_C'`` into a bare catch-rate comparison, reintroducing the
    yes-machine attack the gate exists to kill. Two parts, BOTH binding:

    1. **The PLANNED-n design ceiling.** ``realized_spec_C' < 1 - 2/(R*n_clean_planned)`` (~0.9964
       at R=7, n_clean_planned=80). A spec AT OR ABOVE the ceiling means the arm flags <=~2 expected
       clean cells across the whole stratum -- too few to give J's spec arm any spread.

       **The ceiling is computed from PLANNED n_clean, NEVER realized n_clean (§3.3 NEW-MED-2, the
       backwards-tripwire bug).** Abstention (§3.4) shrinks the realized clean count, which -- if
       the ceiling tracked realized n -- would RAISE the ceiling toward 1 and *loosen* the tripwire,
       exactly backwards: fewer clean items should make a hollow spec arm MORE suspect, not less.
       Fixing the ceiling at the ratified planning n at lock-time means abstention cannot relax it.
       (A realized clean shrinkage >20% triggers the §6.0(c) re-size loop; the corpus is re-sized,
       never the ceiling moved.)

    2. **The variance floor.** ``pvariance(per-item clean flag rate) >= 0.5 * sigma_sq_b_spec``.
       This catches the ``spec_C'=0.96-but-hollow`` case the mean-based ``<0.98`` rev-1 tripwire
       could never see: a non-trivial MEAN spec whose item-to-item flag-rate variance has collapsed
       to near-zero gives the bootstrap no spec spread. ``sigma_sq_b_spec_planning`` is the
       author/eval-stats between-item spec variance the sizing sim was powered at -- INJECTED,
       never derived from the realized artifact (deriving it from the artifact under test would
       make the floor self-satisfying).

    Bound to :func:`~cogworx.eval.youden._sens_spec` / ``_index_cells`` (the EXACT spec-scoring path
    the gate runs), so a drift in how spec is computed moves this check in lockstep. PURE, offline,
    deterministic: no model calls, no substrate, no journal (CANON S1/S2). Decision-independent of
    any verdict -- it reads the artifact's clean cells, runs no oracle.

    :param cells: the 4.4d frozen ``Cell`` artifact (the flat list the bootstrap consumes).
    :param n_clean_planned: the RATIFIED planning clean-item count (e.g. 80); the ceiling's only n.
    :param R: trials per ``(item, arm)`` -- the planning R the ceiling was sized at (e.g. 7).
    :param sigma_sq_b_spec_planning: the planning between-item spec variance ``sigma_sq_b_spec``
        (the ``sb_spec``-derived planning variance); the floor is ``0.5x`` this.
    :param arm: which arm's spec defines the tripwire (default ``"C"`` -- the C' baseline whose
        hollow spec degenerates the binding delta; the gate's spec arm).
    :raises CorpusLockError: if ``n_clean_planned`` or ``R`` is non-positive (a malformed planning
        constant), if the artifact has no clean cells for ``arm`` (nothing to audit -- a hollow
        corpus cannot be cleared), if realized spec is AT OR ABOVE the planned-n ceiling, or if the
        per-item clean flag-rate variance is below the planning-derived floor.
    """
    if n_clean_planned <= 0 or R <= 0:
        raise CorpusLockError(
            f"INV-LOCK-6 spec-ceiling: n_clean_planned={n_clean_planned} and R={R} must both be "
            f"positive -- the planned-n ceiling 1 - 2/(R*n_clean_planned) is undefined otherwise"
        )

    cells_by_item, items_in_stratum = _index_cells(cells)
    clean_ids = items_in_stratum.get("clean", [])
    if not clean_ids:
        raise CorpusLockError(
            "INV-LOCK-6 spec-ceiling: the artifact has no clean-stratum cells -- a corpus with no "
            "auditable clean pool cannot be cleared (the spec arm has nothing to spread over)"
        )

    # The ceiling is a TOTAL function of the PLANNED constants -- computed before, and independent
    # of, the realized clean count. This is the NEW-MED-2 fix: realized n never enters here.
    ceiling = 1.0 - 2.0 / (R * n_clean_planned)

    # Realized spec for `arm` over the clean pool -- the SAME _sens_spec the gate's spec arm runs.
    # error_item_ids is empty: this call reads only spec (index [1]); sens is a degenerate 0.0.
    _, realized_spec = _sens_spec(cells_by_item, [], arm, clean_ids)
    if realized_spec >= ceiling:
        raise CorpusLockError(
            f"INV-LOCK-6 spec-ceiling: realized spec_{arm}={realized_spec:.6f} is AT OR ABOVE the "
            f"planned-n ceiling {ceiling:.6f} (= 1 - 2/(R*n_clean_planned), R={R}, "
            f"n_clean_planned={n_clean_planned}) -- the clean pool is trivial and hollows the spec "
            f"arm of J; the ceiling is fixed at PLANNED n so abstention cannot relax it"
        )

    floor = 0.5 * sigma_sq_b_spec_planning
    rates = _clean_flag_rates(cells_by_item, clean_ids, arm)
    realized_var = statistics.pvariance(rates) if len(rates) >= 1 else 0.0
    if realized_var < floor:
        raise CorpusLockError(
            f"INV-LOCK-6 spec-ceiling: per-item clean flag-rate variance {realized_var:.6g} is "
            f"below the planning-derived floor {floor:.6g} (= 0.5*sigma_sq_b_spec_planning, "
            f"sigma_sq_b_spec_planning={sigma_sq_b_spec_planning:.6g}) -- a non-trivial mean spec "
            f"with collapsed item-to-item variance still hollows the bootstrap's spec spread"
        )


# ===========================================================================
# PIECE G2 -- INV-A0 / INV-A1: the deterministic-oracle arm-A floor (plan §5)
# ===========================================================================
#
# STRUCTURAL STUB. The shuffle-null centering proof (§5) holds ONLY conditional on arm A behaving as
# the floor parent §2/§13.6 assume (sens_A==0 on K, spec_A~1 on clean). The sizing sim scores A
# analytically; the REAL gate corpus must EARN it on the frozen 4.4d artifact. The assertion code
# lands now, bound to `youden._sens_spec`, exercised in tests against synthetic Cell fixtures that
# SATISFY and VIOLATE each floor (negative controls). It fires for real on 4.4d's arm-A Cells + the
# 4.4c-6 spike.


def assert_arm_a_floor(
    cells: Sequence[Cell],
    *,
    n_clean_planned: int,
    R: int,
    error_strata: Sequence[Stratum] = ("K",),
) -> None:
    """Lock precondition INV-A0 / INV-A1 (plan §5): assert the deterministic-oracle arm **A** hits
    its floor on the frozen artifact. The §5 stratum-shuffle null centers at ``E[delta_shuf]=0`` for
    both ``D>A`` and ``D>C'`` ONLY if these two hold; they are asserted at lock so the centering
    claim is EARNED, not assumed (the sim scores A analytically; the gate corpus must not).

    - **INV-A0 (K floor -- arm A flags nothing on the error stratum).** For arm A over the K (error)
      cells, ``sens_A == 0.0`` -- equivalently ``sum(flagged) == 0`` (plan §5, ``youden.py:138`` the
      K loop). A is the pure-oracle floor: on an oracle-BLIND error it flags nothing. If A flags ANY
      K item, that item is mis-stratified -- an "oracle-blind" item the oracle actually reached
      (it belongs in O, not K) -> **refuse lock**. (Generalized to all ``error_strata``; the binding
      error stratum for the D>A floor is K, parent §13.5.)
    - **INV-A1 (clean floor -- arm A barely false-positives).** For arm A over the clean cells,
      ``spec_A >= 1 - tau_A`` with **``tau_A = 2/(R*n_clean_planned)``** (~0.0036 at R=7,
      n_clean_planned=80); A may false-positive on at most ~2 expected clean cells across the whole
      stratum. A clean item A flags is either a mislabeled-clean (it should have failed the §3.7
      noise audit) or an oracle bug; >~2 FPs means A is not the floor the gate's centering assumes
      -> **refuse lock**.

    ``tau_A`` is computed from **PLANNED** ``n_clean`` (the same NEW-MED-2 discipline as
    :func:`assert_spec_ceiling`): abstention must not relax the clean-FP budget. Bound to
    :func:`~cogworx.eval.youden._sens_spec`, the EXACT scoring path the shuffle-null reuses, so a
    floor check and the gate's J read the same sens/spec. PURE, offline, deterministic: no model
    calls, no substrate, no journal (CANON S1/S2). Decision-independent; reads arm-A cells, runs no
    oracle.

    :param cells: the 4.4d frozen ``Cell`` artifact carrying arm ``"A"`` cells on K + clean.
    :param n_clean_planned: the RATIFIED planning clean count; the only n in ``tau_A`` (PLANNED,
        not realized).
    :param R: trials per ``(item, arm)`` -- the planning R ``tau_A`` is sized at.
    :param error_strata: the error strata A must flag nothing on (default ``("K",)`` -- the binding
        D>A error population).
    :raises CorpusLockError: if ``n_clean_planned``/``R`` is non-positive, if arm ``"A"`` is absent
        from an audited stratum (no auditable floor -- a missing arm cannot be cleared), if A flags
        any error-stratum item (INV-A0), or if A's clean spec is below ``1 - tau_A`` (INV-A1).
    """
    if n_clean_planned <= 0 or R <= 0:
        raise CorpusLockError(
            f"INV-A0/A1 arm-A floor: n_clean_planned={n_clean_planned} and R={R} must both be "
            f"positive -- tau_A = 2/(R*n_clean_planned) is undefined otherwise"
        )

    cells_by_item, items_in_stratum = _index_cells(cells)

    # INV-A0 -- arm A flags nothing on every error stratum. Run _sens_spec per error stratum so a
    # mis-stratified item is named by stratum; an empty clean_ids makes spec degenerate (unread).
    for stratum in error_strata:
        error_ids = items_in_stratum.get(stratum, [])
        if not error_ids:
            raise CorpusLockError(
                f"INV-A0 arm-A floor: the artifact has no '{stratum}' error-stratum cells -- arm "
                f"A's floor on {stratum} has nothing to audit (a missing error pool isn't cleared)"
            )
        if not any("A" in cells_by_item[item_id] for item_id in error_ids):
            raise CorpusLockError(
                f"INV-A0 arm-A floor: arm 'A' is absent from the '{stratum}' stratum -- the "
                f"deterministic-oracle floor cannot be asserted on an artifact missing arm A"
            )
        sens_a, _ = _sens_spec(cells_by_item, error_ids, "A", [])
        if sens_a != 0.0:
            raise CorpusLockError(
                f"INV-A0 arm-A floor: arm A has sens_A={sens_a:.6f} > 0 on the '{stratum}' "
                f"stratum -- A flagged an oracle-blind error, so that item is mis-stratified (the "
                f"oracle reached it; it belongs in O, not {stratum}) -> refuse lock"
            )

    # INV-A1 -- arm A barely false-positives on clean. tau_A from PLANNED n_clean (NEW-MED-2).
    clean_ids = items_in_stratum.get("clean", [])
    if not clean_ids:
        raise CorpusLockError(
            "INV-A1 arm-A floor: the artifact has no clean-stratum cells -- arm A's clean floor "
            "has nothing to audit (a missing clean pool cannot be cleared)"
        )
    if not any("A" in cells_by_item[item_id] for item_id in clean_ids):
        raise CorpusLockError(
            "INV-A1 arm-A floor: arm 'A' is absent from the clean stratum -- the deterministic-"
            "oracle floor cannot be asserted on an artifact missing arm A"
        )
    tau_a = 2.0 / (R * n_clean_planned)
    floor = 1.0 - tau_a
    _, spec_a = _sens_spec(cells_by_item, [], "A", clean_ids)
    if spec_a < floor:
        raise CorpusLockError(
            f"INV-A1 arm-A floor: arm A has spec_A={spec_a:.6f} below the clean floor {floor:.6f} "
            f"(= 1 - tau_A, tau_A = 2/(R*n_clean_planned), R={R}, "
            f"n_clean_planned={n_clean_planned}) -- A false-positives on >~2 expected clean cells, "
            f"so it is not the floor the §5 shuffle-null centering assumes; tau_A is fixed at "
            f"PLANNED n so abstention cannot relax it"
        )


# ===========================================================================
# Pod 4.4c-6a -- the three corpus-lock statistical instruments (plan §5, §2.B, §3.3)
# ===========================================================================
#
# These bind the SAME gate-scoring path the gate runs (`youden._index_cells`/`_sens_spec` and the
# pinned `nested_bootstrap_delta`), so a drift in how the gate scores moves these instruments in
# lockstep -- the same discipline `assert_spec_ceiling`/`assert_arm_a_floor` already follow. PURE,
# offline, deterministic: no model calls, no substrate, no journal (CANON S1/S2). All Monte-Carlo
# uses a seeded `random.Random`. Pure stdlib + `cogworx.knowledge.beta`; no numpy in the lock path.
#
# Instrument 1 (§5)  -- the stratum-membership shuffle null (GATES, two orthogonal assertions).
# Instrument 2 (§2.B) -- the per-regime contribution bound (GATES) + LORO (reported-only).
# Instrument 3 (§3.3) -- the KS flag-rate diagnostic (reported-only).


# ---------------------------------------------------------------------------
# INSTRUMENT 1 -- §5 stratum-membership shuffle null
# ---------------------------------------------------------------------------


class _ShuffleFn(Protocol):
    """The stratum-membership shuffle callable shape -- the honest
    :func:`shuffle_stratum_membership` and the test-only leaky variant share it, so
    :func:`shuffle_null_centering` can take either via its ``_shuffle`` injection point."""

    def __call__(
        self,
        cells: Sequence[Cell],
        *,
        paired_ids: Sequence[tuple[int, int]],
        unpaired_ids: Sequence[int],
        rng: Random,
    ) -> list[Cell]: ...


def shuffle_stratum_membership(
    cells: Sequence[Cell],
    *,
    paired_ids: Sequence[tuple[int, int]],
    unpaired_ids: Sequence[int],
    rng: Random,
) -> list[Cell]:
    """ONE stratum-membership permutation -> ONE relabeled `Cell` artifact (plan §5, §5.0).

    The no-leakage control's atom: reassign which ``item_id``s carry ``stratum="K"`` vs ``"clean"``,
    holding the per-stratum marginal counts fixed, **with each item's flag vectors left welded to
    it** (the LOAD-BEARING constraint, §5: move the ``stratum`` field, never regenerate or re-map
    the flags). Under this welding, ``J_shuf^arm = mean_{i in S} f_i - mean_{i notin S} f_i`` for
    every arm, whose expectation over a uniform partition is ``fbar - fbar = 0`` regardless of base
    rate (§5.0 load-bearing identity) -- so an honest corpus centers at 0 and a leaky one does not.

    Two disjoint controls, applied to the SAME artifact:

      - **Within-pair sign-flip (PRIMARY).** Over the matched ``(K, clean)`` bijective pairs
        ``paired_ids`` (the re-validated :class:`BijectionResult.paired`), independently per pair
        with probability 0.5, SWAP the two members' ``stratum`` fields (and only those two). A
        restricted, marginal-count-auto-preserving permutation that holds matched difficulty fixed
        while breaking the label binding -- strictly stronger than the global shuffle (§5 / §3.3).
      - **Fixed-marginal global permutation (FALLBACK).** Over the unpaired pool ``unpaired_ids``
        (orphans + never-paired items), permute the multiset of their current strata among
        themselves (``rng.shuffle`` of the stratum list), holding the K/clean marginal counts fixed.

    The new stratum is welded per ``item_id``: ``model_copy(update={"stratum": ...})`` is applied to
    **every** cell of that item across every arm and trial. Splitting an item's cells across two
    strata would corrupt ``_index_cells``'s urn sizes (the §5 implementation note / the
    ``_index_cells`` split hazard); the per-item welding is pinned by the suite. Cells whose
    ``item_id`` is in NEITHER ``paired_ids`` nor ``unpaired_ids`` (e.g. ``O`` items, which §5 does
    NOT shuffle into K) pass through unchanged.

    PURE: the frozen input cells are untouched; a new list of relabeled copies is returned. Only the
    ``stratum`` field moves -- ``flagged``/``regime``/``route``/``seed``/``trial`` are welded to the
    item. Deterministic in ``rng`` (a seeded :class:`random.Random` gives a byte-identical
    artifact).

    :param paired_ids: the matched ``(K-item, clean-item)`` pairs to sign-flip (the PRIMARY
        control).
    :param unpaired_ids: the unpaired item_ids to globally permute (the FALLBACK control).
    :param rng: the seeded RNG driving both the per-pair coin flips and the global permutation.
    """
    new_stratum: dict[int, Stratum] = {}

    # PRIMARY -- within-pair sign-flip, independently per pair with prob 0.5. Read current strata
    # off the artifact (the first cell of each member suffices -- stratum is item-welded).
    current: dict[int, Stratum] = {}
    for c in cells:
        if c.item_id not in current:
            current[c.item_id] = c.stratum
    for a, b in paired_ids:
        if rng.random() < 0.5:
            new_stratum[a] = current[b]
            new_stratum[b] = current[a]
        else:
            new_stratum[a] = current[a]
            new_stratum[b] = current[b]

    # FALLBACK -- fixed-marginal global permutation over the unpaired pool: shuffle the multiset of
    # their current strata among themselves (K/clean counts preserved by construction).
    pool_strata = [current[i] for i in unpaired_ids]
    rng.shuffle(pool_strata)
    for i, s in zip(unpaired_ids, pool_strata, strict=True):
        new_stratum[i] = s

    # Weld the new stratum onto EVERY cell of each relabeled item; untouched items pass through.
    return [
        c.model_copy(update={"stratum": new_stratum[c.item_id]}) if c.item_id in new_stratum else c
        for c in cells
    ]


def _shuffle_stratum_membership_LEAKY(
    cells: Sequence[Cell],
    *,
    paired_ids: Sequence[tuple[int, int]],
    unpaired_ids: Sequence[int],
    rng: Random,
) -> list[Cell]:
    """The deliberately-LEAKY shuffle variant (§5 mutation test -- instrument validity).

    Identical relabeling to :func:`shuffle_stratum_membership`, EXCEPT it carries flag-SEMANTICS
    with the stratum move instead of welding the flags to the item: whenever an item's ``stratum``
    flips between ``K`` and ``clean``, every flag bit is inverted (``1 - flagged``). This preserves
    the label<->flag association the honest shuffle severs (a high-flag K item relabeled ``clean``
    becomes a low-FP clean item, still scoring "well"), so ``mean(shuffle_delta)`` does NOT cancel
    to ~0 --> :func:`assert_shuffle_null` MUST fire on it. An assertion that cannot be made to fail
    tests nothing; this is the mutation that proves the centering assertion has teeth.

    NOT a public API -- it exists only so the suite can drive the mutation test. (`O`/untouched
    items pass through unchanged, as in the honest shuffle.)
    """
    new_stratum: dict[int, Stratum] = {}
    current: dict[int, Stratum] = {}
    for c in cells:
        if c.item_id not in current:
            current[c.item_id] = c.stratum
    for a, b in paired_ids:
        if rng.random() < 0.5:
            new_stratum[a] = current[b]
            new_stratum[b] = current[a]
        else:
            new_stratum[a] = current[a]
            new_stratum[b] = current[b]
    pool_strata = [current[i] for i in unpaired_ids]
    rng.shuffle(pool_strata)
    for i, s in zip(unpaired_ids, pool_strata, strict=True):
        new_stratum[i] = s

    out: list[Cell] = []
    for c in cells:
        if c.item_id not in new_stratum:
            out.append(c)
            continue
        s_new = new_stratum[c.item_id]
        moved = s_new != current[c.item_id]  # the K<->clean flip carries flag-semantics (the leak)
        flagged = (1 - c.flagged) if moved else c.flagged
        out.append(c.model_copy(update={"stratum": s_new, "flagged": flagged}))
    return out


class ShuffleNullResult(BaseModel):
    """The collected output of :func:`shuffle_null_centering` over ``n_shuffles`` permutations
    (plan §5; Finding-1 fix). The §5 null runs as TWO independent shuffle passes per shuffle idx --
    both the FULL §5 relabeling (within-pair sign-flip + fixed-marginal global permute), differing
    ONLY in their derived RNG draw. They are not two different nulls; they are two independent draws
    of the SAME null, read by orthogonal assertions (Finding-1: centering and coverage-rate must not
    share an artifact, else the paired sign-flip's un-calibrated tail false-refuses an honest
    difficulty-confounded corpus via the global-null-derived 0.08 ceiling).

    Per binding delta (keyed by an opaque delta label, e.g. ``"D>A"`` / ``"D>C'"``):

      - ``paired_point_estimates`` -- Artifact A (paired draw): the per-shuffle δ point estimate
        (``mean_delta``), one per shuffle. Drives the CENTERING gate.
      - ``paired_ci_bounds`` -- Artifact A (paired draw): the per-shuffle δ-CI ``(lo, hi)``. Drives
        the paired-coverage REPORT (report-only -- the paired sign-flip's restricted permutation has
        a different, un-calibrated tail than the 0.08 ceiling was derived for).
      - ``global_ci_bounds`` -- Artifact B (global draw, independent RNG): the per-shuffle δ-CI
        ``(lo, hi)``. Drives the COVERAGE-RATE gate (the smooth global-permutation tail the 0.08
        ceiling was derived against).

    Frozen. CENTERING (first moment, Artifact A) and COVERAGE-RATE (tail, Artifact B) are
    STATISTICALLY ORTHOGONAL and gate off DIFFERENT artifacts; paired-coverage (tail, Artifact A) is
    report-only."""

    model_config = ConfigDict(frozen=True)

    n_shuffles: int
    paired_point_estimates: dict[str, tuple[float, ...]]
    paired_ci_bounds: dict[str, tuple[tuple[float, float], ...]]
    global_ci_bounds: dict[str, tuple[tuple[float, float], ...]]


_GLOBAL_PASS_SEED_SALT = 0x6C0BA1  # "global" -- the Artifact-B RNG-stream separator (Finding-1).


def shuffle_null_centering(
    cells: Sequence[Cell],
    *,
    paired_ids: Sequence[tuple[int, int]],
    unpaired_ids: Sequence[int],
    deltas: Mapping[str, tuple[str, str]],
    n_shuffles: int = 200,
    seed: int,
    error_strata: Sequence[Stratum] = ("K",),
    n_outer: int,
    bootstrap_seed: int = 0xB007,
    _shuffle: _ShuffleFn | None = None,
) -> ShuffleNullResult:
    """Run the §5 stratum-membership shuffle null as TWO independent shuffle passes per shuffle idx
    (Finding-1 fix). For each of ``n_shuffles`` (>=200) indices:

      - **Artifact A (paired draw).** Relabel ONCE with the full §5 shuffle (within-pair sign-flip +
        fixed-marginal global permute) under the A-stream RNG; run
        :func:`~cogworx.eval.youden.nested_bootstrap_delta` for EVERY delta on that ONE artifact ->
        the paired point estimates (CENTERING) + paired CIs (paired-coverage REPORT).
      - **Artifact B (global draw).** Relabel AGAIN with the SAME full §5 shuffle under an
        INDEPENDENT B-stream RNG (the A-stream seed salted with :data:`_GLOBAL_PASS_SEED_SALT`);
        run the deltas on that artifact -> the global CIs (COVERAGE-RATE gate).

    Both passes are the full §5 null (sign-flip + global-permute); they differ ONLY in their RNG
    draw, so they are two independent draws of the SAME null, not two different nulls. The Finding-1
    fix is that CENTERING and COVERAGE-RATE no longer share one artifact: the paired sign-flip's
    restricted permutation has an un-calibrated tail the global-null-derived 0.08 ceiling must NOT
    gate, so coverage runs on the smooth global draw and the paired tail is report-only.

    ``deltas`` maps an opaque delta label to ``(arm_a, arm_b)`` (e.g. ``{"D>A": ("D", "A"),
    "D>C'": ("D", "C")}``). Each delta's :func:`nested_bootstrap_delta` runs at the gate
    ``quantile`` via the look-corrected CI the gate uses; here it defaults to the function's own
    0.025 (the within-run Bonferroni half); the caller threads a tighter quantile by wrapping.

    The ``n_shuffles`` draws SHARE one frozen ``cells`` artifact (each shuffle relabels a COPY), so
    they are positively correlated -- the §5.0 honest caveat (see :func:`assert_shuffle_null`).

    :param deltas: label -> ``(arm_a, arm_b)`` for every binding delta to null.
    :param n_shuffles: number of permutations (>=200; the §5.0 MC-SE bar -- ~0.006-0.013 at 200).
    :param seed: seeds the master RNG that derives each shuffle pass's RNG (reproducible). The
        B-pass master is ``seed ^ _GLOBAL_PASS_SEED_SALT`` so its draws differ from the A-pass.
    :param error_strata: the error population for ``nested_bootstrap_delta`` (default ``("K",)``).
    :param n_outer: outer-bootstrap iterations per delta per shuffle (the cost driver -- keep small
        in unit tests, sim-wide >=10,000 in the real gate). NB: this now runs over TWO passes.
    :param bootstrap_seed: the per-shuffle bootstrap seed base (XOR'd with the shuffle index; the
        B-pass uses ``bootstrap_seed ^ _GLOBAL_PASS_SEED_SALT`` so its bootstrap draws differ).
    :param _shuffle: test-only injection point for the LEAKY shuffle variant (mutation test);
        defaults to the honest :func:`shuffle_stratum_membership`. Never pass in production. Drives
        BOTH passes (so the mutation fires on the centering gate, where its teeth are).
    """
    shuffle_fn: _ShuffleFn = shuffle_stratum_membership if _shuffle is None else _shuffle
    master_a = Random(seed)
    master_b = Random(seed ^ _GLOBAL_PASS_SEED_SALT)
    boot_b = bootstrap_seed ^ _GLOBAL_PASS_SEED_SALT

    paired_points: dict[str, list[float]] = {label: [] for label in deltas}
    paired_cis: dict[str, list[tuple[float, float]]] = {label: [] for label in deltas}
    global_cis: dict[str, list[tuple[float, float]]] = {label: [] for label in deltas}

    for k in range(n_shuffles):
        # Artifact A -- the paired-centering draw.
        rng_a = Random(master_a.randrange(2**31))
        relabeled_a = shuffle_fn(cells, paired_ids=paired_ids, unpaired_ids=unpaired_ids, rng=rng_a)
        # Artifact B -- the independent global-coverage draw (same full §5 null, separate stream).
        rng_b = Random(master_b.randrange(2**31))
        relabeled_b = shuffle_fn(cells, paired_ids=paired_ids, unpaired_ids=unpaired_ids, rng=rng_b)

        for label, (arm_a, arm_b) in deltas.items():
            mean_a, lo_a, hi_a = nested_bootstrap_delta(
                relabeled_a,
                arm_a=arm_a,
                arm_b=arm_b,
                error_strata=error_strata,
                n_outer=n_outer,
                seed=bootstrap_seed ^ k,
            )
            paired_points[label].append(mean_a)
            paired_cis[label].append((lo_a, hi_a))

            _, lo_b, hi_b = nested_bootstrap_delta(
                relabeled_b,
                arm_a=arm_a,
                arm_b=arm_b,
                error_strata=error_strata,
                n_outer=n_outer,
                seed=boot_b ^ k,
            )
            global_cis[label].append((lo_b, hi_b))

    return ShuffleNullResult(
        n_shuffles=n_shuffles,
        paired_point_estimates={label: tuple(v) for label, v in paired_points.items()},
        paired_ci_bounds={label: tuple(v) for label, v in paired_cis.items()},
        global_ci_bounds={label: tuple(v) for label, v in global_cis.items()},
    )


class ShuffleNullReport(BaseModel):
    """The report-only read-out :func:`assert_shuffle_null` returns when the gating assertions PASS
    (Finding-1 fix). Carries, per binding delta, the PAIRED-coverage tail that is REPORT-ONLY --
    NEVER gated. The within-pair sign-flip is a restricted, difficulty-matched permutation whose
    tail geometry differs from the smooth global permutation the 0.08 ceiling was calibrated
    against, so its rate is surfaced for the reviewer (the same spirit as the raw ``k/n`` exposure)
    but cannot refuse the lock.

      - ``paired_k`` -- per delta, ``k_paired`` = #shuffles whose PAIRED δ-CI has ``lo > 0`` (the
        leakage direction). Reads Artifact A's ``paired_ci_bounds``.
      - ``paired_coverage_ucb95`` -- per delta, the Beta-UCB ``1 - beta.lcb(n-k+1, k+1, 0.05)`` on
        the paired exclusion rate. EXPOSED, never compared to a ceiling.

    Frozen. ``n_shuffles`` is carried so ``paired_k/n`` is recoverable by the reader."""

    model_config = ConfigDict(frozen=True)

    n_shuffles: int
    paired_k: dict[str, int]
    paired_coverage_ucb95: dict[str, float]


def assert_shuffle_null(
    result: ShuffleNullResult,
    *,
    coverage_ucb_max: float = 0.08,
    centering_abs_max: float = 0.05,
) -> ShuffleNullReport:
    """Lock precondition (plan §5; Finding-1 fix): assert the stratum-shuffle null is clean for
    EVERY delta. Two statistically-orthogonal GATES, each read off a DIFFERENT artifact, plus one
    report-only tail -- a corpus must pass BOTH gates (they fail on opposite mutations, so passing
    requires genuine no-leakage). On pass, returns a :class:`ShuffleNullReport` carrying the
    report-only paired-coverage tail.

    - **CENTERING (Q-A) -- the first moment, on Artifact A (the PAIRED draw).**
      ``mean = fmean(paired_point_estimates)``; ``se = pstdev(...)/sqrt(n_shuffles)``; refuse unless
      ``abs(mean) < centering_abs_max`` **AND** the normal-approx 95% interval of the shuffle-δ MEAN
      (``mean ± 1.96·se``) contains 0. Centering is provably confound-robust -- E[J_shuf]=0
      term-by-term regardless of any difficulty confound -- and is the load-bearing leak detector,
      so it gates on the strong within-pair sign-flip (Finding-1: the LEAKY mutation fires it).
      This is a normal-approx on the shuffle-δ MEAN -- NOT a percentile of the draws; the ``|mean| <
      0.05`` belt-and-suspenders floor handles the degenerate tiny-sigma case where the CI is a
      hair-thin interval that could exclude 0 at a near-zero mean.

    - **COVERAGE-RATE (Q-B) -- the tail, on Artifact B (the GLOBAL draw).** ``k = #shuffles whose
      per-shuffle GLOBAL δ-CI has lo > 0`` (the LEAKAGE direction ONLY -- ``hi < 0`` is not a
      leakage signature and is NOT counted); ``ucb95 = 1.0 - beta.lcb(n_shuffles - k + 1, k + 1,
      quantile=0.05)`` (the Bayes-Laplace ``Beta(k+1, n-k+1)`` 95th percentile via the lcb symmetry
      identity, S2-clean -- no scipy); refuse if ``ucb95 > coverage_ucb_max``. **The 0.08 ceiling
      was derived for the GLOBAL null's smooth 2.5% one-sided-high tail** (k>=10 at n=200 -> ~3%
      honest false-refuse), so coverage MUST run on the global draw, not the paired one (Finding-1:
      the paired sign-flip's restricted tail false-refuses an honest difficulty-confounded corpus
      against this ceiling). A UCB (not an LCB) is the right tool: it asks "could the true exclusion
      rate plausibly be as high as a leaky rate?" The raw ``k/n`` is exposed on the message so the
      bound is not laundered.

      **The 0.08 global ceiling is PROVISIONAL** -- it is the placeholder pending an eval-stats
      Monte-Carlo (MC-2) that pins the ceiling against the realized cross-shuffle correlation before
      the gate locks. Do NOT treat it as final-calibrated.

    - **paired-coverage -- REPORT-ONLY, on Artifact A (the PAIRED draw).** ``k_paired`` + its
      Beta-UCB are computed off ``paired_ci_bounds`` and returned on the :class:`ShuffleNullReport`.
      It is NEVER compared to a ceiling and NEVER raises -- the paired sign-flip's restricted draw
      has a different, un-calibrated tail than the global ceiling assumes; it is surfaced for the
      reviewer, not gated (Finding-1).

    **Honest caveat (§5.0):** the ``n_shuffles`` outcomes are NOT iid -- they share the one frozen
    artifact, so ``k``'s exclusions are positively correlated and the Beta-UCB UNDERSTATES the true
    uncertainty (effective n < ``n_shuffles``); the ~3% false-refuse is a LOWER bound on the real
    rate. The 0.08 ceiling's ~2.2x slack over nominal absorbs moderate correlation; the raw ``k/n``
    is reported alongside the bound. The mutation test (the LEAKY variant) proves the centering
    assertion can be made to fire -- an assertion that cannot fail tests nothing.

    :returns: a :class:`ShuffleNullReport` with the report-only paired-coverage tail per delta.
    :raises CorpusLockError: for any delta failing centering (mean magnitude or CI excludes 0, on
        the PAIRED draw) or coverage-rate (``ucb95 > coverage_ucb_max``, on the GLOBAL draw).
    """
    n = result.n_shuffles

    # CENTERING gate -- first moment of the PAIRED draw (Artifact A).
    for label, points in result.paired_point_estimates.items():
        mean = statistics.fmean(points)
        se = statistics.pstdev(points) / math.sqrt(n) if n > 0 else 0.0
        lo_ci, hi_ci = mean - 1.96 * se, mean + 1.96 * se
        if not (abs(mean) < centering_abs_max and lo_ci <= 0.0 <= hi_ci):
            raise CorpusLockError(
                f"§5 shuffle-null centering: delta '{label}' has mean(shuffle_δ)={mean:.6g} "
                f"(95% interval [{lo_ci:.6g}, {hi_ci:.6g}], se={se:.6g}) -- the first moment does "
                f"not center at 0 (|mean| >= {centering_abs_max} or the interval excludes 0); "
                f"flags track the label through something other than the binding the gate claims"
            )

    # COVERAGE-RATE gate -- tail of the GLOBAL draw (Artifact B); the 0.08 ceiling's calibrated null
    for label, ci_series in result.global_ci_bounds.items():
        k = sum(1 for lo, _ in ci_series if lo > 0.0)  # leakage-direction exclusions ONLY
        ucb95 = 1.0 - beta.lcb(n - k + 1, k + 1, quantile=0.05)
        if ucb95 > coverage_ucb_max:
            raise CorpusLockError(
                f"§5 shuffle-null coverage-rate: delta '{label}' has k={k}/{n} leakage-direction "
                f"GLOBAL δ-CI exclusions (lo>0); the Beta 95%-UCB on the true exclusion rate "
                f"ucb95={ucb95:.6g} exceeds the ceiling {coverage_ucb_max} (raw rate {k / n:.4g}) "
                f"-- a leaky subset drives lo>0 in too many shuffles even if the mean cancels"
            )

    # paired-coverage -- REPORT-ONLY tail of the PAIRED draw (Artifact A); NEVER gated (Finding-1).
    paired_k: dict[str, int] = {}
    paired_ucb95: dict[str, float] = {}
    for label, ci_series in result.paired_ci_bounds.items():
        kp = sum(1 for lo, _ in ci_series if lo > 0.0)
        paired_k[label] = kp
        paired_ucb95[label] = 1.0 - beta.lcb(n - kp + 1, kp + 1, quantile=0.05)

    return ShuffleNullReport(
        n_shuffles=n, paired_k=paired_k, paired_coverage_ucb95=paired_ucb95
    )


# ---------------------------------------------------------------------------
# INSTRUMENT 2 -- §2.B per-regime contribution bound + LORO
# ---------------------------------------------------------------------------

REGIME_CONTRIBUTION_MAX_SHARE: float = 0.30
"""The §2.B per-regime contribution ceiling -- no single error regime may contribute more than this
share of the pooled K-sensitivity (level-share, plan §2.B, eval-stats Q-C).

**This is the eval-stats Q-C CORRECTION of the plan's rev-3 0.40.** eval-stats reproduced that the
plan's 0.40 gives ~0% honest false-refuse and near-zero catch power (too cold), while **0.30
delivers the intended ~2% honest false-refuse** at ~16 items/regime. The knee analysis: 25% -> ~9%
false-refuse (too hot), 30% -> ~2%, 40% -> ~0% (too cold). PROVISIONAL on the realized ``sens_D``
from 4.4c-0 -- re-derive the knee at the realized sensitivity, not the planning point, before gate
locks. Named (not a magic number) so a re-derivation moves exactly one constant."""

REGIME_ABSTAIN_MAX_FRACTION: float = 0.15
"""The §2.C abstain-fraction ceiling -- the STRUCTURAL floor under the second-author K-regime audit
(Finding-2). The per-regime contribution bound (:data:`REGIME_CONTRIBUTION_MAX_SHARE`) only sees
ATTRIBUTED regimes; a loaded regime mis-tagged as K-abstain (``regime==""``) is counted in the
pooled δ but EXCLUDED from the shares, so it could evade the contribution bound by dumping into the
unattributable bucket. This cap refuses lock when ``abstain_excluded / n_error`` exceeds it: belt
to the second-author audit's suspenders -- a structural bound that holds even if the audit mis-tags.
It caps the SIZE of the blind spot the contribution bound cannot see.

PROVISIONAL on the realized 4.4c-4 honest abstention rate -- re-derive 0.15 against the realized
honest-abstain fraction before the gate locks (the placeholder assumes <=15% of K items legitimately
abstain; if realized honest abstention runs higher, this cap false-refuses and must move). Named so
a re-derivation moves exactly one constant."""


class RegimeContributionReport(BaseModel):
    """The §2.B per-regime contribution shares (level-share of pooled K-sensitivity) for the
    artifact log. ``shares`` maps each regime to its share of the pooled effect; ``max_share`` is
    the ceiling the lock enforced; ``abstain_excluded`` is the count of ``regime==""`` (K-abstain,
    §2.C) error items counted in the pooled δ but EXCLUDED from the per-regime shares. Frozen."""

    model_config = ConfigDict(frozen=True)

    shares: dict[str, float]
    max_share: float
    abstain_excluded: int


def assert_regime_contribution(
    cells: Sequence[Cell],
    *,
    arm_b: str = "A",
    error_stratum: Stratum = "K",
    max_share: float = REGIME_CONTRIBUTION_MAX_SHARE,
    abstain_max_fraction: float = REGIME_ABSTAIN_MAX_FRACTION,
) -> RegimeContributionReport:
    """Lock precondition (plan §2.B, eval-stats Q-C): the per-regime contribution bound. Refuses
    corpus-lock (loud :exc:`CorpusLockError`) when any single error regime carries more than
    ``max_share`` of the pooled error-stratum sensitivity -- the un-confounded core of HIGH-1 (a
    planter loading the easy regimes D happens to win, defeating the count-balance + content-hash).

    **BASIS (eval-stats Q-C, binding): level-share of pooled K-SENSITIVITY, NOT paired-δ share.**
    Per regime ``r``: ``share_r = sens_r * n_r / sum_r(sens_r * n_r)`` where ``sens_r`` is the
    per-regime :func:`~cogworx.eval.youden._sens_spec` sensitivity over regime-``r``'s error subset
    (a POINT estimate -- no bootstrap) and ``n_r`` is that subset's item count. Refuse lock if any
    ``share_r > max_share``.

    ``regime==""`` items (K-abstain, §2.C: an item whose K-regime tag the second-author audit could
    not confirm) are counted in the pooled δ (they remain error-stratum cells the bootstrap scores)
    but are EXCLUDED from the per-regime shares (unattributable, not forced into a regime). Their
    count is returned on the report for the artifact log.

    **The abstain-fraction cap (Finding-2): the STRUCTURAL floor under the §2.C second-author
    audit.** The contribution bound above sees only ATTRIBUTED regimes, so a regime mis-tagged
    as K-abstain (dumped into ``regime==""``) is counted in the pooled δ but invisible to the share
    ceiling -- a blind spot the audit alone (which could mis-tag) does not structurally close. This
    function REFUSES lock (loud :exc:`CorpusLockError`) if
    ``abstain_excluded / n_error > abstain_max_fraction`` (default
    :data:`REGIME_ABSTAIN_MAX_FRACTION` = 0.15). Belt to the second-author audit's suspenders: a
    mis-tagged K-abstain item cannot evade the contribution bound by dumping into the unattributable
    bucket, because the SIZE of that bucket is itself capped. PROVISIONAL on the realized 4.4c-4
    honest-abstain fraction (see :data:`REGIME_ABSTAIN_MAX_FRACTION`).

    The ``arm_b`` parameter pins the binding reduction's comparison arm. **The eval-stats Q-C ruling
    was framed on the D>A reduction (J_A ≡ 0 on K -> δ = sens_D), where pure sens-share IS the
    contribution**, so the sens-share basis is the only wired basis and ``arm_b`` defaults to ``A``.
    The C' paired-δ basis is NOT wired (S12/YAGNI): a non-default ``arm_b`` is REFUSED (Finding-3),
    the two bases are not interchangeable and silently switching would mis-measure the contribution.
    **If a future caller needs the paired-δ form for the C' arm, FLAG it to eval-stats** before
    wiring a second basis; do not pass a non-default ``arm_b`` to coerce it.

    Bound to :func:`~cogworx.eval.youden._sens_spec` / ``_index_cells`` (the EXACT scoring path the
    gate runs). PURE, offline, deterministic: no model calls, no substrate, no journal (CANON
    S1/S2).

    :param arm_b: the comparison arm. MUST be the default ``"A"`` -- the sens-share basis is pinned
        on the D>A reduction (J_A ≡ 0 -> δ = sens_D); the C' paired-δ basis is NOT wired (Finding-3,
        S12/YAGNI). A non-default value is REFUSED.
    :param error_stratum: the error stratum whose sensitivity is partitioned by regime (default
        ``"K"`` -- the binding D>A error population).
    :param max_share: the contribution ceiling (default :data:`REGIME_CONTRIBUTION_MAX_SHARE`).
    :param abstain_max_fraction: the K-abstain fraction ceiling (default
        :data:`REGIME_ABSTAIN_MAX_FRACTION`); ``abstain_excluded / n_error`` above this refuses
        lock.
    :raises CorpusLockError: if ``arm_b`` is non-default (Finding-3, the unwired C' basis), if the
        artifact has no error-stratum cells (nothing to audit), if the K-abstain fraction exceeds
        ``abstain_max_fraction`` (Finding-2), if the pooled sensitivity weight is zero (a degenerate
        all-miss arm), or if any regime's share exceeds ``max_share``.
    """
    if arm_b != "A":
        raise CorpusLockError(
            f"§2.B regime-contribution: arm_b={arm_b!r} is not the default 'A' -- the sens-share "
            f"basis is pinned on the D>A reduction (J_A ≡ 0 on K -> δ = sens_D); the C' paired-δ "
            f"basis is NOT wired (Finding-3, S12/YAGNI). Flag to eval-stats before switching bases "
            f"-- do not coerce the C' form via arm_b."
        )

    cells_by_item, items_in_stratum = _index_cells(cells)
    error_ids = items_in_stratum.get(error_stratum, [])
    if not error_ids:
        raise CorpusLockError(
            f"§2.B regime-contribution: the artifact has no '{error_stratum}' error-stratum cells "
            f"-- the per-regime contribution bound has nothing to audit"
        )

    # The arm whose K-sensitivity is partitioned by regime is arm_a's binding error arm. The gate's
    # D>A reduction makes this arm D; we read the regime of each error item off its cells (regime is
    # item-welded, like stratum -- the first error cell of the item names it).
    arm = _arm_a_of_error(cells, error_stratum)

    regime_of: dict[int, str] = {}
    for c in cells:
        if c.stratum == error_stratum and c.item_id not in regime_of:
            regime_of[c.item_id] = c.regime

    ids_by_regime: dict[str, list[int]] = defaultdict(list)
    abstain_excluded = 0
    for item_id in error_ids:
        r = regime_of.get(item_id, "")
        if r == "":
            abstain_excluded += 1  # counted in pooled δ, EXCLUDED from per-regime shares (§2.C)
            continue
        ids_by_regime[r].append(item_id)

    # Finding-2 -- the STRUCTURAL abstain-fraction cap. Caps the SIZE of the unattributable bucket
    # the per-regime share ceiling cannot see, so a loaded regime cannot evade the bound by
    # mis-tagging into regime=="". Belt to the §2.C second-author audit's suspenders.
    n_error = len(error_ids)
    abstain_fraction = abstain_excluded / n_error if n_error > 0 else 0.0
    if abstain_fraction > abstain_max_fraction:
        raise CorpusLockError(
            f"§2.C regime-abstain: K-abstain fraction {abstain_fraction:.6g} "
            f"({abstain_excluded}/{n_error} regime=='' items) exceeds the ceiling "
            f"{abstain_max_fraction} -- too many error items dumped into the unattributable "
            f"bucket; a loaded regime mis-tagged as K-abstain could evade the per-regime "
            f"contribution bound by hiding there (Finding-2, structural floor under the audit)"
        )

    weights: dict[str, float] = {}
    for r, ids in ids_by_regime.items():
        sens_r, _ = _sens_spec(cells_by_item, ids, arm, [])
        weights[r] = sens_r * len(ids)

    total = sum(weights.values())
    if total <= 0.0:
        raise CorpusLockError(
            f"§2.B regime-contribution: the pooled sensitivity weight over '{error_stratum}' is "
            f"{total:.6g} (<=0) -- the error arm flags nothing on the attributed regimes, so no "
            f"contribution share is defined (a degenerate all-miss arm cannot be cleared)"
        )

    shares = {r: w / total for r, w in weights.items()}
    for r, share in sorted(shares.items()):
        if share > max_share:
            raise CorpusLockError(
                f"§2.B regime-contribution: regime '{r}' contributes share={share:.6g} of the "
                f"pooled '{error_stratum}'-sensitivity, exceeding the ceiling {max_share} -- a "
                f"single regime carrying more than its fair share loads the pooled J (HIGH-1)"
            )

    return RegimeContributionReport(
        shares=shares, max_share=max_share, abstain_excluded=abstain_excluded
    )


def _arm_a_of_error(cells: Sequence[Cell], error_stratum: Stratum) -> str:
    """The single arm whose error-stratum sensitivity the contribution bound partitions by regime.

    The §2.B ruling is framed on the D>A reduction where ``δ = sens_D``, so the contribution is read
    off arm D -- the error arm that actually flags. A frozen 4.4d artifact carries multiple arms; we
    pick the arm with the HIGHEST pooled error-stratum sensitivity (arm D by construction, the one
    whose regime-loading the bound defends against). Deterministic: ties break on arm label."""
    cells_by_item, items_in_stratum = _index_cells(cells)
    error_ids = items_in_stratum.get(error_stratum, [])
    arms = sorted({c.arm for c in cells if c.stratum == error_stratum})
    best_arm = arms[0]
    best_sens = -1.0
    for arm in arms:
        sens, _ = _sens_spec(cells_by_item, error_ids, arm, [])
        if sens > best_sens:
            best_sens = sens
            best_arm = arm
    return best_arm


class LoroReport(BaseModel):
    """Leave-one-regime-out (LORO) robustness read-out (plan §2.B part 1, REPORTED-ONLY). Per regime
    ``r``: ``collapses[r]`` is True iff the binding pooled δ-CI lower bound goes ``<= 0`` when all
    regime-``r`` error items are dropped. ``full_lo`` is the all-regime pooled δ-CI lower bound (the
    baseline). Frozen.

    REPORTED-ONLY: a collapse is a flag to INSPECT, never an automatic refuse (LORO-as-a-gate has
    ~30% honest false-refuse, ``1-(1-0.07)^5 ≈ 0.30`` -- MED-D). The §2.B per-regime contribution
    bound (:func:`assert_regime_contribution`) is the actual single-regime lock defense."""

    model_config = ConfigDict(frozen=True)

    full_lo: float
    collapses: dict[str, bool]


def report_loro(
    cells: Sequence[Cell],
    *,
    arm_b: str,
    error_stratum: Stratum = "K",
    n_outer: int,
    seed: int,
    arm_a: str | None = None,
) -> LoroReport:
    """Compute the §2.B leave-one-regime-out (LORO) robustness report -- **REPORTED-ONLY, NEVER
    RAISES** (plan §2.B part 1, MED-D demote).

    For each regime ``r`` present on the error stratum, recompute the binding pooled δ-CI with all
    regime-``r`` error items dropped, and report whether the lo bound collapses (``<= 0``). The
    all-regime baseline ``full_lo`` is reported alongside.

    **Why reported-only (MED-D / red-team third pass):** as a LOCK gate ("any single-regime collapse
    => refuse") this false-refuses an HONEST corpus with ~30% probability -- when the binding margin
    is sized to barely clear Δmin, dropping ~16 of 80 K items can tip any one of the 5 regimes'
    leave-1-out CIs below 0 by sampling noise alone, and the gate fires if ANY of the 5 does
    (``1-(1-0.07)^5 ≈ 0.30``). The contribution bound (:func:`assert_regime_contribution`, gating)
    is the actual single-regime defense; LORO is logged for the reviewer.

    Drops are applied by filtering the FLAT cell list on ``c.regime`` (every cell of a dropped item
    goes -- item-granular, §3.4), then re-running
    :func:`~cogworx.eval.youden.nested_bootstrap_delta` on the survivors. PURE, offline,
    deterministic: no model calls, no substrate (CANON S1/S2).

    :param arm_b: the comparison arm for the binding δ (e.g. ``"A"`` for D>A, ``"C"`` for D>C').
    :param arm_a: the antithesis arm (defaults to the highest-sensitivity error arm -- arm D).
    :param error_stratum: the error stratum dropped-by-regime (default ``"K"``).
    :param n_outer: outer-bootstrap iterations (keep small in unit tests).
    :param seed: the bootstrap seed (reproducible).
    """
    a = arm_a if arm_a is not None else _arm_a_of_error(cells, error_stratum)
    _, full_lo, _ = nested_bootstrap_delta(
        cells, arm_a=a, arm_b=arm_b, error_strata=(error_stratum,), n_outer=n_outer, seed=seed
    )

    regimes = sorted(
        {c.regime for c in cells if c.stratum == error_stratum and c.regime != ""}
    )
    collapses: dict[str, bool] = {}
    for r in regimes:
        # Drop every cell of a regime-r ERROR item (item-granular). Clean/other-stratum cells stay.
        survivors = [
            c for c in cells if not (c.stratum == error_stratum and c.regime == r)
        ]
        _, lo, _ = nested_bootstrap_delta(
            survivors,
            arm_a=a,
            arm_b=arm_b,
            error_strata=(error_stratum,),
            n_outer=n_outer,
            seed=seed,
        )
        collapses[r] = lo <= 0.0

    return LoroReport(full_lo=full_lo, collapses=collapses)


# ---------------------------------------------------------------------------
# INSTRUMENT 3 -- §3.3 KS flag-rate diagnostic (reported-only)
# ---------------------------------------------------------------------------


class KSReport(BaseModel):
    """The §3.3 one-sided KS flag-rate diagnostic read-out (REPORTED-ONLY). ``ks_statistic`` is the
    one-sided Kolmogorov-Smirnov statistic between the per-item flag-rate distributions of the error
    stratum and the clean stratum for one arm; ``d_crit`` is the
    ``1.22·sqrt((n_K+n_clean)/(n_K·n_clean))`` critical value (≈0.19 at n=80); ``exceeds_crit`` is
    ``ks_statistic > d_crit`` (a sanity flag, NOT a refusal). ``n_error`` / ``n_clean`` are the
    per-item-rate sample sizes. Frozen.

    REPORTED-ONLY -- a flag-rate-match read-out the matched-sibling construction (§3.3 item 1) makes
    the actual difficulty control; the KS is circular as a gate (flag rate is an arm OUTPUT, not an
    item property -- MED-1) so it NEVER blocks the lock."""

    model_config = ConfigDict(frozen=True)

    ks_statistic: float
    d_crit: float
    exceeds_crit: bool
    n_error: int
    n_clean: int


def ks_flag_rate_diagnostic(
    cells: Sequence[Cell],
    *,
    arm: str = "C",
    error_stratum: Stratum = "K",
) -> KSReport:
    """The §3.3 reported-only KS flag-rate diagnostic -- **NEVER BLOCKS the lock** (MED-1).

    A sanity read on whether the error and clean pools produce comparable per-item flag-rate
    distributions for ``arm`` (the spec arm by default). Per-item flag rate ``p̂_i = #flagged/R``
    over the error stratum vs the clean stratum (generalizing :func:`_clean_flag_rates` to ANY
    stratum -- the single source of truth for the per-item rate). The one-sided KS statistic is the
    max positive gap between the two empirical CDFs;
    ``D_crit = 1.22·sqrt((n_K+n_clean)/(n_K·n_clean))``.

    Pure :mod:`statistics` stdlib -- no scipy, no numpy. It is LOGGED, never refuses: the
    matched-sibling construction (§3.3) is the difficulty control; KS is a flag-rate-match read-out
    only (flag rate is an arm output, not an item property -- gating on it is circular, MED-1).

    :param arm: the arm whose flag-rate distributions are compared (default ``"C"``, the spec arm).
    :param error_stratum: the error stratum compared against ``"clean"`` (default ``"K"``).
    """
    cells_by_item, items_in_stratum = _index_cells(cells)
    error_ids = items_in_stratum.get(error_stratum, [])
    clean_ids = items_in_stratum.get("clean", [])

    error_rates = sorted(_clean_flag_rates(cells_by_item, error_ids, arm))
    clean_rates = sorted(_clean_flag_rates(cells_by_item, clean_ids, arm))
    n_e, n_c = len(error_rates), len(clean_rates)

    if n_e == 0 or n_c == 0 or (n_e + n_c) == 0:
        # Degenerate -- nothing to compare. Report a zero statistic + a non-finite (inf) d_crit.
        return KSReport(
            ks_statistic=0.0,
            d_crit=float("inf"),
            exceeds_crit=False,
            n_error=n_e,
            n_clean=n_c,
        )

    # One-sided KS: the max of (CDF_clean - CDF_error) over the pooled support (the positive gap).
    grid = sorted(set(error_rates) | set(clean_rates))
    ks = 0.0
    for x in grid:
        cdf_e = sum(1 for r in error_rates if r <= x) / n_e
        cdf_c = sum(1 for r in clean_rates if r <= x) / n_c
        ks = max(ks, cdf_c - cdf_e)

    d_crit = 1.22 * math.sqrt((n_e + n_c) / (n_e * n_c))
    return KSReport(
        ks_statistic=ks,
        d_crit=d_crit,
        exceeds_crit=ks > d_crit,
        n_error=n_e,
        n_clean=n_c,
    )


__all__ = [
    "MASTER_SEED",
    "REGIME_ABSTAIN_MAX_FRACTION",
    "REGIME_CONTRIBUTION_MAX_SHARE",
    "RESIDUAL_EPSILON_UNAUDITED",
    "BijectionResult",
    "CorpusLockError",
    "ExecEnvIdentity",
    "KSReport",
    "LoroReport",
    "MeasurementFingerprint",
    "RegimeContributionReport",
    "ShuffleNullReport",
    "ShuffleNullResult",
    "assert_arm_a_floor",
    "assert_no_contamination",
    "assert_no_verifiable_claim",
    "assert_regime_contribution",
    "assert_shuffle_null",
    "assert_spec_ceiling",
    "build_fingerprint",
    "content_hash",
    "ks_flag_rate_diagnostic",
    "lock_corpus",
    "read_exec_env",
    "read_git_sha",
    "report_loro",
    "revalidate_bijection",
    "shuffle_null_centering",
    "shuffle_stratum_membership",
]
