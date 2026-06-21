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
"""

from __future__ import annotations

import hashlib
import json
import locale
import platform
import statistics
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from importlib import metadata

from pydantic import BaseModel, ConfigDict, Field

from cogworx.eval.corpus import CorpusItem
from cogworx.eval.youden import Cell, Stratum, _index_cells, _sens_spec

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


__all__ = [
    "MASTER_SEED",
    "RESIDUAL_EPSILON_UNAUDITED",
    "BijectionResult",
    "CorpusLockError",
    "ExecEnvIdentity",
    "MeasurementFingerprint",
    "assert_arm_a_floor",
    "assert_no_contamination",
    "assert_spec_ceiling",
    "build_fingerprint",
    "content_hash",
    "lock_corpus",
    "read_exec_env",
    "read_git_sha",
    "revalidate_bijection",
]
