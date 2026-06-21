"""The Phase-4 GATE verdict assembly (Pod 4.4e-0; plan §5/§13.5).

:func:`score_gate` is the PURE verdict function: given the frozen 4.4d
:class:`~cogworx.eval.youden.Cell` artifact, the locked
:class:`~cogworx.eval.corpus.CorpusItem` truth table, and the corpus
:class:`~cogworx.eval.lock.MeasurementFingerprint`, it drives the corpus-lock instruments (lock.py)
and the binding nested-bootstrap comparison (youden.py) and assembles a single content-addressable
:class:`GateVerdict`. It DECIDES nothing the instruments do not already encode — it ORDERS them into
the architect's tiered gate and short-circuits on the FIRST failing tier.

THE TIERED GATE (short-circuit; the first failing tier sets ``verdict_status``):

  - **Tier 0 — fingerprint integrity (-> INSTRUMENT_INVALID).** The fingerprint must be audited
    (``epsilon_audited``); if an ``expected_fingerprint`` is supplied, its ``digest`` must match. An
    unaudited or mismatched instrument is REFUSED before any statistic runs.
  - **Tier 1 — instrument-validity gates (-> INSTRUMENT_INVALID).** The arm-A floor
    (:func:`~cogworx.eval.lock.assert_arm_a_floor`), the clean-spec ceiling
    (:func:`~cogworx.eval.lock.assert_spec_ceiling`), the per-regime contribution bound
    (:func:`~cogworx.eval.lock.assert_regime_contribution`), and the §5 stratum-shuffle null
    (``shuffle_null_centering`` + :func:`~cogworx.eval.lock.assert_shuffle_null`) over **D>C' only**
    (the floor delta D>A is EXCLUDED — the degenerate floor arm A has no §5 cancellation partner;
    its validity is carried by the arm-A floor + the Tier-2 positive control). Each raises
    :class:`~cogworx.eval.lock.CorpusLockError`; the scorer CATCHES it and REPORTS a failed
    :class:`InstrumentGateReport` — it never crashes. Any failure -> INSTRUMENT_INVALID.
  - **Tier 2 — controls (-> INSTRUMENT_INVALID).** The positive-control arm's Youden-J lower bound
    must clear ``j_pos_min`` (the instrument can detect a KNOWN effect), and the B strawman must
    behave (its binding δ-CI must NOT clear the threshold — a yes-machine that "wins" means the
    instrument is mis-wired). Read off the Cell artifact. The positive control is a DEDICATED arm
    (default ``"PC"`` — the scripted known-flag-pattern stub
    :func:`~cogworx.eval.runner.scripted_executor` emits under this label in the live 5-arm run),
    NOT the arm-under-test D: reading the control off D is a circularity (the gate would certify the
    instrument on the very arm whose effect it is measuring). A MISSING ``"PC"`` (or
    ``strawman_arm``) means the pipeline didn't run the required control -> INSTRUMENT_INVALID (the
    missing-arm path, below).

THE SCORER NEVER CRASHES ON A MISSING ARM. Every
:func:`~cogworx.eval.youden.nested_bootstrap_delta` read (the two Tier-2 controls AND the two Tier-4
binding deltas) names an arm that MUST be present in the Cell artifact. If the artifact is MISSING
that arm, the bootstrap raises (``StopIteration`` / ``KeyError`` off the empty per-item arm urn);
the scorer CATCHES it and produces an ``INSTRUMENT_INVALID`` verdict with an
:class:`InstrumentGateReport` naming the absent arm — it REPORTS, never crashes (the module's
load-bearing contract).
  - **Tier 3 — lineage budget (-> BUDGET_EXHAUSTED).** If ``lineage_look_count >= look_budget_max``
    the look budget is spent; the gate short-circuits (the journal READ is 4.4e-1's async wrapper —
    here the count is a passed-in int).
  - **Tier 4 — the binding comparison (-> PASS / FAIL).** Build the binding K pool = K-stratum cells
    EXCLUDING converted-O (:func:`~cogworx.eval.youden.is_converted_o`) AND detK (corpus JOIN on
    :attr:`~cogworx.eval.corpus.CorpusItem.detk`). Run
    :func:`~cogworx.eval.youden.nested_bootstrap_delta` for **D>A** and **D>C'** at
    ``quantile=0.025`` (the ÷2 within-run Bonferroni half). PASS iff BOTH binding δ lower bounds
    clear ``gate_threshold`` AND every Tier-0..2 gate passed AND the budget is not exhausted;
    otherwise FAIL.

THE EXCLUSION IS LOAD-BEARING (the converted-O / detK firewall): converted-O is a selection-biased-
EASIER slice of K (CF-4.4c-CONVERTER-SELECTION) and detK is the deterministic collusion probe; both
inflate the binding δ if left in. They are excluded from the binding K pool ONLY — they remain in
the Cell artifact the Tier-1 instruments score (those run their own stratum partition). The detK
exclusion reads the LOCKED corpus (option A), never the pre-label ``PlantedItem``.

``gate_threshold`` defaults to 0.0 — the architect's sign-gate recommendation (``lo > 0``); it is a
PARAM so the final eval-stats value is one-line config. ``j_pos_min`` (0.8) and the control arm
labels are PARAMS for the same reason (the Cell-artifact arm labels for the positive control + the B
strawman are an open 4.4e seam — see the function docstrings).

PURE, offline, deterministic: no model calls, no substrate, no journal (CANON S1/S2). All
Monte-Carlo is the seeded :func:`~cogworx.eval.youden.nested_bootstrap_delta`; no numpy/scipy. The
verdict is a frozen, content-addressable pydantic model.

Contract changelog (CANON §6.1):
  - 2026-06-21 (Pod 4.4e-0): initial — :func:`score_gate` (the tiered GATE verdict assembly) +
    the frozen report types (:class:`GateVerdict`, :class:`DeltaCI`, :class:`InstrumentGateReport`,
    :class:`ControlReport`, :class:`LineageBudgetStatus`, :class:`ReportedOnlyBundle`). New module;
    no existing callers. Reuses the landed lock instruments + the pinned nested bootstrap. PURE.
  - 2026-06-21 (Pod 4.4e-1): added :func:`score_gate_live` — the async budget-enforcement wrapper
    that READS the design-lineage look budget off the durable journal (ONE
    :meth:`~cogworx.substrate.journal.Journal.read_design_lineage_budget` touch, NO append — S6
    exactly-once; ``run_and_stamp`` owns the append) and delegates to the pure :func:`score_gate`.
    :class:`LineageBudgetStatus` widened: ``budget_max`` -> ``look_budget_max`` (the field is new at
    4.4e-0, no external callers) and additive ``path`` (``"ceiling-v1"``) + ``caveat`` (the verbatim
    CF-4.4c-ADAPTIVE-LADDER scope string) carried on EVERY verdict (eval-stats S9-honesty). The
    journal READ is the ONLY I/O on the module; the verdict decision stays in the pure core (S1).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Sequence
from random import Random
from typing import Literal

from pydantic import BaseModel, ConfigDict, model_validator

from cogworx.eval.corpus import CorpusItem
from cogworx.eval.lock import (
    CorpusLockError,
    MeasurementFingerprint,
    ShuffleNullResult,
    assert_arm_a_floor,
    assert_regime_contribution,
    assert_shuffle_null,
    assert_spec_ceiling,
    shuffle_null_centering,
)
from cogworx.eval.youden import Cell, is_converted_o, nested_bootstrap_delta
from cogworx.substrate.journal import Journal

# The deterministic-oracle FLOOR arm: it flags nothing (sens_A≡0 on K, spec_A≈1 on clean — certified
# by assert_arm_a_floor / INV-A0/A1), so J_A ≡ 0 with zero variance. It is the binding-comparison
# baseline for D>A, but it is NOT a valid §5-shuffle cancellation partner (see _SHUFFLE_DELTAS).
_FLOOR_ARM = "A"

# The two binding deltas the Tier-4 PASS gate clears (plan §5/§13.5): D beats the deterministic-
# oracle floor A, and D beats the neutral-second-look C' (labeled "C" in the Cell artifact — the
# diet-stripped C is not a binding arm; the C' full-diet neutral reviewer is the gate's binding
# baseline, lock.py:957). Binding PASS requires BOTH lower bounds clear the threshold.
_BINDING_DELTAS: dict[str, tuple[str, str]] = {"D>A": ("D", _FLOOR_ARM), "D>C'": ("D", "C")}

# The DEDICATED positive-control arm label (plan §13.4 #7). The live 5-arm run MUST emit a known-
# flag-pattern stub (cogworx.eval.runner.scripted_executor) under this label — a non-model arm that
# flags exactly the planted-error item_ids, so its D>A-style J-LCB certifies the instrument can
# detect a KNOWN effect WITHOUT reading the control off arm-under-test D (the circularity Finding 2
# fixes: certifying the instrument on the very arm whose effect the gate measures). A MISSING "PC"
# arm means the pipeline skipped the required control -> INSTRUMENT_INVALID (the missing-arm path).
_POSITIVE_CONTROL_ARM = "PC"

# The §5 membership-shuffle null gates ONLY arm-vs-arm deltas whose ``arm_b`` is NON-degenerate —
# carrying the SAME K/clean confound D does, so the §5.0 cancellation E[J_a]-E[J_b]=0 holds under
# shuffle (architect ruling, Pod 4.4e-0). D>A is EXCLUDED: the degenerate floor (J_A==0) has no
# partner, so its single-arm coverage tail is mis-calibrated and false-refuses any honest D-effect.
# D>A's instrument validity is carried instead by assert_arm_a_floor + the Tier-2 positive control.
# Derived (not hardcoded) by dropping every binding delta whose arm_b is the floor arm, so a future
# corpus that makes A non-degenerate re-admits D>A automatically rather than staying excluded by
# coincidence.
_SHUFFLE_DELTAS: dict[str, tuple[str, str]] = {
    label: arms for label, arms in _BINDING_DELTAS.items() if arms[1] != _FLOOR_ARM
}

# The look-corrected within-run quantile for the binding CI: the ÷2 Bonferroni half of the
# two-sided 95% CI (plan §3.9-B), so each of the two binding deltas spends alpha/2. The shuffle-null
# reuses the nested-bootstrap default; the binding comparison threads this explicitly.
_BINDING_QUANTILE = 0.025

# The K_max design-lineage look ceiling (eval-stats: ratified at 10, FWER~=0.40 — a LOOSE,
# correct-direction bound, NOT a tight adaptive-data-analysis guarantee; the caveat below carries
# the scope). The scorer default ``look_budget_max`` matches this.
_LOOK_BUDGET_MAX = 10

# The verbatim CF-4.4c-ADAPTIVE-LADDER caveat — carried on EVERY LineageBudgetStatus so the bound's
# loose, within-config-only scope travels with the verdict (eval-stats S9-honesty). The across-night
# correction is a Bonferroni approximation, not a true adaptive bound; the budget is monotone only
# WITHIN a planning_variance_config_hash; lineage membership is self-declared.
_ADAPTIVE_LADDER_CAVEAT = (
    "the across-night correction is a Bonferroni approximation; a true adaptive-data-analysis "
    "bound needs measurement-fold rotation (R4-a) or holdout rotation — deferred. The budget is "
    "monotone only WITHIN a planning_variance_config_hash (a §6.0 re-size mints a fresh budget); "
    "lineage membership is self-declared — cross-config monotonicity + mechanical lineage-parent "
    "binding are deferred (CF-4.4c-ADAPTIVE-LADDER)."
)


VerdictStatus = Literal["PASS", "FAIL", "INSTRUMENT_INVALID", "BUDGET_EXHAUSTED"]


class DeltaCI(BaseModel):
    """One binding paired-Youden's-J delta CI (``mean``, ``lo``, ``hi``) from
    :func:`~cogworx.eval.youden.nested_bootstrap_delta`. ``lo`` is the look-corrected lower bound
    the gate compares to ``gate_threshold``. Frozen."""

    model_config = ConfigDict(frozen=True)

    mean: float
    lo: float
    hi: float


class InstrumentGateReport(BaseModel):
    """The pass/fail read-out of ONE Tier-1 instrument-validity gate. ``passed`` is True iff the
    gate did not refuse; ``detail`` carries the :class:`~cogworx.eval.lock.CorpusLockError` message
    on a refusal (empty on pass). The scorer CATCHES the lock error and reports it here — it never
    lets an instrument crash the verdict. Frozen."""

    model_config = ConfigDict(frozen=True)

    name: str
    passed: bool
    detail: str = ""


class ControlReport(BaseModel):
    """The Tier-2 control read-out. ``positive_control`` carries the positive-control arm's δ-CI
    lower bound and whether it cleared ``j_pos_min`` (the instrument can detect a KNOWN effect);
    ``strawman_behaves`` is True iff the B strawman did NOT clear the threshold (a yes-machine that
    "wins" is a mis-wired instrument). ``passed`` is the conjunction. Frozen."""

    model_config = ConfigDict(frozen=True)

    positive_control_lo: float
    positive_control_min: float
    positive_control_cleared: bool
    strawman_lo: float
    strawman_behaves: bool
    passed: bool


class LineageBudgetStatus(BaseModel):
    """The Tier-3 design-lineage look-budget status, carried on EVERY verdict (pass / fail /
    exhausted) so the spent-vs-ceiling distance is ALWAYS observable — "approaching K_max" is a
    reviewer signal, not only the moment it trips (eval-stats' S9-honesty requirement).

    ``look_count`` is the lineage looks already spent (the journal READ of
    :meth:`~cogworx.substrate.journal.Journal.read_design_lineage_budget`, supplied by
    :func:`score_gate_live`); ``look_budget_max`` is the K_max ceiling; ``exhausted`` is
    ``look_count >= look_budget_max``. ``path`` pins the budget instrument version
    (``"ceiling-v1"``). ``caveat`` carries the verbatim CF-4.4c-ADAPTIVE-LADDER scope/limit string
    so the loose, correct-direction nature of the bound travels WITH every verdict. Frozen."""

    model_config = ConfigDict(frozen=True)

    look_count: int
    look_budget_max: int
    exhausted: bool
    path: str = "ceiling-v1"
    caveat: str = _ADAPTIVE_LADDER_CAVEAT


class ReportedOnlyBundle(BaseModel):
    """The non-gating diagnostics carried alongside the verdict for the reviewer (NEVER gate the
    outcome). ``binding_pool_size`` is the count of K-stratum cells surviving the converted-O +
    detK exclusion; ``excluded_converted_o`` / ``excluded_detk`` count the rows removed (so the
    firewall's bite is auditable). ``shuffle_paired_k`` surfaces the §5 report-only paired-coverage
    tail per delta (from :class:`~cogworx.eval.lock.ShuffleNullReport`). Frozen."""

    model_config = ConfigDict(frozen=True)

    binding_pool_size: int
    excluded_converted_o: int
    excluded_detk: int
    shuffle_paired_k: dict[str, int] = {}


class GateVerdict(BaseModel):
    """The assembled Phase-4 GATE verdict (plan §5/§13.5). Frozen + content-addressable.

      - ``passed`` is True ONLY for ``verdict_status == "PASS"`` — an instrument-invalid or
        budget-exhausted run is NOT a pass (and is NOT a clean FAIL either; the distinction is
        load-bearing, plan §13.5).
      - ``verdict_status`` is the tiered outcome: the FIRST failing tier sets it.
      - ``binding_deltas`` carries the two binding δ-CIs (``D>A``, ``D>C'``) when Tier 4 ran (empty
        if a higher tier short-circuited).
      - ``instrument_gates`` / ``controls`` / ``lineage`` are the per-tier reports;
        ``reported_only`` carries the non-gating diagnostics.

    :attr:`digest` is the content-addressed identity over the verdict's load-bearing fields."""

    model_config = ConfigDict(frozen=True)

    passed: bool
    verdict_status: VerdictStatus
    binding_deltas: dict[str, DeltaCI] = {}
    instrument_gates: tuple[InstrumentGateReport, ...] = ()
    controls: ControlReport | None = None
    lineage: LineageBudgetStatus
    reported_only: ReportedOnlyBundle

    @model_validator(mode="after")
    def _passed_couples_to_status(self) -> GateVerdict:
        """``passed`` is True iff ``verdict_status == "PASS"`` — the S9 invariant lives in the TYPE,
        not only in the :func:`score_gate` ``_verdict`` factory. A directly-constructed or
        deserialized ``GateVerdict(passed=True, verdict_status="FAIL")`` (or the inverse) RAISES, so
        no consumer can mint a verdict whose boolean and status disagree (Finding 4)."""
        if self.passed != (self.verdict_status == "PASS"):
            raise ValueError(
                f"passed={self.passed} contradicts verdict_status={self.verdict_status!r}: "
                f"passed must be True iff verdict_status == 'PASS'"
            )
        return self

    @property
    def digest(self) -> str:
        """A content-addressed SHA-256 over the canonical serialization of the verdict — so a
        verdict can be pinned/journaled by identity. Deterministic: identical verdicts -> identical
        digest."""
        payload = self.model_dump(mode="json")
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _binding_k_pool(
    cells: Sequence[Cell], corpus: Sequence[CorpusItem]
) -> tuple[list[Cell], int, int]:
    """The binding K pool: every K-stratum cell EXCEPT converted-O and detK rows, plus the clean
    cells the bootstrap's spec arm needs (untouched). Returns ``(pool, n_excluded_converted_o,
    n_excluded_detk)``.

    The exclusion is load-bearing (the converted-O / detK firewall): converted-O is a
    selection-biased-easier slice of K (CF-4.4c-CONVERTER-SELECTION) and detK is the deterministic
    collusion probe; both inflate the binding δ. converted-O reads the per-cell provenance flag
    (:func:`~cogworx.eval.youden.is_converted_o`); detK JOINs the LOCKED corpus by ``item_id`` on
    :attr:`~cogworx.eval.corpus.CorpusItem.detk` (option A — off the locked truth table, never the
    pre-label ``PlantedItem``). Only K-stratum cells are filtered — clean / O / A-arm cells pass
    through so the instruments and the spec arm see the full artifact."""
    detk_ids = {item.item_id for item in corpus if item.detk}
    pool: list[Cell] = []
    n_conv = 0
    n_detk = 0
    for c in cells:
        if c.stratum == "K":
            if is_converted_o(c):
                n_conv += 1
                continue
            if c.item_id in detk_ids:
                n_detk += 1
                continue
        pool.append(c)
    return pool, n_conv, n_detk


def _global_unpaired(cells: Sequence[Cell]) -> list[int]:
    """Every K + clean ``item_id`` — the FALLBACK global-permutation pool the §5 shuffle null uses
    when no matched-sibling bijection is supplied. Items in neither K nor clean (e.g. O) are NOT
    shuffled into K (§5)."""
    return sorted({c.item_id for c in cells if c.stratum in ("K", "clean")})


class _MissingArm(Exception):
    """Raised when a binding/control read names an arm absent from the Cell artifact. Carries the
    arm label so the scorer can REPORT it (an INSTRUMENT_INVALID verdict naming the missing arm)
    rather than crash — the load-bearing never-crash contract."""

    def __init__(self, arm: str) -> None:
        super().__init__(arm)
        self.arm = arm


def _guarded_delta(
    pool: Sequence[Cell], *, arm_a: str, arm_b: str, n_outer: int, seed: int
) -> tuple[float, float, float]:
    """:func:`~cogworx.eval.youden.nested_bootstrap_delta` for ``arm_a`` vs ``arm_b`` over the
    binding K pool, mapping an ABSENT arm to :class:`_MissingArm` (the scorer's never-crash
    contract).

    A missing arm surfaces as ``StopIteration`` / ``KeyError`` from the bootstrap's per-item arm
    urn (``next(iter(cells_by_item[item_id][arm]))`` over an empty arm dict). The arm is present
    iff at least one pool cell carries it; rather than rely on the bootstrap raising on the FIRST
    draw (a degenerate pool could resample around the gap), we check membership up front and raise
    the typed :class:`_MissingArm` deterministically. ``error_strata`` is fixed to ``("K",)`` and
    the look-corrected ``_BINDING_QUANTILE`` is threaded — every gate read shares the binding
    population and the ÷2 within-run quantile."""
    arms_present = {c.arm for c in pool}
    for arm in (arm_a, arm_b):
        if arm not in arms_present:
            raise _MissingArm(arm)
    try:
        return nested_bootstrap_delta(
            pool,
            arm_a=arm_a,
            arm_b=arm_b,
            error_strata=("K",),
            n_outer=n_outer,
            seed=seed,
            quantile=_BINDING_QUANTILE,
        )
    except (StopIteration, KeyError) as exc:
        missing = arm_a if arm_a not in arms_present else arm_b
        raise _MissingArm(missing) from exc


def score_gate(
    cells: Sequence[Cell],
    corpus: Sequence[CorpusItem],
    fingerprint: MeasurementFingerprint,
    *,
    expected_fingerprint: MeasurementFingerprint | None = None,
    lineage_look_count: int = 0,
    n_clean_planned: int = 80,
    R: int = 7,
    gate_threshold: float = 0.0,
    n_outer: int = 10_000,
    look_budget_max: int = _LOOK_BUDGET_MAX,
    sigma_sq_b_spec_planning: float,
    shuffle_seed: int,
    positive_control_arm: str = _POSITIVE_CONTROL_ARM,
    strawman_arm: str = "B",
    j_pos_min: float = 0.8,
    n_shuffles: int = 200,
    shuffle_paired_ids: Sequence[tuple[int, int]] | None = None,
    shuffle_unpaired_ids: Sequence[int] | None = None,
    shuffle_result: ShuffleNullResult | None = None,
) -> GateVerdict:
    """Assemble the Phase-4 GATE verdict over the frozen Cell artifact + the locked corpus (PURE).

    The tiered gate (module docstring) runs in order and SHORT-CIRCUITS on the first failing tier,
    which sets :attr:`GateVerdict.verdict_status`. The scorer REPORTS instrument refusals (catches
    :class:`~cogworx.eval.lock.CorpusLockError`); it never crashes on a refused gate.

    :param cells: the frozen 4.4d :class:`~cogworx.eval.youden.Cell` artifact.
    :param corpus: the LOCKED :class:`~cogworx.eval.corpus.CorpusItem` truth table — JOINed by
        ``item_id`` for the detK exclusion (option A).
    :param fingerprint: the corpus :class:`~cogworx.eval.lock.MeasurementFingerprint`. Its
        :attr:`~cogworx.eval.lock.MeasurementFingerprint.epsilon_audited` MUST be True (Tier 0).
    :param expected_fingerprint: if given, its ``digest`` must equal ``fingerprint``'s — a mismatch
        is a tampered/stale instrument (Tier 0 refuse).
    :param lineage_look_count: design-lineage looks already spent (passed in; the journal READ is
        4.4e-1's async wrapper). ``>= look_budget_max`` -> BUDGET_EXHAUSTED (Tier 3).
    :param n_clean_planned: the RATIFIED planning clean-item count (the spec-ceiling / arm-A-floor
        n; NEVER realized n — the NEW-MED-2 discipline lives in the instruments).
    :param R: planning trials per ``(item, arm)`` (the ceiling / ``tau_A`` sizing R).
    :param gate_threshold: the binding δ lower-bound the gate clears (default 0.0, the sign gate
        ``lo > 0``; a PARAM so the eval-stats value is one-line config).
    :param n_outer: outer-bootstrap iterations for the binding deltas (small in unit tests).
    :param look_budget_max: the design-lineage look ceiling (Tier 3).
    :param sigma_sq_b_spec_planning: the planning between-item spec variance the spec-ceiling floor
        is sized against (INJECTED — never derived from the artifact under test).
    :param shuffle_seed: the §5 shuffle-null master seed.
    :param positive_control_arm: the DEDICATED positive-control arm label whose δ-CI lower bound
        must clear ``j_pos_min`` (Tier 2). Defaults to ``"PC"`` — the live 5-arm run MUST emit a
        known-flag-pattern stub (:func:`~cogworx.eval.runner.scripted_executor`) under this label.
        Reading the control off the arm-under-test D would be a circularity (Finding 2); the control
        is a distinct arm. A MISSING ``"PC"`` arm -> INSTRUMENT_INVALID (the pipeline didn't run the
        required control). Kept a PARAM so the live run supplies the label.
    :param strawman_arm: the Cell-artifact arm label of the B judge strawman, which must NOT clear
        ``gate_threshold`` on the binding K pool (Tier 2). Defaults to ``"B"``. A MISSING arm ->
        INSTRUMENT_INVALID. Kept a PARAM so the live run supplies the label.
    :param j_pos_min: the positive-control δ-CI lower-bound floor (default 0.8). PARAM pending the
        eval-stats J_pos read.
    :param n_shuffles: §5 shuffle permutations (>=200 in the real gate; the coverage-rate ceiling
        is calibrated at 200 — see :func:`~cogworx.eval.lock.assert_shuffle_null`).
    :param shuffle_paired_ids: the matched ``(K-item, clean-item)`` sibling pairs for the §5
        within-pair sign-flip (the PRIMARY, difficulty-robust centering control,
        :class:`~cogworx.eval.lock.BijectionResult.paired`). A LOCK-TIME input the scorer does NOT
        re-derive. ``None`` -> no sign-flip pairs (the global fallback alone). On a real D-effect
        the global-only null false-refuses a difficulty-confounded corpus (the §5.0 honest caveat /
        Finding-1), so the binding gate SHOULD supply the bijection pairs.
    :param shuffle_unpaired_ids: the unpaired global-permutation pool (orphans + never-paired
        items). ``None`` -> every K + clean ``item_id`` (the full global fallback).
    :param shuffle_result: a pre-computed :class:`~cogworx.eval.lock.ShuffleNullResult` to gate
        instead of running the live §5 null. ``None`` -> compute live via
        :func:`~cogworx.eval.lock.shuffle_null_centering`. The injection seam (mirrors
        ``shuffle_null_centering``'s ``_shuffle`` point) lets a caller/test that already has the
        null skip the bootstrap re-run; the gate logic is identical either way.
    """
    lineage = LineageBudgetStatus(
        look_count=lineage_look_count,
        look_budget_max=look_budget_max,
        exhausted=lineage_look_count >= look_budget_max,
    )

    pool, n_conv, n_detk = _binding_k_pool(cells, corpus)
    pool_size = sum(1 for c in pool if c.stratum == "K")

    def _verdict(
        status: VerdictStatus,
        *,
        gates: tuple[InstrumentGateReport, ...] = (),
        controls: ControlReport | None = None,
        deltas: dict[str, DeltaCI] | None = None,
        shuffle_paired_k: dict[str, int] | None = None,
    ) -> GateVerdict:
        return GateVerdict(
            passed=status == "PASS",
            verdict_status=status,
            binding_deltas=deltas or {},
            instrument_gates=gates,
            controls=controls,
            lineage=lineage,
            reported_only=ReportedOnlyBundle(
                binding_pool_size=pool_size,
                excluded_converted_o=n_conv,
                excluded_detk=n_detk,
                shuffle_paired_k=shuffle_paired_k or {},
            ),
        )

    # ---- Tier 0 — fingerprint integrity ----
    fp_gates: list[InstrumentGateReport] = []
    if not fingerprint.epsilon_audited:
        fp_gates.append(
            InstrumentGateReport(
                name="fingerprint-audited",
                passed=False,
                detail="residual_epsilon is the unaudited sentinel — the corpus has no clean bill",
            )
        )
        return _verdict("INSTRUMENT_INVALID", gates=tuple(fp_gates))
    if expected_fingerprint is not None and fingerprint.digest != expected_fingerprint.digest:
        fp_gates.append(
            InstrumentGateReport(
                name="fingerprint-digest",
                passed=False,
                detail=(
                    f"fingerprint digest {fingerprint.digest} != expected "
                    f"{expected_fingerprint.digest} — tampered or stale instrument"
                ),
            )
        )
        return _verdict("INSTRUMENT_INVALID", gates=tuple(fp_gates))
    fp_gates.append(InstrumentGateReport(name="fingerprint-audited", passed=True))
    if expected_fingerprint is not None:
        fp_gates.append(InstrumentGateReport(name="fingerprint-digest", passed=True))

    # ---- Tier 1 — instrument-validity gates ----
    gates = list(fp_gates)
    shuffle_paired_k: dict[str, int] = {}

    def _run_gate(name: str, fn: Callable[[], object]) -> bool:
        try:
            fn()
        except CorpusLockError as exc:
            gates.append(InstrumentGateReport(name=name, passed=False, detail=str(exc)))
            return False
        gates.append(InstrumentGateReport(name=name, passed=True))
        return True

    ok_a = _run_gate(
        "arm-a-floor",
        lambda: assert_arm_a_floor(cells, n_clean_planned=n_clean_planned, R=R),
    )
    ok_spec = _run_gate(
        "spec-ceiling",
        lambda: assert_spec_ceiling(
            cells,
            n_clean_planned=n_clean_planned,
            R=R,
            sigma_sq_b_spec_planning=sigma_sq_b_spec_planning,
        ),
    )
    ok_regime = _run_gate(
        "regime-contribution",
        lambda: assert_regime_contribution(cells),
    )

    paired_ids = list(shuffle_paired_ids) if shuffle_paired_ids is not None else []
    unpaired_ids = (
        list(shuffle_unpaired_ids)
        if shuffle_unpaired_ids is not None
        else _global_unpaired(cells)
    )
    ok_shuffle = True
    try:
        result = (
            shuffle_result
            if shuffle_result is not None
            else shuffle_null_centering(
                cells,
                paired_ids=paired_ids,
                unpaired_ids=unpaired_ids,
                deltas=_SHUFFLE_DELTAS,
                n_shuffles=n_shuffles,
                seed=shuffle_seed,
                n_outer=n_outer,
            )
        )
        report = assert_shuffle_null(result)
        shuffle_paired_k = dict(report.paired_k)
        gates.append(InstrumentGateReport(name="shuffle-null", passed=True))
    except CorpusLockError as exc:
        gates.append(InstrumentGateReport(name="shuffle-null", passed=False, detail=str(exc)))
        ok_shuffle = False

    if not (ok_a and ok_spec and ok_regime and ok_shuffle):
        return _verdict(
            "INSTRUMENT_INVALID", gates=tuple(gates), shuffle_paired_k=shuffle_paired_k
        )

    def _missing_arm_verdict(arm: str, *, controls: ControlReport | None = None) -> GateVerdict:
        gates.append(
            InstrumentGateReport(
                name="arm-present",
                passed=False,
                detail=(
                    f"arm {arm!r} is ABSENT from the Cell artifact — the pipeline did not run a "
                    f"required arm; the verdict cannot be read"
                ),
            )
        )
        return _verdict(
            "INSTRUMENT_INVALID",
            gates=tuple(gates),
            controls=controls,
            shuffle_paired_k=shuffle_paired_k,
        )

    # ---- Tier 2 — controls ----
    # Both controls read the FIREWALLED binding K pool (the same population the Tier-4 binding
    # deltas score) — so the positive control certifies the instrument on the SAME rows the verdict
    # is read off, not an easier converted-O/detK-inflated slice. The positive control is a
    # DEDICATED arm (default "PC"), never the arm-under-test D (the Finding-2 circularity). A
    # missing control/binding arm -> INSTRUMENT_INVALID (the scorer reports, never crashes).
    try:
        _, pc_lo, _ = _guarded_delta(
            pool,
            arm_a=positive_control_arm,
            arm_b="A",
            n_outer=n_outer,
            seed=shuffle_seed ^ 0x90517,
        )
        _, sm_lo, _ = _guarded_delta(
            pool,
            arm_a=strawman_arm,
            arm_b="A",
            n_outer=n_outer,
            seed=shuffle_seed ^ 0xB57A4,
        )
    except _MissingArm as exc:
        return _missing_arm_verdict(exc.arm)
    pc_cleared = pc_lo > j_pos_min
    strawman_behaves = sm_lo <= gate_threshold
    controls = ControlReport(
        positive_control_lo=pc_lo,
        positive_control_min=j_pos_min,
        positive_control_cleared=pc_cleared,
        strawman_lo=sm_lo,
        strawman_behaves=strawman_behaves,
        passed=pc_cleared and strawman_behaves,
    )
    if not controls.passed:
        return _verdict(
            "INSTRUMENT_INVALID",
            gates=tuple(gates),
            controls=controls,
            shuffle_paired_k=shuffle_paired_k,
        )

    # ---- Tier 3 — lineage budget ----
    if lineage.exhausted:
        return _verdict(
            "BUDGET_EXHAUSTED",
            gates=tuple(gates),
            controls=controls,
            shuffle_paired_k=shuffle_paired_k,
        )

    # ---- Tier 4 — the binding comparison (D>A, D>C') on the firewalled K pool ----
    rng = Random(shuffle_seed)
    deltas: dict[str, DeltaCI] = {}
    all_lo_clear = True
    for label, (arm_a, arm_b) in _BINDING_DELTAS.items():
        try:
            mean, lo, hi = _guarded_delta(
                pool,
                arm_a=arm_a,
                arm_b=arm_b,
                n_outer=n_outer,
                seed=rng.randrange(2**31),
            )
        except _MissingArm as exc:
            return _missing_arm_verdict(exc.arm, controls=controls)
        deltas[label] = DeltaCI(mean=mean, lo=lo, hi=hi)
        if lo <= gate_threshold:
            all_lo_clear = False

    status: VerdictStatus = "PASS" if all_lo_clear else "FAIL"
    return _verdict(
        status,
        gates=tuple(gates),
        controls=controls,
        deltas=deltas,
        shuffle_paired_k=shuffle_paired_k,
    )


async def score_gate_live(
    cells: Sequence[Cell],
    corpus: Sequence[CorpusItem],
    fingerprint: MeasurementFingerprint,
    *,
    journal: Journal,
    planning_variance_config_hash: str,
    design_lineage_chain: Sequence[str],
    expected_fingerprint: MeasurementFingerprint | None = None,
    n_clean_planned: int = 80,
    R: int = 7,
    gate_threshold: float = 0.0,
    n_outer: int = 10_000,
    look_budget_max: int = _LOOK_BUDGET_MAX,
    sigma_sq_b_spec_planning: float,
    shuffle_seed: int,
    positive_control_arm: str = _POSITIVE_CONTROL_ARM,
    strawman_arm: str = "B",
    j_pos_min: float = 0.8,
    n_shuffles: int = 200,
    shuffle_paired_ids: Sequence[tuple[int, int]] | None = None,
    shuffle_unpaired_ids: Sequence[int] | None = None,
    shuffle_result: ShuffleNullResult | None = None,
) -> GateVerdict:
    """The LIVE GATE entry (Pod 4.4e-1): READ the design-lineage look budget off the durable
    journal, then delegate to the PURE :func:`score_gate` with the count threaded in. The ONLY I/O
    seam over the otherwise-pure verdict assembly (CANON S1/S6).

    It does EXACTLY ONE journal touch — a single
    :meth:`~cogworx.substrate.journal.Journal.read_design_lineage_budget` READ — and feeds the
    result to the pure scorer as ``lineage_look_count``. **It does NOT call
    :meth:`~cogworx.substrate.journal.Journal.append_design_look`**: the look was already recorded
    by :func:`~cogworx.eval.runner.run_and_stamp` at MEASUREMENT time (S6 exactly-once); a second
    append here would double-count the look and falsely inflate the spent budget. The journal touch
    is the budget READ only; the verdict DECISION stays entirely in the pure core (S1 —
    model/decision work off the durable write-path; this reads, it does not write).

    The ``(planning_variance_config_hash, design_lineage_chain)`` key MUST be the SAME pair
    ``run_and_stamp`` charged the look against — the budget is the count of DISTINCT measurement
    fingerprints under that key (the §3.9-B garden-of-forking-paths bound). Every pass-through knob
    forwards verbatim to :func:`score_gate`; see that function's docstring for their semantics.

    :param journal: the durable :class:`~cogworx.substrate.journal.Journal` seam (S6); only its
        budget READ is exercised.
    :param planning_variance_config_hash: the §3.9-B ledger key's config component.
    :param design_lineage_chain: the §3.9-B ledger key's lineage component (the append-only list of
        config git-SHAs for this design line).
    :returns: the assembled :class:`GateVerdict` (``BUDGET_EXHAUSTED`` iff the READ count meets the
        ceiling; otherwise the pure tiered verdict).
    """
    look_count = await journal.read_design_lineage_budget(
        planning_variance_config_hash=planning_variance_config_hash,
        design_lineage_chain=design_lineage_chain,
    )
    return score_gate(
        cells,
        corpus,
        fingerprint,
        expected_fingerprint=expected_fingerprint,
        lineage_look_count=look_count,
        n_clean_planned=n_clean_planned,
        R=R,
        gate_threshold=gate_threshold,
        n_outer=n_outer,
        look_budget_max=look_budget_max,
        sigma_sq_b_spec_planning=sigma_sq_b_spec_planning,
        shuffle_seed=shuffle_seed,
        positive_control_arm=positive_control_arm,
        strawman_arm=strawman_arm,
        j_pos_min=j_pos_min,
        n_shuffles=n_shuffles,
        shuffle_paired_ids=shuffle_paired_ids,
        shuffle_unpaired_ids=shuffle_unpaired_ids,
        shuffle_result=shuffle_result,
    )


__all__ = [
    "ControlReport",
    "DeltaCI",
    "GateVerdict",
    "InstrumentGateReport",
    "LineageBudgetStatus",
    "ReportedOnlyBundle",
    "VerdictStatus",
    "score_gate",
    "score_gate_live",
]
