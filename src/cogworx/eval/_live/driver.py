"""The 1-key GATE bring-up driver — the L6 capstone composition root (Pod 4.4-live L6).

:func:`run_bring_up` is a LINEAR, staged async pipeline that wires L0-L5 (settings / roster /
cache / budget / corpus) into an end-to-end run on a single model family: preflight the roster ->
build the deterministic bring-up corpus -> wire a budgeted, cached model behind every model arm ->
run the 5-ish arms and stamp the design-lineage look (S6) -> report the HOLD stages this pod
defers -> run the CRN re-size diagnostic over whichever binding pairs are actually present ->
score the GATE -> write the run's honest manifest. Nothing here is re-derived: every stage calls a
landed L0-L5 / ``runner`` / ``arms`` / ``scorer`` seam. This module composes; it does not decide
statistics.

WHY A 1-KEY RUN CAN NEVER PASS (the point of this pod)
--------------------------------------------------------
:func:`~cogworx.eval._live.corpus.build_bring_up_corpus` plants NO cross-family LLM-K pool (there
is only one model family under contract), so the binding K population (K minus detK minus
converted-O, :mod:`cogworx.eval.scorer`'s Tier-4 firewall) is provably empty, and the corpus
fingerprint's ``residual_epsilon`` is the :data:`~cogworx.eval.lock.RESIDUAL_EPSILON_UNAUDITED`
sentinel (no contamination audit has run over it). :func:`~cogworx.eval.scorer.score_gate`'s
Tier 0 (fingerprint integrity) refuses on the unaudited sentinel before any statistic runs; even
if it did not, Tier 2's positive-control read against the empty binding-K pool would refuse it at
Tier 2. Both are structural, not incidental — a 1-key bring-up is STRUCTURALLY unable to emit
PASS. This driver does not special-case that outcome: it calls the real, unmodified
:func:`~cogworx.eval.scorer.score_gate_live` and lets the tiered gate speak for itself.

THE CELL-ARTIFACT ARM-LABEL CROSSOVER (read before touching the arm map)
--------------------------------------------------------------------------
:mod:`cogworx.eval.arms` names its two neutral-framing factories by DIET:
:func:`~cogworx.eval.arms.make_c_prime_executor` (full diet) and
:func:`~cogworx.eval.arms.make_c_executor` (diet-STRIPPED; shares the SAME instruction constant and
``_run_dialectic_arm`` code path, differing only in the diet it self-applies).
:mod:`cogworx.eval.scorer` instead names its arms by CELL-ARTIFACT ROLE: its ``D>C'`` binding delta
reads ``Cell.arm == cogworx.eval.scorer.BINDING_BASELINE_ARM`` (``"C"``) for "the neutral-second-
look reviewer that is the gate's binding baseline". This driver stamps
``Cell.arm == BINDING_BASELINE_ARM`` (``"C"``) from :func:`~cogworx.eval.arms.make_c_prime_executor`
(full diet -- the scorer's binding baseline) and ``Cell.arm == "C_stripped"`` from
:func:`~cogworx.eval.arms.make_c_executor` (diet-stripped -- reported-only, never a binding arm; a
NON-primed label, deliberately distinct from the scorer's ``"D>C'"`` delta-label string so the two
can never be confused with each other or silently swapped). :data:`_MODEL_ARM_FACTORIES` keys off
the imported :data:`~cogworx.eval.scorer.BINDING_BASELINE_ARM` constant rather than a bare re-typed
``"C"`` literal, so the crossover cannot silently re-swap (CANON §6.1 2026-07-02).

THE DIET WELD (CANON §6.1 2026-07-02 -- superseded ``_diet_wrap``)
-----------------------------------------------------------------------
:func:`~cogworx.eval.runner.run_arms` builds ONE generic :class:`~cogworx.eval.runner.ArmInput` per
item (``runner._arm_input``, carrying the real ``test_code`` and the un-stripped
``experiment_design``) and hands the SAME object to every arm executor -- ``ArmExecutor`` is
arm-agnostic by contract (:mod:`cogworx.eval.runner` docstring), so ``run_arms`` has no per-arm
projection hook. This driver previously closed that gap with a local ``_diet_wrap`` adapter (a
composition-root-only "glue" seam). The diet transform now lives INSIDE every
``make_*_executor`` factory itself (each self-applies the shared, idempotent
``arms._project_diet`` to whatever ``ArmInput`` it receives -- see ``arms.py``'s module docstring,
"THE DIET WELD"), so ``_diet_wrap`` is no longer needed here: :func:`_build_arm_executors` wires the
raw factories directly over the budgeted model, and the diet firewall (the C-vs-C' strip, the
``test_code`` drop for every model arm) is enforced structurally by ``arms.py``, not by this
composition root.

S1/S2/S4 POSTURE: the only I/O this module performs is the L3 JSONL idempotency cache, the S6
journal touches inside ``run_and_stamp``/``score_gate_live`` (both already-landed seams), and the
manifest/Cell-artifact file writes at the end. No model call happens outside a DI'd
:class:`~cogworx.model.base.Model` (S4); the real DeepSeek adapter and the real Timescale journal
are constructed ONLY behind ``__main__`` (L7, Jim-gated) -- this module stays importable and fully
unit-testable with a stub model + :class:`~cogworx.testing.doubles.InMemoryJournal` at import time.

Contract changelog (CANON §6.1):
  - 2026-07-02 (Pod 4.4-live L6): initial -- ``run_bring_up`` / ``BringUpManifest`` / the CLI
    ``main``. New module; no existing callers. Additive new public surface only.
  - 2026-07-02 (wiring fix): DELETED ``_diet_wrap`` -- the ``arms.py`` factories now self-apply the
    diet weld (see ``arms.py``'s "THE DIET WELD"), so this composition root no longer needs its own
    adapter. ``_MODEL_ARM_FACTORIES`` narrowed from ``Mapping[str, tuple[factory, strip: bool]]``
    to ``Mapping[str, factory]`` (the strip flag moved into each factory). The reported-only
    stripped-C Cell arm is RENAMED ``"C'"`` -> ``"C_stripped"`` (a non-primed label, never read by
    the scorer -- verified by grep -- so this only touches this driver's arm map, its L3 cache
    filename, and its manifest telemetry); the binding-baseline Cell arm ``"C"`` is now keyed off
    the imported
    :data:`~cogworx.eval.scorer.BINDING_BASELINE_ARM` constant rather than a bare literal. Breaking
    for any out-of-tree reader of ``manifest.json``'s ``arm_telemetry`` that matched on the literal
    ``"C'"`` label; no other caller of this module exists yet (L7 is Jim-gated, unbuilt).
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sys
from collections.abc import Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from cogworx.cost.budget import BudgetGuard
from cogworx.eval._live.budget import build_budgeted_model
from cogworx.eval._live.cache import cached_executor
from cogworx.eval._live.corpus import build_bring_up_corpus
from cogworx.eval._live.roster import RosterReport, RosterUnsound, preflight_roster
from cogworx.eval._live.settings import GateRunMode, GateRunSettings, load_gate_run_settings
from cogworx.eval.arms import (
    make_b_executor,
    make_c_executor,
    make_c_prime_executor,
    make_d_executor,
)
from cogworx.eval.corpus import CorpusItem
from cogworx.eval.lock import (
    RESIDUAL_EPSILON_UNAUDITED,
    MeasurementFingerprint,
    build_fingerprint,
    read_git_sha,
)
from cogworx.eval.runner import (
    MASTER_SEED,
    ArmExecutor,
    CrnResizeDiagnostic,
    arm_a_executor,
    crn_resize_diagnostic,
    run_and_stamp,
    scripted_executor,
)
from cogworx.eval.scorer import BINDING_BASELINE_ARM, GateVerdict, score_gate_live
from cogworx.eval.youden import Cell
from cogworx.model.base import Model
from cogworx.substrate.journal import Journal
from cogworx.testing.doubles import InMemoryJournal

__all__ = ["ArmTelemetry", "BringUpManifest", "HeldNote", "main", "run_bring_up"]

logger = logging.getLogger(__name__)

# The eval-stats sizing-sim planning between-item spec variance (``tests/eval/test_lock.py``'s
# ``_SIGMA_SQ_B_SPEC``) -- irrelevant to THIS run's outcome (Tier 0 refuses before Tier 1 ever
# reads it) but a required, honestly-sourced ``score_gate`` param, not an arbitrary placeholder.
_SIGMA_SQ_B_SPEC_PLANNING = 0.028

_ModelArmFactory = Callable[[Model], ArmExecutor]

#: Cell-artifact arm label -> the ``arms.py`` factory to drive it. Each factory now self-applies its
#: own diet weld (CANON §6.1 2026-07-02 -- see the module docstring's "THE DIET WELD"), so this map
#: carries no separate strip flag. See "CELL-ARTIFACT ARM-LABEL CROSSOVER" for why
#: :data:`~cogworx.eval.scorer.BINDING_BASELINE_ARM` (``"C"``) maps to ``make_c_prime_executor``
#: (full diet -- the scorer's binding baseline) and ``"C_stripped"`` maps to ``make_c_executor``
#: (diet-stripped -- reported-only, never a binding arm).
_MODEL_ARM_FACTORIES: Mapping[str, _ModelArmFactory] = {
    "D": make_d_executor,
    BINDING_BASELINE_ARM: make_c_prime_executor,
    "C_stripped": make_c_executor,
    "B": make_b_executor,
}

#: The candidate CRN re-size pairs (label, arm_a, arm_b) -- mirrors the scorer's binding deltas plus
#: the deferred D-D' cross-family pair. D' is never wired in a 1-key bring-up (no second family), so
#: this pair is ALWAYS reported "not measured", never crashed on. The label string ``"D>C'"``
#: mirrors :mod:`cogworx.eval.scorer`'s own ``_BINDING_DELTAS`` key verbatim (an opaque diagnostic
#: name); its ``arm_b`` is :data:`~cogworx.eval.scorer.BINDING_BASELINE_ARM`, the SAME constant the
#: scorer binds against -- never the reported-only ``"C_stripped"`` arm.
_CRN_CANDIDATE_PAIRS: tuple[tuple[str, str, str], ...] = (
    ("D>A", "D", "A"),
    ("D>C'", "D", BINDING_BASELINE_ARM),
    ("D>D'", "D", "D'"),
)


class HeldNote(BaseModel):
    """One explicitly-deferred/held item this bring-up run does NOT resolve -- emitted so a held
    item is a manifest-visible record, never a silent skip."""

    model_config = ConfigDict(frozen=True)

    name: str
    status: Literal["HELD", "DEFERRED"]
    detail: str


class ArmTelemetry(BaseModel):
    """Per-arm realized telemetry over the emitted Cell artifact: how many cells an arm produced and
    what fraction of them were flagged."""

    model_config = ConfigDict(frozen=True)

    arm: str
    cell_count: int
    flag_rate: float


class BringUpManifest(BaseModel):
    """The bring-up run's honest record (Pod 4.4-live L6). Written to ``out_dir/manifest.json``.

    Carries the roster banner (:attr:`roster`, echoed from the run's start-of-log preflight), the
    assembled :class:`~cogworx.eval.scorer.GateVerdict`, the corpus fingerprint digest, realized
    per-arm telemetry, budget spend, the CRN re-size diagnostic (plus which pairs went unmeasured),
    and the explicit HELD/DEFERRED notes (:class:`HeldNote`) -- never a silent skip.
    """

    model_config = ConfigDict(frozen=True)

    mode: GateRunMode
    roster: RosterReport
    verdict: GateVerdict
    fingerprint_digest: str
    cell_count: int
    arm_telemetry: tuple[ArmTelemetry, ...]
    budget_spent_usd: float
    budget_calls: int
    crn_diagnostic: CrnResizeDiagnostic
    crn_not_measured: tuple[str, ...]
    held: tuple[HeldNote, ...]
    generated_at: datetime


def _safe_filename(label: str) -> str:
    return label.replace("'", "prime")


def _build_arm_executors(
    corpus: Sequence[CorpusItem],
    *,
    model: Model,
    settings: GateRunSettings,
    guard: BudgetGuard,
    fingerprint_digest: str,
    out_dir: Path,
) -> dict[str, ArmExecutor]:
    """Wire the bring-up arm map: A (deterministic oracle) + PC (scripted positive control) are
    non-model and uncached; B/C/C_stripped/D are the ``arms.py`` model factories over a budgeted
    model, each cache-wrapped (L3) under its own Cell-artifact arm label. Each factory self-applies
    its own diet weld (CANON §6.1 2026-07-02), so no separate diet-projection wrap is needed here.
    NO D' (deferred -- needs a second model family)."""
    budgeted = build_budgeted_model(
        model,
        guard=guard,
        price_table=settings.arm_family.price_per_mtok,
        max_output_tokens=settings.arm_family.max_output_tokens,
    )
    positive_control_ids = frozenset(item.item_id for item in corpus if item.is_error == 1)

    executors: dict[str, ArmExecutor] = {
        "A": arm_a_executor,
        "PC": scripted_executor(positive_control_ids),
    }
    for label, factory in _MODEL_ARM_FACTORIES.items():
        executors[label] = cached_executor(
            factory(budgeted),
            arm=label,
            fingerprint_digest=fingerprint_digest,
            cache_path=out_dir / f"cache_{_safe_filename(label)}.jsonl",
        )
    return executors


def _planning_variance_config_hash(settings: GateRunSettings) -> str:
    """The §3.9-B ledger key's config component -- a stable hash over the arm-family model identity
    + price basis (the knobs that define "this planning-variance configuration" for a bring-up
    run)."""
    payload = json.dumps(
        {
            "mode": settings.mode,
            "arm_model_pro": settings.arm_family.model_pro,
            "arm_model_flash": settings.arm_family.model_flash,
            "price_basis": settings.price_basis,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _crn_diagnostic(
    cells: Sequence[Cell], *, n_outer: int, seed: int
) -> tuple[CrnResizeDiagnostic, tuple[str, ...]]:
    """Run the CRN re-size diagnostic over whichever of :data:`_CRN_CANDIDATE_PAIRS` have BOTH arms
    present in the Cell artifact; every absent-arm pair (D-D' in a 1-key bring-up) is reported
    "not measured" rather than handed to :func:`~cogworx.eval.runner.crn_resize_diagnostic`, which
    has no missing-arm guard and would raise (``StopIteration``) on one."""
    present_arms = {c.arm for c in cells}
    measured_pairs = [
        pair for pair in _CRN_CANDIDATE_PAIRS if pair[1] in present_arms and pair[2] in present_arms
    ]
    measured_labels = {pair[0] for pair in measured_pairs}
    not_measured = tuple(
        f"{label}: arm {arm_b!r} absent from the Cell artifact (a bring-up run wires only one "
        "model family -- the cross-family adversary D' is deferred)"
        for label, _arm_a, arm_b in _CRN_CANDIDATE_PAIRS
        if label not in measured_labels
    )
    diagnostic = crn_resize_diagnostic(cells, pairs=measured_pairs, n_outer=n_outer, seed=seed)
    return diagnostic, not_measured


def _held_notes() -> tuple[HeldNote, ...]:
    """The two HOLD stages this pod defers -- emitted explicitly, never silently skipped."""
    return (
        HeldNote(
            name="plant-size-vs-realized-conv-rate",
            status="DEFERRED",
            detail=(
                "the 4.4c-5 lock decision to re-size the cross-family LLM-K plant "
                "(P >= 56/(1 - conv_rate), re-ratified once a live converter run yields a "
                "realized K->O conversion rate) does not apply to this single-family bring-up "
                "corpus -- no cross-family LLM-K pool is planted here (L5). Carried forward to "
                "the first binding multi-family run."
            ),
        ),
        HeldNote(
            name="before-lock-provisionals",
            status="HELD",
            detail=(
                "the 4.4c-6a provisional constants (the 0.08 global shuffle-null coverage "
                "ceiling pending eval-stats Monte-Carlo sizing; the 0.30/0.15/K_max "
                "regime-contribution and look-budget bounds pending realized sens_D/abstention "
                "from a binding run) are neither confirmed nor exercised by this bring-up run -- "
                "Tier-0's unaudited-epsilon refusal short-circuits before any Tier-1 instrument "
                "that would read them."
            ),
        ),
    )


def _build_manifest(
    *,
    settings: GateRunSettings,
    report: RosterReport,
    verdict: GateVerdict,
    fingerprint: MeasurementFingerprint,
    cells: Sequence[Cell],
    guard: BudgetGuard,
    crn_diag: CrnResizeDiagnostic,
    crn_not_measured: tuple[str, ...],
    held: tuple[HeldNote, ...],
) -> BringUpManifest:
    by_arm: dict[str, list[Cell]] = {}
    for c in cells:
        by_arm.setdefault(c.arm, []).append(c)
    arm_telemetry = tuple(
        ArmTelemetry(
            arm=arm,
            cell_count=len(arm_cells),
            flag_rate=(sum(c.flagged for c in arm_cells) / len(arm_cells)) if arm_cells else 0.0,
        )
        for arm, arm_cells in sorted(by_arm.items())
    )
    return BringUpManifest(
        mode=settings.mode,
        roster=report,
        verdict=verdict,
        fingerprint_digest=fingerprint.digest,
        cell_count=len(cells),
        arm_telemetry=arm_telemetry,
        budget_spent_usd=guard.spent_usd,
        budget_calls=guard.calls,
        crn_diagnostic=crn_diag,
        crn_not_measured=crn_not_measured,
        held=held,
        generated_at=datetime.now(UTC),
    )


async def run_bring_up(
    settings: GateRunSettings,
    *,
    model: Model,
    journal: Journal,
    out_dir: Path,
    seed: int = MASTER_SEED,
    git_sha: str | None = None,
    R: int = 7,
    n_o_pairs: int = 8,
    n_detk_pairs: int = 8,
    gate_threshold: float = 0.0,
    gate_n_outer: int = 2_000,
    gate_n_shuffles: int = 200,
    crn_n_outer: int = 200,
) -> BringUpManifest:
    """Run the 1-key GATE bring-up end to end and return its honest :class:`BringUpManifest`.

    A LINEAR staged pipeline (each stage logged by name): preflight the roster (raises
    :class:`~cogworx.eval._live.roster.RosterUnsound` uncaught for a ``mode="binding"`` roster that
    is blocked, or for an inconsistent ``mode="bring-up"`` roster reporting itself binding-eligible)
    -> build + lock + fingerprint the deterministic bring-up corpus -> wire a budgeted, L3-cached
    model behind every model arm -> run the arms and stamp the design-lineage look (S6, exactly one
    ``append_design_look``) -> emit the HELD/DEFERRED notes -> the CRN re-size diagnostic over
    whichever binding pairs are present -> score the GATE (expected ``INSTRUMENT_INVALID`` -- see
    the module docstring) -> write ``out_dir/cells.json`` + ``out_dir/manifest.json``.

    :param model: the DI'd :class:`~cogworx.model.base.Model` driving every model arm (S4) -- a stub
        in tests, the real DeepSeek adapter only behind ``__main__``.
    :param journal: the DI'd :class:`~cogworx.substrate.journal.Journal` (S6) -- an in-memory
        double in tests; the real Timescale journal is L7 (Jim-gated), never connected here.
    :param out_dir: where the L3 idempotency caches, the Cell artifact, and the manifest land.
        Created (with parents) if it does not already exist.
    :param seed: the corpus-content + CRN master seed (default
        :data:`~cogworx.eval.runner.MASTER_SEED`).
    :param git_sha: the INV-LOCK-2 commit pin folded into the fingerprint; ``None`` ->
        :func:`~cogworx.eval.lock.read_git_sha` (the live reading).
    :param R: trials per ``(item, arm)`` for the 5-arm run (default 7, matching
        :func:`~cogworx.eval.runner.run_and_stamp`'s own default).
    :param n_o_pairs: deterministic-O pairs to plant (see
        :func:`~cogworx.eval._live.corpus.build_bring_up_corpus`).
    :param n_detk_pairs: deterministic detK pairs to plant (ditto).
    :param gate_threshold: the scorer's binding δ lower-bound threshold (passed through).
    :param gate_n_outer: outer-bootstrap iterations for the scorer's binding deltas.
    :param gate_n_shuffles: §5 shuffle-null permutations for the scorer.
    :param crn_n_outer: outer-bootstrap iterations for the CRN re-size diagnostic.
    :returns: the assembled, written :class:`BringUpManifest`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("stage 1/8 preflight: mode=%s", settings.mode)
    report = preflight_roster(settings)
    logger.info(
        "roster banner: families_present=%s binding_blocked_reasons=%s",
        report.families_present,
        report.binding_blocked_reasons,
    )

    logger.info("stage 2/8 corpus")
    resolved_git_sha = git_sha if git_sha is not None else read_git_sha()
    corpus, _ = build_bring_up_corpus(
        seed=seed, n_o_pairs=n_o_pairs, n_detk_pairs=n_detk_pairs, git_sha=resolved_git_sha
    )
    fingerprint = build_fingerprint(
        corpus,
        git_sha=resolved_git_sha,
        master_seed=seed,
        residual_epsilon=RESIDUAL_EPSILON_UNAUDITED,
    )

    logger.info("stage 3/8 model wiring: fingerprint=%s", fingerprint.digest)
    guard = BudgetGuard(max_usd=settings.phase_b_max_usd)
    arm_executors = _build_arm_executors(
        corpus,
        model=model,
        settings=settings,
        guard=guard,
        fingerprint_digest=fingerprint.digest,
        out_dir=out_dir,
    )

    logger.info("stage 4/8 arms run")
    config_hash = _planning_variance_config_hash(settings)
    lineage_chain = (resolved_git_sha,)
    cells, run_fingerprint = await run_and_stamp(
        corpus,
        arm_executors=arm_executors,
        journal=journal,
        planning_variance_config_hash=config_hash,
        design_lineage_chain=lineage_chain,
        git_sha=resolved_git_sha,
        master_seed=seed,
        residual_epsilon=RESIDUAL_EPSILON_UNAUDITED,
        R=R,
    )

    logger.info("stage 5/8 HOLD stages")
    held = _held_notes()
    for note in held:
        logger.info("held: %s [%s] %s", note.name, note.status, note.detail)

    logger.info("stage 6/8 CRN diagnostic")
    crn_diag, crn_not_measured = _crn_diagnostic(cells, n_outer=crn_n_outer, seed=seed)
    for reason in crn_not_measured:
        logger.info("crn pair not measured: %s", reason)

    logger.info("stage 7/8 score")
    n_clean_planned = sum(1 for item in corpus if item.stratum == "clean")
    verdict = await score_gate_live(
        cells,
        corpus,
        run_fingerprint,
        journal=journal,
        planning_variance_config_hash=config_hash,
        design_lineage_chain=lineage_chain,
        n_clean_planned=n_clean_planned,
        R=R,
        gate_threshold=gate_threshold,
        n_outer=gate_n_outer,
        sigma_sq_b_spec_planning=_SIGMA_SQ_B_SPEC_PLANNING,
        shuffle_seed=seed,
        n_shuffles=gate_n_shuffles,
    )
    logger.info("verdict: %s", verdict.verdict_status)

    logger.info("stage 8/8 manifest")
    manifest = _build_manifest(
        settings=settings,
        report=report,
        verdict=verdict,
        fingerprint=run_fingerprint,
        cells=cells,
        guard=guard,
        crn_diag=crn_diag,
        crn_not_measured=crn_not_measured,
        held=held,
    )
    (out_dir / "cells.json").write_text(
        json.dumps([c.model_dump(mode="json") for c in cells], indent=2),
        encoding="utf-8",
    )
    (out_dir / "manifest.json").write_text(
        manifest.model_dump_json(indent=2),
        encoding="utf-8",
    )
    return manifest


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="cog-worx Phase-4 GATE bring-up driver")
    parser.add_argument("--config", required=True, type=Path, help="path to the TOML run-config")
    parser.add_argument("--mode", required=True, choices=["bring-up", "binding"])
    parser.add_argument("--out-dir", required=True, type=Path, dest="out_dir")
    parser.add_argument("--seed", type=int, default=MASTER_SEED)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Builds the REAL DeepSeek model + an in-memory journal (real Timescale is L7,
    Jim-gated -- never connected here) ONLY inside this function, so importing this module never
    requires provider credentials or a running substrate."""
    args = _parse_args(argv)
    settings = load_gate_run_settings(args.config)
    if settings.mode != args.mode:
        print(
            f"--mode {args.mode!r} does not match the run-config's mode {settings.mode!r} -- "
            "refusing rather than silently running the config's mode",
            file=sys.stderr,
        )
        return 1

    from cogworx.model.providers.openai_compat import DEEPSEEK_CAPABILITIES, OpenAICompatModel

    model = OpenAICompatModel(settings.arm_family, capabilities=DEEPSEEK_CAPABILITIES)
    journal = InMemoryJournal()

    try:
        asyncio.run(
            run_bring_up(
                settings,
                model=model,
                journal=journal,
                out_dir=args.out_dir,
                seed=args.seed,
            )
        )
    except RosterUnsound as exc:
        print(f"roster refused: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
