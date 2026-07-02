"""The 5-arm GATE runner skeleton — the deterministic Cell-emitter (Pod 4.4d-0; plan §1).

This module is the offline, pure, deterministic loop that turns a LOCKED
:class:`~cogworx.eval.corpus.CorpusItem` corpus into the flat :class:`~cogworx.eval.youden.Cell`
artifact the Phase-4 GATE statistics consume (``nested_bootstrap_delta`` / ``assert_arm_a_floor`` /
the §5 shuffle null). It emits ``item x arm x trial`` cells; THE MODEL NEVER TOUCHES THIS PATH.

S1 POSTURE (the #1 red-team target — read before touching this module):
  **Eval-time machinery. Produces the frozen Cell artifact from a frozen corpus; not on any live
  write-path. ZERO journal I/O on the emission path, ZERO StageContext, ZERO concrete-provider
  import (S1/S4).** Every model-bearing arm is a DEPENDENCY-INJECTED :class:`ArmExecutor`
  (added at 4.4d-1) — this skeleton (4.4d-0) carries only the two NON-MODEL arms and STUB executors:
    - **arm A** (the deterministic oracle floor) binds
      :func:`cogworx.eval._authoring.run_frozen_check` — the journal-free frozen kernel that
      deliberately bypasses the F7 taint latch (there is no live run to taint). It is NEVER
      ``CodeOracle.evaluate`` (that needs a journal + StageContext — the live verification path). A
      test pins that no ``StageContext`` / ``Journal`` / ``CodeOracle`` symbol is reachable here.
    - the positive-control arm is a scripted, deterministic :class:`ArmExecutor` (no model) that
      proves the runner -> scorer plumbing carries signal (plan §13.4 #7).

GROUND TRUTH IS THE CORPUS, NEVER THE ARM (S9 — the load-bearing emission invariant):
  ``Cell.stratum`` / ``Cell.regime`` / ``Cell.converted_o`` are read off the :class:`CorpusItem`
  (the frozen, second-author-audited ground truth) and welded onto EVERY cell of that item across
  every arm and trial. The :class:`ArmExecutor` returns ONLY :class:`ArmOutcome` (``flagged`` +
  ``route``) — it is STRUCTURALLY unable to set a stratum/regime/converted_o (an arm that "decides"
  its own ground truth is the S9 self-report violation the gate exists to forbid). The per-item
  welding is what the §5 stratum-membership shuffle depends on (it moves the ``stratum`` field while
  the flag vectors stay welded to the item); a runner that read stratum from the arm would corrupt
  that shuffle's marginals.

CRN (common random numbers — the pairing contract ``nested_bootstrap_delta`` cancels luck on):
  ``crn_seed(master_seed, item_id, trial)`` is ARM-INDEPENDENT — every arm replays the SAME
  ``(item, trial)`` luck. The paired delta cancels that shared seed, so a flaky arm earns a wide CI
  rather than a free win. The seed is the CRN pairing unit recorded on each :class:`Cell`.

Pure stdlib + pydantic (+ the injected arm executors); no substrate, no numpy/scipy (CANON S1, S2,
S4). Reuses the landed seams (``Cell`` from :mod:`cogworx.eval.youden`; ``CorpusItem`` /
``ConvertedPlanterStamp`` from :mod:`cogworx.eval.corpus`; ``run_frozen_check`` from
:mod:`cogworx.eval._authoring`; ``MASTER_SEED`` from :mod:`cogworx.eval.lock`) — reused, never
redefined.

Contract changelog (CANON §6.1):
  - 2026-06-21 (Pod 4.4d-0): initial — the 5-arm runner skeleton (ArmExecutor / ArmOutcome /
    ArmInput / run_arms / crn_seed + the two non-model executors: arm_a_executor + scripted). New
    module; no existing callers. Additive new public surface only. The model arms (C / C' / D / D')
    implement :class:`ArmExecutor` at 4.4d-1; this pod builds + tests the seam with stub executors.
  - 2026-06-21 (Pod 4.4d-2): the fingerprint stamp + the single post-run design-lineage journal
    append (``run_and_stamp``) and the scoped CRN ``rho_arm`` re-size diagnostic
    (``crn_resize_diagnostic`` / ``CrnResizeDiagnostic`` / ``ArmPairRho``). Additive new public
    surface only; no existing caller touched. ``run_and_stamp`` is the FIRST async function in this
    module (it awaits the durable journal — S6). Budget ENFORCEMENT stays at 4.4e: ``run_and_stamp``
    RECORDS exactly one design look, it does NOT read or gate on the look count.
  - 2026-06-21 (Pod 4.4d-1): ``ArmInput`` gains ``experiment_design: str = ""`` — the thesis's
    proposed experiment, problem-frame data (NOT ground truth), the C-vs-C'/D info-diet axis the
    model arms condition on. ``_arm_input`` projects it from ``item.thesis.experiment_design``.
    Additive per CANON §6.1: the ``""`` default leaves every pre-existing ``ArmInput`` construction
    (arm A + scripted + their fixtures) byte-identical; arm A ignores the field. The model arms (C /
    C' / D / D' / B) live in :mod:`cogworx.eval.arms` and consume it through
    :func:`cogworx.eval.arms.project_arm_input`.
"""

from __future__ import annotations

import asyncio
import hashlib
from collections.abc import Mapping, Sequence
from typing import Protocol

from pydantic import BaseModel, ConfigDict

from cogworx.eval._authoring import run_frozen_check
from cogworx.eval.corpus import ConvertedPlanterStamp, CorpusItem
from cogworx.eval.lock import (
    MASTER_SEED,
    RESIDUAL_EPSILON_UNAUDITED,
    ExecEnvIdentity,
    MeasurementFingerprint,
    build_fingerprint,
)
from cogworx.eval.youden import Cell, Stratum, realized_variance_diagnostic
from cogworx.substrate.journal import Journal

__all__ = [
    "MASTER_SEED",
    "ArmExecutor",
    "ArmInput",
    "ArmOutcome",
    "ArmPairRho",
    "CrnResizeDiagnostic",
    "arm_a_executor",
    "crn_resize_diagnostic",
    "crn_seed",
    "run_and_stamp",
    "run_arms",
    "scripted_executor",
]


class ArmInput(BaseModel):
    """The per-item projection an :class:`ArmExecutor` receives — the problem the arm reasons over.

    It carries ONLY the item's problem frame, the proposed solution, the proposed experiment
    design, and the author-frozen test — the inputs an arm needs to render a flag. It deliberately
    carries NO ground-truth field (``stratum`` / ``is_error`` / ``error_regime`` /
    ``label_source``): an arm that could read the item's true label would be grading itself (S9).
    The runner reads ground truth off the :class:`~cogworx.eval.corpus.CorpusItem` directly and
    welds it onto the :class:`Cell`; the arm never sees it.

    ``experiment_design`` is the thesis's proposed experiment — problem-frame data, NOT ground truth
    (it is the C-vs-C'/D info-diet axis the model arms condition on; arm A ignores it). It defaults
    to ``""`` so the non-model arms (A / scripted) and their fixtures construct ``ArmInput`` without
    it unchanged; :func:`_arm_input` populates it from ``item.thesis.experiment_design`` for the
    run-driven path, and :func:`cogworx.eval.arms.project_arm_input` strips it to a distinct
    sentinel for the diet-restricted arms (C).
    """

    model_config = ConfigDict(frozen=True)

    item_id: int
    problem_statement: str
    completion_criterion: str
    problem_type: str
    proposed_solution: str
    experiment_design: str = ""
    test_code: str | None


class ArmOutcome(BaseModel):
    """One arm's outcome for one ``(item, trial)`` — the ONLY thing an :class:`ArmExecutor` returns.

    ``flagged`` is the single outcome bit the gate scorer reads (1 = the arm raised a flag on this
    item); ``route`` is audit/diagnostic only. There is DELIBERATELY no ground-truth field here: an
    arm structurally CANNOT set the stratum/regime/converted_o — those come from the corpus item
    (S9). This is the structural wall the runner welds ground truth across.
    """

    model_config = ConfigDict(frozen=True)

    flagged: int
    route: str


class ArmExecutor(Protocol):
    """The injected per-arm seam: ``(arm_input, seed) -> ArmOutcome`` (mirrors
    :class:`cogworx.eval.labeling.OracleProbe`).

    The model arms (C / C' / D / D') implement this at 4.4d-1 behind a provider-agnostic
    :class:`~cogworx.model.base.Model` (S4 — the runner imports no concrete provider). At 4.4d-0 the
    only concrete implementations are the two NON-MODEL arms: :func:`arm_a_executor` (the
    deterministic-oracle floor) and :func:`scripted_executor` (the positive control + test stubs).

    ``seed`` is the CRN pairing unit (see :func:`crn_seed`): every arm receives the SAME seed for a
    given ``(item, trial)``, so a paired delta cancels the shared luck. A deterministic arm (arm A)
    ignores it; a stochastic arm (a model arm at temperature) routes it into its sampling stream
    so a re-run reproduces the artifact byte-for-byte.
    """

    def __call__(self, arm_input: ArmInput, seed: int) -> ArmOutcome: ...


def crn_seed(master_seed: int, item_id: int, trial: int) -> int:
    """The CRN (common-random-numbers) seed for one ``(item, trial)`` — ARM-INDEPENDENT by contract.

    A stable, deterministic hash (blake2b over the ``(master_seed, item_id, trial)`` tuple's
    canonical bytes) -> a non-negative int. It does NOT depend on the arm: every arm replays the
    SAME ``(item, trial)`` luck, which is exactly the pairing
    :func:`~cogworx.eval.youden.nested_bootstrap_delta` cancels on (a flaky arm earns a wide CI, not
    a free win). Reproducible across runs/processes
    (blake2b is stdlib and platform-stable; we never hash the address-randomized ``hash()``).

    The digest is truncated to 8 bytes -> a 64-bit unsigned int, ample entropy for an arm's RNG seed
    while staying a plain ``int`` the :class:`~cogworx.eval.youden.Cell` records.
    """
    payload = f"{master_seed}:{item_id}:{trial}".encode()
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big")


def _arm_input(item: CorpusItem) -> ArmInput:
    """Project a :class:`CorpusItem` to the :class:`ArmInput` an arm reasons over — the problem
    only, NEVER the ground-truth label. The S9 firewall: ``stratum`` / ``is_error`` /
    ``error_regime`` are read by the RUNNER (to weld onto the Cell), never handed to the arm."""
    return ArmInput(
        item_id=item.item_id,
        problem_statement=item.frame.problem_statement,
        completion_criterion=item.frame.completion_criterion,
        problem_type=item.frame.problem_type,
        proposed_solution=item.thesis.proposed_solution,
        experiment_design=item.thesis.experiment_design,
        test_code=item.test_code,
    )


def _is_converted_o(item: CorpusItem) -> bool:
    """Read the converted-O provenance marker off the corpus item: True iff its planter is the
    K->O converter's :class:`~cogworx.eval.corpus.ConvertedPlanterStamp` (Pod 4.4c-3.5). The runner
    welds this onto every cell so :func:`cogworx.eval.youden.is_converted_o` can keep converted-O
    OUT of the binding D>A/D>C' delta (a selection-biased-easier slice of K). It is a
    provenance-only marker — the bootstrap NEVER reads it (it partitions on ``stratum``)."""
    return isinstance(item.planter, ConvertedPlanterStamp)


def run_arms(
    corpus: Sequence[CorpusItem],
    *,
    arm_executors: Mapping[str, ArmExecutor],
    R: int = 7,
    master_seed: int = MASTER_SEED,
) -> list[Cell]:
    """Emit the flat :class:`~cogworx.eval.youden.Cell` artifact for the GATE (plan §1).

    For every ``item x arm x trial in range(R)``:

      1. ``seed = crn_seed(master_seed, item.item_id, trial)`` — ARM-INDEPENDENT, so every arm
         replays the same ``(item, trial)`` luck (the CRN pairing contract).
      2. build the :class:`ArmInput` projection (problem only — never the ground-truth label).
      3. ``outcome = arm_executors[arm](arm_input, seed)`` — the arm returns ONLY ``flagged`` +
         ``route``.
      4. emit ``Cell(...)`` with ``flagged`` / ``route`` from the arm, but
         ``stratum`` / ``regime`` / ``converted_o`` welded FROM THE CORPUS ITEM (S9 — the arm never
         sets ground truth).

    PURE, offline, deterministic: no journal, no StageContext, no substrate, no model on this path
    (the model lives INSIDE a DI'd arm executor, added at 4.4d-1). Re-running with the same corpus +
    executors + ``master_seed`` reproduces a byte-identical artifact (CRN determinism).

    :param corpus: the LOADED, frozen corpus items (one stratum/regime/converted_o ground truth per
        item). The loader/lock guard the corpus upstream; this runner TRUSTS those fields.
    :param arm_executors: the arm label -> :class:`ArmExecutor` map. The 5 GATE arms are
        A / C / C' / D / D'; 4.4d-0 drives it with arm A + a positive control + stubs. The arm LABEL
        becomes ``Cell.arm`` verbatim — the runner does not invent arm names.
    :param R: trials per ``(item, arm)`` (the planning R; default 7). Each trial gets its own
        arm-independent CRN seed.
    :param master_seed: the CRN master seed folded into every ``(item, trial)`` seed (default
        :data:`~cogworx.eval.lock.MASTER_SEED`); a different master seed yields a different output.
    :returns: the flat ``Cell`` list, ordered ``item -> arm -> trial`` (a stable, deterministic
        emission order). The bootstrap re-indexes it, so order is for reproducibility, not scoring.
    """
    cells: list[Cell] = []
    for item in corpus:
        arm_input = _arm_input(item)
        # Ground truth is welded from the CORPUS ITEM, once per item — never from any arm (S9).
        stratum = item.stratum
        regime = item.error_regime
        converted_o = _is_converted_o(item)
        for arm, executor in arm_executors.items():
            for trial in range(R):
                seed = crn_seed(master_seed, item.item_id, trial)
                outcome = executor(arm_input, seed)
                cells.append(
                    Cell(
                        item_id=item.item_id,
                        stratum=stratum,
                        arm=arm,
                        trial=trial,
                        seed=seed,
                        flagged=outcome.flagged,
                        route=outcome.route,
                        regime=regime,
                        converted_o=converted_o,
                    )
                )
    return cells


def arm_a_executor(arm_input: ArmInput, seed: int) -> ArmOutcome:
    """Arm **A** — the deterministic-oracle floor (NON-MODEL, deterministic; plan §5).

    Binds :func:`cogworx.eval._authoring.run_frozen_check` (the journal-free frozen kernel, NEVER
    ``CodeOracle.evaluate``) over ``(proposed_solution, test_code)`` and flags iff the frozen oracle
    raises an **O-catch**: ``is_executable ∧ valid_check ∧ ¬holds`` — the SAME predicate
    :func:`cogworx.eval.conversion._is_o_catch` and :func:`cogworx.eval.labeling.assign_stratum` use
    to call an error O. This is load-bearing for the §5 centering proof:

      - on a **K** (oracle-BLIND) item the frozen test does NOT catch the error (``holds`` is True,
        or the check is non-executable / noise) -> ``flagged = 0`` -> ``sens_A == 0`` on K
        (INV-A0). K is *defined* as the complement of the O-catch, so a correctly-stratified K item
        CANNOT make arm A flag; if it does, the lock's :func:`cogworx.eval.lock.assert_arm_a_floor`
        refuses — that item was mis-stratified (the oracle reached it; it belongs in O).
      - on a **clean** item the frozen test PASSES (``holds`` is True) -> ``flagged = 0`` ->
        ``spec_A ~= 1`` (INV-A1).
      - on an **O** item the oracle CATCHES the error -> ``flagged = 1``.

    A noise verdict (``¬valid_check``, e.g. a timeout / no tests collected) is NOT an O-catch and so
    does NOT flag — mirroring the labeler's polarity exactly (a caught error is
    ``valid_check ∧ ¬holds``; a noise verdict is never a catch). DETERMINISTIC across trials: the
    frozen kernel is a pure function of ``(solution, test)``, so ``seed`` is a no-op (every trial of
    a given item emits the identical outcome — A has no trial variance, by design).

    An item with ``test_code is None`` (a non-code / pure-K item) has nothing for the frozen oracle
    to run -> a non-executable verdict -> not an O-catch -> ``flagged = 0`` (the oracle is blind to
    it, exactly as K demands). ``run_frozen_check`` handles the empty-test case; we pass ``""``.
    """
    _ = seed  # arm A is deterministic — the CRN seed is a no-op (no trial variance, by design).
    verdict = run_frozen_check(arm_input.proposed_solution, arm_input.test_code or "")
    is_o_catch = verdict.is_executable and verdict.valid_check and not verdict.holds
    flagged = 1 if is_o_catch else 0
    return ArmOutcome(flagged=flagged, route="flag" if flagged else "pass")


def scripted_executor(flag_item_ids: frozenset[int]) -> ArmExecutor:
    """Build a NON-MODEL, deterministic :class:`ArmExecutor` with a KNOWN flag pattern (plan §13.4
    #7 positive control).

    The returned executor flags exactly the items whose ``item_id`` is in ``flag_item_ids`` (on
    every trial), passes every other item, and ignores ``seed`` (deterministic). Wiring it as a
    pseudo-arm with ``flag_item_ids`` = the planted-error item_ids proves the runner -> scorer
    plumbing carries SIGNAL end-to-end: a flag-the-errors arm must score a high J, so a green test
    on it rules out a silently-broken emitter (e.g. one that drops ``flagged`` or mis-welds it). It
    also serves as the generic stub for the seam tests (any scripted flag pattern).

    DETERMINISTIC and offline: a pure function of ``(item_id, flag_item_ids)``; no model, no
    substrate, no journal.
    """

    def _executor(arm_input: ArmInput, seed: int) -> ArmOutcome:
        _ = seed  # scripted = deterministic; the CRN seed is a no-op.
        flagged = 1 if arm_input.item_id in flag_item_ids else 0
        return ArmOutcome(flagged=flagged, route="flag" if flagged else "pass")

    return _executor


async def run_and_stamp(
    corpus: Sequence[CorpusItem],
    *,
    arm_executors: Mapping[str, ArmExecutor],
    journal: Journal,
    planning_variance_config_hash: str,
    design_lineage_chain: Sequence[str],
    git_sha: str,
    exec_env: ExecEnvIdentity | None = None,
    master_seed: int = MASTER_SEED,
    residual_epsilon: float = RESIDUAL_EPSILON_UNAUDITED,
    R: int = 7,
) -> tuple[list[Cell], MeasurementFingerprint]:
    """Run the 5-arm GATE, stamp the :class:`MeasurementFingerprint`, and record EXACTLY ONE
    design-lineage "look" on the durable journal (Pod 4.4d-2; plan §3.9-B). S6 exactly-once.

    The post-run spine that ties the frozen Cell artifact to the §3.9-B look-budget ledger:

      1. ``fp = build_fingerprint(corpus, ...)`` over the LOCKED corpus. ``build_fingerprint``
         REFUSES a never-locked item (loud :exc:`ValueError` naming the ``item_id``) — this is the
         lock precondition guard, reused, NOT re-implemented here. It runs FIRST so a never-locked
         corpus is refused before any arm work. ``corpus`` MUST already be the output of the
         :mod:`cogworx.eval.lock` pipeline (``lock_corpus`` is the only writer of ``content_hash``);
         this runner does NOT lock — minting a fingerprint on a corpus that never passed the
         contamination / bijection / floor preconditions would be a stale clean bill (S9/S12).
      2. ``cells = run_arms(corpus, ...)`` — the pure, offline Cell emission (no journal, no model
         on this path; the model lives inside a DI'd :class:`ArmExecutor`).
      3. EXACTLY ONE ``await journal.append_design_look(...)`` recording the
         ``(planning_variance_config_hash, design_lineage_chain)`` -> ``fp.digest`` look. ONE look =
         one ``(config, corpus)`` measurement: the append is idempotent on ``(key, fingerprint)``,
         so re-running the same ``(config, corpus)`` appends the same digest and the distinct-look
         count is unchanged (S6). NEVER per-arm / per-item.

    Budget ENFORCEMENT stays at 4.4e (S12 structural stub): this RECORDS the look — it does NOT
    read :meth:`~cogworx.substrate.journal.Journal.read_design_lineage_budget` and does NOT gate on
    the count. The CRN re-size diagnostic (:func:`crn_resize_diagnostic`) is a SEPARATE reported
    read-out this function does not call.

    :param corpus: the LOCKED, frozen corpus (every item carries a computed ``content_hash``).
    :param arm_executors: the arm label -> :class:`ArmExecutor` map (the 5 GATE arms).
    :param journal: the durable :class:`~cogworx.substrate.journal.Journal` seam (S6).
    :param planning_variance_config_hash: the §3.9-B ledger key's config component (the
        planning-variance config the look is charged against).
    :param design_lineage_chain: the §3.9-B ledger key's lineage component — the append-only list
        of config git-SHAs for this design line. The key is the ``(config_hash, chain)`` tuple, NOT
        the corpus identity, so a corpus re-roll under the same config keeps accumulating looks.
    :param git_sha: the INV-LOCK-2 commit pin folded into the fingerprint (required; not implicit).
    :param exec_env: the execution-environment fence; ``None`` -> ``read_exec_env()`` (live path).
    :param master_seed: the CRN master seed folded into the fingerprint + every ``(item, trial)``.
    :param residual_epsilon: the residual-contamination ε folded into the fingerprint (sentinel by
        default until 4.4c-5b's audit fills it).
    :param R: trials per ``(item, arm)`` (default 7).
    :returns: ``(cells, fingerprint)`` — the flat Cell artifact and the stamped fingerprint.
    """
    fingerprint = build_fingerprint(
        corpus,
        git_sha=git_sha,
        exec_env=exec_env,
        master_seed=master_seed,
        residual_epsilon=residual_epsilon,
    )
    # ``run_arms`` is a SYNCHRONOUS driver, but the model-backed arm executors call
    # ``asyncio.run(...)`` internally (``arms.py`` — the ``Model`` seam is async). Running it
    # directly on this coroutine's event-loop thread would nest ``asyncio.run`` inside a running
    # loop -> ``RuntimeError``. Offload to a worker thread so each executor's ``asyncio.run`` owns a
    # fresh loop. Scripted (non-async) arms are unaffected. (S6 append below is unchanged.)
    cells = await asyncio.to_thread(
        run_arms, corpus, arm_executors=arm_executors, R=R, master_seed=master_seed
    )
    await journal.append_design_look(
        planning_variance_config_hash=planning_variance_config_hash,
        design_lineage_chain=design_lineage_chain,
        fingerprint=fingerprint.digest,
    )
    return cells, fingerprint


class ArmPairRho(BaseModel):
    """One binding arm-pair's CRN pairing-credit read-out (Pod 4.4d-2; eval-stats B2).

    ``rho_arm`` is :attr:`cogworx.eval.youden.VarianceDiagnostic.rho_arm` for the
    ``(arm_a, arm_b)`` pair: the CRN credit ``1 - Var(delta_paired)/(Var(J_a)+Var(J_b))``. Pairing
    bought variance reduction iff ``rho_arm > 0``.

    ``exempt`` is the D-A by-construction exemption: arm A is a DETERMINISTIC constant on K
    (``sens_A == 0`` -> ``J_A == 0`` -> ``Var(J_A) == 0`` by construction, ``youden.py``), so for
    the D-A pair ``rho_arm ~= 0`` is CORRECT, NOT a CRN failure. An exempt pair NEVER triggers the
    re-size (a trigger that fired on D-A would false-alarm every run). This is a STRUCTURAL fact of
    arm A's zero-variance floor, not a tuning knob — do not generalize it to other low-variance
    pairs or the trigger silently disarms for a genuinely under-credited stochastic arm.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    arm_a: str
    arm_b: str
    rho_arm: float
    exempt: bool


class CrnResizeDiagnostic(BaseModel):
    """The scoped CRN ``rho_arm`` re-size diagnostic over the binding arm-pairs (Pod 4.4d-2; plan
    §13.6 POST-RUN-1 re-size contract).

    A REPORTED-ONLY read-out, NEVER a binding gate (S9 — a measurement's own realized variance must
    not auto-gate the measurement; the §6.0(c) re-size is an eval-stats/human decision, held
    provisional). It records no journal I/O.

    ``triggered`` is ``True`` iff ANY NON-exempt (stochastic) pair has ``rho_arm <= 0`` — the
    CRN-failure re-size trigger (the planning ``rho`` assumptions were wrong; n was over-credited).
    The D-A pair is EXEMPT (arm A's zero-variance floor makes ``rho_arm ~= 0`` correct-by-
    construction), so it is reported but never contributes to ``triggered``.
    """

    model_config = ConfigDict(frozen=True)

    per_pair: tuple[ArmPairRho, ...]

    @property
    def triggered(self) -> bool:
        """``True`` iff any NON-exempt pair earned no CRN credit (``rho_arm <= 0``) — the re-size
        trigger. The D-A exemption keeps a by-construction zero from false-alarming every run."""
        return any(p.rho_arm <= 0.0 and not p.exempt for p in self.per_pair)


def crn_resize_diagnostic(
    cells: Sequence[Cell],
    *,
    pairs: Sequence[tuple[str, str, str]],
    error_strata: Sequence[Stratum] = ("K",),
    n_outer: int,
    seed: int,
    deterministic_constant_arm: str = "A",
) -> CrnResizeDiagnostic:
    """Compute the scoped CRN ``rho_arm`` re-size diagnostic per binding arm-pair (Pod 4.4d-2;
    eval-stats B2). REPORTED-ONLY — never a binding gate, never appends to the journal.

    For each ``(label, arm_a, arm_b)`` in ``pairs``, runs
    :func:`cogworx.eval.youden.realized_variance_diagnostic` (the SAME nested-bootstrap draws the
    gate scores) and reads its ``rho_arm`` CRN-credit. A pair earns the re-size trigger iff
    ``rho_arm <= 0`` AND it is not exempt.

    THE D-A EXEMPTION (load-bearing): a pair whose ``arm_a`` OR ``arm_b`` is the
    ``deterministic_constant_arm`` (arm A by default) is EXEMPT. Arm A is a deterministic constant
    on the error stratum (``sens_A == 0`` -> ``J_A == 0`` -> ``Var(J_A) == 0`` by construction), so
    ``rho_arm ~= 0`` for the D-A pair is correct, not a CRN failure; a trigger that fired on D-A
    would false-alarm every run. Exempt pairs are still REPORTED (their ``rho_arm`` is computed and
    carried), they just never contribute to :attr:`CrnResizeDiagnostic.triggered`.

    PURE, offline, deterministic: no model, no substrate, no journal (CANON S1/S2). Decision-
    independent of any verdict — it reads the frozen Cell artifact's realized variance, runs no
    oracle.

    :param cells: the frozen Cell artifact (the runner's output).
    :param pairs: the binding ``(label, arm_a, arm_b)`` pairs to diagnose (e.g.
        ``("D>A", "D", "A")``, ``("D>C'", "D", "C'")``, ``("D>D'", "D", "D'")``). The label is
        opaque and carried verbatim onto :class:`ArmPairRho`.
    :param error_strata: the error population for sens (default ``("K",)``).
    :param n_outer: outer bootstrap resamples (passed through to ``realized_variance_diagnostic``).
    :param seed: the bootstrap seed (passed through; determinism).
    :param deterministic_constant_arm: the arm whose presence in a pair exempts it (default "A").
    :returns: the per-pair :class:`CrnResizeDiagnostic` with its derived ``triggered`` flag.
    """
    per_pair: list[ArmPairRho] = []
    for label, arm_a, arm_b in pairs:
        diag = realized_variance_diagnostic(
            cells,
            arm_a=arm_a,
            arm_b=arm_b,
            error_strata=error_strata,
            n_outer=n_outer,
            seed=seed,
        )
        exempt = deterministic_constant_arm in (arm_a, arm_b)
        per_pair.append(
            ArmPairRho(
                label=label,
                arm_a=arm_a,
                arm_b=arm_b,
                rho_arm=diag.rho_arm,
                exempt=exempt,
            )
        )
    return CrnResizeDiagnostic(per_pair=tuple(per_pair))
