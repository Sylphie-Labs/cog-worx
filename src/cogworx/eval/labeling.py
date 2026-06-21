"""Offline corpus-labeling pipeline for the Phase-4 GATE corpus (Pod 4.4c-4; plan §3.4/§4/§5).

This module promotes pre-label :class:`~cogworx.eval.planting.PlantedItem`s into labeled
:class:`~cogworx.eval.corpus.CorpusItem`s: it assigns the BINDING ``stratum`` mechanically from a
frozen-oracle probe, re-verifies clean labels (C1 oracle-reachable / C2 human), routes oracle-blind
labels and K-regime audits to an injected human-adjudication harness, and attaches the resulting
label + provenance. The planter (4.4c-3) owns pre-label truth; THIS pod owns the label.

S1 POSTURE (the #1 red-team target — read before touching this module):
  **Authoring-time tooling. Runs offline to produce the frozen corpus; not on any live write-path.
  ZERO journal I/O, ZERO live-model drive, ZERO StageContext — same S1-clean posture as
  ``planting.py``.** Every external seam is DEPENDENCY-INJECTED:
    - the oracle probe is an :class:`OracleProbe` ``(solution_code, test_code) -> Verdict`` whose
      default binds :func:`cogworx.eval._authoring.run_frozen_check` — the journal-free frozen
      kernel that deliberately bypasses the F7 taint latch (there is no run/drive to taint). It is
      NEVER ``CodeOracle.evaluate`` (that needs a journal + StageContext — the live path). A test
      pins that no ``StageContext`` / ``Journal`` symbol is reachable from this module.
    - the human adjudication is an injected ``adjudicate`` / ``tie_break`` callback pair (stubs in
      tests) — never a live model. ``labeling.py`` constructs no provenance from a model/judge
      response and imports no judge / antithesis symbol (S9: never trust a model's self-report).

THE HARD WALL (enforced structurally):
  - :func:`assign_stratum` has NO access to ``adjudicate`` — stratum is mechanical, never
    human-judged.
  - :func:`adjudicate_item` has NO access to ``probe`` — an oracle-blind label is the human's job.
  They meet ONLY in :func:`promote_corpus`, which routes a clean item to EXACTLY ONE of {C1, C2} by
  ``oracle_reachable``.

DEFERRED to 4.4c-5 (do NOT build here — this pod only produces the inputs):
  matched-sibling bijection re-validation after abstention drops (INV-7 — this pod only emits
  ``dropped_ids``), ``content_hash`` / git-SHA pin (carried ``""``), the noise-audit LLM-flag→ε
  pass, INV-A0/INV-A1 (need :class:`~cogworx.eval.youden.Cell`s from 4.4d), and the lock asserts.

Pure stdlib + pydantic (+ the injected probe/adjudication seams); no substrate (CANON S1, S2, S4).
Reuses the landed seams (``CorpusItem`` / label provenance / ``RegimeAdjudication`` from
:mod:`cogworx.eval.corpus`; ``PlantedItem`` / ``_derive_o_regime`` from
:mod:`cogworx.eval.planting`; ``Verdict`` from :mod:`cogworx.verification.contracts`) — reused.

Contract changelog (CANON §6.1):
  - 2026-06-20 (Pod 4.4c-4): initial — the corpus-labeling pipeline (assign_stratum / reverify_clean
    / adjudicate_item / promote_corpus + the frozen result/request types). New module; no existing
    callers. Additive new public surface only.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict

from cogworx.eval._authoring import run_frozen_check
from cogworx.eval.corpus import (
    Adjudication,
    CorpusItem,
    DeterministicPlanterStamp,
    HumanLabelProvenance,
    OracleLabelProvenance,
    RegimeAdjudication,
)
from cogworx.eval.planting import (
    DETK_PROBE_OPERATOR,
    PlantedItem,
    PlantedPair,
    _derive_o_regime,
)
from cogworx.verification.contracts import Verdict

log = logging.getLogger(__name__)


# ===========================================================================
# Injected seams (the hard-wall) — the oracle probe + the human adjudication callbacks
# ===========================================================================


class OracleProbe(Protocol):
    """The injected frozen-oracle seam: ``(solution_code, test_code) -> Verdict``.

    The default binding is :func:`cogworx.eval._authoring.run_frozen_check` (the journal-free frozen
    kernel). It is DELIBERATELY NOT ``CodeOracle.evaluate`` — that needs a journal + a stage
    context (the live verification path); this is the offline kernel, the S1 boundary. The probe
    MUST return a frozen-sourced verdict (``test_provenance == "frozen"``): a thesis-sourced verdict
    is DISQUALIFIED (the §13.4 #5 self-test-laundering guard)."""

    def __call__(self, solution_code: str, test_code: str) -> Verdict: ...


class AdjudicateCallback(Protocol):
    """An injected human-adjudication callback: ``request -> tuple of >=2 Adjudication``.

    Returns the independent adjudicators' verdicts for an :class:`AdjudicationRequest`. STATELESS /
    re-entrant by contract (4.4c-5 reuses :func:`adjudicate_item` for the noise-audit subsample) —
    it must hold no corpus state. In tests this is a stub returning scripted verdicts; in production
    it is the human-in-the-loop harness. It NEVER calls a model on the labeling path."""

    def __call__(self, request: AdjudicationRequest) -> tuple[Adjudication, ...]: ...


class TieBreakCallback(Protocol):
    """An injected tie-breaker callback (architect / Jim): ``request -> Adjudication``.

    Resolves a split decision (the two primary adjudicators disagree, neither abstaining). A
    tie-breaker ``abstain`` DROPS the item, exactly as a primary abstention does."""

    def __call__(self, request: AdjudicationRequest) -> Adjudication: ...


class RegimeAdjudicateCallback(Protocol):
    """An injected K-regime audit callback: ``request -> RegimeAdjudication``.

    A SECOND author audits a K-error's planter-proposed ``error_regime`` (eval-stats §6.D). Its
    verdict vocabulary is distinct from existence adjudication (``confirm-regime`` /
    ``reassign-regime`` / ``abstain``) and its ``abstain`` KEEPS the item (regime → ``""``), unlike
    an existence ``abstain`` which drops it."""

    def __call__(self, request: AdjudicationRequest) -> RegimeAdjudication: ...


# ===========================================================================
# Frozen request / result types
# ===========================================================================


class StratumAssignment(BaseModel):
    """The mechanical, oracle-determined stratum decision for one item (§3.4).

    ``stratum`` is set from the frozen-oracle ``verdict`` ALONE — :func:`assign_stratum` NEVER reads
    ``item.candidate_stratum`` to set it. ``candidate_agreed`` is reported-only: a disagreement is
    LOGGED, never an error (the oracle overrules the planter's proposal by design).
    """

    model_config = ConfigDict(frozen=True)

    provisional_id: int
    stratum: Literal["O", "K", "clean"]
    oracle_reachable: bool
    verdict: Verdict
    candidate_agreed: bool


class CleanLabel(BaseModel):
    """A re-verified clean label + its provenance (§3.4). Emitted by :func:`reverify_clean` for a
    clean item that earns a C1 (oracle) or C2 (human) stamp. ``label_source`` and the provenance
    union member agree by construction (the :class:`CorpusItem` validator re-checks this)."""

    model_config = ConfigDict(frozen=True)

    provisional_id: int
    label_source: Literal["oracle", "human"]
    label_provenance: OracleLabelProvenance | HumanLabelProvenance


class AbstentionDrop(BaseModel):
    """The drop signal: an item the labeling pipeline REFUSES to promote (INV-3/INV-5). A clean item
    that earns neither a C1 nor a C2 stamp, an existence adjudication that abstains/splits-to-error,
    or any item whose label cannot be honestly established. Carries the ``provisional_id`` (all its
    cells are dropped, item-granular) + a ``reason`` for the audit trail."""

    model_config = ConfigDict(frozen=True)

    provisional_id: int
    reason: str


class AdjudicationRequest(BaseModel):
    """The frozen input to the human harness (§3.4). STATELESS — it carries only what an adjudicator
    needs to render a verdict, holding no pipeline state (so :func:`adjudicate_item` is re-entrant).

    ``purpose`` distinguishes the two verdict jobs (§6.C/§6.D): ``existence`` (is there an error /
    is this clean — folds into :class:`HumanLabelProvenance`) vs ``regime`` (the K-stratum
    error-regime audit — uses :class:`RegimeAdjudication`).
    """

    model_config = ConfigDict(frozen=True)

    provisional_id: int
    purpose: Literal["existence", "regime"]
    problem_statement: str
    proposed_solution: str
    candidate_regime: str = ""
    """For a ``regime`` request: the planter-proposed regime the second author audits. Empty for an
    ``existence`` request."""


class AdjudicationOutcome(BaseModel):
    """The resolved result of the human harness for one item (§3.4).

    For an ``existence`` request: ``existence_verdict`` is the consensus/tie-broken
    ``{error, clean, abstain}`` and ``provenance`` carries the :class:`HumanLabelProvenance` (the
    recorded adjudications + any tie-breaker). For a ``regime`` request: ``regime_adjudication`` is
    the SOLE source of the K-item's ``error_regime``. ``dropped`` is True iff an existence abstain
    (primary both-abstain, split-to-error tie-break abstain) drops the item.
    """

    model_config = ConfigDict(frozen=True)

    provisional_id: int
    purpose: Literal["existence", "regime"]
    existence_verdict: Literal["error", "clean", "abstain"] | None = None
    provenance: HumanLabelProvenance | None = None
    regime_adjudication: RegimeAdjudication | None = None
    dropped: bool = False


class PromotionResult(BaseModel):
    """The output of :func:`promote_corpus` (§4/§5). ``promoted`` are the labeled corpus items;
    ``dropped_ids`` are the abstention/refuse-to-promote drops (item-granular); ``regime_audit``
    records every K-regime :class:`RegimeAdjudication`; ``stratum_drift`` reports every item whose
    oracle-assigned stratum disagreed with the planter's ``candidate_stratum`` (reported-only audit,
    never an error). ``content_hash`` stays ``""`` on every promoted item — 4.4c-5 locks it."""

    model_config = ConfigDict(frozen=True)

    promoted: tuple[CorpusItem, ...]
    dropped_ids: frozenset[int]
    regime_audit: tuple[RegimeAdjudication, ...]
    stratum_drift: tuple[StratumAssignment, ...]


class RegimeExistenceError(ValueError):
    """A K-error carried a regime verdict with no corresponding ``error`` existence
    :class:`Adjudication` (§6.C). Error-existence MUST be established FIRST; a regime verdict is
    meaningful only once existence == ``error``. Refuse-to-promote."""


class ORegimeMismatchError(ValueError):
    """An O item's ``error_regime`` disagreed with :func:`_derive_o_regime` over its planter
    operators (§2.C fail-fast). For an O item the regime derives MECHANICALLY off the operator;
    the author's tag is NEVER trusted. Refuse-to-promote that item (re-run at the 4.4c-5 lock)."""


# ===========================================================================
# Component 1 — assign_stratum (mechanical, oracle-determined; NO adjudicate access)
# ===========================================================================


def assign_stratum(
    item: PlantedItem, *, probe: OracleProbe = run_frozen_check
) -> StratumAssignment:
    """Assign the BINDING ``stratum`` mechanically from the frozen-oracle probe (§3.4, §6.A).

    Binds ``probe(item.thesis.proposed_solution, item.test_code)`` and reads the binding predicates
    off the resulting :class:`Verdict` (note ``valid_check`` is the NOISE bit — a CAUGHT error is
    ``valid_check=True ∧ holds=False``; a noise verdict ``valid_check=False`` is NOT a catch):

    ===================  ==========  =========================================================
    case                 is_error    predicate on the frozen-oracle Verdict
    ===================  ==========  =========================================================
    O-error              1           ``is_executable ∧ valid_check ∧ ¬holds``
    K-error              1           else (the complement of the O-catch)
    clean (any)          0           ``stratum = "clean"`` always
    ``oracle_reachable``  any        ``is_executable ∧ valid_check``  (the C1/C2 tier driver)
    ===================  ==========  =========================================================

    NEVER reads ``item.candidate_stratum`` to set ``stratum``; it only reports whether the oracle
    AGREED with the planter's proposal (``candidate_agreed``) and LOGS a disagreement (the oracle
    overrules by design). A thesis-sourced verdict (``test_provenance != "frozen"``) is DISQUALIFIED
    — this is the offline frozen kernel, and a self-test-launderable verdict is an S9 violation.

    This component has NO access to the human-adjudication seam — stratum is mechanical, never
    human-judged (the hard wall).
    """
    test_code = item.test_code or ""
    verdict = probe(item.thesis.proposed_solution, test_code)

    if verdict.test_provenance != "frozen":
        raise ValueError(
            f"assign_stratum: probe for item {item.provisional_id} returned a "
            f"{verdict.test_provenance!r}-sourced verdict; the frozen kernel must produce a "
            f"'frozen'-sourced verdict (self-test-laundering guard, §13.4 #5)"
        )

    oracle_reachable = verdict.is_executable and verdict.valid_check

    stratum: Literal["O", "K", "clean"]
    if item.is_error == 0:
        stratum = "clean"
    elif verdict.is_executable and verdict.valid_check and not verdict.holds:
        # O-error: the frozen oracle CAUGHT the planted error (a real, executable refutation).
        stratum = "O"
    else:
        # K-error: oracle-blind — not executable, OR a noise verdict (¬valid_check), OR the planted
        # error survived the frozen test (holds). The complement of the O-catch.
        stratum = "K"

    candidate_agreed = stratum == item.candidate_stratum
    if not candidate_agreed:
        log.info(
            "labeling.assign_stratum.candidate_overruled",
            extra={
                "provisional_id": item.provisional_id,
                "candidate_stratum": item.candidate_stratum,
                "assigned_stratum": stratum,
            },
        )

    return StratumAssignment(
        provisional_id=item.provisional_id,
        stratum=stratum,
        oracle_reachable=oracle_reachable,
        verdict=verdict,
        candidate_agreed=candidate_agreed,
    )


# ===========================================================================
# Component 3 — adjudicate_item (the human harness; NO probe access; stateless/re-entrant)
# ===========================================================================


def _resolve_existence(
    request: AdjudicationRequest,
    *,
    adjudicate: AdjudicateCallback,
    tie_break: TieBreakCallback,
) -> AdjudicationOutcome:
    """Resolve an existence adjudication: >=2 independent adjudicators, tie-broken on a split.

    Consensus (both agree, non-abstain) ⟹ that verdict. Split (disagree, neither abstaining) ⟹ the
    tie-breaker. Any abstention (a primary abstains, both abstain, OR the tie-breaker abstains) ⟹
    DROP the item (§6.C). The recorded :class:`HumanLabelProvenance` holds every adjudication + the
    tie-breaker id when one was consulted."""
    adjudications = adjudicate(request)
    if len(adjudications) < 2:
        raise ValueError(
            f"adjudicate_item: existence request for item {request.provisional_id} needs >=2 "
            f"independent adjudicators, got {len(adjudications)}"
        )

    verdicts = [a.verdict for a in adjudications]

    # Any primary abstention drops the item (an honest "I cannot establish this", §6.C).
    if "abstain" in verdicts:
        return AdjudicationOutcome(
            provisional_id=request.provisional_id,
            purpose="existence",
            existence_verdict="abstain",
            provenance=HumanLabelProvenance(adjudications=tuple(adjudications)),
            dropped=True,
        )

    # Consensus: every non-abstaining adjudicator agrees.
    if len(set(verdicts)) == 1:
        return AdjudicationOutcome(
            provisional_id=request.provisional_id,
            purpose="existence",
            existence_verdict=verdicts[0],
            provenance=HumanLabelProvenance(adjudications=tuple(adjudications)),
            dropped=False,
        )

    # Split (error vs clean, neither abstaining) → tie-breaker.
    breaker = tie_break(request)
    if breaker.verdict == "abstain":
        return AdjudicationOutcome(
            provisional_id=request.provisional_id,
            purpose="existence",
            existence_verdict="abstain",
            provenance=HumanLabelProvenance(
                adjudications=(*adjudications, breaker), tie_breaker_id=breaker.adjudicator_id
            ),
            dropped=True,
        )
    return AdjudicationOutcome(
        provisional_id=request.provisional_id,
        purpose="existence",
        existence_verdict=breaker.verdict,
        provenance=HumanLabelProvenance(
            adjudications=(*adjudications, breaker), tie_breaker_id=breaker.adjudicator_id
        ),
        dropped=False,
    )


def adjudicate_item(
    request: AdjudicationRequest,
    *,
    adjudicate: AdjudicateCallback,
    tie_break: TieBreakCallback,
    regime_adjudicate: RegimeAdjudicateCallback | None = None,
) -> AdjudicationOutcome:
    """The human-adjudication harness (§3.4) — STATELESS / re-entrant (4.4c-5 reuses it for the
    noise-audit subsample; it holds NO corpus state). Has NO access to the oracle ``probe`` — an
    oracle-blind label is the human's job (the hard wall).

    Routes on ``request.purpose``:
      - ``existence`` (error-existence / C2-clean): >=2 independent adjudicators
        (``{error,clean,abstain}``), ``architect``/Jim as tie-breaker. Abstain (both, or the
        tie-breaker abstains) ⟹ DROP (all cells). Consensus ⟹ that verdict; split ⟹ tie-breaker.
        Folds into :class:`HumanLabelProvenance`.
      - ``regime`` (K-regime audit): the second-author :class:`RegimeAdjudication`
        (``{confirm-regime,reassign-regime,abstain}``). ``confirm-regime`` keeps the planter tag,
        ``reassign-regime`` makes ``reassigned_regime`` the regime, ``abstain`` blanks it (``""``)
        (the item STAYS — opposite of an existence abstain). The :class:`RegimeAdjudication` is the
        SOLE source of a K item's ``error_regime``, never the planter stamp.
    """
    if request.purpose == "existence":
        return _resolve_existence(request, adjudicate=adjudicate, tie_break=tie_break)

    # purpose == "regime"
    if regime_adjudicate is None:
        raise ValueError(
            f"adjudicate_item: a 'regime' request for item {request.provisional_id} needs a "
            f"regime_adjudicate callback"
        )
    regime = regime_adjudicate(request)
    return AdjudicationOutcome(
        provisional_id=request.provisional_id,
        purpose="regime",
        regime_adjudication=regime,
        dropped=False,
    )


# ===========================================================================
# Component 2 — reverify_clean (C1 reuse-the-verdict / C2 route-to-human; NO second probe)
# ===========================================================================


def reverify_clean(
    item: PlantedItem,
    assignment: StratumAssignment,
    *,
    adjudicate: AdjudicateCallback,
    tie_break: TieBreakCallback,
) -> CleanLabel | AbstentionDrop:
    """Re-verify a CLEAN item's label, routing to C1 (oracle) or C2 (human) by ``oracle_reachable``.

    Only for clean items (``item.is_error == 0`` / ``assignment.stratum == "clean"``). The
    refuse-to-promote guard (INV-3): a clean item that earns NEITHER a C1 nor a C2 stamp is DROPPED.

      - **C1 (oracle-reachable):** REUSE the carried ``assignment.verdict`` (§6.B — the kernel
        is deterministic; no second probe call). Predicate ``is_executable ∧ valid_check ∧ holds`` →
        :class:`OracleLabelProvenance` (``test_provenance="frozen"``, ``label_source="oracle"``). A
        carried verdict that does NOT satisfy the predicate is a corpus defect — DROPPED. An
        oracle-reachable clean item NEVER reaches the human callback.
      - **C2 (oracle-blind):** route to the human harness → :class:`HumanLabelProvenance`,
        ``label_source="human"`` if the consensus/tie-broken verdict is ``clean``; an abstain or a
        split-to-``error`` is an ABSTENTION-DROP.

    Takes NO ``probe``: C1 REUSES the carried ``assignment.verdict`` (§6.B — the kernel is
    deterministic; never a second probe call), and C2 is the human's job.
    """
    if item.is_error != 0:
        raise ValueError(
            f"reverify_clean: item {item.provisional_id} is not clean (is_error="
            f"{item.is_error}); only clean items are re-verified here"
        )

    if assignment.oracle_reachable:
        # C1 — reuse the carried verdict (no second probe). Predicate: executable ∧ valid ∧ holds.
        v = assignment.verdict
        if v.is_executable and v.valid_check and v.holds:
            return CleanLabel(
                provisional_id=item.provisional_id,
                label_source="oracle",
                label_provenance=OracleLabelProvenance(
                    returncode=0 if v.holds else 1,
                    test_provenance="frozen",
                    holds=v.holds,
                    valid_check=v.valid_check,
                    oracle_id="run_frozen_check",
                ),
            )
        # Oracle-reachable but the clean item did NOT pass its frozen test — refuse to promote
        # (a clean item that fails its own test is a corpus defect, not a C1 label).
        return AbstentionDrop(
            provisional_id=item.provisional_id,
            reason=(
                f"C1 refuse-to-promote: oracle-reachable clean item failed its frozen test "
                f"(holds={v.holds}, valid_check={v.valid_check})"
            ),
        )

    # C2 — oracle-blind: route to the human harness.
    request = AdjudicationRequest(
        provisional_id=item.provisional_id,
        purpose="existence",
        problem_statement=item.frame.problem_statement,
        proposed_solution=item.thesis.proposed_solution,
    )
    outcome = adjudicate_item(request, adjudicate=adjudicate, tie_break=tie_break)
    if outcome.dropped or outcome.existence_verdict != "clean":
        return AbstentionDrop(
            provisional_id=item.provisional_id,
            reason=(
                f"C2 abstention-drop: human existence verdict "
                f"{outcome.existence_verdict!r} (not 'clean')"
            ),
        )
    assert outcome.provenance is not None  # a non-dropped existence outcome always carries one
    return CleanLabel(
        provisional_id=item.provisional_id,
        label_source="human",
        label_provenance=outcome.provenance,
    )


# ===========================================================================
# Component 4 — promote_corpus (the orchestrator; the ONLY place probe + adjudicate meet)
# ===========================================================================


def _build_corpus_item(
    item: PlantedItem,
    *,
    stratum: Literal["O", "K", "clean"],
    oracle_reachable: bool,
    label_source: Literal["oracle", "human"],
    label_provenance: OracleLabelProvenance | HumanLabelProvenance,
    error_regime: str,
) -> CorpusItem:
    """Carry the planter's pre-label truth through into a labeled :class:`CorpusItem`, attaching the
    binding ``stratum`` / ``oracle_reachable`` / label / provenance / ``error_regime``. The hash
    stays ``""`` — 4.4c-5 locks it."""
    return CorpusItem(
        item_id=item.provisional_id,
        frame=item.frame,
        thesis=item.thesis,
        test_code=item.test_code,
        is_error=item.is_error,
        label_source=label_source,
        label_provenance=label_provenance,
        stratum=stratum,
        oracle_reachable=oracle_reachable,
        error_regime=error_regime,
        difficulty=item.difficulty,
        matched_sibling_id=item.matched_sibling_id,
        split=item.split,
        planter=item.planter,
        content_hash="",
    )


def _o_operators(item: PlantedItem) -> tuple[str, ...]:
    """The deterministic planter operators behind an O item (for the §2.C regime cross-check). An O
    item must come from a :class:`DeterministicPlanterStamp`; anything else is a corpus defect."""
    stamp = item.planter
    if not isinstance(stamp, DeterministicPlanterStamp):
        raise ORegimeMismatchError(
            f"O item {item.provisional_id} carries a non-deterministic planter stamp "
            f"{type(stamp).__name__}; an O-stratum item must derive its regime from operators"
        )
    return stamp.operators


def _promote_error_item(
    item: PlantedItem,
    assignment: StratumAssignment,
    *,
    adjudicate: AdjudicateCallback,
    tie_break: TieBreakCallback,
    regime_adjudicate: RegimeAdjudicateCallback,
    derive_o_regime: Callable[[tuple[str, ...]], str],
    regime_audit: list[RegimeAdjudication],
) -> CorpusItem | AbstentionDrop:
    """Promote one ERROR item (is_error==1) given its mechanical stratum assignment.

    O-stratum: the regime derives MECHANICALLY off the planter operators (§2.C fail-fast) — the
    author's ``error_regime`` is cross-checked and a mismatch HARD-FAILS that item. The label is the
    oracle catch (``label_source="oracle"`` with the carried frozen verdict).

    K-stratum: ``label_source="human"`` with an explicit ``error`` existence :class:`Adjudication`
    (§6.C) — established FIRST. THEN the K-regime second-author audit
    (:class:`RegimeAdjudication`) sets ``error_regime`` (its SOLE source). detK probe items are
    carried through untouched (regime already derives off the real operator; no K-audit).
    """
    if assignment.stratum == "O":
        operators = _o_operators(item)
        derived = derive_o_regime(operators)
        if item.error_regime != derived:
            raise ORegimeMismatchError(
                f"O item {item.provisional_id}: author error_regime {item.error_regime!r} != "
                f"derived {derived!r} from operators {operators!r} (§2.C fail-fast)"
            )
        return _build_corpus_item(
            item,
            stratum="O",
            oracle_reachable=assignment.oracle_reachable,
            label_source="oracle",
            label_provenance=OracleLabelProvenance(
                returncode=1,  # an O catch is a failing frozen test (holds is False)
                test_provenance="frozen",
                holds=assignment.verdict.holds,
                valid_check=assignment.verdict.valid_check,
                oracle_id="run_frozen_check",
            ),
            error_regime=derived,
        )

    # K-stratum. detK probe items: carried through untouched (regime derives off the real operator).
    is_detk_probe = (
        isinstance(item.planter, DeterministicPlanterStamp)
        and DETK_PROBE_OPERATOR in item.planter.operators
    )

    # §6.C: error-existence FIRST. A K-error carries an explicit `error` existence Adjudication.
    existence_request = AdjudicationRequest(
        provisional_id=item.provisional_id,
        purpose="existence",
        problem_statement=item.frame.problem_statement,
        proposed_solution=item.thesis.proposed_solution,
    )
    existence = adjudicate_item(existence_request, adjudicate=adjudicate, tie_break=tie_break)
    if existence.dropped or existence.existence_verdict != "error":
        return AbstentionDrop(
            provisional_id=item.provisional_id,
            reason=(
                f"K-error existence not established: human verdict "
                f"{existence.existence_verdict!r} (need 'error')"
            ),
        )
    assert existence.provenance is not None
    # §6.C invariant guard: the human provenance MUST carry an `error` existence Adjudication.
    if not any(a.verdict == "error" for a in existence.provenance.adjudications):
        raise RegimeExistenceError(
            f"K-error {item.provisional_id} has no 'error' existence Adjudication in its human "
            f"provenance — a regime verdict is meaningless without established existence (§6.C)"
        )

    if is_detk_probe:
        # detK: regime already derives off the real operator at planting time; no K-audit.
        error_regime = item.error_regime
    else:
        # K-regime second-author audit — the SOLE source of error_regime.
        regime_request = AdjudicationRequest(
            provisional_id=item.provisional_id,
            purpose="regime",
            problem_statement=item.frame.problem_statement,
            proposed_solution=item.thesis.proposed_solution,
            candidate_regime=item.error_regime,
        )
        regime_outcome = adjudicate_item(
            regime_request,
            adjudicate=adjudicate,
            tie_break=tie_break,
            regime_adjudicate=regime_adjudicate,
        )
        ra = regime_outcome.regime_adjudication
        assert ra is not None
        regime_audit.append(ra)
        if ra.verdict == "confirm-regime":
            error_regime = item.error_regime
        elif ra.verdict == "reassign-regime":
            assert ra.reassigned_regime is not None  # enforced by the RegimeAdjudication validator
            error_regime = ra.reassigned_regime
        else:  # abstain → error_regime = ""
            error_regime = ""

    return _build_corpus_item(
        item,
        stratum="K",
        oracle_reachable=assignment.oracle_reachable,
        label_source="human",
        label_provenance=existence.provenance,
        error_regime=error_regime,
    )


def _no_regime(request: AdjudicationRequest) -> RegimeAdjudication:
    """Default regime callback: refuses. A ``regime`` request reached the harness with no
    ``regime_adjudicate`` wired — a programming error, surfaced loudly rather than silently
    abstaining (which would hide a missing audit seam)."""
    raise ValueError(
        f"promote_corpus: a K-regime audit was requested for item {request.provisional_id} but no "
        f"regime_adjudicate callback was provided"
    )


def promote_corpus(
    pairs: Sequence[PlantedPair],
    singletons: Sequence[PlantedItem],
    *,
    probe: OracleProbe = run_frozen_check,
    adjudicate: AdjudicateCallback,
    tie_break: TieBreakCallback,
    regime_adjudicate: RegimeAdjudicateCallback = _no_regime,
    derive_o_regime: Callable[[tuple[str, ...]], str] = _derive_o_regime,
) -> PromotionResult:
    """Orchestrate the corpus-labeling pipeline (§4/§5) — the ONLY place ``probe`` + ``adjudicate``
    meet (the hard wall). Per item: assign stratum → re-verify clean / attach error provenance →
    adjudicate where needed → build :class:`CorpusItem`, carrying through the planter's pre-label
    truth (``is_error``, ``difficulty``, ``matched_sibling_id``, ``split``, ``planter``) and the
    binding label fields. ``content_hash`` stays ``""`` (4.4c-5 locks it).

    Invariants enforced here:
      - **INV-3 (refuse-to-promote unverified clean):** a clean item earning neither C1 nor C2 is
        DROPPED, not promoted.
      - **INV-5 (item-granular abstention-drop):** ``dropped_ids`` holds exactly the dropped
        ``provisional_id``s; ``promoted`` excludes them.
      - **§6.C (existence-before-regime):** a K-error carries an explicit ``error`` existence
        :class:`Adjudication`; a regime verdict with no established existence is rejected.
      - **§2.C (O-regime fail-fast):** an O item whose ``error_regime`` disagrees with
        ``derive_o_regime(planter.operators)`` HARD-FAILS (refuse to promote that item).

    DEFERRED to 4.4c-5 (NOT done here): the post-abstention matched-sibling bijection re-validation
    (this only emits ``dropped_ids``), the ``content_hash`` / git-SHA lock.
    """
    all_items: list[PlantedItem] = []
    for pair in pairs:
        all_items.append(pair.error_item)
        all_items.append(pair.clean_item)
    all_items.extend(singletons)

    promoted: list[CorpusItem] = []
    dropped: set[int] = set()
    regime_audit: list[RegimeAdjudication] = []
    stratum_drift: list[StratumAssignment] = []

    for item in all_items:
        assignment = assign_stratum(item, probe=probe)
        if not assignment.candidate_agreed:
            stratum_drift.append(assignment)

        if item.is_error == 0:
            # Clean item → C1 (oracle) or C2 (human) by oracle_reachable.
            result = reverify_clean(
                item, assignment, adjudicate=adjudicate, tie_break=tie_break
            )
            if isinstance(result, AbstentionDrop):
                dropped.add(result.provisional_id)
                continue
            promoted.append(
                _build_corpus_item(
                    item,
                    stratum="clean",
                    oracle_reachable=assignment.oracle_reachable,
                    label_source=result.label_source,
                    label_provenance=result.label_provenance,
                    error_regime="",
                )
            )
            continue

        # Error item → O (mechanical regime) or K (human existence + regime audit).
        err_result = _promote_error_item(
            item,
            assignment,
            adjudicate=adjudicate,
            tie_break=tie_break,
            regime_adjudicate=regime_adjudicate,
            derive_o_regime=derive_o_regime,
            regime_audit=regime_audit,
        )
        if isinstance(err_result, AbstentionDrop):
            dropped.add(err_result.provisional_id)
            continue
        promoted.append(err_result)

    return PromotionResult(
        promoted=tuple(promoted),
        dropped_ids=frozenset(dropped),
        regime_audit=tuple(regime_audit),
        stratum_drift=tuple(stratum_drift),
    )


__all__ = [
    "AbstentionDrop",
    "AdjudicateCallback",
    "AdjudicationOutcome",
    "AdjudicationRequest",
    "CleanLabel",
    "ORegimeMismatchError",
    "OracleProbe",
    "PromotionResult",
    "RegimeAdjudicateCallback",
    "RegimeExistenceError",
    "StratumAssignment",
    "TieBreakCallback",
    "adjudicate_item",
    "assign_stratum",
    "promote_corpus",
    "reverify_clean",
]
