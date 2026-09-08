"""Adversarial K->O converter for the Phase-4 GATE corpus (Pod 4.4c-3.5; plan §3.5/§4/§7).

The converter attempts to FLIP an LLM-planted K-stratum error (oracle-blind by construction) into an
executable O-stratum error by having an adversarial panel + counter-example adversary SYNTHESIZE a
discriminating test: one that the planted-wrong solution FAILS and its clean sibling PASSES. A
candidate that clears the two-sided gate AND a mandatory determinism re-check rewrites the K error
into an O-shaped :class:`~cogworx.eval.planting.PlantedItem` carrying a
:class:`~cogworx.eval.corpus.ConvertedPlanterStamp`; an item no round can flip is emitted UNCHANGED
as residual-K. The output feeds the EXISTING, unchanged
:func:`cogworx.eval.labeling.promote_corpus` (its ``ConvertedPlanterStamp`` O-branch already handles
the converted items; its human K-branch handles the residuals).

S1 POSTURE (the #1 red-team target — read before touching this module):
  **Authoring-time tooling. Runs offline to produce the frozen corpus; not on any live write-path.
  ZERO journal I/O, ZERO live-model drive, ZERO StageContext — same S1-clean posture as
  ``planting.py`` / ``labeling.py``.** Every external seam is DEPENDENCY-INJECTED and provider-
  agnostic (S4 — no concrete provider import):
    - the adversary panel (:class:`AdversaryPanel`) proposes ADVISORY probe-family hypotheses —
      never a label, never counting toward conversion.
    - the counter-example adversary (:class:`CounterExampleAdversary`) synthesizes a candidate
      ``test_code``.
    - the gate executor is the SAME :class:`~cogworx.eval.labeling.OracleProbe` the labeler uses
      (default :func:`cogworx.eval._authoring.run_frozen_check` — the journal-free frozen kernel).
      It is NEVER ``CodeOracle.evaluate`` (that needs a journal + StageContext — the live path). The
      converter and the labeler MUST agree on what "executable" means, so the probe TYPE is reused,
      never re-invented. A test pins no ``StageContext`` / ``Journal`` / ``CodeOracle`` symbol is
      reachable from this module.

THE GATE (a PLAIN BOOLEAN — eval-stats: a hard executed predicate, NO multiplicity correction):
  A candidate converts iff, over FOUR probe calls:
    - ``v_err  = probe(error.proposed_solution,  candidate)`` is ``is_executable ∧ valid_check ∧
      ¬holds`` (the planted-wrong solution FAILS the SYNTHESIZED candidate — an O-catch), AND
    - ``v_clean = probe(clean.proposed_solution, candidate)`` is ``is_executable ∧ valid_check ∧
      holds`` (the clean sibling PASSES it — the anti-laundering guard vs an ``assert False`` test),
      AND
    - the AUTHOR-ANCHOR (HOLE#1, architect-ruled) over the ORIGINAL author-frozen ``item.test_code``
      (identical on the error item and its clean sibling): ``a_clean = probe(clean,
      item.test_code)`` PASSES and ``a_err = probe(error, item.test_code)`` FAILS — i.e. the
      author's frozen test independently AGREES with the planter's ``is_error`` claim. Without this,
      an adversary could
      synthesize a real, deterministic, gate-passing test that discriminates on an INCIDENTAL
      property (e.g. int-vs-float return when the real bug is a missing divide), minting an
      ``oracle`` label that encodes the WRONG intent. Frozen-test AGREEMENT is sufficient — this is
      NOT a behavioral-equivalence relation (rejected: it would break eval-stats' no-multiplicity
      posture). An item with ``test_code is None`` cannot be author-anchored and is NOT convertible
      (routed to residual-K, excluded from the ``conv_rate_llmK`` denominator).
  THEN a MANDATORY determinism re-check (eval-stats, load-bearing): re-run the CANDIDATE probe ONCE
  MORE on both solutions; if ``holds`` disagrees for EITHER on the re-run, the candidate is REJECTED
  (flaky test) and rounds continue. This enforces the deterministic-faithful precondition the
  no-multiplicity ruling rests on. There is NO ``quantile`` / beta / bootstrap call here.

  SCOPE OF THE DETERMINISM RE-CHECK (CF-4.4c-CONVERTER-ENV-PIN, honesty note): the re-check proves
  IN-PROCESS stability ONLY (the same probe, same interpreter, same wall-clock window yields the
  same ``holds``). It does NOT establish ENVIRONMENT-INDEPENDENCE: a test sensitive to wall-clock,
  float-repr, locale, or package-version could still flip across environments and pass this
  re-check. Environment-independence is a SEPARATE assumed precondition, delivered by the 4.4c-5
  lock fingerprint-fencing the execution environment (NOT this module's job). The author-anchor
  probes are deterministic-by-construction (the seed was self-checked against ``test_code`` at
  planting) and are deliberately EXCLUDED from this re-check.

Pure stdlib + pydantic (+ the injected panel / adversary / probe seams); no substrate (CANON S1, S2,
S4). Reuses the landed seams (``PlantedItem`` / ``PlantedPair`` / ``is_detk`` from
:mod:`cogworx.eval.planting`; ``ConvertedPlanterStamp`` from :mod:`cogworx.eval.corpus`;
``OracleProbe`` from :mod:`cogworx.eval.labeling`; ``Verdict`` from
:mod:`cogworx.verification.contracts`) — reused, not redefined.

Contract changelog (CANON §6.1):
  - 2026-06-21 (Pod 4.4c-3.5): initial — the adversarial K->O converter (convert_k_pool + the panel
    + adversary seams + the frozen round/result/audit types + PanelFamilyCollision). New module; no
    existing callers. Additive new public surface only; the downstream ``promote_corpus``
    ``ConvertedPlanterStamp`` branch (python-expert, 4.4c-3.5) consumes its output unchanged.
  - 2026-06-21 (Pod 4.4c-3.5, HOLE#1 fix — BEHAVIOR CHANGE): the gate gains a CO-EQUAL third arm,
    the AUTHOR-ANCHOR (``_author_anchor`` over ``item.test_code``), closing the wrong-intent
    label-laundering hole. ``_gate`` now takes four verdicts. An item with ``test_code is None`` is
    NOT convertible (routed to residual-K, excluded from the ``conv_rate_llmK`` denominator). Two
    honesty notes added: CF-4.4c-CONVERTER-ENV-PIN (the determinism re-check proves in-process
    stability only, not environment-independence — 4.4c-5 fingerprint-fences the env) and
    CF-4.4c-CONVERTER-SELECTION(ii) (``model_family`` string-disjointness is
    necessary-not-sufficient for distributional independence; correlated pretraining shares blinds).
  - 2026-06-21 (Pod 4.4c-3.5, DOWNSTREAM 4.4d CONTRACT): a converted item carries a
    ``ConvertedPlanterStamp`` (``injector_kind="converted"``). 4.4d MUST emit
    ``Cell.converted_o=True`` (read off the ``ConvertedPlanterStamp``) for every cell of a converted
    item, and MUST use :func:`cogworx.eval.youden.is_converted_o` to keep converted-O OUT of the
    binding D>A/D>C' delta — converted-O is a selection-biased-easier slice of K
    (CF-4.4c-CONVERTER-SELECTION). The bootstrap NEVER reads ``converted_o`` (it partitions on
    ``stratum``); it is a provenance-only marker.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Final, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from cogworx.eval._authoring import run_frozen_check
from cogworx.eval.corpus import ConvertedPlanterStamp, LLMPlanterStamp
from cogworx.eval.labeling import OracleProbe
from cogworx.eval.planting import PlantedItem, PlantedPair, is_detk
from cogworx.verification.contracts import Verdict

log = logging.getLogger(__name__)

K_ROUNDS: Final = 8
"""eval-stats ruling: every K item gets up to 8 conversion rounds; immutable per corpus lock (folded
into the MeasurementFingerprint at 4.4c-5 — the converter only EXPOSES it). NO data-dependent
stopping; the only early exit is at the FIRST gate-passing + determinism-re-checked candidate."""


# ===========================================================================
# Injected seams (provider-agnostic, S4) — the adversary panel + the counter-example adversary.
# The gate executor reuses cogworx.eval.labeling.OracleProbe (NOT a new probe type).
# ===========================================================================


class PanelHypothesis(BaseModel):
    """One probe-family's ADVISORY hypothesis about how the planted error might be caught.

    ADVISORY ONLY: a hypothesis is NEVER a label and NEVER counts toward conversion — only an
    executed probe verdict (the two-sided gate) can convert an item. It carries the proposing
    ``family`` (for the audit trail) + a free-text ``rationale`` the adversary may condition on."""

    model_config = ConfigDict(frozen=True)

    family: str
    rationale: str


class AdversaryPanel(Protocol):
    """The injected advisory panel seam: ``propose(item, clean_sibling, round_index)``.

    N probe families each emit a :class:`PanelHypothesis` — advisory ONLY (never a label, never
    counts toward conversion). In tests this is a stub returning scripted hypotheses; in production
    it is the multi-family LLM panel behind a provider-agnostic :class:`~cogworx.model.base.Model`
    (S4 — this module imports no concrete provider)."""

    def propose(
        self, item: PlantedItem, clean_sibling: PlantedItem, round_index: int
    ) -> tuple[PanelHypothesis, ...]: ...


class CounterExampleAdversary(Protocol):
    """The injected adversary seam: ``synthesize(...) -> candidate test_code``.

    Given the error item, its clean sibling, the panel's advisory hypotheses, and the round index,
    returns a candidate ``test_code`` string. In tests this is a stub returning scripted candidates;
    in production it is the counter-example LLM behind a provider-agnostic
    :class:`~cogworx.model.base.Model` (S4)."""

    def synthesize(
        self,
        item: PlantedItem,
        clean_sibling: PlantedItem,
        hypotheses: tuple[PanelHypothesis, ...],
        round_index: int,
    ) -> str: ...


# ===========================================================================
# Frozen config / round / result / audit types
# ===========================================================================


class PanelFamilyCollision(RuntimeError):
    """Family-disjointness violation (§3.5 detectability). Raised at :class:`PanelConfig`
    construction when a panel family or the adversary family collides with the FORBIDDEN set
    (the planter family + every arm family: thesis / antithesis / C-prime / C / D / D-prime) or
    with EACH OTHER. A colliding converter would be in-distribution with an arm and could launder a
    planted K error into an apparent O catch — a LOUD, logged failure, never a silent same-family
    stamp."""


class PanelConfig(BaseModel):
    """Frozen converter config (§3.5). Construction RAISES :exc:`PanelFamilyCollision` if any panel
    family or the adversary family collides with ``forbidden_families`` (planter + all arm families)
    or with each other — the family-disjointness invariant.

    ``degrade`` mirrors :class:`~cogworx.eval.planting.SameFamilyFallback`'s loud-degrade posture:
      - ``"abstain-pool"`` (default): if fewer than ``len(panel_families) + 1`` DISJOINT families
        are available (the panel + adversary cannot be made disjoint from the forbidden set),
        convert NOTHING — route the entire K pool to residual, set ``degraded=True`` on the result,
        log loudly. A degraded panel NEVER gains converter authority.
      - ``"raise"``: raise :exc:`PanelFamilyCollision` instead.
    """

    model_config = ConfigDict(frozen=True)

    panel_families: tuple[str, ...]
    adversary_family: str
    forbidden_families: frozenset[str]
    degrade: Literal["abstain-pool", "raise"] = "abstain-pool"
    k_rounds: int = K_ROUNDS

    def _check_disjoint(self) -> None:
        """Raise :exc:`PanelFamilyCollision` on any family collision (panel/adversary vs forbidden,
        or panel/adversary internal duplicate). Called at construction; the convert path re-checks
        the same condition for its degrade routing."""
        used = (*self.panel_families, self.adversary_family)
        # internal duplicate (a family appearing twice across panel+adversary)
        if len(set(used)) != len(used):
            raise PanelFamilyCollision(
                f"panel/adversary families are not internally distinct: {used!r}"
            )
        clash = set(used) & self.forbidden_families
        if clash:
            raise PanelFamilyCollision(
                f"panel/adversary families {sorted(clash)!r} collide with the forbidden set "
                f"(planter + arm families) {sorted(self.forbidden_families)!r}: a same-family "
                f"converter is in-distribution with an arm and defeats §3.5 detectability"
            )
        # HONESTY NOTE (CF-4.4c-CONVERTER-SELECTION(ii)): ``model_family`` STRING-disjointness is
        # NECESSARY-NOT-SUFFICIENT for true distributional independence. Two families with distinct
        # name strings can still share blind spots through correlated pretraining (shared corpora,
        # shared base checkpoints, shared RLHF recipes), so a "disjoint" converter may collude with
        # an arm on the same errors despite passing this check. This guard catches only the
        # name-identity case; the residual correlated-blind-spot risk is tracked as
        # CF-4.4c-CONVERTER-SELECTION(ii) (lift trigger: the live cross-family converter run).

    def __init__(self, **data: object) -> None:
        super().__init__(**data)
        self._check_disjoint()


class RoundArtifact(BaseModel):
    """One conversion round's executed record (audit trail). Carries the candidate test, BOTH gate
    verdicts, and whether this round converted the item. Mutation-resistant audit: a converted
    item's winning round is the FIRST ``converted=True`` artifact."""

    model_config = ConfigDict(frozen=True)

    round_index: int
    candidate_test: str
    v_err: Verdict
    v_clean: Verdict
    converted: bool


class ConversionAudit(BaseModel):
    """Per-item conversion audit + the corpus conversion-rate fields (eval-stats §8).

    ``rounds`` holds every executed :class:`RoundArtifact` per converted/attempted item, keyed by
    the error item's ``provisional_id``. The three rates are computed over the WHOLE attempted pool:
      - ``conv_rate_llmK`` (PRIMARY): converted LLM-planted-K / all LLM-planted-K, EXCLUDING detK.
      - ``conv_rate_detK``: converted detK / all detK (reported-only).
      - ``conv_rate_all``: converted / all attempted (excluding nothing).
    The ≥0.50 re-ground flag / conversion↔arm correlation is a 4.4d/4.4e concern; this audit only
    exposes the rates honestly."""

    model_config = ConfigDict(frozen=True)

    rounds: dict[int, tuple[RoundArtifact, ...]]
    conv_rate_llmK: float
    conv_rate_detK: float
    conv_rate_all: float


class ConversionResult(BaseModel):
    """The output of :func:`convert_k_pool` (§4). ``pairs`` / ``singletons`` are the CONVERTED items
    rewritten as O-shaped :class:`~cogworx.eval.planting.PlantedItem`s (a converted pair keeps its
    clean sibling unchanged); ``residual_pairs`` / ``residual_singletons`` are the items no round
    could flip, emitted UNCHANGED as K. ``conversion_audit`` records every round + the rates;
    ``degraded`` is True iff a degraded panel routed the whole pool to residual.

    The converted + residual streams together feed the EXISTING, unchanged ``promote_corpus``."""

    model_config = ConfigDict(frozen=True)

    pairs: tuple[PlantedPair, ...]
    singletons: tuple[PlantedItem, ...]
    residual_pairs: tuple[PlantedPair, ...]
    residual_singletons: tuple[PlantedItem, ...]
    conversion_audit: ConversionAudit
    degraded: bool


# ===========================================================================
# The gate + determinism re-check (plain booleans; NO quantile / beta / bootstrap)
# ===========================================================================


def _is_o_catch(v: Verdict) -> bool:
    """The O-catch predicate (MUST agree with ``labeling.assign_stratum``): the planted-wrong
    solution FAILS an executable, valid check (``is_executable ∧ valid_check ∧ ¬holds``)."""
    return v.is_executable and v.valid_check and not v.holds


def _is_clean_pass(v: Verdict) -> bool:
    """The clean-pass predicate (the anti-laundering half): the clean sibling PASSES an executable,
    valid check (``is_executable ∧ valid_check ∧ holds``). An ``assert False``-style test that fails
    BOTH solutions is rejected here."""
    return v.is_executable and v.valid_check and v.holds


def _author_anchor(a_err: Verdict, a_clean: Verdict) -> bool:
    """The author-anchor predicate (HOLE#1 anti-wrong-intent guard, architect-ruled): the
    synthesized candidate test discriminates the planted-wrong solution from its clean sibling, but
    NOTHING in the two-sided gate ties that discrimination to the item's ORIGINAL author-frozen
    intent. An adversary could synthesize a real, deterministic, gate-passing test that
    discriminates on an INCIDENTAL property (e.g. int-vs-float return when the real bug is a missing
    divide), minting an ``oracle`` label that encodes the WRONG intent.

    The anchor re-probes BOTH solutions against ``item.test_code`` (the seed's author-frozen
    completion-criterion test, identical on the error item and its clean sibling) and requires the
    author's frozen test to AGREE with the planter's ``is_error`` claim: the clean sibling PASSES
    the author test (``_is_clean_pass``) AND the planted-wrong solution FAILS it (``_is_o_catch``).
    When the author test does NOT itself fail the planted error, the conversion is laundering a
    wrong intent — REJECTED.

    Frozen-test AGREEMENT is sufficient (architect ruling): this is NOT a behavioral-equivalence
    relation between the author test and the synthesized candidate (that was explicitly rejected —
    it would break eval-stats' no-multiplicity posture). The author-anchor probes are
    deterministic-by-construction (the seed was self-checked against ``test_code`` at planting), so
    they are NOT added to the determinism re-check loop (that stays scoped to the two candidate
    probes)."""
    return _is_clean_pass(a_clean) and _is_o_catch(a_err)


def _gate(v_err: Verdict, v_clean: Verdict, a_err: Verdict, a_clean: Verdict) -> bool:
    """The conversion gate — a PLAIN BOOLEAN over four executed verdicts (eval-stats: a hard
    executed predicate, no multiplicity correction). Converts iff (the candidate is a discriminating
    O-test: the error FAILS it AND the clean sibling PASSES it) AND the author-anchor holds (the
    item's ORIGINAL author-frozen test independently AGREES with the planter's ``is_error`` claim).
    The author-anchor is a CO-EQUAL requirement that kills wrong-intent conversions (HOLE#1)."""
    return _is_o_catch(v_err) and _is_clean_pass(v_clean) and _author_anchor(a_err, a_clean)


# ===========================================================================
# Per-item conversion
# ===========================================================================


def _converted_stamp(
    item: PlantedItem, *, adversary_family: str, winning_round: int
) -> ConvertedPlanterStamp:
    """Build the :class:`ConvertedPlanterStamp` for a flipped item — carrying the ORIGINAL LLM
    planting identity (honest provenance survives the conversion) + the adversary family + the
    winning round. The error item MUST carry an :class:`LLMPlanterStamp` (only LLM-planted K is
    converted; detK is excluded upstream)."""
    stamp = item.planter
    if not isinstance(stamp, LLMPlanterStamp):
        raise TypeError(
            f"convert: item {item.provisional_id} carries a {type(stamp).__name__}, not an "
            f"LLMPlanterStamp — only LLM-planted K errors are convertible (detK excluded upstream)"
        )
    return ConvertedPlanterStamp(
        planter_model_family=stamp.model_family,
        planter_model_id=stamp.model_id,
        adversary_family=adversary_family,
        winning_round=winning_round,
    )


def _rewrite_as_o(
    item: PlantedItem, *, test_code: str, stamp: ConvertedPlanterStamp
) -> PlantedItem:
    """Rewrite a flipped K error as an O-shaped :class:`PlantedItem`: ``test_code`` <- the winning
    candidate, ``candidate_stratum`` <- ``"O"``, ``planter`` <- the converted stamp. Everything else
    (frame, thesis, is_error, error_regime CANDIDATE, difficulty, sibling, split) is carried through
    unchanged. ``error_regime`` stays a CANDIDATE — ``promote_corpus`` second-author-audits it."""
    return item.model_copy(
        update={"test_code": test_code, "candidate_stratum": "O", "planter": stamp}
    )


def _attempt_convert(
    item: PlantedItem,
    clean_sibling: PlantedItem,
    *,
    panel: AdversaryPanel,
    adversary: CounterExampleAdversary,
    probe: OracleProbe,
    config: PanelConfig,
) -> tuple[PlantedItem | None, tuple[RoundArtifact, ...]]:
    """Run up to ``config.k_rounds`` conversion rounds on ONE error item. Returns
    ``(converted_item_or_None, round_artifacts)``. Early-success exits at the FIRST round whose
    candidate clears the two-sided gate AND the mandatory determinism re-check.

    Per round: panel proposes (advisory) -> adversary synthesizes a candidate -> probe TWICE for
    the candidate gate, then probe TWICE MORE against ``item.test_code`` (the author-frozen test)
    for the author-anchor (HOLE#1). On a gate pass, the MANDATORY determinism re-check re-runs the
    CANDIDATE probe ONCE MORE on both solutions; a ``holds`` disagreement on either REJECTS the
    candidate (flaky) and continues. The author-anchor probes are deterministic-by-construction (the
    seed was self-checked against ``test_code``), so they are deliberately NOT in that re-check."""
    artifacts: list[RoundArtifact] = []
    err_sol = item.thesis.proposed_solution
    clean_sol = clean_sibling.thesis.proposed_solution
    # The author-frozen test is identical on the error item and its clean sibling (mutate-then-
    # revert). `_convertible` already routed a `test_code is None` item to residual, so this is a
    # non-None str here; assert as a load-bearing precondition (the anchor cannot run without it).
    author_test = item.test_code
    assert author_test is not None, (
        f"_attempt_convert: item {item.provisional_id} has test_code=None — it must have been "
        f"routed to residual by _convertible (the author-anchor cannot be evaluated without it)"
    )

    for r in range(1, config.k_rounds + 1):
        hypotheses = panel.propose(item, clean_sibling, r)
        candidate = adversary.synthesize(item, clean_sibling, hypotheses, r)
        v_err = probe(err_sol, candidate)
        v_clean = probe(clean_sol, candidate)
        # Author-anchor (HOLE#1): re-probe BOTH solutions against the ORIGINAL author-frozen test,
        # NEVER the synthesized candidate. Frozen-test agreement with the planter's is_error claim.
        a_err = probe(err_sol, author_test)
        a_clean = probe(clean_sol, author_test)

        passed = _gate(v_err, v_clean, a_err, a_clean)
        converted = False
        if passed:
            # MANDATORY determinism re-check (eval-stats, load-bearing): re-run the CANDIDATE probe
            # ONCE MORE; a `holds` disagreement on EITHER solution rejects the candidate as flaky
            # (continue rounds). This proves IN-PROCESS stability ONLY (CF-4.4c-CONVERTER-ENV-PIN —
            # see the module docstring): it does NOT establish environment-independence (wall-clock,
            # float-repr, locale, package-version), which is a SEPARATE assumed precondition the
            # 4.4c-5 lock delivers by fingerprint-fencing the execution env. The author-anchor
            # probes are deterministic-by-construction and are NOT part of this re-check.
            v_err2 = probe(err_sol, candidate)
            v_clean2 = probe(clean_sol, candidate)
            if v_err2.holds == v_err.holds and v_clean2.holds == v_clean.holds:
                converted = True
            else:
                log.warning(
                    "conversion.flaky_candidate_rejected",
                    extra={
                        "provisional_id": item.provisional_id,
                        "round_index": r,
                        "v_err_holds": (v_err.holds, v_err2.holds),
                        "v_clean_holds": (v_clean.holds, v_clean2.holds),
                    },
                )

        artifacts.append(
            RoundArtifact(
                round_index=r,
                candidate_test=candidate,
                v_err=v_err,
                v_clean=v_clean,
                converted=converted,
            )
        )

        if converted:
            stamp = _converted_stamp(
                item, adversary_family=config.adversary_family, winning_round=r
            )
            return _rewrite_as_o(item, test_code=candidate, stamp=stamp), tuple(artifacts)

    return None, tuple(artifacts)


# ===========================================================================
# Conversion-rate reporting (eval-stats §8)
# ===========================================================================


def _conversion_rates(
    *,
    llmk_total: int,
    llmk_converted: int,
    detk_total: int,
    detk_converted: int,
) -> tuple[float, float, float]:
    """Compute ``(conv_rate_llmK, conv_rate_detK, conv_rate_all)``. ``conv_rate_llmK`` (PRIMARY)
    EXCLUDES detK from its denominator; ``conv_rate_all`` is over the whole attempted pool. An empty
    denominator yields ``0.0`` (no attempts, no rate)."""
    all_total = llmk_total + detk_total
    all_converted = llmk_converted + detk_converted
    conv_rate_llmK = llmk_converted / llmk_total if llmk_total else 0.0
    conv_rate_detK = detk_converted / detk_total if detk_total else 0.0
    conv_rate_all = all_converted / all_total if all_total else 0.0
    return conv_rate_llmK, conv_rate_detK, conv_rate_all


# ===========================================================================
# Public entry — convert_k_pool
# ===========================================================================


def _residual_audit() -> ConversionAudit:
    """The audit for a fully-residual (degraded, or all-failed-with-no-attempts) pool: empty rounds,
    zero rates. Built when a degraded panel routes the whole pool to residual."""
    return ConversionAudit(rounds={}, conv_rate_llmK=0.0, conv_rate_detK=0.0, conv_rate_all=0.0)


def convert_k_pool(
    pairs: Sequence[PlantedPair],
    singletons: Sequence[PlantedItem],
    *,
    panel: AdversaryPanel,
    adversary: CounterExampleAdversary,
    probe: OracleProbe = run_frozen_check,
    config: PanelConfig,
) -> ConversionResult:
    """Attempt to flip K-stratum error items to executable O, partitioning the pool into converted +
    residual-K (§4). The output feeds the EXISTING, unchanged ``promote_corpus``.

    Routing per item:
      - **detK items** (``is_detk``) are EXCLUDED from the converter entirely — passed straight
        through UNTOUCHED to residual; the panel is NEVER consulted for them.
      - **same-family items** (an item whose :class:`LLMPlanterStamp` family equals a panel /
        adversary family) are SKIPPED -> residual (logged) — a same-family converter is
        in-distribution with the planter.
      - **clean siblings / clean singletons** (``is_error == 0``) are never converted; a clean
        sibling rides along with its pair's converted-or-residual routing.
      - **LLM-planted K errors** are run through up to ``config.k_rounds`` conversion rounds; the
        first round clearing the two-sided gate + determinism re-check flips the item to O.

    Loud-degrade (mirrors :class:`~cogworx.eval.planting.SameFamilyFallback`): under
    ``degrade="abstain-pool"`` a config that cannot form ``len(panel_families) + 1`` families
    disjoint from the forbidden set converts NOTHING — the whole pool is routed to residual,
    ``degraded=True``, logged. (Construction already raised on a flat collision; this is the
    insufficient-disjoint-families guard.) Under ``degrade="raise"`` the shortfall raises
    :exc:`PanelFamilyCollision`.
    """
    # --- degrade gate: need (N panel + 1 adversary) DISTINCT families, all disjoint from forbidden.
    used = (*config.panel_families, config.adversary_family)
    enough_disjoint = len(set(used)) == len(used) and not (set(used) & config.forbidden_families)
    if not enough_disjoint:
        if config.degrade == "raise":
            raise PanelFamilyCollision(
                f"insufficient disjoint families: panel+adversary {used!r} vs forbidden "
                f"{sorted(config.forbidden_families)!r}"
            )
        log.error(
            "conversion.degraded_abstain_pool",
            extra={
                "panel_families": config.panel_families,
                "adversary_family": config.adversary_family,
            },
        )
        return ConversionResult(
            pairs=(),
            singletons=(),
            residual_pairs=tuple(pairs),
            residual_singletons=tuple(singletons),
            conversion_audit=_residual_audit(),
            degraded=True,
        )

    panel_family_set = set(used)

    out_pairs: list[PlantedPair] = []
    out_singletons: list[PlantedItem] = []
    residual_pairs: list[PlantedPair] = []
    residual_singletons: list[PlantedItem] = []
    rounds: dict[int, tuple[RoundArtifact, ...]] = {}

    # conversion-rate counters (over the ATTEMPTED LLM-planted-K + detK error pool).
    llmk_total = 0
    llmk_converted = 0
    detk_total = 0
    detk_converted = 0

    def _convertible(item: PlantedItem) -> bool:
        """An error item is convertible iff it is an LLM-planted K error (not detK, not same-family,
        not a deterministic stamp) AND carries an author-frozen ``test_code``. Clean items are never
        convertible.

        ``test_code is None`` -> NOT convertible (HOLE#1): the author-anchor cannot be evaluated
        without the seed's author-frozen test, so such an item is routed to residual-K (logged) and
        is EXCLUDED from the ``conv_rate_llmK`` denominator — the same posture as the singleton skip
        (it was never honestly attempted)."""
        if item.is_error == 0:
            return False
        if is_detk(item):
            return False
        if not isinstance(item.planter, LLMPlanterStamp):
            return False
        if item.test_code is None:
            return False
        return item.planter.model_family not in panel_family_set

    # --- pairs: convert the error member; the clean sibling rides along ---
    for pair in pairs:
        err = pair.error_item
        if is_detk(err):
            detk_total += 1
            residual_pairs.append(pair)
            continue
        if not _convertible(err):
            # same-family / non-LLM / no-author-test error → residual (logged).
            if (
                isinstance(err.planter, LLMPlanterStamp)
                and err.test_code is not None
                and err.planter.model_family in panel_family_set
            ):
                log.warning(
                    "conversion.same_family_skip",
                    extra={
                        "provisional_id": err.provisional_id,
                        "planter_family": err.planter.model_family,
                    },
                )
            elif isinstance(err.planter, LLMPlanterStamp) and err.test_code is None:
                # HOLE#1: an LLM-planted K error with no author-frozen test cannot be
                # author-anchored → residual, NEVER attempted, EXCLUDED from conv_rate_llmK.
                log.warning(
                    "conversion.no_author_test_skip",
                    extra={"provisional_id": err.provisional_id},
                )
            residual_pairs.append(pair)
            continue

        llmk_total += 1
        converted, artifacts = _attempt_convert(
            err, pair.clean_item, panel=panel, adversary=adversary, probe=probe, config=config
        )
        rounds[err.provisional_id] = artifacts
        if converted is not None:
            llmk_converted += 1
            out_pairs.append(PlantedPair(error_item=converted, clean_item=pair.clean_item))
        else:
            residual_pairs.append(pair)

    # --- singletons: pass through unchanged ---
    # The two-sided gate REQUIRES a co-located clean sibling (the clean solution MUST pass the
    # candidate test — the anti-laundering half). A singleton error has no clean sibling to gate
    # against, so it cannot be honestly converted: every singleton rides through to residual,
    # untouched, never attempted (matched pairs are the planter's output unit; an unpaired K error
    # stays K). detK singletons are counted in the detK denominator (reported-only); LLM-planted-K
    # singletons are NOT attempted and so are NOT counted in the conv_rate_llmK denominator.
    for item in singletons:
        if item.is_error == 1 and is_detk(item):
            detk_total += 1
        residual_singletons.append(item)

    conv_rate_llmK, conv_rate_detK, conv_rate_all = _conversion_rates(
        llmk_total=llmk_total,
        llmk_converted=llmk_converted,
        detk_total=detk_total,
        detk_converted=detk_converted,
    )

    return ConversionResult(
        pairs=tuple(out_pairs),
        singletons=tuple(out_singletons),
        residual_pairs=tuple(residual_pairs),
        residual_singletons=tuple(residual_singletons),
        conversion_audit=ConversionAudit(
            rounds=rounds,
            conv_rate_llmK=conv_rate_llmK,
            conv_rate_detK=conv_rate_detK,
            conv_rate_all=conv_rate_all,
        ),
        degraded=False,
    )


__all__ = [
    "K_ROUNDS",
    "AdversaryPanel",
    "ConversionAudit",
    "ConversionResult",
    "CounterExampleAdversary",
    "PanelConfig",
    "PanelFamilyCollision",
    "PanelHypothesis",
    "RoundArtifact",
    "convert_k_pool",
]
