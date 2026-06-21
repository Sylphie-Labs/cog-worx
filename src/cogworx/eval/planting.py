"""Pre-label planting pipeline for the Phase-4 GATE corpus (Pod 4.4c-3; plan §3.5/§3.6/§4/§7).

The planter's output unit is the **mutate-then-revert pair**: one error item (a mutated solution)
and its corrected clean sibling, co-located in the same split. :class:`PlantedItem` is the
*pre-label* record — it carries ONLY the truth the planter owns at planting time. The binding
``stratum``, ``label_source``, ``label_provenance``, ``oracle_reachable``, and ``content_hash`` are
NOT on this model: they are set downstream.

Pipeline boundary (4.4c-3 owns PRE-LABEL truth only):
  - The planter owns ``is_error`` (1 mutated / 0 clean sibling) and proposes ``candidate_stratum``.
  - ``candidate_stratum`` is a PROPOSAL, audit-only — 4.4c-4's oracle assigns the BINDING
    ``stratum`` and may overrule the proposal. NEVER read it as the binding stratum field.
  - 4.4c-4 promotes :class:`PlantedItem` -> :class:`cogworx.eval.corpus.CorpusItem`, attaching the
    label + provenance (forbidden here: ``CorpusItem``'s ``@model_validator`` requires a
    ``label_provenance`` and a mechanically-assigned ``stratum``, both FALSE at planting time).
  - ``content_hash`` is assigned at lock (4.4c-5); the stratum re-assignment + oracle
    re-verification is 4.4c-4. This schema computes, verifies, and re-runs nothing.

S1 POSTURE (the #1 red-team target — read before touching this module):
  **Authoring-time tooling. Runs offline to produce the frozen corpus; not on any live write-path.
  No journal writes, no replay. S1-clean.** The injectors here either (a) mutate a known-correct
  seed's source via :mod:`ast` (pure, no I/O), or (b) call a model ONLY through the injected
  ``Model`` seam (S4) to author K-error candidates — never on a live drive, never against a journal.
  The seed self-check runs the executable oracle's *journal-free* kernel
  (:func:`cogworx.eval._authoring.run_frozen_check`), which deliberately bypasses the F7 taint latch
  because there is no run/drive to taint — that helper lives in the private offline-authoring module
  and the live verification path MUST still go through ``CodeOracle.evaluate``. ``planting.py`` does
  zero journal I/O; a test pins no ``Journal`` / ``RunContext`` symbol is reachable from its calls.

Pure stdlib + pydantic (+ the injected ``Model`` seam for the K injector); no substrate (CANON S2,
S4). Reuses the landed seams (``OracleFrame`` / ``Thesis`` from
:mod:`cogworx.verification.contracts`; ``PlanterStamp`` / ``DifficultyMarker`` from
:mod:`cogworx.eval.corpus`; ``Model`` from :mod:`cogworx.model.base`) — it does NOT redefine them.

Contract changelog (CANON §6.1):
  - 2026-06-20 (Pod 4.4c-3): initial — PlantedItem + PlantedPair (the pre-label planter record + its
    mutate-then-revert pair unit). New module; no existing callers. Additive new types only.
  - 2026-06-20 (Pod 4.4c-3): planting pipeline — ``OPERATOR_REGIME_TABLE`` (Final) + ``Operator`` +
    ``_derive_o_regime`` (the §2.C regime core; UNKNOWN operator RAISES — the un-gameable guarantee
    is unknown-operator rejection ONLY, see ``_derive_o_regime``'s scope note); the
    deterministic ``OInjector`` (operator-based ``ast`` mutation, one regime-class per item) and the
    cross-family ``KInjector`` (oracle-blind, via the ``Model`` seam, S4); the detK collusion-probe
    builder (``DETK_PROBE_OPERATOR`` sentinel, reported-only, excludable from binding deltas);
    ``ANSWER_BEARING_FIELDS`` (Final, the antithesis-projection strip list — declared here,
    projected at 4.4d); and ``draw_splits`` (the eval-stats stratified shuffle-then-take split
    draw). Additive new public surface; no existing callers.
  - 2026-06-20 (Pod 4.4c-3, doc-only ADDITIVE): the §2.C "un-gameable" claim on
    ``_derive_o_regime`` is NARROWED to UNKNOWN-operator rejection only (architect ruling (b); a
    mis-tagged KNOWN operator is undetected — v1 safety rests on the single ``operator`` binding in
    ``OInjector.emit`` / ``build_detk_pair``). No behavior change. Tracked as CF-4.4c-REGIME-MISTAG.
  - 2026-06-20 (Pod 4.4c-3, ADDITIVE): ``draw_splits`` clean-stratum deficit is now CLAMPED to
    ``[0, n_clean_singletons]`` (eval-stats option (a)) — fixes a false ``InfeasibleSplitError`` on
    an honest corpus (1 K-pair + 1 O-pair). K/O strata keep the (unreachable-on-valid-input)
    ``_check_deficit`` defensive assert. No public-surface change.
  - 2026-06-21 (Pod 4.4c-6b fix, ADDITIVE): ``K_REGIME_TO_SPLIT_BUCKET`` (Final) +
    ``canonical_split_regime`` (pure read-through) — the regime->canon split projection closing the
    4.4c-6b integration gap: a ``KInjector`` corpus carrying RAW ``K_ERROR_KINDS`` tags
    (``spec-misread`` / ``silent-degradation``) previously crashed / monocultured ``draw_splits``
    (its step-1 cell key read the raw candidate tag, which is not a canonical split bucket).
    ``draw_splits`` now keys its step-1 cells on ``canonical_split_regime(err)``: O/detK items pass
    through unchanged (behavior-preserving for the already-passing paths), each K kind round-robins
    across TWO buckets (eval-stats A1: a 2->1 scalar map is degenerate, permanently zeroing one
    bucket). LAUNDERING WALL: ``canonical_split_regime`` is PURE — it NEVER writes the bucket back
    to ``item.error_regime``; the raw candidate tag survives verbatim into the §2.C second-author
    audit, and the bucket is allocation-only (NEVER reaches ``Cell.regime`` / the §2.B bound).
    Additive new public surface; the O/detK draw is byte-identical.
"""

from __future__ import annotations

import ast
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from random import Random
from typing import Final, Literal

from pydantic import BaseModel, ConfigDict, model_validator

from cogworx.eval.corpus import (
    DeterministicPlanterStamp,
    DifficultyMarker,
    LLMPlanterStamp,
    PlanterStamp,
)
from cogworx.model.base import ChatMessage, Model, Usage
from cogworx.verification.contracts import OracleFrame, Thesis

log = logging.getLogger(__name__)


class PlantedItem(BaseModel):
    """One pre-label planted item — the truth the planter owns, nothing downstream (plan §4).

    The binding ``stratum``, ``label_source``, ``label_provenance``, ``oracle_reachable``, and
    ``content_hash`` are deliberately ABSENT: an oracle assigns them at 4.4c-4/5. The planter owns
    ``is_error`` (ground truth: 1 mutated, 0 clean sibling) and PROPOSES ``candidate_stratum`` —
    audit-only, overruled by 4.4c-4's oracle. ``candidate_stratum`` is NEVER the binding stratum.
    """

    model_config = ConfigDict(frozen=True)

    provisional_id: int
    frame: OracleFrame
    thesis: Thesis
    test_code: str | None
    candidate_stratum: Literal["O", "K", "clean"]
    is_error: int
    planter: PlanterStamp
    error_regime: str = ""
    difficulty: DifficultyMarker
    matched_sibling_id: int | None
    split: Literal["tuning", "measurement"]


class PlantedPair(BaseModel):
    """The planter's output unit — a mutate-then-revert pair (plan §3.3/§4).

    The structural pair invariants are unconstructable to violate: the error/clean roles, mutual
    sibling references, the co-location (same-split) invariant, and the clean member's
    ``candidate_stratum == "clean"``. The split DRAW itself is test-qa-expert's planter code; this
    validator only makes an inconsistent pair fail construction.
    """

    model_config = ConfigDict(frozen=True)

    error_item: PlantedItem
    clean_item: PlantedItem

    @model_validator(mode="after")
    def _pair_invariants(self) -> PlantedPair:
        if self.error_item.is_error != 1:
            raise ValueError(
                f"error_item.is_error must be 1, got {self.error_item.is_error}"
            )
        if self.clean_item.is_error != 0:
            raise ValueError(
                f"clean_item.is_error must be 0, got {self.clean_item.is_error}"
            )
        if self.clean_item.candidate_stratum != "clean":
            raise ValueError(
                f"clean_item.candidate_stratum must be 'clean', got "
                f"{self.clean_item.candidate_stratum!r}"
            )
        if self.error_item.matched_sibling_id != self.clean_item.provisional_id:
            raise ValueError(
                f"error_item.matched_sibling_id {self.error_item.matched_sibling_id} does not "
                f"name clean_item.provisional_id {self.clean_item.provisional_id}"
            )
        if self.clean_item.matched_sibling_id != self.error_item.provisional_id:
            raise ValueError(
                f"clean_item.matched_sibling_id {self.clean_item.matched_sibling_id} does not "
                f"name error_item.provisional_id {self.error_item.provisional_id}"
            )
        if self.error_item.split != self.clean_item.split:
            raise ValueError(
                f"pair must be co-located in one split: error_item.split "
                f"{self.error_item.split!r} != clean_item.split {self.clean_item.split!r}"
            )
        return self


# ===========================================================================
# Piece 2 — OPERATOR_REGIME_TABLE (Final) + _derive_o_regime (§2.C un-gameable core)
# ===========================================================================

Operator = Literal[
    "arithmetic-swap",
    "relational-swap",
    "boundary",
    "statement-deletion",
    "constant-replacement",
    "sign-flip",
    "unit",
]
"""The deterministic-mutation operator vocabulary (the keys of ``OPERATOR_REGIME_TABLE``)."""

OPERATOR_REGIME_TABLE: Final[dict[str, str]] = {
    "arithmetic-swap": "logic-wrong",
    "relational-swap": "logic-wrong",
    "boundary": "edge-case-miss",
    "statement-deletion": "edge-case-miss",
    "constant-replacement": "off-by-semantics",
    "sign-flip": "off-by-semantics",
    "unit": "off-by-semantics",
}
"""Frozen operator -> error-regime mapping (plan §2.C). The §2.C guarantee is SCOPED to
UNKNOWN-operator rejection: an operator NOT in this table has no regime — :func:`_derive_o_regime`
RAISES rather than silently defaulting. It does NOT validate that the STAMPED operator matches the
actual AST mutation (a mis-tagged KNOWN operator is undetected here — see ``_derive_o_regime``'s
scope note and CF-4.4c-REGIME-MISTAG). Reserved control tokens (``__``-prefixed) are NEVER keys here
(asserted at import) and are stripped before any table read (the detK sentinel firewall, eval-stats
ruling)."""

# eval-stats firewall invariant: a real mutation operator can never collide with a control sentinel.
assert not any(op.startswith("__") for op in OPERATOR_REGIME_TABLE), (
    "OPERATOR_REGIME_TABLE keys must never use the reserved '__' control-token namespace"
)

DETK_PROBE_OPERATOR: Final = "__detK_collusion_probe__"
"""Reserved control sentinel marking a detK collusion-probe item (HIGH-3 / §7). It rides in a
:class:`DeterministicPlanterStamp`'s ``operators`` ALONGSIDE the genuine off-by-semantics operator;
it is a control token, NOT a mutation operator, so :func:`_derive_o_regime` strips it before the
in-table lookup (the probe's regime still derives off the REAL operator). 4.4d reads this token as
the exclusion key that keeps detK out of ``nested_bootstrap_delta``'s binding deltas (eval-stats:
the stamp survives the oracle's ``stratum`` re-assignment, which is exactly the leak it must close).
"""


def _real_operators(operators: tuple[str, ...]) -> tuple[str, ...]:
    """Strip reserved ``__``-prefixed control tokens (e.g. the detK sentinel), leaving only genuine
    mutation operators. The un-gameable in-table guarantee is enforced on THIS remainder."""
    return tuple(op for op in operators if not op.startswith("__"))


def _derive_o_regime(operators: tuple[str, ...]) -> str:
    """Map a deterministic item's ``operators`` to its error-regime (the §2.C regime core).

    SCOPE OF THE "un-gameable" GUARANTEE (architect ruling (b), claim narrowed; no AST-diff
    cross-check): this function validates STAMP -> regime, NOT mutation -> stamp. The un-gameable
    property is **UNKNOWN-operator rejection ONLY** — an operator absent from
    :data:`OPERATOR_REGIME_TABLE` RAISES (no silent default). A MIS-TAGGED KNOWN operator (source
    mutated with operator A but stamped operator B, both in the table) is NOT detected here and
    would yield B's regime. v1 safety does NOT rest on this function catching that: it rests on BOTH
    the stamp AND the mutation deriving from ONE ``operator`` binding in :meth:`OInjector.emit` and
    :func:`build_detk_pair` (so they cannot diverge there). Hand-built items or a SECOND injector
    are OUT OF SCOPE — tracked as CF-4.4c-REGIME-MISTAG (lift trigger: a 2nd deterministic injector
    lands OR items become authorable outside ``OInjector`` / ``build_detk_pair`` ⇒ add an AST-delta
    cross-check). The binding is pinned by the ``*_share_one_operator_binding`` stamp-binding tests.

    Control tokens (``__``-prefixed, e.g. :data:`DETK_PROBE_OPERATOR`) are stripped first; the
    regime derives off the genuine mutation operator(s). Then:
      * an operator not in :data:`OPERATOR_REGIME_TABLE` RAISES ``KeyError`` (NO silent default —
        the §2.C guarantee);
      * an empty remainder (only control tokens, no real mutation) RAISES ``ValueError`` (a
        deterministic item with no genuine error is a corpus defect — e.g. a bare detK sentinel);
      * >1 distinct *regime class* across the real operators RAISES ``ValueError`` (the architect's
        one-regime-class-per-item rule, so the derivation is unambiguous; operator MIX is achieved
        ACROSS items, never stacked within one).

    With one regime-class per item, the returned regime is that single class. This is the function
    4.4c-5/6 lock-asserts ``error_regime == _derive_o_regime(operators)`` against; frozen here.
    """
    real = _real_operators(operators)
    if not real:
        raise ValueError(
            f"_derive_o_regime: no genuine mutation operator in {operators!r} (only control "
            f"tokens) — a deterministic item must carry a real error"
        )
    regimes = set()
    for op in real:
        if op not in OPERATOR_REGIME_TABLE:
            raise KeyError(
                f"_derive_o_regime: operator {op!r} is not in OPERATOR_REGIME_TABLE "
                f"(no silent default — §2.C un-gameable guarantee)"
            )
        regimes.add(OPERATOR_REGIME_TABLE[op])
    if len(regimes) != 1:
        raise ValueError(
            f"_derive_o_regime: operators {real!r} span {len(regimes)} regime classes {regimes!r}; "
            f"exactly one regime-class per item is required (architect ruling)"
        )
    return regimes.pop()


# ===========================================================================
# Piece 1 — Deterministic-mutation O injector (operator-based ``ast`` mutation, §3.6)
# ===========================================================================

# Each operator maps to an AST node-transform. ONE regime-bearing operator class is applied per
# emitted item (architect ruling) so _derive_o_regime is unambiguous. Mutations are syntactic and
# oracle-checkable: the mutated source must FAIL the seed's frozen test while the original passes.

_ARITH_SWAP: Final[dict[type[ast.operator], type[ast.operator]]] = {
    ast.Add: ast.Sub,
    ast.Sub: ast.Add,
    ast.Mult: ast.FloorDiv,
    ast.Div: ast.Mult,
    ast.FloorDiv: ast.Mult,
    ast.Mod: ast.Mult,
}
_REL_SWAP: Final[dict[type[ast.cmpop], type[ast.cmpop]]] = {
    ast.Lt: ast.LtE,
    ast.LtE: ast.Lt,
    ast.Gt: ast.GtE,
    ast.GtE: ast.Gt,
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
}


class MutationFailed(ValueError):
    """The requested operator found no applicable site in the seed source (the seed shape does not
    support that mutation). Surfaced so the corpus author re-routes the operator to a fitting seed —
    NOT a silent no-op (which would emit a "mutated" item identical to its clean sibling)."""


class _FirstSiteMutator(ast.NodeTransformer):
    """Apply exactly ONE operator-driven mutation at the first applicable site (deterministic:
    leftmost-outermost by ``ast.walk`` order). Records whether a site was hit so the injector can
    raise :exc:`MutationFailed` on a no-op."""

    def __init__(self, operator: Operator) -> None:
        self.operator = operator
        self.applied = False

    def _binop(self, node: ast.BinOp) -> ast.AST:
        if not self.applied and type(node.op) in _ARITH_SWAP:
            node.op = _ARITH_SWAP[type(node.op)]()
            self.applied = True
        return node

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if self.operator == "arithmetic-swap":
            return self._binop(node)
        return node

    def visit_Compare(self, node: ast.Compare) -> ast.AST:
        self.generic_visit(node)
        if self.operator in ("relational-swap", "boundary") and not self.applied and node.ops:
            op0 = type(node.ops[0])
            if op0 in _REL_SWAP:
                node.ops[0] = _REL_SWAP[op0]()
                self.applied = True
        return node

    def visit_Constant(self, node: ast.Constant) -> ast.AST:
        if self.applied:
            return node
        value = node.value
        # bool is an int subclass — exclude it so a True/False literal is never "mutated".
        if isinstance(value, bool):
            return node
        if self.operator == "constant-replacement" and isinstance(value, int):
            node.value = value + 1
            self.applied = True
        elif self.operator == "sign-flip" and isinstance(value, (int, float)):
            node.value = -value
            self.applied = True
        elif self.operator == "unit" and isinstance(value, (int, float)):
            # A unit error: scale by a wrong constant factor (e.g. seconds<->minutes).
            node.value = type(value)(value * 60)
            self.applied = True
        return node

    def visit_UnaryOp(self, node: ast.UnaryOp) -> ast.AST:
        self.generic_visit(node)
        if self.operator == "sign-flip" and not self.applied and isinstance(node.op, ast.USub):
            # -x  ->  x  (drop the negation)
            return node.operand
        return node


def _delete_first_statement(tree: ast.Module) -> bool:
    """statement-deletion: drop the first deletable statement inside the first function body
    (leaving >=1 statement so the source still parses/imports). Returns whether a deletion fired."""
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and len(node.body) >= 2:
            # Prefer deleting a non-return statement; never empty the body.
            for i, stmt in enumerate(node.body):
                if not isinstance(stmt, ast.Return) and len(node.body) >= 2:
                    del node.body[i]
                    return True
    return False


def mutate_source(source: str, operator: Operator) -> str:
    """Apply ONE ``operator`` to ``source`` via ``ast`` and return the mutated source.

    Deterministic (first applicable site). Raises :exc:`MutationFailed` if the operator has no
    applicable site (a silent no-op would emit a fake error identical to its clean sibling — the
    poison the §3.6 realism check exists to catch). Pure: no I/O, no model call.
    """
    tree = ast.parse(source)
    if operator == "statement-deletion":
        if not _delete_first_statement(tree):
            raise MutationFailed(
                f"operator 'statement-deletion' found no deletable statement in source:\n{source}"
            )
    else:
        mutator = _FirstSiteMutator(operator)
        tree = mutator.visit(tree)
        if not mutator.applied:
            raise MutationFailed(
                f"operator {operator!r} found no applicable site in source:\n{source}"
            )
    ast.fix_missing_locations(tree)
    return ast.unparse(tree)


@dataclass(frozen=True)
class Seed:
    """One known-correct seed problem: a frame, a correct thesis, and the author-frozen test code
    (plan seed-corpus §). The O injector mutates ``thesis.proposed_solution``; the clean sibling is
    the un-mutated original (mutate-then-revert)."""

    seed_id: int
    frame: OracleFrame
    thesis: Thesis
    test_code: str


def _difficulty(seed: Seed, *, is_matched_sibling: bool) -> DifficultyMarker:
    """A reported-only difficulty marker derived from surface complexity (AST node count). Both
    pair members share the same difficulty (the sibling-matching axis, §3.3)."""
    nodes = sum(1 for _ in ast.walk(ast.parse(seed.thesis.proposed_solution)))
    band: Literal["easy", "medium", "hard"] = (
        "easy" if nodes < 25 else "medium" if nodes < 60 else "hard"
    )
    return DifficultyMarker(
        planted_difficulty=band, surface_complexity=nodes, is_matched_sibling=is_matched_sibling
    )


class OInjector:
    """Deterministic-mutation O injector (operator-based ``ast`` mutation, §3.6) — NO model calls.

    Mutates a known-correct seed's ``proposed_solution`` with ONE regime-bearing operator (architect
    ruling), pairs the mutated item with its un-mutated original as the clean sibling, and emits a
    :class:`PlantedPair`. ``error_regime`` is set from :func:`_derive_o_regime` over the single
    applied operator. The operator MIX (§3.6 realism cap) is achieved ACROSS items by the corpus
    author's operator schedule, never stacked within one item.
    """

    def emit(
        self, seed: Seed, operator: Operator, *, error_id: int, clean_id: int
    ) -> PlantedPair:
        """Emit one O pair from ``seed`` under ``operator``. ``error_id``/``clean_id`` are the
        author-assigned provisional ids (the pair's mutual sibling references). Raises
        :exc:`MutationFailed` if ``operator`` has no site in this seed (re-route it to a fitting
        seed). The split is left at the planter default ``"tuning"`` — :func:`draw_splits` assigns
        the binding split later and co-locates both members."""
        mutated_solution = mutate_source(seed.thesis.proposed_solution, operator)
        regime = _derive_o_regime((operator,))
        stamp = DeterministicPlanterStamp(operators=(operator,))
        error_item = PlantedItem(
            provisional_id=error_id,
            frame=seed.frame,
            thesis=seed.thesis.model_copy(update={"proposed_solution": mutated_solution}),
            test_code=seed.test_code,
            candidate_stratum="O",
            is_error=1,
            planter=stamp,
            error_regime=regime,
            difficulty=_difficulty(seed, is_matched_sibling=True),
            matched_sibling_id=clean_id,
            split="tuning",
        )
        clean_item = PlantedItem(
            provisional_id=clean_id,
            frame=seed.frame,
            thesis=seed.thesis,
            test_code=seed.test_code,
            candidate_stratum="clean",
            is_error=0,
            planter=stamp,
            error_regime="",
            difficulty=_difficulty(seed, is_matched_sibling=True),
            matched_sibling_id=error_id,
            split="tuning",
        )
        return PlantedPair(error_item=error_item, clean_item=clean_item)


# ===========================================================================
# Piece 4 — Deterministic-K collusion-probe sub-pool (detK, HIGH-3 / §7)
# ===========================================================================

DETK_MIN_POOL: Final = 30
"""eval-stats floor: detK is an ADDITIONAL pool of >=30 items (never carved from the binding 56
measurement K — carving would drop binding K below the n=30 power floor and bust the gate). Below 30
the positive collusion signature is unreliable."""


def build_detk_pair(
    seed: Seed, operator: Operator, *, error_id: int, clean_id: int
) -> PlantedPair:
    """Build one detK collusion-probe pair (§7): an off-by-semantics deterministic mutation on an
    oracle-blind item, stamped distinguishably so 4.4d/4.4e exclude it from binding deltas.

    The mutation is GENUINE (a real off-by-semantics error the arms must catch) — only the stamp
    differs: the :class:`DeterministicPlanterStamp` carries :data:`DETK_PROBE_OPERATOR` next to the
    real operator, and ``candidate_stratum`` is ``"K"`` (oracle-blind, K-shaped; audit-only). The
    regime still derives off the REAL operator (off-by-semantics). All detK items are
    ``split="measurement"`` (eval-stats), reported-only — the stamp token is the exclusion key that
    survives the oracle's ``stratum`` re-assignment at 4.4c-4.
    """
    if OPERATOR_REGIME_TABLE.get(operator) != "off-by-semantics":
        raise ValueError(
            f"detK requires an off-by-semantics operator, got {operator!r} "
            f"(regime {OPERATOR_REGIME_TABLE.get(operator)!r})"
        )
    mutated_solution = mutate_source(seed.thesis.proposed_solution, operator)
    stamp = DeterministicPlanterStamp(operators=(operator, DETK_PROBE_OPERATOR))
    regime = _derive_o_regime(stamp.operators)  # strips the sentinel -> off-by-semantics
    error_item = PlantedItem(
        provisional_id=error_id,
        frame=seed.frame,
        thesis=seed.thesis.model_copy(update={"proposed_solution": mutated_solution}),
        test_code=seed.test_code,
        candidate_stratum="K",
        is_error=1,
        planter=stamp,
        error_regime=regime,
        difficulty=_difficulty(seed, is_matched_sibling=True),
        matched_sibling_id=clean_id,
        split="measurement",
    )
    clean_item = PlantedItem(
        provisional_id=clean_id,
        frame=seed.frame,
        thesis=seed.thesis,
        test_code=seed.test_code,
        candidate_stratum="clean",
        is_error=0,
        planter=stamp,
        error_regime="",
        difficulty=_difficulty(seed, is_matched_sibling=True),
        matched_sibling_id=error_id,
        split="measurement",
    )
    return PlantedPair(error_item=error_item, clean_item=clean_item)


def is_detk(item: PlantedItem) -> bool:
    """The detK exclusion predicate (eval-stats firewall): True iff ``item`` is a deterministic
    collusion probe. 4.4d uses this to keep detK items OUT of the ``Cell`` list that
    ``nested_bootstrap_delta`` indexes — they never enter ``error_pools["K"]`` despite their binding
    ``stratum="K"``. Anchored on the immutable planter stamp, NOT on stratum/regime."""
    return (
        isinstance(item.planter, DeterministicPlanterStamp)
        and DETK_PROBE_OPERATOR in item.planter.operators
    )


# ===========================================================================
# Piece 3 — Cross-family LLM K injector (§3.5) — talks to models ONLY via the Model seam (S4)
# ===========================================================================

MAX_PLANTING_CALLS: Final = 600
"""S11 ceiling on the total number of ``Model.complete`` calls a planting RUN may make (plan §3.5).
The run entry point sums :class:`Usage` from each ``ModelResponse`` and logs the total; the real
cross-family run is DEFERRED (needs 2nd-family creds; Jim ratifies the USD ceiling). This pipeline +
its stub-Model tests exist NOW; no live calls are made here."""

MAX_RETRIES_PER_ITEM: Final = 3
"""S11 per-item retry cap (plan §3.5). A K candidate that fails to parse into a well-formed
``PlantedItem`` is retried at most this many times before the item is skipped (logged), so one bad
seed cannot burn the whole :data:`MAX_PLANTING_CALLS` budget."""

# K errors are oracle-blind by construction (no operator signature — that is WHY they are K).
K_ERROR_KINDS: Final = ("spec-misread", "silent-degradation")


class SameFamilyFallback(RuntimeError):
    """CF-4.4c-PLANTER loud fallback: the injected planting ``Model`` shares a family with the
    thesis or antithesis arm, which would make a planted K error in-distribution with an arm and
    defeat the §3.5 cross-family detectability requirement. This is a LOUD, logged failure — never a
    silent same-family stamp."""


class KInjector:
    """Cross-family LLM K injector (§3.5) — plants oracle-blind K errors via the ``Model`` seam.

    The planting model is taken by DEPENDENCY INJECTION (a ``Model``); this class NEVER imports a
    concrete provider (provider is the call-site's choice, S4). It stamps :class:`LLMPlanterStamp`
    from the injected model's identity and sets a CANDIDATE ``error_regime`` (second-author-audited
    at 4.4c-5, NOT operator-derived) with ``candidate_stratum="K"``.

    CF-4.4c-PLANTER (loud fallback): the planting family MUST differ from both the thesis and the
    antithesis arm family. A same-family config raises :exc:`SameFamilyFallback` at build time
    and logs — never a silent same-family stamp (the §3.5 detectability requirement).

    S11: the run entry point bounds total calls + per-item retries (see :data:`MAX_PLANTING_CALLS` /
    :data:`MAX_RETRIES_PER_ITEM`) and sums ``Usage``. The constructor wires the build-time family
    check + log; live calls are DEFERRED.
    """

    def __init__(
        self,
        model: Model,
        *,
        model_family: str,
        model_id: str,
        thesis_family: str,
        antithesis_family: str,
    ) -> None:
        self._model = model
        self._model_family = model_family
        self._model_id = model_id
        # CF-4.4c-PLANTER build-time check — loud, logged, never silent.
        forbidden = {thesis_family, antithesis_family}
        if model_family in forbidden:
            log.error(
                "planting.k_injector.same_family_fallback",
                extra={
                    "planting_family": model_family,
                    "thesis_family": thesis_family,
                    "antithesis_family": antithesis_family,
                },
            )
            raise SameFamilyFallback(
                f"planting family {model_family!r} collides with an arm family "
                f"{{thesis={thesis_family!r}, antithesis={antithesis_family!r}}}: a same-family "
                f"planted K error is in-distribution with an arm and defeats §3.5 detectability"
            )
        self._stamp = LLMPlanterStamp(model_family=model_family, model_id=model_id)

    @property
    def stamp(self) -> LLMPlanterStamp:
        return self._stamp

    @staticmethod
    def _prompt(seed: Seed, error_kind: str) -> list[ChatMessage]:
        return [
            ChatMessage(
                role="system",
                content=(
                    "You author a subtly-wrong variant of a correct solution. The variant must be "
                    "oracle-blind: no mechanical operator signature, a plausible-looking but wrong "
                    f"solution exhibiting a {error_kind} error. Return only the variant solution."
                ),
            ),
            ChatMessage(role="user", content=seed.thesis.proposed_solution),
        ]

    async def plant(
        self, seed: Seed, error_kind: str, *, error_id: int, clean_id: int
    ) -> tuple[PlantedPair, Usage]:
        """Plant ONE oracle-blind K pair from ``seed`` via the injected ``Model`` (S4). Returns the
        pair + the call's :class:`Usage` (the run entry point sums these for the S11 ceiling).

        The clean sibling is the un-mutated original (mutate-then-revert). ``error_regime`` on the
        error member is a CANDIDATE (the ``error_kind``, second-author-audited at 4.4c-5 — NOT
        operator-derived; K errors have no operator signature)."""
        if error_kind not in K_ERROR_KINDS:
            raise ValueError(
                f"unknown K error_kind {error_kind!r}; expected one of {K_ERROR_KINDS}"
            )
        response = await self._model.complete(messages=self._prompt(seed, error_kind))
        wrong_solution = response.text
        if not wrong_solution or not wrong_solution.strip():
            raise ValueError(
                f"K injector: model returned empty text for seed {seed.seed_id} "
                f"(error_kind={error_kind!r}) — retry at the run level (S11 cap)"
            )
        error_item = PlantedItem(
            provisional_id=error_id,
            frame=seed.frame,
            thesis=seed.thesis.model_copy(update={"proposed_solution": wrong_solution}),
            test_code=seed.test_code,
            candidate_stratum="K",
            is_error=1,
            planter=self._stamp,
            error_regime=error_kind,  # CANDIDATE — audited at 4.4c-5, never operator-derived
            difficulty=_difficulty(seed, is_matched_sibling=True),
            matched_sibling_id=clean_id,
            split="tuning",
        )
        clean_item = PlantedItem(
            provisional_id=clean_id,
            frame=seed.frame,
            thesis=seed.thesis,
            test_code=seed.test_code,
            candidate_stratum="clean",
            is_error=0,
            planter=self._stamp,
            error_regime="",
            difficulty=_difficulty(seed, is_matched_sibling=True),
            matched_sibling_id=error_id,
            split="tuning",
        )
        return PlantedPair(error_item=error_item, clean_item=clean_item), response.usage


# ===========================================================================
# Piece 5 — ANSWER_BEARING_FIELDS (Final) + strip-field list
# (port tess judge_calibration.py:161-168)
# ===========================================================================

ANSWER_BEARING_FIELDS: Final = ("thesis.experiment_design",)
"""The canonical answer-bearing field the antithesis input projection must HIDE (the C'-vs-D
info-diet axis). Ported from tess ``judge_calibration.py:161-168`` (tess strips exactly
``experiment_design``). ``test_code`` is held on the CorpusItem but is NEVER in the arm-input
projection (it names the planted error), so it is not listed here.

4.4c-3 DECLARES this strip list as pure DATA. 4.4d performs the actual projection
``thesis.model_copy(update={experiment_design: hidden})`` — this module projects nothing."""


# ===========================================================================
# The split draw — eval-stats stratified shuffle-then-take (implemented VERBATIM, deterministic)
# ===========================================================================

SPLIT_SEED: Final = 0xC067_0440
"""The one fixed split-draw seed (frozen — it folds into ``content_hash`` at lock, 4.4c-5). Every
``Random`` in :func:`draw_splits` is seeded from ``hash((SPLIT_SEED, ...))`` so the draw is fully
deterministic and reproducible from this constant alone."""

SPLIT_TARGET_MEASUREMENT: Final = 0.70
"""TARGET measurement fraction (plan split-draw §)."""

_REGIMES: Final = ("logic-wrong", "edge-case-miss", "off-by-semantics")

K_REGIME_TO_SPLIT_BUCKET: Final[dict[str, tuple[str, ...]]] = {
    "spec-misread": ("logic-wrong", "edge-case-miss"),
    "silent-degradation": ("off-by-semantics", "edge-case-miss"),
}
"""Frozen projection of the 2 oracle-blind K error kinds (:data:`K_ERROR_KINDS`) onto the canonical
split buckets (:data:`_REGIMES`) — ALLOCATION-ONLY (the §2.C split-draw cell axis), NEVER a label.

Why each K kind maps across TWO buckets (eval-stats A1): a scalar 2-input -> 1-bucket map is
DEGENERATE by cardinality. Two K kinds onto two distinct singletons can reach at most 2 of the 3
canonical buckets, so one bucket (here ``edge-case-miss``) is PERMANENTLY zeroed — a silent 2-regime
K monoculture that biases the K split marginal. Mapping each kind across two buckets makes the union
image cover all 3 ``_REGIMES`` (``edge-case-miss`` is shared), so the deterministic round-robin in
:func:`canonical_split_regime` keeps every K-split bucket reachable.

This is the SPLIT-DRAW allocation axis ONLY. It is NEVER written back to ``item.error_regime`` and
MUST NOT reach ``Cell.regime`` or the §2.B contribution bound: the raw ``spec-misread`` /
``silent-degradation`` candidate tag survives verbatim into the §2.C second-author audit. eval-stats
targets (A1)."""

# Firewall: the projection's domain is EXACTLY the K kinds, and its image lies inside the canonical
# split buckets — so a drift in either vocabulary trips at import, not silently in the draw.
assert set(K_REGIME_TO_SPLIT_BUCKET) == set(K_ERROR_KINDS), (
    "K_REGIME_TO_SPLIT_BUCKET domain must be exactly K_ERROR_KINDS"
)
assert all(
    bucket in _REGIMES
    for buckets in K_REGIME_TO_SPLIT_BUCKET.values()
    for bucket in buckets
), "K_REGIME_TO_SPLIT_BUCKET image must lie inside the canonical split buckets (_REGIMES)"
assert {b for buckets in K_REGIME_TO_SPLIT_BUCKET.values() for b in buckets} == set(_REGIMES), (
    "K_REGIME_TO_SPLIT_BUCKET image must COVER all 3 canonical buckets (no permanent monoculture)"
)


def canonical_split_regime(item: PlantedItem) -> str:
    """Project ``item.error_regime`` onto a canonical SPLIT bucket (allocation-only; PURE read).

    The split-draw cells (:func:`draw_splits` step 1) are keyed on the canonical :data:`_REGIMES`.
    Deterministic-O and detK items already carry a canonical ``error_regime`` (operator-derived via
    :func:`_derive_o_regime`), so they pass through UNCHANGED — behavior-preserving for the O/detK
    paths. A cross-family LLM-K item carries a CANDIDATE ``error_regime`` in :data:`K_ERROR_KINDS`
    (``spec-misread`` / ``silent-degradation``), which is NOT a canonical split bucket — left raw
    it crashes / monocultures the draw (the 4.4c-6b integration gap). Such an item is projected onto
    one of its kind's two buckets (:data:`K_REGIME_TO_SPLIT_BUCKET`) by a DETERMINISTIC round-robin
    keyed on the item's stable ``provisional_id`` and folded into :data:`SPLIT_SEED`, so the
    assignment is reproducible and keeps all 3 K-split buckets reachable.

    LAUNDERING WALL (the red-team-critical invariant): this is a PURE function — it NEVER mutates
    ``item.error_regime``. The candidate ``spec-misread`` / ``silent-degradation`` tag MUST survive
    verbatim into the §2.C second-author audit (labeling reads ``item.error_regime`` as the
    candidate). The returned bucket is consumed ONLY by the split allocation; it must NEVER reach
    ``Cell.regime`` or the §2.B contribution bound.

    LOUD on the unknown (mirrors :func:`_derive_o_regime`): an ``error_regime`` that is neither a
    canonical bucket nor a known K kind RAISES :exc:`ValueError` — no silent default.
    """
    regime = item.error_regime
    if regime in _REGIMES:
        return regime  # already canonical (deterministic-O / detK) — pass through unchanged
    if regime in K_REGIME_TO_SPLIT_BUCKET:
        buckets = K_REGIME_TO_SPLIT_BUCKET[regime]
        # Deterministic round-robin keyed on the stable id, folded into SPLIT_SEED so the bucket
        # choice is reproducible from the frozen seed alone (no mutation of item.error_regime).
        idx = hash((SPLIT_SEED, "k_split_bucket", regime, item.provisional_id)) % len(buckets)
        return buckets[idx]
    raise ValueError(
        f"canonical_split_regime: error_regime {regime!r} (item {item.provisional_id}) is neither "
        f"a canonical split bucket {_REGIMES!r} nor a known K kind "
        f"{tuple(K_REGIME_TO_SPLIT_BUCKET)!r} — no silent default (mirrors _derive_o_regime's "
        f"loud-on-unknown discipline)"
    )


def _hamilton_apportion(total: int, weights: Sequence[float], keys: Sequence[str]) -> list[int]:
    """Largest-remainder (Hamilton) apportionment of ``total`` across cells with the given
    ``weights``, ties broken by lowest ``keys`` sort order (frozen). Guarantees the per-cell
    allocations sum EXACTLY to ``total`` (0.7*16=11.2 over 5 cells -> {12,11,11,11,11}=56, never
    5*11=55). ``weights`` need not be normalized; they are the pool sizes scaled by 0.70."""
    if total == 0:
        return [0] * len(weights)
    floors = [int(w) for w in weights]
    remainders = [w - f for w, f in zip(weights, floors, strict=True)]
    deficit = total - sum(floors)
    # Distribute the remaining `deficit` seats to the largest remainders; ties -> lowest key order.
    order = sorted(range(len(weights)), key=lambda i: (-remainders[i], keys[i]))
    alloc = list(floors)
    for i in order[:deficit]:
        alloc[i] += 1
    return alloc


class InfeasibleSplitError(ValueError):
    """An UNREACHABLE-BY-CONSTRUCTION tripwire (split-draw step 2, K/O strata only): a K or O
    stratum's measurement deficit is negative or exceeds its available singletons. Raised here;
    re-asserted at the 4.4c-5 lock.

    SCOPE (eval-stats re-ruling, post-red-team): this guard fires ONLY on an INTERNALLY-INCONSISTENT
    artifact (a mis-count where placement exceeds stratum size), NEVER on an imbalanced-but-VALID
    corpus. K and O are SINGLE-rounding strata (measurement count is one ``round(target*size)``
    over a single pool), so ``0 <= deficit <= |singletons|`` is provably reachable on any
    schema-valid K/O input; eval-stats verified this. It is therefore cheap defensive insurance
    against an internal drift, NOT a refuse-to-lock.

    The earlier false-refuse on an HONEST corpus (1 K-pair + 1 O-pair, no clean singletons) came
    from applying this guard to the CLEAN stratum, which is a UNION of K-siblings + O-siblings
    (placed by per-stratum-ROUNDED inheritance: two independent round-ups need not equal one round
    of the sum), so clean's deficit can fall outside ``[0, |singletons|]`` by +/-1. Clean is now
    CLAMPED (eval-stats option (a)) and never routed through this guard. The check is factored into
    :func:`_check_deficit` so it is directly unit-testable on an infeasible ``(size, placed,
    singletons)`` triple."""


def _check_deficit(stratum: str, deficit: int, n_singletons: int) -> None:
    """Raise :exc:`InfeasibleSplitError` unless ``0 <= deficit <= n_singletons`` (the step-2
    feasibility condition for the K and O strata ONLY — clean is clamped, never checked).

    On a schema-valid K/O corpus this guard is unreachable (single rounding keeps the deficit in
    range); it fires only on an internally-inconsistent placement (placement exceeds stratum size).
    Factored out so the guard is unit-testable without constructing an infeasible corpus."""
    if not (0 <= deficit <= n_singletons):
        raise InfeasibleSplitError(
            f"stratum {stratum!r}: measurement deficit {deficit} is outside "
            f"[0, {n_singletons}] singletons (refuse-to-lock; re-asserted at 4.4c-5)"
        )


def _split_seeded_rng(*parts: object) -> Random:
    """A ``Random`` seeded deterministically from ``SPLIT_SEED`` + the given parts."""
    return Random(hash((SPLIT_SEED, *parts)))


def draw_splits(
    pairs: Sequence[PlantedPair],
    singletons: Sequence[PlantedItem],
    *,
    target: float = SPLIT_TARGET_MEASUREMENT,
) -> dict[int, Literal["tuning", "measurement"]]:
    """Assign each item a split via the eval-stats stratified shuffle-then-take draw (verbatim).

    Returns ``{provisional_id: split}`` for every item across ``pairs`` (both members) and
    ``singletons``. NOT per-pair Bernoulli coins (which blow the regime marginals); per
    ``(stratum, regime)`` error-cell, shuffle-then-take a Hamilton-apportioned count.

    Step 1 — per ``(stratum in {K,O}, regime)`` error-cell:
      * ``pairs_c`` = pairs whose ERROR member is in the cell;
      * ``n_meas_c`` = Hamilton-apportioned share of ``round(target * stratum_total)`` so the regime
        cells sum to the stratum target (not the rounded-per-cell sum);
      * stable-sort ``pairs_c`` by error ``provisional_id``, then ``Random(hash(SPLIT_SEED, stratum,
        regime)).shuffle``; first ``n_meas_c`` pairs -> measurement, rest -> tuning;
      * BOTH pair members inherit the same split (co-location; the clean sibling follows the error).

    Step 2 — singleton top-up per stratum ``S in {K,O,clean}``:
      * ``deficit_S`` = ``round(target*|S|)`` minus the measurement count already placed in ``S`` by
        step 1.
      * For K and O (single-rounding strata): ``0 <= deficit_S <= |singletons_S|`` is asserted
        (:exc:`InfeasibleSplitError`) — but the assert is unreachable on a schema-valid corpus and
        is defensive insurance against an internal mis-count, NOT a refuse-to-lock on an imbalanced
        corpus (eval-stats re-ruling).
      * For clean (a UNION of K-siblings + O-siblings placed by per-stratum-ROUNDED inheritance, so
        two independent round-ups need not equal one round of the sum): the raw deficit is CLAMPED
        to ``[0, |singletons_clean|]`` (eval-stats option (a) + both-directions clamp), NOT checked.
        Clean's marginal is "0.70 within +/-1 item (pairing-induced rounding), exact at 80/80."
      * stable-sort singletons by ``provisional_id``, ``Random(hash(SPLIT_SEED,"singleton",S))
        .shuffle``; first ``deficit_S`` -> measurement, rest -> tuning. ``clean`` has NO regime
        sub-strata (siblings carry ``error_regime=""``) — its top-up is NOT stratified by regime.
    """
    assignment: dict[int, Literal["tuning", "measurement"]] = {}

    # --- Step 1: error-cell stratified shuffle-then-take over PAIRS ---
    for stratum in ("K", "O"):
        cell_pairs: dict[str, list[PlantedPair]] = {r: [] for r in _REGIMES}
        for pair in pairs:
            err = pair.error_item
            if err.candidate_stratum == stratum:
                # ALLOCATION-ONLY canonical bucket: K candidate tags (spec-misread /
                # silent-degradation) are projected onto a canonical split bucket here; O/detK pass
                # through unchanged. The raw err.error_regime is NEVER mutated (laundering wall) —
                # the candidate tag survives verbatim into the §2.C second-author audit.
                bucket = canonical_split_regime(err)
                cell_pairs[bucket].append(pair)
        stratum_total = sum(len(v) for v in cell_pairs.values())
        n_meas_stratum = round(target * stratum_total)
        weights = [target * len(cell_pairs[r]) for r in _REGIMES]
        allocs = _hamilton_apportion(n_meas_stratum, weights, _REGIMES)
        for regime, n_meas_c in zip(_REGIMES, allocs, strict=True):
            cps = sorted(cell_pairs[regime], key=lambda p: p.error_item.provisional_id)
            _split_seeded_rng(stratum, regime).shuffle(cps)
            for i, pair in enumerate(cps):
                split: Literal["tuning", "measurement"] = (
                    "measurement" if i < n_meas_c else "tuning"
                )
                assignment[pair.error_item.provisional_id] = split
                assignment[pair.clean_item.provisional_id] = split

    # --- Step 2: singleton top-up per stratum (clean NOT regime-stratified) ---
    by_stratum: dict[str, list[PlantedItem]] = {"K": [], "O": [], "clean": []}
    for item in singletons:
        by_stratum[item.candidate_stratum].append(item)
    # Measurement already placed per stratum by step 1 (from the paired items).
    placed_meas: dict[str, int] = {"K": 0, "O": 0, "clean": 0}
    for pair in pairs:
        if assignment.get(pair.error_item.provisional_id) == "measurement":
            placed_meas[pair.error_item.candidate_stratum] += 1
            placed_meas[pair.clean_item.candidate_stratum] += 1

    for stratum in ("K", "O", "clean"):
        sgs = by_stratum[stratum]
        paired_in_stratum = sum(
            1
            for pair in pairs
            for member in (pair.error_item, pair.clean_item)
            if member.candidate_stratum == stratum
        )
        stratum_size = len(sgs) + paired_in_stratum
        raw_deficit = round(target * stratum_size) - placed_meas[stratum]
        if stratum == "clean":
            # clean is a UNION of K-siblings + O-siblings placed by per-stratum-ROUNDED inheritance;
            # two independent round-ups need not equal one round of the combined sum, so a single
            # round(target*size) deficit can be ±1 off the count the paired top-up supplied. Clean's
            # marginal is "0.70 within ±1 item (pairing-induced rounding), exact at 80/80."
            #
            # DEFENSIVE, NOT LOAD-BEARING: the placement loop below is range-saturating in
            # BOTH directions — `i < deficit` promotes nobody for deficit<=0 and can't exceed
            # len(ordered) for deficit>len(ordered) — so clamped and raw deficit yield the
            # IDENTICAL assignment on every input (verified: reverting the clamp leaves all
            # tests green; eval-stats 4.4c-3 re-ruling, option B). The clamp is kept only as an
            # explicit, self-documenting bound matching the K/O `_check_deficit` defensive
            # posture; it does NOT absorb the rounding skew (the empty-loop/saturation does).
            # Clean is not routed through `_check_deficit` because its ±1 round-up skew is a
            # VALID corpus, not an internal mis-count.
            deficit = max(0, min(raw_deficit, len(sgs)))
        else:
            # K and O are single-rounding strata: 0 <= deficit <= n_singletons is provably reachable
            # only on a schema-VALID corpus, so _check_deficit stays as cheap defensive insurance
            # against an internal mis-count (placement exceeds size), NOT a refuse-to-lock on an
            # imbalanced-but-valid corpus.
            deficit = raw_deficit
            _check_deficit(stratum, deficit, len(sgs))
        ordered = sorted(sgs, key=lambda it: it.provisional_id)
        _split_seeded_rng("singleton", stratum).shuffle(ordered)
        for i, item in enumerate(ordered):
            assignment[item.provisional_id] = "measurement" if i < deficit else "tuning"

    return assignment


__all__ = [
    "ANSWER_BEARING_FIELDS",
    "DETK_MIN_POOL",
    "DETK_PROBE_OPERATOR",
    "K_ERROR_KINDS",
    "K_REGIME_TO_SPLIT_BUCKET",
    "MAX_PLANTING_CALLS",
    "MAX_RETRIES_PER_ITEM",
    "OPERATOR_REGIME_TABLE",
    "SPLIT_SEED",
    "SPLIT_TARGET_MEASUREMENT",
    "InfeasibleSplitError",
    "KInjector",
    "MutationFailed",
    "OInjector",
    "Operator",
    "PlantedItem",
    "PlantedPair",
    "SameFamilyFallback",
    "Seed",
    "build_detk_pair",
    "canonical_split_regime",
    "draw_splits",
    "is_detk",
    "mutate_source",
]
