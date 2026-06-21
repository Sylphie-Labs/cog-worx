"""Frozen corpus-item schema for the Phase-4 GATE eval corpus (Pod 4.4c-2; plan §4).

The :class:`CorpusItem` is the *ground-truth-about-the-item* record the gate corpus is built from
— the problem frame, the proposed thesis, the author-frozen test, the label and its provenance, the
mechanically-assigned stratum, and the planting/independence provenance. It carries **no run /
behavior fields** — those are :class:`cogworx.eval.youden.Cell` fields produced at 4.4d. Every model
here is frozen (hashable, deterministic) and pure stdlib + pydantic: this is eval-time machinery,
no model calls and no numpy/scipy (CANON S2).

The label provenance and the planter are **discriminated unions**, not optional-field soup: an
oracle label carries the oracle invocation result, a human label carries its adjudications; a
deterministic planter carries its mutation operators, an LLM planter carries its model identity.
The discriminator keys (``kind`` / ``injector_kind``) are the routing tags.

Landed-seam bindings (load-bearing — verified at build):
  - :attr:`CorpusItem.frame` / :attr:`CorpusItem.thesis` ARE the landed
    :class:`cogworx.verification.contracts.OracleFrame` / ``Thesis`` (``contracts.py:36,50``).
  - :attr:`CorpusItem.error_regime` mirrors :attr:`cogworx.eval.youden.Cell.regime` (``str = ""``,
    reported-only) and feeds it at scoring time.
  - :attr:`OracleLabelProvenance.test_provenance` deliberately NARROWS
    :attr:`cogworx.verification.contracts.Verdict.test_provenance`
    (``Literal["thesis","frozen","n/a"]``, ``contracts.py:102``) to ``Literal["frozen"]`` — the
    §13.4 #5 self-test-laundering guard: an oracle label is only trustworthy from the author-frozen
    test corpus, never the thesis-supplied test.

Contract changelog (CANON §6.1):
  - 2026-06-20 (Pod 4.4c-2): initial — CorpusItem + LabelProvenance / PlanterStamp discriminated
    unions + DifficultyMarker + Adjudication. New module; no existing callers. ``content_hash`` is
    carried opaque (default ``""``, "not yet locked"); lock-time (4.4c-5) populates it.
  - 2026-06-21 (Pod 4.4c-3.5, ADDITIVE): ``ConvertedPlanterStamp`` (``injector_kind="converted"``)
    added to the :data:`PlanterStamp` union — the K→O converter's provenance member (honest LLM
    planting identity + adversary family + winning round). A converted item is O-by-execution but
    second-author-regime-audited (no operator cross-check). Additive union member only; the existing
    deterministic/LLM members and their discrimination are unchanged.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from cogworx.verification.contracts import OracleFrame, Thesis

# ---------------------------------------------------------------------------
# Label provenance — discriminated union on `kind` (oracle invocation | human adjudication)
# ---------------------------------------------------------------------------


class OracleLabelProvenance(BaseModel):
    """The result of the oracle invocation that produced an oracle label (§3.4).

    ``test_provenance`` is NARROWED to ``Literal["frozen"]`` — an oracle label is only authoritative
    from the author-frozen test corpus (the §13.4 #5 self-test-laundering guard). Do NOT widen this
    to :attr:`cogworx.verification.contracts.Verdict.test_provenance`'s full domain.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["oracle"] = "oracle"
    returncode: int
    test_provenance: Literal["frozen"]
    holds: bool
    valid_check: bool
    oracle_id: str


class Adjudication(BaseModel):
    """One adjudicator's per-item verdict (§3.4).

    ``abstain`` is first-class — an honest "I cannot establish this" is structurally distinct from a
    ``clean`` verdict, mirroring the dialectic's honest-abstention discipline (Pod 4.2).
    """

    model_config = ConfigDict(frozen=True)

    adjudicator_id: str
    verdict: Literal["error", "clean", "abstain"]
    rationale: str
    timestamp: datetime


class RegimeAdjudication(BaseModel):
    """A second author's K-stratum error-regime audit verdict (eval-stats §6.D, Pod 4.4c-4).

    SEPARATE from :class:`Adjudication` by design: this records the regime second-author audit
    over the K-stratum, with its own verdict vocabulary. Do NOT fold this into ``Adjudication`` —
    the two ``abstain`` semantics are opposite (an :class:`Adjudication` ``abstain`` DROPS the item;
    a regime ``abstain`` KEEPS it in the pooled δ), and widening ``Adjudication.verdict`` would
    conflate them. This record is the single source of truth for a K-item's ``error_regime``.

    Downstream semantics:
      - ``confirm-regime`` keeps the planter's tag (the :class:`DeterministicPlanterStamp` /
        :class:`LLMPlanterStamp`-derived regime stands).
      - ``reassign-regime`` makes :attr:`reassigned_regime` the SOLE source of
        :attr:`CorpusItem.error_regime` — never the planter stamp.
      - ``abstain`` ⟹ ``error_regime=""``: the item STAYS in the pooled δ but is excluded from the
        §2.B per-regime contribution check.
    """

    model_config = ConfigDict(frozen=True)

    adjudicator_id: str
    verdict: Literal["confirm-regime", "reassign-regime", "abstain"]
    reassigned_regime: str | None = None
    """The new regime when ``verdict == "reassign-regime"``; MUST be ``None`` for
    ``confirm-regime`` / ``abstain``. When set, it is the SOLE source of
    :attr:`CorpusItem.error_regime`."""
    rationale: str
    timestamp: datetime

    @model_validator(mode="after")
    def _reassign_iff_regime_present(self) -> RegimeAdjudication:
        """``verdict == "reassign-regime"`` ⟺ ``reassigned_regime is not None`` — a reassignment
        MUST carry the new regime; a confirm/abstain MUST NOT."""
        if (self.verdict == "reassign-regime") != (self.reassigned_regime is not None):
            raise ValueError(
                f"verdict {self.verdict!r} disagrees with reassigned_regime "
                f"{self.reassigned_regime!r}: reassign-regime requires a regime, "
                f"confirm/abstain forbid one"
            )
        return self


class HumanLabelProvenance(BaseModel):
    """The adjudications behind a human label (§3.4). ``adjudications`` holds >=1 record; a
    ``tie_breaker_id`` names the adjudicator who resolved a split decision, if any."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["human"] = "human"
    adjudications: tuple[Adjudication, ...]
    tie_breaker_id: str | None = None


LabelProvenance = Annotated[
    OracleLabelProvenance | HumanLabelProvenance,
    Field(discriminator="kind"),
]

# ---------------------------------------------------------------------------
# Planter stamp — discriminated union on `injector_kind` (deterministic mutation | LLM)
# ---------------------------------------------------------------------------


class DeterministicPlanterStamp(BaseModel):
    """A deterministic mutation planter (research A8). ``operators`` are OPAQUE strings here — the
    operator vocabulary is validated at 4.4c-3, NOT in this schema."""

    model_config = ConfigDict(frozen=True)

    injector_kind: Literal["deterministic-mutation"] = "deterministic-mutation"
    operators: tuple[str, ...]


class LLMPlanterStamp(BaseModel):
    """An LLM error-injection planter (research A8), stamped with the planting model's identity."""

    model_config = ConfigDict(frozen=True)

    injector_kind: Literal["llm"] = "llm"
    model_family: str
    model_id: str


class ConvertedPlanterStamp(BaseModel):
    """A K-to-O conversion planter (Pod 4.4c-3.5): an LLM-planted K ERROR whose O-ness — an
    executable test that catches it — was SYNTHESIZED by the adversarial converter, not by a
    mechanical mutation operator.

    The ERROR provenance stays honest: :attr:`planter_model_family` / :attr:`planter_model_id` carry
    the ORIGINAL :class:`LLMPlanterStamp` identity that planted the K error, so a converted item
    keeps its real planting lineage. :attr:`adversary_family` is the converter family that authored
    the winning test, and :attr:`winning_round` is the converter round at which that test first made
    the error oracle-reachable.

    LABELING CONSEQUENCE (load-bearing — read before touching the O-branch in
    :mod:`cogworx.eval.labeling`): a converted item is O-by-execution (``label_source="oracle"``:
    the adversary's synthesized test made it oracle-reachable and the oracle still decided), but it
    has NO mutation operator — so its :attr:`CorpusItem.error_regime` is SECOND-AUTHOR-AUDITED (the
    same :class:`RegimeAdjudication` path K items use), NOT operator-derived. The §2.C operator
    cross-check (:func:`cogworx.eval.planting._derive_o_regime` over the planter operators) is
    SKIPPED for a converted item BY CONSTRUCTION — there are no operators to derive a regime."""

    model_config = ConfigDict(frozen=True)

    injector_kind: Literal["converted"] = "converted"
    planter_model_family: str
    planter_model_id: str
    adversary_family: str
    winning_round: int


PlanterStamp = Annotated[
    DeterministicPlanterStamp | LLMPlanterStamp | ConvertedPlanterStamp,
    Field(discriminator="injector_kind"),
]

# ---------------------------------------------------------------------------
# Difficulty marker — frozen, reported-only (NOT a gate input)
# ---------------------------------------------------------------------------


class DifficultyMarker(BaseModel):
    """Planted-difficulty marker (A7) + clean-difficulty for sibling matching (§3.3).

    Reported-only — NOT a gate input. ``surface_complexity`` is an author-supplied item property
    (e.g. AST nodes / LoC); it is NEVER an arm output (no flag-rate, no arm behavior leaks in here).
    """

    model_config = ConfigDict(frozen=True)

    planted_difficulty: Literal["easy", "medium", "hard"]
    surface_complexity: int
    is_matched_sibling: bool = False


# ---------------------------------------------------------------------------
# CorpusItem — the ground-truth-about-the-item record (plan §4)
# ---------------------------------------------------------------------------


class CorpusItem(BaseModel):
    """One gate-corpus item — ground truth ABOUT the item only, no run/behavior fields (plan §4).

    Behavior fields ``(item, trial, arm)`` are :class:`cogworx.eval.youden.Cell` fields produced at
    4.4d; this record is frozen into the corpus hash. ``is_error`` is audit/lock/ε only — the scorer
    reads ``stratum`` (``is_error`` is redundant with it for scoring, §12 CRITICAL-1).
    """

    model_config = ConfigDict(frozen=True)

    item_id: int
    frame: OracleFrame
    thesis: Thesis
    test_code: str | None
    is_error: int
    label_source: Literal["oracle", "human"]
    label_provenance: LabelProvenance
    stratum: Literal["O", "K", "clean"]
    oracle_reachable: bool
    error_regime: str = ""
    difficulty: DifficultyMarker
    matched_sibling_id: int | None
    split: Literal["tuning", "measurement"]
    planter: PlanterStamp
    content_hash: str = ""
    """Opaque contamination-lock hash (plan §3.8/§3.9). Carried OPAQUE here — 4.4c-2 does NOT
    compute or verify it; ``""`` means "not yet locked". Lock-time (4.4c-5) hashes over
    ``frame+thesis+test_code+is_error+stratum+regime+split`` and folds the corpus-level aggregate +
    git-SHA into the ``MeasurementFingerprint`` recorded on the TimescaleDB design-lineage ledger
    (plan §3.9-B; S6 durability)."""

    @model_validator(mode="after")
    def _label_source_matches_provenance(self) -> CorpusItem:
        """S5 guard: the declared ``label_source`` MUST agree with the provenance union member —
        an oracle label carries oracle provenance, a human label carries human provenance."""
        if self.label_source != self.label_provenance.kind:
            raise ValueError(
                f"label_source {self.label_source!r} disagrees with "
                f"label_provenance.kind {self.label_provenance.kind!r}"
            )
        return self


# ---------------------------------------------------------------------------
# load_corpus — structural-only corpus loader (Pod 4.4c-2; plan §3.2/§3.3/§3.9/§5)
# ---------------------------------------------------------------------------


class CorpusLoadError(ValueError):
    """A structural guard (G1-G4) rejected a corpus item at load time. The message names the
    offending ``item_id`` and which guard failed. This is NOT a re-verification failure (4.4c-4)
    nor a contamination/lock failure (4.4c-5/6) — those are out of this loader's scope."""


def _is_clean(item: CorpusItem) -> bool:
    """A clean item per the scorer's reading: ``stratum=="clean"`` (canonical) OR ``is_error==0``
    (audit redundancy, §12 CRITICAL-1). Either marks the item as the clean-label population."""
    return item.stratum == "clean" or item.is_error == 0


def load_corpus(
    items: Sequence[CorpusItem], *, measurement_run: bool = False
) -> list[CorpusItem]:
    """Structural-only corpus loader (Pod 4.4c-2; plan §3.2/§3.3/§3.9/§5).

    A PURE structural validator over already-constructed frozen :class:`CorpusItem`s. It enforces
    exactly four guards (G1-G4) and the split firewall (G3), then returns the loadable subset.
    Input is an in-memory ``Sequence[CorpusItem]`` — there is **no file I/O, no JSON
    deserialization** (YAGNI; keeps the load path I/O-free per CANON S1).

    ENFORCED NOW (this loader, 4.4c-2):
      - **G1 — clean-item label-provenance stamp present + well-shaped.** A clean item
        (``stratum=="clean"`` or ``is_error==0``) MUST carry a label-provenance stamp: an oracle
        stamp (``kind=="oracle"``, the C1 record) OR a human stamp (``kind=="human"`` with >=1
        adjudication, the C2 record). Checks **presence + shape only** — it does NOT re-run the
        oracle or verify the label's correctness.
      - **G2 — non-judge label source.** ``label_source in {"oracle","human"}`` and matches
        ``label_provenance.kind``. (The frozen model already enforces the match at construction and
        already forbids a ``judge`` source structurally; this loader guard is defense-in-depth.)
      - **G3 — split firewall (highest-value guard).** Items with ``split=="measurement"`` are
        PHYSICALLY filtered out unless ``measurement_run=True``. Default load returns tuning items
        only; ``measurement_run=True`` returns the measurement items (the ``--measurement-run``
        path).
      - **G4 — frozen-artifact well-formedness (STATIC reference integrity only).** Unique
        ``item_id`` across the loaded set; ``matched_sibling_id``, if set, must (a) reference an
        ``item_id`` present in the loaded set, (b) be symmetric (A names B => B names A), and (c) be
        opposite-label (one member error-stratum ``{"O","K"}``, the other ``clean``); ``test_code``
        is set iff the item is code-domain (``frame.problem_type=="code"``).

    Optional belt-and-suspenders: under ``measurement_run=True`` the loader asserts
    ``content_hash != ""`` as a never-locked tripwire — it does NOT recompute or verify the hash
    value (that is 4.4c-6's spike).

    DEFERRED — explicitly NOT in this loader (do not read G4 as the full §5 guard):
      - Mechanical stratum (re)assignment / running CodeOracle to assign O/K or re-verify C1 =>
        **4.4c-4**. The loader TRUSTS ``stratum``/``oracle_reachable``; it recomputes nothing.
      - C1/C2 re-verification of the clean label's *truth* => **4.4c-4** (G1 checks shape only).
      - Contamination disjointness (§3.8 — tuning/measurement item_id set-disjointness, content-hash
        log overlap) => **4.4c-5**.
      - git-SHA / ``content_hash`` recompute-and-refuse => **4.4c-5/6** (hash carried opaque here).
      - **CF-1 (critical):** G4 is STATIC reference integrity ONLY. The §5 *post-abstention* sibling
        bijection re-validation — dissolve a pair when one member is abstention-dropped and demote
        the survivor to the unpaired pool carrying its real stratum — is **4.4c-5**. Drops do not
        exist until labeling, so the bijection is NOT locked by this loader. Do not read G4 as the
        full §5 guard.

    Zero model calls, zero oracle invocations, recomputes no stratum, recomputes no hash, reads no
    journal.

    :raises CorpusLoadError: on any G1/G2/G4 violation, naming the offending ``item_id`` + guard.
    """
    # G3 — split firewall. Filter FIRST: every later guard (uniqueness, sibling references) is
    # evaluated over the loaded subset only, never the filtered-out half.
    wanted_split: Literal["tuning", "measurement"] = (
        "measurement" if measurement_run else "tuning"
    )
    loaded = [item for item in items if item.split == wanted_split]

    loaded_ids = {item.item_id for item in loaded}

    # G4(a) — unique item_id across the loaded set.
    seen: set[int] = set()
    for item in loaded:
        if item.item_id in seen:
            raise CorpusLoadError(f"G4: duplicate item_id {item.item_id} in loaded set")
        seen.add(item.item_id)

    by_id = {item.item_id: item for item in loaded}

    for item in loaded:
        # G2 — non-judge label source that matches the provenance union member. The frozen model
        # already enforces both; this re-asserts as defense-in-depth (a fail here means a model
        # invariant was bypassed).
        if item.label_source not in ("oracle", "human"):
            raise CorpusLoadError(
                f"G2: item_id {item.item_id} has non-{{oracle,human}} "
                f"label_source {item.label_source!r}"
            )
        if item.label_source != item.label_provenance.kind:
            raise CorpusLoadError(
                f"G2: item_id {item.item_id} label_source {item.label_source!r} "
                f"disagrees with label_provenance.kind {item.label_provenance.kind!r}"
            )

        # G1 — clean items must carry a well-shaped label-provenance stamp. Shape only.
        if _is_clean(item):
            prov = item.label_provenance
            if isinstance(prov, HumanLabelProvenance):
                if len(prov.adjudications) < 1:
                    raise CorpusLoadError(
                        f"G1: clean item_id {item.item_id} has a human label stamp with "
                        f"zero adjudications (the C2 record needs >=1)"
                    )
            elif not isinstance(prov, OracleLabelProvenance):
                # Unreachable given the frozen discriminated union, but explicit for the guard.
                raise CorpusLoadError(
                    f"G1: clean item_id {item.item_id} has no oracle/human label-provenance stamp"
                )

        # G4(d) — test_code present iff code-domain item.
        is_code = item.frame.problem_type == "code"
        if is_code and item.test_code is None:
            raise CorpusLoadError(
                f"G4: code-domain item_id {item.item_id} is missing test_code"
            )
        if not is_code and item.test_code is not None:
            raise CorpusLoadError(
                f"G4: non-code item_id {item.item_id} carries test_code "
                f"(pure-K/non-code items must have test_code=None)"
            )

        # Optional tripwire — never-locked artifact in a measurement run.
        if measurement_run and item.content_hash == "":
            raise CorpusLoadError(
                f"G4: measurement item_id {item.item_id} has an empty content_hash "
                f"(never-locked artifact in a measurement run)"
            )

    # G4(b)+(c) — matched-sibling reference integrity: present, symmetric, opposite-label.
    for item in loaded:
        sib_id = item.matched_sibling_id
        if sib_id is None:
            continue
        if sib_id not in loaded_ids:
            raise CorpusLoadError(
                f"G4: item_id {item.item_id} names matched_sibling_id {sib_id} "
                f"which is absent from the loaded set"
            )
        sib = by_id[sib_id]
        if sib.matched_sibling_id != item.item_id:
            raise CorpusLoadError(
                f"G4: asymmetric sibling — item_id {item.item_id} names {sib_id} but "
                f"item_id {sib_id} names {sib.matched_sibling_id}"
            )
        # Opposite-label: exactly one member is clean, the other error-stratum.
        if _is_clean(item) == _is_clean(sib):
            raise CorpusLoadError(
                f"G4: same-label sibling pair (item_id {item.item_id}, {sib_id}) — a "
                f"matched pair must be one clean + one error-stratum (mutate-then-revert)"
            )

    return loaded


__all__ = [
    "Adjudication",
    "ConvertedPlanterStamp",
    "CorpusItem",
    "CorpusLoadError",
    "DeterministicPlanterStamp",
    "DifficultyMarker",
    "HumanLabelProvenance",
    "LLMPlanterStamp",
    "LabelProvenance",
    "OracleLabelProvenance",
    "PlanterStamp",
    "RegimeAdjudication",
    "load_corpus",
]
