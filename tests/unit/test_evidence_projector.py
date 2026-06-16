"""Unit tests for VerificationEvidenceProjector (CANON S1, S5, S6, S9 — F1).

Projector honesty: a committed oracle-verdict step with an inference-sourced verdict produces
evidence with source="inference" (never laundered to confidence=1.0 confirmed); an executable
tool-sourced verdict produces source="tool", EvidenceType="tool_proof".  Assertions are on the
derived posterior / source, NOT the raw evidence_for count (events accumulate; the posterior is
the idempotent surface — plan §5/S2).

Tests:
  P1 oracle-verdict / inference source → inference evidence, no confirmed
  P2 oracle-verdict / tool source      → tool_proof evidence, confirmed
  P3 antithesis-verdict / no oracle    → inference survival evidence
  P4 non-verdict kind                  → skipped (control flow, no evidence)
  P5 idempotency                       → re-tick after cursor-stall = no posterior change
  P6 cursor advances atomically        → after tick, cursor is at the scanned step
  P7 empty journal                     → returns 0, cursor untouched
  P8 source_authority ordering         → executable > inference (contractual)
  P9 antithesis BROKE / ABSTAINED      → record_for returns None, nothing written
  P10 claim-node accumulation          → two verdicts about the SAME claim → one node + two events
"""

from __future__ import annotations

from datetime import UTC, datetime

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.knowledge.evidence import EVIDENCE_BASE_WEIGHTS
from cogworx.knowledge.identity import claim_id_for
from cogworx.loop.result import Done
from cogworx.substrate.journal import StepRecord
from cogworx.testing.doubles import InMemoryEntityKG, InMemoryJournal
from cogworx.verification.contracts import Verdict
from cogworx.verification.evidence_projector import (
    _AUTHORITY_EXECUTABLE,
    _AUTHORITY_INFERENCE,
    EVIDENCE_PROJECTOR_CONSUMER,
    VerificationEvidenceProjector,
)
from cogworx.verification.honest_failure import AntithesisDisposition, AntithesisVerdict

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_T0 = datetime(2026, 6, 15, 12, 0, 0, tzinfo=UTC)
_RUN = "run-test-001"
_SESSION = "session-test-001"
_PATHWAY = "dialectic_pathway"
_CLAIM_TEXT = "The output of sort([3,1,2]) is [1,2,3]."


def _sys_prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


def _oracle_artifact(verdict: Verdict, verifiable_claim: str = _CLAIM_TEXT) -> Artifact:
    """Build the ExperimentStage artifact shape (kind='oracle-verdict')."""
    data = verdict.model_dump(mode="json")
    data["verifiable_claim"] = verifiable_claim
    return Artifact(
        kind="oracle-verdict", produced_by="experiment", provenance=_sys_prov(), data=data
    )


def _antithesis_artifact(av: AntithesisVerdict, verifiable_claim: str = _CLAIM_TEXT) -> Artifact:
    """Build the AntithesisStage artifact shape (kind='antithesis-verdict')."""
    data = av.model_dump(mode="json")
    data["verifiable_claim"] = verifiable_claim
    return Artifact(
        kind="antithesis-verdict", produced_by="antithesis", provenance=_sys_prov(), data=data
    )


def _control_artifact() -> Artifact:
    return Artifact(kind="dialectic-route", produced_by="evaluate", provenance=_sys_prov(), data={})


async def _start(journal: InMemoryJournal, run_id: str = _RUN) -> None:
    await journal.start_run(
        run_id,
        _SESSION,
        pathway_id=_PATHWAY,
        pathway_version=1,
        pathway_fingerprint="fp",
    )


async def _commit(
    journal: InMemoryJournal,
    *,
    run_id: str = _RUN,
    step_index: int,
    stage: str,
    output: Artifact,
) -> None:
    await journal.commit_step(
        StepRecord(
            run_id=run_id,
            step_index=step_index,
            stage_name=stage,
            result=Done(output=output),
            committed_at=_T0,
        )
    )


def _projector(journal: InMemoryJournal, kg: InMemoryEntityKG) -> VerificationEvidenceProjector:
    return VerificationEvidenceProjector(journal=journal, entity_kg=kg)


def _expected_claim_id(verifiable_claim: str = _CLAIM_TEXT) -> str:
    return claim_id_for(
        subject=verifiable_claim,
        predicate="verified_by",
        object_repr=verifiable_claim,
    )


# ---------------------------------------------------------------------------
# P1 — oracle-verdict / inference source → inference evidence (NO laundering)
# ---------------------------------------------------------------------------


async def test_p1_inference_oracle_verdict_produces_inference_evidence() -> None:
    """A model-judge verdict (source='inference') MUST NOT produce confirmed evidence.

    This is the core F1 anti-laundering assertion (plan §5 spike criterion 5).
    The projector reads verdict.source directly; it never upgrades inference → confirmed.
    """
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=True, valid_check=True, reasoning="looks right", source="inference")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    count = await _projector(journal, kg).tick()

    # A model-judge oracle with holds=True, valid_check=True, is_executable=False → record_for
    # returns None (F2 — a judge-only pass produces NO truth-posterior evidence).
    assert count == 0
    claim_id = _expected_claim_id()
    claim = await kg.get_claim(claim_id)
    assert claim is None  # no evidence written — the judge pass is routing-only


async def test_p1b_inference_oracle_verdict_not_holds_produces_no_evidence() -> None:
    """A model-judge verdict with holds=False also produces no truth evidence (routing-only)."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=False, valid_check=True, reasoning="failed", source="inference")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    count = await _projector(journal, kg).tick()
    assert count == 0


# ---------------------------------------------------------------------------
# P2 — oracle-verdict / tool source → tool_proof evidence, confirmed, source_authority=1.0
# ---------------------------------------------------------------------------


async def test_p2_executable_oracle_pass_produces_tool_proof() -> None:
    """Executable oracle pass (tool-sourced, holds=True) → tool_proof confirmed evidence."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=True, valid_check=True, reasoning="tests passed", source="tool")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    count = await _projector(journal, kg).tick()
    assert count == 1

    claim_id = _expected_claim_id()
    claim = await kg.get_claim(claim_id)
    assert claim is not None
    assert claim.epistemic_type == "confirmed"

    events = await kg.evidence_for(claim_id)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "tool_proof"
    assert ev.polarity == "+"
    assert ev.source_id == f"{_RUN}:0"
    assert ev.source_authority == _AUTHORITY_EXECUTABLE
    assert ev.base_weight == EVIDENCE_BASE_WEIGHTS["tool_proof"]
    assert ev.run_id == _RUN
    assert ev.stage == "experiment"


async def test_p2b_executable_oracle_fail_produces_refutation() -> None:
    """An executable oracle fail (source='tool', holds=False) → refutation '-', confirmed."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=False, valid_check=True, reasoning="tests failed", source="tool")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    count = await _projector(journal, kg).tick()
    assert count == 1

    claim_id = _expected_claim_id()
    events = await kg.evidence_for(claim_id)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "refutation"
    assert ev.polarity == "-"
    assert ev.source_authority == _AUTHORITY_EXECUTABLE


# ---------------------------------------------------------------------------
# P3 — antithesis-verdict / oracle_backed=False → inference survival evidence
# ---------------------------------------------------------------------------


async def test_p3_antithesis_could_not_break_without_oracle_produces_inference_survival() -> None:
    """v1 antithesis (model adversary, oracle_backed=False) → inference antithesis_survival.

    This tests the §6 adapter + projector chain:
      AntithesisVerdict(COULD_NOT_BREAK, oracle_backed=False)
      → verdict_from_antithesis → Verdict(holds=True, source='inference')
      → record_for(role='antithesis') → VerificationRecord('antithesis_survival', '+', 'inference')
      → evidence with source_authority=0.5 (NEVER executable/confirmed laundering).
    """
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    av = AntithesisVerdict(disposition=AntithesisDisposition.COULD_NOT_BREAK, oracle_backed=False)
    await _commit(journal, step_index=0, stage="antithesis", output=_antithesis_artifact(av))

    count = await _projector(journal, kg).tick()
    assert count == 1

    claim_id = _expected_claim_id()
    claim = await kg.get_claim(claim_id)
    assert claim is not None
    # First-write-wins: inference claim (COULD_NOT_BREAK with no oracle backing → inference)
    assert claim.epistemic_type == "inference"

    events = await kg.evidence_for(claim_id)
    assert len(events) == 1
    ev = events[0]
    assert ev.type == "antithesis_survival"
    assert ev.polarity == "+"
    # NEVER laundered to confirmed / authority=1.0
    assert ev.source_authority == _AUTHORITY_INFERENCE
    assert ev.source_authority < _AUTHORITY_EXECUTABLE  # ordering contract


# ---------------------------------------------------------------------------
# P4 — non-verdict artifact kind → skipped
# ---------------------------------------------------------------------------


async def test_p4_control_step_produces_no_evidence() -> None:
    """A non-verdict kind (e.g. 'dialectic-route') is skipped as ordinary control flow."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    await _commit(journal, step_index=0, stage="evaluate", output=_control_artifact())

    count = await _projector(journal, kg).tick()
    assert count == 0
    # Cursor still advances to the scanned step (same as trial projector's P0-2 fix).
    cursor = await kg.read_cursor(EVIDENCE_PROJECTOR_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == (_RUN, 0)


# ---------------------------------------------------------------------------
# P5 — idempotency: re-tick after cursor-stall = no posterior change
# ---------------------------------------------------------------------------


async def test_p5_idempotency_re_tick_no_posterior_change() -> None:
    """Re-ticking after the cursor is at the frontier produces no change (exactly-once into KG)."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=True, valid_check=True, reasoning="ok", source="tool")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    proj = _projector(journal, kg)
    first = await proj.tick()
    assert first == 1

    # Subsequent ticks on the same static journal: cursor is past the frontier → 0-row reads.
    for _ in range(3):
        again = await proj.tick()
        assert again == 0

    # The evidence count has NOT grown beyond the first tick.
    claim_id = _expected_claim_id()
    events = await kg.evidence_for(claim_id)
    assert len(events) == 1


# ---------------------------------------------------------------------------
# P6 — cursor advances atomically with the evidence write
# ---------------------------------------------------------------------------


async def test_p6_cursor_advances_atomically() -> None:
    """After tick(), cursor sits at the last scanned step; evidence + cursor are co-written."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=True, valid_check=True, reasoning="ok", source="tool")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    assert await kg.read_cursor(EVIDENCE_PROJECTOR_CONSUMER) is None

    await _projector(journal, kg).tick()

    cursor = await kg.read_cursor(EVIDENCE_PROJECTOR_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == (_RUN, 0)
    assert cursor.commit_ordinal >= 1
    # Evidence was written in the same tick.
    assert await kg.get_claim(_expected_claim_id()) is not None


# ---------------------------------------------------------------------------
# P7 — empty journal: returns 0, cursor untouched
# ---------------------------------------------------------------------------


async def test_p7_empty_journal_returns_zero_cursor_untouched() -> None:
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    count = await _projector(journal, kg).tick()

    assert count == 0
    assert await kg.read_cursor(EVIDENCE_PROJECTOR_CONSUMER) is None


# ---------------------------------------------------------------------------
# P8 — source_authority ordering: executable > inference (contractual invariant)
# ---------------------------------------------------------------------------


def test_p8_source_authority_ordering() -> None:
    """The contractual ordering: executable > inference must hold (never reverse)."""
    assert _AUTHORITY_EXECUTABLE > _AUTHORITY_INFERENCE
    assert _AUTHORITY_EXECUTABLE <= 1.0
    assert _AUTHORITY_INFERENCE >= 0.0


# ---------------------------------------------------------------------------
# P9 — antithesis BROKE / ABSTAINED → record_for returns None, nothing written
# ---------------------------------------------------------------------------


async def test_p9_antithesis_broke_produces_no_evidence() -> None:
    """A model-claimed antithesis break drives routing, not truth evidence (S9).

    The verdict_from_antithesis adapter maps BROKE → holds=False (valid_check=True), and
    record_for(role='antithesis') returns None for holds=False — so no evidence is written.
    """
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    av = AntithesisVerdict(
        disposition=AntithesisDisposition.BROKE,
        breakage="The function returns None on empty input.",
        oracle_backed=False,
    )
    await _commit(journal, step_index=0, stage="antithesis", output=_antithesis_artifact(av))

    count = await _projector(journal, kg).tick()
    assert count == 0
    # Cursor still advances (scanned step consumed, nothing to write).
    cursor = await kg.read_cursor(EVIDENCE_PROJECTOR_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == (_RUN, 0)


async def test_p9b_antithesis_abstained_produces_no_evidence() -> None:
    """An ABSTAINED antithesis (valid_check=False via adapter) yields no evidence."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    av = AntithesisVerdict(disposition=AntithesisDisposition.ABSTAINED, oracle_backed=False)
    await _commit(journal, step_index=0, stage="antithesis", output=_antithesis_artifact(av))

    count = await _projector(journal, kg).tick()
    assert count == 0


# ---------------------------------------------------------------------------
# P10 — claim-node accumulation: two verdicts about the SAME claim → one node, two events
# ---------------------------------------------------------------------------


async def test_p10_same_claim_accumulates_on_one_node() -> None:
    """Two verdicts about the same verifiable_claim → ONE claim node, TWO evidence events.

    This validates the claim-node identity discipline (plan §5 S1): claim_id_for is derived
    from the claim subject, not per-verdict data, so re-derivation collides by design.
    """
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    # Step 0: executable oracle pass.
    v0 = Verdict(holds=True, valid_check=True, reasoning="pass", source="tool")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(v0))

    # Step 1: antithesis COULD_NOT_BREAK (same verifiable_claim).
    av1 = AntithesisVerdict(disposition=AntithesisDisposition.COULD_NOT_BREAK, oracle_backed=False)
    await _commit(journal, step_index=1, stage="antithesis", output=_antithesis_artifact(av1))

    proj = _projector(journal, kg)
    count = await proj.tick()
    assert count == 2

    # ONE claim node.
    claim_id = _expected_claim_id()
    claim = await kg.get_claim(claim_id)
    assert claim is not None

    # TWO evidence events accumulated (not merged).
    events = await kg.evidence_for(claim_id)
    assert len(events) == 2
    types = {ev.type for ev in events}
    assert "tool_proof" in types
    assert "antithesis_survival" in types

    # Idempotent re-tick: still two events.
    count2 = await proj.tick()
    assert count2 == 0
    events2 = await kg.evidence_for(claim_id)
    assert len(events2) == 2


# ---------------------------------------------------------------------------
# P11 — missing verifiable_claim field in artifact data → skipped gracefully
# ---------------------------------------------------------------------------


async def test_p11_missing_verifiable_claim_skipped() -> None:
    """If the artifact data has no 'verifiable_claim' key, the step is skipped (no crash)."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=True, valid_check=True, reasoning="ok", source="tool")
    # Omit verifiable_claim from data.
    data = verdict.model_dump(mode="json")
    art = Artifact(
        kind="oracle-verdict", produced_by="experiment", provenance=_sys_prov(), data=data
    )
    await _commit(journal, step_index=0, stage="experiment", output=art)

    count = await _projector(journal, kg).tick()
    assert count == 0
    # Cursor still advances (the step was scanned; no evidence to write).
    cursor = await kg.read_cursor(EVIDENCE_PROJECTOR_CONSUMER)
    assert cursor is not None


# ---------------------------------------------------------------------------
# P12 — system-sourced oracle verdict → same as tool (also executable, authority=1.0)
# ---------------------------------------------------------------------------


async def test_p12_system_source_is_executable() -> None:
    """A 'system'-sourced verdict is also executable → confirmed evidence, authority=1.0."""
    journal, kg = InMemoryJournal(), InMemoryEntityKG()
    await _start(journal)

    verdict = Verdict(holds=True, valid_check=True, reasoning="engine check", source="system")
    await _commit(journal, step_index=0, stage="experiment", output=_oracle_artifact(verdict))

    count = await _projector(journal, kg).tick()
    assert count == 1

    claim_id = _expected_claim_id()
    events = await kg.evidence_for(claim_id)
    assert len(events) == 1
    ev = events[0]
    assert ev.source_authority == _AUTHORITY_EXECUTABLE
    assert ev.type == "tool_proof"
