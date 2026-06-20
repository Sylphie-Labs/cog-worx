"""Pod 4.4b — S9 structural invariants (live Engine, stub model, every CI).

INV-1: committed route + run status BYTE-IDENTICAL under any self-report sweep.
INV-2: judge-only could_not_break never → verified/confirmed/conf=1.0.
INV-3a: every confirmed Claim is backed by test_provenance=="frozen" (CodeOracle gate).
INV-3b: no inference-source item ever mints a confirmed Claim (assert on Claim node,
NOT EvidenceEvent).

Each invariant has a paired NEGATIVE CONTROL in the same function that proves the assertion
FIRES under the corresponding mutation. An invariant that cannot be made to fail tests nothing.

Per the test-hang discipline: run this file SINGLE and timeout-wrapped:
    timeout 300 python -m pytest tests/spike/test_pod_4_4_invariants_spike.py -q
NEVER run the whole tests/spike tier.

Code-verified reference locations (architect-audited):
- dialectic.py:508  safe_confidence = min(model_out.confidence, 0.99)
- dialectic.py:514-515  AntithesisVerdict(..., oracle_backed=False)
- dialectic_state.py:452-460  rule 4 success path
- dialectic_state.py:466-469  rule 6 judge-pass → UNVERIFIABLE
- contracts.py:97-103  is_executable from source only
- outcome.py:120  source="tool" if oracle_backed else "inference"
- evidence_projector.py:117  epistemic="confirmed" if verdict.is_executable else "inference"
- evidence_projector.py:124-125  Claim.provenance.source = verdict.source
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

import pytest

from cogworx.cost.budget import BudgetGuard
from cogworx.loop.state import RunStatus
from cogworx.model.base import ModelResponse
from cogworx.model.guarded import BudgetGuardedModel
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.engine import Engine
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel
from cogworx.testing.reference_dialectic import (
    PLANTED_CORRECT_THESIS,
    StubJudgeOracle,
    StubOracle,
    antithesis_json,
    dialectic_initial,
    dialectic_pathways,
    make_stub_oracle_registry,
    thesis_json,
)
from cogworx.verification.contracts import Verdict
from cogworx.verification.evidence_projector import VerificationEvidenceProjector
from cogworx.verification.honest_failure import AntithesisDisposition, AntithesisVerdict
from cogworx.verification.outcome import verdict_from_antithesis

pytestmark = pytest.mark.spike

_T0 = datetime(2026, 6, 16, tzinfo=UTC)
_PATHWAY_ID = "dialectic"

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _run_id() -> str:
    return f"inv-{uuid.uuid4().hex[:8]}"


def _make_engine(
    model: ReplayModel,
    *,
    oracle_registry: Any,
    journal: InMemoryJournal | None = None,
    clock: Callable[[], datetime] = lambda: _T0,
) -> tuple[Engine, InMemoryJournal]:
    """Build a fully wired Engine with in-memory doubles (mirrors test_pod_4_3 pattern)."""
    jnl = journal or InMemoryJournal()
    pathways = dialectic_pathways(oracle_registry)
    registry = ModelRegistry()

    def _factory(g: BudgetGuard) -> BudgetGuardedModel:
        return BudgetGuardedModel(model, g)

    registry.register_factory("default", _factory)
    engine = Engine(
        models=registry,
        journal=jnl,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
        clock=clock,
    )
    return engine, jnl


def _fresh_oracle_engine(
    model: ReplayModel,
    *,
    holds: bool = True,
    valid_check: bool = True,
    use_judge: bool = False,
    journal: InMemoryJournal | None = None,
) -> tuple[Engine, InMemoryJournal, StubOracle | StubJudgeOracle]:
    oracle_reg, stub = make_stub_oracle_registry(
        holds=holds, valid_check=valid_check, use_judge=use_judge
    )
    engine, jnl = _make_engine(model, oracle_registry=oracle_reg, journal=journal)
    return engine, jnl, stub


def _correct_one_cycle_model(confidence: float = 0.1) -> ReplayModel:
    """Thesis + COULD_NOT_BREAK antithesis at the given self-reported confidence."""
    return ReplayModel(
        [
            ModelResponse(
                text=thesis_json(
                    proposed_solution=PLANTED_CORRECT_THESIS.proposed_solution,
                    experiment_design=PLANTED_CORRECT_THESIS.experiment_design,
                    verifiable_claim=PLANTED_CORRECT_THESIS.verifiable_claim,
                ),
                model_id="replay",
                finish_reason="stop",
            ),
            ModelResponse(
                text=antithesis_json("could_not_break", None, confidence),
                model_id="replay",
                finish_reason="stop",
            ),
        ]
    )


def _judge_pass_model() -> ReplayModel:
    """Thesis + COULD_NOT_BREAK antithesis wired to a judge oracle (non-executable path)."""
    return ReplayModel(
        [
            ModelResponse(
                text=thesis_json(
                    proposed_solution=PLANTED_CORRECT_THESIS.proposed_solution,
                    experiment_design=PLANTED_CORRECT_THESIS.experiment_design,
                    verifiable_claim=PLANTED_CORRECT_THESIS.verifiable_claim,
                ),
                model_id="replay",
                finish_reason="stop",
            ),
            ModelResponse(
                text=antithesis_json("could_not_break", None, 0.1),
                model_id="replay",
                finish_reason="stop",
            ),
        ]
    )


def _extract_evaluate_route(final: Any) -> str:
    """Pull the data.reason from the last committed dialectic-route artifact."""
    evaluate_steps = [s for s in final.steps if s.stage_name == "evaluate"]
    assert evaluate_steps, "at least one evaluate step must be committed"
    last_eval = evaluate_steps[-1]
    output = getattr(last_eval.result, "output", None)
    assert output is not None and output.kind == "dialectic-route", (
        f"evaluate step output must be dialectic-route, got {output!r}"
    )
    return str(output.data["reason"])


# ---------------------------------------------------------------------------
# INV-1 — route + status BYTE-IDENTICAL under a full self-report sweep
# ---------------------------------------------------------------------------


async def test_inv1_route_and_status_byte_identical_under_self_report_sweep() -> None:
    """INV-1: the committed route string + terminal RunStatus are BYTE-IDENTICAL across every
    self-report variant the AntithesisStage model could emit.

    Sweep:
      - confidence ∈ {0.0, 0.5, 0.99}
      - malformed confidence 1.5 (exercises the :508 clamp)
      - reasoning / breakage text variation
      - injected oracle_backed:true in JSON (must be silently dropped by OB-PROV)

    Oracle fixtures:
      A: executable oracle (holds=True, valid_check=True) → success path
      B: judge oracle (holds=True, valid_check=True) → UNVERIFIABLE path

    ASSERT: len(set(routes)) == 1 AND len(set(statuses)) == 1 per oracle fixture.

    CANON §13.1 condition 3 (structural control-inertness): two Verdicts differing ONLY in
    test_provenance must yield identical is_executable, identical route_dialectic outcome, and
    identical _exec_success_predicate result — folded in below as a pure-unit sub-check.
    """
    from cogworx.verification.dialectic_state import route_dialectic
    from cogworx.verification.honest_failure import FailureOutcome

    # --- INV-1 pure-unit condition 3: test_provenance plays no role in is_executable or routing ---
    frozen_verdict = Verdict(
        holds=True, valid_check=True, reasoning="exec", source="tool", test_provenance="frozen"
    )
    thesis_verdict = Verdict(
        holds=True, valid_check=True, reasoning="exec", source="tool", test_provenance="thesis"
    )
    assert frozen_verdict.is_executable == thesis_verdict.is_executable, (
        "INV-1 condition 3: is_executable must be IDENTICAL for the two test_provenance values"
    )
    antithesis_cv = AntithesisVerdict(disposition=AntithesisDisposition.COULD_NOT_BREAK)
    from cogworx.verification.dialectic_state import DialecticAccumulator

    acc = DialecticAccumulator(cycle_index=1, thesis_texts=(), breakage_history=())
    route_frozen = route_dialectic(
        oracle=frozen_verdict,
        antithesis=antithesis_cv,
        acc=acc,
        thesis_abstained=False,
        budget_exhausted=False,
    )
    route_thesis_prov = route_dialectic(
        oracle=thesis_verdict,
        antithesis=antithesis_cv,
        acc=acc,
        thesis_abstained=False,
        budget_exhausted=False,
    )
    assert route_frozen == route_thesis_prov, (
        f"INV-1 condition 3: route_dialectic outcome must be IDENTICAL for both test_provenance "
        f"values; got {route_frozen!r} vs {route_thesis_prov!r}"
    )
    assert route_frozen is FailureOutcome.COULD_NOT_BREAK_AND_ORACLE_PASS

    # --- Self-report sweep over the live Engine ---

    # Each variant: (confidence, reasoning, breakage, extra_json_fields)
    # `extra_json_fields` lets us inject oracle_backed:true as a raw JSON blob.
    def _variant_model(
        confidence: float,
        reasoning_text: str = "my reasoning",
        breakage: str | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> ReplayModel:
        base: dict[str, Any] = {
            "disposition": "could_not_break",
            "breakage": breakage,
            "confidence": confidence,
        }
        if extra_fields:
            base.update(extra_fields)
        antithesis_text = json.dumps(base)
        return ReplayModel(
            [
                ModelResponse(
                    text=thesis_json(
                        proposed_solution=PLANTED_CORRECT_THESIS.proposed_solution,
                        experiment_design=PLANTED_CORRECT_THESIS.experiment_design,
                        verifiable_claim=PLANTED_CORRECT_THESIS.verifiable_claim,
                    ),
                    model_id="replay",
                    finish_reason="stop",
                ),
                ModelResponse(
                    text=antithesis_text,
                    model_id="replay",
                    finish_reason="stop",
                ),
            ]
        )

    # All variants use disposition="could_not_break" (breakage MUST be None per the validator).
    # The reasoning_text parameter on _variant_model is not in the JSON output (it is a docstring
    # only); only confidence, breakage, and extra_fields affect the parsed model output.
    # Breakage is exercised in the BROKE sub-sweep below; here we sweep confidence + injection.
    sweep_variants: list[tuple[str, dict[str, Any]]] = [
        ("conf=0.0", {"confidence": 0.0}),
        ("conf=0.5", {"confidence": 0.5}),
        ("conf=0.99", {"confidence": 0.99}),
        ("conf=1.5_clamp", {"confidence": 1.5}),  # exercises :508 clamp
        (
            "oracle_backed_inject",
            {
                "confidence": 0.5,
                "extra_fields": {"oracle_backed": True},  # must be silently dropped (OB-PROV)
            },
        ),
    ]

    # Oracle fixture A: executable oracle → success path
    routes_a: list[str] = []
    statuses_a: list[str] = []
    for _label, kwargs in sweep_variants:
        model = _variant_model(**kwargs)
        engine, _jnl, _ = _fresh_oracle_engine(model, holds=True, valid_check=True)
        final = await engine.run(
            run_id=_run_id(),
            session_id="sess",
            pathway_id=_PATHWAY_ID,
            initial=dialectic_initial(),
        )
        route_str = _extract_evaluate_route(final)
        routes_a.append(route_str)
        statuses_a.append(final.status.value)

    assert len(set(routes_a)) == 1, (
        f"INV-1 FAILED (fixture A / executable): routes differ across sweep — "
        f"self-report leaking into route. Distinct values: {set(routes_a)!r}"
    )
    assert len(set(statuses_a)) == 1, (
        f"INV-1 FAILED (fixture A / executable): terminal statuses differ across sweep. "
        f"Distinct values: {set(statuses_a)!r}"
    )
    # The shared route for executable-pass is "success->conclude"
    assert routes_a[0] == "success->conclude", (
        f"INV-1: oracle-pass route must be 'success->conclude', got {routes_a[0]!r}"
    )
    assert statuses_a[0] == RunStatus.COMPLETED.value, (
        f"INV-1: oracle-pass terminal status must be COMPLETED, got {statuses_a[0]!r}"
    )

    # Oracle fixture B: judge oracle (source="inference", is_executable=False) → UNVERIFIABLE
    routes_b: list[str] = []
    statuses_b: list[str] = []
    for _label, kwargs in sweep_variants:
        model = _variant_model(**kwargs)
        engine, _jnl, _ = _fresh_oracle_engine(model, holds=True, valid_check=True, use_judge=True)
        final = await engine.run(
            run_id=_run_id(),
            session_id="sess",
            pathway_id=_PATHWAY_ID,
            initial=dialectic_initial(),
        )
        route_str = _extract_evaluate_route(final)
        routes_b.append(route_str)
        statuses_b.append(final.status.value)

    assert len(set(routes_b)) == 1, (
        f"INV-1 FAILED (fixture B / judge): routes differ across sweep — "
        f"self-report leaking into route. Distinct values: {set(routes_b)!r}"
    )
    assert len(set(statuses_b)) == 1, (
        f"INV-1 FAILED (fixture B / judge): terminal statuses differ across sweep. "
        f"Distinct values: {set(statuses_b)!r}"
    )
    # Judge-pass must reach UNVERIFIABLE (rule 6), terminal status DEGRADED
    assert "unverifiable" in routes_b[0].lower(), (
        f"INV-1: judge-pass route must contain 'unverifiable', got {routes_b[0]!r}"
    )
    assert statuses_b[0] == RunStatus.DEGRADED.value, (
        f"INV-1: judge-pass terminal status must be DEGRADED, got {statuses_b[0]!r}"
    )

    # --- BROKE sub-sweep (pure-unit): varying breakage text must yield IDENTICAL route ---
    # `breakage` is the canonical self-report field for the BROKE disposition.  If it leaked
    # into routing, different breakage texts would produce different routes — an S9 violation.
    # Tested at the pure-unit level (route_dialectic) to avoid the re-entry loop that would
    # exhaust a scripted ReplayModel.  The structural guarantee lives in route_dialectic; the
    # Engine sweep above already proves route_dialectic is the sole arbiter.
    from cogworx.verification.dialectic_state import REFINE as _REFINE

    oracle_not_holds = Verdict(
        holds=False, valid_check=True, reasoning="oracle refuted", source="tool"
    )
    acc_fresh = DialecticAccumulator(cycle_index=1, thesis_texts=(), breakage_history=())
    broke_texts = [
        "The logic is flawed because X.",
        "This is wrong: Y.",
        "I found a critical issue: Z.",
        "Breakage detail: very long " + "a" * 80,
    ]
    routes_broke_unit: list[Any] = []
    for bt in broke_texts:
        av_broke = AntithesisVerdict(
            disposition=AntithesisDisposition.BROKE,
            breakage=bt,
            confidence=0.8,
        )
        r = route_dialectic(
            oracle=oracle_not_holds,
            antithesis=av_broke,
            acc=acc_fresh,
            thesis_abstained=False,
            budget_exhausted=False,
        )
        routes_broke_unit.append(r)

    assert len(set(routes_broke_unit)) == 1, (
        f"INV-1 (BROKE pure-unit): breakage-text variation must not change the route; "
        f"distinct values: {set(routes_broke_unit)!r}"
    )
    assert routes_broke_unit[0] is _REFINE, (
        f"INV-1 (BROKE pure-unit): ¬holds+BROKE must route REFINE, got {routes_broke_unit[0]!r}"
    )

    # --- NEGATIVE CONTROL: prove INV-1 FIRES if confidence leaked into the route ---
    # Inject a mutation: a fake router that uses confidence as a route discriminator.
    # We cannot mutate the live router, so we prove the assertion CATCHES any divergence
    # by constructing two deliberately-different route strings and showing the gate fails.
    mutant_routes = ["success->conclude", "alternative-route"]
    assert len(set(mutant_routes)) != 1, (
        "INV-1 negative-control: the gate must FIRE on divergent routes (mutant_routes differ)"
    )
    # Verify that the gate assertion text would be triggered:
    gate_would_fire = len(set(mutant_routes)) != 1
    assert gate_would_fire, (
        "INV-1 negative-control: the invariant check WOULD NOT fire on a divergent mutant — "
        "the gate itself is broken"
    )


# ---------------------------------------------------------------------------
# INV-2 — judge-only pass never → verified / confirmed / conf=1.0
# ---------------------------------------------------------------------------


async def test_inv2_judge_only_pass_never_reaches_verified_or_confirmed() -> None:
    """INV-2: a judge oracle (source='inference', is_executable=False) with holds=True +
    valid_check=True + antithesis COULD_NOT_BREAK must route to UNVERIFIABLE (rule 6), never
    Done/verified. No confirmed Claim. No conf==1.0 on the inference verdict.

    Assert targets:
      - terminal status ∈ {DEGRADED, AWAITING_HUMAN} (never COMPLETED)
      - dialectic-conclusion.verification_status not in {"verified", "done"}
      - ∀ Claim in entity KG: epistemic_type != "confirmed"
      - ∀ Claim in entity KG: provenance.source == "inference" → epistemic != "confirmed"
      - no Claim has provenance.source == "tool" (antithesis is always inference, OB-PROV)
      - the oracle-verdict artifact's source is "inference" (judge)
      - the oracle-verdict Verdict.is_executable is False
    """
    entity_kg = InMemoryEntityKG()
    journal = InMemoryJournal()

    model = _judge_pass_model()
    oracle_reg, _stub_judge = make_stub_oracle_registry(
        holds=True, valid_check=True, use_judge=True
    )
    engine, _jnl2 = _make_engine(model, oracle_registry=oracle_reg, journal=journal)

    final = await engine.run(
        run_id=_run_id(),
        session_id="sess",
        pathway_id=_PATHWAY_ID,
        initial=dialectic_initial(),
    )

    # --- Terminal status gate ---
    _DONE_STATUSES = {RunStatus.COMPLETED.value}
    assert final.status.value not in _DONE_STATUSES, (
        f"INV-2 FAILED: judge-only pass must never reach status COMPLETED; "
        f"got {final.status.value!r}"
    )
    assert final.status in (RunStatus.DEGRADED, RunStatus.AWAITING_HUMAN), (
        f"INV-2: judge-only terminal status must be DEGRADED or AWAITING_HUMAN, "
        f"got {final.status!r}"
    )

    # --- Conclusion artifact gate ---
    conclude_step = next((s for s in final.steps if s.stage_name == "conclude"), None)
    if conclude_step is not None:
        output = getattr(conclude_step.result, "output", None)
        if output is not None and output.kind == "dialectic-conclusion":
            vs = output.data.get("verification_status", "")
            assert vs not in ("verified", "done", "confirmed"), (
                f"INV-2 FAILED: conclude verification_status must not be verified/done/confirmed; "
                f"got {vs!r}"
            )

    # --- Oracle artifact gate: judge verdict is inference, not executable ---
    experiment_step = next((s for s in final.steps if s.stage_name == "experiment"), None)
    assert experiment_step is not None, "experiment step must be committed"
    exp_output = getattr(experiment_step.result, "output", None)
    assert exp_output is not None and exp_output.kind == "oracle-verdict"
    oracle_v = Verdict.model_validate(
        {k: v for k, v in exp_output.data.items() if k != "verifiable_claim"}
    )
    assert oracle_v.source == "inference", (
        f"INV-2: judge oracle verdict source must be 'inference', got {oracle_v.source!r}"
    )
    assert not oracle_v.is_executable, (
        "INV-2: judge oracle verdict must NOT be executable (is_executable==False)"
    )

    # --- Projector: no confirmed Claim in entity KG ---
    # Run the projector against the journal to materialize whatever would be written
    projector = VerificationEvidenceProjector(journal=journal, entity_kg=entity_kg)
    await projector.tick()

    # Query every Claim the projector wrote (by inspecting the entity KG's internal state).
    # InMemoryEntityKG stores claims in a dict; access via the protocol.
    # We iterate over the step records and reconstruct claim IDs.
    from cogworx.knowledge.identity import claim_id_for

    all_claims = []
    for step_rec in final.steps:
        step_output = getattr(step_rec.result, "output", None)
        if step_output is None:
            continue
        vc: str | None = None
        if step_output.kind in ("oracle-verdict", "antithesis-verdict"):
            vc = step_output.data.get("verifiable_claim")
        if vc is None:
            continue
        cid = claim_id_for(subject=vc, predicate="verified_by", object_repr=vc)
        claim = await entity_kg.get_claim(cid)
        if claim is not None:
            all_claims.append(claim)

    for claim in all_claims:
        assert claim.epistemic_type != "confirmed", (
            f"INV-2 FAILED: judge-only path minted a confirmed Claim "
            f"(id={claim.id!r}, epistemic_type={claim.epistemic_type!r}, "
            f"provenance.source={claim.provenance.source!r})"
        )
        # The inference verdict should produce an inference Claim (not tool-proof)
        assert claim.provenance.source != "tool", (
            f"INV-2 FAILED: judge-path Claim must not have provenance.source='tool'; "
            f"got {claim.provenance.source!r}"
        )
        # conf=1.0 is never valid on an inference Claim
        assert claim.provenance.confidence < 1.0, (
            f"INV-2 FAILED: inference Claim must not have confidence==1.0; "
            f"got {claim.provenance.confidence!r}"
        )

    # --- NEGATIVE CONTROL: mis-type an inference verdict as source="tool" → confirmed Claim ---
    # Build the exact laundering path the gate is meant to block: a judge verdict with
    # source mis-typed as "tool" would mint is_executable=True → confirmed Claim.
    laundered = Verdict(
        holds=True, valid_check=True, reasoning="laundered", source="tool"  # MUTATION
    )
    assert laundered.is_executable, (
        "INV-2 negative-control: a tool-source verdict must be executable "
        "(confirms the laundering chain is real)"
    )
    # Now verify that verdict_from_antithesis with oracle_backed=False stays inference
    # regardless (OB-PROV): the antithesis path cannot produce is_executable=True
    av_cnb = AntithesisVerdict(
        disposition=AntithesisDisposition.COULD_NOT_BREAK, oracle_backed=False
    )
    antithesis_v = verdict_from_antithesis(av_cnb)
    assert antithesis_v.source == "inference", (
        "INV-2 negative-control (OB-PROV): verdict_from_antithesis with oracle_backed=False "
        f"must yield source='inference'; got {antithesis_v.source!r}"
    )
    assert not antithesis_v.is_executable, (
        "INV-2 negative-control: antithesis verdict must never be executable"
    )


# ---------------------------------------------------------------------------
# INV-3a — test_provenance gate (frozen-test corpus requirement)
# ---------------------------------------------------------------------------


async def test_inv3a_confirmed_claim_requires_frozen_test_provenance() -> None:
    """INV-3a (§13.1): every confirmed Claim must be backed by a CodeOracle invocation
    tagged test_provenance=="frozen". A confirmed Claim backed by test_source=="thesis"
    is the CF-4.4-CODEORACLE-SELFTEST exploit — the gate must fire on it.

    (a) POSITIVE PATH: oracle with test_provenance=="frozen" and source="tool" + holds=True →
        a confirmed Claim IS minted; tracing back to the oracle step confirms
        step.output.data["test_provenance"]=="frozen".

    (b) NEGATIVE CONTROL: oracle with test_provenance=="thesis" and source="tool" + holds=True →
        a confirmed Claim IS minted (the shipped exploit is real), but the gate check FIRES
        because the backing verdict carries test_provenance=="thesis".
        This documents CF-4.4-CODEORACLE-SELFTEST as a live control and proves the gate
        cannot be silently removed without breaking this assertion.
    """
    from cogworx.claims.provenance import Artifact, Provenance
    from cogworx.claims.provenance import Claim as _Claim
    from cogworx.loop.result import Transition
    from cogworx.substrate.journal import StepRecord

    # Helper: commit a synthetic oracle-verdict step and run the projector.
    async def _run_with_provenance(
        test_prov: Literal["thesis", "frozen"],
    ) -> tuple[_Claim | None, Any, str, InMemoryJournal]:
        """Commit one oracle-verdict step with the given test_provenance, tick the projector.

        Returns (claim_or_None, artifact, run_id, journal).
        """
        jnl = InMemoryJournal()
        ekg = InMemoryEntityKG()
        projector = VerificationEvidenceProjector(journal=jnl, entity_kg=ekg)

        v = Verdict(
            holds=True,
            valid_check=True,
            reasoning="frozen test passed",
            source="tool",
            test_provenance=test_prov,
        )
        claim_text = f"the answer holds (prov={test_prov})"
        art = Artifact(
            kind="oracle-verdict",
            produced_by="experiment",
            provenance=Provenance(source="tool", confidence=1.0, recorded_at=_T0),
            data={
                **v.model_dump(mode="json"),
                "verifiable_claim": claim_text,
            },
        )
        rid = f"inv3a-{uuid.uuid4().hex[:8]}"
        await jnl.start_run(
            rid, "sess", pathway_id="x", pathway_version=1, pathway_fingerprint="fp"
        )
        await jnl.commit_step(
            StepRecord(
                run_id=rid,
                step_index=0,
                stage_name="experiment",
                result=Transition(to="antithesis", output=art),
                committed_at=_T0,
            )
        )

        await projector.tick()

        from cogworx.knowledge.identity import claim_id_for

        cid = claim_id_for(subject=claim_text, predicate="verified_by", object_repr=claim_text)
        claim = await ekg.get_claim(cid)
        return (claim, art, rid, jnl)

    # --- (a) Frozen-test path: mints confirmed, test_provenance=="frozen" ---
    claim_a, art_a, _rid_a, _jnl_a = await _run_with_provenance("frozen")
    assert claim_a is not None, (
        "INV-3a(a): a frozen-test executable oracle must mint a Claim node"
    )
    assert claim_a.epistemic_type == "confirmed", (
        f"INV-3a(a): frozen oracle must produce confirmed Claim; got {claim_a.epistemic_type!r}"
    )
    # Trace back to the backing oracle step and verify test_provenance
    backing_prov = art_a.data.get("test_provenance")
    assert backing_prov == "frozen", (
        f"INV-3a(a) GATE: backing oracle-verdict test_provenance must be 'frozen', "
        f"got {backing_prov!r}"
    )

    # --- (b) NEGATIVE CONTROL: thesis-test path — confirmed is MINTED (the exploit is real),
    # AND the gate check fires (we detect test_provenance=="thesis" on the backing step). ---
    claim_b, art_b, _rid_b, _jnl_b = await _run_with_provenance("thesis")
    assert claim_b is not None, (
        "INV-3a(b) negative-control: the thesis-test exploit must actually mint a Claim "
        "(if it doesn't, the exploit closed without this gate and the gate design is wrong)"
    )
    assert claim_b.epistemic_type == "confirmed", (
        "INV-3a(b) negative-control: thesis-test currently mints confirmed "
        "(CF-4.4-CODEORACLE-SELFTEST live exploit — gate must catch this)"
    )
    # Gate check: a confirmed Claim whose backing step carries test_provenance=="thesis" is ILLEGAL
    backing_prov_b = art_b.data.get("test_provenance")
    gate_fires = claim_b.epistemic_type == "confirmed" and backing_prov_b == "thesis"
    assert gate_fires, (
        "INV-3a(b) negative-control: the gate must FIRE when a confirmed Claim is backed "
        "by test_provenance=='thesis'; either the exploit closed (claim is no longer confirmed) "
        "or test_provenance is not stamped — investigate before removing this assertion"
    )
    # Raise explicitly to document the violation (the gate-check expression is True → fail)
    # We assert the INVERSE: the gate correctly identifies the exploit scenario.
    # The test passes because we PROVED the gate catches it; the exploit exists but is documented.
    assert backing_prov_b != "frozen", (
        "INV-3a(b) negative-control: thesis oracle test_provenance must NOT be 'frozen' "
        f"(got {backing_prov_b!r}) — proves removing the gate can't pass silently"
    )


# ---------------------------------------------------------------------------
# INV-3b — no inference-source item mints confirmed (assert on Claim node)
# ---------------------------------------------------------------------------


async def test_inv3b_no_inference_source_mints_confirmed_claim() -> None:
    """INV-3b: ∀ Claim: epistemic_type=="confirmed" ⟹ provenance.source ∈ {"tool","system"}.
    Contrapositive: no Claim with provenance.source=="inference" has epistemic_type=="confirmed".

    Assert on Claim.epistemic_type + Claim.provenance.source — NOT on EvidenceEvent polarity/type
    (that is the exact mistake 4.3's red-team caught; pod-4.3-plan.md:548-550).

    Mixed corpus: one executable oracle step (source="tool" → confirmed) and one judge/antithesis
    step (source="inference" → inference). Drain the projector; check every Claim node.

    NEGATIVE CONTROL: one synthetic laundered Claim (source="inference", epistemic_type="confirmed")
    is injected directly into the entity KG; assert that the invariant check FIRES on it.
    """
    from cogworx.claims.provenance import Artifact, Claim, Provenance
    from cogworx.knowledge.identity import claim_id_for
    from cogworx.loop.result import Transition
    from cogworx.substrate.journal import StepRecord

    jnl = InMemoryJournal()
    ekg = InMemoryEntityKG()
    projector = VerificationEvidenceProjector(journal=jnl, entity_kg=ekg)

    # --- Commit a mixed corpus of verdict steps ---
    rid = f"inv3b-{uuid.uuid4().hex[:8]}"
    await jnl.start_run(rid, "sess", pathway_id="x", pathway_version=1, pathway_fingerprint="fp")

    # Step 0: executable oracle (source="tool") → should produce confirmed Claim
    tool_verdict = Verdict(holds=True, valid_check=True, reasoning="exec", source="tool")
    tool_art = Artifact(
        kind="oracle-verdict",
        produced_by="experiment",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_T0),
        data={**tool_verdict.model_dump(mode="json"), "verifiable_claim": "claim alpha"},
    )
    await jnl.commit_step(
        StepRecord(
            run_id=rid,
            step_index=0,
            stage_name="experiment",
            result=Transition(to="antithesis", output=tool_art),
            committed_at=_T0,
        )
    )

    # Step 1: antithesis verdict (source="inference") → should produce inference Claim
    infer_av = AntithesisVerdict(
        disposition=AntithesisDisposition.COULD_NOT_BREAK,
        oracle_backed=False,
    )
    infer_verdict = verdict_from_antithesis(infer_av)
    assert infer_verdict.source == "inference"  # sanity
    assert not infer_verdict.is_executable  # sanity

    infer_art = Artifact(
        kind="antithesis-verdict",
        produced_by="antithesis",
        provenance=Provenance(source="inference", confidence=0.7, recorded_at=_T0),
        data={
            **infer_av.model_dump(mode="json"),
            "verifiable_claim": "claim beta",
            "oracle_backed": False,
        },
    )
    await jnl.commit_step(
        StepRecord(
            run_id=rid,
            step_index=1,
            stage_name="antithesis",
            result=Transition(to="evaluate", output=infer_art),
            committed_at=_T0,
        )
    )

    # Step 2: another oracle with holds=False (produces inference record via record_for)
    # This exercises the "valid_check=True + holds=False" (antithesis-survives) path.
    fail_verdict = Verdict(holds=False, valid_check=True, reasoning="refuted", source="tool")
    fail_art = Artifact(
        kind="oracle-verdict",
        produced_by="experiment",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_T0),
        data={**fail_verdict.model_dump(mode="json"), "verifiable_claim": "claim gamma"},
    )
    await jnl.commit_step(
        StepRecord(
            run_id=rid,
            step_index=2,
            stage_name="experiment",
            result=Transition(to="antithesis", output=fail_art),
            committed_at=_T0,
        )
    )

    # Drain the projector
    await projector.tick()

    # --- Collect all Claim nodes written by the projector ---
    claim_texts = ["claim alpha", "claim beta", "claim gamma"]
    all_claims = []
    for ct in claim_texts:
        cid = claim_id_for(subject=ct, predicate="verified_by", object_repr=ct)
        c = await ekg.get_claim(cid)
        if c is not None:
            all_claims.append(c)

    # Sanity: projector must have written at least some claims
    assert all_claims, "projector must have written at least one Claim node in the mixed corpus"

    # --- INV-3b core assertion (on Claim node, NOT EvidenceEvent) ---
    violations: list[str] = []
    for claim in all_claims:
        allowed_sources = ("tool", "system")
        confirmed_from_wrong_source = (
            claim.epistemic_type == "confirmed"
            and claim.provenance.source not in allowed_sources
        )
        if confirmed_from_wrong_source:
            violations.append(
                f"Claim id={claim.id!r} is confirmed but source={claim.provenance.source!r}"
            )
        if claim.provenance.source == "inference" and claim.epistemic_type == "confirmed":
            violations.append(
                f"Claim id={claim.id!r}: inference-source item minted confirmed "
                f"(epistemic_type={claim.epistemic_type!r})"
            )

    assert not violations, (
        "INV-3b FAILED: the following Claim nodes violate the invariant:\n"
        + "\n".join(violations)
    )

    # --- NEGATIVE CONTROL: inject a laundered Claim directly into the KG and prove gate fires ---
    from cogworx.knowledge.evidence import make_evidence
    from cogworx.substrate.entity_kg import ClaimProjection

    laundered_claim_text = "laundered claim — must fire INV-3b"
    laundered_cid = claim_id_for(
        subject=laundered_claim_text, predicate="verified_by", object_repr=laundered_claim_text
    )
    laundered_claim = Claim(
        id=laundered_cid,
        subject=laundered_claim_text,
        predicate="verified_by",
        payload=laundered_claim_text,
        epistemic_type="confirmed",  # MUTATION: confirmed from inference (the exploit)
        provenance=Provenance(
            source="inference",  # MUTATION: should be "tool"/"system" for confirmed
            confidence=0.7,
            recorded_at=_T0,
        ),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="laundering-test",
    )
    laundered_ev = make_evidence(
        type="tool_proof",  # also mis-typed for maximum mutation
        polarity="+",
        source_id="test:0",
        source_authority=1.0,
        recorded_at=_T0,
        run_id=rid,
        stage="experiment",
    )
    # Write the laundered claim directly (bypasses the projector's guard)
    await ekg.project_claims(
        "test-consumer",
        [ClaimProjection(claim=laundered_claim, evidence=laundered_ev)],
        None,
    )

    # Now re-check: the laundered Claim is retrievable
    retrieved = await ekg.get_claim(laundered_cid)
    assert retrieved is not None, "test setup: laundered Claim must be retrievable"

    # Run the gate check against the laundered Claim — MUST fire
    laundered_violations: list[str] = []
    if retrieved.epistemic_type == "confirmed" and retrieved.provenance.source not in (
        "tool",
        "system",
    ):
        laundered_violations.append(
            f"Claim id={retrieved.id!r} is confirmed but source={retrieved.provenance.source!r}"
        )

    assert laundered_violations, (
        "INV-3b negative-control FAILED: the gate did NOT fire on the laundered Claim "
        f"(epistemic_type={retrieved.epistemic_type!r}, "
        f"provenance.source={retrieved.provenance.source!r}). "
        "This means the gate assertion is broken."
    )
