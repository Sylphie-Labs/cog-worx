"""Reusable CANON invariant property-suites (CANON S1, S5, S6, S9).

These are the mechanical checks the Feature Test Bundle (ROADMAP, Definition of Done) auto-applies
to every feature that touches the corresponding invariant. They are importable assertion utilities,
``Journal`` wrappers — not one-off tests — so a downstream pod wires its feature into the reference
loop and gets the invariants enforced for free:

- **S1** — no model call on the write path: ``CommitSpyJournal`` snapshots the model's call count
  around every ``commit_step`` and trips if a commit invoked the model.
- **S5** — every substrate write carries provenance + epistemic type: structural walk over a run's
  committed step outputs, plus a constructive proof that the claim/artifact types reject a missing
  provenance.
- **S6** — resume never re-calls the model: ``CrashAfterStepJournal`` durably commits a step then
  simulates the process dying; the chaos harness resumes over the same journal with a zero-response
  ``ReplayModel`` and proves the committed work is replayed, not recomputed.
- **S9** — control flow is a function of ``StageResult`` + graph, never of the model's words: run
  the same graph under two wildly different model texts and assert an identical committed path.

A feature that cannot be driven through these wrappers is a coupling smell (S8): it cannot be
lesioned, so it does not harden (S12). Surface that to ``architect`` rather than weaken the suite.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from datetime import UTC, datetime, timedelta

import pydantic

from cogworx.claims.provenance import Artifact, Claim
from cogworx.knowledge.evidence import EVIDENCE_BASE_WEIGHTS, EvidenceEvent
from cogworx.loop.state import RunStatus
from cogworx.runtime.engine import Engine
from cogworx.substrate.entity_kg import ClaimProjection, EntityKG, ScoredClaim
from cogworx.substrate.journal import (
    Journal,
    ProjectedStep,
    ProjectionCursor,
    RunState,
    StepRecord,
    Timer,
)
from cogworx.testing.fake_model import ReplayModel

EngineFactory = Callable[[Journal, ReplayModel], Engine]
"""Builds an ``Engine`` given the journal it should drive and the model to spy on/exhaust.

The factory owns wiring the ``PathwayRegistry`` into the engine, so the resume harness can build a
FRESH engine B over the same registry — modelling true cold cross-process resume with no in-process
graph carried over."""


class InvariantViolation(AssertionError):
    """Raised when a CANON invariant property-suite is violated."""


class SimulatedCrash(Exception):
    """A simulated process death used by the S6 chaos harness (not a real error)."""


# --------------------------------------------------------------------------------------------------
# S1 — no model call on the write path
# --------------------------------------------------------------------------------------------------


class CommitSpyJournal:
    """A ``Journal`` wrapper that fails if a model call happens during a commit (S1).

    Wraps an inner ``Journal`` and the ``ReplayModel`` spy driving the same run. On every
    ``commit_step`` it snapshots ``model.call_count`` immediately before and after the inner commit
    and asserts it did not increase — a model call on the write/commit path is an S1 violation. All
    other journal methods delegate to the inner journal unchanged, so any run driven through this
    wrapper gets S1 enforced mechanically.
    """

    def __init__(self, *, inner: Journal, model: ReplayModel) -> None:
        self._inner = inner
        self._model = model

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        await self._inner.start_run(
            run_id,
            session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint,
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._inner.set_run_status(run_id, status)

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def commit_step(self, record: StepRecord) -> None:
        before = self._model.call_count
        await self._inner.commit_step(record)
        after = self._model.call_count
        if after != before:
            raise InvariantViolation(
                "S1 violation: the model was called on the write path during commit of step "
                f"{record.stage_name!r} (call_count {before} -> {after}); writes must be a pure "
                "persist with no model-heavy work on the hot path"
            )

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

    async def committed_steps_after(
        self, cursor: ProjectionCursor | None, *, limit: int
    ) -> Sequence[ProjectedStep]:
        return await self._inner.committed_steps_after(cursor, limit=limit)

    async def load_run(self, run_id: str) -> RunState | None:
        return await self._inner.load_run(run_id)

    async def set_timer(self, timer: Timer) -> None:
        await self._inner.set_timer(timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return await self._inner.due_timers(now)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        return await self._inner.claim_due_timers(now, lease_ttl=lease_ttl)

    async def cancel_timer(self, timer_id: str) -> None:
        await self._inner.cancel_timer(timer_id)

    async def cancel_timers_for_run(self, run_id: str) -> None:
        await self._inner.cancel_timers_for_run(run_id)

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return await self._inner.get_run_status(run_id)

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.increment_attempt(run_id, step_index)

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.read_attempt(run_id, step_index)

    async def record_human_input(self, run_id: str, step_index: int, answer: Artifact) -> None:
        await self._inner.record_human_input(run_id, step_index, answer)

    async def read_human_input(self, run_id: str, step_index: int) -> Artifact | None:
        return await self._inner.read_human_input(run_id, step_index)

    async def set_run_tainted(self, run_id: str) -> None:
        await self._inner.set_run_tainted(run_id)

    async def append_design_look(
        self,
        *,
        planning_variance_config_hash: str,
        design_lineage_chain: Sequence[str],
        fingerprint: str,
    ) -> None:
        await self._inner.append_design_look(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,
            fingerprint=fingerprint,
        )

    async def read_design_lineage_budget(
        self,
        *,
        planning_variance_config_hash: str,
        design_lineage_chain: Sequence[str],
    ) -> int:
        return await self._inner.read_design_lineage_budget(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,
        )


async def assert_no_model_on_write_path(
    *,
    engine_factory: EngineFactory,
    inner_journal: Journal,
    model: ReplayModel,
    pathway_id: str,
    initial: Artifact,
    run_id: str = "s1-run",
    session_id: str = "s1-sess",
) -> RunState:
    """Drive a run through a ``CommitSpyJournal`` and assert the spy never trips (S1).

    ``engine_factory(journal, model)`` builds an ``Engine`` (with its ``PathwayRegistry`` wired)
    over the given journal and model; the factory lets the caller wire their feature's pathway while
    this helper owns the spying journal. ``pathway_id`` names the registered pathway to run; the
    final ``RunState`` is returned on success.
    """
    spy = CommitSpyJournal(inner=inner_journal, model=model)
    engine = engine_factory(spy, model)
    return await engine.run(
        run_id=run_id, session_id=session_id, pathway_id=pathway_id, initial=initial
    )


# --------------------------------------------------------------------------------------------------
# S5 — every substrate write carries provenance + epistemic type
# --------------------------------------------------------------------------------------------------


def assert_run_writes_carry_provenance(state: RunState) -> None:
    """Assert every committed step output in ``state`` carries provenance (S5).

    Walks ``state.steps``; for each step's ``result`` that carries an ``output`` artifact
    (transition / done / degraded always do; await-human may be ``None``) asserts the artifact's
    ``provenance`` is present. Artifacts require provenance by type, so this is a structural
    belt-and-suspenders check that also documents the invariant at the run level.
    """
    for step in state.steps:
        result = step.result
        output = getattr(result, "output", None)
        if output is None:
            continue
        if output.provenance is None:
            raise InvariantViolation(
                f"S5 violation: step {step.stage_name!r} committed an output artifact with no "
                "provenance; every substrate write must carry provenance + epistemic type"
            )


def assert_claim_requires_provenance() -> None:
    """Prove the claim/artifact types reject a missing provenance (S5).

    Constructs a ``Claim`` and an ``Artifact`` without provenance and asserts each raises
    ``pydantic.ValidationError`` — the type, not a runtime check, is what enforces S5.
    """
    # Build via ``model_validate`` from a dict deliberately MISSING ``provenance`` — this exercises
    # the type's required-field enforcement without a source-level ``# type: ignore``.
    raised_for_artifact = False
    try:
        Artifact.model_validate({"kind": "x", "produced_by": "test"})
    except pydantic.ValidationError:
        raised_for_artifact = True
    if not raised_for_artifact:
        raise InvariantViolation(
            "S5 violation: Artifact accepted construction without provenance; the type must "
            "require provenance on every claim-bearing write"
        )

    raised_for_claim = False
    try:
        Claim.model_validate(
            {
                "id": "c1",
                "subject": "s",
                "payload": "p",
                "epistemic_type": "inference",
                "valid_from": datetime(2026, 1, 1),
                "ingest_time": datetime(2026, 1, 1),
                "created_by": "test",
            }
        )
    except pydantic.ValidationError:
        raised_for_claim = True
    if not raised_for_claim:
        raise InvariantViolation(
            "S5 violation: Claim accepted construction without provenance; the type must require "
            "provenance + epistemic typing on every claim"
        )


# --------------------------------------------------------------------------------------------------
# S6 — resume never re-calls the model (the durability invariant)
# --------------------------------------------------------------------------------------------------


class CrashAfterStepJournal:
    """A ``Journal`` wrapper that durably commits, then simulates a crash after a chosen stage (S6).

    Delegates ``commit_step`` to the inner journal so the step IS durably committed, then — if the
    just-committed step's ``stage_name`` equals ``crash_after_stage`` — raises ``SimulatedCrash`` to
    model the process dying after the commit but before the runner advances. All other methods
    delegate. Driving an engine through this wrapper produces a journal that holds the committed
    prefix exactly as a real crash would leave it, so a fresh engine can resume over the same inner
    journal and prove exactly-once / no-model-recall behaviour.
    """

    def __init__(self, *, inner: Journal, crash_after_stage: str) -> None:
        self._inner = inner
        self._crash_after_stage = crash_after_stage

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        await self._inner.start_run(
            run_id,
            session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            pathway_fingerprint=pathway_fingerprint,
        )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        await self._inner.set_run_status(run_id, status)

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        return await self._inner.compare_and_set_run_status(run_id, expect=expect, new=new)

    async def commit_step(self, record: StepRecord) -> None:
        await self._inner.commit_step(record)
        if record.stage_name == self._crash_after_stage:
            raise SimulatedCrash(
                f"simulated crash after durable commit of stage {record.stage_name!r}"
            )

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        return await self._inner.read_step(run_id, step_index)

    async def committed_steps_after(
        self, cursor: ProjectionCursor | None, *, limit: int
    ) -> Sequence[ProjectedStep]:
        return await self._inner.committed_steps_after(cursor, limit=limit)

    async def load_run(self, run_id: str) -> RunState | None:
        return await self._inner.load_run(run_id)

    async def set_timer(self, timer: Timer) -> None:
        await self._inner.set_timer(timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return await self._inner.due_timers(now)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        return await self._inner.claim_due_timers(now, lease_ttl=lease_ttl)

    async def cancel_timer(self, timer_id: str) -> None:
        await self._inner.cancel_timer(timer_id)

    async def cancel_timers_for_run(self, run_id: str) -> None:
        await self._inner.cancel_timers_for_run(run_id)

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        return await self._inner.get_run_status(run_id)

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.increment_attempt(run_id, step_index)

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        return await self._inner.read_attempt(run_id, step_index)

    async def record_human_input(self, run_id: str, step_index: int, answer: Artifact) -> None:
        await self._inner.record_human_input(run_id, step_index, answer)

    async def read_human_input(self, run_id: str, step_index: int) -> Artifact | None:
        return await self._inner.read_human_input(run_id, step_index)

    async def set_run_tainted(self, run_id: str) -> None:
        await self._inner.set_run_tainted(run_id)

    async def append_design_look(
        self,
        *,
        planning_variance_config_hash: str,
        design_lineage_chain: Sequence[str],
        fingerprint: str,
    ) -> None:
        await self._inner.append_design_look(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,
            fingerprint=fingerprint,
        )

    async def read_design_lineage_budget(
        self,
        *,
        planning_variance_config_hash: str,
        design_lineage_chain: Sequence[str],
    ) -> int:
        return await self._inner.read_design_lineage_budget(
            planning_variance_config_hash=planning_variance_config_hash,
            design_lineage_chain=design_lineage_chain,
        )


_TERMINAL_STATUSES = (
    RunStatus.COMPLETED,
    RunStatus.AWAITING_HUMAN,
    RunStatus.DEGRADED,
    RunStatus.FAILED,
)


async def assert_resume_never_recalls_model(
    *,
    build_engine: Callable[[Journal, ReplayModel], Engine],
    initial: Artifact,
    pathway_id: str,
    crash_after_stage: str,
    shared_journal: Journal,
    scripted_model: ReplayModel,
    pathway_version: int = 1,
    run_id: str = "s6-run",
    session_id: str = "s6-sess",
) -> RunState:
    """The reusable kill-mid-run chaos harness (S6) — COLD resume over a durable journal.

    1. Build engine **A** over a ``CrashAfterStepJournal(inner=shared_journal, ...)`` and the
       ``scripted_model``; ``run`` the named ``pathway_id`` and assert it raises ``SimulatedCrash``
       — the ``crash_after_stage`` step is durably committed to ``shared_journal``, then the run is
       killed mid-flight (the crash models the process dying after commit, before the run advances).
    2. Build a FRESH engine **B** over the SAME ``shared_journal`` (now holding the committed
       prefix) with a fresh **zero-response** ``ReplayModel`` (so ANY model call raises
       ``ReplayExhaustedError``) and ``resume(run_id)``. Engine B carries NO in-process graph: it
       rehydrates the pathway from its ``PathwayRegistry`` using the run's stored ``pathway_id``
       pointer — true cold cross-process resume. ``build_engine`` MUST wire the SAME registry into
       both engines (that is the only shared state besides the journal).
    3. Assert resume reaches a terminal status (normally ``COMPLETED``), engine B's model
       ``call_count == 0`` (no committed model-bearing stage re-invoked), and the steps committed
       before the crash are byte-for-byte identical in the journal after resume (read both ways).

    The caller shapes the registered pathway so the crash lands AFTER a model-bearing stage commits
    and BEFORE a later stage — then a zero-response model on resume proves the committed model call
    was never repeated. ``build_engine(journal, model)`` builds an ``Engine`` with the pathway
    registry wired in.
    """
    # 1. Engine A: run until the durable-commit-then-crash.
    crash_journal = CrashAfterStepJournal(inner=shared_journal, crash_after_stage=crash_after_stage)
    engine_a = build_engine(crash_journal, scripted_model)
    crashed = False
    try:
        await engine_a.run(
            run_id=run_id,
            session_id=session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            initial=initial,
        )
    except SimulatedCrash:
        crashed = True
    if not crashed:
        raise InvariantViolation(
            "S6 harness: engine A did not crash; the run completed before "
            f"crash_after_stage={crash_after_stage!r} could fire"
        )

    pre_resume = await shared_journal.load_run(run_id)
    if pre_resume is None:
        raise InvariantViolation("S6 harness: journal lost the run after the simulated crash")
    committed_before = pre_resume.steps
    crash_index = _index_of_stage(committed_before, crash_after_stage)
    if crash_index is None:
        raise InvariantViolation(
            f"S6 harness: stage {crash_after_stage!r} was not durably committed before the crash"
        )

    # 2. Engine B: a FRESH engine resumes over the SAME journal with a zero-response model (any call
    #    raises). The graph is rehydrated from the registry via the run's stored pathway pointer.
    zero_response_model = ReplayModel([])
    engine_b = build_engine(shared_journal, zero_response_model)
    final = await engine_b.resume(run_id)

    # 3. Assertions: terminal, no model re-call, replayed steps unchanged.
    if final.status not in _TERMINAL_STATUSES:
        raise InvariantViolation(
            f"S6 violation: resume did not reach a terminal status (got {final.status!r})"
        )
    if zero_response_model.call_count != 0:
        raise InvariantViolation(
            f"S6 violation: resume re-called the model {zero_response_model.call_count} time(s); "
            "a step committed before the crash must be replayed from the journal, never recomputed"
        )

    replayed = final.steps[: len(committed_before)]
    if tuple(replayed) != tuple(committed_before):
        raise InvariantViolation(
            "S6 violation: replayed step records differ from what was committed before the crash; "
            "resume must read stored outputs verbatim"
        )
    return final


def _index_of_stage(steps: Sequence[StepRecord], stage_name: str) -> int | None:
    for index, step in enumerate(steps):
        if step.stage_name == stage_name:
            return index
    return None


# --------------------------------------------------------------------------------------------------
# S9 — no self-report used as a control signal
# --------------------------------------------------------------------------------------------------


async def assert_control_independent_of_model_text(
    *,
    build_engine: Callable[[Journal, ReplayModel], Engine],
    pathway_id: str,
    initial: Artifact,
    journal_factory: Callable[[], Journal],
    model_a: ReplayModel,
    model_b: ReplayModel,
    pathway_version: int = 1,
    run_id: str = "s9-run",
    session_id: str = "s9-sess",
) -> None:
    """Assert the committed control path is identical under wildly different model text (S9).

    Runs the SAME registered pathway twice — once with ``model_a`` (whose response text screams
    control words like "STOP" / "transition to intake" / "confidence 0.0") and once with ``model_b``
    (plain) — each over its own fresh journal from ``journal_factory``. The model's words must never
    steer the loop: control flow is a function of ``StageResult`` + graph. Asserts the SEQUENCE of
    committed ``stage_name``s is identical across both runs. ``build_engine`` wires the
    ``PathwayRegistry`` (holding ``pathway_id``) into the engine.

    ``model_a`` and ``model_b`` must be scripted to return responses that shape the SAME
    ``StageResult`` (same transitions / terminal kinds) — only the free-text differs — so any
    divergence in the committed path is attributable to the model's words, which would be the
    violation.
    """

    async def _control_path(model: ReplayModel) -> tuple[str, ...]:
        journal = journal_factory()
        engine = build_engine(journal, model)
        state = await engine.run(
            run_id=run_id,
            session_id=session_id,
            pathway_id=pathway_id,
            pathway_version=pathway_version,
            initial=initial,
        )
        return tuple(step.stage_name for step in state.steps)

    path_a = await _control_path(model_a)
    path_b = await _control_path(model_b)
    if path_a != path_b:
        raise InvariantViolation(
            "S9 violation: the committed control path changed with the model's response text "
            f"({path_a} vs {path_b}); control flow must be a function of StageResult + graph, "
            "never of the model's self-report"
        )


# --------------------------------------------------------------------------------------------------
# S5 — RecordingEntityKG + substrate invariant auditor
# --------------------------------------------------------------------------------------------------


class RecordingEntityKG:
    """EntityKG wrapper that records every claim id that crosses a write surface (S5 audit).

    Delegates all EntityKG methods to the inner instance, intercepting the write surfaces to record
    the claim ids they touch:

    - ``write_claim``: records the returned claim id.
    - ``project_claims``: records each ``ClaimProjection.claim.id`` before delegation.
    - ``add_evidence``: records the ``claim_id`` argument.

    All other methods delegate to the inner instance unchanged. The recorded set is append-only so
    the auditor can verify provenance completeness post-run without modifying any claim data.
    """

    def __init__(self, inner: EntityKG) -> None:
        self._inner = inner
        self.recorded_claim_ids: set[str] = set()

    async def write_claim(self, claim: Claim, *, evidence: EvidenceEvent) -> str:
        """Delegate to inner and record the returned claim id."""
        cid = await self._inner.write_claim(claim, evidence=evidence)
        self.recorded_claim_ids.add(cid)
        return cid

    async def add_evidence(self, claim_id: str, event: EvidenceEvent) -> None:
        """Delegate to inner and record the claim_id."""
        await self._inner.add_evidence(claim_id, event)
        self.recorded_claim_ids.add(claim_id)

    async def project_claims(
        self,
        consumer: str,
        writes: Sequence[ClaimProjection],
        progress: ProjectionCursor | None,
    ) -> None:
        """Record each projected claim id, then delegate to inner."""
        for cp in writes:
            self.recorded_claim_ids.add(cp.claim.id)
        await self._inner.project_claims(consumer, writes, progress)

    async def get_claim(self, claim_id: str) -> Claim | None:
        return await self._inner.get_claim(claim_id)

    async def evidence_for(self, claim_id: str) -> Sequence[EvidenceEvent]:
        return await self._inner.evidence_for(claim_id)

    async def claims_about(
        self,
        entity: str,
        *,
        limit: int = 20,
        as_of: datetime | None = None,
        scope: str | None = None,
    ) -> Sequence[ScoredClaim]:
        return await self._inner.claims_about(entity, limit=limit, as_of=as_of, scope=scope)

    async def claims_by_similarity(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
        scope: str | None = None,
    ) -> Sequence[ScoredClaim]:
        return await self._inner.claims_by_similarity(
            embedding, k=k, min_score=min_score, scope=scope
        )

    async def claims_full_text(
        self,
        query: str,
        *,
        k: int = 10,
        scope: str | None = None,
        as_of: datetime | None = None,
    ) -> Sequence[ScoredClaim]:
        return await self._inner.claims_full_text(query, k=k, scope=scope, as_of=as_of)

    async def resolution_candidates(
        self,
        subject: str,
        predicate: str,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
        scope: str | None = None,
    ) -> Sequence[Claim]:
        return await self._inner.resolution_candidates(
            subject, predicate, embedding=embedding, k=k, scope=scope
        )

    async def write_contradiction(self, claim_id_a: str, claim_id_b: str) -> None:
        await self._inner.write_contradiction(claim_id_a, claim_id_b)

    async def contradictions_of(self, claim_id: str) -> Sequence[Claim]:
        return await self._inner.contradictions_of(claim_id)

    async def invalidate_claim(self, claim_id: str, *, valid_to: datetime) -> None:
        await self._inner.invalidate_claim(claim_id, valid_to=valid_to)

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        return await self._inner.read_cursor(consumer)


async def assert_s5_substrate_invariants(
    kg: EntityKG,
    claim_ids: Iterable[str],
) -> None:
    """Assert S5 invariants for every claim id. Raises InvariantViolation on first failure.

    For each claim id checks:
      1. The claim exists in the KG.
      2. At least one evidence event is attached.
      3. Each evidence event has a known type, positive weight, non-empty source_id, and
         non-None recorded_at.
      4. The claim's epistemic_type is one of the recognised values.
      5. The claim's provenance.source is non-None and non-empty.
      6. The claim's valid_from is not None and is timezone-aware UTC.
    """
    _known_epistemic = {"observation", "inference", "confirmed"}

    for claim_id in claim_ids:
        claim = await kg.get_claim(claim_id)
        if claim is None:
            raise InvariantViolation(f"S5: claim {claim_id} not found")

        evidence = await kg.evidence_for(claim_id)
        if len(evidence) < 1:
            raise InvariantViolation(f"S5: claim {claim_id} has no evidence")

        for event in evidence:
            if event.type not in EVIDENCE_BASE_WEIGHTS:
                raise InvariantViolation(
                    f"S5: unknown evidence type {event.type!r} on claim {claim_id}"
                )
            if event.base_weight <= 0:
                raise InvariantViolation(
                    f"S5: non-positive base_weight {event.base_weight} on claim {claim_id}"
                )
            if not event.source_id:
                raise InvariantViolation(f"S5: empty source_id on evidence for claim {claim_id}")
            if event.recorded_at is None:
                raise InvariantViolation(f"S5: None recorded_at on evidence for claim {claim_id}")

        if claim.epistemic_type not in _known_epistemic:
            raise InvariantViolation(
                f"S5: unknown epistemic_type {claim.epistemic_type!r} on claim {claim_id}"
            )

        if not claim.provenance.source:
            raise InvariantViolation(f"S5: empty provenance.source on claim {claim_id}")

        if claim.valid_from is None:
            raise InvariantViolation(f"S5: None valid_from on claim {claim_id}")
        vf = claim.valid_from
        if vf.tzinfo is None or vf.utcoffset() is None:
            raise InvariantViolation(f"S5: valid_from is not timezone-aware on claim {claim_id}")
        # Normalize to UTC and check offset is zero.
        if vf.astimezone(UTC).utcoffset() != timedelta(0):
            raise InvariantViolation(f"S5: valid_from is not UTC on claim {claim_id}")


__all__ = [
    "CommitSpyJournal",
    "CrashAfterStepJournal",
    "EngineFactory",
    "InvariantViolation",
    "RecordingEntityKG",
    "SimulatedCrash",
    "assert_claim_requires_provenance",
    "assert_control_independent_of_model_text",
    "assert_no_model_on_write_path",
    "assert_resume_never_recalls_model",
    "assert_run_writes_carry_provenance",
    "assert_s5_substrate_invariants",
]
