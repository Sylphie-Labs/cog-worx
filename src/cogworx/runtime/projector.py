"""The trial projector — committed journal steps to procedural-KG trials, off-write-path (S1/S6).

This is the WRITE-side integration that turns committed journal ``StepRecord``s into ``(:Trial)``
nodes (architect P0.2: a Trial is a PROJECTION of a committed step, never written inline in a stage
body). It is SWEEPER-SHAPED (:class:`cogworx.runtime.sweeper.Sweeper`): it owns no model and does no
commits on the hot path — each tick polls the TimescaleDB journal for newly committed steps and
MERGEs a trial per procedure step into Neo4j, advancing a Neo4j-resident cursor.

Why a projector and not an inline write (S1): writing the trial inside the stage would put a Neo4j
round-trip on the latency-critical write path and couple the posterior write to the stage's success.
Projecting off the committed journal means the trial write is replay-safe (MERGE on the step PK is
idempotent), crash-safe (the cursor advances in the batch's own txn), and free of the model loop.

CURSOR MODEL (COMMIT-ORDER, SINGLE FENCED READ — the commit_xid resolution):
  The cursor (:class:`ProjectionCursor`) is the tuple ``(commit_ordinal, run_id, step_index)``,
  where ``commit_ordinal`` is the journal's DB-assigned commit-order integer (Postgres
  ``commit_xid``). It is monotonic-in-commit-order and never regresses (lexicographic tuple-max), so
  each tick does ONE bounded, visibility-fenced journal read:
    * ``committed_steps_after(cursor, limit=batch_limit)`` — the strict keyset ``(commit_ordinal,
      run_id, step_index) > cursor`` (cold start: from the beginning), oldest-first, capped at
      ``batch_limit``, FENCED so it never consumes a row an in-flight txn could still commit beneath
      (the sequence-visibility race; the fence lives in the journal adapter SQL). A strict keyset
      bound advances past many rows that share one ordinal (no livelock) and advances even on a
      zero-trial control batch (P0-2).
  There is no band, no lookback, no second read: ``commit_ordinal`` imposes a single total order, so
  the fenced forward read is total. A late / out-of-order wall-clock ``committed_at`` cannot drop a
  row — the fence holds it back until it is committed, then a later tick consumes it in order.
  In ONE ``project_batch`` Neo4j txn: MERGE the staged trials and advance the cursor to the
  tuple-max with the LAST SCANNED row (the last forward row).

CORRECTNESS MODEL (the load-bearing journal facts):
  * The journal is write-once: ``commit_step`` is INSERT … ON CONFLICT DO NOTHING, and a FAILED
    attempt commits NOTHING (only the durable attempt counter climbs). So one committed step == one
    final outcome; there is no "flip" (an earlier failed attempt left no row) and no double-count.
  * ``project_batch`` MERGEs trials on ``trial_id = f"{run_id}:{step_index}"`` first-write-wins, so
    a re-projection after a crash (the cursor never advanced) is a harmless no-op.
  * The cursor advances IN THE SAME Neo4j txn as the trial MERGEs, so a crash can never leave the
    cursor ahead of unwritten trials (exactly-once into Neo4j).

OUTCOME RESOLUTION (S9 — structure over self-report; NEVER infer from ``result.kind``):
  1. STAMPED — the stage stamped ``output.data["outcome"]`` (+ ``procedure_id``, ``problem_type``).
     This is the authority. A failing procedure that routes to a fixup stage commits a
     ``Transition`` whose ``kind`` reads as success, so the stamped outcome — not the result kind —
     is consulted.
  2. SYNTHESISED FAILURE — an engine-synthesised retry-exhaustion ``Degraded`` (``output.kind ==
     "retry-exhausted"``) carries ``failure_class``/``attempts`` but NO ``procedure_id`` (the engine
     does not know the mapping). The projector recovers ``(procedure_id, problem_type)`` from the
     static :class:`~cogworx.knowledge.procedural_registry.ProcedureRegistry` keyed by the failed
     step's ``(pathway, stage)`` — the survivorship-bias fix (hard failures still count).
  3. SKIP — a committed step whose stage is not a declared procedure and carries no stamped outcome
     is ordinary control flow, not a trial.

KNOWN LIMITATION (FAILED-at-seq, flagged for Jim): a run that exhausts retries with
``on_exhausted="fail"`` commits NOTHING at the failing ``seq`` — the run record flips to FAILED but
there is no journal row for the projector to observe (the journal exposes only committed steps, and
the table is kept pure: no per-consumer column, no status-change outbox). So a hard FAILED-at-seq
failure is NOT projected in v1 → it does not contribute its failure trial, leaving that specific
posterior biased upward. The exhaustion-``Degrade`` failure path (``on_exhausted="degrade"``) IS
projected fully (it commits a step). Observing never-committed terminal failures needs a run-status
projection seam (a status-change outbox or a FAILED-run scan joined against the pathway graph to
recover the stalled stage) that crosses into engine/pathway-graph ownership; deferred, not silently
dropped. ``tests/unit/test_trial_projector.py::test_failed_at_seq_not_projected_known_limitation``
pins this gap so it cannot regress unnoticed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from cogworx.claims.provenance import Provenance
from cogworx.knowledge.procedural_registry import ProcedureRegistry
from cogworx.substrate.journal import Journal, ProjectedStep, ProjectionCursor, StepRecord
from cogworx.substrate.procedural_kg import (
    Outcome,
    ProceduralKG,
    TrialWrite,
    ordinal_max,
)

__all__ = [
    "DEFAULT_PROJECTION_CONSUMER",
    "ResolvedOutcome",
    "TrialProjector",
    "resolve_outcome",
]

DEFAULT_PROJECTION_CONSUMER = "procedural-kg/trial-projector"
"""The default cursor consumer key. Each projection consumer owns its own ``(:ProjectionCursor)`` so
the journal table stays pure (S3) and a future entity-KG / world-model projector reads from its own
watermark without interfering."""


@dataclass(frozen=True)
class ResolvedOutcome:
    """The trial identity + outcome resolved for one committed step (the projector's per-step plan).

    ``procedure_id``/``problem_type`` are the deterministic framework-assigned ids; ``outcome`` is
    the stamped or synthesised polarity. A step that is not a procedure trial resolves to ``None``
    (see :func:`resolve_outcome`), never a :class:`ResolvedOutcome`.
    """

    procedure_id: str
    problem_type: str
    outcome: Outcome


def resolve_outcome(
    step: StepRecord, *, pathway_id: str, registry: ProcedureRegistry
) -> ResolvedOutcome | None:
    """Resolve a committed step to a trial outcome, or ``None`` if it is not a procedure trial.

    Order (S9): the STAMPED ``output.data["outcome"]`` is authoritative; only an engine-synthesised
    retry-exhaustion ``Degraded`` falls back to the registry for a synthesised failure. ``kind`` is
    NEVER consulted for the outcome — a fixup-routing ``Transition`` would falsely read as success.

    A stamped outcome must carry its own ``procedure_id`` and ``problem_type`` (the stage knows
    them — it stamped them); a stamped outcome missing either is a malformed stamp and raises,
    rather than silently guessing from the registry (fail-loud, S9).
    """
    output = getattr(step.result, "output", None)
    data = output.data if output is not None else {}

    stamped = data.get("outcome")
    if stamped is not None:
        if stamped not in ("success", "failure"):
            raise ValueError(
                f"step {step.run_id!r}:{step.step_index} stamped a non-canonical outcome "
                f"{stamped!r}; expected 'success' or 'failure'"
            )
        procedure_id = data.get("procedure_id")
        problem_type = data.get("problem_type")
        if not isinstance(procedure_id, str) or not isinstance(problem_type, str):
            raise ValueError(
                f"step {step.run_id!r}:{step.step_index} stamped outcome {stamped!r} but is "
                "missing a string procedure_id/problem_type; a stamped outcome must carry both"
            )
        return ResolvedOutcome(
            procedure_id=procedure_id, problem_type=problem_type, outcome=stamped
        )

    # No stamp: the only trial we synthesise is an engine exhaustion-degrade (a hard failure with no
    # procedure_id). Everything else is ordinary control flow the projector skips.
    if output is not None and output.kind == "retry-exhausted":
        declaration = registry.get(pathway_id, step.stage_name)
        if declaration is None:
            return None
        return ResolvedOutcome(
            procedure_id=declaration.procedure_id,
            problem_type=declaration.problem_type,
            outcome="failure",
        )
    return None


class TrialProjector:
    """Polls the journal for committed steps and projects each procedure step into a ``(:Trial)``.

    Sweeper-shaped (S1): no model, no hot-path commit. The ``STEP_COMMITTED`` engine event MAY
    trigger an extra ``tick`` to cut latency, but the journal poll is the correctness backstop — the
    projector never depends on an event arriving (the engine emits it observationally, non-deduped).
    """

    def __init__(
        self,
        *,
        journal: Journal,
        procedural_kg: ProceduralKG,
        registry: ProcedureRegistry,
        consumer: str = DEFAULT_PROJECTION_CONSUMER,
        batch_limit: int = 256,
    ) -> None:
        self._journal = journal
        self._kg = procedural_kg
        self._registry = registry
        self._consumer = consumer
        self._batch_limit = batch_limit
        # Resolve run -> pathway once per tick; a run's pathway pointer is immutable for its life.
        self._pathway_cache: dict[str, str | None] = {}

    async def tick(self) -> int:
        """Project one batch via the single fenced commit-order read; return the trials staged.

        ONE bounded journal read per tick: ``committed_steps_after(cursor, limit=batch_limit)`` —
        the STRICT keyset read STRICTLY past the cursor, capped at ``batch_limit``,
        visibility-fenced on ``commit_xid`` (the adapter forbids consuming any row an in-flight txn
        could still commit beneath). It advances past many rows sharing one ordinal (no livelock)
        and advances even on a zero-trial control batch (P0-2).

        Then in ONE ``project_batch`` txn: stage a trial per resolved PROCEDURE row and advance the
        cursor to the tuple-max with the LAST SCANNED row (never regress; unchanged if the read was
        empty). An idle tick (no rows) returns 0 and leaves the cursor untouched.
        """
        self._pathway_cache.clear()
        cursor = await self._kg.read_cursor(self._consumer)

        scanned = tuple(await self._journal.committed_steps_after(cursor, limit=self._batch_limit))
        if not scanned:
            return 0

        staged = await self._stage(scanned)

        new_cursor = ordinal_max(cursor, _cursor_of(scanned[-1]))
        await self._kg.project_batch(self._consumer, trials=staged, progress=new_cursor)
        return len(staged)

    async def _stage(self, scanned: Sequence[ProjectedStep]) -> list[TrialWrite]:
        staged: list[TrialWrite] = []
        for projected in scanned:
            step = projected.record
            pathway_id = await self._pathway_for(step.run_id)
            if pathway_id is None:
                continue
            resolved = resolve_outcome(step, pathway_id=pathway_id, registry=self._registry)
            if resolved is None:
                continue
            staged.append(
                TrialWrite(
                    trial_id=f"{step.run_id}:{step.step_index}",
                    procedure_id=resolved.procedure_id,
                    problem_type=resolved.problem_type,
                    outcome=resolved.outcome,
                    occurred_at=step.committed_at,
                    provenance=_trial_provenance(step.committed_at),
                )
            )
        return staged

    async def _pathway_for(self, run_id: str) -> str | None:
        if run_id not in self._pathway_cache:
            run = await self._journal.load_run(run_id)
            self._pathway_cache[run_id] = run.pathway_id if run is not None else None
        return self._pathway_cache[run_id]


def _cursor_of(projected: ProjectedStep) -> ProjectionCursor:
    """The cursor ``(commit_ordinal, run_id, step_index)`` of a scanned step."""
    return ProjectionCursor(
        commit_ordinal=projected.commit_ordinal,
        run_id=projected.record.run_id,
        step_index=projected.record.step_index,
    )


def _trial_provenance(committed_at: datetime) -> Provenance:
    """Build the trial provenance: ``source="system"`` (a deterministic, zero-model control event —
    a first-hand observation of our OWN execution, not a model claim) at full confidence (S5)."""
    return Provenance(source="system", confidence=1.0, recorded_at=committed_at)
