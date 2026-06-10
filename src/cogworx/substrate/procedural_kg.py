"""Procedural-KG seam — the thin internal Protocol for tests/mocks (CANON S3).

This is the engine-shaped seam for the procedural knowledge graph on Neo4j: which procedures work on
which problem types, learned from trials. It is the test/mock surface only — NOT a backend-
portability layer (S3 violation). The real implementation will be
``cogworx.adapters.neo4j_procedural_kg.Neo4jProceduralKG`` (a later task); a parity-held
``InMemoryProceduralKG`` double is the test seam.

Graph semantics (to be enforced by BOTH the real adapter AND the in-memory double):
  - (:Procedure) and (:ProblemType) nodes are addressed by deterministic, collision-proof ids
    (:mod:`cogworx.knowledge.procedural_identity`); MERGE on those ids so re-declaration is
    idempotent.
  - (:Procedure)-[:APPLIES_TO]->(:ProblemType) is TOPOLOGY ONLY — it carries NO stored counters and
    NO stored posterior. The success rate is DERIVED AT READ from the trials (S1, S6).
  - (:Trial) nodes are the immutable EVENT SOURCE. One Trial = one committed journal step. They
    ACCUMULATE (MERGE on ``trial_id``, never UPDATE); a re-projection of the same step is a no-op.
  - The Beta posterior over an edge is derived from its trials at read via
    :func:`cogworx.knowledge.procedural_confidence.procedure_success` (the single computation path).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import datetime
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict

from cogworx.claims.provenance import Provenance
from cogworx.knowledge.procedural_confidence import ProcedureSuccess
from cogworx.substrate.journal import ProjectionCursor

__all__ = [
    "CursorAdvance",
    "Outcome",
    "ProblemType",
    "ProceduralKG",
    "Procedure",
    "ScoredProcedure",
    "Trial",
    "TrialWrite",
    "ordinal_ge",
    "ordinal_max",
]

# A stamped STRUCTURAL fact, NOT a model inference. The projecting stage stamps the committed
# outcome (S9); success is NEVER read off ``result.kind`` (a failing procedure routing to a fixup
# stage still commits a Transition, which would read as success). Maps to polarity at read:
# success -> "+", failure -> "-".
Outcome = Literal["success", "failure"]


class Procedure(BaseModel):
    """A learnable unit of behaviour — a ``(pathway, stage)`` scored for success per problem type.

    ``id`` is deterministic + framework-assigned via
    :func:`cogworx.knowledge.procedural_identity.procedure_id_for` (S9: NEVER model output).
    ``label`` is the human-readable name; identity is the id, not the label.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    label: str


class ProblemType(BaseModel):
    """A category a procedure is applied to. Posterior is per ``(procedure, problem_type)`` edge.

    ``id`` is deterministic + framework-assigned via
    :func:`cogworx.knowledge.procedural_identity.problem_type_id_for` (S9: NEVER model output).
    ``embedding`` is nullable now: the dense/BM25 recall channels over problem types are deferred to
    Pod 2.5, but the field exists so the seam does not change shape when they land.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    label: str
    embedding: tuple[float, ...] | None = None


class Trial(BaseModel):
    """An immutable record that a procedure was applied to a problem type with a stamped outcome.

    One Trial == one committed journal step. ``trial_id`` IS the journal step PK
    ``f"{run_id}:{step_index}"`` (architect P0.2) so MERGE on it is exactly-once into Neo4j and a
    re-run of the off-path projector is idempotent. ``attempt`` is deliberately NOT in the key: a
    failed attempt commits nothing, so a committed step is already the final outcome.

    Validated + frozen like :class:`~cogworx.knowledge.evidence.EvidenceEvent`: an invalid trial can
    never be persisted or replayed.
    """

    model_config = ConfigDict(frozen=True)

    trial_id: str
    """``f"{run_id}:{step_index}"`` — the committed journal step PK (the MERGE key)."""
    run_id: str
    """The DEDUP GRAIN for the posterior (``source_id = run_id``). One run = one contribution per
    polarity, regardless of how many times a cyclic run applied the procedure (S6 / eval-stats)."""
    step_index: int
    procedure_id: str
    problem_type: str
    outcome: Outcome
    """A stamped structural fact read off the COMMITTED step (S9) — never inferred from
    ``result.kind``."""
    occurred_at: datetime
    """UTC; the committed-at of the journal step. Naive datetimes are interpreted as UTC at the
    boundary, tz-aware are converted to UTC (the substrate datetime contract)."""
    provenance: Provenance
    """``source="system"`` (deterministic control event, zero-model). Epistemically a trial is a
    first-hand *observation* of our own execution, not a model claim — but note ``Provenance`` has
    no ``epistemic_type`` field today (it lives on ``Claim``); the observation grade is conveyed
    structurally by ``source="system"``, not by a typed field (S5). See the Pod-2.0 C3 carry-forward
    on whether epistemic typing should become a field on every claim type."""


class TrialWrite(BaseModel):
    """One trial staged for a batch projection — the per-trial payload
    :meth:`ProceduralKG.project_batch` MERGEs (the fields :meth:`ProceduralKG.record_trial` takes,
    minus the cursor).

    The projector stages one of these per resolved procedure step, then hands the whole sequence
    (plus the last-scanned cursor) to ``project_batch`` for one atomic txn.
    """

    model_config = ConfigDict(frozen=True)

    trial_id: str
    procedure_id: str
    problem_type: str
    outcome: Outcome
    occurred_at: datetime
    provenance: Provenance


def ordinal_ge(a: ProjectionCursor, b: ProjectionCursor) -> bool:
    """Return whether ``a >= b`` under lexicographic ``(commit_ordinal, run_id, step_index)`` order.

    The SINGLE source of the cursor comparison shared by the adapter and the in-memory double so
    they can never drift on monotonic-advance semantics. The stored cursor advances to the tuple
    ``max``: keep ``a`` iff ``ordinal_ge(a, b)``, else move to ``b``.
    """
    return (a.commit_ordinal, a.run_id, a.step_index) >= (b.commit_ordinal, b.run_id, b.step_index)


def ordinal_max(a: ProjectionCursor | None, b: ProjectionCursor | None) -> ProjectionCursor | None:
    """Return the lexicographic max of two cursors, treating ``None`` as the empty (lowest) value.

    The cursor's never-regress rule, single-sourced so the projector advances identically to the way
    the adapter/double store it. ``ordinal_max(None, x) == x`` (cold start), ``ordinal_max(x, None)
    == x`` (an empty read leaves the cursor unchanged).
    """
    if a is None:
        return b
    if b is None:
        return a
    return a if ordinal_ge(a, b) else b


class CursorAdvance(BaseModel):
    """A request to advance a projection cursor IN THE SAME TXN as the batch write (S6).

    The off-write-path projector passes this so the ``(:ProjectionCursor {consumer})`` cursor and
    the ``(:Trial)`` batch it just projected commit ATOMICALLY — a crash can never leave the cursor
    ahead of trials that were never written (exactly-once into Neo4j; spike a6). The advance is
    MONOTONIC: the stored cursor moves to the lexicographic ``max`` of itself and ``cursor``
    (:func:`ordinal_ge`), so an out-of-order commit never regresses it.
    """

    model_config = ConfigDict(frozen=True)

    consumer: str
    """The cursor's owner key. Each projection consumer (trial projector, future entity-KG / world
    model) owns its OWN cursor node — the journal table stays pure (no per-consumer column, S3)."""
    cursor: ProjectionCursor
    """The cursor of the LAST SCANNED step (procedure or not) — the cursor advances to the
    lexicographic max of this and its current value. This is the LAST scanned row, NOT the max of
    the projected trials, so it advances even on a zero-trial control batch (fixes P0-2)."""


class ScoredProcedure(BaseModel):
    """A candidate procedure for a problem type with its read-derived success posterior.

    Carries the deduped trial count (``success.n_trials``) the promotion floor MUST use — the raw
    ``(:Trial)`` count is never surfaced here, so a caller cannot accidentally gate on it.
    """

    model_config = ConfigDict(frozen=True)

    procedure: Procedure
    problem_type: ProblemType
    success: ProcedureSuccess
    """The derived Beta posterior over this edge: ``success_rate``, ``alpha``/``beta`` for Thompson
    sampling, ``variance`` for the LCB gate, and ``n_trials`` (deduped) for the promotion floor."""
    promoted: bool = False
    """Whether this edge has cleared the promotion gate. Set by the promotion read surface (a later
    task); defaults False until that lands."""


@runtime_checkable
class ProceduralKG(Protocol):
    """The Neo4j procedural-KG seam (S3 — thin internal test/mock surface, NOT a portability layer).

    CONTRACT SEMANTICS (to be enforced by both the real adapter and the in-memory double):

    project_batch
      The off-write-path projector's PER-TICK atomic unit (S1/S6). In ONE managed Neo4j txn it
      MERGEs every trial in the batch (ON CREATE only, first-write-wins) THEN advances the cursor
      to the last SCANNED step. A crash mid-batch commits NOTHING (neither trials nor the advance);
      the next tick re-reads and re-projects (a4 no-op). A ZERO-TRIAL control batch STILL advances
      the cursor in its own txn — a burst of non-procedure control steps can never strand the
      watermark at cold start (fixes P0-2).

    record_trial
      A single-trial write surface (the parity suite's per-write use). MERGE on ``trial_id``
      (= ``f"{run_id}:{step_index}"``) so it is idempotent — re-projecting the same committed step
      stores nothing new, and the stored payload of an already-present trial is NEVER updated (no
      ``ON MATCH SET``; first-write-wins). MERGEs the (:Procedure), (:ProblemType), and
      topology-only [:APPLIES_TO] edge as a side effect; the edge carries no counters.
      ``procedure_id`` / ``problem_type`` MUST be the deterministic framework-assigned ids (S9 —
      never model output). The per-tick projection uses ``project_batch``, not ``record_trial``.

    Posterior derivation (DERIVE-AT-READ, S1/S6)
      No success rate, count, or posterior is ever stored. ``posterior`` and ``candidates`` derive
      the Beta posterior at read from the edge's trials via
      :func:`cogworx.knowledge.procedural_confidence.procedure_success`, deduped at
      ``source_id = run_id`` per polarity. The promotion floor counts ``ProcedureSuccess.n_trials``
      (deduped), NOT raw trials.

    Datetime contract
      Naive datetimes are interpreted as UTC (not rejected). Tz-aware datetimes are converted to
      UTC. The projection cursor's ``commit_ordinal`` is a plain integer (the journal's commit-order
      key), stored as a Neo4j integer — NOT a lexicographic ISO string.

    S9 identity discipline (interim)
      ``procedure_id`` and ``problem_type`` MUST be assigned by framework code (the static
      ``(pathway, stage) -> procedure_id`` registry; a declaration-time problem-type label) and
      NEVER taken from model output — self-reported procedure identity would let a model split/merge
      procedures to inflate its own promotion. Structural enforcement (the registry) is the
      projector's job; this is the interim discipline, mirroring entity-KG's ``source_id`` contract.
    """

    async def project_batch(
        self,
        consumer: str,
        *,
        trials: Sequence[TrialWrite],
        progress: ProjectionCursor | None,
    ) -> None:
        """Atomically MERGE a batch of trials THEN advance the cursor, in ONE Neo4j txn.

        Every trial is MERGEd ON CREATE only (first-write-wins, like :meth:`record_trial`). Then
        the ``(:ProjectionCursor {consumer})`` cursor advances to the lexicographic ``max`` of its
        stored value and ``progress`` (:func:`ordinal_max`) — it NEVER regresses. A ``None``
        argument leaves the stored cursor UNCHANGED (an empty read).

        The cursor lives as ``commit_ordinal/run_id/step_index`` on the node. Because the trial
        MERGEs and the advance share one txn, a crash mid-batch commits nothing and the next tick
        re-reads + reprojects (a4 idempotent). ``trials`` MAY be empty: a zero-trial control batch
        advances the cursor alone so a burst of non-procedure steps still moves the watermark
        (P0-2). ``progress`` is the LAST SCANNED row's cursor, NOT the max of ``trials``.
        """
        ...

    async def record_trial(
        self,
        *,
        trial_id: str,
        procedure_id: str,
        problem_type: str,
        outcome: Outcome,
        occurred_at: datetime,
        provenance: Provenance,
        cursor: CursorAdvance | None = None,
    ) -> None:
        """MERGE one trial (idempotent on ``trial_id``) plus its procedure/problem-type/edge graph.

        First-write-wins: an already-present ``trial_id`` is a no-op, never an update. Records no
        derived value — the posterior is computed at read. The full :class:`Trial` is reconstructed
        by the adapter from these fields (``run_id``/``step_index`` are split from ``trial_id``).

        When ``cursor`` is supplied, the projection cursor is advanced (monotonically, by the
        lexicographic ``(commit_ordinal, run_id, step_index)`` tuple) IN THE SAME TRANSACTION as the
        trial MERGE (S6). The per-tick projection uses :meth:`project_batch`; ``record_trial`` is
        the parity suite's single-write surface and is typically called with ``cursor=None``.
        """
        ...

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        """Return the consumer's projection cursor, or ``None`` if never advanced.

        The last step the consumer has scanned, in commit order. The off-write-path projector reads
        it at the start of a tick and polls the journal for committed steps STRICTLY past it
        (``(commit_ordinal, run_id, step_index) > cursor``). ``None`` means a cold start (read from
        the beginning).
        """
        ...

    async def get_trial(self, trial_id: str) -> Trial | None:
        """Return the trial by id, or ``None``."""
        ...

    async def trials_for(self, procedure_id: str, problem_type: str) -> Sequence[Trial]:
        """Return all trials on the ``(procedure_id, problem_type)`` edge, in occurrence order."""
        ...

    async def posterior(self, procedure_id: str, problem_type: str) -> ProcedureSuccess:
        """Derive the Beta-posterior success summary for one edge at read.

        Empty edge → prior-only (success_rate 0.5, n_trials 0). The single computation path: deduped
        at ``source_id = run_id`` per polarity, ``n_trials`` is the deduped count for the promotion
        floor.
        """
        ...

    async def candidates(
        self,
        problem_type: str,
        *,
        limit: int = 20,
        promoted_only: bool = False,
        rng: random.Random | None = None,
    ) -> Sequence[ScoredProcedure]:
        """Return scored candidate procedures for a problem type, ranked for the read surface.

        Each :class:`ScoredProcedure` has ``promoted`` stamped fresh from the promotion gate
        (derive-at-read, no stored flag); ``promoted_only`` filters to edges that have cleared the
        gate (for high-stakes consumers).

        The PRODUCTION ranking (``rng is None``) is deterministic posterior-mean greedy: candidates
        are ranked by the Beta posterior mean DESC, tie-broken by procedure id. The Beta(1, 1) prior
        gives a fresh edge mean 0.5, so the first selections over a new problem type explore before
        exploiting — no RNG needed. This is what ships after the Pod 2.1 spike.

        ``rng`` is the OPT-IN seam for the DEFERRED Thompson read surface: supplying a seeded RNG
        switches ranking to a per-edge ``theta ~ Beta(alpha, beta)`` draw (and a pinned seed yields
        byte-identical ordering across the adapter and the double — the determinism bet). Thompson
        did not beat a fair baseline in the Pod 2.1 spike and is kept callable for a future spike;
        it is not the default.
        """
        ...
