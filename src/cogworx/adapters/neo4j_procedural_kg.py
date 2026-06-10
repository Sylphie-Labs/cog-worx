"""Neo4j adapter for the procedural knowledge graph (CANON S1, S3, S5, S6).

Extends :class:`cogworx.adapters.neo4j_graph.Neo4jGraphStore` (shares driver, config, aclose) and
fully implements the :class:`cogworx.substrate.procedural_kg.ProceduralKG` Protocol. This is the
graph knowledge layer ONLY (S3): the durable journal lives in TimescaleDB and the latent space in
pgvector — neither belongs here.

Graph model
-----------
``(:Procedure {id, label})`` and ``(:ProblemType {id, label, embedding?})`` are MERGE-d on their
deterministic ids (:mod:`cogworx.knowledge.procedural_identity`) so re-declaration is idempotent.

``(:Procedure)-[:APPLIES_TO]->(:ProblemType)`` is TOPOLOGY ONLY — it carries NO stored counters and
NO stored posterior (D1: derive-at-read). The success rate is re-derived from trials at read.

``(:Trial {...})`` is the immutable EVENT SOURCE. Each Trial attaches via TWO typed edges:
  ``(:Trial)-[:OF_PROCEDURE]->(:Procedure)`` and ``(:Trial)-[:AGAINST]->(:ProblemType)``.

Why two edges (not "Trial on the APPLIES_TO edge"): Neo4j relationships cannot be endpoints of
other relationships, so "a Trial hanging off the edge" would force either reifying APPLIES_TO into a
node (contradicting topology-only) or stamping trial data as edge properties (mutable, not
event-sourced). Two typed edges keep the Trial a first-class immutable node and make ``trials_for``
a clean two-edge intersection on the ``(procedure_id, problem_type)`` pair.

Hard invariants
---------------
record_trial MERGEs on ``trial_id`` with ``ON CREATE`` ONLY — there is NO ``ON MATCH SET`` (D5 /
spike a5: first-write-wins). Re-calling with the same ``trial_id`` but a different payload leaves
the stored Trial UNCHANGED. The inherited blanket-``SET`` write back-door (``upsert_claim``) is
SEALED to ``NotImplementedError`` (mirrors entity-KG FIX 1 / 2.0 red-team #1) — a caller cannot
reach a blanket-SET surface on this class.

Posterior derivation routes through
:func:`cogworx.knowledge.procedural_confidence.procedure_success` VERBATIM (the single computation
path); the adapter never reimplements Beta math and never stores a derived value.

Datetime contract: ``occurred_at`` is UTC-normalised via :func:`to_utc` before storage (naive → UTC,
tz-aware → converted) so stored ISO strings share the +00:00 offset and lexicographic order ==
temporal order (used by ``trials_for``'s ORDER BY).
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_entity_kg import to_utc
from cogworx.adapters.neo4j_graph import Neo4jGraphStore, _as_source
from cogworx.claims.provenance import Provenance
from cogworx.knowledge.procedural_confidence import (
    ProcedureSuccess,
    TrialOutcome,
    procedure_success,
)
from cogworx.knowledge.procedural_promotion import PromotionPolicy
from cogworx.substrate.journal import ProjectionCursor
from cogworx.substrate.procedural_kg import (
    CursorAdvance,
    Outcome,
    ProblemType,
    Procedure,
    ScoredProcedure,
    Trial,
    TrialWrite,
)
from cogworx.substrate.procedural_selection import (
    DEFAULT_PROMOTION_POLICY,
    posterior_mean_select,
    thompson_select,
)

if TYPE_CHECKING:
    from neo4j import AsyncManagedTransaction, Record

__all__ = ["Neo4jProceduralKG", "select_candidates", "split_trial_id"]


# ---------------------------------------------------------------------------
# trial_id parsing
# ---------------------------------------------------------------------------


def split_trial_id(trial_id: str) -> tuple[str, int]:
    """Split a ``trial_id`` into ``(run_id, step_index)``.

    The journal step PK is ``f"{run_id}:{step_index}"`` (architect P0.2). ``step_index`` is an
    integer and therefore contains no colon, so splitting at the LAST colon is unambiguous even when
    ``run_id`` itself contains colons. Raises ``ValueError`` if the id has no colon or the suffix is
    not a non-negative integer — a malformed id can never be persisted (mirrors the Trial validity
    contract).
    """
    sep = trial_id.rfind(":")
    if sep < 0:
        raise ValueError(
            f"malformed trial_id {trial_id!r}: expected 'f{{run_id}}:{{step_index}}' "
            "(no ':' separator found)"
        )
    run_id = trial_id[:sep]
    step_raw = trial_id[sep + 1 :]
    if not run_id:
        raise ValueError(f"malformed trial_id {trial_id!r}: empty run_id before ':'")
    try:
        step_index = int(step_raw)
    except ValueError as exc:
        raise ValueError(
            f"malformed trial_id {trial_id!r}: step_index {step_raw!r} is not an integer"
        ) from exc
    if step_index < 0:
        raise ValueError(f"malformed trial_id {trial_id!r}: step_index {step_index} is negative")
    # Reject non-canonical integer spellings ("01", "+1", " 1") so the id is byte-stable.
    if step_raw != str(step_index):
        raise ValueError(
            f"malformed trial_id {trial_id!r}: step_index {step_raw!r} is not canonical "
            f"(expected {step_index!r})"
        )
    return run_id, step_index


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

_PROCEDURE_CONSTRAINT = """
CREATE CONSTRAINT procedure_id_unique IF NOT EXISTS
FOR (p:Procedure) REQUIRE p.id IS UNIQUE
"""

_PROBLEM_TYPE_CONSTRAINT = """
CREATE CONSTRAINT problem_type_id_unique IF NOT EXISTS
FOR (pt:ProblemType) REQUIRE pt.id IS UNIQUE
"""

_TRIAL_CONSTRAINT = """
CREATE CONSTRAINT trial_id_unique IF NOT EXISTS
FOR (t:Trial) REQUIRE t.trial_id IS UNIQUE
"""

_CURSOR_CONSTRAINT = """
CREATE CONSTRAINT projection_cursor_consumer_unique IF NOT EXISTS
FOR (c:ProjectionCursor) REQUIRE c.consumer IS UNIQUE
"""

# ---------------------------------------------------------------------------
# Write-path Cypher
# ---------------------------------------------------------------------------

# record_trial: MERGE the Procedure / ProblemType nodes (idempotent re-declaration, coalesce the
# label so a re-declared node keeps its first label), MERGE the topology-only APPLIES_TO edge (no
# properties), then MERGE the Trial on its unique trial_id with ON CREATE ONLY.
#
# CRITICAL (spike a5, D5): there is NO `ON MATCH SET`. A second record_trial with the same trial_id
# but a different payload is a no-op — the stored Trial is immutable, first-write-wins. The two
# attachment edges are also MERGE-d so re-projecting the same step never duplicates them.
_RECORD_TRIAL_CYPHER = """
MERGE (p:Procedure {id: $procedure_id})
  ON CREATE SET p.label = $procedure_label
MERGE (pt:ProblemType {id: $problem_type})
  ON CREATE SET pt.label = $problem_type_label, pt.embedding = $problem_type_embedding
MERGE (p)-[:APPLIES_TO]->(pt)
MERGE (t:Trial {trial_id: $trial_id})
  ON CREATE SET
    t.run_id           = $run_id,
    t.step_index       = $step_index,
    t.procedure_id     = $procedure_id,
    t.problem_type     = $problem_type,
    t.outcome          = $outcome,
    t.occurred_at      = $occurred_at,
    t.prov_source      = $prov_source,
    t.prov_source_ref  = $prov_source_ref,
    t.prov_confidence  = $prov_confidence,
    t.prov_evidence    = $prov_evidence,
    t.prov_recorded_at = $prov_recorded_at
MERGE (t)-[:OF_PROCEDURE]->(p)
MERGE (t)-[:AGAINST]->(pt)
"""

# Advance a projection cursor MONOTONICALLY by the lexicographic (commit_ordinal, run_id,
# step_index) tuple — NEVER regress. commit_ordinal is a plain Neo4j INTEGER (the journal's
# commit_xid), so the list comparison [a,b,c] < [x,y,z] orders correctly without an ISO-string hack.
# The CASE advances only when the new cursor is strictly greater than the stored one (tuple-max).
# Run in the SAME managed transaction as the trial MERGE(s) so the cursor and the trials commit
# atomically (S6, spike a6). $commit_ordinal is non-null here (record_trial only advances when a
# CursorAdvance is supplied); the null-guard covers project_batch's empty-read case.
_ADVANCE_CURSOR_CYPHER = """
MERGE (c:ProjectionCursor {consumer: $consumer})
WITH c, ($commit_ordinal IS NOT NULL AND
         (c.commit_ordinal IS NULL OR
          [c.commit_ordinal, c.cursor_run_id, c.cursor_step_index]
            < [$commit_ordinal, $cursor_run_id, $cursor_step_index])) AS advance
SET c.commit_ordinal    = CASE WHEN advance THEN $commit_ordinal    ELSE c.commit_ordinal END,
    c.cursor_run_id     = CASE WHEN advance THEN $cursor_run_id     ELSE c.cursor_run_id END,
    c.cursor_step_index = CASE WHEN advance THEN $cursor_step_index ELSE c.cursor_step_index END
"""

# ---------------------------------------------------------------------------
# Read-path Cypher
# ---------------------------------------------------------------------------

_READ_CURSOR_CYPHER = """
MATCH (c:ProjectionCursor {consumer: $consumer})
RETURN c.commit_ordinal    AS commit_ordinal,
       c.cursor_run_id     AS cursor_run_id,
       c.cursor_step_index AS cursor_step_index
"""

_GET_TRIAL_CYPHER = """
MATCH (t:Trial {trial_id: $trial_id})
RETURN t.trial_id        AS trial_id,
       t.run_id          AS run_id,
       t.step_index      AS step_index,
       t.procedure_id    AS procedure_id,
       t.problem_type    AS problem_type,
       t.outcome         AS outcome,
       t.occurred_at     AS occurred_at,
       t.prov_source     AS prov_source,
       t.prov_source_ref AS prov_source_ref,
       t.prov_confidence AS prov_confidence,
       t.prov_evidence   AS prov_evidence,
       t.prov_recorded_at AS prov_recorded_at
"""

# All trials on one edge, in occurrence order. The intersection on (procedure_id, problem_type) is
# expressed as two MATCHes from the Trial so the same edge identity used by record_trial is read.
# ORDER BY occurred_at relies on the UTC-normalised ISO strings (lexicographic == temporal).
_TRIALS_FOR_CYPHER = """
MATCH (t:Trial)-[:OF_PROCEDURE]->(:Procedure {id: $procedure_id})
MATCH (t)-[:AGAINST]->(:ProblemType {id: $problem_type})
RETURN t.trial_id        AS trial_id,
       t.run_id          AS run_id,
       t.step_index      AS step_index,
       t.procedure_id    AS procedure_id,
       t.problem_type    AS problem_type,
       t.outcome         AS outcome,
       t.occurred_at     AS occurred_at,
       t.prov_source     AS prov_source,
       t.prov_source_ref AS prov_source_ref,
       t.prov_confidence AS prov_confidence,
       t.prov_evidence   AS prov_evidence,
       t.prov_recorded_at AS prov_recorded_at
ORDER BY t.occurred_at ASC, t.trial_id ASC
"""

# Minimal projection for posterior derivation: only (run_id, outcome) per trial reaches the math.
_POSTERIOR_TRIALS_CYPHER = """
MATCH (t:Trial)-[:OF_PROCEDURE]->(:Procedure {id: $procedure_id})
MATCH (t)-[:AGAINST]->(:ProblemType {id: $problem_type})
RETURN t.run_id AS run_id, t.outcome AS outcome
"""

# candidates: every ProblemType-targeting procedure with its label, plus the (run_id, outcome) of
# each trial on that edge so the posterior is derived per edge in Python. Each row is one edge.
_CANDIDATES_CYPHER = """
MATCH (p:Procedure)-[:APPLIES_TO]->(pt:ProblemType {id: $problem_type})
OPTIONAL MATCH (t:Trial)-[:OF_PROCEDURE]->(p)
OPTIONAL MATCH (t)-[:AGAINST]->(pt)
WITH p, pt, t
WHERE t IS NULL OR (t)-[:AGAINST]->(pt)
WITH p, pt, collect({run_id: t.run_id, outcome: t.outcome}) AS trial_rows
RETURN p.id          AS procedure_id,
       p.label       AS procedure_label,
       pt.id         AS problem_type_id,
       pt.label      AS problem_type_label,
       pt.embedding  AS problem_type_embedding,
       trial_rows    AS trial_rows
ORDER BY p.id ASC
"""

# Reset: wipe procedural-KG labels. DETACH so the APPLIES_TO / OF_PROCEDURE / AGAINST edges go too.
_RESET_TRIAL_CYPHER = "MATCH (t:Trial) DETACH DELETE t"
_RESET_PROCEDURE_CYPHER = "MATCH (p:Procedure) DETACH DELETE p"
_RESET_PROBLEM_TYPE_CYPHER = "MATCH (pt:ProblemType) DETACH DELETE pt"
_RESET_CURSOR_CYPHER = "MATCH (c:ProjectionCursor) DETACH DELETE c"


def _row_to_trial_outcome(row: Any) -> TrialOutcome:
    """Map a (run_id, outcome) result row onto the minimal TrialOutcome for the Beta math."""
    data = row.data() if hasattr(row, "data") else dict(row)
    return TrialOutcome(run_id=str(data["run_id"]), success=data["outcome"] == "success")


def _rows_to_posterior(rows: list[Any]) -> ProcedureSuccess:
    """Derive the edge posterior from (run_id, outcome) rows via the single computation path."""
    return procedure_success(_row_to_trial_outcome(r) for r in rows)


def _as_outcome(value: Any) -> Outcome:
    text = str(value)
    if text == "success":
        return "success"
    if text == "failure":
        return "failure"
    raise ValueError(f"unknown outcome persisted in Neo4j: {text!r}")


def _record_to_trial(record: Any) -> Trial:
    """Rebuild a full :class:`Trial` from a Cypher result row."""
    data = record.data() if hasattr(record, "data") else dict(record)
    provenance = Provenance(
        source=_as_source(data["prov_source"]),
        source_ref=data["prov_source_ref"],
        confidence=float(data["prov_confidence"]),
        evidence=tuple(data["prov_evidence"] or ()),
        recorded_at=datetime.fromisoformat(data["prov_recorded_at"]),
    )
    return Trial(
        trial_id=str(data["trial_id"]),
        run_id=str(data["run_id"]),
        step_index=int(data["step_index"]),
        procedure_id=str(data["procedure_id"]),
        problem_type=str(data["problem_type"]),
        outcome=_as_outcome(data["outcome"]),
        occurred_at=datetime.fromisoformat(data["occurred_at"]),
        provenance=provenance,
    )


class Neo4jProceduralKG(Neo4jGraphStore):
    """Procedural knowledge graph on Neo4j — event-sourced trials with derive-at-read posterior.

    Extends :class:`~cogworx.adapters.neo4j_graph.Neo4jGraphStore` (driver lifecycle, ``aclose``)
    and implements the :class:`~cogworx.substrate.procedural_kg.ProceduralKG` Protocol. Performs
    zero model calls (S1): ids and outcomes are framework-assigned and passed in.
    """

    def __init__(
        self,
        *,
        settings: SubstrateSettings | None = None,
        uri: str | None = None,
        user: str | None = None,
        password: str | None = None,
        procedure_labels: dict[str, str] | None = None,
        problem_type_labels: dict[str, str] | None = None,
        promotion_policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
        rng: random.Random | None = None,
    ) -> None:
        super().__init__(settings=settings, uri=uri, user=user, password=password)
        # The Protocol's record_trial does not carry human labels (ids are the identity); the
        # adapter resolves a stored label, falling back to the id, via these optional registries.
        # Mirrors the static (pathway, stage) -> procedure_id declaration registry (D2/S9).
        self._procedure_labels = dict(procedure_labels or {})
        self._problem_type_labels = dict(problem_type_labels or {})
        self._promotion_policy = promotion_policy
        # OPT-IN Thompson RNG. None (the default) => candidates uses the production deterministic
        # posterior-mean greedy selector. A caller who wants the DEFERRED Thompson read surface
        # supplies an rng here (or per-call); a pinned seed makes the Thompson order reproducible.
        self._rng = rng

    async def ensure_schema(self, *, embedding_dim: int | None = None) -> None:
        """Idempotently create the procedural-KG constraints, ADDITIVELY over the base schema.

        Calls the base constraint (UNIQUE :Claim {id}) first so this class coexists with the entity
        KG in the same community-edition database, then adds UNIQUE (:Procedure {id}),
        UNIQUE (:ProblemType {id}), UNIQUE (:Trial {trial_id}). ``embedding_dim`` is accepted for
        signature parity with the entity KG; the ProblemType vector index lands in Pod 2.5.
        """
        await super().ensure_schema()

        async def _write_ddl(tx: AsyncManagedTransaction) -> None:
            await tx.run(_PROCEDURE_CONSTRAINT)
            await tx.run(_PROBLEM_TYPE_CONSTRAINT)
            await tx.run(_TRIAL_CONSTRAINT)
            await tx.run(_CURSOR_CONSTRAINT)

        async with self._connection.session() as session:
            await session.execute_write(_write_ddl)

    async def reset(self) -> None:
        """Drop all :Trial, :Procedure, :ProblemType nodes (and the base :Claim wipe)."""
        await super().reset()

        async def _wipe(tx: AsyncManagedTransaction) -> None:
            await tx.run(_RESET_TRIAL_CYPHER)
            await tx.run(_RESET_PROCEDURE_CYPHER)
            await tx.run(_RESET_PROBLEM_TYPE_CYPHER)
            await tx.run(_RESET_CURSOR_CYPHER)

        async with self._connection.session() as session:
            await session.execute_write(_wipe)

    # -----------------------------------------------------------------------
    # Sealed write-seam (mirror entity-KG FIX 1 / 2.0 red-team #1)
    # -----------------------------------------------------------------------

    async def upsert_claim(self, claim: Any) -> str:
        """Sealed: the procedural KG's only write surface is record_trial.

        Raises ``NotImplementedError`` unconditionally. The inherited Phase-0
        ``Neo4jGraphStore.upsert_claim`` performs a blanket SET of every field on a (:Claim) node.
        Exposing it on a procedural-KG adapter is a back-door around Trial immutability (the
        ``ON CREATE``-only MERGE / first-write-wins contract). Use ``record_trial`` instead.
        """
        raise NotImplementedError(
            "Neo4jProceduralKG.upsert_claim is disabled. "
            "The procedural KG's only write surface is record_trial "
            "(ON CREATE-only MERGE: trials are immutable, first-write-wins). "
            "The inherited Phase-0 Neo4jGraphStore.upsert_claim is a blanket-SET back-door."
        )

    # -----------------------------------------------------------------------
    # ProceduralKG Protocol implementation
    # -----------------------------------------------------------------------

    async def project_batch(
        self,
        consumer: str,
        *,
        trials: Sequence[TrialWrite],
        progress: ProjectionCursor | None,
    ) -> None:
        """MERGE a batch of trials THEN advance the cursor, in ONE managed Neo4j txn (S6).

        Each trial is validated (frozen pydantic) into a full :class:`Trial` before any I/O so a
        malformed id never reaches Neo4j, then MERGEd ON CREATE only (first-write-wins). LAST, in
        the same txn: ``progress`` advances tuple-max (``None`` leaves it unchanged). A crash
        mid-batch commits nothing and the next tick re-reads + reprojects (a4). An empty ``trials``
        sequence
        still advances the cursor (a zero-trial control batch moves the watermark — P0-2).
        """
        batch = [self._record_trial_params_for(write) for write in trials]
        cursor_params = self._write_cursor_params(consumer, progress)

        async def _work(tx: AsyncManagedTransaction) -> None:
            for params in batch:
                await tx.run(_RECORD_TRIAL_CYPHER, **params)
            await tx.run(_ADVANCE_CURSOR_CYPHER, **cursor_params)

        async with self._connection.session() as session:
            await session.execute_write(_work)

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
        """MERGE one trial (idempotent on ``trial_id``, first-write-wins) plus its node/edge graph.

        Validates ``trial_id`` format (``f"{run_id}:{step_index}"``) and reconstructs the full
        :class:`Trial` (validating it) before any I/O so a malformed id never reaches Neo4j. The
        APPLIES_TO edge is topology-only; no derived value is stored.

        When ``cursor`` is supplied, the projection cursor is advanced monotonically in the SAME
        managed transaction as the trial MERGE (both ``tx.run`` calls share one txn -> atomic
        commit, S6 / spike a6). The per-tick projection uses :meth:`project_batch`.
        """
        params = self._record_trial_params_for(
            TrialWrite(
                trial_id=trial_id,
                procedure_id=procedure_id,
                problem_type=problem_type,
                outcome=outcome,
                occurred_at=occurred_at,
                provenance=provenance,
            )
        )
        cursor_params = (
            self._cursor_params(cursor.consumer, cursor.cursor) if cursor is not None else None
        )

        async def _work(tx: AsyncManagedTransaction) -> None:
            await tx.run(_RECORD_TRIAL_CYPHER, **params)
            if cursor_params is not None:
                await tx.run(_ADVANCE_CURSOR_CYPHER, **cursor_params)

        async with self._connection.session() as session:
            await session.execute_write(_work)

    def _record_trial_params_for(self, write: TrialWrite) -> dict[str, Any]:
        # Construct the Trial so pydantic validates the payload (frozen + typed) before persistence,
        # mirroring entity-KG's identity assertion-before-I/O discipline.
        run_id, step_index = split_trial_id(write.trial_id)
        trial = Trial(
            trial_id=write.trial_id,
            run_id=run_id,
            step_index=step_index,
            procedure_id=write.procedure_id,
            problem_type=write.problem_type,
            outcome=write.outcome,
            occurred_at=to_utc(write.occurred_at),
            provenance=write.provenance,
        )
        return _record_trial_params(
            trial,
            procedure_label=self._procedure_labels.get(write.procedure_id, write.procedure_id),
            problem_type_label=self._problem_type_labels.get(
                write.problem_type, write.problem_type
            ),
        )

    @staticmethod
    def _cursor_params(consumer: str, cursor: ProjectionCursor) -> dict[str, Any]:
        # commit_ordinal is a plain integer (the journal's commit_xid); no ISO-string hack. The
        # cursor is never null here (record_trial only advances when a CursorAdvance is supplied).
        return {
            "consumer": consumer,
            "commit_ordinal": cursor.commit_ordinal,
            "cursor_run_id": cursor.run_id,
            "cursor_step_index": cursor.step_index,
        }

    @staticmethod
    def _write_cursor_params(consumer: str, progress: ProjectionCursor | None) -> dict[str, Any]:
        # Nullable cursor triple for _ADVANCE_CURSOR_CYPHER: a null commit_ordinal leaves the stored
        # cursor unchanged (an empty read). commit_ordinal is a plain integer.
        return {
            "consumer": consumer,
            "commit_ordinal": progress.commit_ordinal if progress is not None else None,
            "cursor_run_id": progress.run_id if progress is not None else None,
            "cursor_step_index": progress.step_index if progress is not None else None,
        }

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        """Return the consumer's projection cursor (None if never advanced)."""

        async def _work(tx: AsyncManagedTransaction) -> Record | None:
            result = await tx.run(_READ_CURSOR_CYPHER, consumer=consumer)
            return await result.single()

        async with self._connection.session() as session:
            record = await session.execute_read(_work)
        if record is None:
            return None
        data = record.data() if hasattr(record, "data") else dict(record)
        if data["commit_ordinal"] is None:
            return None
        return ProjectionCursor(
            commit_ordinal=int(data["commit_ordinal"]),
            run_id=str(data["cursor_run_id"]),
            step_index=int(data["cursor_step_index"]),
        )

    async def get_trial(self, trial_id: str) -> Trial | None:
        """Return the trial by id, or ``None``."""

        async def _work(tx: AsyncManagedTransaction) -> Record | None:
            result = await tx.run(_GET_TRIAL_CYPHER, trial_id=trial_id)
            return await result.single()

        async with self._connection.session() as session:
            record = await session.execute_read(_work)
        return _record_to_trial(record) if record is not None else None

    async def trials_for(self, procedure_id: str, problem_type: str) -> Sequence[Trial]:
        """Return all trials on the ``(procedure_id, problem_type)`` edge, in occurrence order."""

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(
                _TRIALS_FOR_CYPHER, procedure_id=procedure_id, problem_type=problem_type
            )
            return [r async for r in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)
        return tuple(_record_to_trial(r) for r in records)

    async def posterior(self, procedure_id: str, problem_type: str) -> ProcedureSuccess:
        """Derive the Beta-posterior success summary for one edge at read (empty → prior-only)."""

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(
                _POSTERIOR_TRIALS_CYPHER, procedure_id=procedure_id, problem_type=problem_type
            )
            return [r async for r in result]

        async with self._connection.session() as session:
            rows = await session.execute_read(_work)
        return _rows_to_posterior(rows)

    async def candidates(
        self,
        problem_type: str,
        *,
        limit: int = 20,
        promoted_only: bool = False,
        rng: random.Random | None = None,
    ) -> Sequence[ScoredProcedure]:
        """Return scored candidate procedures for a problem type, ranked for the read surface.

        Each candidate's ``promoted`` is stamped fresh from the adapter's promotion policy
        (derive-at-read); ``promoted_only`` keeps only promoted edges. The PRODUCTION ranking is
        deterministic posterior-mean greedy (no RNG). Passing ``rng`` (or constructing the adapter
        with one) opts into the DEFERRED Thompson read surface instead, drawing
        ``theta ~ Beta(alpha, beta)`` per edge; a pinned seed then gives deterministic order —
        identical to the in-memory double, which calls the SAME :func:`select_candidates`.
        """

        async def _work(tx: AsyncManagedTransaction) -> list[Record]:
            result = await tx.run(_CANDIDATES_CYPHER, problem_type=problem_type)
            return [r async for r in result]

        async with self._connection.session() as session:
            records = await session.execute_read(_work)

        scored: list[ScoredProcedure] = []
        for record in records:
            data = record.data() if hasattr(record, "data") else dict(record)
            rows = list(data["trial_rows"] or [])
            # OPTIONAL MATCH on a procedure with no trials yields a single {run_id:null,...} row.
            outcomes = [
                TrialOutcome(run_id=str(r["run_id"]), success=r["outcome"] == "success")
                for r in rows
                if r.get("run_id") is not None
            ]
            success = procedure_success(outcomes)
            embedding_raw = data["problem_type_embedding"]
            embedding = tuple(float(v) for v in embedding_raw) if embedding_raw else None
            scored.append(
                ScoredProcedure(
                    procedure=Procedure(
                        id=str(data["procedure_id"]), label=str(data["procedure_label"])
                    ),
                    problem_type=ProblemType(
                        id=str(data["problem_type_id"]),
                        label=str(data["problem_type_label"]),
                        embedding=embedding,
                    ),
                    success=success,
                    promoted=False,
                )
            )

        return select_candidates(
            scored,
            rng=rng if rng is not None else self._rng,
            limit=limit,
            promoted_only=promoted_only,
            policy=self._promotion_policy,
        )  # rng None => production posterior-mean greedy; rng set => deferred Thompson


# ---------------------------------------------------------------------------
# Shared selection (so the adapter and the double select identically)
# ---------------------------------------------------------------------------


def select_candidates(
    scored: list[ScoredProcedure],
    *,
    rng: random.Random | None,
    limit: int,
    promoted_only: bool,
    policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
) -> tuple[ScoredProcedure, ...]:
    """Candidate selection shared by the adapter and the in-memory double (single-sourced).

    Single-sourced here so the two implementations can NEVER drift on ranking or promotion: both
    call this with the same ``policy``, so the order and the ``promoted`` stamp are byte-identical.

    The PRODUCTION default (``rng is None``) is deterministic posterior-mean greedy
    (:func:`~cogworx.substrate.procedural_selection.posterior_mean_select`) — what ships after the
    Pod 2.1 spike. Thompson sampling
    (:func:`~cogworx.substrate.procedural_selection.thompson_select`) is DEFERRED and used ONLY when
    a caller explicitly opts in by passing an ``rng`` (it did not beat a fair baseline in the spike;
    kept callable for a future spike). This wrapper only fixes the call shape the adapter/double
    share.
    """
    if rng is None:
        return posterior_mean_select(
            scored,
            limit=limit,
            promoted_only=promoted_only,
            policy=policy,
        )
    return thompson_select(
        scored,
        rng=rng,
        limit=limit,
        promoted_only=promoted_only,
        policy=policy,
    )


def _record_trial_params(
    trial: Trial,
    *,
    procedure_label: str,
    problem_type_label: str,
) -> dict[str, Any]:
    """Build the parameter dict for the record_trial Cypher.

    ``occurred_at`` is already UTC on the Trial; ``recorded_at`` is UTC-normalised here so all
    stored ISO strings share the +00:00 offset.
    """
    prov = trial.provenance
    return {
        "trial_id": trial.trial_id,
        "run_id": trial.run_id,
        "step_index": trial.step_index,
        "procedure_id": trial.procedure_id,
        "procedure_label": procedure_label,
        "problem_type": trial.problem_type,
        "problem_type_label": problem_type_label,
        "problem_type_embedding": None,
        "outcome": trial.outcome,
        "occurred_at": to_utc(trial.occurred_at).isoformat(),
        "prov_source": prov.source,
        "prov_source_ref": prov.source_ref,
        "prov_confidence": prov.confidence,
        "prov_evidence": list(prov.evidence),
        "prov_recorded_at": to_utc(prov.recorded_at).isoformat(),
    }
