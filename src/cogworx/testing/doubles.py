"""In-memory substrate doubles for the Test Kit (CANON S3, S6).

Reference implementations of the distinct substrate seams (journal / graph / latent / entity-KG) —
kept separate so the doubles exercise the same engine-shaped contracts the real adapters do (S3, no
flattening ``Store``). ``InMemoryJournal`` is the S6 reference behaviour: exactly-once on
``(run_id, step_index)``, with the run's status PERSISTED on the run record (the authority — not
derived from the last step) and the run's pathway pointer stored for cold resume.
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable, Sequence
from datetime import UTC, datetime, timedelta

from cogworx.adapters.neo4j_procedural_kg import select_candidates, split_trial_id
from cogworx.claims.provenance import Artifact, Claim, Provenance
from cogworx.knowledge.confidence import ClaimConfidence, claim_confidence
from cogworx.knowledge.evidence import EvidenceEvent
from cogworx.knowledge.identity import claim_id_for, normalize_topic_part
from cogworx.knowledge.latent_activation import ActivationParams, ActivationRow, select_hot_ids
from cogworx.knowledge.procedural_confidence import (
    ProcedureSuccess,
    TrialOutcome,
    procedure_success,
)
from cogworx.knowledge.procedural_promotion import PromotionPolicy
from cogworx.loop.state import RunStatus
from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.journal import (
    ProjectedStep,
    ProjectionCursor,
    RunState,
    StepRecord,
    Timer,
)
from cogworx.substrate.latent import LatentMatch, LatentRecord, Tier, TierSweepResult
from cogworx.substrate.procedural_kg import (
    CursorAdvance,
    Outcome,
    ProblemType,
    Procedure,
    ScoredProcedure,
    Trial,
    TrialWrite,
    ordinal_ge,
)
from cogworx.substrate.procedural_selection import DEFAULT_PROMOTION_POLICY


class _RunLog:
    def __init__(
        self,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        self.session_id = session_id
        self.pathway_id = pathway_id
        self.pathway_version = pathway_version
        self.pathway_fingerprint = pathway_fingerprint
        self.status = RunStatus.RUNNING
        self.steps: dict[int, StepRecord] = {}


class InMemoryJournal:
    """An in-memory ``Journal``: exactly-once positional commits + persisted run status."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunLog] = {}
        self._timers: dict[str, Timer] = {}
        # The FAILURE counter keyed (run_id, step_index) — mirrors the adapter's step_attempts row.
        self._attempts: dict[tuple[str, int], int] = {}
        # HITL answers keyed (run_id, step_index) — FIRST-ANSWER-WINS (setdefault, never overwrite).
        self._human_inputs: dict[tuple[str, int], Artifact] = {}
        # The in-memory mirror of Postgres' commit_xid: a strictly increasing per-commit ordinal,
        # assigned in commit ORDER, keyed by the step's (run_id, step_index). There is no MVCC here
        # (commits are synchronous, single-threaded under asyncio), so no visibility fence is needed
        # — the strict-keyset forward read over this ordinal is total and matches the FENCED adapter
        # read on quiescent data.
        self._commit_ordinals: dict[tuple[str, int], int] = {}
        self._next_ordinal = 1

    async def start_run(
        self,
        run_id: str,
        session_id: str,
        *,
        pathway_id: str,
        pathway_version: int,
        pathway_fingerprint: str,
    ) -> None:
        if run_id not in self._runs:
            self._runs[run_id] = _RunLog(
                session_id,
                pathway_id=pathway_id,
                pathway_version=pathway_version,
                pathway_fingerprint=pathway_fingerprint,
            )

    async def set_run_status(self, run_id: str, status: RunStatus) -> None:
        log = self._runs.get(run_id)
        if log is not None:
            log.status = status

    async def compare_and_set_run_status(
        self, run_id: str, *, expect: RunStatus, new: RunStatus
    ) -> bool:
        # Atomic under asyncio: the read and the write straddle NO ``await``, so no other task can
        # interleave between them — mirrors the adapter's single row-locked conditional UPDATE.
        log = self._runs.get(run_id)
        if log is None or log.status is not expect:
            return False
        log.status = new
        return True

    async def commit_step(self, record: StepRecord) -> None:
        log = self._runs.get(record.run_id)
        if log is None:
            return
        if record.step_index in log.steps:
            return
        log.steps[record.step_index] = record
        # Assign the commit ordinal in commit order (the in-memory mirror of commit_xid). Exactly
        # once: a duplicate (run_id, step_index) returns above before reaching here, so the ordinal
        # is never reassigned.
        self._commit_ordinals[(record.run_id, record.step_index)] = self._next_ordinal
        self._next_ordinal += 1

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        log = self._runs.get(run_id)
        if log is None:
            return None
        return log.steps.get(step_index)

    async def committed_steps_after(
        self, cursor: ProjectionCursor | None, *, limit: int
    ) -> Sequence[ProjectedStep]:
        # The COMMIT-ORDER projection read seam (S1): strict keyset over (commit_ordinal, run_id,
        # step_index) — the in-memory mirror of the FENCED commit_xid adapter read. No fence is
        # needed (no in-flight txns); the ordinal is monotonic-in-commit-order, so a single forward
        # read is total. Ordered by the lexicographic tuple, capped at ``limit``.
        def key(run_id: str, step_index: int) -> tuple[int, str, int]:
            return (self._commit_ordinals[(run_id, step_index)], run_id, step_index)

        lo = (
            (cursor.commit_ordinal, cursor.run_id, cursor.step_index)
            if cursor is not None
            else None
        )
        rows = [record for log in self._runs.values() for record in log.steps.values()]
        scanned = [r for r in rows if lo is None or key(r.run_id, r.step_index) > lo]
        scanned.sort(key=lambda r: key(r.run_id, r.step_index))
        return tuple(
            ProjectedStep(record=r, commit_ordinal=self._commit_ordinals[(r.run_id, r.step_index)])
            for r in scanned[:limit]
        )

    async def load_run(self, run_id: str) -> RunState | None:
        log = self._runs.get(run_id)
        if log is None:
            return None
        steps = tuple(log.steps[index] for index in sorted(log.steps))
        current_stage = steps[-1].stage_name if steps else None
        return RunState(
            run_id=run_id,
            session_id=log.session_id,
            status=log.status,
            pathway_id=log.pathway_id,
            pathway_version=log.pathway_version,
            pathway_fingerprint=log.pathway_fingerprint,
            current_stage=current_stage,
            steps=steps,
        )

    async def set_timer(self, timer: Timer) -> None:
        # Idempotent on timer_id (mirrors the adapter's ON CONFLICT DO NOTHING): a re-armed Wait is
        # a no-op, so a committed Wait that replays can never resurrect a cancelled timer.
        self._timers.setdefault(timer.timer_id, timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return tuple(timer for timer in self._timers.values() if timer.wake_at <= now)

    async def claim_due_timers(self, now: datetime, *, lease_ttl: timedelta) -> Sequence[Timer]:
        # Atomic-by-single-threadedness lease: stamp claimed_at on each due, unclaimed-or-stale
        # timer and return the stamped copies. The row SURVIVES — deleted only by cancel_* — a crash
        # between claim and advance is recoverable (the next stale-lease sweep re-fires it).
        stale_before = now - lease_ttl
        claimed: list[Timer] = []
        for timer_id, timer in self._timers.items():
            if timer.wake_at > now:
                continue
            if timer.claimed_at is not None and timer.claimed_at > stale_before:
                continue
            leased = timer.model_copy(update={"claimed_at": now})
            self._timers[timer_id] = leased
            claimed.append(leased)
        return tuple(claimed)

    async def cancel_timer(self, timer_id: str) -> None:
        self._timers.pop(timer_id, None)

    async def cancel_timers_for_run(self, run_id: str) -> None:
        for timer_id in [tid for tid, t in self._timers.items() if t.run_id == run_id]:
            del self._timers[timer_id]

    async def get_run_status(self, run_id: str) -> RunStatus | None:
        log = self._runs.get(run_id)
        return log.status if log is not None else None

    async def increment_attempt(self, run_id: str, step_index: int) -> int:
        # Atomic under asyncio (no await between read and write) — mirrors the adapter's single
        # INSERT … ON CONFLICT DO UPDATE attempt = attempt + 1 RETURNING attempt.
        key = (run_id, step_index)
        count = self._attempts.get(key, 0) + 1
        self._attempts[key] = count
        return count

    async def read_attempt(self, run_id: str, step_index: int) -> int:
        return self._attempts.get((run_id, step_index), 0)

    async def record_human_input(self, run_id: str, step_index: int, answer: Artifact) -> None:
        # FIRST-ANSWER-WINS: setdefault is atomic under asyncio (no await between read and write) —
        # mirrors the adapter's ON CONFLICT (run_id, step_index) DO NOTHING.
        self._human_inputs.setdefault((run_id, step_index), answer)

    async def read_human_input(self, run_id: str, step_index: int) -> Artifact | None:
        return self._human_inputs.get((run_id, step_index))


class InMemoryGraphStore:
    """An in-memory ``GraphStore``. Recall (``neighbors``) is Phase 2 — kept thin here."""

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}

    async def upsert_claim(self, claim: Claim) -> str:
        self._claims[claim.id] = claim
        return claim.id

    async def get_claim(self, claim_id: str) -> Claim | None:
        return self._claims.get(claim_id)

    async def neighbors(self, claim_id: str, *, limit: int = 20) -> Sequence[Claim]:
        return ()


class InMemoryLatentStore:
    """An in-memory ``LatentStore`` — full Pod 2.2 Protocol (put/record_use/search/sweep_tiers).

    Implements the same contract as ``PgLatentStore`` without a database. All state is in plain
    dicts; usage metadata (use_count, last_used_at, tier, created_at) is kept separately from
    content so ``put`` (content-only) and ``record_use`` (usage-only) are structurally isolated —
    the same invariant the adapter enforces via separate SQL clauses.
    """

    def __init__(self, *, clock: Callable[[], datetime] | None = None) -> None:
        self._records: dict[str, LatentRecord] = {}
        self._use_count: dict[str, int] = {}
        self._last_used_at: dict[str, datetime] = {}
        self._created_at: dict[str, datetime] = {}
        self._tier: dict[str, Tier] = {}
        self._clock: Callable[[], datetime] = clock or (lambda: datetime.now(UTC))

    async def put(self, record: LatentRecord) -> None:
        """Insert-or-replace content; never touches use_count / last_used_at / tier / created_at."""
        now = self._clock()
        if record.id not in self._records:
            self._use_count[record.id] = 0
            self._last_used_at[record.id] = now
            self._created_at[record.id] = now
            self._tier[record.id] = "cold"
        self._records[record.id] = record

    async def record_use(self, ids: Sequence[str]) -> int:
        """Atomically increment use_count + advance last_used_at for each known id."""
        now = self._clock()
        count = 0
        for id_ in ids:
            if id_ in self._records:
                self._use_count[id_] = self._use_count.get(id_, 0) + 1
                self._last_used_at[id_] = now
                count += 1
        return count

    async def search(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        tier: Tier | None = None,
    ) -> Sequence[LatentMatch]:
        """Exact cosine top-k; ``tier`` scopes to hot/cold only (None = all)."""
        query = tuple(embedding)
        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0.0 or not self._records:
            return ()
        now = self._clock()
        matches: list[LatentMatch] = []
        for id_, record in self._records.items():
            if tier is not None and self._tier.get(id_, "cold") != tier:
                continue
            score = self._cosine(query, query_norm, record.embedding)
            if score is None:
                continue
            matches.append(
                LatentMatch(
                    record=record,
                    score=score,
                    tier=self._tier.get(id_, "cold"),
                    use_count=self._use_count.get(id_, 0),
                    last_used_at=self._last_used_at.get(id_, now),
                )
            )
        matches.sort(key=lambda m: (-m.score, m.record.id))
        return tuple(matches[:k])

    async def sweep_tiers(self, *, now: datetime, hot_capacity: int) -> TierSweepResult:
        """Re-assign hot/cold tiers using ACT-R activation ranking. Pure-Python reference."""
        rows = [
            ActivationRow(
                id=id_,
                use_count=self._use_count.get(id_, 0),
                last_used_at=self._last_used_at.get(id_, now),
            )
            for id_ in self._records
        ]
        params = ActivationParams(hot_capacity=hot_capacity)
        hot_ids = select_hot_ids(rows, now, params)
        promoted = demoted = 0
        for id_ in self._records:
            current = self._tier.get(id_, "cold")
            should_hot = id_ in hot_ids
            if should_hot and current == "cold":
                self._tier[id_] = "hot"
                promoted += 1
            elif not should_hot and current == "hot":
                self._tier[id_] = "cold"
                demoted += 1
        hot_size = sum(1 for t in self._tier.values() if t == "hot")
        return TierSweepResult(promoted=promoted, demoted=demoted, hot_size=hot_size)

    @staticmethod
    def _cosine(
        query: tuple[float, ...], query_norm: float, candidate: tuple[float, ...]
    ) -> float | None:
        if len(candidate) != len(query):
            return None
        candidate_norm = math.sqrt(sum(value * value for value in candidate))
        if candidate_norm == 0.0:
            return None
        dot = sum(a * b for a, b in zip(query, candidate, strict=True))
        return dot / (query_norm * candidate_norm)


def _to_utc_mem(dt: datetime) -> datetime:
    """Naive → attach UTC; tz-aware → convert to UTC. Mirrors the adapter's to_utc()."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


class InMemoryEntityKG:
    """An in-memory :class:`~cogworx.substrate.entity_kg.EntityKG` for deterministic unit tests.

    Implements the full EntityKG contract in dicts — no Neo4j, no model calls. Cosine similarity
    for ``claims_by_similarity`` is computed in pure Python (same algorithm as InMemoryLatentStore).

    Same identity-discipline ValueError, same invalidate first-wins, same as_of filtering, same
    lineage weakest-link (recursive ancestor walk capped at depth 5 with cycle guard) as the real
    adapter. Confidence is always re-derived — never stored.

    Datetime contract: naive datetimes are interpreted as UTC (not rejected). Tz-aware datetimes
    are converted to UTC before comparison. This matches the Neo4jEntityKG adapter's behaviour.
    """

    def __init__(self) -> None:
        self._claims: dict[str, Claim] = {}
        # evidence events indexed by claim_id — append-only
        self._evidence: dict[str, list[EvidenceEvent]] = {}
        # contradiction pairs stored as frozenset so both directions query the same set
        self._contradictions: set[frozenset[str]] = set()

    async def upsert_claim(self, claim: Claim) -> str:
        """Sealed: raises NotImplementedError. Use write_claim instead.

        Matches the real adapter's sealed write-seam (FIX 1). The Phase-0 upsert_claim bypasses
        identity discipline, immutable-on-match, and bi-temporal honesty.
        """
        raise NotImplementedError(
            "InMemoryEntityKG.upsert_claim is disabled. "
            "The entity KG's only write surface is write_claim (identity discipline + "
            "immutable-on-match + bi-temporal honesty). "
            "The inherited Phase-0 upsert_claim bypasses all three invariants."
        )

    async def write_claim(self, claim: Claim, *, evidence: EvidenceEvent) -> str:
        """MERGE claim + record evidence. Raises ValueError on identity-discipline failure."""
        _assert_identity_mem(claim)
        if claim.id not in self._claims:
            self._claims[claim.id] = claim
        if claim.id not in self._evidence:
            self._evidence[claim.id] = []
        self._evidence[claim.id].append(evidence)
        return claim.id

    async def add_evidence(self, claim_id: str, event: EvidenceEvent) -> None:
        """Append an evidence event. Raises ValueError if claim_id unknown."""
        if claim_id not in self._claims:
            raise ValueError(f"add_evidence: unknown claim_id {claim_id!r}")
        self._evidence[claim_id].append(event)

    async def get_claim(self, claim_id: str) -> Claim | None:
        """Return the claim by id, or None."""
        return self._claims.get(claim_id)

    async def evidence_for(self, claim_id: str) -> Sequence[EvidenceEvent]:
        """Return all evidence events for the claim in creation order."""
        return tuple(self._evidence.get(claim_id, []))

    async def claims_about(
        self,
        entity: str,
        *,
        limit: int = 20,
        as_of: datetime | None = None,
    ) -> Sequence[ScoredClaim]:
        """Return scored claims where entity is subject or object, newest-first."""
        # UTC-normalise as_of so naive datetimes are accepted (matches adapter contract).
        as_of_utc = _to_utc_mem(as_of) if as_of is not None else None
        results: list[ScoredClaim] = []
        seen: set[str] = set()
        # Sort by ingest_time DESC for consistent ordering
        for claim in sorted(self._claims.values(), key=lambda c: c.ingest_time, reverse=True):
            if claim.id in seen:
                continue
            if claim.subject != entity and claim.object_entity != entity:
                continue
            if as_of_utc is not None:
                # Normalise claim datetimes to UTC for comparison so mixed-offset claims compare
                # correctly. This mirrors what the Neo4j adapter does at write time.
                vf = _to_utc_mem(claim.valid_from)
                if vf > as_of_utc:
                    continue
                if claim.valid_to is not None and _to_utc_mem(claim.valid_to) <= as_of_utc:
                    continue
            seen.add(claim.id)
            own_conf = claim_confidence(self._evidence.get(claim.id, []))
            lineage_min = self._lineage_min(claim.id, own_conf.confidence, depth=5)
            results.append(
                ScoredClaim(
                    claim=claim,
                    confidence=own_conf,
                    lineage_min_confidence=lineage_min,
                )
            )
            if len(results) >= limit:
                break
        return tuple(results)

    async def claims_by_similarity(
        self,
        embedding: Sequence[float],
        *,
        k: int = 10,
        min_score: float = 0.70,
    ) -> Sequence[ScoredClaim]:
        """Return up to k scored claims nearest to embedding by raw cosine (>= min_score)."""
        query = tuple(embedding)
        query_norm = math.sqrt(sum(v * v for v in query))
        if query_norm == 0.0:
            return ()

        scored: list[tuple[float, ScoredClaim]] = []
        for claim in self._claims.values():
            if claim.embedding is None:
                continue
            raw = _cosine_sim(query, query_norm, claim.embedding)
            if raw is None or raw < min_score:
                continue
            own_conf = claim_confidence(self._evidence.get(claim.id, []))
            lineage_min = self._lineage_min(claim.id, own_conf.confidence, depth=5)
            scored.append(
                (
                    raw,
                    ScoredClaim(
                        claim=claim,
                        confidence=own_conf,
                        lineage_min_confidence=lineage_min,
                        similarity=raw,
                    ),
                )
            )

        scored.sort(key=lambda pair: pair[0], reverse=True)
        return tuple(sc for _, sc in scored[:k])

    async def resolution_candidates(
        self,
        subject: str,
        predicate: str,
        *,
        embedding: Sequence[float] | None = None,
        k: int = 5,
    ) -> Sequence[Claim]:
        """Return candidate claims by exact topic-column match, topped up by vector if needed."""
        subject_norm = normalize_topic_part(subject)
        predicate_norm = normalize_topic_part(predicate)

        # Primary: exact (subject_norm, predicate_norm), newest first
        topic_hits = sorted(
            [
                c
                for c in self._claims.values()
                if normalize_topic_part(c.subject) == subject_norm
                and normalize_topic_part(c.predicate or "") == predicate_norm
            ],
            key=lambda c: c.ingest_time,
            reverse=True,
        )

        seen: set[str] = {c.id for c in topic_hits[:k]}
        candidates: list[Claim] = list(topic_hits[:k])

        if len(candidates) < k and embedding is not None:
            query = tuple(embedding)
            query_norm = math.sqrt(sum(v * v for v in query))
            if query_norm > 0.0:
                vec_hits: list[tuple[float, Claim]] = []
                for claim in self._claims.values():
                    if claim.id in seen or claim.embedding is None:
                        continue
                    raw = _cosine_sim(query, query_norm, claim.embedding)
                    if raw is not None and raw >= 0.70:
                        vec_hits.append((raw, claim))
                vec_hits.sort(key=lambda pair: pair[0], reverse=True)
                for _, claim in vec_hits:
                    if claim.id not in seen:
                        seen.add(claim.id)
                        candidates.append(claim)
                        if len(candidates) >= k:
                            break

        return tuple(candidates[:k])

    async def write_contradiction(self, claim_id_a: str, claim_id_b: str) -> None:
        """Record a CONTRADICTS pair (idempotent, semantically undirected)."""
        self._contradictions.add(frozenset({claim_id_a, claim_id_b}))

    async def contradictions_of(self, claim_id: str) -> Sequence[Claim]:
        """Return all claims that contradict the given claim."""
        result: list[Claim] = []
        for pair in self._contradictions:
            if claim_id in pair:
                other_ids = pair - {claim_id}
                for other_id in other_ids:
                    claim = self._claims.get(other_id)
                    if claim is not None:
                        result.append(claim)
        return tuple(result)

    async def invalidate_claim(self, claim_id: str, *, valid_to: datetime) -> None:
        """Set valid_to (first-invalidation-wins). Raises ValueError if claim unknown.

        valid_to is UTC-normalised before storage (matches adapter contract).
        """
        claim = self._claims.get(claim_id)
        if claim is None:
            raise ValueError(f"invalidate_claim: unknown claim_id {claim_id!r}")
        # First-invalidation-wins: only update if currently None.
        if claim.valid_to is None:
            self._claims[claim_id] = claim.model_copy(update={"valid_to": _to_utc_mem(valid_to)})

    def _lineage_min(
        self,
        claim_id: str,
        own_confidence: float,
        *,
        depth: int,
        _visited: frozenset[str] | None = None,
    ) -> float:
        """Weakest-link lineage confidence (recursive DERIVED_FROM walk, capped at depth 5).

        Cycle guard via _visited so a diamond lineage (same ancestor reachable via multiple paths)
        neither double-counts nor loops infinitely.
        """
        if depth == 0:
            return own_confidence
        visited = _visited if _visited is not None else frozenset()
        visited = visited | {claim_id}

        claim = self._claims.get(claim_id)
        if claim is None:
            return own_confidence

        min_conf = own_confidence
        for parent_id in claim.provenance.evidence:
            if parent_id in visited:
                continue
            parent_claim = self._claims.get(parent_id)
            if parent_claim is None:
                # Skeleton/unwritten ancestor: treat as prior-only confidence (Beta(1,1).mean = 0.5)
                # so the lineage weakest-link correctly penalises claims derived from unresolved
                # parents. This matches the Neo4j adapter's behaviour (_derive_confidence sees an
                # ancestor with null evidence rows → claim_confidence([]) = 0.5).
                min_conf = min(min_conf, 0.5)
                continue
            parent_events = self._evidence.get(parent_id, [])
            parent_conf = claim_confidence(parent_events).confidence
            # Recurse to collect the parent's own lineage min.
            parent_lineage_min = self._lineage_min(
                parent_id,
                parent_conf,
                depth=depth - 1,
                _visited=visited,
            )
            min_conf = min(min_conf, parent_lineage_min)

        return min_conf


def _assert_identity_mem(claim: Claim) -> None:
    """Enforce identity discipline in the in-memory double (same rule as the real adapter)."""
    object_repr = claim.object_entity if claim.object_entity is not None else claim.payload
    expected = claim_id_for(claim.subject, claim.predicate or "", object_repr)
    if claim.id != expected:
        raise ValueError(
            f"claim.id {claim.id!r} does not match expected "
            f"claim_id_for({claim.subject!r}, {claim.predicate!r}, {object_repr!r}) = {expected!r}"
        )


def _cosine_sim(
    query: tuple[float, ...], query_norm: float, candidate: tuple[float, ...]
) -> float | None:
    """Raw cosine similarity in [-1, 1]. Returns None if candidate is zero-vector or mismatched."""
    if len(candidate) != len(query):
        return None
    candidate_norm = math.sqrt(sum(v * v for v in candidate))
    if candidate_norm == 0.0:
        return None
    dot = sum(a * b for a, b in zip(query, candidate, strict=True))
    return dot / (query_norm * candidate_norm)


def _confidence_from_claims(
    claims: list[Claim],
    evidence_store: dict[str, list[EvidenceEvent]],
) -> dict[str, ClaimConfidence]:
    """Batch-derive ClaimConfidence for a list of claims from the evidence store."""
    return {c.id: claim_confidence(evidence_store.get(c.id, [])) for c in claims}


class InMemoryProceduralKG:
    """In-memory :class:`~cogworx.substrate.procedural_kg.ProceduralKG` for deterministic tests.

    Held to BEHAVIOURAL PARITY with :class:`~cogworx.adapters.neo4j_procedural_kg.Neo4jProceduralKG`
    by the shared parity suite. Implements the full ProceduralKG contract in dicts — no Neo4j, no
    model calls (S1). The posterior is ALWAYS re-derived via the single computation path
    (:func:`cogworx.knowledge.procedural_confidence.procedure_success`) — never stored. The
    ``APPLIES_TO`` topology is tracked as a set; no counters live on it.

    Hard invariants matched to the adapter:
      - ``record_trial`` MERGEs on ``trial_id`` first-write-wins (setdefault); a re-call with the
        same id and a different payload is a no-op (spike a5).
      - ``upsert_claim`` is SEALED to NotImplementedError (2.0 red-team #1 / entity-KG FIX 1).
      - ``trial_id`` is validated + split via the SAME :func:`split_trial_id` the adapter uses.
      - ``occurred_at`` is UTC-normalised (naive → UTC, tz-aware → converted) before storage so the
        occurrence order and the run-level dedup grain are identical to the adapter.
      - ``candidates`` selection reuses the adapter's :func:`select_candidates` verbatim, so the
        production deterministic posterior-mean greedy order (and the opt-in seeded Thompson order)
        are byte-identical across both implementations.
    """

    def __init__(
        self,
        *,
        procedure_labels: dict[str, str] | None = None,
        problem_type_labels: dict[str, str] | None = None,
        promotion_policy: PromotionPolicy = DEFAULT_PROMOTION_POLICY,
        rng: random.Random | None = None,
    ) -> None:
        # Trials keyed by trial_id — FIRST-WRITE-WINS (setdefault), never updated.
        self._trials: dict[str, Trial] = {}
        # Topology-only APPLIES_TO edges: set of (procedure_id, problem_type). No counters.
        self._applies_to: set[tuple[str, str]] = set()
        # Declared node labels (coalesce-on-create: first label wins), id-keyed.
        self._procedure_labels = dict(procedure_labels or {})
        self._problem_type_labels = dict(problem_type_labels or {})
        # Insertion order of declaration, mirroring ON CREATE label population.
        self._procedures_seen: dict[str, None] = {}
        self._problem_types_seen: dict[str, None] = {}
        # Projection cursors keyed by consumer — the in-memory mirror of (:ProjectionCursor). The
        # cursor advances MONOTONICALLY (tuple-max via ordinal_ge) in the same logical "txn" as the
        # trial write(s) (no await between them, so a concurrent task can't observe a half batch).
        self._progress: dict[str, ProjectionCursor] = {}
        self._promotion_policy = promotion_policy
        # OPT-IN Thompson RNG. None => candidates uses production posterior-mean greedy; an injected
        # rng opts into the deferred Thompson read surface (mirrors the adapter).
        self._rng = rng

    async def upsert_claim(self, claim: Claim) -> str:
        """Sealed: raises NotImplementedError. Use record_trial instead.

        Matches the adapter's sealed write-seam (2.0 red-team #1): the inherited Phase-0
        blanket-SET ``upsert_claim`` is a back-door around Trial immutability (first-write-wins).
        """
        raise NotImplementedError(
            "InMemoryProceduralKG.upsert_claim is disabled. "
            "The procedural KG's only write surface is record_trial "
            "(ON CREATE-only MERGE: trials are immutable, first-write-wins). "
            "The inherited Phase-0 upsert_claim is a blanket-SET back-door."
        )

    async def project_batch(
        self,
        consumer: str,
        *,
        trials: Sequence[TrialWrite],
        progress: ProjectionCursor | None,
    ) -> None:
        """MERGE a batch of trials THEN advance the cursor in one atomic step (S6).

        Mirrors the adapter's single-txn ``project_batch``: each trial MERGEs first-write-wins, then
        the cursor advances tuple-max (``None`` leaves it unchanged). No ``await`` between the trial
        writes and the cursor write, so a concurrent task cannot observe a half-applied batch. An
        empty ``trials`` sequence still advances the cursor (a zero-trial control batch — P0-2).
        """
        for write in trials:
            self._merge_trial(write)
        self._advance_progress(consumer, progress)

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
        """MERGE one trial (idempotent on trial_id, first-write-wins) + node/edge graph.

        Validates trial_id format before any state change so a malformed id never enters the store.
        When ``cursor`` is supplied the keyset advances monotonically in the same atomic step as the
        trial MERGE (mirrors the adapter's same-txn advance, S6 / spike a6). The per-tick projection
        uses ``project_batch``.
        """
        self._merge_trial(
            TrialWrite(
                trial_id=trial_id,
                procedure_id=procedure_id,
                problem_type=problem_type,
                outcome=outcome,
                occurred_at=occurred_at,
                provenance=provenance,
            )
        )
        if cursor is not None:
            self._advance_progress(cursor.consumer, cursor.cursor)

    def _merge_trial(self, write: TrialWrite) -> None:
        run_id, step_index = split_trial_id(write.trial_id)
        trial = Trial(
            trial_id=write.trial_id,
            run_id=run_id,
            step_index=step_index,
            procedure_id=write.procedure_id,
            problem_type=write.problem_type,
            outcome=write.outcome,
            occurred_at=_to_utc_mem(write.occurred_at),
            provenance=write.provenance,
        )
        # MERGE node labels (first-label-wins) + topology edge as a side effect, ALWAYS — even when
        # the trial is a duplicate (mirrors the Cypher MERGE-then-MERGE-trial order: the
        # Procedure/ProblemType/APPLIES_TO MERGEs run before the ON-CREATE trial guard).
        self._procedures_seen.setdefault(write.procedure_id, None)
        self._problem_types_seen.setdefault(write.problem_type, None)
        self._procedure_labels.setdefault(write.procedure_id, write.procedure_id)
        self._problem_type_labels.setdefault(write.problem_type, write.problem_type)
        self._applies_to.add((write.procedure_id, write.problem_type))
        # FIRST-WRITE-WINS: setdefault never overwrites an existing trial_id (no ON MATCH SET).
        self._trials.setdefault(write.trial_id, trial)

    def _advance_progress(self, consumer: str, cursor: ProjectionCursor | None) -> None:
        # None leaves the stored cursor unchanged (an empty read).
        if cursor is None:
            return
        current = self._progress.get(consumer)
        # Tuple-max via the SHARED ordinal_ge so the double and the adapter can never drift on the
        # never-regress rule: keep current iff it is >= incoming, else advance.
        if current is None or not ordinal_ge(current, cursor):
            self._progress[consumer] = cursor

    async def read_cursor(self, consumer: str) -> ProjectionCursor | None:
        """Return the consumer's projection cursor (None if never advanced)."""
        return self._progress.get(consumer)

    async def get_trial(self, trial_id: str) -> Trial | None:
        """Return the trial by id, or None."""
        return self._trials.get(trial_id)

    async def trials_for(self, procedure_id: str, problem_type: str) -> Sequence[Trial]:
        """Return all trials on the edge, in occurrence order (occurred_at, then trial_id)."""
        edge_trials = [
            t
            for t in self._trials.values()
            if t.procedure_id == procedure_id and t.problem_type == problem_type
        ]
        edge_trials.sort(key=lambda t: (_to_utc_mem(t.occurred_at), t.trial_id))
        return tuple(edge_trials)

    async def posterior(self, procedure_id: str, problem_type: str) -> ProcedureSuccess:
        """Derive the Beta-posterior success summary for one edge at read (empty → prior-only)."""
        outcomes = [
            TrialOutcome(run_id=t.run_id, success=t.outcome == "success")
            for t in self._trials.values()
            if t.procedure_id == procedure_id and t.problem_type == problem_type
        ]
        return procedure_success(outcomes)

    async def candidates(
        self,
        problem_type: str,
        *,
        limit: int = 20,
        promoted_only: bool = False,
        rng: random.Random | None = None,
    ) -> Sequence[ScoredProcedure]:
        """Return scored candidate procedures for a problem type, ranked for the read surface.

        Identical selection to the adapter via the shared :func:`select_candidates`: ``promoted`` is
        stamped fresh from the promotion policy. With no ``rng`` (production) the order is
        deterministic posterior-mean greedy; passing ``rng`` (or constructing with one) opts into
        the deferred Thompson read surface and a pinned seed gives byte-identical order to
        :class:`~cogworx.adapters.neo4j_procedural_kg.Neo4jProceduralKG`.
        """
        pt_label = self._problem_type_labels.get(problem_type, problem_type)
        problem_node = ProblemType(id=problem_type, label=pt_label, embedding=None)

        scored: list[ScoredProcedure] = []
        for procedure_id, edge_pt in self._applies_to:
            if edge_pt != problem_type:
                continue
            outcomes = [
                TrialOutcome(run_id=t.run_id, success=t.outcome == "success")
                for t in self._trials.values()
                if t.procedure_id == procedure_id and t.problem_type == problem_type
            ]
            success = procedure_success(outcomes)
            scored.append(
                ScoredProcedure(
                    procedure=Procedure(
                        id=procedure_id,
                        label=self._procedure_labels.get(procedure_id, procedure_id),
                    ),
                    problem_type=problem_node,
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
        )


__all__ = [
    "InMemoryEntityKG",
    "InMemoryGraphStore",
    "InMemoryJournal",
    "InMemoryLatentStore",
    "InMemoryProceduralKG",
]
