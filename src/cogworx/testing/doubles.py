"""In-memory substrate doubles for the Test Kit (CANON S3, S6).

Reference implementations of the distinct substrate seams (journal / graph / latent) — kept separate
so the doubles exercise the same engine-shaped contracts the real adapters do (S3, no flattening
``Store``). ``InMemoryJournal`` is the S6 reference behaviour: exactly-once on the idempotency key,
status derived from the committed steps.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime

from cogworx.claims.provenance import Claim
from cogworx.loop.state import RunStatus
from cogworx.substrate.journal import RunState, StepRecord, Timer
from cogworx.substrate.latent import LatentMatch, LatentRecord


class _RunLog:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.steps: list[StepRecord] = []
        self.seen_keys: set[str] = set()


class InMemoryJournal:
    """An in-memory ``Journal`` with exactly-once commits and status derived from steps."""

    def __init__(self) -> None:
        self._runs: dict[str, _RunLog] = {}
        self._timers: list[Timer] = []

    async def start_run(self, run_id: str, session_id: str) -> None:
        if run_id not in self._runs:
            self._runs[run_id] = _RunLog(session_id)

    async def commit_step(self, record: StepRecord) -> None:
        log = self._runs.setdefault(record.run_id, _RunLog(record.run_id))
        if record.idempotency_key in log.seen_keys:
            return
        log.seen_keys.add(record.idempotency_key)
        log.steps.append(record)

    async def read_step(self, run_id: str, step_id: str) -> StepRecord | None:
        log = self._runs.get(run_id)
        if log is None:
            return None
        for record in log.steps:
            if record.step_id == step_id:
                return record
        return None

    async def load_run(self, run_id: str) -> RunState | None:
        log = self._runs.get(run_id)
        if log is None:
            return None
        steps = tuple(log.steps)
        status = self._derive_status(steps)
        current_stage = steps[-1].stage_name if steps else None
        return RunState(
            run_id=run_id,
            session_id=log.session_id,
            status=status,
            current_stage=current_stage,
            steps=steps,
        )

    @staticmethod
    def _derive_status(steps: Sequence[StepRecord]) -> RunStatus:
        if not steps:
            return RunStatus.PENDING
        result = steps[-1].result
        match result.kind:
            case "done":
                return RunStatus.COMPLETED
            case "await-human":
                return RunStatus.AWAITING_HUMAN
            case "degraded":
                return RunStatus.DEGRADED if result.to is None else RunStatus.RUNNING
            case "transition":
                return RunStatus.RUNNING

    async def set_timer(self, timer: Timer) -> None:
        self._timers.append(timer)

    async def due_timers(self, now: datetime) -> Sequence[Timer]:
        return tuple(timer for timer in self._timers if timer.wake_at <= now)


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
    """An in-memory ``LatentStore`` with cosine-similarity nearest-neighbour search."""

    def __init__(self) -> None:
        self._records: dict[str, LatentRecord] = {}

    async def upsert(self, record: LatentRecord) -> None:
        self._records[record.id] = record

    async def search(self, embedding: Sequence[float], *, k: int = 10) -> Sequence[LatentMatch]:
        query = tuple(embedding)
        query_norm = math.sqrt(sum(value * value for value in query))
        if query_norm == 0.0 or not self._records:
            return ()
        matches: list[LatentMatch] = []
        for record in self._records.values():
            score = self._cosine(query, query_norm, record.embedding)
            if score is None:
                continue
            matches.append(LatentMatch(record=record, score=score))
        matches.sort(key=lambda match: match.score, reverse=True)
        return tuple(matches[:k])

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


__all__ = [
    "InMemoryGraphStore",
    "InMemoryJournal",
    "InMemoryLatentStore",
]
