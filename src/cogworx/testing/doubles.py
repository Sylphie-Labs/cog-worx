"""In-memory substrate doubles for the Test Kit (CANON S3, S6).

Reference implementations of the distinct substrate seams (journal / graph / latent) — kept separate
so the doubles exercise the same engine-shaped contracts the real adapters do (S3, no flattening
``Store``). ``InMemoryJournal`` is the S6 reference behaviour: exactly-once on ``(run_id,
step_index)``, with the run's status PERSISTED on the run record (the authority — not derived from
the last step) and the run's pathway pointer stored for cold resume.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from datetime import datetime, timedelta

from cogworx.claims.provenance import Artifact, Claim
from cogworx.loop.state import RunStatus
from cogworx.substrate.journal import RunState, StepRecord, Timer
from cogworx.substrate.latent import LatentMatch, LatentRecord


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

    async def read_step(self, run_id: str, step_index: int) -> StepRecord | None:
        log = self._runs.get(run_id)
        if log is None:
            return None
        return log.steps.get(step_index)

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

    async def record_human_input(
        self, run_id: str, step_index: int, answer: Artifact
    ) -> None:
        # FIRST-ANSWER-WINS: setdefault is atomic under asyncio (no await between read and write) —
        # mirrors the adapter's ON CONFLICT (run_id, step_index) DO NOTHING.
        self._human_inputs.setdefault((run_id, step_index), answer)

    async def read_human_input(
        self, run_id: str, step_index: int
    ) -> Artifact | None:
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
