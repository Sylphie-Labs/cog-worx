"""The claim extractor — committed journal steps to entity-KG claims, off-write-path (S1/S6).

Sweeper-shaped: reads committed steps from the journal, groups by session, calls the model for
extraction, then writes claims AND advances the cursor in ONE atomic Neo4j transaction via
:meth:`~cogworx.substrate.entity_kg.EntityKG.project_claims` (D6 — same txn, exactly-once
state even when the model is non-deterministic).

Independent journal consumer (D5): does NOT consume the episodes table.  Owns its own
(:ProjectionCursor {consumer: DEFAULT_EXTRACTION_CONSUMER}) in Neo4j, read via
:meth:`~cogworx.substrate.entity_kg.EntityKG.read_cursor`.  S8-independent of EpisodeProjector:
either sweeper can be disabled without affecting the other.

CURSOR MODEL (single-store, single-txn — D6):
  cursor read: entity_kg.read_cursor(consumer)  [Neo4j (:ProjectionCursor)]
  cursor write: entity_kg.project_claims(consumer, writes, progress=cursor)  [same Neo4j txn]
  A crash before the txn commits leaves NO claims AND NO cursor advance — at-least-once model
  cost, exactly-once KG state.  On re-tick the model is re-called; MERGE is idempotent
  (first-write-wins for the claim node, evidence accumulates).

SESSION GROUPING:
  Steps are grouped by run's session_id; one model call per session group per tick.  A run with
  no session_id maps to the empty string (session-less) — treated as its own group.

FAIL-STALL, NOT FAIL-SILENT (D6):
  On model failure, JSON parse failure, or malformed turn stamp: no claims written, cursor NOT
  advanced.  The extractor stalls at the current watermark until the next tick.

OPTIONAL LATENT (D9):
  If both ``latent_store`` and ``embedder`` are provided, each user-turn across the batch is
  embedded and put to the latent store with id ``f"episode:{episode_id}"``.  Embedder failures
  are swallowed and logged (S8 graceful degradation) — a latent miss is never a tick failure.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, datetime

from cogworx.knowledge.episodes import Turn, turns_of
from cogworx.knowledge.extraction import (
    ExtractionResult,
    mint_extraction_claims,
    render_transcript,
    validate_raw_model_json,
)
from cogworx.knowledge.source_registry import SourceRegistry
from cogworx.model.base import ChatMessage, Model
from cogworx.substrate.entity_kg import ClaimProjection, EntityKG
from cogworx.substrate.journal import Journal, ProjectedStep, ProjectionCursor
from cogworx.substrate.latent import LatentRecord, LatentStore

__all__ = ["DEFAULT_EXTRACTION_CONSUMER", "ClaimExtractor"]

_log = logging.getLogger(__name__)

DEFAULT_EXTRACTION_CONSUMER = "entity-kg/episode-extractor"
"""The default cursor consumer key.

Each projection consumer owns its own cursor row in ``cogworx_projection_cursors`` so the
journal table stays pure (S3) and any two sweepers advance from their own watermarks without
interfering.
"""

_EXTRACTION_SYSTEM = (
    "Extract factual claims from the conversation. "
    "Return ONLY valid JSON with schema: "
    '{"claims": [{"subject": "...", "predicate": "...", "object": "...",'
    ' "supporting_turn_index": 0}]}. '
    "Extract only facts stated by the user. "
    "Use short normalized strings for subject/predicate/object."
)


def _now_utc() -> datetime:
    return datetime.now(tz=UTC)


def _cursor_of(projected: ProjectedStep) -> ProjectionCursor:
    """The cursor ``(commit_ordinal, run_id, step_index)`` of a scanned step."""
    return ProjectionCursor(
        commit_ordinal=projected.commit_ordinal,
        run_id=projected.record.run_id,
        step_index=projected.record.step_index,
    )


class ClaimExtractor:
    """Polls the journal for committed steps and projects user-turn claims into the entity KG.

    Sweeper-shaped (S1): each :meth:`tick` does ONE bounded, visibility-fenced journal read,
    groups the steps by session, calls the model once per session group (flash tier), mints
    claims via the pure extraction core, then writes claims AND advances the cursor atomically
    in ONE Neo4j transaction via
    :meth:`~cogworx.substrate.entity_kg.EntityKG.project_claims` (D6).

    A failed model call or JSON parse stalls the extractor (no cursor advance, no claims
    written) — fail-stall, not fail-silent (D6).

    The optional ``latent_store`` + ``embedder`` pair puts user-turn embeddings idempotently to
    the latent space after the claim write succeeds.  Embedder errors degrade silently (S8).
    """

    def __init__(
        self,
        *,
        journal: Journal,
        entity_kg: EntityKG,
        model: Model,
        source_registry: SourceRegistry,
        latent_store: LatentStore | None = None,
        embedder: Callable[[Sequence[str]], Awaitable[Sequence[tuple[float, ...]]]] | None = None,
        consumer: str = DEFAULT_EXTRACTION_CONSUMER,
        batch_limit: int = 64,
        default_source_authority: float = 0.8,
    ) -> None:
        self._journal = journal
        self._entity_kg = entity_kg
        self._model = model
        self._source_registry = source_registry
        self._latent_store = latent_store
        self._embedder = embedder
        self._consumer = consumer
        self._batch_limit = batch_limit
        self._default_source_authority = default_source_authority
        # Resolve run -> session_id once per tick; a run's session_id is immutable for its life.
        self._session_cache: dict[str, str | None] = {}

    async def tick(self) -> int:
        """Extract one batch; return the claims written.

        Steps:
        1. Read cursor from Neo4j (entity_kg.read_cursor).
        2. Fetch one batch of committed steps from the journal.
        3. Group by session; call the model once per session group (flash tier).
        4. Mint claims via the pure extraction core.
        5. Write all claims + advance cursor atomically in ONE Neo4j txn (project_claims, D6).
        6. Optional: embed user turns and put to the latent store (S8 degradation).

        Returns the count of claims written (0 if nothing to do or nothing extracted).

        Raises:
            ValueError: on a malformed ``"turns"`` stamp or non-JSON model response — the
                cursor is NOT advanced (fail-stall, D6).
            Exception: if the model call raises — the cursor is NOT advanced (fail-stall).
        """
        self._session_cache.clear()

        cursor = await self._entity_kg.read_cursor(self._consumer)
        scanned = tuple(await self._journal.committed_steps_after(cursor, limit=self._batch_limit))
        if not scanned:
            return 0

        # Group scanned steps by session.  Session-less runs map to "".
        groups = await self._group_by_session(scanned)

        # Per-session model call + mint — any exception propagates (fail-stall, no advance).
        all_projections: list[ClaimProjection] = []
        latent_targets: list[tuple[str, str]] = []  # (episode_id, content) for user turns

        recorded_at = _now_utc()

        for session_id, session_steps in groups.items():
            projections, latent = await self._extract_session(
                session_id, session_steps, recorded_at
            )
            all_projections.extend(projections)
            latent_targets.extend(latent)

        new_cursor = _cursor_of(scanned[-1])

        # 5. Write claims AND advance cursor atomically in ONE Neo4j txn (D6 — exactly-once).
        await self._entity_kg.project_claims(self._consumer, all_projections, new_cursor)

        # 6. Optional latent store — S8: degrade silently on any error.
        if latent_targets and self._latent_store is not None and self._embedder is not None:
            await self._put_latent(latent_targets)

        return len(all_projections)

    async def _group_by_session(
        self, scanned: Sequence[ProjectedStep]
    ) -> dict[str, list[ProjectedStep]]:
        """Group scanned steps by session_id (empty string for session-less runs)."""
        groups: dict[str, list[ProjectedStep]] = {}
        for projected in scanned:
            session_id = await self._session_for(projected.record.run_id) or ""
            groups.setdefault(session_id, []).append(projected)
        return groups

    async def _extract_session(
        self,
        session_id: str,
        session_steps: list[ProjectedStep],
        recorded_at: datetime,
    ) -> tuple[list[ClaimProjection], list[tuple[str, str]]]:
        """Extract claims for one session group; return (projections, latent_targets).

        ``latent_targets`` is a list of ``(episode_id, content)`` for user-role turns across
        the session — populated only when a latent store + embedder are present.

        Raises ValueError on malformed turn stamps (propagates from turns_of).
        Raises on model errors (propagates — fail-stall).
        """
        # Collect all turns across all steps in this session group, with their episode_ids.
        all_turns: list[Turn] = []
        episode_ids_by_turn: list[str] = []

        for projected in session_steps:
            step = projected.record
            turns = turns_of(step)  # raises ValueError on malformed stamp (fail-loud)
            for i, turn in enumerate(turns):
                episode_id = f"{step.run_id}:{step.step_index}:{i}"
                all_turns.append(turn)
                episode_ids_by_turn.append(episode_id)

        if not all_turns:
            return [], []

        # Declare the source for this session (framework code, never model output — S9).
        source_decl = self._source_registry.declare(
            "human",
            session_id or "__session_less__",
            authority=self._default_source_authority,
        )

        # Build transcript and call model (flash tier — S11 cost discipline).
        transcript = render_transcript(all_turns, session_id=session_id)
        system_msg = ChatMessage(role="system", content=_EXTRACTION_SYSTEM)
        user_msg = ChatMessage(role="user", content=transcript)

        response = await self._model.complete(messages=[system_msg, user_msg], tier="flash")

        raw_text = response.text or ""
        try:
            raw_json = json.loads(raw_text)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"claim_extractor: model returned non-JSON for session {session_id!r} "
                f"(response length={len(raw_text)}); stalling (fail-stall, D6)"
            ) from exc

        raw_items = validate_raw_model_json(raw_json)

        result: ExtractionResult = mint_extraction_claims(
            raw_items,
            turns=all_turns,
            source_decl=source_decl,
            episode_ids_by_turn=episode_ids_by_turn,
            recorded_at=recorded_at,
        )

        if result.rejected_count:
            _log.debug(
                "claim_extractor: session %r: %d items rejected during minting",
                session_id,
                result.rejected_count,
            )

        projections = [
            ClaimProjection(claim=claim, evidence=evidence) for claim, evidence in result.pairs
        ]

        # Collect user-turn latent targets (D9 — only when latent store is wired).
        latent_targets: list[tuple[str, str]] = []
        if self._latent_store is not None and self._embedder is not None:
            for turn, episode_id in zip(all_turns, episode_ids_by_turn, strict=True):
                if turn.role == "user":
                    latent_targets.append((episode_id, turn.content))

        return projections, latent_targets

    async def _session_for(self, run_id: str) -> str | None:
        if run_id not in self._session_cache:
            run = await self._journal.load_run(run_id)
            self._session_cache[run_id] = run.session_id if run is not None else None
        return self._session_cache[run_id]

    async def _put_latent(self, targets: list[tuple[str, str]]) -> None:
        """Embed user-turn contents and put to the latent store; swallow all errors (S8)."""
        assert self._latent_store is not None
        assert self._embedder is not None
        try:
            contents = [content for _, content in targets]
            embeddings = await self._embedder(contents)
            for (episode_id, _), embedding in zip(targets, embeddings, strict=False):
                record = LatentRecord(
                    id=f"episode:{episode_id}",
                    embedding=tuple(embedding),
                )
                await self._latent_store.put(record)
        except Exception:
            _log.warning(
                "claim_extractor: latent store put failed (S8 degradation — claims were written)",
                exc_info=True,
            )
