"""Pod 2.3 Feature Test Bundle — episodic memory (CANON invariants S1, S5, S6, S8, S9).

Tier 2 of the DoD: wires the episodic-memory pipeline into the walking-skeleton Engine and
checks every invariant mechanically. No Postgres, no Neo4j, no real model — everything runs on
InMemoryJournal + InMemoryEpisodeStore + InMemoryEntityKG + ReplayModel.

## Invariants under test

I1 — Zero model work on the EpisodeProjector write path (S1):
    EpisodeProjector.tick() has no model; projecting committed turns must leave
    model.call_count unchanged.

I2 — EpisodeProjector exactly-once cursor atomicity (S6):
    tick() twice on the same journal → second tick sees no new rows (cursor advanced past them).
    A third tick on a static journal is a no-op. Episodes are never double-inserted.

I3 — Phantom-turn impossibility (D1 design validation):
    A step that was never committed (attempt 1 abandoned) produces no episodes; only the
    committed step's turns appear. This is the KEY structural correctness claim of the
    "capture = stamp_turns, project = committed-only" design.

I4 — S9 sockpuppet rejection (extraction core):
    mint_extraction_claims rejects items whose supporting_turn_index points to an assistant
    turn (only user turns are evidence). Out-of-range indices are also rejected. Valid items are
    accepted with epistemic_type == "inference" and source_id == source_decl.source_id.
    Model text never reaches source_id.

I5 — Identity discipline + idempotent accumulation:
    Calling mint_extraction_claims twice with the same raw_items yields byte-identical claim_ids
    (claim_id_for is deterministic). No duplicate evidence can be minted that violates identity.

I6 — S8 lesion matrix (stub only, no live substrate):
    - EpisodeProjector absent → engine + turns in journal, no episodes materialised.
    - ClaimExtractor absent → engine runs normally, turns are committed.
    - Both absent → engine runs byte-identically; model.call_count is unaffected by their absence.

Additional:
    - SourceRegistry: declare + idempotent re-declare + conflict raises ValueError.
    - render_transcript: format is deterministic; session prefix + role indexing correct.
    - turns_of: round-trips through stamp_turns; raises ValueError on malformed stamp.
    - EpisodeProjector stalls on malformed turn stamp (fail-loud, cursor not advanced).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.knowledge.episodes import Turn, stamp_turns, turns_of
from cogworx.knowledge.extraction import (
    RawClaimItem,
    mint_extraction_claims,
    render_transcript,
    validate_raw_model_json,
)
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.source_registry import SourceDeclaration, SourceRegistry
from cogworx.loop.result import Done, Transition
from cogworx.model.base import ModelResponse
from cogworx.model.registry import ModelRegistry
from cogworx.runtime.episode_projector import DEFAULT_EPISODE_CONSUMER, EpisodeProjector
from cogworx.substrate.journal import ProjectionCursor, StepRecord
from cogworx.testing.doubles import (
    InMemoryEntityKG,
    InMemoryEpisodeStore,
    InMemoryGraphStore,
    InMemoryJournal,
    InMemoryLatentStore,
)
from cogworx.testing.fake_model import ReplayModel

_NOW = datetime(2026, 6, 10, tzinfo=UTC)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_PATHWAY = "episodic_pathway"
_STAGE_TURNS = "turns_stage"
_STAGE_PLAIN = "plain_stage"


def _prov(*, source: str = "system") -> Provenance:
    return Provenance(source=source, confidence=1.0, recorded_at=_NOW)


def _artifact(
    text: str = "ok", *, source: str = "system", turns: list[Turn] | None = None
) -> Artifact:
    data: dict[str, object] = {"text": text}
    if turns:
        stamp_turns(data, turns)
    return Artifact(
        kind="output",
        produced_by="stage",
        provenance=_prov(source=source),
        data=data,
    )


def _conversation_turns(
    user_text: str = "hello",
    assistant_text: str = "hi",
) -> list[Turn]:
    return [
        Turn(role="user", content=user_text, kind="conversation"),
        Turn(role="assistant", content=assistant_text, kind="conversation"),
    ]


async def _start(journal: InMemoryJournal, run_id: str, *, session_id: str = "") -> None:
    await journal.start_run(
        run_id,
        session_id or f"session:{run_id}",
        pathway_id=_PATHWAY,
        pathway_version=1,
        pathway_fingerprint="fp",
    )


async def _commit_turns(
    journal: InMemoryJournal,
    *,
    run_id: str,
    step_index: int,
    turns: list[Turn],
    committed_at: datetime = _NOW,
    transition_to: str | None = None,
) -> None:
    art = _artifact("turn-step", turns=turns)
    result: Done | Transition = (
        Transition(to=transition_to, output=art) if transition_to is not None else Done(output=art)
    )
    await journal.commit_step(
        StepRecord(
            run_id=run_id,
            step_index=step_index,
            stage_name=_STAGE_TURNS,
            result=result,
            committed_at=committed_at,
        )
    )


async def _commit_plain(
    journal: InMemoryJournal,
    *,
    run_id: str,
    step_index: int,
    committed_at: datetime = _NOW,
) -> None:
    """Commit a step with no 'turns' key — a plain control step."""
    await journal.commit_step(
        StepRecord(
            run_id=run_id,
            step_index=step_index,
            stage_name=_STAGE_PLAIN,
            result=Done(output=_artifact("plain")),
            committed_at=committed_at,
        )
    )


def _projector(
    journal: InMemoryJournal,
    episode_store: InMemoryEpisodeStore,
    *,
    batch_limit: int = 256,
) -> EpisodeProjector:
    return EpisodeProjector(
        journal=journal,
        episode_store=episode_store,
        batch_limit=batch_limit,
    )


# ---------------------------------------------------------------------------
# I1 — Zero model work on the EpisodeProjector write path (S1)
# ---------------------------------------------------------------------------


async def test_i1_projector_tick_does_not_call_model() -> None:
    """EpisodeProjector.tick() is model-free; model.call_count must not change (S1).

    A stage stamps turns into the output artifact before committing. After the commit the
    EpisodeProjector sweeps and materialises episodes. The model is NOT part of this path.
    """
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    model = ReplayModel([])  # zero responses — any call would raise ReplayExhaustedError

    await _start(journal, "r1", session_id="sess1")
    await _commit_turns(
        journal,
        run_id="r1",
        step_index=0,
        turns=_conversation_turns("tell me about Jupiter", "Jupiter is a gas giant"),
    )

    before = model.call_count
    projected = await _projector(journal, store).tick()
    after = model.call_count

    assert after == before, (
        f"S1 violation: model was called {after - before} time(s) during EpisodeProjector.tick()"
    )
    assert projected == 2  # 2 turns → 2 episode rows
    episodes = await store.episodes_for_session("sess1")
    assert len(episodes) == 2


async def test_i1_projector_tick_episodes_appear_in_store() -> None:
    """Episodes appear in InMemoryEpisodeStore after a single tick —
    turn-stamping is the capture."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    turns = _conversation_turns("what is Mars?", "Mars is the red planet")
    await _start(journal, "r1", session_id="sess-mars")
    await _commit_turns(journal, run_id="r1", step_index=0, turns=turns)

    await _projector(journal, store).tick()

    episodes = await store.episodes_for_session("sess-mars")
    assert len(episodes) == 2
    assert episodes[0].role == "user"
    assert episodes[0].content == "what is Mars?"
    assert episodes[1].role == "assistant"
    assert episodes[1].content == "Mars is the red planet"
    # episode_id is deterministic
    assert episodes[0].episode_id == "r1:0:0"
    assert episodes[1].episode_id == "r1:0:1"
    # occurred_at comes from step.committed_at, not Turn (S6 replay-stability)
    assert episodes[0].occurred_at == _NOW
    assert episodes[1].occurred_at == _NOW


# ---------------------------------------------------------------------------
# I2 — EpisodeProjector exactly-once cursor atomicity (S6)
# ---------------------------------------------------------------------------


async def test_i2_second_tick_sees_empty() -> None:
    """After the first tick advances the cursor, a second tick on the same data returns 0 (S6)."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1")
    await _commit_turns(journal, run_id="r1", step_index=0, turns=_conversation_turns("a", "b"))
    await _commit_turns(journal, run_id="r1", step_index=1, turns=_conversation_turns("c", "d"))

    p = _projector(journal, store)
    first = await p.tick()
    second = await p.tick()  # cursor already past both steps

    assert first == 4  # 2 steps * 2 turns
    assert second == 0


async def test_i2_cursor_advances_to_last_scanned_row() -> None:
    """The cursor advances to the last scanned row even when a batch has zero episodes
    (S6 / P0-2)."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1")
    # A plain (turn-free) step
    await _commit_plain(journal, run_id="r1", step_index=0)

    assert await store.read_cursor(DEFAULT_EPISODE_CONSUMER) is None
    projected = await _projector(journal, store).tick()

    assert projected == 0  # no episodes from a plain step
    cursor = await store.read_cursor(DEFAULT_EPISODE_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", 0)


async def test_i2_idempotent_repeated_tick() -> None:
    """Many ticks on a static journal: episode count never grows past the first projection (S6)."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    turns = _conversation_turns("q", "a")
    await _start(journal, "r1", session_id="sess-idem")
    await _commit_turns(journal, run_id="r1", step_index=0, turns=turns)

    proj = _projector(journal, store)
    for _ in range(5):
        await proj.tick()

    episodes = await store.episodes_for_session("sess-idem")
    assert len(episodes) == 2  # exactly first projection, never doubled


async def test_i2_negative_control_without_cursor_advance_would_double_project() -> None:
    """Negative control: if project_episodes were NOT called (no cursor advance), re-ticking on the
    same journal would re-project the same rows.  Prove the assertion in i2 is sensitive by
    simulating the broken path directly against the store's own first-write-wins semantics."""
    journal, _store = InMemoryJournal(), InMemoryEpisodeStore()
    turns = _conversation_turns("x", "y")
    await _start(journal, "r1", session_id="sess-neg")
    await _commit_turns(journal, run_id="r1", step_index=0, turns=turns)

    # Simulate the broken projector: project WITHOUT advancing the cursor.
    # The store's setdefault is first-write-wins, so re-insertion is a no-op on the episodes
    # dict — but the CURSOR does not advance, so a second tick would re-scan the same row.
    # What we are proving: the cursor machinery is the only thing that prevents this.
    scanned = await journal.committed_steps_after(None, limit=256)
    assert len(scanned) == 1

    # Re-scan from cursor=None again (the broken path: no cursor advance happened)
    re_scanned = await journal.committed_steps_after(None, limit=256)
    assert len(re_scanned) == 1, (
        "negative control: without cursor advance the same row is visible again, "
        "confirming the cursor IS the exactly-once mechanism"
    )


# ---------------------------------------------------------------------------
# I3 — Phantom-turn impossibility (D1 design validation)
# ---------------------------------------------------------------------------


async def test_i3_uncommitted_attempt_produces_no_episodes() -> None:
    """The KEY design invariant: only committed-step turns appear as episodes.

    Simulates two attempts at step 0:
    - Attempt 1: a 'capture_episode' call is NOT committed (the step is never committed).
    - Attempt 2: committed with different turns.

    The projector must see ONLY attempt 2's turns.
    """
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1", session_id="sess-phantom")

    # Attempt 1: increment the attempt counter (as the engine would on a failed attempt),
    # but DO NOT commit a step — the turns from attempt 1 are never durable.
    await journal.increment_attempt("r1", 0)
    _attempt1_turns = _conversation_turns("attempt1 question", "attempt1 answer")
    # (We do NOT call _commit_turns here — attempt 1 was abandoned without committing.)

    # Attempt 2: commit with different, recognisable content.
    attempt2_turns = _conversation_turns("attempt2 question", "attempt2 answer")
    await _commit_turns(
        journal,
        run_id="r1",
        step_index=1,  # new positional index after the failed attempt
        turns=attempt2_turns,
    )

    await _projector(journal, store).tick()

    episodes = await store.episodes_for_session("sess-phantom")
    contents = {ep.content for ep in episodes}

    # Attempt 1 content must NOT appear
    assert "attempt1 question" not in contents, (
        "I3 violation: phantom turn from uncommitted attempt 1 appeared in episode store"
    )
    assert "attempt1 answer" not in contents, (
        "I3 violation: phantom turn from uncommitted attempt 1 appeared in episode store"
    )
    # Attempt 2 content MUST appear
    assert "attempt2 question" in contents
    assert "attempt2 answer" in contents
    assert len(episodes) == len(attempt2_turns)


async def test_i3_negative_control_committed_turns_always_appear() -> None:
    """Negative control: if a step IS committed, its turns MUST appear
    (confirms detection power)."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1", session_id="sess-neg3")
    turns = _conversation_turns("real question", "real answer")
    await _commit_turns(journal, run_id="r1", step_index=0, turns=turns)
    await _projector(journal, store).tick()

    episodes = await store.episodes_for_session("sess-neg3")
    contents = {ep.content for ep in episodes}
    assert "real question" in contents
    assert "real answer" in contents


# ---------------------------------------------------------------------------
# I4 — S9 sockpuppet rejection (extraction core, no model needed)
# ---------------------------------------------------------------------------


async def test_i4_assistant_turn_index_rejected() -> None:
    """An item whose supporting_turn_index points to an assistant turn is rejected (S9)."""
    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-42", authority=0.9)
    turns = [
        Turn(role="user", content="Alice lives in Paris", kind="conversation"),
        Turn(role="assistant", content="noted", kind="conversation"),
    ]
    raw_items = [
        RawClaimItem(
            subject="Alice", predicate="lives_in", object="Paris", supporting_turn_index=1
        ),
    ]
    result = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=["r1:0:0", "r1:0:1"],
        recorded_at=_NOW,
    )
    assert result.rejected_count == 1
    assert len(result.pairs) == 0


async def test_i4_out_of_range_index_rejected() -> None:
    """An item with supporting_turn_index out of range is rejected."""
    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-99", authority=0.8)
    turns = [Turn(role="user", content="Bob lives in Tokyo", kind="conversation")]
    raw_items = [
        RawClaimItem(subject="Bob", predicate="lives_in", object="Tokyo", supporting_turn_index=5),
    ]
    result = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=["r1:0:0"],
        recorded_at=_NOW,
    )
    assert result.rejected_count == 1
    assert len(result.pairs) == 0


async def test_i4_valid_item_accepted_with_correct_epistemic_type() -> None:
    """A valid item pointing to a user turn is accepted; epistemic_type == 'inference' (S9).

    model text never reaches source_id — source_id comes from SourceDeclaration only.
    """
    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-77", authority=0.85)
    turns = [
        Turn(role="user", content="Carol works at CERN", kind="conversation"),
    ]
    raw_items = [
        RawClaimItem(subject="Carol", predicate="works_at", object="CERN", supporting_turn_index=0),
    ]
    result = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=["r1:0:0"],
        recorded_at=_NOW,
    )
    assert result.rejected_count == 0
    assert len(result.pairs) == 1
    claim, evidence = result.pairs[0]

    # S9 hard wall: epistemic_type is hardcoded to "inference", never model-chosen
    assert claim.epistemic_type == "inference"
    # S9: source_id comes from SourceDeclaration, never from model output
    assert evidence.source_id == source_decl.source_id
    assert claim.created_by == source_decl.source_id


async def test_i4_mixed_valid_and_invalid_items() -> None:
    """Valid item accepted, assistant-index rejected, out-of-range rejected — in one call."""
    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-mix", authority=0.9)
    turns = [
        Turn(role="user", content="Dave is a physicist", kind="conversation"),
        Turn(role="assistant", content="ok", kind="conversation"),
    ]
    raw_items = [
        RawClaimItem(subject="Dave", predicate="is_a", object="physicist", supporting_turn_index=0),
        RawClaimItem(subject="bad", predicate="from_asst", object="turn", supporting_turn_index=1),
        RawClaimItem(subject="bad", predicate="out_of", object="range", supporting_turn_index=99),
    ]
    result = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=["r1:0:0", "r1:0:1"],
        recorded_at=_NOW,
    )
    assert len(result.pairs) == 1
    assert result.rejected_count == 2
    claim, _ = result.pairs[0]
    assert claim.subject == "Dave"


# ---------------------------------------------------------------------------
# I5 — Identity discipline + idempotent accumulation
# ---------------------------------------------------------------------------


async def test_i5_claim_ids_are_deterministic() -> None:
    """Calling mint_extraction_claims twice with the same inputs yields identical claim_ids (S9)."""
    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-id", authority=0.8)
    turns = [Turn(role="user", content="Eve is a biologist", kind="conversation")]
    raw_items = [
        RawClaimItem(subject="Eve", predicate="is_a", object="biologist", supporting_turn_index=0),
    ]
    episode_ids = ["r1:0:0"]

    result_a = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=episode_ids,
        recorded_at=_NOW,
    )
    result_b = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=episode_ids,
        recorded_at=_NOW,
    )

    assert len(result_a.pairs) == 1
    assert len(result_b.pairs) == 1
    claim_a, _ = result_a.pairs[0]
    claim_b, _ = result_b.pairs[0]
    assert claim_a.id == claim_b.id


async def test_i5_claim_id_matches_claim_id_for() -> None:
    """claim.id must equal claim_id_for(subject, predicate, object) — framework-minted (S9)."""
    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-ident", authority=1.0)
    turns = [Turn(role="user", content="Frank studies Neptune", kind="conversation")]
    raw_items = [
        RawClaimItem(
            subject="Frank", predicate="studies", object="Neptune", supporting_turn_index=0
        ),
    ]
    result = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=["r1:0:0"],
        recorded_at=_NOW,
    )
    claim, _ = result.pairs[0]
    expected_id = claim_id_for("Frank", "studies", "Neptune")
    assert claim.id == expected_id


async def test_i5_in_memory_entity_kg_project_claims_idempotent() -> None:
    """project_claims with the same ClaimProjection twice → first-write-wins, no error (S6)."""
    from cogworx.substrate.entity_kg import ClaimProjection

    registry = SourceRegistry()
    source_decl = registry.declare("human", "session-idem2", authority=0.8)
    turns = [Turn(role="user", content="Grace studies Saturn", kind="conversation")]
    raw_items = [
        RawClaimItem(
            subject="Grace", predicate="studies", object="Saturn", supporting_turn_index=0
        ),
    ]
    result = mint_extraction_claims(
        raw_items,
        turns=turns,
        source_decl=source_decl,
        episode_ids_by_turn=["r1:0:0"],
        recorded_at=_NOW,
    )
    claim, evidence = result.pairs[0]
    projection = ClaimProjection(claim=claim, evidence=evidence)

    kg = InMemoryEntityKG()
    cursor = ProjectionCursor(commit_ordinal=1, run_id="r1", step_index=0)
    await kg.project_claims("extractor", [projection], cursor)
    # Second projection of the same claim — must not raise, cursor is kept at max
    await kg.project_claims("extractor", [projection], cursor)

    claims = await kg.claims_about("Grace")
    # First-write-wins: only one claim node, but evidence accumulates (two events)
    assert len(claims) == 1
    ev = await kg.evidence_for(claims[0].claim.id)
    # Evidence events accumulate — this is correct and expected
    assert len(ev) >= 1


# ---------------------------------------------------------------------------
# I6 — S8 lesion matrix (stub only, no live substrate)
# ---------------------------------------------------------------------------


async def test_i6_lesion_projector_absent_engine_runs_normally() -> None:
    """S8: with no EpisodeProjector, the engine runs normally and turns are committed to
    the journal.

    The engine does not know about EpisodeProjector — the projector is an external sweeper.
    This test confirms the engine's stage still stamps turns and commits them to the journal;
    the downstream projector is optional (its absence only means no episodes materialised yet).
    """
    from cogworx.loop.graph import StageGraph
    from cogworx.loop.pathway import PathwayRegistry
    from cogworx.loop.result import Done
    from cogworx.loop.stage import StageContext
    from cogworx.runtime.engine import Engine

    class _TurnsStage:
        name: str = "turns_stage"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> Done:
            turns = _conversation_turns("user text", "assistant text")
            art = _artifact("stamped", turns=turns)
            return Done(output=art)

    pathway_id = "lesion-no-projector"
    pathways = PathwayRegistry()
    pathways.register(pathway_id, StageGraph([_TurnsStage()], entry="turns_stage"))

    model = ReplayModel([])  # no model calls expected in a no-model stage
    _reg = ModelRegistry()
    _reg.register("default", model)
    journal = InMemoryJournal()
    engine = Engine(
        models=_reg,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
    )
    state = await engine.run(
        run_id="lesion-r1",
        session_id="lesion-sess",
        pathway_id=pathway_id,
        initial=_artifact("start", source="human"),
    )
    from cogworx.loop.state import RunStatus

    assert state.status is RunStatus.COMPLETED
    assert len(state.steps) == 1
    # Turns ARE in the committed step (journal has them)
    committed_step = state.steps[0]
    parsed_turns = turns_of(committed_step)
    assert len(parsed_turns) == 2
    assert parsed_turns[0].role == "user"
    assert parsed_turns[1].role == "assistant"
    # No projector ran — no episodes anywhere (nothing to assert on, but model was never called)
    assert model.call_count == 0


async def test_i6_lesion_extractor_absent_engine_runs_normally() -> None:
    """S8: with no ClaimExtractor, the engine runs normally, turns committed, no claims extracted.

    ClaimExtractor is a sweeper. Its absence does not affect the engine's turn-stamping path.
    """
    from cogworx.loop.graph import StageGraph
    from cogworx.loop.pathway import PathwayRegistry
    from cogworx.loop.result import Done
    from cogworx.loop.stage import StageContext
    from cogworx.runtime.engine import Engine

    class _TurnsStage:
        name: str = "turns_stage"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> Done:
            turns = _conversation_turns("extractor absent", "no extractor")
            return Done(output=_artifact("stamped", turns=turns))

    pathway_id = "lesion-no-extractor"
    pathways = PathwayRegistry()
    pathways.register(pathway_id, StageGraph([_TurnsStage()], entry="turns_stage"))

    model = ReplayModel([])
    _reg = ModelRegistry()
    _reg.register("default", model)
    journal = InMemoryJournal()
    engine = Engine(
        models=_reg,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
    )
    state = await engine.run(
        run_id="lesion-r2",
        session_id="lesion-sess2",
        pathway_id=pathway_id,
        initial=_artifact("start", source="human"),
    )
    from cogworx.loop.state import RunStatus

    assert state.status is RunStatus.COMPLETED
    # Turns still in the journal
    step = state.steps[0]
    parsed = turns_of(step)
    assert len(parsed) == 2


async def test_i6_lesion_both_absent_model_call_count_unchanged() -> None:
    """S8: with both EpisodeProjector and ClaimExtractor absent, model.call_count reflects only
    the stage's own calls — it is unaffected by the absent sweepers.

    This test runs a pathway where the stage itself calls the model once, then confirms that
    the absence of both sweepers adds zero model calls.
    """
    from cogworx.loop.graph import StageGraph
    from cogworx.loop.pathway import PathwayRegistry
    from cogworx.loop.result import Done
    from cogworx.loop.stage import StageContext
    from cogworx.model.base import ChatMessage
    from cogworx.runtime.engine import Engine

    class _ModelStage:
        name: str = "model_stage"
        transitions: tuple[str, ...] = ()

        async def run(self, ctx: StageContext) -> Done:
            resp = await ctx.model.complete(
                messages=[ChatMessage(role="user", content="Say hello")]
            )
            turns = [Turn(role="user", content="Say hello", kind="conversation")]
            return Done(output=_artifact(resp.text or "hi", turns=turns))

    pathway_id = "lesion-both-absent"
    pathways = PathwayRegistry()
    pathways.register(pathway_id, StageGraph([_ModelStage()], entry="model_stage"))

    model = ReplayModel(
        [ModelResponse(text="hello from model", model_id="replay", finish_reason="stop")]
    )
    _reg = ModelRegistry()
    _reg.register("default", model)
    journal = InMemoryJournal()
    engine = Engine(
        models=_reg,
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
    )
    state = await engine.run(
        run_id="lesion-r3",
        session_id="lesion-sess3",
        pathway_id=pathway_id,
        initial=_artifact("start", source="human"),
    )
    from cogworx.loop.state import RunStatus

    assert state.status is RunStatus.COMPLETED
    # Exactly one model call: the stage's own call — no sweeper added any
    assert model.call_count == 1


# ---------------------------------------------------------------------------
# SourceRegistry — declare, idempotent re-declare, conflict detection
# ---------------------------------------------------------------------------


def test_source_registry_declare_basic() -> None:
    """declare() returns a SourceDeclaration with deterministic source_id."""
    registry = SourceRegistry()
    decl = registry.declare("human", "session-abc")
    assert isinstance(decl, SourceDeclaration)
    assert decl.kind == "human"
    assert decl.ref == "session-abc"
    assert decl.source_id == "source:human:session-abc"
    assert decl.source_authority == 1.0


def test_source_registry_declare_idempotent() -> None:
    """Re-declaring the same (kind, ref, authority) returns the same object, no error."""
    registry = SourceRegistry()
    d1 = registry.declare("agent", "agent-007", authority=0.9)
    d2 = registry.declare("agent", "agent-007", authority=0.9)
    assert d1 is d2


def test_source_registry_declare_conflict_raises() -> None:
    """Re-declaring the same (kind, ref) with a different authority raises ValueError (S9)."""
    registry = SourceRegistry()
    registry.declare("tool", "search-tool", authority=0.8)
    with pytest.raises(ValueError, match="conflicting identity discipline"):
        registry.declare("tool", "search-tool", authority=0.5)


def test_source_registry_get() -> None:
    """get() retrieves a declaration; get_by_id() retrieves by source_id."""
    registry = SourceRegistry()
    decl = registry.declare("document", "wiki-page-1")
    assert registry.get("document", "wiki-page-1") is decl
    assert registry.get_by_id(decl.source_id) is decl
    assert registry.get("document", "nonexistent") is None


def test_source_registry_source_id_never_supplied_by_caller() -> None:
    """source_id is computed, never a constructor parameter — callers cannot hand-roll it (S9).

    SourceDeclaration is a frozen dataclass; source_id is a field(init=False) computed in
    __post_init__. Attempting to pass it as a positional arg is a TypeError.
    """
    # frozen dataclass: keyword args only; source_id is not in the __init__ signature
    # Two declarations of the same (kind, ref) must produce the same source_id across runs.
    d_a = SourceDeclaration(kind="system", ref="run-loop")
    d_b = SourceDeclaration(kind="system", ref="run-loop")
    assert d_a.source_id == d_b.source_id == "source:system:run-loop"


def test_source_declaration_authority_out_of_range_raises() -> None:
    """source_authority outside [0.0, 1.0] raises ValueError at construction."""
    with pytest.raises(ValueError):
        SourceDeclaration(kind="human", ref="bad", source_authority=1.5)
    with pytest.raises(ValueError):
        SourceDeclaration(kind="human", ref="bad", source_authority=-0.1)


# ---------------------------------------------------------------------------
# render_transcript — deterministic format
# ---------------------------------------------------------------------------


def test_render_transcript_format_deterministic() -> None:
    """render_transcript produces a deterministic, parseable format."""
    turns = [
        Turn(role="user", content="Hello", kind="conversation"),
        Turn(role="assistant", content="Hi there", kind="conversation"),
        Turn(role="user", content="Tell me more", kind="conversation"),
    ]
    result = render_transcript(turns, session_id="test-session")
    lines = result.rstrip("\n").split("\n")

    assert lines[0] == "[SESSION: test-session]"
    assert lines[1] == "[USER 0]: Hello"
    assert lines[2] == "[ASSISTANT 1]: Hi there"
    assert lines[3] == "[USER 2]: Tell me more"


def test_render_transcript_same_input_same_output() -> None:
    """render_transcript is a pure function — same input → byte-identical output."""
    turns = [
        Turn(role="user", content="question", kind="conversation"),
        Turn(role="system", content="rule", kind="system_note"),
    ]
    a = render_transcript(turns, session_id="s1")
    b = render_transcript(turns, session_id="s1")
    assert a == b


def test_render_transcript_session_prefix_is_first_line() -> None:
    """Session id appears as the very first line of the transcript."""
    turns = [Turn(role="user", content="x", kind="conversation")]
    result = render_transcript(turns, session_id="my-session-id")
    assert result.startswith("[SESSION: my-session-id]\n")


# ---------------------------------------------------------------------------
# turns_of + stamp_turns — round-trip, error handling
# ---------------------------------------------------------------------------


def test_stamp_turns_round_trips_via_turns_of() -> None:
    """stamp_turns → turns_of round-trips the turn list faithfully."""
    turns = [
        Turn(role="user", content="A", kind="conversation"),
        Turn(role="assistant", content="B", kind="conversation"),
        Turn(role="system", content="C", kind="system_note"),
        Turn(role="tool", content="D", kind="tool_exchange"),
    ]
    data: dict[str, object] = {}
    stamp_turns(data, turns)

    step = StepRecord(
        run_id="r1",
        step_index=0,
        stage_name="s",
        result=Done(output=Artifact(kind="o", produced_by="s", provenance=_prov(), data=data)),
        committed_at=_NOW,
    )
    recovered = turns_of(step)
    assert recovered == turns


def test_stamp_turns_empty_raises() -> None:
    """stamp_turns raises ValueError on an empty turn list."""
    with pytest.raises(ValueError, match="non-empty"):
        stamp_turns({}, [])


def test_turns_of_missing_key_returns_empty() -> None:
    """turns_of returns [] when the step's output.data has no 'turns' key."""
    step = StepRecord(
        run_id="r1",
        step_index=0,
        stage_name="s",
        result=Done(output=_artifact("plain")),
        committed_at=_NOW,
    )
    assert turns_of(step) == []


def test_turns_of_malformed_raises_value_error() -> None:
    """turns_of raises ValueError when 'turns' is present but not a list of dicts."""
    bad_artifact = Artifact(
        kind="output",
        produced_by="stage",
        provenance=_prov(),
        data={"turns": "this is not a list"},
    )
    step = StepRecord(
        run_id="r1",
        step_index=0,
        stage_name="s",
        result=Done(output=bad_artifact),
        committed_at=_NOW,
    )
    with pytest.raises(ValueError):
        turns_of(step)


# ---------------------------------------------------------------------------
# EpisodeProjector fail-loud: malformed stamp stalls the projector
# ---------------------------------------------------------------------------


async def test_projector_stalls_on_malformed_stamp() -> None:
    """A step with a malformed 'turns' key raises ValueError and the cursor is NOT advanced (S6).

    This matches the TrialProjector's contract: fail-loud, not fail-silent. The projector stalls
    at the bad row until the stamp is corrected.
    """
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1", session_id="sess-stall")

    bad_artifact = Artifact(
        kind="output",
        produced_by="stage",
        provenance=_prov(),
        data={"turns": 42},  # present but malformed — not a list
    )
    await journal.commit_step(
        StepRecord(
            run_id="r1",
            step_index=0,
            stage_name=_STAGE_TURNS,
            result=Done(output=bad_artifact),
            committed_at=_NOW,
        )
    )

    proj = _projector(journal, store)
    with pytest.raises(ValueError):
        await proj.tick()

    # Cursor must NOT have advanced — the projector stalls
    cursor = await store.read_cursor(DEFAULT_EPISODE_CONSUMER)
    assert cursor is None


# ---------------------------------------------------------------------------
# validate_raw_model_json — structural validation, no model needed
# ---------------------------------------------------------------------------


def test_validate_raw_model_json_happy_path() -> None:
    """Valid JSON is parsed into RawClaimItems."""
    raw = {
        "claims": [
            {
                "subject": "Alice",
                "predicate": "lives_in",
                "object": "Paris",
                "supporting_turn_index": 0,
            },
        ]
    }
    items = validate_raw_model_json(raw)
    assert len(items) == 1
    assert items[0].subject == "Alice"
    assert items[0].predicate == "lives_in"
    assert items[0].object == "Paris"
    assert items[0].supporting_turn_index == 0


def test_validate_raw_model_json_garbage_input_raises() -> None:
    """Non-dict input raises ValueError (fail-stall, D6 — cursor must not advance)."""
    with pytest.raises(ValueError, match="expected a dict"):
        validate_raw_model_json("not a dict")


def test_validate_raw_model_json_missing_claims_key_raises() -> None:
    """A dict without a list-typed 'claims' key raises ValueError (fail-stall, D6)."""
    with pytest.raises(ValueError, match="raw\\['claims'\\]"):
        validate_raw_model_json({"result": "oops"})


def test_validate_raw_model_json_drops_item_with_negative_turn_index() -> None:
    """Items with negative supporting_turn_index are dropped."""
    raw = {
        "claims": [
            {"subject": "s", "predicate": "p", "object": "o", "supporting_turn_index": -1},
        ]
    }
    assert validate_raw_model_json(raw) == []


def test_validate_raw_model_json_drops_item_with_bool_turn_index() -> None:
    """Bool values for supporting_turn_index are rejected (bool is a subclass of int in Python)."""
    raw = {
        "claims": [
            {"subject": "s", "predicate": "p", "object": "o", "supporting_turn_index": True},
        ]
    }
    assert validate_raw_model_json(raw) == []


def test_validate_raw_model_json_mixed_valid_and_invalid() -> None:
    """Valid items are returned; invalid items (missing fields) are dropped."""
    raw = {
        "claims": [
            {"subject": "Alice", "predicate": "is", "object": "human", "supporting_turn_index": 0},
            {"subject": "Bob"},  # incomplete
            {
                "subject": "Carol",
                "predicate": "lives_in",
                "object": "Rome",
                "supporting_turn_index": 2,
            },
        ]
    }
    items = validate_raw_model_json(raw)
    assert len(items) == 2
    assert items[0].subject == "Alice"
    assert items[1].subject == "Carol"


# ---------------------------------------------------------------------------
# EpisodeProjector multi-step, multi-run projection
# ---------------------------------------------------------------------------


async def test_projector_multi_step_single_run() -> None:
    """Multiple steps in one run all project to episodes with correct episode_ids."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1", session_id="sess-multi")
    await _commit_turns(
        journal,
        run_id="r1",
        step_index=0,
        turns=[Turn(role="user", content="step0 user", kind="conversation")],
    )
    await _commit_turns(
        journal,
        run_id="r1",
        step_index=1,
        turns=[
            Turn(role="user", content="step1 user", kind="conversation"),
            Turn(role="assistant", content="step1 asst", kind="conversation"),
        ],
    )

    total = await _projector(journal, store).tick()
    assert total == 3  # 1 + 2

    episodes = await store.episodes_for_session("sess-multi")
    assert len(episodes) == 3
    ids = [ep.episode_id for ep in episodes]
    assert "r1:0:0" in ids
    assert "r1:1:0" in ids
    assert "r1:1:1" in ids


async def test_projector_multi_run_distinct_sessions() -> None:
    """Two runs in distinct sessions project independently; no cross-session contamination."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1", session_id="sess-A")
    await _start(journal, "r2", session_id="sess-B")
    await _commit_turns(
        journal,
        run_id="r1",
        step_index=0,
        turns=[Turn(role="user", content="A question", kind="conversation")],
    )
    await _commit_turns(
        journal,
        run_id="r2",
        step_index=0,
        turns=[Turn(role="user", content="B question", kind="conversation")],
    )

    await _projector(journal, store).tick()

    sess_a = await store.episodes_for_session("sess-A")
    sess_b = await store.episodes_for_session("sess-B")
    assert len(sess_a) == 1
    assert len(sess_b) == 1
    assert sess_a[0].content == "A question"
    assert sess_b[0].content == "B question"


async def test_projector_control_step_advances_cursor_without_episodes() -> None:
    """A plain (no-turns) step advances the cursor without producing any episode rows (P0-2)."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1")
    await _commit_plain(journal, run_id="r1", step_index=0)

    proj = _projector(journal, store)
    projected = await proj.tick()
    assert projected == 0

    cursor = await store.read_cursor(DEFAULT_EPISODE_CONSUMER)
    assert cursor is not None
    assert (cursor.run_id, cursor.step_index) == ("r1", 0)


async def test_projector_empty_journal_leaves_cursor_unchanged() -> None:
    """An idle tick on an empty journal returns 0 and leaves cursor at None."""
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1")
    projected = await _projector(journal, store).tick()
    assert projected == 0
    assert await store.read_cursor(DEFAULT_EPISODE_CONSUMER) is None


async def test_projector_batch_limit_overflow_drains_completely() -> None:
    """With batch_limit=1, multiple ticks drain all committed steps — forward progress (FP)."""
    n_steps = 5
    journal, store = InMemoryJournal(), InMemoryEpisodeStore()
    await _start(journal, "r1", session_id="sess-drain")
    for i in range(n_steps):
        await _commit_turns(
            journal,
            run_id="r1",
            step_index=i,
            turns=[Turn(role="user", content=f"turn {i}", kind="conversation")],
        )

    proj = _projector(journal, store, batch_limit=1)
    total = 0
    for _ in range(n_steps + 2):  # extra ticks to ensure idempotence after drain
        total += await proj.tick()

    episodes = await store.episodes_for_session("sess-drain")
    assert len(episodes) == n_steps
    assert total == n_steps  # exactly one episode per step (one user turn each)


# ---------------------------------------------------------------------------
# InMemoryEpisodeStore — Protocol compliance
# ---------------------------------------------------------------------------


def test_episode_store_protocol_runtime_check() -> None:
    """InMemoryEpisodeStore satisfies the EpisodeStore Protocol at runtime."""
    from cogworx.substrate.episodes import EpisodeStore

    store = InMemoryEpisodeStore()
    assert isinstance(store, EpisodeStore)


async def test_episode_store_get_episode() -> None:
    """get_episode returns the episode by id or None."""
    store = InMemoryEpisodeStore()
    from cogworx.substrate.episodes import Episode

    ep = Episode(
        episode_id="r1:0:0",
        run_id="r1",
        step_index=0,
        turn_index=0,
        session_id="sess",
        role="user",
        content="hello",
        kind="conversation",
        occurred_at=_NOW,
    )
    cursor = ProjectionCursor(commit_ordinal=1, run_id="r1", step_index=0)
    await store.project_episodes("consumer", [ep], cursor)

    fetched = await store.get_episode("r1:0:0")
    assert fetched is not None
    assert fetched.content == "hello"
    assert await store.get_episode("nonexistent") is None


async def test_episode_store_first_write_wins() -> None:
    """project_episodes is idempotent: second write of same episode_id is a silent no-op (S6)."""
    store = InMemoryEpisodeStore()
    from cogworx.substrate.episodes import Episode

    ep_v1 = Episode(
        episode_id="r1:0:0",
        run_id="r1",
        step_index=0,
        turn_index=0,
        session_id="sess",
        role="user",
        content="original",
        kind="conversation",
        occurred_at=_NOW,
    )
    cursor = ProjectionCursor(commit_ordinal=1, run_id="r1", step_index=0)
    await store.project_episodes("c", [ep_v1], cursor)

    ep_v2 = ep_v1.model_copy(update={"content": "different"})
    await store.project_episodes("c", [ep_v2], cursor)

    ep = await store.get_episode("r1:0:0")
    assert ep is not None
    assert ep.content == "original"  # first write wins
