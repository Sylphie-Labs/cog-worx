"""Integration tests for the Neo4j procedural-KG adapter (CANON S1, S3, S5, S6).

These hit a REAL Neo4j (``docker compose up -d`` first) and ERROR — not silently pass — when it is
unreachable. They only run under ``-m integration``. Each case gets an ephemeral, wiped graph via
the per-case fixture, verifying additivity (the procedural-KG labels coexist with the entity-KG /
base :Claim schema) and per-case isolation.

Covers (mirrors the unit double suite so the SAME behaviour is asserted against both):
  - record_trial round-trip: trial node + provenance + run_id/step_index split → get_trial exact.
  - Idempotent accumulation: re-projecting the same step → one :Trial, derived posterior unchanged.
  - First-write-wins (spike a5): re-MERGE with a FLIPPED outcome → stored outcome UNCHANGED, and the
    same probed at the posterior level (no silent counter mutation).
  - APPLIES_TO is topology-only: the edge carries no posterior/counter properties.
  - Posterior derive-at-read + run-level dedup grain (50 same-run successes = ONE contribution).
  - Out-of-order trial arrival; empty-edge prior-only; UTC datetime contract (naive + offset).
  - candidates ordering + promoted_only.
  - Sealed blanket-SET back-door (upsert_claim raises).
  - Parity spot-checks: identical scenario against InMemoryProceduralKG and Neo4jProceduralKG →
    identical posterior + candidates ordering.
"""

from __future__ import annotations

import random
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta, timezone

import pytest
from neo4j import AsyncManagedTransaction

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_procedural_kg import Neo4jProceduralKG
from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for
from cogworx.substrate.procedural_kg import ProceduralKG
from cogworx.testing.doubles import InMemoryProceduralKG

pytestmark = pytest.mark.integration

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)

_PID = procedure_id_for("math_pathway", "solve_stage")
_PT = problem_type_id_for("word problem")


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jProceduralKG]:
    """Real Neo4jProceduralKG, schema ensured (additively), wiped per case."""
    adapter = Neo4jProceduralKG(settings=settings)
    await adapter.ensure_schema()
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


def _prov() -> Provenance:
    return Provenance(source="system", confidence=1.0, recorded_at=_T0)


async def _record(
    kg: ProceduralKG,
    *,
    trial_id: str,
    outcome: str,
    procedure_id: str = _PID,
    problem_type: str = _PT,
    occurred_at: datetime = _T0,
) -> None:
    await kg.record_trial(
        trial_id=trial_id,
        procedure_id=procedure_id,
        problem_type=problem_type,
        outcome=outcome,  # type: ignore[arg-type]
        occurred_at=occurred_at,
        provenance=_prov(),
    )


# ---------------------------------------------------------------------------
# Round-trip
# ---------------------------------------------------------------------------


async def test_record_trial_round_trip(kg: Neo4jProceduralKG) -> None:
    """record_trial → get_trial reconstructs the full Trial, splitting run_id/step_index."""
    await _record(kg, trial_id="sess:abc:42", outcome="success", occurred_at=_T1)
    trial = await kg.get_trial("sess:abc:42")
    assert trial is not None
    assert trial.run_id == "sess:abc"
    assert trial.step_index == 42
    assert trial.procedure_id == _PID
    assert trial.problem_type == _PT
    assert trial.outcome == "success"
    assert trial.occurred_at == _T1
    assert trial.provenance.source == "system"


async def test_get_trial_unknown_returns_none(kg: Neo4jProceduralKG) -> None:
    """get_trial returns None for an unknown id."""
    assert await kg.get_trial("nope:0") is None


# ---------------------------------------------------------------------------
# Idempotent accumulation + first-write-wins (spike a5)
# ---------------------------------------------------------------------------


async def test_record_trial_idempotent(kg: Neo4jProceduralKG) -> None:
    """Re-projecting the same committed step → exactly one :Trial node."""
    await _record(kg, trial_id="r1:0", outcome="success")
    await _record(kg, trial_id="r1:0", outcome="success")
    trials = await kg.trials_for(_PID, _PT)
    assert len(trials) == 1


async def test_record_trial_first_write_wins_on_flip(kg: Neo4jProceduralKG) -> None:
    """Spike a5: re-MERGE same trial_id with a FLIPPED outcome → stored outcome UNCHANGED.

    Probes the ON-CREATE-only MERGE: there is no ON MATCH SET, so a different payload for an
    already-present trial_id is a no-op.
    """
    await _record(kg, trial_id="r1:0", outcome="success")
    await _record(kg, trial_id="r1:0", outcome="failure")  # flip — must be a no-op
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.outcome == "success"


async def test_first_write_wins_does_not_move_posterior(kg: Neo4jProceduralKG) -> None:
    """The flipped re-MERGE leaves the derived posterior identical (no silent counter mutation)."""
    await _record(kg, trial_id="r1:0", outcome="success")
    before = await kg.posterior(_PID, _PT)
    await _record(kg, trial_id="r1:0", outcome="failure")
    after = await kg.posterior(_PID, _PT)
    assert (before.alpha, before.beta) == (after.alpha, after.beta)
    assert after.alpha == pytest.approx(2.0)
    assert after.beta == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# APPLIES_TO is topology-only (no counters/posterior on the edge)
# ---------------------------------------------------------------------------


async def test_applies_to_edge_has_no_properties(kg: Neo4jProceduralKG) -> None:
    """The APPLIES_TO edge carries NO stored properties (topology only — D1)."""
    await _record(kg, trial_id="r1:0", outcome="success")

    async def _probe(tx: AsyncManagedTransaction) -> list[dict[str, object]]:
        result = await tx.run(
            "MATCH (:Procedure {id: $pid})-[r:APPLIES_TO]->(:ProblemType {id: $pt}) "
            "RETURN properties(r) AS props",
            pid=_PID,
            pt=_PT,
        )
        return [rec.data() async for rec in result]

    async with kg._connection.session() as session:
        rows = await session.execute_read(_probe)
    assert rows, "APPLIES_TO edge should exist"
    assert rows[0]["props"] == {}, "APPLIES_TO must be topology-only (no counters/posterior)"


# ---------------------------------------------------------------------------
# Posterior — derive-at-read, dedup grain, prior-only
# ---------------------------------------------------------------------------


async def test_posterior_empty_edge_prior_only(kg: Neo4jProceduralKG) -> None:
    """Empty edge → prior-only: success_rate 0.5, n_trials 0."""
    post = await kg.posterior(_PID, _PT)
    assert post.success_rate == pytest.approx(0.5)
    assert post.n_trials == 0


async def test_posterior_run_level_dedup(kg: Neo4jProceduralKG) -> None:
    """50 successes from ONE run = one contribution, not 50 (grain dedup at source_id=run_id)."""
    for i in range(50):
        await _record(kg, trial_id=f"cyc:{i}", outcome="success")
    post = await kg.posterior(_PID, _PT)
    assert post.n_trials == 1
    assert post.alpha == pytest.approx(2.0)
    assert post.beta == pytest.approx(1.0)
    assert len(await kg.trials_for(_PID, _PT)) == 50  # raw trials still stored


async def test_posterior_distinct_runs(kg: Neo4jProceduralKG) -> None:
    """Distinct runs each contribute once per polarity."""
    await _record(kg, trial_id="r1:0", outcome="success")
    await _record(kg, trial_id="r2:0", outcome="success")
    await _record(kg, trial_id="r3:0", outcome="failure")
    post = await kg.posterior(_PID, _PT)
    assert post.n_trials == 3
    assert post.alpha == pytest.approx(3.0)
    assert post.beta == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Out-of-order arrival + occurrence order
# ---------------------------------------------------------------------------


async def test_trials_for_out_of_order_arrival(kg: Neo4jProceduralKG) -> None:
    """Trials arriving out of order are returned in occurrence order."""
    await _record(kg, trial_id="r1:2", outcome="success", occurred_at=_T2)
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=_T0)
    await _record(kg, trial_id="r1:1", outcome="failure", occurred_at=_T1)
    trials = await kg.trials_for(_PID, _PT)
    assert [t.occurred_at for t in trials] == [_T0, _T1, _T2]


# ---------------------------------------------------------------------------
# Datetime contract
# ---------------------------------------------------------------------------


async def test_naive_occurred_at_interpreted_as_utc(kg: Neo4jProceduralKG) -> None:
    """A naive occurred_at is interpreted as UTC."""
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=datetime(2026, 6, 9, 12))
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.occurred_at == datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)


async def test_offset_occurred_at_converted_to_utc(kg: Neo4jProceduralKG) -> None:
    """A tz-aware occurred_at is converted to UTC (and the stored ISO string is +00:00)."""
    plus_two = timezone(timedelta(hours=2))
    offset = datetime(2026, 6, 9, 14, tzinfo=plus_two)  # == 12:00Z
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=offset)
    trial = await kg.get_trial("r1:0")
    assert trial is not None
    assert trial.occurred_at == datetime(2026, 6, 9, 12, 0, 0, tzinfo=UTC)


async def test_occurrence_order_mixed_offsets(kg: Neo4jProceduralKG) -> None:
    """Mixed-offset trials sort by true instant (lexicographic == temporal after UTC-normalise)."""
    plus_two = timezone(timedelta(hours=2))
    later = datetime(2026, 6, 9, 13, tzinfo=UTC)
    earlier = datetime(2026, 6, 9, 14, tzinfo=plus_two)  # == 12:00Z
    await _record(kg, trial_id="r1:1", outcome="failure", occurred_at=later)
    await _record(kg, trial_id="r1:0", outcome="success", occurred_at=earlier)
    trials = await kg.trials_for(_PID, _PT)
    assert [t.trial_id for t in trials] == ["r1:0", "r1:1"]


# ---------------------------------------------------------------------------
# candidates
# ---------------------------------------------------------------------------


async def test_candidates_empty_returns_empty(kg: Neo4jProceduralKG) -> None:
    """candidates on an unknown problem type returns empty."""
    assert await kg.candidates(_PT) == ()


async def test_candidates_default_is_posterior_mean_greedy_deterministic(
    kg: Neo4jProceduralKG,
) -> None:
    """The production default (no rng) is deterministic posterior-mean greedy on the adapter."""
    for i in range(5):
        await _record(
            kg, trial_id=f"r{i}:0", outcome="success", procedure_id=procedure_id_for("p", f"s{i}")
        )
    order_a = [c.procedure.id for c in await kg.candidates(_PT)]
    order_b = [c.procedure.id for c in await kg.candidates(_PT)]
    assert order_a == order_b
    assert len(order_a) == 5


async def test_candidates_explicit_thompson_seeded_is_deterministic(kg: Neo4jProceduralKG) -> None:
    """The opt-in Thompson path: same seed → identical ordering against the real adapter."""
    for i in range(5):
        await _record(
            kg, trial_id=f"r{i}:0", outcome="success", procedure_id=procedure_id_for("p", f"s{i}")
        )
    order_a = [c.procedure.id for c in await kg.candidates(_PT, rng=random.Random(7))]
    order_b = [c.procedure.id for c in await kg.candidates(_PT, rng=random.Random(7))]
    assert order_a == order_b
    assert len(order_a) == 5


async def test_candidates_limit_and_promoted_only(kg: Neo4jProceduralKG) -> None:
    """candidates respects limit; promoted_only keeps only edges that clear the gate.

    strong gets 12 distinct-run successes → Beta(13,1), LCB clears 0.70 at the q=0.025 gate.
    (8/8 no longer promotes after the Pod 2.1 gate tightening.)
    """
    strong = procedure_id_for("p", "strong")
    for i in range(12):
        await _record(kg, trial_id=f"s{i}:0", outcome="success", procedure_id=strong)
    for i in range(4):
        await _record(
            kg, trial_id=f"w{i}:0", outcome="success", procedure_id=procedure_id_for("p", f"w{i}")
        )
    assert len(await kg.candidates(_PT, limit=3)) == 3
    promoted = await kg.candidates(_PT, promoted_only=True)
    assert [c.procedure.id for c in promoted] == [strong]


# ---------------------------------------------------------------------------
# Sealed write-seam (2.0 red-team #1)
# ---------------------------------------------------------------------------


async def test_upsert_claim_raises_not_implemented(kg: Neo4jProceduralKG) -> None:
    """Neo4jProceduralKG.upsert_claim raises NotImplementedError (sealed blanket-SET back-door)."""
    claim = Claim(
        id="x",
        subject="s",
        payload="p",
        epistemic_type="observation",
        provenance=_prov(),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    with pytest.raises(NotImplementedError, match="record_trial"):
        await kg.upsert_claim(claim)


# ---------------------------------------------------------------------------
# Schema additivity
# ---------------------------------------------------------------------------


async def test_ensure_schema_additive_with_entity_kg(settings: SubstrateSettings) -> None:
    """Procedural-KG constraints coexist with the entity-KG schema in the same DB (additive).

    Both ensure_schema calls must succeed against the same community-edition instance.
    """
    from cogworx.adapters.neo4j_entity_kg import Neo4jEntityKG

    entity = Neo4jEntityKG(settings=settings)
    proc = Neo4jProceduralKG(settings=settings)
    try:
        await entity.ensure_schema(embedding_dim=4)
        await proc.ensure_schema()
        # Re-running both is idempotent (IF NOT EXISTS).
        await entity.ensure_schema(embedding_dim=4)
        await proc.ensure_schema()
        # The procedural adapter still works after the entity schema is present.
        await proc.reset()
        await _record(proc, trial_id="r1:0", outcome="success")
        post = await proc.posterior(_PID, _PT)
        assert post.n_trials == 1
    finally:
        await entity.aclose()
        await proc.aclose()


# ---------------------------------------------------------------------------
# Parity: InMemoryProceduralKG vs Neo4jProceduralKG (the SAME scenario, both sides)
# ---------------------------------------------------------------------------


async def test_parity_posterior_and_candidates(kg: Neo4jProceduralKG) -> None:
    """Same scenario against both → identical posterior + identical candidates ordering.

    Verifies the double has not drifted from the adapter on the load-bearing read surface:
    derive-at-read posterior, run-level dedup, and the interim candidate ordering.
    """
    mem = InMemoryProceduralKG()

    high = procedure_id_for("p", "high")
    mid = procedure_id_for("p", "mid")
    low = procedure_id_for("p", "low")

    scenario = [
        # high: 3 distinct-run successes
        ("high:0", "success", high),
        ("hi2:0", "success", high),
        ("hi3:0", "success", high),
        # mid: one run, 10 successes (dedup → 1 contribution)
        *[(f"midrun:{i}", "success", mid) for i in range(10)],
        # low: 1 success, 2 distinct-run failures
        ("lo1:0", "success", low),
        ("lo2:0", "failure", low),
        ("lo3:0", "failure", low),
    ]

    for impl in (kg, mem):
        for trial_id, outcome, pid in scenario:
            await _record(impl, trial_id=trial_id, outcome=outcome, procedure_id=pid)

    # Posterior parity per edge.
    for pid in (high, mid, low):
        neo = await kg.posterior(pid, _PT)
        dbl = await mem.posterior(pid, _PT)
        assert neo == dbl, f"posterior parity failure for {pid}"

    # mid deduped to ONE contribution despite 10 trials.
    assert (await kg.posterior(mid, _PT)).n_trials == 1
    assert len(await kg.trials_for(mid, _PT)) == 10

    # Candidates Thompson ordering + posteriors parity: the SAME seed must give the SAME order
    # across both implementations (the determinism + parity bet), even though candidates arrive in
    # different iteration orders (Neo4j ORDER BY p.id vs the double's set iteration).
    neo_cands = await kg.candidates(_PT, rng=random.Random(42))
    mem_cands = await mem.candidates(_PT, rng=random.Random(42))
    assert [c.procedure.id for c in neo_cands] == [c.procedure.id for c in mem_cands]
    neo_rates = {c.procedure.id: c.success.success_rate for c in neo_cands}
    mem_rates = {c.procedure.id: c.success.success_rate for c in mem_cands}
    assert neo_rates == mem_rates
    neo_promoted = {c.procedure.id: c.promoted for c in neo_cands}
    mem_promoted = {c.procedure.id: c.promoted for c in mem_cands}
    assert neo_promoted == mem_promoted


async def test_parity_first_write_wins(kg: Neo4jProceduralKG) -> None:
    """First-write-wins is identical in both implementations (flip → no-op, same stored outcome)."""
    mem = InMemoryProceduralKG()
    for impl in (kg, mem):
        await _record(impl, trial_id="r1:0", outcome="success")
        await _record(impl, trial_id="r1:0", outcome="failure")  # flip — no-op in both

    neo_trial = await kg.get_trial("r1:0")
    mem_trial = await mem.get_trial("r1:0")
    assert neo_trial is not None and mem_trial is not None
    assert neo_trial.outcome == mem_trial.outcome == "success"
    assert await kg.posterior(_PID, _PT) == await mem.posterior(_PID, _PT)
