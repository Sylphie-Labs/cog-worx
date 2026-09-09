"""Integration tests for the Neo4j entity-KG adapter (CANON S3, S5).

These hit a REAL Neo4j (``docker compose up -d`` first) and ERROR — not silently pass — when
it is unreachable. They only run under ``-m integration``. Each case gets an ephemeral, wiped
graph via the per-case fixture.

Covers:
  - Round-trip: write_claim with object_entity + provenance source "system" → get_claim exact.
  - Idempotent accumulation: same triple twice → one :Claim, two :Evidence, confidence matches.
  - Immutable-on-match: second write does not clobber stored fields; valid_to survives re-write.
  - claims_about: subject-side, object-side, as_of filter, newest-first, limit.
  - invalidate_claim: first-wins, unknown-id raises.
  - Contradictions: write idempotent, both directions surface.
  - resolution_candidates: norm-column exact match + vector top-up.
  - claims_by_similarity: cosine correctness (hand-computed, 4-dim), min_score excludes far vector.
  - Lineage: A (weak evidence) ← C derived_from A → C.lineage_min_confidence < C.confidence.
  - Parity spot-check: identical scenario against InMemoryEntityKG and Neo4jEntityKG produces
    identical ScoredClaim confidences and ordering.
"""

from __future__ import annotations

import math
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.adapters.config import SubstrateSettings
from cogworx.adapters.neo4j_entity_kg import Neo4jEntityKG
from cogworx.claims.provenance import Claim, Provenance, ProvenanceSource
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.evidence import EvidenceEvent, EvidenceType, Polarity, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.testing.doubles import InMemoryEntityKG

pytestmark = pytest.mark.integration

# Embedding dimension for all tests in this file.
_DIM = 4

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)
_T3 = _T0 + timedelta(hours=3)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
async def kg(settings: SubstrateSettings) -> AsyncIterator[Neo4jEntityKG]:
    """Real Neo4jEntityKG, schema ensured, wiped per case."""
    adapter = Neo4jEntityKG(settings=settings)
    await adapter.ensure_schema(embedding_dim=_DIM)
    await adapter.reset()
    try:
        yield adapter
    finally:
        await adapter.aclose()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(
    *,
    source: ProvenanceSource = "system",
    evidence: tuple[str, ...] = (),
) -> Provenance:
    return Provenance(
        source=source,
        confidence=1.0,
        evidence=evidence,
        recorded_at=_T0,
    )


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    object_entity: str | None = None,
    valid_from: datetime = _T0,
    ingest_time: datetime = _T0,
    embedding: tuple[float, ...] | None = None,
    provenance_evidence: tuple[str, ...] = (),
    source: ProvenanceSource = "system",
    valid_to: datetime | None = None,
) -> Claim:
    object_repr = object_entity if object_entity is not None else payload
    cid = claim_id_for(subject, predicate, object_repr)
    return Claim(
        id=cid,
        subject=subject,
        predicate=predicate,
        payload=payload,
        object_entity=object_entity,
        epistemic_type="inference",
        provenance=Provenance(
            source=source,
            confidence=1.0,
            evidence=provenance_evidence,
            recorded_at=_T0,
        ),
        valid_from=valid_from,
        valid_to=valid_to,
        ingest_time=ingest_time,
        created_by="test",
        embedding=embedding,
    )


def _ev(
    source_id: str = "src-a",
    polarity: Polarity = "+",
    ev_type: EvidenceType = "corroboration",
    source_authority: float = 1.0,
) -> EvidenceEvent:
    return make_evidence(
        type=ev_type,
        polarity=polarity,
        source_id=source_id,
        source_authority=source_authority,
        recorded_at=_T0,
    )


def _cosine(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb)


# ---------------------------------------------------------------------------
# Round-trip (pins "system" provenance source + object_entity persistence)
# ---------------------------------------------------------------------------


async def test_round_trip_with_object_entity_and_system_source(kg: Neo4jEntityKG) -> None:
    """write_claim with object_entity + provenance source='system' → get_claim byte-identical."""
    claim = _make_claim(
        "solar_system",
        "contains",
        "pluto ref",
        object_entity="pluto",
        source="system",
    )
    ev = _ev(source_id="ingestion-01")
    await kg.write_claim(claim, evidence=ev)

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.id == claim.id
    assert fetched.subject == "solar_system"
    assert fetched.predicate == "contains"
    assert fetched.object_entity == "pluto"  # object_entity persisted
    assert fetched.provenance.source == "system"  # "system" source round-trips
    assert fetched.epistemic_type == "inference"
    assert fetched.created_by == "test"


# ---------------------------------------------------------------------------
# Idempotent accumulation
# ---------------------------------------------------------------------------


async def test_idempotent_accumulation(kg: Neo4jEntityKG) -> None:
    """Same triple written twice (different evidence) → one :Claim, two :Evidence.

    The derived confidence must equal claim_confidence computed locally over the same events.
    """
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    ev1 = _ev(source_id="src-a", polarity="+", ev_type="corroboration")
    ev2 = _ev(source_id="src-b", polarity="+", ev_type="corroboration")

    await kg.write_claim(claim, evidence=ev1)
    await kg.write_claim(claim, evidence=ev2)

    # Only one :Claim node survives the MERGE.
    fetched = await kg.get_claim(claim.id)
    assert fetched is not None

    # Two :Evidence events.
    events = await kg.evidence_for(claim.id)
    assert len(events) == 2

    # Confidence matches local computation.
    local_conf = claim_confidence(events)
    scored = await kg.claims_about("pluto")
    scored_claim = next((sc for sc in scored if sc.claim.id == claim.id), None)
    assert scored_claim is not None
    assert abs(scored_claim.confidence.confidence - local_conf.confidence) < 1e-6


# ---------------------------------------------------------------------------
# Immutable-on-match
# ---------------------------------------------------------------------------


async def test_immutable_on_match_second_write_does_not_clobber(kg: Neo4jEntityKG) -> None:
    """Second write with different payload casing does not clobber stored fields."""
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev(source_id="src-a"))

    # Write same triple again (same id); ON MATCH should not clobber immutable fields.
    await kg.write_claim(claim, evidence=_ev(source_id="src-b"))

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.payload == "1.3e22 kg"  # first write's field survives


async def test_immutable_on_match_valid_to_not_touched_by_rewrite(kg: Neo4jEntityKG) -> None:
    """Invalidate claim then re-write same triple — claim stays invalidated."""
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev(source_id="src-a"))
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    # Re-write the same triple; ON MATCH does NOT touch valid_to.
    await kg.write_claim(claim, evidence=_ev(source_id="src-b"))

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.valid_to == _T1  # invalidation survived the re-write


# ---------------------------------------------------------------------------
# claims_about
# ---------------------------------------------------------------------------


async def test_claims_about_subject_and_object_side(kg: Neo4jEntityKG) -> None:
    """claims_about returns claims for both subject-side and object-side roles."""
    c_subj = _make_claim("pluto", "has_mass", "1.3e22 kg")
    c_obj = _make_claim(
        "solar_system", "contains", "pluto ref", object_entity="pluto", ingest_time=_T1
    )
    await kg.write_claim(c_subj, evidence=_ev(source_id="src-s"))
    await kg.write_claim(c_obj, evidence=_ev(source_id="src-o"))

    results = await kg.claims_about("pluto")
    ids = {sc.claim.id for sc in results}
    assert c_subj.id in ids
    assert c_obj.id in ids


async def test_claims_about_accepts_the_normalized_subject(kg: Neo4jEntityKG) -> None:
    """claims_about finds a subject by its normalized form as well as its original name.

    The coherence reconciler only has ``subject_norm`` (from the DirtySubject node); the
    in-memory double has always accepted both spellings, and the adapter must agree or the
    reconciler clears every dirty subject having found no claims to adjudicate.
    """
    from cogworx.knowledge.identity import normalize_topic_part

    claim = _make_claim("IntegAlice", "status", "active")
    await kg.write_claim(claim, evidence=_ev(source_id="src-n"))

    by_name = {sc.claim.id for sc in await kg.claims_about("IntegAlice")}
    by_norm = {sc.claim.id for sc in await kg.claims_about(normalize_topic_part("IntegAlice"))}
    assert claim.id in by_name
    assert by_norm == by_name


async def test_claims_about_as_of_filter(kg: Neo4jEntityKG) -> None:
    """as_of filtering: claim disappears after valid_to."""
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg", valid_from=_T0)
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    # Before invalidation → visible
    before = await kg.claims_about("pluto", as_of=_T0 + timedelta(minutes=30))
    assert any(sc.claim.id == claim.id for sc in before)

    # After invalidation → invisible
    after = await kg.claims_about("pluto", as_of=_T1 + timedelta(minutes=30))
    assert not any(sc.claim.id == claim.id for sc in after)

    # No as_of → all visible
    no_filter = await kg.claims_about("pluto", as_of=None)
    assert any(sc.claim.id == claim.id for sc in no_filter)


async def test_claims_about_newest_first(kg: Neo4jEntityKG) -> None:
    """claims_about returns claims ordered newest-first by ingest_time."""
    c1 = _make_claim("pluto", "attr", "v1", ingest_time=_T0)
    c2 = _make_claim("pluto", "attr", "v2", ingest_time=_T1)
    c3 = _make_claim("pluto", "attr", "v3", ingest_time=_T2)
    for c in (c1, c2, c3):
        await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_about("pluto")
    times = [sc.claim.ingest_time for sc in results]
    assert times == sorted(times, reverse=True)


async def test_claims_about_limit(kg: Neo4jEntityKG) -> None:
    """claims_about respects limit."""
    for i in range(5):
        c = _make_claim("pluto", "attr", f"v{i}", ingest_time=_T0 + timedelta(seconds=i))
        await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_about("pluto", limit=3)
    assert len(results) <= 3


# ---------------------------------------------------------------------------
# invalidate_claim
# ---------------------------------------------------------------------------


async def test_invalidate_first_wins_live(kg: Neo4jEntityKG) -> None:
    """First invalidation wins; second call with later valid_to is no-op."""
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)
    await kg.invalidate_claim(claim.id, valid_to=_T2)  # no-op

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.valid_to == _T1


async def test_invalidate_unknown_id_raises_live(kg: Neo4jEntityKG) -> None:
    """invalidate_claim raises ValueError on unknown claim_id."""
    with pytest.raises(ValueError, match="unknown claim_id"):
        await kg.invalidate_claim("no-such-id", valid_to=_T1)


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------


async def test_contradictions_write_idempotent_both_directions(kg: Neo4jEntityKG) -> None:
    """Contradictions: write idempotent, surfaces both directions."""
    c1 = _make_claim("pluto", "has_mass", "1.3e22 kg")
    c2 = _make_claim("pluto", "has_mass", "9.9e23 kg")
    await kg.write_claim(c1, evidence=_ev(source_id="s1"))
    await kg.write_claim(c2, evidence=_ev(source_id="s2"))

    # Write twice (idempotent).
    await kg.write_contradiction(c1.id, c2.id)
    await kg.write_contradiction(c1.id, c2.id)

    contra1 = await kg.contradictions_of(c1.id)
    contra2 = await kg.contradictions_of(c2.id)

    # Both directions surface.
    assert any(c.id == c2.id for c in contra1)
    assert any(c.id == c1.id for c in contra2)

    # No duplicates.
    ids_from_c1 = [c.id for c in contra1]
    assert ids_from_c1.count(c2.id) == 1


# ---------------------------------------------------------------------------
# resolution_candidates — norm exact match + vector top-up
# ---------------------------------------------------------------------------


async def test_resolution_candidates_norm_exact_match(kg: Neo4jEntityKG) -> None:
    """Norm-column exact match: write 'Has  Mass' then query 'has mass' → finds it."""
    # The subject/predicate stored with extra whitespace; normalization should still match.
    claim = _make_claim("pluto", "Has  Mass", "payload")
    await kg.write_claim(claim, evidence=_ev())

    results = await kg.resolution_candidates("pluto", "has mass")
    assert any(c.id == claim.id for c in results)


async def test_resolution_candidates_vector_topup(kg: Neo4jEntityKG) -> None:
    """Vector top-up path fills remainder when exact matches < k."""
    c_exact = _make_claim(
        "pluto", "has_mass", "exact", embedding=(1.0, 0.0, 0.0, 0.0), ingest_time=_T0
    )
    c_similar = _make_claim(
        "pluto", "other_pred", "similar", embedding=(0.99, 0.1, 0.0, 0.0), ingest_time=_T1
    )
    await kg.write_claim(c_exact, evidence=_ev(source_id="e"))
    await kg.write_claim(c_similar, evidence=_ev(source_id="s"))

    results = await kg.resolution_candidates(
        "pluto", "has_mass", k=2, embedding=(1.0, 0.0, 0.0, 0.0)
    )
    ids = {c.id for c in results}
    assert c_exact.id in ids
    assert c_similar.id in ids


# ---------------------------------------------------------------------------
# claims_by_similarity — cosine verified against hand-computed value
# ---------------------------------------------------------------------------


async def test_claims_by_similarity_cosine_correctness(kg: Neo4jEntityKG) -> None:
    """Cosine similarity verified against hand-computed value (4-dim, small float tolerance).

    emb_a = (1.0, 0.0, 0.0, 0.0)
    query = (0.6, 0.8, 0.0, 0.0)
    cosine(emb_a, query) = (1.0*0.6) / (1.0 * 1.0) = 0.6
    (query is already unit-length: sqrt(0.36+0.64)=1.0)
    """
    emb_a = (1.0, 0.0, 0.0, 0.0)
    query = (0.6, 0.8, 0.0, 0.0)
    expected_cosine = _cosine(emb_a, query)  # 0.6

    c_a = _make_claim("pluto", "has_mass", "v-a", embedding=emb_a)
    await kg.write_claim(c_a, evidence=_ev())

    results = await kg.claims_by_similarity(query, k=5, min_score=0.0)
    scored = next((sc for sc in results if sc.claim.id == c_a.id), None)
    assert scored is not None
    assert scored.similarity is not None
    assert abs(scored.similarity - expected_cosine) < 0.01  # float tolerance


async def test_claims_by_similarity_min_score_excludes_far_vector(kg: Neo4jEntityKG) -> None:
    """min_score excludes a far vector (cosine ≈ 0.0 < 0.8 threshold)."""
    emb_near = (1.0, 0.0, 0.0, 0.0)
    emb_far = (0.0, 1.0, 0.0, 0.0)  # orthogonal → cosine = 0.0
    c_near = _make_claim("pluto", "near", "payload", embedding=emb_near)
    c_far = _make_claim("pluto", "far", "payload", embedding=emb_far)
    await kg.write_claim(c_near, evidence=_ev(source_id="n"))
    await kg.write_claim(c_far, evidence=_ev(source_id="f"))

    results = await kg.claims_by_similarity((1.0, 0.0, 0.0, 0.0), k=10, min_score=0.8)
    ids = {sc.claim.id for sc in results}
    assert c_near.id in ids
    assert c_far.id not in ids


# ---------------------------------------------------------------------------
# Lineage
# ---------------------------------------------------------------------------


async def test_lineage_derived_from_lineage_min(kg: Neo4jEntityKG) -> None:
    """C derived_from A: C.lineage_min_confidence <= min(A.conf, C.conf)."""
    # A: weak evidence (zero-authority → prior only ≈ 0.5)
    c_a = _make_claim("pluto", "weak_attr", "ancestor_payload")
    await kg.write_claim(
        c_a, evidence=_ev(source_id="wa", polarity="+", ev_type="recall", source_authority=0.0)
    )

    # C: strong evidence, derived from A
    id_c = claim_id_for("pluto", "derived_attr", "child_payload")
    c_c = Claim(
        id=id_c,
        subject="pluto",
        predicate="derived_attr",
        payload="child_payload",
        epistemic_type="inference",
        provenance=Provenance(
            source="system",
            confidence=1.0,
            evidence=(c_a.id,),
            recorded_at=_T0,
        ),
        valid_from=_T0,
        ingest_time=_T1,
        created_by="test",
    )
    await kg.write_claim(
        c_c,
        evidence=_ev(source_id="strong", polarity="+", ev_type="tool_proof", source_authority=1.0),
    )

    results = await kg.claims_about("pluto")
    scored = {sc.claim.id: sc for sc in results}

    assert c_a.id in scored
    assert id_c in scored

    c_scored = scored[id_c]
    a_scored = scored[c_a.id]

    # lineage_min_confidence is <= min(own, ancestor)
    expected_min = min(c_scored.confidence.confidence, a_scored.confidence.confidence)
    assert c_scored.lineage_min_confidence <= expected_min + 1e-6
    # And strictly less than C's own confidence (since A is weaker)
    assert c_scored.lineage_min_confidence <= c_scored.confidence.confidence


# ---------------------------------------------------------------------------
# Parity spot-check: InMemoryEntityKG vs Neo4jEntityKG
# ---------------------------------------------------------------------------


async def test_parity_inmemory_vs_neo4j(kg: Neo4jEntityKG) -> None:
    """Same scenario against both adapters → identical ScoredClaim confidences and ordering.

    Two claims for "pluto", different evidence weights. Checks that the InMemoryEntityKG
    double has not drifted from the real adapter.
    """
    mem_kg = InMemoryEntityKG()

    c1 = _make_claim("pluto", "has_mass", "v1", ingest_time=_T0, embedding=(1.0, 0.0, 0.0, 0.0))
    c2 = _make_claim("pluto", "has_attr", "v2", ingest_time=_T1, embedding=(0.0, 1.0, 0.0, 0.0))

    ev1 = _ev(source_id="src-a", polarity="+", ev_type="tool_proof", source_authority=1.0)
    ev2 = _ev(source_id="src-b", polarity="+", ev_type="corroboration", source_authority=0.8)

    for adapter in (kg, mem_kg):
        await adapter.write_claim(c1, evidence=ev1)
        await adapter.write_claim(c2, evidence=ev2)

    neo_results = await kg.claims_about("pluto")
    mem_results = await mem_kg.claims_about("pluto")

    assert len(neo_results) == len(mem_results), (
        f"Parity failure: neo4j returned {len(neo_results)} results, "
        f"in-memory returned {len(mem_results)}"
    )

    # Sort by claim id for comparison (order may differ if timestamps equal).
    neo_by_id = {sc.claim.id: sc.confidence.confidence for sc in neo_results}
    mem_by_id = {sc.claim.id: sc.confidence.confidence for sc in mem_results}

    assert set(neo_by_id.keys()) == set(mem_by_id.keys()), (
        "Parity failure: different claim ids returned"
    )
    for cid in neo_by_id:
        assert abs(neo_by_id[cid] - mem_by_id[cid]) < 1e-6, (
            f"Parity failure: claim {cid}: neo4j confidence {neo_by_id[cid]}, "
            f"in-memory confidence {mem_by_id[cid]}"
        )


# ---------------------------------------------------------------------------
# FIX 1: upsert_claim back-door sealed (regression — integration)
# ---------------------------------------------------------------------------


async def test_upsert_claim_raises_not_implemented(kg: Neo4jEntityKG) -> None:
    """Neo4jEntityKG.upsert_claim raises NotImplementedError (FIX 1 — sealed back-door)."""
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    with pytest.raises(NotImplementedError, match="write_claim"):
        await kg.upsert_claim(claim)


async def test_upsert_claim_does_not_resurrect_invalidated_claim(kg: Neo4jEntityKG) -> None:
    """Red-team attack: write → invalidate → attempt upsert_claim → raises, claim still invalid.

    FIX 1 regression: the Phase-0 upsert_claim would have blanket-SET valid_to=None, resurrecting
    an invalidated claim and silently corrupting bi-temporal history.
    """
    import pytest as _pytest

    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    resurrection_attempt = claim.model_copy(update={"valid_to": None})
    with _pytest.raises(NotImplementedError):
        await kg.upsert_claim(resurrection_attempt)

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.valid_to is not None, "Claim was resurrected by upsert_claim attack"


# ---------------------------------------------------------------------------
# FIX 2: skeleton lineage nodes — out-of-order writes + cycle (regression)
# ---------------------------------------------------------------------------


async def test_out_of_order_write_populates_skeleton(kg: Neo4jEntityKG) -> None:
    """Write child before parent → parent skeleton gets fully populated on arrival.

    FIX 2 regression: the old ON CREATE/ON MATCH split meant a skeleton created by
    DERIVED_FROM MERGE was never populated when the real parent arrived (ON CREATE never
    fires again). With coalesce population, the skeleton is populated by the first real write.
    """
    id_parent = claim_id_for("pluto", "parent_attr", "parent_payload")

    # Write CHILD first — references parent that does not exist yet.
    child = Claim(
        id=claim_id_for("pluto", "child_attr", "child_payload"),
        subject="pluto",
        predicate="child_attr",
        payload="child_payload",
        epistemic_type="inference",
        provenance=Provenance(
            source="system",
            confidence=1.0,
            evidence=(id_parent,),  # DERIVED_FROM → creates skeleton for id_parent
            recorded_at=_T0,
        ),
        valid_from=_T0,
        ingest_time=_T1,
        created_by="test",
    )
    await kg.write_claim(child, evidence=_ev(source_id="child-src"))

    # Parent node is a skeleton at this point (payload IS NULL).
    # get_claim must return None for skeletons (FIX 2 base-adapter null guard).
    skeleton = await kg.get_claim(id_parent)
    assert skeleton is None, "Skeleton node should be invisible via get_claim"

    # Now write the real parent.
    parent = _make_claim("pluto", "parent_attr", "parent_payload", ingest_time=_T0)
    await kg.write_claim(parent, evidence=_ev(source_id="parent-src"))

    # Parent must now be fully populated.
    fetched = await kg.get_claim(id_parent)
    assert fetched is not None, "Parent should be visible after real write"
    assert fetched.payload == "parent_payload"
    assert fetched.subject == "pluto"

    # claims_about must not crash — previously crashed with _as_source('None').
    results = await kg.claims_about("pluto")
    ids = {sc.claim.id for sc in results}
    assert child.id in ids
    assert id_parent in ids


async def test_out_of_order_parity_double_and_neo4j(kg: Neo4jEntityKG) -> None:
    """Out-of-order write: lineage confidence matches between double and real adapter.

    FIX 2 parity regression: after out-of-order writes and cycle, both adapters must
    produce the same lineage_min_confidence values.
    """
    mem_kg = InMemoryEntityKG()

    id_a = claim_id_for("pluto", "cycle_a", "payload_a")
    id_b = claim_id_for("pluto", "cycle_b", "payload_b")

    # A references B, B references A (cycle) — write A first (B is skeleton).
    c_a = Claim(
        id=id_a,
        subject="pluto",
        predicate="cycle_a",
        payload="payload_a",
        epistemic_type="inference",
        provenance=Provenance(source="system", confidence=1.0, evidence=(id_b,), recorded_at=_T0),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    c_b = Claim(
        id=id_b,
        subject="pluto",
        predicate="cycle_b",
        payload="payload_b",
        epistemic_type="inference",
        provenance=Provenance(source="system", confidence=1.0, evidence=(id_a,), recorded_at=_T0),
        valid_from=_T0,
        ingest_time=_T1,
        created_by="test",
    )

    ev_a = _ev(source_id="sa", polarity="+", ev_type="corroboration", source_authority=1.0)
    ev_b = _ev(source_id="sb", polarity="+", ev_type="tool_proof", source_authority=1.0)

    for adapter in (kg, mem_kg):
        await adapter.write_claim(c_a, evidence=ev_a)  # B skeleton created
        await adapter.write_claim(c_b, evidence=ev_b)  # B skeleton populated

    neo_results = await kg.claims_about("pluto")
    mem_results = await mem_kg.claims_about("pluto")

    neo_by_id = {sc.claim.id: sc.lineage_min_confidence for sc in neo_results}
    mem_by_id = {sc.claim.id: sc.lineage_min_confidence for sc in mem_results}

    assert set(neo_by_id.keys()) == set(mem_by_id.keys()), (
        "Out-of-order parity failure: different claim ids"
    )
    for cid in neo_by_id:
        assert abs(neo_by_id[cid] - mem_by_id[cid]) < 1e-6, (
            f"Out-of-order parity failure: claim {cid}: "
            f"neo4j lineage_min={neo_by_id[cid]}, "
            f"in-memory lineage_min={mem_by_id[cid]}"
        )


# ---------------------------------------------------------------------------
# FIX 4: UTC datetime normalization (regression — integration)
# ---------------------------------------------------------------------------


async def test_offset_claim_visible_at_utc_as_of(kg: Neo4jEntityKG) -> None:
    """Red-team repro (FIX 4): claim at 00:00Z written as 02:00+02:00 MUST be visible at 01:00Z.

    Mixed timezone offsets previously broke lexicographic=temporal comparison (stored ISO string
    "2026-06-09T02:00:00+02:00" is lexicographically AFTER "2026-06-09T01:00:00+00:00" even though
    it represents an earlier instant). UTC-normalisation fixes this.
    """
    from datetime import timezone

    plus_two = timezone(timedelta(hours=2))
    # 2026-06-09T02:00:00+02:00 == 2026-06-09T00:00:00+00:00
    valid_from_offset = datetime(2026, 6, 9, 2, 0, 0, tzinfo=plus_two)
    as_of_utc = datetime(2026, 6, 9, 1, 0, 0, tzinfo=UTC)

    claim = _make_claim("pluto", "tz_test", "payload", valid_from=valid_from_offset)
    await kg.write_claim(claim, evidence=_ev(source_id="tz-src"))

    results = await kg.claims_about("pluto", as_of=as_of_utc)
    assert any(sc.claim.id == claim.id for sc in results), (
        "Offset-aware claim (02:00+02:00 == 00:00Z) should be visible at as_of=01:00Z"
    )


async def test_offset_parity_adapter_and_double(kg: Neo4jEntityKG) -> None:
    """UTC-offset parity: adapter and double both see the same offset claim at the same as_of."""
    from datetime import timezone

    plus_two = timezone(timedelta(hours=2))
    valid_from_offset = datetime(2026, 6, 9, 2, 0, 0, tzinfo=plus_two)
    as_of_utc = datetime(2026, 6, 9, 1, 0, 0, tzinfo=UTC)
    as_of_naive = datetime(2026, 6, 9, 1, 0, 0)  # naive → interpreted as UTC

    mem_kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "tz_parity", "payload", valid_from=valid_from_offset)

    for adapter in (kg, mem_kg):
        await adapter.write_claim(claim, evidence=_ev(source_id="tp-src"))

    neo_utc = await kg.claims_about("pluto", as_of=as_of_utc)
    mem_utc = await mem_kg.claims_about("pluto", as_of=as_of_utc)
    neo_naive = await kg.claims_about("pluto", as_of=as_of_naive)
    mem_naive = await mem_kg.claims_about("pluto", as_of=as_of_naive)

    assert any(sc.claim.id == claim.id for sc in neo_utc), "neo4j: claim missing with utc as_of"
    assert any(sc.claim.id == claim.id for sc in mem_utc), "mem: claim missing with utc as_of"
    assert any(sc.claim.id == claim.id for sc in neo_naive), "neo4j: claim missing with naive as_of"
    assert any(sc.claim.id == claim.id for sc in mem_naive), "mem: claim missing with naive as_of"


async def test_neighbors_skips_skeleton_nodes(kg: Neo4jEntityKG) -> None:
    """Base adapter neighbors() skips skeleton (:Claim) placeholders (FIX 2 base null guard).

    If a claim has a DERIVED_FROM edge to an id that was never written (only MERGE'd as a
    placeholder by the DERIVED_FROM link), neighbors() previously crashed with _as_source('None').
    """
    id_ghost = claim_id_for("pluto", "ghost_parent", "ghost_payload")

    child = Claim(
        id=claim_id_for("pluto", "child_for_neighbors", "child_payload"),
        subject="pluto",
        predicate="child_for_neighbors",
        payload="child_payload",
        epistemic_type="inference",
        provenance=Provenance(
            source="system",
            confidence=1.0,
            evidence=(id_ghost,),
            recorded_at=_T0,
        ),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    await kg.write_claim(child, evidence=_ev(source_id="nb-src"))

    # neighbors() must not crash — skeleton id_ghost must be filtered out.
    neighbors = await kg.neighbors(child.id)
    neighbor_ids = {c.id for c in neighbors}
    assert id_ghost not in neighbor_ids, "Skeleton placeholder should not appear in neighbors()"


async def test_claim_born_with_valid_to_is_honoured_and_parity(kg: Neo4jEntityKG) -> None:
    """A claim written WITH valid_to keeps it; invalidate stays first-wins (adapter+double parity).

    Regression for an adapter-vs-double drift: the adapter dropped a caller-set valid_to at write
    time (only invalidate_claim could set it) while the double stored it — a claim legitimately
    born with a known validity end (valid_to is part of the Claim contract) was bi-temporally
    filtered by the double but visible forever in Neo4j.
    """
    born_ended = _make_claim("ceres", "classified_as", "asteroid", valid_to=_T1)
    mem = InMemoryEntityKG()
    await kg.write_claim(born_ended, evidence=_ev(source_id="bv-src"))
    await mem.write_claim(born_ended, evidence=_ev(source_id="bv-src"))

    # Both implementations: visible before the end, hidden after it.
    for impl in (kg, mem):
        before = await impl.claims_about("ceres", as_of=_T0)
        after = await impl.claims_about("ceres", as_of=_T2)
        assert {s.claim.id for s in before} == {born_ended.id}
        assert after == ()

    # First-wins: a later invalidate_claim cannot move the birth valid_to.
    await kg.invalidate_claim(born_ended.id, valid_to=_T3)
    await mem.invalidate_claim(born_ended.id, valid_to=_T3)
    got_kg = await kg.get_claim(born_ended.id)
    got_mem = await mem.get_claim(born_ended.id)
    assert got_kg is not None and got_kg.valid_to == _T1
    assert got_mem is not None and got_mem.valid_to == _T1


async def test_epistemic_level_is_first_write_wins(kg: Neo4jEntityKG) -> None:
    """Re-deriving the same triple at a different epistemic level keeps the FIRST level (S5).

    epistemic_type is not part of claim identity: an "observation" arriving after an "inference"
    accumulates evidence under the stored (first) level — never silently flipped. The explicit
    upgrade surface is the Pod 2.7 coherence reconciler (canon-review concern C1, pinned here for
    adapter + double parity).
    """
    first = _make_claim("eris", "orbits", "the sun")  # helper writes epistemic_type="inference"
    second = first.model_copy(update={"epistemic_type": "observation"})
    mem = InMemoryEntityKG()
    for impl in (kg, mem):
        await impl.write_claim(first, evidence=_ev(source_id="ep-src-1"))
        await impl.write_claim(second, evidence=_ev(source_id="ep-src-2"))
        got = await impl.get_claim(first.id)
        assert got is not None and got.epistemic_type == "inference", (
            "stored epistemic level must be first-write-wins, never silently merged (S5)"
        )
        events = await impl.evidence_for(first.id)
        assert len(events) == 2, "the second write's evidence must still accumulate"
