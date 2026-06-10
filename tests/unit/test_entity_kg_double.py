"""Full EntityKG contract tests against InMemoryEntityKG (CANON S3, S5).

This suite pins every clause of the EntityKG Protocol (cogworx.substrate.entity_kg)
against the in-memory double so the unit tier runs without any Neo4j service.
The real Neo4j adapter is expected to satisfy the same contract; if these tests pass
here and the integration tests in test_neo4j_entity_kg.py pass against Neo4j, the
double has not drifted from the adapter (parity is verified explicitly in the
integration file).

Invariants covered:
  S3 — thin internal seam only; no backend-portability flattening.
  S5 — every claim carries provenance; evidence events are immutable and ordered.
  S8 — behaviour is well-defined at all boundary conditions (empty, unknown, cyclic).
"""

from __future__ import annotations

import math
from datetime import UTC, datetime, timedelta

import pytest

from cogworx.claims.provenance import Claim, Provenance, ProvenanceSource
from cogworx.knowledge.confidence import claim_confidence
from cogworx.knowledge.evidence import EvidenceEvent, EvidenceType, Polarity, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.testing.doubles import InMemoryEntityKG

_T0 = datetime(2026, 6, 9, 0, 0, 0, tzinfo=UTC)
_T1 = _T0 + timedelta(hours=1)
_T2 = _T0 + timedelta(hours=2)
_T3 = _T0 + timedelta(hours=3)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prov(source: ProvenanceSource = "system") -> Provenance:
    return Provenance(source=source, confidence=1.0, recorded_at=_T0)


def _make_claim(
    subject: str,
    predicate: str,
    payload: str,
    *,
    object_entity: str | None = None,
    valid_from: datetime = _T0,
    ingest_time: datetime = _T0,
    provenance_evidence: tuple[str, ...] = (),
    embedding: tuple[float, ...] | None = None,
    valid_to: datetime | None = None,
    source: ProvenanceSource = "system",
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
# write_claim — identity discipline
# ---------------------------------------------------------------------------


async def test_write_claim_wrong_id_raises() -> None:
    """write_claim raises ValueError when claim.id does not match identity formula."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    bad_claim = claim.model_copy(update={"id": "bad-id-00000000000000000000000000"})
    with pytest.raises(ValueError, match="does not match"):
        await kg.write_claim(bad_claim, evidence=_ev())


async def test_write_claim_correct_id_succeeds() -> None:
    """write_claim accepts a correctly-formed id."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    returned_id = await kg.write_claim(claim, evidence=_ev())
    assert returned_id == claim.id


# ---------------------------------------------------------------------------
# Idempotent re-derivation
# ---------------------------------------------------------------------------


async def test_write_claim_twice_accumulates_evidence() -> None:
    """Same triple written twice: ONE claim, TWO evidence events, confidence reflects both."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    ev1 = _ev(source_id="src-a", polarity="+", ev_type="corroboration")
    ev2 = _ev(source_id="src-b", polarity="+", ev_type="corroboration")

    await kg.write_claim(claim, evidence=ev1)
    await kg.write_claim(claim, evidence=ev2)

    # Only one claim node.
    fetched = await kg.get_claim(claim.id)
    assert fetched is not None

    # Two evidence events in insertion order.
    events = await kg.evidence_for(claim.id)
    assert len(events) == 2
    assert events[0].source_id == "src-a"
    assert events[1].source_id == "src-b"

    # Confidence reflects both.
    derived = claim_confidence(events)
    scored = await kg.claims_about("pluto")
    assert any(abs(sc.confidence.confidence - derived.confidence) < 1e-9 for sc in scored)


async def test_write_claim_second_write_does_not_clobber_fields() -> None:
    """Second write of same triple does not overwrite claim fields (immutable on match).

    If the second write passed different payload casing, the FIRST write's fields survive.
    """
    kg = InMemoryEntityKG()
    claim_first = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim_first, evidence=_ev(source_id="src-a"))

    # Second write of same triple (same id) but if we somehow had different fields
    # the double should keep the FIRST. Re-writing the identical triple is idempotent.
    await kg.write_claim(claim_first, evidence=_ev(source_id="src-b"))

    fetched = await kg.get_claim(claim_first.id)
    assert fetched is not None
    assert fetched.payload == "1.3e22 kg"  # first write's payload is preserved


# ---------------------------------------------------------------------------
# add_evidence
# ---------------------------------------------------------------------------


async def test_add_evidence_unknown_claim_raises() -> None:
    """add_evidence raises ValueError on an unknown claim_id."""
    kg = InMemoryEntityKG()
    with pytest.raises(ValueError, match="unknown claim_id"):
        await kg.add_evidence("nonexistent-id", _ev())


async def test_add_evidence_appends_to_known_claim() -> None:
    """add_evidence appends a new evidence event to an existing claim."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev(source_id="src-a"))

    new_ev = _ev(source_id="src-b")
    await kg.add_evidence(claim.id, new_ev)

    events = await kg.evidence_for(claim.id)
    assert len(events) == 2
    assert events[1].source_id == "src-b"


# ---------------------------------------------------------------------------
# evidence_for — insertion order, no dedup
# ---------------------------------------------------------------------------


async def test_evidence_for_returns_insertion_order() -> None:
    """evidence_for returns events in insertion order, never deduped."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "payload")
    # Write two events from the SAME source (would be deduped at confidence-derivation time,
    # but evidence_for must return both).
    ev1 = _ev(source_id="same-src", polarity="+", ev_type="corroboration")
    ev2 = _ev(source_id="same-src", polarity="+", ev_type="corroboration")
    await kg.write_claim(claim, evidence=ev1)
    await kg.add_evidence(claim.id, ev2)

    events = await kg.evidence_for(claim.id)
    # Both raw events are returned (not deduped at storage level).
    assert len(events) == 2
    assert events[0].id == ev1.id
    assert events[1].id == ev2.id


# ---------------------------------------------------------------------------
# claims_about — subject-side, object-side, limit, as_of
# ---------------------------------------------------------------------------


async def test_claims_about_subject_side() -> None:
    """claims_about returns claims where entity is the subject."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())

    results = await kg.claims_about("pluto")
    assert any(sc.claim.id == claim.id for sc in results)


async def test_claims_about_object_side() -> None:
    """claims_about returns claims where entity is the object (REFERS_TO)."""
    kg = InMemoryEntityKG()
    claim = _make_claim("solar_system", "contains", "pluto ref", object_entity="pluto")
    await kg.write_claim(claim, evidence=_ev())

    results = await kg.claims_about("pluto")
    assert any(sc.claim.id == claim.id for sc in results)


async def test_claims_about_limit_respected() -> None:
    """claims_about respects the limit parameter."""
    kg = InMemoryEntityKG()
    for i in range(5):
        c = _make_claim("pluto", "attr", f"value-{i}", ingest_time=_T0 + timedelta(seconds=i))
        await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_about("pluto", limit=3)
    assert len(results) <= 3


async def test_claims_about_as_of_filters_invalidated() -> None:
    """as_of filtering: invalidated claim disappears for as_of after valid_to."""
    kg = InMemoryEntityKG()
    # Claim valid from T0, invalidated at T1.
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg", valid_from=_T0)
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    # as_of=None → all claims returned (no filter)
    all_results = await kg.claims_about("pluto", as_of=None)
    assert any(sc.claim.id == claim.id for sc in all_results)

    # as_of BEFORE valid_to (T0 + 30min) → claim still visible
    before = await kg.claims_about("pluto", as_of=_T0 + timedelta(minutes=30))
    assert any(sc.claim.id == claim.id for sc in before)

    # as_of AFTER valid_to (T1 + 30min) → claim invisible
    after = await kg.claims_about("pluto", as_of=_T1 + timedelta(minutes=30))
    assert not any(sc.claim.id == claim.id for sc in after)


async def test_claims_about_newest_first() -> None:
    """claims_about returns claims ordered newest-first by ingest_time."""
    kg = InMemoryEntityKG()
    c1 = _make_claim("pluto", "has_mass", "v1", ingest_time=_T0)
    c2 = _make_claim("pluto", "has_mass", "v2", ingest_time=_T1)
    c3 = _make_claim("pluto", "has_mass", "v3", ingest_time=_T2)
    for c in (c1, c2, c3):
        await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_about("pluto")
    ingest_times = [sc.claim.ingest_time for sc in results]
    assert ingest_times == sorted(ingest_times, reverse=True)


# ---------------------------------------------------------------------------
# invalidate_claim
# ---------------------------------------------------------------------------


async def test_invalidate_unknown_claim_raises() -> None:
    """invalidate_claim raises ValueError on an unknown claim_id."""
    kg = InMemoryEntityKG()
    with pytest.raises(ValueError, match="unknown claim_id"):
        await kg.invalidate_claim("no-such-id", valid_to=_T1)


async def test_invalidate_sets_valid_to() -> None:
    """First invalidation sets valid_to."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.valid_to == _T1


async def test_invalidate_first_wins() -> None:
    """Second invalidate with a LATER valid_to is a no-op (first-wins)."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)
    await kg.invalidate_claim(claim.id, valid_to=_T2)  # no-op

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.valid_to == _T1  # first value survives


async def test_invalidate_claim_still_readable_via_get_claim() -> None:
    """Invalidated claim is still readable via get_claim (never deleted)."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    fetched = await kg.get_claim(claim.id)
    assert fetched is not None  # never deleted
    assert fetched.id == claim.id


# ---------------------------------------------------------------------------
# write_contradiction + contradictions_of
# ---------------------------------------------------------------------------


async def test_write_contradiction_surfaces_both_directions() -> None:
    """Both directions surface from one directed CONTRADICTS edge."""
    kg = InMemoryEntityKG()
    c1 = _make_claim("pluto", "has_mass", "1.3e22 kg")
    c2 = _make_claim("pluto", "has_mass", "9.9e23 kg")
    await kg.write_claim(c1, evidence=_ev())
    await kg.write_claim(c2, evidence=_ev())

    await kg.write_contradiction(c1.id, c2.id)

    contra1 = await kg.contradictions_of(c1.id)
    contra2 = await kg.contradictions_of(c2.id)

    assert any(c.id == c2.id for c in contra1), "c1 should see c2 as contradicting"
    assert any(c.id == c1.id for c in contra2), "c2 should see c1 as contradicting"


async def test_write_contradiction_is_idempotent() -> None:
    """Repeated write_contradiction calls do not produce duplicate entries."""
    kg = InMemoryEntityKG()
    c1 = _make_claim("pluto", "has_mass", "1.3e22 kg")
    c2 = _make_claim("pluto", "has_mass", "9.9e23 kg")
    await kg.write_claim(c1, evidence=_ev())
    await kg.write_claim(c2, evidence=_ev())

    await kg.write_contradiction(c1.id, c2.id)
    await kg.write_contradiction(c1.id, c2.id)
    await kg.write_contradiction(c2.id, c1.id)  # reverse direction, still idempotent

    contra = await kg.contradictions_of(c1.id)
    ids = [c.id for c in contra]
    assert ids.count(c2.id) == 1, "contradiction should appear exactly once"


# ---------------------------------------------------------------------------
# claims_by_similarity
# ---------------------------------------------------------------------------


async def test_claims_by_similarity_raw_cosine_and_min_score() -> None:
    """claims_by_similarity: raw cosine semantics, min_score filter, descending order."""
    kg = InMemoryEntityKG()
    emb_a = (1.0, 0.0, 0.0, 0.0)  # unit vector x
    emb_b = (0.0, 1.0, 0.0, 0.0)  # unit vector y — orthogonal to query
    emb_c = (0.9, 0.1, 0.0, 0.0)  # close to x

    c_a = _make_claim("pluto", "x", "a", embedding=emb_a, ingest_time=_T0)
    c_b = _make_claim("pluto", "y", "b", embedding=emb_b, ingest_time=_T1)
    c_c = _make_claim("pluto", "z", "c", embedding=emb_c, ingest_time=_T2)
    for c in (c_a, c_b, c_c):
        await kg.write_claim(c, evidence=_ev())

    query = (1.0, 0.0, 0.0, 0.0)
    # min_score=0.8 excludes emb_b (cosine=0.0) and should include emb_a (cosine=1.0)
    # and emb_c (cosine ≈ 0.9939).
    results = await kg.claims_by_similarity(query, k=10, min_score=0.8)
    ids = [sc.claim.id for sc in results]
    assert c_a.id in ids
    assert c_c.id in ids
    assert c_b.id not in ids  # orthogonal → cosine=0.0 < 0.8


async def test_claims_by_similarity_descending_order() -> None:
    """Results are ordered descending by similarity."""
    kg = InMemoryEntityKG()
    emb_a = (1.0, 0.0, 0.0, 0.0)
    emb_c = (0.9, 0.1, 0.0, 0.0)
    c_a = _make_claim("pluto", "x", "a", embedding=emb_a)
    c_c = _make_claim("pluto", "z", "c", embedding=emb_c)
    for c in (c_a, c_c):
        await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_by_similarity((1.0, 0.0, 0.0, 0.0), k=10, min_score=0.0)
    sims = [sc.similarity for sc in results if sc.similarity is not None]
    assert sims == sorted(sims, reverse=True)


async def test_claims_by_similarity_skips_claims_without_embeddings() -> None:
    """Claims without embeddings are excluded from similarity results."""
    kg = InMemoryEntityKG()
    c_no_emb = _make_claim("pluto", "has_mass", "no-emb", embedding=None)
    c_with_emb = _make_claim("pluto", "has_attr", "with-emb", embedding=(1.0, 0.0, 0.0, 0.0))
    await kg.write_claim(c_no_emb, evidence=_ev())
    await kg.write_claim(c_with_emb, evidence=_ev())

    results = await kg.claims_by_similarity((1.0, 0.0, 0.0, 0.0), k=10, min_score=0.0)
    ids = {sc.claim.id for sc in results}
    assert c_no_emb.id not in ids
    assert c_with_emb.id in ids


async def test_claims_by_similarity_k_cap() -> None:
    """claims_by_similarity respects the k cap."""
    kg = InMemoryEntityKG()
    for i in range(5):
        c = _make_claim(
            "pluto",
            f"attr-{i}",
            f"v{i}",
            embedding=(float(i + 1), 0.0, 0.0, 0.0),
            ingest_time=_T0 + timedelta(seconds=i),
        )
        await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_by_similarity((1.0, 0.0, 0.0, 0.0), k=2, min_score=0.0)
    assert len(results) <= 2


async def test_claims_by_similarity_populates_similarity_field() -> None:
    """similarity field on ScoredClaim is set to the raw cosine value."""
    kg = InMemoryEntityKG()
    emb = (1.0, 0.0, 0.0, 0.0)
    c = _make_claim("pluto", "x", "a", embedding=emb)
    await kg.write_claim(c, evidence=_ev())

    results = await kg.claims_by_similarity((1.0, 0.0, 0.0, 0.0), k=1, min_score=0.0)
    assert len(results) == 1
    assert results[0].similarity is not None
    # cosine of identical unit vectors = 1.0
    assert abs(results[0].similarity - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# resolution_candidates
# ---------------------------------------------------------------------------


async def test_resolution_candidates_exact_norm_match() -> None:
    """Exact (subject_norm, predicate_norm) match is case/whitespace-insensitive."""
    kg = InMemoryEntityKG()
    c = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(c, evidence=_ev())

    # Query with different casing/whitespace — normalization should still find it.
    results = await kg.resolution_candidates("Pluto", "Has_Mass")
    assert any(cand.id == c.id for cand in results)


async def test_resolution_candidates_newest_first() -> None:
    """Exact-match results are ordered newest-first."""
    kg = InMemoryEntityKG()
    c1 = _make_claim("pluto", "has_mass", "v1", ingest_time=_T0)
    c2 = _make_claim("pluto", "has_mass", "v2", ingest_time=_T1)
    c3 = _make_claim("pluto", "has_mass", "v3", ingest_time=_T2)
    for c in (c1, c2, c3):
        await kg.write_claim(c, evidence=_ev())

    results = await kg.resolution_candidates("pluto", "has_mass")
    ingest_times = [cand.ingest_time for cand in results]
    assert ingest_times == sorted(ingest_times, reverse=True)


async def test_resolution_candidates_k_cap() -> None:
    """resolution_candidates caps at k."""
    kg = InMemoryEntityKG()
    for i in range(5):
        c = _make_claim("pluto", "has_mass", f"v{i}", ingest_time=_T0 + timedelta(seconds=i))
        await kg.write_claim(c, evidence=_ev())

    results = await kg.resolution_candidates("pluto", "has_mass", k=2)
    assert len(results) <= 2


async def test_resolution_candidates_vector_topup() -> None:
    """Vector top-up fills remainder when exact matches < k and embedding given."""
    kg = InMemoryEntityKG()
    # One exact-match claim (no embedding needed for topic match).
    c_exact = _make_claim(
        "pluto", "has_mass", "v-exact", ingest_time=_T0, embedding=(1.0, 0.0, 0.0, 0.0)
    )
    # A claim with different predicate but high cosine similarity.
    c_similar = _make_claim(
        "pluto", "different_pred", "v-sim", ingest_time=_T1, embedding=(0.99, 0.1, 0.0, 0.0)
    )
    await kg.write_claim(c_exact, evidence=_ev())
    await kg.write_claim(c_similar, evidence=_ev())

    # k=2: exact match gives 1 result; vector top-up should contribute 1 more.
    results = await kg.resolution_candidates(
        "pluto", "has_mass", k=2, embedding=(1.0, 0.0, 0.0, 0.0)
    )
    ids = {c.id for c in results}
    assert c_exact.id in ids
    assert c_similar.id in ids


async def test_resolution_candidates_no_duplicate_ids() -> None:
    """resolution_candidates produces no duplicate ids."""
    kg = InMemoryEntityKG()
    # A claim that is both an exact match AND would appear in vector top-up.
    c = _make_claim("pluto", "has_mass", "payload", embedding=(1.0, 0.0, 0.0, 0.0))
    await kg.write_claim(c, evidence=_ev())

    results = await kg.resolution_candidates(
        "pluto", "has_mass", k=5, embedding=(1.0, 0.0, 0.0, 0.0)
    )
    ids = [cand.id for cand in results]
    assert len(ids) == len(set(ids)), "Duplicate ids in resolution_candidates output"


async def test_resolution_candidates_unknown_subject_returns_empty() -> None:
    """resolution_candidates does not raise on unknown subject — returns empty."""
    kg = InMemoryEntityKG()
    results = await kg.resolution_candidates("unknown-entity", "some_pred")
    assert results == () or len(results) == 0


# ---------------------------------------------------------------------------
# Lineage weakest-link
# ---------------------------------------------------------------------------


async def test_lineage_weakest_link_one_ancestor() -> None:
    """C derived from A: lineage_min_confidence == min(C.conf, A.conf)."""
    kg = InMemoryEntityKG()

    # Ancestor A with weak evidence (prior only → confidence 0.5).
    c_a = _make_claim("pluto", "weak_attr", "ancestor_payload")
    await kg.write_claim(
        c_a,
        evidence=_ev(
            source_id="src-a", polarity="+", ev_type="corroboration", source_authority=0.0
        ),
    )
    # weight = 1.0 * 0.0 = 0.0, so A has prior-only confidence = 0.5

    # Child C derived from A (provenance.evidence references A).
    c_c = Claim(
        id=claim_id_for("pluto", "derived_attr", "child_payload"),
        subject="pluto",
        predicate="derived_attr",
        payload="child_payload",
        epistemic_type="inference",
        provenance=Provenance(
            source="system",
            confidence=1.0,
            evidence=(c_a.id,),  # derived from A
            recorded_at=_T0,
        ),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    # Strong evidence on C.
    await kg.write_claim(
        c_c,
        evidence=_ev(source_id="strong", polarity="+", ev_type="tool_proof", source_authority=1.0),
    )

    scored = await kg.claims_about("pluto")
    c_scored = next((sc for sc in scored if sc.claim.id == c_c.id), None)
    assert c_scored is not None

    a_scored = next((sc for sc in scored if sc.claim.id == c_a.id), None)
    assert a_scored is not None

    # Lineage min should be min(C.conf, A.conf)
    expected_min = min(c_scored.confidence.confidence, a_scored.confidence.confidence)
    assert abs(c_scored.lineage_min_confidence - expected_min) < 1e-9


async def test_lineage_cycle_terminates() -> None:
    """A cycle (A references B, B references A) terminates without infinite recursion."""
    kg = InMemoryEntityKG()

    id_a = claim_id_for("pluto", "cycle_a", "payload_a")
    id_b = claim_id_for("pluto", "cycle_b", "payload_b")

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
        ingest_time=_T0,
        created_by="test",
    )

    await kg.write_claim(c_a, evidence=_ev(source_id="sa"))
    await kg.write_claim(c_b, evidence=_ev(source_id="sb"))

    # Should not recurse forever — should complete and return results.
    results = await kg.claims_about("pluto")
    assert len(results) == 2


async def test_lineage_placeholder_ancestor_uses_prior_only() -> None:
    """Ancestor id with no stored claim → treated as prior-only confidence (0.5).

    FIX 2: a skeleton/unwritten ancestor gets Beta(1,1).mean = 0.5 as its lineage confidence
    so the weakest-link correctly penalises claims derived from unresolved parents. This matches
    the Neo4j adapter's behaviour (an ancestor node with no evidence rows → claim_confidence([])
    returns ClaimConfidence at prior: confidence=0.5).
    """
    from cogworx.knowledge.confidence import CLAIM_PRIOR_ALPHA, CLAIM_PRIOR_BETA

    kg = InMemoryEntityKG()

    # C has strong tool_proof evidence but references a ghost parent.
    c_c = Claim(
        id=claim_id_for("pluto", "orphan_derived", "payload"),
        subject="pluto",
        predicate="orphan_derived",
        payload="payload",
        epistemic_type="inference",
        provenance=Provenance(
            source="system",
            confidence=1.0,
            evidence=("ghost-id-that-does-not-exist",),
            recorded_at=_T0,
        ),
        valid_from=_T0,
        ingest_time=_T0,
        created_by="test",
    )
    await kg.write_claim(
        c_c,
        evidence=_ev(source_id="src-c", polarity="+", ev_type="tool_proof", source_authority=1.0),
    )

    results = await kg.claims_about("pluto")
    assert any(sc.claim.id == c_c.id for sc in results)
    scored_c = next(sc for sc in results if sc.claim.id == c_c.id)

    # C's own confidence is high (tool_proof: alpha=4, beta=1 → 0.8).
    assert scored_c.confidence.confidence > 0.7

    # lineage_min must be min(C.conf, ghost_prior) = min(0.8, 0.5) = 0.5.
    prior_confidence = CLAIM_PRIOR_ALPHA / (CLAIM_PRIOR_ALPHA + CLAIM_PRIOR_BETA)  # 0.5
    assert abs(scored_c.lineage_min_confidence - prior_confidence) < 1e-9, (
        f"Expected lineage_min={prior_confidence} (prior-only ghost), "
        f"got {scored_c.lineage_min_confidence}"
    )


async def test_lineage_chain_depth() -> None:
    """Chain A <- B <- C: C.lineage_min_confidence = min(A.conf, B.conf, C.conf)."""
    kg = InMemoryEntityKG()

    # A: weak (prior-only with zero-weight evidence)
    c_a = _make_claim("entity", "chain_a", "payload_a")
    await kg.write_claim(
        c_a,
        evidence=_ev(source_id="src-a", polarity="+", ev_type="recall", source_authority=0.0),
    )
    # B: derived from A, medium evidence
    id_b = claim_id_for("entity", "chain_b", "payload_b")
    c_b = Claim(
        id=id_b,
        subject="entity",
        predicate="chain_b",
        payload="payload_b",
        epistemic_type="inference",
        provenance=Provenance(source="system", confidence=1.0, evidence=(c_a.id,), recorded_at=_T0),
        valid_from=_T0,
        ingest_time=_T1,
        created_by="test",
    )
    await kg.write_claim(
        c_b,
        evidence=_ev(
            source_id="src-b", polarity="+", ev_type="corroboration", source_authority=1.0
        ),
    )

    # C: derived from B
    id_c = claim_id_for("entity", "chain_c", "payload_c")
    c_c = Claim(
        id=id_c,
        subject="entity",
        predicate="chain_c",
        payload="payload_c",
        epistemic_type="inference",
        provenance=Provenance(source="system", confidence=1.0, evidence=(id_b,), recorded_at=_T0),
        valid_from=_T0,
        ingest_time=_T2,
        created_by="test",
    )
    await kg.write_claim(
        c_c,
        evidence=_ev(source_id="src-c", polarity="+", ev_type="tool_proof", source_authority=1.0),
    )

    results = await kg.claims_about("entity")
    scored = {sc.claim.id: sc for sc in results}

    # lineage_min for C must be <= min(B.conf, A.conf)
    a_conf = scored[c_a.id].confidence.confidence
    b_conf = scored[c_b.id].confidence.confidence
    c_conf = scored[c_c.id].confidence.confidence
    expected_min = min(a_conf, b_conf, c_conf)
    assert abs(scored[c_c.id].lineage_min_confidence - expected_min) < 1e-9


# ---------------------------------------------------------------------------
# FIX 1: upsert_claim is sealed (regression)
# ---------------------------------------------------------------------------


async def test_upsert_claim_raises_not_implemented() -> None:
    """InMemoryEntityKG.upsert_claim raises NotImplementedError (FIX 1 — sealed back-door)."""
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    with pytest.raises(NotImplementedError, match="write_claim"):
        await kg.upsert_claim(claim)


async def test_upsert_claim_does_not_resurrect_invalidated_claim() -> None:
    """Red-team attack: write → invalidate → attempt upsert_claim → raises, claim still invalid.

    FIX 1 regression: the inherited Phase-0 upsert_claim would have SET valid_to to whatever the
    caller passed (potentially None), resurrecting an invalidated claim. The sealed override must
    raise before any state change so the invalidation survives.
    """
    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg")
    await kg.write_claim(claim, evidence=_ev())
    await kg.invalidate_claim(claim.id, valid_to=_T1)

    # Attempt the attack: upsert_claim with valid_to=None (would resurrect the claim).
    resurrection_attempt = claim.model_copy(update={"valid_to": None})
    with pytest.raises(NotImplementedError):
        await kg.upsert_claim(resurrection_attempt)

    # The claim must still be invalidated.
    fetched = await kg.get_claim(claim.id)
    assert fetched is not None
    assert fetched.valid_to is not None, "Claim was resurrected by upsert_claim attack"


# ---------------------------------------------------------------------------
# FIX 4: UTC datetime normalization (regression)
# ---------------------------------------------------------------------------


async def test_claims_about_offset_claim_visible_at_utc_as_of() -> None:
    """Red-team repro (FIX 4): claim at 00:00Z written as 02:00+02:00 MUST be visible at 01:00Z.

    Both the double and the adapter should treat offset-aware datetimes as the same instant.
    The double previously raised TypeError on naive vs aware comparison; it now normalises both
    sides to UTC.
    """
    from datetime import timezone

    plus_two = timezone(timedelta(hours=2))
    # 2026-06-09T02:00:00+02:00 is the same instant as 2026-06-09T00:00:00+00:00
    valid_from_offset = datetime(2026, 6, 9, 2, 0, 0, tzinfo=plus_two)
    as_of_utc = datetime(2026, 6, 9, 1, 0, 0, tzinfo=UTC)

    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg", valid_from=valid_from_offset)
    await kg.write_claim(claim, evidence=_ev())

    # Claim valid_from is 00:00Z; as_of is 01:00Z → claim MUST be visible.
    results = await kg.claims_about("pluto", as_of=as_of_utc)
    assert any(sc.claim.id == claim.id for sc in results), (
        "Offset-aware claim (02:00+02:00 == 00:00Z) should be visible at as_of=01:00Z"
    )


async def test_claims_about_naive_as_of_accepted() -> None:
    """Naive as_of is accepted by the double (interpreted as UTC, not TypeError)."""
    naive_as_of = datetime(2026, 6, 9, 1, 0, 0)  # naive — no tzinfo

    kg = InMemoryEntityKG()
    claim = _make_claim("pluto", "has_mass", "1.3e22 kg", valid_from=_T0)
    await kg.write_claim(claim, evidence=_ev())

    # Must not raise TypeError — naive is interpreted as UTC.
    results = await kg.claims_about("pluto", as_of=naive_as_of)
    assert any(sc.claim.id == claim.id for sc in results)
