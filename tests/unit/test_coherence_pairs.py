"""Unit tests for cogworx.coherence.pairs — pure, no I/O."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from cogworx.claims.provenance import Claim, Provenance
from cogworx.coherence.pairs import candidate_pairs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_NOW = datetime(2026, 1, 1, tzinfo=UTC)
_PAST = datetime(2025, 1, 1, tzinfo=UTC)
_FUTURE = datetime(2027, 1, 1, tzinfo=UTC)

_PROV = Provenance(
    source="tool",
    confidence=0.9,
    recorded_at=_NOW,
)

_EMB_A: tuple[float, ...] = (1.0, 0.0, 0.0)
_EMB_B: tuple[float, ...] = (0.8, 0.6, 0.0)  # cosine with A ≈ 0.8
_EMB_ORTHO: tuple[float, ...] = (0.0, 1.0, 0.0)  # cosine with A = 0.0


def _claim(
    cid: str,
    *,
    predicate: str = "pred:foo",
    payload: str = "value-a",
    valid_from: datetime = _PAST,
    valid_to: datetime | None = None,
    embedding: tuple[float, ...] | None = None,
    status: str = "active",
) -> Claim:
    return Claim(
        id=cid,
        subject="subject:x",
        predicate=predicate,
        payload=payload,
        epistemic_type="inference",
        provenance=_PROV,
        valid_from=valid_from,
        valid_to=valid_to,
        ingest_time=_NOW,
        created_by="test",
        embedding=embedding,
        status=status,
    )


def _adj_id(id_a: str, id_b: str) -> str:
    key = "\x1f".join(sorted([id_a, id_b]))
    return "adj:" + hashlib.sha256(key.encode()).hexdigest()[:32]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_disjoint_window_drop() -> None:
    """Claims whose validity windows don't overlap produce no candidates."""
    # a: [2025-01, 2025-06], b: [2026-01, open) — no overlap
    a = _claim(
        "a1",
        valid_from=datetime(2025, 1, 1, tzinfo=UTC),
        valid_to=datetime(2025, 6, 1, tzinfo=UTC),
    )
    b = _claim(
        "b1",
        valid_from=datetime(2026, 1, 1, tzinfo=UTC),
    )
    report = candidate_pairs([a, b])
    assert len(report.pairs) == 0
    assert report.dropped_disjoint_window == 1


def test_same_predicate_always_candidate() -> None:
    """Same-predicate claims (no embedding) always produce a candidate pair."""
    a = _claim("a2", predicate="pred:age", payload="30")
    b = _claim("b2", predicate="pred:age", payload="31")
    report = candidate_pairs([a, b])
    assert len(report.pairs) == 1
    pair = report.pairs[0]
    assert pair.reason == "same-predicate"
    assert pair.cosine is None
    assert {pair.a_id, pair.b_id} == {"a2", "b2"}


def test_cross_predicate_cosine_gate_pass() -> None:
    """Cross-predicate pair with cosine above threshold is admitted."""
    # _EMB_A · _EMB_B cosine ≈ 0.8, threshold=0.5
    a = _claim("a3", predicate="pred:height", embedding=_EMB_A)
    b = _claim("b3", predicate="pred:tallness", embedding=_EMB_B)
    report = candidate_pairs([a, b], cosine_threshold=0.5)
    assert len(report.pairs) == 1
    pair = report.pairs[0]
    assert pair.reason == "embedding"
    assert pair.cosine is not None
    assert pair.cosine > 0.5


def test_cross_predicate_cosine_gate_fail() -> None:
    """Cross-predicate pair with cosine below threshold is dropped."""
    # _EMB_A · _EMB_B cosine ≈ 0.8, threshold=0.9 → rejected
    a = _claim("a4", predicate="pred:height", embedding=_EMB_A)
    b = _claim("b4", predicate="pred:tallness", embedding=_EMB_B)
    report = candidate_pairs([a, b], cosine_threshold=0.9)
    assert len(report.pairs) == 0
    assert report.dropped_no_embedding == 0  # embeddings present; not a no-emb drop


def test_cross_predicate_no_embedding_counted() -> None:
    """Cross-predicate pair where one claim lacks an embedding is dropped and counted."""
    a = _claim("a5", predicate="pred:height", embedding=_EMB_A)
    b = _claim("b5", predicate="pred:weight")  # no embedding
    report = candidate_pairs([a, b])
    assert len(report.pairs) == 0
    assert report.dropped_no_embedding == 1


def test_no_good_cache_skips() -> None:
    """A pair whose adjudication id is in skip_adjudication_ids is dropped."""
    a = _claim("a6", predicate="pred:age", payload="30")
    b = _claim("b6", predicate="pred:age", payload="31")
    adj = _adj_id("a6", "b6")
    report = candidate_pairs([a, b], skip_adjudication_ids=frozenset({adj}))
    assert len(report.pairs) == 0
    assert report.dropped_cached == 1


def test_identical_payload_same_predicate_drop() -> None:
    """Re-derivation: same predicate + same payload → not a conflict."""
    a = _claim("a7", predicate="pred:age", payload="30")
    b = _claim("b7", predicate="pred:age", payload="30")
    report = candidate_pairs([a, b])
    assert len(report.pairs) == 0
    assert report.dropped_disjoint_window == 0
    assert report.dropped_cached == 0
    assert report.dropped_no_embedding == 0


def test_determinism() -> None:
    """Calling candidate_pairs twice with the same input yields identical output."""
    claims = [
        _claim("d1", predicate="pred:age", payload="30"),
        _claim("d2", predicate="pred:age", payload="31"),
        _claim("d3", predicate="pred:height", embedding=_EMB_A),
        _claim("d4", predicate="pred:tallness", embedding=_EMB_B),
    ]
    r1 = candidate_pairs(claims, cosine_threshold=0.5)
    r2 = candidate_pairs(claims, cosine_threshold=0.5)
    assert r1 == r2


def test_single_claim_no_pairs() -> None:
    """A single-element input always produces an empty report."""
    report = candidate_pairs([_claim("solo")])
    assert len(report.pairs) == 0
    assert report.candidate_claim_ids == ()
    assert report.dropped_disjoint_window == 0
    assert report.dropped_no_embedding == 0
    assert report.dropped_cached == 0


def test_candidate_claim_ids_is_sorted_union() -> None:
    """candidate_claim_ids is the sorted union of all ids that appear in any pair."""
    a = _claim("z1", predicate="pred:age", payload="30")
    b = _claim("m2", predicate="pred:age", payload="31")
    c = _claim("a3", predicate="pred:age", payload="32")
    report = candidate_pairs([a, b, c])
    # Three same-predicate/different-payload pairs → all three ids should appear
    assert set(report.candidate_claim_ids) == {"z1", "m2", "a3"}
    assert list(report.candidate_claim_ids) == sorted(["z1", "m2", "a3"])


def test_inactive_claims_excluded() -> None:
    """Claims with status != 'active' are excluded from pair generation."""
    a = _claim("e1", predicate="pred:age", payload="30", status="active")
    b = _claim("e2", predicate="pred:age", payload="31", status="defeasibly-defeated")
    report = candidate_pairs([a, b])
    assert len(report.pairs) == 0


def test_none_predicate_both_falls_to_no_same_predicate() -> None:
    """Two claims with predicate=None must NOT trigger same-predicate logic.

    predicate=None means no structural predicate is present.  Without a predicate the claims
    cannot share one, so there is no same-predicate candidate.  Cross-predicate via embedding
    also falls out: both predicates are None, so neither has an embedding that differs by
    predicate.  The pair is dropped and counted as dropped_no_embedding (no embedding → no
    cross-predicate candidate path).
    """
    # Predicate=None, same subject, overlapping validity — no embedding.
    a = Claim(
        id="np1",
        subject="subject:x",
        predicate=None,
        payload="value-a",
        epistemic_type="inference",
        provenance=_PROV,
        valid_from=_PAST,
        valid_to=None,
        ingest_time=_NOW,
        created_by="test",
        embedding=None,
    )
    b = Claim(
        id="np2",
        subject="subject:x",
        predicate=None,
        payload="value-b",
        epistemic_type="inference",
        provenance=_PROV,
        valid_from=_PAST,
        valid_to=None,
        ingest_time=_NOW,
        created_by="test",
        embedding=None,
    )
    report = candidate_pairs([a, b])
    assert len(report.pairs) == 0, (
        f"predicate=None pair should not be a candidate, got pairs={report.pairs!r}"
    )
    assert report.dropped_no_embedding >= 1, (
        "predicate=None cross-predicate pair must be counted as dropped_no_embedding"
    )
