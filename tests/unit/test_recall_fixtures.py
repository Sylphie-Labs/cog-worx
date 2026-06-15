"""Unit tests for the Phase 2 gate corpus fixtures (recall_fixtures.py).

asyncio_mode = "auto" (pyproject.toml) — no @pytest.mark.asyncio needed.
"""

from __future__ import annotations

from cogworx.testing.doubles import InMemoryEntityKG
from cogworx.testing.recall_fixtures import (
    basis,
    build_gate_corpus,
    build_ge4_contradiction,
    vec_toward,
)


def test_vec_toward_exact_cosine() -> None:
    """vec_toward produces a vector with EXACTLY the specified cosine to the topic axis."""
    for axis in range(8):
        for noise in range(8):
            if axis == noise:
                continue
            for cos_val in [0.40, 0.72, 0.95, 0.96, 0.97, 0.98]:
                v = vec_toward(axis, cos_val, noise)
                # e[axis] is a basis vector so dot(v, e[axis]) = v[axis]
                actual_cos = v[axis]
                assert abs(actual_cos - cos_val) < 1e-9, (
                    f"axis={axis} noise={noise} cos={cos_val}: "
                    f"expected {cos_val!r}, got {actual_cos!r}"
                )


def test_vec_toward_is_unit() -> None:
    """vec_toward produces a unit vector."""
    v = vec_toward(0, 0.95, 6)
    norm = sum(x * x for x in v) ** 0.5
    assert abs(norm - 1.0) < 1e-9, f"norm={norm!r}, expected 1.0"


def test_vec_toward_is_unit_all_combinations() -> None:
    """vec_toward is a unit vector for all axis/noise/cos combinations in the corpus."""
    for axis in range(8):
        for noise in range(8):
            if axis == noise:
                continue
            for cos_val in [0.40, 0.72, 0.95, 0.96, 0.97, 0.98]:
                v = vec_toward(axis, cos_val, noise)
                norm = sum(x * x for x in v) ** 0.5
                assert abs(norm - 1.0) < 1e-9, (
                    f"axis={axis} noise={noise} cos={cos_val}: norm={norm!r}"
                )


def test_vec_toward_has_correct_dim() -> None:
    """vec_toward returns a tuple of the correct length."""
    assert len(vec_toward(0, 0.95, 6)) == 8
    assert len(vec_toward(0, 0.95, 3, dim=4)) == 4


def test_vec_toward_noise_axis_component() -> None:
    """The noise axis component equals sqrt(1 - cos^2)."""
    import math

    cos_val = 0.95
    v = vec_toward(0, cos_val, 6)
    expected_noise = math.sqrt(1.0 - cos_val * cos_val)
    assert abs(v[6] - expected_noise) < 1e-9


def test_basis_is_unit() -> None:
    """basis(n) has v[n] == 1.0 and all others == 0.0."""
    v = basis(3)
    assert v[3] == 1.0
    assert sum(v) == 1.0


def test_basis_correct_axis() -> None:
    """basis(n) has exactly the correct hot component."""
    for axis in range(8):
        v = basis(axis)
        assert v[axis] == 1.0
        for other in range(8):
            if other != axis:
                assert v[other] == 0.0


async def test_build_gate_corpus_claim_ids() -> None:
    """All expected corpus claims are written and have valid (non-None) ids."""
    kg = InMemoryEntityKG()
    corpus = await build_gate_corpus(kg)

    for name in ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "DX"]:
        assert name in corpus.claim_ids, f"{name} missing from corpus.claim_ids"
        assert corpus.claim_ids[name] is not None, f"{name} has None claim_id"

    for i in range(1, 21):
        key = f"D{i}"
        assert key in corpus.claim_ids, f"{key} missing from corpus.claim_ids"
        assert corpus.claim_ids[key] is not None, f"{key} has None claim_id"


async def test_build_gate_corpus_claims_exist_in_kg() -> None:
    """Every corpus claim id can be retrieved from the KG."""
    kg = InMemoryEntityKG()
    corpus = await build_gate_corpus(kg)

    for name, cid in corpus.claim_ids.items():
        claim = await kg.get_claim(cid)
        assert claim is not None, f"claim {name!r} ({cid!r}) not found in KG"


async def test_build_gate_corpus_claims_have_evidence() -> None:
    """Every corpus claim has at least one evidence event."""
    kg = InMemoryEntityKG()
    corpus = await build_gate_corpus(kg)

    for name, cid in corpus.claim_ids.items():
        ev = await kg.evidence_for(cid)
        assert len(ev) >= 1, f"claim {name!r} ({cid!r}) has no evidence"


async def test_build_gate_corpus_ids_are_unique() -> None:
    """All claim ids in the corpus are unique (no two names map to the same id)."""
    kg = InMemoryEntityKG()
    corpus = await build_gate_corpus(kg)

    ids = list(corpus.claim_ids.values())
    assert len(ids) == len(set(ids)), "Duplicate claim ids in corpus"


async def test_build_ge4_contradiction_writes_two_claims() -> None:
    """build_ge4_contradiction writes LISBON and MADRID to the KG."""
    kg = InMemoryEntityKG()
    await build_gate_corpus(kg)
    ge4 = await build_ge4_contradiction(kg)

    assert "LISBON" in ge4.claim_ids
    assert "MADRID" in ge4.claim_ids
    assert ge4.claim_ids["LISBON"] is not None
    assert ge4.claim_ids["MADRID"] is not None

    lisbon = await kg.get_claim(ge4.claim_ids["LISBON"])
    madrid = await kg.get_claim(ge4.claim_ids["MADRID"])
    assert lisbon is not None
    assert madrid is not None
    assert lisbon.payload == "Lisbon"
    assert madrid.payload == "Madrid"


async def test_build_ge4_contradiction_only_two_keys() -> None:
    """build_ge4_contradiction returns a corpus with exactly LISBON and MADRID."""
    kg = InMemoryEntityKG()
    await build_gate_corpus(kg)
    ge4 = await build_ge4_contradiction(kg)
    assert set(ge4.claim_ids.keys()) == {"LISBON", "MADRID"}
