"""Unit tests for Pod 2.4 — ScopedKG, world_model, user_model (CANON S3, S5, S7).

All tests use InMemoryEntityKG + ScopeRegistry — no Neo4j, no model calls.
asyncio_mode = "auto" (pyproject.toml), so no @pytest.mark.asyncio needed.

Invariants under test:
  K1 — Scope isolation: world and user claims have different IDs; views don't cross.
  K2 — assert_fact returns a claim_id matching claim_id_for(..., scope=...).
  K3 — Stored claim carries the correct scope field.
  K4 — assert_fact is idempotent (same ID, evidence accumulates).
  K5 — add_evidence cross-scope raises ValueError("out of scope").
  K6 — invalidate cross-scope raises ValueError.
  K7 — get_claim view semantics: out-of-scope returns None, in-scope returns the Claim.
  K8 — S7 token discipline via ScopeRegistry.
  K9 — Injectable clock: valid_from and evidence.recorded_at match fixed time.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.scoped_kg import ScopedKG, user_model, world_model
from cogworx.knowledge.scopes import ScopeRegistry
from cogworx.knowledge.source_registry import SourceDeclaration, SourceRegistry
from cogworx.testing.doubles import InMemoryEntityKG

_FIXED_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_CLOCK = lambda: _FIXED_NOW  # noqa: E731


def _source_decl(
    kind: str = "tool", ref: str = "test-source", authority: float = 0.9
) -> SourceDeclaration:
    registry = SourceRegistry()
    return registry.declare(kind, ref, authority=authority)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# K1 — Scope isolation: world vs user claims are distinct
# ---------------------------------------------------------------------------


async def test_k1_world_and_user_claims_have_different_ids() -> None:
    """Same triple in world and user scope produces two different claim IDs (scope-in-identity)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-u")

    world_id = await wm.assert_fact(
        subject="Paris",
        predicate="capital_of",
        obj="France",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )
    user_id = await um.assert_fact(
        subject="Paris",
        predicate="capital_of",
        obj="France",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-u",
    )

    assert world_id != user_id


async def test_k1_wm_get_claim_returns_world_claim() -> None:
    """wm.get_claim(world_id) returns the claim; um.get_claim(world_id) returns None
    (view semantics)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-u")

    world_id = await wm.assert_fact(
        subject="Paris",
        predicate="capital_of",
        obj="France",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    assert await wm.get_claim(world_id) is not None
    assert await um.get_claim(world_id) is None


async def test_k1_claims_about_scope_filtering() -> None:
    """wm.claims_about returns only world-scope claims, not user-scope claims."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-u")

    await wm.assert_fact(
        subject="Paris",
        predicate="capital_of",
        obj="France",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )
    await um.assert_fact(
        subject="Paris",
        predicate="home_of",
        obj="alice",
        epistemic_type="inference",
        source=source,
        created_by="agent-u",
    )

    world_claims = await wm.claims_about("Paris")
    user_claims = await um.claims_about("Paris")

    # World view sees only world claims
    assert len(world_claims) == 1
    assert world_claims[0].claim.scope == "world"

    # User view sees only user claims
    assert len(user_claims) == 1
    assert user_claims[0].claim.scope == "user:alice"


# ---------------------------------------------------------------------------
# K2 — assert_fact returns claim_id matching claim_id_for(..., scope=...)
# ---------------------------------------------------------------------------


async def test_k2_world_claim_id_matches_claim_id_for() -> None:
    """World-scope claim ID equals claim_id_for(s, p, o, scope='world')."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    wm = world_model(kg, registry, owner="agent-w")

    returned_id = await wm.assert_fact(
        subject="Berlin",
        predicate="capital_of",
        obj="Germany",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    expected_id = claim_id_for("Berlin", "capital_of", "Germany", scope="world")
    assert returned_id == expected_id


async def test_k2_user_claim_id_matches_claim_id_for() -> None:
    """User-scope claim ID equals claim_id_for(s, p, o, scope='user:jim')."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    um = user_model(kg, registry, "jim", owner="agent-u")

    returned_id = await um.assert_fact(
        subject="Jim",
        predicate="lives_in",
        obj="Berlin",
        epistemic_type="inference",
        source=source,
        created_by="agent-u",
    )

    expected_id = claim_id_for("Jim", "lives_in", "Berlin", scope="user:jim")
    assert returned_id == expected_id


# ---------------------------------------------------------------------------
# K3 — Stored claim carries the correct scope field
# ---------------------------------------------------------------------------


async def test_k3_stored_claim_scope_is_world() -> None:
    """A claim written via world_model has claim.scope == 'world'."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    wm = world_model(kg, registry, owner="agent-w")

    world_id = await wm.assert_fact(
        subject="Tokyo",
        predicate="capital_of",
        obj="Japan",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    claim = await kg.get_claim(world_id)
    assert claim is not None
    assert claim.scope == "world"


async def test_k3_stored_claim_scope_is_user_id() -> None:
    """A claim written via user_model has claim.scope == 'user:<user_id>'."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    um = user_model(kg, registry, "carol", owner="agent-u")

    user_id_val = await um.assert_fact(
        subject="Carol",
        predicate="works_at",
        obj="CERN",
        epistemic_type="inference",
        source=source,
        created_by="agent-u",
    )

    claim = await kg.get_claim(user_id_val)
    assert claim is not None
    assert claim.scope == "user:carol"


# ---------------------------------------------------------------------------
# K4 — assert_fact is idempotent (same ID, evidence accumulates)
# ---------------------------------------------------------------------------


async def test_k4_assert_fact_twice_same_id() -> None:
    """Calling assert_fact twice with the same triple+scope returns the same claim ID."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    wm = world_model(kg, registry, owner="agent-w")

    id1 = await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )
    id2 = await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    assert id1 == id2


async def test_k4_assert_fact_twice_evidence_accumulates() -> None:
    """Second assert_fact with the same triple accumulates evidence (not duplicate claim)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    wm = world_model(kg, registry, owner="agent-w")

    cid = await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )
    await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    evidence = await kg.evidence_for(cid)
    # write_claim always appends exactly one evidence event per call — two calls → two events.
    assert len(evidence) == 2


async def test_k4_negative_control_no_accumulate_bug(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: monkeypatch write_claim to skip the second evidence append.

    Under the broken implementation only 1 evidence event lands, proving the == 2 assertion
    above has real teeth (it would FAIL against the patched code).
    """
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    wm = world_model(kg, registry, owner="agent-w")

    original_write_claim = InMemoryEntityKG.write_claim
    call_count: dict[str, int] = {"n": 0}

    async def _no_accumulate_write_claim(
        self: InMemoryEntityKG, claim: Claim, *, evidence: EvidenceEvent
    ) -> str:
        call_count["n"] += 1
        if call_count["n"] == 1:
            # First call: behave normally
            return await original_write_claim(self, claim, evidence=evidence)
        else:
            # Subsequent calls: merge the claim node but skip the evidence append
            # (simulates a broken idempotent-no-append bug)
            if claim.id not in self._claims:
                self._claims[claim.id] = claim
            return claim.id

    monkeypatch.setattr(InMemoryEntityKG, "write_claim", _no_accumulate_write_claim)

    cid = await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )
    await wm.assert_fact(
        subject="Rome",
        predicate="capital_of",
        obj="Italy",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    evidence = await kg.evidence_for(cid)
    # Under the mutant the second write skips the evidence append → only 1 event.
    # This proves the positive assertion (== 2) would FAIL against this implementation.
    assert len(evidence) == 1


# ---------------------------------------------------------------------------
# K5 — add_evidence cross-scope raises ValueError
# ---------------------------------------------------------------------------


async def test_k5_add_evidence_cross_scope_raises() -> None:
    """um.add_evidence on a world-scope claim raises ValueError containing 'out of scope'."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-u")

    world_id = await wm.assert_fact(
        subject="London",
        predicate="capital_of",
        obj="UK",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    evidence_event = make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source.source_id,
        source_authority=source.source_authority,
        recorded_at=_FIXED_NOW,
    )

    with pytest.raises(ValueError, match="out of scope"):
        await um.add_evidence(world_id, evidence_event)


async def test_k5_add_evidence_same_scope_succeeds() -> None:
    """um.add_evidence on a user-scope claim (same scope) succeeds."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    um = user_model(kg, registry, "alice", owner="agent-u")

    user_cid = await um.assert_fact(
        subject="Alice",
        predicate="likes",
        obj="cats",
        epistemic_type="inference",
        source=source,
        created_by="agent-u",
    )

    extra_event = make_evidence(
        type="corroboration",
        polarity="+",
        source_id=source.source_id,
        source_authority=source.source_authority,
        recorded_at=_FIXED_NOW,
    )

    # Must not raise
    await um.add_evidence(user_cid, extra_event)
    evidence = await kg.evidence_for(user_cid)
    assert len(evidence) >= 2  # original + extra


# ---------------------------------------------------------------------------
# K6 — invalidate cross-scope raises ValueError
# ---------------------------------------------------------------------------


async def test_k6_invalidate_cross_scope_raises() -> None:
    """um.invalidate on a world-scope claim raises ValueError."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-u")

    world_id = await wm.assert_fact(
        subject="Athens",
        predicate="capital_of",
        obj="Greece",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    with pytest.raises(ValueError):
        await um.invalidate(world_id)


async def test_k6_invalidate_same_scope_succeeds() -> None:
    """wm.invalidate on a world-scope claim succeeds (sets valid_to)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    # Construct ScopedKG directly so we can inject the fixed clock.
    token = registry.claim_write_token("world", owner="agent-w")
    wm = ScopedKG(kg, token, _clock=_CLOCK)

    world_id = await wm.assert_fact(
        subject="Athens",
        predicate="capital_of",
        obj="Greece",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    # Must not raise
    await wm.invalidate(world_id)

    claim = await kg.get_claim(world_id)
    assert claim is not None
    assert claim.valid_to is not None


# ---------------------------------------------------------------------------
# K7 — get_claim view semantics: out-of-scope returns None
# ---------------------------------------------------------------------------


async def test_k7_get_claim_out_of_scope_returns_none() -> None:
    """get_claim returns None for a claim that exists but belongs to a different scope."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-u")

    world_id = await wm.assert_fact(
        subject="Oslo",
        predicate="capital_of",
        obj="Norway",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    # From the user's view, the world claim is invisible — not a KeyError, just None
    result = await um.get_claim(world_id)
    assert result is None


async def test_k7_get_claim_unknown_id_returns_none() -> None:
    """get_claim returns None for an entirely unknown claim ID."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    wm = world_model(kg, registry, owner="agent-w")

    result = await wm.get_claim("deadbeef" * 4)  # 32-char hex, but unknown
    assert result is None


async def test_k7_get_claim_in_scope_returns_claim() -> None:
    """get_claim returns the Claim object for an in-scope claim."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    wm = world_model(kg, registry, owner="agent-w")

    world_id = await wm.assert_fact(
        subject="Madrid",
        predicate="capital_of",
        obj="Spain",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    claim = await wm.get_claim(world_id)
    assert claim is not None
    assert claim.id == world_id
    assert claim.subject == "Madrid"


# ---------------------------------------------------------------------------
# K8 — S7 token discipline via ScopeRegistry
# ---------------------------------------------------------------------------


def test_k8_same_owner_reclaims_token() -> None:
    """Same owner reclaiming the same scope returns a token with identical scope and owner."""
    registry = ScopeRegistry()
    t1 = registry.claim_write_token("world", owner="agent-x")
    t2 = registry.claim_write_token("world", owner="agent-x")
    assert t1.scope is t2.scope
    assert t1.owner == t2.owner


def test_k8_different_owner_raises() -> None:
    """Different owner claiming the same scope raises ValueError (S7)."""
    registry = ScopeRegistry()
    registry.claim_write_token("world", owner="agent-a")
    with pytest.raises(ValueError, match="exactly one write-token"):
        registry.claim_write_token("world", owner="agent-b")


async def test_k8_world_model_different_owner_raises() -> None:
    """world_model called twice with different owners raises ValueError (S7)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    world_model(kg, registry, owner="agent-a")
    with pytest.raises(ValueError, match="exactly one write-token"):
        world_model(kg, registry, owner="agent-b")


# ---------------------------------------------------------------------------
# K9 — Injectable clock: valid_from and evidence.recorded_at match fixed time
# ---------------------------------------------------------------------------


async def test_k9_injectable_clock_valid_from() -> None:
    """assert_fact with a fixed clock produces claim.valid_from == fixed_time."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    # Construct ScopedKG directly to inject the fixed clock.
    token = registry.claim_write_token("world", owner="agent-w")
    wm = ScopedKG(kg, token, _clock=_CLOCK)

    cid = await wm.assert_fact(
        subject="Vienna",
        predicate="capital_of",
        obj="Austria",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    claim = await kg.get_claim(cid)
    assert claim is not None
    assert claim.valid_from == _FIXED_NOW


async def test_k9_injectable_clock_evidence_recorded_at() -> None:
    """The EvidenceEvent recorded_at matches the fixed clock."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source_decl()
    # Construct ScopedKG directly to inject the fixed clock.
    token = registry.claim_write_token("world", owner="agent-w")
    wm = ScopedKG(kg, token, _clock=_CLOCK)

    cid = await wm.assert_fact(
        subject="Vienna",
        predicate="capital_of",
        obj="Austria",
        epistemic_type="confirmed",
        source=source,
        created_by="agent-w",
    )

    evidence = await kg.evidence_for(cid)
    assert len(evidence) >= 1
    assert evidence[0].recorded_at == _FIXED_NOW
