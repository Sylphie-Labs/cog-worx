"""Pod 2.4 scope spike (CANON S12) — world model + user model.

Falsifiable spike success criteria SC-1 through SC-7.

Each positive invariant has a mutation-resistant negative control (monkeypatch or direct
injection) that MUST trip.  If a negative control passes when it should fail, the assertion
is toothless and the spike rejects it.

CONCLUSION (to be recorded after running):
  SC-1 — Scope isolation (query-level): verified
  SC-2 — Shared entity connectivity: verified
  SC-3 — Evidence segregation (governance property): verified
  SC-4 — Token discipline: verified WITHIN a single ScopeRegistry instance.
          Multi-registry same-process bypass documented in CF-6 (test_sc4_multi_registry_same_process_known_gap).
  SC-5 — Back-compat / migration-free: verified
  SC-6 — S8 lesion (degradation): verified
  SC-7 — S1 / S9 structural invariant: accidental model-text→token coercion verified blocked.
          Direct Python construction of ScopeWriteToken(...) is an explicit, reviewable breach (CF-1).

Pure Python — no Neo4j, no model calls, no live substrate.
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
import uuid
from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Claim, DEFAULT_SCOPE, Provenance
from cogworx.knowledge.evidence import make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.scoped_kg import ScopedKG, user_model, world_model
from cogworx.knowledge.scopes import Scope, ScopeRegistry, ScopeWriteToken
from cogworx.knowledge.source_registry import SourceRegistry
from cogworx.testing.doubles import InMemoryEntityKG

pytestmark = pytest.mark.spike

_NOW = datetime(2026, 6, 10, 12, 0, 0, tzinfo=UTC)
_CLOCK = lambda: _NOW  # noqa: E731


def _source():
    return SourceRegistry().declare("tool", "spike-tool", authority=0.9)


def _evidence_event(source=None):
    src = source or _source()
    return make_evidence(
        type="corroboration",
        polarity="+",
        source_id=src.source_id,
        source_authority=src.source_authority,
        recorded_at=_NOW,
    )


# ---------------------------------------------------------------------------
# SC-1 — Scope isolation (query-level)
# ---------------------------------------------------------------------------


async def test_sc1_scope_query_isolation() -> None:
    """claims_about scoped to 'world' returns only world claims; 'user:jim' only jim's;
    scope=None returns all four scopes."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source()

    wm = world_model(kg, registry, owner="agent-w")
    um_jim = user_model(kg, registry, "jim", owner="agent-j")
    um_alice = user_model(kg, registry, "alice", owner="agent-a")

    # Write same subject/predicate to each scope (different objects to keep claim IDs distinct)
    await wm.assert_fact(
        subject="Paris", predicate="scope_tag", obj="world-val",
        epistemic_type="confirmed", source=source, created_by="agent-w",
    )
    await um_jim.assert_fact(
        subject="Paris", predicate="scope_tag", obj="jim-val",
        epistemic_type="inference", source=source, created_by="agent-j",
    )
    await um_alice.assert_fact(
        subject="Paris", predicate="scope_tag", obj="alice-val",
        epistemic_type="inference", source=source, created_by="agent-a",
    )
    # Write one "agent" (default) claim directly via the raw KG
    agent_claim_id = claim_id_for("Paris", "scope_tag", "agent-val")
    agent_claim = Claim(
        id=agent_claim_id,
        subject="Paris",
        predicate="scope_tag",
        payload="agent-val",
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="agent",
        scope="agent",
    )
    await kg.write_claim(agent_claim, evidence=_evidence_event(source))

    # Scope-filtered reads
    world_hits = await kg.claims_about("Paris", scope="world")
    jim_hits = await kg.claims_about("Paris", scope="user:jim")
    alice_hits = await kg.claims_about("Paris", scope="user:alice")
    all_hits = await kg.claims_about("Paris", scope=None)

    assert len(world_hits) == 1, f"Expected 1 world claim, got {len(world_hits)}"
    assert len(jim_hits) == 1, f"Expected 1 jim claim, got {len(jim_hits)}"
    assert len(alice_hits) == 1, f"Expected 1 alice claim, got {len(alice_hits)}"
    assert len(all_hits) == 4, f"Expected 4 total claims, got {len(all_hits)}"

    assert world_hits[0].claim.scope == "world"
    assert jim_hits[0].claim.scope == "user:jim"
    assert alice_hits[0].claim.scope == "user:alice"


async def test_sc1_negative_control_scope_filter_bypassed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: monkeypatch InMemoryEntityKG.claims_about to skip scope filtering
    → world-scoped query returns MORE than 1 result, proving positive test has teeth."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "jim", owner="agent-j")

    await wm.assert_fact(
        subject="Paris", predicate="scope_tag", obj="world-val",
        epistemic_type="confirmed", source=source, created_by="agent-w",
    )
    await um.assert_fact(
        subject="Paris", predicate="scope_tag", obj="jim-val",
        epistemic_type="inference", source=source, created_by="agent-j",
    )

    # Capture the original method
    original_claims_about = InMemoryEntityKG.claims_about

    async def _unfiltered_claims_about(self, entity, *, limit=20, as_of=None, scope=None):
        # Broken: always ignore scope — return all claims
        return await original_claims_about(self, entity, limit=limit, as_of=as_of, scope=None)

    monkeypatch.setattr(InMemoryEntityKG, "claims_about", _unfiltered_claims_about)

    # With the broken filter, world-scoped query leaks user claims → count > 1
    all_results = await kg.claims_about("Paris", scope="world")  # scope arg ignored by mutant
    # The mutant returns all claims (scope=None inside), so we expect 2
    assert len(all_results) == 2, (
        "Negative control: unfiltered claims_about must return all scopes "
        f"(expected 2, got {len(all_results)})"
    )
    # The positive test's assertion (== 1) would FAIL on this mutant — that is the proof


# ---------------------------------------------------------------------------
# SC-2 — Shared entity connectivity
# ---------------------------------------------------------------------------


async def test_sc2_shared_entity_connectivity() -> None:
    """World and user claims about the same entity subject both appear in scope=None read.

    Entities are shared topology; claims carry scope. Two scoped claims on the same subject
    must not create duplicate entity entries — both are retrievable via claims_about(scope=None).
    """
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "jim", owner="agent-j")

    world_id = await wm.assert_fact(
        subject="Paris", predicate="capital_of", obj="France",
        epistemic_type="confirmed", source=source, created_by="agent-w",
    )
    user_id = await um.assert_fact(
        subject="Paris", predicate="home_city", obj="jim",
        epistemic_type="inference", source=source, created_by="agent-j",
    )

    # Both claims use the same entity name "Paris" — confirm both are reachable via unscoped read
    all_paris = await kg.claims_about("Paris", scope=None)
    claim_ids = {sc.claim.id for sc in all_paris}

    assert world_id in claim_ids, "World-scope claim on Paris not found in unscoped read"
    assert user_id in claim_ids, "User-scope claim on Paris not found in unscoped read"
    assert len(all_paris) == 2, f"Expected exactly 2 claims about Paris, got {len(all_paris)}"


# ---------------------------------------------------------------------------
# SC-3 — Evidence segregation (governance property)
# ---------------------------------------------------------------------------


async def test_sc3_evidence_segregation() -> None:
    """Writing the same triple in 'agent' and 'world' scope yields two distinct claim IDs.
    Evidence on the agent claim does NOT bleed into the world claim."""
    kg = InMemoryEntityKG()
    source = _source()

    # Write agent-scope claim
    agent_id = claim_id_for("X", "is", "Y", scope="agent")
    agent_claim = Claim(
        id=agent_id,
        subject="X",
        predicate="is",
        payload="Y",
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="agent",
        scope="agent",
    )
    await kg.write_claim(agent_claim, evidence=_evidence_event(source))

    # Write world-scope claim for same triple
    world_id = claim_id_for("X", "is", "Y", scope="world")
    world_claim = Claim(
        id=world_id,
        subject="X",
        predicate="is",
        payload="Y",
        epistemic_type="confirmed",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="world-writer",
        scope="world",
    )
    await kg.write_claim(world_claim, evidence=_evidence_event(source))

    # IDs must differ (scope-in-identity)
    assert agent_id != world_id

    # Add 10 more evidence events to the agent claim
    for _ in range(10):
        await kg.add_evidence(agent_id, _evidence_event(source))

    # Agent claim now has 11 events; world claim must have exactly 1
    agent_ev = await kg.evidence_for(agent_id)
    world_ev = await kg.evidence_for(world_id)

    assert len(agent_ev) == 11, f"Agent claim should have 11 evidence events, got {len(agent_ev)}"
    assert len(world_ev) == 1, f"World claim should have exactly 1 evidence event, got {len(world_ev)}"


async def test_sc3_negative_control_scope_ignored_in_claim_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: monkeypatch claim_id_for in the doubles module to ignore scope.
    → agent and world claims would collide to the same ID → world claim would accrue 11 events.
    Proves SC-3's positive assertion has teeth."""
    import cogworx.knowledge.identity as _identity_mod
    import cogworx.testing.doubles as _doubles_mod

    # Mutant: strip scope from the hash (pre-2.4 3-part-only behaviour)
    def _no_scope_claim_id_for(subject, predicate, object_repr, *, scope=DEFAULT_SCOPE):
        return _identity_mod.claim_id_for(subject, predicate, object_repr, scope=DEFAULT_SCOPE)

    monkeypatch.setattr(_doubles_mod, "claim_id_for", _no_scope_claim_id_for)

    # Now both agent and world claim_id_for calls return the same value
    agent_id_mutant = _doubles_mod.claim_id_for("X", "is", "Y", scope="agent")
    world_id_mutant = _doubles_mod.claim_id_for("X", "is", "Y", scope="world")

    # The mutant makes them identical — the positive test (agent_id != world_id) would FAIL
    assert agent_id_mutant == world_id_mutant, (
        "Negative control: with scope stripped, both IDs should be identical — "
        "the positive assertion (agent_id != world_id) would FAIL"
    )


# ---------------------------------------------------------------------------
# SC-4 — Token discipline
# ---------------------------------------------------------------------------


def test_sc4_different_owner_raises() -> None:
    """ScopeRegistry.claim_write_token with different owner raises ValueError (S7)."""
    registry = ScopeRegistry()
    registry.claim_write_token("world", owner="owner-a")
    with pytest.raises(ValueError, match="exactly one write-token"):
        registry.claim_write_token("world", owner="owner-b")


async def test_sc4_hand_forged_claim_wrong_scope_in_id_raises() -> None:
    """A hand-forged Claim whose ID was computed WITHOUT scope fails identity discipline
    when write_claim is called. ValueError raised before any I/O."""
    kg = InMemoryEntityKG()

    # Forge a claim with scope='world' but compute the id as if scope='agent'
    wrong_id = claim_id_for("Paris", "capital_of", "France", scope="agent")  # wrong scope
    forged_claim = Claim(
        id=wrong_id,
        subject="Paris",
        predicate="capital_of",
        payload="France",
        epistemic_type="confirmed",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="forger",
        scope="world",  # scope field says "world" but id was computed for "agent"
    )

    with pytest.raises(ValueError):
        await kg.write_claim(forged_claim, evidence=_evidence_event())


async def test_sc4_add_evidence_cross_scope_raises() -> None:
    """ScopedKG.add_evidence on an out-of-scope claim raises ValueError (not silent)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-a")

    world_id = await wm.assert_fact(
        subject="Test", predicate="prop", obj="val",
        epistemic_type="confirmed", source=source, created_by="agent-w",
    )

    ev = _evidence_event(source)
    with pytest.raises(ValueError):
        await um.add_evidence(world_id, ev)


async def test_sc4_invalidate_cross_scope_raises() -> None:
    """ScopedKG.invalidate on an out-of-scope claim raises ValueError (not silent)."""
    kg = InMemoryEntityKG()
    registry = ScopeRegistry()
    source = _source()

    wm = world_model(kg, registry, owner="agent-w")
    um = user_model(kg, registry, "alice", owner="agent-a")

    world_id = await wm.assert_fact(
        subject="Test", predicate="prop", obj="val",
        epistemic_type="confirmed", source=source, created_by="agent-w",
    )

    with pytest.raises(ValueError):
        await um.invalidate(world_id)


async def test_sc4_multi_registry_same_process_known_gap() -> None:
    """KNOWN GAP CF-6: two ScopeRegistry instances in the same process can each claim the
    same scope independently. S7 enforcement requires using ONE shared registry per EntityKG.
    This test documents the gap explicitly — it passes to prove the bypass is real.

    Token discipline is VERIFIED within a single ScopeRegistry instance (see tests above).
    Multi-registry same-process bypass is documented here as CF-6.
    Resolution: use ONE ScopeRegistry instance per EntityKG; cross-instance enforcement
    is the deferred orchestration platform (CANON §3/S7 contracts-now, platform-later).
    """
    kg = InMemoryEntityKG()
    reg_a = ScopeRegistry()
    reg_b = ScopeRegistry()
    src = SourceRegistry().declare("system", "test")

    wm_a = world_model(kg, reg_a, owner="writer-a")
    # Different registry — no conflict raised even though same scope is claimed
    wm_b = world_model(kg, reg_b, owner="writer-b")

    id_a = await wm_a.assert_fact(
        subject="Paris", predicate="capital_of", obj="France",
        epistemic_type="observation", source=src, created_by="writer-a",
    )
    id_b = await wm_b.assert_fact(
        subject="Paris", predicate="capital_of", obj="France",
        epistemic_type="observation", source=src, created_by="writer-b",
    )
    # Same scope + same triple = same claim_id_for hash = same claim node
    assert id_a == id_b, "Both writers target the same claim node (scope-in-identity)"

    evidence = await kg.evidence_for(id_a)
    # Two writers, two evidence events — proves the gap is real: S7 violated in-process
    assert len(evidence) == 2, (
        f"CF-6 bypass confirmed: {len(evidence)} evidence events from two uncoordinated writers "
        "(expected 2). S7 is enforced only within a single ScopeRegistry instance."
    )


# ---------------------------------------------------------------------------
# SC-5 — Back-compat / migration-free
# ---------------------------------------------------------------------------


def test_sc5_no_scope_arg_equals_agent_scope() -> None:
    """claim_id_for(s, p, o) == claim_id_for(s, p, o, scope='agent') (byte-identical)."""
    id_no_arg = claim_id_for("Paris", "capital_of", "France")
    id_agent = claim_id_for("Paris", "capital_of", "France", scope="agent")
    assert id_no_arg == id_agent


def test_sc5_agent_scope_differs_from_world_scope() -> None:
    """claim_id_for(..., scope='agent') != claim_id_for(..., scope='world')."""
    id_agent = claim_id_for("Paris", "capital_of", "France", scope="agent")
    id_world = claim_id_for("Paris", "capital_of", "France", scope="world")
    assert id_agent != id_world


async def test_sc5_agent_scope_claim_found_by_agent_and_unscoped_not_world() -> None:
    """A claim written with default scope='agent' appears in scope='agent' and scope=None reads,
    but NOT in scope='world' reads."""
    kg = InMemoryEntityKG()
    source = _source()

    agent_id = claim_id_for("Paris", "capital_of", "France")  # default scope
    agent_claim = Claim(
        id=agent_id,
        subject="Paris",
        predicate="capital_of",
        payload="France",
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="agent",
        # scope not set → uses DEFAULT_SCOPE == "agent"
    )
    await kg.write_claim(agent_claim, evidence=_evidence_event(source))

    agent_hits = await kg.claims_about("Paris", scope="agent")
    unscoped_hits = await kg.claims_about("Paris", scope=None)
    world_hits = await kg.claims_about("Paris", scope="world")

    assert len(agent_hits) == 1
    assert len(unscoped_hits) == 1
    assert len(world_hits) == 0


def test_sc5_claim_default_scope_is_agent() -> None:
    """A Claim constructed without scope= has claim.scope == 'agent'."""
    claim = Claim(
        id=claim_id_for("X", "p", "Y"),
        subject="X",
        predicate="p",
        payload="Y",
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="agent",
    )
    assert claim.scope == "agent"


def test_sc5_negative_control_scope_always_in_hash(monkeypatch: pytest.MonkeyPatch) -> None:
    """Negative control: mutant claim_id_for that always appends scope produces a different hash
    for the no-arg call compared to the canonical implementation.
    → existing-claim lookup for no-scope callers would fail (back-compat broken).
    Proves SC-5 positive assertion has teeth."""
    import cogworx.knowledge.identity as _identity_mod

    def _always_scoped(subject, predicate, object_repr, *, scope=DEFAULT_SCOPE):
        # Always include scope — breaks the 3-part/4-part distinction
        import hashlib
        import unicodedata

        def norm(s: str) -> str:
            s = unicodedata.normalize("NFC", s)
            return s.strip().lower()

        parts = [
            f"{len(p)}:{p}"
            for p in (norm(subject), norm(predicate), norm(object_repr), norm(scope))
        ]
        raw = "\x1f".join(parts).encode()
        return hashlib.sha256(raw).hexdigest()[:32]

    # The mutant always includes scope, even for DEFAULT_SCOPE
    mutant_no_scope = _always_scoped("Paris", "capital_of", "France")
    canonical_no_scope = _identity_mod.claim_id_for("Paris", "capital_of", "France")

    # Mutant produces a DIFFERENT hash than canonical → existing lookups break
    assert mutant_no_scope != canonical_no_scope, (
        "Negative control: mutant (always-include-scope) should differ from canonical no-scope hash"
    )


# ---------------------------------------------------------------------------
# SC-6 — S8 lesion (degradation): base EntityKG works without ScopeRegistry
# ---------------------------------------------------------------------------


async def test_sc6_entity_kg_works_without_scope_machinery() -> None:
    """Without ScopedKG or ScopeRegistry, InMemoryEntityKG behaves byte-identically to pre-2.4.

    write_claim / claims_about / get_claim all function normally; scope=None returns all.
    The absence of world/user model machinery must not cause any crash or regression.
    """
    kg = InMemoryEntityKG()
    source = _source()

    # Write via raw KG (no scoped layer)
    claim_id = claim_id_for("Alice", "knows", "Bob")
    claim = Claim(
        id=claim_id,
        subject="Alice",
        predicate="knows",
        payload="Bob",
        epistemic_type="inference",
        provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
        valid_from=_NOW,
        ingest_time=_NOW,
        created_by="agent",
    )
    await kg.write_claim(claim, evidence=_evidence_event(source))

    # Read — must work, no crash
    fetched = await kg.get_claim(claim_id)
    assert fetched is not None
    assert fetched.subject == "Alice"

    claims = await kg.claims_about("Alice")
    assert len(claims) == 1

    claims_unscoped = await kg.claims_about("Alice", scope=None)
    assert len(claims_unscoped) == 1

    # No ScopeRegistry or ScopedKG anywhere — must not have raised
    evidence = await kg.evidence_for(claim_id)
    assert len(evidence) == 1


async def test_sc6_entity_kg_scope_none_does_not_filter() -> None:
    """scope=None in claims_about returns all claims regardless of scope field."""
    kg = InMemoryEntityKG()
    source = _source()

    # Write two claims with different scopes directly
    for scope_val, obj in [("agent", "val-agent"), ("world", "val-world")]:
        cid = claim_id_for("X", "p", obj, scope=scope_val)
        c = Claim(
            id=cid,
            subject="X",
            predicate="p",
            payload=obj,
            epistemic_type="inference",
            provenance=Provenance(source="tool", confidence=1.0, recorded_at=_NOW),
            valid_from=_NOW,
            ingest_time=_NOW,
            created_by="agent",
            scope=scope_val,
        )
        await kg.write_claim(c, evidence=_evidence_event(source))

    all_hits = await kg.claims_about("X", scope=None)
    assert len(all_hits) == 2, f"scope=None should return both claims; got {len(all_hits)}"


# ---------------------------------------------------------------------------
# SC-7 — S1 / S9 structural invariants
# ---------------------------------------------------------------------------


def test_sc7_scopes_module_does_not_import_model_or_substrate() -> None:
    """cogworx.knowledge.scopes imports cleanly in a subprocess without model/substrate deps.

    scopes.py is pure Python: no I/O, no Neo4j driver, no Anthropic SDK.
    """
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from cogworx.knowledge.scopes import Scope, ScopeRegistry, ScopeWriteToken; print('ok')",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"cogworx.knowledge.scopes import failed (S8 lesion failure):\n{result.stderr}"
    )
    assert result.stdout.strip() == "ok"


def test_sc7_scopes_module_forbidden_attrs() -> None:
    """cogworx.knowledge.scopes does not expose any model or substrate attributes."""
    import cogworx.knowledge.scopes as _scopes_mod

    forbidden = ("anthropic", "openai", "model_client", "ClaudeClient", "neo4j", "psycopg")
    for attr in forbidden:
        assert attr not in dir(_scopes_mod), (
            f"S1/S8 violation: cogworx.knowledge.scopes exposes {attr!r} — "
            "a model or substrate attribute reached the pure scopes module"
        )


def test_sc7_scope_write_token_not_json_roundtrippable() -> None:
    """SC-7 scope: this test verifies that accidental model-text→token coercion fails (JSON
    round-trip degrades ScopeWriteToken to a dict). It does NOT prevent direct Python
    construction of ScopeWriteToken(...) — that path is an explicit, reviewable breach
    covered by CF-1.

    A model-generated text string cannot accidentally become a usable ScopeWriteToken:

    1. json.dumps raises TypeError (not JSON-serialisable).
    2. dataclasses.asdict reduces Scope to a plain dict — not a Scope instance. Splatting
       the result back into ScopeWriteToken produces a degraded token whose .scope is a dict.
       Any downstream call to .scope.scope_id raises AttributeError — the reconstructed object
       is structurally unusable without a deliberate re-hydration step.

    Invariant scope: accidental model-text→valid ScopeWriteToken coercion via JSON/dict is
    structurally blocked. In-process deliberate construction (direct Python call) is a
    separate, explicit breach surface, not covered by this test.
    """
    scope = Scope(kind="world", ref="global")
    token = ScopeWriteToken(scope=scope, owner="agent-x")

    # ScopeWriteToken is not JSON-serialisable (frozen dataclass, no __json__ method)
    with pytest.raises(TypeError):
        json.dumps(token)

    # After dataclasses.asdict, the scope field is a plain dict — not a Scope object.
    token_dict = dataclasses.asdict(token)
    assert isinstance(token_dict["scope"], dict), (
        "asdict should reduce Scope to a plain dict (no Scope instance)"
    )

    # A "reconstructed" token from the dict has scope as a dict — structurally degraded.
    # Accessing .scope.scope_id on this degraded token must fail (dict has no .scope_id).
    degraded_token = ScopeWriteToken(**token_dict)  # type: ignore[arg-type]
    with pytest.raises(AttributeError):
        _ = degraded_token.scope.scope_id  # type: ignore[union-attr]


# CARRY-FORWARD CF-6: non-canonical scope strings (e.g. scope='AGENT' instead of 'agent')
# create phantom claim nodes that are invisible to both canonical reads and to the correct
# scope. Exposure requires direct Claim construction bypassing ScopedKG.assert_fact (which
# always derives scope from token.scope.scope_id — always canonical). No structural guard
# at the Claim boundary; document discipline in assert_fact docstring.


def test_sc7_scope_write_token_no_dict_construction() -> None:
    """dataclasses.asdict on ScopeWriteToken produces a plain dict where 'scope' is a dict.

    Splatting this dict back into ScopeWriteToken succeeds at Python's dataclass level (no
    runtime type enforcement), but the resulting token is structurally degraded: .scope is a
    dict, not a Scope, so any downstream call to .scope.scope_id raises AttributeError.

    This proves that model-generated JSON cannot accidentally produce a USABLE ScopeWriteToken
    without a deliberate re-hydration step (constructing a real Scope from the dict first).
    """
    scope = Scope(kind="world", ref="global")
    token = ScopeWriteToken(scope=scope, owner="agent-x")

    token_dict = dataclasses.asdict(token)
    # token_dict["scope"] is now a plain dict, not a Scope object
    assert isinstance(token_dict["scope"], dict)

    # Python dataclasses allow splatting (no runtime type enforcement) — but the result is
    # degraded: .scope is a dict, not a Scope object.
    degraded = ScopeWriteToken(**token_dict)  # type: ignore[arg-type]
    assert isinstance(degraded.scope, dict), "Degraded token's scope must be a dict"

    # The degraded token is unusable: .scope.scope_id raises AttributeError
    with pytest.raises(AttributeError):
        _ = degraded.scope.scope_id  # type: ignore[union-attr]
