"""Unit tests for Pod 2.4 — Scope, ScopeWriteToken, ScopeRegistry (CANON S7).

No async, no external deps, no substrate.  Pure Python contract tests.

Invariants under test:
  U1 — Scope construction and scope_id derivation rules.
  U2 — Scope is frozen (immutable after construction).
  U3 — ScopeRegistry.declare is idempotent (same object returned).
  U4 — claim_write_token first-call binds owner, same-owner re-call is idempotent.
  U5 — claim_write_token different-owner raises ValueError ("exactly one write-token").
  U6 — writer_of returns None before claim, owner after.
  U7 — DEFAULT_SCOPE is the literal string "agent".
"""

from __future__ import annotations

import pytest

from cogworx.knowledge.scopes import (
    DEFAULT_SCOPE,
    Scope,
    ScopeKind,
    ScopeRegistry,
    ScopeWriteToken,
)


# ---------------------------------------------------------------------------
# U1 — Scope construction + scope_id derivation
# ---------------------------------------------------------------------------


def test_world_scope_id_is_world() -> None:
    """Scope(kind='world', ref='global') always yields scope_id == 'world'."""
    s = Scope(kind="world", ref="global")
    assert s.scope_id == "world"


def test_user_scope_id_is_user_colon_ref() -> None:
    """Scope(kind='user', ref='jim') yields scope_id == 'user:jim'."""
    s = Scope(kind="user", ref="jim")
    assert s.scope_id == "user:jim"


def test_user_scope_id_nfc_normalised() -> None:
    """scope_id for user scopes uses NFC-normalised ref (same as SourceDeclaration convention)."""
    # café with combining acute vs precomposed — both should yield the same scope_id
    ref_decomposed = "café"  # e + combining acute
    ref_precomposed = "café"   # precomposed é
    s_dec = Scope(kind="user", ref=ref_decomposed)
    s_pre = Scope(kind="user", ref=ref_precomposed)
    assert s_dec.scope_id == s_pre.scope_id


def test_user_scope_empty_ref_raises() -> None:
    """Scope(kind='user', ref='') raises ValueError (empty ref)."""
    with pytest.raises(ValueError, match="non-empty"):
        Scope(kind="user", ref="")


def test_user_scope_whitespace_only_ref_raises() -> None:
    """Scope(kind='user', ref='   ') raises ValueError (whitespace-only ref)."""
    with pytest.raises(ValueError, match="non-empty"):
        Scope(kind="user", ref="   ")


def test_world_scope_ref_is_stored() -> None:
    """The ref field is preserved on world scopes."""
    s = Scope(kind="world", ref="global")
    assert s.ref == "global"
    assert s.kind == "world"


def test_user_scope_kind_and_ref_stored() -> None:
    """kind and ref fields are preserved on user scopes."""
    s = Scope(kind="user", ref="alice")
    assert s.kind == "user"
    assert s.ref == "alice"


# ---------------------------------------------------------------------------
# U2 — Scope is frozen (immutable)
# ---------------------------------------------------------------------------


def test_scope_is_frozen_kind() -> None:
    """Scope is a frozen dataclass — mutating kind raises."""
    s = Scope(kind="world", ref="global")
    with pytest.raises((AttributeError, TypeError)):
        s.kind = "user"  # type: ignore[misc]


def test_scope_is_frozen_ref() -> None:
    """Scope is a frozen dataclass — mutating ref raises."""
    s = Scope(kind="user", ref="bob")
    with pytest.raises((AttributeError, TypeError)):
        s.ref = "alice"  # type: ignore[misc]


def test_scope_is_frozen_scope_id() -> None:
    """scope_id cannot be overwritten (computed field, frozen dataclass)."""
    s = Scope(kind="world", ref="global")
    with pytest.raises((AttributeError, TypeError)):
        s.scope_id = "hacked"  # type: ignore[misc]


def test_scope_write_token_is_frozen() -> None:
    """ScopeWriteToken is a frozen dataclass — mutating owner raises."""
    scope = Scope(kind="world", ref="global")
    token = ScopeWriteToken(scope=scope, owner="agent-a")
    with pytest.raises((AttributeError, TypeError)):
        token.owner = "agent-b"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# U3 — ScopeRegistry.declare is idempotent (same object, no ownership claim)
# ---------------------------------------------------------------------------


def test_declare_world_idempotent() -> None:
    """declare('world') twice returns the same Scope object."""
    registry = ScopeRegistry()
    s1 = registry.declare("world")
    s2 = registry.declare("world")
    assert s1 is s2


def test_declare_user_idempotent() -> None:
    """declare('user', 'jim') twice returns the same Scope object."""
    registry = ScopeRegistry()
    s1 = registry.declare("user", "jim")
    s2 = registry.declare("user", "jim")
    assert s1 is s2


def test_declare_different_users_are_distinct() -> None:
    """declare for different user refs returns distinct Scope objects."""
    registry = ScopeRegistry()
    s_jim = registry.declare("user", "jim")
    s_alice = registry.declare("user", "alice")
    assert s_jim is not s_alice
    assert s_jim.scope_id != s_alice.scope_id


def test_declare_world_and_user_are_distinct() -> None:
    """World and user scopes are always distinct even when ref collides textually."""
    registry = ScopeRegistry()
    s_world = registry.declare("world", "global")
    s_user = registry.declare("user", "global")  # valid user with ref="global"
    assert s_world is not s_user
    assert s_world.scope_id == "world"
    assert s_user.scope_id == "user:global"


# ---------------------------------------------------------------------------
# U4 — claim_write_token first-call binds; same-owner re-call is idempotent
# ---------------------------------------------------------------------------


def test_claim_write_token_first_call_returns_token() -> None:
    """First call to claim_write_token returns a ScopeWriteToken with correct scope and owner."""
    registry = ScopeRegistry()
    token = registry.claim_write_token("world", owner="agent-x")
    assert isinstance(token, ScopeWriteToken)
    assert token.owner == "agent-x"
    assert token.scope.scope_id == "world"


def test_claim_write_token_same_owner_idempotent() -> None:
    """Same owner re-claiming the same scope returns the same token content (idempotent)."""
    registry = ScopeRegistry()
    t1 = registry.claim_write_token("world", owner="agent-x")
    t2 = registry.claim_write_token("world", owner="agent-x")
    # Both tokens are equal: same scope and owner
    assert t1.scope is t2.scope
    assert t1.owner == t2.owner


def test_claim_write_token_user_scope_first_call() -> None:
    """claim_write_token for a user scope binds correctly."""
    registry = ScopeRegistry()
    token = registry.claim_write_token("user", "alice", owner="world-agent")
    assert token.scope.scope_id == "user:alice"
    assert token.owner == "world-agent"


# ---------------------------------------------------------------------------
# U5 — different-owner raises ValueError with canonical message fragment
# ---------------------------------------------------------------------------


def test_claim_write_token_different_owner_raises() -> None:
    """Different owner claiming the same scope raises ValueError (S7)."""
    registry = ScopeRegistry()
    registry.claim_write_token("world", owner="agent-a")
    with pytest.raises(ValueError, match="exactly one write-token"):
        registry.claim_write_token("world", owner="agent-b")


def test_claim_write_token_different_owner_user_scope_raises() -> None:
    """Different owner claiming the same user scope raises ValueError (S7)."""
    registry = ScopeRegistry()
    registry.claim_write_token("user", "jim", owner="agent-a")
    with pytest.raises(ValueError, match="exactly one write-token"):
        registry.claim_write_token("user", "jim", owner="agent-b")


def test_claim_write_token_different_users_independent() -> None:
    """Different user scopes allow different owners — the constraint is per-scope_id."""
    registry = ScopeRegistry()
    t1 = registry.claim_write_token("user", "alice", owner="agent-a")
    t2 = registry.claim_write_token("user", "bob", owner="agent-b")
    # No ValueError: different scope_ids, each with their own owner
    assert t1.owner == "agent-a"
    assert t2.owner == "agent-b"


# ---------------------------------------------------------------------------
# U6 — writer_of returns None before claim, owner after
# ---------------------------------------------------------------------------


def test_writer_of_returns_none_before_claim() -> None:
    """writer_of returns None for an undeclared scope."""
    registry = ScopeRegistry()
    result = registry.writer_of("world")
    assert result is None


def test_writer_of_returns_owner_after_claim() -> None:
    """writer_of returns the owner after claim_write_token."""
    registry = ScopeRegistry()
    registry.claim_write_token("world", owner="agent-z")
    assert registry.writer_of("world") == "agent-z"


def test_writer_of_returns_none_for_unclaimed_user_scope() -> None:
    """writer_of returns None for a user scope nobody has claimed."""
    registry = ScopeRegistry()
    # Declare but do not claim
    registry.declare("user", "jim")
    assert registry.writer_of("user", "jim") is None


def test_writer_of_user_scope_returns_owner_after_claim() -> None:
    """writer_of returns the correct owner for a claimed user scope."""
    registry = ScopeRegistry()
    registry.claim_write_token("user", "alice", owner="world-writer")
    assert registry.writer_of("user", "alice") == "world-writer"


def test_writer_of_user_scope_scope_isolation() -> None:
    """writer_of for user:alice does not return the owner of user:bob."""
    registry = ScopeRegistry()
    registry.claim_write_token("user", "alice", owner="owner-a")
    assert registry.writer_of("user", "bob") is None


# ---------------------------------------------------------------------------
# U7 — DEFAULT_SCOPE is the literal string "agent"
# ---------------------------------------------------------------------------


def test_default_scope_is_agent() -> None:
    """DEFAULT_SCOPE must equal the literal string 'agent' — backward-compat invariant."""
    assert DEFAULT_SCOPE == "agent"


def test_default_scope_is_string_type() -> None:
    """DEFAULT_SCOPE is a plain str, not some special type."""
    assert isinstance(DEFAULT_SCOPE, str)
