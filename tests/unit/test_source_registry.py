"""Source registry — declaration-time identity discipline (CANON S9, Pod 2.3).

Pins the SourceRegistry contract: deterministic source_id minting from (kind, ref), NFC
normalisation, idempotent same-declaration, fail-loud rejection of conflicting authority for
the same (kind, ref), and structural isolation of distinct (kind, ref) pairs.
"""

from __future__ import annotations

import unicodedata

import pytest

from cogworx.knowledge.source_registry import SourceDeclaration, SourceRegistry


def test_declare_basic() -> None:
    registry = SourceRegistry()
    decl = registry.declare("human", "alice", authority=0.9)

    assert isinstance(decl, SourceDeclaration)
    assert decl.kind == "human"
    assert decl.ref == "alice"
    assert decl.source_id == "source:human:alice"
    assert decl.source_authority == 0.9


def test_declare_idempotent() -> None:
    registry = SourceRegistry()
    a = registry.declare("human", "alice", authority=0.9)
    b = registry.declare("human", "alice", authority=0.9)

    assert a is b
    assert len(registry) == 1


def test_declare_conflict_raises() -> None:
    registry = SourceRegistry()
    registry.declare("human", "alice", authority=0.9)

    with pytest.raises(ValueError, match="conflicting identity discipline"):
        registry.declare("human", "alice", authority=0.5)


def test_source_id_deterministic() -> None:
    registry = SourceRegistry()
    decl = registry.declare("human", "bob")

    assert decl.source_id == "source:human:bob"


def test_nfc_normalization() -> None:
    # é as NFC (U+00E9) vs NFD decomposed (e + combining accent U+0301) — both must produce
    # the same source_id and count as the same declaration.
    nfc_ref = unicodedata.normalize("NFC", "é")  # é  (precomposed)
    nfd_ref = unicodedata.normalize("NFD", "é")  # e + combining accent (decomposed)
    assert nfc_ref != nfd_ref  # confirm the test inputs actually differ

    registry = SourceRegistry()
    a = registry.declare("human", nfc_ref, authority=1.0)
    b = registry.declare("human", nfd_ref, authority=1.0)

    assert a is b
    assert len(registry) == 1
    assert a.source_id == f"source:human:{nfc_ref}"


def test_get_by_id() -> None:
    registry = SourceRegistry()
    decl = registry.declare("tool", "search_tool", authority=0.8)

    result = registry.get_by_id(decl.source_id)
    assert result is decl


def test_get_unknown() -> None:
    registry = SourceRegistry()

    assert registry.get("human", "nobody") is None
    assert registry.get_by_id("source:human:nobody") is None


def test_authority_validation_above_one() -> None:
    with pytest.raises(ValueError, match="source_authority must be in"):
        SourceDeclaration(kind="human", ref="alice", source_authority=1.1)


def test_authority_validation_below_zero() -> None:
    with pytest.raises(ValueError, match="source_authority must be in"):
        SourceDeclaration(kind="human", ref="alice", source_authority=-0.1)


def test_authority_boundary_values_are_valid() -> None:
    lo = SourceDeclaration(kind="human", ref="lo", source_authority=0.0)
    hi = SourceDeclaration(kind="human", ref="hi", source_authority=1.0)

    assert lo.source_authority == 0.0
    assert hi.source_authority == 1.0


def test_different_kinds_same_ref() -> None:
    registry = SourceRegistry()
    human = registry.declare("human", "alice", authority=0.9)
    tool = registry.declare("tool", "alice", authority=0.7)

    assert human.source_id != tool.source_id
    assert human.source_id == "source:human:alice"
    assert tool.source_id == "source:tool:alice"
    assert len(registry) == 2

    assert registry.get("human", "alice") is human
    assert registry.get("tool", "alice") is tool


def test_contains_protocol() -> None:
    registry = SourceRegistry()
    registry.declare("agent", "planner", authority=1.0)

    assert ("agent", "planner") in registry
    assert ("agent", "unknown") not in registry


def test_iter_and_len() -> None:
    registry = SourceRegistry()
    registry.declare("human", "alice", authority=0.9)
    registry.declare("tool", "search", authority=0.8)
    registry.declare("document", "https://example.com/doc", authority=0.6)

    assert len(registry) == 3
    kinds = {d.kind for d in registry}
    assert kinds == {"human", "tool", "document"}
