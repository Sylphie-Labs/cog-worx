"""Unit tests for cogworx.knowledge.identity (CANON S5).

Pins the content-hash identity rules:
  - normalize_topic_part: NFC + strip/lowercase/collapse-whitespace/strip-punctuation.
  - claim_id_for: applies normalize_topic_part to each component before hashing, then uses
    length-prefixed fields separated by ASCII unit-separator (\\x1f) to prevent collisions.
  - Normalization semantics: "Has Mass" vs "has mass" vs "has  mass" and NFC-vs-combining-accent
    variants all produce the same id; genuinely different words produce different ids.
"""

from __future__ import annotations

import unicodedata

from cogworx.knowledge.identity import claim_id_for, normalize_topic_part

# ---------------------------------------------------------------------------
# normalize_topic_part
# ---------------------------------------------------------------------------


def test_normalize_strips_whitespace() -> None:
    assert normalize_topic_part("  Has Mass  ") == "has mass"


def test_normalize_lowercases() -> None:
    assert normalize_topic_part("HAS_MASS") == "has_mass"


def test_normalize_collapses_internal_whitespace() -> None:
    # Internal tabs and newlines collapse to single space
    assert normalize_topic_part("has\t\tmass") == "has mass"
    assert normalize_topic_part("has\nmass") == "has mass"
    assert normalize_topic_part("has   mass") == "has mass"


def test_normalize_strips_leading_trailing_punctuation() -> None:
    assert normalize_topic_part("—has-mass—") == "has-mass"
    assert normalize_topic_part("...topic...") == "topic"
    assert normalize_topic_part("!!hello!!") == "hello"


def test_normalize_preserves_internal_punctuation() -> None:
    # Hyphens and underscores inside a word are word characters — untouched.
    assert normalize_topic_part("has-mass") == "has-mass"
    assert normalize_topic_part("has_mass") == "has_mass"


def test_normalize_empty_string() -> None:
    assert normalize_topic_part("") == ""


def test_normalize_is_idempotent() -> None:
    cases = [
        "  Has Mass  ",
        "—has-mass—",
        "has   mass",
        "HELLO WORLD",
        "...topic...",
    ]
    for s in cases:
        once = normalize_topic_part(s)
        twice = normalize_topic_part(once)
        assert once == twice, f"normalize_topic_part not idempotent on {s!r}: {once!r} -> {twice!r}"


# ---------------------------------------------------------------------------
# claim_id_for
# ---------------------------------------------------------------------------


def test_claim_id_for_is_32_hex_chars() -> None:
    cid = claim_id_for("pluto", "has_mass", "1.3e22 kg")
    assert len(cid) == 32
    assert all(c in "0123456789abcdef" for c in cid)


def test_claim_id_for_is_deterministic() -> None:
    a = claim_id_for("pluto", "has_mass", "1.3e22 kg")
    b = claim_id_for("pluto", "has_mass", "1.3e22 kg")
    assert a == b


def test_claim_id_for_differs_by_subject() -> None:
    a = claim_id_for("pluto", "has_mass", "payload")
    b = claim_id_for("eris", "has_mass", "payload")
    assert a != b


def test_claim_id_for_differs_by_predicate() -> None:
    a = claim_id_for("pluto", "has_mass", "payload")
    b = claim_id_for("pluto", "has_radius", "payload")
    assert a != b


def test_claim_id_for_differs_by_object_repr() -> None:
    a = claim_id_for("pluto", "has_mass", "1.3e22 kg")
    b = claim_id_for("pluto", "has_mass", "1.4e22 kg")
    assert a != b


def test_claim_id_for_empty_predicate() -> None:
    # When predicate is None the adapter passes "" — the id must still be deterministic.
    a = claim_id_for("pluto", "", "some payload")
    b = claim_id_for("pluto", "", "some payload")
    assert a == b
    # And distinct from a non-empty predicate.
    c = claim_id_for("pluto", "has_mass", "some payload")
    assert a != c


# ---------------------------------------------------------------------------
# Pipe-delimiter collision check (was a bug; now fixed by length-prefix encoding)
# ---------------------------------------------------------------------------
# The implementation uses length-prefix + ASCII unit-separator (\x1f) encoding which prevents
# collisions even when fields contain arbitrary characters including pipe.


def test_claim_id_pipe_delimiter_no_collision() -> None:
    """Pipe in subject must not collide with a different predicate split (delimiter safety)."""
    # Triple A: subject has a pipe
    id_a = claim_id_for("a|b", "c", "d")
    # Triple B: predicate has a pipe (different triple)
    id_b = claim_id_for("a", "b|c", "d")
    # These represent DIFFERENT knowledge triples and MUST NOT collide.
    assert id_a != id_b


# ---------------------------------------------------------------------------
# FIX 5: normalize_topic_part NFC + claim_id_for normalization regression tests
# ---------------------------------------------------------------------------


def test_normalize_nfc_precomposed_vs_combining_accents() -> None:
    """NFC normalization: precomposed é (U+00E9) vs e + combining accent (U+0065 U+0301) same."""
    precomposed = "é"  # é precomposed
    combining = "é"  # e + combining acute accent
    # They look the same and NFC collapses them.
    assert unicodedata.normalize("NFC", precomposed) == unicodedata.normalize("NFC", combining)
    assert normalize_topic_part(precomposed) == normalize_topic_part(combining)


def test_normalize_nfc_cafe_variants() -> None:
    """café variants (NFC vs NFD) normalize to the same form."""
    cafe_nfc = "café"  # precomposed
    cafe_nfd = "café"  # decomposed (e + combining accent)
    assert normalize_topic_part(cafe_nfc) == normalize_topic_part(cafe_nfd)


def test_claim_id_for_case_insensitive() -> None:
    """'Has Mass' and 'has mass' produce the same claim id after normalization."""
    id_upper = claim_id_for("Pluto", "Has Mass", "Big")
    id_lower = claim_id_for("pluto", "has mass", "big")
    assert id_upper == id_lower


def test_claim_id_for_double_space_same_as_single() -> None:
    """'has  mass' (double space) and 'has mass' (single space) produce the same id."""
    id_double = claim_id_for("pluto", "has  mass", "payload")
    id_single = claim_id_for("pluto", "has mass", "payload")
    assert id_double == id_single


def test_claim_id_for_nfc_combining_accent_same_as_precomposed() -> None:
    """NFC-vs-combining-accent variants of the same subject produce the same id."""
    id_precomposed = claim_id_for("café", "serves", "coffee")
    id_combining = claim_id_for("café", "serves", "coffee")
    assert id_precomposed == id_combining


def test_claim_id_for_genuinely_different_words_different_ids() -> None:
    """Genuinely different words still produce different ids (paraphrases not collapsed)."""
    id_a = claim_id_for("pluto", "has_mass", "1.3e22 kg")
    id_b = claim_id_for("eris", "has_mass", "1.3e22 kg")
    id_c = claim_id_for("pluto", "has_radius", "1.3e22 kg")
    assert id_a != id_b
    assert id_a != id_c
