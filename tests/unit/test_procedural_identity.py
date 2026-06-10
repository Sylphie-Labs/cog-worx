"""Unit tests for cogworx.knowledge.procedural_identity (CANON S5, S9).

Pins the deterministic, collision-proof id minting for procedural-KG nodes:
  - procedure_id_for(pathway, stage): NFC + lowercase + whitespace/punctuation normalization, then
    length-prefixed fields separated by ASCII unit-separator (\\x1f) to prevent split collisions.
  - problem_type_id_for(label): same normalization + hashing over a single label.
  - Case / whitespace / NFC variants collapse to one id; genuinely different labels do not collide;
    the (pathway, stage) split is collision-safe even when a component contains the separator.
"""

from __future__ import annotations

import unicodedata

from cogworx.knowledge.procedural_identity import problem_type_id_for, procedure_id_for

# ---------------------------------------------------------------------------
# procedure_id_for — shape + determinism
# ---------------------------------------------------------------------------


def test_procedure_id_is_32_hex_chars() -> None:
    pid = procedure_id_for("solve_math", "decompose")
    assert len(pid) == 32
    assert all(c in "0123456789abcdef" for c in pid)


def test_procedure_id_is_deterministic() -> None:
    a = procedure_id_for("solve_math", "decompose")
    b = procedure_id_for("solve_math", "decompose")
    assert a == b


def test_procedure_id_differs_by_pathway() -> None:
    a = procedure_id_for("solve_math", "decompose")
    b = procedure_id_for("solve_logic", "decompose")
    assert a != b


def test_procedure_id_differs_by_stage() -> None:
    a = procedure_id_for("solve_math", "decompose")
    b = procedure_id_for("solve_math", "verify")
    assert a != b


# ---------------------------------------------------------------------------
# procedure_id_for — normalization (case / whitespace / NFC)
# ---------------------------------------------------------------------------


def test_procedure_id_case_insensitive() -> None:
    upper = procedure_id_for("Solve_Math", "Decompose")
    lower = procedure_id_for("solve_math", "decompose")
    assert upper == lower


def test_procedure_id_whitespace_collapsed() -> None:
    assert procedure_id_for("  solve math  ", "de  compose") == procedure_id_for(
        "solve math", "de compose"
    )


def test_procedure_id_nfc_combining_accent_same_as_precomposed() -> None:
    precomposed = "résumé"  # NFC precomposed
    combining = unicodedata.normalize("NFD", "résumé")  # decomposed
    assert procedure_id_for(precomposed, "stage") == procedure_id_for(combining, "stage")


# ---------------------------------------------------------------------------
# procedure_id_for — split-collision safety
# ---------------------------------------------------------------------------


def test_procedure_id_split_no_collision() -> None:
    """A (pathway, stage) boundary must not be confusable with a different split."""
    id_a = procedure_id_for("a b", "c")
    id_b = procedure_id_for("a", "b c")
    assert id_a != id_b


def test_procedure_id_separator_in_component_no_collision() -> None:
    """A component containing the unit separator cannot forge a different split."""
    id_a = procedure_id_for("a\x1fb", "c")
    id_b = procedure_id_for("a", "b\x1fc")
    assert id_a != id_b


# ---------------------------------------------------------------------------
# problem_type_id_for
# ---------------------------------------------------------------------------


def test_problem_type_id_is_32_hex_chars() -> None:
    tid = problem_type_id_for("Math Word Problem")
    assert len(tid) == 32
    assert all(c in "0123456789abcdef" for c in tid)


def test_problem_type_id_is_deterministic() -> None:
    assert problem_type_id_for("math word problem") == problem_type_id_for("math word problem")


def test_problem_type_id_case_and_whitespace_insensitive() -> None:
    assert problem_type_id_for("  Math Word Problem  ") == problem_type_id_for("math word problem")


def test_problem_type_id_differs_by_label() -> None:
    assert problem_type_id_for("math word problem") != problem_type_id_for("logic puzzle")


def test_problem_type_id_nfc_variants_collapse() -> None:
    nfc = "café tasks"
    nfd = unicodedata.normalize("NFD", "café tasks")
    assert problem_type_id_for(nfc) == problem_type_id_for(nfd)


# ---------------------------------------------------------------------------
# procedure vs problem-type id-space separation
# ---------------------------------------------------------------------------


def test_procedure_and_problem_type_ids_do_not_collide_on_same_text() -> None:
    """A one-part problem-type label must not mint the same id as a (pathway, stage) procedure.

    Length-prefixed arity makes a 1-field hash distinct from a 2-field hash even when the text lines
    up, so the two id spaces never alias.
    """
    assert problem_type_id_for("solve_math") != procedure_id_for("solve_math", "")
