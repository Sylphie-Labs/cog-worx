"""Shared text-validation helpers for the cogworx.context package (Pod 3.4a).

Provides the newline-character frozenset and guard used by personality.py and rules.py
to reject injection attempts via embedded line separators in declared string fields.

Internal module — not part of the cogworx.context public API.

CANON sections cited:
  - S9 — structure over prompting; declared string fields must not carry hidden control chars.
"""

from __future__ import annotations

# Built from explicit codepoints (not literals) to avoid ambiguous-character linter warnings.
# Covers: \n (LF), \r (CR), \x0b (VT), \x0c (FF), U+0085 (NEL),
#          U+2028 (LINE SEP), U+2029 (PARA SEP).
_NEWLINE_CHARS: frozenset[str] = frozenset(
    ["\n", "\r", "\x0b", "\x0c", chr(0x0085), chr(0x2028), chr(0x2029)]
)


def _has_newline(s: str) -> bool:
    """Return True if *s* contains any newline or line-separator character."""
    return not _NEWLINE_CHARS.isdisjoint(s)
