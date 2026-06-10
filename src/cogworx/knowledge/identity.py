"""Content-hash identity for entity-KG claims (CANON S5).

Two derivations of the same (subject, predicate, object) triple must produce the same claim id so
re-derivation across runs accumulates evidence on one node rather than minting a duplicate. The hash
is computed in Python — canonicalization rules live here, not in Cypher. Ported from tess.kg.
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

from cogworx.claims.provenance import DEFAULT_SCOPE as DEFAULT_SCOPE

__all__ = [
    "DEFAULT_SCOPE",
    "claim_id_for",
    "normalize_topic_part",
]

# Leading/trailing: anything that is not a word character (\w) or whitespace.
_LEADING_TRAILING_PUNCT = re.compile(r"^[^\w\s]+|[^\w\s]+$")
_WHITESPACE_RUN = re.compile(r"\s+")


def normalize_topic_part(s: str) -> str:
    """Canonical form for subject/predicate columns used in resolution-candidate lookup.

    Rules (applied in order):
      1. Unicode NFC normalization — ensures that visually identical strings with different
         code-point sequences (e.g. combining accents vs precomposed characters, "café" composed
         vs decomposed) map to the same bytes before any other rule is applied.
      2. Strip surrounding whitespace.
      3. Lowercase.
      4. Collapse internal runs of whitespace to one space.
      5. Strip leading/trailing punctuation (anything that is not a word character or space).

    Deterministic and pure — no I/O, no state. Ported from tess.kg.normalize_topic_part.

    Identity now matches the resolution-column notion of "same topic": casing, extra whitespace,
    and combining-character variants of the same glyph are all treated as identical. PARAPHRASES
    (different words) still mint distinct nodes by design — semantic dedup is the future
    resolution-judge pod's job, not the hash's.
    """
    # NFC first so combining accents collapse to precomposed form before lowercasing.
    s = unicodedata.normalize("NFC", s)
    s = s.strip()
    s = s.lower()
    s = _WHITESPACE_RUN.sub(" ", s)
    s = _LEADING_TRAILING_PUNCT.sub("", s)
    return s


def claim_id_for(
    subject: str,
    predicate: str,
    object_repr: str,
    *,
    scope: str = DEFAULT_SCOPE,
) -> str:
    """Deterministic content-hash id for a (subject, predicate, object[, scope]) tuple.

    ``object_repr`` is ``claim.object_entity`` when the object is itself an entity, otherwise the
    payload text. Two stage writes that derive the same triple collide by design — MERGE on this id
    makes the second write idempotent (evidence accumulates on the existing node).

    ``scope`` partitions the id space between governed views (world model, user model) and the
    default unscoped surface ("agent"). When scope is DEFAULT_SCOPE the output is byte-identical
    to the pre-2.4 three-part hash — no existing claim ids change.

    Returns the first 32 hex characters of the SHA-256 digest, matching tess's convention.

    Encoding: each component is first normalized via :func:`normalize_topic_part` (NFC + lowercase
    + whitespace collapse + punctuation strip), then length-prefixed (``len:<value>``) and separated
    by the ASCII unit separator (``\\x1f``). For scoped claims a 4th length-prefixed part is
    appended. Length-prefix + distinct separator together prevent the pipe-collision bug where a
    field containing the separator character could produce the same raw bytes as a different split.
    The 3-part (unscoped) and 4-part (scoped) encodings cannot collide because the boundary is
    unambiguous in the length-prefix scheme.
    """
    parts = [
        f"{len(s)}:{s}"
        for s in (
            normalize_topic_part(subject),
            normalize_topic_part(predicate),
            normalize_topic_part(object_repr),
        )
    ]
    if scope != DEFAULT_SCOPE:
        scope_norm = normalize_topic_part(scope)
        parts.append(f"{len(scope_norm)}:{scope_norm}")
    raw = "\x1f".join(parts).encode()
    return hashlib.sha256(raw).hexdigest()[:32]
