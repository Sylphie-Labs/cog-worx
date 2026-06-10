"""Deterministic identity for procedural-KG nodes (CANON S5, S9).

A ``(:Procedure)`` and a ``(:ProblemType)`` are addressed by deterministic, collision-proof ids so
that two declarations of the same human-supplied label across runs collide on ONE node (trials
accumulate against it) rather than minting duplicates. The hash is computed in Python — the
canonicalization rules live here, not in Cypher. Sibling of :mod:`cogworx.knowledge.identity`,
reusing its :func:`normalize_topic_part` for label canonicalization.

source DISCIPLINE (S9): procedure ids and problem-type labels are FRAMEWORK/DEV-assigned — a static
``(pathway, stage) -> procedure_id`` registry and a declaration-time problem-type label. They are
NEVER taken from model output. A model-chosen procedure_id is an S9 violation — self-reported
identity gating the success posterior would let a model split or merge procedures to inflate its own
promotion. Structural enforcement (the declaration registry) is the projector's job; this module is
the deterministic minting discipline the framework code calls. Mirrors the ``source_id`` interim
discipline documented on :class:`cogworx.substrate.entity_kg.EntityKG`.
"""

from __future__ import annotations

import hashlib

from cogworx.knowledge.identity import normalize_topic_part

__all__ = [
    "problem_type_id_for",
    "procedure_id_for",
]


def procedure_id_for(pathway: str, stage: str) -> str:
    """Deterministic content-hash id for a ``(pathway, stage)`` procedure.

    A procedure is the unit a Trial scores: the stage's behaviour within a pathway. Two declarations
    of the same ``(pathway, stage)`` collide by design — MERGE on this id makes the projector
    idempotent (trials accumulate against the existing node).

    Returns the first 32 hex characters of the SHA-256 digest, matching
    :func:`cogworx.knowledge.identity.claim_id_for`'s convention.

    Encoding: each component is normalized via :func:`normalize_topic_part` (NFC + lowercase +
    whitespace collapse + punctuation strip), then length-prefixed (``len:<value>``) and separated
    by the ASCII unit separator (``\\x1f``). Length-prefix + distinct separator together prevent the
    split-collision bug where a component containing the separator could produce the same raw bytes
    as a different split (``pathway="a\\x1fb"`` is impossible to confuse with ``stage`` boundaries).
    """
    return _mint(normalize_topic_part(pathway), normalize_topic_part(stage))


def problem_type_id_for(label: str) -> str:
    """Deterministic content-hash id for a ``(:ProblemType)`` from its dev-supplied label.

    The label is the framework-assigned category a procedure is applied to (the posterior is per
    ``(procedure_id, problem_type)`` edge). Normalized + hashed identically to
    :func:`procedure_id_for` so ``"Math Word Problem"`` and ``"math word problem"`` collapse to one
    type. Genuinely different labels mint distinct types by design (no semantic dedup — that is a
    future judge pod's job, not the hash's).
    """
    return _mint(normalize_topic_part(label))


def _mint(*parts: str) -> str:
    encoded = [f"{len(p)}:{p}" for p in parts]
    raw = "\x1f".join(encoded).encode()
    return hashlib.sha256(raw).hexdigest()[:32]
