"""Pure extraction core for episodic claim extraction (CANON S1, S5, S9, Pod 2.3).

This module is PURE — no substrate, no model interface deps. It:
  1. Renders a transcript from a list of turns for model consumption.
  2. Validates raw model JSON output (RawExtractionOutput) structurally.
  3. Mints all trust-bearing claim/evidence fields from framework code — epistemic_type is
     hardcoded to "inference", source_id comes only from SourceDeclaration, claim.id from
     claim_id_for. Model-chosen identity/epistemic fields are structurally unreachable (S9).

The ClaimExtractor (runtime/claim_extractor.py) calls these functions and owns the model call.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from cogworx.claims.provenance import Claim, Provenance
from cogworx.knowledge.episodes import Turn
from cogworx.knowledge.evidence import EvidenceEvent, make_evidence
from cogworx.knowledge.identity import claim_id_for
from cogworx.knowledge.source_registry import SourceDeclaration

__all__ = [
    "ExtractionResult",
    "RawClaimItem",
    "mint_extraction_claims",
    "render_transcript",
    "validate_raw_model_json",
]

_log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawClaimItem:
    """One raw claim as the model emitted it (all fields are strings, unvalidated)."""

    subject: str
    predicate: str
    object: str
    supporting_turn_index: int  # model-chosen; validated against the turn list


@dataclass(frozen=True)
class ExtractionResult:
    """The validated result of one extraction: a list of (Claim, EvidenceEvent) pairs."""

    pairs: tuple[tuple[Claim, EvidenceEvent], ...]
    rejected_count: int  # claims dropped due to validation (wrong turn index, not user turn, etc.)


def render_transcript(turns: Sequence[Turn], *, session_id: str) -> str:
    """Render a list of turns as a transcript string for model input.

    Format: each turn as "[ROLE {i}]: {content}\\n" (role uppercased, 0-indexed).
    Prefixed by "[SESSION: {session_id}]\\n".
    """
    lines: list[str] = [f"[SESSION: {session_id}]"]
    for i, turn in enumerate(turns):
        lines.append(f"[{turn.role.upper()} {i}]: {turn.content}")
    return "\n".join(lines) + "\n"


def validate_raw_model_json(raw: Any) -> list[RawClaimItem]:
    """Parse model JSON output into a list of RawClaimItem.

    Expected model output schema::

        {
            "claims": [
                {"subject": "...", "predicate": "...", "object": "...", "supporting_turn_index": 0}
            ]
        }

    Raises ValueError on schema mismatch (raw not a dict, or raw['claims'] not a list) — the
    caller (ClaimExtractor._extract_session) does not catch ValueError, so this stalls the
    cursor (fail-stall, D6).
    Drops individual items missing required fields
    (subject, predicate, object, supporting_turn_index).
    supporting_turn_index must be a non-negative integer.
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"validate_raw_model_json: expected a dict, got {type(raw).__name__!r}; "
            "stalling (fail-stall, D6)"
        )

    claims_raw = raw.get("claims")
    if not isinstance(claims_raw, list):
        raise ValueError(
            f"validate_raw_model_json: expected raw['claims'] to be a list, "
            f"got {type(claims_raw).__name__!r}; stalling (fail-stall, D6)"
        )

    items: list[RawClaimItem] = []
    for i, item in enumerate(claims_raw):
        if not isinstance(item, dict):
            _log.debug("validate_raw_model_json: claims[%d] is not a dict; dropping", i)
            continue

        subject = item.get("subject")
        predicate = item.get("predicate")
        obj = item.get("object")
        turn_index = item.get("supporting_turn_index")

        if not isinstance(subject, str):
            _log.debug("validate_raw_model_json: claims[%d] missing/invalid 'subject'; dropping", i)
            continue
        if not isinstance(predicate, str):
            _log.debug(
                "validate_raw_model_json: claims[%d] missing/invalid 'predicate'; dropping", i
            )
            continue
        if not isinstance(obj, str):
            _log.debug("validate_raw_model_json: claims[%d] missing/invalid 'object'; dropping", i)
            continue
        if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 0:
            _log.debug(
                "validate_raw_model_json: claims[%d] missing/invalid"
                " 'supporting_turn_index'; dropping",
                i,
            )
            continue

        items.append(
            RawClaimItem(
                subject=subject,
                predicate=predicate,
                object=obj,
                supporting_turn_index=turn_index,
            )
        )

    return items


def mint_extraction_claims(
    raw_items: Sequence[RawClaimItem],
    *,
    turns: Sequence[Turn],
    source_decl: SourceDeclaration,
    episode_ids_by_turn: Sequence[str],  # episode_id for each turn (same index)
    recorded_at: datetime,
    run_id: str | None = None,
    stage: str | None = None,
) -> ExtractionResult:
    """Validate raw items and mint (Claim, EvidenceEvent) pairs from framework code (S9).

    Validation per item:

    - supporting_turn_index must be in range [0, len(turns)-1] — drop if not.
    - turns[supporting_turn_index].role must be "user" — drop if not (v1: user turns only; CF-2).
    - subject, predicate, object must all be non-empty strings — drop if empty.

    Minting applies S9 hard walls:

    - claim.id = claim_id_for(subject, predicate, object) — NEVER model-chosen.
    - claim.epistemic_type = "inference" — HARDCODED; model output ignored.
    - claim.provenance.source = "extraction"; source_ref =
      episode_ids_by_turn[supporting_turn_index].
    - claim.provenance.confidence = source_decl.source_authority.
    - claim.created_by = source_decl.source_id.
    - evidence.source_id = source_decl.source_id.

    Returns ExtractionResult(pairs=..., rejected_count=...).
    """
    n_turns = len(turns)
    pairs: list[tuple[Claim, EvidenceEvent]] = []
    rejected = 0

    for item in raw_items:
        idx = item.supporting_turn_index

        if idx >= n_turns:
            _log.debug(
                "mint_extraction_claims: supporting_turn_index %d out of range [0, %d); dropping",
                idx,
                n_turns,
            )
            rejected += 1
            continue

        if turns[idx].role != "user":
            _log.debug(
                "mint_extraction_claims: supporting_turn_index %d has role %r"
                " (not 'user'); dropping",
                idx,
                turns[idx].role,
            )
            rejected += 1
            continue

        if not item.subject.strip():
            _log.debug("mint_extraction_claims: empty subject; dropping")
            rejected += 1
            continue

        if not item.predicate.strip():
            _log.debug("mint_extraction_claims: empty predicate; dropping")
            rejected += 1
            continue

        if not item.object.strip():
            _log.debug("mint_extraction_claims: empty object; dropping")
            rejected += 1
            continue

        episode_id = episode_ids_by_turn[idx]

        claim = Claim(
            id=claim_id_for(item.subject, item.predicate, item.object),
            subject=item.subject,
            predicate=item.predicate,
            payload=item.object,
            epistemic_type="inference",  # S9: hardcoded, never model-chosen
            provenance=Provenance(
                source="extraction",
                source_ref=episode_id,
                confidence=source_decl.source_authority,
                recorded_at=recorded_at,
            ),
            valid_from=recorded_at,
            ingest_time=recorded_at,
            created_by=source_decl.source_id,
        )

        evidence = make_evidence(
            type="extraction",
            polarity="+",
            source_id=source_decl.source_id,
            source_authority=source_decl.source_authority,
            recorded_at=recorded_at,
            run_id=run_id,
            stage=stage,
        )

        pairs.append((claim, evidence))

    return ExtractionResult(
        pairs=tuple(pairs),
        rejected_count=rejected,
    )
