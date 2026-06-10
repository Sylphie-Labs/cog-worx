"""Per-kind deterministic text rendering for recall results (Pod 2.5, CANON S1).

Pure functions — no I/O, no model calls. Each function produces the ``text`` field that
flows through ``RecallResult.text`` → ``ContextChunk.text`` → the assembled prompt.
"""

from __future__ import annotations

from cogworx.substrate.entity_kg import ScoredClaim
from cogworx.substrate.episodes import Episode
from cogworx.substrate.latent import LatentMatch

__all__ = ["render_claim", "render_episode", "render_latent"]


def render_claim(result: ScoredClaim) -> str:
    """subject predicate payload [epistemic_type, conf=X.XX, scope=Y]"""
    conf = result.confidence.confidence
    return (
        f"{result.claim.subject} {result.claim.predicate} {result.claim.payload}"
        f" [{result.claim.epistemic_type}, conf={conf:.2f}, scope={result.claim.scope}]"
    )


def render_episode(episode: Episode) -> str:
    """[role @ YYYY-MM-DD HH:MM] content"""
    return f"[{episode.role} @ {episode.occurred_at:%Y-%m-%d %H:%M}] {episode.content}"


def render_latent(match: LatentMatch) -> str:
    """payload['text'] if present and a str, otherwise compact repr of the payload."""
    text = match.record.payload.get("text")
    if isinstance(text, str):
        return text
    return repr(match.record.payload)
