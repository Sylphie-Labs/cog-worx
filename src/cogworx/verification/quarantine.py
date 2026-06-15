"""Model-output quarantine channel — nonce-fenced framing of untrusted model text (CANON S10).

DEFENSE-IN-DEPTH, never the primary defense (S10 = security BY STRUCTURE). Thesis output is
model-generated; entering the antithesis stage's context, it is — from the antithesis's point of
view — attacker-influenceable content sitting at the same "level" as its instructions: classic
prompt-injection surface. Unlike external-tier tool results it does NOT trip the Pod 3.2 durable
taint latch (it never crossed an external capability), so it is genuinely new surface (F8).

The STRUCTURAL defenses stay primary and are NOT rebuilt here: the antithesis runs in a fresh
context that receives the thesis ARTIFACT, not its chain-of-thought (the Pod 4.3 context boundary),
and the durable taint latch (``capability/policy.py``, Pod 3.2) is untouched. This channel adds a
*frame*: it wraps untrusted text in BEGIN/END markers tagged with a per-call cryptographic nonce, so
wrapped content cannot spoof the frame — an attacker who emits ``--- END UNTRUSTED OUTPUT ---`` to
"close" the block cannot guess the matching nonce. The consuming prompt's contract: anything between
matched markers is DATA, never instructions. Ported from tess ``oracles/_shared.py``
(``quote_untrusted_output``).

CF-3.2-B is a PRECONDITION (untagged read-tier ingestion does not yet taint/wrap); this marker is
one layer of the same load-bearing defense, never a substitute for closing that gap (S10).

The nonce is per-call and lives on the LIVE prompt-assembly path only; it is never part of replayed
state (S6 replays the model OUTPUT, not the re-assembled prompt), so its randomness is replay-safe.
"""

from __future__ import annotations

import re
import secrets
from typing import Final

_NONCE_HEX_BYTES: Final = 8  # 16 hex chars — unguessable per call (secrets, not a uuid).
_BEGIN: Final = "--- BEGIN UNTRUSTED OUTPUT [{nonce}] ---"
_END: Final = "--- END UNTRUSTED OUTPUT [{nonce}] ---"

# A well-formed block: BEGIN/END must carry the SAME nonce (backreference \1), body is everything
# between (DOTALL). A block whose markers disagree is a spoof attempt and does not match.
_BLOCK_RE: Final = re.compile(
    r"^--- BEGIN UNTRUSTED OUTPUT \[([0-9a-f]+)\] ---\n(.*)\n--- END UNTRUSTED OUTPUT \[\1\] ---$",
    re.DOTALL,
)


def new_nonce() -> str:
    """Mint a fresh, cryptographically unguessable quarantine nonce (16 hex chars)."""
    return secrets.token_hex(_NONCE_HEX_BYTES)


def quarantine(text: str, *, nonce: str | None = None) -> str:
    """Frame model-generated ``text`` as attacker-controlled DATA, not instructions (S10).

    Wraps ``text`` in BEGIN/END markers carrying a per-call nonce. When ``nonce`` is not supplied a
    fresh one is minted and RE-MINTED until it does not occur in ``text`` — so the framed content
    can never carry a matching END marker to break out (an unguessable 64-bit nonce makes this
    collision astronomically unlikely anyway; the loop makes it a guarantee). This is
    defense-in-depth: the consuming model must still be prompted to treat matched-nonce blocks as
    data — it is a layer, not the whole defense (S10).
    """
    if nonce is None:
        nonce = new_nonce()
        while nonce in text:  # guarantee the frame's nonce cannot appear in the framed body
            nonce = new_nonce()
    return f"{_BEGIN.format(nonce=nonce)}\n{text}\n{_END.format(nonce=nonce)}"


def unwrap(block: str) -> str | None:
    """Recover the framed text from a quarantine ``block`` iff its BEGIN/END nonces MATCH; return
    ``None`` if the block is malformed or its markers carry different nonces (a spoof). Round-trips:
    ``unwrap(quarantine(t)) == t``."""
    match = _BLOCK_RE.match(block)
    return match.group(2) if match is not None else None


__all__ = [
    "new_nonce",
    "quarantine",
    "unwrap",
]
