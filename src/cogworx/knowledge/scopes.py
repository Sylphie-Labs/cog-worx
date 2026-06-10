"""Scope governance for the world model and user model (CANON S7).

Clones the SourceRegistry (Pod 2.3) discipline: a ScopeWriteToken is produced only by
ScopeRegistry.claim_write_token; model output cannot construct one structurally.

S7 — exactly one write-token per scope. First call to claim_write_token binds the owner;
any later call with a different owner raises ValueError.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field
from typing import Final, Literal

__all__ = [
    "DEFAULT_SCOPE",
    "Scope",
    "ScopeKind",
    "ScopeRegistry",
    "ScopeWriteToken",
]

ScopeKind = Literal["world", "user"]
DEFAULT_SCOPE: Final[str] = "agent"  # The unscoped default — requires no token


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


@dataclass(frozen=True)
class Scope:
    """Read-side scope handle. scope_id is computed from kind + ref, never caller-supplied."""

    kind: ScopeKind
    ref: str  # "global" for world kind; the user_id for user kind
    scope_id: str = field(init=False)

    def __post_init__(self) -> None:
        if self.kind == "user" and not self.ref.strip():
            raise ValueError("user scope requires a non-empty ref (user_id)")
        if self.kind == "world":
            object.__setattr__(self, "scope_id", "world")
        else:
            object.__setattr__(self, "scope_id", f"user:{_nfc(self.ref)}")


@dataclass(frozen=True)
class ScopeWriteToken:
    """S7 write-token. Produced ONLY by ScopeRegistry.claim_write_token.

    Direct construction is an explicit, reviewable breach of the single-writer contract.
    """

    scope: Scope
    owner: str  # the single authorized writer's name


class ScopeRegistry:
    """In-process ledger enforcing exactly-one-writer-per-scope (CANON S7).

    Idempotent identical declare; raises ValueError on conflicting owner claim.
    """

    def __init__(self) -> None:
        self._scopes: dict[str, Scope] = {}   # scope_id -> Scope
        self._owners: dict[str, str] = {}     # scope_id -> owner

    def declare(self, kind: ScopeKind, ref: str = "global") -> Scope:
        """Return a read-side Scope handle. Idempotent; no ownership claimed."""
        s = Scope(kind=kind, ref=ref)
        self._scopes.setdefault(s.scope_id, s)
        return self._scopes[s.scope_id]

    def claim_write_token(
        self, kind: ScopeKind, ref: str = "global", *, owner: str
    ) -> ScopeWriteToken:
        """Bind owner to this scope and return a write-token.

        First call binds owner; same-owner re-call is idempotent.
        Different-owner re-call raises ValueError (S7: exactly one writer per scope).
        """
        if not owner.strip():
            raise ValueError("owner must be non-empty")
        scope = self.declare(kind, ref)
        existing = self._owners.get(scope.scope_id)
        if existing is None:
            self._owners[scope.scope_id] = owner
        elif existing != owner:
            raise ValueError(
                f"Scope {scope.scope_id!r} already claimed by {existing!r}; "
                f"cannot claim with {owner!r} (S7: exactly one write-token per scope)"
            )
        return ScopeWriteToken(scope=scope, owner=owner)

    def writer_of(self, kind: ScopeKind, ref: str = "global") -> str | None:
        """Return the registered owner for this scope, or None if unclaimed."""
        scope = self.declare(kind, ref)
        return self._owners.get(scope.scope_id)
