"""Source registry — framework-assigned source identity (CANON S9, Pod 2.3).

The structural S9 precondition for episodic extraction: source_id and source_authority are
MINTED by framework code via SourceDeclaration, NEVER taken from model output. The extraction
core accepts SourceDeclaration (not str) so model text cannot reach EvidenceEvent.source_id
without an explicit, reviewable type breach.

Clones ProcedureRegistry's shape (cogworx.knowledge.procedural_registry).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Literal

__all__ = [
    "SourceDeclaration",
    "SourceKind",
    "SourceRegistry",
]

SourceKind = Literal["human", "tool", "document", "agent", "system"]


def _nfc(ref: str) -> str:
    return unicodedata.normalize("NFC", ref)


@dataclass(frozen=True)
class SourceDeclaration:
    """The resolved identity of a declared source (framework-assigned, never model-chosen).

    ``source_id`` is deterministic: ``f"source:{kind}:{_nfc(ref)}"`` — two declarations of the
    same ``(kind, ref)`` across runs collide on the same id so evidence accumulates against one
    source node rather than minting duplicates.

    CONTRACT (S9): instances are produced only by :meth:`SourceRegistry.declare`; the extraction
    core accepts :class:`SourceDeclaration` (not ``str``) for evidence minting so model text
    cannot reach :attr:`~cogworx.knowledge.evidence.EvidenceEvent.source_id` without an
    explicit, reviewable type breach.
    """

    kind: SourceKind
    ref: str
    """Raw reference supplied at declaration: session id, tool id, document URL, user id, etc."""
    source_id: str = field(init=False)
    """Deterministic id: ``f"source:{kind}:{_nfc(ref)}"``.  Computed at construction; not a
    constructor parameter so callers cannot supply a hand-rolled string."""
    source_authority: float = field(default=1.0)
    """[0.0, 1.0]. Scaled by the write path from the caller's authority estimate."""

    def __post_init__(self) -> None:
        # frozen=True means we must use object.__setattr__ to write the computed field.
        object.__setattr__(self, "source_id", f"source:{self.kind}:{_nfc(self.ref)}")
        if not (0.0 <= self.source_authority <= 1.0):
            raise ValueError(
                f"source_authority must be in [0.0, 1.0]; got {self.source_authority!r}"
            )


class SourceRegistry:
    """A declaration-time source identity registry (framework-assigned source_id/source_authority).

    Build once at startup alongside the pathway registry: for each source the framework will stamp
    on evidence events, ``declare(kind, ref, authority=...)``. The episodic extraction core looks
    up a :class:`SourceDeclaration` and stamps its ``source_id`` / ``source_authority`` directly —
    model output never touches those fields.

    ``declare`` semantics:

    * Identical re-declare (same ``kind``, ``ref``, ``authority``) is idempotent — returns the
      existing :class:`SourceDeclaration`, no error.
    * Re-declaring the same ``(kind, ref)`` with a DIFFERENT ``authority`` raises
      :exc:`ValueError` (conflicting identity discipline; fail-loud, S9).
    """

    def __init__(self) -> None:
        self._declarations: dict[tuple[SourceKind, str], SourceDeclaration] = {}
        self._by_id: dict[str, SourceDeclaration] = {}

    def declare(
        self,
        kind: SourceKind,
        ref: str,
        *,
        authority: float = 1.0,
    ) -> SourceDeclaration:
        """Register ``(kind, ref)`` as a source with the given ``authority``.

        ``authority`` is validated to [0.0, 1.0] by :class:`SourceDeclaration.__post_init__`.
        Returns the resolved :class:`SourceDeclaration`.
        """
        declaration = SourceDeclaration(kind=kind, ref=ref, source_authority=authority)
        key: tuple[SourceKind, str] = (kind, _nfc(ref))
        existing = self._declarations.get(key)
        if existing is not None:
            if existing.source_authority != declaration.source_authority:
                raise ValueError(
                    f"source ({kind!r}, {ref!r}) is already declared with authority "
                    f"{existing.source_authority!r}; cannot re-declare with "
                    f"{declaration.source_authority!r} (conflicting identity discipline, S9)"
                )
            return existing
        self._declarations[key] = declaration
        self._by_id[declaration.source_id] = declaration
        return declaration

    def get(self, kind: SourceKind, ref: str) -> SourceDeclaration | None:
        """Return the declaration for ``(kind, ref)``, or ``None`` if not registered."""
        return self._declarations.get((kind, _nfc(ref)))

    def get_by_id(self, source_id: str) -> SourceDeclaration | None:
        """Return the declaration for a ``source_id``, or ``None`` if not registered."""
        return self._by_id.get(source_id)

    def __contains__(self, key: tuple[SourceKind, str]) -> bool:
        kind, ref = key
        return (kind, _nfc(ref)) in self._declarations

    def __iter__(self) -> Iterator[SourceDeclaration]:
        return iter(self._declarations.values())

    def __len__(self) -> int:
        return len(self._declarations)
