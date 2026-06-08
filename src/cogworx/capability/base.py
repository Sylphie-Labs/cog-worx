"""Capability seam — security by structure (CANON S10).

Capabilities are least-privilege, permission-tiered tools. Tool routing happens in code, not as
model-emitted text; the permission tier is what lets the loop drop external-tier tools during read
phases to break the lethal trifecta structurally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

PermissionTier = Literal["read", "write", "external"]


@runtime_checkable
class Capability(Protocol):
    """A permission-tiered, code-routed tool."""

    name: str
    tier: PermissionTier
    input_schema: Mapping[str, Any]

    async def invoke(self, args: Mapping[str, Any]) -> Any: ...


__all__ = [
    "Capability",
    "PermissionTier",
]
