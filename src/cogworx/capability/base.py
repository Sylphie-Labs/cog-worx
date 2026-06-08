"""Capability seam — security by structure (CANON S10).

Capabilities are least-privilege, permission-tiered tools. Tool routing happens in code, not as
model-emitted text; the permission tier is what lets the loop drop external-tier tools during read
phases to break the lethal trifecta structurally.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, Protocol, runtime_checkable

PermissionTier = Literal["read", "write", "external"]


class CapabilityUnavailable(Exception):
    """A dispatched capability cannot be invoked — no registry, unknown, or disabled/lesioned.

    The single uniform signal a degradation-aware stage catches to return ``Degraded`` (S8),
    regardless of *why* the capability is unavailable. The lesion switch (``Registry.disable``) and
    a missing registry both surface here, so graceful degradation does not depend on the caller
    knowing which failure mode occurred.
    """


@runtime_checkable
class Capability(Protocol):
    """A permission-tiered, code-routed tool."""

    name: str
    tier: PermissionTier
    input_schema: Mapping[str, Any]

    async def invoke(self, args: Mapping[str, Any]) -> Any: ...


__all__ = [
    "Capability",
    "CapabilityUnavailable",
    "PermissionTier",
]
