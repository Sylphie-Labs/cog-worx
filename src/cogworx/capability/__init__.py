"""Capability layer (CANON S10): least-privilege, permission-tiered, code-routed tools."""

from __future__ import annotations

from cogworx.capability.base import Capability, PermissionTier
from cogworx.capability.registry import Registry, RegistryError, function_capability

__all__ = [
    "Capability",
    "PermissionTier",
    "Registry",
    "RegistryError",
    "function_capability",
]
