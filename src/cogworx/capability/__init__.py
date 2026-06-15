"""Capability layer (CANON S10): least-privilege, permission-tiered, code-routed tools."""

from __future__ import annotations

from cogworx.capability.base import Capability, CapabilityUnavailable, PermissionTier
from cogworx.capability.policy import (
    DEFAULT_TOOL_POLICY,
    ApprovalRequired,
    StageToolPolicy,
    TaintState,
    TierViolation,
    ToolArgumentError,
    ToolGate,
)
from cogworx.capability.registry import Registry, RegistryError, function_capability
from cogworx.capability.router import (
    ToolLoopLimit,
    ToolResult,
    ToolStatus,
    dispatch_one,
    route_tool_calls,
    run_tool_loop,
    tool_result_messages,
)

__all__ = [
    "DEFAULT_TOOL_POLICY",
    "ApprovalRequired",
    "Capability",
    "CapabilityUnavailable",
    "PermissionTier",
    "Registry",
    "RegistryError",
    "StageToolPolicy",
    "TaintState",
    "TierViolation",
    "ToolArgumentError",
    "ToolGate",
    "ToolLoopLimit",
    "ToolResult",
    "ToolStatus",
    "dispatch_one",
    "function_capability",
    "route_tool_calls",
    "run_tool_loop",
    "tool_result_messages",
]
