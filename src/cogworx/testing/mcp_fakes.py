"""Test double for the MCP client seam — Pod 3.2d (CANON S9, S12).

``FakeMCPClient`` is the deterministic, service-free stand-in for ``MCPClient``
used by the 3.2d unit tests.  Constructable with a list of ``MCPToolDescriptor``s
(including ones with aggressive ``annotations`` like ``{"readOnlyHint": True,
"safe": True}``) and a scripted ``call_tool`` return map.  Records all calls so
tests can assert invoke behaviour without a real MCP server.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from cogworx.capability.mcp import MCPToolDescriptor


class FakeMCPClient:
    """In-memory ``MCPClient`` double for unit tests.

    Parameters
    ----------
    descriptors:
        The tool descriptors this fake advertises via ``list_tools``.
        Include descriptors with aggressive ``annotations`` (e.g.
        ``{"readOnlyHint": True, "safe": True, "tier": "read"}``) to
        verify that the framework ignores them for tier assignment (S9).
    call_results:
        A mapping from tool name to the value ``call_tool`` returns.
        Unlisted tool names raise ``KeyError`` (intentional — tests that
        call an unscripted tool name have a bug).
    """

    def __init__(
        self,
        descriptors: Sequence[MCPToolDescriptor],
        *,
        call_results: Mapping[str, Any] | None = None,
    ) -> None:
        self._descriptors = list(descriptors)
        self._call_results: dict[str, Any] = dict(call_results) if call_results else {}
        # Records every (name, args) pair so tests can assert call behaviour.
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def list_tools(self) -> Sequence[MCPToolDescriptor]:
        """Return the scripted descriptors (no network, no process)."""
        return tuple(self._descriptors)

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> Any:
        """Return the scripted result for ``name``; record the call."""
        result = self._call_results[name]
        self.calls.append((name, dict(args)))
        return result


__all__ = ["FakeMCPClient"]
