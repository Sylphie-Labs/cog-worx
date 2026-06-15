"""MCP capability integration — Pod 3.2d (CANON S2, S9, S10).

cog-worx owns the ``MCPClient`` seam (S2 — own the loop; no SDK owns the control path).
Tier is ALWAYS assigned by ``MCPTierPolicy`` (framework code), never by server-declared
annotations (S9 — structure over prompting; never trust the model's / server's self-report).
Default tier is ``"external"`` + tag ``"untrusted-source"`` because an MCP tool crosses a
process/network boundary; conservative is the only honest default (S10).

Namespacing ``mcp.<server>.<tool>`` prevents MCP tools from shadowing native capabilities.

Contract changelog:
  - 2026-06-12 (Pod 3.2d §6.1): initial — MCPToolDescriptor, MCPClient, MCPTierPolicy,
    MCPCapability, bind_mcp_tools, StdioMCPClient (import-guarded).  Additive new module;
    no existing callers.
  - 2026-06-12 (Pod 3.2 B2): ``bind_mcp_tools`` now calls ``harden_input_schema`` on each
    tool's ``input_schema`` (the single recursive authority) so nested object properties in
    MCP schemas are hardened against extra-key injection, consistent with native caps.
    Additive: no existing callers are broken.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import field as _field
from typing import Any, Protocol, runtime_checkable

from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

from cogworx.capability.base import PermissionTier
from cogworx.capability.schema import harden_input_schema

# ---------------------------------------------------------------------------
# Frozen value types (server metadata — NOT used for tier assignment)
# ---------------------------------------------------------------------------


@dataclass(frozen=True, config=ConfigDict(arbitrary_types_allowed=True))
class MCPToolDescriptor:
    """The server's own description of one tool.

    This is pure metadata recorded as-is from the server.  ``annotations`` may
    contain anything the server declares (``readOnlyHint``, ``safe``, …).  None
    of it is trusted for security purposes (S9 — the tier is assigned by
    ``MCPTierPolicy``, never from this object).
    """

    name: str
    description: str
    input_schema: Mapping[str, Any]
    annotations: dict[str, Any]


# ---------------------------------------------------------------------------
# MCPClient Protocol — cog-worx owns this seam (S2)
# ---------------------------------------------------------------------------


@runtime_checkable
class MCPClient(Protocol):
    """Thin client seam for an MCP server connection.

    cog-worx owns this Protocol (S2 — own the loop).  The ``StdioMCPClient``
    below is one concrete implementation backed by the official ``mcp`` SDK;
    other transports implement the same two-method contract.
    """

    async def list_tools(self) -> Sequence[MCPToolDescriptor]: ...

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> Any: ...


# ---------------------------------------------------------------------------
# MCPTierPolicy — framework-side tier assignment (never from server claims)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MCPTierPolicy:
    """Framework-side tier assignment policy for one MCP server.

    ``overrides`` is a name-keyed map of per-tool tier overrides for tools that
    a developer has explicitly decided to trust at a higher tier.  Every tool
    not in ``overrides`` lands at ``default_tier``.

    ``extra_tags`` are propagated to every capability registered from this
    server (in addition to the mandatory ``"mcp"`` and ``"untrusted-source"``
    tags that are ALWAYS applied — they may not be suppressed here).

    S9 hard rule: tier comes from this policy object (framework code assigned
    at registration time by the developer).  A server declaring
    ``readOnlyHint: true`` / ``"safe": true`` / any other annotation NEVER
    elevates or changes the tier — annotations are recorded in
    ``MCPToolDescriptor.annotations`` for auditing only.
    """

    overrides: Mapping[str, PermissionTier] = _field(default_factory=dict)
    default_tier: PermissionTier = "external"
    extra_tags: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# MCPCapability — implements the Capability Protocol
# ---------------------------------------------------------------------------


class MCPCapability:
    """A ``Capability`` backed by an MCP tool call.

    Tier is assigned by ``MCPTierPolicy`` at registration time; it is never
    derived from the server's ``MCPToolDescriptor.annotations``.  The capability
    always carries tags ``("mcp", "untrusted-source", …)`` so the ``TaintState``
    latch fires on first invocation (S10).

    ``description`` is carried as a plain attribute so ``ToolGate.exposed_specs``
    can surface it via ``getattr(cap, "description", "")`` (Pod 3.1 duck-typed
    contract — additive on concrete classes only, not on the ``Capability``
    Protocol).
    """

    def __init__(
        self,
        *,
        name: str,
        description: str,
        tier: PermissionTier,
        input_schema: Mapping[str, Any],
        client: MCPClient,
        tool_name: str,
    ) -> None:
        self.name = name
        self.description = description
        self.tier = tier
        self.input_schema = dict(input_schema)
        self._client = client
        self._tool_name = tool_name

    async def invoke(self, args: Mapping[str, Any]) -> Any:
        """Delegate to ``client.call_tool`` with the original (un-namespaced) tool name."""
        return await self._client.call_tool(self._tool_name, args)


# ---------------------------------------------------------------------------
# bind_mcp_tools — the registration entry point
# ---------------------------------------------------------------------------


async def bind_mcp_tools(
    registry: Any,
    client: MCPClient,
    *,
    server_name: str,
    tier_policy: MCPTierPolicy,
) -> tuple[str, ...]:
    """Discover all tools from ``client`` and register them in ``registry``.

    Each tool is registered under the namespaced name ``mcp.<server_name>.<tool>``
    with tags ``("mcp", "untrusted-source", *tier_policy.extra_tags)``.  The tier
    is ``tier_policy.overrides.get(tool_name, tier_policy.default_tier)`` —
    NEVER from the server's ``MCPToolDescriptor.annotations``.

    Parameters
    ----------
    registry:
        A ``Registry`` instance.  Typed as ``Any`` to avoid a circular import
        between ``capability.mcp`` and ``capability.registry`` — callers import
        both explicitly.
    client:
        An ``MCPClient`` instance.
    server_name:
        Short identifier for the server (used in the namespaced name and for
        collision prevention).
    tier_policy:
        The framework-side tier-assignment policy for this server.

    Returns
    -------
    tuple[str, ...]
        The namespaced names of every capability registered (in list-tools order).

    S9 invariants (hard-coded):
      1. Tier from ``MCPTierPolicy`` only — server annotations are ignored for tier.
      2. Default tier ``"external"`` + tag ``"untrusted-source"`` — MCP crosses a
         process/network boundary; conservative is the only honest default.
      3. Namespace ``mcp.<server>.<tool>`` — prevents shadowing native capabilities.
      4. ``input_schema`` is carried so the 3.2b router can validate model args —
         it is NOT a security boundary (the tier gate is).
    """
    tags: tuple[str, ...] = ("mcp", "untrusted-source", *tier_policy.extra_tags)
    descriptors = await client.list_tools()
    registered: list[str] = []
    for descriptor in descriptors:
        tool_name = descriptor.name
        namespaced = f"mcp.{server_name}.{tool_name}"
        tier: PermissionTier = tier_policy.overrides.get(tool_name, tier_policy.default_tier)
        cap = MCPCapability(
            name=namespaced,
            description=descriptor.description,
            tier=tier,
            input_schema=harden_input_schema(descriptor.input_schema),
            client=client,
            tool_name=tool_name,
        )
        registry.register(cap, tags=tags)
        registered.append(namespaced)
    return tuple(registered)


# ---------------------------------------------------------------------------
# StdioMCPClient — optional real adapter (import-guarded, never at module top)
# ---------------------------------------------------------------------------


class StdioMCPClient:
    """A thin ``MCPClient`` over the official ``mcp`` package (stdio transport).

    The ``mcp`` package is an **optional** dependency (the ``mcp`` extra).  Its
    import is deliberately deferred to each method so this module can be imported
    without the SDK installed — exactly the pattern the provider adapters follow.

    Install with::

        pip install 'cog-worx[mcp]'

    Parameters
    ----------
    command:
        The executable to launch (e.g. ``"npx"``, ``"uvx"``, ``"python"``).
    args:
        Arguments passed to the command.
    """

    def __init__(self, command: str, args: Sequence[str] = ()) -> None:
        self._command = command
        self._args = list(args)
        self._session: Any = None
        self._context: Any = None

    async def __aenter__(self) -> StdioMCPClient:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'mcp' package is required for StdioMCPClient. "
                "Install it with: pip install 'cog-worx[mcp]'"
            ) from exc

        params = StdioServerParameters(command=self._command, args=self._args)
        self._context = stdio_client(params)
        read, write = await self._context.__aenter__()
        self._session = ClientSession(read, write)
        await self._session.__aenter__()
        await self._session.initialize()
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        if self._session is not None:
            await self._session.__aexit__(*exc_info)
        if self._context is not None:
            await self._context.__aexit__(*exc_info)

    async def list_tools(self) -> Sequence[MCPToolDescriptor]:
        """Return all tools advertised by the server."""
        try:
            from mcp.types import Tool as _MCPTool  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'mcp' package is required for StdioMCPClient. "
                "Install it with: pip install 'cog-worx[mcp]'"
            ) from exc

        if self._session is None:  # pragma: no cover
            raise RuntimeError("StdioMCPClient must be used as an async context manager")
        result = await self._session.list_tools()
        descriptors: list[MCPToolDescriptor] = []
        for tool in result.tools:
            annotations_dict: dict[str, Any] = {}
            raw_ann = getattr(tool, "annotations", None)
            if raw_ann is not None:
                if hasattr(raw_ann, "model_dump"):
                    annotations_dict = {
                        k: v for k, v in raw_ann.model_dump().items() if v is not None
                    }
                elif isinstance(raw_ann, dict):
                    annotations_dict = dict(raw_ann)
            descriptors.append(
                MCPToolDescriptor(
                    name=tool.name,
                    description=getattr(tool, "description", "") or "",
                    input_schema=tool.inputSchema if isinstance(tool.inputSchema, dict) else {},
                    annotations=annotations_dict,
                )
            )
        return descriptors

    async def call_tool(self, name: str, args: Mapping[str, Any]) -> Any:
        """Call a tool by its un-namespaced name and return the raw result."""
        try:
            from mcp.types import CallToolResult as _CallToolResult  # noqa: F401
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "The 'mcp' package is required for StdioMCPClient. "
                "Install it with: pip install 'cog-worx[mcp]'"
            ) from exc

        if self._session is None:  # pragma: no cover
            raise RuntimeError("StdioMCPClient must be used as an async context manager")
        return await self._session.call_tool(name, dict(args))


__all__ = [
    "MCPCapability",
    "MCPClient",
    "MCPTierPolicy",
    "MCPToolDescriptor",
    "StdioMCPClient",
    "bind_mcp_tools",
]
