"""Pod 3.2d Feature Test Bundle — MCP capability integration (CANON S2, S9, S10).

Mutation-resistant assertions throughout.  The spike SC-5 invariants are:

  M1 — Bound tier is ``"external"`` EVEN when descriptor annotations scream read-only/safe.
       Mutation pin: iterate fake annotations through every tier literal — bound tier must
       never vary with annotations.
  M2 — ``MCPTierPolicy.overrides`` is honoured for an explicitly-overridden tool.
  M3 — Name collision with a native capability → registered as ``mcp.<server>.<name>``;
       native capability object identity is unchanged.
  M4 — ``invoke`` round-trips args to ``client.call_tool`` and returns its result.
  M5 — ``additionalProperties`` / schema is carried on the capability so the 3.2b router
       can validate args (exact schema dict asserted, not just presence).
  M6 — Tags always contain ``"mcp"`` and ``"untrusted-source"`` regardless of policy.
  M7 — ``description`` attribute exists on ``MCPCapability`` (duck-typed 3.1 contract).
"""

from __future__ import annotations

from typing import Any, get_args

import pytest

from cogworx.capability.base import PermissionTier
from cogworx.capability.mcp import (
    MCPTierPolicy,
    MCPToolDescriptor,
    bind_mcp_tools,
)
from cogworx.capability.registry import Registry, RegistryError, function_capability
from cogworx.testing.mcp_fakes import FakeMCPClient

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


def _descriptor(
    name: str,
    *,
    schema: dict[str, Any] | None = None,
    annotations: dict[str, Any] | None = None,
) -> MCPToolDescriptor:
    return MCPToolDescriptor(
        name=name,
        description=f"desc:{name}",
        input_schema=schema or {"type": "object", "properties": {"x": {"type": "string"}}},
        annotations=annotations or {},
    )


def _policy(**overrides: PermissionTier) -> MCPTierPolicy:
    return MCPTierPolicy(overrides=overrides)


# ---------------------------------------------------------------------------
# M1 — Tier is always "external" even with aggressive annotations
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m1_tier_external_despite_annotations() -> None:
    """Tier is ``"external"`` regardless of server-declared annotations (S9 hard rule)."""
    aggressive_annotations = {
        "readOnlyHint": True,
        "safe": True,
        "tier": "read",
        "trusted": True,
        "idempotent": True,
    }
    desc = _descriptor("safe_tool", annotations=aggressive_annotations)
    client = FakeMCPClient([desc], call_results={"safe_tool": "result"})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    assert len(names) == 1
    cap = reg.get(names[0])
    assert cap.tier == "external", (
        f"tier must be 'external'; got {cap.tier!r} — annotations must not influence tier (S9)"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("annotation_tier", get_args(PermissionTier))
async def test_m1_mutation_pin_all_tier_literals(annotation_tier: str) -> None:
    """Mutation pin: bound tier never varies with the annotation-declared tier literal."""
    desc = _descriptor("tool", annotations={"tier": annotation_tier, "safe": True})
    client = FakeMCPClient([desc], call_results={"tool": None})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])
    assert cap.tier == "external", (
        f"annotation tier={annotation_tier!r} must not influence bound tier; got {cap.tier!r}"
    )


# ---------------------------------------------------------------------------
# M2 — MCPTierPolicy.overrides is honoured
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m2_overrides_honoured() -> None:
    """An explicit override in MCPTierPolicy is applied to the named tool."""
    descs = [
        _descriptor("trusted_read"),
        _descriptor("default_tool"),
    ]
    client = FakeMCPClient(descs, call_results={"trusted_read": "r", "default_tool": "d"})
    policy = MCPTierPolicy(overrides={"trusted_read": "read"}, default_tier="external")
    reg = Registry()
    await bind_mcp_tools(reg, client, server_name="srv", tier_policy=policy)

    assert reg.get("mcp.srv.trusted_read").tier == "read"
    assert reg.get("mcp.srv.default_tool").tier == "external"


@pytest.mark.asyncio
async def test_m2_overrides_write_tier() -> None:
    """Override to ``"write"`` tier is accepted."""
    desc = _descriptor("write_tool")
    client = FakeMCPClient([desc], call_results={"write_tool": None})
    policy = MCPTierPolicy(overrides={"write_tool": "write"})
    reg = Registry()
    await bind_mcp_tools(reg, client, server_name="srv", tier_policy=policy)
    assert reg.get("mcp.srv.write_tool").tier == "write"


# ---------------------------------------------------------------------------
# M3 — Namespace prevents shadowing; native capability identity is unchanged
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m3_no_native_shadowing() -> None:
    """MCP tool with same name as a native cap uses the namespace; native identity unchanged."""

    async def _native(x: str) -> str:
        return x

    native_cap = function_capability(_native, name="read_file", tier="read")
    reg = Registry()
    reg.register(native_cap)
    native_id = id(reg.get("read_file"))

    desc = _descriptor("read_file")
    client = FakeMCPClient([desc], call_results={"read_file": "mcp_result"})
    await bind_mcp_tools(reg, client, server_name="fs", tier_policy=MCPTierPolicy())

    # Native unchanged
    assert id(reg.get("read_file")) == native_id
    # MCP tool registered under the namespaced name
    mcp_cap = reg.get("mcp.fs.read_file")
    assert mcp_cap is not None
    assert mcp_cap.tier == "external"


@pytest.mark.asyncio
async def test_m3_namespaced_name_format() -> None:
    """Bound names follow ``mcp.<server>.<tool>`` exactly."""
    descs = [_descriptor("alpha"), _descriptor("beta.sub")]
    client = FakeMCPClient(descs, call_results={"alpha": 1, "beta.sub": 2})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="my_server", tier_policy=MCPTierPolicy())
    assert set(names) == {"mcp.my_server.alpha", "mcp.my_server.beta.sub"}


# ---------------------------------------------------------------------------
# M4 — invoke round-trips args and returns the client result
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m4_invoke_round_trips_args() -> None:
    """invoke passes args through to call_tool and returns its result."""
    desc = _descriptor("echo")
    client = FakeMCPClient([desc], call_results={"echo": {"status": "ok"}})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])

    result = await cap.invoke({"x": "hello"})

    assert result == {"status": "ok"}
    assert len(client.calls) == 1
    call_name, call_args = client.calls[0]
    assert call_name == "echo"
    assert call_args == {"x": "hello"}


@pytest.mark.asyncio
async def test_m4_multiple_tools_route_independently() -> None:
    """Each MCPCapability routes to its own tool name (not the namespaced name)."""
    descs = [_descriptor("tool_a"), _descriptor("tool_b")]
    client = FakeMCPClient(descs, call_results={"tool_a": "result_a", "tool_b": "result_b"})
    reg = Registry()
    await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())

    r_a = await reg.get("mcp.srv.tool_a").invoke({"x": "a"})
    r_b = await reg.get("mcp.srv.tool_b").invoke({"x": "b"})

    assert r_a == "result_a"
    assert r_b == "result_b"
    assert [c[0] for c in client.calls] == ["tool_a", "tool_b"]


# ---------------------------------------------------------------------------
# M5 — input_schema is carried (for the 3.2b router's arg validation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m5_schema_carried() -> None:
    """input_schema from the descriptor is present on the bound capability (hardened).

    B2: bind_mcp_tools applies harden_input_schema so the bound schema gains
    ``additionalProperties: false`` at the top level (the absent-injection discipline).
    The descriptor's original schema is NOT mutated (deep-copy contract).
    """
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }
    desc = _descriptor("read_file", schema=schema)
    client = FakeMCPClient([desc], call_results={"read_file": None})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])

    # Hardened schema includes everything from the original PLUS additionalProperties: false.
    expected_hardened = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
        "additionalProperties": False,
    }
    assert cap.input_schema == expected_hardened, (
        "bound schema must be hardened (additionalProperties: false added);"
        f" got {cap.input_schema!r}"
    )
    # Original descriptor schema must NOT be mutated (deep-copy discipline).
    assert "additionalProperties" not in schema, (
        "harden_input_schema must deep-copy; original descriptor schema must be untouched"
    )


@pytest.mark.asyncio
async def test_m5_schema_is_a_dict() -> None:
    """input_schema is a plain dict (not a reference to the descriptor's Mapping)."""
    desc = _descriptor("tool")
    client = FakeMCPClient([desc], call_results={"tool": None})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])
    assert isinstance(cap.input_schema, dict)


# ---------------------------------------------------------------------------
# M6 — Tags always include "mcp" and "untrusted-source"
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m6_mandatory_tags_present() -> None:
    """``"mcp"`` and ``"untrusted-source"`` are always in the registered tags."""
    desc = _descriptor("tool")
    client = FakeMCPClient([desc], call_results={"tool": None})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    tags = reg.tags_of(names[0])
    assert "mcp" in tags
    assert "untrusted-source" in tags


@pytest.mark.asyncio
async def test_m6_extra_tags_appended() -> None:
    """``extra_tags`` from the policy are appended alongside the mandatory tags."""
    desc = _descriptor("tool")
    client = FakeMCPClient([desc], call_results={"tool": None})
    policy = MCPTierPolicy(extra_tags=("my-server", "experimental"))
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=policy)
    tags = reg.tags_of(names[0])
    assert "mcp" in tags
    assert "untrusted-source" in tags
    assert "my-server" in tags
    assert "experimental" in tags


# ---------------------------------------------------------------------------
# M7 — description attribute present (duck-typed 3.1 contract)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_m7_description_attribute_present() -> None:
    """MCPCapability carries a ``description`` attribute for exposed_specs duck-typing."""
    desc = _descriptor("tool")
    client = FakeMCPClient([desc], call_results={"tool": None})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])
    assert hasattr(cap, "description")
    assert isinstance(cap.description, str)
    assert cap.description == "desc:tool"


# ---------------------------------------------------------------------------
# Additional: duplicate registration raises RegistryError (namespacing guard)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_duplicate_server_name_raises() -> None:
    """Registering the same server twice raises RegistryError on the duplicate name."""
    desc = _descriptor("tool")
    client = FakeMCPClient([desc], call_results={"tool": None})
    reg = Registry()
    await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    with pytest.raises(RegistryError, match="already registered"):
        await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())


# ---------------------------------------------------------------------------
# Additional: empty server → empty tuple returned
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_server_returns_empty_tuple() -> None:
    """bind_mcp_tools on a server with no tools returns an empty tuple."""
    client = FakeMCPClient([], call_results={})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="empty", tier_policy=MCPTierPolicy())
    assert names == ()


# ---------------------------------------------------------------------------
# Additional: MCPCapability is a valid Capability (Protocol check)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_mcp_capability_satisfies_protocol() -> None:
    """MCPCapability satisfies the Capability Protocol at runtime."""
    from cogworx.capability.base import Capability

    desc = _descriptor("tool")
    client = FakeMCPClient([desc], call_results={"tool": None})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])
    assert isinstance(cap, Capability)


# ---------------------------------------------------------------------------
# B2 — harden_input_schema: recursive additionalProperties injection (Pod 3.2 B2)
# ---------------------------------------------------------------------------


from cogworx.capability.schema import harden_input_schema  # noqa: E402


def test_harden_top_level_object_without_additionalprops() -> None:
    """Top-level object schema without additionalProperties gets it injected as False."""
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "required": ["x"]}
    hardened = harden_input_schema(schema)
    assert hardened["additionalProperties"] is False
    # Original must be untouched (deep copy).
    assert "additionalProperties" not in schema


def test_harden_existing_additionalprops_true_is_respected() -> None:
    """An explicitly-set additionalProperties=True must NOT be overwritten."""
    schema = {
        "type": "object",
        "properties": {"x": {"type": "string"}},
        "additionalProperties": True,
    }
    hardened = harden_input_schema(schema)
    assert hardened["additionalProperties"] is True


def test_harden_existing_additionalprops_schema_is_respected() -> None:
    """An explicitly-set additionalProperties as a schema dict must NOT be overwritten."""
    extra = {"type": "string"}
    schema = {"type": "object", "properties": {}, "additionalProperties": extra}
    hardened = harden_input_schema(schema)
    assert hardened["additionalProperties"] == extra


def test_harden_nested_object_property() -> None:
    """A nested object property without additionalProperties gets it injected."""
    schema = {
        "type": "object",
        "properties": {
            "outer": {"type": "string"},
            "inner": {
                "type": "object",
                "properties": {"key": {"type": "integer"}},
            },
        },
    }
    hardened = harden_input_schema(schema)
    assert hardened["additionalProperties"] is False
    inner = hardened["properties"]["inner"]
    assert inner["additionalProperties"] is False, "nested object must also be hardened"


def test_harden_bare_type_object_no_properties() -> None:
    """Bare {'type': 'object'} hardens to additionalProperties: false (accepts NO args)."""
    schema: dict[str, Any] = {"type": "object"}
    hardened = harden_input_schema(schema)
    assert hardened["additionalProperties"] is False


def test_harden_deep_nesting_propagates() -> None:
    """Three levels deep: every absent object node gets additionalProperties: false."""
    schema = {
        "type": "object",
        "properties": {
            "level1": {
                "type": "object",
                "properties": {
                    "level2": {
                        "type": "object",
                        "properties": {"leaf": {"type": "string"}},
                    }
                },
            }
        },
    }
    hardened = harden_input_schema(schema)
    assert hardened["additionalProperties"] is False
    l1 = hardened["properties"]["level1"]
    assert l1["additionalProperties"] is False
    l2 = l1["properties"]["level2"]
    assert l2["additionalProperties"] is False


@pytest.mark.asyncio
async def test_b2_mcp_top_level_no_additionalprops_gets_hardened() -> None:
    """MCP tool with properties but no additionalProperties → hardened at top level."""
    desc = MCPToolDescriptor(
        name="tool1",
        description="test",
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
        annotations={},
    )
    client = FakeMCPClient([desc], call_results={"tool1": "ok"})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])
    assert cap.input_schema.get("additionalProperties") is False, (
        "MCP cap missing additionalProperties must be hardened to False"
    )


@pytest.mark.asyncio
async def test_b2_mcp_nested_object_property_gets_hardened() -> None:
    """MCP tool with nested object property lacking additionalProperties → both levels hardened."""
    desc = MCPToolDescriptor(
        name="tool2",
        description="test",
        input_schema={
            "type": "object",
            "properties": {
                "config": {
                    "type": "object",
                    "properties": {"timeout": {"type": "integer"}},
                }
            },
        },
        annotations={},
    )
    client = FakeMCPClient([desc], call_results={"tool2": "ok"})
    reg = Registry()
    names = await bind_mcp_tools(reg, client, server_name="srv", tier_policy=MCPTierPolicy())
    cap = reg.get(names[0])
    assert cap.input_schema.get("additionalProperties") is False
    config_schema = cap.input_schema["properties"]["config"]
    assert config_schema.get("additionalProperties") is False, (
        "nested object in MCP schema must also be hardened"
    )


@pytest.mark.asyncio
async def test_b2_mcp_bare_object_no_args_accepted() -> None:
    """MCP tool with bare {'type':'object'} → hardened;
    dispatch with smuggled keys → invalid_args."""
    from cogworx.capability.policy import StageToolPolicy, TaintState, ToolGate
    from cogworx.capability.router import route_tool_calls
    from cogworx.model.base import ToolCall

    desc = MCPToolDescriptor(
        name="bare_tool",
        description="bare",
        input_schema={"type": "object"},
        annotations={},
    )
    call_log: list[dict[str, Any]] = []

    class _LoggingClient:
        async def list_tools(self) -> list[MCPToolDescriptor]:
            return [desc]

        async def call_tool(self, name: str, args: object) -> str:
            call_log.append({"name": name, "args": args})
            return "reached"

    reg = Registry()
    policy = MCPTierPolicy(default_tier="external")
    names = await bind_mcp_tools(reg, _LoggingClient(), server_name="srv", tier_policy=policy)

    gate = ToolGate(
        reg,
        policy=StageToolPolicy(allowed_tiers=frozenset({"external"}), taint_drops_external=False),
        taint=TaintState(),
    )

    # Smuggled key → invalid_args (hardened schema rejects it); call_tool must NOT be reached.
    tc_bad = ToolCall(id="tc-bad", name=names[0], arguments={"__proto__": "evil"})
    results = await route_tool_calls(gate, reg, [tc_bad])
    assert results[0].status == "invalid_args", (
        f"bare-object hardened schema must reject smuggled keys; got {results[0].status!r}"
    )
    assert call_log == [], "call_tool must NOT be reached on invalid_args"

    # Clean call with empty args IS accepted by bare-object schema + additionalProperties:false.
    tc_clean = ToolCall(id="tc-ok", name=names[0], arguments={})
    results2 = await route_tool_calls(gate, reg, [tc_clean])
    assert results2[0].status == "ok", (
        f"empty-args dispatch on bare-object schema must succeed; got {results2[0].status!r}"
    )
    assert len(call_log) == 1
