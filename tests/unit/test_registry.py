"""S8/S10 tests: the capability registry is discoverable, filterable, and lesionable."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from cogworx.capability.base import Capability, PermissionTier
from cogworx.capability.registry import Registry, RegistryError, function_capability


class _StubCapability:
    def __init__(self, name: str, tier: PermissionTier = "read") -> None:
        self.name = name
        self.tier: PermissionTier = tier
        self.input_schema: Mapping[str, Any] = {}

    async def invoke(self, args: Mapping[str, Any]) -> Any:
        return args


def test_register_get_and_tags() -> None:
    registry = Registry()
    cap = _StubCapability("read_file")
    registry.register(cap, tags=("io", "read"))
    assert registry.get("read_file") is cap
    assert registry.tags_of("read_file") == ("io", "read")


def test_duplicate_registration_raises() -> None:
    registry = Registry()
    registry.register(_StubCapability("dup"))
    with pytest.raises(RegistryError):
        registry.register(_StubCapability("dup"))


def test_list_filters_by_all_tags() -> None:
    registry = Registry()
    registry.register(_StubCapability("a"), tags=("io", "read"))
    registry.register(_StubCapability("b"), tags=("io",))
    both = registry.list(tags=("io", "read"))
    assert tuple(c.name for c in both) == ("a",)
    io_only = registry.list(tags=("io",))
    assert tuple(c.name for c in io_only) == ("a", "b")


def test_disable_removes_from_dispatch_but_not_features() -> None:
    registry = Registry()
    registry.register(_StubCapability("lesion"))
    registry.disable("lesion")
    assert not registry.is_enabled("lesion")
    with pytest.raises(RegistryError):
        registry.get("lesion")
    assert registry.list() == ()
    assert tuple(c.name for c in registry.list(include_disabled=True)) == ("lesion",)
    assert tuple(c.name for c in registry.features()) == ("lesion",)


def test_enable_restores_dispatch() -> None:
    registry = Registry()
    cap = _StubCapability("lesion")
    registry.register(cap)
    registry.disable("lesion")
    registry.enable("lesion")
    assert registry.is_enabled("lesion")
    assert registry.get("lesion") is cap


def test_toggle_unknown_name_raises() -> None:
    registry = Registry()
    with pytest.raises(RegistryError):
        registry.disable("ghost")
    with pytest.raises(RegistryError):
        registry.enable("ghost")


def test_clear_empties_the_registry() -> None:
    registry = Registry()
    registry.register(_StubCapability("a"))
    registry.clear()
    assert registry.names() == ()
    assert registry.features() == ()


async def test_function_capability_builds_a_working_capability() -> None:
    async def echo(text: str, times: int = 1) -> str:
        return text * times

    cap: Capability = function_capability(echo, name="echo", tier="read")
    assert cap.name == "echo"
    assert cap.tier == "read"
    assert isinstance(cap.input_schema, dict)
    assert "text" in cap.input_schema["properties"]
    result = await cap.invoke({"text": "ab", "times": 3})
    assert result == "ababab"
