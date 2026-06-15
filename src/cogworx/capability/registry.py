"""Feature registry — flat, discoverable, lesionable (CANON S8, S10).

A toggleable registry of ``Capability`` objects. ``disable`` is the lesion switch (S8): a disabled
capability is not dispatchable but stays enumerable via ``features()``, which is the auto-enrolment
hook the Test Kit uses to apply invariant suites to every registered feature. Generalized from tess
``tess/tools/registry.py``.

Contract changelog:
  - 2026-06-12 (Pod 3.2b §6.1): ``_build_input_schema`` now injects ``additionalProperties: false``
    into the pydantic-derived schema so extra model-supplied keys are REJECTED by framework-side
    jsonschema validation instead of silently splatted into ``**kwargs``.  Additive: no existing
    callers are broken (the schema becomes stricter, which is the desired direction).
  - 2026-06-12 (Pod 3.2 B2): replaced the manual top-level injection with ``harden_input_schema``
    from ``cogworx.capability.schema`` — the single recursive hardening authority that closes the
    nested-object gap for free.  Additive: no existing callers are broken.
"""

from __future__ import annotations

import inspect
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Any, cast, get_type_hints

from pydantic import ConfigDict, Field, create_model

from cogworx.capability.base import Capability, PermissionTier
from cogworx.capability.schema import harden_input_schema


class RegistryError(Exception):
    """Raised on duplicate registration, lookup of an absent/disabled name, or unknown toggle."""


class Registry:
    """A flat registry of permission-tiered capabilities with a per-name lesion switch."""

    def __init__(self) -> None:
        self._capabilities: dict[str, Capability] = {}
        self._tags: dict[str, tuple[str, ...]] = {}
        self._disabled: set[str] = set()

    def register(self, capability: Capability, *, tags: Sequence[str] = ()) -> None:
        if capability.name in self._capabilities:
            raise RegistryError(f"capability {capability.name!r} already registered")
        self._capabilities[capability.name] = capability
        self._tags[capability.name] = tuple(tags)

    def get(self, name: str) -> Capability:
        if name not in self._capabilities:
            raise RegistryError(f"unknown capability {name!r}")
        if name in self._disabled:
            raise RegistryError(f"capability {name!r} is disabled")
        return self._capabilities[name]

    def list(
        self, *, tags: Sequence[str] = (), include_disabled: bool = False
    ) -> tuple[Capability, ...]:
        wanted = set(tags)
        result: list[Capability] = []
        for name, capability in self._capabilities.items():
            if not include_disabled and name in self._disabled:
                continue
            if wanted and not wanted.issubset(self._tags[name]):
                continue
            result.append(capability)
        return tuple(result)

    def tags_of(self, name: str) -> tuple[str, ...]:
        if name not in self._capabilities:
            raise RegistryError(f"unknown capability {name!r}")
        return self._tags[name]

    def enable(self, name: str) -> None:
        if name not in self._capabilities:
            raise RegistryError(f"unknown capability {name!r}")
        self._disabled.discard(name)

    def disable(self, name: str) -> None:
        if name not in self._capabilities:
            raise RegistryError(f"unknown capability {name!r}")
        self._disabled.add(name)

    def is_enabled(self, name: str) -> bool:
        if name not in self._capabilities:
            raise RegistryError(f"unknown capability {name!r}")
        return name not in self._disabled

    def names(self, *, include_disabled: bool = True) -> tuple[str, ...]:
        return tuple(
            name for name in self._capabilities if include_disabled or name not in self._disabled
        )

    def clear(self) -> None:
        self._capabilities.clear()
        self._tags.clear()
        self._disabled.clear()

    def features(self) -> tuple[Capability, ...]:
        return tuple(self._capabilities.values())


class _FunctionCapability:
    """A ``Capability`` backed by a plain async function with an auto-derived input schema."""

    def __init__(
        self,
        fn: Callable[..., Awaitable[Any]],
        *,
        name: str,
        tier: PermissionTier,
        input_schema: Mapping[str, Any],
    ) -> None:
        self.name = name
        self.tier = tier
        self.input_schema = input_schema
        self._fn = fn

    async def invoke(self, args: Mapping[str, Any]) -> Any:
        return await self._fn(**args)


def function_capability(
    fn: Callable[..., Awaitable[Any]],
    *,
    name: str,
    tier: PermissionTier,
) -> Capability:
    """Build a ``Capability`` from an async function, deriving ``input_schema`` from its hints.

    The schema is generated from the function's type-hinted parameters via ``pydantic.create_model``
    so it cannot drift out of sync with the signature (the tess pattern). ``*args``/``**kwargs`` and
    un-annotated parameters are rejected — capabilities need a closed, typed parameter list.

    (A human-facing ``description`` belongs on the ``Capability`` contract itself; it is added in
    Phase 3 when tool-routing needs it, not bolted on here.)
    """
    schema = _build_input_schema(fn, name)
    return _FunctionCapability(fn, name=name, tier=tier, input_schema=schema)


def _build_input_schema(fn: Callable[..., Awaitable[Any]], name: str) -> dict[str, Any]:
    sig = inspect.signature(fn)
    hints = get_type_hints(fn, include_extras=True)
    fields: dict[str, Any] = {}
    for param_name, param in sig.parameters.items():
        if param.kind in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD):
            raise RegistryError(
                f"capability {name!r}: parameter {param_name!r} is *args/**kwargs; "
                "capabilities need a closed parameter list"
            )
        if param_name not in hints:
            raise RegistryError(
                f"capability {name!r}: parameter {param_name!r} has no type annotation"
            )
        annotation = hints[param_name]
        if param.default is inspect.Parameter.empty:
            fields[param_name] = (annotation, Field(...))
        else:
            fields[param_name] = (annotation, Field(default=param.default))

    model = create_model(
        f"{name}_Input",
        __config__=ConfigDict(arbitrary_types_allowed=True),
        **fields,
    )
    raw = model.model_json_schema()
    schema: dict[str, Any] = cast(dict[str, Any], raw)
    schema.pop("title", None)
    # harden_input_schema (the single recursive authority, B2) injects additionalProperties:false
    # at every absent object node — top-level AND nested — so extra model-supplied keys are
    # rejected by framework-side jsonschema validation rather than splatted into **kwargs (S9).
    return harden_input_schema(schema)


__all__ = [
    "Registry",
    "RegistryError",
    "function_capability",
]
