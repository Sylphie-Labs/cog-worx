"""Input-schema hardening — the single authority for additionalProperties injection (Pod 3.2 B2).

``harden_input_schema`` walks a JSON Schema recursively and injects
``additionalProperties: false`` at every object node where it is ABSENT — so extra model-supplied
keys are rejected by the framework-side jsonschema validation rather than silently forwarded to
``cap.invoke``.

Design rules (mythos ruling, B2):
  - An explicitly-present ``additionalProperties`` (``true`` or a sub-schema) is RESPECTED and
    never overwritten — this is the ABSENCE-only injection discipline.
  - Bare ``{"type": "object"}`` (no ``properties``) hardens to ``{"type": "object",
    "additionalProperties": false}`` — accepts NO args (correct fail-safe for under-specified
    schemas).
  - Every object-shaped sub-schema is walked: ``properties`` values, ``items``, ``prefixItems``,
    ``$defs`` / ``definitions``, ``anyOf`` / ``oneOf`` / ``allOf`` branches, and
    ``patternProperties`` values.
  - The original schema is DEEP-COPIED before mutation so the caller's object is never modified
    in-place (important for MCP descriptors whose ``input_schema`` is carried on the descriptor
    and reused).

Contract changelog:
  - 2026-06-12 (Pod 3.2 B2): new module.  Replaces the manual top-level injection in
    ``registry.py:_build_input_schema`` (which only patched the top level) with a single
    recursive authority.  Additive: no existing callers outside this package.
"""

from __future__ import annotations

import copy
from typing import Any


def harden_input_schema(schema: Any) -> dict[str, Any]:
    """Deep-copy ``schema`` and inject ``additionalProperties: false`` at every absent object node.

    Parameters
    ----------
    schema:
        A JSON Schema mapping (dict).  Non-dict values are returned as-is (guard against
        malformed schemas arriving from MCP servers).

    Returns
    -------
    A new ``dict`` (deep copy) with ``additionalProperties: false`` injected at every object-
    shaped node where it was absent.  Existing values are never touched.
    """
    if not isinstance(schema, dict):
        return dict(schema) if hasattr(schema, "items") else {}
    result: dict[str, Any] = copy.deepcopy(schema)
    _harden_node(result)
    return result


def _is_object_node(node: dict[str, Any]) -> bool:
    """Return True if ``node`` is an object schema (has ``type=="object"`` or ``properties``)."""
    return node.get("type") == "object" or "properties" in node


def _harden_node(node: dict[str, Any]) -> None:
    """Recursively harden a schema node in-place (called on a deep copy)."""
    if not isinstance(node, dict):
        return

    # Inject at this level if it is an object node without an explicit additionalProperties.
    if _is_object_node(node) and "additionalProperties" not in node:
        node["additionalProperties"] = False

    # Walk properties values.
    for prop_schema in node.get("properties", {}).values():
        if isinstance(prop_schema, dict):
            _harden_node(prop_schema)

    # Walk patternProperties values.
    for prop_schema in node.get("patternProperties", {}).values():
        if isinstance(prop_schema, dict):
            _harden_node(prop_schema)

    # Walk items (array item schema — may be a single schema or a list in draft 4).
    items = node.get("items")
    if isinstance(items, dict):
        _harden_node(items)
    elif isinstance(items, list):
        for item in items:
            if isinstance(item, dict):
                _harden_node(item)

    # Walk prefixItems (JSON Schema draft 2020-12 tuple validation).
    for item in node.get("prefixItems", []):
        if isinstance(item, dict):
            _harden_node(item)

    # Walk $defs and definitions.
    for def_schema in node.get("$defs", {}).values():
        if isinstance(def_schema, dict):
            _harden_node(def_schema)
    for def_schema in node.get("definitions", {}).values():
        if isinstance(def_schema, dict):
            _harden_node(def_schema)

    # Walk anyOf / oneOf / allOf branches.
    for keyword in ("anyOf", "oneOf", "allOf"):
        for branch in node.get(keyword, []):
            if isinstance(branch, dict):
                _harden_node(branch)


__all__ = ["harden_input_schema"]
