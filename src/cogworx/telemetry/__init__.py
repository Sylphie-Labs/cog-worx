"""Telemetry layer: thin OTel span helpers with a swappable, app-configured sink."""

from __future__ import annotations

from cogworx.telemetry.spans import get_tracer, span

__all__ = [
    "get_tracer",
    "span",
]
