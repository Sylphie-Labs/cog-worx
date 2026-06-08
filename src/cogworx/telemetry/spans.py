"""OpenTelemetry spans — swappable sink (CANON S3-adjacent observability).

A thin span helper over the OTel API. The framework never configures a global ``TracerProvider`` or
exporter — that is app/ops config. With no provider installed OTel uses a no-op tracer, which is the
correct framework default. Generalized from tess ``telemetry.py``.
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator
from typing import Any

from opentelemetry import trace
from opentelemetry.trace import Span, Tracer

_TRACER_NAME = "cogworx"


def get_tracer() -> Tracer:
    """The single framework-wide tracer; callers carry specifics on span attributes."""
    return trace.get_tracer(_TRACER_NAME)


@contextlib.contextmanager
def span(name: str, **attributes: Any) -> Iterator[Span]:
    """Start ``name`` as the current span, set OTel-permitted attributes, record exceptions.

    Only ``str``/``bool``/``int``/``float`` attribute values are set; anything else is skipped (OTel
    rejects non-primitive attribute values). Exceptions are recorded on the span and re-raised.
    """
    tracer = get_tracer()
    with tracer.start_as_current_span(name) as active:
        for key, value in attributes.items():
            if isinstance(value, str | bool | int | float):
                active.set_attribute(key, value)
        try:
            yield active
        except Exception as exc:
            active.record_exception(exc)
            raise


__all__ = [
    "get_tracer",
    "span",
]
