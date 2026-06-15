"""Unit tests for MemoryPolicy, InjectedMemory, and DEFAULT_MEMORY_POLICY (Pod 2.6).

All tests are pure — no I/O, no model calls, no async.
asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cogworx.injection.policy import (
    DEFAULT_MEMORY_POLICY,
    InjectedMemory,
    MemoryPolicy,
)
from cogworx.recall.results import AssembledContext, ChannelHit, ContextChunk

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_T0_STR = "2026-06-10T00:00:00+00:00"


def _empty_context(budget: int = 2048) -> AssembledContext:
    return AssembledContext(chunks=(), token_count=0, budget=budget, dropped=0)


def _chunk(key: str, kind: str = "claim", text: str = "some text") -> ContextChunk:
    hit = ChannelHit(channel="dense.claims", rank=1, raw_score=0.9)
    return ContextChunk(
        text=text,
        key=key,
        kind=kind,
        hits=(hit,),
        fused_score=0.9,
        relevance_rank=1,
    )


# ---------------------------------------------------------------------------
# 1. DEFAULT_MEMORY_POLICY has expected defaults
# ---------------------------------------------------------------------------


def test_default_memory_policy_token_budget() -> None:
    assert DEFAULT_MEMORY_POLICY.token_budget == 2048


def test_default_memory_policy_min_per_kind() -> None:
    assert DEFAULT_MEMORY_POLICY.min_per_kind == 0


def test_default_memory_policy_required_kinds_empty() -> None:
    assert DEFAULT_MEMORY_POLICY.required_kinds == ()


# ---------------------------------------------------------------------------
# 2. MemoryPolicy is frozen — mutation raises
# ---------------------------------------------------------------------------


def test_memory_policy_is_frozen_raises_on_attribute_set() -> None:
    policy = MemoryPolicy(token_budget=512)
    with pytest.raises((ValidationError, TypeError)):
        policy.token_budget = 1024  # type: ignore[misc]


def test_memory_policy_is_frozen_raises_on_required_kinds_set() -> None:
    policy = MemoryPolicy(required_kinds=("claim",))
    with pytest.raises((ValidationError, TypeError)):
        policy.required_kinds = ("episode",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 3. MemoryPolicy construction with custom values
# ---------------------------------------------------------------------------


def test_memory_policy_custom_values() -> None:
    policy = MemoryPolicy(token_budget=512, min_per_kind=2, required_kinds=("claim", "episode"))
    assert policy.token_budget == 512
    assert policy.min_per_kind == 2
    assert policy.required_kinds == ("claim", "episode")


# ---------------------------------------------------------------------------
# 4. InjectedMemory.status accepts valid literals
# ---------------------------------------------------------------------------


def test_injected_memory_status_ok() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok")
    assert mem.status == "ok"


def test_injected_memory_status_unwired() -> None:
    mem = InjectedMemory(context=_empty_context(), status="unwired")
    assert mem.status == "unwired"


def test_injected_memory_status_below_floor() -> None:
    mem = InjectedMemory(context=_empty_context(), status="below_floor")
    assert mem.status == "below_floor"


def test_injected_memory_status_invalid_raises() -> None:
    with pytest.raises(ValidationError):
        InjectedMemory(
            context=_empty_context(),
            status="not_a_valid_status",
        )


# ---------------------------------------------------------------------------
# 5. missing_kinds populated correctly on below_floor status
# ---------------------------------------------------------------------------


def test_injected_memory_missing_kinds_populated() -> None:
    mem = InjectedMemory(
        context=_empty_context(),
        status="below_floor",
        missing_kinds=("episode", "latent"),
    )
    assert "episode" in mem.missing_kinds
    assert "latent" in mem.missing_kinds


def test_injected_memory_missing_kinds_empty_on_ok() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok", missing_kinds=())
    assert mem.missing_kinds == ()


def test_injected_memory_missing_kinds_defaults_to_empty() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok")
    assert mem.missing_kinds == ()


# ---------------------------------------------------------------------------
# 6. kinds_present computed from chunks correctly
# ---------------------------------------------------------------------------


def test_kinds_present_empty_when_no_chunks() -> None:
    mem = InjectedMemory(context=_empty_context(), status="unwired")
    assert mem.kinds_present == ()


def test_kinds_present_reflects_provided_value() -> None:
    """InjectedMemory.kinds_present is a stored field set by the injector, not auto-computed."""
    mem = InjectedMemory(
        context=_empty_context(),
        status="ok",
        kinds_present=("claim",),
    )
    assert mem.kinds_present == ("claim",)


def test_kinds_present_multiple_kinds() -> None:
    mem = InjectedMemory(
        context=_empty_context(),
        status="ok",
        kinds_present=("claim", "episode", "latent"),
    )
    assert set(mem.kinds_present) == {"claim", "episode", "latent"}


# ---------------------------------------------------------------------------
# 7. InjectedMemory is frozen
# ---------------------------------------------------------------------------


def test_injected_memory_is_frozen() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok")
    with pytest.raises((ValidationError, TypeError)):
        mem.status = "unwired"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# 8. InjectedMemory optional fields default correctly
# ---------------------------------------------------------------------------


def test_injected_memory_latent_uses_recorded_default_zero() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok")
    assert mem.latent_uses_recorded == 0


def test_injected_memory_record_use_error_default_none() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok")
    assert mem.record_use_error is None


def test_injected_memory_channel_status_default_empty() -> None:
    mem = InjectedMemory(context=_empty_context(), status="ok")
    assert mem.channel_status == ()
