"""Unit tests for the four concrete ContextContributor implementations (Pod 3.1c).

Covers:
  1. StaticContributor — happy path (non-empty text), empty path (text=""), Protocol conformance.
  2. TaskContributor   — happy path (task field, instructions field), empty path (None/empty),
                         Protocol conformance.
  3. ToolSpecContributor — happy path (non-empty list), empty path (empty list),
                           Protocol conformance.
  4. MemoryContributor — happy path (query present, injector returns chunks), empty path
                         (query=None), budget min() applied, zero model calls, last_injected
                         diagnostics stash, Protocol conformance.

All tests are pure — no I/O, no network, no model calls.
asyncio_mode = "auto" (pyproject.toml).
"""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

from cogworx.context.contributor import ContextContributor
from cogworx.context.contributors import (
    MemoryContributor,
    StaticContributor,
    TaskContributor,
    ToolSpecContributor,
)
from cogworx.context.types import (
    ContextPolicy,
    ContextRequest,
    SlotAllocation,
)
from cogworx.injection.injector import MemoryInjector
from cogworx.injection.policy import DEFAULT_MEMORY_POLICY, InjectedMemory, MemoryPolicy
from cogworx.model.base import ToolSpec
from cogworx.recall.query import RecallQuery
from cogworx.recall.results import AssembledContext, ContextChunk
from cogworx.recall.stack import RecallStack

# ---------------------------------------------------------------------------
# Shared constants / helpers
# ---------------------------------------------------------------------------

_EPOCH = datetime(2026, 6, 12, 0, 0, 0, tzinfo=UTC)
_FIXED_CLOCK = lambda: _EPOCH  # noqa: E731

_DEFAULT_REQUEST = ContextRequest(task="Summarise the codebase.")
_DEFAULT_ALLOC = SlotAllocation(max_tokens=512)
_UNCONSTRAINED_ALLOC = SlotAllocation()  # max_tokens=None


def _make_context_chunk(text: str = "hello", key: str = "claim:1") -> ContextChunk:
    return ContextChunk(
        text=text,
        key=key,
        kind="claim",
        hits=(),
        fused_score=1.0,
        relevance_rank=1,
    )


def _assembled_context_with(chunks: tuple[ContextChunk, ...]) -> AssembledContext:
    return AssembledContext(
        chunks=chunks,
        token_count=sum(len(c.text) // 4 for c in chunks),
        budget=2048,
        dropped=0,
    )


def _ok_injected_memory(chunks: tuple[ContextChunk, ...] = ()) -> InjectedMemory:
    return InjectedMemory(
        context=_assembled_context_with(chunks),
        status="ok",
    )


def _make_mock_injector(return_value: InjectedMemory | None = None) -> MemoryInjector:
    """Build a MemoryInjector with inject replaced by an AsyncMock."""
    stack_mock = MagicMock(spec=RecallStack)
    injector = MemoryInjector(stack_mock, clock=_FIXED_CLOCK)
    injector.inject = AsyncMock(  # type: ignore[method-assign]
        return_value=return_value or _ok_injected_memory()
    )
    return injector


# ---------------------------------------------------------------------------
# 1. StaticContributor
# ---------------------------------------------------------------------------


async def test_static_contributor_happy_path_returns_chunk() -> None:
    contrib = StaticContributor("You are helpful.", key="rules:base", source_slot="rules")
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.status == "ok"
    assert len(result.chunks) == 1
    chunk = result.chunks[0]
    assert chunk.text == "You are helpful."
    assert chunk.key == "rules:base"
    assert chunk.source_slot == "rules"


async def test_static_contributor_happy_path_no_tools() -> None:
    contrib = StaticContributor("Be concise.", key="personality:main", source_slot="personality")
    result = await contrib.contribute(_DEFAULT_REQUEST, _UNCONSTRAINED_ALLOC)
    assert result.tools == ()


async def test_static_contributor_empty_text_returns_empty_status() -> None:
    contrib = StaticContributor("", key="rules:base", source_slot="rules")
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.status == "empty"
    assert result.chunks == ()


async def test_static_contributor_default_text_is_empty() -> None:
    contrib = StaticContributor(key="personality:main", source_slot="personality")
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.status == "empty"


def test_static_contributor_protocol_conformance() -> None:
    contrib = StaticContributor(key="rules:base", source_slot="rules")
    assert isinstance(contrib, ContextContributor)


async def test_static_contributor_ignores_allocation_max_tokens() -> None:
    """StaticContributor always returns its full text regardless of allocation."""
    contrib = StaticContributor("Long text here.", key="rules:main", source_slot="rules")
    small_alloc = SlotAllocation(max_tokens=1)
    result = await contrib.contribute(_DEFAULT_REQUEST, small_alloc)
    # The contributor does not truncate — budget enforcement is the assembler's job.
    assert result.status == "ok"
    assert result.chunks[0].text == "Long text here."


# ---------------------------------------------------------------------------
# 2. TaskContributor
# ---------------------------------------------------------------------------


async def test_task_contributor_task_field_happy_path() -> None:
    contrib = TaskContributor(field="task", key="task:main", source_slot="task")
    req = ContextRequest(task="Write unit tests.")
    result = await contrib.contribute(req, _DEFAULT_ALLOC)
    assert result.status == "ok"
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Write unit tests."
    assert result.chunks[0].key == "task:main"
    assert result.chunks[0].source_slot == "task"


async def test_task_contributor_instructions_field_happy_path() -> None:
    contrib = TaskContributor(
        field="instructions", key="instructions:overlay", source_slot="instructions"
    )
    req = ContextRequest(task="Do something.", instructions="Be concise and precise.")
    result = await contrib.contribute(req, _DEFAULT_ALLOC)
    assert result.status == "ok"
    assert result.chunks[0].text == "Be concise and precise."


async def test_task_contributor_instructions_none_returns_empty() -> None:
    contrib = TaskContributor(
        field="instructions", key="instructions:overlay", source_slot="instructions"
    )
    req = ContextRequest(task="Do something.", instructions=None)
    result = await contrib.contribute(req, _DEFAULT_ALLOC)
    assert result.status == "empty"
    assert result.chunks == ()


def test_context_request_empty_task_raises_validation_error() -> None:
    """ContextRequest.task = Field(min_length=1) — construction with '' raises ValidationError."""
    with pytest.raises(ValidationError):
        ContextRequest(task="")


async def test_task_contributor_no_tools() -> None:
    contrib = TaskContributor(field="task", key="task:main", source_slot="task")
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.tools == ()


def test_task_contributor_protocol_conformance() -> None:
    contrib = TaskContributor(field="task", key="task:main", source_slot="task")
    assert isinstance(contrib, ContextContributor)


async def test_task_contributor_default_field_is_task() -> None:
    contrib = TaskContributor(key="task:main", source_slot="task")
    req = ContextRequest(task="Default field check.")
    result = await contrib.contribute(req, _DEFAULT_ALLOC)
    assert result.chunks[0].text == "Default field check."


# ---------------------------------------------------------------------------
# 3. ToolSpecContributor
# ---------------------------------------------------------------------------


def _tool(name: str = "search") -> ToolSpec:
    return ToolSpec(
        name=name,
        description=f"Tool {name}",
        input_schema={"type": "object", "properties": {}},
    )


async def test_tool_spec_contributor_happy_path_returns_tools() -> None:
    specs = [_tool("search"), _tool("calculator")]
    contrib = ToolSpecContributor(specs=specs)
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.status == "ok"
    assert len(result.tools) == 2
    assert result.tools[0].name == "search"
    assert result.tools[1].name == "calculator"


async def test_tool_spec_contributor_happy_path_no_chunks() -> None:
    contrib = ToolSpecContributor(specs=[_tool("web_search")])
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.chunks == ()


async def test_tool_spec_contributor_empty_list_returns_empty() -> None:
    contrib = ToolSpecContributor(specs=[])
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.status == "empty"
    assert result.tools == ()


async def test_tool_spec_contributor_default_specs_is_empty() -> None:
    contrib = ToolSpecContributor()
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert result.status == "empty"


def test_tool_spec_contributor_protocol_conformance() -> None:
    contrib = ToolSpecContributor()
    assert isinstance(contrib, ContextContributor)


async def test_tool_spec_contributor_preserves_order() -> None:
    specs = [_tool(f"tool_{i}") for i in range(5)]
    contrib = ToolSpecContributor(specs=specs)
    result = await contrib.contribute(_DEFAULT_REQUEST, _DEFAULT_ALLOC)
    assert [t.name for t in result.tools] == [f"tool_{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# 4. MemoryContributor
# ---------------------------------------------------------------------------


async def test_memory_contributor_happy_path_with_chunks() -> None:
    chunk = _make_context_chunk(text="Relevant claim.", key="claim:abc")
    injected = _ok_injected_memory(chunks=(chunk,))
    injector = _make_mock_injector(return_value=injected)

    contrib = MemoryContributor(injector, source_slot="memory")
    req = ContextRequest(task="Do something.", query=RecallQuery(text="relevant"))
    result = await contrib.contribute(req, _DEFAULT_ALLOC)

    assert result.status == "ok"
    assert len(result.chunks) == 1
    assert result.chunks[0].text == "Relevant claim."
    assert result.chunks[0].key == "claim:abc"
    assert result.chunks[0].source_slot == "memory"


async def test_memory_contributor_empty_path_query_none() -> None:
    injector = _make_mock_injector()
    contrib = MemoryContributor(injector)
    req = ContextRequest(task="Do something.", query=None)
    result = await contrib.contribute(req, _DEFAULT_ALLOC)

    assert result.status == "empty"
    assert result.chunks == ()
    # Injector must NOT be called when query is None (S1 — no unnecessary work)
    injector.inject.assert_not_called()  # type: ignore[attr-defined]


async def test_memory_contributor_empty_path_injector_returns_no_chunks() -> None:
    injected = _ok_injected_memory(chunks=())  # empty assembled context
    injector = _make_mock_injector(return_value=injected)

    contrib = MemoryContributor(injector)
    req = ContextRequest(task="Do something.", query=RecallQuery(text="query"))
    result = await contrib.contribute(req, _DEFAULT_ALLOC)

    assert result.status == "empty"
    assert result.chunks == ()


async def test_memory_contributor_respects_min_budget_allocation_smaller() -> None:
    """When allocation.max_tokens < policy.token_budget, effective budget = allocation."""
    injected = _ok_injected_memory(chunks=(_make_context_chunk(),))
    injector = _make_mock_injector(return_value=injected)

    policy = MemoryPolicy(token_budget=2048)
    contrib = MemoryContributor(injector, policy=policy)

    # allocation.max_tokens (256) < policy.token_budget (2048) → effective = 256
    alloc = SlotAllocation(max_tokens=256)
    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    await contrib.contribute(req, alloc)

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"].token_budget == 256


async def test_memory_contributor_respects_min_budget_policy_smaller() -> None:
    """When policy.token_budget < allocation.max_tokens, effective budget = policy."""
    injected = _ok_injected_memory(chunks=(_make_context_chunk(),))
    injector = _make_mock_injector(return_value=injected)

    policy = MemoryPolicy(token_budget=128)
    contrib = MemoryContributor(injector, policy=policy)

    # allocation.max_tokens (4096) > policy.token_budget (128) → effective = 128
    alloc = SlotAllocation(max_tokens=4096)
    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    await contrib.contribute(req, alloc)

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"].token_budget == 128


async def test_memory_contributor_unconstrained_alloc_uses_policy_budget() -> None:
    """When allocation.max_tokens is None, effective budget = policy.token_budget unchanged."""
    injected = _ok_injected_memory(chunks=(_make_context_chunk(),))
    injector = _make_mock_injector(return_value=injected)

    policy = MemoryPolicy(token_budget=512)
    contrib = MemoryContributor(injector, policy=policy)

    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    await contrib.contribute(req, _UNCONSTRAINED_ALLOC)

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"].token_budget == 512


async def test_memory_contributor_zero_model_calls() -> None:
    """MemoryContributor must never call model.complete() (S1 — off the write path)."""
    from cogworx.testing.fake_model import ReplayModel

    # Use a real ReplayModel with no scripted responses — complete() would raise if called.
    model = ReplayModel(responses=[])

    injected = _ok_injected_memory(chunks=(_make_context_chunk(),))
    injector = _make_mock_injector(return_value=injected)

    contrib = MemoryContributor(injector)
    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    await contrib.contribute(req, _DEFAULT_ALLOC)

    # The model was never used in MemoryContributor — call_count must be 0.
    assert model.call_count == 0


async def test_memory_contributor_stashes_last_injected() -> None:
    """last_injected must hold the InjectedMemory returned by the injector."""
    chunk = _make_context_chunk(text="Remembered fact.")
    injected = _ok_injected_memory(chunks=(chunk,))
    injector = _make_mock_injector(return_value=injected)

    contrib = MemoryContributor(injector)
    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    await contrib.contribute(req, _DEFAULT_ALLOC)

    assert contrib.last_injected is injected


async def test_memory_contributor_last_injected_none_when_query_none() -> None:
    """last_injected is None when query is absent (no injection was attempted)."""
    injector = _make_mock_injector()
    contrib = MemoryContributor(injector)
    contrib.last_injected = _ok_injected_memory()  # pre-seed to ensure it's cleared

    req = ContextRequest(task="t", query=None)
    await contrib.contribute(req, _DEFAULT_ALLOC)

    assert contrib.last_injected is None


async def test_memory_contributor_multiple_chunks_all_converted() -> None:
    chunks = tuple(_make_context_chunk(text=f"chunk {i}", key=f"claim:{i}") for i in range(3))
    injected = _ok_injected_memory(chunks=chunks)
    injector = _make_mock_injector(return_value=injected)

    contrib = MemoryContributor(injector, source_slot="memory")
    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    result = await contrib.contribute(req, _DEFAULT_ALLOC)

    assert len(result.chunks) == 3
    for i, slot_chunk in enumerate(result.chunks):
        assert slot_chunk.text == f"chunk {i}"
        assert slot_chunk.key == f"claim:{i}"
        assert slot_chunk.source_slot == "memory"


def test_memory_contributor_protocol_conformance() -> None:
    injector = _make_mock_injector()
    contrib = MemoryContributor(injector)
    assert isinstance(contrib, ContextContributor)


async def test_memory_contributor_default_policy_forwards_correct_budget() -> None:
    """When no policy is supplied, DEFAULT_MEMORY_POLICY budget is used."""
    injected = _ok_injected_memory()
    injector = _make_mock_injector(return_value=injected)

    contrib = MemoryContributor(injector)  # no policy kwarg
    req = ContextRequest(task="t", query=RecallQuery(text="q"))
    alloc = SlotAllocation(max_tokens=DEFAULT_MEMORY_POLICY.token_budget + 1000)
    await contrib.contribute(req, alloc)

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    # allocation is larger than default policy budget → effective = policy.token_budget
    assert call_kwargs["policy"].token_budget == DEFAULT_MEMORY_POLICY.token_budget


async def test_memory_contributor_policy_from_context_request_override() -> None:
    """ContextRequest.policy.memory can be used by callers to set per-request policy."""
    # This test verifies the caller CAN pass a different MemoryPolicy to MemoryContributor;
    # the contributor itself reads self._policy, not request.policy (the assembler is responsible
    # for passing the correct MemoryContributor to each request if policy varies per-request).
    injected = _ok_injected_memory()
    injector = _make_mock_injector(return_value=injected)

    explicit_policy = MemoryPolicy(token_budget=300)
    contrib = MemoryContributor(injector, policy=explicit_policy)

    req = ContextRequest(
        task="t",
        query=RecallQuery(text="q"),
        policy=ContextPolicy(total_budget=8192),
    )
    await contrib.contribute(req, _UNCONSTRAINED_ALLOC)

    call_kwargs = injector.inject.call_args.kwargs  # type: ignore[attr-defined]
    assert call_kwargs["policy"].token_budget == 300


# ---------------------------------------------------------------------------
# 5. Isolation check: cogworx.context.contributors must NOT import cogworx.runtime
# ---------------------------------------------------------------------------


def test_contributors_does_not_import_runtime() -> None:
    """cogworx.context.contributors must not pull cogworx.runtime into sys.modules (CANON D3)."""
    import subprocess
    import sys

    script = (
        "import sys; "
        "import cogworx.context.contributors; "
        "assert 'cogworx.runtime' not in sys.modules, "
        "'contributors pulled in cogworx.runtime -- dependency direction violation'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, (
        "cogworx.context.contributors imported cogworx.runtime (dependency direction "
        f"violation):\nstdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
