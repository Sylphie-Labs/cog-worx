"""InMemoryJournal taint contract — SC-J1 through SC-J5 (CANON S6, S10).

Tests that ``InMemoryJournal`` correctly implements the taint contract:
``set_run_tainted`` (idempotent monotonic False→True) and ``load_run`` rehydration of
the ``tainted`` field. No real DB adapter; no spike marker.
"""

from __future__ import annotations

from cogworx.testing.doubles import InMemoryJournal

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RUN_ID = "test-run-1"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


async def _start(journal: InMemoryJournal) -> None:
    await journal.start_run(
        RUN_ID,
        "session-1",
        pathway_id="path",
        pathway_version=1,
        pathway_fingerprint="fp",
    )


# ---------------------------------------------------------------------------
# SC-J1 — set_run_tainted flips False → True
# ---------------------------------------------------------------------------


async def test_set_run_tainted_flips_to_true() -> None:
    """set_run_tainted transitions tainted from False to True.

    Mutation killed: an impl that writes to a local variable instead of the stored
    _RunLog, leaving load_run returning tainted=False after the call.
    """
    j = InMemoryJournal()
    await _start(j)
    state_before = await j.load_run(RUN_ID)
    assert state_before is not None
    assert state_before.tainted is False
    await j.set_run_tainted(RUN_ID)
    state_after = await j.load_run(RUN_ID)
    assert state_after is not None
    assert state_after.tainted is True


# ---------------------------------------------------------------------------
# SC-J2 — idempotent double-call
# ---------------------------------------------------------------------------


async def test_set_run_tainted_idempotent() -> None:
    """A second set_run_tainted call must not raise and must leave tainted=True.

    Mutation killed: an impl that raises or resets tainted on a repeat call (non-idempotent).
    """
    j = InMemoryJournal()
    await _start(j)
    await j.set_run_tainted(RUN_ID)
    await j.set_run_tainted(RUN_ID)  # second call must not raise
    state = await j.load_run(RUN_ID)
    assert state is not None
    assert state.tainted is True


# ---------------------------------------------------------------------------
# SC-J3 — load_run rehydrates tainted=True
# ---------------------------------------------------------------------------


async def test_load_run_rehydrates_tainted_true() -> None:
    """load_run returns a RunState with tainted=True after set_run_tainted.

    Mutation killed: load_run ignoring the stored tainted field and always returning
    tainted=False (field not wired into RunState construction).
    """
    j = InMemoryJournal()
    await _start(j)
    await j.set_run_tainted(RUN_ID)
    state = await j.load_run(RUN_ID)
    assert state is not None
    assert state.tainted is True


# ---------------------------------------------------------------------------
# SC-J4 — load_run rehydrates tainted=False (never tainted)
# ---------------------------------------------------------------------------


async def test_load_run_rehydrates_tainted_false() -> None:
    """load_run returns tainted=False for a run that was never tainted.

    Mutation killed: an impl that always returns tainted=True regardless of whether
    set_run_tainted was called (overly broad taint).
    """
    j = InMemoryJournal()
    await _start(j)
    state = await j.load_run(RUN_ID)
    assert state is not None
    assert state.tainted is False


# ---------------------------------------------------------------------------
# SC-J5 — fresh run defaults to untainted
# ---------------------------------------------------------------------------


async def test_fresh_run_default_untainted() -> None:
    """A freshly started run must default to tainted=False with no set_run_tainted call.

    Mutation killed: _RunLog initialising tainted=True by default (every run would be
    born tainted, breaking the monotonic False→True invariant).
    """
    j = InMemoryJournal()
    await _start(j)
    # No set_run_tainted call
    state = await j.load_run(RUN_ID)
    assert state is not None
    assert state.tainted is False
