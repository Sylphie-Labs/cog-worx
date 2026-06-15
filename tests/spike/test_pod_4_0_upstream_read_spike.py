"""Pod 4.0 spike (S12) — the F5 upstream-read seam: ``StageContext.last_output``.

Falsifiable criterion (architect R5 — the crash-correctness of the journal walk is the thing most
likely to be silently wrong): ``last_output`` returns THIS refine cycle's committed verdict, and a
fresh context after a (simulated) mid-cycle resume reads the SAME durable answer — never an
in-process value.

The phase's other Pod 4.0 spike criteria are covered by the verification unit suite: the verdict is
structurally ground-truth-typed (``test_verification_outcome`` / ``Verdict.is_executable``), the
oracle registry degrades to its always-on fallback (``test_verification_oracle``), and the
quarantine channel round-trips (``test_verification_quarantine``).

Per the test-hang discipline: run this file SINGLE and timeout-wrapped — never the whole tests/spike
tier (known cross-test resource leak). This spike is in-memory only (no substrate, no model call).
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from cogworx.claims.provenance import Artifact, Provenance
from cogworx.cost.budget import BudgetGuard
from cogworx.loop.result import Transition
from cogworx.loop.stage import StageContext
from cogworx.runtime.context import RunContext
from cogworx.substrate.journal import StepRecord
from cogworx.testing.doubles import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore
from cogworx.testing.fake_model import echo_model

pytestmark = pytest.mark.spike

_T0 = datetime(2026, 6, 15, tzinfo=UTC)
_RUN = "run-dialectic"


def _artifact(label: str) -> Artifact:
    return Artifact(
        kind="verdict",
        produced_by="antithesis",
        provenance=Provenance(source="inference", confidence=0.5, recorded_at=_T0),
        data={"label": label},
    )


def _step(step_index: int, stage_name: str, label: str, to: str) -> StepRecord:
    return StepRecord(
        run_id=_RUN,
        step_index=step_index,
        stage_name=stage_name,
        result=Transition(to=to, output=_artifact(label)),
        committed_at=_T0,
    )


async def _seed_two_cycles(journal: InMemoryJournal) -> None:
    """Commit two dialectic cycles: thesis/experiment/antithesis/evaluate(refine), then again."""
    await journal.start_run(
        _RUN, "sess", pathway_id="dialectic", pathway_version=1, pathway_fingerprint="fp"
    )
    steps = [
        _step(0, "thesis", "thesis-A", "experiment"),
        _step(1, "experiment", "exp-A", "antithesis"),
        _step(2, "antithesis", "cycle-A", "evaluate"),
        _step(3, "evaluate", "eval-A", "thesis"),  # routed to refine
        _step(4, "thesis", "thesis-B", "experiment"),
        _step(5, "experiment", "exp-B", "antithesis"),
        _step(6, "antithesis", "cycle-B", "evaluate"),
    ]
    for step in steps:
        await journal.commit_step(step)


def _ctx(journal: InMemoryJournal) -> RunContext:
    return RunContext(
        run_id=_RUN,
        session_id="sess",
        model=echo_model("unused"),
        journal=journal,
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        budget=BudgetGuard(),
    )


async def test_last_output_returns_current_cycle_verdict() -> None:
    journal = InMemoryJournal()
    await _seed_two_cycles(journal)

    out = await _ctx(journal).last_output("antithesis")

    assert out is not None
    assert out.data["label"] == "cycle-B"  # the MOST RECENT antithesis, not cycle-A


async def test_last_output_is_crash_correct_after_resume() -> None:
    # Mid-cycle resume: a FRESH context (zero in-process state) over the SAME durable journal must
    # read the same committed answer — the walk is journal-backed, never in-memory (S6).
    journal = InMemoryJournal()
    await _seed_two_cycles(journal)

    resumed = _ctx(journal)  # brand-new RunContext, as after a cold resume
    out = await resumed.last_output("antithesis")

    assert out is not None and out.data["label"] == "cycle-B"


async def test_last_output_none_for_uncommitted_stage() -> None:
    journal = InMemoryJournal()
    await _seed_two_cycles(journal)
    assert await _ctx(journal).last_output("never-ran") is None


async def test_last_output_none_for_unknown_run() -> None:
    # Empty journal: no run record → None, total and never raising (S8).
    assert await _ctx(InMemoryJournal()).last_output("antithesis") is None


def test_runcontext_satisfies_stagecontext_with_last_output() -> None:
    # The new Protocol member is structurally present on the concrete context.
    assert isinstance(_ctx(InMemoryJournal()), StageContext)
