# Session 2026-06-08 — Phase 0: Foundations

**Outcome:** Phase 0 complete and CANON-COMPLIANT. The cross-pod seams are frozen, the Test Kit
ships, the walking skeleton runs, the real polyglot substrate is up, and **Spike Suite 1 passed**
(hardened against a red-team pass). **Phase 1 is unblocked.**

## What landed (commit order on `main`)

1. **Scaffold** — `pyproject` (src-layout, hatchling, Py 3.13, ruff + mypy --strict + pytest), `nox`
   runner, GitHub Actions CI, typed package (`py.typed`). Provider SDKs behind an optional
   `[providers]` extra (the loop + spike run on the Test Kit `ReplayModel`).
2. **Seams, pass A (contract types)** — `Model` (S4), `Claim`/`Provenance`/`Artifact` (S5),
   coordination `Event` + `EVENT_BOUNDARY_MAP` + `Request`/`Reply`/`WriteToken` (S7), `Stage`/
   `StageContext` + `StageResult` union, three thin Store seams (S3), `Capability` (S10).
3. **Seams, pass B (behavioral)** — `BudgetGuard` (S11), `StageGraph` (DAG + termination-by-
   construction) + `Loop`, the toggleable `Registry` (S8/S10), telemetry span helper.
4. **Seam refinement** — `StepRecord` carries the committed `StageResult` (the control decision S6
   resume needs), which forced splitting the result union into the leaf `loop/result.py`.
5. **Test Kit + walking skeleton** — `ReplayModel`, in-memory Store doubles, `RunContext`, the
   `Engine` (intake → one model call → journal commit → done), reference agent, fixtures.
6. **Invariant suites + chaos test** — reusable S1/S5/S6/S9 helpers + Journal wrappers
   (`CommitSpyJournal`, `CrashAfterStepJournal`); kill-mid-run durability test.
7. **Substrate + adapters** — docker-compose (Neo4j + one Postgres = Timescale + pgvector), the
   `TimescaleJournal` / `Neo4jGraphStore` / `PgLatentStore` adapters; 15 integration tests green.
8. **Spike Suite 1 + S8 fix** — the gate's 3 criteria on the live substrate; unified dispatch to a
   single `CapabilityUnavailable` so lesioned capabilities degrade uniformly.
9. **Red-team hardening** — see below.

## The red-team save (why the gate is real)

The red-teamer found **Spike Suite 1 was largely vacuous**: the chaos graph crashed *after* the
model stage's transition committed, so `Engine.resume` forward-jumped past it and the journal-replay
branch never ran — "0 model calls on resume" was a topological accident. Fix: **`resume` re-drives
from `graph.entry`** and replays committed steps from the journal (the faithful S6 semantics). Every
gate assertion is now **mutation-resistant** (disabling replay / removing `ON CONFLICT` / steering on
model text / no-op'ing the lesion switch each make the corresponding test FAIL). CANON review then
returned **COMPLIANT** on S1–S12.

## Carry-forward into Phase 1 (do not lose these)

- **Cyclic step-ids.** The engine keys journal steps by `step_id == stage_name`, so loop-back/cyclic
  pathways would replay forever. The Spine's "graph of stages by default" needs **per-visit step
  ids** before cyclic pathways can be authored. (`runtime/engine.py`.)
- **Cross-process resume.** `Engine.resume` holds the graph in-process (`self._graphs[run_id]`) and
  raises `ResumeError` otherwise. The S6 proof is **in-process** resume over durable rows; cold
  cross-process resume (rebuild the graph from a registry/journal) is Phase 1 — and the chaos
  harness's `register_graph` seam should be replaced by genuine rehydration then.
- **Durable timers.** The `set_timer`/`due_timers` seam + `cogworx_timers` table exist, but the
  **sweeper / retry / pause / fire-and-forget** machinery is Phase 1 — the table's existence is not
  the feature being done.
- **Capability `description`.** Dropped from `function_capability` as a Phase-0 no-op; add it to the
  `Capability` contract when Phase 3 tool-routing needs it.

## Verification

`uv run nox -s lint typecheck test` → ruff + mypy --strict (58 files) + **69 unit** green.
`docker compose up -d` then `uv run pytest -m integration` → **15** green; `-m spike` → **4** green.
Local substrate maps alternate host ports via `.env` (sibling stacks hold 7474/7687/5432).
