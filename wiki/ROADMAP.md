# cog-worx — ROADMAP (Build Order)

> **Conforms to the CANON** (`wiki/CANON.md`). Where this doc and the CANON conflict, the CANON wins.
> **Dependency-ordered, not calendar-ordered.** A feature advances when its **spike passes** (S12), not
> on a date. Phases are dependency layers; within a phase, feature pods run in parallel.

## How to read this

- **A "pod" is a feature.** Each feature is built behind a contract and tested **independently**.
- **A feature is "done" only when it passes its Feature Test Bundle** (see the Definition of Done at the
  bottom) **in isolation.** Independent testability == lesionability (S8): if it can't be isolated for a
  test, that's a coupling smell and it does not harden (S12).
- **Phases gate each other; features within a phase don't.** Don't open more pods than your contracts
  can support (see Parallelism, bottom).

---

## Phase 0 — Foundations (sequential · unblocks every pod)

The job of Phase 0 is to **freeze the seams** and build the **machinery that makes every later feature
independently testable**. Nothing in Phase 1+ starts until Phase 0's contracts are frozen and the
walking skeleton is green.

- [x] **Package skeleton** — `pyproject`, module layout, CI running `ruff` + `mypy --strict` + `pytest`.
- [x] **Freeze the cross-pod contracts (the seams):** `Model` interface (S4); `Stage`/`Loop` contract;
      the three **Store seams** — Neo4j graph, pgvector, Timescale journal (S3); `Capability` interface;
      the **event/coordination types** (port sylphie `packages/shared/src/types/event.types.ts`, S7);
      artifact/claim types carrying **provenance + epistemic typing** (S5).
- [x] **Substrate up:** docker-compose for Neo4j + **one Postgres cluster** (pgvector + TimescaleDB)
      (S3); plus **ephemeral test instances wiped per case** (port tess's isolated instances).
- [x] **The Test Kit** (the deliverable that makes heavy testing repeatable, not heroic): fake/replay
      `Model` client (port tess `llm`/`kg_stub`); ephemeral-substrate fixtures; the **reference agent**;
      the **invariant property-suites** (S1, S5, S6, S9) as reusable parametrized tests.
- [x] **Feature registry** — discoverable, toggleable features auto-enrolled in the relevant suites
      (generalize tess `tools/registry`). Toggleability is also S8.
- [x] **Walking skeleton** — the thinnest end-to-end loop (intake → one model call → journal commit →
      done) driven by one **trivial reference agent**.
- [x] **⛓ GATE — Spike Suite 1:** polyglot substrate capability + exactly-once resume (S6) + a first
      Lesion pass (S8). **Phase 1 is blocked until this passes.** ✅ **PASSED** (hardened against a
      red-team pass; CANON-reviewed COMPLIANT). **Phase 1 is unblocked.**

---

## Phase 1 — The Spine (depends on Phase 0)

The configurable control loop everything else attaches to.

> **Pod 1.0 (durable-execution core) — DONE** ✅ (red-team-hardened, CANON-COMPLIANT): per-visit
> `step_index` keying (cyclic pathways resume exactly-once), code-registered **pathway registry** +
> **true cold cross-process resume** (rehydrate the graph from the journal's `pathway_id`, guarded by a
> structural pathway fingerprint), a structural **step ceiling**, and persisted run status. The
> durability/chaos spike now covers cold resume + cyclic per-visit replay on the live Timescale journal.

> **Pod 1.1 (durable timers + sweeper + pause/resume) — DONE** ✅ (CANON-COMPLIANT, red-team-hardened
> over two passes): a 5th first-class `StageResult` kind **`Wait`** (parks a run on a durable, **leased**
> timer); a **`Sweeper`** that claims due timers and re-drives (no model, no commits); **pause/unpause**
> with cooperative step-boundary stops; and a run-status **CAS** giving exactly-once *execution* under
> concurrent drivers (commit-idempotency alone is not enough). Crashed-mid-drive auto-recovery
> (run-lease + reaper) is **deferred** to a future ops pod (single-active-driver bar; explicit `resume`
> recovers a dead `RUNNING` run — §3/S7). **Next:** 1.2 retries+FSM timeouts · 1.3 await-human resume ·
> 1.4 fire-and-forget.

> **Pod 1.2 (retries + per-stage FSM timeouts) — DONE** ✅ (CANON-COMPLIANT, red-team-hardened): a
> declarative per-stage **`RetryPolicy`** + durable retries — a **failed attempt commits nothing** (only
> a success-class result lands at `seq`, so exactly-once stays pristine), a per-`(run_id, step_index)`
> **attempt counter** survives crashes, backoff reuses the 1.1 timer/sweeper/CAS, and a distinct
> **`RETRYING`** status. **In-process timeouts** (`asyncio.wait_for`); exhaustion **degrades-onward** by
> default (policy-overridable to FAILED); non-retryable exceptions **propagate loud**. Red-team caught +
> fixed a HIGH: the new `exhausted_to` graph edge is now folded into the `pathway_fingerprint` so the S6
> resume divergence guard isn't blind to it. **Next:** 1.3 await-human resume · 1.4 fire-and-forget.

> **Pod 1.3 (durable await-human resume) — DONE** ✅ (CANON-COMPLIANT, red-team-hardened over two
> passes): `AwaitHuman` gained a `to` (a committed await-human now REPLAYS as a plain advance to `to`,
> mirroring `Wait`); a journal seam **`record_human_input`/`read_human_input`** keyed
> `(run_id, step_index)` (**first-answer-wins**, `ON CONFLICT DO NOTHING`, new `cogworx_human_inputs`
> table); pull-based **`ctx.read_human_input`** so a downstream stage routes **structurally** on the
> answer (S9), never on model text; and **`Engine.provide_human_input`** — record the
> `Provenance(source="human")` answer **BEFORE** the CAS `AWAITING_HUMAN→RUNNING` (S5/S6 hard
> ordering), then re-drive exactly-once with no model re-call. Red-team caught + fixed a MED: a stage
> routing to an **undeclared** `to` could strand a run mid-HITL — now a runtime **declared-route guard**
> (`result.to ∈ graph.edges_from(current)`, before commit, all `to`-bearing results) fails it loud at
> first execution. The exactly-once-EXECUTION docstring was scoped honestly (CAS-guarded entrypoints;
> `resume(RUNNING)` is the deferred run-lease/reaper gap). **Next:** 1.4 fire-and-forget.

> **Pod 1.4 (fire-and-forget with feedback) — DONE** ✅ (CANON-COMPLIANT WITH CONCERNS, red-team-
> hardened): `Engine.start()` backgrounds the existing `run()` on a tracked task and returns a
> **`RunHandle`** immediately (in-process only — the JOURNAL owns durability, S6); crash feedback
> via a new **`RUN_CRASHED`** event from the task done-callback (a dropped handle never hides a
> death); `aclose()` drain-loops every background task incl. mid-drain starts. Red-team caught 3
> real defects pre-validation: an **unguarded double-start** (two live drivers, double model call —
> now refused loud), a **pre-journal silent-loss window** (unknown pathway died backgrounded before
> `start_run` — now validated synchronously at `start()`), and **guard overreach** (the old
> start-scoped set missed sweeper-vs-resume + double-resume races — replaced by an in-process
> **drive mutex** registered by EVERY drive entrypoint, before the CAS). Also fixed: a latent
> import cycle (`import cogworx.runtime` first detonated; TYPE_CHECKING fix + 12 subprocess
> first-import regression tests). Cross-INSTANCE exclusion stays honestly deferred (run-lease/
> reaper ops pod). **Phase-1 spine features are all landed; next: the comprehensive Phase-1 gate
> on the live substrate, then Phase 2.**

- [ ] **Configurable graph loop** — DAG-of-stages + FSM-per-stage; **dev-authored pathways, graph by
      default**, not a fixed list. *(graph + dev-authored pathway registry done in 1.0; per-stage FSM
      retry/timeout transitions done in 1.2)*
- [ ] **Durability** — journaled resume, durable timers (`wake_at` + sweeper), **retries**,
      **pause/resume**, **fire-and-forget-with-feedback**, all on the Timescale journal (S6).
      *(cold/cyclic journaled resume done in 1.0; durable timers + sweeper + pause/resume done in 1.1;
      retries + per-stage timeouts done in 1.2; durable await-human resume done in 1.3;
      fire-and-forget with feedback done in 1.4)*
- [ ] **First-class transitions** — `await-human` and `degraded` (S8). *(both are first-class results;
      `Wait` added as a 5th first-class result in 1.1; durable await-human resume done in 1.3 —
      `AwaitHuman.to` + `provide_human_input`)*
- [ ] **⛓ Spike** — durability/chaos test: kill mid-step → resume → exactly-once, **no model re-call**.
      *(core PASSED on live substrate — cold + cyclic; the full gate spans the operation pods)*

---

## Phase 2 — Knowledge & Memory (the differentiator · first pod after the spine)

Deepest, riskiest (the polyglot recall bet), and the most differentiated value. Memory research is
already done in biz-firm, so this is port + generalize.

- [ ] **Procedural KG** (Neo4j) · **Entity KG** (Neo4j)
- [ ] **World model** (long-term memory) · **User model**
- [ ] **Episodic memory**
- [ ] **Latent space on pgvector** — hot/cold tiering (port sylphie)
- [ ] **Recall stack** — dense + BM25 + graph + temporal → rank fusion → rerank → token-budget assembly
- [ ] **Memory injection**
- [ ] **Conversation extraction / classification / storage**
- [ ] **Coherence** — async/batched reconciler; **surface, don't silently delete**
- [ ] **⛓ Spike** — recall-quality eval (LongMemEval-style) + provenance/epistemic-typing invariant (S5)

---

## Phase 3 — Cognition (per step)

- [ ] **Model routing** — per-agent provider selection, graceful capability degradation (S4)
- [ ] **Context assembly** — per-stage, token budget, lost-in-the-middle ordering
- [ ] **Capabilities** — permission-tiered tools, **tool routing in code** (not model-as-text), **MCP**
- [ ] **Personality injection**
- [ ] **Internal unbreakable rules**
- [ ] **⛓ Spike** — structured-output reliability + security-by-structure (S10) + cost guards (S11)

---

## Phase 4 — Verification & Truth

- [ ] **Oracle layer** — deterministic coding, math proofing, simulated chemistry, simulated physics
      (port tess `tools/` + `oracles/`)
- [ ] **Honest failure**
- [ ] **Thesis/antithesis team decision-making**
- [ ] **⛓ Spike** — antithesis catch-rate eval + never-trust-self-report invariant (S9)

---

## Phase 5 — Perception (largely independent · can start earlier with spare hands)

- [ ] **Computer vision + multimodal fusion** (port sylphie `perception-service` /
      `cobeing/layer2_perception`)
- [ ] **Visual working memory**
- [ ] **⛓ Spike** — perception accuracy on a fixture set + degradation (S8)

---

## Cross-cutting — Operations (runs **alongside** every phase, not after)

- [ ] **Observability** — OTel spans (port tess `telemetry`)
- [ ] **Eval harness** — LLM-judge with bias controls (port tess `judge_calibration`)
- [ ] **Safety / permission tiers** (S10)
- [ ] **Graceful degradation / the Lesion Test matrix** (S8 — port sylphie)
- [ ] **Cost / budget ceilings** (S11)

---

## Deferred (per CANON §3 — do not build "as a foundation")

- The **connecting / orchestration system** — coordination is **contracts-only** for now (S7).
- Any **product** wrapper.
- The **`.claude` governance machinery** (`canon` agent, enforce/update-canon skills, hooks, spike
  suites) — Jim sets this up in-repo.
- **Auto prompt-optimizers** and **cross-agent learning** — unless the latest research pulls them
  forward.

---

## Definition of Done — the Feature Test Bundle

A feature does not harden (S12) until it passes, **in isolation**:

1. **Unit** — feature isolated, all deps stubbed.
2. **Deterministic integration** — feature wired into the walking-skeleton loop, everything else
   stubbed/replayed, the reference agent exercising it.
3. **Invariant / property tests (auto-applied)** — the CANON invariants the feature touches, checked
   mechanically: writes carry provenance + epistemic type (S5); nothing on the write path calls a model
   (S1); resume never re-calls the model (S6); no self-report used as a control signal (S9).
4. **Eval** — only if the feature is judgment/model-bearing (recall quality, oracle correctness,
   antithesis catch-rate), with LLM-judge bias controls.
5. **Degradation test** — feature off → system still runs (S8).

**Cadence:** the fast tiers (1–3, 5) are deterministic (stub/replay the model) and run on **every change
in CI**; evals (4) run **nightly / pre-harden**. The determinism bet (CANON §0.4) is exactly what makes
this feasible — most agent frameworks can't test this heavily because they never made it.

---

## Parallelism guidance

- **Phase 0 is sequential and small** — it freezes the contracts; rushing it is how you build the wrong
  abstraction six ways at once.
- After the skeleton is green, open **2–3 feature pods max** until the contracts have survived the
  reference agent, then widen.
- **Operations is always-on** alongside every phase.
- **Entangled features** (memory recall ↔ context assembly ↔ the loop): test the unit in isolation with
  stubs **and** test the seam explicitly — don't fake independence.
