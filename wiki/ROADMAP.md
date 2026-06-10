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

- [x] **Configurable graph loop** — DAG-of-stages + FSM-per-stage; **dev-authored pathways, graph by
      default**, not a fixed list. *(graph + dev-authored pathway registry done in 1.0; per-stage FSM
      retry/timeout transitions done in 1.2)*
- [x] **Durability** — journaled resume, durable timers (`wake_at` + sweeper), **retries**,
      **pause/resume**, **fire-and-forget-with-feedback**, all on the Timescale journal (S6).
      *(cold/cyclic journaled resume done in 1.0; durable timers + sweeper + pause/resume done in 1.1;
      retries + per-stage timeouts done in 1.2; durable await-human resume done in 1.3;
      fire-and-forget with feedback done in 1.4)*
- [x] **First-class transitions** — `await-human` and `degraded` (S8). *(both are first-class results;
      `Wait` added as a 5th first-class result in 1.1; durable await-human resume done in 1.3 —
      `AwaitHuman.to` + `provide_human_input`)*
- [x] **⛓ GATE — Spike** — durability/chaos test: kill mid-step → resume → exactly-once, **no model
      re-call**. ✅ **PASSED on the live substrate, end-to-end across the operation pods**
      (`tests/integration/test_phase1_gate_live.py`): G1 Wait → durable timer row → fresh-engine
      Sweeper cold-wake → COMPLETED · G2 pause beats the fired timer (CAS loses), unpause advances
      past the committed `Wait` · G3 two durable retries across fresh engines, success at the frozen
      `seq`, failed attempts commit nothing · G4 backgrounded crash → `RUN_CRASHED` feedback → cold
      resume COMPLETED — every engine on `ReplayModel([])`, so any model call anywhere fails loud
      (S6/S1 at the live tier). Plus Spike Suite 1 (1.0 core) + the live primitive suites
      (timer-lease, attempt-counter, human-input first-answer-wins). **Phase 1 is COMPLETE; Phase 2
      is unblocked.** Cross-instance run-lease/reaper + event dedup remain the documented ops-pod
      carry-forwards (not Phase-1 gaps).

---

## Phase 2 — Knowledge & Memory (the differentiator · first pod after the spine)

Deepest, riskiest (the polyglot recall bet), and the most differentiated value. Memory research is
already done in biz-firm, so this is port + generalize.

> **Pod plan (dependency-ordered; 2.0 first — it is the claim substrate everything else in this
> phase writes to or reads from):**
>
> - **Pod 2.0 — Entity KG core (Neo4j) — DONE** ✅ (CANON-COMPLIANT WITH CONCERNS, red-team-
>   hardened): event-sourced claims (tess Wave B port) — `(:Entity)`/`(:Claim)`/`(:Evidence)` where
>   evidence ACCUMULATES and Beta confidence is always **derived at read**, never stored;
>   NFC-normalized content-hash claim identity (idempotent re-derivation, enforced structurally at
>   `write_claim`); coalesce-populate writes (immutable-on-match, skeleton lineage nodes populate
>   on arrival and never surface unpopulated); bi-temporal invalidate-don't-delete with all
>   datetimes UTC-normalized at the boundary (naive = UTC); `DERIVED_FROM` weakest-link lineage;
>   `CONTRADICTS` surfaced-not-deleted; dense recall channel (native vector index, raw-cosine
>   contract); `InMemoryEntityKG` double held to adapter parity by test. Red-team killed 6
>   findings pre-validation (inherited-`upsert_claim` back door → sealed; skeleton-node read
>   crashes; unvalidated `base_weight` → NaN confidence; lexicographic as_of drift; unnormalized
>   identity hash splitting evidence; sockpuppet `source_id` → documented S9 discipline).
>   **Carry-forwards:** ① a **structural source registry** (code-assigned `source_id` /
>   `source_authority`) is a **precondition for Pod 2.3** — until then S9 holds only because no
>   model output reaches the write surface; ② **epistemic upgrades** (inference→confirmed) are
>   first-write-wins until the Pod 2.7 reconciler ships the explicit promotion surface (pinned by
>   test); ③ the additive `Claim.object_entity` contract extension needs **Jim's sign-off** on a
>   contract-evolution rule (see session log).
> - **Pod 2.1 — Procedural KG (Neo4j).** `(:Procedure)`/`(:ProblemType)`/`(:Trial)`, promotion gate,
>   Thompson-sampling read surface (port tess procedural KG + biz-firm e1-reflection; drop tess's
>   epoch/MVCC machinery). **Design locked 2026-06-09** (mythos review + architect + eval-stats; Jim
>   sign-off): ① `(:Trial)` nodes are the immutable **event source** and the Beta posterior is
>   **derived at read** — counters are **NOT** stored on `APPLIES_TO` (edge is topology only),
>   mirroring 2.0's "derive at read, never store" and keeping us S6-clean (tess's increment-in-place
>   edge counters rejected as non-idempotent). ② **Deterministic substrate only** — no model on the
>   write path; `procedure_id` framework-assigned (interim discipline like 2.0's `source_id`); the
>   e1 model-driven **consolidator is OUT of 2.1**, deferred to a later pod with its own spike.
>   ③ Trial = **projection of the committed journal `StepRecord`**, written off-path by a
>   sweeper-shaped projector, key `(run_id, step_index)`, `MERGE`-idempotent; outcome **stage-stamped**
>   (`output.data["outcome"]`), never inferred from `result.kind` (S9). ④ Projection cursor is a
>   `(:ProjectionCursor)` node advanced in the **same Neo4j txn** as the Trial MERGE; Timescale polled
>   with `committed_at > cursor − lookback_lag` (injectable/non-monotonic clock). ⑤ Posterior dedups
>   at `source_id = run_id` per `(procedure, problem_type)` edge — and the `n ≥ 5` promotion floor
>   counts **deduped contributions, not raw `(:Trial)` nodes**. ⑥ Static `(pathway, stage) →
>   procedure_id` registry lets the projector **synthesize failure-trials** from retry-exhausted /
>   FAILED-at-`seq` records (else the posterior is biased upward). ⑦ TS is an **invention** from biz-firm
>   e1 (NOT in tess, which is exploitation-only) — the load-bearing unproven claim; ships as a pure fn
>   with injectable seeded RNG + `promoted_only` filter **only if** its spike shows a CI-clean regret
>   win in the near-tie regime, else ship greedy and defer TS (a legitimate S12 outcome). See session
>   log `docs/sessions/2026-06-09-phase-2-pod-2.1.md`.
> - **Pod 2.2 — Latent space on pgvector — DESIGN LOCKED / IN PROGRESS** (2026-06-10, mythos
>   deep-reasoning pass + Jim sign-off on both flags): hot/cold tier membership as a
>   capacity-bounded top-N by ACT-R activation (`ln(1+use_count) − d·ln(max(Δt_hours,ε))`),
>   recomputed by an atomic SQL sweep in `LatentTierSweeper` (mirrors the 1.1/2.1 sweeper
>   pattern). Key design decisions: ① **single table + `tier` column** (two-table and
>   stored-activation-score both rejected; tier flips are row UPDATEs, S8 lesion is clean —
>   sweeper off → everything cold → search byte-identical); ② **seam split** — `put`
>   (idempotent, content-only) and `record_use` (non-idempotent, atomic increment) are separate
>   methods; `LatentRecord` drops `use_count` (write-ignored field = contract lie); ③ **default
>   search is tier-agnostic** (global exact top-k by cosine; tier= filter available; hot-first
>   composite deferred to Pod 2.6 behind the confidence gate — without a similarity threshold
>   hot-first would rank low-similarity hot rows above high-similarity cold rows); ④ d=0.5
>   spike-checked with d ∈ {0.25, 0.5, 1.0} sensitivity (2.1 lcb-constant lesson); ⑤ schema
>   migration: idempotent ALTER ×3 after CREATE (2.1 FIX-2 lesson); live migration test
>   in-scope. Carry-forwards: runner-up margin/confidence gate (CF-1 → 2.6); ANN index tuning
>   (CF-3); d-constant empirical tuning (CF-5). See session log
>   `docs/sessions/2026-06-10-phase-2-pod-2.2.md`.
> - **Pod 2.3 — Episodic memory + conversation capture — DESIGN LOCKED / IMPLEMENTED** ✅
>   (2026-06-10, Jim sign-offs F1–F4): synchronous cheap capture and async batched extraction.
>   Key design decisions: ① **Capture = turn-stamping into step output** (`stamp_turns` writes into
>   the `StepOutput.data` dict on the write path via `D1`; a direct `EpisodeStore.append` call on
>   the write path would be a phantom-turn defect — avoided structurally); ② **`EpisodeProjector`**
>   sweeper polls the Timescale journal (`committed_at > cursor − lookback_lag`) and writes
>   `cogworx_episodes` rows in Postgres — mirrors the 2.1 TrialProjector pattern (visibility-fenced
>   cursor advanced in the same PG txn as the episode MERGE, `MERGE`-idempotent on `(run_id,
>   step_index)`); ③ **`ClaimExtractor`** sweeper polls the `cogworx_episodes` table, fires the
>   model off-path (S1), and writes claims into the entity KG via the new `project_claims` additive
>   seam method (F1 approved by Jim); ④ **`SourceRegistry`** enforces S9 structurally —
>   `SourceDeclaration` type-enforces `source_id` / `source_authority` assignment by framework
>   code, never by model output; the 2.0 carry-forward precondition is satisfied; ⑤ new
>   `EvidenceType` **`"extraction"`** (base_weight 1.0) registered in `EVIDENCE_BASE_WEIGHTS`;
>   ⑥ `project_claims` is an **additive** method on the `EntityKG` seam (F1) — existing callers
>   unaffected (C3 contract-evolution rule). Carry-forwards: CF-1 through CF-11 (see session log
>   `docs/sessions/2026-06-10-phase-2-pod-2.3.md`).
>   **Carry-forwards from 2.0 now satisfied:** structural source registry landed; model-derived
>   claims reach the write surface only through `ClaimExtractor` (off-path, S1-clean).
> - **Pod 2.4 — World model + user model — DONE** ✅ (CANON-COMPLIANT, Jim-approved F1–F4, red-team PASS, 572 unit+spike tests green): `Claim.scope: str = "agent"` (additive, migration-free) with scope joining claim identity (conditional 4th hash part — pre-2.4 IDs byte-identical); `ScopeRegistry`+`ScopeWriteToken` clone the Pod 2.3 `SourceRegistry` discipline (S7 in-process, exactly one writer per scope); `ScopedKG` is a `@final` governed-view class over the unchanged `EntityKG` seam — `assert_fact` mints scoped Claims internally (callers never construct); `world_model`/`user_model` factory functions. `ClaimExtractor` unchanged (routing extraction into user scope = model-as-scope-routing-signal, S9). Spike SC-1–SC-7: scope isolation, shared-entity topology, evidence segregation, token discipline, back-compat, S8 lesion, S1 isolation. Red-team caught + fixed a CRITICAL (vacuous K4 accumulation `>= 1` → `== 2` + negative control) and a HIGH (SC-4 multi-registry same-process S7 bypass now explicitly tested + verdict corrected). **Carry-forwards:** CF-1 (direct token construction, not blocked structurally); CF-3 (vector over-fetch re-query floor); CF-4 (automated agent→user/world promotion → Pod 2.7); CF-5 (cross-process S7 exclusivity → ops pod); CF-6 (multi-registry same-process gap, documented in spike); CF-7 (non-canonical scope strings at Claim boundary).
> - **Pod 2.5 — Recall stack.** dense + BM25 + graph + temporal channels → Reciprocal Rank Fusion →
>   rerank seam → token-budget assembly with lost-in-the-middle ordering (per biz-firm c3/c1).
> - **Pod 2.6 — Memory injection.** Per-stage recall riding 2.5 into `StageContext` (token-budget,
>   minimum-source guarantees; sylphie activation/budget math as reference).
> - **Pod 2.7 — Coherence reconciler.** Async/batched, dirty-subject queue; embedding pre-filter →
>   specificity check → MUS+QuickXplain judge seam; AGM entrenchment ordering; `SUPERSEDES` +
>   `defeasibly-defeated` status — **surface, never silently delete** (per biz-firm c2). Owns the
>   **explicit epistemic-upgrade surface** (2.0 carry-forward: stored levels are first-write-wins
>   until promotion-with-provenance lands here).

- [x] **Procedural KG** (Neo4j) · **Entity KG** (Neo4j) *(entity KG = Pod 2.0 ✅ DONE · procedural
      = Pod 2.1 ✅ DONE — validated 2026-06-10: derive-at-read posterior, trial-as-projection of the
      committed journal via a `commit_xid` visibility-fenced cursor (S6-additive), posterior-mean
      greedy shipped / Thompson deferred. Carry-forwards: S6 FAILED-at-seq failure-trial gap (xfail,
      needs a run-status projection seam); S5 epistemic-type-as-field (C3); recommended future
      durability item already taken via commit_xid)*
- [x] **World model** (long-term memory) · **User model** *(Pod 2.4 ✅ DONE — see above)*
- [x] **Episodic memory** *(Pod 2.3 ✅ DONE — VALIDATED 2026-06-10 (red-team PASS: 4 HIGH/CRITICAL issues found and fixed — fail-stall contract, PgEpisodeStore cursor monotonicity, test isolation, mutation-resistant negative controls))*
- [x] **Latent space on pgvector** — hot/cold tiering *(Pod 2.2 ✅ DONE — validated 2026-06-10:
      ACT-R activation, single-table tier column, LatentTierSweeper, seam split put/record_use)*
- [ ] **Recall stack** — dense + BM25 + graph + temporal → rank fusion → rerank → token-budget assembly
      *(Pod 2.5)*
- [ ] **Memory injection** *(Pod 2.6)*
- [x] **Conversation extraction / classification / storage** *(Pod 2.3 ✅ DONE — see above)*
- [ ] **Coherence** — async/batched reconciler; **surface, don't silently delete** *(Pod 2.7)*
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
