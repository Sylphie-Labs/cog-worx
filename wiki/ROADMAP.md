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

- [ ] **Package skeleton** — `pyproject`, module layout, CI running `ruff` + `mypy --strict` + `pytest`.
- [ ] **Freeze the cross-pod contracts (the seams):** `Model` interface (S4); `Stage`/`Loop` contract;
      the three **Store seams** — Neo4j graph, pgvector, Timescale journal (S3); `Capability` interface;
      the **event/coordination types** (port sylphie `packages/shared/src/types/event.types.ts`, S7);
      artifact/claim types carrying **provenance + epistemic typing** (S5).
- [ ] **Substrate up:** docker-compose for Neo4j + **one Postgres cluster** (pgvector + TimescaleDB)
      (S3); plus **ephemeral test instances wiped per case** (port tess's isolated instances).
- [ ] **The Test Kit** (the deliverable that makes heavy testing repeatable, not heroic): fake/replay
      `Model` client (port tess `llm`/`kg_stub`); ephemeral-substrate fixtures; the **reference agent**;
      the **invariant property-suites** (S1, S5, S6, S9) as reusable parametrized tests.
- [ ] **Feature registry** — discoverable, toggleable features auto-enrolled in the relevant suites
      (generalize tess `tools/registry`). Toggleability is also S8.
- [ ] **Walking skeleton** — the thinnest end-to-end loop (intake → one model call → journal commit →
      done) driven by one **trivial reference agent**.
- [ ] **⛓ GATE — Spike Suite 1:** polyglot substrate capability + exactly-once resume (S6) + a first
      Lesion pass (S8). **Phase 1 is blocked until this passes.**

---

## Phase 1 — The Spine (depends on Phase 0)

The configurable control loop everything else attaches to.

- [ ] **Configurable graph loop** — DAG-of-stages + FSM-per-stage; **dev-authored pathways, graph by
      default**, not a fixed list.
- [ ] **Durability** — journaled resume, durable timers (`wake_at` + sweeper), **retries**,
      **pause/resume**, **fire-and-forget-with-feedback**, all on the Timescale journal (S6).
- [ ] **First-class transitions** — `await-human` and `degraded` (S8).
- [ ] **⛓ Spike** — durability/chaos test: kill mid-step → resume → exactly-once, **no model re-call**.

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
