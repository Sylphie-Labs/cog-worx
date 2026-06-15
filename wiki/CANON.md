# cog-worx — CANON

**This document is law.** Every plan, design, ticket, PR, agent output, and implementation choice is
validated against it. If something contradicts the CANON, it is wrong until **Jim** says otherwise.
The CANON is **immutable unless Jim explicitly approves a change**. It **supersedes every other
document**; where any spec, repo, or note conflicts with the CANON, the CANON wins and the other doc
is corrected.

> **Status: research / alignment.** This is the alignment artifact. Enforcement tooling (a `canon`
> agent, `enforce-canon` / `update-canon` skills, a `canon-check` hook, spike suites) is **not** set up
> yet — Jim configures the `.claude` machinery once he's in the repo. Until then the CANON is enforced
> by reading it.

---

## 0. What cog-worx Is (core philosophy)

1. **A framework, not a product.** cog-worx is the cognitive-architecture **library** developers use to
   build agents fast — the loop, the memory, the verification, the perception, all the QoL machinery,
   out of the box. It ships as a **publishable Python package**. It is *not* an application; there is no
   product wrapped around it (this is the line biz-firm crossed and cog-worx does not).
2. **The connecting system is a later layer — never built in the abstract.** The system that wires
   agents built with the library into a working multi-agent whole comes *after* the framework. cog-worx
   lays down the **contracts** that make that wiring possible later (see S7); it does not build the
   orchestration platform now, for users who don't exist yet.
3. **A synthesis of proven prior art, not invention in the abstract.** Every load-bearing piece is
   ported/generalized from something already working in the sibling repos and validated by a spike
   before it hardens. No piece is carelessly designed.
   - **tess** — the 10-stage Python outer loop, procedural + entity Neo4j KGs, the thesis/antithesis
     honest-failure mechanism, the oracle layer (math / chemistry / physics / pytest), OTel telemetry,
     persona injection.
   - **sylphie** — the executor engine, the perception-service (computer vision + multimodal fusion),
     hot/cold pgvector latent space, graduation, the Lesion Test, the event/coordination contracts
     (`packages/shared/src/types/event.types.ts`).
   - **biz-firm** — the immutable-standards model, the composition primitives (Stage · Capability ·
     Context · Loop), the recall stack, and the research program that backs every decision.
4. **Determinism makes cheap models work.** A well-structured, deterministic framework is the bet that
   lets cheaper/local models do real work. **Structure is the moat, not any one model.**
5. **Config over code for the loop.** Developers author their own pathways — a **graph of stages by
   default**, not a fixed list — and get the cognitive features (memory, KGs, verification, perception,
   durability, personality, rules) for free.

---

## 1. The Immutable Standards (constitutional — only Jim changes them)

Each standard is law. These are adapted from biz-firm's CANON; the deliberate divergences (S2, S3, S7)
are recorded in §7.

### S1 — Model work OFF the write path
Writes are cheap and synchronous (persist + emit + mark dirty); all model-heavy work (extraction,
reflection, coherence reconciliation) runs **asynchronously, batched**, off the hot path.
*Violation:* an LLM call on the write/commit path; "just summarize on write"; per-write extraction.

### S2 — Own the loop
The loop and composition layer are ours. **No agent SDK, no managed durable-execution dependency**
that owns the control loop. OSS is **reference, not dependency** (we may read and port; AGPL/Elastic =
read-only). *Violation:* importing a framework that owns the control loop; a runtime dependency under
the differentiated layer. *(Divergence from biz-firm S2: the blanket "no Postgres" clause is dropped —
see S3.)*

### S3 — Polyglot substrate, each engine for its strength
The substrate is **not** one store and **not** a generic store-abstraction that discards each engine's
power. Each persistence engine is chosen for what it is best at:
- **Neo4j** — the graph knowledge layer: procedural KG, entity KG, world model, user model
  (bi-temporal, provenance-typed). Graph + native vector + full-text give graph/dense/BM25 recall
  channels.
- **Postgres (single cluster)** — `pgvector` for the latent space (sylphie's **hot/cold** tiering ports
  straight over) **and** **TimescaleDB** for the durable journal (hypertable-fast on time-ordered run
  state). One cluster hosts both extensions.
- **OpenTelemetry** — spans (the observability path; sinks are swappable).

The only storage seam is a **thin internal one for tests/mocks**, never a backend-portability layer.
*Violation:* a generic `Store` abstraction that flattens these engines; persisting graph claims outside
Neo4j; the journal outside Timescale; vectors outside pgvector.

### S4 — Model-agnostic
The framework talks to models through a thin `Model` interface; **provider is selectable per agent**
(Claude / DeepSeek / Ollama / any OpenAI-compatible). No hard provider lock anywhere. Premium
capabilities (structured output, caching, streaming, logprobs) are interface capabilities that
**degrade gracefully** when a provider lacks them. *Violation:* a hardcoded provider; logic that only
works on one vendor without a fallback.

### S5 — Provenance + epistemic typing on every claim
Every substrate claim is typed **observation | inference | confirmed**, never silently merged across
levels; every derived fact links to the evidence that produced it ("why did this surface" is a graph
traversal). *Violation:* a write without provenance/epistemic type; an inference stored as a fact.

### S6 — Durable by default; exactly-once resume
A step is "done" **iff its result is committed to the TimescaleDB journal before the runner advances.**
On resume, the runner reads stored outputs — it **never re-calls the model on replay.** Durable timers
= `wake_at` + sweeper. Retries, pause/resume, and fire-and-forget-with-feedback all ride this journal.
*Violation:* in-memory-only progress; replay that re-invokes the model; non-idempotent side effects.

### S7 — Coordination by contract now; the connecting system later
cog-worx ships the **coordination contracts/types** so agents built with it are wireable later, but
does **not** build the orchestration platform now. The contract (from biz-firm S7, sylphie event types)
is two-channel — (1) the shared substrate (broadcast) and (2) direct point-to-point messaging — under
four rules: ① direct calls carry requests/replies, never state (durable facts go to the substrate);
② synchronous need = a tool call, not a message; ③ "waiting for a reply" is a **stage** (non-blocking)
with a timeout that escalates to HITL; ④ every call emits an event the system can witness. **Exactly
one write-token per entity.** *Violation:* building the connecting/orchestration platform now; state
passed agent-to-agent; blocking waits; two writers to one entity.

### S8 — Graceful degradation (the Lesion Test)
Every model/tool path has a fallback chain; the system degrades, it does not fail hard. `await-human`
and `degraded` are first-class loop transitions. *Violation:* a path that throws instead of degrading
when a model/tool is unavailable.

### S9 — Structure over prompting; never trust the model's self-report
Prefer structural mechanisms over prompt wording (constrained decoding, structured judging, retrieval
fusion, plan-as-DAG). The model's self-assessment is **not** a control signal — require
external/structural signals (no self-correction without ground truth, no self-confidence gating, no
model-enforced budgets). The thesis/antithesis mechanism is the structural check on "is this right,"
not the model's own say-so. *Violation:* "ask the model nicely" where a structural guarantee is needed.

### S10 — Security by structure
Least-privilege capabilities (permission tiers); untrusted (web/tool) content is **quarantined**; the
"lethal trifecta" (private data + untrusted content + exfiltration) is broken structurally
(Plan-Then-Execute; drop external-tier tools during read phases), **not** by content filters. Human
approval is gated on *consequential ∧ irreversible ∧ tainted*. *Violation:* filtering as the primary
injection defense; external-tier tools available while reading untrusted content.

### S11 — Cost bounded structurally
Budgets are enforced as **pre-call guards** (the model cannot self-terminate); ceilings are
hard; async work routes to a provider batch/discount path where available. *Violation:* relying on the
model to stop; unbounded loops without a step/cost ceiling.

> **Clarification (2026-06-12, Jim-approved via `/update-canon`):** the enforced **hard unit is the
> drive segment**, not the whole run. A paused/resumed/timer-fired run gets a fresh guard per drive
> segment; each segment is hard-bounded pre-call, and S6 replay never re-bills a committed prefix.
> Cumulative **per-run** ceilings (durable spend accounting journaled across resume) are deferred as
> **CF-3.0-B** — a scheduled future item, not a silent gap. The budget API names this honestly
> (`BudgetPolicy.max_calls_per_drive` / `max_usd_per_drive`).

### S12 — Spike-gated
Nothing load-bearing hardens until its **falsifiable spike** passes. *Violation:* building a sector's
real logic before its spike has passed.

---

## 2. Architecture (the layers)

cog-worx is a layered Python framework. Each layer has proven references in the sibling repos.

- **Spine — the configurable loop.** A control loop that is a **graph of stages by default** (DAG of
  stages + FSM per stage), crash-safe and journaled-resumable; devs author pathways via config.
  `await-human` and `degraded` are first-class transitions. *Refs:* tess `OuterLoop`, sylphie executor.
- **Cognition (one step).** Thin `Model` interface + routing; **Context** assembly per stage
  (token-budget, lost-in-the-middle ordering); **Capabilities** (permission-tiered tools, **tool
  routing in code** not model-as-text, **MCP** support); **personality injection**; **internal
  unbreakable rules**.
- **Knowledge & Memory.** Episodic memory; long-term memory via a **world model**; procedural KG +
  entity KG + user model on **Neo4j**; **latent space on pgvector** (hot/cold); **memory injection**;
  **conversation extraction / classification / storage**. Recall stack: dense + BM25 + graph + temporal
  → rank fusion → rerank → token-budget assembly. Coherence: async/batched reconciler that **surfaces,
  doesn't silently delete**.
- **Perception (computer vision + multimodal).** Vision + audio fusion pipeline and visual working
  memory. *Ref:* sylphie `perception-service` (`cobeing/layer2_perception`).
- **Verification & Truth.** The **oracle** layer — deterministic coding, math proofing, simulated
  chemistry / physics. **Honest failure** + **thesis/antithesis team decision-making**. *Ref:* tess
  `oracles/`, `tools/`, the antithesis mechanism.
- **Operations (the -ilities).** OTel observability; eval/testing harness; safety + permission tiers;
  graceful degradation; cost/budget ceilings; durability (retries, pause, resume,
  fire-and-forget-with-feedback) on the Timescale journal.

---

## 3. Scope & phase boundaries (hard walls)

**In scope — the framework:** every layer in §2, shipped as a configurable, model-agnostic Python
package with the cognitive features available out of the box.

**Deferred — out of scope (do not build "as a foundation"):**
- The **connecting / orchestration system** that wires many agents together (a later layer; cog-worx
  only lays the S7 contracts now).
- Any **product** wrapped around the framework.
- The **`.claude` governance machinery** (`canon` agent, enforce/update-canon skills, hooks, spike
  suites) — **Jim** sets this up in-repo later.
- Auto prompt-optimizers and cross-agent / "master-agent" learning — unless the latest Phase-1 research
  pulls them forward.

---

## 4. Planning & process rules

1. **Research-backed.** Every load-bearing decision traces to the latest Phase-1 research (§5).
2. **Synthesis discipline.** Port/generalize proven pieces from tess / sylphie / biz-firm; do not
   reinvent in the abstract.
3. **Spike before harden** (S12).
4. **Resolve open decisions first.** Work that presupposes an unmade decision is flagged, not started.
5. **Tangible artifact per session;** durable state survives crashes (S6).

---

## 5. Source-of-truth hierarchy

1. **CANON** (this file) — supreme.
2. **Latest Phase-1 project research** — authoritative on conflicts (in practice, biz-firm's
   `wiki/CANON.md` + `research/` is the working proxy for it).
3. **The sibling repos** — `tess`, `sylphie`, `biz-firm` — proven prior art to port from.

If (2)–(3) ever conflict with (1), the CANON wins. Where the latest research conflicts with a repo, the
research wins.

---

## 6. Amendment process

The CANON changes **only** with Jim's explicit approval (document change → impact assessment → Jim
approves → apply). When the enforcement tooling lands, the `canon` agent surfaces gaps and proposed
changes; it never edits the CANON itself. **If the enforcer can rewrite the law, there is no law.**

### 6.1 Contract-evolution rule (C3) — evolving a frozen seam

Frozen cross-pod seams (the `Model` interface, `Stage`/`Loop`, the Store seams, `Capability`, the
event/coordination types, the artifact/claim types — S3–S7) may evolve **additively without
per-change sign-off**; a **breaking** change requires explicit `/update-canon` approval like any other
CANON change.

- **"Additive" is defined structurally:** a change is additive **iff no existing conformant
  implementation or caller becomes non-conformant.**
  - Adding an **optional, defaulted field to a frozen value type** → **additive**.
  - Adding a **member or parameter to a Protocol** → **breaking** (every existing implementer silently
    stops conforming), even when it "looks" additive.
- **Additive changes need no pre-approval but MUST be declared** — a one-line entry in a
  **`Contract changelog`** block in the seam module's docstring, plus the session log. The
  `canon-check.cjs` stop-hook enforces the declaration discipline.
- **Breaking changes** go through the full §6 amendment process (Jim approves).
- **Grandfathered (one-time audit, 2026-06-11):** the pre-rule additive touches —
  `Claim.object_entity` (Pod 2.0), `EntityKG.project_claims` (Pod 2.3), `Claim.scope` (Pod 2.4) — each
  meets the additive definition and is ratified as-is; changelog entries backfilled.

---

## 7. Deltas from biz-firm's CANON (deliberate divergences)

cog-worx inherits biz-firm's standards except for three intentional changes:

1. **Framework-only scope (§0, §3).** biz-firm is "a product *and* a framework, built together"
   (pitch extraction, domain teams, dashboard, auth, inbox). cog-worx is the **framework alone**; the
   product layer and the connecting system are out.
2. **Polyglot substrate (S3) replaces single-Neo4j.** biz-firm's S3 mandated **one Neo4j store in four
   roles** and S2 forbade Postgres. cog-worx splits by strength: **Neo4j** (graph KGs), **Postgres /
   pgvector + TimescaleDB** in one cluster (latent space + durable journal), **OTel** (spans).
   Consequently biz-firm's "done iff committed to the **Neo4j** journal" (S6) becomes the **Timescale**
   journal.
3. **Coordination is contracts-now, platform-later (S7).** biz-firm builds the two-channel coordination
   into a running product. cog-worx ships the **contracts/types** so agents are wireable later, but
   defers the orchestration platform itself.
