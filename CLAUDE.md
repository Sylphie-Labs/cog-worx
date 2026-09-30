# cog-worx — Claude Code working guide

cog-worx is the cognitive-architecture **library** developers use to build agents fast — the loop,
the memory, the verification, the perception, and the QoL machinery, out of the box. It ships as a
**publishable Python package**. It is a **framework, not a product** — there is no application wrapped
around it (the line biz-firm crossed and cog-worx does not). See `wiki/CANON.md` §0.

## The CANON is law
`wiki/CANON.md` is the single source of truth and **supersedes every other doc**. Read it before any
architectural change. Validate changes with the **`/enforce-canon`** skill (spawns the `canon` agent).
Change the CANON **only** via **`/update-canon`** — it is immutable unless **Jim** explicitly approves.

**Source-of-truth order (CANON §5):** `wiki/CANON.md` → latest Phase-1 research (in practice biz-firm's
`wiki/CANON.md` + `research/` is the working proxy) → the sibling repos (`tess`, `sylphie`, `biz-firm`)
as proven prior art to port from. If a lower source conflicts with the CANON, the CANON wins.

## Stack (locked — see CANON §0, S2–S4)
**Python** framework (publishable OSS package) · **own the loop** (no agent SDK, no managed
durable-execution dependency; OSS is reference, not dependency) · **model-agnostic** (Claude / DeepSeek
/ Ollama / any OpenAI-compatible, per agent). **Polyglot substrate, each engine for its strength (S3):**
- **Neo4j** — graph knowledge layer: procedural KG, entity KG, world model, user model (graph + native
  vector + full-text → graph/dense/BM25 recall).
- **Postgres (one cluster)** — `pgvector` for the hot/cold latent space **and** **TimescaleDB** for the
  durable journal.
- **OpenTelemetry** — spans (swappable sinks).

> cog-worx deliberately **diverges from biz-firm** on **S2/S3/S6/S7** (CANON §7): polyglot substrate
> instead of single-Neo4j; the durable journal lives in **TimescaleDB**, not Neo4j; coordination is
> **contracts-now, platform-later**. Don't import biz-firm's single-store assumptions wholesale.

## Working style
The top-level agent is the **coordinator and the ONLY agent that runs on Fable** — it is responsible
for interacting with Jim. It **routes and tracks; it does not decide.** **Every decision —
architecture, design, trade-offs, diagnosis, review verdicts — is delegated to the `architect`
agent.** The coordinator's job is to frame the question crisply for the architect, relay the
architect's decision verbatim as the task spec to the specialist subagents, and run the mechanical
gates (ruff, mypy `--strict`, pytest, CANON checks) on the output before reporting back. If a subagent
result raises a judgment call, the coordinator sends it back to the architect rather than ruling on
it itself.

**Model ladder (mandatory):** every spawned subagent gets an explicit `model:` — never let one
inherit the session model. Pick the cheapest tier the task needs, by what the agent must DO with what
it reads:
- **Haiku** — read to confirm/verify (does X exist, does output match, mechanical checks).
- **Sonnet** — read then summarize (surveys, sweeps, mapping, instruction-following implementation).
- **Opus** — read and reason (design judgment, subtle correctness, adversarial verdicts).
- **Fable** — never for subagents. Agents with a model pinned in their definition (e.g. `architect`)
  keep their pin.
Do not spend big-model tokens on work that doesn't require reasoning.
**Spike-gated (S12):** nothing load-bearing hardens until its falsifiable spike passes. Phase 0
(freeze the seams + the Test Kit) is sequential and unblocks every later pod — see `wiki/ROADMAP.md`.

## Agents (`.claude/agents/`)
- **architect** — **the decision-maker for this repo:** the coordinator routes *all* decisions
  here — architecture, design, trade-offs, diagnosis, review verdicts. Owns the configurable loop,
  composition primitives (Stage · Capability · Context · Loop), stage boundaries, where the model sits,
  the S7 coordination contract, and failure modes. Reasons and frames; delegates implementation.
- **canon** — CANON enforcement / drift detection (use before/after any architectural change or PR).
- **python-expert** — idiomatic Python, the package's public API + packaging/semver, typing, async.
- **neo4j-expert** — the Neo4j graph layer (KGs, bi-temporal, provenance, recall, Cypher) **within** the
  polyglot substrate (journal = TimescaleDB, vectors = pgvector — not Neo4j).
- **eval-stats** — eval harnesses, LLM-judge discipline, statistics, spike success criteria.
- **red-teamer** — break "validated" claims, falsify spikes, coherence + safety adversarials.
- **code-reviewer** — cross-cutting review (CANON compliance, contracts, invariants); reads only.
- **test-qa-expert** — deterministic tests (stub Model + in-memory Store double), the durability/chaos
  test, eval harnesses, integration markers.

## Skills (`.claude/skills/`)
- **enforce-canon** — check a plan/PR/diff against the CANON.
- **update-canon** — propose + apply a CANON change (Jim approves).

## Hooks (`.claude/hooks/`)
- **PreToolUse (Bash)** — blocks raw `rm`/`mv`/redirect against `wiki/CANON.md` (the CANON is immutable;
  changes go through `/update-canon`, CANON §6).
- **Stop → doc-check.cjs** — warn-only; nudges to keep `wiki/ROADMAP.md` / session logs in step with
  `src/` changes.
- **Stop → canon-check.cjs** — spawns Sonnet to check `src/**/*.py` diffs against S1–S12; **blocks**
  completion on a violation.

## Standing rules
Model work OFF the write path (S1) · provenance + epistemic typing on every claim (S5) · durable
exactly-once on the TimescaleDB journal, no model re-call on replay (S6) · own the loop (S2) ·
model-agnostic (S4) · structure over prompting, never trust the model's self-report (S9) · ruff + mypy
`--strict` + pytest clean · port/generalize proven pieces from `tess`/`sylphie`/`biz-firm` rather than
reinventing in the abstract · spike before harden (S12).
