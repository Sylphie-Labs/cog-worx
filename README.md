# cog-worx

The cognitive-architecture **library** developers use to build agents fast — the loop, the memory, the
verification, the perception, and the QoL machinery, out of the box. A **framework, not a product**.

> **Status: Phase 0 (Foundations).** Freezing the cross-pod contracts (the seams) and the Test Kit that
> makes every later feature independently testable. See `wiki/ROADMAP.md`. The `wiki/CANON.md` is law.

## Architecture (the layers)

- **Spine** — a configurable control loop: a graph of stages (DAG) + FSM per stage, crash-safe and
  journaled-resumable. `await-human` and `degraded` are first-class transitions.
- **Cognition** — a thin, model-agnostic `Model` interface; per-stage context assembly; permission-tiered
  capabilities; personality + unbreakable rules.
- **Knowledge & Memory** — procedural/entity KGs + world/user model on Neo4j; hot/cold latent space on
  pgvector; a dense + BM25 + graph + temporal recall stack.
- **Verification & Truth** — the oracle layer; honest failure; thesis/antithesis decision-making.
- **Perception** — computer vision + multimodal fusion; visual working memory.
- **Operations** — OTel observability; eval harness; safety/permission tiers; cost ceilings; durability.

## Polyglot substrate (S3)

Each engine for its strength: **Neo4j** (graph knowledge layer) · **Postgres** one cluster =
`pgvector` (latent space) **+** **TimescaleDB** (durable journal) · **OpenTelemetry** (spans).

## Develop

```bash
uv sync --all-extras          # install (Python 3.13+)
uv run nox -s lint typecheck test sizing   # ruff + mypy --strict + pytest (deterministic, no Docker)
docker compose up -d          # bring up the polyglot substrate
uv run pytest -m integration  # integration + Spike Suite 1 (needs the substrate)
```

The deterministic tiers run with no external services — the Test Kit (`cogworx.testing`) ships a
replay `Model` and in-memory substrate doubles. That determinism bet is what makes cog-worx testable to
a degree most agent frameworks can't reach.
