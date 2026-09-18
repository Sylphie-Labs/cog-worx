# cog-worx

cog-worx is a Python cognitive-architecture **library** for building agents quickly. It provides the
control loop, memory, verification, and supporting machinery out of the box. It is a **framework, not
a product**: there is no application wrapped around it. You write the stages and pathways, and
cog-worx runs them durably, remembers what happened, and checks the results against structural
signals rather than the model's own report.

It is aimed at Python developers building agents who want:

- a loop they own (no agent SDK and no managed durable-execution service underneath),
- crash-safe, exactly-once execution that resumes from a journal without calling the model again,
- a real memory stack (knowledge graphs, a latent vector space, and hybrid recall), and
- deterministic tests, with a replay model and in-memory doubles, so agent behavior can be tested
  without a network or a database.

## Status

Pre-release (`0.0.0`) and not yet published to PyPI. The public API is still settling, so expect
breaking changes.

Development proceeds in dependency-ordered phases, each closed by a falsifiable "spike" gate. From
[`wiki/ROADMAP.md`](wiki/ROADMAP.md):

| Phase | Scope | State |
|---|---|---|
| 0 | Foundations: frozen contracts, substrate, Test Kit, walking skeleton | Complete (Spike Suite 1 passed) |
| 1 | The Spine: configurable graph loop, durability, retries, timers, await-human | Complete |
| 2 | Knowledge and Memory: entity/procedural KGs, latent space, episodic memory, recall, coherence | Complete |
| 3 | Cognition: model routing, context assembly, capabilities (incl. MCP), personality, rules | Complete |
| 4 | Verification and Truth: oracles, honest failure, thesis/antithesis decision-making | In progress |
| 5 | Perception: computer vision, multimodal fusion, visual working memory | Not started |

Phase 4 code (the `cogworx.verification` package, oracles, the dialectic stages, and gate-corpus
tooling in `cogworx.eval`) has landed in `src/`, but the Phase 4 gate has not passed yet. The
cross-cutting Operations track (observability, eval harness, safety tiers, degradation matrix, cost
ceilings) runs alongside every phase.

## Key concepts

- **Stage**: a unit of work with a `name`, the stages it may `transitions` to, and an async `run`
  that returns a first-class result: `Transition`, `Done`, `Wait`, `AwaitHuman`, or `Degraded`.
- **StageGraph and pathways**: stages form a validated graph with a per-stage FSM; cyclic pathways
  (for example a refine loop) are supported and still resume exactly once. Graphs are registered by
  id and version in a `PathwayRegistry`, which is what makes a cold, cross-process resume possible.
- **Engine**: drives a pathway to a terminal `RunState`, committing every step to the journal before
  advancing. The model is never called on the write path or on replay.
- **Model seam**: a thin, model-agnostic `Model` interface with adapters for Claude and any
  OpenAI-compatible endpoint (OpenAI, DeepSeek, Ollama), plus structural pre-call budget guards.
- **Claims and provenance**: every artifact and claim carries provenance and an epistemic type.
- **The CANON**: [`wiki/CANON.md`](wiki/CANON.md) is the project's binding design standard. Its
  invariants (numbered S1 to S12) are enforced by tests and code review, and it supersedes every other
  document, this README included.

## Architecture

```mermaid
flowchart TB
    subgraph app["Your agent"]
        stages["Stages + pathways"]
    end

    subgraph cw["cog-worx"]
        spine["Spine<br/>loop · runtime (Engine, Sweeper)"]
        cognition["Cognition<br/>model · context · capability · cost"]
        memory["Knowledge & Memory<br/>knowledge · recall · injection · coherence"]
        verification["Verification & Truth<br/>verification · oracles · dialectic"]
        seams["Substrate seams<br/>substrate (Protocols) · adapters"]
        testkit["Test Kit<br/>testing (ReplayModel, in-memory doubles)"]
    end

    subgraph sub["Polyglot substrate"]
        neo4j[("Neo4j<br/>knowledge graphs")]
        pg[("Postgres 16<br/>pgvector + TimescaleDB")]
        otel["OpenTelemetry"]
    end

    stages --> spine
    spine --> cognition
    spine --> memory
    spine --> verification
    spine --> seams
    memory --> seams
    seams --> neo4j
    seams --> pg
    cw -. spans via telemetry .-> otel
    testkit -. replaces .-> seams
```

The substrate is polyglot on purpose, each engine chosen for its strength:

- **Neo4j** holds the graph knowledge layer: procedural and entity knowledge graphs plus the world
  and user models (graph, native vector, and full-text indexes feed recall).
- **One Postgres cluster** hosts both **pgvector** (the hot/cold latent space) and **TimescaleDB**
  (the durable run journal).
- **OpenTelemetry** carries spans to a sink the application configures.

Every engine sits behind a Protocol in `cogworx.substrate`, and `cogworx.testing` provides
in-memory implementations of each, so the deterministic test tiers need no services at all.

## Tech stack

| Area | Choice |
|---|---|
| Language | Python 3.13 or newer (CI uses 3.13.15) |
| Packaging | [uv](https://docs.astral.sh/uv/) with a committed `uv.lock`; hatchling build backend; typed (`py.typed`) |
| Core dependencies | pydantic 2, pydantic-settings, structlog, OpenTelemetry API/SDK, jsonschema, neo4j driver, psycopg 3, pgvector |
| Optional extras | `providers` (openai, anthropic), `otlp` (OTLP HTTP exporter), `mcp` (MCP SDK), `sizing` (numpy) |
| Backing services | Neo4j 5; Postgres 16 with TimescaleDB and pgvector (`timescale/timescaledb-ha:pg16`) |
| Tooling | ruff, mypy `--strict`, pytest (with pytest-asyncio and hypothesis), nox |

Provider SDKs are optional: the core never requires one, and the Test Kit's `ReplayModel` stands in
for a real model in tests.

## Prerequisites

- Python 3.13+ (uv can install it for you).
- uv. CI pins uv 0.12.12; any recent release that understands the committed lock should work.
- Docker with Compose, only for the integration and spike tiers.
- Node.js and the `claude` CLI, only if you use the bundled Claude Code configuration (see
  [Claude Code configuration](#claude-code-configuration)).

## Installation

cog-worx is not on PyPI yet. To work on it:

```bash
git clone https://github.com/Sylphie-Labs/cog-worx.git
cd cog-worx
uv sync --all-extras
```

`uv sync --all-extras` creates `.venv` with the dev tools and every optional extra, which is the
environment the full local check suite expects. Plain `uv sync` installs the core and dev tools
without the optional extras.

To use it from another project, depend on a local checkout, for example
`uv add --editable ../cog-worx` (add `--extra providers` if you want the Claude or OpenAI-compatible
adapters).

## Running the backing services

The integration and spike tiers need the polyglot substrate. `docker-compose.yml` starts both
services with local development credentials:

```bash
docker compose up -d        # Neo4j 5 + Postgres 16 (TimescaleDB + pgvector)
docker compose ps           # wait until both report healthy
docker compose down         # stop; add -v to also drop the data volumes
```

| Service | Container | Host ports (default) |
|---|---|---|
| Neo4j | `cogworx-neo4j` | 7474 (HTTP browser), 7687 (Bolt) |
| Postgres | `cogworx-postgres` | 5432 |

`docker/postgres-init.sql` runs on first start and enables the `timescaledb` and `vector`
extensions. The credentials in the
compose file are for local, throwaway containers only.

## Configuration

Substrate endpoints are read by `cogworx.adapters.config.SubstrateSettings` from environment
variables with the `COGWORX_` prefix, or from a `.env` file in the working directory. The defaults
match `docker-compose.yml`, so on a machine with no port conflicts you need no configuration at all.
To override anything, copy `.env.example` to `.env` and uncomment what you need.

| Variable | Default | Purpose |
|---|---|---|
| `COGWORX_NEO4J_URI` | `bolt://localhost:7687` | Bolt URI the Neo4j adapters connect to |
| `COGWORX_NEO4J_USER` | `neo4j` | Neo4j user |
| `COGWORX_NEO4J_PASSWORD` | `cogworx-test` | Neo4j password (local development default) |
| `COGWORX_PG_DSN` | `postgresql://postgres:cogworx-test@localhost:5432/cogworx` | DSN for the single Postgres cluster (pgvector + TimescaleDB) |
| `COGWORX_NEO4J_HTTP_PORT` | `7474` | Host port Compose maps to Neo4j HTTP |
| `COGWORX_NEO4J_BOLT_PORT` | `7687` | Host port Compose maps to Neo4j Bolt |
| `COGWORX_PG_PORT` | `5432` | Host port Compose maps to Postgres |

The first four are read by the library; the three `*_PORT` variables are read only by
`docker compose`. If you move a host port, update the matching URI or DSN too, for example
`COGWORX_NEO4J_BOLT_PORT=17687` together with `COGWORX_NEO4J_URI=bolt://localhost:17687`.

Model providers are configured in code (`cogworx.model.providers`), not through `COGWORX_`
variables. The optional live-provider integration tests read the provider's usual variables
(`ANTHROPIC_API_KEY`, `OPENAI_API_KEY`) and skip when they are absent; the DeepSeek and Ollama live
tests are currently hard-skipped. Never commit `.env`; it is gitignored.

## Usage

A minimal agent: one stage that makes one model call, run by the `Engine` against the Test Kit's
in-memory substrate and a replay model, so it needs no network and no database.

```python
import asyncio
from datetime import UTC, datetime

from cogworx.claims import Artifact, Provenance
from cogworx.loop import Done, StageContext, StageGraph, StageResult
from cogworx.loop.pathway import PathwayRegistry
from cogworx.model import ChatMessage
from cogworx.model.registry import ModelRegistry
from cogworx.runtime import Engine
from cogworx.testing import InMemoryGraphStore, InMemoryJournal, InMemoryLatentStore, echo_model


class Greet:
    """A one-stage pathway: make a single model call and finish."""

    name = "greet"
    transitions: tuple[str, ...] = ()

    async def run(self, ctx: StageContext) -> StageResult:
        response = await ctx.model.complete(
            messages=[ChatMessage(role="user", content="Say hello.")]
        )
        return Done(
            output=Artifact(
                kind="greeting",
                produced_by=self.name,
                provenance=Provenance(
                    source="inference", confidence=1.0, recorded_at=datetime.now(UTC)
                ),
                data={"text": response.text or ""},
            )
        )


async def main() -> None:
    pathways = PathwayRegistry()
    pathways.register("hello", StageGraph([Greet()], entry="greet"), version=1)

    models = ModelRegistry()
    models.register("default", echo_model("Hello from a replayed model."))  # no network

    engine = Engine(
        models=models,
        journal=InMemoryJournal(),
        graph_store=InMemoryGraphStore(),
        latent=InMemoryLatentStore(),
        pathways=pathways,
    )
    state = await engine.run(
        run_id="run-1",
        session_id="session-1",
        pathway_id="hello",
        initial=Artifact(
            kind="user-input",
            produced_by="human",
            provenance=Provenance(source="human", confidence=1.0, recorded_at=datetime.now(UTC)),
        ),
    )
    output = state.steps[-1].result.output
    assert output is not None
    print(state.status)  # completed
    print(output.data["text"])  # Hello from a replayed model.


asyncio.run(main())
```

To run against real services, swap the in-memory doubles for the adapters in `cogworx.adapters`
(Neo4j graph and knowledge graphs, pgvector latent store, TimescaleDB journal) and register a
provider model from `cogworx.model.providers`. For fuller examples, see the reference agents in
`src/cogworx/testing/reference_agent.py` and `src/cogworx/testing/reference_dialectic.py`, and the
walking-skeleton and spike tests under `tests/`.

## Testing

Tests are layered by pytest marker (declared in `pyproject.toml`):

| Marker | Meaning |
|---|---|
| none | Deterministic unit, invariant, and eval tests on the Test Kit's doubles. No services. |
| `integration` | Hits the real substrate (Neo4j and/or Postgres). |
| `spike` | Spike Suite gate tests. Some also carry `integration` and need the substrate. |
| `neo4j`, `postgres` | Subsets of `integration` that need that one service. |
| `slow` | Long-running soak and chaos tests; opt in with `-m slow`. |

Run the tiers through nox, which installs from `uv.lock` with `--locked` and fails if the lock is
stale:

```bash
uv run nox                        # default sessions: lint, typecheck, test, sizing
uv run nox -s test                # deterministic tier: pytest -m "not integration and not spike"
uv run nox -s sizing              # mypy + tests/eval with the numpy `sizing` extra installed
docker compose up -d
uv run nox -s integration         # pytest -m "integration or spike" against the substrate
```

Extra pytest arguments pass through after `--`, for example
`uv run nox -s test -- tests/unit/test_stage_graph.py -x`.

A bare `uv run pytest` selects every marker, including the integration tests, so it needs the
substrate running. Use the nox sessions, or pass `-m "not integration and not spike"` yourself, to
stay service-free. The `sizing` session is the slowest of the defaults, since it runs the numpy
fast-kernel equivalence suite.

## Lint and type-check

```bash
uv run nox -s lint                # ruff check + ruff format --check on src and tests
uv run nox -s typecheck           # mypy --strict (pydantic plugin) on src and tests
uv run ruff format src tests      # apply formatting
```

`typecheck` deliberately runs without numpy to prove the core type-checks without the `sizing`
extra; the `sizing` session re-runs mypy with numpy present.

## Project layout

```
src/cogworx/
  loop/           Stage, StageGraph, results, retry policy, run/stage states
  runtime/        Engine, Sweeper (durable timers), projectors, claim extractor
  model/          Model seam, registry, structured-output ladder, providers/ (Claude, OpenAI-compatible)
  context/        Per-stage context assembly, personality, unbreakable rules
  capability/     Permission-tiered, code-routed tools and MCP binding
  cost/           Structural pre-call budget ceilings
  claims/         Artifacts and claims with provenance and epistemic typing
  knowledge/      Entity and procedural knowledge-graph domain logic, scopes, episodes
  recall/         Dense + BM25 + graph + temporal recall, fusion, rerank, assembly
  injection/      Memory injection into context
  coherence/      Async contradiction detection and reconciliation
  verification/   Oracles, honest failure, thesis/antithesis dialectic (Phase 4)
  eval/           Evaluation statistics and gate tooling for the spike suites
  coordination/   Event and coordination contracts
  substrate/      Engine-shaped Protocols for the graph, journal, latent space, and KGs
  adapters/       Neo4j, pgvector, and TimescaleDB implementations; SubstrateSettings
  telemetry/      OpenTelemetry span helpers
  testing/        The Test Kit: ReplayModel, in-memory doubles, reference agents, pytest fixtures
tests/            unit/, eval/, integration/, spike/, fixtures/
docker/           Postgres init script for Compose
wiki/             CANON.md (design law) and ROADMAP.md (build order)
docs/sessions/    Development session logs
scripts/          check-ci-pins.py (CI pin guard)
noxfile.py        Test, lint, and type-check sessions
```

## Continuous integration

[`.github/workflows/ci.yml`](.github/workflows/ci.yml) runs on pull requests and on pushes to
`main`:

| Job | Pull requests | Pushes to `main` |
|---|---|---|
| `lint`: CI pin guard (`scripts/check-ci-pins.py`) + `nox -s lint` | Yes | Yes |
| `check`: `nox -s typecheck`, `nox -s test`, `nox -s sizing` | No | Yes |
| `integration`: `nox -s integration` against Neo4j and TimescaleDB service containers | No | Yes |

Pull requests are gated on lint only, which keeps them to about a minute. Type errors and test
failures surface on `main` after merge and are fixed in a follow-up PR, so run
`uv run nox` locally before opening a PR.

Everything in CI is pinned: actions by commit SHA, uv and Python by exact version, service images by
exact tag, and Python packages by `uv.lock`. Every program a workflow step runs must be declared in
[`.github/pinned-tools.txt`](.github/pinned-tools.txt), which the pin guard enforces. Dependabot
([`.github/dependabot.yml`](.github/dependabot.yml)) opens weekly grouped PRs for GitHub Actions and
uv dependencies.

## Contributing

1. Branch from `main` with a short prefixed name, for example `fix/...`, `ci/...`, `chore/...`, or
   `docs/...`.
2. Read [`wiki/CANON.md`](wiki/CANON.md) before any architectural change. It is changed only through
   an explicit, maintainer-approved amendment, never as a side effect of a feature PR.
3. Keep `uv run nox` green locally (ruff, mypy `--strict`, deterministic tests), and run
   `uv run nox -s integration` when you touch adapters or the substrate.
4. If you change `pyproject.toml` dependencies, run `uv lock` and commit `uv.lock` in the same PR;
   CI refuses a stale lock.
5. New features follow the roadmap's Definition of Done: unit tests, a deterministic integration test
   through the walking skeleton, the relevant invariant suites, an eval where the feature is
   model-bearing, and a degradation test showing the system still runs with the feature off.
6. Open a PR against `main`. PRs are merged with a merge commit.

## Claude Code configuration

The repository includes [Claude Code](https://docs.anthropic.com/en/docs/claude-code) configuration
under `.claude/` that the maintainers use: specialist agents, CANON skills, and hooks. None of it is
needed to build, test, or use the library.

If you open the repo in Claude Code, be aware of the hooks in `.claude/settings.json`:

- A PreToolUse guard blocks shell commands that would delete, move, or overwrite `wiki/CANON.md`.
- A git guard shows what a destructive git command (for example `reset --hard` or `clean -f`) would
  discard before it runs.
- Stop hooks run Node scripts: `doc-check.cjs` (warn-only reminder to keep the roadmap and session
  logs in step with `src/`), and `canon-check.cjs`, which sends `src/**/*.py` diffs to
  `claude -p` for a CANON review and blocks the stop on a violation. It exits cleanly if the
  `claude` CLI call fails.
- A best-effort session-capture Stop hook for the maintainers' internal tooling; it always exits 0.

These hooks need Node.js on your `PATH`, and the CANON check needs the `claude` CLI.

## Related projects

- **wombat** (Sylphie-Labs), a personal-assistant agent, is the first downstream consumer. It depends
  on cog-worx as a local path dependency, so changes to cog-worx's public API can break it.

## Roadmap

See [`wiki/ROADMAP.md`](wiki/ROADMAP.md) for the full build order, the per-pod status, the
Definition of Done, and deferred work. Detailed per-pod design and validation notes live in
[`docs/sessions/`](docs/sessions/).

## License

Apache License 2.0, as declared in `pyproject.toml`.
