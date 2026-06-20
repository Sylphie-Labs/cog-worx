# 2026-06-19 — codebase-pkg inference pass

First inference run over the cog-worx PKG (graph: `cogworx-neo4j` @ bolt 7694, user `neo4j`,
served by the `codebase-pkg` MCP per `.mcp.json`). The skill default container
`codebase-pkg-neo4j` (7692) holds a **different** project's graph — do not classify there.

## Domain classification (`/classify-pkg-domains`)
814 `Function` nodes (all in `src/cogworx`), 0 remaining unclassified. CANON-aligned taxonomy
(Jim-approved over the default, which has dead `frontend`/`web-api`/`cli` buckets):

| domain | n | | domain | n |
|---|---|---|---|---|
| substrate | 186 | | coherence | 32 |
| testing | 183 | | recall | 31 |
| loop-runtime | 97 | | telemetry-cost | 11 |
| composition | 82 | | eval | 9 |
| verification | 69 | | coordination | 4 |
| model-provider | 56 | | | |
| knowledge-graph | 54 | | | |

`shared-utilities` stayed empty — helpers live inside their owning concern.

## Inference (`/infer-pkg-connections`)
- **Hubs:** 35 function-hubs (0 god-functions), 83 contract-types. Widest contracts:
  `Claim` (52), `ProjectionCursor` (36), `Artifact` (30), `Verdict`, `Model`, `StageContext`.
- **Pipelines:** 7 named, 15 `DATA_FLOWS_TO` edges (most converge on the claim machinery).
- **Bridges:** 56 cross-**domain** `BRIDGES` edges (19 domain pairs). Literal cross-package = 0
  (single package, `packageName='app'`); analysis adapted to key on `domain`.
- **Layers:** 20 modules — domain 8 · infrastructure 6 · application 5 · testing 1. No
  presentation layer (it's a library). Root `src/cogworx` left unset.

### Import cycles (report-only, not persisted)
- `model ↔ model/providers`, `verification ↔ verification/oracles` — benign parent↔subpkg.
- `cost ↔ model`, `loop ↔ substrate` — to confirm.
- **`knowledge ↔ substrate` — REAL runtime coupling** (not `TYPE_CHECKING`). substrate adapters
  import domain types (`EvidenceEvent`/`ClaimConfidence`/`ProcedureSuccess`); knowledge imports
  the `EntityKG` **Protocol** back. Normal adapter pattern, resolves fine — noted, not a bug.

## New convention: `USER_CALLS` (the "user_calls_this" marker)
Dead-code by raw "no incoming `CALLS`" gave 483/814 (59%) — near-total false positives because
cog-worx is a **library** (public API called by external consumers + un-indexed `tests/`) and the
`CALLS` graph is sparse (395 edges; seeder doesn't resolve Protocol/registry/DI dispatch).

Fix (Jim-approved shape): a sentinel-node + dedicated-edge marker, kept **separate from `CALLS`**
so it doesn't skew hub/caller counts:

```
(:ExternalConsumer {kind})-[:USER_CALLS]->(:Function)
```

Three sources (565 edges, 482 distinct functions):
| kind | rule | fns |
|---|---|---|
| `public-api` | name in a subpackage `__all__`, **+** public methods of `__all__` classes | 362 |
| `test-suite` | `domain='testing'` (called by un-indexed `tests/`) | 183 |
| `decorated-hook` | runtime-invoked decorators (`pytest.fixture`, pydantic `*_validator`, `.setter`, `contextmanager`) | 20 |

Re-derivable from `__all__` + the graph on every `/sync-pkg` re-seed.

**Redefined dead-code query:** `no CALLS AND no USER_CALLS` → suspects fell **483 → 92**.
Residual 92 dominated by `substrate` (38) / `model-provider` (20) — Protocol-dispatched adapter
methods the seeder can't yet resolve. **No `possiblyDead` labels written** (Jim's call — known FPs).
Upstream fix: teach the seeder to emit `CALLS` through Protocols/registries/DI.
