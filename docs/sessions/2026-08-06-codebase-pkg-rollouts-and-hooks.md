# 2026-08-06 — codebase-pkg rollouts, MCP-server improvements, hook quieting

Tooling/infrastructure session — no cogworx `src/` changes (the modified `src/` files in the
working tree predate this session). No pod advanced; ROADMAP untouched by design.

## What happened (all work by Sonnet subagents, coordinator-routed)

1. **codebase-pkg v2 rolled out to agent-choreo + banister** (uncommitted in those repos):
   - agent-choreo: slug `agent-choreo-0d29`, neo4j bolt 8042, pg 5739. Only browser-extension
     TS seeded (15 files) — the parser is ts-morph-only, so the 114 Rust files in `crates/`
     are invisible to the graph.
   - banister: slug `banister-0014`, neo4j bolt 7993, pg 5682. 7 files seeded from
     `src/banister/` (tests/ has no `__init__.py` → excluded). Repo has ZERO git commits, so
     the seed's sync-cursor write crashed — incremental sync blocked until an initial commit.
   - Both: `.mcp.json` hand-patched to the local dist, all 6 pg tables bootstrapped, doctor
     0-fail. Known init bugs handled per the wombat playbook.

2. **All 10 items of codebase-pkg's `MCP-SERVER-IMPROVEMENTS.md` implemented** (uncommitted
   in codebase-pkg; suite 402 → 510 pass / 1 skip; dist rebuilt): learning supersede,
   secret redaction, near-dup insert-and-warn @0.85, graphRef validation, `resolveCaveat`,
   session upsert by sessionId, hybrid pg_trgm recall ranking (+`nearRefs`, recency
   tiebreak), `compact` mode + `getRecallItem`, `getSession` + sessionId filter,
   `format:"json"`. Schema changes are additive `IF NOT EXISTS` — live instances upgrade on
   next connection; sessions must restart to see the new tools.

3. **Hooks quieted in THIS repo** (the only cog-worx changes this session):
   - `.claude/hooks/canon-check.cjs` — FAIL now blocks via JSON `decision:"block"` +
     `suppressOutput` instead of a raw exit-2 stderr dump. Same blocking semantics/text.
   - `.claude/hooks/doc-check.cjs` — warning now rides non-blocking
     `hookSpecificOutput.additionalContext` (reaches the model, not the terminal).
   - `.claude/settings.json` inline Bash CANON guard — was emitting JSON + exit 2 (JSON
     ignored per protocol, raw text dumped); now a proper `permissionDecision:"deny"` with
     try/catch. Same deny conditions.
   - `.claude/hooks/codebase-pkg-*.cjs` — `suppressOutput: true` added (synced with the
     codebase-pkg template).

## Follow-ups
- Commit debt in codebase-pkg (v2 wave + improvements + template hooks) awaits Jim.
- banister needs an initial commit; agent-choreo's Rust needs upstream parser support.
- All consumer repos: restart sessions to pick up new MCP tools; run `/sync-pkg`.
- Known Windows fragility in canon-check (pre-existing, unfixed): `git diff -- 'src/**/*.py'`
  single quotes are literal under cmd.exe; works today only because Python is nested under
  `src/cogworx/`.
- 4 upstream codebase-pkg init bugs pending filing: state-block drop, npx `.mcp.json`
  stanza, sync-before-bootstrap, no `agent_*` CLI bootstrap.
