# Enforce CANON

Run a CANON compliance check against a proposal, plan, spike, or code change.

## Usage
```
/enforce-canon                 # check current uncommitted changes (git diff)
/enforce-canon "proposal text" # check a specific proposal
/enforce-canon "Phase 1 spine" # check a phase/pod plan
```

## When to use
- Before any architectural change or new pod/feature build.
- When reviewing a phase or ⛓ spike plan (`wiki/ROADMAP.md`).
- When a proposal feels like it might drift from a CANON Standard — especially the cog-worx
  divergences from biz-firm (S2/S3/S6/S7, CANON §7).
- Automatically at session end (the `canon-check` Stop-hook checks `src/**/*.py` diffs on this path).

## Workflow
1. Read `wiki/CANON.md` **in full** (fresh — never from memory).
2. Identify the subject under review (the diff, the plan, or the pasted proposal).
3. Spawn the **canon** agent to run the full enforcement checklist (Standards S1–S12,
   scope/phase boundaries §3, planning rules §4).
4. Return a structured verdict: **COMPLIANT | NON-COMPLIANT | COMPLIANT WITH CONCERNS**,
   with every violation citing the specific Standard (e.g. `S6`) or section.

## Key rules
- The canon agent always reads the CANON fresh; it never works from memory.
- Enforce **cog-worx's** CANON, not biz-firm's — the journal is TimescaleDB (S6), the substrate is
  polyglot (S3), and the connecting platform is deferred (S7).
- Every violation cites the exact Standard/section. Vague "spirit of the project" appeals are not enough.
- NON-COMPLIANT blocks work until resolved or **Jim** explicitly overrides.
- The canon agent enforces; it never edits the CANON.
