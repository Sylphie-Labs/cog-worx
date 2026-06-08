# Update CANON

Propose and apply a change to `wiki/CANON.md`. **Requires Jim's explicit approval.**

## Usage
```
/update-canon "Resolve open decision: latent-space tiering policy is hot=7d / cold=archive"
/update-canon "Add Standard S13: ..."
/update-canon "Amend S3: add DuckDB for the analytics seam"
```

## When to use
- A CANON gap is found during planning or implementation.
- An open decision gets decided and must become law.
- Jim requests a CANON change.

## Workflow
### Phase 1 — Document the change
Draft: (a) what the CANON currently says, (b) the proposed change, (c) why it's needed,
(d) impact on existing plans / `wiki/ROADMAP.md` / affected layers, and (e) whether it changes any of
the deliberate divergences from biz-firm recorded in CANON §7.

### Phase 2 — Jim's approval
Present the proposal to Jim. **The CANON is immutable unless Jim explicitly approves** (CANON §6).

### Phase 3 — Apply
If approved: edit `wiki/CANON.md`; update any affected docs (especially `wiki/ROADMAP.md` and the §7
deltas if a divergence changed); record the change in a CHANGELOG if one exists; notify relevant agents.

## Key rules
- NEVER modify the CANON without Jim's explicit approval. If the enforcer can rewrite the law, there is
  no law (CANON §6).
- Every change carries a rationale + an impact assessment.
- Keep `wiki/ROADMAP.md` (and the §7 divergence list) in sync with the change.
