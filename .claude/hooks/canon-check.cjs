// Hook: CANON compliance check before session completion.
// Runs on Stop — spawns claude (Sonnet) to verify Python changes in src/
// comply with cog-worx's CANON (the Immutable Standards S1–S12).
// Blocks completion if violations are found.

const { execSync } = require("child_process");

const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const gitOpts = { encoding: "utf-8", timeout: 10000, maxBuffer: 1024 * 1024, cwd: projectDir };

let data = "";
process.stdin.on("data", (chunk) => (data += chunk));
process.stdin.on("end", () => {
  try {
    // Get diff of changed Python files in src/
    let diff = "";
    try {
      diff = execSync("git diff HEAD -- 'src/**/*.py'", gitOpts).trim();
    } catch {
      process.exit(0);
    }

    if (!diff) {
      process.exit(0);
    }

    // Also include untracked new .py files in src/
    let newFiles = "";
    try {
      const untrackedList = execSync(
        'git ls-files --others --exclude-standard -- "src/"',
        { ...gitOpts, timeout: 5000 }
      ).trim();

      if (untrackedList) {
        const pyFiles = untrackedList.split("\n").filter((f) => f.endsWith(".py"));
        for (const f of pyFiles) {
          try {
            const content = execSync(`cat "${f}"`, { ...gitOpts, timeout: 5000 });
            newFiles += `\n--- NEW FILE: ${f} ---\n${content}\n`;
          } catch {
            // skip unreadable files
          }
        }
      }
    } catch {
      // skip
    }

    const fullDiff = diff + newFiles;

    const maxChars = 50000;
    const truncatedDiff =
      fullDiff.length > maxChars
        ? fullDiff.slice(0, maxChars) + "\n\n[TRUNCATED — diff too large]"
        : fullDiff;

    const prompt = `You are the CANON compliance checker for the cog-worx project (a publishable Python agent framework).

Read wiki/CANON.md using the Read tool. Then review the code diff below for violations of the Immutable Standards (S1–S12). cog-worx deliberately diverges from biz-firm on S2/S3/S6/S7 (see CANON §7) — judge against cog-worx's CANON, not biz-firm's.

CHECK FOR THESE SPECIFIC VIOLATIONS:

1. S1 — MODEL WORK ON THE WRITE PATH: any LLM/model call on the write/commit path. Extraction, reflection, and coherence reconciliation must run asynchronously and batched, off the hot path. ("Just summarize on write" / per-write extraction is a violation.)

2. S2 — OWN THE LOOP: importing a framework or managed durable-execution dependency that owns the control loop; a runtime dependency under the differentiated layer. (OSS is reference, not dependency; AGPL/Elastic is read-only.)

3. S3 — POLYGLOT SUBSTRATE: a generic Store abstraction that flattens the engines; graph claims persisted outside Neo4j; the durable journal outside TimescaleDB; latent vectors outside pgvector. Each engine is chosen for its strength — Neo4j = graph KGs, Postgres/pgvector = latent space, TimescaleDB = journal, OTel = spans. The only allowed seam is a thin internal one for tests/mocks.

4. S4 — MODEL-AGNOSTIC: a hardcoded provider; logic that only works for one vendor without graceful degradation. Provider is selectable per agent through a thin Model interface.

5. S5 — PROVENANCE + EPISTEMIC TYPING: a substrate write missing provenance or an epistemic type (observation | inference | confirmed); an inference stored as a fact; claims silently merged across epistemic levels.

6. S6 — DURABLE EXACTLY-ONCE: in-memory-only progress; a step marked done before its result is committed to the TimescaleDB journal; replay/resume that re-invokes the model; non-idempotent side effects.

7. S7 — COORDINATION BY CONTRACT: building the connecting/orchestration platform now (deferred); state passed agent-to-agent instead of through the substrate; blocking waits instead of a non-blocking stage; two writers to one entity (must be exactly one write-token per entity).

8. S9 — STRUCTURE OVER PROMPTING / NEVER TRUST SELF-REPORT: using the model's self-assessment as a control signal (self-correction without ground truth, self-confidence gating, model-enforced budgets); "ask the model nicely" where a structural guarantee is required.

9. S10 — SECURITY BY STRUCTURE: content filtering as the primary prompt-injection defense; external-tier tools available while reading untrusted content (the lethal trifecta must be broken structurally via Plan-Then-Execute / tool-dropping).

10. S11 — COST BOUNDED STRUCTURALLY: relying on the model to self-terminate; an unbounded loop without a step/cost ceiling; budgets not enforced as pre-call guards.

11. S12 — SPIKE-GATED: building a sector's load-bearing logic before its falsifiable spike has passed.

RESPOND WITH EXACTLY ONE OF:

If NO violations found:
CANON_CHECK: PASS

If violations found:
CANON_CHECK: FAIL
- [file:line] Description of violation and which Standard (S#) / section it violates

Nothing else. No preamble, no explanation beyond the violation list.

CODE DIFF TO CHECK:
${truncatedDiff}`;

    let result = "";
    try {
      result = execSync(
        `claude -p --model sonnet --allowedTools "Read,Grep,Glob" --no-session-persistence`,
        {
          input: prompt,
          encoding: "utf-8",
          timeout: 120000,
          maxBuffer: 1024 * 512,
          stdio: ["pipe", "pipe", "pipe"],
          cwd: projectDir,
        }
      ).trim();
    } catch (e) {
      process.stderr.write(
        "CANON CHECK: Could not run compliance check (claude CLI error). Proceeding with warning.\n"
      );
      if (e.stderr) process.stderr.write(e.stderr.toString());
      process.exit(0);
    }

    if (result.includes("CANON_CHECK: FAIL")) {
      process.stderr.write(
        "CANON COMPLIANCE VIOLATION DETECTED\n\n" +
          result +
          "\n\nFix the violations above before completing. The CANON is the single source of truth.\n"
      );
      process.exit(2);
    }

    if (result.includes("CANON_CHECK: PASS")) {
      process.exit(0);
    }

    process.stderr.write(
      "CANON CHECK: Unexpected response from compliance checker:\n" + result.slice(0, 500) + "\n"
    );
    process.exit(0);
  } catch (e) {
    process.stderr.write("CANON CHECK: Hook error — " + e.message + "\n");
    process.exit(0);
  }
});
