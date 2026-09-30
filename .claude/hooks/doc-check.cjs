// Hook: Documentation compliance check before session completion.
// Runs on Stop — reminds the agent to keep cog-worx's docs in step with code.
// Warn-only (never blocks).
//
// Output contract: this hook is non-blocking by design, so it never uses
// {"decision":"block"}. The reminder is still delivered to the model via
// hookSpecificOutput.additionalContext (Stop/SubagentStop support
// non-blocking additionalContext injection — the conversation continues,
// Claude can act on the note) while "suppressOutput":true keeps the raw JSON
// out of the user-facing terminal. Always exit(0); no warnings -> no output.
//
// Loop guard: injecting additionalContext on Stop re-invokes the model, whose
// reply triggers another Stop — so an unchanged warning set must fire at most
// once per session (tmpdir marker keyed by session + warning hash) or the
// hook loops forever on a persistent working-tree condition.

const { execSync } = require("child_process");
const crypto = require("crypto");
const fs = require("fs");
const os = require("os");
const path = require("path");

const projectDir = process.env.CLAUDE_PROJECT_DIR || process.cwd();
const gitOpts = { encoding: "utf-8", timeout: 5000, cwd: projectDir };

let data = "";
process.stdin.on("data", (chunk) => (data += chunk));
process.stdin.on("end", () => {
  try {
    const warnings = [];

    let diffOutput = "";
    try {
      diffOutput = execSync("git diff --name-status HEAD", gitOpts).trim();
    } catch {
      process.exit(0);
    }

    if (!diffOutput) {
      process.exit(0);
    }

    const lines = diffOutput.split("\n").filter(Boolean);
    const changes = lines.map((line) => {
      const [status, ...pathParts] = line.split("\t");
      return { status: status.trim(), path: pathParts.join("\t").replace(/\\/g, "/") };
    });

    // Only care about src/ changes (the framework package)
    const srcChanges = changes.filter((c) => c.path.startsWith("src/"));
    if (srcChanges.length === 0) {
      process.exit(0);
    }

    // Check: a session log for today (docs/sessions/ is optional — only nudge if the dir exists).
    const today = new Date().toISOString().slice(0, 10);
    let sessionDirExists = false;
    let sessionLogExists = false;
    try {
      const sessionFiles = execSync("git diff --name-only HEAD -- docs/sessions/", gitOpts).trim();
      const untrackedSessions = execSync(
        'git ls-files --others --exclude-standard -- "docs/sessions/"',
        gitOpts
      ).trim();
      const trackedSessions = execSync('git ls-files -- "docs/sessions/"', gitOpts).trim();
      sessionDirExists = Boolean(trackedSessions || untrackedSessions);
      const allSessionChanges = (sessionFiles + "\n" + untrackedSessions).trim();
      sessionLogExists = allSessionChanges.includes(today);
    } catch {
      // docs/sessions/ may not exist yet
    }

    if (sessionDirExists && !sessionLogExists) {
      warnings.push(
        `No session log for ${today} in docs/sessions/ — write a brief log of what changed and why`
      );
    }

    // Check: load-bearing src/ changes should keep the ROADMAP checkboxes honest.
    const roadmapTouched = changes.some((c) => c.path.includes("wiki/ROADMAP.md"));
    const addedOrModifiedSrc = srcChanges.some((c) => c.status === "A" || c.status === "M");
    if (addedOrModifiedSrc && !roadmapTouched) {
      warnings.push(
        "src/ changed but wiki/ROADMAP.md was not — if a pod/feature advanced or hardened, tick its box (a feature only hardens once its spike + Feature Test Bundle pass, S12)"
      );
    }

    if (warnings.length > 0) {
      const message =
        "DOCUMENTATION CHECK — before completing, address these items:\n\n" +
        warnings.map((w, i) => `  ${i + 1}. ${w}`).join("\n") +
        "\n\nResolve these, then confirm completion. If an item doesn't apply, explain why.";

      // Fire at most once per session for the same warning set (see loop guard note).
      let sessionId = "unknown";
      try {
        sessionId = JSON.parse(data).session_id || "unknown";
      } catch {}
      const hash = crypto.createHash("sha256").update(message).digest("hex").slice(0, 16);
      const marker = path.join(os.tmpdir(), `cogworx-doc-check-${sessionId}`);
      try {
        if (fs.existsSync(marker) && fs.readFileSync(marker, "utf-8") === hash) {
          process.exit(0); // already warned this session, warnings unchanged
        }
        fs.writeFileSync(marker, hash);
      } catch {}

      process.stdout.write(
        JSON.stringify({
          suppressOutput: true,
          hookSpecificOutput: {
            hookEventName: "Stop",
            additionalContext: message,
          },
        }),
      );
      process.exit(0); // Warn only — non-blocking
    }

    process.exit(0);
  } catch {
    process.exit(0);
  }
});
