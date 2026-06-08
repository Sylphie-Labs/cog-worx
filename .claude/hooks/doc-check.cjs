// Hook: Documentation compliance check before session completion.
// Runs on Stop — reminds the agent to keep cog-worx's docs in step with code.
// Warn-only (never blocks).

const { execSync } = require("child_process");

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
      process.stderr.write(message);
      process.exit(0); // Warn only
    }

    process.exit(0);
  } catch (e) {
    process.exit(0);
  }
});
