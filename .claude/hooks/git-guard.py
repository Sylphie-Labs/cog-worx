#!/usr/bin/env python3
"""git-guard: a PreToolUse hook on Bash that shows what a worktree-destroying
git command would discard, before it runs.

Installed by init_repo.py into a repo's .claude/hooks/ and wired as

    {"matcher": "Bash", "hooks": [{"command": "sh .../hook.sh git-guard"}]}

hook.sh runs this file with python3 (or python under Git Bash) on every
platform; there is no .sh/.ps1 pair (dec_01M21SWDC62XZVBVBMZ745R64K).

WHY. Every other guard in a connected repo is a PreToolUse hook on Edit|Write,
so a Bash tool call is a hole straight through all of them, and the ordinary
way to revert a file -- `git checkout -- path` -- silently took a session's
uncommitted review fixes with it (tik_01M21993QFY45YPTGEH2PXS36J). The point
is not prevention: discarding is often exactly what is meant. The point is
that the loss is VISIBLE before it happens.

CONTRACT.
  exit 0, silent   nothing would be lost; or not a git command this guard
                   knows; or the override prefix is present
  exit 2, stderr   something would be lost: what (git's own words, scoped to
                   the affected paths), the recoverable alternative, and how
                   to proceed deliberately
  exit 1, stderr   the guard itself could not do its job (unparseable
                   command, unexpected error). Claude Code shows the line and
                   the command proceeds: a broken guard must not block every
                   Bash call, but it must not be silent about standing aside.

RULES (what is checked, and what "would be lost" means):
  checkout <paths> / restore <paths>   unstaged changes to those paths
                                       (staged too when a tree-ish is given,
                                       or restore has --staged --worktree)
  checkout -f <branch>                 all uncommitted changes to tracked files
  reset --hard                         all uncommitted changes to tracked files
  clean -f (any spelling)              what `git clean --dry-run` with the same
                                       flags says it would remove
  stash drop / stash clear             the stash entries that would go
  branch -D / --delete --force         commits on that branch that no other
                                       ref (branch, remote, tag) and not HEAD
                                       still reaches
Anything else, including branch switches (git refuses to clobber on its own),
`restore --staged` alone, `reset --soft/--mixed`, `clean -n`, and `stash
pop`, is allowed without comment.

OVERRIDE. Re-run with the prefix `HIVE_GIT_GUARD=allow git ...`. The prefix is
read from the command text, so the intent is in the transcript, and nothing
is remembered between calls.

Compound commands are walked segment by segment (&&, ||, ;, |, backticks,
$(...), newlines); a leading `cd` and git's own `-C` are honoured so the check
runs where the command would. Heredoc bodies are skipped. A report is capped
at 40 lines. Standard library only.
"""
import json
import os
import re
import shlex
import subprocess
import sys

OVERRIDE_VAR = "HIVE_GIT_GUARD"
OVERRIDE_VALUE = "allow"
SEGMENT_BREAKS = {";", "&&", "||", "|", "&", "|&", "(", ")", ";;", "`"}
PUNCTUATION = "();<>|&`"   # shlex's default set plus the backtick
MAX_REPORT_LINES = 40
PREFIX_WORDS = {"env", "command", "exec", "nohup", "time", "builtin"}
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
HEREDOC_RE = re.compile(r"<<-?\s*(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\1")
# Global git options that take a value, and ones that do not.
GIT_GLOBAL_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
GIT_GLOBAL_FLAGS = {"-p", "--paginate", "-P", "--no-pager", "--no-replace-objects",
                    "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
                    "--icase-pathspecs", "--no-optional-locks", "--bare"}


class GuardError(Exception):
    """The guard could not do its job; reported on stderr with exit 1."""


# ---------------------------------------------------------------- parsing

def strip_heredocs(command):
    """Drop heredoc bodies so their lines are not read as commands."""
    out = []
    lines = command.split("\n")
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        m = HEREDOC_RE.search(line)
        i += 1
        if m:
            terminator = m.group(2)
            j = i
            while j < len(lines) and lines[j].strip() != terminator:
                j += 1
            if j < len(lines):
                i = j + 1  # skip the body and the terminator line
            # else: `<<` inside a quoted string, not a heredoc; keep reading
    return out


def segments(command):
    """Split a shell command into simple-command token lists.

    Quotes are honoured (so `echo 'git reset --hard'` is one echo argument),
    operators split, newlines split. Raises GuardError on shell the tokenizer
    cannot read, such as an unterminated quote.
    """
    text = " ; ".join(strip_heredocs(command))
    # posix=True strips an unquoted backslash, which is what Git Bash does
    # before git sees the word, so a Windows path in a command tokenises the
    # way the shell would run it. Quoted forms keep their backslashes.
    lex = shlex.shlex(text, posix=True, punctuation_chars=PUNCTUATION)
    lex.whitespace_split = True
    try:
        tokens = list(lex)
    except ValueError as e:
        raise GuardError(f"could not parse the command ({e}); not checked")
    out, cur = [], []
    skip_next = False
    for tok in tokens:
        if skip_next:
            skip_next = False
            continue
        if tok in SEGMENT_BREAKS:
            if cur:
                out.append(cur)
            cur = []
        elif tok and all(ch in "<>&" for ch in tok):
            # A redirection: the next token is its target, not an argument,
            # and a bare digit just before it (2>&1) is the fd, not an argument.
            if cur and cur[-1].isdigit():
                cur.pop()
            skip_next = True
        else:
            cur.append(tok)
    if cur:
        out.append(cur)
    return out


def split_prefix(seg):
    """Peel `VAR=x ... env command` off a segment. Returns (overridden, rest)."""
    overridden = False
    i = 0
    while i < len(seg):
        tok = seg[i]
        if ASSIGNMENT_RE.match(tok):
            name, _, value = tok.partition("=")
            if name == OVERRIDE_VAR and value == OVERRIDE_VALUE:
                overridden = True
            i += 1
        elif tok in PREFIX_WORDS:
            i += 1
        else:
            break
    return overridden, seg[i:]


def is_git(word):
    base = os.path.basename(word)
    return base == "git" or base == "git.exe"


def parse_git(rest, cwd):
    """(subcommand, args, cwd) for a `git ...` segment, or None."""
    i = 1
    while i < len(rest):
        tok = rest[i]
        if tok in GIT_GLOBAL_WITH_VALUE:
            if tok == "-C" and i + 1 < len(rest):
                cwd = os.path.normpath(os.path.join(cwd, rest[i + 1]))
            i += 2
        elif tok.startswith("--") and "=" in tok and tok.split("=", 1)[0] in GIT_GLOBAL_WITH_VALUE:
            i += 1
        elif tok in GIT_GLOBAL_FLAGS or (tok.startswith("-") and tok != "--"):
            i += 1
        else:
            return tok, rest[i + 1:], cwd
    return None


# ---------------------------------------------------------------- git

def run_git(cwd, *args):
    """git's stdout, or None when git failed or is absent (then there is
    nothing to protect that git itself will not refuse).

    Every call here is read-only. --no-optional-locks keeps `diff` from
    refreshing the index, so the guard never takes index.lock. OSError also
    covers a cwd that does not exist (a `cd` to nowhere): the shell would
    fail too, so allowing silently is right.
    """
    try:
        p = subprocess.run(
            ["git", "--no-optional-locks", *args], cwd=cwd, capture_output=True,
            encoding="utf-8", errors="replace",
        )
    except OSError:
        return None
    if p.returncode != 0:
        return None
    return p.stdout


def options_and_paths(args):
    """Split args into options and non-option words; `--` ends options."""
    opts, words = [], []
    seen_dashdash = False
    for a in args:
        if seen_dashdash:
            words.append(a)
        elif a == "--":
            seen_dashdash = True
        elif a.startswith("-"):
            opts.append(a)
        else:
            words.append(a)
    return opts, words, seen_dashdash


def short_flags(opts):
    """Every single-letter flag present in combined short options."""
    flags = set()
    for o in opts:
        if o.startswith("-") and not o.startswith("--"):
            flags.update(o[1:])
    return flags


def is_commitish(cwd, word):
    """True when git resolves `word` to a commit. Git tries the ref before the
    path, so a branch named like a dirty file is a branch (checkout carries
    the modification over; nothing is lost)."""
    return run_git(cwd, "rev-parse", "--verify", "--quiet", word + "^{commit}") is not None


def only_deletions(cwd, base, paths):
    """True when every change to `paths` is a worktree deletion: then
    `checkout -- path` is the recovery, not the loss."""
    out = run_git(cwd, "diff", "--name-status", *base, "--", *paths)
    if not out or not out.strip():
        return False
    return all(line.startswith("D") for line in out.splitlines())


def quoted_paths(paths):
    return " ".join(shlex.quote(p) for p in paths)


# ---------------------------------------------------------------- rules

class Finding:
    def __init__(self, what, lost, keep, cmd):
        self.what, self.lost, self.keep, self.cmd = what, lost, keep, cmd


def rule_checkout(args, cwd, cmd):
    opts, words, dashdash = options_and_paths(args)
    flags = short_flags(opts)
    if {"b", "B", "p"} & flags or {"--orphan", "--detach", "--patch"} & set(opts):
        return None
    if dashdash:
        # Everything before -- that is not an option is a tree-ish; the
        # paths are exactly what follows it.
        before = [a for a in args[:args.index("--")] if not a.startswith("-")]
        treeish, paths = (before[0] if before else None), args[args.index("--") + 1:]
    else:
        treeish, paths = None, []
        for w in words:
            if treeish is None and is_commitish(cwd, w):
                treeish = w
            elif os.path.exists(os.path.join(cwd, w)):
                paths.append(w)
            elif treeish is None:
                treeish = w
    if not paths:
        if "f" in flags or "--force" in opts:
            lost = run_git(cwd, "diff", "HEAD", "--stat")
            if lost and lost.strip():
                return Finding("would discard every uncommitted change to tracked files",
                               lost, "git stash push", cmd)
        return None
    base = ["HEAD"] if treeish else []
    lost = run_git(cwd, "diff", *base, "--stat", "--", *paths)
    if lost and lost.strip() and not only_deletions(cwd, base, paths):
        return Finding("would discard uncommitted changes", lost,
                       "git stash push -- " + quoted_paths(paths), cmd)
    return None


def rule_restore(args, cwd, cmd):
    opts, words, _ = options_and_paths(args)
    flags = short_flags(opts)
    staged = "--staged" in opts or "S" in flags
    worktree = "--worktree" in opts or "W" in flags or not staged
    if not worktree or "p" in flags or "--patch" in opts:
        return None
    paths = [w for w in words]
    # `--source X` / `-s X` take a value that is not a path.
    for i, a in enumerate(args):
        if a in ("--source", "-s") and i + 1 < len(args) and args[i + 1] in paths:
            paths.remove(args[i + 1])
    if not paths:
        return None
    base = ["HEAD"] if staged else []
    lost = run_git(cwd, "diff", *base, "--stat", "--", *paths)
    if lost and lost.strip() and not only_deletions(cwd, base, paths):
        return Finding("would discard uncommitted changes", lost,
                       "git stash push -- " + quoted_paths(paths), cmd)
    return None


def rule_reset(args, cwd, cmd):
    if "--hard" not in args:
        return None
    lost = run_git(cwd, "diff", "HEAD", "--stat")
    if lost and lost.strip():
        return Finding("would discard every uncommitted change to tracked files",
                       lost, "git stash push", cmd)
    return None


def rule_clean(args, cwd, cmd):
    opts, _, _ = options_and_paths(args)
    flags = short_flags(opts)
    if "n" in flags or "--dry-run" in opts or "i" in flags or "--interactive" in opts:
        return None
    if "f" not in flags and "--force" not in opts:
        return None  # git refuses on its own unless clean.requireForce is off
    # Forward the user's flags minus force and quiet: -q would silence the
    # very lines the check reads, and -f is what the dry-run replaces.
    forwarded = []
    for a in args:
        if a in ("--force", "--quiet"):
            continue
        if a.startswith("-") and not a.startswith("--") and len(a) > 1:
            kept = "".join(ch for ch in a[1:] if ch not in "fq")
            if not kept:
                continue
            a = "-" + kept
        forwarded.append(a)
    would = run_git(cwd, "clean", "--dry-run", *forwarded)
    if would and would.strip():
        return Finding("would delete files git does not track", would,
                       "git stash push --include-untracked   (or review with: git clean -n " + " ".join(args) + ")", cmd)
    return None


def rule_stash(args, cwd, cmd):
    sub = args[0] if args else "push"
    if sub not in ("drop", "clear"):
        return None
    listing = run_git(cwd, "stash", "list")
    if not listing or not listing.strip():
        return None
    lines = listing.splitlines()
    if sub == "drop":
        ref = next((a for a in args[1:] if not a.startswith("-")), None)
        if ref is not None and ref.isdigit():
            ref = "stash@{%s}" % ref
        lines = [l for l in lines if l.startswith(ref + ":")] if ref else lines[:1]
        if not lines:
            return None
        keep = "git stash show -p " + (ref or "stash@{0}") + "   (review, or: git stash branch <name>)"
    else:
        keep = "git stash list   (review each with git stash show -p)"
    return Finding("would delete stashed work", "\n".join(lines) + "\n", keep, cmd)


def rule_branch(args, cwd, cmd):
    opts, names, _ = options_and_paths(args)
    flags = short_flags(opts)
    force_delete = ("D" in flags) or (("d" in flags or "--delete" in opts) and ("f" in flags or "--force" in opts))
    if not force_delete or "r" in flags or "--remotes" in opts:
        return None
    lost_all = []
    for name in names:
        # Lost only if no OTHER ref (branch, remote-tracking, tag) and not
        # HEAD reaches the tip: a branch merged into develop with --no-ff is
        # the everyday gitflow cleanup, and -d refuses it too, which is why
        # people reach for -D there; blocking it would only train the override.
        holders = run_git(cwd, "for-each-ref", "--format=%(refname)", "--contains", name,
                          "refs/heads", "refs/remotes", "refs/tags")
        others = [r for r in (holders or "").split() if r != "refs/heads/" + name]
        if others or run_git(cwd, "merge-base", "--is-ancestor", name, "HEAD") is not None:
            continue
        ahead = run_git(cwd, "log", "--oneline", "-20", "HEAD.." + name, "--")
        if ahead and ahead.strip():
            lost_all.append(f"{name}:\n{ahead}")
    if not lost_all:
        return None
    return Finding("would delete commits not reachable from HEAD", "".join(lost_all),
                   "git tag backup/" + names[0] + " " + names[0] + "   (a tag keeps them reachable)", cmd)


RULES = {
    "checkout": rule_checkout,
    "restore": rule_restore,
    "reset": rule_reset,
    "clean": rule_clean,
    "stash": rule_stash,
    "branch": rule_branch,
}


# ---------------------------------------------------------------- main

def findings_for(command, cwd):
    found = []
    here = cwd
    for seg in segments(command):
        overridden, rest = split_prefix(seg)
        if not rest:
            continue
        if rest[0] in ("cd", "pushd") and len(rest) > 1 and rest[1] != "-":
            here = os.path.normpath(os.path.join(here, os.path.expanduser(rest[1])))
            continue
        if not is_git(rest[0]) or overridden:
            continue
        parsed = parse_git(rest, here)
        if not parsed:
            continue
        sub, args, git_cwd = parsed
        rule = RULES.get(sub)
        if rule is None:
            continue
        finding = rule(args, git_cwd, " ".join(shlex.quote(t) for t in rest))
        if finding:
            found.append(finding)
    return found


def main():
    try:
        payload = json.loads(sys.stdin.read() or "null")
    except ValueError:
        return 0
    if not isinstance(payload, dict) or payload.get("tool_name") != "Bash":
        return 0
    tool_input = payload.get("tool_input")
    command = tool_input.get("command") if isinstance(tool_input, dict) else None
    if not isinstance(command, str) or not command.strip():
        return 0
    cwd = payload.get("cwd") or os.environ.get("CLAUDE_PROJECT_DIR") or os.getcwd()
    try:
        found = findings_for(command, cwd)
    except GuardError as e:
        print(f"git-guard: {e}", file=sys.stderr)
        return 1
    if not found:
        return 0
    for f in found:
        print(f"git-guard: `{f.cmd}` {f.what}:", file=sys.stderr)
        lines = f.lost.rstrip("\n").splitlines()
        shown = lines if len(lines) <= MAX_REPORT_LINES else lines[:MAX_REPORT_LINES]
        for line in shown:
            print("  " + line, file=sys.stderr)
        if len(lines) > MAX_REPORT_LINES:
            print(f"  ... {len(lines) - MAX_REPORT_LINES} more lines", file=sys.stderr)
        print(f"Keep it recoverable instead: {f.keep}", file=sys.stderr)
        print(f"If the loss is intended, re-run with the prefix on the git command itself: "
              f"{OVERRIDE_VAR}={OVERRIDE_VALUE} {f.cmd}", file=sys.stderr)
    if len(found) > 1 or any(f.cmd != command.strip() for f in found):
        print("(In a compound command the prefix goes on that segment, not at the front.)", file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 -- a broken guard must not block every Bash call
        print(f"git-guard: internal error, not checked: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
