#!/usr/bin/env python3
"""secret-guard: a PreToolUse hook on Bash that refuses a command whose
normal output is a secret.

Installed by init_repo.py into a repo's .claude/hooks/ and wired as

    {"matcher": "Bash", "hooks": [{"command": "sh .../hook.sh secret-guard"}]}

hook.sh runs this file with python3 (or python under Git Bash) on every
platform; there is no .sh/.ps1 pair (dec_01M21SWDC62XZVBVBMZ745R64K).

WHY. An agent checking credential plumbing ran `git credential fill`, and git
printed a live OAuth token into the tool output: into the transcript on disk,
one Stop hook away from the hive (tik_01M1MK92XRJ1CZSV52AQ35Q8HA). Nothing
stopped the command, because its success output IS the secret. This guard
stops that class of command and names the check that would have been safe.

CONTRACT.
  exit 0, silent   no rule matched, or the user's allow file covers the match
  exit 2, stderr   a rule matched: which rule, why, the safe alternative, and
                   the allow-file line that would permit it
  exit 1, stderr   the guard itself could not do its job (unparseable
                   command, unexpected error). Claude Code shows the line and
                   the command proceeds: a broken guard must not block every
                   Bash call, but it must not be silent about standing aside.

RULES. The RULES table below is the list; each has an id.
It covers: git credential fill/get and the credential helpers; gh auth token;
keychain lookups; printenv, env, a bare export/declare/typeset/set, and echo
or printf of a plain $NAME whose name looks secret (TOKEN, KEY, PASSWORD...);
cat and the other readers on .env files, private keys, .pem/.key files and
the credential files under the home directory; kubectl/oc secret values; and
docker/podman inspect of the environment. A block message prints the rule and
a canonical form of the command (its program and the words the rule matched),
never the command itself: an inline value must not be copied into the message.

NO OVERRIDE AN AGENT CAN TYPE. git-guard takes a prefix on the command,
because discarding work is often exactly what is meant. Printing a secret
into a transcript is not, and an agent that can add a prefix can wave itself
through. Exceptions live in a file a person edits, outside the repo:

    ${XDG_CONFIG_HOME:-~/.config}/hive/secret-guard-allow     (POSIX)
    %APPDATA%\\hive\\secret-guard-allow                         (Windows)

One entry per line. A `#` at the start of a line, or after whitespace, starts a
comment that runs to the end of the line; a `#` inside a word is part of it.
    docker-inspect                  allows that rule everywhere
    secret-file: cat .env.test      allows it when the command matches the glob
The guard refuses any command that names that file or anything else in its
directory (rule `allow-file`), and that rule cannot itself be allowed. A line
that names an unknown rule, or `allow-file`, or has a rule and a colon but no
pattern, allows nothing; when the guard
blocks a command it says how many such lines there are (not what they say).

WHAT THIS IS NOT. It catches the honest mistake: the command an agent reaches
for when checking a credential. It does not stop a determined workaround
(`cp .env x; cat x`, or a script that opens the file itself). Redaction at
capture (hive_redact.py) is the backstop for what gets past it.

Compound commands are walked segment by segment (&&, ||, ;, |, backticks,
$(...), newlines); a substitution inside double quotes, `sh -c '...'` and
`eval ...` are parsed again; a leading `cd` is honored. Heredoc bodies are
skipped. A command whose standard output goes to /dev/null prints nothing and
is allowed: that is the exit-code check the messages recommend. A pipe or a
substitution is NOT treated that way, because what happens to the output next
cannot be known from the text. Standard library only.
"""
import fnmatch
import json
import os
import re
import shlex
import sys

# The shared tokenizer is imported from this file's own directory. Without
# this, the import would leave a __pycache__/ in every connected repo's
# .claude/hooks/.
sys.dont_write_bytecode = True
try:
    import hive_shell
except ImportError as e:
    print(f"secret-guard: hive_shell.py is missing next to this file ({e}); not checked. "
          "Re-run init_repo.py to restore it.", file=sys.stderr)
    sys.exit(1)

MAX_DEPTH = 3

# ---------------------------------------------------------------- the lists
# Both are data: adding an entry is a one-line change.

# Commands that print the files they are given.
READERS = {"cat", "head", "tail", "less", "more", "bat", "nl", "tac", "grep", "egrep",
           "fgrep", "rg", "sed", "awk", "cut", "sort", "strings", "base64", "xxd", "od",
           "hexdump"}
# Of those, the ones whose first operand is a pattern or script, not a file,
# unless one of these options supplied it instead.
PATTERN_FIRST = {"grep", "egrep", "fgrep", "rg", "sed", "awk"}
PATTERN_OPTIONS = {"-e", "-f", "--regexp", "--file", "--expression"}
# The grep family, and the options with which it prints no file content: only
# a count, a file name, or an exit code.
GREP_FAMILY = {"grep", "egrep", "fgrep", "rg"}
GREP_QUIET_LONG = {"--quiet", "--silent", "--count", "--files-with-matches", "--files-without-match"}
GREP_QUIET_SHORT = set("qclL")
# Short options that take a value: the rest of the cluster, or the next word.
GREP_VALUE_SHORT = set("ABCmefdD")
GREP_VALUE_LONG = {"--regexp", "--file", "--max-count", "--after-context", "--before-context",
                   "--context", "--directories", "--devices", "--exclude", "--include",
                   "--exclude-dir", "--binary-files", "--label"}

# Secret files under the home directory, relative to it.
HOME_FILES = (".aws/credentials", ".netrc", ".claude.json", ".config/gh/hosts.yml", ".npmrc",
              ".docker/config.json", ".git-credentials", ".pgpass", ".kube/config")
# Private key file names, with or without a suffix such as _github or -work
# (the .pub half is public and is not matched).
KEY_NAME_RE = re.compile(r"^id_(?:rsa|dsa|ecdsa|ed25519)(?:[_.-].+)?$")
# .env.<suffix> files that hold examples, not values.
ENV_EXAMPLE_SUFFIXES = {"example", "sample", "template", "dist", "defaults"}

# A variable name looks secret when one of its `_`-separated parts is a word
# from SECRET_PARTS, or it ends with one of SECRET_ENDINGS (AUTHTOKEN,
# PGPASSWORD), or it is or ends with a connection-string name. `KEY` is a
# part, not a substring: MONKEY and KEYBOARD_LAYOUT are not secrets, and
# PWD is the working directory.
SECRET_PARTS = {"TOKEN", "TOKENS", "SECRET", "SECRETS", "PASSWORD", "PASSWD", "PASS", "KEY",
                "KEYS", "APIKEY", "CREDENTIAL", "CREDENTIALS", "CREDS", "AUTH", "DSN", "PAT"}
SECRET_ENDINGS = ("TOKEN", "SECRET", "PASSWORD", "PASSWD", "APIKEY")
URL_NAMES = ("DATABASE_URL", "DB_URL", "REDIS_URL", "MONGODB_URI", "MONGO_URL",
             "CONNECTION_STRING", "CONN_STR")
# An expansion of a variable: `$NAME`, or `${...}` with an optional `#` (length)
# or `!` (indirection) before the name and anything up to the first `}` after.
EXPANSION_RE = re.compile(r"\$(?:\{([#!]?)([A-Za-z_][A-Za-z0-9_]*)([^}]*)\}?|([A-Za-z_][A-Za-z0-9_]*))")
# docker and podman options that come before the subcommand and take a value.
DOCKER_GLOBAL_VALUES = {"--context", "-c", "-H", "--host", "--config", "-l", "--log-level"}
GH_GLOBAL_VALUES = {"-R", "--repo"}

CAPTURE = 'to use the value without showing it, capture it: NAME="$(...)" some-command'

# rule id -> (why it is blocked, the safe alternative)
RULES = {
    "git-credential": (
        "it prints the stored credential",
        "discard the output and check the exit code (... >/dev/null 2>&1; echo $?), "
        "or watch a dummy credential get rejected; "
        + CAPTURE),
    "gh-token": (
        "it prints the GitHub token",
        "gh auth status   (says whether you are logged in, without the token); "
        + CAPTURE),
    "keychain": (
        "it reads a stored password out of the keychain",
        "discard the output and check the exit code (... >/dev/null 2>&1; echo $?); "
        + CAPTURE),
    "env-dump": (
        "it prints environment variables, which is where tokens and keys live",
        'test -n "${NAME:-}" or echo "${NAME:+set}" to check one variable is set; '
        'compgen -e to list names without values'),
    "secret-file": (
        "it prints a file that holds secrets",
        "check that the file exists (test -f), count or test for a line without printing it "
        "(grep -c, grep -q), or read its .example sibling; " + CAPTURE),
    "kubectl-secret": (
        "it prints the secret's values",
        "kubectl describe secret NAME   (shows keys and sizes, not values)"),
    "docker-inspect": (
        "it prints the container's environment, which is where its secrets are",
        "pick the fields you need: docker inspect --format '{{.State.Status}}' NAME"),
    "allow-file": (
        "it touches the secret guard's own allow file",
        "a person edits that file by hand, outside the session"),
}
# The highest-risk commands, matched as plain text when the tokenizer cannot
# read the command at all.
FALLBACK = [
    ("git-credential", "git credential fill|get",
     re.compile(r"\bgit\b[^|;&]*\bcredential\s+(?:fill|get)\b")),
    ("gh-token", "gh auth token", re.compile(r"\bgh\s+auth\s+token\b")),
    ("keychain", "security find-...-password",
     re.compile(r"\bsecurity\s+find-(?:generic|internet)-password\b")),
]


class Finding:
    """A rule that fired. `cmd` is the whole command text: allow patterns match
    it and de-duplication keys on it. `shown` is the only form of the command a
    block message may print: built by the rule from the program's basename and
    the fixed words it matched, never from option values. `file` is the matched
    file word of a secret-file finding, and `match` the text an allow pattern is
    matched against: `cmd`, except for secret-file, where it is `program file`
    so that an entry allows one file and not the whole command. `cmd` and `match`
    hold the full command text: they are matched against, never printed."""

    def __init__(self, rule, cmd, shown, file=None, match=None):
        self.rule, self.cmd, self.shown, self.file = rule, cmd, shown, file
        self.match = cmd if match is None else match


# ---------------------------------------------------------------- paths

def home_dir():
    # The shell's home is $HOME. On Windows os.path.expanduser reads
    # USERPROFILE instead, which Git Bash does not use for `~` or `$HOME`.
    return os.path.normpath(os.environ.get("HOME") or os.path.expanduser("~"))


def allow_file_path():
    if os.name == "nt":
        base = os.environ.get("APPDATA") or os.path.join(home_dir(), "AppData", "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home_dir(), ".config")
    return os.path.normpath(os.path.join(base, "hive", "secret-guard-allow"))


def config_dir():
    """The directory that holds the allow file: the guard's own, protected."""
    return os.path.dirname(allow_file_path())


def leading_variables():
    """(prefix, value) for each variable a word may start with. The shell would
    expand it; the tokenizer only sees the text."""
    xdg = os.environ.get("XDG_CONFIG_HOME") or os.path.join(home_dir(), ".config")
    table = [("${HOME}", home_dir()), ("$HOME", home_dir()),
             ("${XDG_CONFIG_HOME}", xdg), ("$XDG_CONFIG_HOME", xdg)]
    appdata = os.environ.get("APPDATA")
    if appdata:
        table += [("${APPDATA}", appdata), ("$APPDATA", appdata), ("%APPDATA%", appdata)]
    return table


def resolve(word, cwd):
    """The absolute path a command word names: `~`, `$HOME`, `$XDG_CONFIG_HOME`
    and `$APPDATA` (and their `${...}` forms) expanded, a Git Bash drive path
    translated, relative words joined to `cwd`."""
    for prefix, value in [("~", home_dir())] + leading_variables():
        if word == prefix or word[len(prefix):len(prefix) + 1] in ("/", "\\") and word.startswith(prefix):
            word = value + word[len(prefix):]
            break
    return os.path.normpath(os.path.join(cwd, os.path.expanduser(hive_shell.native_path(word))))


def fold(path):
    """`path` in the form every spelling of one file shares. macOS and Windows
    file systems ignore case, so `~/.AWS/credentials` is the credentials file
    there. On a file system that does not, this only makes the guard refuse a
    few names that are different files."""
    return path.lower()


def touches_config_dir(word, cwd):
    path = fold(resolve(word, cwd))
    base = fold(config_dir())
    return path == base or path.startswith(base + os.sep)


def secret_basename(name):
    name = fold(name)
    has_glob = any(c in name for c in "*?[")
    if name == ".env":
        return True
    if name.startswith(".env."):
        return name.rsplit(".", 1)[-1] not in ENV_EXAMPLE_SUFFIXES
    if name.startswith(".env") and has_glob:
        # `.env*` reaches the real files; `.env*.example` reaches only examples.
        return name.rsplit(".", 1)[-1] not in ENV_EXAMPLE_SUFFIXES
    if name == ".envrc":
        return True
    if name.endswith(".pub"):
        return False
    if KEY_NAME_RE.match(name):
        return True
    if name.startswith("id_") and has_glob:
        return True
    return name.endswith((".pem", ".key"))


# A name that ends like this holds a socket path, a count, a file location or the
# like, not the secret itself (MY_TOKEN_FILE, PASS_COUNT).
NOT_SECRET_ENDINGS = ("_SOCK", "_COUNT", "_FILE", "_PATH", "_DIR", "_NAME", "_LEN", "_LENGTH")


def is_secret_name(name):
    """True when an environment variable's name looks like it holds a secret."""
    upper = name.upper()
    if upper.endswith(NOT_SECRET_ENDINGS):
        return False
    return (any(part in SECRET_PARTS for part in upper.split("_"))
            or upper.endswith(SECRET_ENDINGS) or upper.endswith(URL_NAMES))


class Taint:
    """The names that hold a secret in this command: assigned from a secret
    command or from another tainted name. A nested check starts from its
    parent's names and keeps its own additions to itself."""

    def __init__(self, parent=None):
        self.names = set()
        self.order = []       # the names in the order they were added
        self.parent = parent

    def __contains__(self, name):
        node = self
        while node is not None:
            if name in node.names:
                return True
            node = node.parent
        return False

    def add(self, name):
        if name not in self.names:
            self.names.add(name)
            self.order.append(name)

    def forget_since(self, mark):
        """Drop the names added after `mark` (a len(order) taken earlier)."""
        for name in self.order[mark:]:
            self.names.discard(name)
        del self.order[mark:]


def shown_names(text):
    """The names whose value an expansion in `text` can display: every
    expansion except the two that never show it, ${NAME:+word} / ${NAME+word}
    (is it set?) and ${#NAME} (its length). Expansions
    nested in a `${...}` operand are found too."""
    out = []
    for m in EXPANSION_RE.finditer(text):
        if m.group(4):
            out.append(m.group(4))
            continue
        prefix, name, rest = m.group(1), m.group(2), m.group(3)
        if prefix != "#" and not rest.startswith((":+", "+")):
            out.append(name)
        out += shown_names(rest)
    return out


def sensitive(name, tainted):
    return name in tainted or is_secret_name(name)


def is_secret_path(word, cwd):
    path = resolve(word, cwd)
    home = home_dir()
    if any(fold(path) == fold(os.path.join(home, *rel.split("/"))) for rel in HOME_FILES):
        return True
    return secret_basename(os.path.basename(path))


# ---------------------------------------------------------------- parsing

def grep_prints_no_content(args):
    """True when a grep-family command's options mean it prints no file
    content (-q, -c, -l, -L and their long forms). A short-option cluster is
    read left to right and stops at an option that takes a value, since the
    rest of the cluster (or the next word) is that value: `-rqi` is quiet,
    `-eclass` is the pattern `class`."""
    i = 0
    while i < len(args):
        a = args[i]
        i += 1
        if a == "--":
            break
        if a.startswith("--"):
            name, has_value, _ = a.partition("=")
            if name in GREP_QUIET_LONG:
                return True
            if name in GREP_VALUE_LONG and not has_value:
                i += 1
        elif a.startswith("-") and len(a) > 1:
            for k, c in enumerate(a[1:], 1):
                if c in GREP_QUIET_SHORT:
                    return True
                if c in GREP_VALUE_SHORT:
                    if k == len(a) - 1:
                        i += 1
                    break
    return False


def pattern_given_by_option(args):
    """True when -e/-f (or --regexp, --file), alone or inside a short-option
    cluster such as `-ne` or glued to its value (`-eclass`), supplies the
    pattern, so the first operand is a file and not the pattern."""
    for a in args:
        if a == "--":
            break
        if a.startswith("--"):
            if a.split("=", 1)[0] in PATTERN_OPTIONS:
                return True
        elif a.startswith("-") and len(a) > 1:
            for c in a[1:]:
                if c in "ef":
                    return True
                if c in "ABCmdD":
                    break
    return False


def operands(args):
    """The non-option words of an argument list; `--` ends options."""
    out, seen_dashdash = [], False
    for a in args:
        if seen_dashdash:
            out.append(a)
        elif a == "--":
            seen_dashdash = True
        elif not a.startswith("-"):
            out.append(a)
    return out


def option_value(args, short, long):
    """The value of `-s VALUE`, `-sVALUE`, `-s=VALUE`, `--long VALUE` or
    `--long=VALUE`, or None when the option is absent."""
    for i, a in enumerate(args):
        if a == short or a == long:
            return args[i + 1] if i + 1 < len(args) else ""
        if a.startswith(long + "="):
            return a[len(long) + 1:]
        if a.startswith(short) and not a.startswith("--") and len(a) > len(short):
            return a[len(short):].lstrip("=")
    return None


def docker_format_is_safe(fmt):
    """A --format that picks specific fields and none of them is the
    environment or a whole object that contains it."""
    if not fmt or "Env" in fmt:
        return False
    return not re.search(r"\{\{\s*(?:json\s+)?\.(?:Config)?\s*\}\}", fmt)


# ---------------------------------------------------------------- rules

def skip_global_options(args, with_value):
    """`args` without the leading global options of docker/podman/gh, so the
    subcommand comes first. An option in `with_value` takes the next word
    unless it is written `--opt=value`; any other leading option is a flag."""
    i = 0
    while i < len(args) and args[i].startswith("-") and args[i] != "--":
        opt = args[i]
        i += 2 if opt in with_value else 1
    return args[i:]


def secret_file_finding(word, text, program, redirected):
    """One secret-file finding. `program` is the command that reads the file
    (empty for a bare redirection); `redirected` says it is read through `<`.
    Allow patterns match `program file`, the same for both ways of reading."""
    program = program or "(redirection)"
    shown = f"(redirection) < {word}" if redirected else f"{program} {word}"
    return Finding("secret-file", text, shown, word, match=f"{program} {word}")


def printed_secret_variable(args, tainted):
    """The first sensitive NAME an echo/printf argument expands in a way that
    shows its value."""
    for a in args:
        for name in shown_names(a):
            if sensitive(name, tainted):
                return name
    return None


def check_env_dump(cmd, args, ops, text, found, tainted):
    """printenv, export, declare, set, echo and printf: the shell-builtin and
    utility ways to print variables. True when a finding was added."""
    shown = None
    options = [a for a in args if a.startswith("-") and a != "--"]
    if cmd == "printenv":
        if not ops:
            shown = "printenv"
        else:
            name = next((o for o in ops if sensitive(o, tainted)), None)
            shown = f"printenv {name}" if name else None
    elif cmd == "export":
        if not ops:
            shown = "export -p"
    elif cmd in ("declare", "typeset"):
        letters = "".join(o.lstrip("-") for o in options)
        if "f" not in letters and "F" not in letters:
            if not ops or ("p" in letters and any(sensitive(o, tainted) for o in ops)):
                shown = "declare -p"
    elif cmd == "set":
        if not args:
            shown = "set"
    elif cmd in ("echo", "printf"):
        name = printed_secret_variable(args, tainted)
        if name:
            shown = f"{cmd} ${name}"
    if shown:
        found.append(Finding("env-dump", text, shown))
        return True
    return False


def check_command(rest, seg, cwd, depth, found, tainted, unplace=str):
    """Apply every rule to one simple command. `rest` is the segment's tokens
    with its prefix peeled."""
    # Every word is read as the user typed it: a placeholder for a substitution
    # must not reach a message or the text an allow pattern is matched against.
    # The expansion checks (echo, printenv, declare -p) judge the words with the
    # substitutions still cut out: a substitution is its own command, judged on
    # its own, and its text is not part of this command's arguments.
    cut_rest = rest
    rest = [unplace(t) for t in rest]
    reads = [unplace(t) for t in seg.reads]
    writes = [unplace(t) for t in seg.writes]
    text = " ".join(shlex.quote(t) for t in rest)
    cmd = os.path.basename(rest[0]) if rest else ""
    if cmd.endswith(".exe"):
        cmd = cmd[:-4]
    for word in rest[1:] + reads + writes:
        if touches_config_dir(word, cwd):
            found.append(Finding("allow-file", text or "(redirection)",
                                 f"{cmd} (the allow file)" if cmd else "(redirection) (the allow file)"))
            return
    if seg.quiet:
        # Standard output goes to /dev/null: nothing is shown, and this is the
        # exit-code check the messages recommend.
        return
    redirected = [w for w in reads if is_secret_path(w, cwd)]
    for word in redirected:
        found.append(secret_file_finding(word, text + " < " + shlex.quote(word), cmd, True))
    if redirected:
        return
    if not rest:
        return
    args = rest[1:]
    ops = operands(args)

    if cmd in hive_shell.SHELLS and depth < MAX_DEPTH:
        line = hive_shell.shell_c_argument(args)
        if line is not None:
            check(unplace(line), cwd, depth + 1, found, Taint(tainted))
        return
    if cmd == "eval" and depth < MAX_DEPTH:
        check(unplace(" ".join(args)), cwd, depth + 1, found, Taint(tainted))
        return

    if hive_shell.is_git(rest[0]):
        parsed = hive_shell.parse_git(rest, cwd)
        if parsed:
            sub, gargs, _ = parsed
            gops = operands(gargs)
            if sub == "credential" and gops[:1] in (["fill"], ["get"]):
                found.append(Finding("git-credential", text, f"git credential {gops[0]}"))
            elif sub.startswith("credential-") and "get" in gops:
                found.append(Finding("git-credential", text, f"git {sub} get"))
        return
    if cmd.startswith("git-credential-") and "get" in ops:
        found.append(Finding("git-credential", text, f"{cmd} get"))
    elif cmd == "gh":
        gargs = skip_global_options(args, GH_GLOBAL_VALUES)
        gops = operands(gargs)
        if gops[:2] == ["auth", "token"]:
            found.append(Finding("gh-token", text, "gh auth token"))
        elif gops[:2] == ["auth", "status"] and any(a in ("--show-token", "-t") for a in gargs):
            found.append(Finding("gh-token", text, "gh auth status --show-token"))
    elif cmd == "security":
        if ops[:1] in (["find-generic-password"], ["find-internet-password"]):
            found.append(Finding("keychain", text, f"security {ops[0]}"))
    elif cmd in ("kubectl", "oc"):
        names = [part for a in ops for part in a.split(",")]
        if "get" in ops and any(re.match(r"secrets?(?:$|/|\.)", n) for n in names):
            out = option_value(args, "-o", "--output")
            templated = any(a == "--template" or a.startswith("--template=") for a in args)
            if templated or (out is not None and out not in ("name", "wide")):
                found.append(Finding("kubectl-secret", text, f"{cmd} get secret"))
    elif cmd in ("docker", "podman"):
        dops = operands(skip_global_options(args, DOCKER_GLOBAL_VALUES))
        if dops[:1] == ["inspect"] or dops[:2] == ["container", "inspect"]:
            fmt = option_value(args, "-f", "--format")
            if not docker_format_is_safe(fmt):
                found.append(Finding("docker-inspect", text, f"{cmd} inspect"))
    elif check_env_dump(cmd, cut_rest[1:], operands(cut_rest[1:]), text, found, tainted):
        return
    elif cmd in READERS:
        files = ops
        if cmd in PATTERN_FIRST and not pattern_given_by_option(args):
            files = ops[1:]
        if cmd in GREP_FAMILY and grep_prints_no_content(args):
            return
        for word in files:
            if is_secret_path(word, cwd):
                found.append(secret_file_finding(word, text, cmd, False))


# A substitution is cut out of the text the segments are made from and replaced
# by a placeholder word (see hive_shell.cut_substitutions).
PH_OPEN, PH_CLOSE = hive_shell.PH_OPEN, hive_shell.PH_CLOSE
PLACEHOLDER_RE = hive_shell.PLACEHOLDER_RE
CAPTURE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=" + PH_OPEN + r"(\d+)" + PH_CLOSE + "$")
# Builtins whose NAME=value arguments set a variable and print nothing.
CAPTURE_BUILTINS = {"export", "local", "readonly", "declare", "typeset"}
ASSIGN_VALUE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)


def check(command, cwd, depth, found, tainted=None):
    """Walk a command line segment by segment. Raises hive_shell.ParseError.

    Each top-level substitution is cut out of the text, replaced by a
    placeholder word, and checked on its own. One that is only assigned to a
    variable (`NAME="$(cmd)" prog`, `export NAME=$(cmd)`) is captured: its
    value goes into the variable and is not displayed, so only its allow-file
    findings count. The name is tainted, and so is any name assigned from it:
    an expansion of a tainted name that would display its value (echo, printf,
    printenv, declare -p) is a finding, here and in every nested check. Any
    substitution that is not a capture is shown, and all of its findings count.
    A name set by an assignment-only segment or an export stays tainted for the
    rest of the command; one set as a prefix (`X=$(cmd) prog`) only for `prog`."""
    tainted = Taint(tainted)
    spans = hive_shell.substitution_spans(command) if depth < MAX_DEPTH else []
    text = hive_shell.cut_substitutions(command, spans)

    def unplace(word):
        """`word` with each placeholder replaced by the substitution it stands
        for, so a `sh -c` or `eval` argument is parsed as the user wrote it."""
        return hive_shell.restore_substitutions(word, command, spans)

    done = set()

    def check_span(k, name):
        """Check substitution k; `name` is the variable it is captured into."""
        done.add(k)
        inner = []
        check(spans[k][2], cwd, depth + 1, inner, tainted)
        if name is None:
            found.extend(inner)
            return
        kept = [f for f in inner if f.rule == "allow-file"]
        if len(kept) < len(inner):
            tainted.add(name)
        found.extend(kept)

    here = cwd
    for seg in hive_shell.segments_with_redirects(text):
        saw_env, rest, assignments, _ = hive_shell.peel(seg.tokens)
        builtin = bool(rest) and hive_shell.prefix_base(rest[0]) in CAPTURE_BUILTINS
        words = [w for w, ok in assignments if ok] + (rest[1:] if builtin else [])
        captured = {}
        for word in words:
            m = CAPTURE_RE.match(word)
            if m:
                captured[int(m.group(2))] = m.group(1)
        # Left to right: each substitution is checked with the names tainted so far.
        placeholders = sorted({int(m.group(1)) for w in seg.tokens + seg.reads + seg.writes
                               for m in PLACEHOLDER_RE.finditer(w)})
        mark = len(tainted.order)
        for k in placeholders:
            if k not in done:
                check_span(k, captured.get(k))
        for word in [w for w, _ in assignments] + (rest[1:] if builtin else []):
            m = ASSIGN_VALUE_RE.match(word)
            if m and any(sensitive(n, tainted) for n in shown_names(m.group(2))):
                tainted.add(m.group(1))
        if not rest:
            if saw_env and not seg.quiet:
                found.append(Finding("env-dump", " ".join(shlex.quote(unplace(t)) for t in seg.tokens),
                                     "env"))
            check_command([], seg, here, depth, found, tainted, unplace)
        elif rest[0] in ("cd", "pushd") and len(rest) > 1 and rest[1] != "-":
            here = resolve(rest[1], here)
        else:
            check_command(rest, seg, here, depth, found, tainted, unplace)
        if rest and not builtin:
            # `X=$(cmd) prog`: X exists only for prog, so forget it afterwards.
            tainted.forget_since(mark)
    for k in range(len(spans)):          # one in a comment, say: never in a segment
        if k not in done:
            check_span(k, None)


# ---------------------------------------------------------------- allow file

def strip_comment(line):
    """`line` without a trailing comment: a `#` at the start of the line or
    after whitespace starts one. A `#` inside a word stays."""
    for i, c in enumerate(line):
        if c == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line


def load_allows():
    """([(rule, glob_or_None)], ignored) from the user's allow file. A missing
    or unreadable file allows nothing; an `allow-file` entry, an unknown rule
    id, or a `rule:` with an empty pattern is ignored, and `ignored` counts
    such lines (blank and comment-only lines are not counted)."""
    try:
        # utf-8-sig: a file saved by a Windows editor starts with a BOM, which
        # would otherwise turn its first rule id into an unknown one.
        with open(allow_file_path(), encoding="utf-8-sig") as f:
            lines = f.read().splitlines()
    except (OSError, UnicodeDecodeError):
        return [], 0
    out, ignored = [], 0
    for raw in lines:
        line = strip_comment(raw.strip())
        if not line:
            continue
        rule, sep, pattern = line.partition(":")
        rule, pattern = rule.strip(), pattern.strip()
        if rule == "allow-file" or rule not in RULES or (sep and not pattern):
            # unknown rule, the rule that cannot be allowed, or `rule:` with
            # nothing after the colon: not understood, so it allows nothing
            # (a bare `rule` is the rule-wide form)
            ignored += 1
            continue
        out.append((rule, pattern or None))
    return out, ignored


def is_allowed(finding, allows):
    if finding.rule == "allow-file":
        return False
    return any(rule == finding.rule and (glob is None or fnmatch.fnmatchcase(finding.match, glob))
               for rule, glob in allows)


def glob_escape(text):
    """`text` as a glob that matches exactly itself."""
    return re.sub(r"([*?\[])", r"[\1]", text)


# ---------------------------------------------------------------- main

def report(finding):
    """Print a block. Only `shown` (the rule's canonical form) is printed, never
    the command: an inline value in it must not be copied into the transcript."""
    why, instead = RULES[finding.rule]
    rule = finding.rule
    print(f"secret-guard: `{finding.shown}` is blocked (rule {rule}): {why}.", file=sys.stderr)
    print(f"  Instead: {instead}", file=sys.stderr)
    if rule == "allow-file":
        return
    print("  If this is a real exception, a person can allow it by adding a line to this file,",
          file=sys.stderr)
    print("  by hand, outside the session (the guard refuses commands that touch it):", file=sys.stderr)
    print(f"    {allow_file_path()}", file=sys.stderr)
    print(f"    {rule}                    allows this rule everywhere", file=sys.stderr)
    if finding.file:
        print("  or, to allow only this program and file, add exactly:", file=sys.stderr)
        print(f"    {rule}: {glob_escape(finding.match)}", file=sys.stderr)
        print(f"  (`*` can stand for any reader: {rule}: * {glob_escape(finding.file)})",
              file=sys.stderr)
    else:
        print(f"    {rule}: <glob>            allows only commands whose text matches the glob",
              file=sys.stderr)


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
    allows, ignored = load_allows()
    found = []
    try:
        if "\x00" in command:
            raise hive_shell.ParseError("could not parse the command (NUL character); not checked")
        check(command, cwd, 0, found)
    except hive_shell.ParseError as e:
        # The tokenizer could not read it. Still refuse the commands that
        # matter most, matched as plain text; otherwise stand aside, loudly.
        found = [Finding(rule, command.strip(), shown) for rule, shown, pattern in FALLBACK
                 if pattern.search(command)]
        if not found:
            print(f"secret-guard: {e}", file=sys.stderr)
            return 1
    blocked, seen = [], set()
    for f in found:
        if (f.rule, f.cmd, f.shown) in seen or is_allowed(f, allows):
            continue
        seen.add((f.rule, f.cmd, f.shown))
        blocked.append(f)
    if not blocked:
        return 0
    for f in blocked:
        report(f)
    if ignored:
        print(f"  Note: {ignored} line(s) in {allow_file_path()} were not understood and allow nothing.",
              file=sys.stderr)
    return 2


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as e:  # noqa: BLE001 -- a broken guard must not block every Bash call
        print(f"secret-guard: internal error, not checked: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
