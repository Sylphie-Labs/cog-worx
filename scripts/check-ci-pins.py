#!/usr/bin/env python3
"""check-ci-pins: fail the build when CI depends on something whose version floats.

Canonical copy: Sylphie-Labs/hive-client, scripts/check-ci-pins.py, with its
tests in tests/test_check_ci_pins.py. The other repos carry byte-identical
copies. Fix it here first, prove it with a test, then copy it over.

Why this exists: pinning everything once is a snapshot. The next `uses:` line or
the next tool called from a `run:` step floats again, and nothing notices until
GitHub moves the image or the tag under it. This runs on every PR so an unpinned
dependency cannot enter without a deliberate line saying what governs it.

Rules. Each finding prints file:line, the offending text, and how to fix it.

  1. `uses:` must name a full 40-hex commit SHA and carry a `# <version>`
     comment (Dependabot reads the comment to know what to bump). Local
     `./...` references are exempt: the containing commit pins them, and
     `.github/actions/*/action.yml` files are scanned like workflows.
     `docker://` references need an `@sha256:` digest.
  2. No `ubuntu-latest`, `macos-latest`, `windows-latest` or `ubuntu-slim`.
     Name the image. A repo that deliberately tracks a floating label
     declares it in the manifest as `runs-on=<label>`.
  3. An install that bypasses a lockfile must name an exact version:
     pip / uv pip / uv tool / uvx / pipx, npm -g / npx / yarn add / pnpm add,
     cargo install, gem, go install, brew, apt-get, choco. Also
     `session.install(...)` and `session.run_install(...)` in noxfile.py.
     Lockfile-driven installs (`npm ci`, `uv sync --frozen`, `yarn install
     --frozen-lockfile`, `cargo build`) are the pin and are not checked here.
  4. Every program a `run:` step invokes must be declared in
     `.github/pinned-tools.txt` as `name=<what governs its version>`, except
     shell builtins and a fixed list of POSIX utilities. Only sh/bash steps
     are tokenized; a step whose effective shell is pwsh, cmd, python or
     anything else is skipped (the shell comes from the step, the job's or
     the workflow's `defaults.run.shell`, or pwsh on a windows-* runner).
  5. A service container `image:` must carry a tag with an x.y version, or a
     digest.

Usage: python3 scripts/check-ci-pins.py [--root DIR]
Exit 0 when everything is pinned, 1 with findings, 2 on a usage error.
Standard library only, Python 3.8+.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Set, Tuple

MANIFEST = Path(".github") / "pinned-tools.txt"
WORKFLOWS = Path(".github") / "workflows"
ACTIONS = Path(".github") / "actions"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LATEST_RE = re.compile(r"\b(?:(?:ubuntu|macos|windows)-latest(?:-[a-z]+)?|ubuntu-slim)\b")
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
VERSION_TAG_RE = re.compile(r"\d+\.\d+")
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
FUNCTION_DEF_RE = re.compile(r"^\s*(?:function\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*\(\s*\)\s*\{?")
COMMAND_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.+-]*$")
HEREDOC_RE = re.compile(r"(?<!<)<<-?(?!<)\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?")
# `name==1.2.3`, `name[extra]==1.2.3`, or a version held in an f-string constant.
EXACT_PY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]*\])?==(?:\d[\w.!+-]*|\{[A-Za-z_]\w*\})$")
EXACT_UVX_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*@\d+\.\d+\.\d+[\w.+-]*$")
EXACT_NPM_RE = re.compile(r"^(?:@[^/@\s]+/)?[^@\s]+@\d+\.\d+\.\d+[\w.-]*$")
EXACT_APT_RE = re.compile(r"^[^=\s]+=\S+$")
EXACT_GO_RE = re.compile(r"@v?\d+\.\d+\.\d+[\w.-]*$")
EXACT_CARGO_VERSION_RE = re.compile(r"^=?\d+\.\d+\.\d+[\w.+-]*$")
PACKAGE_LIKE_RE = re.compile(r"^[A-Za-z@][A-Za-z0-9._@/\[\]-]*(?:[<>=!~^,][^\s]*)?$")

# Words that never need a manifest entry: shell syntax, builtins, and the POSIX
# utilities every image ships. Deliberately not a list of "things Ubuntu has".
SHELL_WORDS: Set[str] = set(
    """
    : . [ [[ ]] } ) ;; fi done esac
    alias bg break builtin case cd command continue declare echo eval exec exit export fg
    getopts hash jobs kill let local popd printf pushd pwd read readonly return select set shift
    shopt source test times trap true false type typeset ulimit umask unalias unset wait
    awk basename bash cat chgrp chmod chown cmp comm cp cut dash date dd df diff dirname du env
    expr file find fold grep gzip head id join ln ls mkdir mktemp mv nl od paste readlink rm
    rmdir sed seq sh sleep sort split stat tail tar tee touch tr uname uniq wc which xargs xxd zsh
    """.split()
)

# Words that precede another command on the same segment: skip them (and their
# flags) and check what follows.
TRANSPARENT: Set[str] = {
    "sudo", "env", "time", "exec", "nohup", "command", "nice", "xargs",
    "if", "elif", "while", "until", "then", "do", "else", "{", "!",
}
# Flags of the transparent words above that take a value.
TRANSPARENT_VALUE_FLAGS: Set[str] = {"-u", "-g", "-n", "-I", "-L", "-P", "-d", "-a", "-s", "-C", "-S", "-p"}

# Per installer: flags that take a value, so the value is not mistaken for a package.
VALUE_FLAGS: Dict[str, Set[str]] = {
    "py": {
        "-r", "--requirement", "-c", "--constraint", "-i", "--index-url", "--extra-index-url", "-f",
        "--find-links", "-t", "--target", "--prefix", "--root", "--python", "-p", "--group",
        "--only-group", "--extra", "--with", "--from", "--index", "-e", "--editable", "--config-file",
        "--cache-dir", "--build", "-b", "--log", "--proxy", "--platform", "--implementation",
        "--abi", "--python-version", "--pip-args",
    },
    "uvx": {"--from", "--with", "--python", "-p", "--index", "--index-url", "--extra-index-url"},
    "npm": {"--registry", "--prefix", "-C", "--workspace", "-w", "--tag", "--package", "-p", "-c", "--call"},
    "cargo": {
        "--version", "--vers", "--git", "--branch", "--tag", "--rev", "--path", "--features", "-F",
        "--profile", "--config", "-Z", "--index", "--registry", "--root", "--target", "--target-dir",
        "-j", "--jobs",
    },
    "go": {"-tags", "-ldflags", "-p", "-o", "-gcflags", "-asmflags", "-mod", "-modfile", "-overlay"},
    "apt": {"-o", "-t", "--target-release", "-c", "--config-file"},
    "brew": set(),
    "gem": {"-v", "--version", "-s", "--source", "-n", "--bindir"},
    "choco": {"--version", "--source", "-s", "--params", "--install-arguments", "-ia"},
}


class Finding:
    def __init__(self, path: Path, line: int, rule: int, text: str, fix: str) -> None:
        self.path, self.line, self.rule, self.text, self.fix = path, line, rule, text, fix

    def render(self, root: Path) -> str:
        try:
            rel = self.path.relative_to(root)
        except ValueError:
            rel = self.path
        return f"{rel}:{self.line}: rule {self.rule}: {self.text}\n    fix: {self.fix}"


# --------------------------------------------------------------------------- text helpers


def strip_comment(line: str) -> str:
    """Drop a trailing `# ...` that is outside quotes. YAML and sh agree on this shape."""
    quote: Optional[str] = None
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
        elif ch == "#" and (i == 0 or line[i - 1] in " \t"):
            return line[:i]
    return line


def unquote(value: str) -> str:
    value = value.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        return value[1:-1]
    return value


def indent_of(line: str) -> int:
    return len(line) - len(line.lstrip(" "))


def split_segments(line: str) -> List[str]:
    """Split one sh line into command segments at && || ; | ( ) $( ) and backticks.

    A `$( )` inside double quotes is a command too (`echo "sha=$(git rev-parse HEAD)"`),
    so the quote context is suspended for its extent and restored at the `)`.
    """
    segments: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    saved: List[Optional[str]] = []  # quote context to restore at each `)`

    def cut() -> None:
        segments.append("".join(buf))
        buf.clear()

    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        two = line[i : i + 2]
        if quote == "'":
            buf.append(ch)
            if ch == "'":
                quote = None
            i += 1
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < n:
                buf.append(ch)
                buf.append(line[i + 1])
                i += 2
                continue
            if two == "$(":
                cut()
                saved.append(quote)
                quote = None
                i += 2
                continue
            if ch == "`":
                cut()
                i += 1
                continue
            buf.append(ch)
            if ch == '"':
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(line[i + 1])
            i += 2
            continue
        if two in ("&&", "||", ";;"):
            cut()
            i += 2
            continue
        if two == "$(":
            cut()
            saved.append(None)
            i += 2
            continue
        if ch == "(":
            cut()
            saved.append(None)
            i += 1
            continue
        if ch == ")":
            cut()
            if saved:
                quote = saved.pop()
                if quote:
                    buf.append(quote)  # the tail of the string is content, not a command
            i += 1
            continue
        if ch in (";", "|", "&", "`"):
            cut()
            i += 1
            continue
        buf.append(ch)
        i += 1
    cut()
    return [s.strip() for s in segments if s.strip()]


def shell_words(segment: str) -> List[str]:
    """Whitespace split that keeps quoted spans together (quotes retained)."""
    words: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    for ch in segment:
        if quote:
            buf.append(ch)
            if ch == quote:
                quote = None
        elif ch in ("'", '"'):
            quote = ch
            buf.append(ch)
        elif ch in " \t":
            if buf:
                words.append("".join(buf))
                buf = []
        else:
            buf.append(ch)
    if buf:
        words.append("".join(buf))
    return words


def is_path_like(word: str) -> bool:
    return "/" in word or word.startswith(".") or word.startswith("~")


def command_words(segment: str) -> Tuple[Optional[str], List[str]]:
    """Return (command, args) for one sh segment, or (None, []) when it runs no
    checkable program: an assignment, a `for`/`case` header, a variable-dispatched
    command, a redirect, or a path (a repo script is pinned by the commit)."""
    words = shell_words(segment)
    i = 0
    while i < len(words):
        w = words[i]
        if ASSIGNMENT_RE.match(w):
            i += 1
            continue
        if w in ("for", "case", "select", "function"):
            return None, []
        if w in TRANSPARENT:
            i += 1
            while i < len(words) and words[i].startswith("-"):
                if words[i] in TRANSPARENT_VALUE_FLAGS and w in ("sudo", "xargs", "nice", "env"):
                    i += 1
                i += 1
            continue
        break
    if i >= len(words):
        return None, []
    cmd = unquote(words[i])
    if not cmd or is_path_like(cmd) or not COMMAND_NAME_RE.match(cmd):
        return None, []
    return cmd, words[i + 1 :]


# --------------------------------------------------------------------------- workflow model


class RunBlock:
    def __init__(self, path: Path, lines: List[Tuple[int, str]], shell: str) -> None:
        self.path, self.lines, self.shell = path, lines, shell


class Step:
    def __init__(self, lines: List[Tuple[int, str]], job_shell: str, runs_on: str) -> None:
        self.lines, self.job_shell, self.runs_on = lines, job_shell, runs_on


class Workflow:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.raw = path.read_text(encoding="utf-8", errors="replace").splitlines()
        self.code = [strip_comment(line) for line in self.raw]

    def uses(self) -> Iterator[Tuple[int, str, str]]:
        """(line number, value, trailing comment)"""
        for no, (code, raw) in enumerate(zip(self.code, self.raw), 1):
            m = re.match(r"^\s*-?\s*uses:\s*(.+?)\s*$", code)
            if m:
                comment = raw[len(code) :].strip()
                yield no, unquote(m.group(1)), comment

    def images(self) -> Iterator[Tuple[int, str]]:
        for no, code in enumerate(self.code, 1):
            m = re.match(r"^\s*-?\s*image:\s*(.+?)\s*$", code)
            if m:
                yield no, unquote(m.group(1))

    def latest_labels(self) -> Iterator[Tuple[int, str]]:
        for no, code in enumerate(self.code, 1):
            for m in LATEST_RE.finditer(code):
                yield no, m.group(0)

    def run_blocks(self) -> Iterator[RunBlock]:
        """Every `run:` scalar, with the shell that will execute it."""
        for step in self._steps():
            shell = step.job_shell
            for _, code in step.lines:
                m = re.match(r"^\s*-?\s*shell:\s*(.+?)\s*$", code)
                if m:
                    shell = unquote(m.group(1)).lower()
            if not shell:
                shell = "pwsh" if step.runs_on.startswith("windows-") else "bash"
            idx = 0
            while idx < len(step.lines):
                no, code = step.lines[idx]
                m = re.match(r"^(\s*)-?\s*run:\s*(.*)$", code)
                if not m:
                    idx += 1
                    continue
                key_indent = indent_of(code)
                value = m.group(2).strip()
                if value in ("|", "|-", "|+", ">", ">-", ">+"):
                    # YAML block scalar: the content indent is that of its first
                    # non-blank line, and the block ends at the first non-blank
                    # line indented less than that (a sibling key like
                    # `working-directory:` is indented less, so it is not content).
                    block: List[Tuple[int, str]] = []
                    content_indent: Optional[int] = None
                    idx += 1
                    while idx < len(step.lines):
                        bno, _ = step.lines[idx]
                        braw = self.raw[bno - 1]
                        if braw.strip():
                            ind = indent_of(braw)
                            if content_indent is None:
                                if ind <= key_indent:
                                    break
                                content_indent = ind
                            elif ind < content_indent:
                                break
                        block.append((bno, braw))
                        idx += 1
                    if value.startswith(">"):
                        block = _fold(block)
                    yield RunBlock(self.path, block, shell)
                    continue
                yield RunBlock(self.path, [(no, unquote(value))], shell)
                idx += 1

    def _steps(self) -> List[Step]:
        """Group lines into steps under each `steps:` key, remembering the job's
        `runs-on` and the effective `defaults.run.shell` (workflow or job level)."""
        steps: List[Step] = []
        workflow_shell = ""
        job_shell = ""
        runs_on = ""
        jobs_indent: Optional[int] = None
        job_indent: Optional[int] = None
        defaults_indent: Optional[int] = None
        steps_indent: Optional[int] = None
        step_indent: Optional[int] = None
        current: Optional[List[Tuple[int, str]]] = None
        for no, code in enumerate(self.code, 1):
            if not code.strip():
                if current is not None:
                    current.append((no, code))
                continue
            ind = indent_of(code)
            stripped = code.strip()
            if steps_indent is not None and (ind < steps_indent or (ind == steps_indent and not stripped.startswith("- "))):
                steps_indent, step_indent, current = None, None, None
            if defaults_indent is not None and ind <= defaults_indent:
                defaults_indent = None
            if re.match(r"^jobs:\s*$", code):
                jobs_indent = ind
                continue
            if jobs_indent is not None and ind == jobs_indent + 2 and re.match(r"^[A-Za-z0-9_-]+:\s*$", stripped):
                job_indent, job_shell, runs_on = ind, "", ""
                continue
            if job_indent is not None and ind == job_indent + 2:
                m = re.match(r"^runs-on:\s*(.+)$", stripped)
                if m:
                    runs_on = unquote(m.group(1)).lower()
            if re.match(r"^defaults:\s*$", stripped):
                defaults_indent = ind
                continue
            if defaults_indent is not None:
                m = re.match(r"^shell:\s*(.+)$", stripped)
                if m:
                    if job_indent is not None and defaults_indent > job_indent:
                        job_shell = unquote(m.group(1)).lower()
                    else:
                        workflow_shell = unquote(m.group(1)).lower()
                continue
            if re.match(r"^steps:\s*$", stripped):
                steps_indent, step_indent, current = ind, None, None
                continue
            if steps_indent is None:
                continue
            if stripped.startswith("- ") and (step_indent is None or ind == step_indent):
                step_indent = ind
                current = [(no, code)]
                steps.append(Step(current, job_shell or workflow_shell, runs_on))
            elif current is not None:
                current.append((no, code))
        return steps


def _fold(block: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
    """A folded scalar (`>-`) joins lines with spaces, but a blank line is a newline."""
    out: List[Tuple[int, str]] = []
    para: List[Tuple[int, str]] = []
    for no, text in block + [(0, "")]:
        if text.strip():
            para.append((no, strip_comment(text).strip()))
        elif para:
            out.append((para[0][0], " ".join(t for _, t in para)))
            para = []
    return out


# --------------------------------------------------------------------------- manifest


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.tools: Set[str] = set()
        self.tracked_labels: Set[str] = set()
        self.errors: List[Finding] = []
        self.present = path.is_file()
        if not self.present:
            return
        for no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            name, sep, source = line.partition("=")
            name, source = name.strip(), source.strip()
            if not sep or not name or not source:
                self.errors.append(
                    Finding(path, no, 4, f"manifest line {line!r} does not say what governs the version",
                            "write it as name=<what pins it>, e.g. `uv=astral-sh/setup-uv version input`")
                )
                continue
            if name == "runs-on":
                self.tracked_labels.add(source)
            else:
                self.tools.add(name)


# --------------------------------------------------------------------------- rules


def _has_version_comment(comment: str) -> bool:
    return comment.startswith("#") and bool(comment.lstrip("# ").strip())


def check_uses(wf: Workflow) -> Iterator[Finding]:
    for no, value, comment in wf.uses():
        if value.startswith("./"):
            continue
        if value.startswith("docker://"):
            if not DIGEST_RE.search(value):
                yield Finding(wf.path, no, 1, f"uses: {value} has no digest",
                              "pin the image as docker://<image>@sha256:<digest>  # <tag>")
            elif not _has_version_comment(comment):
                yield Finding(wf.path, no, 1, f"uses: {value} has no `# <version>` comment",
                              "append `# <tag>` so a reader knows what the digest is")
            continue
        ref = value.rsplit("@", 1)
        if len(ref) != 2 or not SHA_RE.match(ref[1]):
            repo = "/".join(value.split("@", 1)[0].split("/")[:2])
            tag = ref[1] if len(ref) == 2 else "<tag>"
            yield Finding(wf.path, no, 1, f"uses: {value} is not pinned to a commit SHA",
                          f"replace with {repo}@$(gh api repos/{repo}/commits/{tag} --jq .sha)  # {tag}")
            continue
        if not _has_version_comment(comment):
            yield Finding(wf.path, no, 1, f"uses: {value} has no `# <version>` comment",
                          "append `# <version>` after the SHA; Dependabot bumps the SHA by reading it")


def check_runner_labels(wf: Workflow, manifest: Manifest) -> Iterator[Finding]:
    for no, label in wf.latest_labels():
        if label in manifest.tracked_labels:
            continue
        yield Finding(wf.path, no, 2, f"runner label {label} floats with GitHub's schedule",
                      f"name the image it resolves to today (the run log's \"Runner Image\" line, or "
                      f"github.com/actions/runner-images); or declare `runs-on={label}` in {MANIFEST} "
                      f"if this repo deliberately tracks it, with every tool it uses from the image pinned")


def check_service_images(wf: Workflow) -> Iterator[Finding]:
    for no, image in wf.images():
        if DIGEST_RE.search(image) or "${{" in image:
            continue
        name, sep, tag = image.rpartition(":")
        if not sep or "/" in tag:
            tag = ""
        if not VERSION_TAG_RE.search(tag):
            yield Finding(wf.path, no, 5, f"service image {image} has no x.y version tag",
                          "use an exact tag (e.g. postgres:17.9, neo4j:5.26.30-community) or an @sha256 digest")


def _positionals(kind: str, args: Sequence[str]) -> List[str]:
    value_flags = VALUE_FLAGS.get(kind, set())
    out: List[str] = []
    skip_next = False
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            skip_next = a in value_flags
            continue
        out.append(unquote(a))
    return out


def _flag_value(args: Sequence[str], names: Sequence[str]) -> Optional[str]:
    for i, a in enumerate(args):
        for name in names:
            if a == name and i + 1 < len(args):
                return unquote(args[i + 1])
            if a.startswith(name + "="):
                return unquote(a[len(name) + 1 :])
    return None


def _exact_findings(kind: str, args: Sequence[str]) -> List[str]:
    """Names of package arguments that lack an exact version, for one install command."""
    if kind == "cargo":
        version = _flag_value(args, ("--version", "--vers"))
        if _flag_value(args, ("--path",)) is not None:
            return []
        if _flag_value(args, ("--git",)) is not None:
            pinned = _flag_value(args, ("--rev", "--tag")) is not None
            return [] if pinned else [_flag_value(args, ("--git",)) or "--git"]
        if version is not None:
            return [] if EXACT_CARGO_VERSION_RE.match(version) else [f"--version {version}"]
        return [p for p in _positionals(kind, args) if not is_path_like(p)]
    if kind == "uvx":
        spec = _flag_value(args, ("--from",))
        if spec is not None:
            return [] if (EXACT_PY_RE.match(spec) or EXACT_UVX_RE.match(spec)) else [spec]
        first = _positionals(kind, args)[:1]
        return [p for p in first if not (EXACT_PY_RE.match(p) or EXACT_UVX_RE.match(p))]
    bad: List[str] = []
    for p in _positionals(kind, args):
        if kind == "go":
            if not EXACT_GO_RE.search(p):
                bad.append(p)
            continue
        if is_path_like(p) or p.endswith((".txt", ".toml", ".json", ".whl", ".tar.gz", ".deb", ".gem")):
            continue
        if not PACKAGE_LIKE_RE.match(p) or p[0].isdigit():
            continue
        if kind == "py" and not EXACT_PY_RE.match(p):
            bad.append(p)
        elif kind == "npm" and not EXACT_NPM_RE.match(p):
            bad.append(p)
        elif kind == "apt" and not EXACT_APT_RE.match(p):
            bad.append(p)
        elif kind in ("brew", "gem", "choco"):
            bad.append(p)
    return bad


def _after(args: Sequence[str], word: str) -> List[str]:
    return list(args[list(args).index(word) + 1 :])


def install_kind(cmd: str, args: Sequence[str]) -> Tuple[Optional[str], List[str], str]:
    """Classify an install command. Returns (kind, args after the subcommand, human name)."""
    a = list(args)
    sub = [w for w in a if not w.startswith("-")]
    if cmd in ("pip", "pip3") and sub[:1] == ["install"]:
        return "py", _after(a, "install"), "pip install"
    if cmd in ("python", "python3") and a[:2] == ["-m", "pip"] and "install" in sub:
        return "py", _after(a, "install"), "pip install"
    if cmd == "uv" and sub[:2] in (["pip", "install"], ["tool", "install"]):
        return "py", _after(a, "install"), "uv " + sub[0] + " install"
    if cmd == "uvx" or (cmd == "uv" and sub[:2] == ["tool", "run"]):
        rest = a if cmd == "uvx" else _after(a, "run")
        return "uvx", rest, "uvx"
    if cmd == "pipx" and sub[:1] in (["install"], ["run"]):
        return "py", _positionals("py", _after(a, sub[0]))[:1], "pipx " + sub[0]
    if cmd == "npm" and sub[:1] in (["install"], ["i"], ["add"]):
        rest = _after(a, sub[0])
        if not [w for w in rest if not w.startswith("-")]:
            return None, [], ""  # bare `npm install` is governed by package-lock.json
        return "npm", rest, "npm install"
    if cmd == "npx":
        return "npm", _positionals("npm", a)[:1], "npx"
    if cmd == "yarn" and (sub[:1] == ["add"] or sub[:2] == ["global", "add"]):
        return "npm", _after(a, "add"), "yarn add"
    if cmd == "pnpm" and sub[:1] in (["add"], ["dlx"]):
        return "npm", _after(a, sub[0]), "pnpm " + sub[0]
    if cmd == "cargo" and sub[:1] == ["install"]:
        return "cargo", _after(a, "install"), "cargo install"
    if cmd == "go" and sub[:1] == ["install"]:
        return "go", _after(a, "install"), "go install"
    if cmd == "gem" and sub[:1] == ["install"]:
        has_v = _flag_value(a, ("-v", "--version")) is not None
        return (None if has_v else "gem"), _after(a, "install"), "gem install"
    if cmd == "brew" and sub[:1] == ["install"]:
        return "brew", _after(a, "install"), "brew install"
    if cmd in ("apt-get", "apt") and sub[:1] == ["install"]:
        return "apt", _after(a, "install"), f"{cmd} install"
    if cmd == "choco" and sub[:1] == ["install"]:
        has_v = _flag_value(a, ("--version",)) is not None
        return (None if has_v else "choco"), _after(a, "install"), "choco install"
    return None, [], ""


FIX_BY_KIND = {
    "py": "name an exact version, e.g. name==1.2.3 (or install from the lockfile: uv sync --frozen)",
    "uvx": "name an exact version, e.g. uvx name@1.2.3 or uvx --from 'name==1.2.3' name",
    "npm": "name an exact version, e.g. name@1.2.3 (or install from the lockfile: npm ci)",
    "cargo": "pass --version x.y.z --locked (or --git with --rev/--tag)",
    "go": "name an exact version, e.g. module@v1.2.3",
    "apt": "pin the package as name=version, or fetch a pinned build yourself (see hive-client tests/shellcheck.sh)",
    "brew": "brew cannot pin a version; fetch a pinned build yourself (see hive-client tests/shellcheck.sh)",
    "gem": "pass -v x.y.z",
    "choco": "pass --version x.y.z",
}


def sh_commands(block: RunBlock) -> Iterator[Tuple[int, str, List[str]]]:
    """(line, command, args) for every command a run block executes."""
    joined: List[Tuple[int, str]] = []
    pending: Optional[Tuple[int, str]] = None
    for no, text in block.lines:
        code = strip_comment(text.rstrip("\n")).rstrip()
        if pending is not None:
            pending = (pending[0], pending[1][:-1] + " " + code.strip())
        else:
            pending = (no, code)
        if pending[1].endswith("\\"):
            continue
        joined.append(pending)
        pending = None
    if pending is not None:
        joined.append(pending)

    defined = {m.group(1) for _, text in joined for m in [FUNCTION_DEF_RE.match(text)] if m}
    heredoc: Optional[str] = None
    for no, text in joined:
        stripped = text.strip()
        if heredoc is not None:
            if stripped == heredoc:
                heredoc = None
            continue
        if not stripped:
            continue
        code = FUNCTION_DEF_RE.sub("", stripped, count=1)
        m = HEREDOC_RE.search(code)
        if m:
            heredoc = m.group(1)
            code = code[: m.start()]
        for seg in split_segments(code):
            cmd, args = command_words(seg)
            if cmd is None or cmd in defined:
                continue
            yield no, cmd, args


def check_run_blocks(wf: Workflow, manifest: Manifest) -> Iterator[Finding]:
    for block in wf.run_blocks():
        if block.shell.split()[0] not in ("bash", "sh"):
            continue
        for no, cmd, args in sh_commands(block):
            kind, pkgs, human = install_kind(cmd, args)
            if kind is not None:
                for bad in _exact_findings(kind, pkgs):
                    yield Finding(wf.path, no, 3, f"{human} {bad}: no exact version", FIX_BY_KIND[kind])
            if cmd in SHELL_WORDS or cmd in manifest.tools:
                continue
            yield Finding(wf.path, no, 4, f"`{cmd}` is run here but {MANIFEST} does not declare it",
                          f"add `{cmd}=<what governs its version>` to {MANIFEST} (an action's input, a "
                          f"pinned fetch script, or the named runner image if it deliberately floats)")


NOX_CALL_RE = re.compile(r"session\.(install|run_install)\s*(\((?:[^()]|\([^()]*\))*\))")


def check_noxfile(root: Path) -> Iterator[Finding]:
    nox = root / "noxfile.py"
    if not nox.is_file():
        return
    text = nox.read_text(encoding="utf-8")
    for m in NOX_CALL_RE.finditer(text):
        line = text.count("\n", 0, m.start()) + 1
        strings = [unquote(s) for s in re.findall(r"""("[^"]*"|'[^']*')""", m.group(2))]
        if m.group(1) == "install":
            kind, pkgs, human = "py", strings, "session.install"
        else:
            if not strings:
                continue
            kind, pkgs, human = install_kind(strings[0], strings[1:])
            if kind is None:
                continue
            human = "session.run_install " + human
        for bad in _exact_findings(kind, pkgs):
            yield Finding(nox, line, 3, f"{human} {bad}: no exact version", FIX_BY_KIND[kind])


# --------------------------------------------------------------------------- driver


def workflow_files(root: Path) -> List[Path]:
    files: List[Path] = []
    if (root / WORKFLOWS).is_dir():
        files.extend((root / WORKFLOWS).glob("*.y*ml"))
    if (root / ACTIONS).is_dir():
        files.extend((root / ACTIONS).rglob("action.y*ml"))
    return sorted(files)


def run(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    manifest = Manifest(root / MANIFEST)
    findings.extend(manifest.errors)
    files = workflow_files(root)
    for path in files:
        wf = Workflow(path)
        findings.extend(check_uses(wf))
        findings.extend(check_runner_labels(wf, manifest))
        findings.extend(check_service_images(wf))
        findings.extend(check_run_blocks(wf, manifest))
    findings.extend(check_noxfile(root))
    if files and not manifest.present and any(f.rule == 4 for f in findings):
        findings.insert(0, Finding(root / MANIFEST, 0, 4, "manifest is missing",
                                   f"create {MANIFEST} with one `name=<what governs its version>` line per tool listed below"))
    return findings


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Fail when CI depends on a floating version.")
    parser.add_argument("--root", default=".", help="repository root (default: current directory)")
    args = parser.parse_args(argv)
    root = Path(args.root).resolve()
    if not root.is_dir():
        print(f"check-ci-pins: {root} is not a directory", file=sys.stderr)
        return 2
    findings = run(root)
    if not findings:
        print("check-ci-pins: every action, runner image, service image, install and tool is pinned or declared.")
        return 0
    findings.sort(key=lambda f: (str(f.path), f.line, f.rule))
    for f in findings:
        print(f.render(root))
    by_rule = sorted({f.rule for f in findings})
    print(f"check-ci-pins: {len(findings)} finding(s), rule(s) {', '.join(map(str, by_rule))}. See the header of this script for the rules.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
