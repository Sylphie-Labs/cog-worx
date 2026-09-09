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
     `./...` references are exempt: the containing commit pins them.
     `docker://` references need an `@sha256:` digest.
  2. No `ubuntu-latest`, `macos-latest`, or `windows-latest`. Name the image.
     A repo that deliberately tracks a floating label declares it in the
     manifest as `runs-on=<label>`.
  3. An install that bypasses a lockfile must name an exact version:
     pip / uv pip / uv tool / uvx / pipx, npm -g / npx / yarn add / pnpm add,
     cargo install, gem, go install, brew, apt-get, choco. Also
     `session.install(...)` and `session.run_install(...)` in noxfile.py.
     Lockfile-driven installs (`npm ci`, `uv sync --frozen`, `yarn install
     --frozen-lockfile`, `cargo build`) are the pin and are not checked here.
  4. Every program a `run:` step invokes must be declared in
     `.github/pinned-tools.txt` as `name=<what governs its version>`, except
     shell builtins and a fixed list of POSIX utilities. Steps with an explicit
     PowerShell or cmd shell are skipped: the tokenizer is a sh tokenizer.
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
from typing import Iterable, Iterator, List, Optional, Sequence, Set, Tuple

MANIFEST = Path(".github") / "pinned-tools.txt"
WORKFLOWS = Path(".github") / "workflows"

SHA_RE = re.compile(r"^[0-9a-f]{40}$")
LATEST_RE = re.compile(r"\b(?:ubuntu|macos|windows)-latest(?:-[a-z]+)?\b")
DIGEST_RE = re.compile(r"@sha256:[0-9a-f]{64}$")
VERSION_TAG_RE = re.compile(r"\d+\.\d+")
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
EXACT_PY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]*\])?==\d[\w.!+*-]*$")
EXACT_NPM_RE = re.compile(r"^(?:@[^/@\s]+/)?[^@\s]+@\d[\w.-]*$")
EXACT_APT_RE = re.compile(r"^[^=\s]+=\S+$")
EXACT_GO_RE = re.compile(r"@v?\d[\w.-]*$")
PACKAGE_LIKE_RE = re.compile(r"^[A-Za-z@][A-Za-z0-9._@/\[\]-]*(?:[<>=!~^,][^\s]*)?$")

# Words that never need a manifest entry: shell syntax, builtins, and the POSIX
# utilities every image ships. Deliberately not a list of "things Ubuntu has".
SHELL_WORDS: Set[str] = set(
    """
    : . [ [[ ]] ! { } ( )
    alias bg break builtin case cd command continue declare do done echo elif else esac eval exec
    exit export fg fi for function getopts hash if in jobs let local popd printf pushd pwd read
    readonly return select set shift shopt source test then time times trap true false type
    typeset ulimit umask unalias unset until wait while
    awk basename bash cat chgrp chmod chown cmp comm cp cut dash date dd df diff dirname du env
    expr file find fold grep gzip head id join kill ln ls mkdir mktemp mv nl od paste readlink rm
    rmdir sed seq sh sleep sort split stat tail tar tee touch tr uname uniq wc which xargs xxd zsh
    """.split()
)

# Words that wrap another command: skip them (and their flags) and check what follows.
TRANSPARENT: Set[str] = {"sudo", "env", "time", "exec", "nohup", "command", "nice", "xargs", "if", "elif", "while", "until", "!"}
# Flags of the transparent words above that take a value.
TRANSPARENT_VALUE_FLAGS: Set[str] = {"-u", "-g", "-n", "-I", "-L", "-P", "-d", "-a", "-s", "-C", "-S", "-p"}

INSTALL_VALUE_FLAGS: Set[str] = {
    "-r", "--requirement", "-c", "--constraint", "-i", "--index-url", "--extra-index-url", "-f",
    "--find-links", "-t", "--target", "--prefix", "--root", "--python", "-p", "--group",
    "--only-group", "--extra", "--with", "--from", "--version", "--vers", "-v", "--index",
    "--registry", "--git", "--branch", "--tag", "--rev", "--path", "--features", "--profile",
    "--config", "-Z", "--cask", "--package", "--source", "-s", "--tags", "--ldflags",
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
    """Split one sh line into command segments at && || ; | $( ( ) outside quotes."""
    segments: List[str] = []
    buf: List[str] = []
    quote: Optional[str] = None
    i = 0
    n = len(line)
    while i < n:
        ch = line[i]
        if quote:
            buf.append(ch)
            if ch == "\\" and quote == '"' and i + 1 < n:
                buf.append(line[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            # $'...' is a quote too; the dollar is already in buf and harmless.
            quote = ch
            buf.append(ch)
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            buf.append(ch)
            buf.append(line[i + 1])
            i += 2
            continue
        two = line[i : i + 2]
        if two in ("&&", "||", "$(", ";;"):
            segments.append("".join(buf))
            buf = []
            i += 2
            continue
        if ch in (";", "|", "(", ")", "&"):
            segments.append("".join(buf))
            buf = []
            i += 1
            continue
        buf.append(ch)
        i += 1
    segments.append("".join(buf))
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
    cmd = words[i]
    if cmd[0] in "$\"'`{}()<>0123456789-" or is_path_like(cmd) or cmd.startswith("${{"):
        return None, []
    return cmd, words[i + 1 :]


# --------------------------------------------------------------------------- workflow model


class RunBlock:
    def __init__(self, path: Path, lines: List[Tuple[int, str]], shell: Optional[str]) -> None:
        self.path, self.lines, self.shell = path, lines, shell


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
        """Every `run:` scalar, with the `shell:` of the step it belongs to."""
        steps = self._steps()
        for step in steps:
            shell = None
            for no, code in step:
                m = re.match(r"^\s*-?\s*shell:\s*(.+?)\s*$", code)
                if m:
                    shell = unquote(m.group(1)).lower()
            idx = 0
            while idx < len(step):
                no, code = step[idx]
                m = re.match(r"^(\s*)-?\s*run:\s*(.*)$", code)
                if not m:
                    idx += 1
                    continue
                key_indent = indent_of(code)
                value = m.group(2).strip()
                if value in ("|", "|-", "|+", ">", ">-", ">+"):
                    block: List[Tuple[int, str]] = []
                    idx += 1
                    while idx < len(step):
                        bno, bcode = step[idx]
                        braw = self.raw[bno - 1]
                        if braw.strip() and indent_of(braw) <= key_indent:
                            break
                        block.append((bno, braw))
                        idx += 1
                    if value.startswith(">"):
                        # Folded scalar: one logical command line, reported at its first line.
                        joined = " ".join(strip_comment(t).strip() for _, t in block if t.strip())
                        block = [(block[0][0], joined)] if block else []
                    yield RunBlock(self.path, block, shell)
                    continue
                yield RunBlock(self.path, [(no, unquote(value))], shell)
                idx += 1

    def _steps(self) -> List[List[Tuple[int, str]]]:
        """Group lines into steps: a step starts at a `- ` item under a `steps:` key."""
        steps: List[List[Tuple[int, str]]] = []
        step_indent: Optional[int] = None
        current: Optional[List[Tuple[int, str]]] = None
        in_steps = False
        steps_indent = 0
        for no, code in enumerate(self.code, 1):
            if not code.strip():
                if current is not None:
                    current.append((no, code))
                continue
            ind = indent_of(code)
            if re.match(r"^\s*steps:\s*$", code):
                in_steps, steps_indent, step_indent, current = True, ind, None, None
                continue
            if in_steps and ind <= steps_indent:
                in_steps, current = False, None
            if not in_steps:
                continue
            if code.lstrip().startswith("- ") and (step_indent is None or ind == step_indent):
                step_indent = ind
                current = [(no, code)]
                steps.append(current)
            elif current is not None:
                current.append((no, code))
        return steps


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


def check_uses(wf: Workflow) -> Iterator[Finding]:
    for no, value, comment in wf.uses():
        if value.startswith("./"):
            continue
        if value.startswith("docker://"):
            if not DIGEST_RE.search(value):
                yield Finding(wf.path, no, 1, f"uses: {value} has no digest",
                              "pin the image as docker://<image>@sha256:<digest>  # <tag>")
            elif not comment.startswith("#") or len(comment) < 3:
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
        if not comment.startswith("#") or len(comment.lstrip("# ")) == 0:
            yield Finding(wf.path, no, 1, f"uses: {value} has no `# <version>` comment",
                          "append `# <version>` after the SHA; Dependabot bumps the SHA by reading it")


def check_runner_labels(wf: Workflow, manifest: Manifest) -> Iterator[Finding]:
    for no, label in wf.latest_labels():
        if label in manifest.tracked_labels:
            continue
        family = label.split("-")[0]
        named = {"ubuntu": "ubuntu-24.04", "macos": "macos-26", "windows": "windows-2025"}[family]
        yield Finding(wf.path, no, 2, f"runner label {label} floats with GitHub's schedule",
                      f"name the image (today {label} is {named}); or declare `runs-on={label}` in {MANIFEST} "
                      f"if this repo deliberately tracks it, with every tool it uses from the image pinned")


def check_service_images(wf: Workflow) -> Iterator[Finding]:
    for no, image in wf.images():
        if DIGEST_RE.search(image):
            continue
        name, sep, tag = image.rpartition(":")
        if not sep or "/" in tag:
            tag = ""
        if not VERSION_TAG_RE.search(tag):
            yield Finding(wf.path, no, 5, f"service image {image} has no x.y version tag",
                          "use an exact tag (e.g. postgres:17.9, neo4j:5.26.30-community) or an @sha256 digest")


def _exact_findings(kind: str, args: Sequence[str]) -> List[str]:
    """Names of package arguments that lack an exact version, for one install command."""
    bad: List[str] = []
    skip_next = False
    positional: List[str] = []
    for a in args:
        if skip_next:
            skip_next = False
            continue
        if a.startswith("-"):
            if a in INSTALL_VALUE_FLAGS:
                skip_next = True
            continue
        positional.append(unquote(a))
    if kind == "cargo":
        has_version = any(a in ("--version", "--vers") for a in args) or any(
            a.startswith("--version=") or a.startswith("--vers=") for a in args
        )
        has_git = any(a in ("--git", "--path") or a.startswith("--git=") or a.startswith("--path=") for a in args)
        return [] if has_version or has_git else [p for p in positional if not is_path_like(p)]
    for p in positional:
        if is_path_like(p) or p.endswith((".txt", ".toml", ".json", ".whl", ".tar.gz")):
            continue
        if not PACKAGE_LIKE_RE.match(p) or p[0].isdigit():
            continue
        if kind == "py" and not EXACT_PY_RE.match(p):
            bad.append(p)
        elif kind == "npm" and not EXACT_NPM_RE.match(p):
            bad.append(p)
        elif kind == "apt" and not EXACT_APT_RE.match(p):
            bad.append(p)
        elif kind == "go" and not EXACT_GO_RE.search(p):
            bad.append(p)
        elif kind == "none":
            bad.append(p)
    return bad


def install_kind(cmd: str, args: Sequence[str]) -> Tuple[Optional[str], List[str], str]:
    """Classify an install command. Returns (kind, package args, human name)."""
    a = list(args)
    sub = [w for w in a if not w.startswith("-")]
    if cmd in ("pip", "pip3") and sub[:1] == ["install"]:
        return "py", a[a.index("install") + 1 :], "pip install"
    if cmd in ("python", "python3") and a[:3] == ["-m", "pip", "install"]:
        return "py", a[3:], "pip install"
    if cmd == "uv" and sub[:2] in (["pip", "install"], ["tool", "install"]):
        idx = a.index("install")
        return "py", a[idx + 1 :], "uv " + sub[0] + " install"
    if cmd == "uvx":
        return "py", a[:1] if a and not a[0].startswith("-") else _first_positional(a), "uvx"
    if cmd == "pipx" and sub[:1] in (["install"], ["run"]):
        return "py", _first_positional(a[a.index(sub[0]) + 1 :]), "pipx " + sub[0]
    if cmd == "npm" and sub[:1] in (["install"], ["i"], ["add"]):
        rest = a[a.index(sub[0]) + 1 :]
        pkgs = [w for w in rest if not w.startswith("-")]
        if not pkgs:
            return None, [], ""  # bare `npm install` is governed by package-lock.json
        return "npm", rest, "npm install"
    if cmd == "npx":
        return "npm", _first_positional(a), "npx"
    if cmd == "yarn" and (sub[:1] == ["add"] or sub[:2] == ["global", "add"]):
        return "npm", a[a.index("add") + 1 :], "yarn add"
    if cmd == "pnpm" and sub[:1] in (["add"], ["dlx"]):
        return "npm", a[a.index(sub[0]) + 1 :], "pnpm " + sub[0]
    if cmd == "cargo" and sub[:1] == ["install"]:
        return "cargo", a[a.index("install") + 1 :], "cargo install"
    if cmd == "go" and sub[:1] == ["install"]:
        return "go", a[a.index("install") + 1 :], "go install"
    if cmd == "gem" and sub[:1] == ["install"]:
        has_v = any(w in ("-v", "--version") or w.startswith("--version=") for w in a)
        return (None if has_v else "none"), a[a.index("install") + 1 :], "gem install"
    if cmd == "brew" and sub[:1] == ["install"]:
        return "none", a[a.index("install") + 1 :], "brew install"
    if cmd in ("apt-get", "apt") and sub[:1] == ["install"]:
        return "apt", a[a.index("install") + 1 :], f"{cmd} install"
    if cmd == "choco" and sub[:1] == ["install"]:
        has_v = any(w in ("--version",) or w.startswith("--version=") for w in a)
        return (None if has_v else "none"), a[a.index("install") + 1 :], "choco install"
    return None, [], ""


def _first_positional(args: Sequence[str]) -> List[str]:
    skip = False
    for w in args:
        if skip:
            skip = False
            continue
        if w.startswith("-"):
            skip = w in INSTALL_VALUE_FLAGS
            continue
        return [w]
    return []


FIX_BY_KIND = {
    "py": "name an exact version, e.g. name==1.2.3 (or install from the lockfile: uv sync --frozen)",
    "npm": "name an exact version, e.g. name@1.2.3 (or install from the lockfile: npm ci)",
    "cargo": "pass --version x.y.z --locked",
    "go": "name an exact version, e.g. module@v1.2.3",
    "apt": "pin the package as name=version, or fetch a pinned build yourself (see hive-client tests/shellcheck.sh)",
    "none": "this installer cannot pin a version; fetch a pinned build yourself (see hive-client tests/shellcheck.sh)",
}


def sh_commands(block: RunBlock) -> Iterator[Tuple[int, str, List[str]]]:
    """(line, command, args) for every command a run block executes."""
    lines = [(no, text) for no, text in block.lines]
    joined: List[Tuple[int, str]] = []
    pending: Optional[Tuple[int, str]] = None
    for no, text in lines:
        text = text.rstrip("\n")
        if pending is not None:
            pending = (pending[0], pending[1][:-1] + " " + text.strip())
        else:
            pending = (no, text)
        if pending[1].rstrip().endswith("\\"):
            continue
        joined.append(pending)
        pending = None
    if pending is not None:
        joined.append(pending)

    heredoc: Optional[str] = None
    for no, text in joined:
        stripped = text.strip()
        if heredoc is not None:
            if stripped == heredoc:
                heredoc = None
            continue
        code = strip_comment(text).strip()
        if not code:
            continue
        m = re.search(r"<<-?\s*['\"]?([A-Za-z_][A-Za-z0-9_]*)['\"]?", code)
        if m:
            heredoc = m.group(1)
            code = code[: m.start()]
        for seg in split_segments(code):
            cmd, args = command_words(seg)
            if cmd is None:
                continue
            yield no, cmd, args


def check_run_blocks(wf: Workflow, manifest: Manifest) -> Iterator[Finding]:
    for block in wf.run_blocks():
        if block.shell and block.shell.split()[0] in ("pwsh", "powershell", "cmd"):
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


def check_noxfile(root: Path) -> Iterator[Finding]:
    nox = root / "noxfile.py"
    if not nox.is_file():
        return
    text = nox.read_text(encoding="utf-8")
    for m in re.finditer(r"session\.(install|run_install)\s*\((.*?)\)\s*$", text, re.S | re.M):
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


def run(root: Path) -> List[Finding]:
    findings: List[Finding] = []
    manifest = Manifest(root / MANIFEST)
    findings.extend(manifest.errors)
    workflows = sorted(p for p in (root / WORKFLOWS).glob("*.y*ml")) if (root / WORKFLOWS).is_dir() else []
    for path in workflows:
        wf = Workflow(path)
        findings.extend(check_uses(wf))
        findings.extend(check_runner_labels(wf, manifest))
        findings.extend(check_service_images(wf))
        findings.extend(check_run_blocks(wf, manifest))
    findings.extend(check_noxfile(root))
    if workflows and not manifest.present and any(f.rule == 4 for f in findings):
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
