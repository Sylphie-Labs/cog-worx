"""hive_shell: reads a shell command the way the guards need it read.

Shared by git-guard.py and secret-guard.py, which sit next to this file in a
repo's .claude/hooks/. Imported, never run. It lives in one place so the two
guards cannot drift on what a command means: a wrapper one of them sees
through and the other does not is a hole in the second.

What it does: splits a command line into simple commands (segments), with
quotes honored and heredoc bodies dropped. One pre-pass is the only reader of
the raw text: it drops comments, joins `\\`-newline continuations, and keeps a
quoted or escaped operator a plain word (`echo ">" x` redirects nothing). A
command inside a process substitution (`<(cmd)`) is a segment of its own. It
also finds the command substitutions inside double quotes that the tokenizer
cannot see; peels `VAR=x env sudo timeout 5 command ...` prefixes off a segment
(one function, peel, for both guards); and finds
the subcommand of a `git ...` segment past git's own global options. It does
not run anything. Standard library only.
"""
import os
import re
import shlex

SEGMENT_BREAKS = {";", "&&", "||", "|", "&", "|&", "(", ")", ";;", "`"}
PUNCTUATION = "();<>|&`"   # shlex's default set plus the backtick
# Words that run the command after them, so the command is what a guard must
# read. A prefix word's own options (and which take a value) and the operands
# it takes before the command (`timeout 5 cmd`: the duration) are skipped.
PREFIX_WORDS = {"env", "command", "exec", "nohup", "time", "builtin",
                "sudo", "timeout", "nice", "stdbuf", "noglob"}
PREFIX_OPTION_VALUES = {
    "sudo": {"-u", "-g", "-h", "-p", "-C", "-D", "-r", "-t", "-U", "--user", "--group"},
    "env": {"-u", "-C", "-S", "--unset", "--chdir", "--split-string"},
    "timeout": {"-s", "--signal", "-k", "--kill-after"},
    "nice": {"-n", "--adjustment"},
    "stdbuf": {"-i", "-o", "-e"},
}
PREFIX_OPERANDS = {"timeout": 1}
# After these, a `NAME=value` word still sets a variable for the command; after
# any other prefix word the shell would try to run it.
CAPTURE_PREFIX = {"env", "sudo", "time"}
# Shells whose `-c` argument is a command line of its own.
SHELLS = {"sh", "bash", "zsh", "dash", "ksh"}
# Words that open a compound command's body or negate a pipeline. A segment
# can start with one (`then git reset --hard`), and the command is behind it.
SHELL_KEYWORDS = {"if", "then", "elif", "else", "do", "while", "until", "!", "{"}
# shlex strips quotes, so a quoted `>` would look like a redirect. Before
# tokenizing, each quoted or escaped punctuation character is swapped for a
# private-use stand-in, and swapped back in the words that come out.
_STAND_IN = {ch: chr(0xE000 + i) for i, ch in enumerate(PUNCTUATION)}
_RESTORE = {v: k for k, v in _STAND_IN.items()}
# Every operator the tokenizer can hand over, longest first within a length.
# shlex returns a run of adjacent punctuation (`;(`, `&&(`) as ONE token, so a
# run is cut back into these.
OPERATORS = ("&>>", "<<<", "&&", "||", ";;", "|&", "&>", ">>", ">&", ">|", "<<", "<&", "<>",
             ";", "|", "&", "(", ")", "<", ">", "`")
ASSIGNMENT_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")
# `<<WORD`, `<<'WORD'`, `<<"WORD"`, `<<\WORD` (and `<<-`). Not `<<<`, a here-string.
# Group 1 is a backslash, group 2 a quote: either means the body is literal.
HEREDOC_RE = re.compile(r"(?<!<)<<(?!<)-?[ \t]*(\\?)(['\"]?)([A-Za-z_][A-Za-z0-9_]*)\2")
# Global git options that take a value, and ones that do not.
GIT_GLOBAL_WITH_VALUE = {"-C", "-c", "--git-dir", "--work-tree", "--namespace", "--exec-path"}
GIT_GLOBAL_FLAGS = {"-p", "--paginate", "-P", "--no-pager", "--no-replace-objects",
                    "--literal-pathspecs", "--glob-pathspecs", "--noglob-pathspecs",
                    "--icase-pathspecs", "--no-optional-locks", "--bare"}


class ParseError(Exception):
    """The command could not be tokenized (an unterminated quote, say)."""


def strip_heredocs(command):
    """Drop heredoc bodies so their lines are not read as commands. Several
    heredocs on one line are consumed in order."""
    out = []
    lines = command.split("\n")
    failed = set()        # terminators with no line after this point
    i = 0
    while i < len(lines):
        line = lines[i]
        out.append(line)
        i += 1
        for m in HEREDOC_RE.finditer(line):
            terminator = m.group(3)
            if terminator in failed:
                continue
            j = i
            while j < len(lines) and lines[j].strip() != terminator:
                j += 1
            if j < len(lines):
                i = j + 1  # skip the body and the terminator line
            else:
                failed.add(terminator)   # `<<` inside a quoted string, not a heredoc
    return out


class Segment:
    """One simple command.

    tokens  its words, starting with any `VAR=x` or `env` prefix
    reads   files it redirects input from (`< file`, `<> file`)
    writes  files it redirects output to (`> file`, `>> file`, `>| file`,
            `&> file`, `2>&file`)
    quiet   True when, after all its redirects, its standard output ends up in
            /dev/null, so whatever it would print is not shown. The last
            redirect of standard output wins. Standard output sent to another
            descriptor (`>&2`) is not tracked and counts as not quiet. A
            redirect after a subshell (`(cmd) >/dev/null`) belongs to the
            subshell, not to the commands inside, so those are not quiet.
    """
    __slots__ = ("tokens", "reads", "writes", "quiet")

    def __init__(self):
        self.tokens, self.reads, self.writes, self.quiet = [], [], [], False

    def __bool__(self):
        return bool(self.tokens or self.reads or self.writes)


REDIRECT_OPS = {"<", "<>", "<<", "<<<", "<&", ">", ">>", ">|", "&>", "&>>", ">&"}


def _operator_tokens(tokens):
    """Cut every token made only of punctuation into operators, greedy longest
    match, so `;(` is `;` then `(`. Other tokens pass through. Linear in the
    total length: each step looks at most three characters ahead."""
    for tok in tokens:
        if not tok or any(ch not in PUNCTUATION for ch in tok):
            yield tok
            continue
        i, n = 0, len(tok)
        while i < n:
            for width in (3, 2, 1):
                if tok[i:i + width] in OPERATORS:
                    yield tok[i:i + width]
                    i += width
                    break
            else:       # cannot happen: every punctuation character is an operator
                yield tok[i]
                i += 1


_BOUNDARY = set(";|&()<>")   # unquoted characters after which a `#` starts a comment


def _mask_quoted(text):
    """Read the raw text once, the way a POSIX shell does, and return what
    shlex should see. In one linear walk that tracks quote state:

    - quoted or backslash-escaped punctuation becomes its stand-in, so it is
      a word and not an operator;
    - a backslash followed by a newline, outside single quotes, is removed
      with the newline (a line continuation);
    - an unquoted `#` that begins a word starts a comment, which is dropped up
      to the next newline; any other `#` is an ordinary character;
    - an unquoted newline becomes ` ; `; a quoted one stays a newline.

    Raises ParseError if the text already holds a stand-in character.
    """
    if any(0xE000 <= ord(ch) < 0xE000 + len(PUNCTUATION) for ch in text):
        raise ParseError("could not parse the command (reserved character); not checked")
    out = []
    quote = None
    word_start = True     # True where a `#` would begin a comment
    i, n = 0, len(text)
    while i < n:
        ch = text[i]
        if quote == "'":
            if ch == "'":
                quote = None
            out.append(_STAND_IN.get(ch, ch))
        elif ch == "\\" and i + 1 < n:
            nxt = text[i + 1]
            i += 1
            if nxt != "\n":
                out.append(ch)
                out.append(_STAND_IN.get(nxt, nxt))
                word_start = False
        elif quote == '"':
            if ch == '"':
                quote = None
            out.append(_STAND_IN.get(ch, ch))
        elif ch in "'\"":
            quote = ch
            out.append(ch)
            word_start = False
        elif ch == "#" and word_start:
            while i + 1 < n and text[i + 1] != "\n":
                i += 1
        elif ch == "\n":
            out.append(" ; ")
            word_start = True
        else:
            out.append(ch)
            word_start = ch.isspace() or ch in _BOUNDARY
        i += 1
    return "".join(out)


_RESTORE_TABLE = str.maketrans(_RESTORE)


def _restore(word):
    return word.translate(_RESTORE_TABLE)


def segments_with_redirects(command):
    """Split a shell command into simple commands, as a list of Segment.

    A heredoc terminator, a here-string and a file-descriptor target (`2>&1`)
    are not files and are dropped. Runs of operators (`;(`, `&&(`) are split.

    Quotes are honored (so `echo 'git reset --hard'` is one echo argument),
    operators split, newlines split. Raises ParseError on shell the tokenizer
    cannot read, such as an unterminated quote.
    """
    text = _mask_quoted("\n".join(strip_heredocs(command)))
    # posix=True strips an unquoted backslash, which is what Git Bash does
    # before git sees the word, so a Windows path in a command tokenizes the
    # way the shell would run it. Quoted forms keep their backslashes.
    lex = shlex.shlex(text, posix=True, punctuation_chars=PUNCTUATION)
    lex.whitespace_split = True
    lex.commenters = ""    # _mask_quoted already dropped the comments
    try:
        tokens = list(lex)
    except ValueError as e:
        raise ParseError(f"could not parse the command ({e}); not checked")
    out = []
    seg = Segment()
    pending = None   # (op, fd) when the next token is a redirection's target
    for tok in _operator_tokens(tokens):
        if pending and tok == "(":
            pending = None      # `<(cmd)`, `>(cmd)`: a process substitution, not a target
        if pending:
            op, fd = pending
            pending = None
            stdout = "&" in op and op != ">&" or fd in (None, "1")
            if op in ("<", "<>"):
                seg.reads.append(_restore(tok))
            elif op == ">&":
                if tok.isdigit() or tok == "-":
                    # A descriptor, not a file. Stdout now follows another
                    # descriptor, which is not tracked: not quiet.
                    if stdout:
                        seg.quiet = False
                else:
                    seg.writes.append(_restore(tok))
                    if stdout:
                        seg.quiet = tok == "/dev/null"
            elif op not in ("<<", "<<<", "<&"):    # those are not files
                seg.writes.append(_restore(tok))
                if stdout:
                    seg.quiet = tok == "/dev/null"
            continue
        if tok in SEGMENT_BREAKS:
            if seg:
                out.append(seg)
            seg = Segment()
        elif tok in REDIRECT_OPS:
            # A redirection: the next token is its target, not an argument,
            # and a bare digit just before it (2>&1) is the fd, not an argument.
            fd = None
            if seg.tokens and seg.tokens[-1].isdigit():
                fd = seg.tokens.pop()
            pending = (tok, fd)
        else:
            seg.tokens.append(_restore(tok))
    if seg:
        out.append(seg)
    return out


def segments(command):
    """The token lists of segments_with_redirects, redirections dropped."""
    return [seg.tokens for seg in segments_with_redirects(command) if seg.tokens]


def _heredoc_body(command, start, name, failed):
    """(body_start, body_end, resume): the body of a heredoc whose body begins
    at `start`, up to the line that is `name`, and where scanning resumes (after
    that line); or None when no such line follows. `failed` remembers names
    already searched for, so a command of many unterminated markers stays linear."""
    if name in failed:
        return None
    n = len(command)
    pos = start
    while pos <= n:
        nl = command.find("\n", pos)
        end = n if nl < 0 else nl
        if command[pos:end].strip() == name:
            return start, pos, min(end + 1, n)
        if nl < 0:
            break
        pos = nl + 1
    failed.add(name)
    return None


def _skip_heredocs(command, start, names, failed):
    """Where scanning resumes after the bodies of `names`, opened on the line
    before `start`, in order; -1 when one has no terminator line."""
    for name in names:
        found = _heredoc_body(command, start, name, failed)
        if not found:
            return -1
        start = found[2]
    return start


def substitution_spans(command):
    """(start, end, body) for each top-level `$(...)` and backtick substitution
    the shell would run, where `command[start:end]` is the whole substitution
    including its delimiters. Not inside single quotes, where they are literal;
    not in the body of a heredoc whose delimiter is quoted or escaped
    (`<<'EOF'`), where the shell expands nothing. `$((` is arithmetic, and an
    unterminated one gives nothing.

    segments() already sees an unquoted substitution, because parentheses and
    backticks split segments. It cannot see one inside double quotes, which
    the tokenizer hands over as a single word, and that is the common shape:
    `echo "token: $(gh auth token)"`. Nested substitutions stay inside their
    parent's body; the caller parses each body again. One linear pass.
    """
    out = []
    i, n = 0, len(command)
    in_single = in_double = False
    pending = []          # heredocs opened on this line: (name, literal)
    failed = set()
    plain_until = 0       # inside a bare heredoc body, quotes are ordinary text
    regions = []          # heredoc bodies found, in order: (start, end, resume, literal)
    while i < n:
        while regions and i >= regions[0][0]:
            _, body_end, resume, literal = regions.pop(0)
            if literal:
                i = max(i, resume)          # nothing in the body is expanded: step over it
            else:
                plain_until = max(plain_until, body_end)
        if i >= n:
            break
        if plain_until and i >= plain_until:
            plain_until = 0
            in_single = in_double = False
        plain = i < plain_until
        ch = command[i]
        if in_single:
            in_single = ch != "'"
        elif ch == "\\":
            i += 1                      # the next character is escaped
        elif ch == "'" and not in_double and not plain:
            in_single = True
        elif ch == '"' and not plain:
            in_double = not in_double
        elif ch == "$" and command[i + 1:i + 2] == "(" and command[i + 2:i + 3] != "(":
            # Inside the body, a parenthesis in quotes is text, not nesting, and
            # a heredoc is opaque: its lines are not read for quotes or parens.
            depth, j = 1, i + 2
            q = None
            heredocs = []
            while j < n and depth:
                c = command[j]
                if q == "'":
                    if c == "'":
                        q = None
                elif c == "\\":
                    j += 1
                elif q == '"':
                    if c == '"':
                        q = None
                elif c in "'\"":
                    q = c
                elif c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
                elif c == "<" and command.startswith("<<", j):
                    m = HEREDOC_RE.match(command, j)
                    if m:
                        heredocs.append(m.group(3))
                        j = m.end() - 1
                elif c == "\n" and heredocs:
                    j = _skip_heredocs(command, j + 1, heredocs, failed)
                    heredocs = []
                    if j < 0:           # a heredoc never ends: neither does this substitution
                        j, depth = n, 1
                        break
                    j -= 1
                j += 1
            if depth == 0:
                out.append((i, j, command[i + 2:j - 1]))
            i = j - 1
        elif ch == "`":
            j = i + 1
            heredocs = []
            while j < n and command[j] != "`":
                c = command[j]
                if c == "\\":
                    j += 1
                elif c == "<" and command.startswith("<<", j):
                    m = HEREDOC_RE.match(command, j)
                    if m:
                        heredocs.append(m.group(3))
                        j = m.end() - 1
                elif c == "\n" and heredocs:
                    j = _skip_heredocs(command, j + 1, heredocs, failed)
                    heredocs = []
                    if j < 0:
                        j = n
                        break
                    j -= 1
                j += 1
            if j < n:
                out.append((i, j + 1, command[i + 1:j]))
            i = j
        elif ch == "<" and not in_double and not plain and command.startswith("<<", i):
            m = HEREDOC_RE.match(command, i)
            if m:
                pending.append((m.group(3), bool(m.group(1) or m.group(2))))
                i = m.end() - 1
        elif ch == "\n" and pending and not in_double and not plain:
            start = i + 1
            for name, literal in pending:
                found = _heredoc_body(command, start, name, failed)
                if found:
                    body_start, body_end, start = found
                    regions.append((body_start, body_end, start, literal))
            pending = []
        i += 1
    return out


# A stand-in word for a substitution cut out of a command line. Private-use
# characters: the tokenizer reads them as ordinary word characters, and they
# are not its own punctuation stand-ins. A command that already holds one
# cannot be cut (callers raise ParseError).
PH_OPEN, PH_CLOSE = "\ue100", "\ue101"
PLACEHOLDER_RE = re.compile(PH_OPEN + r"(\d+)" + PH_CLOSE)


def cut_substitutions(command, spans):
    """`command` with each span of substitution_spans replaced by a placeholder
    word, so the segments of what is left never see inside a substitution."""
    if PH_OPEN in command or PH_CLOSE in command:
        raise ParseError("could not parse the command (reserved character); not checked")
    pieces, last = [], 0
    for k, (start, end, _) in enumerate(spans):
        pieces += [command[last:start], f"{PH_OPEN}{k}{PH_CLOSE}"]
        last = end
    pieces.append(command[last:])
    return "".join(pieces)


def restore_substitutions(word, command, spans):
    """`word` with each placeholder replaced by the substitution it stands for."""
    return PLACEHOLDER_RE.sub(
        lambda m: command[spans[int(m.group(1))][0]:spans[int(m.group(1))][1]], word)


def prefix_base(word):
    """A prefix word's name: its basename, without a trailing .exe."""
    base = os.path.basename(word)
    return base[:-4] if base.endswith(".exe") else base


def peel(tokens):
    """Peel `VAR=x env sudo -u me timeout 5 command ...` off a segment.

    Returns (saw_env, rest, assignments, reaching):
      saw_env      an `env` was peeled: with nothing left after it, env prints
                   the environment
      rest         the command word and its arguments
      assignments  [(`NAME=value`, capture_ok)] for each assignment peeled;
                   capture_ok says it sits where the shell or env or sudo
                   sets it for the command (before any prefix word, or right
                   after env, sudo or time) and not where the shell would try
                   to run it (after command, nohup, timeout...)
      reaching     the capture_ok assignments that survive to the command:
                   sudo, `env -i` and `env -` clear what came before them

    A prefix word matches by basename (`/usr/bin/env`, `env.exe`); a shell
    keyword (then, do, !, {...) must match exactly (`./then` is a command).
    One pass by index over the one list: a command can carry thousands of
    prefix words.
    """
    saw_env = False
    assignments = []
    dropped = 0          # assignments before the last environment-clearing word
    capture_ok = True
    i, n = 0, len(tokens)
    while i < n:
        tok = tokens[i]
        if ASSIGNMENT_RE.match(tok):
            assignments.append((tok, capture_ok))
            i += 1
            continue
        if tok in SHELL_KEYWORDS:
            i += 1
            continue
        word = prefix_base(tok)
        if word not in PREFIX_WORDS:
            break
        i += 1
        capture_ok = word in CAPTURE_PREFIX
        saw_env = saw_env or word == "env"
        if word == "sudo":
            dropped = len(assignments)
        takes_value = PREFIX_OPTION_VALUES.get(word, ())
        own_operands = PREFIX_OPERANDS.get(word, 0)
        has_options = word in PREFIX_OPTION_VALUES and i < n and tokens[i].startswith("-")
        if not has_options and not (own_operands and i < n):
            continue
        while i < n and tokens[i].startswith("-"):
            opt = tokens[i]
            i += 1
            if word == "env" and opt in ("-i", "--ignore-environment", "-"):
                dropped = len(assignments)
            if opt in takes_value and i < n:
                i += 1
        i = min(i + own_operands, n)
    reaching = [tok for tok, ok in assignments[dropped:] if ok]
    return saw_env, tokens[i:], assignments, reaching


def shell_c_argument(args):
    """The command line given to `sh -c ...` (also `-lc`, `-xc`), or None."""
    for i, a in enumerate(args):
        if a.startswith("-") and not a.startswith("--") and "c" in a[1:]:
            return args[i + 1] if i + 1 < len(args) else None
    return None


DRIVE_RE = re.compile(r"/([A-Za-z])(?=/|$)")


def native_path(word, windows=None):
    """`word` as this platform names the path. Git Bash spells a Windows drive
    as a directory (`/c/Users/me` is `C:/Users/me`), and Python on Windows does
    not know that. Its other paths (`/tmp`, `/usr/bin`) have no fixed Windows
    name and are returned as written, as is every word off Windows."""
    if windows is None:
        windows = os.name == "nt"
    m = DRIVE_RE.match(word) if windows else None
    return m.group(1) + ":" + (word[2:] or "/") if m else word


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
