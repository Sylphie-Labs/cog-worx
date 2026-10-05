#!/usr/bin/env python3
"""hive_redact: cut the next slice of a session transcript and redact secrets
from it before the scribe reads it.

Installed by init_repo.py into a repo's .claude/hooks/ and called by both
capture scripts (hive-scribe-capture.sh through transcript_cut in _lib.sh, and
hive-scribe-capture.ps1 directly):

    python3 hive_redact.py cut <transcript> <from_offset> <out_file>

WHY. The capture prompt asks the scribe not to reproduce secrets, but that is
prose applied by a model after it has already read them. A token that reached
the transcript (tik_01M1MK92XRJ1CZSV52AQ35Q8HA) was one Stop hook away from
the hive. Redaction here is deterministic and happens before the slice exists
on disk, so the scribe never sees the value.

CONTRACT.
  exit 0   stdout line 1: the end offset in the ORIGINAL transcript (the next
           pass starts there). Line 2: what was redacted, as
           `github-token=1 jwt=2`, or empty. <out_file> holds the redacted
           slice. When there is no complete new line, line 1 is <from_offset>
           and nothing is written.
  other    nothing was written. The caller skips the pass and leaves its
           offset alone, so the same bytes are retried on the next Stop.

The cut ends at the last newline, so no JSONL record is split. Offsets count
bytes of the original file; redaction changes the slice's length, never the
offsets. A transcript shorter than <from_offset> was rewritten: start from 0.

WHAT IS REDACTED. The KINDS table below: private-key, github-token,
gitlab-token, google-api-key, npm-token, stripe-key, api-key, aws-access-key,
slack-token, hive-key, jwt, bearer-token, basic-auth, url-credential and
assignment. Each match becomes [REDACTED:<kind>]. The kernel-side check
(tik_01M2TX3JKJ4XCCH7N6V2BQZFPE) should mirror these kinds: Python and Rust
cannot share the code, so the list of kinds is the contract.

SPANS. find_spans(line) gives the same matches as positions in the line. It
is for sweep_secrets.py in hive-client, which reports and overwrites secrets
that are already on disk. redact_line and find_spans are one loop (_scan), so
what capture redacts and what the sweep finds cannot differ.

A MASK IS NOT A SECRET. A value that is only `*`, or that begins with a token's
prefix (zero to eleven letters, `_` or `-`) and four or more `*`, where what
follows the stars is the end of the value or a byte that is not a letter, digit
or `*`, is what the sweep's scrub writes (`ghs_****`, and `ghs_****;echo` when
punctuation followed the token). It is left alone: there is nothing in it to
redact. Stars followed by letters or digits (`pass****word9`) are not a mask.

The slice stays valid JSONL. A transcript record is one line of JSON, so a
secret sits inside a JSON string, where a quote is `\\"` and a newline is
`\\n`. No replacement contains a quote or a backslash, and a match never
starts or stops inside an escape pair, because the patterns either stop at
the first quote or backslash (tokens, url-credential, unquoted values) or
consume escape pairs whole. Two kinds consume pairs whole: the private-key
block (it runs to its END line, or to the end of the string when the block is
cut off, which in doubly-encoded JSON is the end of the outer string), and a
quoted `assignment` value (it runs to the same closing quote sequence it was
opened with, at most 512 characters, and never past a bare quote or the end of
the line; with no closing quote it is read as an unquoted value).

`assignment`, `bearer-token` and `basic-auth` (the header form) are shapes
around a value, so they redact only a value that reads as a literal secret
(see _assignment_is_literal, _bearer_is_literal and _basic_is_literal). For an
assignment that means: a quoted value with a letter and a digit, or with no
whitespace and no `/` that does not start with `$`, `<` or `{`; or an unquoted
value (it runs to whitespace, a quote, a backslash or a bracket, so
punctuation is part of it) with a letter and a digit, or assigned to an
UPPER_CASE name with a bare `=`, no `/` and no leading `$`. A bearer or basic
value needs a letter and a digit, or 20 or more letters and digits. Code like
`token = get_token(args)` is left alone. The `curl -u user:PASSWORD` form of
basic-auth is redacted without that test, but only inside a curl command.

Over-redaction is accepted: a false positive costs the scribe one detail, a
false negative is a leak. Standard library only.
"""
import os
import re
import sys

# A token starts at a word boundary. `\b` is not enough: inside a JSON string
# a token on its own line is preceded by the two characters `\n`, and `n` is a
# word character. So: not after a word character, or right after an escape.
_LEFT = rb"(?:(?<![A-Za-z0-9_])|(?<=\\[nrt]))"
# The same, for prefixes that are ordinary English after a hyphen ("task-...").
_LEFT_NO_DASH = rb"(?:(?<![A-Za-z0-9_-])|(?<=\\[nrt]))"

# One JSON-string character: anything but a quote or a backslash, or a whole
# escape pair. Matching pairs whole is what keeps a cut from landing inside one.
_STR_CHAR = rb'(?:[^"\\]|\\.)'

_ASSIGN_NAMES = (
    rb"(?:password|passwd|passphrase|secret|token|api[_-]?key|access[_-]?key|private[_-]?key)"
)

_BEARER = re.compile(rb"(?i)(?P<keep>\bBearer\s+)(?P<secret>[A-Za-z0-9._~+/=-]{16,})")
_BASIC_HEADER = re.compile(rb"(?i)(?P<keep>\bBasic\s+)(?P<secret>[A-Za-z0-9._~+/=-]{16,})")
# curl -u user:PASSWORD / --user user:PASSWORD. Only inside a curl command,
# within a bounded distance of it, so `docker run -u 1000:1000` and
# `sudo -u deploy` are left alone and the pattern stays linear.
_CURL_USER = re.compile(
    rb"(?P<keep>\bcurl\b[^\r\n]{0,300}?\s(?:--user[\s=]+|-u\s*)\\*[\"']?"
    rb"[^\s:\"'\\]{1,128}:)(?P<secret>[^\s\"'\\]{1,256})")

# (kind, pattern). Order matters: the specific shapes run first, so the
# generic `assignment` rule sees their placeholders, not their values.
KINDS = [
    # One pass, linear: the body stops at the first END line, and the END line
    # is optional, so a block cut off before its end is redacted to the end of
    # its string. The pattern cannot fail once the header matched, so a long
    # run of headers is consumed by one match, not rescanned from each one.
    ("private-key", re.compile(
        rb"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----"
        rb"(?:(?!-----END [A-Z0-9 ]*PRIVATE KEY-----)" + _STR_CHAR + rb")*"
        rb"(?:-----END [A-Z0-9 ]*PRIVATE KEY-----)?")),
    ("github-token", re.compile(
        _LEFT + rb"(?:gh[pousr]_[A-Za-z0-9]{36,}|github_pat_[A-Za-z0-9_]{22,})")),
    ("gitlab-token", re.compile(_LEFT + rb"glpat-[A-Za-z0-9_-]{20,}")),
    ("google-api-key", re.compile(_LEFT + rb"AIza[0-9A-Za-z_-]{35}")),
    ("npm-token", re.compile(_LEFT + rb"npm_[A-Za-z0-9]{36}")),
    ("stripe-key", re.compile(_LEFT + rb"(?:sk|rk)_(?:live|test)_[A-Za-z0-9]{16,}")),
    # sk-ant-..., sk-proj-...: at least one digit, so a hyphenated phrase that
    # happens to start with "sk-" is left alone.
    ("api-key", re.compile(
        _LEFT_NO_DASH + rb"sk-(?=[A-Za-z0-9_-]*[0-9])[A-Za-z0-9_-]{20,}")),
    ("aws-access-key", re.compile(_LEFT + rb"(?:AKIA|ASIA)[0-9A-Z]{16}(?![A-Za-z0-9])")),
    ("slack-token", re.compile(_LEFT + rb"xox[baprs]-[A-Za-z0-9-]{10,}")),
    ("hive-key", re.compile(_LEFT + rb"hive_[A-Za-z0-9]{20,}")),
    ("jwt", re.compile(
        _LEFT + rb"eyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")),
    ("bearer-token", _BEARER),
    ("basic-auth", _BASIC_HEADER),
    ("basic-auth", _CURL_USER),
    # scheme://user:PASSWORD@host -- only the password goes. The password may
    # hold an `@`: it runs to the LAST `@` of the authority.
    ("url-credential", re.compile(
        rb"(?P<keep>(?<=://)[^\s/:@\"\\]{0,128}:)(?P<secret>[^\s/\"\\]{1,256})(?=@)")),
    # NAME=value / "name": "value" / name: value, where the name says secret.
    # This pattern matches the name and the separator only; _assignment_values
    # reads the value. The name's tail is bounded and its head is not matched
    # at all, so a long run of word characters costs linear time, not
    # quadratic. The tail takes no dot, so `secret-guard.py:50:...` in grep
    # output is not an assignment.
    ("assignment", re.compile(
        rb"(?i)(?P<keep>(?P<name>" + _ASSIGN_NAMES + rb"[A-Za-z0-9_-]{0,40})"
        rb"(?P<q1>(?:\\*[\"'])?)(?P<sep>\s*[=:]\s*)(?P<q2>(?:\\*[\"'])?))")),
]

_NOT_A_SECRET = re.compile(rb"(?i)^(?:true|false|null|none|[0-9.]+)$")
_HEAD = re.compile(rb"[A-Za-z0-9_-]{0,64}$")
_ALNUM = re.compile(rb"[A-Za-z0-9]+")
_MARKER = b"[REDACTED:"
# A mask, not a secret: nothing but `*`, or a value that BEGINS with zero to
# eleven of `[A-Za-z_-]` and four or more `*`, after which comes the end of the
# value or a byte that is not a letter, digit or `*` (`ghs_****`, `sk-ant-****`,
# `ghs_****;echo`). It is what the sweep's scrub leaves behind, so a scrubbed
# file must not match again. Stars followed by letters or digits are not a
# mask: `pass****word9` still holds a secret. Use `.match`.
_MASK = re.compile(rb"\*+\Z|[A-Za-z_-]{0,11}\*{4,}(?![A-Za-z0-9*])")

_QUOTED_MAX = 512        # longest quoted value read
_UNQUOTED_WINDOW = 256   # unquoted value read in one bounded step
# A quote, with the run of backslashes before it; or a raw line break. The
# lookbehind keeps a search from starting inside a run, so a long run of
# backslashes is read once, not once per position.
_QUOTE_EVENT = re.compile(rb"(?<!\\)(\\*+)([\"'])|[\r\n]")
_UNQUOTED = re.compile(rb"[^\s\"'\\()\[\]{}<>]*")
_UNQUOTED_WINDOWED = re.compile(rb"[^\s\"'\\()\[\]{}<>]{0,%d}" % _UNQUOTED_WINDOW)


def _has_letter_and_digit(value):
    return bool(re.search(rb"[A-Za-z]", value)) and bool(re.search(rb"[0-9]", value))


def _quoted_value_end(line, start, quote, backslashes):
    """Where a quoted value that begins at `start` ends, or None. `quote` is the
    opening quote character and `backslashes` the length of the escape run in
    front of it (0 in plain text, 1 inside a JSON string, 3 inside JSON inside
    JSON). The value ends at the same sequence: that many backslashes and the
    quote, the run being exactly that long. A `"` with a shorter run is a
    boundary of an enclosing string, so the value must not cross it; one with
    a longer run is an escaped character of the value. A raw line break, or
    more than _QUOTED_MAX characters, means no closing quote: None."""
    pos = start
    limit = start + _QUOTED_MAX + backslashes + 1
    while True:
        m = _QUOTE_EVENT.search(line, pos, limit)
        if m is None or m.group(2) is None:
            return None
        run, q = len(m.group(1)), m.group(2)
        if q == quote and run == backslashes:
            return m.start() if m.start() - start <= _QUOTED_MAX else None
        if q == b'"' and (run < backslashes if quote == b'"' else run == 0):
            return None
        pos = m.end()


def _assignment_is_literal(m, value, quoted):
    """An assignment's value is redacted unless it reads as code or prose. Not
    a bare number or keyword, and of at least 8 characters, and:
    - quoted: it has a letter and a digit; or it has no whitespace and no `/`
      and does not start with `$`, `<` or `{` (`password: "correcthorse"`, but
      not a path or a template such as "${DB_PASSWORD}");
    - unquoted: it has a letter and a digit; or the name is env-style (no
      lowercase letter anywhere in it, head included), the separator is a bare
      `=`, and the value has no `/` and does not start with `$`
      (`DB_PASSWORD=value`).
    Lowercase unquoted code like `password=password` or `token = getToken` is
    left alone."""
    if len(value) < 8 or _NOT_A_SECRET.match(value):
        return False
    if _has_letter_and_digit(value):
        return True
    if quoted:
        return (not re.search(rb"\s", value) and b"/" not in value
                and value[:1] not in (b"$", b"<", b"{"))
    if m.group("q1") or m.group("q2") or m.group("sep") != b"=":
        return False
    if b"/" in value or value.startswith(b"$"):
        return False
    # The pattern does not match the name's head (that would be an unbounded
    # prefix); read it backward from the match start, bounded.
    head = _HEAD.search(m.string[max(0, m.start() - 64):m.start()]).group(0)
    return not re.search(rb"[a-z]", head + m.group("name"))


def _assignment_values(pattern, line):
    """[(start, end)] of the value of every secret-named assignment in `line`.
    The pattern finds the name and separator; the value is read here: up to
    the closing quote for a quoted one, else up to whitespace, a quote, a
    backslash or a bracket."""
    found = []
    scan = 0      # where the next search starts
    while True:
        m = pattern.search(line, scan)
        if m is None:
            break
        start = m.end()
        quoted = False
        end = None
        q2 = m.group("q2")
        if q2:
            qlen = len(q2.lstrip(b"\\"))
            end = _quoted_value_end(line, start, q2[-1:], len(q2) - qlen)
            quoted = end is not None
        if end is None:
            end = _UNQUOTED_WINDOWED.match(line, start).end()
        value = line[start:end]
        if (value.startswith(_MARKER) or _MASK.match(value)
                or not _assignment_is_literal(m, value, quoted)):
            scan = m.end()
            continue
        if not quoted and end - start == _UNQUOTED_WINDOW:
            end = _UNQUOTED.match(line, end).end()      # take the rest of a long value
        found.append((start, end))
        scan = end
    return found


def _bearer_is_literal(m):
    """A bearer value is redacted when it has a letter and a digit, or is 20 or
    more letters and digits only. "Bearer authentication-is-required" is left
    alone: it has hyphens and no digit."""
    value = m.group("secret")
    return _has_letter_and_digit(value) or (
        len(value) >= 20 and _ALNUM.fullmatch(value) is not None)


def _basic_is_literal(m):
    """The same test for a Basic header value. Base64 padding (`=`) at the end
    does not count against the letters-and-digits form: the credentials
    `user:password` encode to 20 or more letters, often with no digit."""
    value = m.group("secret")
    return _has_letter_and_digit(value) or (
        len(value.rstrip(b"=")) >= 20 and _ALNUM.fullmatch(value.rstrip(b"=")) is not None)


# Patterns whose match is redacted only when the value reads as a literal
# secret. The curl form of basic-auth is a shape that is always redacted.
_LITERAL_CHECKS = {_BEARER: _bearer_is_literal, _BASIC_HEADER: _basic_is_literal}
_SCANNERS = {"assignment": _assignment_values}


def _matches(kind, pattern, line):
    """[(start, end)] in `line`, in order: the part of each match of this kind
    that is replaced. A value that is already a marker, or a mask, is not a
    match."""
    if kind in _SCANNERS:
        return _SCANNERS[kind](pattern, line)
    has_secret = "secret" in pattern.groupindex
    check = _LITERAL_CHECKS.get(pattern)
    found = []
    for m in pattern.finditer(line):
        if not has_secret:
            found.append(m.span())
            continue
        secret = m.group("secret")
        if secret.startswith(_MARKER) or _MASK.match(secret):
            continue
        if check is not None and not check(m):
            continue
        found.append(m.span("secret"))
    return found


def _replace(line, spans, found, kind):
    """Replace each (start, end) of `found` in `line` with this kind's marker.
    `spans` describes the markers already in `line`: (start, end, orig_start,
    orig_end, kind), in order, where start:end is the marker in `line` and
    orig_start:orig_end is what it replaced in the original line. Returns the
    new line and the new list.

    Offsets are mapped back through the earlier markers, so every span is in
    the original line's offsets however much the line has changed length. A
    match that touches earlier markers swallows them: one span, of this kind,
    covering all of what they covered."""
    marker = b"[REDACTED:" + kind.encode("ascii") + b"]"
    out = []
    new_spans = []
    pos = 0       # everything before this in `line` is copied or replaced
    shift = 0     # (offset in the new line) - (offset in `line`), at `pos`
    i = 0         # the next earlier marker not yet carried over or swallowed
    for start, end in found:
        while i < len(spans) and spans[i][1] <= start:
            s = spans[i]
            new_spans.append((s[0] + shift, s[1] + shift, s[2], s[3], s[4]))
            i += 1
        # Past the last marker before `start`, an offset in `line` and one in
        # the original differ by a constant.
        delta = spans[i - 1][3] - spans[i - 1][1] if i else 0
        orig_start = start + delta
        orig_end = end + delta
        while i < len(spans) and spans[i][0] < end:
            s = spans[i]
            orig_start = min(orig_start, s[2])
            orig_end = max(end + (s[3] - s[1]), s[3])
            i += 1
        out.append(line[pos:start])
        out.append(marker)
        new_start = start + shift
        new_spans.append((new_start, new_start + len(marker), orig_start, orig_end, kind))
        shift += len(marker) - (end - start)
        pos = end
    for s in spans[i:]:
        new_spans.append((s[0] + shift, s[1] + shift, s[2], s[3], s[4]))
    out.append(line[pos:])
    return b"".join(out), new_spans


def _scan(line, counts=None):
    """(the line with every match replaced, the spans). One loop gives both,
    so what capture redacts and what the sweep finds cannot differ. The kinds
    run in table order and each sees the markers of the ones before it."""
    spans = []
    for kind, pattern in KINDS:
        found = _matches(kind, pattern, line)
        if not found:
            continue
        if counts is not None:
            counts[kind] = counts.get(kind, 0) + len(found)
        line, spans = _replace(line, spans, found, kind)
    return line, spans


def redact_line(line, counts):
    """One transcript line with every match replaced. `counts` (kind -> n) is
    updated in place."""
    return _scan(line, counts)[0]


def find_spans(line):
    """[(start, end, kind), ...] for every secret in `line`, in order, in the
    line's own byte offsets. start:end is the part redact_line replaces with
    a marker: the whole match, or its `secret` group, or an assignment's
    value. The sweep (sweep_secrets.py in hive-client) is built on this."""
    return [(s[2], s[3], s[4]) for s in _scan(line)[1]]


def redact(data):
    """(redacted bytes, counts). Line by line, so no match can span two
    records and swallow the record boundary between them."""
    counts = {}
    out = [redact_line(line, counts) for line in data.splitlines(keepends=True)]
    return b"".join(out), counts


def format_counts(counts):
    return " ".join(f"{kind}={counts[kind]}" for kind in sorted(counts))


def cut(transcript, start, out_file):
    """Write the redacted slice; return (end_offset, counts)."""
    with open(transcript, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        if size < start:
            start = 0
        f.seek(start)
        data = f.read(size - start)
    end = data.rfind(b"\n")
    if end < 0:
        return start, {}
    data = data[:end + 1]
    redacted, counts = redact(data)
    # Temp file and rename: a reader never sees a half-written slice, and a
    # failure part-way leaves no slice at all.
    tmp = out_file + ".tmp"
    try:
        with open(tmp, "wb") as o:
            o.write(redacted)
        os.replace(tmp, out_file)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return start + len(data), counts


def main(argv):
    if len(argv) != 5 or argv[1] != "cut":
        print("usage: hive_redact.py cut <transcript> <from_offset> <out_file>", file=sys.stderr)
        return 2
    try:
        start = int(argv[3])
    except ValueError:
        print(f"hive_redact: from_offset is not a number: {argv[3]!r}", file=sys.stderr)
        return 2
    end, counts = cut(argv[2], start, argv[4])
    sys.stdout.write(f"{end}\n{format_counts(counts)}\n")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv))
    except Exception as e:  # noqa: BLE001 -- any failure must be a non-zero exit, never a partial slice
        print(f"hive_redact: {type(e).__name__}: {e}", file=sys.stderr)
        sys.exit(1)
