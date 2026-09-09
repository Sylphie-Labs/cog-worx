# Shared helpers for the POSIX hook implementations. Sourced, never run.
#
# JSON parsing goes through Python 3 rather than jq: the hive-client README
# already requires Python 3 on every platform, while jq is not standard on a
# fresh macOS install. There is deliberately no fallback to a bare `python`:
# on distros where that is Python 2, json.load yields unicode objects and the
# parsed strings come back JSON-quoted, which breaks every path check silently.

_py() {
    python3 "$@"
}

# stop_hook_fields <<< "$json" -- prints three lines: stop_hook_active
# (true/false), session_id, transcript_path. One interpreter launch instead
# of three; this runs on the synchronous Stop path of every turn. Prints
# nothing and fails when the payload is not a JSON object.
stop_hook_fields() {
    _py -c '
import json, sys
try:
    d = json.load(sys.stdin)
except Exception:
    sys.exit(1)
if not isinstance(d, dict):
    sys.exit(1)
def s(v):
    return v if isinstance(v, str) else ""
print("true" if d.get("stop_hook_active") is True else "false")
print(s(d.get("session_id")))
print(s(d.get("transcript_path")))
'
}

# hive_log <file> <message> -- appends one timestamped line. Shared by hook
# mode and capture mode so every diagnostic has the same shape.
hive_log() {
    printf '[%s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$2" >>"$1"
}

# _fc_usable <path> -- true when <path> is an executable REGULAR file. `[ -x ]`
# alone is also true for a directory, which `command -v` would have skipped, so a
# directory named "claude" in an earlier PATH entry would beat the real binary
# later on PATH -- capture is then disabled for as long as that directory exists,
# while a working claude sits one PATH entry away.
#
# It is NOT true that this failure is invisible: capture logs
# "pass FAILED (claude exit 126)" with the path. An earlier comment said that
# AND that the offset never advances; only the first half was wrong. The offset
# really does advance only on rc=0 (hive-scribe-capture.sh:119), so nothing is
# lost and the next Stop retries the same slice. The reason to reject a
# directory is the masking, not a lost offset.
_fc_usable() {
    [ -f "$1" ] && [ -x "$1" ]
}

# usable_claude_bin <path> -- true when <path> is an ABSOLUTE path to an
# executable regular file: the contract find_claude returns, and the one any
# caller must hold an inherited CLAUDE_BIN to before exec'ing it.
usable_claude_bin() {
    case "$1" in
        /*) _fc_usable "$1" ;;
        *) return 1 ;;
    esac
}

# find_claude -- prints the ABSOLUTE path to the claude binary. Honors
# HIVE_CLAUDE_BIN when set (exclusively: an operator override, and what the
# test harness uses to simulate "not installed"). Otherwise PATH, then the
# known install locations, because PATH alone misses the alias-style local
# install (`claude migrate-installer` puts the binary at
# ~/.claude/local/claude and adds a shell alias that a non-interactive
# /bin/sh never sees).
#
# A relative PATH entry -- "." , "bin", or an EMPTY component, which POSIX
# defines as the current directory -- must never win: HOOK mode resolves the
# binary with cwd still set to the project directory, because nothing in hook.sh
# or the hook-mode branch cds at all, so a repo-checked-in "claude" would run
# with the user's credentials and the repo's hive-scribe key. Capture mode cds
# to $STATE_DIR/work first and is the safer of the two -- an earlier version of
# this comment credited capture mode's cd with CREATING the hazard, when it is
# what avoids it. That is the whole of the
# invariant here: components are trusted iff they start with "/". It does NOT
# make an absolute path safe -- npm, yarn and pnpm prepend node_modules/.bin
# as an ABSOLUTE path, so a dependency's bin shim still passes this test.
# See tik_01M221A74YJJ8HY8M973CB7GHB.
#
# This walks PATH itself rather than asking `command -v`, because the shells
# disagree about what `command -v` prints for a relative entry and so the
# answer cannot be inferred from the result's shape. Under bash in POSIX mode
# -- which is /bin/sh on macOS, and how the installed hook is launched -- it
# resolves the entry against the cwd and returns an ABSOLUTE path, so a
# leading-slash test accepts exactly the binary it was written to reject.
# dash and zsh return it relative. Inspecting the PATH component instead is
# the same answer on every shell. See tik_01M21Y44H5KTEQSNM60W61DNJD.
#
# The split is done with ${.%%:*} rather than `IFS=: ; for _d in $PATH`,
# because that loop is not portable in the direction this function needs:
# zsh does not field-split an unquoted expansion, so it would iterate ONCE
# with the whole PATH as a single word and silently skip the walk entirely.
# Unquoted $PATH is also glob-expanded, which would return a directory that
# is not on PATH at all. This form splits identically on sh, dash, bash and
# zsh, and needs no $IFS save/restore (which cannot restore "unset" anyway, and
# would leave the caller with field splitting disabled). It does still leave
# _fc_rest and _d set in the calling shell -- and _c when the fallback loop runs,
# though not on the PATH-hit path; what it no longer touches is a variable with
# shell SEMANTICS attached.
find_claude() {
    if [ -n "${HIVE_CLAUDE_BIN:-}" ]; then
        # Held to the same two rules as a PATH entry, because this is the one
        # lookup an operator sets by hand. A relative override is rejected, not
        # returned: this function's contract is an ABSOLUTE path, and
        # hive-scribe-capture.sh does `cd "$work_dir"` six lines before calling
        # it, so a relative value would resolve somewhere neither of us chose.
        usable_claude_bin "$HIVE_CLAUDE_BIN" || return 1
        printf '%s' "$HIVE_CLAUDE_BIN"
        return 0
    fi
    # ${PATH:-} and not $PATH: this function already uses ${HIVE_CLAUDE_BIN:-}
    # above, and a bare reference aborts a caller running under `set -u`.
    _fc_rest=${PATH:-}
    while [ -n "$_fc_rest" ]; do
        case "$_fc_rest" in
            *:*) _d=${_fc_rest%%:*}; _fc_rest=${_fc_rest#*:} ;;
            *)   _d=$_fc_rest;       _fc_rest= ;;
        esac
        # Only absolute components are trusted. An empty component (PATH=":/bin",
        # "/bin::/usr/bin") means the cwd and is skipped by the same test.
        case "$_d" in
            /*)
                if _fc_usable "$_d/claude"; then
                    printf '%s' "$_d/claude"
                    return 0
                fi
                ;;
        esac
    done
    # ${HOME:-} for the same reason as ${PATH:-} above: a bare $HOME killed a
    # `set -u` caller here, seventeen lines after the comment promising it
    # would not. With HOME unset these become "/.claude/local/claude", which
    # _fc_usable rejects like any other missing file.
    for _c in "${HOME:-}/.claude/local/claude" "${HOME:-}/.local/bin/claude" \
              /usr/local/bin/claude /opt/homebrew/bin/claude; do
        if _fc_usable "$_c"; then
            printf '%s' "$_c"
            return 0
        fi
    done
    return 1
}

# transcript_cut <transcript> <from_offset> <out_file> -- copies the bytes of
# <transcript> from <from_offset> up to (and including) the last newline in
# the file into <out_file>, so the cut never splits a JSONL record. Prints the
# end offset that was cut to (the next pass starts there). Prints <from_offset>
# unchanged and writes nothing when there is no complete new line. If the
# transcript has shrunk below <from_offset> (rewritten), starts over from 0.
transcript_cut() {
    _py -c '
import sys
tp, start, out = sys.argv[1], int(sys.argv[2]), sys.argv[3]
with open(tp, "rb") as f:
    f.seek(0, 2)
    size = f.tell()
    if size < start:
        start = 0
    f.seek(start)
    data = f.read(size - start)
end = data.rfind(b"\n")
if end < 0:
    sys.stdout.write(str(start))
    sys.exit(0)
data = data[:end + 1]
with open(out, "wb") as o:
    o.write(data)
sys.stdout.write(str(start + len(data)))
' "$1" "$2" "$3"
}

# pid_alive <pid> -- true when a process with that pid exists.
pid_alive() {
    [ -n "$1" ] && kill -0 "$1" 2>/dev/null
}

# read_offset <marker_file> -- prints the transcript byte offset recorded on
# the marker's first line, or 0 when the file is missing or not a number.
read_offset() {
    _off=$(sed -n 1p "$1" 2>/dev/null | tr -d ' \r\n')
    case "$_off" in ''|*[!0-9]*) _off=0 ;; esac
    printf '%s' "$_off"
}

# busy_is_live <busy_file> -- true when the capture pass that wrote this lock
# is still running: the pid on line 1 is alive AND the file is under an hour
# old. The age check guards against pid reuse after a hard reboot or kill,
# where a dead pass's number lands on some unrelated long-lived process.
busy_is_live() {
    _pid=$(sed -n 1p "$1" 2>/dev/null | tr -d ' \r\n')
    pid_alive "$_pid" || return 1
    [ -z "$(find "$1" -mmin +60 2>/dev/null)" ]
}

# spawn_detached <cmd...> -- runs the command in the background, detached
# from this hook: its own session (so a process-group TERM aimed at the
# hook's tree does not kill a pass mid-run), stdin from /dev/null (nohup only
# redirects a TTY, and the hook's stdin is Claude Code's JSON pipe), output
# discarded. python3 does the setsid so the behaviour is the same on macOS
# (no setsid binary) and Linux. Prints the child's pid.
spawn_detached() {
    nohup python3 -c 'import os, sys
os.setsid()
os.execvp(sys.argv[1], sys.argv[1:])' "$@" </dev/null >/dev/null 2>&1 &
    printf '%s' "$!"
}
