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

# find_claude -- prints the ABSOLUTE path to the claude binary. Honors
# HIVE_CLAUDE_BIN when set (exclusively: an operator override, and what the
# test harness uses to simulate "not installed"). Otherwise PATH, then the
# known install locations, because `command -v` alone misses the alias-style
# local install (`claude migrate-installer` puts the binary at
# ~/.claude/local/claude and adds a shell alias that a non-interactive
# /bin/sh never sees). A relative PATH entry (node_modules/.bin, ".") makes
# `command -v` return a relative path under dash; that is rejected rather
# than resolved, since capture mode changes directory before exec and a
# repo-checked-in "claude" must never run with the user's credentials.
find_claude() {
    if [ -n "${HIVE_CLAUDE_BIN:-}" ]; then
        [ -x "$HIVE_CLAUDE_BIN" ] || return 1
        printf '%s' "$HIVE_CLAUDE_BIN"
        return 0
    fi
    _c=$(command -v claude 2>/dev/null)
    case "$_c" in
        /*) if [ -x "$_c" ]; then printf '%s' "$_c"; return 0; fi ;;
    esac
    for _c in "$HOME/.claude/local/claude" "$HOME/.local/bin/claude" \
              /usr/local/bin/claude /opt/homebrew/bin/claude; do
        if [ -x "$_c" ]; then
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
