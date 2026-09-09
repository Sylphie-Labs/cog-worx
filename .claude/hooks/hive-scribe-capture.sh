#!/bin/sh
# Stop hook: launches the hive-scribe capture session (hive-scribe.prompt.md).
# POSIX sibling of hive-scribe-capture.ps1. Hook mode reads the Stop-hook JSON on
# stdin, debounces on transcript growth, claims a per-session lock, and spawns
# capture mode detached so the user's session never waits. Capture mode runs
# `claude -p` from a neutral working directory OUTSIDE the repo -- project hooks
# therefore never fire for the scribe's own session, which is what prevents
# capture-of-the-capture recursion.
#
# Each pass hands the scribe only the transcript lines added since the previous
# successful pass (cut on a record boundary), never the whole transcript, so a
# long session that crosses the debounce threshold several times does not
# re-record the same decisions and tickets on every pass. The scribe merges each
# delta into the one session memory it upserts (see hive-scribe.prompt.md).
#
# Exactly-once rules, so no slice of transcript is ever skipped:
#   - the offset advances only after `claude` exits 0 for that slice; a failed
#     pass leaves it alone and the same range is re-cut on the next Stop;
#   - capture mode keeps looping while the transcript has grown by another
#     debounce worth since its last pass, so a Stop that arrived while a pass was
#     running (and was therefore turned away) is still picked up by that pass;
#   - a lock file with the running pass's pid serialises passes per session.
#
# State, per session id, under $STATE_DIR:
#   <sid>.last         byte offset of the transcript captured so far (one line)
#   <sid>.busy         pid of the capture pass currently running
#   <sid>.delta.jsonl  the slice handed to the running pass; removed when it ends
#   <sid>.log          combined output of every pass
#
# TEMPLATE: this file is installed by init_repo.py into a target repo's
# .claude/hooks/. The slug placeholder below is replaced by init_repo.py at
# install time with the target hive project's slug. The capture prompt is read
# from the installed repo itself (.claude/hooks/hive-scribe.prompt.md, next to where
# this script lives once installed).
#
# Usage:  hive-scribe-capture.sh                                 (hook mode)
#         hive-scribe-capture.sh __capture <sid> <transcript>    (internal)
#
# Environment from hook mode to capture mode: CLAUDE_BIN (resolved binary),
# re-validated on arrival because the variable may equally come from the
# ambient environment. HIVE_CLAUDE_BIN, if set, overrides every find_claude
# lookup; it must be an absolute path to an executable file, and a value that
# is not is reported rather than silently ignored. Note it is NOT consulted
# when CLAUDE_BIN is already set -- capture mode then never calls find_claude.
#
# `here` is the logical path of this script's directory (pwd keeps symlinks),
# symlink into a dotfiles repo; `$here/..` would resolve physically and miss.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/_lib.sh"

PROJECT_SLUG='cog-worx'   # replaced by init_repo.py at install time
STATE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hive-scribe"
DEBOUNCE=30000        # bytes of new transcript before another pass is worth running
MAX_PASSES_PER_RUN=8  # bound on the catch-up loop inside one capture process

# log_setup_once <sid> <message> -- a setup diagnostic that must appear exactly
# once per session. Deduped through a MARKER FILE, not by grepping the session
# log: that log is also where every scribe pass's stdout and stderr is appended,
# so a key matched against its contents can be satisfied by a pass that merely
# MENTIONED the phrase -- and the scribe summarises sessions working on this
# very repo. The one diagnostic that exists to make a "connected but silent"
# repo diagnosable would then be the one thing suppressed.
log_setup_once() {
    _lso_marker="$STATE_DIR/$1.setup-logged"
    [ -f "$_lso_marker" ] && return 0
    hive_log "$STATE_DIR/$1.log" "$2"
    : > "$_lso_marker" 2>/dev/null || :
}

if [ "$1" = "__capture" ]; then
    # ---- capture mode: the scribe session itself ----
    sid=$2
    tp=$3
    HIVE_SCRIBE=1
    export HIVE_SCRIBE
    busy="$STATE_DIR/$sid.busy"
    marker="$STATE_DIR/$sid.last"
    delta="$STATE_DIR/$sid.delta.jsonl"
    log_file="$STATE_DIR/$sid.log"
    trap 'rm -f "$busy" "$delta"' EXIT
    log() { hive_log "$log_file" "$1"; }
    work_dir="$STATE_DIR/work"
    mkdir -p "$work_dir" || exit 0
    cd "$work_dir" || exit 0
    # The capture prompt sits in this hook's own directory, the same way
    # the PowerShell variant finds it, so no project-dir argument is needed.
    agent_path="$here/hive-scribe.prompt.md"
    agent_def=$(cat "$agent_path" 2>/dev/null) || { log "capture prompt missing at $agent_path"; exit 0; }
    [ -n "$agent_def" ] || { log "capture prompt is empty at $agent_path"; exit 0; }
    # An inherited CLAUDE_BIN is RE-VALIDATED, not trusted. Hook mode exports one
    # it resolved through find_claude, so through the shipped flow this value is
    # always already checked -- but capture mode re-enters as a fresh /bin/sh and
    # cannot tell that apart from a value the ambient environment supplied, and
    # `__capture` is directly invocable. Defence in depth rather than a live hole.
    #
    # A relative value is rejected even though the `cd "$work_dir"` above has
    # moved OUT of the repo: it would resolve under $STATE_DIR/work, which is
    # not the repo but is not a location anyone chose either. The repo-cwd
    # exposure is hook mode's, not this branch's -- see find_claude's own
    # comment. `${CLAUDE_BIN:-}` for symmetry with the guarded expansions in
    # _lib.sh, not because any template sets `set -u` today.
    if [ -n "${CLAUDE_BIN:-}" ]; then
        usable_claude_bin "$CLAUDE_BIN" || {
            log_setup_once "$sid" "CLAUDE_BIN is set but is not an absolute path to an executable file: $CLAUDE_BIN; capture skipped"
            exit 0
        }
    else
        CLAUDE_BIN=$(find_claude) || { log "claude binary not found"; exit 0; }
    fi

    n=0
    while [ "$n" -lt "$MAX_PASSES_PER_RUN" ]; do
        n=$((n + 1))
        [ -f "$tp" ] || break
        last=$(read_offset "$marker")
        size=$(wc -c < "$tp" | tr -d ' ')
        [ "$size" -lt "$last" ] && last=0   # transcript rewritten: start over
        # The first pass of this run was admitted by the hook's debounce; later
        # ones (catching up on Stops that arrived mid-pass) must earn it again.
        if [ "$n" -gt 1 ] && [ $((size - last)) -lt "$DEBOUNCE" ]; then break; fi
        end=$(transcript_cut "$tp" "$last" "$delta") || break
        case "$end" in ''|*[!0-9]*) break ;; esac
        [ "$end" -gt "$last" ] && [ -s "$delta" ] || break
        if [ "$last" -gt 0 ]; then
            scope="Continuation pass: the file holds only the transcript lines added since the previous pass, which already captured everything before them."
        else
            scope="First pass: the file holds the whole transcript so far."
        fi
        prompt="Capture this session into the hive. Session id: $sid. Transcript file to read: $delta. $scope The session memory's external_ref is session:$sid and its tag is session-$sid. This session belongs to hive project '$PROJECT_SLUG'. Pass project: '$PROJECT_SLUG' on every memory_write, ticket_create, decision_record, ticket_list, decision_list and memory_query call."
        log "pass start: bytes [$last,$end)"
        "$CLAUDE_BIN" -p "$prompt" \
            --append-system-prompt "$agent_def" \
            --model sonnet \
            --max-turns 30 \
            --allowedTools 'Read,mcp__hive-scribe__decision_record,mcp__hive-scribe__memory_write,mcp__hive-scribe__memory_query,mcp__hive-scribe__ticket_create,mcp__hive-scribe__ticket_list,mcp__hive-scribe__decision_list,mcp__hive-scribe__activity_list,mcp__hive-scribe__activity_summary' >>"$log_file" 2>&1
        rc=$?
        if [ "$rc" -eq 0 ]; then
            printf '%s\n' "$end" > "$marker"
            log "pass ok: offset now $end"
        else
            log "pass FAILED (claude exit $rc): offset stays at $last; bytes [$last,$end) will be retried on the next Stop"
            break
        fi
    done
    exit 0
fi

# ---- hook mode: must always exit 0 quickly and never block the session ----
# (hook.sh also guarantees exit 0 around this whole script; the checks below
# keep the common path cheap and the failure paths logged.)
[ "$HIVE_SCRIBE" = "1" ] && exit 0
# python3 is the one hard dependency. Its absence (or a stub that fails, like
# the macOS CLT placeholder before the tools are installed) is the one setup
# fault that would otherwise leave no trace at all, so it gets a log line of
# its own, written once, before anything that needs python3 to parse.
if ! python3 -c 'import json' >/dev/null 2>&1; then
    mkdir -p "$STATE_DIR" 2>/dev/null || exit 0
    grep -qsF "python3 not usable" "$STATE_DIR/hook-errors.log" \
        || hive_log "$STATE_DIR/hook-errors.log" "python3 not usable on this machine; hive-scribe capture is disabled until it is installed"
    exit 0
fi

stdin=$(cat)
[ -n "$stdin" ] || exit 0
fields=$(printf '%s' "$stdin" | stop_hook_fields) || exit 0
active=$(printf '%s\n' "$fields" | sed -n 1p)
sid=$(printf '%s\n' "$fields" | sed -n 2p)
tp=$(printf '%s\n' "$fields" | sed -n 3p)
[ "$active" = "true" ] && exit 0
[ -n "$sid" ] && [ -n "$tp" ] || exit 0
case "$sid" in *[!0-9A-Za-z_-]*) exit 0 ;; esac
[ -f "$tp" ] || exit 0

mkdir -p "$STATE_DIR" || exit 0


# Debounce: only capture when the transcript has grown >= DEBOUNCE since the
# last successful pass. A missing marker (first pass) always qualifies.
marker="$STATE_DIR/$sid.last"
last=$(read_offset "$marker")
size=$(wc -c < "$tp" | tr -d ' ')
[ "$size" -lt "$last" ] && last=0   # transcript rewritten: start over
if [ -f "$marker" ] && [ $((size - last)) -lt "$DEBOUNCE" ]; then exit 0; fi

# One pass at a time per session. A running pass catches up on its own (see
# capture mode), so this Stop can simply be turned away.
busy="$STATE_DIR/$sid.busy"
if [ -f "$busy" ]; then
    if busy_is_live "$busy"; then exit 0; fi
    rm -f "$busy"
fi
# A missing claude binary is the one setup failure that used to leave no
# trace at all; log it once per session so a "connected but silent" repo can
# be diagnosed from the state dir. Resolved here, after the debounce and lock
# checks, because it is only needed at spawn time.
CLAUDE_BIN=$(find_claude) || {
    # Distinguish "nothing found" from "your override was rejected". Reporting
    # the second as the first told an operator who had just set
    # HIVE_CLAUDE_BIN to a path that plainly exists to set HIVE_CLAUDE_BIN --
    # and this line is deduped per session, so there was no other signal.
    if [ -n "${HIVE_CLAUDE_BIN:-}" ]; then
        why="HIVE_CLAUDE_BIN is set but is not an absolute path to an executable file: $HIVE_CLAUDE_BIN; capture skipped"
    else
        why="claude binary not found on PATH or in the known install locations (set HIVE_CLAUDE_BIN to override); capture skipped"
    fi
    log_setup_once "$sid" "$why"
    exit 0
}
export CLAUDE_BIN

printf '%s\n' "$$" > "$busy"   # claim before spawning so a racing Stop backs off

self="$here/$(basename -- "$0")"
child=$(spawn_detached /bin/sh "$self" __capture "$sid" "$tp")
printf '%s\n' "$child" > "$busy"   # hand the lock to the pass that now owns it
exit 0
