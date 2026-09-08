#!/bin/sh
# Stop hook: launches the hive-scribe capture session (.claude/agents/hive-scribe.md).
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
# delta into the one session memory it upserts (see hive-scribe.md).
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
# install time with the target hive project's slug. The agent definition is read
# from the installed repo itself (.claude/agents/hive-scribe.md, next to where
# this script lives once installed).
#
# Usage:  hive-scribe-capture.sh                                 (hook mode)
#         hive-scribe-capture.sh __capture <sid> <transcript>    (internal)
#
# Environment from hook mode to capture mode: CLAUDE_BIN (resolved binary).
# HIVE_CLAUDE_BIN, if set, overrides the binary lookup everywhere.
#
# `here` is the logical path of this script's directory (pwd keeps symlinks),
# so `${here%/*}` is the .claude directory even when .claude/hooks is a
# symlink into a dotfiles repo; `$here/..` would resolve physically and miss.
here=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
. "$here/_lib.sh"

PROJECT_SLUG='cog-worx'   # replaced by init_repo.py at install time
STATE_DIR="${XDG_CACHE_HOME:-$HOME/.cache}/hive-scribe"
DEBOUNCE=30000        # bytes of new transcript before another pass is worth running
MAX_PASSES_PER_RUN=8  # bound on the catch-up loop inside one capture process

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
    # The agent definition lives next to this hook's directory, the same way
    # the PowerShell variant finds it, so no project-dir argument is needed.
    agent_path="${here%/*}/agents/hive-scribe.md"
    agent_def=$(cat "$agent_path" 2>/dev/null) || { log "agent definition missing at $agent_path"; exit 0; }
    [ -n "$agent_def" ] || { log "agent definition is empty at $agent_path"; exit 0; }
    [ -n "$CLAUDE_BIN" ] || CLAUDE_BIN=$(find_claude) || { log "claude binary not found"; exit 0; }

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
            --allowedTools 'Read,mcp__hive-scribe__*' >>"$log_file" 2>&1
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
    grep -qsF "claude binary not found" "$STATE_DIR/$sid.log" \
        || hive_log "$STATE_DIR/$sid.log" "claude binary not found on PATH or in the known install locations (set HIVE_CLAUDE_BIN to override); capture skipped"
    exit 0
}
export CLAUDE_BIN

printf '%s\n' "$$" > "$busy"   # claim before spawning so a racing Stop backs off

self="$here/$(basename -- "$0")"
child=$(spawn_detached /bin/sh "$self" __capture "$sid" "$tp")
printf '%s\n' "$child" > "$busy"   # hand the lock to the pass that now owns it
exit 0
