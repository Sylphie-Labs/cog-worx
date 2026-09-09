#!/bin/sh
# Cross-platform hook dispatcher.
#
# settings.json invokes every hook through this one entry point:
#     sh "$CLAUDE_PROJECT_DIR/.claude/hooks/hook.sh" <name> [args...]
#
# On Windows (Git Bash / MSYS, which Claude Code already relies on) it hands
# off to the PowerShell implementation. Everywhere else it runs the POSIX
# sibling. Arguments are forwarded verbatim on both branches; each script owns
# its own parameter contract.
#
# EXIT STATUS. The dispatcher is transparent: the implementation's status is
# returned unchanged. Connected repos route their own blocking guards through
# this entry point, and the status IS the contract -- Claude Code reads exit 2
# from a PreToolUse hook as "block this tool call" and from a Stop hook as
# "block this stop". A status swallowed here approves whatever the guard
# refused, while the reason text still reaches stderr, so it looks healthy in
# a log. That is how hive-mind's protect-starter, test-gate and fmt-clippy
# were all silently disabled (tik_01M23MKPY4127HTEMPXS09EF05).
#
# The ONE exception is hive-scribe-capture, which is always reported as 0.
# Capture is best-effort by design and logs its own failures under the
# per-session state dir, but it cannot contain a failure that happens before
# its first line runs: a missing _lib.sh or a CRLF checkout makes the shell
# itself exit 2 (dash) or 1 (bash), which would block every Stop until someone
# noticed. Only the caller can absorb that, so it lives here, keyed by name.
# Nothing else is contained -- a repo-local hook that wants best-effort
# semantics must exit 0 on its own. tests/hook_test.sh case G pins the
# exception and case K pins the rule.
#
# stdin carries the hook event JSON and is inherited by the implementation.
name=$1
[ -n "$name" ] || exit 0
shift

# finish <status>: exit with the implementation's status, or 0 for the one
# contained hook.
finish() {
    [ "$name" = "hive-scribe-capture" ] && exit 0
    exit "$1"
}

dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd) || exit 0

case "$(uname -s 2>/dev/null)" in
    MINGW* | MSYS* | CYGWIN*)
        ps1="$dir/$name.ps1"
        [ -f "$ps1" ] || exit 0
        # powershell.exe needs a Windows path; MSYS usually converts /c/... in
        # arguments automatically, but not when MSYS_NO_PATHCONV is set and
        # never under Cygwin. cygpath is present on all three.
        if command -v cygpath >/dev/null 2>&1; then
            ps1=$(cygpath -w "$ps1" 2>/dev/null) || ps1="$dir/$name.ps1"
        fi
        # -File returns the script's own `exit N` as powershell's status.
        powershell -NoProfile -ExecutionPolicy Bypass -File "$ps1" "$@"
        finish $?
        ;;
esac

impl="$dir/$name.sh"
[ -f "$impl" ] || exit 0
/bin/sh "$impl" "$@"
finish $?
