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
# This dispatcher ALWAYS exits 0. A Stop hook that exits 2 makes Claude Code
# block the stop and feed stderr back to Claude on every turn, and a shell
# error (missing helper file, CRLF checkout, bad arithmetic) would otherwise
# surface exactly that way. Capture is best-effort by design; the
# implementation logs its own failures under the per-session state dir.
#
# stdin carries the hook event JSON and is inherited by the implementation.
name=$1
[ -n "$name" ] || exit 0
shift

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
        powershell -NoProfile -ExecutionPolicy Bypass -File "$ps1" "$@"
        exit 0
        ;;
esac

impl="$dir/$name.sh"
[ -f "$impl" ] || exit 0
/bin/sh "$impl" "$@"
exit 0
