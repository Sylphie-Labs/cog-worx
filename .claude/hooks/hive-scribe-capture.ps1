# Stop hook: launches the hive-scribe capture session (see hive-scribe.prompt.md).
# Hook mode reads the Stop-hook JSON on stdin, debounces on transcript growth, claims a
# per-session lock, and spawns capture mode detached so the user's session never waits.
# Capture mode runs `claude -p` from a neutral working directory OUTSIDE the repo --
# project hooks therefore never fire for the scribe's own session, which is what
# prevents capture-of-the-capture recursion.
#
# Each pass hands the scribe only the transcript lines added since the previous
# successful pass (cut on a record boundary), never the whole transcript, so a long
# session that crosses the debounce threshold several times does not re-record the
# same decisions and tickets on every pass. The scribe merges each delta into the one
# session memory it upserts (see hive-scribe.prompt.md).
#
# Exactly-once rules, so no slice of transcript is ever skipped:
#   - the offset advances only after `claude` exits 0 for that slice; a failed pass
#     leaves it alone and the same range is re-cut on the next Stop;
#   - capture mode keeps looping while the transcript has grown by another debounce
#     worth since its last pass, so a Stop that arrived while a pass was running (and
#     was therefore turned away) is still picked up by that pass;
#   - a lock file with the running pass's pid serialises passes per session.
#
# State, per session id, under $stateDir:
#   <sid>.last         byte offset of the transcript captured so far (one line)
#   <sid>.busy         pid of the capture pass currently running
#   <sid>.delta.jsonl  the slice handed to the running pass; removed when it ends
#   <sid>.log          combined output of every pass
#
# TEMPLATE: this file is installed by init_repo.py into a target repo's
# .claude/hooks/. The slug placeholder below is replaced by init_repo.py at
# install time with the target hive project's slug. The capture prompt is read
# from the installed repo itself (.claude/hooks/hive-scribe.prompt.md, next to where this
# script lives once installed).
param(
    [switch]$Capture,
    [string]$SessionId,
    [string]$TranscriptPath
)

$projectSlug = 'cog-worx'   # replaced by init_repo.py at install time
$stateDir = Join-Path $env:LOCALAPPDATA 'hive-scribe'
$debounce = 30000        # bytes of new transcript before another pass is worth running
$maxPassesPerRun = 8     # bound on the catch-up loop inside one capture process

# Offset recorded on the marker's first line, or 0 when missing / not a number.
function Read-Offset([string]$marker) {
    if (Test-Path $marker) {
        $line = Get-Content $marker -TotalCount 1
        if ($line -match '^\d+$') { return [int64]$line }
    }
    return [int64]0
}

# Copies bytes [$start, end) of $tp into $out, where end is just past the last
# newline in the file, so the cut never splits a JSONL record. Returns the end
# offset; returns $start (and writes nothing) when there is no complete new line.
# A transcript shorter than $start was rewritten: starts over from 0.
function Cut-Transcript([string]$tp, [int64]$start, [string]$out) {
    $fs = [System.IO.File]::Open($tp, 'Open', 'Read', 'ReadWrite')
    try {
        $size = $fs.Length
        if ($size -lt $start) { $start = 0 }
        $len = [int]($size - $start)
        if ($len -le 0) { return $start }
        $null = $fs.Seek($start, 'Begin')
        $buf = New-Object byte[] $len
        $n = 0
        while ($n -lt $len) {
            $r = $fs.Read($buf, $n, $len - $n)
            if ($r -le 0) { break }
            $n += $r
        }
    } finally {
        $fs.Dispose()
    }
    if ($n -le 0) { return $start }
    $end = [Array]::LastIndexOf($buf, [byte]10, $n - 1)
    if ($end -lt 0) { return $start }
    $o = [System.IO.File]::Open($out, 'Create', 'Write', 'None')
    try { $o.Write($buf, 0, $end + 1) } finally { $o.Dispose() }
    return [int64]($start + $end + 1)
}

# True when the pass that wrote this lock is still running: pid alive AND the
# file under an hour old (guards against pid reuse after a reboot or hard kill).
function Test-BusyLive([string]$busy) {
    $line = Get-Content $busy -TotalCount 1
    if ($line -notmatch '^\d+$') { return $false }
    if (-not (Get-Process -Id ([int]$line) -ErrorAction SilentlyContinue)) { return $false }
    return ((Get-Item $busy).LastWriteTime -gt (Get-Date).AddHours(-1))
}

if ($Capture) {
    # ---- capture mode: the scribe session itself ----
    if (-not $SessionId -or $SessionId -notmatch '^[0-9A-Za-z_-]+$') { exit 0 }

    if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Force $stateDir | Out-Null }
    $logFile = Join-Path $stateDir ($SessionId + '.log')
    $busy = Join-Path $stateDir ($SessionId + '.busy')
    $marker = Join-Path $stateDir ($SessionId + '.last')
    $delta = Join-Path $stateDir ($SessionId + '.delta.jsonl')
    function Write-Log([string]$msg) { Add-Content -Path $logFile -Value ('[{0}] {1}' -f (Get-Date -Format 'o'), $msg) }

    try {
        $env:HIVE_SCRIBE = '1'
        $workDir = Join-Path $stateDir 'work'
        if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Force $workDir | Out-Null }
        Set-Location $workDir
        $promptPath = Join-Path $PSScriptRoot 'hive-scribe.prompt.md'
        $agentDef = Get-Content $promptPath -Raw -ErrorAction SilentlyContinue
        if (-not $agentDef) {
            # The POSIX sibling logs this; without it a Windows box captures nothing
            # and leaves no trace of why. Newly reachable until every repo has been
            # re-run through the installer after the prompt moved out of agents/.
            Write-Log "capture prompt missing or empty at $promptPath"
        }
        if ($agentDef) {
            for ($n = 1; $n -le $maxPassesPerRun; $n++) {
                if (-not (Test-Path $TranscriptPath)) { break }
                [int64]$last = Read-Offset $marker
                [int64]$size = (Get-Item $TranscriptPath).Length
                if ($size -lt $last) { $last = 0 }   # transcript rewritten: start over
                # The first pass of this run was admitted by the hook's debounce; later
                # ones (catching up on Stops that arrived mid-pass) must earn it again.
                if ($n -gt 1 -and ($size - $last) -lt $debounce) { break }
                [int64]$end = Cut-Transcript $TranscriptPath $last $delta
                if ($end -le $last) { break }
                if ($last -gt 0) {
                    $scope = 'Continuation pass: the file holds only the transcript lines added since the previous pass, which already captured everything before them.'
                } else {
                    $scope = 'First pass: the file holds the whole transcript so far.'
                }
                $prompt = "Capture this session into the hive. Session id: $SessionId. Transcript file to read: $delta. $scope The session memory's external_ref is session:$SessionId and its tag is session-$SessionId. This session belongs to hive project '$projectSlug'. Pass project: '$projectSlug' on every memory_write, ticket_create, decision_record, ticket_list, decision_list and memory_query call."
                Write-Log "pass start: bytes [$last,$end)"
                & claude -p $prompt `
                    --append-system-prompt $agentDef `
                    --model sonnet `
                    --max-turns 30 `
                    --allowedTools 'Read,mcp__hive-scribe__decision_record,mcp__hive-scribe__memory_write,mcp__hive-scribe__memory_query,mcp__hive-scribe__ticket_create,mcp__hive-scribe__ticket_list,mcp__hive-scribe__decision_list,mcp__hive-scribe__activity_list,mcp__hive-scribe__activity_summary' *>> $logFile
                if ($LASTEXITCODE -eq 0) {
                    Set-Content -Path $marker -Value $end -Encoding ascii
                    Write-Log "pass ok: offset now $end"
                } else {
                    Write-Log "pass FAILED (claude exit $LASTEXITCODE): offset stays at $last; bytes [$last,$end) will be retried on the next Stop"
                    break
                }
            }
        }
    } catch {
        Write-Log $_.Exception.ToString()
    } finally {
        Remove-Item -Path $busy -ErrorAction SilentlyContinue
        Remove-Item -Path $delta -ErrorAction SilentlyContinue
    }
    exit 0
}

# ---- hook mode: must always exit 0 quickly and never block the session ----
try {
    if ($env:HIVE_SCRIBE -eq '1') { exit 0 }

    $stdin = [Console]::In.ReadToEnd()
    $in = $stdin | ConvertFrom-Json
    if ($in.stop_hook_active) { exit 0 }
    $sid = $in.session_id
    $tp = $in.transcript_path
    if (-not $sid -or -not $tp) { exit 0 }
    if ($sid -notmatch '^[0-9A-Za-z_-]+$') { exit 0 }
    if (-not (Test-Path $tp)) { exit 0 }

    if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Force $stateDir | Out-Null }

    # Debounce: only capture when the transcript has grown >= $debounce since the
    # last successful pass. A missing marker (first pass) always qualifies.
    $marker = Join-Path $stateDir ($sid + '.last')
    $hasMarker = Test-Path $marker
    [int64]$last = Read-Offset $marker
    [int64]$size = (Get-Item $tp).Length
    if ($size -lt $last) { $last = 0 }   # transcript rewritten: start over
    if ($hasMarker -and (($size - $last) -lt $debounce)) { exit 0 }

    # One pass at a time per session. A running pass catches up on its own (see
    # capture mode), so this Stop can simply be turned away.
    $busy = Join-Path $stateDir ($sid + '.busy')
    if (Test-Path $busy) {
        if (Test-BusyLive $busy) { exit 0 }
        Remove-Item -Path $busy -ErrorAction SilentlyContinue
    }
    Set-Content -Path $busy -Value $PID -Encoding ascii   # claim before spawning so a racing Stop backs off

    $self = $MyInvocation.MyCommand.Path
    $child = Start-Process powershell -WindowStyle Hidden -PassThru -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $self),
        '-Capture', '-SessionId', $sid, '-TranscriptPath', ('"{0}"' -f $tp)
    )
    if ($child) { Set-Content -Path $busy -Value $child.Id -Encoding ascii }   # hand the lock to the pass that owns it
} catch {
    # Capture is best-effort; a broken hook must never block the user's session.
}
exit 0
