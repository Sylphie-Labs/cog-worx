# Stop hook: launches the hive-scribe capture session (see .claude/agents/hive-scribe.md).
# Hook mode reads the Stop-hook JSON on stdin, debounces on transcript growth, and spawns
# capture mode detached so the user's session never waits. Capture mode runs `claude -p`
# from a neutral working directory OUTSIDE the repo — project hooks therefore never fire
# for the scribe's own session, which is what prevents capture-of-the-capture recursion.
#
# TEMPLATE: this file is installed by onboard/install.ps1 into a target repo's
# .claude/hooks/. The slug placeholder below is replaced by onboard/install.ps1 at
# install time with the target hive project's slug. The agent definition is read
# from the installed repo itself (.claude/agents/hive-scribe.md, next to where this
# script lives once installed).
param(
    [switch]$Capture,
    [string]$SessionId,
    [string]$TranscriptPath
)

$projectSlug = 'cog-worx'   # replaced by onboard/install.ps1 at install time
$stateDir = Join-Path $env:LOCALAPPDATA 'hive-scribe'

if ($Capture) {
    # ---- capture mode: the scribe session itself ----
    if (-not $SessionId -or $SessionId -notmatch '^[0-9A-Za-z_-]+$') { exit 0 }

    if (-not (Test-Path $stateDir)) { New-Item -ItemType Directory -Force $stateDir | Out-Null }
    $logFile = Join-Path $stateDir ($SessionId + '.log')

    try {
        $env:HIVE_SCRIBE = '1'
        $workDir = Join-Path $stateDir 'work'
        if (-not (Test-Path $workDir)) { New-Item -ItemType Directory -Force $workDir | Out-Null }
        Set-Location $workDir
        $agentDef = Get-Content (Join-Path (Split-Path $PSScriptRoot) 'agents\hive-scribe.md') -Raw
        $prompt = "Capture this session into the hive. Session id: $SessionId. Transcript file to read: $TranscriptPath. Follow your instructions; the session memory external_ref must be exactly session:$SessionId. This session belongs to hive project '$projectSlug'. Pass project: '$projectSlug' on every memory_write, ticket_create, decision_record, ticket_list, decision_list and memory_query call."
        & claude -p $prompt `
            --append-system-prompt $agentDef `
            --model sonnet `
            --max-turns 30 `
            --allowedTools 'Read,mcp__hive-scribe__*' *>> $logFile
    } catch {
        Add-Content -Path $logFile -Value ('[{0}] {1}' -f (Get-Date -Format 'o'), $_.Exception.ToString())
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

    # Debounce: only capture when the transcript has grown >= 30KB since last capture.
    $size = (Get-Item $tp).Length
    $marker = Join-Path $stateDir ($sid + '.last')
    if (Test-Path $marker) {
        $last = [int64](Get-Content $marker -TotalCount 1)
        if (($size - $last) -lt 30000) { exit 0 }
    }
    Set-Content -Path $marker -Value $size -Encoding ascii

    $self = $MyInvocation.MyCommand.Path
    Start-Process powershell -WindowStyle Hidden -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"{0}"' -f $self),
        '-Capture', '-SessionId', $sid, '-TranscriptPath', ('"{0}"' -f $tp)
    )
} catch {
    # Capture is best-effort; a broken hook must never block the user's session.
}
exit 0
