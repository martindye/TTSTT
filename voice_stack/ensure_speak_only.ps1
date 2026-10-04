# ensure_speak_only.ps1 — idempotent "make the TTS-only speak voice running"
# for a workspace. The TTS-only sibling of ensure_coding_voice.ps1:
# pocket TTS on the CPU, NO microphone, NO STT — zero GPU, so it coexists
# with the 350k-context LLM. Run it and do nothing else; it handles every
# starting state:
#
#   - already running, right workspace  -> reports it, changes nothing
#   - stale / dead / wrong workspace    -> stops it, starts fresh
#   - not running at all                -> starts it
# and it exits 0 only when the bridge is actually ready (or already was).
#
#   powershell -NoProfile -File voice_stack\ensure_speak_only.ps1 `
#       -Workspace 'C:\Users\press\OneDrive\Projects\DSH_TESTS'
#
#   [OK] exit 0 — the voice speaks now.
#   [ER] exit 1 — not running; reason + log tails are printed. Fix, re-run.
#
# Extra arguments are passed straight to the bridge, e.g.:
#   -Workspace 'C:\...' --tts-voice vera
#
# Note: the workspace is where the SESSION lives (the chat's workspace, i.e.
# the coding agent's working directory), not where this script lives.

param(
    [Parameter(Mandatory = $true, Position = 0)]
    [string]$Workspace,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$ExtraArgs
)

$ErrorActionPreference = "Stop"

# DSH home: $DSH_HOME or <user profile>\.dsh (portable across machines).
$DshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $env:USERPROFILE ".dsh" }
$LogDir  = Join-Path $DshHome "logs"
$LogFile = Join-Path $LogDir "speak_only.log"
$ErrLog  = Join-Path $LogDir "speak_only.err.log"
$SupLog  = Join-Path $LogDir "speak_only_supervisor.log"
$StateFile = Join-Path $LogDir "speak_only.state"
$StopFlag  = Join-Path $LogDir "speak_only.STOP"
$SupScript = Join-Path $PSScriptRoot "supervise_speak_only.ps1"

function Quote([string]$s) {
    if ($s -match '\s') { "`"$s`"" } else { $s }
}

function Read-State() {
    if (-not (Test-Path $StateFile)) { return $null }
    try { return (Get-Content $StateFile -Raw | ConvertFrom-Json) } catch { return $null }
}

function Say($msg) { Write-Output ("{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $msg) }

function Fail([string]$reason) {
    Say "FAILED: $reason"
    Say "--- speak_only.log (tail) ---"
    if (Test-Path $LogFile) { Get-Content $LogFile -Tail 25 }
    if (Test-Path $ErrLog) {
        $err = Get-Content $ErrLog -Tail 15
        if ($err) { Say "--- speak_only.err.log (tail) ---"; $err | ForEach-Object { Say $_ } }
    }
    if (Test-Path $SupLog) {
        Say "--- supervisor log (tail) ---"
        (Get-Content $SupLog -Tail 8) | ForEach-Object { Say $_ }
    }
    exit 1
}

# ------------------------------------------------------------- current
$wsNorm = $Workspace.TrimEnd('\', '/').ToLower()
$state = Read-State
$bridgePid = 0; $supPid = 0; $stateWs = ""; $stateArgs = ""
if ($state) {
    $bridgePid = [int]$state.bridge_pid
    $supPid    = [int]$state.supervisor_pid
    $stateWs   = ([string]$state.workspace).TrimEnd('\', '/').ToLower()
    # Old state files have no args field: that reads as "".
    $stateArgs = [string]$state.args
}
$bridgeAlive = ($bridgePid -gt 0) -and (Get-Process -Id $bridgePid -ErrorAction SilentlyContinue)
$supAlive    = ($supPid -gt 0) -and (Get-Process -Id $supPid -ErrorAction SilentlyContinue)
$logFresh = (Test-Path $LogFile) -and
    (((Get-Item $LogFile).LastWriteTime -gt (Get-Date).AddSeconds(-90)))

# "Already right" means same workspace AND same bridge extras (engine/voice).
$extraNorm = ((@($ExtraArgs | Where-Object { $_ }) -join " ").Trim()).ToLower()
$argsMatch = ($stateArgs -eq $extraNorm)
if ($bridgeAlive -and $supAlive -and $logFresh -and
        ($stateWs -ceq $wsNorm) -and $argsMatch) {
    $cur = ""
    $tail = Get-Content $LogFile -Tail 400 -ErrorAction SilentlyContinue
    $m = [regex]::Matches((($tail | Out-String) -replace "`r", ""),
                         'speaking into (?:session )?(session-[0-9a-f-]+)')
    if ($m.Count -gt 0) { $cur = $m[$m.Count - 1].Groups[1].Value }
    Say "ALREADY RUNNING - speaking into $cur (workspace: $wsNorm). Nothing to do."
    exit 0
}

# ---------------------------------------------------------------- stop
Say "stopping any existing speak-only stack (workspace: $wsNorm)"
New-Item -ItemType File -Force -Path $StopFlag | Out-Null   # sentinel FIRST
if ($bridgeAlive) {
    Stop-Process -Id $bridgePid -Force -ErrorAction SilentlyContinue
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Process -Id $bridgePid -ErrorAction SilentlyContinue) -and
           (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
    if (Get-Process -Id $bridgePid -ErrorAction SilentlyContinue) {
        Fail "old bridge (pid $bridgePid) would not die - not starting a second one"
    }
}
if ($supAlive) {
    $deadline = (Get-Date).AddSeconds(15)
    while ((Get-Process -Id $supPid -ErrorAction SilentlyContinue) -and
           (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
    if (Get-Process -Id $supPid -ErrorAction SilentlyContinue) {
        Stop-Process -Id $supPid -Force -ErrorAction SilentlyContinue
    }
}
Remove-Item $StateFile -Force -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- start
if (-not (Test-Path $SupScript)) { Fail "supervisor script missing: $SupScript" }
Say "starting speak-only supervisor (workspace: $wsNorm)"
# The supervisor prepends `-X utf8 -m voice_stack.speak_only` itself; pass
# ONLY the bridge options here (passing -m again makes argparse die).
# Launch from a copy in %TEMP%: under the current WDAC/Intune policy,
# `powershell -File` on a script under OneDrive hangs forever (03/10/2026:
# the process stays alive but never executes a line). A copy outside
# OneDrive starts normally; it gets the real root via -ProjectRoot.
$projectRoot = Split-Path $PSScriptRoot -Parent
$supCopy = Join-Path $env:TEMP "supervise_speak_only.ps1"
Copy-Item -Path $SupScript -Destination $supCopy -Force
$extra = @($ExtraArgs | Where-Object { $_ })   # $null when no extras given
$bridgeArgs = @("--workspace", $Workspace) + $extra
$supArgs = @("-NoProfile", "-File", (Quote $supCopy), "-ProjectRoot", (Quote $projectRoot)) +
    @($bridgeArgs | ForEach-Object { Quote $_ })
Start-Process -FilePath "powershell" -ArgumentList $supArgs `
    -WindowStyle Hidden | Out-Null

# 1) The supervisor writes the state file within a couple of seconds.
$deadline = (Get-Date).AddSeconds(30)
while (-not (Test-Path $StateFile) -and (Get-Date) -lt $deadline) {
    Start-Sleep -Milliseconds 500
}
if (-not (Test-Path $StateFile)) {
    Fail "supervisor did not come up (no state file after 30 s)"
}

# 2) The bridge needs a few seconds to tens of seconds (TTS model load,
#    follow stream open). No STT to load, so this is usually quicker than
#    the full coding voice.
Say "waiting for bridge to become ready"
$deadline = (Get-Date).AddSeconds(90)
$target = ""
while ((Get-Date) -lt $deadline) {
    Start-Sleep -Seconds 2
    if (Test-Path $LogFile) {
        $tail = Get-Content $LogFile -Tail 300 -ErrorAction SilentlyContinue
        if ($tail) {
            $joined = (($tail | Out-String) -replace "`r", "")
            # "ready" line: "ready - speaking into session session-<id>"
            if ($joined -match 'ready') {
                $hit = [regex]::Matches($joined,
                        'speaking into (?:session )?(session-[0-9a-f-]+)')
                if ($hit.Count -gt 0) {
                    $target = $hit[$hit.Count - 1].Groups[1].Value
                    break
                }
            }
        }
    }
    if (-not (Test-Path $StateFile)) { break }   # supervisor died
}
if (-not $target) {
    if (-not (Test-Path $StateFile)) {
        Fail "supervisor exited before the bridge became ready"
    }
    Fail "bridge did not report ready within 90 s"
}
Say "STARTED - speaking into $target (workspace: $wsNorm). The voice is TTS-only (pocket, CPU, no mic)."
exit 0
