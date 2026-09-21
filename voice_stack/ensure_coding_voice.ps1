# ensure_coding_voice.ps1 — idempotent "make the coding voice running" for a
# workspace. Run it and do nothing else; it handles every starting state:
#
#   - already running, right workspace  -> reports it, changes nothing
#   - stale / dead / wrong workspace    -> stops it, starts fresh
#   - not running at all                -> starts it
# and it exits 0 only when the bridge is actually ready (or already was).
#
#   powershell -NoProfile -File voice_stack\ensure_coding_voice.ps1 `
#       -Workspace 'C:\Users\press\OneDrive\Projects\DSH_TESTS'
#
#   [OK] exit 0 — speak now (the printed session is what it speaks into).
#   [ER] exit 1 — not running; the reason and the relevant log tails are
#        printed. Fix that, run this again.
#
# Extra arguments are passed straight to the bridge, e.g.:
#   -Workspace 'C:\...' --tts-voice vera --stt-model accurate
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

$LogDir    = "C:\Users\press\.dsh\logs"
$LogFile   = Join-Path $LogDir "coding_voice.log"
$ErrLog    = Join-Path $LogDir "coding_voice.err.log"
$SupLog    = Join-Path $LogDir "coding_voice_supervisor.log"
$StateFile = Join-Path $LogDir "coding_voice.state"
$StopFlag  = Join-Path $LogDir "coding_voice.STOP"
$SupScript = Join-Path $PSScriptRoot "supervise_coding_voice.ps1"

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
    Say "--- coding_voice.log (tail) ---"
    if (Test-Path $LogFile) { Get-Content $LogFile -Tail 25 }
    if (Test-Path $ErrLog) {
        $err = Get-Content $ErrLog -Tail 15
        if ($err) { Say "--- coding_voice.err.log (tail) ---"; $err | ForEach-Object { Say $_ } }
    }
    if (Test-Path $SupLog) {
        Say "--- supervisor log (tail) ---"
        (Get-Content $SupLog -Tail 8) | ForEach-Object { Say $_ }
    }
    exit 1
}

$wsNorm = $Workspace.TrimEnd('\', '/').ToLower()

# ---------------------------------------------------------------- current
$state = Read-State
$bridgePid = 0; $supPid = 0; $stateWs = ""
if ($state) {
    $bridgePid = [int]$state.bridge_pid
    $supPid    = [int]$state.supervisor_pid
    $stateWs   = ([string]$state.workspace).TrimEnd('\', '/').ToLower()
}
$bridgeAlive = ($bridgePid -gt 0) -and (Get-Process -Id $bridgePid -ErrorAction SilentlyContinue)
$supAlive    = ($supPid -gt 0) -and (Get-Process -Id $supPid -ErrorAction SilentlyContinue)
$logFresh = (Test-Path $LogFile) -and
    (((Get-Item $LogFile).LastWriteTime -gt (Get-Date).AddSeconds(-90)))

if ($bridgeAlive -and $supAlive -and $logFresh -and ($stateWs -ceq $wsNorm)) {
    # Last "speaking into session-..." line in the log is where it talks now.
    $cur = ""
    $tail = Get-Content $LogFile -Tail 400 -ErrorAction SilentlyContinue
    # ready line: "ready - speaking into session session-<id>"
    # auto-follow: "auto-follow: speaking into session-<id>"
    $m = [regex]::Matches((($tail | Out-String) -replace "`r", ""),
                         'speaking into (?:session )?(session-[0-9a-f-]+)')
    if ($m.Count -gt 0) { $cur = $m[$m.Count - 1].Groups[1].Value }
    Say "ALREADY RUNNING - speaking into $cur (workspace: $wsNorm). Nothing to do."
    exit 0
}

# ---------------------------------------------------------------- stop
Say "stopping any existing stack (workspace: $wsNorm)"
New-Item -ItemType File -Force -Path $StopFlag | Out-Null   # sentinel FIRST
if ($bridgeAlive) {
    Stop-Process -Id $bridgePid -Force -ErrorAction SilentlyContinue
    $deadline = (Get-Date).AddSeconds(20)
    while ((Get-Process -Id $bridgePid -ErrorAction SilentlyContinue) -and
           (Get-Date) -lt $deadline) { Start-Sleep -Milliseconds 500 }
    if (Get-Process -Id $bridgePid -ErrorAction SilentlyContinue) {
        Fail "old bridge (pid $bridgePid) would not die - not starting a second one"
    }
} elseif (-not $state) {
    # Legacy stack (older supervisor wrote no state file): no other python
    # is expected to be running; kill whatever python is the bridge.
    $py = @(Get-Process python -ErrorAction SilentlyContinue)
    if ($py.Count -gt 0) {
        Say "NOTE: no state file found - stopping $($py.Count) python process(es)"
        $py | Stop-Process -Force -ErrorAction SilentlyContinue
        Start-Sleep -Seconds 1
    }
}
if ($supAlive) {
    # The supervisor notices the sentinel after the bridge exits; give it a
    # moment, then take it down if it is stuck.
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
Say "starting supervisor (workspace: $wsNorm)"
# The supervisor prepends `-X utf8 -m voice_stack.coding_voice` itself;
# pass ONLY the bridge options here (passing -m again makes argparse die).
$extra = @($ExtraArgs | Where-Object { $_ })   # $null when no extras given
$bridgeArgs = @("--workspace", $Workspace) + $extra
$supArgs = @("-NoProfile", "-File", (Quote $SupScript)) +
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

# 2) The bridge needs ~10-40 s (STT + TTS load, follow stream open).
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
Say "STARTED - speaking into $target (workspace: $wsNorm). You can speak now."
exit 0
