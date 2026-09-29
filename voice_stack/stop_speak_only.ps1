# stop_speak_only.ps1 - stop the TTS-only speak stack, for good.
#
# The supervisor restarts the bridge 8 s after any death, so killing the
# python alone always loses: stop sentinel FIRST, then kill bridge and
# supervisor, then verify and clean up. Idempotent: safe when nothing runs.
#
#   powershell -NoProfile -File voice_stack\stop_speak_only.ps1

param(
    [string]$DshHome = "",
    [int]$WaitSeconds = 20
)

$ErrorActionPreference = "Continue"

if (-not $DshHome) {
    $DshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $env:USERPROFILE ".dsh" }
}
$LogDir    = Join-Path $DshHome "logs"
$StopFlag  = Join-Path $LogDir "speak_only.STOP"
$StateFile = Join-Path $LogDir "speak_only.state"

function Say([string]$m) { Write-Output ("{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

# 1) Sentinel FIRST - the supervisor checks it before every (re)start.
New-Item -ItemType File -Force -Path $StopFlag | Out-Null
Say "stop sentinel set"

# 2) PIDs from the state file (may be absent).
$bridgePid = 0
$supPid = 0
if (Test-Path $StateFile) {
    try {
        $s = Get-Content $StateFile -Raw | ConvertFrom-Json
        $bridgePid = [int]$s.bridge_pid
        $supPid    = [int]$s.supervisor_pid
    } catch { }
}

# 3) Kill the bridge by PID, else by its exact command line (never blind
#    at-all-python; and the specific 'voice_stack.speak_only' pattern so the
#    full coding voice, if running, is untouched).
$killed = $false
if ($bridgePid -gt 0) {
    if (Stop-Process -Id $bridgePid -Force -ErrorAction SilentlyContinue) { $killed = $true }
    if ($killed) { Say "bridge (pid $bridgePid) stopped" }
}
if (-not $killed) {
    $voiced = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine -like '*voice_stack.speak_only*' })
    if ($voiced.Count -gt 0) {
        foreach ($p in $voiced) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            Say "bridge (pid $($p.ProcessId)) stopped"
            $killed = $true
        }
    } elseif ($killed -eq $false) {
        Say "no speak-only bridge running"
    }
}

# 4) The supervisor exits on the sentinel; give it the deadline, force if stuck.
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    $supAlive = ($supPid -gt 0) -and (Get-Process -Id $supPid -ErrorAction SilentlyContinue)
    $anyVoicePy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                    Where-Object { $_.CommandLine -like '*voice_stack.speak_only*' }).Count
    if ((-not $supAlive) -and ($anyVoicePy -eq 0) -and (-not (Test-Path $StateFile))) { break }
    Start-Sleep -Milliseconds 500
}
if (($supPid -gt 0) -and (Get-Process -Id $supPid -ErrorAction SilentlyContinue)) {
    Stop-Process -Id $supPid -Force -ErrorAction SilentlyContinue
    Say "supervisor (pid $supPid) force-stopped"
}
Remove-Item $StateFile -Force -ErrorAction SilentlyContinue

# 5) Verify and tidy.
Start-Sleep -Milliseconds 500
$left = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -like '*voice_stack.speak_only*' })
if ($left.Count -gt 0) {
    Say "WARNING: speak-only python still running: $($left.ProcessId -join ', ')"
    exit 1
}
Remove-Item $StopFlag -Force -ErrorAction SilentlyContinue
Say "speak-only voice stopped - nothing will restart it"
exit 0
