# stop_coding_voice.ps1 - stop the coding-voice stack, for good.
#
# The supervisor (supervise_coding_voice.ps1) restarts the bridge 8 s after
# any death, so killing the python alone always loses: this script does the
# whole stop in the safe order -
#
#   1. stop sentinel FIRST (the supervisor checks it before every (re)start)
#   2. kill the bridge (PID from coding_voice.state; python by command line)
#   3. wait for the supervisor to notice and exit; force-kill it if stuck
#   4. remove the state file, confirm nothing is left, clean the sentinel
#
# Idempotent: safe to run when nothing is running (reports and exits 0).
# This is the target of the spoken commands "stop speech" / "voice off".
#
#   powershell -NoProfile -File voice_stack\stop_coding_voice.ps1

param(
    [string]$DshHome = "",
    [int]$WaitSeconds = 20
)

$ErrorActionPreference = "Continue"

if (-not $DshHome) {
    $DshHome = if ($env:DSH_HOME) { $env:DSH_HOME } else { Join-Path $env:USERPROFILE ".dsh" }
}
$LogDir    = Join-Path $DshHome "logs"
$StopFlag  = Join-Path $LogDir "coding_voice.STOP"
$StateFile = Join-Path $LogDir "coding_voice.state"

function Say([string]$m) { Write-Output ("{0} {1}" -f (Get-Date -Format "HH:mm:ss"), $m) }

if (-not (Test-Path $LogDir)) { New-Item -ItemType Directory -Force -Path $LogDir | Out-Null }

# 1) Sentinel FIRST - no bridge can be (re)started while it exists.
New-Item -ItemType File -Force -Path $StopFlag | Out-Null
Say "stop sentinel set"

# 2) PIDs from the state file the supervisor writes (may be absent = legacy).
$bridgePid = 0
$supPid = 0
if (Test-Path $StateFile) {
    try {
        $s = Get-Content $StateFile -Raw | ConvertFrom-Json
        $bridgePid = [int]$s.bridge_pid
        $supPid = [int]$s.supervisor_pid
    } catch { }
}

# 3) Kill the bridge: by PID when known, else by command line (never blind
#    at-all-python - this box runs other python occasionally).
$killed = $false
if ($bridgePid -gt 0) {
    if (Stop-Process -Id $bridgePid -Force -ErrorAction SilentlyContinue) { $killed = $true }
    if ($killed) { Say "bridge (pid $bridgePid) stopped" }
}
if (-not $killed) {
    $voiced = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                Where-Object { $_.CommandLine -like '*voice_stack*' })
    if ($voiced.Count -gt 0) {
        foreach ($p in $voiced) {
            Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue
            Say "bridge (pid $($p.ProcessId)) stopped"
            $killed = $true
        }
    } elseif (-not (Get-Process python -ErrorAction SilentlyContinue)) {
        Say "no bridge running"
    }
}

# 4) The supervisor exits on its own when it sees the sentinel; give it the
#    deadline, then force it down if it is stuck (e.g. mid-restart-sleep).
$deadline = (Get-Date).AddSeconds($WaitSeconds)
while ((Get-Date) -lt $deadline) {
    $supAlive = ($supPid -gt 0) -and (Get-Process -Id $supPid -ErrorAction SilentlyContinue)
    $anyVoicePy = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
                    Where-Object { $_.CommandLine -like '*voice_stack*' }).Count
    if ((-not $supAlive) -and ($anyVoicePy -eq 0) -and (-not (Test-Path $StateFile))) { break }
    Start-Sleep -Milliseconds 500
}
if (($supPid -gt 0) -and (Get-Process -Id $supPid -ErrorAction SilentlyContinue)) {
    Stop-Process -Id $supPid -Force -ErrorAction SilentlyContinue
    Say "supervisor (pid $supPid) force-stopped"
}
Remove-Item $StateFile -Force -ErrorAction SilentlyContinue

# 5) Verify and tidy: a lingering sentinel is harmless (the supervisor clears
#    a stale one at its next start), but remove it so state is unambiguous.
Start-Sleep -Milliseconds 500
$left = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue |
          Where-Object { $_.CommandLine -like '*voice_stack*' })
if ($left.Count -gt 0) {
    Say "WARNING: voice python still running: $($left.ProcessId -join ', ')"
    exit 1
}
Remove-Item $StopFlag -Force -ErrorAction SilentlyContinue
Say "voice stack stopped - nothing will restart it"
exit 0
