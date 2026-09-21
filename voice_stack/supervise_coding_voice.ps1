# Supervisor for the coding-voice bridge.
#
# Runs the bridge and restarts it if it dies unexpectedly, so the voice
# never dies silently. Every (re)start keeps the previous run's log as
# coding_voice.prev.log so a crash always leaves its trace behind.
#
# While running it writes a state file (coding_voice.state) holding the
# supervisor's and the bridge's PIDs; ensure_coding_voice.ps1 and the stop
# protocol use it, so stopping never needs WMI. The state file is removed
# when the supervisor exits.
#
#   Start (idempotent — use the ensure script, which also starts this):
#       powershell -NoProfile -File voice_stack\ensure_coding_voice.ps1 `
#           -Workspace 'C:\path\to\DSH workspace'
#   Stop:
#       New-Item <DASHOME>\logs\coding_voice.STOP -ItemType File -Force
#       <then Stop-Process the bridge_pid from coding_voice.state>
#       (DASHOME = $DSH_HOME or ~/.dsh)
#
# Any extra arguments are passed straight to the bridge (e.g. --tts-voice).

param(
    [string]$LogFile = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
)

$ErrorActionPreference = "Continue"

if (-not $LogFile) {
    $dshHome = if ($env:DSH_HOME) { $env:DSH_HOME }
               else { Join-Path $env:USERPROFILE ".dsh" }
    $LogFile = Join-Path (Join-Path $dshHome "logs") "coding_voice.log"
}

# Offline-first: every model this bridge uses (both Kyutai STT models, the
# pocket-tts voices) is already in the local Hugging Face cache, and the
# huggingface.co freshness HEAD-checks have a bad habit of hanging startup
# on flaky DNS/network. Set to 0 to allow downloads (new model, first use).
$env:HF_HUB_OFFLINE = "1"

$logDir = Split-Path $LogFile -Parent
$stopFlag = Join-Path $logDir "coding_voice.STOP"
$supLog = Join-Path $logDir "coding_voice_supervisor.log"
$prevLog = [IO.Path]::ChangeExtension($LogFile, ".prev.log")
$errLog = [IO.Path]::ChangeExtension($LogFile, ".err.log")
$prevErrLog = [IO.Path]::ChangeExtension($prevLog, ".prev.err.log")
$stateFile = Join-Path $logDir "coding_voice.state"
$projectRoot = Split-Path $PSScriptRoot -Parent   # this file sits in voice_stack\

function LogSup($msg) {
    $line = "{0} supervisor: {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $supLog -Value $line -Encoding utf8
    Write-Output $line
}

function Write-State([int]$BridgePid) {
    $ws = ""
    for ($i = 0; $i -lt $PyArgs.Count - 1; $i++) {
        if ($PyArgs[$i] -eq "--workspace") { $ws = $PyArgs[$i + 1] }
    }
    $s = [ordered]@{
        supervisor_pid = $PID
        bridge_pid     = $BridgePid
        workspace      = $ws
        started_at     = (Get-Date).ToString("o")
    }
    Set-Content -Path $stateFile -Value ($s | ConvertTo-Json) -Encoding utf8
}

function Clear-State() {
    Remove-Item $stateFile -Force -ErrorAction SilentlyContinue
}

New-Item -ItemType Directory -Force -Path $logDir | Out-Null
if (Test-Path $stopFlag) {
    Remove-Item $stopFlag -Force
    LogSup "cleared stale stop sentinel from a previous run"
}

$attempt = 0
while ($true) {
    if (Test-Path $stopFlag) {
        LogSup "stop sentinel present - exiting without restart"
        break
    }
    $attempt++
    if ($attempt -gt 1) {
        # Keep the trace of the run that just died, then start fresh.
        if (Test-Path $LogFile) { Move-Item $LogFile $prevLog -Force }
        if (Test-Path $errLog)  { Move-Item $errLog $prevErrLog -Force }
        LogSup "starting bridge (restart $([string]($attempt - 1)))"
        # Give the dead instance time to release the microphone.
        Start-Sleep -Seconds 8
    } else {
        Set-Content -Path $LogFile -Value "" -NoNewline
        Set-Content -Path $errLog -Value "" -NoNewline
    }
    # -PassThru gives us the bridge PID for the state file; the redirected
    # streams go to the logs (stdout -> the main log, stderr -> the err log).
    $p = Start-Process -FilePath "python" `
        -ArgumentList (@("-X", "utf8", "-m", "voice_stack.coding_voice") + $PyArgs) `
        -WorkingDirectory $projectRoot -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput $LogFile -RedirectStandardError $errLog
    if (-not $p) {
        LogSup "Start-Process failed: $($Error[0].Exception.Message)"
        break
    }
    Write-State $p.Id
    LogSup "bridge started (pid $($p.Id))"
    $p.WaitForExit()
    $code = $p.ExitCode
    LogSup "python exited code=$code"
    if ($code -eq 0) {
        LogSup "clean exit - supervisor exiting"
        break
    }
    if (Test-Path $stopFlag) {
        LogSup "stop sentinel present - not restarting"
        break
    }
    LogSup "abnormal exit - restarting in 8s"
    Start-Sleep -Seconds 8
}

Remove-Item $stopFlag -Force -ErrorAction SilentlyContinue
Clear-State
LogSup "supervisor stopped"
