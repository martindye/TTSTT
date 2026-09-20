# Supervisor for the coding-voice bridge.
#
# Runs the bridge and restarts it if it dies unexpectedly, so the voice
# never dies silently. Every (re)start keeps the previous run's log as
# coding_voice.prev.log so a crash always leaves its trace behind.
#
# Stop protocol: create the STOP sentinel file, then kill the python
# process. The supervisor sees the sentinel and exits without restarting.
#
#   Start (from the TTSTT project dir):
#       powershell -NoProfile -File voice_stack\supervise_coding_voice.ps1
#   Stop:
#       New-Item C:\Users\press\.dsh\logs\coding_voice.STOP -ItemType File -Force
#       <then kill the python process>
#
# Any extra arguments are passed straight to the bridge (e.g. --tts-voice).

param(
    [string]$LogFile = "C:\Users\press\.dsh\logs\coding_voice.log",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
)

$ErrorActionPreference = "Continue"
$logDir = Split-Path $LogFile -Parent
$stopFlag = Join-Path $logDir "coding_voice.STOP"
$supLog = Join-Path $logDir "coding_voice_supervisor.log"
$prevLog = [IO.Path]::ChangeExtension($LogFile, ".prev.log")

function LogSup($msg) {
    $line = "{0} supervisor: {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $supLog -Value $line -Encoding utf8
    Write-Output $line
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
        if (Test-Path $LogFile) {
            Move-Item $LogFile $prevLog -Force
        }
        LogSup "starting bridge (restart $([string]($attempt - 1)))"
        # Give the dead instance time to release the microphone.
        Start-Sleep -Seconds 8
    } else {
        if (Test-Path $LogFile) {
            Set-Content -Path $LogFile -Value "" -NoNewline
        }
    }
    python -X utf8 -m voice_stack.coding_voice @PyArgs 2>&1 |
        Out-File -FilePath $LogFile -Encoding utf8 -Append
    $code = $LASTEXITCODE
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
LogSup "supervisor stopped"
