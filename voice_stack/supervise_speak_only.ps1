# supervise_speak_only.ps1 — supervisor for the TTS-only speak bridge.
#
# New-file twin of supervise_coding_voice.ps1, for the voice_stack.speak_only
# module (pocket TTS on CPU; no microphone, no STT — zero GPU usage, so it
# coexists with the 350k-context LLM). It keeps its own log/state/sentinel
# names (speak_only.*) so it can never collide with the full coding-voice
# stack (coding_voice.*).
#
#   powershell -NoProfile -File voice_stack\supervise_speak_only.ps1 `
#       --workspace 'C:\path\to\DSH workspace'
#
# Any extra arguments are passed straight to the bridge (e.g. --tts-voice).
# Do not run this directly — use ensure_speak_only.ps1.

param(
    [string]$LogFile = "",
    # When launched from a copy outside OneDrive (see ensure_speak_only.ps1),
    # name the real project root: $PSScriptRoot would be the copy's folder.
    [string]$ProjectRoot = "",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PyArgs
)

$ErrorActionPreference = "Continue"

$moduleName = "voice_stack.speak_only"

if (-not $LogFile) {
    $dshHome = if ($env:DSH_HOME) { $env:DSH_HOME }
               else { Join-Path $env:USERPROFILE ".dsh" }
    $LogFile = Join-Path (Join-Path $dshHome "logs") "speak_only.log"
}

$logDir     = Split-Path $LogFile -Parent
$stopFlag   = Join-Path $logDir "speak_only.STOP"
$supLog     = Join-Path $logDir "speak_only_supervisor.log"
$prevLog    = [IO.Path]::ChangeExtension($LogFile, ".prev.log")
$errLog     = [IO.Path]::ChangeExtension($LogFile, ".err.log")
$prevErrLog = [IO.Path]::ChangeExtension($prevLog, ".prev.err.log")
$stateFile  = Join-Path $logDir "speak_only.state"
# this file sits in voice_stack\; a TEMP copy gets the real root via -ProjectRoot
$projectRoot = if ($ProjectRoot) { $ProjectRoot } else { Split-Path $PSScriptRoot -Parent }

function LogSup($msg) {
    $line = "{0} speak-only supervisor: {1}" -f (Get-Date -Format "HH:mm:ss"), $msg
    Add-Content -Path $supLog -Value $line -Encoding utf8
    Write-Output $line
}

# Offline-first: the TTS-only bridge needs exactly one model family
# (the pocket TTS weights). Warm cache -> never touch the network.
$hubCache = Join-Path $env:USERPROFILE ".cache\huggingface\hub"
$pocketModel = "models--kyutai--pocket-tts-without-voice-cloning"
if (Test-Path (Join-Path $hubCache $pocketModel)) {
    $env:HF_HUB_OFFLINE = "1"
} else {
    $env:HF_HUB_OFFLINE = "0"
    LogSup "HF cache not warm (pocket TTS) - allowing model download this run"
}

function Write-State([int]$BridgePid) {
    $ws = ""
    $extra = @()
    $skip = $false
    for ($i = 0; $i -lt $PyArgs.Count; $i++) {
        if ($skip) { $skip = $false; continue }
        if ($PyArgs[$i] -eq "--workspace") {
            $ws = $PyArgs[$i + 1]
            $skip = $true
        } else {
            $extra += $PyArgs[$i]
        }
    }
    $s = [ordered]@{
        supervisor_pid = $PID
        bridge_pid     = $BridgePid
        workspace      = $ws
        args           = ($extra -join " ")
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
        Start-Sleep -Seconds 8
    } else {
        Set-Content -Path $LogFile -Value "" -NoNewline
        Set-Content -Path $errLog  -Value "" -NoNewline
    }
    $p = Start-Process -FilePath "python" `
        -ArgumentList (@("-X", "utf8", "-m", $moduleName) + $PyArgs) `
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
LogSup "speak-only supervisor stopped"
