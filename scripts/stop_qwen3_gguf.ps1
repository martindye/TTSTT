# Stop only this project's resident Qwen speech server, never llama-server.
param(
    # State file written by the engine's ensure_server(). Default: the
    # original VoiceDesign server; pass the base-clone state to stop that.
    [string]$StatePath = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path $PSScriptRoot -Parent
if (-not $StatePath) { $StatePath = Join-Path $repoRoot 'tests/qwen3_gguf_server.json' }
if (-not (Test-Path -LiteralPath $StatePath)) { return }
$state = Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json
# Alias comes from the state when present; old state files predate the
# field and carry no alias, so fall back to the original design alias.
$expectedAlias = 'ttstt-qwen3-voicedesign-q8'
if ($state.PSObject.Properties['alias'] -and $state.alias) { $expectedAlias = $state.alias }
$expectedExe = [IO.Path]::GetFullPath((Join-Path $repoRoot 'tools/qwentts.cpp/build-win/Release/tts-server.exe'))
$speechProcess = Get-CimInstance Win32_Process -Filter "ProcessId = $([int]$state.pid)"
if ($speechProcess) {
    if ($speechProcess.ExecutablePath -ine $expectedExe -or
        $speechProcess.CommandLine -notlike "*$expectedAlias*") {
        throw 'Saved PID belongs to another process; refusing to stop it.'
    }
    Stop-Process -Id $speechProcess.ProcessId
    Write-Output "Qwen speech server stopped ($expectedAlias)."
}
Remove-Item -LiteralPath $StatePath
