# hibernate-machine.ps1 - enable hibernation (if needed), then hibernate.
#
# Run ELEVATED (the UAC prompt is expected). Sequence:
#   1. powercfg /h on            (creates hiberfil.sys if hibernation is off)
#   2. wait 3 s, verify hiberfil
#   3. wait 20 s (so the caller can finish its message)
#   4. SetSuspendState 0,1,0     (hibernate; 0 = hibernate, 1 = force)
#
# Log: C:\Users\press\.dsh\logs\hibernate-attempt.log
# If hiberfil cannot be created (disk full / no admin), it aborts WITHOUT
# hibernating and says so in the log.

$ErrorActionPreference = "Continue"
$log = "C:\Users\press\.dsh\logs\hibernate-attempt.log"
New-Item -ItemType Directory -Force -Path (Split-Path $log) | Out-Null
Add-Content $log ("{0} hibernate sequence started (elevated, pid {1})" -f (Get-Date), $PID)

powercfg /h on 2>&1 | ForEach-Object { Add-Content $log "powercfg: $_" }
Start-Sleep -Seconds 3

if (Test-Path C:\hiberfil.sys) {
    $sz = [math]::Round((Get-Item C:\hiberfil.sys).Length / 1GB, 1)
    Add-Content $log ("hiberfil present ({0} GB) - hibernating in 20 s" -f $sz)
    Start-Sleep -Seconds 20
    rundll32 powrprof.dll,SetSuspendState 0,1,0
    Add-Content $log "SetSuspendState issued"
} else {
    Add-Content $log "hiberfil missing after powercfg /h on - ABORTING, machine left running"
}
