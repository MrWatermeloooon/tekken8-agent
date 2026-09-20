param(
    [Parameter(Mandatory = $true)]
    [string]$RunDir
)

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$runPath = [System.IO.Path]::GetFullPath((Join-Path $repo $RunDir))
$sessionPath = Join-Path $runPath "session.json"
if (-not (Test-Path -LiteralPath $sessionPath)) {
    throw "Missing training session metadata: $sessionPath"
}

$session = Get-Content -LiteralPath $sessionPath -Raw | ConvertFrom-Json
$trainer = Get-Process -Id $session.trainer_pid -ErrorAction SilentlyContinue
if ($null -eq $trainer) {
    Write-Host "Trainer PID $($session.trainer_pid) is not running."
    exit 0
}
if ($trainer.ProcessName -ne "t8_v2_train") {
    throw "PID $($session.trainer_pid) belongs to $($trainer.ProcessName), not t8_v2_train. Refusing to stop it."
}

Stop-Process -Id $trainer.Id
Write-Host "Stopped trainer PID $($trainer.Id). The supervisor will release sleep prevention."
