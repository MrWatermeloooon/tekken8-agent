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
$supervisor = Get-Process -Id $session.supervisor_pid -ErrorAction SilentlyContinue
$metricsPath = Join-Path $runPath "metrics.jsonl"

[pscustomobject]@{
    SessionStatus = $session.status
    TrainerRunning = $null -ne $trainer
    TrainerPid = $session.trainer_pid
    SupervisorRunning = $null -ne $supervisor
    SupervisorPid = $session.supervisor_pid
    StartedAtUtc = $session.started_at_utc
    RunDirectory = $runPath
}

if (Test-Path -LiteralPath $metricsPath) {
    Write-Host "`nLatest metric:"
    Get-Content -LiteralPath $metricsPath -Tail 1
} else {
    Write-Host "`nMetrics have not been written yet."
}

$stderr = if ($session.stderr) { [string]$session.stderr } else { Join-Path $runPath "trainer.stderr.log" }
if ((Test-Path -LiteralPath $stderr) -and (Get-Item -LiteralPath $stderr).Length -gt 0) {
    Write-Host "`nLatest stderr:"
    Get-Content -LiteralPath $stderr -Tail 10
}

if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) {
    Write-Host "`nGPU:"
    & nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader
}
