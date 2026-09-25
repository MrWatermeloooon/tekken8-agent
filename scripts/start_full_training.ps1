# Windows wrapper: python scripts/training.py start --help
param(
    [string]$RunDir = "",
    [int]$Seed = 2027,
    [int]$Updates = 100,
    [int]$AnnealUpdates = 100,
    [int]$Environments = 4096,
    [int]$Horizon = 128,
    [int]$Epochs = 4,
    [int]$Minibatch = 4096,
    [switch]$NoVisualizer
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
if (-not $RunDir) { $RunDir = "runs\roster_visual_shaped_seed$Seed" }
$arguments = @("start", "--run-dir", $RunDir, "--seed", $Seed, "--updates", $Updates,
    "--anneal-updates", $AnnealUpdates, "--envs", $Environments, "--horizon", $Horizon, "--epochs", $Epochs,
    "--minibatch", $Minibatch)
if ($NoVisualizer) { $arguments += "--no-visualizer" }
& $python (Join-Path $repo "scripts\training.py") @arguments
exit $LASTEXITCODE
