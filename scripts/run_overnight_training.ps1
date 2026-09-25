# Windows wrapper for the cross-platform supervisor: python scripts/training.py run --help
param(
    [string]$RunDir = "",
    [int]$Seed = 20260722,
    [int]$Updates = 1000000,
    [int]$AnnealUpdates = 1000000,
    [int]$Environments = 4096,
    [int]$Horizon = 128,
    [int]$Epochs = 4,
    [int]$Minibatch = 4096,
    [int]$CheckpointInterval = 100,
    [int]$EvalInterval = 100,
    [int]$EvalEpisodes = 256,
    [string]$ResumeCheckpoint = ""
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
if (-not $RunDir) { $RunDir = "runs\overnight_roster_visual_shaped_seed$Seed" }
$arguments = @("run", "--run-dir", $RunDir, "--seed", $Seed, "--updates", $Updates,
    "--anneal-updates", $AnnealUpdates, "--envs", $Environments, "--horizon", $Horizon, "--epochs", $Epochs,
    "--minibatch", $Minibatch, "--checkpoint-interval", $CheckpointInterval, "--eval-interval", $EvalInterval,
    "--eval-episodes", $EvalEpisodes)
if ($ResumeCheckpoint) { $arguments += @("--resume", $ResumeCheckpoint) }
& $python (Join-Path $repo "scripts\training.py") @arguments
exit $LASTEXITCODE
