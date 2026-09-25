# Windows wrapper: python scripts/training.py seed-matrix --help
param(
    [int[]]$Seeds = @(2027, 2028, 2029, 2030, 2031),
    [string]$Prefix = "seed_matrix",
    [int]$Updates = 36000,
    [int]$CurriculumUpdates = 2000,
    [int]$AnnealUpdates = 36000,
    [int]$Environments = 32768,
    [int]$Horizon = 128,
    [int]$Epochs = 4,
    [int]$Minibatch = 131072,
    [int]$CheckpointInterval = 100,
    [int]$EvalInterval = 100,
    [int]$EvalEpisodes = 256,
    [string]$Trainer = "",
    # Extra trainer options, identical for every seed (for example '--regression-guard','pause').
    [string[]]$ExtraArgs = @()
)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$arguments = @("seed-matrix", "--seeds") + $Seeds + @("--prefix", $Prefix, "--updates", $Updates,
    "--curriculum-updates", $CurriculumUpdates, "--anneal-updates", $AnnealUpdates, "--envs", $Environments,
    "--horizon", $Horizon, "--epochs", $Epochs, "--minibatch", $Minibatch,
    "--checkpoint-interval", $CheckpointInterval, "--eval-interval", $EvalInterval, "--eval-episodes", $EvalEpisodes)
if ($Trainer) { $arguments += @("--trainer", $Trainer) }
if ($ExtraArgs.Count -gt 0) { $arguments += @("--extra") + $ExtraArgs }
& $python (Join-Path $repo "scripts\training.py") @arguments
exit $LASTEXITCODE
