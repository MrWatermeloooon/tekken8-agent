# Windows wrapper: python scripts/training.py phase0 --help
param(
    [int]$Updates = 100,
    [int]$AnnealUpdates = 100,
    [int]$Envs = 4096,
    [int]$Horizon = 128,
    [int]$Epochs = 4,
    [int]$Minibatch = 4096,
    [int]$EvalInterval = 10,
    [int]$EvalEpisodes = 256,
    [string]$Label = 'phase0_heldout_v2_visual',
    [string]$Seeds = '2027,2028,2029',
    [ValidateSet('visual', 'privileged')]
    [string]$ObservationMode = 'visual',
    [string]$BuildDirectory = 'build',
    [bool]$ResumeIncomplete = $true
)
$ErrorActionPreference = 'Stop'
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$arguments = @("phase0", "--seeds") + ($Seeds.Split(',') | ForEach-Object { $_.Trim() }) + @(
    "--label", $Label, "--updates", $Updates, "--anneal-updates", $AnnealUpdates, "--envs", $Envs,
    "--horizon", $Horizon, "--epochs", $Epochs, "--minibatch", $Minibatch, "--eval-interval", $EvalInterval,
    "--eval-episodes", $EvalEpisodes, "--observation-mode", $ObservationMode,
    "--trainer", (Join-Path $repo "$BuildDirectory\Release\t8_v2_train.exe"))
if (-not $ResumeIncomplete) { $arguments += "--no-resume-incomplete" }
& $python (Join-Path $repo "scripts\training.py") @arguments
exit $LASTEXITCODE
