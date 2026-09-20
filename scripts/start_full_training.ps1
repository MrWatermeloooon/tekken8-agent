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
Set-Location $repo

$trainer = Join-Path $repo "build\Release\t8_v2_train.exe"
$feed = Join-Path $repo "build\Release\t8_v2_visualizer_feed.exe"
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "Missing $trainer. Build Release first."
}
if (-not $NoVisualizer -and -not (Test-Path -LiteralPath $feed)) {
    throw "Missing $feed. Build Release first."
}
if ($Environments % 16 -ne 0) {
    throw "Environments must be a multiple of 16."
}
if (($Environments * $Horizon) -lt $Minibatch) {
    throw "Minibatch cannot exceed the rollout sample count."
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = "runs\roster_visual_shaped_seed$Seed"
}

$metrics = Join-Path $RunDir "metrics.jsonl"
$checkpoints = Join-Path $RunDir "checkpoints"
if ((Test-Path -LiteralPath $metrics) -or
    ((Test-Path -LiteralPath $checkpoints) -and
     (Get-ChildItem -LiteralPath $checkpoints -Force | Select-Object -First 1))) {
    throw "Run directory already contains training artifacts: $RunDir. Use resume or choose another RunDir."
}

$viewer = $null
try {
    if (-not $NoVisualizer) {
        $python = Join-Path $repo ".venv\Scripts\pythonw.exe"
        if (-not (Test-Path -LiteralPath $python)) {
            $python = (Get-Command pythonw.exe -ErrorAction Stop).Source
        }
        $viewerArguments = @(
            "scripts\visualize_v2.py",
            "--follow-dir", (Join-Path $RunDir "checkpoints"),
            "--observation-mode", "visual",
            "--opponent-character", "reina",
            "--opponent-archetype", "rushdown"
        )
        $viewer = Start-Process `
            -FilePath $python `
            -ArgumentList $viewerArguments `
            -WorkingDirectory $repo `
            -PassThru
    }

    $trainerArguments = @(
        "--envs", "$Environments",
        "--horizon", "$Horizon",
        "--updates", "$Updates",
        "--anneal-updates", "$AnnealUpdates",
        "--epochs", "$Epochs",
        "--minibatch", "$Minibatch",
        "--opponents", "roster",
        "--curriculum-stage", "auto",
        "--observation-mode", "visual",
        "--reward", "shaped",
        "--seed", "$Seed",
        "--checkpoint-interval", "1",
        "--eval-interval", "10",
        "--eval-episodes", "256",
        "--run-dir", "$RunDir"
    )
    & $trainer @trainerArguments
    if ($LASTEXITCODE -ne 0) {
        throw "V2 trainer exited with code $LASTEXITCODE"
    }
}
finally {
    if ($null -ne $viewer -and -not $viewer.HasExited) {
        Stop-Process -Id $viewer.Id
    }
}
