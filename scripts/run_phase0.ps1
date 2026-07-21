param(
    [int]$Updates = 100,
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
$root = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $root "$BuildDirectory\Release\t8_v2_train.exe"
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "Trainer not built: $trainer"
}

$seedValues = $Seeds.Split(',') | ForEach-Object { [int]$_.Trim() }
foreach ($seed in $seedValues) {
    foreach ($reward in @('shaped', 'sparse')) {
        $runDir = Join-Path $root "runs\${Label}_${reward}_seed${seed}"
        $resumeCheckpoint = $null
        if (Test-Path -LiteralPath $runDir) {
            $artifacts = Get-ChildItem -LiteralPath $runDir -Force -ErrorAction SilentlyContinue
            if ($artifacts) {
                $metricsPath = Join-Path $runDir 'metrics.jsonl'
                if (-not $ResumeIncomplete -or -not (Test-Path -LiteralPath $metricsPath)) {
                    throw "Refusing to reuse ambiguous/non-empty Phase 0 run directory: $runDir"
                }
                $lastMetric = Get-Content -LiteralPath $metricsPath -Tail 1 | ConvertFrom-Json
                $completedUpdate = [int]$lastMetric.update
                if ($completedUpdate -eq $Updates) {
                    Write-Host "Phase 0 run already complete: reward=$reward seed=$seed"
                    continue
                }
                if ($completedUpdate -le 0 -or $completedUpdate -gt $Updates) {
                    throw "Invalid completed update $completedUpdate in $metricsPath"
                }
                $resumeCheckpoint = Join-Path $runDir "checkpoints\update_${completedUpdate}.t8ppo"
                $resumeState = Join-Path $runDir "checkpoints\update_${completedUpdate}.t8state"
                if (-not (Test-Path -LiteralPath $resumeCheckpoint) -or
                    -not (Test-Path -LiteralPath $resumeState)) {
                    throw "Incomplete Phase 0 run is missing exact-resume artifacts: $runDir"
                }
            }
        }
        $trainerArgs = @(
            '--envs', $Envs, '--horizon', $Horizon, '--updates', $Updates,
            '--epochs', $Epochs, '--minibatch', $Minibatch, '--seed', $seed,
            '--eval-interval', $EvalInterval, '--eval-episodes', $EvalEpisodes,
            '--observation-mode', $ObservationMode, '--reward', $reward,
            '--run-dir', $runDir
        )
        if ($resumeCheckpoint) {
            $trainerArgs += @('--resume', $resumeCheckpoint)
            Write-Host "Resuming Phase 0 run: reward=$reward seed=$seed update=$completedUpdate"
        }
        & $trainer @trainerArgs
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 0 run failed: reward=$reward seed=$seed"
        }
    }
}

& python (Join-Path $root 'tools\analyze_phase0.py') `
    --runs-root (Join-Path $root 'runs') --label $Label `
    --output (Join-Path $root "docs\${Label}_report.md")
if ($LASTEXITCODE -ne 0) {
    throw "Phase 0 analysis failed for label=$Label"
}
