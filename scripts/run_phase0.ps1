param(
    [int]$Updates = 100,
    [int]$Envs = 4096,
    [int]$Horizon = 128,
    [int]$Epochs = 4,
    [int]$Minibatch = 4096
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $root 'build-gpu\Release\t8_v2_train.exe'
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "Trainer not built: $trainer"
}

foreach ($seed in @(2027, 2028, 2029)) {
    foreach ($reward in @('shaped', 'sparse')) {
        $runDir = Join-Path $root "runs\phase0_${reward}_seed${seed}"
        & $trainer --envs $Envs --horizon $Horizon --updates $Updates `
            --epochs $Epochs --minibatch $Minibatch --seed $seed `
            --reward $reward --run-dir $runDir
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 0 run failed: reward=$reward seed=$seed"
        }
    }
}
