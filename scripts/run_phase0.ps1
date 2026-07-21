param(
    [int]$Updates = 100,
    [int]$Envs = 4096,
    [int]$Horizon = 128,
    [int]$Epochs = 4,
    [int]$Minibatch = 4096,
    [int]$EvalInterval = 10,
    [int]$EvalEpisodes = 256,
    [string]$Label = 'phase0_scripted',
    [string]$Seeds = '2027,2028,2029'
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$trainer = Join-Path $root 'build-gpu\Release\t8_v2_train.exe'
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "Trainer not built: $trainer"
}

$seedValues = $Seeds.Split(',') | ForEach-Object { [int]$_.Trim() }
foreach ($seed in $seedValues) {
    foreach ($reward in @('shaped', 'sparse')) {
        $runDir = Join-Path $root "runs\${Label}_${reward}_seed${seed}"
        & $trainer --envs $Envs --horizon $Horizon --updates $Updates `
            --epochs $Epochs --minibatch $Minibatch --seed $seed `
            --eval-interval $EvalInterval --eval-episodes $EvalEpisodes `
            --reward $reward --run-dir $runDir
        if ($LASTEXITCODE -ne 0) {
            throw "Phase 0 run failed: reward=$reward seed=$seed"
        }
    }
}
