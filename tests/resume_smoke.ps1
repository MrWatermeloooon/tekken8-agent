param(
    [Parameter(Mandatory = $true)]
    [string]$Trainer
)

$ErrorActionPreference = 'Stop'
$root = Join-Path ([System.IO.Path]::GetTempPath()) ("t8_v2_resume_" + [guid]::NewGuid().ToString('N'))
$resumed = Join-Path $root 'resumed'
$reference = Join-Path $root 'reference'
$corrupt = Join-Path $root 'corrupt'

function Invoke-Trainer([string[]]$Arguments) {
    & $Trainer @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Trainer exited with code $LASTEXITCODE"
    }
}

function Assert-TrainerFails([string[]]$Arguments, [string]$Message) {
    & $Trainer @Arguments
    if ($LASTEXITCODE -eq 0) {
        throw $Message
    }
}

try {
    $common = @(
        '--envs', '32', '--horizon', '8', '--epochs', '1', '--minibatch', '128',
        '--eval-interval', '1', '--eval-episodes', '16', '--checkpoint-interval', '1',
        '--reward', 'sparse', '--seed', '6501'
    )
    Invoke-Trainer ($common + @('--updates', '1', '--run-dir', $resumed))
    $checkpoint = Join-Path $resumed 'checkpoints\update_1.t8ppo'

    Copy-Item -LiteralPath $resumed -Destination $corrupt -Recurse
    Add-Content -LiteralPath (Join-Path $corrupt 'metrics.jsonl') -Value '{"update":2' -NoNewline
    Assert-TrainerFails `
        ($common + @('--updates', '2', '--run-dir', $corrupt, '--resume',
                     (Join-Path $corrupt 'checkpoints\update_1.t8ppo'))) `
        'Trainer accepted an incomplete metrics row'
    Assert-TrainerFails `
        ($common + @('--updates', '1', '--learning-rate', 'NaN',
                     '--run-dir', (Join-Path $root 'invalid_nan'))) `
        'Trainer accepted a non-finite learning rate'

    Invoke-Trainer ($common + @('--updates', '2', '--run-dir', $resumed, '--resume', $checkpoint))
    Invoke-Trainer ($common + @('--updates', '2', '--run-dir', $reference))

    $resumedHash = (Get-FileHash -Algorithm SHA256 (Join-Path $resumed 'checkpoints\update_2.t8ppo')).Hash
    $referenceHash = (Get-FileHash -Algorithm SHA256 (Join-Path $reference 'checkpoints\update_2.t8ppo')).Hash
    if ($resumedHash -ne $referenceHash) {
        throw "Resumed checkpoint does not match uninterrupted checkpoint"
    }
} finally {
    if (Test-Path -LiteralPath $root) {
        Remove-Item -LiteralPath $root -Recurse -Force
    }
}
