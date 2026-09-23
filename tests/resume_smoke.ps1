param(
    [Parameter(Mandatory = $true)]
    [string]$Trainer
)

$ErrorActionPreference = 'Stop'
$root = Join-Path ([System.IO.Path]::GetTempPath()) ("t8_v2_resume_" + [guid]::NewGuid().ToString('N'))
$resumed = Join-Path $root 'resumed'
$reference = Join-Path $root 'reference'
$corrupt = Join-Path $root 'corrupt'
$selfPlayResumed = Join-Path $root 'selfplay_resumed'
$selfPlayReference = Join-Path $root 'selfplay_reference'

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

function Get-Sha256([string]$Path) {
    $sha256 = [System.Security.Cryptography.SHA256]::Create()
    $stream = [System.IO.File]::OpenRead($Path)
    try {
        return ([System.BitConverter]::ToString($sha256.ComputeHash($stream))).Replace('-', '')
    } finally {
        $stream.Dispose()
        $sha256.Dispose()
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

    $resumedHash = Get-Sha256 (Join-Path $resumed 'checkpoints\update_2.t8ppo')
    $referenceHash = Get-Sha256 (Join-Path $reference 'checkpoints\update_2.t8ppo')
    if ($resumedHash -ne $referenceHash) {
        throw "Resumed checkpoint does not match uninterrupted checkpoint"
    }

    $selfPlayCommon = @(
        '--envs', '32', '--horizon', '8', '--epochs', '1', '--minibatch', '128',
        '--eval-interval', '1', '--eval-episodes', '16', '--checkpoint-interval', '1',
        '--reward', 'sparse', '--seed', '7619', '--curriculum-updates', '4'
    )
    Invoke-Trainer ($selfPlayCommon + @('--updates', '4', '--run-dir', $selfPlayResumed))
    $selfPlayCheckpoint = Join-Path $selfPlayResumed 'checkpoints\update_4.t8ppo'
    Invoke-Trainer ($selfPlayCommon + @(
        '--updates', '5', '--run-dir', $selfPlayResumed, '--resume', $selfPlayCheckpoint))
    Invoke-Trainer ($selfPlayCommon + @('--updates', '5', '--run-dir', $selfPlayReference))

    $selfPlayResumedHash = Get-Sha256 (
        Join-Path $selfPlayResumed 'checkpoints\update_5.t8ppo')
    $selfPlayReferenceHash = Get-Sha256 (
        Join-Path $selfPlayReference 'checkpoints\update_5.t8ppo')
    if ($selfPlayResumedHash -ne $selfPlayReferenceHash) {
        throw "Resumed self-play checkpoint does not match uninterrupted checkpoint"
    }

    # Regression guard: thresholds that trigger on any held-out drop. The
    # reference run rolls back at update 3 and pauses (exit 3) at update 4; a
    # run stopped exactly at the rollback must resume into the same update-4
    # checkpoint.
    $guardCommon = @(
        '--envs', '64', '--horizon', '16', '--epochs', '1', '--minibatch', '256',
        '--eval-interval', '1', '--eval-episodes', '32', '--checkpoint-interval', '1',
        '--seed', '9123', '--regression-score-drop', '0', '--regression-style-drop', '1',
        '--regression-max-side-gap', '1', '--regression-patience', '1',
        '--regression-max-rollbacks', '1'
    )
    $guardReference = Join-Path $root 'guard_reference'
    $guardResumed = Join-Path $root 'guard_resumed'
    & $Trainer @($guardCommon + @('--updates', '30', '--run-dir', $guardReference))
    if ($LASTEXITCODE -ne 3) { throw "Regression guard did not pause with exit code 3 (got $LASTEXITCODE)" }
    $guardRows = Get-Content -LiteralPath (Join-Path $guardReference 'metrics.jsonl')
    if (-not ($guardRows | Where-Object { $_ -match '"action":"rollback"' })) {
        throw "Regression guard never rolled back before pausing"
    }
    $pauseRow = $guardRows[-1]
    if ($pauseRow -notmatch '"action":"pause"') { throw "Final guard row is not a pause" }
    $pauseUpdate = [int]([regex]::Match($pauseRow, '"update":(\d+)').Groups[1].Value)
    $rollbackUpdate = [int]([regex]::Match(
        ($guardRows | Where-Object { $_ -match '"action":"rollback"' } | Select-Object -First 1),
        '"update":(\d+)').Groups[1].Value)
    Invoke-Trainer ($guardCommon + @('--updates', "$rollbackUpdate", '--run-dir', $guardResumed))
    & $Trainer @($guardCommon + @('--updates', '30', '--run-dir', $guardResumed, '--resume',
                                  (Join-Path $guardResumed "checkpoints\update_$rollbackUpdate.t8ppo")))
    if ($LASTEXITCODE -ne 3) { throw "Resumed guard run did not pause (exit $LASTEXITCODE)" }
    $guardReferenceHash = Get-Sha256 (Join-Path $guardReference "checkpoints\update_$pauseUpdate.t8ppo")
    $guardResumedHash = Get-Sha256 (Join-Path $guardResumed "checkpoints\update_$pauseUpdate.t8ppo")
    if ($guardReferenceHash -ne $guardResumedHash) {
        throw "Checkpoint after a resumed rollback does not match the uninterrupted run"
    }
} finally {
    if (Test-Path -LiteralPath $root) {
        Remove-Item -LiteralPath $root -Recurse -Force
    }
}
