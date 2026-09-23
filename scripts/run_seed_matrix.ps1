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

# Runs one long training per seed, sequentially, with identical options, then
# aggregates held-out results across seeds (tools\aggregate_seed_matrix.py).
# Re-running the same command skips finished seeds and resumes partial ones
# from their newest checkpoint. A regression-guard pause (exit 3) is recorded
# and the matrix moves on to the next seed.

$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
if (-not $Trainer) {
    $Trainer = @("build-gpu\Release\t8_v2_train.exe", "build\Release\t8_v2_train.exe") |
        ForEach-Object { Join-Path $repo $_ } | Where-Object { Test-Path -LiteralPath $_ } |
        Select-Object -First 1
}
if (-not $Trainer -or -not (Test-Path -LiteralPath $Trainer)) { throw "Trainer executable not found." }
if ($Seeds.Count -lt 1) { throw "Provide at least one seed." }

if (-not ("SleepControl.NativeMethods" -as [type])) {
    Add-Type -Namespace SleepControl -Name NativeMethods -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
}
$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]0x00000001

$statusPath = Join-Path $repo "runs\${Prefix}_status.json"
$status = [ordered]@{}
function Save-Status {
    [ordered]@{
        prefix = $Prefix
        trainer = $Trainer
        seeds = $Seeds
        updates = $Updates
        options = @($commonArguments)
        updated_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        runs = $status
    } | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $statusPath -Encoding UTF8
}

$commonArguments = @(
    "--envs", "$Environments", "--horizon", "$Horizon", "--updates", "$Updates",
    "--anneal-updates", "$AnnealUpdates", "--curriculum-updates", "$CurriculumUpdates",
    "--epochs", "$Epochs", "--minibatch", "$Minibatch",
    "--opponents", "roster", "--curriculum-stage", "auto",
    "--observation-mode", "visual", "--reward", "shaped",
    "--checkpoint-interval", "$CheckpointInterval",
    "--eval-interval", "$EvalInterval", "--eval-episodes", "$EvalEpisodes"
) + $ExtraArgs

[void][SleepControl.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED)
try {
    foreach ($seed in $Seeds) {
        $runDir = "runs\${Prefix}_seed$seed"
        $runPath = Join-Path $repo $runDir
        $checkpoints = Join-Path $runPath "checkpoints"
        $final = Join-Path $checkpoints "update_$Updates.t8ppo"
        if (Test-Path -LiteralPath $final) {
            $status["$seed"] = [ordered]@{ run_directory = $runDir; state = "completed"; exit_code = 0 }
            Save-Status
            continue
        }
        $arguments = $commonArguments + @("--seed", "$seed", "--run-dir", $runDir)
        $latest = if (Test-Path -LiteralPath $checkpoints) {
            Get-ChildItem -LiteralPath $checkpoints -Filter "update_*.t8ppo" |
                Sort-Object { [int]($_.BaseName -replace '^update_', '') } | Select-Object -Last 1
        }
        if ($latest) { $arguments += @("--resume", $latest.FullName) }
        New-Item -ItemType Directory -Path $runPath -Force | Out-Null
        $stamp = Get-Date -Format yyyyMMdd_HHmmss
        $stdout = Join-Path $runPath "trainer_$stamp.stdout.log"
        $stderr = Join-Path $runPath "trainer_$stamp.stderr.log"
        $status["$seed"] = [ordered]@{
            run_directory = $runDir; state = "running"; resumed_from = $(if ($latest) { $latest.Name } else { $null })
            started_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        }
        Save-Status
        Write-Host "seed $seed -> $runDir$(if ($latest) { " (resuming $($latest.Name))" })"
        $process = Start-Process -FilePath $Trainer -ArgumentList $arguments -WorkingDirectory $repo `
            -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru
        $processHandle = $process.Handle  # keeps ExitCode available in PowerShell 5.1
        $process.WaitForExit()
        $process.Refresh()
        $state = switch ($process.ExitCode) { 0 { "completed" } 3 { "paused_by_regression_guard" } default { "failed" } }
        $status["$seed"].state = $state
        $status["$seed"].exit_code = $process.ExitCode
        $status["$seed"].completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        Save-Status
        Write-Host "seed $seed $state (exit $($process.ExitCode))"
    }
}
finally {
    [void][SleepControl.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS)
}

$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
$runDirectories = $Seeds | ForEach-Object { Join-Path $repo "runs\${Prefix}_seed$_" }
& $python (Join-Path $repo "tools\aggregate_seed_matrix.py") @runDirectories `
    --output (Join-Path $repo "runs\${Prefix}_report")
