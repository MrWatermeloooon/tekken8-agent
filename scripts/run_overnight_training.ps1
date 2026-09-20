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
Set-Location $repo

$trainer = Join-Path $repo "build\Release\t8_v2_train.exe"
if (-not (Test-Path -LiteralPath $trainer)) {
    throw "Missing $trainer. Build Release first."
}
if ($Environments % 16 -ne 0) {
    throw "Environments must be a multiple of 16."
}
if (($Environments * $Horizon) -lt $Minibatch) {
    throw "Minibatch cannot exceed the rollout sample count."
}
if ($Updates -lt 1 -or $CheckpointInterval -lt 1 -or $EvalInterval -lt 1) {
    throw "Updates and checkpoint/evaluation intervals must be positive."
}
if ([string]::IsNullOrWhiteSpace($RunDir)) {
    $RunDir = "runs\overnight_roster_visual_shaped_seed$Seed"
}

$runPath = [System.IO.Path]::GetFullPath((Join-Path $repo $RunDir))
$repoPath = [System.IO.Path]::GetFullPath($repo).TrimEnd('\') + '\'
if (-not $runPath.StartsWith($repoPath, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "RunDir must be inside the repository: $repo"
}

$metrics = Join-Path $runPath "metrics.jsonl"
$checkpoints = Join-Path $runPath "checkpoints"
$hasArtifacts = (Test-Path -LiteralPath $metrics) -or
    ((Test-Path -LiteralPath $checkpoints) -and
     (Get-ChildItem -LiteralPath $checkpoints -Force | Select-Object -First 1))
$resumePath = $null
if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    if ($hasArtifacts) {
        throw "Run directory already contains training artifacts: $runPath"
    }
} else {
    $resumePath = [System.IO.Path]::GetFullPath((Join-Path $repo $ResumeCheckpoint))
    if (-not (Test-Path -LiteralPath $resumePath)) {
        throw "Missing resume checkpoint: $resumePath"
    }
    $statePath = [System.IO.Path]::ChangeExtension($resumePath, ".t8state")
    if (-not (Test-Path -LiteralPath $statePath)) {
        throw "Missing exact-resume trainer state: $statePath"
    }
    if (-not (Test-Path -LiteralPath $metrics)) {
        throw "Missing metrics ledger for exact resume: $metrics"
    }
}

New-Item -ItemType Directory -Path $runPath -Force | Out-Null
$logSuffix = if ($null -ne $resumePath) { ".resume_$(Get-Date -Format yyyyMMdd_HHmmss)" } else { "" }
$stdout = Join-Path $runPath "trainer$logSuffix.stdout.log"
$stderr = Join-Path $runPath "trainer$logSuffix.stderr.log"
$sessionPath = Join-Path $runPath "session.json"
if (Test-Path -LiteralPath $sessionPath) {
    $archivedSession = Join-Path $runPath "session.previous_$(Get-Date -Format yyyyMMdd_HHmmss).json"
    Copy-Item -LiteralPath $sessionPath -Destination $archivedSession
}

if (-not ("SleepControl.NativeMethods" -as [type])) {
    Add-Type -Namespace SleepControl -Name NativeMethods -MemberDefinition @"
[System.Runtime.InteropServices.DllImport("kernel32.dll")]
public static extern uint SetThreadExecutionState(uint esFlags);
"@
}

$ES_CONTINUOUS = [uint32]2147483648
$ES_SYSTEM_REQUIRED = [uint32]0x00000001
$sleepState = [SleepControl.NativeMethods]::SetThreadExecutionState(
    $ES_CONTINUOUS -bor $ES_SYSTEM_REQUIRED
)
if ($sleepState -eq 0) {
    throw "Windows refused the request to prevent system sleep."
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
    "--checkpoint-interval", "$CheckpointInterval",
    "--eval-interval", "$EvalInterval",
    "--eval-episodes", "$EvalEpisodes",
    "--run-dir", "$RunDir"
)
if ($null -ne $resumePath) {
    $trainerArguments += @("--resume", "$ResumeCheckpoint")
}

$process = $null
$startedAt = (Get-Date).ToUniversalTime().ToString("o")
try {
    $process = Start-Process `
        -FilePath $trainer `
        -ArgumentList $trainerArguments `
        -WorkingDirectory $repo `
        -RedirectStandardOutput $stdout `
        -RedirectStandardError $stderr `
        -WindowStyle Hidden `
        -PassThru
    # Force PowerShell 5.1 to retain the native process handle so ExitCode remains available.
    $processHandle = $process.Handle

    [ordered]@{
        status = "running"
        started_at_utc = $startedAt
        supervisor_pid = $PID
        trainer_pid = $process.Id
        run_directory = $runPath
        stdout = $stdout
        stderr = $stderr
        sleep_prevention = "system"
        configuration = [ordered]@{
            seed = $Seed
            updates = $Updates
            environments = $Environments
            horizon = $Horizon
            epochs = $Epochs
            minibatch = $Minibatch
            opponents = "roster"
            curriculum_stage = "auto"
            observation_mode = "visual"
            reward = "shaped"
            checkpoint_interval = $CheckpointInterval
            eval_interval = $EvalInterval
            eval_episodes = $EvalEpisodes
            resume_checkpoint = $resumePath
        }
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $sessionPath -Encoding UTF8

    $process.WaitForExit()
    $process.Refresh()
    $exitCode = $process.ExitCode
    [ordered]@{
        status = $(if ($exitCode -eq 0) { "completed" } else { "failed" })
        started_at_utc = $startedAt
        completed_at_utc = (Get-Date).ToUniversalTime().ToString("o")
        supervisor_pid = $PID
        trainer_pid = $process.Id
        exit_code = $exitCode
        run_directory = $runPath
        stdout = $stdout
        stderr = $stderr
        sleep_prevention = "released"
    } | ConvertTo-Json -Depth 3 | Set-Content -LiteralPath $sessionPath -Encoding UTF8

    if ($exitCode -ne 0) {
        throw "V2 trainer exited with code $exitCode. See $stderr"
    }
}
finally {
    [void][SleepControl.NativeMethods]::SetThreadExecutionState($ES_CONTINUOUS)
}
