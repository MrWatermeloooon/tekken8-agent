# Windows wrapper: python scripts/training.py stop --run-dir <run>
param([Parameter(Mandatory = $true)][string]$RunDir)
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
$python = Join-Path $repo ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { $python = "python" }
& $python (Join-Path $repo "scripts\training.py") stop --run-dir $RunDir
exit $LASTEXITCODE
