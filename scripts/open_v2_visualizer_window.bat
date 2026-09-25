@echo off
rem Windows wrapper: python scripts\training.py visualize [--follow-dir <run>\checkpoints]
setlocal
cd /d "%~dp0.."
set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"
if "%~1"=="" (
    "%PYTHON%" scripts\training.py visualize
) else (
    "%PYTHON%" scripts\training.py visualize --follow-dir "%~1"
)
if errorlevel 1 pause
endlocal
