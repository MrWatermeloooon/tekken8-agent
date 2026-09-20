@echo off
setlocal
cd /d "%~dp0.."

if not exist "build\Release\t8_v2_visualizer_feed.exe" (
    echo Missing build\Release\t8_v2_visualizer_feed.exe
    echo Build V2 first with: cmake --build build --config Release --parallel
    pause
    exit /b 1
)

set "PYTHON=.venv\Scripts\python.exe"
if not exist "%PYTHON%" set "PYTHON=python"

if "%~1"=="" (
    "%PYTHON%" scripts\visualize_v2.py --opponent-character reina --opponent-archetype rushdown
) else (
    "%PYTHON%" scripts\visualize_v2.py --follow-dir "%~1" --opponent-character reina --opponent-archetype rushdown
)

endlocal
