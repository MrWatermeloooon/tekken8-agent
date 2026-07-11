@echo off
title Tekken V11 Continuous Training
cd /d "%~dp0\.."
call scripts\run_v11_training.bat
echo.
echo Training stopped. Press any key to close this window.
pause >nul
