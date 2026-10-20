@echo off
rem Runs every test that does not need Tekken 8, one after another, with a
rem PASS/FAIL summary at the end (logs in test_logs\). See scripts\run_all_tests.py.
cd /d "%~dp0"
set "PY=%~dp0.venv\Scripts\python.exe"
if not exist "%PY%" set "PY=python"
"%PY%" scripts\run_all_tests.py %*
set RESULT=%ERRORLEVEL%
echo.
pause
exit /b %RESULT%
