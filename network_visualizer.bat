@echo off
rem Neural network visualizer. Extra arguments pass through, e.g.: network_visualizer.bat --run runs\r2_base_DP
cd /d "%~dp0"
".venv\Scripts\python.exe" scripts\network_visualizer.py %*
if errorlevel 1 pause
