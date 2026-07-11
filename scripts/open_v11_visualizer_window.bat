@echo off
title Tekken V11 Visualizer
cd /d "%~dp0\.."
".venv\Scripts\python.exe" scripts\visualize_sim.py --p1 checkpoint --checkpoint checkpoints\mixed_curriculum_v11_lateral_stall_fix_fast\iter_001.zip --follow-dir checkpoints\mixed_curriculum_v11_lateral_stall_fix_fast --p2 scripted --p2-scripted rushdown --speed 4
echo.
echo Visualizer stopped. Press any key to close this window.
pause >nul
