@echo off
cd /d "%~dp0\.."
".venv\Scripts\python.exe" scripts\visualize_sim.py --p1 checkpoint --checkpoint checkpoints\visual_student_selfplay_v2\iter_002.zip --follow-dir checkpoints\visual_student_selfplay_v3 --p2 scripted --p2-scripted nightmare --rotate-p2 mixed --rotation-checkpoint-dir checkpoints\visual_student_selfplay_v2 --rotation-scripted-rate 0.40 --speed 4
pause
