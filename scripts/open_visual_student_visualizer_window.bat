@echo off
cd /d "%~dp0\.."
".venv\Scripts\python.exe" scripts\visualize_sim.py --p1 checkpoint --checkpoint checkpoints\visual_student_v3.zip --follow-dir checkpoints\visual_student_selfplay_v1 --p2 scripted --p2-scripted ace --rotate-p2 mixed --rotation-checkpoint-dir checkpoints\visual_student_selfplay_v1 --rotation-scripted-rate 0.40 --speed 4
pause
