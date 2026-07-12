@echo off
cd /d "%~dp0\.."

".venv\Scripts\python.exe" -u scripts\train_sim_ppo_selfplay.py ^
  --observation-mode visual ^
  --iterations 1000000 ^
  --timesteps-per-iteration 65536 ^
  --bootstrap-iterations 0 ^
  --scripted-sample-rate 0.40 ^
  --scripted-curriculum ^
  --curriculum-start-stage 3 ^
  --elo-sampling ^
  --elo-episodes-per-pair 1 ^
  --eval-episodes 28 ^
  --per-opponent-eval-episodes 2 ^
  --detailed-eval-interval 5 ^
  --checkpoint-eval-episodes 1 ^
  --latest-checkpoint-rate 0.40 ^
  --best-checkpoint-rate 0.40 ^
  --old-sample-rate 0.20 ^
  --n-envs 16 ^
  --vec-env dummy ^
  --n-steps 256 ^
  --batch-size 1024 ^
  --device cpu ^
  --initial-checkpoint checkpoints\visual_student_selfplay_v1\iter_188.zip ^
  --initial-pool-dir checkpoints\visual_student_selfplay_v1 ^
  --initial-ratings runs\visual_student_selfplay_v1\metrics.json ^
  --checkpoint-dir checkpoints\visual_student_selfplay_v2 ^
  --final-checkpoint checkpoints\visual_student_selfplay_v2_final.zip ^
  --run-dir runs\visual_student_selfplay_v2 ^
  > logs\visual_student_selfplay_v2.out.log ^
  2> logs\visual_student_selfplay_v2.err.log
