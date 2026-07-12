@echo off
cd /d "%~dp0\.."

".venv\Scripts\python.exe" -u scripts\train_sim_ppo_selfplay.py ^
  --iterations 1000000 ^
  --timesteps-per-iteration 32768 ^
  --bootstrap-iterations 0 ^
  --scripted-sample-rate 0.34 ^
  --scripted-curriculum ^
  --curriculum-start-stage 3 ^
  --elo-sampling ^
  --elo-episodes-per-pair 1 ^
  --eval-episodes 8 ^
  --checkpoint-eval-episodes 1 ^
  --per-opponent-eval-episodes 3 ^
  --detailed-eval-interval 5 ^
  --latest-checkpoint-rate 0.50 ^
  --best-checkpoint-rate 0.50 ^
  --old-sample-rate 0.0 ^
  --n-envs 16 ^
  --vec-env dummy ^
  --n-steps 256 ^
  --batch-size 1024 ^
  --device cpu ^
  --initial-checkpoint checkpoints\mixed_curriculum_v11_lateral_stall_fix_fast\iter_002.zip ^
  --initial-pool-dir checkpoints\full_selfplay_stalemate_v7_fast ^
  --checkpoint-dir checkpoints\mixed_curriculum_v11_lateral_stall_fix_fast ^
  --final-checkpoint checkpoints\sim_ppo_mixed_curriculum_v11_lateral_stall_fix_fast.zip ^
  --run-dir runs\mixed_curriculum_v11_lateral_stall_fix_fast ^
  --initial-ratings runs\full_selfplay_stalemate_v7_fast\metrics.json ^
  > logs\mixed_curriculum_v11_lateral_stall_fix_fast.out.log ^
  2> logs\mixed_curriculum_v11_lateral_stall_fix_fast.err.log
