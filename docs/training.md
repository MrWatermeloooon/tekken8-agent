# GPU PPO training

## Single run

```powershell
build-gpu/Release/t8_v2_train.exe `
  --reward shaped --seed 2027 `
  --envs 4096 --horizon 128 --updates 100 `
  --epochs 4 --minibatch 4096 `
  --run-dir runs/phase0_shaped_seed2027
```

Use `--reward sparse` for the terminal-only objective. Both modes use the same
CUDA simulator, initial policy seed, frozen opponent, rollout count, optimizer,
and evaluation sequence. Only the reward tensor consumed by GAE differs.

`--resume <checkpoint.t8ppo>` restores network weights, Adam moments, and the
optimizer step. `--smoke` selects a small two-update validation configuration.

## Phase 0 paired baseline

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase0.ps1
```

The script runs seeds 2027, 2028, and 2029 for both shaped and sparse reward.
Do not advance to return redistribution until all six runs finish and the
comparison records median and interquartile range for:

- frozen-opponent win rate versus environment steps;
- equal-step and wall-clock performance;
- final win rate and learning-curve area;
- stability indicators: KL, clip fraction, entropy, gradient norm, and value loss.

Smoke runs prove plumbing only and cannot satisfy the Phase 0 gate.

## Artifacts

Each run directory contains:

- `metrics.jsonl`: one JSON object per update with optimization metrics and
  periodic frozen evaluation results;
- `checkpoints/update_N.t8ppo`: versioned network weights plus Adam state.

The evaluator uses win/loss/draw outcomes, never shaped training reward. This
keeps optimization and selection/audit signals separate.
