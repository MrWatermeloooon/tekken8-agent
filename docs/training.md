# GPU PPO training

## Single run

```powershell
build-gpu/Release/t8_v2_train.exe `
  --reward shaped --seed 2027 `
  --envs 4096 --horizon 128 --updates 100 `
  --epochs 4 --minibatch 4096 `
  --run-dir runs/phase0_scripted_shaped_seed2027
```

Use `--reward sparse` for the terminal-only objective. Both modes use the same
CUDA simulator, initial policy seed, frozen opponent, rollout count, optimizer,
and evaluation sequence. Only the reward tensor consumed by GAE differs. The
frozen `scripted_v1` mixture rotates eight GPU-native styles across lanes:
pressure, keepout, defense/punishment, grappling, lows, evasion, turtling, and
adaptive mixed offense.

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

When the default cadence saturates before the first evaluation, use the
high-resolution protocol:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/run_phase0.ps1 `
  -Updates 100 -Envs 512 -Horizon 32 -Epochs 2 -Minibatch 1024 `
  -EvalInterval 1 -EvalEpisodes 256 -Label phase0_scripted_highres
```

Use `-Seeds '2030,2031'` to extend an existing protocol to five total seeds
without rerunning the first three.

## Artifacts

Each run directory contains:

- `metrics.jsonl`: one JSON object per update with optimization metrics and
  periodic frozen evaluation results;
- `checkpoints/update_N.t8ppo`: versioned network weights plus Adam state.

The evaluator uses win/loss/draw outcomes, never shaped training reward. This
keeps optimization and selection/audit signals separate.
