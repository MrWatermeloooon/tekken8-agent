# GPU PPO training

## Guarded full-run launcher

The one-command launcher uses the documented full-roster visual configuration, refuses to
overwrite existing artifacts, and opens the independent checkpoint-following CUDA visualizer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\start_full_training.ps1 `
  -RunDir runs\roster_visual_shaped_seed2027 `
  -Seed 2027
```

Add `-NoVisualizer` for maximum isolated throughput. Use the native resume command below after an
interruption; the launcher intentionally never guesses which checkpoint should be resumed.

For an effectively open-ended overnight run with periodic checkpoints and Windows sleep
prevention, start the supervisor in a background PowerShell process:

```powershell
$run = "runs\overnight_roster_visual_shaped_$(Get-Date -Format yyyyMMdd_HHmmss)"
Start-Process powershell.exe -WindowStyle Hidden -ArgumentList @(
  '-NoProfile', '-ExecutionPolicy', 'Bypass',
  '-File', 'scripts\run_overnight_training.ps1',
  '-RunDir', $run
)
```

The default target is one million PPO updates, so the job keeps running until it is stopped or
completes. It uses 4096 CUDA environments, the visual temporal observation, the full roster,
automatic curriculum, and shaped reward. Check and stop it cleanly with:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\training_status.ps1 -RunDir $run
powershell -ExecutionPolicy Bypass -File scripts\stop_training.ps1 -RunDir $run
```

To continue the same run after an interruption, first make sure `metrics.jsonl` ends at the chosen
checkpoint update, then pass both the original run directory and checkpoint:

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_overnight_training.ps1 `
  -RunDir $run `
  -ResumeCheckpoint "$run\checkpoints\update_100.t8ppo"
```

## Observation and reward modes

Roster training is the default (`--opponents roster`). `--observation-mode visual` starts with the
13 screen-compatible health/position/distance/motion/hit/attack features and augments them on GPU
with character identity, archetype, and an eight-decision temporal stack, producing 95 features.
`privileged` similarly augments the 19 simulator features to 101. `--opponents legacy` retains the
old 13/19 contracts for controlled comparisons.

`--reward shaped` consumes dense combat rewards. `--reward sparse` consumes only terminal
win/loss/draw outcomes. Evaluation always uses sparse outcomes regardless of training reward.
Shaped rewards are scaled by `0.01` only when they enter PPO, keeping critic targets near unit
scale without changing simulator rewards or behavior metrics. Sparse rewards remain unscaled.
The clipped policy/value objective follows the core algorithm described in the
[PPO paper](https://arxiv.org/abs/1707.06347), with target-KL stopping and gradient clipping.
Max-frame boundaries bootstrap the post-step value as truncations; knockout, stalemate, and
no-action boundaries remain hard terminals.

Training is balanced in blocks of 16 lanes: eight matched profiles with the Jun learner as P1 and
the same profiles with Jun as P2. Starts are deterministically randomized by seed. Profile tables,
assignments, temporal history, character IDs, character move lookup, and actions remain in VRAM.
New profiles are applied only when their lane resets, so an unfinished round never changes
character. Evaluation uses equal P1/P2 episodes, fair timeout draws, a fixed seed sequence, fixed
catalog coverage, and the separate `HeldOutV2` behavior set. Evaluation does not alter training
opponent priorities.

The automatic curriculum divides `--curriculum-updates` (default 100) into four stages: Jun
fundamentals, rotating character groups, the full roster, and checkpoint self-play. Extending
`--updates` therefore cannot move a resumed run backward. In the fourth stage, parallel lanes use
the frozen latest checkpoint for 80% of matches and the highest held-out win-rate older checkpoint
for 20%. Use `--curriculum-stage 1|2|3|4` to pin a stage.

## Example

```powershell
build\Release\t8_v2_train.exe `
  --opponents roster --observation-mode visual --reward shaped --seed 2027 `
  --envs 4096 --horizon 128 --updates 100 `
  --epochs 4 --minibatch 4096 `
  --learning-rate 0.0003 --final-learning-rate 0.00003 --anneal-updates 100 `
  --entropy-coefficient 0.01 --final-entropy-coefficient 0.001 `
  --run-dir runs\visual_shaped_seed2027
```

The environment and evaluation counts must be multiples of 16. `--smoke` selects a two-update
plumbing check and is not evidence of learning quality.

The trainer also exposes `--reward-scale`, `--gamma`, `--gae-lambda`, `--clip-range`,
`--value-clip-range`, `--target-kl`, `--value-coefficient`, and `--max-gradient-norm`.
`--anneal-updates` is independent of `--updates`, so extending and resuming a run does not alter
the optimizer schedule of updates that already happened. `--curriculum-updates` provides the same
stability for curriculum stages.

## Checkpoints and exact resume

Every checkpoint interval writes two atomic artifacts:

- `update_N.t8ppo`: architecture, weights, Adam moments, optimizer step, payload size, and checksum.
- `update_N.t8state`: completed update, step/time counters, all run-defining options, every GPU
  simulator state, profile assignment, executed-action history, and both self-play temporal states.

The simulator refreshes derived observations and masks after state upload. PPO reductions use a
fixed order, so a resumed two-update run is regression-tested against an uninterrupted run by
SHA-256. Existing checkpoints and non-empty run directories are never silently overwritten.

## CUDA correctness audit

Run NVIDIA Compute Sanitizer on small, representative workloads after a Release build. The
[official guide](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html) recommends
starting with memcheck; V2 also exercises synchronization, initialization, and race tools.

```powershell
compute-sanitizer --tool memcheck --leak-check full --error-exitcode 99 `
  build\Release\t8_v2_training_benchmark.exe `
  --envs 16 --horizon 4 --updates 1 --minibatch 32 --epochs 1 --visual
compute-sanitizer --tool synccheck --error-exitcode 99 `
  build\Release\t8_v2_rollout_tests.exe
compute-sanitizer --tool initcheck --error-exitcode 99 `
  build\Release\t8_v2_rollout_tests.exe
```

Racecheck can be much slower than normal execution; run it per executable with an explicit time
bound so one heavily instrumented kernel family does not hide results from other targets.

## Metrics

`metrics.jsonl` contains one ordered row per update. The ledger is rewritten through an atomic
temporary-file replacement so an interrupted write leaves a complete previous or new version.
Optimization fields include policy/value loss, entropy, approximate KL, clip fraction, gradient
norm, minibatches, completed epochs, and target-KL early-stop status. Evaluation rows include total,
P1, P2, and per-style outcomes plus timeouts, stalemates, frames, and damage dealt/taken.
`evaluation` is masked argmax and `evaluation_stochastic` samples the trained categorical policy.
`training_opponent`, `latest_checkpoint_update`, and `best_older_checkpoint_update` identify the
self-play pool used for each rollout.
Roster runs
also update `matchup_matrix.json` with every character/archetype cell, draw-aware score, matchup Elo,
and forgetting flags. The Python `MatchupEvaluation` exporter adds punishment, throw-break,
low-defense, string-interruption, sidestep, Heat-defense, and wall-escape rates in JSON/CSV.

## Phase 0 paired baseline

```powershell
powershell -ExecutionPolicy Bypass -File scripts\run_phase0.ps1 `
  -Updates 100 -Envs 512 -Horizon 32 -Epochs 2 -Minibatch 1024 `
  -EvalInterval 1 -EvalEpisodes 256 `
  -Seeds '2027,2028,2029,2030,2031'
```

The launcher skips completed runs and exactly resumes an incomplete run only when its last valid
metrics update has both matching checkpoint artifacts. It otherwise refuses ambiguous non-empty
directories. After all shaped/sparse runs finish it invokes `tools/analyze_phase0.py`, which
requires at least three seeds and refuses mixed benchmark or observation protocols. It reports
median/IQR for final performance, learning-curve AUC, side, style, behavior, wall-clock, and PPO
stability metrics.

The completed five-seed visual baseline is in the
[generated report](phase0_heldout_v2_visual_report.md). Shaped reward
improved normalized learning-curve AUC (median 0.649 vs. 0.568), but both modes ended at median
0.672 held-out win rate. That supports no Phase 1 promotion under the declared gate.

Do not promote return redistribution, hyperparameter PBT, or league training from smoke results or
the archived V1/in-distribution Phase 0 numbers.
