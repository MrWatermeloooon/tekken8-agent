# GPU PPO training

## Observation and reward modes

`--observation-mode visual` trains the deployable 13-feature policy from screen-compatible
health, positions, distance, motion, hit events, and attack-likelihood signals. These tensors are
generated and routed on GPU. `privileged` uses the 19-feature simulator-state contract and is
useful as a teacher/oracle, but is not directly deployable from screen capture.

`--reward shaped` consumes dense combat rewards. `--reward sparse` consumes only terminal
win/loss/draw outcomes. Evaluation always uses sparse outcomes regardless of training reward.
The clipped policy/value objective follows the core algorithm described in the
[PPO paper](https://arxiv.org/abs/1707.06347), with target-KL stopping and gradient clipping.

Training is balanced in blocks of 16 lanes: eight styles with the learner as P1 and the same eight
with the learner as P2. Starts are deterministically randomized by seed. Evaluation uses a distinct
held-out eight-style suite, equal P1/P2 episodes, fair timeout draws, and a fixed seed sequence.

## Example

```powershell
build\Release\t8_v2_train.exe `
  --observation-mode visual --reward shaped --seed 2027 `
  --envs 4096 --horizon 128 --updates 100 `
  --epochs 4 --minibatch 4096 `
  --run-dir runs\visual_shaped_seed2027
```

The environment and evaluation counts must be multiples of 16. `--smoke` selects a two-update
plumbing check and is not evidence of learning quality.

## Checkpoints and exact resume

Every checkpoint interval writes two atomic artifacts:

- `update_N.t8ppo`: architecture, weights, Adam moments, optimizer step, payload size, and checksum.
- `update_N.t8state`: completed update, step/time counters, all run-defining options, and every GPU
  simulator state serialized field-by-field.

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
Optimization fields include policy/value
loss, entropy, approximate KL, clip fraction, gradient norm, minibatches, completed epochs, and
target-KL early-stop status. Evaluation rows include total, P1, P2, and per-style outcomes plus
timeouts, stalemates, frames, and damage dealt/taken.

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
