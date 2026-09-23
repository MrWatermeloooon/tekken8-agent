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

`--observation-norm on|off` and `--return-norm on|off` (both default `on`) control the standard
PPO normalizers. Observation normalization keeps a running per-feature mean/variance inside the
model (clipped to ±10). The statistics are frozen while a rollout is collected and trained on,
then updated from that rollout, and they are saved in the checkpoint (format V4) so the live
Python agent applies the same transform. Return normalization divides rewards by the running
standard deviation of the discounted return, applied on top of `--reward-scale`. New networks use
orthogonal initialization. V3 checkpoints still load, with normalization disabled. To resume a run
that began before these options existed, pass `--observation-norm off --return-norm off`.

`--promotion-max-side-gap` (default 0.10; 1 disables it) allows an older checkpoint to become the
self-play best-older opponent only if its held-out |P1 − P2| win-rate gap is within the limit.
Metrics now also record `decision_fraction` and `decision_entropy`. The old `entropy` averages
in forced single-action frames, so it mostly measures how often the policy has a choice.
`--promotion-tie-band` (default 1) treats older checkpoints scoring within that many binomial
standard errors of the best as tied, and promotes the newest of them.

The regression guard (`--regression-guard rollback|pause|off`, default `rollback`) checks every
deterministic held-out evaluation against a reference. The reference is the best evaluation so far
that passed every check and has a saved checkpoint. An evaluation counts as regressed when any of
these hold:

- the win rate falls more than `--regression-score-drop` (0.15) below the reference;
- the weakest held-out style (the exploitability proxy) falls more than `--regression-style-drop`
  (0.30) below the reference's weakest style;
- the |P1 − P2| gap exceeds `--regression-max-side-gap` (0.20).

After `--regression-patience` (3) consecutive regressed evaluations, the learner's weights and Adam
state are restored from the reference. This happens up to `--regression-max-rollbacks` (2) times
per reference, then the trainer pauses. In `pause` mode it pauses on the first trigger. A pause
writes a checkpoint and exits with code 3; resume it normally. The checkpoint at the triggering
update keeps the evaluated weights for inspection, and the trainer state records that a rollback
happened, so resume stays exact. Each decision is logged under `regression_guard` in
`metrics.jsonl`.

`matchup_matrix.json` and the curriculum scheduler's prioritized matchmaking now receive
learner win/loss/draw outcomes from every finished scripted-opponent training episode. Held-out
evaluations stay separate, and self-play episodes are not counted.

`--observation-mode screen` trains on the screen-only contract `screen-matchup-95-v1`, which is
defined in `include/t8_v2/screen_observation.hpp`. Every input is something the live screen
pipeline can measure:

- health from the HUD;
- fighter positions, with distance and velocity derived from them;
- health-drop hit events;
- two binary detections per fighter: activity (animating) and attack cue (an attack is visibly
  coming out).

The 8-step history drops the opponent's move ID, hit level, stance, and repeat count, and holds
observed opponent activity and attack cue, the two uncertainty levels, how long the opponent has
been visibly active, outcome, distance, and opponent velocity. The `visual` mode, by contrast,
derives its attack likelihood from exact move range and active frames and puts the true move ID in
the history. Neither exists live.

Measurement error is simulated per environment lane:

- Gaussian position noise with sigma drawn from [0, `--screen-position-sigma-max`] (default 1.0
  stage units);
- small HUD health noise;
- detection flips at a rate drawn from [0, `--screen-event-error-max`] (default 0.30).

Each lane's two levels are also inputs, so the policy learns how much to trust its measurements.
Live, the same inputs come from `config/live_screen.yaml` (`screen_position_sigma`,
`screen_event_error`). Measure them with `scripts/calibrate_screen_uncertainty.py`: the `distance`
subcommand compares against the Practice-mode distance readout, and `events` compares against a
hand-labelled recording. `live_vision_play.py` refuses a screen checkpoint until both values are
set.

Every held-out evaluation also writes three flat CSVs to the run directory:

- `evaluations.csv`: one row per evaluation, with deterministic and stochastic win rate, P1/P2
  split, draw rate, weakest style, mean damage, the self-play opponents in use, and the guard
  action;
- `evaluation_styles.csv`: per policy, side, and style, with episodes, wins, losses, and draws;
- `evaluation_characters.csv`: per opponent roster character. With 256 episodes this is about 6
  per character, so use `--probe-checkpoint` with more episodes for character-level conclusions.

On resume, rows written after the resume checkpoint are dropped. For runs from before these
exports, `python tools\export_evaluations.py <run-dir>` backfills the first two files from
`metrics.jsonl` with identical columns (per-character data was never recorded for those runs).
`tools\render_readme_figures.ps1` reads `evaluations.csv`.

Multi-seed runs: `scripts\run_seed_matrix.ps1` trains one run per seed with identical options.
The defaults are seeds 2027–2031, 36,000 updates, and the overnight run's shape with
`--curriculum-updates 2000`. Seeds run one after another. Re-running the same command skips
finished seeds and resumes partial ones, and a regression-guard pause is recorded before moving
to the next seed. Afterwards it runs `tools\aggregate_seed_matrix.py`, which reports each seed's
final (last-10) score, peak, minimum, maximum drawdown, fraction of evaluations at 80% or above,
side gap, weakest style, and guard actions, plus the median, mean with a 95% CI, and range across
seeds. The report states whether a learning-quality claim is supported: at least five seeds, all
reaching the target, with a target of at least 10,000 updates.

`--probe-checkpoint <file> [--probe-episodes N]` runs the held-out suite on one checkpoint with
the same run-defining options and prints a JSON report of win rates, per-side choice fraction,
per-choice entropy, and action mix. It does not write anything. See
[selfplay_oscillation_diagnosis.md](selfplay_oscillation_diagnosis.md).

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
