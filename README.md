# Tekken 8 Agent V3

A CUDA-first reinforcement-learning project for training character-aware fighting-game policies,
validating them in a deterministic simulator, and testing them against Tekken 8 through screen
capture and virtual-controller output.

> [!IMPORTANT]
> This project does not read Tekken 8 memory, modify the game, or reproduce the complete game
> engine. Training uses an independent Tekken-like simulator. Live play uses only screen-derived
> observations and controller inputs, and must be tested in offline modes where automation is
> permitted.

## Project status

V3 introduces the data and policy architecture needed to move beyond a fixed six-attack action
space:

- 6,393 documented move rows compiled across 41 sourced fighters;
- stable character-specific move IDs and a versioned move-catalog contract;
- 18 universal movement and defense actions plus variable per-character move candidates;
- a CUDA parametric PPO policy that scores legal move-property vectors;
- legal-action masks for recovery, posture, stance, Heat, Rage, and character resources;
- a scalar combat oracle for launches, combos, scaling, tornado, walls, armor, crushes, parries,
  throws, recoverable health, Heat, and Rage;
- deterministic command execution for motions, chords, holds, releases, strings, stance inputs,
  just-frame separators, and facing conversion;
- strict per-character validation reports and offline Practice-mode command checks;
- screen-based live inference with capture freshness checks, persistent guard/movement holds,
  cancellable action sequences, and round-history reset.

The full-move path is deliberately gated. Imported frame data does not contain every measurement
needed to model collision, tracking, active frames, transitions, or character mechanics exactly.
A fighter is not enabled for full-move training until its data, CPU scenarios, CUDA parity,
representative combo routes, and offline Practice validation all pass.

The current native trainer still provides a compatibility rollout path using six combat slots per
character. It is useful for PPO, self-play, scheduling, evaluation, and throughput work, but it is
not presented as complete full-moveset Tekken training.

## Architecture

```text
documented move data              screen capture
          |                             |
          v                             v
 versioned move compiler        observable-state estimator
          |                             |
          v                             |
character move candidates              |
          |                             |
          +----------+------------------+
                     v
          shared state encoder
                     |
          parametric move scorer
                     |
                     v
             legal move/action
                     |
          +----------+-----------+
          |                      |
          v                      v
   CUDA simulation      controller command executor
          |                      |
          v                      v
 PPO + self-play             Tekken 8 offline
```

The value function is state-only. The policy builds a state query, compares it with each legal
candidate's encoded properties, and samples only from the current fighter's valid candidates.
Checkpoint metadata binds a policy to its roster version, move-catalog hash, observation contract,
action contract, and feature dimensions.

## Repository map

| Path | Purpose |
|---|---|
| `src/train.cpp` | Native PPO training, evaluation, self-play, checkpointing, and exact resume |
| `src/ppo.cu` | CUDA actor-critic, parametric move scoring, GAE, PPO, and Adam |
| `src/gpu_sim.cu` | Batched compatibility simulator used for rollout collection |
| `src/full_combat.cpp` | Scalar full-combat correctness oracle |
| `src/opponents.cu` | Scripted profiles and checkpoint-opponent routing |
| `src/temporal.cu` | Character and temporal observation encoding |
| `src/t8_agent/moves/` | Move compiler, notation parser, masks, and validation gates |
| `src/t8_agent/live/` | Checkpoint loading and screen-observable live policy runtime |
| `src/t8_agent/io/` | Screen capture and virtual-controller backends |
| `data/characters/` | Saved character move sources and provenance |
| `data/generated/` | Compiled catalogs, reports, profiles, and hashes |
| `scripts/` | Training, visualization, calibration, validation, and live-play entry points |
| `tests/` | CPU, CUDA, PPO, resume, catalog, controller, vision, and runtime tests |

## Requirements

Core native training:

- Windows 10 or 11
- NVIDIA GPU
- CUDA Toolkit 13.1 or a compatible toolkit
- Visual Studio 2022 with Desktop development with C++
- CMake 3.24 or newer
- Python 3.10 or newer

Live testing additionally requires DXcam, OpenCV, `keyboard`, `vgamepad`, and a functioning
ViGEmBus installation. Live dependencies are not required for simulator training.

## Build

For an RTX 5070 Ti or another Blackwell GPU using architecture 120:

```powershell
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
```

For another GPU, omit `-DCMAKE_CUDA_ARCHITECTURES=120` to use the repository defaults or provide
the architecture matching that device.

Install the Python development environment:

```powershell
python -m pip install -e ".[dev]"
```

## Test

```powershell
ctest --test-dir build -C Release --output-on-failure
python -m pytest tests -q
```

The native suite covers scalar behavior, CUDA parity, policy inference, PPO updates, exact resume,
catalog loading, full-combat mechanics, and headless visualization. The Python suite covers move
compilation, legality rules, live checkpoint loading, screen freshness, controller cancellation,
runtime state, roster scheduling, and optional vision components.

## Move catalog

Rebuild the saved roster and compiled catalogs:

```powershell
python tools\import_roster_frame_data.py --allow-missing
python tools\generate_opponent_catalog.py --data-root data
python tools\compile_full_move_catalog.py
```

Expected generated totals:

| Item | Count |
|---|---:|
| Roster records | 42 |
| Fighters with sourced move data | 41 |
| Documented move rows | 6,393 |
| Opponent profiles | 2,100 |
| Universal actions | 18 |

Bob remains explicitly unavailable because the saved source snapshot has no move endpoint. No
generic substitute is generated. Roger Jr. is not treated as a playable fighter until released,
documented data is available, and the same validation gates pass.

Inspect a fighter's gate status:

```powershell
python tools\check_character_gate.py jun
python scripts\validate_move_commands.py jun --dry-run
```

Missing or contradictory fields remain visible and block promotion. They are never silently
replaced with generic attack properties.

## Offline move validation

With Tekken 8 open in Practice mode and its input-history display visible:

```powershell
python scripts\validate_move_commands.py jun --confirm-offline
```

The validator walks through compiled commands and records results under `data/validation/`:

- `F8`: pass
- `F9`: fail
- `F10`: skip
- `Esc`: stop safely

Validation records are written atomically. A parsed command is not automatically considered a
correct in-game command.

## Training

Run a short compatibility smoke test:

```powershell
build\Release\t8_v2_train.exe --smoke `
  --opponents roster `
  --learner-character jun `
  --observation-mode visual `
  --reward shaped `
  --seed 2027 `
  --run-dir runs\smoke_jun_2027
```

Start a larger run and launch the checkpoint-following visualizer:

```powershell
powershell -NoProfile -ExecutionPolicy Bypass `
  -File scripts\start_full_training.ps1 `
  -RunDir runs\jun_visual_2027 `
  -Seed 2027
```

Equivalent native command:

```powershell
build\Release\t8_v2_train.exe `
  --envs 4096 `
  --horizon 128 `
  --updates 100 `
  --curriculum-updates 100 `
  --epochs 4 `
  --minibatch 4096 `
  --opponents roster `
  --learner-character jun `
  --full-move-catalog data\generated\full_move_catalog.csv `
  --curriculum-stage auto `
  --observation-mode visual `
  --reward shaped `
  --seed 2027 `
  --run-dir runs\jun_visual_2027
```

`--learner-character all` remains blocked until at least one complete full-move character passes
measured CPU/CUDA parity and Practice validation. This prevents a compatibility run from being
mistaken for full-roster move training.

### Self-play and evaluation

The final curriculum stage uses frozen policy checkpoints as opponents:

- 80% latest-self checkpoint;
- 20% strongest older checkpoint selected by held-out evaluation.

Training balances the learner across P1 and P2. Held-out profiles are separate from training
matchmaking and do not change the training scheduler. Metrics include reward, win/draw/loss rates,
matchup results, behavior statistics, Elo, and forgetting checks.

### Resume

Use the same run-defining options and pass both the existing run directory and checkpoint:

```powershell
build\Release\t8_v2_train.exe `
  --envs 4096 `
  --horizon 128 `
  --updates 200 `
  --curriculum-updates 100 `
  --epochs 4 `
  --minibatch 4096 `
  --opponents roster `
  --learner-character jun `
  --full-move-catalog data\generated\full_move_catalog.csv `
  --curriculum-stage auto `
  --observation-mode visual `
  --reward shaped `
  --seed 2027 `
  --run-dir runs\jun_visual_2027 `
  --resume runs\jun_visual_2027\checkpoints\update_100.t8ppo
```

Checkpoint format v3 rejects policies from the previous fixed-action contract with an explicit
incompatibility error. Exact resume also validates trainer options, catalog identity, roster
version, learner character, scheduler state, simulation state, and random-number state.

## Visualizer

Watch a standalone CUDA matchup:

```powershell
.\.venv\Scripts\python.exe scripts\visualize_v2.py `
  --opponent-character reina `
  --opponent-archetype rushdown
```

Follow the newest checkpoint from a training run:

```powershell
.\.venv\Scripts\python.exe scripts\visualize_v2.py `
  --follow-dir runs\jun_visual_2027\checkpoints `
  --observation-mode visual `
  --opponent-character reina `
  --opponent-archetype rushdown
```

The visualizer runs a separate CUDA evaluation feed and does not slow the trainer's rollout path.

## Live offline testing

Install the optional live dependencies and verify the environment:

```powershell
python -m pip install -e ".[live]"
python scripts\calibrate_live_screen.py --automatic
python scripts\check_live_setup.py
```

Begin with controller output disabled:

```powershell
python scripts\live_vision_play.py --dry-run --agent v2 `
  --ppo-checkpoint runs\jun_visual_2027\checkpoints\update_100.t8ppo `
  --player 1 `
  --opponent-character reina `
  --opponent-archetype movement_specialist
```

Controls:

- `F8`: enable or pause controller output
- `F7`: reverse facing after a side switch
- `F6`: manually reset policy and perception history

The live runtime pauses output when fresh capture frames stop arriving. Movement and guard are held
persistently between decisions, attack sequences are cancellable, and new rounds reset temporal
history. Start in Practice mode, verify the capture regions and facing, and keep the emergency pause
hotkey reachable.

## Benchmarks

```powershell
build\Release\t8_v2_gpu_benchmark.exe --envs 262144 --steps 2000
build\Release\t8_v2_training_benchmark.exe --envs 4096 --horizon 128 --updates 5
```

Performance depends on the GPU, CUDA build, observation contract, opponent mix, and PPO settings.
Record the exact commit and command alongside benchmark results.

## Documentation

- [Full-roster architecture](docs/full_roster_move_architecture.md)
- [Roster curriculum](docs/roster_curriculum.md)
- [Training and evaluation](docs/training.md)
- [CUDA visualizer](docs/visualizer.md)
- [Live runtime](docs/live_runtime.md)
- [Roadmap](docs/roadmap.md)
- [Frame-data provenance](data/README.md)

## Scope and safety

- Use controller automation only in offline modes or where it is explicitly allowed.
- Do not treat simulator win rate as proof of real-game strength.
- Keep unmeasured mechanics marked unknown until verified against the current game patch.
- Tekken and its characters are property of their respective owners. This is an independent
  research project and is not affiliated with Bandai Namco Entertainment.
