# Tekken 8 Agent

A reinforcement-learning bot for Tekken 8. A CUDA simulator trains one policy that plays all 41
sourced fighters with their full move lists. The same policy then plays the real game in offline
modes, from screen capture through a virtual controller.

> [!WARNING]
> **Experimental and unfinished.** The simulator, trainer and live tools work, but the simulator
> runs on partly provisional data. The bot has not yet been tested in full matches against the
> real game. Simulator win rates are not real-game strength.

> [!IMPORTANT]
> The bot never reads or modifies the game's memory or files. It sees only what is on screen and
> plays only through a virtual controller. Use it only in offline modes (Practice, local versus)
> where automation is allowed.

## How it works

```text
 move data (6,393 moves, 41 fighters)            screen capture (60 fps)
              |                                           |
   move compiler + binder                  pose tracker, HUD and Practice readers
              |                                           |
   frame-stepped combat engine                screen observation (same features
   (one CUDA/C++ source, bitwise parity)       the trainer used, with its noise)
              |                                           |
   PPO self-play, 32,768 fights at once                   |
              |                                           |
              +-------------> one policy <----------------+
                                   |
                 scores every legal move of the fighter
                                   |
                     virtual Xbox pad (ViGEm) -> Tekken 8
```

- **One policy, every fighter.** Each fighter's moves are encoded by their properties (startup,
  recovery, advantage, hit levels, damage, crush, Heat and Rage use, and more). The policy scores
  every move that is legal right now and picks one. It learns from these properties, not from move
  IDs, so it can learn its own combos.
- **The engine is one source for CPU and GPU.** The frame step is written once as
  `__host__ __device__` code. The CPU engine and the CUDA simulator run it with bitwise-identical
  results (checked on randomized fights with `--fmad=false`).
- **Trained the way it will play.** The live bot is trained only on what the screen can show:
  positions, posture, health, recoverable health, its own move in progress and the opponent's
  pose. The live setup's flaws are simulated too: position noise, misreads, 2 to 6 frames of
  capture latency, a 12-frame perception delay and 8-frame button holds. A policy trained without
  this noise lost badly when tested with it.

## Status

| Part | State |
|---|---|
| Move data | 6,393 moves for 41 fighters (Bob has no source data). About 98.5% bind to the engine; the 96 that don't are listed by `t8_v2_full_move_inspect --all --list-unbound`. |
| Combat engine | Launches, juggles, walls, floors, scaling, Heat, Rage, armor, crushes, parries, throws and throw breaks, recoverable health, stances and back-turned states. Much geometry (range, tracking, active frames) still uses stand-in values. |
| Training | Main run: 44.4B steps, about 575K steps/s on an RTX 5070 Ti, screen-only observation with live noise (`full-move-screen-noise-151-v4`). |
| Practice validation | 925 moves confirmed in the game (startup and damage) for 12 fighters: Jun, Jin, Kazuya, King, Paul, Law, Lars, Xiaoyu, Leroy, Asuka, Lili, Bryan. |
| Live perception | Pose tracking, posture, health and recoverable health work. Distance is calibrated to 0.28 m RMS. The Heat gauge read is unreliable and is not used. |
| Opponent-move recognizer | First version: predicts an incoming attack, its height and its timing from the opponent's pose. On moves it never saw: recall 0.84, precision 0.72, level accuracy 0.58, timing error about 6 frames. Not yet part of the policy's input. |
| Full matches in the real game | Not done yet. |

### What the experiments showed

Each comparison is a fork of the same run, trained for the same number of steps, then played
head-to-head in the simulator over several seeds.

- **Entropy floor 1.0** beat 1.5 in 70% of games. 0.5 won more often (79%) but leaned on one move.
- **Training with live noise** is essential: in noisy fights the noise-trained policy won 86%
  against a clean-trained one.
- **Screen-only observation** costs a lot without the opponent's move: the screen-only bot won
  only about 6% against a noise-trained bot that sees everything. That missing input is what the
  move recognizer is for.
- **A memory of recent moves** and **self-play scenarios** gave no measurable gain. They are kept
  as opt-in flags.
- **Deeper networks** (3 layers, or separate heads) gave nothing at 5B steps and ran 25% slower,
  so the policy stays at 2 × 512.

## Repository layout

| Path | Contents |
|---|---|
| `include/t8_v2/`, `src/*.cpp`, `src/*.cu` | Combat engine, CUDA simulator, PPO, trainers, arena, inspect tool, T8Train launcher |
| `src/t8_agent/moves/` | Move compiler, command-notation parser, legality, validation and provenance |
| `src/t8_agent/live/` | Live runtime: observation builder, Practice panel and HUD readers, menu automation |
| `src/t8_agent/io/` | Screen capture, pose tracker, virtual-controller backends (Windows and Linux) |
| `data/characters/` | Saved move sources per fighter |
| `data/generated/` | Compiled catalog, engine bindings, stances and candidate lists |
| `data/character_modules/` | Stance and resource state machines per fighter |
| `data/validation/` | Moves confirmed in Practice mode, with provenance |
| `scripts/` | Training, visualizers, calibration, move validation, data collection, live play |
| `tools/` | Catalog building, evaluation, ranking, sweeps, recognizer training |
| `tests/`, `benchmarks/` | C++, CUDA and Python tests; throughput benchmarks |

## Requirements

- Windows 10 or 11 with Visual Studio 2022 (Desktop development with C++), or Linux with GCC 13+
- NVIDIA GPU and CUDA Toolkit 12.8+ (13.x tested; Blackwell needs 12.8+)
- CMake 3.24+ and Python 3.10+

For live play you also need Tekken 8 and the ViGEmBus driver. The live Python packages (DXcam,
OpenCV, `keyboard`, `vgamepad`, PyTorch) install with the `live` extra. The pose model
(YOLOv8n-pose through `ultralytics`) is downloaded separately into `models/`.

## Build and test

```powershell
cmake -S . -B build-gpu -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build-gpu --config Release --parallel
python -m pip install -e ".[dev]"
```

Use the architecture of your GPU (120 is the RTX 50 series). Without CUDA,
`-DT8_V2_ENABLE_CUDA=OFF` builds and tests only the CPU parts.

```powershell
ctest --test-dir build-gpu -C Release --output-on-failure
python -m pytest tests -q
```

`run_all_tests.bat` runs everything that doesn't need the game, one step after another: build,
C++ and Python tests, CPU-only build, data reproducibility, Compute Sanitizer and training smokes.
It prints a PASS/FAIL summary and writes logs to `test_logs/`. Tests that need a GPU have the
CTest label `gpu`.

## Training

Run `T8Train.exe` from the repository root. On first run it writes `training.ini`, then trains
one policy for all fighters in `runs/train_all`. It resumes automatically, keeps the PC awake and
shows a live report. Keys while it runs:

| Key | Action |
|---|---|
| `V` | Open the fight visualizer on the current champion |
| `N` | Open the neural-network visualizer |
| `C` | Rank the champion's characters (which it plays best) |
| `Q` | Save a checkpoint and quit |

Every 500M steps the newest checkpoint plays the current champion. It is promoted at 53% or
better; after two results under 45%, training rolls back to the champion.

The defaults in `training.ini` are the winners of the experiments above:
- PPO: 32,768 environments, minibatch 16,384, 2 × 512 network, learning rate 5e-4 annealed to
  1e-4, TF32, entropy floor 1.0;
- self-play with an annealed opponent pool;
- screen observation with live noise, 8-frame holds and 12-frame perception delay.

The trainer can also be run directly:

```powershell
build-gpu\Release\t8_v2_full_move_train.exe --help
```

Useful options:
- `--fork CHECKPOINT` starts an A/B branch from an existing run;
- `--memory`, `--live-noise`, `--screen-observation` and `--self-play-scenarios` toggle the
  experimental inputs;
- `--exploit-target CHECKPOINT` trains a best response against a frozen policy.

## Evaluating

```powershell
build-gpu\Release\t8_v2_full_move_arena.exe --allow-provisional --mixed --sample `
  --perception-delay 12 --hold-frames 8 --output arena.csv a.t8ppo b.t8ppo
python tools\rank_checkpoints.py arena.csv            # Bradley-Terry ranking with bootstrap intervals
python tools\character_ranking.py --run runs\train_all # which fighter the policy plays best
python tools\fundamentals.py --run runs\train_all      # throws, throw breaks, punishes, counter-attacks
python tools\move_usage.py --run runs\train_all        # how varied the policy's move choice is
build-gpu\Release\t8_v2_full_move_inspect.exe --all --blockers
```

Scripted bots are no longer a useful benchmark, because trained policies beat every tier almost
100% of the time. Strength is measured head-to-head in the arena across several seeds.

## Visualizers

```powershell
.venv\Scripts\python.exe scripts\visualizer.py --run runs\train_all
network_visualizer.bat
```

`visualizer.py` opens a desktop window where the champion fights itself across random matchups.
It shows a Tekken-style HUD, the move names and frame data, and the policy's top choices with
their probabilities.

`network_visualizer.bat` shows the policy as a classic node-and-edge network, lit up by each real
decision. It reloads automatically when a new champion appears.

## Real game (offline only)

Start Tekken 8 in Practice mode and unplug your own controller; the virtual pad takes over.

```powershell
python -m pip install -e ".[live]"
python scripts\check_live_setup.py
python scripts\auto_calibrate_distance.py --write-config  # fits screen distance to the Practice readout
python scripts\validate_moves.py --character jun --write-ledger
python scripts\collect_characters.py                      # every fighter: validation + pose recordings
python tools\train_move_recognizer.py                     # opponent-move recognizer from the recordings
python scripts\live_vision_play.py --dry-run --agent full-move `
  --ppo-checkpoint runs\train_all\champion\update_20504.t8ppo --controlled-character jun
```

- `validate_moves.py` performs each move against the dummy and checks the Practice panel's startup
  and damage against the data. Matching moves are written to `data/validation/`.
- `collect_characters.py` switches fighters through the pause menu by reading the names on the
  select screen. For each fighter it validates the moves and records both fighters' pose skeletons.
  Progress is kept, so a stopped run resumes where it left off.
- `live_vision_play.py` refuses checkpoints whose fighter hasn't passed validation, except in
  `--dry-run`. In the game, `F8` pauses or resumes controller output and `F7` flips facing. Output
  stops on its own whenever fresh frames stop arriving.

Screen regions are set in `config/live_screen.yaml`; `config/live_screen.example.yaml` documents
every key.

## Known limits and next steps

- **Move geometry is provisional.** Range, tracking, active frames and some recovery are stand-in
  values. Engine-wide constants (scaling, Heat duration, juggle timing) are not measured yet.
- **Practice validation covers 12 fighters** and only moves done from a standing start. Stance,
  crouch, Heat, Rage and sidestep moves need the controller to set up the state first.
- **The overnight collector got stuck** at character select after Bryan. The remaining 29 fighters
  still need that navigation fixed and a rerun.
- **The opponent-move recognizer** was trained on two fighters. It needs the full dataset, then to
  be wired into the screen observation (features 50-57), followed by a retrain.
- **The engine holds one stance at a time.** Moves that need a mode and a stance together
  (Yoshimitsu's NSS.KIN, Heihachi's Warrior Instinct versions) are not used. Counter-hit-only and
  airborne-only follow-ups are not modeled either.
- **Full real-game matches** against the CPU and in local versus are the next milestone.

## Author

Branden Agnello ([@MrWatermeloooon](https://github.com/MrWatermeloooon))

Tekken and its characters are the property of Bandai Namco Entertainment. This is an independent
project and is not affiliated with or endorsed by Bandai Namco. Move data comes from community
frame-data sources and the author's own Practice-mode measurements.
