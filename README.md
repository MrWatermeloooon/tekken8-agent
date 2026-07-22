# Tekken 8 Agent V2

GPU-first reinforcement learning for training one Jun policy against a full-roster,
character-conditioned Tekken 8 simulation.

V2 contains a custom CUDA environment, scripted opponent population, temporal matchup encoder,
PPO trainer, evaluation system, exact checkpoint/resume support, and a screen-based live inference
runtime. The rollout and optimization hot path stays in GPU memory.

> [!IMPORTANT]
> This repository does **not** emulate Tekken 8 or provide direct access to its internal game state.
> Training happens in a purpose-built, deterministic Tekken-like simulator. The live runtime reads
> screen-derived features and sends controller inputs. Policies still require real-game validation.

V1 is preserved on the [`v1` branch](https://github.com/MrWatermeloooon/tekken8-agent/tree/v1).
The `v2` branch is the default branch and contains all current work; there are no duplicated
`v1/` or `v2/` source directories.

## Current capabilities

- Jun learner versus all 42 fighters in the 2026-07-21 playable-roster snapshot.
- 50 probabilistic profiles per character: 10 archetypes x 5 variations, or 2,100 total.
- Character-specific six-move CUDA combat tables distilled from public frame data.
- Device-resident simulation, observations, profile assignments, actions, rewards, rollouts,
  generalized advantage estimation, and PPO optimization.
- Side-balanced training with the same matchup represented as learner P1 and learner P2.
- 95-feature deployable visual-matchup observations and 101-feature privileged observations.
- Eight-decision opponent history covering action/move identity, animation phase, stance, hit level,
  delay, outcome, distance, and lateral movement.
- Four-stage curriculum with uncertainty, weakness, regression, and exploit-severity scheduling.
- Per-character/archetype evaluation, draw-aware scores, matchup Elo, behavior metrics, and
  catastrophic-forgetting detection.
- Checksummed atomic policy checkpoints and byte-exact trainer resume state.
- Torch CPU/CUDA live inference for legacy 13-feature and roster-temporal 95-feature checkpoints.
- CPU/CUDA parity, PPO numerical tests, exact-resume tests, smoke training, and sanitizer workflows.

## Architecture

```text
roster + frame data + archetypes
              |
              v
      2,100 opponent profiles
              |
              v
CUDA simulator -> temporal matchup encoder -> CUDA actor/critic
      |                                         |
      +------------ rewards + masks ------------+
                                                |
                                                v
                                     rollout -> GAE -> PPO/Adam
                                                |
                          checkpoints + metrics + matchup matrix
                                                |
                                                v
                                  screen-based live inference
```

The scalar C++ simulator is a deterministic correctness oracle. It is not used for production
rollout collection.

## Repository layout

| Path | Purpose |
|---|---|
| `src/gpu_sim.cu` | Massively batched character-aware CUDA simulator |
| `src/opponents.cu` | GPU scripted/profiled opponent population |
| `src/temporal.cu` | GPU character, archetype, and temporal observation encoder |
| `src/train.cpp` | Native PPO training, evaluation, checkpointing, and resume |
| `src/t8_agent/roster/` | Python catalog, scheduler, temporal encoder, and evaluation exports |
| `src/t8_agent/live/` | Native checkpoint loader and live Torch inference |
| `data/characters/` | Imported character frame-data snapshots with provenance |
| `data/character_modules/` | Generated matchup knowledge for every character |
| `data/generated/` | Runtime profile/move catalogs and SHA-256 manifest |
| `tests/` | CPU, CUDA, PPO, resume, roster, and live-runtime tests |
| `tools/` | Reproducible roster import and catalog-generation utilities |

## Requirements

- Windows 10/11
- NVIDIA GPU
- CUDA Toolkit 13.1 or a compatible toolkit
- Visual Studio 2022 with Desktop development with C++
- CMake 3.24+
- Python 3.10+

The default CMake configuration emits native images for Turing, Ampere, Ada, and Blackwell, with
PTX in the newest image. For the validated RTX 5070 Ti build, use CUDA architecture `120`.

## Build and test

```powershell
git switch v2
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
ctest --test-dir build -C Release --output-on-failure

python -m pip install -e ".[dev]"
python -m pytest tests -q
```

For another GPU, omit `-DCMAKE_CUDA_ARCHITECTURES=120` to use the repository defaults or pass the
architecture appropriate for that device.

## Rebuild the roster data

The playable roster comes from the
[official Tekken fighter list](https://tekken.com/fighters). Frame data is imported from the public
[TekkenDocs API](https://tekkendocs.com/api/t8/characters) with attribution, retrieval dates, and
source hashes. The source API currently has no Bob frame-data entry, so Bob is explicitly marked
unavailable and receives documented abstract values only for the six simulator move slots.

```powershell
python tools\import_roster_frame_data.py --allow-missing
python tools\generate_opponent_catalog.py --data-root data
```

Generation must produce:

- 42 characters
- 50 profiles per character
- 2,100 total profiles
- 252 character-move rows

The exact counts and hashes are stored in `data/generated/manifest.json`.

## Training

### Fast smoke test

Use a new run directory each time; V2 intentionally refuses to overwrite training artifacts.

```powershell
build\Release\t8_v2_train.exe --smoke `
  --opponents roster `
  --observation-mode visual `
  --reward shaped `
  --seed 2027 `
  --run-dir runs\smoke_roster_visual_2027
```

### Full run

```powershell
build\Release\t8_v2_train.exe `
  --envs 4096 `
  --horizon 128 `
  --updates 100 `
  --epochs 4 `
  --minibatch 4096 `
  --opponents roster `
  --curriculum-stage auto `
  --observation-mode visual `
  --reward shaped `
  --seed 2027 `
  --run-dir runs\roster_visual_shaped_2027
```

### Resume exactly

Use the same run-defining options. `--updates` is the final target update, not the number of
additional updates.

```powershell
build\Release\t8_v2_train.exe `
  --envs 4096 `
  --horizon 128 `
  --updates 100 `
  --epochs 4 `
  --minibatch 4096 `
  --opponents roster `
  --curriculum-stage auto `
  --observation-mode visual `
  --reward shaped `
  --seed 2027 `
  --run-dir runs\roster_visual_shaped_2027 `
  --resume runs\roster_visual_shaped_2027\checkpoints\update_40.t8ppo
```

V2 rejects option drift, corrupted checkpoints, incomplete metric rows, and accidental artifact
overwrite.

### Observation modes

| Opponents | Mode | Features | Intended use |
|---|---:|---:|---|
| `roster` | `visual` | 95 | Deployable roster-temporal policy |
| `roster` | `privileged` | 101 | Simulator teacher/oracle |
| `legacy` | `visual` | 13 | Controlled legacy comparison |
| `legacy` | `privileged` | 19 | Controlled legacy oracle comparison |

### Curriculum stages

1. Jun fundamentals
2. Character groups
3. Full roster
4. Adversarial/weakness-focused league

Use `--curriculum-stage auto` for the four-stage schedule or pin a stage with `1`, `2`, `3`, or
`4` for a controlled experiment.

### Run artifacts

Each run directory contains:

- `metrics.jsonl` — ordered PPO and evaluation metrics
- `matchup_matrix.json` — all character/archetype outcomes, scores, Elo, and forgetting flags
- `checkpoints/update_N.t8ppo` — integrity-protected policy and optimizer state
- `checkpoints/update_N.t8state` — exact simulator, opponent, scheduler, temporal, and RNG state

## Live screen inference

Install the CUDA-enabled Torch build explicitly before installing the live dependencies. This
prevents pip from silently selecting a CPU-only wheel.

```powershell
python -m pip uninstall -y torch
python -m pip install --no-cache-dir torch==2.13.0 `
  --index-url https://download.pytorch.org/whl/cu130
python -m pip install -e ".[live]"
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"

Copy-Item config\live_screen.example.yaml config\live_screen.yaml
python scripts\live_vision_play.py --dry-run --agent v2 `
  --ppo-checkpoint runs\roster_visual_shaped_2027\checkpoints\update_100.t8ppo `
  --opponent-character reina `
  --opponent-archetype movement_specialist
```

A 95-feature checkpoint requires the current opponent character and archetype. A legacy 13-feature
checkpoint does not. Start with `--dry-run`; controller output is paused by default. Confirm screen
capture, player side, and facing before enabling controller output.

## Should this project use DIAMBRA?

**Not as the primary Tekken 8 training environment.** DIAMBRA Arena provides a polished
Gymnasium-compatible API, pixels plus RAM observations, one/two-player modes, and self-play-friendly
retro fighting-game environments. However, its official game list includes
[Tekken Tag Tournament](https://docs.diambra.ai/envs/games/tektagt/), not Tekken 8. It also runs
emulated games through a Docker-oriented environment stack rather than this project's
massively-batched CUDA simulator.

DIAMBRA could still be useful as an **optional, isolated research benchmark** for:

- validating generic fighting-game representations;
- testing Gymnasium-compatible wrappers;
- comparing league/self-play scheduling on a real emulated game;
- pretraining visual encoders before Tekken 8-specific fine-tuning.

It should not become a core dependency or replace the current simulator. Tekken Tag Tournament has
a different roster, tag mechanics, observation contract, and action space, so direct policy or
matchup-knowledge transfer would be unreliable. See the
[DIAMBRA overview](https://docs.diambra.ai/) and
[official supported-game list](https://docs.diambra.ai/envs/games/) for current details.

## Benchmarks

```powershell
build\Release\t8_v2_gpu_benchmark.exe --envs 262144 --steps 2000
build\Release\t8_v2_training_benchmark.exe --envs 4096 --horizon 128 --updates 5
```

The earlier 13-feature visual build measured a five-run median of 1.95 million environment
decisions/s and 9.08 million PPO sample-visits/s on an RTX 5070 Ti. The simulator-only median was
425.64 million decisions/s (1.70 billion simulated frames/s). These are historical local results,
not guarantees for the current roster-temporal configuration. Rebenchmark the exact commit and
hardware used for an experiment.

## Documentation

- [Full-roster curriculum](docs/roster_curriculum.md)
- [Training and evaluation](docs/training.md)
- [Roadmap and promotion gates](docs/roadmap.md)
- [Frame-data provenance](data/README.md)
- [Held-out Phase 0 report](docs/phase0_heldout_v2_visual_report.md)

## Safety and scope

- Use live input only where automation is allowed.
- Keep the live controller paused until capture and directional facing are verified.
- Treat simulator performance as a hypothesis to test in the real game, not proof of mastery.
- Tekken and its characters are property of their respective owners; this repository is an
  independent research project.
