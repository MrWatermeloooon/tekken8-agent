# Tekken 8 Agent

Local-only research scaffold for building a Tekken 8 self-play agent.

This repository is for offline lab work and local versus testing only. Do not use
it for online play, matchmaking, anti-cheat bypassing, or anything that violates
the game or platform rules.

## Roadmap Focus

The source roadmap says Tekken 8 is the capstone because there is no ready-made
Gym environment. We are now taking the faster route: build a compact
Jun-focused simulator first, train self-play there at high speed, then use
DIAMBRA and real Tekken 8 as validation/calibration targets.

1. Build a fast Tekken-lite simulator with spacing, walls, frame data, blocking,
   whiffs, hitstun/blockstun, and round outcomes.
2. Train self-play policies in many parallel simulator instances.
3. Use DIAMBRA/Tekken Tag as a fighting-game reference environment.
4. Use Tekken 8 computer vision and virtual controller input for final
   validation and live local play.

Current target choices:

- Usage: offline local training and local versus only.
- Tekken 8 character: Jun Kazama.
- Main training backend: fast surrogate simulator.
- Tekken 8 state extraction: computer vision first.
- DIAMBRA role: optional/reference on-ramp, not the main training bottleneck.

## Current Scaffold

- `src/t8_agent/core/types.py` defines the state/action/reward data shapes.
- `src/t8_agent/env/mock_env.py` is a fake environment for testing loops before
  Tekken integration works.
- `src/t8_agent/sim/tekken_lite.py` is the fast surrogate simulator for
  simulator-first self-play.
- `src/t8_agent/sim/moves.py` contains the first Jun-style frame-data move
  table.
- `src/t8_agent/sim/opponents.py` contains the default hard scripted
  curriculum: poke, rushdown, turtle, whiff-punish, keepout, frame-trap, and
  anti-throw.
- `data/characters/jun.yaml` starts the Jun character catalog: core moves,
  training tags, and starter combo routes sourced from TekkenDocs.
- `src/t8_agent/io/input_backend.py` defines the controller output interface.
- `src/t8_agent/io/state_backend.py` defines the game-state reader interface.
- `scripts/run_sim_episode.py` runs fast simulator episodes without Tekken 8.
- `scripts/diambra_random_episode.py` runs a DIAMBRA random-policy smoke test.
- `scripts/check_setup.py` reports local dependency/setup status.
- `docs/tekken8_implementation_plan.md` captures the Tekken-only plan from the
  PDF roadmap.
- `docs/diambra_onramp.md` captures the DIAMBRA setup and first milestones.

## Quick Smoke Test

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
python scripts/run_sim_episode.py --episodes 5
```

## First Training

This starts the lightweight simulator trainer and writes a checkpoint:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_linear.py --generations 8 --population 16 --episodes-per-candidate 3 --eval-episodes 10 --max-decisions 1000 --checkpoint checkpoints\sim_linear_policy.npz
```

This is prototype training, not the final bot. It is meant to verify that the
simulator has a learnable signal before we add PPO and larger self-play pools.
The default training opponents intentionally exclude the old random policy
because it saturated too quickly.

PPO training is available through MaskablePPO:

```powershell
.\.venv\Scripts\python -m pip install -e ".[rl]"
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo.py --timesteps 20000 --checkpoint checkpoints\sim_ppo_policy.zip
```

Checkpoint-pool PPO self-play:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 4 --timesteps-per-iteration 5000
```

Use Elo-weighted checkpoint sampling:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 4 --timesteps-per-iteration 5000 --elo-sampling
```

Plot curves and rank checkpoint pools:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\plot_training_curves.py --metrics runs\ppo_selfplay_latest\metrics.json
.\.venv\Scripts\python scripts\rank_ppo_checkpoints.py --checkpoint-dir checkpoints\selfplay
```

Evaluate and bug-check:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\evaluate_sim_policy.py --checkpoint checkpoints\sim_linear_policy.npz --episodes 50
.\.venv\Scripts\python scripts\bug_check.py
```

## Visualizer

Watch the trained simulator policy fight a scripted opponent:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\visualize_sim.py --checkpoint checkpoints\sim_linear_policy.npz
```

Controls:

- `Space`: pause/resume
- `R`: reset episode
- `+` / `-`: speed up or slow down

The older mock environment still exists for interface tests:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
python -m t8_agent.env.mock_env
```

## Offline Live Test

The repo now includes a first real-game bridge for offline/local testing:
screen capture through `dxcam`, virtual controller output through `vgamepad`,
and an `F8` hotkey toggle.

```powershell
.\.venv\Scripts\python -m pip install -e ".[live]"
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\live_play.py --dry-run
```

See `docs/live_game_test.md` before enabling controller output. The current live
agent is a scripted controller/screen test; calibrated CV is still needed before
the simulator PPO can play from real Tekken 8 pixels.

## Character Curriculum

Jun should become strong first, then we add one matchup character at a time.
Inspect the current Jun catalog with:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\character_catalog.py data\characters\jun.yaml
```

See `docs/character_curriculum.md` for the full character-by-character plan.

## DIAMBRA On-Ramp

DIAMBRA currently requires a free account, Docker Desktop, the `diambra` CLI,
`diambra-arena`, and a valid supported ROM. The repo does not include ROMs.

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/check_setup.py
python -m diambra arena list-roms
python -m diambra arena check-roms C:\path\to\roms\tektagt.zip
python -m diambra run -r C:\path\to\roms python scripts/diambra_random_episode.py --game tektagt --characters Jun Jin --render
```

## Next Decisions

Before wiring the real game, decide:

- Which Jun TekkenDocs entries should be promoted from raw catalog to simulator
  actions first?
- Which reward terms still create abusable behavior after the harder scripted
  curriculum?
- Which real Tekken 8 observations should calibrate the simulator first:
  movement speed, health/damage, move timing, or wall spacing?
