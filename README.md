# Tekken 8 Self-Play Agent

An offline, local-only research project for building a fighting-game agent, using a fast custom simulator as the primary training ground before validating against DIAMBRA Arena and real Tekken 8.

> **Usage policy:** This project is for offline local training and local versus play only. No online play, matchmaking, anti-cheat bypassing, or stealth behavior.

## Why a Custom Simulator?

Training directly against a real game is slow and hard to debug. This project trains first in **Tekken-lite**, a compact fighting-game simulator built from scratch, which gives:

- Episodes that run far faster than real time
- Many parallel environments for self-play
- Easy-to-inspect rewards
- A place to validate policies before touching a real game client

Tekken-lite currently models:

- Health, round timer, and win/loss termination
- 2.5D spacing with stage walls and body collision
- Startup / active / recovery frame data
- High / mid / low / throw hit levels, with high and low blocking
- Hitstun, blockstun, whiffs, and a launch flag
- A small Jun Kazama–style move set: jab, df1, f2, db3, hopkick, throw

## Project Structure

- `src/t8_agent/sim/` — the Tekken-lite simulator, move data, scripted opponents, action space, and observation vectors
- `src/t8_agent/train/` — training code: a CEM-based linear policy baseline and MaskablePPO self-play
- `src/t8_agent/env/` — a minimal mock environment used for testing loops
- `src/t8_agent/io/` — input/state backend interfaces (mock implementations included; real backends are not part of this repo)
- `scripts/` — runnable entry points for training, evaluation, visualization, and setup checks
- `docs/` — additional documentation on the simulator, training, and the visualizer

## Installation

```bash
python -m venv .venv
source .venv/bin/activate    # on Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install -e ".[rl]"        # MaskablePPO / stable-baselines3
pip install -e ".[diambra]"   # DIAMBRA Arena on-ramp
pip install -e ".[cv]"        # computer-vision backend deps
pip install -e ".[gamepad]"   # virtual controller input
```

## Quick Start

Run a few fast simulator episodes with no training required:

```bash
export PYTHONPATH=src        # on Windows: $env:PYTHONPATH = "src"
python scripts/run_sim_episode.py --episodes 5
```

## Training

### Baseline: Linear Policy (CEM)

A lightweight optimizer that runs without a GPU and is useful for sanity-checking the reward signal:

```bash
python scripts/train_sim_linear.py \
  --generations 8 --population 16 \
  --episodes-per-candidate 3 --eval-episodes 10 \
  --max-decisions 1000 \
  --checkpoint checkpoints/sim_linear_policy.npz
```

### MaskablePPO

Requires the `rl` extra:

```bash
python scripts/train_sim_ppo.py --timesteps 20000 --checkpoint checkpoints/sim_ppo_policy.zip
```

### PPO Self-Play

Trains against a growing pool of scripted opponents and past checkpoints:

```bash
python scripts/train_sim_ppo_selfplay.py --iterations 4 --timesteps-per-iteration 5000
```

## Evaluation

```bash
python scripts/evaluate_sim_policy.py --checkpoint checkpoints/sim_linear_policy.npz --episodes 50
python scripts/evaluate_sim_ppo.py --checkpoint checkpoints/sim_ppo_policy.zip --episodes 50
```

## Visualizer

Watch any policy fight a scripted opponent in a live Tkinter window:

```bash
python scripts/visualize_sim.py --checkpoint checkpoints/sim_linear_policy.npz
```

Controls: `Space` pause/resume, `R` reset episode, `+` / `-` speed up/slow down.

See `docs/visualizer.md` for more run modes (scripted vs. scripted, random chaos testing, headless smoke tests).

## Scripted Opponents

Five hand-tuned baseline opponents are available for training and evaluation: `random`, `poke`, `turtle`, `rushdown`, `whiff_punish`. See `src/t8_agent/sim/opponents.py`.

## DIAMBRA On-Ramp

[DIAMBRA Arena](https://diambra.ai) is used as an optional reference environment to validate the training loop against a supported fighting-game framework before targeting a real game client. Setup requires Docker Desktop, a DIAMBRA account, and a legally obtained ROM (not included in this repo). See `docs/diambra_onramp.md` for details.

```bash
python scripts/check_setup.py
```

## Testing

```bash
pytest
python scripts/bug_check.py   # compiles, runs tests, and smoke-tests scripts end to end
```

## Documentation

- `docs/simulator_first_plan.md` — design rationale for the simulator-first approach
- `docs/training.md` — training workflow details
- `docs/visualizer.md` — visualizer usage
- `docs/diambra_onramp.md` — DIAMBRA setup and milestones
- `docs/tekken8_implementation_plan.md` — plan for extending this toward a real game client (state extraction, input injection, self-play, calibration)

## Disclaimer

This is a research/hobby project. It does not include any game assets, ROMs, or licensed content. Any real-game integration is intended strictly for offline, local, single-player calibration and local versus play, and must comply with the applicable game's terms of use.
