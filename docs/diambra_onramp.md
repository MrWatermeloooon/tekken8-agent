# DIAMBRA On-Ramp

DIAMBRA is the proving ground before the custom Tekken 8 environment. The point
is to validate the self-play/training loop in a supported fighting-game
environment before spending weeks on Tekken 8 computer vision.

## Current Source Check

The DIAMBRA Arena README describes:

- A standard Gym/Gymnasium-style Python API.
- Discrete gamepad actions.
- Observations with screen pixels plus game-specific RAM values.
- Single-player and two-player modes.
- Supported games including `tektagt` for Tekken Tag Tournament.
- Installation with Docker Desktop, `diambra`, and `diambra-arena`.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[diambra,dev]"
```

Before running DIAMBRA CLI commands in PowerShell, activate the venv:

```powershell
.\.venv\Scripts\Activate.ps1
```

You also need:

- A free DIAMBRA account.
- Docker Desktop running.
- A valid ROM for a DIAMBRA-supported game.

This repo does not include ROMs.

## Validate ROMs

```powershell
python -m diambra arena list-roms
python -m diambra arena check-roms C:\path\to\roms\tektagt.zip
```

For Tekken Tag Tournament, DIAMBRA reports the original ROM name as
`tektagtac.zip`, but it must be renamed to `tektagt.zip`.

## First Episode

```powershell
python -m diambra run -r C:\path\to\roms python scripts/diambra_random_episode.py --game tektagt --characters Jun Jin --render
```

If this works, DIAMBRA can launch the emulator, connect Python, step the
environment, and render frames.

## Milestones

1. Random-policy episode runs for `tektagt`.
2. Script logs observation keys, reward, and termination status.
3. Add a two-player/self-play sample.
4. Save episode traces.
5. Train a tiny baseline agent.
6. Port the training loop shape back to the Tekken 8 CV environment.
