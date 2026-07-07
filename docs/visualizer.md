# Simulator Visualizer

The simulator visualizer is a Tkinter window for watching policies fight inside
the fast Tekken-lite environment.

## Run

```powershell
cd "D:\tekken 8"
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\visualize_sim.py --checkpoint checkpoints\sim_linear_policy.npz
```

## Controls

- `Space`: pause/resume.
- `R`: reset the episode.
- `+`: increase simulation speed.
- `-`: decrease simulation speed.

## Useful Modes

Checkpoint policy versus active rushdown scripted opponent:

```powershell
.\.venv\Scripts\python scripts\visualize_sim.py --p1 checkpoint --p2 scripted --p2-scripted rushdown
```

Scripted versus scripted:

```powershell
.\.venv\Scripts\python scripts\visualize_sim.py --p1 scripted --p1-scripted poke --p2 scripted --p2-scripted turtle
```

Random chaos test:

```powershell
.\.venv\Scripts\python scripts\visualize_sim.py --p1 random --p2 random
```

Headless smoke test:

```powershell
.\.venv\Scripts\python scripts\visualize_sim.py --headless-steps 100
```
