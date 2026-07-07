# Training

We can start prototype simulator training now.

The current trainer is a lightweight Cross-Entropy Method style optimizer over a
linear policy. It is not the final learning setup, but it is useful because it:

- Runs without GPU/PyTorch.
- Saves real checkpoints.
- Tests whether simulator rewards produce learnable behavior.
- Gives us a baseline before PPO/self-play.

## Train A Prototype Policy

```powershell
cd "D:\tekken 8"
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_linear.py --generations 8 --population 16 --episodes-per-candidate 3 --eval-episodes 10 --max-decisions 1000 --checkpoint checkpoints\sim_linear_policy.npz
```

Expected output looks like:

```text
generation=1 score=...
...
saved=checkpoints\sim_linear_policy.npz eval_score=... eval_win_rate=...
```

## What This Means

A good result here means the simulator supports basic learnable fighting
behavior against scripted opponents. It does not mean the policy is ready for
Tekken 8.

## Watch A Checkpoint

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\visualize_sim.py --checkpoint checkpoints\sim_linear_policy.npz
```

The visualizer shows health, timer, spacing, walls, current actions, move
commands, hitstun, blockstun, whiffs, and attack range.

The default visualizer opponent is `rushdown`, so the pink fighter should walk
in and attack rather than waiting for whiff-punish openings.

## Evaluate A Checkpoint

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\evaluate_sim_policy.py --checkpoint checkpoints\sim_linear_policy.npz --episodes 50
```

## Bug Check

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\bug_check.py
```

## Next Training Upgrade

1. Add richer Jun moves and better scripted opponents.
2. Add self-play checkpoint opponents.
3. Add PPO once the simulator reward is stable.
4. Evaluate sim-trained policies against DIAMBRA and then real Tekken 8 CV.

## PPO Training

Install the RL dependencies:

```powershell
.\.venv\Scripts\python -m pip install -e ".[rl]"
```

Train MaskablePPO:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo.py --timesteps 20000 --checkpoint checkpoints\sim_ppo_policy.zip
```

Evaluate:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\evaluate_sim_ppo.py --checkpoint checkpoints\sim_ppo_policy.zip --episodes 50
```

## PPO Self-Play

Train in chunks and add saved PPO checkpoints into the opponent pool:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 4 --timesteps-per-iteration 5000
```

The pool samples scripted opponents, mostly recent checkpoints, and occasionally
older checkpoints.
