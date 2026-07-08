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

The default scripted curriculum now skips the easy `random` opponent and trains
against `poke`, `rushdown`, `turtle`, `whiff_punish`, `keepout`, `frame_trap`,
and `anti_throw`. Use `--opponents random` only for smoke tests or chaos checks.
The simulator reward also discounts throw-only damage, rewards clean blocks,
penalizes blocked attacks and whiffs more strongly, and adds a round-end health
margin bonus so the policy has to win cleanly instead of farming one trick.

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

1. Add richer Jun moves and unsafe-on-block punish windows.
2. Run longer PPO self-play against the harder curriculum.
3. Track whether checkpoint-pool win rate stays informative instead of
   saturating.
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

PPO training uses `VecNormalize(norm_obs=True, norm_reward=True)` by default.
Each PPO checkpoint saves a matching `.vecnormalize.pkl` file beside the `.zip`
so evaluation, visualization, and checkpoint-pool opponents can use the same
observation scale. Pass `--no-normalize` only for debugging.

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

By default this is hybrid self-play: the pool samples the harder scripted
opponents some of the time, mostly recent checkpoints, and occasionally older
checkpoints. This keeps a stable curriculum while the checkpoint pool is still
small.

Run full checkpoint-pool self-play after one bootstrap iteration:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 8 --timesteps-per-iteration 10000 --full-self-play --bootstrap-iterations 1
```

In full self-play mode, the first checkpoint is bootstrapped from scripted
opponents. Once a checkpoint exists and bootstrap is over, training samples only
from saved older versions. Use `--old-sample-rate` to keep older checkpoints in
the mix and `--max-recent` to control the recent-checkpoint window.

Use parallel simulator games when CPU/RAM has headroom:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 20 --timesteps-per-iteration 20000 --full-self-play --bootstrap-iterations 1 --old-sample-rate 0.25 --elo-sampling --n-envs 8 --n-steps 256 --batch-size 256
```

`--n-envs` runs multiple simulator games per PPO rollout. On a 32 GB machine,
start with 8 parallel games, then raise it if CPU stays below about 85% and RAM
still has comfortable headroom.

For the main self-play curriculum, mostly train against the latest previous
self, with some fights against the strongest older checkpoint:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 20 --timesteps-per-iteration 32768 --full-self-play --bootstrap-iterations 1 --elo-sampling --latest-checkpoint-rate 0.80 --best-checkpoint-rate 0.20 --n-envs 16 --n-steps 256 --batch-size 512
```

With those rates, once at least one previous checkpoint exists, scripted
opponents are removed after bootstrap, 80% of checkpoint opponents are the most
recent previous version, and 20% are the highest-Elo older version.

Adjust the hybrid scripted mix:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 8 --timesteps-per-iteration 10000 --scripted-sample-rate 0.20 --old-sample-rate 0.25
```

Use Elo-weighted checkpoint sampling:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\train_sim_ppo_selfplay.py --iterations 8 --timesteps-per-iteration 10000 --full-self-play --bootstrap-iterations 1 --old-sample-rate 0.25 --elo-sampling
```

Each self-play run writes:

- `metrics.json` with scripted and checkpoint-pool evaluation per iteration.
- The active self-play mode and effective scripted sample rate per iteration.
- `curves.png` with win-rate and reward curves.
- `.vecnormalize.pkl` files beside PPO checkpoints when normalization is on.

Re-plot an existing run:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\plot_training_curves.py --metrics runs\ppo_selfplay_latest\metrics.json
```

Rank a checkpoint pool with Elo:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\rank_ppo_checkpoints.py --checkpoint-dir checkpoints\selfplay
```
