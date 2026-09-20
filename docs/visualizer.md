# V2 CUDA simulator visualizer

The V2 visualizer watches the actual CUDA simulator. It does not copy the retired V1 Python
environment and it does not insert rendering or device-to-host transfers into the PPO rollout loop.

The native `t8_v2_visualizer_feed` executable runs a separate 16-lane evaluation batch, optionally
loads a native `.t8ppo` checkpoint, and streams lane-zero state as JSON. `scripts/visualize_v2.py`
draws that stream in a Tkinter window.

## Build

```powershell
cmake -S . -B build -A x64 -DCMAKE_CUDA_ARCHITECTURES=120
cmake --build build --config Release --parallel
```

## Watch scripted policies

```powershell
.\.venv\Scripts\python.exe scripts\visualize_v2.py `
  --opponent-character reina `
  --opponent-archetype rushdown
```

You can also double-click `scripts\open_v2_visualizer_window.bat`.

## Watch one checkpoint

The observation mode must match the checkpoint that produced the policy.

```powershell
.\.venv\Scripts\python.exe scripts\visualize_v2.py `
  --checkpoint runs\roster_visual_shaped_2027\checkpoints\update_10.t8ppo `
  --observation-mode visual `
  --opponent-character reina `
  --opponent-archetype movement_specialist
```

## Follow a training run

The viewer starts with a scripted learner if the first checkpoint has not been written yet. It
automatically restarts its independent feed whenever a newer atomic `.t8ppo` file appears.

```powershell
.\.venv\Scripts\python.exe scripts\visualize_v2.py `
  --follow-dir runs\roster_visual_shaped_2027\checkpoints `
  --observation-mode visual `
  --opponent-character reina `
  --opponent-archetype rushdown
```

Or pass the checkpoint directory to the launcher:

```powershell
scripts\open_v2_visualizer_window.bat runs\roster_visual_shaped_2027\checkpoints
```

Controls:

- Space pauses the display.
- R restarts the independent simulator feed.
- `+` and `-` change playback speed.

## Headless validation

```powershell
.\.venv\Scripts\python.exe scripts\visualize_v2.py --headless-steps 64
```

This is also covered by CTest. A successful run prints `headless_ok` with the final episode, frame,
and health values.

## Performance isolation

The viewer is intentionally not baked into the 4,096-environment trainer. It runs a separate small
CUDA evaluation process and only that process downloads one lane for drawing. Training remains
device-resident. Close the viewer if every last percent of training throughput is required.
