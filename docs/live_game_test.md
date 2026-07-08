# Live Offline Game Test

This bridge is for local/offline Practice or Versus testing only. Do not run it
in online matchmaking or any mode where automation could affect another player.

## Install Live Dependencies

The live bridge needs screen capture, hotkeys, and a virtual Xbox controller:

```powershell
cd "D:\tekken 8"
.\.venv\Scripts\python -m pip install -e ".[live]"
```

`vgamepad` also needs the ViGEmBus driver installed on Windows. If the script
can capture the screen but cannot create a controller, install ViGEmBus and
restart the machine.

## First Dry Run

Open Tekken 8 in offline Practice mode, keep the game focused, then run:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\live_play.py --dry-run --self-test
.\.venv\Scripts\python scripts\live_play.py --dry-run
```

The self-test captures one frame and exits. The second command opens the hotkey
loop: press `F8` to start/pause, and press `F12` to quit. Dry-run mode captures
the screen and prints actions, but does not press the controller.

## Controller Test

After dry-run works:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\live_play.py
```

The scripted live agent is still available with `--agent scripted`; it is meant
to prove that hotkeys, screen capture, and controller output work in the real
game. Checkpoint mode can load the simulator PPO, but it still needs calibrated
computer vision before it can make smart decisions from real Tekken 8 pixels.

To test the newest PPO checkpoint instead of the scripted controller test:

```powershell
$env:PYTHONPATH = "D:\tekken 8\src"
.\.venv\Scripts\python scripts\live_play.py --agent checkpoint --checkpoint latest --dry-run
.\.venv\Scripts\python scripts\live_play.py --agent checkpoint --checkpoint latest
```

`--checkpoint latest` searches under `checkpoints/`. It intentionally ignores
throwaway smoke runs under `runs/`.

## Health Bar Calibration

Copy the example config:

```powershell
Copy-Item config\live_screen.example.yaml config\live_screen.yaml
```

Edit the regions as `[left, top, right, bottom]` screen pixels:

```yaml
p1_health_region: [100, 80, 760, 120]
p2_health_region: [1160, 80, 1820, 120]
```

Then run:

```powershell
.\.venv\Scripts\python scripts\live_play.py --dry-run --screen-config config\live_screen.yaml
```

The printed `p1_hp` and `p2_hp` should move when health changes. Once that is
stable, the next step is real character/spacing detection so the PPO policy can
replace the scripted live-test agent.
