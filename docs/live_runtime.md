# Live screen and virtual-controller runtime

The live runtime deploys a trained visual checkpoint. It captures the screen, estimates the
13-feature visual state and 95-feature roster-temporal state, runs Torch inference on CUDA, and can
send the selected action through a virtual Xbox 360 controller.

It is deliberately separate from PPO training. The trainer learns in thousands of parallel CUDA
simulators; one real Tekken 8 window is used for dry-run validation, policy evaluation, and later
real-game data collection.

## 1. Create the project environment

The `--system-site-packages` option reuses the validated CUDA Torch installation instead of
downloading another large wheel.

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install -e ".[live]"
```

## 2. Install the virtual-controller driver

`vgamepad` requires ViGEmBus on Windows. ViGEmBus is retired but its final Windows 10/11 installer
is still available from the signed
[official Nefarius release](https://github.com/nefarius/ViGEmBus/releases/tag/v1.22.0).

Installing a kernel driver is a machine-level change. Verify the downloaded executable's Windows
signature is valid and names `Nefarius Software Solutions e.U.` before approving its UAC prompt.
After installation, this neutral-state check must report `controller_ok`:

```powershell
.\.venv\Scripts\python.exe scripts\check_live_setup.py
```

The check creates a virtual controller only long enough to report a neutral state. It never presses
a gameplay button.

## 3. Calibrate screen capture

Generate a safe resolution-matched starting configuration:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_live_screen.py --automatic
```

Automatic regions are for dry-run setup only. Controller mode refuses them unless
`--allow-automatic-calibration` is explicitly supplied.

Before enabling controller output, open Tekken 8 in Practice mode and perform the interactive
calibration. Select the P1 health bar, P2 health bar, left-fighter search area, and right-fighter
search area in that order:

```powershell
.\.venv\Scripts\python.exe scripts\calibrate_live_screen.py
```

The resulting `config/live_screen.yaml` is machine-specific and intentionally ignored by Git.

## 4. Dry-run the capture path

Start with the rule policy and no controller:

```powershell
.\.venv\Scripts\python.exe scripts\live_vision_play.py `
  --dry-run `
  --agent rule `
  --seconds 30 `
  --screen-config config\live_screen.yaml
```

Then validate a trained V2 checkpoint, still without controller output:

```powershell
.\.venv\Scripts\python.exe scripts\live_vision_play.py `
  --dry-run `
  --agent v2 `
  --ppo-checkpoint runs\roster_visual_shaped_2027\checkpoints\update_100.t8ppo `
  --opponent-character reina `
  --opponent-archetype movement_specialist
```

## 5. Offline controller test

Remove `--dry-run` only in Practice/offline play. Output begins paused. Press F8 to enable or pause
the controller. Press F7 whenever the fighters switch sides; V2 now flips controller directions,
fighter positions, and temporal motion identity together.

Never use controller automation where it violates game rules or service terms.

For an explicit offline input-map check, the calibration tool requires a safety acknowledgment:

```powershell
.\.venv\Scripts\python.exe scripts\controller_calibration.py `
  --confirm-offline `
  --start-delay 5
```

## Optional real-game capture dataset

Record resized frames and timestamped 13-feature pseudo-labels without sending controller input:

```powershell
.\.venv\Scripts\python.exe scripts\record_live_vision.py `
  --seconds 60 `
  --record-fps 15
```

Sessions are written beneath `artifacts/live_capture/`, which is ignored by Git. This is the safe
bridge toward later real-game vision fine-tuning; it does not slow or contaminate simulator PPO
training.
