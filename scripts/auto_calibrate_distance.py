"""Automated distance calibration: the virtual controller walks the fighter, the screen is read.

    python scripts/auto_calibrate_distance.py [--steps 40] [--delay 8] [--write-config]

Offline Practice mode only (controller automation is for offline modes; README "Scope and
safety"). Set up: Practice mode, your character as P1 on the left, the dummy standing, the info
panel showing "Distance from Opponent", and the virtual controller (ViGEmBus; check with
scripts/check_live_setup.py) controlling P1. Do not touch your own controller while it runs;
Esc stops it at once (all inputs released).

First it claims the game: the virtual pad presses Menu once (this answers Tekken's "Press
Menu on the controller" prompt after you unplug your own controller, or opens the pause menu if
there is none), and again if a menu is still covering the screen (the distance readout is dimmed
and unreadable under a menu). It then restarts the Practice round (View + A) so both fighters
stand at the start, P1 on the left about 1.9 m apart, and checks that a step back really moves
your fighter before calibrating (stopping with a message if not). A fit that covers less than
--min-range metres is reported but never written to the config.

It dashes in close, then walks back in growing steps until the distance stops increasing (the
wall), then walks forward in steps until close again. After each step it releases, lets the
fighters settle, and records the pose tracker's gap / fighter height next to the distance the
game shows (read with t8_agent.live.practice_readout). It then fits distance = slope * ratio +
intercept, reports the error, writes runs/distance_calibration_auto/fit.json, and with
--write-config updates pose_distance_slope / pose_distance_intercept in the screen config.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t8_agent.io.pose_tracker import PoseFighterTracker  # noqa: E402
from t8_agent.live.practice_readout import PracticeDistanceReader  # noqa: E402


def fresh_frame(camera, timeout: float = 0.5):
    end = time.perf_counter() + timeout
    while time.perf_counter() < end:
        frame = camera.grab()
        if frame is not None:
            return frame
        time.sleep(0.002)
    return None


def hold(controller, action: str, seconds: float, stop) -> None:
    """Holds a movement for `seconds` (heartbeats keep the controller's hold alive).

    The controller's watchdog switches it off after hold_timeout without a heartbeat (a live-play
    safety), which happens during menu presses and settling pauses, so re-enable it first."""
    controller.set_enabled(True)
    end = time.perf_counter() + seconds
    while time.perf_counter() < end and not stop():
        controller.send_action(action)
        controller.heartbeat()
        time.sleep(0.02)
    controller.release_all()


def measure(camera, tracker: PoseFighterTracker, reader: PracticeDistanceReader, reads: int = 4) -> dict | None:
    """Median of a few reads once the fighters have settled."""
    ratios, shown = [], []
    for _ in range(reads):
        frame = fresh_frame(camera)
        if frame is None:
            continue
        pair = tracker.update(frame)
        value = reader.read_frame(frame[:, :, ::-1].copy())  # the reader takes BGR like the saved crops
        if pair is not None and tracker.fresh and value is not None:  # never a repeated stale pair
            heights = [f.box[3] - f.box[1] for f in pair]
            ratios.append((pair[1].x - pair[0].x) / max(1.0, sum(heights) / 2))
            shown.append(value)
        time.sleep(0.03)
    if len(ratios) < 2 or max(shown) - min(shown) > 0.02:
        return None  # not settled, or a read failed
    return {"ratio": float(np.median(ratios)), "distance": float(np.median(shown))}


def readable(camera, reader: PracticeDistanceReader, tries: int = 6) -> float | None:
    """The readout if it reads consistently (a menu over the screen dims it to unreadable)."""
    values = []
    for _ in range(tries):
        frame = fresh_frame(camera)
        if frame is not None:
            value = reader.read_frame(frame[:, :, ::-1].copy())
            if value is not None:
                values.append(value)
        time.sleep(0.05)
    return float(np.median(values)) if len(values) >= tries // 2 else None


def claim_game(controller, camera, reader: PracticeDistanceReader) -> bool:
    """Make the virtual pad the active controller and return to the fight: Menu, then Menu
    again only if a menu still covers the screen."""
    for attempt in range(2):
        controller.press_menu()
        time.sleep(1.5)
        if readable(camera, reader) is not None:
            return True
        print(f"menu press {attempt + 1}: the screen is still covered", flush=True)
    return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--steps", type=int, default=40, help="maximum steps each way")
    parser.add_argument("--delay", type=float, default=8, help="seconds to switch to the game first")
    parser.add_argument("--screen-config", type=Path, default=Path("config/live_screen.yaml"))
    parser.add_argument("--out", type=Path, default=Path("runs/distance_calibration_auto"))
    parser.add_argument("--write-config", action="store_true")
    parser.add_argument("--min-range", type=float, default=2.5,
                        help="metres of distance a fit must span before --write-config writes it")
    parser.add_argument("--max-rms", type=float, default=0.4,
                        help="largest RMS error (m) of a fit --write-config writes")
    args = parser.parse_args()
    import dxcam
    import keyboard
    from t8_agent.io.controller_backend import VGamepadInputBackend

    config = yaml.safe_load(args.screen_config.read_text(encoding="utf-8")) or {}
    tracker = PoseFighterTracker(config)
    reader = PracticeDistanceReader()
    camera = dxcam.create(output_idx=0)
    controller = VGamepadInputBackend(facing=1, tap_seconds=0.035, asynchronous=True, hold_timeout=0.25)
    stop = lambda: keyboard.is_pressed("esc")  # noqa: E731
    samples: list[dict] = []
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        print(f"starting in {args.delay:.0f} s: switch to Tekken (Esc stops)", flush=True)
        time.sleep(args.delay)
        controller.set_enabled(True)
        if not claim_game(controller, camera, reader):
            print("could not get back to the fight (menu or prompt still showing): "
                  "close it by hand and run again", flush=True)
            return 1
        # From a known start: Practice restart puts P1 on the left about 1.9 m away, mid-stage,
        # where a step back must open the distance (no wall behind, no body contact).
        controller.press_practice_restart()
        time.sleep(2.0)
        moved = None
        before = readable(camera, reader)
        hold(controller, "walk_back", 0.8, stop)
        time.sleep(0.5)
        after = readable(camera, reader)
        print(f"pad check walk_back: {before} -> {after}", flush=True)
        if before is not None and after is not None and after - before >= 0.2:
            moved = ("walk_back", before, after)
        if moved is None:
            frame = fresh_frame(camera)
            if frame is not None:
                import cv2
                cv2.imwrite(str(args.out / "pad_check_failed.png"), frame[:, :, ::-1])
            print("the virtual pad does not seem to move P1 (neither a step back nor forward changed the "
                  f"distance; screenshot in {args.out / 'pad_check_failed.png'})", flush=True)
            return 1
        print(f"the virtual pad controls P1 ({moved[0]}: {moved[1]:.2f} -> {moved[2]:.2f} m)", flush=True)
        for _ in range(3):  # close in
            hold(controller, "dash_forward", 0.15, stop)
            time.sleep(0.25)
        for phase, action in (("back", "walk_back"), ("forward", "walk_forward")):
            stalled = 0
            last = None
            for step in range(args.steps):
                if stop():
                    raise KeyboardInterrupt
                time.sleep(0.45)  # settle
                sample = measure(camera, tracker, reader)
                if sample is not None:
                    sample.update(phase=phase, step=step)
                    samples.append(sample)
                    print(json.dumps(sample), flush=True)
                    moved = last is None or abs(sample["distance"] - last) > 0.03
                    stalled = 0 if moved else stalled + 1
                    last = sample["distance"]
                    if phase == "forward" and sample["distance"] < 1.0:
                        break
                if stalled >= 2:
                    break  # the wall (or the dummy) stops it
                hold(controller, action, 0.3 + 0.03 * min(step, 10), stop)
    except KeyboardInterrupt:
        print("stopped", flush=True)
    finally:
        controller.release_all()
        controller.close()
    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    if len(samples) < 8:
        print(f"only {len(samples)} usable samples: no fit")
        return 1
    ratio = np.array([s["ratio"] for s in samples])
    truth = np.array([s["distance"] for s in samples])
    slope, intercept = np.polyfit(ratio, truth, 1)
    error = np.polyval([slope, intercept], ratio) - truth
    fit = {"slope": float(slope), "intercept": float(intercept), "rms_m": float(np.sqrt((error ** 2).mean())),
           "max_m": float(np.abs(error).max()), "samples": len(samples),
           "range_m": [float(truth.min()), float(truth.max())]}
    (args.out / "fit.json").write_text(json.dumps(fit, indent=1), encoding="utf-8")
    print(f"fit: distance = {slope:.4f} * ratio + {intercept:.4f}; rms {fit['rms_m']:.3f} m, max {fit['max_m']:.2f} m "
          f"over {fit['range_m'][0]:.2f}-{fit['range_m'][1]:.2f} m ({len(samples)} samples)")
    if args.write_config and fit["range_m"][1] - fit["range_m"][0] < args.min_range:
        print(f"not writing the config: the samples span only {fit['range_m'][1] - fit['range_m'][0]:.2f} m "
              f"(need {args.min_range} m)")
    elif args.write_config and fit["rms_m"] > args.max_rms:
        print(f"not writing the config: rms {fit['rms_m']:.2f} m is above {args.max_rms} m")
    elif args.write_config:
        text = args.screen_config.read_text(encoding="utf-8")
        lines = [line for line in text.splitlines()
                 if not line.startswith(("pose_distance_slope:", "pose_distance_intercept:"))]
        lines += [f"pose_distance_slope: {slope:.4f}", f"pose_distance_intercept: {intercept:.4f}"]
        args.screen_config.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"updated {args.screen_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
