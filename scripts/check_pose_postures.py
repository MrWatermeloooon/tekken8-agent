"""Measures the pose tracker's posture readings against postures the virtual pad commands.

    python scripts/check_pose_postures.py [--delay 5] [--distances 3]

Offline Practice mode, your character as P1, the dummy standing, your own controller
unplugged (the pad claims the game itself; Esc stops). At each distance (restart, then walk
back) the pad makes P1 stand, crouch, and jump, and every pose reading in those windows is
compared with the commanded posture (P1) and with standing (the dummy). Prints confusion
tables and writes runs/pose_posture_check.json; the screen-observation contract reads both
fighters' postures from this tracker (full_core::kScreenViewFeatures 8-11, 14-17).
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
import time

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_calibrate_distance import claim_game, fresh_frame, hold  # noqa: E402
from t8_agent.io.pose_tracker import PoseFighterTracker  # noqa: E402
from t8_agent.live.practice_readout import PracticeDistanceReader  # noqa: E402


def sample(camera, tracker, seconds: float, controller=None) -> list[tuple[str, str]]:
    """(P1 posture, P2 posture) for every fresh pose reading during `seconds`."""
    readings = []
    end = time.perf_counter() + seconds
    while time.perf_counter() < end:
        if controller is not None:
            controller.heartbeat()
        frame = fresh_frame(camera)
        if frame is None:
            continue
        pair = tracker.update(frame)
        if pair is not None and tracker.fresh:
            readings.append((pair[0].posture, pair[1].posture))  # P1 is on the left after a restart
    return readings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--delay", type=float, default=5)
    parser.add_argument("--distances", type=int, default=3)
    parser.add_argument("--screen-config", type=Path, default=Path("config/live_screen.yaml"))
    args = parser.parse_args()
    import dxcam
    import keyboard
    from t8_agent.io.controller_backend import VGamepadInputBackend

    config = yaml.safe_load(args.screen_config.read_text(encoding="utf-8")) or {}
    tracker = PoseFighterTracker(config)
    camera = dxcam.create(output_idx=0)
    reader = PracticeDistanceReader()
    controller = VGamepadInputBackend(facing=1, tap_seconds=0.035, asynchronous=True, hold_timeout=0.25)
    stop = lambda: keyboard.is_pressed("esc")  # noqa: E731
    own: dict[str, Counter] = {"standing": Counter(), "crouching": Counter(), "airborne": Counter()}
    opponent: Counter = Counter()
    try:
        time.sleep(args.delay)
        if not claim_game(controller, camera, reader):
            print("could not get back to the fight", flush=True)
            return 1
        for distance in range(args.distances):
            controller.press_practice_restart()
            time.sleep(2.2)
            if distance:
                hold(controller, "walk_back", 0.6 * distance, stop)
                time.sleep(0.4)
            tracker.reset_round()
            sample(camera, tracker, 1.0)  # learn the standing height and floor first
            for posture, readings in (("standing", sample(camera, tracker, 1.5)),):
                for mine, theirs in readings:
                    own[posture][mine] += 1
                    opponent[theirs] += 1
            # Crouch: hold down; skip the first 0.25 s of the transition.
            controller.set_enabled(True)
            end = time.perf_counter() + 1.8
            started = time.perf_counter()
            while time.perf_counter() < end and not stop():
                controller.send_action("crouch")
                controller.heartbeat()
                if time.perf_counter() - started > 0.25:
                    frame = fresh_frame(camera)
                    pair = tracker.update(frame) if frame is not None else None
                    if pair is not None and tracker.fresh:
                        own["crouching"][pair[0].posture] += 1
                        opponent[pair[1].posture] += 1
                time.sleep(0.01)
            controller.release_all()
            time.sleep(0.6)
            # Jump: up; the airborne window is roughly 0.15-0.55 s after the press.
            for _ in range(3):
                controller.set_enabled(True)
                controller.send_action("jump")
                pressed = time.perf_counter()
                while time.perf_counter() - pressed < 0.7:
                    controller.heartbeat()
                    frame = fresh_frame(camera)
                    elapsed = time.perf_counter() - pressed
                    pair = tracker.update(frame) if frame is not None else None
                    if pair is not None and tracker.fresh and 0.15 <= elapsed <= 0.55:
                        own["airborne"][pair[0].posture] += 1
                        opponent[pair[1].posture] += 1
                time.sleep(0.8)
            print(f"distance step {distance} done", flush=True)
    finally:
        controller.release_all()
        controller.close()
    report = {"own": {k: dict(v) for k, v in own.items()}, "opponent_standing": dict(opponent)}
    Path("runs/pose_posture_check.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    for posture, counts in own.items():
        total = sum(counts.values())
        right = counts.get(posture, 0)
        print(f"P1 commanded {posture:9}: read correctly {right}/{total} ({right / max(total, 1):.0%})  {dict(counts)}")
    total = sum(opponent.values())
    print(f"dummy (always standing): read standing {opponent.get('standing', 0)}/{total} "
          f"({opponent.get('standing', 0) / max(total, 1):.0%})  {dict(opponent)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
