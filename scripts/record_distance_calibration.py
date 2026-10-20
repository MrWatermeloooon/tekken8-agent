"""Records pose distance measurements next to Practice mode's "Distance from Opponent" readout.

    python scripts/record_distance_calibration.py [--seconds 60] [--every 2] [--delay 10]
                                                  [--out runs/distance_calibration]

Every --every seconds it saves the pose tracker's measurements (the fighters' on-screen
distance in pixels, their on-screen heights, the raw stage-unit distance) to samples.jsonl
and a crop of the Practice info panel (where the readout is) as sample_NNN.png, so the true
distance can be read off each crop. Move the fighters to different distances while it runs,
pausing briefly at each. Tekken's camera zooms with distance, so the fit uses the gap
divided by the fighters' height as well as the raw gap.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from t8_agent.io.pose_tracker import PoseFighterTracker  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--seconds", type=float, default=60)
    parser.add_argument("--every", type=float, default=2)
    parser.add_argument("--delay", type=float, default=10)
    parser.add_argument("--out", type=Path, default=Path("runs/distance_calibration"))
    parser.add_argument("--screen-config", type=Path, default=Path("config/live_screen.yaml"))
    args = parser.parse_args()
    import dxcam
    config = yaml.safe_load(args.screen_config.read_text(encoding="utf-8")) or {}
    tracker = PoseFighterTracker(config)
    camera = dxcam.create(output_idx=0)
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"starting in {args.delay:.0f} s; move the fighters between distances, pausing at each", flush=True)
    time.sleep(args.delay)
    samples = []
    end = time.perf_counter() + args.seconds
    while time.perf_counter() < end:
        frame = None
        while frame is None:
            frame = camera.grab()
        pair = tracker.update(frame)
        height, width = frame.shape[:2]
        index = len(samples)
        # The Practice info panel (bottom left), where "Distance from Opponent" is shown.
        crop = frame[int(height * 0.58):int(height * 0.92), int(width * 0.03):int(width * 0.28)]
        cv2.imwrite(str(args.out / f"sample_{index:03d}.png"), cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        record = {"sample": index, "time": time.time(), "found": pair is not None}
        if pair is not None:
            left, right = pair
            heights = [f.box[3] - f.box[1] for f in pair]
            record.update(gap_px=right.x - left.x, height_px=sum(heights) / 2, width_px=width,
                          left_x=left.x, right_x=right.x, postures=[left.posture, right.posture])
        samples.append(record)
        print(json.dumps(record), flush=True)
        time.sleep(max(0.0, args.every - 0.05))
    (args.out / "samples.jsonl").write_text("\n".join(json.dumps(s) for s in samples) + "\n", encoding="utf-8")
    print(f"saved {len(samples)} samples to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
