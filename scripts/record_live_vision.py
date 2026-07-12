from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2

from t8_agent.io.screen_backend import DxcamScreenStateBackend
from t8_agent.vision.temporal import TemporalScreenEstimator


def main() -> int:
    parser = argparse.ArgumentParser(description="Record screen-only Tekken clips and temporal pseudo-labels.")
    parser.add_argument("--screen-config", default="config/live_screen.yaml")
    parser.add_argument("--output-dir", default="data/live_vision")
    parser.add_argument("--seconds", type=float, default=60.0)
    parser.add_argument("--record-fps", type=float, default=15.0)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=180)
    args = parser.parse_args()

    session = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
    frames_dir = session / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    backend = DxcamScreenStateBackend(config_path=args.screen_config)
    estimator = TemporalScreenEstimator(
        p1_region=backend.p1_body_region,
        p2_region=backend.p2_body_region,
    )
    interval = 1.0 / args.record_fps
    deadline = time.perf_counter() + args.seconds
    next_frame = time.perf_counter()
    records = 0
    metadata_path = session / "frames.jsonl"
    try:
        with metadata_path.open("w", encoding="utf-8") as metadata:
            while time.perf_counter() < deadline:
                state = backend.read()
                frame = backend.last_frame
                if frame is None:
                    continue
                estimate = estimator.update(state, frame)
                frame_name = f"{records:07d}.jpg"
                resized = cv2.resize(frame, (args.width, args.height), interpolation=cv2.INTER_AREA)
                cv2.imwrite(str(frames_dir / frame_name), cv2.cvtColor(resized, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 88])
                metadata.write(json.dumps({"frame": frame_name, "time": time.time(), **estimate.to_dict()}) + "\n")
                records += 1
                next_frame += interval
                time.sleep(max(0.0, next_frame - time.perf_counter()))
    finally:
        backend.close()
    (session / "manifest.json").write_text(
        json.dumps({"frames": records, "fps": args.record_fps, "width": args.width, "height": args.height}, indent=2),
        encoding="utf-8",
    )
    print(f"recorded={records} session={session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
