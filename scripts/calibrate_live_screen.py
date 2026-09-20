from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import dxcam
import yaml


def _capture(output_idx: int, attempts: int = 40):
    camera = dxcam.create(output_idx=output_idx)
    camera.start(target_fps=30, video_mode=True)
    try:
        for _ in range(attempts):
            frame = camera.get_latest_frame()
            if frame is not None:
                return frame.copy()
            time.sleep(0.025)
    finally:
        camera.stop()
    raise RuntimeError("DXcam did not return a desktop frame")


def _region_from_roi(roi: tuple[int, int, int, int]) -> list[int]:
    x, y, width, height = [int(value) for value in roi]
    if width <= 0 or height <= 0:
        raise RuntimeError("a required region selection was cancelled")
    return [x, y, x + width, y + height]


def _automatic_regions(width: int, height: int) -> dict[str, list[int]]:
    # Conservative 16:9 Tekken 8 HUD/body defaults. They make the bridge
    # immediately testable; interactive calibration remains the accuracy gate.
    return {
        "p1_health_region": [int(width * 0.06), int(height * 0.055), int(width * 0.44), int(height * 0.11)],
        "p2_health_region": [int(width * 0.56), int(height * 0.055), int(width * 0.94), int(height * 0.11)],
        "p1_body_region": [0, int(height * 0.20), width // 2, int(height * 0.98)],
        "p2_body_region": [width // 2, int(height * 0.20), width, int(height * 0.98)],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Create the local Tekken 8 screen-capture calibration.")
    parser.add_argument("--output", type=Path, default=Path("config/live_screen.yaml"))
    parser.add_argument("--output-idx", type=int, default=0)
    parser.add_argument(
        "--automatic",
        action="store_true",
        help="Write conservative 16:9 defaults without opening ROI selection windows.",
    )
    args = parser.parse_args()

    frame = _capture(args.output_idx)
    height, width = frame.shape[:2]
    if args.automatic:
        regions = _automatic_regions(width, height)
        source = "automatic_16_9_defaults"
    else:
        prompts = (
            ("p1_health_region", "Select the P1 health bar, then press Enter"),
            ("p2_health_region", "Select the P2 health bar, then press Enter"),
            ("p1_body_region", "Select the LEFT fighter search area, then press Enter"),
            ("p2_body_region", "Select the RIGHT fighter search area, then press Enter"),
        )
        regions = {}
        try:
            for key, prompt in prompts:
                regions[key] = _region_from_roi(cv2.selectROI(prompt, frame, showCrosshair=True))
        finally:
            cv2.destroyAllWindows()
        source = "interactive_roi"

    config = {
        "calibration_source": source,
        "calibration_resolution": [width, height],
        **regions,
        "capture_fps": 60,
        "position_sample_stride": 4,
        "position_distance_scale": 1.0,
        "motion_threshold": 0.015,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    print(f"wrote {args.output} resolution={width}x{height} source={source}")
    if args.automatic:
        print("Run again without --automatic while Tekken 8 is visible before enabling controller output.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
