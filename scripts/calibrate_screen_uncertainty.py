"""Measures the live screen pipeline's error for screen-only checkpoints.

Screen-only policies ("screen-matchup-95-v1") receive two uncertainty inputs:
the position noise level (stage units) and the event-detection error rate.
In training these are simulated; live, they must describe this capture
setup. This tool measures both and writes them to the screen config:

  distance  Pairs the pipeline's distance estimate with the "Distance from
            Opponent" value Tekken 8 shows in Practice mode (enable it in the
            Practice display settings). Move the fighters between samples.
            With independent per-fighter position noise, distance error has
            variance 2 * sigma^2, so sigma = RMS(error) / sqrt(2). Bias is
            included in the RMS (the policy is told the true total error); a
            large bias means position_distance_scale needs recalibration.

  events    Compares detections in a session recorded by
            scripts/record_live_vision.py against hand labels. The labels are
            a JSONL file with one {"frame": "0000012.jpg", "opponent_activity":
            0|1, "opponent_attack": 0|1} per labelled frame. The error is the
            overall disagreement rate, matching the simulator's symmetric flip
            model.

  manual    Writes explicit values (recorded as manual, not measured).

Examples:
  python scripts/calibrate_screen_uncertainty.py distance --samples 12
  python scripts/calibrate_screen_uncertainty.py events artifacts/live_capture/<session> labels.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from pathlib import Path

import yaml

MINIMUM_DISTANCE_SAMPLES = 8
MINIMUM_EVENT_LABELS = 50


def position_sigma_from_pairs(pairs: list[tuple[float, float]]) -> dict[str, float]:
    """pairs: (estimated distance, true distance) in stage units."""
    if len(pairs) < 2:
        raise ValueError("need at least two distance samples")
    errors = [estimate - truth for estimate, truth in pairs]
    bias = sum(errors) / len(errors)
    rms = math.sqrt(sum(error * error for error in errors) / len(errors))
    return {"samples": len(pairs), "bias": bias, "rms": rms, "sigma": rms / math.sqrt(2.0)}


def event_error_from_labels(
    detections: list[tuple[bool, bool]],
    labels: list[tuple[bool, bool]],
) -> dict[str, float]:
    """detections/labels: (opponent activity, opponent attack cue) per frame."""
    if len(detections) != len(labels) or not labels:
        raise ValueError("detections and labels must be non-empty and aligned")
    disagreements = sum(
        (detected[0] != truth[0]) + (detected[1] != truth[1]) for detected, truth in zip(detections, labels)
    )
    error = disagreements / (2.0 * len(labels))
    return {"labels": len(labels), "error": min(error, 0.5)}


def update_config(path: Path, values: dict) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    config.update(values)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def _detections_for_session(session: Path, motion_threshold: float, attack_threshold: float,
                            opponent_is_p2: bool) -> dict[str, tuple[bool, bool]]:
    detections = {}
    prefix = "p2" if opponent_is_p2 else "p1"
    with (session / "frames.jsonl").open(encoding="utf-8") as records:
        for line in records:
            record = json.loads(line)
            detections[record["frame"]] = (
                record[f"{prefix}_motion"] >= motion_threshold,
                record[f"{prefix}_attack_likelihood"] >= attack_threshold,
            )
    return detections


def calibrate_distance(args: argparse.Namespace) -> dict:
    from t8_agent.io.screen_backend import DxcamScreenStateBackend

    backend = DxcamScreenStateBackend(config_path=args.screen_config, p1_on_left=args.facing == 1)
    pairs: list[tuple[float, float]] = []
    try:
        print("Practice mode, 'Distance from Opponent' visible. Reposition the fighters between samples.")
        while len(pairs) < args.samples:
            answer = input(f"[{len(pairs) + 1}/{args.samples}] Press Enter to capture (q to stop): ").strip()
            if answer.lower() == "q":
                break
            estimate = backend.read().distance
            shown = input(f"  estimated {estimate:.2f}; distance shown in game: ").strip()
            try:
                pairs.append((float(estimate), float(shown)))
            except ValueError:
                print("  not a number; sample discarded")
    finally:
        backend.close()
    if len(pairs) < MINIMUM_DISTANCE_SAMPLES:
        raise SystemExit(f"need at least {MINIMUM_DISTANCE_SAMPLES} samples, got {len(pairs)}")
    result = position_sigma_from_pairs(pairs)
    print(f"distance bias {result['bias']:+.3f}, RMS {result['rms']:.3f} -> position sigma {result['sigma']:.3f}")
    if abs(result["bias"]) > 0.5 * result["rms"] and abs(result["bias"]) > 0.2:
        print("warning: error is mostly bias; recalibrate position_distance_scale first")
    return {"screen_position_sigma": round(result["sigma"], 4),
            "screen_position_calibration": {"method": "practice_distance_readout", **result,
                                            "date": dt.date.today().isoformat()}}


def calibrate_events(args: argparse.Namespace) -> dict:
    config = yaml.safe_load(Path(args.screen_config).read_text(encoding="utf-8")) or {}
    detections = _detections_for_session(
        args.session, float(config.get("motion_threshold", 0.015)), args.attack_threshold, args.opponent == 2)
    paired_detections, paired_labels = [], []
    with args.labels.open(encoding="utf-8") as labels:
        for line in labels:
            if not line.strip():
                continue
            label = json.loads(line)
            if label["frame"] not in detections:
                raise SystemExit(f"labelled frame {label['frame']} is not in {args.session}")
            paired_detections.append(detections[label["frame"]])
            paired_labels.append((bool(label["opponent_activity"]), bool(label["opponent_attack"])))
    if len(paired_labels) < MINIMUM_EVENT_LABELS:
        raise SystemExit(f"need at least {MINIMUM_EVENT_LABELS} labelled frames, got {len(paired_labels)}")
    result = event_error_from_labels(paired_detections, paired_labels)
    print(f"event detection error {result['error']:.3f} over {result['labels']} labelled frames")
    return {"screen_event_error": round(result["error"], 4),
            "screen_event_calibration": {"method": "hand_labelled_session", "session": str(args.session),
                                         **result, "date": dt.date.today().isoformat()}}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--screen-config", type=Path, default=Path("config/live_screen.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    distance = commands.add_parser("distance")
    distance.add_argument("--samples", type=int, default=12)
    distance.add_argument("--facing", type=int, choices=[-1, 1], default=1)
    events = commands.add_parser("events")
    events.add_argument("session", type=Path)
    events.add_argument("labels", type=Path)
    events.add_argument("--opponent", type=int, choices=[1, 2], default=2)
    events.add_argument("--attack-threshold", type=float, default=0.5)
    manual = commands.add_parser("manual")
    manual.add_argument("--position-sigma", type=float, required=True)
    manual.add_argument("--event-error", type=float, required=True)
    args = parser.parse_args()

    if args.command == "distance":
        values = calibrate_distance(args)
    elif args.command == "events":
        values = calibrate_events(args)
    else:
        if args.position_sigma < 0 or not 0 <= args.event_error <= 0.5:
            parser.error("--position-sigma must be >= 0 and --event-error in [0, 0.5]")
        values = {"screen_position_sigma": args.position_sigma, "screen_event_error": args.event_error,
                  "screen_uncertainty_calibration": {"method": "manual", "date": dt.date.today().isoformat()}}
    update_config(args.screen_config, values)
    print(f"updated {args.screen_config}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
