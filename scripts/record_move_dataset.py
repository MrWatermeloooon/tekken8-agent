"""Records pose sequences of known moves: training data for an opponent-move recognizer.

    python scripts/record_move_dataset.py --character jun [--reps 3] [--limit N] [--delay 5]

Offline Practice mode, the character as P1, the dummy standing, your own controller unplugged
(the pad claims the game; Esc stops). For every Practice-validated move of the character
(data/validation/<character>.json: its input and startup are confirmed in the game) it
restarts, steps in, and records both fighters' pose skeletons (17 keypoints, boxes) at the
tracker's rate for about two seconds around the move, with the time the input was sent. It
also records idle, walking, crouching, and dashing as moves that are not attacks. Each sample
is one .npz in runs/move_dataset/<character>/ with its labels (move id, hit levels, startup,
damage) so a recognizer can learn "an attack is coming, at this height, in N frames" from the
skeleton alone, the screen observation's missing piece (the opponent's move, features 50-57).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_calibrate_distance import claim_game, fresh_frame, hold  # noqa: E402
from t8_agent.io.pose_tracker import PoseFighterTracker  # noqa: E402
from t8_agent.live.practice_readout import PracticeDistanceReader  # noqa: E402

NON_ATTACKS = ("neutral", "walk_forward", "walk_back", "crouch", "dash_forward", "dash_back")


def record(camera, tracker, controller, seconds: float, start_action, stop, held: bool = False,
           on_frame=None) -> dict:
    """Pose frames for `seconds`; start_action() runs after 0.3 s (and keeps running every frame
    when `held`: a walk or crouch) and its time is kept. on_frame(frame) sees every new frame."""
    times, p1, p2, boxes, sent_at = [], [], [], [], None
    began = time.perf_counter()
    while time.perf_counter() - began < seconds and not stop():
        controller.heartbeat()
        if time.perf_counter() - began >= 0.3 and (sent_at is None or held):
            start_action()
            if sent_at is None:
                sent_at = time.perf_counter() - began
        frame = fresh_frame(camera)
        if frame is None:
            continue
        if on_frame is not None:
            on_frame(frame)
        pair = tracker.update(frame)
        if pair is None or not tracker.fresh:
            continue
        times.append(time.perf_counter() - began)
        p1.append(pair[0].keypoints)
        p2.append(pair[1].keypoints)
        boxes.append([list(pair[0].box), list(pair[1].box)])
    return {"times": np.array(times, np.float32), "p1": np.array(p1, np.float32), "p2": np.array(p2, np.float32),
            "boxes": np.array(boxes, np.float32), "sent_at": sent_at or 0.0}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--character", required=True)
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--first-rep", type=int, default=0, help="number the repetitions from here (add to a dataset)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=5)
    parser.add_argument("--out", type=Path, default=Path("runs/move_dataset"))
    parser.add_argument("--screen-config", type=Path, default=Path("config/live_screen.yaml"))
    args = parser.parse_args()
    import dxcam
    import keyboard
    from t8_agent.io.controller_backend import VGamepadInputBackend
    from t8_agent.moves.candidates import build_candidate_table
    from t8_agent.moves.catalog import load_compiled_catalog

    catalog = load_compiled_catalog(Path("data/generated/full_move_catalog.json"))
    moves = {move["stable_id"]: move for move in catalog.moves}
    table = build_candidate_table(catalog, args.character)
    ledger = json.loads((Path("data/validation") / f"{args.character}.json").read_text(encoding="utf-8"))
    validated = [sid for sid, record in ledger["moves"].items() if record.get("status") == "pass"
                 and sid in table.stable_ids]
    if args.limit:
        validated = validated[:args.limit]
    out = args.out / args.character
    out.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(args.screen_config.read_text(encoding="utf-8")) or {}
    tracker = PoseFighterTracker(config)
    camera = dxcam.create(output_idx=0)
    reader = PracticeDistanceReader()
    controller = VGamepadInputBackend(
        facing=1, tap_seconds=0.035, asynchronous=True, hold_timeout=0.25, string_gap_seconds=14 / 60,
        string_first_gap_seconds=8 / 60, string_gap_growth_seconds=4 / 60, dash_button_delay_seconds=4 / 60)
    stop = lambda: keyboard.is_pressed("esc")  # noqa: E731
    plan = [(sid, rep) for rep in range(args.first_rep, args.first_rep + args.reps) for sid in list(validated) + [f"universal:{n}" for n in NON_ATTACKS]]
    saved = 0
    try:
        print(f"{len(plan)} samples ({len(validated)} moves + {len(NON_ATTACKS)} non-attacks, x{args.reps}); "
              f"starting in {args.delay:.0f} s (Esc stops)", flush=True)
        time.sleep(args.delay)
        if not claim_game(controller, camera, reader):
            print("could not get back to the fight", flush=True)
            return 1
        for count, (sid, rep) in enumerate(plan, 1):
            if stop():
                break
            controller.press_practice_restart()
            time.sleep(2.2)
            tracker.reset_round()
            hold(controller, "dash_forward", 0.15, stop)
            time.sleep(0.35)
            controller.set_enabled(True)
            if sid.startswith("universal:"):
                name = sid.split(":", 1)[1]
                sample = record(camera, tracker, controller, 2.0, lambda: controller.send_action(name), stop,
                                held=name in ("walk_forward", "walk_back", "crouch"))
                labels = {"stable_id": sid, "attack": False, "hit_levels": [], "startup": None, "damage": []}
            else:
                index = table.stable_ids.index(sid)
                sample = record(camera, tracker, controller, 2.2,
                                lambda: controller.send_command(table.commands[index], move_id=sid), stop)
                move = moves.get(sid) or moves.get(sid.split("~")[0])
                startup = (move.get("startup") or {}).get("min")
                labels = {"stable_id": sid, "attack": True, "hit_levels": move.get("hit_levels") or [],
                          "startup": startup, "damage": move.get("damage") or [],
                          "measured_startup": ledger["moves"][sid].get("measured_startup")}
            controller.release_all()
            if len(sample["times"]) < 10:
                print(f"[{count}/{len(plan)}] {sid}: too few pose frames, skipped", flush=True)
                continue
            name = f"{sid.replace(':', '_').replace('~', '-')}_{rep}.npz"
            np.savez_compressed(out / name, times=sample["times"], p1=sample["p1"], p2=sample["p2"],
                                boxes=sample["boxes"], sent_at=np.float32(sample["sent_at"]),
                                labels=json.dumps(labels))
            saved += 1
            if count % 10 == 0:
                print(f"[{count}/{len(plan)}] saved {saved}", flush=True)
    finally:
        controller.release_all()
        controller.close()
    print(f"saved {saved} samples to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
