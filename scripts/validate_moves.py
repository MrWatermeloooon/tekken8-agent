"""Automated Practice-mode move validation: the virtual controller performs each move, the
Practice info panel is read, and the result is compared with our frame data.

    python scripts/validate_moves.py --character jun [--moves jun:6,jun:43] [--limit N] [--delay 8]
                                     [--patch 3.02.02] [--write-ledger]

Offline Practice mode only. Setup: the character as P1, the dummy standing, the info panel
showing (Attack Startup Frames, Frame Advantage, Damage), your own controller unplugged; the
virtual pad claims the game itself (scripts/auto_calibrate_distance.py explains how). Esc stops
at once.

For each move testable from a standing start (a parsed, certain command with no stance, crouch,
sidestep, Heat or Rage requirement), it restarts the round (View + A), steps in, performs the
move, and reads Total Damage, Damage, Attack Startup Frames, and Frame Advantage. A hit shows on the Damage
line as one of the move's hit damages (Total Damage, at the panel's top edge, misreads); a whiff
is retried once from closer. A single hit passes when its startup is within a frame of
our data and its damage matches; a string passes when the panel's damage (the last hit that
landed) is its last hit's, and is "partial k/n" when it stopped at hit k (input timing).
Results go to runs/move_validation/<character>.jsonl with every reading; --write-ledger records
passes and failures in data/validation/<character>.json (method practice_mode, reviewer
auto_validator, unreviewed) so the live gate can unlock validated moves.
"""
from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from auto_calibrate_distance import claim_game, fresh_frame, hold, readable  # noqa: E402
from t8_agent.live.practice_readout import (  # noqa: E402
    PracticeDistanceReader, PracticeLineReader, panel_crop,
)


def read_panel(camera, lines: PracticeLineReader) -> dict:
    """The panel's values, read twice and kept only where both reads agree."""
    reads = []
    for _ in range(2):
        frame = fresh_frame(camera)
        if frame is None:
            continue
        panel = panel_crop(frame[:, :, ::-1].copy())
        reads.append({line: lines.read(panel, line)
                      for line in ("total_damage", "damage", "startup", "frame_advantage")})
        time.sleep(0.05)
    if len(reads) < 2:
        return {}
    return {key: value for key, value in reads[0].items() if reads[1].get(key) == value}


def testable(command) -> bool:
    """A parsed command with certain inputs and no state requirement (a CommandSpec or its dict)."""
    field = (lambda obj, name, default=None: obj.get(name, default) if isinstance(obj, dict)
             else getattr(obj, name, default))
    steps = field(command, "steps", ()) or ()
    return (field(command, "parser_status") == "parsed" and bool(steps) and not field(command, "requirements", ())
            and not any(field(step, "uncertain", False) or field(step, "requirement") for step in steps))


def perform(controller, table, index: int, stop) -> bool:
    """Sends a move and keeps the controller's watchdog fed until the input has played out."""
    controller.set_enabled(True)
    sent = controller.send_command(table.commands[index], move_id=table.stable_ids[index])
    end = time.perf_counter() + 1.8
    while time.perf_counter() < end and not stop():
        controller.heartbeat()
        time.sleep(0.02)
    controller.release_all()
    return bool(sent)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--character", required=True)
    parser.add_argument("--moves", help="comma-separated stable ids (default: every testable move)")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--delay", type=float, default=8)
    parser.add_argument("--patch", default="3.02.02", help="the game version shown on screen")
    parser.add_argument("--write-ledger", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="skip moves already recorded in the log since --since (default today), and "
                             "include those records in the ledger")
    parser.add_argument("--since", default=None, help="ISO date for --resume (default today)")
    parser.add_argument("--out", type=Path, default=Path("runs/move_validation"))
    parser.add_argument("--string-gap-frames", type=float, default=14,
                        help="frames before each later input of a string (controller string_gap_seconds)")
    parser.add_argument("--string-first-gap-frames", type=float, default=8,
                        help="frames before a string's first follow-up input")
    parser.add_argument("--string-gap-growth-frames", type=float, default=4,
                        help="extra frames for each later follow-up input")
    parser.add_argument("--dash-button-delay-frames", type=float, default=4,
                        help="frames a held direction (f,F+2) is pressed alone before its button")
    args = parser.parse_args()
    import dxcam
    import keyboard
    from t8_agent.io.controller_backend import VGamepadInputBackend
    from t8_agent.moves.candidates import build_candidate_table
    from t8_agent.moves.catalog import load_compiled_catalog

    catalog = load_compiled_catalog(Path("data/generated/full_move_catalog.json"))
    moves = {move["stable_id"]: move for move in catalog.moves}
    table = build_candidate_table(catalog, args.character)
    chosen = set(args.moves.split(",")) if args.moves else None
    indices = [i for i, command in enumerate(table.commands)
               if not table.is_universal(i) and command is not None and testable(command)
               and (chosen is None or table.stable_ids[i] in chosen)]
    results = []
    if args.resume and (args.out / f"{args.character}.jsonl").exists():
        since = time.mktime(time.strptime(args.since or date.today().isoformat(), "%Y-%m-%d"))
        latest: dict[str, dict] = {}
        for line in (args.out / f"{args.character}.jsonl").read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            if record.get("time", 0) >= since:
                latest[record["stable_id"]] = record  # the most recent result of each move
        wanted = {table.stable_ids[i] for i in indices}
        results = [record for record in latest.values() if record["stable_id"] in wanted]
        done = {record["stable_id"] for record in results}
        indices = [i for i in indices if table.stable_ids[i] not in done]
        print(f"resuming: {len(done)} moves already recorded", flush=True)
    if args.limit:
        indices = indices[:args.limit]
    camera = dxcam.create(output_idx=0)
    distance_reader = PracticeDistanceReader()
    lines = PracticeLineReader()
    controller = VGamepadInputBackend(
        facing=1, tap_seconds=0.035, asynchronous=True, hold_timeout=0.25,
        string_gap_seconds=args.string_gap_frames / 60.0,
        string_first_gap_seconds=args.string_first_gap_frames / 60.0,
        string_gap_growth_seconds=args.string_gap_growth_frames / 60.0,
        dash_button_delay_seconds=args.dash_button_delay_frames / 60.0)
    stop = lambda: keyboard.is_pressed("esc")  # noqa: E731
    args.out.mkdir(parents=True, exist_ok=True)
    log_path = args.out / f"{args.character}.jsonl"
    previous = len(results)  # resumed records are already in the log
    try:
        print(f"{len(indices)} moves to test; starting in {args.delay:.0f} s: switch to Tekken (Esc stops)", flush=True)
        time.sleep(args.delay)
        if not claim_game(controller, camera, distance_reader):
            print("could not get back to the fight: close the menu by hand and run again", flush=True)
            return 1
        for count, index in enumerate(indices, 1):
            if stop():
                print("stopped", flush=True)
                break
            stable_id = table.stable_ids[index]
            # A variant (jun:60~FC: the same move ending crouched) has its base move's data.
            move = moves.get(stable_id) or moves.get(stable_id.split("~")[0])
            if move is None:
                print(f"[{count}/{len(indices)}] {stable_id}: no catalog data, skipped", flush=True)
                continue
            startup = move.get("startup") or {}
            expected = (startup.get("min"), startup.get("max"))
            reading, attempt = {}, 0
            for attempt, dashes in enumerate((1, 2)):  # from a step in; closer on a retry
                controller.press_practice_restart()
                time.sleep(2.2)
                for _ in range(dashes):
                    hold(controller, "dash_forward", 0.15, stop)
                    time.sleep(0.35)
                time.sleep(0.4)
                if not perform(controller, table, index, stop):
                    reading = {"error": "not sent"}
                    break
                time.sleep(0.3)
                reading = read_panel(camera, lines)
                damages = [float(value) for value in (move.get("damage") or [])]
                if reading.get("damage") is not None and any(abs(v - reading["damage"]) < 0.5 for v in damages):
                    break  # the Damage line shows one of this move's hits: it landed
            measured = reading.get("startup")
            damages = [float(value) for value in (move.get("damage") or [])]
            shown = reading.get("damage")
            # The panel shows the last hit that landed: its damage tells how far a string got. Its
            # startup is that hit's timing, so only a single hit is compared with our (first-hit) startup.
            reached = max((k for k, value in enumerate(damages) if shown is not None and abs(value - shown) < 0.5),
                          default=None)
            if reached is None:
                status = "whiff"
            elif len(damages) <= 1:
                in_range = (measured is not None and expected[0] is not None
                            and expected[0] - 1 <= measured <= (expected[1] or expected[0]) + 1)
                status = "pass" if in_range and (not damages or reached == 0) else "fail"
            elif reached == len(damages) - 1:
                status = "pass"
            elif reached is not None:
                status = f"partial {reached + 1}/{len(damages)}"
            else:
                status = "fail"
            record = {"stable_id": stable_id, "command": move["command"]["raw"], "name": move.get("name"),
                      "status": status, "expected_startup": expected, "reading": reading,
                      "expected_damage": move.get("damage"), "tries": attempt + 1}
            results.append(record)
            print(f"[{count}/{len(indices)}] {stable_id:9} {move['command']['raw']:<14} {status:5} "
                  f"startup {measured} (data {expected[0]}-{expected[1]}), damage {reading.get('damage')} "
                  f"(data {(move.get('damage') or [float('nan')])[0]:g}), "
                  f"advantage {reading.get('frame_advantage')}", flush=True)
    finally:
        controller.release_all()
        controller.close()
        with log_path.open("a", encoding="utf-8") as handle:
            for record in results[previous:]:
                handle.write(json.dumps({"time": time.time(), **record}) + "\n")
    passed = sum(r["status"] == "pass" for r in results)
    print(f"{passed}/{len(results)} passed; whiffs {sum(r['status'] == 'whiff' for r in results)}, "
          f"startup mismatches {sum(r['status'] == 'fail' for r in results)}; log {log_path}")
    if args.write_ledger and results:
        ledger_path = Path("data/validation") / f"{args.character}.json"
        ledger = (json.loads(ledger_path.read_text(encoding="utf-8")) if ledger_path.exists()
                  else {"schema_version": 1, "moves": {}, "scenarios": {}, "routes": {}})
        provenance = {"patch": args.patch, "recorded_at": date.today().isoformat(), "method": "practice_mode",
                      "confidence": "medium", "reviewer": "auto_validator", "review_status": "unreviewed"}
        for record in results:
            if record["status"] == "whiff" or record["status"].startswith("partial"):
                continue  # inconclusive (a missed hit or an input timing problem), not a data failure
            ledger["moves"][record["stable_id"]] = {
                "status": "pass" if record["status"] == "pass" else "fail",
                "measured_startup": record["reading"].get("startup"),
                "measured_damage": record["reading"].get("damage"),
                "measured_hit_advantage": record["reading"].get("frame_advantage"),
                "provenance": provenance,
            }
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger_path.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        print(f"updated {ledger_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
