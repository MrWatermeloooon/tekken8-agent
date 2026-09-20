from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from t8_agent.io.controller_backend import VGamepadInputBackend  # noqa: E402
from t8_agent.moves.catalog import load_compiled_catalog  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Walk a character's compiled commands through offline Tekken Practice mode.")
    parser.add_argument("character")
    parser.add_argument("--confirm-offline", action="store_true")
    parser.add_argument("--catalog", type=Path,
                        default=REPO_ROOT / "data/generated/full_move_catalog.json")
    parser.add_argument("--validation-root", type=Path,
                        default=REPO_ROOT / "data/validation")
    parser.add_argument("--start-delay", type=float, default=5.0)
    parser.add_argument("--between-seconds", type=float, default=1.0)
    parser.add_argument("--facing", type=int, choices=(-1, 1), default=1)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if not args.dry_run and not args.confirm_offline:
        parser.error("--confirm-offline is required before sending controller inputs")
    catalog = load_compiled_catalog(args.catalog)
    character = next((row for row in catalog.characters if row["slug"] == args.character), None)
    if character is None:
        parser.error(f"unknown released character: {args.character}")
    moves = [move for move in catalog.moves if move["character_id"] == character["id"]]
    args.validation_root.mkdir(parents=True, exist_ok=True)
    ledger_path = args.validation_root / f"{args.character}.json"
    ledger = {"schema_version": 1, "character": args.character, "moves": {},
              "scenarios": {}, "routes": {}}
    if ledger_path.exists():
        ledger.update(json.loads(ledger_path.read_text(encoding="utf-8")))
    if args.dry_run:
        print(json.dumps({"character": args.character, "moves": len(moves),
                          "executable": sum(move["command"]["parser_status"] == "parsed"
                                            for move in moves)}, indent=2))
        return 0

    try:
        import keyboard
    except ImportError as exc:
        raise RuntimeError('install live dependencies with: pip install -e ".[live]"') from exc
    controller = VGamepadInputBackend(facing=args.facing, asynchronous=True, hold_timeout=2.0)
    try:
        print("Focus offline Practice mode. F8=pass, F9=fail, F10=skip, Esc=stop.", flush=True)
        time.sleep(args.start_delay)
        for move in moves:
            stable_id = move["stable_id"]
            if (ledger["moves"].get(stable_id) or {}).get("status") == "pass":
                continue
            if move["command"]["parser_status"] != "parsed":
                ledger["moves"][stable_id] = {"status": "blocked_parser"}
                continue
            print(f'{stable_id} | {move["name"]} | {move["command"]["raw"]}', flush=True)
            controller.set_enabled(True)
            controller.send_command(move["command"], move_id=stable_id)
            time.sleep(args.between_seconds)
            while True:
                if keyboard.is_pressed("esc"):
                    return 130
                if keyboard.is_pressed("f8"):
                    status = "pass"
                    break
                if keyboard.is_pressed("f9"):
                    status = "fail"
                    break
                if keyboard.is_pressed("f10"):
                    status = "skip"
                    break
                time.sleep(0.03)
            ledger["moves"][stable_id] = {
                "status": status,
                "validated_at": datetime.now(timezone.utc).isoformat(),
                "command": move["command"]["raw"],
            }
            temporary = ledger_path.with_suffix(".json.tmp")
            temporary.write_text(json.dumps(ledger, indent=2) + "\n", encoding="utf-8")
            temporary.replace(ledger_path)
            while any(keyboard.is_pressed(key) for key in ("f8", "f9", "f10")):
                time.sleep(0.03)
    finally:
        controller.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
