"""How a full-move checkpoint fights itself: activity, round endings, contacts.

    python tools/fight_stats.py [--run runs/train_all | --checkpoint PATH] [--matchups 10] [--sample]

Plays --matchups random matchups (two rounds each, the checkpoint on both
sides) through t8_v2_full_move_feed and reports how rounds end (KO, timeout
won on health, timeout with equal health), how soon the first contact comes,
contact types, and how much of the time the fighters do nothing (standing,
walking back, or crouching without a move). --sample draws moves from the
policy's probabilities as training does; the default is greedy (always the
most likely move), which is what evaluation and the viewer show.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
ROSTER = ["jun", "king", "kazuya", "paul", "law", "lili", "dragunov", "asuka", "bryan", "reina", "jin", "hwoarang"]


def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--matchups", type=int, default=10)
    parser.add_argument("--sample", action="store_true")
    parser.add_argument("--seed", type=int, default=3)
    args = parser.parse_args()
    checkpoint = args.checkpoint or newest(args.run / "champion") or newest(args.run / "checkpoints")
    if checkpoint is None:
        raise SystemExit("no checkpoint")
    feed = REPO / "build-gpu" / "Release" / "t8_v2_full_move_feed.exe"
    rng = random.Random(args.seed)
    endings, contacts = Counter(), Counter()
    first_contact, idle, sidestep, fighter_frames, rounds = [], 0, 0, 0, 0
    for index in range(args.matchups):
        a, b = rng.choice(ROSTER), rng.choice(ROSTER)
        command = [str(feed), "--character", a, "--opponent", b, "--checkpoint", str(checkpoint),
                   "--opponent-checkpoint", str(checkpoint), "--frames", "7200", "--interval-ms", "0",
                   "--seed", str(args.seed * 100 + index), "--allow-provisional"] + (["--sample"] if args.sample else [])
        output = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout
        frames = [json.loads(line) for line in output.splitlines() if line.startswith("{")]
        current_round, first, ended = None, None, False
        for frame in frames:
            if frame["round"] != current_round:
                if current_round is not None:
                    first_contact.append(first)
                current_round, first, ended = frame["round"], None, False
                rounds += 1
            for fighter in frame["fighters"]:
                fighter_frames += 1
                if not fighter["move"] and fighter["universal"] in (0, 2, 5):
                    idle += 1
                if not fighter["move"] and fighter["universal"] in (6, 7):
                    sidestep += 1
            if frame["contact_age"] == 0:
                for contact in frame["contacts"]:
                    contacts[contact["contact"]] += 1
                    if first is None:
                        first = frame["frame"]
            if not ended and (frame["winner"] >= 0 or frame["frame"] >= frame["max_frames"]):
                ended = True
                health = [f["health"] for f in frame["fighters"]]
                if min(health) <= 0:
                    endings["KO"] += 1
                elif health[0] != health[1]:
                    endings["timeout, won on health"] += 1
                else:
                    endings["timeout, equal health"] += 1
        first_contact.append(first)
    complete = sum(endings.values())
    no_contact = sum(1 for f in first_contact if f is None)
    seconds = sorted(f / 60 for f in first_contact if f is not None)
    print(f"{checkpoint.name}, {'sampled' if args.sample else 'greedy'} play, {complete} finished rounds")
    for name, count in endings.most_common():
        print(f"  {name}: {count} ({count / max(complete, 1):.0%})")
    print(f"  rounds without any contact: {no_contact} of {len(first_contact)}")
    if seconds:
        print(f"  first contact: median {seconds[len(seconds) // 2]:.1f} s")
    print(f"  contacts: {dict(contacts.most_common())}")
    print(f"  doing nothing (standing, backing off, crouching): {idle / max(fighter_frames, 1):.0%} of the time")
    print(f"  sidestepping: {sidestep / max(fighter_frames, 1):.0%} of the time")
    return 0


if __name__ == "__main__":
    sys.exit(main())
