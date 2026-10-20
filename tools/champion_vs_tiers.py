"""How a checkpoint does against each tier of scripted bots (sampled play).

    python tools/champion_vs_tiers.py [--checkpoint PATH | --run runs/train_all] [--fights 12] [--seed 1]

Plays the checkpoint against random bots of every tier (random characters, both
sides) through t8_v2_full_move_feed and prints its round score per tier (draws
half). The bots are useful sparring only while the score is well below 100%.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import random
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bot_ladder import CHARACTERS, FEED, REPO, TIERS, TIER_NAMES  # noqa: E402


def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


PERCEPTION_DELAY = 0  # set from --perception-delay: must match how the checkpoint was trained


def fight(checkpoint: Path, bot: int, own: str, other: str, seed: int, bot_first: bool) -> tuple[float, int]:
    """The checkpoint's round score and finished rounds; the checkpoint is P1 unless bot_first."""
    command = [str(FEED), "--frames", "10800", "--interval-ms", "0", "--seed", str(seed), "--allow-provisional",
               "--sample", "--perception-delay", str(PERCEPTION_DELAY)]
    if bot_first:  # the bot plays P1 (no checkpoint); the checkpoint plays P2
        command += ["--character", other, "--opponent", own, "--bot", str(bot), "--opponent-checkpoint", str(checkpoint)]
    else:
        command += ["--character", own, "--opponent", other, "--checkpoint", str(checkpoint), "--opponent-bot", str(bot)]
    out = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout
    score, rounds, current, last = 0.0, 0, None, None
    side = 1 if bot_first else 0
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        frame = json.loads(line)
        if current is not None and frame["round"] != current and last is not None:
            health = [f["health"] for f in last["fighters"]]
            mine, theirs = health[side], health[1 - side]
            score += 1.0 if mine > theirs else 0.5 if mine == theirs else 0.0
            rounds += 1
        current, last = frame["round"], frame
    return score, rounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--fights", type=int, default=12)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--perception-delay", type=int, default=0, help="the checkpoint's training --perception-delay")
    args = parser.parse_args()
    global PERCEPTION_DELAY
    PERCEPTION_DELAY = args.perception_delay
    checkpoint = args.checkpoint or newest(args.run / "champion")
    if checkpoint is None:
        raise SystemExit("no champion")
    rng = random.Random(args.seed)
    jobs = [(tier, rng.choice(TIERS[tier]), rng.choice(CHARACTERS), rng.choice(CHARACTERS), rng.randrange(1 << 30),
             index % 2 == 1) for tier in range(len(TIERS)) for index in range(args.fights)]
    with ThreadPoolExecutor(max(1, min(6, (os.cpu_count() or 4) - 2))) as pool:
        results = list(pool.map(lambda job: fight(checkpoint, *job[1:]), jobs))
    print(f"{checkpoint} against each tier (round score, draws half):")
    for tier in range(len(TIERS)):
        score = sum(r[0] for job, r in zip(jobs, results) if job[0] == tier)
        rounds = sum(r[1] for job, r in zip(jobs, results) if job[0] == tier)
        print(f"  {TIER_NAMES[tier]:<16} {score / max(rounds, 1):5.0%} of {rounds} rounds")
    return 0


if __name__ == "__main__":
    sys.exit(main())
