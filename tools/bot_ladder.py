"""Round robin of the tiered scripted bots: are higher tiers stronger?

    python tools/bot_ladder.py [--fights 8] [--seed 1] [--bots]

Plays every pair of tiers (a random bot of each tier, random characters, both
sides) through t8_v2_full_move_feed and prints each tier's round score against
each other tier (draws count half). With --bots, every bot also plays a fixed
set of fights against the tier-3 rushdown, for a per-bot strength list.
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

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
FEED = REPO / "build-gpu" / "Release" / f"t8_v2_full_move_feed{EXE}"
# Bot ids per tier (include/t8_v2/full_move_contract.hpp, scripted_profile): the 30
# hand-tuned bots, then 43 generated bots per tier (ids 30+, tier = (id - 30) // 43).
HANDMADE = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9, 10], [11, 12, 13, 14, 15, 16], [17, 18, 19, 20, 21, 22],
            [23, 24, 25], [26, 27, 28, 29]]
GENERATED_PER_TIER = 43
TIERS = [bots + list(range(30 + tier * GENERATED_PER_TIER, 30 + (tier + 1) * GENERATED_PER_TIER))
         for tier, bots in enumerate(HANDMADE)]
TIER_NAMES = ["0 dummy", "1 beginner", "2 intermediate", "3 advanced", "4 expert", "5 master", "6 pro"]
BOT_NAMES = ["walking dummy", "standing dummy", "crouching dummy", "masher", "low spammer", "throw spammer",
             "backdasher"] + [f"{style}" for style in ("rushdown", "turtle", "keep-away", "mix-up")] + \
    [f"{style}" for style in ("rushdown", "turtle", "keep-away", "sidestepper", "mix-up", "heat user")] * 2 + \
    ["rushdown", "turtle", "all-rounder", "movement", "pressure", "defense", "mix-up"]
CHARACTERS = ["jun", "king", "kazuya", "paul", "law", "lili", "dragunov", "asuka", "bryan", "reina", "jin",
              "hwoarang", "feng", "nina", "leroy", "claudio", "lars", "steve"]


def tier_of(bot: int) -> int:
    return next(index for index, bots in enumerate(TIERS) if bot in bots)


def fight(p1_bot: int, p2_bot: int, p1: str, p2: str, seed: int, frames: int = 10800) -> tuple[float, int]:
    """P1's round score (draws half) and the number of finished rounds."""
    command = [str(FEED), "--character", p1, "--opponent", p2, "--bot", str(p1_bot), "--opponent-bot", str(p2_bot),
               "--frames", str(frames), "--interval-ms", "0", "--seed", str(seed), "--allow-provisional"]
    out = subprocess.run(command, cwd=REPO, capture_output=True, text=True, check=True).stdout
    score, rounds, current, last = 0.0, 0, None, None
    for line in out.splitlines():
        if not line.startswith("{"):
            continue
        frame = json.loads(line)
        if current is not None and frame["round"] != current and last is not None:
            health = [f["health"] for f in last["fighters"]]
            score += 1.0 if health[0] > health[1] else 0.5 if health[0] == health[1] else 0.0
            rounds += 1
        current, last = frame["round"], frame
    return score, rounds


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--fights", type=int, default=8, help="fights per tier pair (each up to 3 rounds)")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--bots", action="store_true", help="also rank every bot against the tier-3 rushdown")
    args = parser.parse_args()
    rng = random.Random(args.seed)
    jobs = []
    for a in range(len(TIERS)):
        for b in range(a + 1, len(TIERS)):
            for fight_index in range(args.fights):
                p1, p2 = rng.choice(CHARACTERS), rng.choice(CHARACTERS)
                bot_a, bot_b = rng.choice(TIERS[a]), rng.choice(TIERS[b])
                swap = fight_index % 2 == 1  # both sides
                jobs.append((a, b, swap, (bot_b, bot_a, p2, p1) if swap else (bot_a, bot_b, p1, p2),
                             rng.randrange(1 << 30)))
    with ThreadPoolExecutor(max(1, (os.cpu_count() or 4) - 2)) as pool:
        results = list(pool.map(lambda job: fight(*job[3], job[4]), jobs))
    table = {}
    for (a, b, swap, _, _), (score, rounds) in zip(jobs, results):
        a_score = (rounds - score) if swap else score  # tier a's rounds
        total = table.setdefault((a, b), [0.0, 0])
        total[0] += a_score
        total[1] += rounds
    print("Round score of the row tier against the column tier (draws half):\n")
    print(" " * 16 + "".join(f"{name.split()[0]:>7}" for name in TIER_NAMES))
    for row in range(len(TIERS)):
        cells = []
        for column in range(len(TIERS)):
            if row == column:
                cells.append("     -")
                continue
            a, b = min(row, column), max(row, column)
            score, rounds = table[(a, b)]
            share = score / rounds if rounds else float("nan")
            cells.append(f"{(share if row == a else 1 - share):7.0%}")
        print(f"{TIER_NAMES[row]:<16}" + "".join(cells))
    ordered = all(table[(a, a + 1)][0] / max(table[(a, a + 1)][1], 1) < 0.5 for a in range(len(TIERS) - 1))
    print("\nEach tier beats the one below it:", "yes" if ordered else "NO")
    if args.bots:
        reference = 11  # tier-3 rushdown
        rng = random.Random(args.seed + 1)
        bot_jobs = [(bot, rng.choice(CHARACTERS), rng.choice(CHARACTERS), rng.randrange(1 << 30), index % 2 == 1)
                    for bot in range(sum(len(t) for t in TIERS)) for index in range(args.fights)]
        with ThreadPoolExecutor(max(1, (os.cpu_count() or 4) - 2)) as pool:
            bot_results = list(pool.map(
                lambda j: fight(*((reference, j[0], j[2], j[1]) if j[4] else (j[0], reference, j[1], j[2])), j[3]),
                bot_jobs))
        strength: dict[int, list[float]] = {}
        for (bot, _, _, _, swap), (score, rounds) in zip(bot_jobs, bot_results):
            total = strength.setdefault(bot, [0.0, 0])
            total[0] += (rounds - score) if swap else score
            total[1] += rounds
        print("\nEvery bot against the tier-3 rushdown:")
        for bot, (score, rounds) in sorted(strength.items(), key=lambda item: item[1][0] / max(item[1][1], 1)):
            name = BOT_NAMES[bot] if bot < len(BOT_NAMES) else "generated"
            print(f"  {bot:3d} tier {tier_of(bot)} {name:<16} {score / max(rounds, 1):5.0%} of {rounds} rounds")
    return 0 if ordered else 1


if __name__ == "__main__":
    sys.exit(main())
