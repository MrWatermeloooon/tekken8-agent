"""Which characters the bot plays best: a ranking of every character it learned.

    python tools/character_ranking.py [--run runs/train_all | --checkpoint PATH] [--rounds 32] [--pause]

The checkpoint (default: the run's newest champion) plays itself in every matchup of its
characters through t8_v2_full_move_arena --character-ranking: each ordered pair --rounds
times (half on each side), sampled play, with the perception delay and hold frames its
run trained with. Because the same policy plays both sides, a character's score is how
well the bot plays that character against the rest of the cast (draws count half). It
mixes the bot's skill with the character and with the simulator's stand-in move geometry,
so it is "the bot's best character", not a real Tekken tier list.

Prints the ranking with a 95% margin, each character's best and worst matchup, and writes
<run>/character_ranking.md and <run>/character_ranking_matchups.csv.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
from pathlib import Path
import re
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
EXE = ".exe" if os.name == "nt" else ""
ARENA = REPO / "build-gpu" / "Release" / f"t8_v2_full_move_arena{EXE}"


def newest(directory: Path) -> Path | None:
    found = [p for p in directory.glob("update_*.t8ppo") if p.with_suffix(".t8fmstate").exists()]
    return max(found, key=lambda p: int(p.stem.split("_")[1])) if found else None


def run_setting(checkpoint: Path, key: str, fallback: int) -> int:
    for run_json in (checkpoint.parent.parent / "run.json", checkpoint.parent / "run.json"):
        if run_json.exists():
            match = re.search(rf";{key}=(\d+)", run_json.read_text(encoding="utf-8"))
            return int(match.group(1)) if match else fallback
    return fallback


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--run", type=Path, default=REPO / "runs" / "train_all")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--rounds", type=int, default=32, help="rounds per ordered pair of characters (even)")
    parser.add_argument("--seed", type=int, default=31337)
    parser.add_argument("--pause", action="store_true", help="wait for Enter before closing (T8Train's C key)")
    args = parser.parse_args()
    try:
        return rank(args)
    finally:
        if args.pause:
            input("\nPress Enter to close.")


def rank(args: argparse.Namespace) -> int:
    run = args.run if args.run.is_absolute() else REPO / args.run
    checkpoint = args.checkpoint or newest(run / "champion") or newest(run / "checkpoints")
    if checkpoint is None:
        print(f"no champion or checkpoint in {run} yet")
        return 1
    output_dir = checkpoint.parent.parent
    perception = run_setting(checkpoint, "perception", 0)
    hold = run_setting(checkpoint, "hold", 4)
    matchups = output_dir / "character_ranking_matchups.csv"
    print(f"Ranking the characters of {checkpoint.parent.parent.name}/{checkpoint.parent.name}/{checkpoint.name}")
    print(f"(self-play in every matchup, {args.rounds} rounds per pair, sampled play, perception delay {perception}, "
          f"hold {hold}; this takes about a minute)\n", flush=True)
    command = [str(ARENA), "--character-ranking", "--output", str(matchups), "--rounds-per-pair", str(args.rounds),
               "--seed", str(args.seed), "--sample", "--perception-delay", str(perception), "--hold-frames", str(hold),
               "--allow-provisional", str(checkpoint)]
    if subprocess.run(command, cwd=REPO).returncode != 0:
        print("the arena failed")
        return 1
    rows = list(csv.DictReader(matchups.open(encoding="utf-8")))
    # Score per character from both directions: its own rows, and the rows where it is the opponent.
    score: dict[str, list[float]] = {}
    pair: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        a, b = r["character"], r["opponent"]
        wins, losses, draws = int(r["wins"]), int(r["losses"]), int(r["draws"])
        if a == b:
            continue
        for who, against, points in ((a, b, wins + draws / 2), (b, a, losses + draws / 2)):
            total = wins + losses + draws
            score.setdefault(who, [0.0, 0])
            score[who][0] += points
            score[who][1] += total
            cell = pair.setdefault((who, against), [0.0, 0])
            cell[0] += points
            cell[1] += total
    ranked = sorted(score, key=lambda c: score[c][0] / max(score[c][1], 1), reverse=True)
    lines = []
    header = f"{'#':>3}  {'character':<12} {'win rate':>9}  {'+-95%':>5}   {'best matchup':<22} worst matchup"
    print(header)
    print("-" * len(header))
    for place, character in enumerate(ranked, 1):
        points, total = score[character]
        rate = points / max(total, 1)
        margin = 1.96 * math.sqrt(max(rate * (1 - rate), 1e-9) / max(total, 1))
        against = {b: v[0] / max(v[1], 1) for (a, b), v in pair.items() if a == character}
        best = max(against, key=against.get)
        worst = min(against, key=against.get)
        line = (f"{place:>3}  {character:<12} {rate:>9.1%}  {margin:>5.1%}   {best + f' {against[best]:.0%}':<22} "
                f"{worst} {against[worst]:.0%}")
        print(line)
        lines.append(f"| {place} | {character} | {rate:.1%} | ±{margin:.1%} | {best} {against[best]:.0%} | "
                     f"{worst} {against[worst]:.0%} |")
    print(f"\nBest character: {ranked[0]}   Worst: {ranked[-1]}")
    print("Characters whose margins overlap are not reliably different; raise --rounds to separate them.")
    report = output_dir / "character_ranking.md"
    report.write_text("\n".join([
        f"# Character ranking: {checkpoint.name}", "",
        f"Self-play in every matchup, {args.rounds} rounds per ordered pair, sampled play, perception delay "
        f"{perception}, hold {hold}. Win rate against the rest of the cast (draws half). Matchups: "
        f"{matchups.name}.", "",
        "| # | character | win rate | 95% | best matchup | worst matchup |", "|---:|---|---:|---:|---|---|",
        *lines, ""]), encoding="utf-8")
    print(f"Saved {report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
