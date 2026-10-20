"""Draw-aware ratings for a checkpoint pool from an arena table.

    t8_v2_full_move_arena --output runs/x/arena.csv ckpt_a.t8ppo ckpt_b.t8ppo ...
    python tools/rank_checkpoints.py runs/x/arena.csv [--anchor 0] [--bootstraps 400]

Fits a Bradley-Terry model by maximum likelihood (Hunter's MM iterations) in
which a draw counts as half a win for each side, so ratings are on the Elo
scale (400 points = 10:1 odds) and draws pull ratings together. Every pair
also gets one virtual draw, which keeps unbeaten or winless checkpoints at a
finite rating. 95% intervals come from a parametric bootstrap over each
pair's win/loss/draw counts with a fixed seed, so the same table always gives
the same report. Writes <table>.ratings.json and .md next to the table.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random


def fit(pairs: list[tuple[int, int, float, float, float]], count: int, iterations: int = 2000) -> list[float]:
    """Bradley-Terry strengths (geometric mean 1) from (a, b, wins_a, wins_b, draws)."""

    wins = [0.0] * count
    games: dict[tuple[int, int], float] = {}
    for a, b, wins_a, wins_b, draws in pairs:
        # One virtual draw per pair regularizes perfect records.
        total = wins_a + wins_b + draws + 1.0
        wins[a] += wins_a + 0.5 * (draws + 1.0)
        wins[b] += wins_b + 0.5 * (draws + 1.0)
        games[(a, b)] = games.get((a, b), 0.0) + total
    strength = [1.0] * count
    for _ in range(iterations):
        updated = []
        for i in range(count):
            denominator = sum(n / (strength[i] + strength[j if i == k else k])
                              for (k, j), n in games.items() if i in (k, j))
            updated.append(wins[i] / denominator if denominator > 0 else strength[i])
        scale = math.exp(sum(math.log(value) for value in updated) / count)
        updated = [value / scale for value in updated]
        if max(abs(math.log(u / s)) for u, s in zip(updated, strength)) < 1e-12:
            strength = updated
            break
        strength = updated
    return strength


def to_elo(strength: list[float], anchor: int, base: float = 1000.0) -> list[float]:
    return [base + 400.0 * math.log10(value / strength[anchor]) for value in strength]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("table", type=Path)
    parser.add_argument("--anchor", type=int, default=0, help="Entrant fixed at 1000 (default: the first).")
    parser.add_argument("--bootstraps", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261007)
    args = parser.parse_args()
    with args.table.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    pairs = [(int(row["a"]), int(row["b"]), float(row["wins_a"]), float(row["wins_b"]), float(row["draws"]))
             for row in rows]
    count = 1 + max(max(a, b) for a, b, *_ in pairs)
    names_path = args.table.with_suffix(".entrants.csv")
    names = [str(index) for index in range(count)]
    if names_path.exists():
        with names_path.open(encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                names[int(row["index"])] = row["checkpoint"]
    ratings = to_elo(fit(pairs, count), args.anchor)

    rng = random.Random(args.seed)
    samples: list[list[float]] = []
    for _ in range(args.bootstraps):
        resampled = []
        for a, b, wins_a, wins_b, draws in pairs:
            total = int(wins_a + wins_b + draws)
            weights = (wins_a, wins_b, draws)
            draws_of = rng.choices((0, 1, 2), weights=weights, k=total) if total else []
            resampled.append((a, b, draws_of.count(0), draws_of.count(1), draws_of.count(2)))
        samples.append(to_elo(fit(resampled, count, 300), args.anchor))
    intervals = []
    for index in range(count):
        values = sorted(sample[index] for sample in samples)
        intervals.append((values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))])
                         if values else (ratings[index], ratings[index]))
    order = sorted(range(count), key=lambda index: -ratings[index])
    report = {
        "table": str(args.table), "anchor": names[args.anchor], "bootstraps": args.bootstraps, "seed": args.seed,
        "model": "Bradley-Terry, draws as half wins, one virtual draw per pair, Elo scale",
        "ratings": [{"rank": rank + 1, "entrant": names[index], "elo": round(ratings[index], 1),
                     "ci95": [round(intervals[index][0], 1), round(intervals[index][1], 1)]}
                    for rank, index in enumerate(order)],
    }
    args.table.with_suffix(".ratings.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    lines = ["| Rank | Checkpoint | Elo | 95% interval |", "|---:|---|---:|---:|"]
    for entry in report["ratings"]:
        lines.append(f"| {entry['rank']} | {entry['entrant']} | {entry['elo']:.0f} | "
                     f"{entry['ci95'][0]:.0f} to {entry['ci95'][1]:.0f} |")
    args.table.with_suffix(".ratings.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
