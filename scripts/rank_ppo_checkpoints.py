from __future__ import annotations

import argparse
import json
from pathlib import Path

from t8_agent.train.elo import rank_checkpoints


def main() -> int:
    parser = argparse.ArgumentParser(description="Rank PPO checkpoints with a simple Elo round robin.")
    parser.add_argument("--checkpoint-dir", default="checkpoints/selfplay")
    parser.add_argument("--episodes-per-pair", type=int, default=2)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=8080)
    parser.add_argument("--k", type=float, default=32.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    checkpoints = sorted(Path(args.checkpoint_dir).glob("*.zip"))
    if len(checkpoints) < 2:
        raise SystemExit("need at least two PPO checkpoints to rank")

    ratings = rank_checkpoints(
        checkpoint_paths=checkpoints,
        episodes_per_pair=args.episodes_per_pair,
        seed=args.seed,
        max_decisions=args.max_decisions,
        k=args.k,
    )

    ranking = [
        {"checkpoint": str(path), "elo": ratings[str(path)]}
        for path in sorted(checkpoints, key=lambda item: ratings[str(item)], reverse=True)
    ]
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(ranking, indent=2), encoding="utf-8")
    for item in ranking:
        print(f"elo={item['elo']:.1f} checkpoint={item['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
