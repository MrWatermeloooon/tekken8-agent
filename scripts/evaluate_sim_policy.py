from __future__ import annotations

import argparse
import json
from pathlib import Path

from t8_agent.sim.opponents import DEFAULT_SCRIPTED_OPPONENTS
from t8_agent.train.cem import evaluate_policy
from t8_agent.train.linear_policy import LinearPolicy


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a lightweight simulator checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/sim_linear_policy.npz")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=9001)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_SCRIPTED_OPPONENTS,
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    policy = LinearPolicy.load(checkpoint)
    result = evaluate_policy(
        policy=policy,
        episodes=args.episodes,
        seed=args.seed,
        max_decisions=args.max_decisions,
        opponent_names=args.opponents,
    )
    payload = {
        "checkpoint": str(checkpoint),
        "episodes": args.episodes,
        "seed": args.seed,
        "max_decisions": args.max_decisions,
        "opponents": args.opponents,
        "score": result.score,
        "win_rate": result.win_rate,
        "avg_reward": result.avg_reward,
        "avg_frames": result.avg_frames,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"checkpoint={checkpoint} "
        f"score={result.score:.2f} "
        f"win_rate={result.win_rate:.2f} "
        f"avg_reward={result.avg_reward:.2f} "
        f"avg_frames={result.avg_frames:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
