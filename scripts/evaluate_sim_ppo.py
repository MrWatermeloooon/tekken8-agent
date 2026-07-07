from __future__ import annotations

import argparse
import json
from pathlib import Path

from t8_agent.train.ppo_eval import evaluate_maskable_model


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a MaskablePPO Tekken-lite checkpoint.")
    parser.add_argument("--checkpoint", default="checkpoints/sim_ppo_policy.zip")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=3333)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=["poke", "rushdown", "turtle", "whiff_punish", "random"],
    )
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install with: .\\.venv\\Scripts\\python -m pip install -e \".[rl]\""
        ) from exc

    model = MaskablePPO.load(args.checkpoint)
    result = evaluate_maskable_model(
        model=model,
        episodes=args.episodes,
        seed=args.seed,
        max_decisions=args.max_decisions,
        opponent_names=args.opponents,
    )
    payload = {
        "checkpoint": args.checkpoint,
        "episodes": args.episodes,
        "seed": args.seed,
        "max_decisions": args.max_decisions,
        "opponents": args.opponents,
        "win_rate": result.win_rate,
        "avg_reward": result.avg_reward,
        "avg_frames": result.avg_frames,
    }
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"checkpoint={args.checkpoint} "
        f"win_rate={result.win_rate:.2f} "
        f"avg_reward={result.avg_reward:.2f} "
        f"avg_frames={result.avg_frames:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
