from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import strftime

from t8_agent.sim.opponents import DEFAULT_SCRIPTED_OPPONENTS
from t8_agent.train.cem import evaluate_policy, train_cem


def main() -> int:
    parser = argparse.ArgumentParser(description="Train a lightweight linear policy in Tekken-lite.")
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--elite-fraction", type=float, default=0.25)
    parser.add_argument("--noise-std", type=float, default=1.0)
    parser.add_argument("--episodes-per-candidate", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--checkpoint", default="checkpoints/sim_linear_policy.npz")
    parser.add_argument("--run-dir", default=None, help="Directory for metrics/history output.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_SCRIPTED_OPPONENTS,
        help="Scripted opponent names.",
    )
    args = parser.parse_args()
    run_dir = Path(args.run_dir or f"runs/sim_linear_{strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)

    policy, history = train_cem(
        generations=args.generations,
        population=args.population,
        elite_fraction=args.elite_fraction,
        noise_std=args.noise_std,
        episodes_per_candidate=args.episodes_per_candidate,
        seed=args.seed,
        max_decisions=args.max_decisions,
        opponent_names=args.opponents,
    )
    final = evaluate_policy(
        policy=policy,
        episodes=args.eval_episodes,
        seed=args.seed + 999_000,
        max_decisions=args.max_decisions,
        opponent_names=args.opponents,
    )
    checkpoint = Path(args.checkpoint)
    policy.save(checkpoint)
    metrics = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "generations": args.generations,
        "population": args.population,
        "elite_fraction": args.elite_fraction,
        "noise_std": args.noise_std,
        "episodes_per_candidate": args.episodes_per_candidate,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "max_decisions": args.max_decisions,
        "opponents": args.opponents,
        "final": {
            "score": final.score,
            "win_rate": final.win_rate,
            "avg_reward": final.avg_reward,
            "avg_frames": final.avg_frames,
        },
        "history": [
            {
                "generation": idx,
                "score": item.score,
                "win_rate": item.win_rate,
                "avg_reward": item.avg_reward,
                "avg_frames": item.avg_frames,
            }
            for idx, item in enumerate(history, start=1)
        ],
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"saved={checkpoint} "
        f"run_dir={run_dir} "
        f"eval_score={final.score:.2f} "
        f"eval_win_rate={final.win_rate:.2f} "
        f"eval_avg_reward={final.avg_reward:.2f} "
        f"eval_avg_frames={final.avg_frames:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
