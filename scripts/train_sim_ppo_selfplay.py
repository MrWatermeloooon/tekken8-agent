from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import strftime

from t8_agent.train.ppo_eval import evaluate_maskable_model
from t8_agent.train.ppo_opponents import OpponentPool
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MaskablePPO with a simple checkpoint opponent pool.")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--timesteps-per-iteration", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--checkpoint-dir", default="checkpoints/selfplay")
    parser.add_argument("--final-checkpoint", default="checkpoints/sim_ppo_selfplay_policy.zip")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--old-sample-rate", type=float, default=0.15)
    parser.add_argument("--max-recent", type=int, default=8)
    parser.add_argument(
        "--scripted-opponents",
        nargs="+",
        default=["poke", "rushdown", "turtle", "whiff_punish", "random"],
    )
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install with: .\\.venv\\Scripts\\python -m pip install -e \".[rl]\""
        ) from exc

    run_dir = Path(args.run_dir or f"runs/sim_ppo_selfplay_{strftime('%Y%m%d_%H%M%S')}")
    checkpoint_dir = Path(args.checkpoint_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: list[Path] = []
    model = None
    history = []

    for iteration in range(1, args.iterations + 1):
        pool = OpponentPool(
            scripted_names=args.scripted_opponents,
            checkpoint_paths=checkpoint_paths,
            old_checkpoint_sample_rate=args.old_sample_rate,
            max_recent_checkpoints=args.max_recent,
        )
        env = TekkenLiteSingleAgentEnv(
            opponent_names=args.scripted_opponents,
            seed=args.seed + iteration,
            max_decisions=args.max_decisions,
            opponent_sampler=pool.sample,
        )
        if model is None:
            model = MaskablePPO(
                "MlpPolicy",
                env,
                learning_rate=args.learning_rate,
                n_steps=args.n_steps,
                batch_size=args.batch_size,
                gamma=args.gamma,
                seed=args.seed,
                verbose=1,
            )
        else:
            model.set_env(env)

        model.learn(total_timesteps=args.timesteps_per_iteration, reset_num_timesteps=False, progress_bar=False)
        checkpoint = checkpoint_dir / f"iter_{iteration:03d}.zip"
        model.save(checkpoint)
        checkpoint_paths.append(checkpoint)
        evaluation = evaluate_maskable_model(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed + 700_000 + iteration * 1000,
            max_decisions=args.max_decisions,
            opponent_names=args.scripted_opponents,
        )
        item = {
            "iteration": iteration,
            "checkpoint": str(checkpoint),
            "pool_size": len(checkpoint_paths),
            "eval_win_rate": evaluation.win_rate,
            "eval_avg_reward": evaluation.avg_reward,
            "eval_avg_frames": evaluation.avg_frames,
        }
        history.append(item)
        print(
            f"iteration={iteration} checkpoint={checkpoint} pool_size={len(checkpoint_paths)} "
            f"eval_win_rate={evaluation.win_rate:.2f} eval_avg_reward={evaluation.avg_reward:.2f}"
        )

    final_checkpoint = Path(args.final_checkpoint)
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_checkpoint)
    metrics = {
        "final_checkpoint": str(final_checkpoint),
        "run_dir": str(run_dir),
        "iterations": args.iterations,
        "timesteps_per_iteration": args.timesteps_per_iteration,
        "scripted_opponents": args.scripted_opponents,
        "old_sample_rate": args.old_sample_rate,
        "max_recent": args.max_recent,
        "history": history,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"saved={final_checkpoint} run_dir={run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
