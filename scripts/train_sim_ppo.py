from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import strftime

from t8_agent.sim.opponents import DEFAULT_SCRIPTED_OPPONENTS
from t8_agent.train.ppo_eval import evaluate_maskable_model
from t8_agent.train.ppo_opponents import vecnormalize_path
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MaskablePPO on Tekken-lite.")
    parser.add_argument("--timesteps", type=int, default=20_000)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--checkpoint", default="checkpoints/sim_ppo_policy.zip")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument("--tensorboard", action="store_true", help="Write TensorBoard logs under the run directory.")
    parser.add_argument("--no-normalize", action="store_true", help="Disable VecNormalize observation/reward normalization.")
    parser.add_argument(
        "--opponents",
        nargs="+",
        default=DEFAULT_SCRIPTED_OPPONENTS,
    )
    args = parser.parse_args()

    try:
        from sb3_contrib import MaskablePPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install with: .\\.venv\\Scripts\\python -m pip install -e \".[rl]\""
        ) from exc

    run_dir = Path(args.run_dir or f"runs/sim_ppo_{strftime('%Y%m%d_%H%M%S')}")
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_env = DummyVecEnv(
        [
            lambda: TekkenLiteSingleAgentEnv(
                opponent_names=args.opponents,
                seed=args.seed,
                max_decisions=args.max_decisions,
            )
        ]
    )
    env = raw_env if args.no_normalize else VecNormalize(raw_env, norm_obs=True, norm_reward=True)
    model = MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=args.gamma,
        seed=args.seed,
        verbose=1,
        tensorboard_log=str(run_dir / "tb") if args.tensorboard else None,
    )
    model.learn(total_timesteps=args.timesteps, progress_bar=False)

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(checkpoint)
    normalizer_path = None
    if isinstance(env, VecNormalize):
        normalizer_path = vecnormalize_path(checkpoint)
        env.save(str(normalizer_path))

    evaluation = evaluate_maskable_model(
        model=model,
        episodes=args.eval_episodes,
        seed=args.seed + 500_000,
        max_decisions=args.max_decisions,
        opponent_names=args.opponents,
        normalizer=env if isinstance(env, VecNormalize) else None,
    )
    metrics = {
        "checkpoint": str(checkpoint),
        "run_dir": str(run_dir),
        "timesteps": args.timesteps,
        "seed": args.seed,
        "max_decisions": args.max_decisions,
        "n_steps": args.n_steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "normalize": not args.no_normalize,
        "vecnormalize": str(normalizer_path) if normalizer_path else None,
        "opponents": args.opponents,
        "eval": {
            "win_rate": evaluation.win_rate,
            "avg_reward": evaluation.avg_reward,
            "avg_frames": evaluation.avg_frames,
        },
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(
        f"saved={checkpoint} run_dir={run_dir} "
        f"eval_win_rate={evaluation.win_rate:.2f} "
        f"eval_avg_reward={evaluation.avg_reward:.2f} "
        f"eval_avg_frames={evaluation.avg_frames:.0f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
