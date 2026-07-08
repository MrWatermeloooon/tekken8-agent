from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from time import strftime

from t8_agent.sim.opponents import DEFAULT_SCRIPTED_OPPONENTS
from t8_agent.train.curves import plot_selfplay_metrics
from t8_agent.train.elo import rank_checkpoints
from t8_agent.train.ppo_eval import evaluate_maskable_model, evaluate_model_vs_checkpoints
from t8_agent.train.ppo_opponents import OpponentPool
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


def make_selfplay_env(
    *,
    scripted_opponents: list[str],
    checkpoint_paths: list[Path],
    checkpoint_ratings: dict[str, float],
    use_elo_sampling: bool,
    target_rating: float | None,
    scripted_sample_rate: float,
    old_sample_rate: float,
    max_recent: int,
    seed: int,
    max_decisions: int,
):
    def _init() -> TekkenLiteSingleAgentEnv:
        pool = OpponentPool(
            scripted_names=scripted_opponents,
            checkpoint_paths=checkpoint_paths,
            checkpoint_ratings=checkpoint_ratings if use_elo_sampling else None,
            target_rating=target_rating,
            scripted_sample_rate=scripted_sample_rate,
            old_checkpoint_sample_rate=old_sample_rate,
            max_recent_checkpoints=max_recent,
            rng=random.Random(seed),
        )
        return TekkenLiteSingleAgentEnv(
            opponent_names=scripted_opponents,
            seed=seed,
            max_decisions=max_decisions,
            opponent_sampler=pool.sample,
        )

    return _init


def build_training_env(
    *,
    scripted_opponents: list[str],
    checkpoint_paths: list[Path],
    checkpoint_ratings: dict[str, float],
    use_elo_sampling: bool,
    target_rating: float | None,
    scripted_sample_rate: float,
    old_sample_rate: float,
    max_recent: int,
    seed: int,
    max_decisions: int,
    n_envs: int,
    vec_env: str,
):
    if n_envs == 1:
        return make_selfplay_env(
            scripted_opponents=scripted_opponents,
            checkpoint_paths=list(checkpoint_paths),
            checkpoint_ratings=dict(checkpoint_ratings),
            use_elo_sampling=use_elo_sampling,
            target_rating=target_rating,
            scripted_sample_rate=scripted_sample_rate,
            old_sample_rate=old_sample_rate,
            max_recent=max_recent,
            seed=seed,
            max_decisions=max_decisions,
        )()

    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

    env_fns = [
        make_selfplay_env(
            scripted_opponents=scripted_opponents,
            checkpoint_paths=list(checkpoint_paths),
            checkpoint_ratings=dict(checkpoint_ratings),
            use_elo_sampling=use_elo_sampling,
            target_rating=target_rating,
            scripted_sample_rate=scripted_sample_rate,
            old_sample_rate=old_sample_rate,
            max_recent=max_recent,
            seed=seed + env_idx * 10_000,
            max_decisions=max_decisions,
        )
        for env_idx in range(n_envs)
    ]
    if vec_env == "dummy":
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns, start_method="spawn")


def main() -> int:
    parser = argparse.ArgumentParser(description="Train MaskablePPO with a simple checkpoint opponent pool.")
    parser.add_argument("--iterations", type=int, default=4)
    parser.add_argument("--timesteps-per-iteration", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=6060)
    parser.add_argument("--max-decisions", type=int, default=1200)
    parser.add_argument("--n-steps", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel simulator games for PPO rollouts.")
    parser.add_argument(
        "--vec-env",
        choices=["subproc", "dummy"],
        default="subproc",
        help="Vector-env backend used when --n-envs is greater than 1.",
    )
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--checkpoint-eval-episodes", type=int, default=4)
    parser.add_argument("--checkpoint-dir", default="checkpoints/selfplay")
    parser.add_argument("--final-checkpoint", default="checkpoints/sim_ppo_selfplay_policy.zip")
    parser.add_argument("--run-dir", default=None)
    parser.add_argument(
        "--scripted-sample-rate",
        type=float,
        default=0.35,
        help="Probability of sampling a scripted opponent after checkpoints exist.",
    )
    parser.add_argument(
        "--full-self-play",
        action="store_true",
        help="After bootstrap, train only against the checkpoint pool instead of scripted opponents.",
    )
    parser.add_argument(
        "--bootstrap-iterations",
        type=int,
        default=1,
        help="Number of early iterations allowed to use scripted opponents before full self-play takes over.",
    )
    parser.add_argument("--old-sample-rate", type=float, default=0.15)
    parser.add_argument("--max-recent", type=int, default=8)
    parser.add_argument("--elo-sampling", action="store_true", help="Sample checkpoint opponents near the latest Elo rating.")
    parser.add_argument("--elo-episodes-per-pair", type=int, default=1)
    parser.add_argument(
        "--scripted-opponents",
        nargs="+",
        default=DEFAULT_SCRIPTED_OPPONENTS,
    )
    args = parser.parse_args()
    if not 0.0 <= args.scripted_sample_rate <= 1.0:
        parser.error("--scripted-sample-rate must be between 0 and 1")
    if not 0.0 <= args.old_sample_rate <= 1.0:
        parser.error("--old-sample-rate must be between 0 and 1")
    if args.max_recent < 1:
        parser.error("--max-recent must be at least 1")
    if args.bootstrap_iterations < 0:
        parser.error("--bootstrap-iterations must be 0 or greater")
    if args.n_envs < 1:
        parser.error("--n-envs must be at least 1")

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
    checkpoint_ratings: dict[str, float] = {}
    model = None
    history = []

    for iteration in range(1, args.iterations + 1):
        if args.full_self_play and checkpoint_paths and iteration > args.bootstrap_iterations:
            effective_scripted_sample_rate = 0.0
        else:
            effective_scripted_sample_rate = args.scripted_sample_rate
        env = build_training_env(
            scripted_opponents=args.scripted_opponents,
            checkpoint_paths=checkpoint_paths,
            checkpoint_ratings=checkpoint_ratings,
            use_elo_sampling=args.elo_sampling,
            target_rating=checkpoint_ratings.get(str(checkpoint_paths[-1])) if args.elo_sampling and checkpoint_paths else None,
            scripted_sample_rate=effective_scripted_sample_rate,
            old_sample_rate=args.old_sample_rate,
            max_recent=args.max_recent,
            seed=args.seed + iteration,
            max_decisions=args.max_decisions,
            n_envs=args.n_envs,
            vec_env=args.vec_env,
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
            previous_env = model.get_env()
            model.set_env(env)
            if previous_env is not None:
                previous_env.close()

        model.learn(total_timesteps=args.timesteps_per_iteration, reset_num_timesteps=False, progress_bar=False)
        checkpoint = checkpoint_dir / f"iter_{iteration:03d}.zip"
        model.save(checkpoint)
        previous_checkpoints = list(checkpoint_paths)
        checkpoint_paths.append(checkpoint)
        if args.elo_sampling and len(checkpoint_paths) >= 2:
            checkpoint_ratings = rank_checkpoints(
                checkpoint_paths=checkpoint_paths,
                episodes_per_pair=args.elo_episodes_per_pair,
                seed=args.seed + 900_000 + iteration * 1000,
                max_decisions=args.max_decisions,
            )
        scripted_eval = evaluate_maskable_model(
            model=model,
            episodes=args.eval_episodes,
            seed=args.seed + 700_000 + iteration * 1000,
            max_decisions=args.max_decisions,
            opponent_names=args.scripted_opponents,
        )
        checkpoint_eval = evaluate_model_vs_checkpoints(
            model=model,
            checkpoint_paths=previous_checkpoints,
            episodes_per_checkpoint=args.checkpoint_eval_episodes,
            seed=args.seed + 800_000 + iteration * 1000,
            max_decisions=args.max_decisions,
        )
        item = {
            "iteration": iteration,
            "checkpoint": str(checkpoint),
            "pool_size": len(checkpoint_paths),
            "n_envs": args.n_envs,
            "vec_env": args.vec_env if args.n_envs > 1 else "single",
            "full_self_play": args.full_self_play,
            "bootstrap_iterations": args.bootstrap_iterations,
            "scripted_sample_rate": args.scripted_sample_rate,
            "effective_scripted_sample_rate": effective_scripted_sample_rate,
            "scripted_eval_win_rate": scripted_eval.win_rate,
            "scripted_eval_avg_reward": scripted_eval.avg_reward,
            "scripted_eval_avg_frames": scripted_eval.avg_frames,
            "checkpoint_eval_win_rate": checkpoint_eval.win_rate if checkpoint_eval else None,
            "checkpoint_eval_avg_reward": checkpoint_eval.avg_reward if checkpoint_eval else None,
            "checkpoint_eval_avg_frames": checkpoint_eval.avg_frames if checkpoint_eval else None,
            "latest_checkpoint_elo": checkpoint_ratings.get(str(checkpoint)) if checkpoint_ratings else None,
        }
        history.append(item)
        print(
            f"iteration={iteration} checkpoint={checkpoint} pool_size={len(checkpoint_paths)} "
            f"n_envs={args.n_envs} "
            f"scripted_sample_rate={effective_scripted_sample_rate:.2f} "
            f"scripted_win_rate={scripted_eval.win_rate:.2f} scripted_reward={scripted_eval.avg_reward:.2f} "
            f"checkpoint_win_rate={checkpoint_eval.win_rate if checkpoint_eval else 'na'}"
        )

    final_checkpoint = Path(args.final_checkpoint)
    final_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    model.save(final_checkpoint)
    metrics = {
        "final_checkpoint": str(final_checkpoint),
        "run_dir": str(run_dir),
        "iterations": args.iterations,
        "timesteps_per_iteration": args.timesteps_per_iteration,
        "n_envs": args.n_envs,
        "vec_env": args.vec_env if args.n_envs > 1 else "single",
        "checkpoint_eval_episodes": args.checkpoint_eval_episodes,
        "scripted_opponents": args.scripted_opponents,
        "scripted_sample_rate": args.scripted_sample_rate,
        "full_self_play": args.full_self_play,
        "bootstrap_iterations": args.bootstrap_iterations,
        "old_sample_rate": args.old_sample_rate,
        "max_recent": args.max_recent,
        "elo_sampling": args.elo_sampling,
        "elo_episodes_per_pair": args.elo_episodes_per_pair,
        "checkpoint_ratings": checkpoint_ratings,
        "history": history,
    }
    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    curves_path = plot_selfplay_metrics(metrics_path)
    print(f"saved={final_checkpoint} run_dir={run_dir} curves={curves_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
