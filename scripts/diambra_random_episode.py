from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any


def summarize_observation(observation: Any) -> str:
    if isinstance(observation, Mapping):
        parts = []
        for key, value in observation.items():
            shape = getattr(value, "shape", None)
            parts.append(f"{key}={shape or type(value).__name__}")
        return ", ".join(parts)
    shape = getattr(observation, "shape", None)
    return f"observation={shape or type(observation).__name__}"


def run(game_id: str, render: bool, steps: int, seed: int, characters: tuple[str, ...] | None) -> None:
    import diambra.arena
    from diambra.arena import EnvironmentSettings

    render_mode = "human" if render else None
    env_settings = EnvironmentSettings(seed=seed, characters=characters)
    env = diambra.arena.make(game_id, env_settings=env_settings, render_mode=render_mode)
    observation, info = env.reset(seed=seed)
    print(
        f"reset game={game_id} characters={characters or 'random'} "
        f"obs={summarize_observation(observation)} info_keys={list(info.keys())}"
    )

    total_reward = 0.0
    for step_idx in range(1, steps + 1):
        if render:
            env.render()
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        if step_idx == 1 or terminated or truncated:
            print(
                f"step={step_idx} reward={reward} total={total_reward:.3f} "
                f"terminated={terminated} truncated={truncated} "
                f"obs={summarize_observation(observation)}"
            )
        if terminated or truncated:
            break

    env.close()
    print(f"done steps={step_idx} total_reward={total_reward:.3f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a random DIAMBRA episode.")
    parser.add_argument("--game", default="tektagt", help="DIAMBRA game id.")
    parser.add_argument("--render", action="store_true", help="Render the emulator window.")
    parser.add_argument("--steps", type=int, default=1000, help="Maximum environment steps.")
    parser.add_argument("--seed", type=int, default=42, help="Environment seed.")
    parser.add_argument(
        "--characters",
        nargs="*",
        default=["Jun", "Jin"],
        help="Character names for games that support character selection.",
    )
    args = parser.parse_args()

    characters = tuple(args.characters) if args.characters else None
    run(game_id=args.game, render=args.render, steps=args.steps, seed=args.seed, characters=characters)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
