from __future__ import annotations

import argparse

from t8_agent.sim import SimAction, TekkenLiteEnv


ATTACKS = [
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]


def simple_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    state = env.state
    me = state.p1 if player == 1 else state.p2
    distance = state.distance
    if me.busy:
        return SimAction.NEUTRAL
    if distance > 1.15:
        return SimAction.WALK_FORWARD
    if distance < 0.52:
        return env.rng.choice([SimAction.THROW, SimAction.WALK_BACK, SimAction.BLOCK_HIGH])
    if env.rng.random() < 0.18:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice(ATTACKS[:-1])


def random_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    _ = player
    return env.sample_action()


def run(episodes: int, seed: int, max_decisions: int) -> None:
    env = TekkenLiteEnv(seed=seed)
    wins = {1: 0, 2: 0, None: 0}
    total_reward = 0.0
    for episode_idx in range(1, episodes + 1):
        env.reset(seed=seed + episode_idx)
        ep_reward = 0.0
        for _ in range(max_decisions):
            p1_action = simple_policy(env, player=1)
            p2_action = random_policy(env, player=2)
            result = env.step(p1_action, p2_action)
            ep_reward += result.reward_p1
            if result.terminated or result.truncated:
                break
        wins[env.state.winner] += 1
        total_reward += ep_reward
        print(
            f"episode={episode_idx} winner={env.state.winner} "
            f"frames={env.state.frame} p1_hp={env.state.p1.health:.1f} "
            f"p2_hp={env.state.p2.health:.1f} reward_p1={ep_reward:.2f}"
        )
    print(f"summary episodes={episodes} wins={wins} avg_reward_p1={total_reward / episodes:.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run fast Tekken-lite simulator episodes.")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-decisions", type=int, default=2000)
    args = parser.parse_args()
    run(episodes=args.episodes, seed=args.seed, max_decisions=args.max_decisions)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
