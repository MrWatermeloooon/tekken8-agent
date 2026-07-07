from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from t8_agent.sim.opponents import SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv
from t8_agent.train.linear_policy import LinearPolicy


@dataclass(frozen=True)
class EvaluationResult:
    score: float
    win_rate: float
    avg_reward: float
    avg_frames: float


def evaluate_policy(
    policy: LinearPolicy,
    episodes: int,
    seed: int,
    max_decisions: int,
    opponent_names: list[str],
) -> EvaluationResult:
    total_reward = 0.0
    wins = 0
    frames = 0
    for episode_idx in range(episodes):
        env = TekkenLiteEnv(seed=seed + episode_idx)
        env.reset(seed=seed + episode_idx)
        opponent = SCRIPTED_POLICIES[opponent_names[episode_idx % len(opponent_names)]]
        episode_reward = 0.0
        for _ in range(max_decisions):
            p1_action = _safe_policy_action(policy, env)
            p2_action = opponent(env, 2)
            result = env.step(p1_action, p2_action)
            episode_reward += result.reward_p1
            if result.terminated or result.truncated:
                break
        total_reward += episode_reward
        wins += int(env.state.winner == 1)
        frames += env.state.frame
    win_rate = wins / episodes
    avg_reward = total_reward / episodes
    avg_frames = frames / episodes
    score = avg_reward + 25.0 * win_rate
    return EvaluationResult(score=score, win_rate=win_rate, avg_reward=avg_reward, avg_frames=avg_frames)


def train_cem(
    generations: int,
    population: int,
    elite_fraction: float,
    noise_std: float,
    episodes_per_candidate: int,
    seed: int,
    max_decisions: int,
    opponent_names: list[str],
) -> tuple[LinearPolicy, list[EvaluationResult]]:
    rng = np.random.default_rng(seed)
    mean = LinearPolicy.zeros().weights
    elite_count = max(1, int(population * elite_fraction))
    history: list[EvaluationResult] = []
    best_policy = LinearPolicy(weights=mean.copy())
    best_result = EvaluationResult(score=float("-inf"), win_rate=0.0, avg_reward=0.0, avg_frames=0.0)

    for generation in range(1, generations + 1):
        candidates = []
        for candidate_idx in range(population):
            weights = mean + rng.normal(0.0, noise_std, size=mean.shape).astype(np.float32)
            policy = LinearPolicy(weights=weights)
            result = evaluate_policy(
                policy=policy,
                episodes=episodes_per_candidate,
                seed=seed + generation * 10_000 + candidate_idx * 100,
                max_decisions=max_decisions,
                opponent_names=opponent_names,
            )
            candidates.append((result, weights))
            if result.score > best_result.score:
                best_result = result
                best_policy = LinearPolicy(weights=weights.copy())

        candidates.sort(key=lambda item: item[0].score, reverse=True)
        elites = candidates[:elite_count]
        mean = np.mean([weights for _, weights in elites], axis=0).astype(np.float32)
        generation_best = candidates[0][0]
        history.append(generation_best)
        print(
            f"generation={generation} "
            f"score={generation_best.score:.2f} "
            f"win_rate={generation_best.win_rate:.2f} "
            f"avg_reward={generation_best.avg_reward:.2f} "
            f"avg_frames={generation_best.avg_frames:.0f}"
        )

    return best_policy, history


def _safe_policy_action(policy: LinearPolicy, env: TekkenLiteEnv) -> SimAction:
    if env.state.p1.busy:
        return SimAction.NEUTRAL
    return policy.act(env, player=1)
