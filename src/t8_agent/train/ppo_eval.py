from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from t8_agent.sim.action_space import index_to_action, legal_action_mask
from t8_agent.sim.observations import vector_observation
from t8_agent.sim.tekken_lite import TekkenLiteEnv
from t8_agent.train.ppo_opponents import PpoCheckpointOpponent
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


@dataclass(frozen=True)
class PpoEvalResult:
    win_rate: float
    avg_reward: float
    avg_frames: float


def _clean_p1_win(winner: object, info: dict) -> bool:
    return winner == 1 and not info.get("timed_out") and not info.get("stalemate")


def _normalize_obs(normalizer, obs):
    if normalizer is None:
        return obs
    return normalizer.normalize_obs(obs)


def evaluate_maskable_model(
    model,
    episodes: int,
    seed: int,
    max_decisions: int,
    opponent_names: list[str],
    normalizer=None,
) -> PpoEvalResult:
    env = TekkenLiteSingleAgentEnv(opponent_names=opponent_names, seed=seed, max_decisions=max_decisions)
    wins = 0
    total_reward = 0.0
    total_frames = 0
    for episode_idx in range(episodes):
        obs, _info = env.reset(seed=seed + episode_idx)
        episode_reward = 0.0
        terminated = False
        truncated = False
        last_info = {}
        while not (terminated or truncated):
            action, _state = model.predict(_normalize_obs(normalizer, obs), deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, last_info = env.step(int(action))
            episode_reward += reward
        wins += int(_clean_p1_win(last_info.get("winner"), last_info))
        total_reward += episode_reward
        total_frames += int(last_info.get("frame", env.sim.state.frame))
    return PpoEvalResult(
        win_rate=wins / episodes,
        avg_reward=total_reward / episodes,
        avg_frames=total_frames / episodes,
    )


def evaluate_model_vs_checkpoints(
    model,
    checkpoint_paths: list[str | Path],
    episodes_per_checkpoint: int,
    seed: int,
    max_decisions: int,
    normalizer=None,
) -> PpoEvalResult | None:
    if not checkpoint_paths:
        return None

    wins = 0
    episodes = 0
    total_reward = 0.0
    total_frames = 0
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        opponent = PpoCheckpointOpponent(checkpoint_path)
        for episode_idx in range(episodes_per_checkpoint):
            env = TekkenLiteEnv(seed=seed + checkpoint_idx * 10_000 + episode_idx)
            env.reset(seed=seed + checkpoint_idx * 10_000 + episode_idx)
            episode_reward = 0.0
            last_info = {}
            for _ in range(max_decisions):
                p1_action, _ = model.predict(
                    _normalize_obs(normalizer, vector_observation(env.state, env.config, player=1)),
                    deterministic=True,
                    action_masks=legal_action_mask(env.state, player=1),
                )
                result = env.step(index_to_action(int(p1_action)), opponent(env, 2))
                last_info = result.info
                episode_reward += result.reward_p1
                if result.terminated or result.truncated:
                    break
            wins += int(_clean_p1_win(env.state.winner, last_info))
            episodes += 1
            total_reward += episode_reward
            total_frames += env.state.frame

    return PpoEvalResult(
        win_rate=wins / episodes,
        avg_reward=total_reward / episodes,
        avg_frames=total_frames / episodes,
    )
