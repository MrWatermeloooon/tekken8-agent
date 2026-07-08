from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from t8_agent.sim.action_space import index_to_action, legal_action_mask
from t8_agent.sim.observations import vector_observation
from t8_agent.sim.tekken_lite import TekkenLiteEnv
from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


@dataclass(frozen=True)
class PpoEvalResult:
    win_rate: float
    avg_reward: float
    avg_frames: float


def evaluate_maskable_model(model, episodes: int, seed: int, max_decisions: int, opponent_names: list[str]) -> PpoEvalResult:
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
            action, _state = model.predict(obs, deterministic=True, action_masks=env.action_masks())
            obs, reward, terminated, truncated, last_info = env.step(int(action))
            episode_reward += reward
        wins += int(last_info.get("winner") == 1)
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
) -> PpoEvalResult | None:
    if not checkpoint_paths:
        return None

    from sb3_contrib import MaskablePPO

    wins = 0
    episodes = 0
    total_reward = 0.0
    total_frames = 0
    for checkpoint_idx, checkpoint_path in enumerate(checkpoint_paths):
        opponent = MaskablePPO.load(checkpoint_path)
        for episode_idx in range(episodes_per_checkpoint):
            env = TekkenLiteEnv(seed=seed + checkpoint_idx * 10_000 + episode_idx)
            env.reset(seed=seed + checkpoint_idx * 10_000 + episode_idx)
            episode_reward = 0.0
            for _ in range(max_decisions):
                p1_action, _ = model.predict(
                    vector_observation(env.state, env.config, player=1),
                    deterministic=True,
                    action_masks=legal_action_mask(env.state, player=1),
                )
                p2_action, _ = opponent.predict(
                    vector_observation(env.state, env.config, player=2),
                    deterministic=True,
                    action_masks=legal_action_mask(env.state, player=2),
                )
                result = env.step(index_to_action(int(p1_action)), index_to_action(int(p2_action)))
                episode_reward += result.reward_p1
                if result.terminated or result.truncated:
                    break
            wins += int(env.state.winner == 1)
            episodes += 1
            total_reward += episode_reward
            total_frames += env.state.frame

    return PpoEvalResult(
        win_rate=wins / episodes,
        avg_reward=total_reward / episodes,
        avg_frames=total_frames / episodes,
    )
