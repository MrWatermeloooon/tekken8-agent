from __future__ import annotations

from dataclasses import dataclass

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
