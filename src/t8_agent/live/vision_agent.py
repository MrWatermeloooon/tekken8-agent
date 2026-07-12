from __future__ import annotations

import random
from pathlib import Path

import numpy as np

from t8_agent.sim.tekken_lite import SimAction
from t8_agent.vision.temporal import VisualEstimate
from t8_agent.sim.action_space import action_count, index_to_action
from t8_agent.train.ppo_opponents import vecnormalize_path


class LiveVisionAgent:
    """Conservative screen-only baseline used to collect safe real-game trajectories."""

    def __init__(self, seed: int = 8080) -> None:
        self.rng = random.Random(seed)
        self.tick = 0

    def act(self, estimate: VisualEstimate) -> SimAction:
        self.tick += 1
        if estimate.p1_health_ratio <= 0.02:
            return SimAction.NEUTRAL
        if estimate.p2_attack_likelihood > 0.35 and estimate.distance < 2.6:
            return SimAction.BLOCK_LOW if self.tick % 5 == 0 else SimAction.BLOCK_HIGH
        if estimate.distance > 2.4:
            return SimAction.DASH_FORWARD
        if estimate.p2_hit_event:
            return self.rng.choice([SimAction.DF1, SimAction.F2, SimAction.HOPKICK])
        if estimate.distance < 1.0 and self.tick % 7 == 0:
            return SimAction.THROW
        return self.rng.choice(
            [
                SimAction.JAB,
                SimAction.DF1,
                SimAction.DB3,
                SimAction.F2,
                SimAction.BLOCK_HIGH,
                SimAction.WALK_BACK,
            ]
        )


class LiveVisualPpoAgent:
    def __init__(self, checkpoint: str | Path, *, deterministic: bool = False) -> None:
        from sb3_contrib import MaskablePPO

        self.checkpoint = Path(checkpoint)
        self.model = MaskablePPO.load(self.checkpoint, device="cpu")
        if tuple(self.model.observation_space.shape) != (13,):
            raise ValueError(f"visual PPO checkpoint must use 13 observations: {self.model.observation_space.shape}")
        self.normalizer = None
        normalizer_path = vecnormalize_path(self.checkpoint)
        if normalizer_path.exists():
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv

            dummy = DummyVecEnv([lambda: TekkenLiteSingleAgentEnv(observation_mode="visual")])
            self.normalizer = VecNormalize.load(str(normalizer_path), dummy)
            self.normalizer.training = False
            self.normalizer.norm_reward = False
        self.deterministic = deterministic

    def act(self, estimate: VisualEstimate) -> SimAction:
        observation = estimate.to_vector()
        if self.normalizer is not None:
            observation = self.normalizer.normalize_obs(observation)
        action, _state = self.model.predict(
            observation,
            deterministic=self.deterministic,
            action_masks=np.ones(action_count(), dtype=bool),
        )
        return index_to_action(int(action))
