from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from t8_agent.sim.observations import observation_size, vector_observation
from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv

ACTION_SPACE = [
    SimAction.NEUTRAL,
    SimAction.WALK_FORWARD,
    SimAction.WALK_BACK,
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]


@dataclass
class LinearPolicy:
    weights: np.ndarray

    @classmethod
    def zeros(cls) -> "LinearPolicy":
        return cls(weights=np.zeros((len(ACTION_SPACE), observation_size()), dtype=np.float32))

    def act(self, env: TekkenLiteEnv, player: int, temperature: float = 0.0) -> SimAction:
        obs = vector_observation(env.state, env.config, player)
        logits = self.weights @ obs
        if temperature > 0.0:
            noise = np.array(
                [env.rng.normalvariate(0.0, temperature) for _ in ACTION_SPACE],
                dtype=np.float32,
            )
            logits = logits + noise
        action_idx = int(np.argmax(logits))
        return ACTION_SPACE[action_idx]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, weights=self.weights, actions=np.array([action.value for action in ACTION_SPACE]))

    @classmethod
    def load(cls, path: str | Path) -> "LinearPolicy":
        data = np.load(path, allow_pickle=False)
        weights = data["weights"].astype(np.float32)
        expected_shape = (len(ACTION_SPACE), observation_size())
        if weights.shape != expected_shape:
            raise ValueError(f"checkpoint weights shape {weights.shape} does not match expected {expected_shape}")
        return cls(weights=weights)
