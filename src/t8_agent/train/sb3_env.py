from __future__ import annotations

from collections.abc import Callable, Sequence

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from t8_agent.sim.action_space import action_count, index_to_action, legal_action_mask
from t8_agent.sim.observations import observation_size, vector_observation
from t8_agent.sim.opponents import DEFAULT_SCRIPTED_OPPONENTS, SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import SimAction, SimConfig, TekkenLiteEnv

OpponentSampler = Callable[[], tuple[str, Callable[[TekkenLiteEnv, int], SimAction]]]


class TekkenLiteSingleAgentEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        opponent_names: Sequence[str] | None = None,
        seed: int | None = None,
        max_decisions: int = 1200,
        config: SimConfig | None = None,
        opponent_sampler: OpponentSampler | None = None,
    ) -> None:
        super().__init__()
        self.sim = TekkenLiteEnv(config=config, seed=seed)
        self.opponent_names = list(opponent_names or DEFAULT_SCRIPTED_OPPONENTS)
        unknown = [name for name in self.opponent_names if name not in SCRIPTED_POLICIES]
        if unknown:
            known = ", ".join(sorted(SCRIPTED_POLICIES))
            raise ValueError(f"unknown opponent(s) {unknown}; known: {known}")
        self.max_decisions = max_decisions
        self.opponent_sampler = opponent_sampler
        self.decision_count = 0
        self.opponent_name = self.opponent_names[0]
        self.opponent_policy = SCRIPTED_POLICIES[self.opponent_name]
        self.action_space = spaces.Discrete(action_count())
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(observation_size(),),
            dtype=np.float32,
        )

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        self.sim.reset(seed=seed)
        self.decision_count = 0
        options = options or {}
        if self.opponent_sampler is not None:
            self.opponent_name, self.opponent_policy = self.opponent_sampler()
        else:
            self.opponent_name = options.get("opponent_name") or self.sim.rng.choice(self.opponent_names)
            self.opponent_policy = SCRIPTED_POLICIES[self.opponent_name]
        return self._obs(), {"opponent_name": self.opponent_name}

    def step(self, action: int):
        p1_action = index_to_action(int(action))
        if not self.action_masks()[int(action)]:
            p1_action = SimAction.NEUTRAL
        p2_action = self.opponent_policy(self.sim, 2)
        result = self.sim.step(p1_action, p2_action)
        self.decision_count += 1
        truncated = result.truncated or self.decision_count >= self.max_decisions
        info = dict(result.info)
        info.update(
            {
                "winner": result.state.winner or 0,
                "opponent_name": self.opponent_name,
                "p1_health": result.state.p1.health,
                "p2_health": result.state.p2.health,
            }
        )
        return self._obs(), float(result.reward_p1), bool(result.terminated), bool(truncated), info

    def action_masks(self) -> np.ndarray:
        return legal_action_mask(self.sim.state, player=1)

    def _obs(self) -> np.ndarray:
        return vector_observation(self.sim.state, self.sim.config, player=1)
