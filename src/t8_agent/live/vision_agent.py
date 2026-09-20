from __future__ import annotations

import random
import numpy as np
from t8_agent.sim.actions import SimAction
from t8_agent.vision.temporal import VisualEstimate
from t8_agent.live.v2_policy import LiveV2GpuAgent


class LiveVisionAgent:
    """Conservative screen-only baseline used to collect safe real-game trajectories."""

    def __init__(self, seed: int = 8080) -> None:
        self.rng = random.Random(seed)
        self.tick = 0

    def reset_episode(self) -> None:
        self.tick = 0

    def act(self, estimate: VisualEstimate, *, action_mask: np.ndarray | None = None) -> SimAction:
        from t8_agent.sim.action_space import ACTION_SPACE

        action = self._choose_action(estimate)
        if action_mask is not None and not action_mask[ACTION_SPACE.index(action)]:
            for fallback in (SimAction.BLOCK_HIGH, SimAction.NEUTRAL):
                if action_mask[ACTION_SPACE.index(fallback)]:
                    return fallback
        return action

    def _choose_action(self, estimate: VisualEstimate) -> SimAction:
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


LiveVisualPpoAgent = LiveV2GpuAgent
