from __future__ import annotations

import random

from t8_agent.core.types import GameState
from t8_agent.sim.tekken_lite import SimAction


class LiveScriptedAgent:
    """Small live-test policy for validating screen capture and controller output."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self.tick = 0

    def act(self, state: GameState) -> SimAction:
        self.tick += 1
        if state.round_over or state.p1.health <= 0:
            return SimAction.NEUTRAL
        if self.tick % 7 == 0:
            return self.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW, SimAction.WALK_BACK])
        if self.tick % 3 == 0:
            return SimAction.WALK_FORWARD
        return self.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.F2])
