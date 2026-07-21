from __future__ import annotations

from abc import ABC, abstractmethod

from t8_agent.core.types import GameState, PlayerState


class StateBackend(ABC):
    @abstractmethod
    def read(self) -> GameState:
        """Read the latest Tekken match state."""

    def close(self) -> None:
        return None


class MockStateBackend(StateBackend):
    def __init__(self) -> None:
        self.state = GameState(
            p1=PlayerState(health=180.0, position_x=-1.0),
            p2=PlayerState(health=180.0, position_x=1.0, facing=-1),
            round_timer=60.0,
        )

    def read(self) -> GameState:
        return self.state

    def set_state(self, state: GameState) -> None:
        self.state = state
