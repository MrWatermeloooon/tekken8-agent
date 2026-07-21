from __future__ import annotations

from abc import ABC, abstractmethod

from t8_agent.core.types import DiscreteAction


class InputBackend(ABC):
    @abstractmethod
    def send(self, action: DiscreteAction) -> None:
        """Send one agent action to Player 1."""

    @abstractmethod
    def release_all(self) -> None:
        """Return the controller to neutral."""

    def close(self) -> None:
        self.release_all()


class MockInputBackend(InputBackend):
    def __init__(self) -> None:
        self.actions: list[DiscreteAction] = []

    def send(self, action: DiscreteAction) -> None:
        self.actions.append(action)

    def release_all(self) -> None:
        self.actions.append(DiscreteAction.NEUTRAL)
