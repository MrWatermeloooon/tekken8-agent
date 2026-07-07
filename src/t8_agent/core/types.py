from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping


class DiscreteAction(str, Enum):
    NEUTRAL = "neutral"
    WALK_FORWARD = "walk_forward"
    WALK_BACK = "walk_back"
    CROUCH = "crouch"
    JUMP = "jump"
    LEFT_PUNCH = "left_punch"
    RIGHT_PUNCH = "right_punch"
    LEFT_KICK = "left_kick"
    RIGHT_KICK = "right_kick"


@dataclass(frozen=True)
class PlayerState:
    health: float
    position_x: float
    position_y: float = 0.0
    facing: int = 1
    move_id: int | str | None = None
    is_attacking: bool = False
    is_blocking: bool = False
    is_in_hitstun: bool = False


@dataclass(frozen=True)
class GameState:
    p1: PlayerState
    p2: PlayerState
    round_timer: float
    round_over: bool = False
    winner: int | None = None
    raw: Mapping[str, float | int | bool | str] | None = None

    @property
    def distance(self) -> float:
        return abs(self.p1.position_x - self.p2.position_x)


@dataclass(frozen=True)
class StepResult:
    observation: GameState
    reward: float
    terminated: bool
    truncated: bool
    info: Mapping[str, float | int | bool | str]
