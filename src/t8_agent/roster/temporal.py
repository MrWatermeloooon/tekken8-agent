from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import math

import numpy as np


CHARACTER_EMBEDDING_SIZE = 8
ARCHETYPE_COUNT = 10
HISTORY_LENGTH = 8
TEMPORAL_FEATURES = 8


@dataclass(frozen=True)
class TemporalFrame:
    """Opponent context available at one policy decision."""

    move_id: int = 0
    animation_phase: float = 0.0
    stance_id: int = 0
    hit_level: int = 0
    delay_frames: int = 0
    outcome: float = 0.0
    distance: float = 0.0
    side_movement: float = 0.0

    def vector(self) -> np.ndarray:
        return np.asarray(
            [
                # The native CUDA encoder uses the 24-action simulation ID
                # (0..23) as its compact move identity.
                np.clip(self.move_id / 23.0, 0.0, 1.0),
                np.clip(self.animation_phase, 0.0, 1.0),
                np.clip(self.stance_id / 31.0, 0.0, 1.0),
                np.clip(self.hit_level / 4.0, 0.0, 1.0),
                np.clip(self.delay_frames / 60.0, 0.0, 1.0),
                np.clip(self.outcome, -1.0, 1.0),
                np.clip(self.distance / 7.2, 0.0, 1.0),
                np.clip(self.side_movement, -1.0, 1.0),
            ],
            dtype=np.float32,
        )


class MatchupObservationEncoder:
    """Eight-decision frame stack with character and style conditioning."""

    def __init__(self, *, base_size: int = 13, history_length: int = HISTORY_LENGTH) -> None:
        if base_size <= 0 or history_length < 4:
            raise ValueError("base_size must be positive and history_length must be at least four")
        self.base_size = base_size
        self.history_length = history_length
        self._history: deque[np.ndarray] = deque(maxlen=history_length)

    @property
    def observation_size(self) -> int:
        return self.base_size + CHARACTER_EMBEDDING_SIZE + ARCHETYPE_COUNT + self.history_length * TEMPORAL_FEATURES

    def reset(self) -> None:
        self._history.clear()

    def encode(
        self,
        base_observation: np.ndarray,
        *,
        opponent_character_id: int,
        opponent_archetype_id: int,
        frame: TemporalFrame,
    ) -> np.ndarray:
        base = np.asarray(base_observation, dtype=np.float32)
        if base.shape != (self.base_size,):
            raise ValueError(f"base observation must have shape ({self.base_size},), got {base.shape}")
        if opponent_character_id < 0:
            raise ValueError("opponent_character_id must be non-negative")
        if not 0 <= opponent_archetype_id < ARCHETYPE_COUNT:
            raise ValueError(f"opponent_archetype_id must be in [0, {ARCHETYPE_COUNT})")
        self._history.append(frame.vector())
        missing = self.history_length - len(self._history)
        history = [np.zeros(TEMPORAL_FEATURES, dtype=np.float32) for _ in range(missing)]
        history.extend(self._history)
        archetype = np.zeros(ARCHETYPE_COUNT, dtype=np.float32)
        archetype[opponent_archetype_id] = 1.0
        return np.concatenate(
            [base, _character_embedding(opponent_character_id), archetype, *history]
        ).astype(np.float32, copy=False)


def _character_embedding(character_id: int) -> np.ndarray:
    """Unique six-bit identity plus smooth cyclic coordinates for 64 slots."""
    bits = [1.0 if character_id & (1 << bit) else -1.0 for bit in range(6)]
    angle = 2.0 * math.pi * (character_id % 64) / 64.0
    return np.asarray([*bits, math.sin(angle), math.cos(angle)], dtype=np.float32)
