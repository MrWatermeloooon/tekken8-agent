from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from t8_agent.core.types import GameState
from t8_agent.io.screen_backend import ScreenRegion


@dataclass(frozen=True)
class VisualEstimate:
    p1_health_ratio: float
    p2_health_ratio: float
    p1_x: float
    p2_x: float
    distance: float
    p1_velocity: float
    p2_velocity: float
    p1_motion: float
    p2_motion: float
    p1_hit_event: bool
    p2_hit_event: bool
    p1_attack_likelihood: float
    p2_attack_likelihood: float

    def to_dict(self) -> dict[str, float | bool]:
        return asdict(self)

    def to_vector(self) -> np.ndarray:
        return np.asarray(
            [
                self.p1_health_ratio,
                self.p2_health_ratio,
                self.p1_x,
                self.p2_x,
                self.distance,
                self.p1_velocity,
                self.p2_velocity,
                self.p1_motion,
                self.p2_motion,
                float(self.p1_hit_event),
                float(self.p2_hit_event),
                self.p1_attack_likelihood,
                self.p2_attack_likelihood,
            ],
            dtype=np.float32,
        )


class TemporalScreenEstimator:
    def __init__(
        self,
        *,
        max_health: float = 180.0,
        p1_region: ScreenRegion | None = None,
        p2_region: ScreenRegion | None = None,
        motion_threshold: float = 0.015,
    ) -> None:
        self.max_health = max_health
        self.p1_region = p1_region
        self.p2_region = p2_region
        self.motion_threshold = motion_threshold
        self.previous_gray: np.ndarray | None = None
        self.previous_state: GameState | None = None

    def update(self, state: GameState, frame: np.ndarray) -> VisualEstimate:
        gray = _gray_small(frame)
        height, width = frame.shape[:2]
        p1_region = self.p1_region or ScreenRegion(0, int(height * 0.2), width // 2, height)
        p2_region = self.p2_region or ScreenRegion(width // 2, int(height * 0.2), width, height)
        p1_motion = self._motion_for_region(gray, p1_region, width, height)
        p2_motion = self._motion_for_region(gray, p2_region, width, height)
        # Camera movement and hit shake affect both screen halves. Remove the
        # shared component so the agent does not treat its own move as a new
        # opponent attack and enter a repeated-attack feedback loop.
        shared_motion = min(p1_motion, p2_motion)
        p1_attack_motion = max(0.0, p1_motion - shared_motion)
        p2_attack_motion = max(0.0, p2_motion - shared_motion)

        previous = self.previous_state
        p1_velocity = 0.0 if previous is None else state.p1.position_x - previous.p1.position_x
        p2_velocity = 0.0 if previous is None else state.p2.position_x - previous.p2.position_x
        p1_health_drop = 0.0 if previous is None else previous.p1.health - state.p1.health
        p2_health_drop = 0.0 if previous is None else previous.p2.health - state.p2.health
        distance = state.distance

        self.previous_gray = gray
        self.previous_state = state
        return VisualEstimate(
            p1_health_ratio=float(np.clip(state.p1.health / self.max_health, 0.0, 1.0)),
            p2_health_ratio=float(np.clip(state.p2.health / self.max_health, 0.0, 1.0)),
            p1_x=float(state.p1.position_x),
            p2_x=float(state.p2.position_x),
            distance=float(distance),
            p1_velocity=float(p1_velocity),
            p2_velocity=float(p2_velocity),
            p1_motion=p1_motion,
            p2_motion=p2_motion,
            p1_hit_event=p1_health_drop > 1.0,
            p2_hit_event=p2_health_drop > 1.0,
            p1_attack_likelihood=_attack_likelihood(p1_attack_motion, distance, self.motion_threshold),
            p2_attack_likelihood=_attack_likelihood(p2_attack_motion, distance, self.motion_threshold),
        )

    def _motion_for_region(self, gray: np.ndarray, region: ScreenRegion, width: int, height: int) -> float:
        if self.previous_gray is None:
            return 0.0
        scale_x = gray.shape[1] / width
        scale_y = gray.shape[0] / height
        left = int(region.left * scale_x)
        right = int(region.right * scale_x)
        top = int(region.top * scale_y)
        bottom = int(region.bottom * scale_y)
        current = gray[top:bottom, left:right]
        previous = self.previous_gray[top:bottom, left:right]
        if current.size == 0 or current.shape != previous.shape:
            return 0.0
        return float(np.abs(current.astype(np.int16) - previous.astype(np.int16)).mean() / 255.0)


def _gray_small(frame: np.ndarray) -> np.ndarray:
    import cv2

    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return cv2.resize(gray, (320, 180), interpolation=cv2.INTER_AREA)


def _attack_likelihood(motion: float, distance: float, threshold: float) -> float:
    proximity = float(np.clip(1.0 - distance / 4.0, 0.0, 1.0))
    motion_score = float(np.clip((motion - threshold) / max(threshold * 3.0, 1e-6), 0.0, 1.0))
    return motion_score * (0.35 + 0.65 * proximity)
