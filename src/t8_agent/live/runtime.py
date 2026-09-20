from __future__ import annotations

import time

import numpy as np

from t8_agent.sim.action_space import ACTION_SPACE
from t8_agent.sim.actions import SimAction
from t8_agent.vision.temporal import VisualEstimate


class LiveActionState:
    """Conservative repeat-attack mask, not an oracle for in-game legality."""

    RECOVERY_LIMITS = {
        SimAction.JAB: 0.28, SimAction.DF1: 0.40, SimAction.F2: 0.52,
        SimAction.DB3: 0.55, SimAction.HOPKICK: 0.65, SimAction.THROW: 0.50,
    }

    def __init__(self, player: int = 1) -> None:
        self.player = player
        self.reset_episode()

    def reset_episode(self) -> None:
        self.started_at = 0.0
        self.busy_until = 0.0
        self.saw_motion = False
        self.quiet_frames = 0

    def record_executed(self, action: SimAction, *, now: float | None = None) -> None:
        duration = self.RECOVERY_LIMITS.get(action)
        if duration is not None:
            self.started_at = time.perf_counter() if now is None else now
            self.busy_until = self.started_at + duration
            self.saw_motion = False
            self.quiet_frames = 0

    def action_mask(self, estimate: VisualEstimate, *, input_busy: bool = False,
                    now: float | None = None) -> np.ndarray:
        now = time.perf_counter() if now is None else now
        mask = np.ones(len(ACTION_SPACE), dtype=bool)
        own_motion = estimate.p1_motion if self.player == 1 else estimate.p2_motion
        own_hit = estimate.p1_hit_event if self.player == 1 else estimate.p2_hit_event
        if own_hit:
            # The attempted move was interrupted; do not retain its old recovery lock.
            self.reset_episode()
        elif now < self.busy_until:
            self.saw_motion |= own_motion >= 0.02
            self.quiet_frames = self.quiet_frames + 1 if own_motion < 0.008 else 0
            if self.saw_motion and self.quiet_frames >= 2 and now - self.started_at >= 0.10:
                self.busy_until = now
        if now < self.busy_until or own_hit:
            for action in self.RECOVERY_LIMITS:
                mask[ACTION_SPACE.index(action)] = False
        if input_busy:
            mask[:] = False
            mask[ACTION_SPACE.index(SimAction.NEUTRAL)] = True
        return mask


class RoundTracker:
    """Debounce HUD resets; keep KO and health-restoration frames out of history."""

    def __init__(self, confirm_frames: int = 3) -> None:
        self.confirm_frames = confirm_frames
        self.reset_episode()

    def reset_episode(self) -> None:
        self.minimum: np.ndarray | None = None
        self.pending = 0
        self.ko_frames = 0
        self.ended = False
        self.ready = False

    def update(self, p1_health: float, p2_health: float) -> str:
        health = np.asarray([p1_health, p2_health], dtype=float)
        if not np.isfinite(health).all() or np.any((health < 0) | (health > 1)):
            return "waiting"
        if self.minimum is None:
            self.minimum = health.copy()
        if not self.ready:
            if health.min() <= 0.02:
                self.pending = 0
                return "waiting"
            self.pending += 1
            if self.pending < self.confirm_frames:
                return "waiting"
            self.ready = True
            self.pending = 0
            self.minimum = health.copy()
            return "reset"
        restored = bool(np.all(health >= 0.95) and np.any(health - self.minimum >= 0.10))
        if restored:
            self.pending += 1
            if self.pending >= self.confirm_frames:
                self.minimum = health.copy()
                self.pending = self.ko_frames = 0
                self.ended = False
                return "reset"
            return "waiting"
        self.pending = 0
        self.minimum = np.minimum(self.minimum, health)
        self.ko_frames = self.ko_frames + 1 if health.min() <= 0.02 else 0
        self.ended |= self.ko_frames >= 2
        return "waiting" if self.ended or self.ko_frames else "active"
