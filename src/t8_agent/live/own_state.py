"""Estimates of the controlled fighter's state for full-move legality.

The screen alone does not show a fighter's stance, posture, or recovery, but
the runtime knows exactly which commands it sent. ``OwnStateTracker`` follows
the engine binding data (data/generated/full_combat_bindings.csv) for each
dispatched move: it is busy for the move's startup plus recovery, then ends in
the move's documented result (stance on hit, on block, or by default; crouched;
back-turned). Being hit interrupts the move and clears any stance; the fighter
is unknown-airborne/knocked-down until the hit's stun could have ended.

``HudGauge`` reads a calibrated HUD bar (the Heat gauge) as a fill ratio, and
``RageEstimate`` applies the health threshold. Every estimate that cannot be
established stays ``None`` (unknown), which keeps the moves that depend on it
masked (t8_agent.live.full_move.live_legal_mask).
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from t8_agent.live.full_move import ObservedMoveState
from t8_agent.moves.candidates import UNIVERSAL_PREFIX

FRAME_SECONDS = 1.0 / 60.0
# Time after being hit during which the fighter may still be airborne or down;
# provisional until measured (the engine's knockdown and wake-up timings are too).
UNSETTLED_AFTER_HIT_SECONDS = 1.5
ACTIVE_FRAMES_ALLOWANCE = 2


@dataclass(frozen=True)
class MoveTiming:
    startup: int
    recovery: int
    result_stance: str
    result_stance_on_hit: str
    result_stance_on_block: str
    result_crouching: bool
    result_back_turned: bool

    @property
    def total_frames(self) -> int:
        return self.startup + ACTIVE_FRAMES_ALLOWANCE + self.recovery


def load_move_timings(bindings_path: str | Path, character: str) -> dict[str, MoveTiming]:
    timings: dict[str, MoveTiming] = {}
    with Path(bindings_path).open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["character"] != character or not row["startup"] or not row["recovery"]:
                continue
            timings[row["stable_id"]] = MoveTiming(
                int(row["startup"]), int(row["recovery"]), row["result_stance"].upper(),
                row["result_stance_on_hit"].upper(), row["result_stance_on_block"].upper(),
                row["result_crouching"] == "1", row["result_back_turned"] == "1")
    return timings


@dataclass
class OwnStateTracker:
    timings: Mapping[str, MoveTiming]
    stance: str | None = ""
    posture: str | None = "standing"
    back_turned: bool | None = False
    busy_until: float = 0.0
    hit_at: float | None = None
    _pending: MoveTiming | None = field(default=None, repr=False)
    _pending_outcome: str | None = field(default=None, repr=False)

    def reset_round(self) -> None:
        """A new round starts standing, out of any stance."""
        self.stance, self.posture, self.back_turned = "", "standing", False
        self.busy_until, self.hit_at = 0.0, None
        self._pending = self._pending_outcome = None

    def on_dispatch(self, candidate_id: str, now: float) -> None:
        if candidate_id.startswith(UNIVERSAL_PREFIX):
            name = candidate_id[len(UNIVERSAL_PREFIX):]
            if name in {"crouch", "block_low"}:
                self.posture = "crouching"
            elif name in {"neutral", "stand", "walk_forward", "walk_back", "dash_forward", "dash_back",
                          "sidestep_left", "sidestep_right", "block_high", "jump"}:
                self.posture = "standing"
                self.stance = ""          # any movement leaves a stance
                self.back_turned = False  # and turns a back-turned fighter around
            return
        timing = self.timings.get(candidate_id)
        if timing is None:
            # No documented timing: the result cannot be predicted.
            self.stance = self.posture = self.back_turned = None
            self._pending = None
            self.busy_until = now + 1.0
            return
        self._pending = timing
        self._pending_outcome = None
        self.busy_until = now + timing.total_frames * FRAME_SECONDS

    def move_progress(self, now: float) -> tuple[bool, float]:
        """Whether the fighter is in a move it sent, and how many frames of it are left (the
        simulator's own-move features 58 and 59; universal actions are not moves)."""
        if self._pending is None or now >= self.busy_until:
            return False, 0.0
        return True, (self.busy_until - now) / FRAME_SECONDS

    def on_outcome(self, outcome: str) -> None:
        """Execution feedback for the pending move: "hit", "blocked", or another outcome."""
        self._pending_outcome = outcome

    def on_own_hit(self, now: float) -> None:
        self.hit_at = now
        self._pending = None
        self.stance, self.back_turned = "", False
        self.busy_until = max(self.busy_until, now)

    def observe(self, now: float, *, busy: bool, heat: bool | None = None, rage: bool | None = None,
                situations: frozenset[str] = frozenset()) -> ObservedMoveState:
        if self._pending is not None and now >= self.busy_until:
            timing = self._pending
            if self._pending_outcome == "hit" and timing.result_stance_on_hit:
                self.stance = timing.result_stance_on_hit
            elif self._pending_outcome == "blocked" and timing.result_stance_on_block:
                self.stance = timing.result_stance_on_block
            else:
                self.stance = timing.result_stance
            self.posture = "crouching" if timing.result_crouching else "standing"
            self.back_turned = timing.result_back_turned
            self._pending = None
        settled = self.hit_at is None or now - self.hit_at >= UNSETTLED_AFTER_HIT_SECONDS
        return ObservedMoveState(
            busy=busy or now < self.busy_until,
            posture=self.posture if settled else None,
            stance=self.stance if settled else None,
            heat=heat,
            rage=rage,
            airborne=False if settled else None,
            knocked_down=False if settled else None,
            situations=situations,
        )


@dataclass(frozen=True)
class HudGauge:
    """A horizontal HUD bar: the fraction of columns showing the bar's color."""

    left: int
    top: int
    right: int
    bottom: int
    color: tuple[int, int, int]
    tolerance: int = 40
    column_fill: float = 0.4

    @classmethod
    def from_config(cls, config: Mapping[str, Any], key: str) -> "HudGauge | None":
        value = config.get(key)
        if not value:
            return None
        return cls(int(value["left"]), int(value["top"]), int(value["right"]), int(value["bottom"]),
                   tuple(int(channel) for channel in value["color"]), int(value.get("tolerance", 40)))

    def fill(self, frame: np.ndarray) -> float | None:
        crop = np.asarray(frame)[self.top:self.bottom, self.left:self.right, :3].astype(np.int16)
        if crop.size == 0:
            return None
        distance = np.abs(crop - np.asarray(self.color, dtype=np.int16)).max(axis=2)
        columns = (distance <= self.tolerance).mean(axis=0) >= self.column_fill
        return float(columns.mean())


class HeatEstimate:
    """Heat is active while the gauge drains after having been full.

    Heat Burst and Heat Engagers fill nothing: they start the drain, so the
    signal is a full gauge that then shrinks. Unknown until the gauge has been
    seen full in this round.
    """

    def __init__(self, gauge: HudGauge | None, *, full: float = 0.95, drop: float = 0.03) -> None:
        self.gauge, self.full, self.drop = gauge, full, drop
        self.reset_round()

    def reset_round(self) -> None:
        self.seen_full = False
        self.previous: float | None = None
        self.active = False
        self.spent = False

    def update(self, frame: np.ndarray) -> bool | None:
        if self.gauge is None:
            return None
        fill = self.gauge.fill(frame)
        if fill is None:
            return None
        if fill >= self.full and not self.active and not self.spent:
            self.seen_full = True
        if self.seen_full and not self.spent and fill < self.full - self.drop:
            self.active = True  # a full gauge visibly draining
        if self.active and fill <= 0.01:
            self.active = False
            self.spent = True  # Heat is once per round
        self.previous = fill
        if self.active:
            return True
        return False if self.seen_full or self.spent else None


@dataclass(frozen=True)
class RageEstimate:
    """Rage starts when health falls to the threshold (provisional: 25%), once per round.

    After the round's Rage Art the state is unknown, since the screen cannot
    tell a spent Rage from an unspent one without a Rage Art detector.
    """

    threshold: float = 0.25

    def update(self, own_health_ratio: float, *, rage_art_used: bool = False) -> bool | None:
        if rage_art_used:
            return None
        return own_health_ratio <= self.threshold and own_health_ratio > 0.0


ROSTER_SIZE = 42
FULL_MOVE_STATE_FEATURES = 64
FULL_MOVE_OBSERVATION_SIZE = FULL_MOVE_STATE_FEATURES + 2 * ROSTER_SIZE


def screen_full_move_observation(estimate, observed: ObservedMoveState, *, player: int,
                                 own_character: int | None = None, opponent_character: int | None = None,
                                 stance_count: int = 3, stance_index: int | None = None,
                                 attack_cue_threshold: float = 0.5,
                                 noise_levels: tuple[float, float, float] | None = None,
                                 own_pose: str | None = None, opponent_pose: str | None = None,
                                 own_move: tuple[bool, float] = (False, 0.0),
                                 rage_threshold: float = 0.25, hud=None) -> np.ndarray:
    """A full-move observation (the 148-feature layout: 64 state features, then one-hot
    roster ids of the fighter and the opponent) built from screen estimates and the own-state
    tracker.

    It fills exactly the features a --screen-observation checkpoint trains on
    (full_core::kScreenViewFeatures in include/t8_v2/full_move_contract.hpp, which
    tests/test_screen_full_move.py checks against this function): HUD health, distance, own
    standing/crouching, can act, Heat, Rage, stance, and the opponent's attack cue; since v4
    both fighters' postures from the pose model (own_pose / opponent_pose: standing, crouching,
    airborne, grounded; the tracker's posture when the pose is unknown), the move the fighter
    sent and the frames left of it (own_move, from OwnStateTracker.move_progress), and the
    opponent's Rage by its health; with ``hud`` (a t8_agent.live.hud_reader.HudReading), both
    recoverable health bars and the health bars from the HUD reader (Heat is not read: its
    gauges misread during hit flashes and Heat activation). Everything else is
    0, as in that training. A privileged
    checkpoint has never seen this input distribution, so it is for dry runs only.

    noise_levels (position sigma in metres, detection error rate, input latency in frames, from
    calibration) are appended for a --live-noise checkpoint (151 inputs).
    """

    vector = np.zeros(FULL_MOVE_OBSERVATION_SIZE, dtype=np.float32)
    own_health = estimate.p1_health_ratio if player == 1 else estimate.p2_health_ratio
    other_health = estimate.p2_health_ratio if player == 1 else estimate.p1_health_ratio
    vector[0], vector[1] = own_health, other_health
    vector[4] = min(1.0, max(0.0, estimate.distance / 4.0))
    postures = {"standing": 0, "crouching": 1, "airborne": 2, "grounded": 3}
    own_posture = own_pose if own_pose in postures else observed.posture
    if own_posture in postures:
        vector[8 + postures[own_posture]] = 1.0
    if opponent_pose in postures:
        vector[14 + postures[opponent_pose]] = 1.0
    if other_health > 0.0 and other_health <= rage_threshold:
        vector[32] = 1.0  # the opponent's Rage, by the same health rule as our own (RageEstimate)
    if own_move[0]:
        vector[58] = 1.0
        vector[59] = min(1.0, max(0.0, own_move[1] / 60.0))
    vector[24] = 0.0 if observed.busy else 1.0
    if hud is not None:
        own_side, other_side = (1, 2) if player == 1 else (2, 1)
        vector[0] = getattr(hud, f"p{own_side}_health")
        vector[1] = getattr(hud, f"p{other_side}_health")
        vector[2] = getattr(hud, f"p{own_side}_recoverable")
        vector[3] = getattr(hud, f"p{other_side}_recoverable")
        vector[32] = 1.0 if 0.0 < vector[1] <= rage_threshold else 0.0  # the opponent's Rage by the HUD health
    vector[29] = 1.0 if observed.rage else 0.0
    if observed.stance:
        vector[33] = 1.0
        if stance_index is not None:
            vector[34] = (stance_index + 1) / (stance_count + 1)
    # The opponent's attack cue (the simulator's "opponent attacking", seen after the perception delay).
    opponent_attack = estimate.p2_attack_likelihood if player == 1 else estimate.p1_attack_likelihood
    vector[50] = 1.0 if opponent_attack >= attack_cue_threshold else 0.0
    vector[63] = 1.0
    if own_character is not None and 0 <= own_character < ROSTER_SIZE:
        vector[FULL_MOVE_STATE_FEATURES + own_character] = 1.0
    if opponent_character is not None and 0 <= opponent_character < ROSTER_SIZE:
        vector[FULL_MOVE_STATE_FEATURES + ROSTER_SIZE + opponent_character] = 1.0
    if noise_levels is not None:  # full_core::encode_noise_levels
        sigma, error, latency = noise_levels
        levels = np.clip([sigma / 0.5, error / 0.3, latency / 10.0], 0.0, 1.0).astype(np.float32)
        vector = np.concatenate([vector, levels])
    return vector
