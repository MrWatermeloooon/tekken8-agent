"""The live screen observation fills exactly what --screen-observation training sees."""
from __future__ import annotations

from pathlib import Path
import re

import numpy as np

from t8_agent.live.own_state import (
    ObservedMoveState, OwnStateTracker, load_move_timings, screen_full_move_observation,
)
from t8_agent.live.hud_reader import HudReading
from t8_agent.vision.temporal import VisualEstimate

REPO_ROOT = Path(__file__).resolve().parents[1]
CONTRACT = REPO_ROOT / "include" / "t8_v2" / "full_move_contract.hpp"
BINDINGS = REPO_ROOT / "data" / "generated" / "full_combat_bindings.csv"


def screen_view_features() -> set[int]:
    """full_core::kScreenViewFeatures, read from the C++ contract."""
    match = re.search(r"kScreenViewFeatures\[\]\s*=\s*\{([^}]*)\}", CONTRACT.read_text(encoding="utf-8"))
    assert match, "kScreenViewFeatures not found in full_move_contract.hpp"
    return {int(value) for value in match.group(1).split(",")}


def estimate(**overrides) -> VisualEstimate:
    values = dict(p1_health_ratio=0.7, p2_health_ratio=0.4, p1_x=-0.5, p2_x=0.5, distance=1.3,
                  p1_velocity=0.2, p2_velocity=-0.3, p1_motion=0.5, p2_motion=0.5, p1_hit_event=True,
                  p2_hit_event=True, p1_attack_likelihood=0.9, p2_attack_likelihood=0.9)
    values.update(overrides)
    return VisualEstimate(**values)


def test_live_features_are_exactly_the_screen_view():
    allowed = screen_view_features()
    # Everything the live side can report, switched on across a few calls (every posture the pose
    # model reports for both fighters, a move in progress, the opponent in Rage): the filled
    # state features must be the training screen view, all of it.
    filled: set[int] = set()
    for own_pose, opponent_pose in (("standing", "crouching"), ("crouching", "standing"),
                                    ("airborne", "grounded"), ("grounded", "airborne")):
        observed = ObservedMoveState(busy=False, posture="standing", stance="IZU", heat=True, rage=True)
        hud = HudReading(p1_health=0.7, p1_recoverable=0.1, p2_health=0.2, p2_recoverable=0.15,
                         p1_heat=0.5 if own_pose in ("standing", "airborne") else 1.0, p2_heat=0.5)
        vector = screen_full_move_observation(estimate(p2_health_ratio=0.2), observed, player=1, own_character=3,
                                              opponent_character=31, stance_index=0, own_pose=own_pose,
                                              opponent_pose=opponent_pose, own_move=(True, 20.0), hud=hud)
        new = {int(index) for index in np.flatnonzero(vector[:64])}
        assert new <= allowed, f"live fills features training never sees: {sorted(new - allowed)}"
        filled |= new
    assert filled == allowed, f"training sees features live never fills: {sorted(allowed - filled)}"


def test_pose_postures_and_own_move_land_where_the_simulator_puts_them():
    observed = ObservedMoveState(busy=False, posture="standing")
    vector = screen_full_move_observation(estimate(), observed, player=1, own_pose="airborne",
                                          opponent_pose="grounded", own_move=(True, 30.0))
    assert vector[10] == 1.0 and vector[8:14].sum() == 1.0, "own airborne"
    assert vector[17] == 1.0 and vector[14:20].sum() == 1.0, "opponent grounded"
    assert vector[58] == 1.0 and abs(vector[59] - 0.5) < 1e-6, "30 frames left of the own move"
    unknown = screen_full_move_observation(estimate(), ObservedMoveState(busy=False, posture="crouching"), player=1)
    assert unknown[9] == 1.0, "no pose: the tracker's posture"


def test_attack_cue_is_the_opponents_and_thresholded():
    observed = ObservedMoveState(busy=False, posture="standing")
    p1 = screen_full_move_observation(estimate(p1_attack_likelihood=0.0, p2_attack_likelihood=0.8), observed,
                                      player=1, attack_cue_threshold=0.5)
    p2 = screen_full_move_observation(estimate(p1_attack_likelihood=0.0, p2_attack_likelihood=0.8), observed,
                                      player=2, attack_cue_threshold=0.5)
    assert p1[50] == 1.0 and p2[50] == 0.0
    low = screen_full_move_observation(estimate(p2_attack_likelihood=0.3), observed, player=1, attack_cue_threshold=0.5)
    assert low[50] == 0.0


def test_noise_levels_are_appended_like_the_simulator():
    observed = OwnStateTracker(load_move_timings(BINDINGS, "jun")).observe(0.0, busy=False)
    vector = screen_full_move_observation(estimate(), observed, player=1, noise_levels=(0.25, 0.15, 4))
    assert vector.shape == (151,)
    # full_core::encode_noise_levels: sigma / 0.5, error / 0.3, latency / 10.
    assert np.allclose(vector[148:], [0.5, 0.5, 0.4])


def test_only_screen_checkpoints_may_drive_the_controller():
    from t8_agent.live import full_move
    source = Path(full_move.__file__).read_text(encoding="utf-8")
    assert 'contract.startswith("full-move-") and not contract.startswith("full-move-screen")' in source



def test_hud_fills_health_and_recoverable_health():
    hud = HudReading(p1_health=0.8, p1_recoverable=0.1, p2_health=0.6, p2_recoverable=0.2, p1_heat=1.0, p2_heat=0.0)
    vector = screen_full_move_observation(estimate(), ObservedMoveState(busy=False, posture="standing"), player=1, hud=hud)
    assert [round(float(v), 3) for v in vector[0:4]] == [0.8, 0.6, 0.1, 0.2]
    assert vector[26] == vector[27] == vector[28] == vector[31] == 0.0, "Heat is not read from the screen yet"
