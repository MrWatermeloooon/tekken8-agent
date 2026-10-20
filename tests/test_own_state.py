from __future__ import annotations

from pathlib import Path

import numpy as np

from t8_agent.live.full_move import live_legal_mask
from t8_agent.live.own_state import (
    HeatEstimate, HudGauge, OwnStateTracker, RageEstimate, load_move_timings, screen_full_move_observation,
)
from t8_agent.moves.candidates import build_candidate_table
from t8_agent.moves.catalog import compile_catalog
from t8_agent.vision.temporal import VisualEstimate

REPO_ROOT = Path(__file__).resolve().parents[1]
BINDINGS = REPO_ROOT / "data" / "generated" / "full_combat_bindings.csv"


def estimate(**overrides) -> VisualEstimate:
    values = dict(p1_health_ratio=1.0, p2_health_ratio=1.0, p1_x=-0.5, p2_x=0.5, distance=1.0,
                  p1_velocity=0.0, p2_velocity=0.0, p1_motion=0.0, p2_motion=0.0, p1_hit_event=False,
                  p2_hit_event=False, p1_attack_likelihood=0.0, p2_attack_likelihood=0.0)
    values.update(overrides)
    return VisualEstimate(**values)


def test_tracker_follows_sent_moves_into_their_stances():
    timings = load_move_timings(BINDINGS, "jun")
    one_one = timings["jun:7"]  # 1,1: ends in Izumo
    assert one_one.result_stance == "IZU" and one_one.startup == 10
    tracker = OwnStateTracker(timings)
    observed = tracker.observe(0.0, busy=False)
    assert observed.stance == "" and observed.posture == "standing" and observed.airborne is False
    tracker.on_dispatch("jun:7", 0.0)
    assert tracker.observe(0.05, busy=False).busy, "busy for the move's duration"
    settled = tracker.observe(one_one.total_frames / 60.0 + 0.01, busy=False)
    assert not settled.busy and settled.stance == "IZU"

    catalog = compile_catalog(REPO_ROOT / "data")
    table = build_candidate_table(catalog, "jun")
    mask = live_legal_mask(table, catalog, settled, require_validated=False)
    legal = {table.stable_ids[index] for index in np.flatnonzero(mask)}
    izu_moves = {move["stable_id"] for move in catalog.moves if move["legal_state"]["stances"] == ["IZU"]}
    plain_moves = {move["stable_id"] for move in catalog.moves if move["stable_id"].startswith("jun:")
                   and not move["legal_state"]["stances"] and not move["legal_state"]["situations"]
                   and not move["legal_state"]["requires_heat"]}
    assert legal & izu_moves, "IZU moves become legal in IZU"
    assert not legal & plain_moves, "moves from neutral are not legal while in a stance"

    # Any movement leaves the stance; being hit interrupts and makes the state unknown for a while.
    tracker.on_dispatch("universal:walk_back", 1.0)
    assert tracker.observe(1.0, busy=False).stance == ""
    tracker.on_dispatch("jun:7", 2.0)
    tracker.on_own_hit(2.05)
    unsettled = tracker.observe(2.2, busy=False)
    assert unsettled.stance is None and unsettled.airborne is None
    assert tracker.observe(4.0, busy=False).stance == "", "a hit cancels the stance the move would have entered"


def test_tracker_uses_on_hit_results_and_unknown_timings():
    timings = load_move_timings(BINDINGS, "jun")
    on_hit = next(stable_id for stable_id, timing in timings.items()
                  if timing.result_stance_on_hit and timing.result_stance_on_hit != timing.result_stance)
    tracker = OwnStateTracker(timings)
    tracker.on_dispatch(on_hit, 0.0)
    tracker.on_outcome("hit")
    assert tracker.observe(10.0, busy=False).stance == timings[on_hit].result_stance_on_hit
    tracker.on_dispatch("jun:not-a-move", 20.0)
    assert tracker.observe(30.0, busy=False).stance is None, "an undocumented move leaves the state unknown"
    tracker.reset_round()
    assert tracker.observe(0.0, busy=False).stance == ""


def test_hud_gauge_heat_and_rage_estimates():
    frame = np.zeros((20, 100, 3), dtype=np.uint8)
    gauge = HudGauge(left=0, top=5, right=100, bottom=15, color=(250, 120, 20), tolerance=30)
    heat = HeatEstimate(gauge)
    assert heat.update(frame) is None, "unknown before the gauge was seen full"
    frame[5:15, :, :] = (250, 120, 20)
    assert gauge.fill(frame) == 1.0 and heat.update(frame) is False
    frame[5:15, 70:, :] = 0
    assert abs(gauge.fill(frame) - 0.7) < 1e-9 and heat.update(frame) is True, "a draining full gauge is Heat"
    frame[5:15, :, :] = 0
    assert heat.update(frame) is False, "an empty gauge ends Heat for the round"
    assert HudGauge.from_config({}, "p1_heat_gauge") is None
    configured = HudGauge.from_config({"p1_heat_gauge": {"left": 1, "top": 2, "right": 3, "bottom": 4,
                                                         "color": [1, 2, 3]}}, "p1_heat_gauge")
    assert configured == HudGauge(1, 2, 3, 4, (1, 2, 3))
    rage = RageEstimate(0.25)
    assert rage.update(0.2) is True and rage.update(0.5) is False and rage.update(0.2, rage_art_used=True) is None


def test_screen_full_move_observation_layout():
    tracker = OwnStateTracker(load_move_timings(BINDINGS, "jun"))
    observed = tracker.observe(0.0, busy=False, heat=True, rage=False)
    vector = screen_full_move_observation(estimate(p1_health_ratio=0.8, p2_health_ratio=0.3, distance=2.0),
                                          observed, player=1, own_character=3, opponent_character=31)
    assert vector.shape == (148,) and vector.dtype == np.float32
    assert vector[64 + 3] == 1.0 and vector[64 + 42 + 31] == 1.0 and vector[64:].sum() == 2.0
    assert vector[0] == np.float32(0.8) and vector[1] == np.float32(0.3) and vector[4] == 0.5
    assert vector[8] == 1.0 and vector[24] == 1.0 and vector[63] == 1.0
    assert vector[26] == 0.0, "Heat is not read from the screen (screen contract v4)"
    p2 = screen_full_move_observation(estimate(p1_health_ratio=0.8, p2_health_ratio=0.3), observed, player=2)
    assert p2[0] == np.float32(0.3) and p2[1] == np.float32(0.8)


def test_heat_stays_spent_for_the_round():
    frame = np.zeros((20, 100, 3), dtype=np.uint8)
    heat = HeatEstimate(HudGauge(left=0, top=5, right=100, bottom=15, color=(250, 120, 20), tolerance=30))
    frame[5:15, :, :] = (250, 120, 20)
    heat.update(frame)
    frame[5:15, 50:, :] = 0
    assert heat.update(frame) is True
    frame[5:15, :, :] = 0
    assert heat.update(frame) is False and heat.update(frame) is False, "an empty gauge after Heat stays spent"
    heat.reset_round()
    assert heat.update(frame) is None
