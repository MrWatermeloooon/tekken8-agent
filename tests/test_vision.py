from __future__ import annotations

import numpy as np
import torch

from t8_agent.core.types import GameState, PlayerState
from t8_agent.live.vision_agent import LiveVisionAgent
from t8_agent.sim.tekken_lite import SimAction
from t8_agent.vision.model import LearnedTemporalEstimator, TemporalStateNet
from t8_agent.vision.temporal import TemporalScreenEstimator, VisualEstimate


def _state(p2_health: float = 180.0) -> GameState:
    return GameState(
        p1=PlayerState(health=180.0, position_x=-1.0),
        p2=PlayerState(health=p2_health, position_x=1.0, facing=-1),
        round_timer=60.0,
    )


def test_temporal_estimator_detects_motion_and_health_drop() -> None:
    estimator = TemporalScreenEstimator()
    first = np.zeros((180, 320, 3), dtype=np.uint8)
    second = first.copy()
    second[50:130, 180:260] = 255

    estimator.update(_state(), first)
    estimate = estimator.update(_state(p2_health=170.0), second)

    assert estimate.p2_hit_event is True
    assert estimate.p2_motion > 0.0
    assert estimate.to_vector().shape == (13,)


def test_vision_agent_approaches_at_long_range() -> None:
    estimate = VisualEstimate(1.0, 1.0, -2.0, 2.0, 4.0, 0.0, 0.0, 0.0, 0.0, False, False, 0.0, 0.0)

    assert LiveVisionAgent().act(estimate) == SimAction.DASH_FORWARD


def test_temporal_state_model_output_shape() -> None:
    model = TemporalStateNet()

    output = model(torch.zeros((2, 4, 3, 180, 320)))

    assert output.shape == (2, 13)


def test_learned_estimator_waits_for_complete_clip(tmp_path) -> None:
    checkpoint = tmp_path / "vision.pt"
    TemporalStateNet().save(checkpoint)
    estimator = LearnedTemporalEstimator(checkpoint, clip_length=2)
    frame = np.zeros((180, 320, 3), dtype=np.uint8)

    assert estimator.update(frame) is None
    assert estimator.update(frame) is not None
