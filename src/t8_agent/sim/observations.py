from __future__ import annotations

import numpy as np

from t8_agent.sim.moves import JUN_MOVES
from t8_agent.sim.tekken_lite import FighterRuntime, SimConfig, SimState


def observation_size() -> int:
    return 18


def vector_observation(state: SimState, config: SimConfig, player: int) -> np.ndarray:
    own, opponent = _fighters_for_player(state, player)
    forward = 1.0 if player == 1 else -1.0
    signed_distance = (opponent.x - own.x) * forward
    own_forward_wall = (config.stage_half_width - own.x) if player == 1 else (own.x + config.stage_half_width)
    own_back_wall = (own.x + config.stage_half_width) if player == 1 else (config.stage_half_width - own.x)

    return np.array(
        [
            own.health / config.max_health,
            opponent.health / config.max_health,
            signed_distance / (config.stage_half_width * 2.0),
            state.distance / (config.stage_half_width * 2.0),
            own_forward_wall / (config.stage_half_width * 2.0),
            own_back_wall / (config.stage_half_width * 2.0),
            state.frame / config.max_frames,
            min(1.0, own.hitstun / 60.0),
            min(1.0, opponent.hitstun / 60.0),
            min(1.0, own.blockstun / 60.0),
            min(1.0, opponent.blockstun / 60.0),
            1.0 if own.move_key is not None else 0.0,
            1.0 if opponent.move_key is not None else 0.0,
            _move_frames_remaining(own),
            _move_frames_remaining(opponent),
            1.0 if own.guard is not None else 0.0,
            1.0 if opponent.guard is not None else 0.0,
            1.0,
        ],
        dtype=np.float32,
    )


def _fighters_for_player(state: SimState, player: int) -> tuple[FighterRuntime, FighterRuntime]:
    if player == 1:
        return state.p1, state.p2
    if player == 2:
        return state.p2, state.p1
    raise ValueError(f"player must be 1 or 2, got {player}")


def _move_frames_remaining(fighter: FighterRuntime) -> float:
    if fighter.move_key is None:
        return 0.0
    move = JUN_MOVES[fighter.move_key]
    remaining = max(0, move.total_frames - fighter.move_frame)
    return min(1.0, remaining / move.total_frames)
