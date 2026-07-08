from __future__ import annotations

import numpy as np

from t8_agent.sim.tekken_lite import FighterRuntime, SimAction, SimState

# Full real-game-ish vocabulary. Not all of these are promoted into the current
# PPO training action space yet, because old checkpoints use the compact space.
FULL_ACTION_SPACE = list(SimAction)

ACTION_SPACE = [
    SimAction.NEUTRAL,
    SimAction.WALK_FORWARD,
    SimAction.WALK_BACK,
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]

ATTACK_OR_GUARD_ACTIONS = {
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
}


def action_count() -> int:
    return len(ACTION_SPACE)


def action_to_index(action: SimAction) -> int:
    return ACTION_SPACE.index(action)


def index_to_action(index: int) -> SimAction:
    return ACTION_SPACE[index]


def legal_action_mask(state: SimState, player: int) -> np.ndarray:
    fighter = _fighter_for_player(state, player)
    mask = np.ones(action_count(), dtype=bool)
    if state.round_over:
        mask[:] = False
        mask[action_to_index(SimAction.NEUTRAL)] = True
        return mask
    if fighter.busy:
        mask[:] = False
        mask[action_to_index(SimAction.NEUTRAL)] = True
    return mask


def _fighter_for_player(state: SimState, player: int) -> FighterRuntime:
    if player == 1:
        return state.p1
    if player == 2:
        return state.p2
    raise ValueError(f"player must be 1 or 2, got {player}")
