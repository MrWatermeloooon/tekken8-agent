from t8_agent.sim.actions import SimAction

FULL_ACTION_SPACE = list(SimAction)

ACTION_SPACE = [
    SimAction.NEUTRAL,
    SimAction.WALK_FORWARD,
    SimAction.WALK_BACK,
    SimAction.DASH_FORWARD,
    SimAction.DASH_BACK,
    SimAction.CROUCH,
    SimAction.STAND,
    SimAction.JUMP,
    SimAction.SIDESTEP_LEFT,
    SimAction.SIDESTEP_RIGHT,
    SimAction.SIDEWALK_LEFT,
    SimAction.SIDEWALK_RIGHT,
    SimAction.BLOCK_HIGH,
    SimAction.BLOCK_LOW,
    SimAction.LOW_PARRY,
    SimAction.THROW_BREAK_1,
    SimAction.THROW_BREAK_2,
    SimAction.THROW_BREAK_1_2,
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]

def action_count() -> int:
    return len(ACTION_SPACE)


def action_to_index(action: SimAction) -> int:
    return ACTION_SPACE.index(action)


def index_to_action(index: int) -> SimAction:
    return ACTION_SPACE[index]
