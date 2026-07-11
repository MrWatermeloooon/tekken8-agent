from t8_agent.sim import SimAction, TekkenLiteEnv
from t8_agent.sim.action_space import ACTION_SPACE, action_to_index


def test_busy_fighter_can_only_choose_neutral() -> None:
    env = TekkenLiteEnv(seed=12)
    env.reset()
    env.step(SimAction.JAB, SimAction.NEUTRAL)

    mask = env.legal_action_mask(player=1)

    assert mask[action_to_index(SimAction.NEUTRAL)]
    assert not mask[action_to_index(SimAction.JAB)]
    assert mask.sum() == 1


def test_ready_fighter_has_full_action_mask() -> None:
    env = TekkenLiteEnv(seed=13)
    env.reset()

    mask = env.legal_action_mask(player=1)

    assert mask.all()


def test_training_action_space_includes_movement_and_blocking() -> None:
    assert SimAction.WALK_BACK in ACTION_SPACE
    assert SimAction.DASH_BACK in ACTION_SPACE
    assert SimAction.JUMP in ACTION_SPACE
    assert SimAction.SIDESTEP_LEFT in ACTION_SPACE
    assert SimAction.SIDESTEP_RIGHT in ACTION_SPACE
    assert SimAction.SIDEWALK_LEFT in ACTION_SPACE
    assert SimAction.SIDEWALK_RIGHT in ACTION_SPACE
    assert SimAction.BLOCK_HIGH in ACTION_SPACE
    assert SimAction.BLOCK_LOW in ACTION_SPACE
    assert SimAction.THROW_BREAK_1 in ACTION_SPACE
    assert SimAction.THROW_BREAK_2 in ACTION_SPACE
    assert SimAction.THROW_BREAK_1_2 in ACTION_SPACE
