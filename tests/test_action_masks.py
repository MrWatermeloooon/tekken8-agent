from t8_agent.sim import SimAction, TekkenLiteEnv
from t8_agent.sim.action_space import action_to_index


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
