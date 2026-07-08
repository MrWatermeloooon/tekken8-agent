from t8_agent.sim import SimAction, TekkenLiteEnv
from t8_agent.sim.observations import observation_size, vector_observation


def test_vector_observation_includes_move_progress_features() -> None:
    env = TekkenLiteEnv(seed=15)
    env.reset()
    env.step(SimAction.JAB, SimAction.NEUTRAL)

    obs = vector_observation(env.state, env.config, player=1)

    assert obs.shape == (observation_size(),)
    assert observation_size() == 18
    assert obs[13] > 0.0
    assert obs[14] == 0.0
