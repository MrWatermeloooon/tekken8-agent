from t8_agent.sim import SimAction, TekkenLiteEnv
from t8_agent.sim.observations import observation_size, vector_observation, visual_observation_size, visual_vector_observation


def test_vector_observation_includes_move_progress_features() -> None:
    env = TekkenLiteEnv(seed=15)
    env.reset()
    env.step(SimAction.JAB, SimAction.NEUTRAL)

    obs = vector_observation(env.state, env.config, player=1)

    assert obs.shape == (observation_size(),)
    assert observation_size() == 19
    assert obs[13] > 0.0
    assert obs[14] == 0.0


def test_vector_observation_marks_incoming_throw_threat() -> None:
    env = TekkenLiteEnv(seed=16)
    env.reset()
    env.state = env.state.__class__(
        p1=env.state.p1,
        p2=env.state.p2.__class__(health=180.0, x=0.48, move_key="throw", move_frame=6),
        frame=0,
    )

    obs = vector_observation(env.state, env.config, player=1)

    assert obs[17] == 1.0


def test_visual_observation_size_matches_vector() -> None:
    env = TekkenLiteEnv(seed=17)

    obs = visual_vector_observation(env.state, env.config, player=1)

    assert visual_observation_size() == 13
    assert obs.shape == (13,)
