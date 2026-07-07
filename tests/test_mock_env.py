from t8_agent.core.types import DiscreteAction
from t8_agent.env.mock_env import MockTekkenEnv


def test_mock_env_runs_one_step() -> None:
    env = MockTekkenEnv()
    initial = env.reset()
    result = env.step(DiscreteAction.WALK_FORWARD)

    assert result.observation.p1.position_x > initial.p1.position_x
    assert not result.terminated
    assert "damage_dealt" in result.info


def test_mock_env_can_terminate_round() -> None:
    env = MockTekkenEnv(max_steps=1000)
    env.reset()

    result = None
    for _ in range(1000):
        env.state = env._transition(env.state, DiscreteAction.WALK_FORWARD)
        result = env.step(DiscreteAction.LEFT_PUNCH)
        if result.terminated or result.truncated:
            break

    assert result is not None
    assert result.terminated or result.truncated
