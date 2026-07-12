from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv
from t8_agent.sim.action_space import action_to_index
from t8_agent.sim.tekken_lite import SimAction


def test_sb3_env_reset_step_and_mask() -> None:
    env = TekkenLiteSingleAgentEnv(opponent_names=["rushdown"], seed=14, max_decisions=20)

    obs, info = env.reset(seed=14)
    mask = env.action_masks()
    next_obs, reward, terminated, truncated, step_info = env.step(0)

    assert obs.shape == env.observation_space.shape
    assert next_obs.shape == env.observation_space.shape
    assert mask.shape == (env.action_space.n,)
    assert info["opponent_name"] == "rushdown"
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert "winner" in step_info


def test_sb3_visual_env_uses_screen_compatible_observation() -> None:
    env = TekkenLiteSingleAgentEnv(opponent_names=["random"], observation_mode="visual")

    observation, _info = env.reset()

    assert observation.shape == (13,)
    assert env.observation_space.shape == (13,)


def test_repeated_committed_attack_penalty_escalates() -> None:
    env = TekkenLiteSingleAgentEnv(opponent_names=["turtle"], repeat_attack_penalty=1.0)
    env.reset(seed=3)
    penalties = []
    for _ in range(3):
        while env.sim.state.p1.busy:
            env.step(action_to_index(SimAction.NEUTRAL))
        _obs, _reward, _terminated, _truncated, info = env.step(action_to_index(SimAction.F2))
        penalties.append(info["repeat_attack_penalty"])

    assert penalties == [0.0, 0.0, 1.0]
