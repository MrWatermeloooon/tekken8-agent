from t8_agent.train.sb3_env import TekkenLiteSingleAgentEnv


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
