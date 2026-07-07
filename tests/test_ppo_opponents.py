import pytest

from t8_agent.train.ppo_opponents import OpponentPool


def test_opponent_pool_samples_scripted_policy_without_checkpoints() -> None:
    pool = OpponentPool(scripted_names=["rushdown"])

    name, policy = pool.sample()

    assert name == "rushdown"
    assert callable(policy)


def test_opponent_pool_rejects_unknown_scripted_policy() -> None:
    with pytest.raises(ValueError, match="unknown opponent"):
        OpponentPool(scripted_names=["missing"])
