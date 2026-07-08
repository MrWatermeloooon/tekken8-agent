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


def test_opponent_pool_can_sample_by_rating(tmp_path) -> None:
    first = tmp_path / "a.zip"
    second = tmp_path / "b.zip"
    first.write_text("placeholder", encoding="utf-8")
    second.write_text("placeholder", encoding="utf-8")
    pool = OpponentPool(
        scripted_names=["rushdown"],
        checkpoint_paths=[first, second],
        checkpoint_ratings={str(first): 1000.0, str(second): 1400.0},
        target_rating=1400.0,
    )

    sampled = pool._sample_by_rating()

    assert sampled in {first, second}
