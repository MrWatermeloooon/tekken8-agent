import random

import pytest

from t8_agent.sim.tekken_lite import SimAction
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


def test_opponent_pool_can_disable_scripted_sampling_when_checkpoints_exist(tmp_path) -> None:
    checkpoint = tmp_path / "iter_001.zip"
    checkpoint.write_text("placeholder", encoding="utf-8")
    pool = OpponentPool(
        scripted_names=["rushdown"],
        checkpoint_paths=[checkpoint],
        scripted_sample_rate=0.0,
        rng=random.Random(123),
    )
    pool._load = lambda path: (lambda env, player: SimAction.NEUTRAL)  # type: ignore[method-assign]

    name, policy = pool.sample()

    assert name == "checkpoint:iter_001.zip"
    assert callable(policy)


def test_opponent_pool_can_force_scripted_sampling_when_checkpoints_exist(tmp_path) -> None:
    checkpoint = tmp_path / "iter_001.zip"
    checkpoint.write_text("placeholder", encoding="utf-8")
    pool = OpponentPool(
        scripted_names=["rushdown"],
        checkpoint_paths=[checkpoint],
        scripted_sample_rate=1.0,
        rng=random.Random(123),
    )

    name, policy = pool.sample()

    assert name == "rushdown"
    assert callable(policy)


def test_opponent_pool_rejects_invalid_sample_rates() -> None:
    with pytest.raises(ValueError, match="scripted_sample_rate"):
        OpponentPool(scripted_names=["rushdown"], scripted_sample_rate=-0.1)
    with pytest.raises(ValueError, match="old_checkpoint_sample_rate"):
        OpponentPool(scripted_names=["rushdown"], old_checkpoint_sample_rate=1.1)
    with pytest.raises(ValueError, match="max_recent_checkpoints"):
        OpponentPool(scripted_names=["rushdown"], max_recent_checkpoints=0)
