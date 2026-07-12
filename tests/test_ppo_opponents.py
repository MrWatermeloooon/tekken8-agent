import random

import pytest

from scripts.train_sim_ppo_selfplay import scripted_curriculum_names
from scripts.visualize_sim import OpponentRotator
from t8_agent.sim.opponents import (
    DEFAULT_SCRIPTED_OPPONENTS,
    SCRIPTED_BASE_POLICIES,
    SCRIPTED_DIFFICULTY_TIERS,
    SCRIPTED_POLICIES,
)
from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv
from t8_agent.train.ppo_opponents import OpponentPool


def test_opponent_pool_samples_scripted_policy_without_checkpoints() -> None:
    pool = OpponentPool(scripted_names=["rushdown"])

    name, policy = pool.sample()

    assert name == "rushdown"
    assert callable(policy)


def test_opponent_pool_balances_scripted_opponents_per_cycle() -> None:
    names = ["random", "poke", "rushdown", "turtle"]
    pool = OpponentPool(scripted_names=names, rng=random.Random(123))

    first_cycle = [pool.sample()[0] for _ in names]
    second_cycle = [pool.sample()[0] for _ in names]

    assert sorted(first_cycle) == sorted(names)
    assert sorted(second_cycle) == sorted(names)


def test_scripted_curriculum_unlocks_all_opponents_by_stage_three() -> None:
    configured = ["random", "poke", "whiff_punish", "ace"]

    assert scripted_curriculum_names(configured, 1) == ["random", "poke"]
    assert scripted_curriculum_names(configured, 2) == ["random", "poke", "whiff_punish"]
    assert scripted_curriculum_names(configured, 3) == configured


def test_visualizer_scripted_rotation_is_balanced() -> None:
    names = ["random", "poke", "rushdown"]
    rotator = OpponentRotator(
        mode="scripted",
        scripted_names=names,
        checkpoint_dir=None,
        scripted_rate=1.0,
        seed=123,
    )

    labels = [rotator.next().label for _ in names]

    assert sorted(labels) == sorted(names)


def test_opponent_pool_rejects_unknown_scripted_policy() -> None:
    with pytest.raises(ValueError, match="unknown opponent"):
        OpponentPool(scripted_names=["missing"])


def test_default_scripted_roster_matches_registered_policies() -> None:
    assert DEFAULT_SCRIPTED_OPPONENTS
    assert len(DEFAULT_SCRIPTED_OPPONENTS) == len(set(DEFAULT_SCRIPTED_OPPONENTS))
    assert set(DEFAULT_SCRIPTED_OPPONENTS).issubset(SCRIPTED_POLICIES)
    assert {
        "anti_throw",
        "keepout",
        "frame_trap",
        "low_spammer",
        "grappler",
        "zoner",
        "throw_looper",
        "sparring_partner",
        "ace",
        "stone_wall",
        "fifty_fifty_master",
        "whiff_reaper",
        "corner_king",
        "the_gatekeeper",
        "nightmare",
        "the_apprentice",
    }.issubset(DEFAULT_SCRIPTED_OPPONENTS)
    assert len(DEFAULT_SCRIPTED_OPPONENTS) == 58
    assert set(SCRIPTED_POLICIES) == set(SCRIPTED_BASE_POLICIES)
    assert all(SCRIPTED_POLICIES[name] is not SCRIPTED_BASE_POLICIES[name] for name in SCRIPTED_POLICIES)
    tiered = [name for names in SCRIPTED_DIFFICULTY_TIERS.values() for name in names]
    assert sorted(tiered) == sorted(DEFAULT_SCRIPTED_OPPONENTS)
    assert len(tiered) == len(set(tiered))


@pytest.mark.parametrize("opponent_name", DEFAULT_SCRIPTED_OPPONENTS)
def test_every_scripted_opponent_returns_an_action(opponent_name: str) -> None:
    env = TekkenLiteEnv(seed=123)
    env.reset(seed=123)

    action = SCRIPTED_POLICIES[opponent_name](env, 2)

    assert isinstance(action, SimAction)


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


def test_opponent_pool_respects_zero_target_rating(tmp_path) -> None:
    first = tmp_path / "a.zip"
    second = tmp_path / "b.zip"
    first.write_text("placeholder", encoding="utf-8")
    second.write_text("placeholder", encoding="utf-8")
    pool = OpponentPool(
        scripted_names=["rushdown"],
        checkpoint_paths=[first, second],
        checkpoint_ratings={str(first): 0.0, str(second): 1000.0},
        target_rating=0.0,
        rng=random.Random(0),
    )

    sampled = pool._sample_by_rating()

    assert sampled == first


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


def test_opponent_pool_can_force_best_checkpoint_by_rating(tmp_path) -> None:
    first = tmp_path / "iter_001.zip"
    second = tmp_path / "iter_002.zip"
    first.write_text("placeholder", encoding="utf-8")
    second.write_text("placeholder", encoding="utf-8")
    pool = OpponentPool(
        scripted_names=["rushdown"],
        checkpoint_paths=[first, second],
        checkpoint_ratings={str(first): 1800.0, str(second): 1100.0},
        scripted_sample_rate=0.0,
        best_checkpoint_sample_rate=1.0,
        rng=random.Random(123),
    )
    pool._load = lambda path: (lambda env, player: SimAction.NEUTRAL)  # type: ignore[method-assign]

    name, policy = pool.sample()

    assert name == "checkpoint:iter_001.zip"
    assert callable(policy)


def test_opponent_pool_can_force_latest_checkpoint(tmp_path) -> None:
    first = tmp_path / "iter_001.zip"
    second = tmp_path / "iter_002.zip"
    first.write_text("placeholder", encoding="utf-8")
    second.write_text("placeholder", encoding="utf-8")
    pool = OpponentPool(
        scripted_names=["rushdown"],
        checkpoint_paths=[first, second],
        checkpoint_ratings={str(first): 1800.0, str(second): 1100.0},
        scripted_sample_rate=0.0,
        latest_checkpoint_sample_rate=1.0,
        rng=random.Random(123),
    )
    pool._load = lambda path: (lambda env, player: SimAction.NEUTRAL)  # type: ignore[method-assign]

    name, policy = pool.sample()

    assert name == "checkpoint:iter_002.zip"
    assert callable(policy)


def test_opponent_pool_rejects_invalid_sample_rates() -> None:
    with pytest.raises(ValueError, match="scripted_sample_rate"):
        OpponentPool(scripted_names=["rushdown"], scripted_sample_rate=-0.1)
    with pytest.raises(ValueError, match="best_checkpoint_sample_rate"):
        OpponentPool(scripted_names=["rushdown"], best_checkpoint_sample_rate=-0.1)
    with pytest.raises(ValueError, match="latest_checkpoint_sample_rate"):
        OpponentPool(scripted_names=["rushdown"], latest_checkpoint_sample_rate=1.1)
    with pytest.raises(ValueError, match="at most 1"):
        OpponentPool(
            scripted_names=["rushdown"],
            best_checkpoint_sample_rate=0.8,
            latest_checkpoint_sample_rate=0.3,
        )
    with pytest.raises(ValueError, match="old_checkpoint_sample_rate"):
        OpponentPool(scripted_names=["rushdown"], old_checkpoint_sample_rate=1.1)
    with pytest.raises(ValueError, match="max_recent_checkpoints"):
        OpponentPool(scripted_names=["rushdown"], max_recent_checkpoints=0)
