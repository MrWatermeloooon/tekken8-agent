from t8_agent.sim import SimAction, TekkenLiteEnv
from t8_agent.sim.opponents import DEFAULT_SCRIPTED_OPPONENTS, SCRIPTED_POLICIES
from t8_agent.sim.tekken_lite import FighterRuntime, SimState


def advance(env: TekkenLiteEnv, p1_action: SimAction, p2_action: SimAction, decisions: int = 20):
    result = None
    for idx in range(decisions):
        action = p1_action if idx == 0 else SimAction.NEUTRAL
        result = env.step(action, p2_action)
    assert result is not None
    return result


def walk_into_range(env: TekkenLiteEnv, decisions: int = 12) -> None:
    for _ in range(decisions):
        env.step(SimAction.WALK_FORWARD, SimAction.WALK_FORWARD)


def first_decision_then_neutral(env: TekkenLiteEnv, p1_action: SimAction, p2_action: SimAction, decisions: int = 20):
    result = None
    for idx in range(decisions):
        result = env.step(
            p1_action if idx == 0 else SimAction.NEUTRAL,
            p2_action if idx == 0 else SimAction.NEUTRAL,
        )
    assert result is not None
    return result


def test_mid_attack_damages_in_range() -> None:
    env = TekkenLiteEnv(seed=1)
    env.reset()
    walk_into_range(env)

    result = advance(env, SimAction.DF1, SimAction.NEUTRAL)

    assert result.state.p2.health < env.config.max_health
    assert result.info["damage_to_p2"] >= 0.0


def test_high_block_stops_mid_damage() -> None:
    env = TekkenLiteEnv(seed=2)
    env.reset()
    walk_into_range(env)

    result = None
    blocked = False
    for idx in range(20):
        action = SimAction.DF1 if idx == 0 else SimAction.NEUTRAL
        result = env.step(action, SimAction.BLOCK_HIGH)
        blocked = blocked or result.info["p2_blocks"] > 0
    assert result is not None

    assert result.state.p2.health == env.config.max_health
    assert blocked


def test_low_block_stops_low_but_not_mid() -> None:
    env = TekkenLiteEnv(seed=3)
    env.reset()
    walk_into_range(env)
    low_result = advance(env, SimAction.DB3, SimAction.BLOCK_LOW)
    assert low_result.state.p2.health == env.config.max_health

    env.reset()
    walk_into_range(env)
    mid_result = advance(env, SimAction.DF1, SimAction.BLOCK_LOW)
    assert mid_result.state.p2.health < env.config.max_health


def test_whiff_does_not_damage_out_of_range() -> None:
    env = TekkenLiteEnv(seed=4)
    env.reset()
    for _ in range(40):
        env.step(SimAction.WALK_BACK, SimAction.WALK_BACK)

    result = advance(env, SimAction.JAB, SimAction.NEUTRAL)

    assert result.state.p2.health == env.config.max_health
    assert result.state.p1.whiffs >= 1


def test_round_can_terminate() -> None:
    env = TekkenLiteEnv(seed=5)
    env.reset()
    result = None
    for _ in range(5000):
        result = env.step(SimAction.HOPKICK, SimAction.NEUTRAL)
        if result.terminated:
            break

    assert result is not None
    assert result.terminated
    assert result.state.winner == 1


def test_p2_rushdown_can_damage_p1() -> None:
    env = TekkenLiteEnv(seed=6)
    env.reset()
    p2_policy = SCRIPTED_POLICIES["rushdown"]
    damage_to_p1 = 0.0

    for _ in range(400):
        result = env.step(SimAction.NEUTRAL, p2_policy(env, 2))
        damage_to_p1 += float(result.info["damage_to_p1"])
        if damage_to_p1 > 0:
            break

    assert damage_to_p1 > 0
    assert env.state.p1.health < env.config.max_health


def test_default_curriculum_uses_harder_scripted_opponents() -> None:
    assert "random" not in DEFAULT_SCRIPTED_OPPONENTS
    assert {"keepout", "frame_trap", "anti_throw"}.issubset(DEFAULT_SCRIPTED_OPPONENTS)
    assert all(name in SCRIPTED_POLICIES for name in DEFAULT_SCRIPTED_OPPONENTS)


def test_hard_scripted_opponents_can_apply_pressure() -> None:
    for idx, name in enumerate(["keepout", "frame_trap", "anti_throw"], start=1):
        env = TekkenLiteEnv(seed=100 + idx)
        env.reset()
        policy = SCRIPTED_POLICIES[name]
        damage_to_p1 = 0.0

        for _ in range(600):
            result = env.step(SimAction.NEUTRAL, policy(env, 2))
            damage_to_p1 += float(result.info["damage_to_p1"])
            if damage_to_p1 > 0:
                break

        assert damage_to_p1 > 0, name


def test_block_reward_favors_defender() -> None:
    env = TekkenLiteEnv(seed=8)
    env.reset()
    walk_into_range(env)

    result = None
    for idx in range(20):
        action = SimAction.DF1 if idx == 0 else SimAction.NEUTRAL
        result = env.step(action, SimAction.BLOCK_HIGH)
        if result.info["p2_blocks"] > 0:
            break
    assert result is not None

    assert result.info["p2_blocks"] > 0
    assert result.reward_p2 > result.reward_p1


def test_throw_damage_is_discounted_as_training_reward() -> None:
    env = TekkenLiteEnv(seed=9)
    env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=180.0, x=0.48),
        frame=0,
    )

    result = None
    for idx in range(20):
        action = SimAction.THROW if idx == 0 else SimAction.NEUTRAL
        result = env.step(action, SimAction.NEUTRAL)
        if result.info["p1_throw_damage"] > 0:
            break
    assert result is not None

    assert result.info["p1_throw_damage"] > 0
    assert result.reward_p1 < float(result.info["damage_to_p2"]) * 0.5


def test_simultaneous_trade_is_symmetric() -> None:
    env = TekkenLiteEnv(seed=7)
    env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=180.0, x=0.82),
        frame=0,
    )

    result = advance(env, SimAction.JAB, SimAction.JAB)

    assert result.state.p1.health == 173.0
    assert result.state.p2.health == 173.0
    assert result.info["damage_to_p1"] >= 0.0
    assert result.info["damage_to_p2"] >= 0.0


def test_round_win_reward_scales_with_remaining_health() -> None:
    full_health_env = TekkenLiteEnv(seed=11)
    full_health_env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=1.0, x=0.82),
        frame=0,
    )
    low_health_env = TekkenLiteEnv(seed=12)
    low_health_env.state = SimState(
        p1=FighterRuntime(health=1.0, x=0.0),
        p2=FighterRuntime(health=1.0, x=0.82),
        frame=0,
    )

    full_health_result = advance(full_health_env, SimAction.JAB, SimAction.NEUTRAL)
    low_health_result = advance(low_health_env, SimAction.JAB, SimAction.NEUTRAL)

    assert full_health_result.terminated
    assert low_health_result.terminated
    assert full_health_result.reward_p1 > low_health_result.reward_p1


def test_whiff_punish_bonus_rewards_hitting_recovery() -> None:
    env = TekkenLiteEnv(seed=13)
    env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=180.0, x=0.82, move_key="jab", move_frame=13, has_hit=False),
        frame=0,
    )

    result = None
    for idx in range(20):
        action = SimAction.JAB if idx == 0 else SimAction.NEUTRAL
        result = env.step(action, SimAction.NEUTRAL)
        if result.info["p1_whiff_punish_bonus"] > 0:
            break
    assert result is not None

    assert result.info["damage_to_p2"] > 0
    assert result.info["p1_whiff_punish_bonus"] > 0


def test_sidestepping_can_make_linear_attack_whiff() -> None:
    env = TekkenLiteEnv(seed=10)
    env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0, y=0.5),
        p2=FighterRuntime(health=180.0, x=0.82, y=0.0),
        frame=0,
    )

    result = advance(env, SimAction.JAB, SimAction.NEUTRAL)

    assert result.state.p2.health == 180.0
    assert result.state.p1.whiffs >= 1


def test_jump_avoids_lows_and_throws_but_not_mids() -> None:
    low_env = TekkenLiteEnv(seed=14)
    low_env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=180.0, x=0.82),
        frame=0,
    )
    low_result = first_decision_then_neutral(low_env, SimAction.JUMP, SimAction.DB3)
    assert low_result.state.p1.health == 180.0

    throw_env = TekkenLiteEnv(seed=15)
    throw_env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=180.0, x=0.48),
        frame=0,
    )
    throw_result = first_decision_then_neutral(throw_env, SimAction.JUMP, SimAction.THROW)
    assert throw_result.state.p1.health == 180.0

    mid_env = TekkenLiteEnv(seed=16)
    mid_env.state = SimState(
        p1=FighterRuntime(health=180.0, x=0.0),
        p2=FighterRuntime(health=180.0, x=0.82),
        frame=0,
    )
    mid_result = first_decision_then_neutral(mid_env, SimAction.JUMP, SimAction.DF1)
    assert mid_result.state.p1.health < 180.0
