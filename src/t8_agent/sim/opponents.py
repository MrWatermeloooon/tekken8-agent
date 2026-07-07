from __future__ import annotations

from collections.abc import Callable

from t8_agent.sim.tekken_lite import SimAction, TekkenLiteEnv

ScriptedPolicy = Callable[[TekkenLiteEnv, int], SimAction]

ATTACKS = [
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]


def random_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    _ = player
    return env.sample_action()


def poke_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own = env.state.p1 if player == 1 else env.state.p2
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.05:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.18:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])


def turtle_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own = env.state.p1 if player == 1 else env.state.p2
    opponent = env.state.p2 if player == 1 else env.state.p1
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.3:
        return SimAction.WALK_FORWARD
    if opponent.move_key is not None:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    if env.rng.random() < 0.25:
        return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])


def whiff_punish_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own = env.state.p1 if player == 1 else env.state.p2
    opponent = env.state.p2 if player == 1 else env.state.p1
    if own.busy:
        return SimAction.NEUTRAL
    if opponent.move_key is not None and env.state.distance < 1.25:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    if env.state.distance < 0.65:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.BLOCK_HIGH, SimAction.THROW])
    if env.state.distance > 1.25:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.DF1, SimAction.F2])


def rushdown_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own = env.state.p1 if player == 1 else env.state.p2
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.82:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.08:
        return SimAction.BLOCK_HIGH
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


SCRIPTED_POLICIES: dict[str, ScriptedPolicy] = {
    "random": random_policy,
    "poke": poke_policy,
    "rushdown": rushdown_policy,
    "turtle": turtle_policy,
    "whiff_punish": whiff_punish_policy,
}
