from __future__ import annotations

from collections.abc import Callable

from t8_agent.sim.moves import HitLevel, JUN_MOVES
from t8_agent.sim.tekken_lite import FighterRuntime, SimAction, TekkenLiteEnv

ScriptedPolicy = Callable[[TekkenLiteEnv, int], SimAction]

DEFAULT_SCRIPTED_OPPONENTS = [
    "poke",
    "rushdown",
    "turtle",
    "whiff_punish",
    "keepout",
    "frame_trap",
    "anti_throw",
]

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
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if opponent.move_key is not None and env.state.distance < 1.05:
        return _guard_against(env, player)
    if env.state.distance > 1.05:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.18:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])


def turtle_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.3:
        return SimAction.WALK_FORWARD
    if opponent.move_key is not None:
        return _guard_against(env, player)
    if env.rng.random() < 0.25:
        return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])


def whiff_punish_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if _opponent_is_punishable(env, player) and env.state.distance < 1.25:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    if opponent.move_key is not None and env.state.distance < 1.1:
        return _guard_against(env, player)
    if env.state.distance < 0.65:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.BLOCK_HIGH, SimAction.THROW])
    if env.state.distance > 1.25:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.DF1, SimAction.F2])


def rushdown_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if opponent.move_key is not None and env.state.distance < 0.95:
        return env.rng.choice([_guard_against(env, player), SimAction.JAB])
    if env.state.distance > 0.82:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.08:
        return SimAction.BLOCK_HIGH
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


def keepout_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if _opponent_is_punishable(env, player) and env.state.distance < 1.25:
        return env.rng.choice([SimAction.F2, SimAction.DF1, SimAction.HOPKICK])
    if opponent.move_key is not None and env.state.distance < 1.2:
        return _guard_against(env, player)
    if env.state.distance < 0.68:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.WALK_BACK, SimAction.DF1, SimAction.BLOCK_HIGH])
    if env.state.distance > 1.25:
        return SimAction.WALK_FORWARD
    if env.state.distance > 0.95:
        return env.rng.choice([SimAction.F2, SimAction.F2, SimAction.DF1, SimAction.WALK_BACK])
    return env.rng.choice([SimAction.DF1, SimAction.DB3, SimAction.JAB, SimAction.BLOCK_HIGH])


def frame_trap_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if _opponent_is_punishable(env, player) and env.state.distance < 1.25:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    if opponent.move_key is not None and env.state.distance < 1.05:
        return _guard_against(env, player)
    if env.state.distance > 0.95:
        return SimAction.WALK_FORWARD
    if env.state.distance < 0.55:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.JAB, SimAction.DF1])

    roll = env.rng.random()
    if roll < 0.35:
        return SimAction.JAB
    if roll < 0.65:
        return SimAction.DF1
    if roll < 0.82:
        return SimAction.DB3
    if roll < 0.92:
        return SimAction.THROW
    return SimAction.BLOCK_HIGH


def anti_throw_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if _opponent_is_punishable(env, player) and env.state.distance < 1.25:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.DF1])
    if opponent.move_key is not None and env.state.distance < 1.05:
        return _guard_against(env, player)
    if env.state.distance < 0.58:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.WALK_BACK, SimAction.HOPKICK, SimAction.DF1])
    if env.state.distance > 1.15:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.DF1, SimAction.DB3, SimAction.F2, SimAction.BLOCK_HIGH])


def _fighters(env: TekkenLiteEnv, player: int) -> tuple[FighterRuntime, FighterRuntime]:
    own = env.state.p1 if player == 1 else env.state.p2
    opponent = env.state.p2 if player == 1 else env.state.p1
    return own, opponent


def _guard_against(env: TekkenLiteEnv, player: int) -> SimAction:
    _own, opponent = _fighters(env, player)
    if opponent.move_key is not None:
        move = JUN_MOVES[opponent.move_key]
        if move.hit_level == HitLevel.LOW:
            return SimAction.BLOCK_LOW
        if move.hit_level == HitLevel.THROW and env.state.distance < 0.58:
            return SimAction.WALK_BACK
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])


def _opponent_is_punishable(env: TekkenLiteEnv, player: int) -> bool:
    _own, opponent = _fighters(env, player)
    if opponent.move_key is None:
        return False
    move = JUN_MOVES[opponent.move_key]
    remaining = max(0, move.total_frames - opponent.move_frame)
    active_is_over = opponent.move_frame > move.startup + move.active
    return active_is_over and remaining >= 8


SCRIPTED_POLICIES: dict[str, ScriptedPolicy] = {
    "random": random_policy,
    "poke": poke_policy,
    "rushdown": rushdown_policy,
    "turtle": turtle_policy,
    "whiff_punish": whiff_punish_policy,
    "keepout": keepout_policy,
    "frame_trap": frame_trap_policy,
    "anti_throw": anti_throw_policy,
}
