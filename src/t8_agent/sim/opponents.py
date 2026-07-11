from __future__ import annotations

from collections.abc import Callable

from t8_agent.sim.moves import HitLevel, JUN_MOVES
from t8_agent.sim.tekken_lite import FighterRuntime, SimAction, TekkenLiteEnv

ScriptedPolicy = Callable[[TekkenLiteEnv, int], SimAction]

DEFAULT_SCRIPTED_OPPONENTS = [
    "random",
    "poke",
    "rushdown",
    "turtle",
    "whiff_punish",
    "keepout",
    "frame_trap",
    "anti_throw",
    "low_spammer",
    "footsies",
    "grappler",
    "counter_hitter",
    "mixup",
    "wall_pressure",
    "adaptive",
    "launcher_spammer",
    "panic_blocker",
    "backdash_bait",
    "jab_spammer",
    "zoner",
    "hit_confirm",
    "throw_looper",
    "safe_poke",
    "yolo_offense",
    "flowchart",
    "spacing_ghost",
    "sparring_partner",
    "ace",
]

ATTACKS = [
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]

LATERAL_MOVEMENT = [
    SimAction.SIDESTEP_LEFT,
    SimAction.SIDESTEP_RIGHT,
    SimAction.SIDEWALK_LEFT,
    SimAction.SIDEWALK_RIGHT,
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
    if env.rng.random() < 0.08:
        return env.rng.choice(LATERAL_MOVEMENT)
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
    if env.rng.random() < 0.12:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, *LATERAL_MOVEMENT])
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
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.SIDESTEP_LEFT, SimAction.SIDESTEP_RIGHT, SimAction.BLOCK_HIGH, SimAction.THROW])
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
        return SimAction.DASH_FORWARD if env.state.distance > 1.25 and env.rng.random() < 0.35 else SimAction.WALK_FORWARD
    if env.rng.random() < 0.10:
        return env.rng.choice([SimAction.SIDESTEP_LEFT, SimAction.SIDESTEP_RIGHT])
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
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.SIDESTEP_LEFT, SimAction.SIDESTEP_RIGHT, SimAction.DF1, SimAction.BLOCK_HIGH])
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
        return SimAction.DASH_FORWARD if env.state.distance > 1.35 and env.rng.random() < 0.25 else SimAction.WALK_FORWARD
    if env.state.distance < 0.55:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.SIDESTEP_LEFT, SimAction.SIDESTEP_RIGHT, SimAction.JAB, SimAction.DF1])

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
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.JUMP, SimAction.HOPKICK, SimAction.DF1])
    if env.state.distance > 1.15:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.DF1, SimAction.DB3, SimAction.F2, SimAction.BLOCK_HIGH])


def low_spammer_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.0:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.12:
        return SimAction.BLOCK_LOW
    return env.rng.choice([SimAction.DB3, SimAction.DB3, SimAction.JAB, SimAction.THROW])


def footsies_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if distance > 1.3:
        return SimAction.WALK_FORWARD
    if distance < 0.9:
        return SimAction.WALK_BACK
    if env.rng.random() < 0.35:
        return env.rng.choice([SimAction.JAB, SimAction.F2])
    return SimAction.WALK_BACK if env.rng.random() < 0.5 else SimAction.NEUTRAL


def grappler_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.9:
        return SimAction.WALK_FORWARD
    if opponent.move_key is not None:
        return _guard_against(env, player)
    return env.rng.choice([SimAction.THROW, SimAction.THROW, SimAction.DF1, SimAction.JAB])


def counter_hitter_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if opponent.move_key is not None and distance < 1.3:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2])
    if distance > 1.2:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return SimAction.WALK_BACK
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW, SimAction.NEUTRAL])


def mixup_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.05:
        return SimAction.WALK_FORWARD
    roll = env.rng.random()
    if roll < 0.22:
        return SimAction.DB3
    if roll < 0.44:
        return SimAction.JAB
    if roll < 0.60:
        return SimAction.THROW
    if roll < 0.75:
        return SimAction.DF1
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])


def wall_pressure_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    forward = 1.0 if player == 1 else -1.0
    opponent_back_wall = (
        env.config.stage_half_width - opponent.x if forward == 1.0 else opponent.x + env.config.stage_half_width
    )
    if env.state.distance > 0.95:
        return SimAction.WALK_FORWARD
    if opponent_back_wall < 0.6 and env.rng.random() < 0.5:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])


def adaptive_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.1:
        return SimAction.WALK_FORWARD
    lead = own.health - opponent.health
    if lead > 20:
        if env.rng.random() < 0.5:
            return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
        return SimAction.WALK_BACK
    if lead < -20:
        return env.rng.choice([SimAction.HOPKICK, SimAction.THROW, SimAction.F2])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.BLOCK_HIGH])


def launcher_spammer_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.85:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.75:
        return SimAction.HOPKICK
    return env.rng.choice([SimAction.JAB, SimAction.BLOCK_HIGH])


def panic_blocker_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if opponent.move_key is not None or env.rng.random() < 0.55:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    if env.state.distance > 1.2:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DB3])


def backdash_bait_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance < 1.15:
        return SimAction.WALK_BACK
    if opponent.move_key is not None:
        return env.rng.choice([SimAction.F2, SimAction.DF1, SimAction.HOPKICK])
    if env.rng.random() < 0.2:
        return SimAction.WALK_FORWARD
    return SimAction.NEUTRAL


def jab_spammer_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.85:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.85:
        return SimAction.JAB
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.THROW])


def zoner_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if distance < 0.95:
        return SimAction.WALK_BACK
    if distance > 1.25:
        return SimAction.WALK_FORWARD
    if opponent.move_key is not None:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.WALK_BACK])
    if env.rng.random() < 0.4:
        return SimAction.F2
    return SimAction.WALK_BACK


def hit_confirm_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if opponent.hitstun > 0 and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.DF1])
    if opponent.blockstun > 0:
        return env.rng.choice([SimAction.THROW, SimAction.WALK_BACK])
    if distance > 1.05:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.15:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return SimAction.JAB


def throw_looper_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if distance > 0.55:
        if opponent.move_key is not None:
            return SimAction.BLOCK_HIGH
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.THROW, SimAction.THROW, SimAction.THROW, SimAction.JAB])


def safe_poke_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.0:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.3:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice([SimAction.JAB, SimAction.DF1])


def yolo_offense_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.9:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.THROW, SimAction.DF1])


def flowchart_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    sequence = [
        SimAction.WALK_FORWARD,
        SimAction.WALK_FORWARD,
        SimAction.JAB,
        SimAction.DF1,
        SimAction.BLOCK_HIGH,
        SimAction.BLOCK_LOW,
        SimAction.THROW,
        SimAction.WALK_BACK,
    ]
    index = (env.state.frame // env.config.decision_frames) % len(sequence)
    return sequence[index]


def spacing_ghost_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, _opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if distance > 1.5:
        return SimAction.WALK_FORWARD
    if distance < 0.5:
        return SimAction.WALK_BACK
    roll = env.rng.random()
    if roll < 0.3:
        return SimAction.WALK_FORWARD
    if roll < 0.6:
        return SimAction.WALK_BACK
    if roll < 0.75:
        return env.rng.choice([SimAction.JAB, SimAction.DB3])
    if roll < 0.85:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return SimAction.NEUTRAL


def sparring_partner_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance

    if opponent.move_key is not None and distance < 1.3 and env.rng.random() < 0.75:
        move = JUN_MOVES[opponent.move_key]
        if move.hit_level == HitLevel.LOW:
            return SimAction.BLOCK_LOW
        if move.hit_level in (HitLevel.HIGH, HitLevel.MID):
            return SimAction.BLOCK_HIGH

    if opponent.hitstun > 0 and distance < 1.3 and env.rng.random() < 0.7:
        return env.rng.choice([SimAction.F2, SimAction.DF1])

    if distance > 1.2:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return env.rng.choice([SimAction.THROW, SimAction.WALK_BACK, SimAction.JAB])

    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW, SimAction.BLOCK_HIGH])


def ace_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    own, opponent = _fighters(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    forward = 1.0 if player == 1 else -1.0

    if opponent.move_key is not None and distance < 1.3:
        move = JUN_MOVES[opponent.move_key]
        if move.hit_level == HitLevel.LOW:
            return SimAction.BLOCK_LOW
        if move.hit_level in (HitLevel.HIGH, HitLevel.MID):
            return SimAction.BLOCK_HIGH
        return SimAction.WALK_BACK

    if opponent.hitstun > 0 and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    if opponent.blockstun > 0 and distance < 1.0:
        return env.rng.choice([SimAction.THROW, SimAction.WALK_BACK])

    opponent_back_wall = (
        env.config.stage_half_width - opponent.x if forward == 1.0 else opponent.x + env.config.stage_half_width
    )
    lead = own.health - opponent.health

    if distance > 1.2:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return env.rng.choice([SimAction.THROW, SimAction.JAB, SimAction.WALK_BACK])
    if opponent_back_wall < 0.6:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    if lead < -15:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.THROW])
    if lead > 15:
        return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.BLOCK_HIGH])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


def _fighters(env: TekkenLiteEnv, player: int) -> tuple[FighterRuntime, FighterRuntime]:
    own = env.state.p1 if player == 1 else env.state.p2
    opponent = env.state.p2 if player == 1 else env.state.p1
    return own, opponent


def _guard_against(env: TekkenLiteEnv, player: int) -> SimAction:
    _own, opponent = _fighters(env, player)
    if opponent.move_key is not None:
        move = JUN_MOVES[opponent.move_key]
        if move.hit_level == HitLevel.LOW:
            return env.rng.choice([SimAction.BLOCK_LOW, SimAction.LOW_PARRY, SimAction.JUMP])
        if move.hit_level == HitLevel.THROW and env.state.distance < 0.58:
            return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.JUMP, SimAction.THROW_BREAK_1, SimAction.THROW_BREAK_2])
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
    "low_spammer": low_spammer_policy,
    "footsies": footsies_policy,
    "grappler": grappler_policy,
    "counter_hitter": counter_hitter_policy,
    "mixup": mixup_policy,
    "wall_pressure": wall_pressure_policy,
    "adaptive": adaptive_policy,
    "launcher_spammer": launcher_spammer_policy,
    "panic_blocker": panic_blocker_policy,
    "backdash_bait": backdash_bait_policy,
    "jab_spammer": jab_spammer_policy,
    "zoner": zoner_policy,
    "hit_confirm": hit_confirm_policy,
    "throw_looper": throw_looper_policy,
    "safe_poke": safe_poke_policy,
    "yolo_offense": yolo_offense_policy,
    "flowchart": flowchart_policy,
    "spacing_ghost": spacing_ghost_policy,
    "sparring_partner": sparring_partner_policy,
    "ace": ace_policy,
}
