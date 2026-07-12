from __future__ import annotations

from collections.abc import Callable

from t8_agent.sim.moves import HitLevel, JUN_MOVES
from t8_agent.sim.tekken_lite import FighterRuntime, SimAction, TekkenLiteEnv

ScriptedPolicy = Callable[[TekkenLiteEnv, int], SimAction]

ATTACKS = [
    SimAction.JAB,
    SimAction.DF1,
    SimAction.F2,
    SimAction.DB3,
    SimAction.HOPKICK,
    SimAction.THROW,
]


def _own_and_opponent(env: TekkenLiteEnv, player: int) -> tuple[FighterRuntime, FighterRuntime]:
    return (env.state.p1, env.state.p2) if player == 1 else (env.state.p2, env.state.p1)


def _forward_sign(player: int) -> float:
    return 1.0 if player == 1 else -1.0


def _back_wall_distance(env: TekkenLiteEnv, x: float, forward: float) -> float:
    """Distance from x to the wall behind a fighter facing `forward`."""
    return (x + env.config.stage_half_width) if forward == 1.0 else (env.config.stage_half_width - x)


def _incoming_defense(opponent: FighterRuntime, distance: float) -> SimAction | None:
    """Returns the correct block for a genuinely live threat, or None if
    there's nothing to fear right now (opponent isn't attacking, is out of
    range, or is already past their active frames and harmless)."""
    if opponent.move_key is None:
        return None
    move = JUN_MOVES[opponent.move_key]
    if distance > move.range + 0.15:
        return None
    if opponent.move_frame > move.startup + move.active:
        return None
    if move.hit_level == HitLevel.LOW:
        return SimAction.BLOCK_LOW
    if move.hit_level in (HitLevel.HIGH, HitLevel.MID):
        return SimAction.BLOCK_HIGH
    return None


def _opponent_is_recovering(opponent: FighterRuntime) -> bool:
    """True once the opponent's active hitbox frames are over and they're
    just sitting in recovery: a free window to attack into."""
    if opponent.move_key is None:
        return False
    move = JUN_MOVES[opponent.move_key]
    return opponent.move_frame > move.startup + move.active


def random_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Uniform random legal-ish action. Baseline chaos opponent."""
    _ = player
    return env.sample_action()


def poke_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Closes to range and throws out safe pokes, blocks occasionally."""
    own = env.state.p1 if player == 1 else env.state.p2
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.05:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.18:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])


def turtle_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Defensive: blocks reactively to opponent movement, rarely initiates."""
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
    """Sits just outside range, waits for an opening, punishes hard."""
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
    """Closes distance relentlessly and keeps attacking."""
    own = env.state.p1 if player == 1 else env.state.p2
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.82:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.08:
        return SimAction.BLOCK_HIGH
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])



def keepout_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Controls long range and punishes approaches."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if _opponent_is_recovering(opponent) and distance < 1.25:
        return env.rng.choice([SimAction.F2, SimAction.DF1, SimAction.HOPKICK])
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance < 0.68:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.SIDESTEP_LEFT, SimAction.SIDESTEP_RIGHT])
    if distance > 1.25:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.F2, SimAction.DF1, SimAction.WALK_BACK])


def frame_trap_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Uses fast pressure and mixed timing at close range."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if _opponent_is_recovering(opponent) and distance < 1.25:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.95:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


def anti_throw_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Avoids point-blank throw range and challenges grab attempts."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if _opponent_is_recovering(opponent) and distance < 1.25:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.DF1])
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance < 0.58:
        return env.rng.choice([SimAction.WALK_BACK, SimAction.DASH_BACK, SimAction.JUMP, SimAction.HOPKICK])
    if distance > 1.15:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.DF1, SimAction.DB3, SimAction.F2, SimAction.BLOCK_HIGH])


def low_spammer_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Leans heavily on the low poke, mixes in throws to beat crouch blocking."""
    own, _opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.0:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.12:
        return SimAction.BLOCK_LOW
    return env.rng.choice([SimAction.DB3, SimAction.DB3, SimAction.JAB, SimAction.THROW])


def footsies_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Hovers at the edge of mid-poke range, pokes and retreats (hit-and-run)."""
    own, _opponent = _own_and_opponent(env, player)
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
    """Beelines for throw range and prioritizes throws over strikes up close."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.9:
        return SimAction.WALK_FORWARD
    if opponent.move_key is not None:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice([SimAction.THROW, SimAction.THROW, SimAction.DF1, SimAction.JAB])


def counter_hitter_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Patient whiff-punisher that leans on the launcher for max reward."""
    own, opponent = _own_and_opponent(env, player)
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
    """Unpredictable blend of high/low/throw with weighted random selection."""
    own, _opponent = _own_and_opponent(env, player)
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
    """Tracks the opponent's distance to their back wall and leans on big
    damage once they're cornered, otherwise plays a normal poke game."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    forward = 1.0 if player == 1 else -1.0
    opponent_back_wall = (
        (env.config.stage_half_width - opponent.x) if forward == 1.0 else (opponent.x + env.config.stage_half_width)
    )
    if env.state.distance > 0.95:
        return SimAction.WALK_FORWARD
    if opponent_back_wall < 0.6 and env.rng.random() < 0.5:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3])


def adaptive_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Plays safe while ahead on health, gambles for big damage while behind."""
    own, opponent = _own_and_opponent(env, player)
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
    """Goes for the launcher on nearly every opening. High risk, high reward,
    and a good dummy for practicing whiff punishment against."""
    own, _opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.85:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.75:
        return SimAction.HOPKICK
    return env.rng.choice([SimAction.JAB, SimAction.BLOCK_HIGH])


def panic_blocker_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Blocks the moment the opponent looks active, only pokes in small windows."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if opponent.move_key is not None or env.rng.random() < 0.55:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    if env.state.distance > 1.2:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DB3])


def backdash_bait_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Walks back to bait the opponent into whiffing, then punishes forward pressure."""
    own, opponent = _own_and_opponent(env, player)
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
    """Pure tempo control: spams the fast, low-risk jab and rarely commits
    to anything bigger. Good for testing how well a policy handles constant
    low-damage pressure without overreacting."""
    own, _opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.85:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.85:
        return SimAction.JAB
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.THROW])


def zoner_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Keeps the opponent at long range and only engages with f2, the
    longest-range move. Backs off the instant the opponent closes in rather
    than trading up close."""
    own, opponent = _own_and_opponent(env, player)
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
    """Reads the outcome of its own last poke before committing again: if the
    opponent is still in hitstun it follows up for real damage, if they're in
    blockstun (meaning the poke got blocked) it backs off to a safer option
    instead of getting punished."""
    own, opponent = _own_and_opponent(env, player)
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
    """A throw specialist: only closes into point-blank range and loops
    throws there, blocking everything else while walking in."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if distance > 0.55:
        if opponent.move_key is not None:
            return SimAction.BLOCK_HIGH
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.THROW, SimAction.THROW, SimAction.THROW, SimAction.JAB])


def safe_poke_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Never over-commits: only ever throws out jab or df1, never goes for
    the riskier hopkick or throw. Low variance, hard to punish, good baseline
    for testing patient offense."""
    own, _opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 1.0:
        return SimAction.WALK_FORWARD
    if env.rng.random() < 0.3:
        return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW])
    return env.rng.choice([SimAction.JAB, SimAction.DF1])


def yolo_offense_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Never blocks, always swings for the highest damage options available.
    A punching bag that's great for training a policy's punish game and
    defense, since it never plays it safe."""
    own, _opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    if env.state.distance > 0.9:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.THROW, SimAction.DF1])


def flowchart_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Fully deterministic and non-reactive: cycles through a fixed action
    sequence based on frame count regardless of game state. Not a strong
    opponent, but a stable, repeatable benchmark for regression testing since
    it contains no randomness and doesn't react to the opponent at all."""
    own, _opponent = _own_and_opponent(env, player)
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
    """Erratic, unpredictable footwork with no clear pattern: randomly
    advances, retreats, or holds ground. Stresses a policy's ability to
    control neutral against spacing it can't easily read, rather than
    testing its offense."""
    own, _opponent = _own_and_opponent(env, player)
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
    """A strong, well-rounded curriculum step below ace: the same overall
    game plan (spacing, whiff punishing, hit confirms, move-aware blocking)
    but with imperfect execution, so a policy can build up to facing ace
    instead of hitting a wall immediately."""
    own, opponent = _own_and_opponent(env, player)
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

    return env.rng.choice(
        [SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW, SimAction.BLOCK_HIGH]
    )


def ace_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """The strongest scripted opponent in the roster. Combines whiff
    punishing, hit-confirmed follow-ups, wall awareness, health-adaptive
    aggression, and move-aware blocking (reads the opponent's active move
    key and blocks the correct height) into one cohesive game plan. Its one
    real weakness is that nothing in this sim can block a throw, so a
    trained policy has to find that gap. Meant to be a genuine final exam,
    not just another archetype to mix into a training pool."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    forward = 1.0 if player == 1 else -1.0

    # React to a telegraphed attack: block the correct height every time.
    if opponent.move_key is not None and distance < 1.3:
        move = JUN_MOVES[opponent.move_key]
        if move.hit_level == HitLevel.LOW:
            return SimAction.BLOCK_LOW
        if move.hit_level in (HitLevel.HIGH, HitLevel.MID):
            return SimAction.BLOCK_HIGH
        # It's a throw: can't block it, try to be out of range next time.
        return SimAction.WALK_BACK

    # Capitalize immediately if the last poke landed clean.
    if opponent.hitstun > 0 and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])

    # If it got blocked, don't get punished back; reset with a safer option.
    if opponent.blockstun > 0 and distance < 1.0:
        return env.rng.choice([SimAction.THROW, SimAction.WALK_BACK])

    opponent_back_wall = (
        (env.config.stage_half_width - opponent.x) if forward == 1.0 else (opponent.x + env.config.stage_half_width)
    )
    lead = own.health - opponent.health

    if distance > 1.2:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return env.rng.choice([SimAction.THROW, SimAction.JAB, SimAction.WALK_BACK])
    if opponent_back_wall < 0.6:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    if lead < -15:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.THROW])    if lead > 15:
        return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.BLOCK_HIGH])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])



def stone_wall_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Blocks live threats correctly and answers with compact safe pokes."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 1.05:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW, SimAction.JAB, SimAction.DF1])


def iron_curtain_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Defends first and punishes recovery instead of swinging freely."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if _opponent_is_recovering(opponent) and distance < 1.25:
        return env.rng.choice([SimAction.DF1, SimAction.F2])
    if distance > 1.2:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW, SimAction.WALK_BACK])


def untouchable_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Uses spacing and reactive guard to make reckless approaches whiff."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    if distance < 1.0:
        return SimAction.WALK_BACK
    if distance > 1.3:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.NEUTRAL, SimAction.WALK_BACK, SimAction.F2])


def the_wall_ii_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Advanced turtle that mixes low parries and throw checks into defense."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense == SimAction.BLOCK_LOW and env.rng.random() < 0.35:
        return SimAction.LOW_PARRY
    if defense is not None:
        return defense
    if distance > 1.1:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.BLOCK_HIGH, SimAction.BLOCK_LOW, SimAction.DF1, SimAction.THROW])


def patience_incarnate_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Waits for a verified recovery window before taking meaningful risks."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK, SimAction.DF1])
    if distance > 1.25:
        return SimAction.WALK_FORWARD
    if distance < 0.75:
        return SimAction.WALK_BACK
    return SimAction.NEUTRAL


def no_free_lunch_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Defends real attacks and immediately taxes unsafe recovery."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2])
    if distance > 1.05:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.BLOCK_HIGH])


def jab_engine_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Maintains fast jab pressure while reacting to incoming attacks."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.9:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.JAB, SimAction.JAB, SimAction.DF1, SimAction.THROW])


def pressure_wall_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Combines correct defense with persistent close-range pressure."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.85:
        return SimAction.WALK_FORWARD
    if opponent.hitstun > 0:
        return env.rng.choice([SimAction.DF1, SimAction.F2])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


def suffocator_prime_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Closes distance aggressively and attacks the instant it's in range,
    with near-zero downtime between actions."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.95:
        return SimAction.WALK_FORWARD
    if _opponent_is_recovering(opponent):
        return env.rng.choice([SimAction.DF1, SimAction.F2])
    return env.rng.choice([SimAction.JAB, SimAction.DB3, SimAction.THROW])


def death_by_inches_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Favors low-risk, low-damage pokes over commitment moves, grinding
    health down slowly while never leaving an opening of its own."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.9:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DB3])


def no_gaps_v2_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Never idles: always attacking, closing distance, or blocking. No
    neutral frames for the opponent to exploit."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.95:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


def hyper_aggro_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Always in your face, always ready to block: extreme close-range
    rushdown backed by correct defensive reads."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.7:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.THROW, SimAction.DB3])


########################################################################
# Tier: perfect mixups - punishes only ever blocking one height
########################################################################


def fifty_fifty_master_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Near-even weighted high/mid/low/throw mixup at close range, always
    with real damage behind it. Guessing wrong is expensive."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 1.0:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.DF1, SimAction.DB3, SimAction.THROW, SimAction.F2])


def guess_wrong_die_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Commits hard to hopkick/f2/throw mixups. Guess the wrong height and
    it's a huge chunk of health, every time."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.95:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.THROW])


def unreadable_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Pure entropy across the whole moveset with no discernible pattern,
    forcing the opponent to react to what's actually happening rather than
    predicting a habit."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 1.05:
        return SimAction.WALK_FORWARD
    return env.rng.choice(
        [SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW, SimAction.F2, SimAction.HOPKICK]
    )


def crossup_illusion_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Alternates deliberately between mid and low pokes at the same range
    and tempo, so they look identical right up until it's too late to
    block correctly."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 1.0:
        return SimAction.WALK_FORWARD
    return SimAction.DB3 if env.rng.random() < 0.5 else SimAction.DF1


def throw_or_die_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Alternates between safe mid pokes and throws right at throw range,
    punishing anyone who just holds block and never moves."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 0.55:
        return SimAction.WALK_FORWARD if distance > 1.0 else SimAction.DF1
    return SimAction.THROW if env.rng.random() < 0.55 else SimAction.JAB


def adaptive_mixup_v2_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Leans into heavier mixups when behind on health, plays a tighter
    patient game when ahead, always with correct defense underneath."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    lead = own.health - opponent.health
    if distance > 1.05:
        return SimAction.WALK_FORWARD
    if lead < -15:
        return env.rng.choice([SimAction.HOPKICK, SimAction.THROW, SimAction.F2])
    if lead > 15:
        return env.rng.choice([SimAction.JAB, SimAction.DF1])
    return env.rng.choice([SimAction.DF1, SimAction.DB3, SimAction.THROW])


########################################################################
# Tier: punish / whiff specialists - brutal on any overextension
########################################################################


def whiff_reaper_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Attacks with the biggest available option the instant the opponent
    is caught in recovery; otherwise defends and waits."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2])
    if distance > 1.2:
        return SimAction.WALK_FORWARD
    return SimAction.NEUTRAL


def launch_or_bust_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Near-exclusively goes for the launcher on any real opening. Patient
    to set up, brutal when it lands."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if _opponent_is_recovering(opponent) and distance < 0.9:
        return SimAction.HOPKICK
    if distance > 0.85:
        return SimAction.WALK_FORWARD
    if distance < 0.5:
        return SimAction.WALK_BACK
    return SimAction.NEUTRAL


def backfoot_sniper_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Plays a patient backward-spacing game, then instantly punishes any
    forward approach with f2 or the launcher."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance < 1.1:
        return SimAction.WALK_BACK
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return SimAction.F2
    if env.rng.random() < 0.15:
        return SimAction.F2
    return SimAction.NEUTRAL


def range_denier_v2_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Sits right at f2 range and denies any comfortable approach,
    punishing overextension instantly."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 1.25:
        return SimAction.WALK_FORWARD
    if distance < 1.05:
        return SimAction.WALK_BACK
    if env.rng.random() < 0.3:
        return SimAction.F2
    return SimAction.NEUTRAL


def second_wind_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Always follows up the instant a hit lands, maximizing damage from
    every single opening it gets."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if opponent.hitstun > 0 and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    if distance > 1.1:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.JAB, SimAction.DF1])


def the_closer_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Turtles hard while ahead on health, hunts aggressively for whiff
    punishes the moment it falls behind."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    lead = own.health - opponent.health
    if _opponent_is_recovering(opponent) and distance < 1.2:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    if lead >= 0:
        return SimAction.WALK_BACK if distance < 1.2 else SimAction.NEUTRAL
    if distance > 1.1:
        return SimAction.WALK_FORWARD
    return env.rng.choice([SimAction.DF1, SimAction.THROW])


########################################################################
# Tier: positional / wall specialists
########################################################################


def corner_king_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Actively walks the opponent toward their own back wall, then unloads
    maximum damage the moment they're cornered."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    back_wall = _back_wall_distance(env, opponent.x, -_forward_sign(player))
    if distance > 1.0:
        return SimAction.WALK_FORWARD
    if back_wall < 0.6:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2])
    return env.rng.choice([SimAction.JAB, SimAction.DF1])


def spacing_general_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Controls neutral tightly: never lets the opponent settle into a
    comfortable range, while defending correctly against anything real."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    if distance > 1.15:
        return SimAction.WALK_FORWARD
    if distance < 0.8:
        return SimAction.WALK_BACK
    if env.rng.random() < 0.3:
        return env.rng.choice([SimAction.DF1, SimAction.JAB])
    return SimAction.NEUTRAL


def anchor_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """Holds a stable mid-range position and pokes from it, while actively
    avoiding ever getting pinned against its own back wall."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense
    own_back_wall = _back_wall_distance(env, own.x, _forward_sign(player))
    if own_back_wall < 0.7:
        return SimAction.WALK_FORWARD
    if distance > 1.1:
        return SimAction.WALK_FORWARD
    if distance < 0.7:
        return SimAction.WALK_BACK
    return env.rng.choice([SimAction.JAB, SimAction.DB3])


########################################################################
# Tier: gauntlet finishers - the hardest bots in the roster
########################################################################


def the_gatekeeper_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """A more precise version of ace: perfect defense, hit-confirm
    follow-ups, whiff punishing during the opponent's actual recovery
    window (not just anywhere they're mid-move), throw mixups at point
    blank, and wall + health awareness."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense

    if opponent.hitstun > 0 and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return env.rng.choice([SimAction.F2, SimAction.DF1, SimAction.HOPKICK])

    back_wall = _back_wall_distance(env, opponent.x, -_forward_sign(player))
    lead = own.health - opponent.health

    if distance > 1.15:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return env.rng.choice([SimAction.THROW, SimAction.JAB])
    if back_wall < 0.6:
        return env.rng.choice([SimAction.F2, SimAction.HOPKICK])
    if lead < -15:
        return env.rng.choice([SimAction.HOPKICK, SimAction.THROW])
    if lead > 15:
        return env.rng.choice([SimAction.JAB, SimAction.DF1])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


def nightmare_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """The hardest bot in the roster. Everything the_gatekeeper does, but
    faster to close distance, more willing to gamble on big damage, and
    less forgiving of any hesitation."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    defense = _incoming_defense(opponent, distance)
    if defense is not None:
        return defense

    if opponent.hitstun > 0 and distance < 1.3:
        return SimAction.HOPKICK if env.rng.random() < 0.6 else SimAction.F2
    if _opponent_is_recovering(opponent) and distance < 1.3:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2, SimAction.DF1])

    back_wall = _back_wall_distance(env, opponent.x, -_forward_sign(player))

    if distance > 1.0:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return env.rng.choice([SimAction.THROW, SimAction.THROW, SimAction.JAB])
    if back_wall < 0.7:
        return env.rng.choice([SimAction.HOPKICK, SimAction.F2])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW, SimAction.F2])


def the_apprentice_policy(env: TekkenLiteEnv, player: int) -> SimAction:
    """A stepping stone between ace/sparring_partner and
    the_gatekeeper/nightmare: the same refined fundamentals, but with
    imperfect execution so a policy isn't thrown straight at the hardest
    bots in the roster."""
    own, opponent = _own_and_opponent(env, player)
    if own.busy:
        return SimAction.NEUTRAL
    distance = env.state.distance
    if env.rng.random() < 0.85:
        defense = _incoming_defense(opponent, distance)
        if defense is not None:
            return defense

    if opponent.hitstun > 0 and distance < 1.3 and env.rng.random() < 0.7:
        return env.rng.choice([SimAction.F2, SimAction.DF1])
    if _opponent_is_recovering(opponent) and distance < 1.2 and env.rng.random() < 0.7:
        return env.rng.choice([SimAction.DF1, SimAction.F2])

    if distance > 1.1:
        return SimAction.WALK_FORWARD
    if distance < 0.55:
        return env.rng.choice([SimAction.THROW, SimAction.JAB, SimAction.WALK_BACK])
    return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])


SCRIPTED_BASE_POLICIES: dict[str, ScriptedPolicy] = {
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
    "stone_wall": stone_wall_policy,
    "iron_curtain": iron_curtain_policy,
    "untouchable": untouchable_policy,
    "the_wall_ii": the_wall_ii_policy,
    "patience_incarnate": patience_incarnate_policy,
    "no_free_lunch": no_free_lunch_policy,
    "jab_engine": jab_engine_policy,
    "pressure_wall": pressure_wall_policy,
    "suffocator_prime": suffocator_prime_policy,
    "death_by_inches": death_by_inches_policy,
    "no_gaps_v2": no_gaps_v2_policy,
    "hyper_aggro": hyper_aggro_policy,
    "fifty_fifty_master": fifty_fifty_master_policy,
    "guess_wrong_die": guess_wrong_die_policy,
    "unreadable": unreadable_policy,
    "crossup_illusion": crossup_illusion_policy,
    "throw_or_die": throw_or_die_policy,
    "adaptive_mixup_v2": adaptive_mixup_v2_policy,
    "whiff_reaper": whiff_reaper_policy,
    "launch_or_bust": launch_or_bust_policy,
    "backfoot_sniper": backfoot_sniper_policy,
    "range_denier_v2": range_denier_v2_policy,
    "second_wind": second_wind_policy,
    "the_closer": the_closer_policy,
    "corner_king": corner_king_policy,
    "spacing_general": spacing_general_policy,
    "anchor": anchor_policy,
    "the_gatekeeper": the_gatekeeper_policy,
    "nightmare": nightmare_policy,
    "the_apprentice": the_apprentice_policy,
}

SCRIPTED_DIFFICULTY_TIERS: dict[int, list[str]] = {
    1: [
        "random", "poke", "rushdown", "turtle", "keepout",
        "frame_trap", "anti_throw", "low_spammer", "footsies", "grappler",
        "mixup", "launcher_spammer", "panic_blocker", "jab_spammer",
        "safe_poke", "yolo_offense", "flowchart", "spacing_ghost",
    ],
    2: [
        "whiff_punish", "counter_hitter", "wall_pressure", "adaptive", "backdash_bait",
        "zoner", "hit_confirm", "throw_looper", "sparring_partner", "stone_wall",
        "iron_curtain", "jab_engine", "pressure_wall", "suffocator_prime",
        "death_by_inches", "no_gaps_v2", "hyper_aggro", "fifty_fifty_master",
        "unreadable", "crossup_illusion", "throw_or_die", "spacing_general",
        "anchor", "the_apprentice",
    ],
    3: [
        "ace", "untouchable", "the_wall_ii", "patience_incarnate",
        "no_free_lunch", "guess_wrong_die", "adaptive_mixup_v2", "whiff_reaper",
        "launch_or_bust", "backfoot_sniper", "range_denier_v2", "second_wind",
        "the_closer", "corner_king", "the_gatekeeper", "nightmare",
    ],
}


def _harden_policy(policy: ScriptedPolicy, *, reaction_rate: float, reaction_frames: int) -> ScriptedPolicy:
    def hardened(env: TekkenLiteEnv, player: int) -> SimAction:
        own, opponent = _own_and_opponent(env, player)
        if own.busy:
            return SimAction.NEUTRAL
        distance = env.state.distance
        punish_rate = min(0.98, reaction_rate + 0.18)
        if _opponent_is_recovering(opponent) and distance < 1.3 and env.rng.random() < punish_rate:
            return env.rng.choice([SimAction.DF1, SimAction.F2, SimAction.HOPKICK])
        if opponent.hitstun > 0 and distance < 1.3 and env.rng.random() < punish_rate:
            return env.rng.choice([SimAction.DF1, SimAction.F2, SimAction.HOPKICK])
        if opponent.blockstun > 0 and distance < 1.0 and env.rng.random() < reaction_rate:
            return env.rng.choice([SimAction.DB3, SimAction.THROW, SimAction.DF1])
        if opponent.move_key is not None and opponent.move_frame >= reaction_frames and env.rng.random() < reaction_rate:
            defense = _incoming_defense(opponent, distance)
            if defense is not None:
                return defense
        action = policy(env, player)
        if action == SimAction.NEUTRAL and distance < 1.1 and env.rng.random() < reaction_rate * 0.35:
            return env.rng.choice([SimAction.JAB, SimAction.DF1, SimAction.DB3, SimAction.THROW])
        return action

    hardened.__name__ = f"hardened_{policy.__name__}"
    return hardened


_HARDEST_BOTS = {
    "ace", "the_gatekeeper", "nightmare", "untouchable", "the_wall_ii",
    "patience_incarnate", "no_free_lunch",
}
_INTERMEDIATE_BOTS = set(SCRIPTED_DIFFICULTY_TIERS[2])

SCRIPTED_POLICIES: dict[str, ScriptedPolicy] = {
    name: _harden_policy(
        policy,
        reaction_rate=0.95 if name in _HARDEST_BOTS else 0.85 if name in _INTERMEDIATE_BOTS else 0.68,
        reaction_frames=1 if name in _HARDEST_BOTS else 2 if name in _INTERMEDIATE_BOTS else 3,
    )
    for name, policy in SCRIPTED_BASE_POLICIES.items()
}

DEFAULT_SCRIPTED_OPPONENTS = list(SCRIPTED_BASE_POLICIES)
