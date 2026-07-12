from __future__ import annotations

import random
from dataclasses import dataclass, replace
from enum import Enum
from typing import Mapping

from t8_agent.core.types import GameState, PlayerState, StepResult
from t8_agent.sim.moves import HitLevel, JUN_MOVES, MoveSpec


class SimAction(str, Enum):
    NEUTRAL = "neutral"
    WALK_FORWARD = "walk_forward"
    WALK_BACK = "walk_back"
    DASH_FORWARD = "dash_forward"
    DASH_BACK = "dash_back"
    CROUCH = "crouch"
    STAND = "stand"
    JUMP = "jump"
    SIDESTEP_LEFT = "sidestep_left"
    SIDESTEP_RIGHT = "sidestep_right"
    SIDEWALK_LEFT = "sidewalk_left"
    SIDEWALK_RIGHT = "sidewalk_right"
    BLOCK_HIGH = "block_high"
    BLOCK_LOW = "block_low"
    LOW_PARRY = "low_parry"
    THROW_BREAK_1 = "throw_break_1"
    THROW_BREAK_2 = "throw_break_2"
    THROW_BREAK_1_2 = "throw_break_1p2"
    JAB = "jab"
    DF1 = "df1"
    F2 = "f2"
    DB3 = "db3"
    HOPKICK = "hopkick"
    THROW = "throw"
    HEAT_BURST = "heat_burst"
    HEAT_SMASH = "heat_smash"
    RAGE_ART = "rage_art"


@dataclass(frozen=True)
class SimConfig:
    max_health: float = 180.0
    stage_half_width: float = 3.6
    decision_frames: int = 4
    max_frames: int = 60 * 60
    walk_speed: float = 0.025
    dash_speed: float = 0.07
    jump_frames: int = 32
    throw_break_frames: int = 9
    sidestep_speed: float = 0.045
    sidewalk_speed: float = 0.075
    lateral_return_speed: float = 0.012
    sidestep_evasion_width: float = 0.33
    body_radius: float = 0.22
    stall_distance: float = 1.5
    max_stall_frames: int = 360
    no_action_timeout_frames: int = 300
    no_action_timeout_penalty: float = 320.0
    round_win_base_reward: float = 40.0
    round_win_health_reward: float = 60.0
    round_loss_penalty: float = 160.0
    timeout_ahead_penalty: float = 60.0
    timeout_even_penalty: float = 120.0
    timeout_behind_penalty: float = 240.0
    stalemate_penalty: float = 260.0
    damage_dealt_scale: float = 0.65
    damage_taken_scale: float = 0.85
    throw_reward_scale: float = 0.70
    successful_throw_reward: float = 1.0
    throw_break_reward: float = 0.35
    throw_broken_penalty: float = -0.20
    whiff_punish_reward_scale: float = 0.15
    block_reward: float = 0.12
    blocked_attack_penalty: float = -0.08
    idle_penalty: float = -0.02
    far_spacing_penalty: float = 0.0
    lateral_passivity_penalty: float = -0.04
    wall_camping_penalty: float = -0.05
    late_round_passivity_penalty: float = -0.03
    whiff_penalty: float = -0.18


@dataclass(frozen=True)
class FighterRuntime:
    health: float
    x: float
    y: float = 0.0
    guard: HitLevel | None = None
    move_key: str | None = None
    move_frame: int = 0
    has_hit: bool = False
    hitstun: int = 0
    blockstun: int = 0
    airborne: int = 0
    throw_break_active: int = 0
    launches_taken: int = 0
    whiffs: int = 0

    @property
    def busy(self) -> bool:
        return self.move_key is not None or self.hitstun > 0 or self.blockstun > 0 or self.airborne > 0


@dataclass(frozen=True)
class SimState:
    p1: FighterRuntime
    p2: FighterRuntime
    frame: int
    stall_frames: int = 0
    no_action_frames: int = 0
    round_over: bool = False
    winner: int | None = None

    @property
    def distance(self) -> float:
        return abs(self.p2.x - self.p1.x)


@dataclass(frozen=True)
class SimStepResult:
    state: SimState
    observation: GameState
    reward_p1: float
    reward_p2: float
    terminated: bool
    truncated: bool
    info: Mapping[str, float | int | bool | str]


@dataclass(frozen=True)
class AttackCheck:
    in_range: bool
    blocked: bool
    damage: float
    throw_broken: bool = False


class TekkenLiteEnv:
    """Fast Jun-focused surrogate simulator for self-play experiments.

    The model is intentionally compact: it is not trying to clone Tekken 8.
    It captures the learning-relevant fundamentals first: spacing, walls,
    frame commitment, whiffs, blocking, damage, and round outcomes.
    """

    def __init__(self, config: SimConfig | None = None, seed: int | None = None) -> None:
        self.config = config or SimConfig()
        self.rng = random.Random(seed)
        self.state = self._initial_state()

    def reset(self, seed: int | None = None) -> GameState:
        if seed is not None:
            self.rng.seed(seed)
        self.state = self._initial_state()
        return self._to_observation(self.state)

    def step(self, p1_action: SimAction, p2_action: SimAction) -> SimStepResult:
        previous = self.state
        damage_to_p1 = 0.0
        damage_to_p2 = 0.0
        p1_whiffs = 0
        p2_whiffs = 0
        p1_blocks = 0
        p2_blocks = 0
        p1_throw_damage = 0.0
        p2_throw_damage = 0.0
        p1_throw_breaks = 0
        p2_throw_breaks = 0
        p1_whiff_punish_bonus = 0.0
        p2_whiff_punish_bonus = 0.0

        state = self._start_actions(previous, p1_action, p2_action)
        for _ in range(self.config.decision_frames):
            state, frame_info = self._advance_frame(state, p1_action, p2_action)
            damage_to_p1 += frame_info["damage_to_p1"]
            damage_to_p2 += frame_info["damage_to_p2"]
            p1_throw_damage += frame_info["p1_throw_damage"]
            p2_throw_damage += frame_info["p2_throw_damage"]
            p1_throw_breaks += int(frame_info["p1_throw_break"])
            p2_throw_breaks += int(frame_info["p2_throw_break"])
            p1_whiff_punish_bonus += frame_info["p1_whiff_punish_bonus"]
            p2_whiff_punish_bonus += frame_info["p2_whiff_punish_bonus"]
            p1_whiffs += int(frame_info["p1_whiff"])
            p2_whiffs += int(frame_info["p2_whiff"])
            p1_blocks += int(frame_info["p1_block"])
            p2_blocks += int(frame_info["p2_block"])
            if state.round_over:
                break

        no_action = self._is_no_action_window(previous, state, p1_action, p2_action, damage_to_p1, damage_to_p2)
        no_action_frames = state.no_action_frames + self.config.decision_frames if no_action else 0
        if no_action_frames >= self.config.no_action_timeout_frames and not state.round_over:
            state = replace(state, no_action_frames=no_action_frames, round_over=True, winner=None)
        else:
            state = replace(state, no_action_frames=no_action_frames)
        self.state = state
        truncated = state.frame >= self.config.max_frames and not state.round_over
        timed_out = state.round_over and state.frame >= self.config.max_frames and state.p1.health > 0.0 and state.p2.health > 0.0
        stalemate = state.round_over and state.winner is None
        no_action_timeout = state.round_over and state.no_action_frames >= self.config.no_action_timeout_frames
        reward_damage_to_p2 = (damage_to_p2 - p1_throw_damage) + self.config.throw_reward_scale * p1_throw_damage
        reward_damage_to_p1 = (damage_to_p1 - p2_throw_damage) + self.config.throw_reward_scale * p2_throw_damage
        reward_p1 = self.config.damage_dealt_scale * reward_damage_to_p2 - self.config.damage_taken_scale * damage_to_p1
        reward_p2 = self.config.damage_dealt_scale * reward_damage_to_p1 - self.config.damage_taken_scale * damage_to_p2
        if p1_throw_damage > 0.0:
            reward_p1 += self.config.successful_throw_reward
        if p2_throw_damage > 0.0:
            reward_p2 += self.config.successful_throw_reward
        reward_p1 += p1_whiff_punish_bonus
        reward_p2 += p2_whiff_punish_bonus

        if state.winner == 1 and not timed_out:
            win_reward = self._scaled_win_reward(state.p1.health)
            reward_p1 += win_reward
            reward_p2 -= win_reward
            reward_p2 -= self.config.round_loss_penalty
        elif state.winner == 2 and not timed_out:
            win_reward = self._scaled_win_reward(state.p2.health)
            reward_p1 -= win_reward
            reward_p1 -= self.config.round_loss_penalty
            reward_p2 += win_reward
        if state.round_over:
            health_margin = (state.p1.health - state.p2.health) / self.config.max_health
            if timed_out:
                reward_p1 -= self._scaled_timeout_penalty(health_margin)
                reward_p2 -= self._scaled_timeout_penalty(-health_margin)
            if stalemate:
                reward_p1 -= self.config.stalemate_penalty
                reward_p2 -= self.config.stalemate_penalty
            if no_action_timeout:
                reward_p1 -= self.config.no_action_timeout_penalty
                reward_p2 -= self.config.no_action_timeout_penalty

        if p1_action == SimAction.NEUTRAL:
            reward_p1 += self.config.idle_penalty
        if p2_action == SimAction.NEUTRAL:
            reward_p2 += self.config.idle_penalty
        lateral_actions = {
            SimAction.SIDESTEP_LEFT,
            SimAction.SIDESTEP_RIGHT,
            SimAction.SIDEWALK_LEFT,
            SimAction.SIDEWALK_RIGHT,
        }
        no_contact = damage_to_p1 == 0.0 and damage_to_p2 == 0.0 and p1_blocks == 0 and p2_blocks == 0
        if no_contact and p1_action in lateral_actions and p2_whiffs == 0:
            reward_p1 += self.config.lateral_passivity_penalty
        if no_contact and p2_action in lateral_actions and p1_whiffs == 0:
            reward_p2 += self.config.lateral_passivity_penalty
        if state.distance > 1.5:
            if p1_action not in {SimAction.WALK_FORWARD, SimAction.DASH_FORWARD}:
                reward_p1 += self.config.far_spacing_penalty
            if p2_action not in {SimAction.WALK_FORWARD, SimAction.DASH_FORWARD}:
                reward_p2 += self.config.far_spacing_penalty
        own_wall = self.config.stage_half_width - 0.35
        if state.p1.x < -own_wall and p1_action not in {SimAction.WALK_FORWARD, SimAction.DASH_FORWARD}:
            reward_p1 += self.config.wall_camping_penalty
        if state.p2.x > own_wall and p2_action not in {SimAction.WALK_FORWARD, SimAction.DASH_FORWARD}:
            reward_p2 += self.config.wall_camping_penalty
        if state.frame > self.config.max_frames * 0.65 and damage_to_p1 == 0.0 and damage_to_p2 == 0.0:
            if p1_action in {SimAction.NEUTRAL, SimAction.WALK_BACK, SimAction.DASH_BACK}:
                reward_p1 += self.config.late_round_passivity_penalty
            if p2_action in {SimAction.NEUTRAL, SimAction.WALK_BACK, SimAction.DASH_BACK}:
                reward_p2 += self.config.late_round_passivity_penalty
        reward_p1 += self.config.block_reward * p1_blocks
        reward_p2 += self.config.block_reward * p2_blocks
        reward_p1 += self.config.blocked_attack_penalty * p2_blocks
        reward_p2 += self.config.blocked_attack_penalty * p1_blocks
        reward_p1 += self.config.throw_break_reward * p1_throw_breaks
        reward_p2 += self.config.throw_break_reward * p2_throw_breaks
        reward_p1 += self.config.throw_broken_penalty * p2_throw_breaks
        reward_p2 += self.config.throw_broken_penalty * p1_throw_breaks
        reward_p1 += self.config.whiff_penalty * p1_whiffs
        reward_p2 += self.config.whiff_penalty * p2_whiffs

        observation = self._to_observation(state)
        return SimStepResult(
            state=state,
            observation=observation,
            reward_p1=reward_p1,
            reward_p2=reward_p2,
            terminated=state.round_over,
            truncated=truncated,
            info={
                "damage_to_p1": damage_to_p1,
                "damage_to_p2": damage_to_p2,
                "p1_throw_damage": p1_throw_damage,
                "p2_throw_damage": p2_throw_damage,
                "p1_throw_breaks": p1_throw_breaks,
                "p2_throw_breaks": p2_throw_breaks,
                "p1_whiff_punish_bonus": p1_whiff_punish_bonus,
                "p2_whiff_punish_bonus": p2_whiff_punish_bonus,
                "p1_whiffs": p1_whiffs,
                "p2_whiffs": p2_whiffs,
                "p1_blocks": p1_blocks,
                "p2_blocks": p2_blocks,
                "timed_out": timed_out,
                "stalemate": stalemate,
                "no_action_timeout": no_action_timeout,
                "stall_frames": state.stall_frames,
                "no_action_frames": state.no_action_frames,
                "frame": state.frame,
            },
        )

    def step_single_agent(self, p1_action: SimAction, opponent_action: SimAction) -> StepResult:
        result = self.step(p1_action, opponent_action)
        return StepResult(
            observation=result.observation,
            reward=result.reward_p1,
            terminated=result.terminated,
            truncated=result.truncated,
            info=result.info,
        )

    def legal_actions(self) -> list[SimAction]:
        return list(SimAction)

    def legal_action_mask(self, player: int):
        from t8_agent.sim.action_space import legal_action_mask

        return legal_action_mask(self.state, player)

    def sample_action(self) -> SimAction:
        return self.rng.choice(self.legal_actions())

    def _start_actions(self, state: SimState, p1_action: SimAction, p2_action: SimAction) -> SimState:
        p1 = self._start_action(state.p1, p1_action)
        p2 = self._start_action(state.p2, p2_action)
        return replace(state, p1=p1, p2=p2)

    def _start_action(self, fighter: FighterRuntime, action: SimAction) -> FighterRuntime:
        if fighter.busy:
            return fighter
        if action == SimAction.BLOCK_HIGH:
            return replace(fighter, guard=HitLevel.MID, throw_break_active=0)
        if action == SimAction.BLOCK_LOW:
            return replace(fighter, guard=HitLevel.LOW, throw_break_active=0)
        if action == SimAction.CROUCH:
            return replace(fighter, guard=HitLevel.LOW, throw_break_active=0)
        if action == SimAction.LOW_PARRY:
            return replace(fighter, guard=HitLevel.LOW, throw_break_active=0)
        if action == SimAction.STAND:
            return replace(fighter, guard=HitLevel.MID, throw_break_active=0)
        if action == SimAction.NEUTRAL:
            return replace(fighter, guard=HitLevel.MID, throw_break_active=0)
        if action == SimAction.JUMP:
            return replace(fighter, guard=None, airborne=self.config.jump_frames, throw_break_active=0)
        if action in {SimAction.THROW_BREAK_1, SimAction.THROW_BREAK_2, SimAction.THROW_BREAK_1_2}:
            return replace(fighter, guard=None, throw_break_active=self.config.throw_break_frames)
        move_key = _ACTION_TO_MOVE.get(action)
        if move_key is not None:
            return replace(fighter, guard=None, move_key=move_key, move_frame=0, has_hit=False, throw_break_active=0)
        return replace(fighter, guard=None, throw_break_active=0)

    def _advance_frame(
        self,
        state: SimState,
        p1_action: SimAction,
        p2_action: SimAction,
    ) -> tuple[SimState, dict[str, float | bool]]:
        p1 = self._tick_timers(state.p1)
        p2 = self._tick_timers(state.p2)

        if not p1.busy:
            p1 = self._move(p1, p1_action, forward=1)
        if not p2.busy:
            p2 = self._move(p2, p2_action, forward=-1)
        p1, p2 = self._separate_and_clip(p1, p2)

        damage_to_p1 = 0.0
        damage_to_p2 = 0.0
        p1_throw_damage = 0.0
        p2_throw_damage = 0.0
        p1_throw_break = False
        p2_throw_break = False
        p1_whiff_punish_bonus = 0.0
        p2_whiff_punish_bonus = 0.0
        p1_whiff = False
        p2_whiff = False
        p1_block = False
        p2_block = False

        p1_attack = self._active_move(p1)
        p2_attack = self._active_move(p2)

        p1_check = (
            self._check_attack(attacker=p1, defender=p2, move=p1_attack)
            if p1_attack is not None and not p1.has_hit
            else None
        )
        p2_check = (
            self._check_attack(attacker=p2, defender=p1, move=p2_attack)
            if p2_attack is not None and not p2.has_hit
            else None
        )

        p1_punishes_recovery = p1_check is not None and p1_check.damage > 0.0 and self._is_whiff_recovery(p2)
        p2_punishes_recovery = p2_check is not None and p2_check.damage > 0.0 and self._is_whiff_recovery(p1)

        if p1_attack is not None and p1_check is not None:
            p1, p2 = self._apply_attack(attacker=p1, defender=p2, move=p1_attack, check=p1_check, direction=1)
            damage_to_p2 += p1_check.damage
            if p1_attack.hit_level == HitLevel.THROW:
                p1_throw_damage += p1_check.damage
                p2_throw_break = p1_check.throw_broken
            if p1_punishes_recovery:
                p1_whiff_punish_bonus += self.config.whiff_punish_reward_scale * p1_check.damage
            p2_block = p1_check.blocked
        if p2_attack is not None and p2_check is not None:
            p2, p1 = self._apply_attack(attacker=p2, defender=p1, move=p2_attack, check=p2_check, direction=-1)
            damage_to_p1 += p2_check.damage
            if p2_attack.hit_level == HitLevel.THROW:
                p2_throw_damage += p2_check.damage
                p1_throw_break = p2_check.throw_broken
            if p2_punishes_recovery:
                p2_whiff_punish_bonus += self.config.whiff_punish_reward_scale * p2_check.damage
            p1_block = p2_check.blocked

        p1, p1_whiff = self._finish_move_if_needed(p1, p1_check is not None and not p1_check.in_range)
        p2, p2_whiff = self._finish_move_if_needed(p2, p2_check is not None and not p2_check.in_range)

        next_state = replace(state, p1=p1, p2=p2, frame=state.frame + 1)
        next_state = self._check_stalemate(next_state, damage_to_p1=damage_to_p1, damage_to_p2=damage_to_p2)
        next_state = self._check_round_over(next_state)
        return next_state, {
            "damage_to_p1": damage_to_p1,
            "damage_to_p2": damage_to_p2,
            "p1_throw_damage": p1_throw_damage,
            "p2_throw_damage": p2_throw_damage,
            "p1_throw_break": p1_throw_break,
            "p2_throw_break": p2_throw_break,
            "p1_whiff_punish_bonus": p1_whiff_punish_bonus,
            "p2_whiff_punish_bonus": p2_whiff_punish_bonus,
            "p1_whiff": p1_whiff,
            "p2_whiff": p2_whiff,
            "p1_block": p1_block,
            "p2_block": p2_block,
        }

    def _tick_timers(self, fighter: FighterRuntime) -> FighterRuntime:
        hitstun = max(0, fighter.hitstun - 1)
        blockstun = max(0, fighter.blockstun - 1)
        airborne = max(0, fighter.airborne - 1)
        throw_break_active = max(0, fighter.throw_break_active - 1)
        move_frame = fighter.move_frame
        if fighter.move_key is not None:
            move_frame += 1
        y = fighter.y
        if abs(y) <= self.config.lateral_return_speed:
            y = 0.0
        elif y > 0:
            y -= self.config.lateral_return_speed
        else:
            y += self.config.lateral_return_speed
        return replace(
            fighter,
            hitstun=hitstun,
            blockstun=blockstun,
            airborne=airborne,
            throw_break_active=throw_break_active,
            move_frame=move_frame,
            y=y,
        )

    def _move(self, fighter: FighterRuntime, action: SimAction, forward: int) -> FighterRuntime:
        x = fighter.x
        if action == SimAction.WALK_FORWARD:
            x += self.config.walk_speed * forward
        elif action == SimAction.WALK_BACK:
            x -= self.config.walk_speed * forward
        elif action == SimAction.DASH_FORWARD:
            x += self.config.dash_speed * forward
        elif action == SimAction.DASH_BACK:
            x -= self.config.dash_speed * forward
        y = fighter.y
        if action == SimAction.SIDESTEP_LEFT:
            y += self.config.sidestep_speed
        elif action == SimAction.SIDESTEP_RIGHT:
            y -= self.config.sidestep_speed
        elif action == SimAction.SIDEWALK_LEFT:
            y += self.config.sidewalk_speed
        elif action == SimAction.SIDEWALK_RIGHT:
            y -= self.config.sidewalk_speed
        return replace(fighter, x=x, y=max(-1.0, min(1.0, y)))

    def _scaled_win_reward(self, winner_health: float) -> float:
        health_ratio = max(0.0, min(1.0, winner_health / self.config.max_health))
        return self.config.round_win_base_reward + self.config.round_win_health_reward * health_ratio

    def _scaled_timeout_penalty(self, health_margin: float) -> float:
        margin = max(-1.0, min(1.0, health_margin))
        if margin > 0.0:
            return self.config.timeout_ahead_penalty + (
                self.config.timeout_even_penalty - self.config.timeout_ahead_penalty
            ) * (1.0 - margin)
        if margin < 0.0:
            return self.config.timeout_even_penalty + (
                self.config.timeout_behind_penalty - self.config.timeout_even_penalty
            ) * (-margin)
        return self.config.timeout_even_penalty

    def _separate_and_clip(self, p1: FighterRuntime, p2: FighterRuntime) -> tuple[FighterRuntime, FighterRuntime]:
        min_gap = self.config.body_radius * 2
        p1_x = max(-self.config.stage_half_width, min(self.config.stage_half_width, p1.x))
        p2_x = max(-self.config.stage_half_width, min(self.config.stage_half_width, p2.x))
        if p2_x - p1_x < min_gap:
            center = (p1_x + p2_x) / 2.0
            p1_x = center - min_gap / 2.0
            p2_x = center + min_gap / 2.0
        p1_x = max(-self.config.stage_half_width, min(self.config.stage_half_width, p1_x))
        p2_x = max(-self.config.stage_half_width, min(self.config.stage_half_width, p2_x))
        return replace(p1, x=p1_x), replace(p2, x=p2_x)

    def _active_move(self, fighter: FighterRuntime) -> MoveSpec | None:
        if fighter.move_key is None:
            return None
        move = JUN_MOVES[fighter.move_key]
        if fighter.move_frame > move.startup and fighter.move_frame <= move.startup + move.active:
            return move
        return None

    def _check_attack(self, attacker: FighterRuntime, defender: FighterRuntime, move: MoveSpec) -> AttackCheck:
        if abs(defender.x - attacker.x) > move.range:
            return AttackCheck(in_range=False, blocked=False, damage=0.0)
        if move.hit_level != HitLevel.THROW and abs(defender.y - attacker.y) > self.config.sidestep_evasion_width:
            return AttackCheck(in_range=False, blocked=False, damage=0.0)
        if defender.airborne > 0 and move.hit_level in {HitLevel.LOW, HitLevel.THROW}:
            return AttackCheck(in_range=False, blocked=False, damage=0.0)
        if move.hit_level == HitLevel.THROW and defender.throw_break_active > 0:
            return AttackCheck(in_range=True, blocked=True, damage=0.0, throw_broken=True)

        blocked = self._is_blocked(defender.guard, move.hit_level)
        return AttackCheck(in_range=True, blocked=blocked, damage=0.0 if blocked else move.damage)

    @staticmethod
    def _is_whiff_recovery(fighter: FighterRuntime) -> bool:
        if fighter.move_key is None or fighter.has_hit:
            return False
        move = JUN_MOVES[fighter.move_key]
        return fighter.move_frame > move.startup + move.active

    def _apply_attack(
        self,
        attacker: FighterRuntime,
        defender: FighterRuntime,
        move: MoveSpec,
        check: AttackCheck,
        direction: int,
    ) -> tuple[FighterRuntime, FighterRuntime]:
        if not check.in_range:
            return attacker, defender

        attacker = replace(attacker, has_hit=True)
        if check.blocked:
            defender = replace(defender, blockstun=max(defender.blockstun, move.blockstun))
            attacker, defender = self._apply_pushback(attacker, defender, direction, move.pushback * 0.6)
            return attacker, defender

        launches_taken = defender.launches_taken + int(move.launches)
        defender = replace(
            defender,
            health=max(0.0, defender.health - check.damage),
            hitstun=max(defender.hitstun, move.hitstun),
            blockstun=0,
            guard=None,
            move_key=None,
            move_frame=0,
            has_hit=False,
            launches_taken=launches_taken,
        )
        attacker, defender = self._apply_pushback(attacker, defender, direction, move.pushback)
        return attacker, defender

    @staticmethod
    def _is_blocked(guard: HitLevel | None, hit_level: HitLevel) -> bool:
        if hit_level == HitLevel.THROW:
            return False
        if guard == HitLevel.LOW:
            return hit_level == HitLevel.LOW
        if guard == HitLevel.MID:
            return hit_level in {HitLevel.HIGH, HitLevel.MID}
        return False

    def _apply_pushback(
        self,
        attacker: FighterRuntime,
        defender: FighterRuntime,
        direction: int,
        amount: float,
    ) -> tuple[FighterRuntime, FighterRuntime]:
        defender_x = defender.x + amount * direction
        defender_x = max(-self.config.stage_half_width, min(self.config.stage_half_width, defender_x))
        return attacker, replace(defender, x=defender_x)

    def _finish_move_if_needed(self, fighter: FighterRuntime, active_whiff: bool) -> tuple[FighterRuntime, bool]:
        if fighter.move_key is None:
            return fighter, False
        move = JUN_MOVES[fighter.move_key]
        whiffed = False
        if active_whiff and not fighter.has_hit and fighter.move_frame == move.startup + move.active:
            whiffed = True
            fighter = replace(fighter, whiffs=fighter.whiffs + 1)
        if fighter.move_frame >= move.total_frames:
            fighter = replace(fighter, move_key=None, move_frame=0, has_hit=False)
        return fighter, whiffed

    def _check_round_over(self, state: SimState) -> SimState:
        if state.round_over:
            return state
        winner = None
        if state.p1.health <= 0.0:
            winner = 2
        elif state.p2.health <= 0.0:
            winner = 1
        elif state.frame >= self.config.max_frames:
            winner = 1 if state.p1.health >= state.p2.health else 2
        return replace(state, round_over=winner is not None, winner=winner)

    def _check_stalemate(self, state: SimState, damage_to_p1: float, damage_to_p2: float) -> SimState:
        if state.round_over:
            return state
        no_damage = damage_to_p1 == 0.0 and damage_to_p2 == 0.0
        disengaged = state.distance > self.config.stall_distance
        no_active_attack = state.p1.move_key is None and state.p2.move_key is None
        stall_frames = state.stall_frames + 1 if no_damage and disengaged and no_active_attack else 0
        if stall_frames >= self.config.max_stall_frames:
            return replace(state, stall_frames=stall_frames, round_over=True, winner=None)
        return replace(state, stall_frames=stall_frames)

    @staticmethod
    def _is_no_action_window(
        previous: SimState,
        state: SimState,
        p1_action: SimAction,
        p2_action: SimAction,
        damage_to_p1: float,
        damage_to_p2: float,
    ) -> bool:
        passive = {SimAction.NEUTRAL, SimAction.STAND, SimAction.CROUCH}
        if damage_to_p1 > 0.0 or damage_to_p2 > 0.0:
            return False
        if p1_action not in passive or p2_action not in passive:
            return False
        return not (
            previous.p1.busy
            or previous.p2.busy
            or state.p1.busy
            or state.p2.busy
        )

    def _to_observation(self, state: SimState) -> GameState:
        round_timer = max(0.0, (self.config.max_frames - state.frame) / 60.0)
        return GameState(
            p1=PlayerState(
                health=state.p1.health,
                position_x=state.p1.x,
                position_y=state.p1.y,
                facing=1,
                move_id=state.p1.move_key,
                is_attacking=state.p1.move_key is not None,
                is_blocking=state.p1.guard is not None or state.p1.blockstun > 0,
                is_in_hitstun=state.p1.hitstun > 0,
            ),
            p2=PlayerState(
                health=state.p2.health,
                position_x=state.p2.x,
                position_y=state.p2.y,
                facing=-1,
                move_id=state.p2.move_key,
                is_attacking=state.p2.move_key is not None,
                is_blocking=state.p2.guard is not None or state.p2.blockstun > 0,
                is_in_hitstun=state.p2.hitstun > 0,
            ),
            round_timer=round_timer,
            round_over=state.round_over,
            winner=state.winner,
            raw={
                "frame": state.frame,
                "p1_blockstun": state.p1.blockstun,
                "p2_blockstun": state.p2.blockstun,
                "p1_airborne": state.p1.airborne,
                "p2_airborne": state.p2.airborne,
                "p1_throw_break_active": state.p1.throw_break_active,
                "p2_throw_break_active": state.p2.throw_break_active,
                "p1_throw_threat": self._is_throw_threat(state.p1),
                "p2_throw_threat": self._is_throw_threat(state.p2),
                "p1_hitstun": state.p1.hitstun,
                "p2_hitstun": state.p2.hitstun,
                "p1_whiffs": state.p1.whiffs,
                "p2_whiffs": state.p2.whiffs,
                "p1_launches_taken": state.p1.launches_taken,
                "p2_launches_taken": state.p2.launches_taken,
                "p1_y": state.p1.y,
                "p2_y": state.p2.y,
            },
        )

    @staticmethod
    def _is_throw_threat(fighter: FighterRuntime) -> bool:
        if fighter.move_key is None:
            return False
        move = JUN_MOVES[fighter.move_key]
        return move.hit_level == HitLevel.THROW and fighter.move_frame <= move.startup + move.active

    def _initial_state(self) -> SimState:
        return SimState(
            p1=FighterRuntime(health=self.config.max_health, x=-0.85),
            p2=FighterRuntime(health=self.config.max_health, x=0.85),
            frame=0,
        )


_ACTION_TO_MOVE: dict[SimAction, str] = {
    SimAction.JAB: "jab",
    SimAction.DF1: "df1",
    SimAction.F2: "f2",
    SimAction.DB3: "db3",
    SimAction.HOPKICK: "hopkick",
    SimAction.THROW: "throw",
}
