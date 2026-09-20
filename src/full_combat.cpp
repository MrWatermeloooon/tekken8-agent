#include "t8_v2/full_combat.hpp"

#include <algorithm>
#include <cmath>

namespace t8::v2 {
namespace {

bool has_flag(const FullCombatMove& move, FullCombatFlag flag) {
    return (move.flags & static_cast<std::uint32_t>(flag)) != 0;
}

bool blocked(HitLevel guard, HitLevel level) {
    if (level == HitLevel::Throw) return false;
    if (guard == HitLevel::Low) return level == HitLevel::Low;
    if (guard == HitLevel::Mid) return level == HitLevel::High || level == HitLevel::Mid;
    return false;
}

FullAttackOutcome check(
    const FullFighterState& attacker,
    const FullFighterState& defender,
    const FullCombatMove& attack,
    const std::optional<FullCombatMove>& defense_move) {
    FullAttackOutcome outcome{};
    outcome.legal = FullCombatOracle::legal(attacker, attack);
    if (!outcome.legal || std::abs(defender.x - attacker.x) > attack.range) return outcome;
    if (!has_flag(attack, FullHoming)) {
        const double delta = defender.axis - attacker.axis;
        if (delta < -attack.tracking_left || delta > attack.tracking_right) return outcome;
    }
    if (defense_move) {
        if ((attack.hit_level == HitLevel::High && has_flag(*defense_move, FullHighCrush)) ||
            ((attack.hit_level == HitLevel::Low || attack.hit_level == HitLevel::Throw) &&
             has_flag(*defense_move, FullLowCrush))) {
            outcome.crushed = true;
            return outcome;
        }
        if ((has_flag(*defense_move, FullParry) || has_flag(*defense_move, FullReversal)) &&
            attack.hit_level != HitLevel::Low && attack.hit_level != HitLevel::Throw) {
            outcome.parried = true;
            return outcome;
        }
        outcome.armored = has_flag(*defense_move, FullPowerCrush) &&
            attack.hit_level != HitLevel::Low && attack.hit_level != HitLevel::Throw;
    }
    outcome.landed = true;
    outcome.blocked = blocked(defender.guard, attack.hit_level);
    if (!outcome.blocked) {
        const double scale = defender.combo_count == 0
            ? 1.0 : std::max(0.30, 1.0 - 0.10 * static_cast<double>(defender.combo_count));
        outcome.damage = attack.damage * scale;
    }
    return outcome;
}

void spend_and_restore(FullFighterState& fighter, const FullCombatMove& move) {
    fighter.resource = std::max(0.0, fighter.resource + move.resource_delta);
    fighter.health = std::max(1.0, fighter.health - move.health_cost);
    const double restored = std::min(move.health_restore, fighter.recoverable_health);
    fighter.health = std::min(180.0, fighter.health + restored);
    fighter.recoverable_health -= restored;
    if (move.requires_heat || has_flag(move, FullHeatDash) || has_flag(move, FullHeatSmash)) {
        fighter.heat = std::max(0.0, fighter.heat - 1.0);
    }
    if (move.requires_rage || has_flag(move, FullRageArt)) fighter.rage = false;
}

void apply(
    FullFighterState& attacker,
    FullFighterState& defender,
    FullStageState& stage,
    const FullCombatMove& move,
    FullAttackOutcome& outcome,
    int direction) {
    if (!outcome.legal) return;
    spend_and_restore(attacker, move);
    if (!outcome.landed || outcome.blocked || outcome.parried || outcome.crushed) return;
    defender.health = std::max(0.0, defender.health - outcome.damage);
    defender.recoverable_health = std::min(
        180.0 - defender.health, defender.recoverable_health + outcome.damage * 0.30);
    if (!outcome.armored) {
        if (has_flag(move, FullLauncher)) {
            defender.posture = Posture::Airborne;
            defender.combo_count = std::max(1, defender.combo_count + 1);
        } else if (defender.combo_count > 0) {
            ++defender.combo_count;
        }
        if (has_flag(move, FullTornado) && !defender.tornado_used) {
            defender.tornado_used = true;
            defender.posture = Posture::Airborne;
        }
    }
    defender.x = std::clamp(
        defender.x + move.pushback * static_cast<double>(direction),
        -stage.half_width, stage.half_width);
    const bool at_wall = std::abs(defender.x) >= stage.half_width - 1e-9;
    outcome.wall_splat = at_wall && has_flag(move, FullWallSplat);
    defender.wall_splat = outcome.wall_splat;
    if (at_wall && has_flag(move, FullWallBreak)) stage.wall_broken = true;
    if (has_flag(move, FullFloorBreak)) stage.floor_broken = true;
    if (at_wall && has_flag(move, FullBalconyBreak)) stage.balcony_broken = true;
    if (has_flag(move, FullHeatEngager)) attacker.heat = std::max(attacker.heat, 1.0);
}

}  // namespace

bool FullCombatOracle::legal(const FullFighterState& fighter, const FullCombatMove& move) {
    return move.validated && move.startup > 0 && move.active > 0 && move.recovery >= 0 &&
        move.damage >= 0.0 && move.range > 0.0 && fighter.health > move.health_cost &&
        fighter.posture == move.required_posture &&
        (move.required_stance < 0 || fighter.stance == move.required_stance) &&
        fighter.resource >= move.required_resource && (!move.requires_heat || fighter.heat > 0.0) &&
        (!move.requires_rage || fighter.rage);
}

FullExchangeOutcome FullCombatOracle::resolve(
    FullCombatState& state,
    const std::optional<FullCombatMove>& p1_move,
    const std::optional<FullCombatMove>& p2_move) {
    const FullCombatState snapshot = state;
    FullExchangeOutcome result{};
    if (p1_move) result.p1 = check(snapshot.p1, snapshot.p2, *p1_move, p2_move);
    if (p2_move) result.p2 = check(snapshot.p2, snapshot.p1, *p2_move, p1_move);
    if (p1_move) apply(state.p1, state.p2, state.stage, *p1_move, result.p1, 1);
    if (p2_move) apply(state.p2, state.p1, state.stage, *p2_move, result.p2, -1);
    return result;
}

}  // namespace t8::v2
