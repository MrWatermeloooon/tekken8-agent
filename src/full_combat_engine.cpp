#include "t8_v2/full_combat_engine.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string_view>

namespace t8::v2 {

int FullMoveSpec::total_frames() const noexcept {
    int last_active = 0;
    for (const auto& hit : hits) last_active = std::max(last_active, hit.first_active_frame + hit.active_frames - 1);
    return last_active + recovery;
}

std::string_view to_string(FullContact contact) noexcept {
    switch (contact) {
        case FullContact::Hit: return "hit";
        case FullContact::CounterHit: return "counter_hit";
        case FullContact::Blocked: return "blocked";
        case FullContact::Crushed: return "crushed";
        case FullContact::Parried: return "parried";
        case FullContact::Armored: return "armored";
        case FullContact::Evaded: return "evaded";
        case FullContact::ThrowBroken: return "throw_broken";
        case FullContact::Reversed: return "reversed";
        case FullContact::ThrowHeld: return "throw_held";
    }
    return "unknown";
}

FullLevelClass level_class(FullHitLevel level) noexcept {
    switch (level) {
        case FullHitLevel::High: return FullLevelHigh;
        case FullHitLevel::Mid:
        case FullHitLevel::SpecialMid: return FullLevelMid;
        case FullHitLevel::Low:
        case FullHitLevel::SpecialLow: return FullLevelLow;
        case FullHitLevel::Throw: return FullLevelThrow;
    }
    return FullLevelMid;
}

std::size_t level_class_index(FullHitLevel level) noexcept {
    switch (level_class(level)) {
        case FullLevelHigh: return 0;
        case FullLevelMid: return 1;
        case FullLevelLow: return 2;
        case FullLevelThrow: return 3;
    }
    return 1;
}

namespace {

bool is_strike(FullHitLevel level) noexcept { return level != FullHitLevel::Throw; }
bool is_low(FullHitLevel level) noexcept {
    return level == FullHitLevel::Low || level == FullHitLevel::SpecialLow;
}

bool is_throw_break(FullUniversal action) noexcept {
    return action == FullUniversal::ThrowBreak || action == FullUniversal::ThrowBreak1 ||
        action == FullUniversal::ThrowBreak2;
}

bool breaks(FullThrowBreak rule, FullUniversal input) noexcept {
    switch (rule) {
        case FullThrowBreak::None: return false;
        case FullThrowBreak::One: return input == FullUniversal::ThrowBreak1;
        case FullThrowBreak::Two: return input == FullUniversal::ThrowBreak2;
        case FullThrowBreak::OneTwo: return input == FullUniversal::ThrowBreak;
        case FullThrowBreak::OneOrTwo:
            return input == FullUniversal::ThrowBreak1 || input == FullUniversal::ThrowBreak2;
    }
    return false;
}

int universal_duration(FullUniversal action, const FullEngineConfig& config) {
    switch (action) {
        case FullUniversal::DashForward: return config.dash_frames;
        case FullUniversal::Backdash: return config.backdash_frames;
        case FullUniversal::SidestepLeft:
        case FullUniversal::SidestepRight: return config.sidestep_frames;
        case FullUniversal::HeatDash: return config.heat_dash_frames;
        default: return 1;  // Idle, walking, and crouching are re-chosen every frame.
    }
}

// A decided contact, applied after both fighters' contacts are known so a
// same-frame trade resolves from one snapshot.
struct Decision {
    int attacker = 0;
    std::size_t move_index = 0;
    std::optional<std::size_t> defense_move;
    std::size_t hit_index = 0;
    FullContact contact = FullContact::Hit;
};

}  // namespace

FullCombatEngine::FullCombatEngine(std::array<std::vector<FullMoveSpec>, 2> movesets, FullEngineConfig config,
                                   std::array<FullCharacterRules, 2> rules)
    : movesets_(std::move(movesets)), config_(config), rules_(std::move(rules)) {
    if (!(config_.half_width > 0.0) || !(config_.body_radius >= 0.0) || config_.dash_frames <= 0 ||
        config_.backdash_frames <= 0 || config_.sidestep_frames <= 0 || config_.launch_air_frames <= 0 ||
        config_.throw_break_window <= 0 || config_.heat_dash_frames <= 0 || config_.heat_duration_frames <= 0 ||
        !(config_.max_health > 0.0) || config_.rage_health_fraction < 0.0 || config_.rage_health_fraction > 1.0) {
        throw std::invalid_argument("full combat engine configuration is invalid");
    }
    for (const auto& moveset : movesets_) {
        for (const auto& move : moveset) {
            // Stance entries have no hits; reactive outcomes need the hit they deal.
            if (move.hits.size() > 32 || move.recovery < 0 || (move.hits.empty() && move.recovery <= 0) ||
                (move.reactive && move.hits.empty())) {
                throw std::invalid_argument("move " + move.stable_id + " needs 0-32 hits and a valid recovery");
            }
            int previous = 0;
            for (const auto& hit : move.hits) {
                if (hit.first_active_frame <= previous || hit.active_frames <= 0 || !(hit.reach > 0.0) ||
                    hit.damage < 0.0 || hit.tracking_left < 0.0 || hit.tracking_right < 0.0 ||
                    hit.chip_damage < 0.0 || (hit.heat_chip_damage && *hit.heat_chip_damage < 0.0)) {
                    throw std::invalid_argument("move " + move.stable_id + " has an invalid or unordered hit");
                }
                previous = hit.first_active_frame;
            }
            if (move.self_damage < 0.0 || move.self_recoverable < 0.0 || move.self_recoverable > move.self_damage ||
                move.restore_health_hit < 0.0 || move.restore_recoverable_hit < 0.0 ||
                move.restore_recoverable_block < 0.0 || move.reversal_damage < 0.0 ||
                (move.launch_air_frames && *move.launch_air_frames <= 0) || move.heat_cost_frames < 0 ||
                move.resource_gain_start < 0.0 || move.resource_gain_hit < 0.0 ||
                move.resource_gain_airborne_hit < 0.0 || move.resource_gain_block < 0.0 ||
                move.resource_gain_heat_activation < 0.0 || move.install_damage_bonus < 0.0 ||
                (move.install_reach && !(*move.install_reach > 0.0))) {
                throw std::invalid_argument("move " + move.stable_id + " has invalid resource data");
            }
        }
    }
    for (const auto& rule : rules_) {
        if (rule.resource_max < 0.0 || (rule.install_threshold && *rule.install_threshold > rule.resource_max)) {
            throw std::invalid_argument("character resource rules are invalid");
        }
        for (const auto& stance : rule.stances) {
            if (stance.pulse_interval_frames < 0 || (stance.max_frames && *stance.max_frames <= 0)) {
                throw std::invalid_argument("stance " + stance.name + " has invalid timing");
            }
        }
    }
    reset();
}

void FullCombatEngine::next_round(double p1_x, double p2_x) {
    const std::array<double, 2> carried = {state_.fighters[0].resource, state_.fighters[1].resource};
    reset(p1_x, p2_x);
    for (int player = 0; player < 2; ++player) {
        if (rules_[static_cast<std::size_t>(player)].resource_persists) {
            gain_resource(player, carried[static_cast<std::size_t>(player)]);
        }
    }
}

const FullStanceRule* FullCombatEngine::stance_rule(int player) const {
    const auto& fighter = state_.fighters.at(static_cast<std::size_t>(player));
    const auto& stances = rules_.at(static_cast<std::size_t>(player)).stances;
    if (fighter.stance < 0 || static_cast<std::size_t>(fighter.stance) >= stances.size()) return nullptr;
    return &stances[static_cast<std::size_t>(fighter.stance)];
}

std::optional<std::size_t> FullCombatEngine::find_move(int player, const std::string& stable_id) const {
    if (stable_id.empty()) return std::nullopt;
    const auto& moveset = movesets_.at(static_cast<std::size_t>(player));
    for (std::size_t index = 0; index < moveset.size(); ++index) {
        if (moveset[index].stable_id == stable_id) return index;
    }
    return std::nullopt;
}

void FullCombatEngine::gain_resource(int player, double amount) {
    auto& fighter = state_.fighters[static_cast<std::size_t>(player)];
    const auto& rule = rules_[static_cast<std::size_t>(player)];
    if (!(rule.resource_max > 0.0) || !(amount > 0.0)) return;
    fighter.resource = std::min(rule.resource_max, fighter.resource + amount);
    if (rule.install_threshold && fighter.resource >= *rule.install_threshold) fighter.installed = true;
}

void FullCombatEngine::reset(double p1_x, double p2_x) {
    state_ = {};
    state_.stage.left_wall = -config_.half_width;
    state_.stage.right_wall = config_.half_width;
    state_.fighters[0].x = p1_x;
    state_.fighters[1].x = p2_x;
    for (auto& fighter : state_.fighters) {
        fighter.health = config_.max_health;
        fighter.action_frame = universal_duration(FullUniversal::Idle, config_);  // actionable at once
    }
}

const FullMoveSpec& FullCombatEngine::move(int player, std::size_t index) const {
    return movesets_.at(static_cast<std::size_t>(player)).at(index);
}

bool FullCombatEngine::actionable(int player) const {
    const auto& fighter = state_.fighters.at(static_cast<std::size_t>(player));
    if (fighter.stun > 0 || fighter.crumpled || fighter.throw_break_frames > 0 || fighter.pending_throw_hit >= 0) {
        return false;
    }
    if (fighter.posture != FullPosture::Standing && fighter.posture != FullPosture::Crouching) return false;
    if (fighter.move) return fighter.action_frame >= move(player, *fighter.move).total_frames();
    return fighter.action_frame >= universal_duration(fighter.universal, config_);
}

bool FullCombatEngine::legal(int player, const FullInput& input) const {
    if (state_.winner >= 0) return false;
    const auto& fighter = state_.fighters.at(static_cast<std::size_t>(player));
    if (input.move) {
        if (*input.move >= movesets_[static_cast<std::size_t>(player)].size()) return false;
        const auto& spec = move(player, *input.move);
        if (spec.reactive) return false;
        if (spec.requires_sidestep) {
            // Performed out of a sidestep in progress.
            const bool sidestepping = !fighter.move && fighter.action_frame >= 1 && fighter.stun == 0 &&
                !fighter.crumpled && fighter.throw_break_frames == 0 && fighter.pending_throw_hit < 0 &&
                (fighter.universal == FullUniversal::SidestepLeft || fighter.universal == FullUniversal::SidestepRight);
            if (!sidestepping) return false;
        } else if (!actionable(player)) {
            return false;
        }
        const auto& other = state_.fighters.at(static_cast<std::size_t>(1 - player));
        if (spec.requires_back_turned != fighter.back_turned) return false;  // back-turned: only BT moves
        if (spec.requires_opponent_back_turned && !other.back_turned) return false;
        if (spec.requires_back_to_wall) {
            const bool facing_right = other.x >= fighter.x;
            const double wall = facing_right ? state_.stage.left_wall : state_.stage.right_wall;
            if (std::abs(fighter.x - wall) > config_.back_to_wall_distance) return false;
        }
        if (spec.requires_opponent_left_side || spec.requires_opponent_right_side) {
            // Offset of this fighter as seen by the opponent, which faces it: negative is the
            // opponent's left (the tracking convention), so a large negative offset exposes its left side.
            const double toward = fighter.x >= other.x ? 1.0 : -1.0;
            const double distance = std::max(std::abs(fighter.x - other.x), 1e-9);
            const double side = (fighter.axis - other.axis) * toward / distance;
            if (spec.requires_opponent_left_side && side > -config_.side_exposure_ratio) return false;
            if (spec.requires_opponent_right_side && side < config_.side_exposure_ratio) return false;
        }
        return spec.required_posture == fighter.posture &&
            (spec.required_stance < 0 || spec.required_stance == fighter.stance) &&
            (!spec.requires_heat || fighter.heat) && (!spec.consumes_heat || fighter.heat) &&
            (spec.heat_cost_frames == 0 || fighter.heat) &&
            (!spec.engages_heat || (fighter.heat_available && !fighter.heat)) &&
            (!(spec.requires_rage || spec.consumes_rage) || fighter.rage);
    }
    switch (input.universal) {
        case FullUniversal::WakeupStand:
        case FullUniversal::WakeupRoll:
            return fighter.posture == FullPosture::Grounded && fighter.grounded_elapsed >= config_.knockdown_frames;
        case FullUniversal::TechRoll:
            return fighter.posture == FullPosture::Grounded && fighter.techable &&
                fighter.grounded_elapsed < config_.tech_window_frames;
        case FullUniversal::ThrowBreak:
        case FullUniversal::ThrowBreak1:
        case FullUniversal::ThrowBreak2:
            return fighter.throw_break_frames > 0 && !fighter.throw_break_attempted;
        case FullUniversal::HeatDash: {
            if (!fighter.move || !fighter.heat || !fighter.heat_dash_from) return false;
            const auto& spec = move(player, *fighter.move);
            return *fighter.heat_dash_from == FullContact::Blocked ? spec.heat_dash_block_advantage.has_value()
                                                                   : spec.heat_dash_hit_advantage.has_value();
        }
        default:
            return actionable(player);
    }
}

void FullCombatEngine::deal_damage(int target, double damage, bool recoverable, bool can_ko,
                                   FullFrameEvents& events) {
    auto& fighter = state_.fighters[static_cast<std::size_t>(target)];
    if (!can_ko) damage = std::min(damage, std::max(0.0, fighter.health - 1.0));
    damage = std::max(0.0, damage);
    fighter.health -= damage;
    if (recoverable) fighter.recoverable_health += damage;
    fighter.recoverable_health = std::clamp(fighter.recoverable_health, 0.0, config_.max_health - fighter.health);
    if (fighter.health > 0.0 && fighter.rage_available &&
        fighter.health <= config_.rage_health_fraction * config_.max_health) {
        fighter.rage = true;
        fighter.rage_available = false;
        events.rage_activated[static_cast<std::size_t>(target)] = true;
    }
}

namespace {

void regain(FullFighter& fighter, double recoverable, double health, double max_health) {
    const double restored = std::min(std::max(0.0, recoverable), fighter.recoverable_health);
    fighter.recoverable_health -= restored;
    fighter.health = std::min(max_health, fighter.health + restored + std::max(0.0, health));
    fighter.recoverable_health = std::min(fighter.recoverable_health, max_health - fighter.health);
}

}  // namespace

void FullCombatEngine::activate_heat(int player, FullFrameEvents& events) {
    auto& fighter = state_.fighters[static_cast<std::size_t>(player)];
    if (!fighter.heat_available || fighter.heat) return;
    fighter.heat = true;
    fighter.heat_available = false;
    fighter.heat_frames = config_.heat_duration_frames;
    events.heat_activated[static_cast<std::size_t>(player)] = true;
}

void FullCombatEngine::break_throw(int thrower, FullFrameEvents& events) {
    auto& attacker = state_.fighters[static_cast<std::size_t>(thrower)];
    auto& defender = state_.fighters[static_cast<std::size_t>(1 - thrower)];
    const std::size_t hit_index = static_cast<std::size_t>(attacker.pending_throw_hit);
    const bool side_switch = attacker.move && move(thrower, *attacker.move).side_switch_on_break;
    events.contacts.push_back({thrower, hit_index, FullContact::ThrowBroken, 0.0});
    for (auto* fighter : {&attacker, &defender}) {
        fighter->move.reset();
        fighter->universal = FullUniversal::Idle;
        fighter->action_frame = universal_duration(FullUniversal::Idle, config_);
        fighter->stun = config_.throw_break_recovery;
        fighter->in_blockstun = true;  // guard holds while recovering from the break
        fighter->heat_dash_from.reset();
    }
    attacker.pending_throw_hit = -1;
    defender.throw_break_frames = 0;
    if (side_switch) std::swap(attacker.x, defender.x);
    const double direction = defender.x >= attacker.x ? 1.0 : -1.0;
    attacker.x = std::max(state_.stage.left_wall, std::min(state_.stage.right_wall, attacker.x - direction * 0.2));
    defender.x = std::max(state_.stage.left_wall, std::min(state_.stage.right_wall, defender.x + direction * 0.2));
}

void FullCombatEngine::release_throw(int thrower) {
    auto& fighter = state_.fighters[static_cast<std::size_t>(thrower)];
    auto& held = state_.fighters[static_cast<std::size_t>(1 - thrower)];
    if (fighter.pending_throw_hit < 0) return;
    fighter.pending_throw_hit = -1;
    held.throw_break_frames = 0;
    held.throw_break_attempted = false;
}

void FullCombatEngine::apply_parry_outcome(int parrier_index, std::size_t outcome, int attacker_index,
                                           FullFrameEvents& events) {
    auto& parrier = state_.fighters[static_cast<std::size_t>(parrier_index)];
    auto& attacker = state_.fighters[static_cast<std::size_t>(attacker_index)];
    const auto& spec = move(parrier_index, outcome);
    double damage = 0.0;
    for (const auto& hit : spec.hits) damage += hit.damage;
    deal_damage(attacker_index, damage, spec.recoverable_only, !spec.recoverable_only, events);
    release_throw(attacker_index);
    attacker.move.reset();
    attacker.universal = FullUniversal::Idle;
    attacker.stance = -1;
    attacker.heat_dash_from.reset();
    attacker.in_blockstun = false;
    const int frames = spec.recovery > 0 ? spec.recovery : config_.parry_outcome_frames;
    switch (spec.hit_effect) {
        case FullHitEffect::Launch:
            attacker.stun = 0;
            attacker.posture = FullPosture::Airborne;
            attacker.posture_frames = spec.launch_air_frames.value_or(
                frames + spec.hit_advantage > 0 ? frames + spec.hit_advantage : config_.launch_air_frames);
            attacker.combo_hits = 1;
            break;
        case FullHitEffect::Knockdown:
        case FullHitEffect::Crumple:
            attacker.stun = 0;
            attacker.posture = FullPosture::Grounded;
            attacker.grounded_elapsed = 0;
            attacker.techable = false;
            break;
        case FullHitEffect::Stun:
            attacker.stun = std::max(0, frames + spec.hit_advantage);
            break;
    }
    // The parrying fighter plays the outcome: its hits have resolved, `frames` of
    // recovery remain, and it ends in the outcome's stance.
    release_throw(parrier_index);
    parrier.move = outcome;
    parrier.universal = FullUniversal::Idle;
    parrier.stance = -1;
    parrier.stun = 0;
    parrier.in_blockstun = false;
    parrier.heat_dash_from.reset();
    parrier.action_frame = std::max(0, spec.total_frames() - frames);
    parrier.hits_landed = spec.hits.size() >= 32 ? ~0U : (1U << spec.hits.size()) - 1U;
    parrier.move_hit = true;
    parrier.move_blocked = false;
    parrier.resource_gained = true;
    if (spec.heat_cost_frames > 0 && parrier.heat) {
        parrier.heat_frames -= spec.heat_cost_frames;
        if (parrier.heat_frames <= 0) {
            parrier.heat = false;
            parrier.heat_frames = 0;
        }
    }
    regain(parrier, spec.restore_recoverable_hit, spec.restore_health_hit, config_.max_health);
    gain_resource(parrier_index, spec.resource_gain_start + spec.resource_gain_hit);
    if (spec.removes_recoverable) attacker.recoverable_health = 0.0;
}

void FullCombatEngine::apply_contact(int attacker_index, std::size_t move_index,
                                     std::optional<std::size_t> defense_move, std::size_t hit_index,
                                     FullContact contact, int elapsed, FullFrameEvents& events) {
    auto& stage = state_.stage;
    auto& attacker = state_.fighters[static_cast<std::size_t>(attacker_index)];
    auto& defender = state_.fighters[static_cast<std::size_t>(1 - attacker_index)];
    const auto& spec = move(attacker_index, move_index);
    const auto& hit = spec.hits[hit_index];
    const bool last_hit = hit_index + 1 == spec.hits.size();
    FullContactEvent event{attacker_index, hit_index, contact, 0.0};
    // Published advantage assumes contact on the hit's first active frame;
    // stun is fixed from there, so a later (meaty) contact gains frames. A
    // throw resolves `elapsed` frames after it connected (its break window).
    const int remaining = std::max(0, spec.total_frames() - hit.first_active_frame - elapsed);
    const auto push_defender = [&](double distance) {
        // Away from the attacker, judged at push time (after any side switch).
        const double direction = defender.x >= attacker.x ? 1.0 : -1.0;
        defender.x += direction * distance;
        const double wall = direction > 0.0 ? stage.right_wall : stage.left_wall;
        const double overflow = direction > 0.0 ? defender.x - wall : wall - defender.x;
        if (overflow > 0.0) {
            defender.x = wall;
            attacker.x -= direction * overflow;  // pushback against a wall moves the attacker back
        }
    };
    const auto enable_heat_dash = [&](FullContact from) {
        if (!last_hit) return;  // a multi-hit engager activates Heat and allows Heat Dash after its last hit
        if (spec.heat_engager && attacker.heat_available && !attacker.heat) {
            activate_heat(attacker_index, events);
            gain_resource(attacker_index, spec.resource_gain_heat_activation);
        }
        // Engagers, and Heat versions that publish Heat Dash data, can Heat Dash while in Heat.
        const bool has_dash = spec.heat_dash_block_advantage || spec.heat_dash_hit_advantage;
        if (attacker.heat && has_dash) attacker.heat_dash_from = from;
    };
    const auto scaled = [&](double damage, bool counter) {
        damage *= counter ? config_.counter_hit_damage_scale : 1.0;
        if (attacker.rage) damage *= config_.rage_damage_scale;
        if (defender.combo_hits > 0) {
            damage *= std::max(config_.combo_scale_floor, 1.0 - config_.combo_scale_step * defender.combo_hits);
        }
        if (defender.wall_scaled) damage *= config_.wall_combo_scale;
        return damage;
    };

    switch (contact) {
        case FullContact::Evaded:
        case FullContact::Crushed:
        case FullContact::ThrowBroken:
        case FullContact::ThrowHeld:
            break;
        case FullContact::Parried: {
            // A parry with a published outcome plays it (Jun: GEN.P); otherwise the attacker is stunned.
            const auto level = level_class_index(hit.level);
            std::string outcome;
            if (defense_move) {
                const auto& defense = move(1 - attacker_index, *defense_move);
                const bool heat_parry = defender.move_in_heat && (defense.heat_parry_levels & level_class(hit.level)) &&
                    defense.heat_parry.contains(defender.action_frame);
                outcome = heat_parry ? defense.heat_parry_outcome : defense.parry_outcomes[level];
                const auto& stances = rules_[static_cast<std::size_t>(1 - attacker_index)].stances;
                if (outcome.empty() && defense.result_stance >= 0 &&
                    static_cast<std::size_t>(defense.result_stance) < stances.size()) {
                    outcome = stances[static_cast<std::size_t>(defense.result_stance)].parry_outcomes[level];
                }
            } else if (const auto* rule = stance_rule(1 - attacker_index)) {
                outcome = rule->parry_outcomes[level];
            }
            if (const auto index = find_move(1 - attacker_index, outcome)) {
                const double before = attacker.health;
                apply_parry_outcome(1 - attacker_index, *index, attacker_index, events);
                event.damage = before - attacker.health;
                break;
            }
            release_throw(attacker_index);
            attacker.stance = -1;
            attacker.move.reset();
            attacker.universal = FullUniversal::Idle;
            attacker.action_frame = universal_duration(FullUniversal::Idle, config_);
            attacker.stun = config_.parry_stun_frames;
            attacker.in_blockstun = false;
            attacker.heat_dash_from.reset();
            break;
        }
        case FullContact::Reversed: {
            const double damage = defense_move ? move(1 - attacker_index, *defense_move).reversal_damage : 0.0;
            deal_damage(attacker_index, damage, false, true, events);
            event.damage = damage;
            release_throw(attacker_index);
            attacker.move.reset();
            attacker.universal = FullUniversal::Idle;
            attacker.stance = -1;
            attacker.stun = 0;
            attacker.heat_dash_from.reset();
            attacker.posture = FullPosture::Grounded;
            attacker.grounded_elapsed = 0;
            attacker.techable = false;
            // The reversing fighter recovers with its guard up.
            defender.move.reset();
            defender.universal = FullUniversal::Idle;
            defender.action_frame = universal_duration(FullUniversal::Idle, config_);
            defender.stun = config_.reversal_recovery;
            defender.in_blockstun = true;
            break;
        }
        case FullContact::Blocked: {
            defender.stun = std::max(0, remaining + spec.block_advantage);
            defender.in_blockstun = defender.stun > 0;
            attacker.move_blocked = true;
            if (last_hit && !attacker.resource_gained) {
                attacker.resource_gained = true;
                gain_resource(attacker_index, spec.resource_gain_block);
            }
            const double chip = attacker.installed && last_hit && spec.install_chip_damage ? *spec.install_chip_damage
                : attacker.heat && hit.heat_chip_damage ? *hit.heat_chip_damage : hit.chip_damage;
            if (chip > 0.0) deal_damage(1 - attacker_index, chip, true, false, events);
            event.damage = chip;
            regain(attacker, config_.recoverable_regain_block * hit.damage +
                       (last_hit ? spec.restore_recoverable_block : 0.0), 0.0, config_.max_health);
            enable_heat_dash(FullContact::Blocked);
            push_defender(spec.pushback_block);
            break;
        }
        case FullContact::Armored:
        case FullContact::Hit:
        case FullContact::CounterHit: {
            const bool counter = contact == FullContact::CounterHit;
            const bool juggle = defender.posture == FullPosture::Airborne || defender.posture == FullPosture::WallSplat;
            const bool defender_back_turned = defender.back_turned;
            const FullPosture defender_posture = defender.posture;
            double base = hit.damage + (attacker.installed && last_hit ? spec.install_damage_bonus : 0.0);
            if (last_hit && spec.rage_art_max_damage) {
                // Provisional: grows linearly from the listed damage at full health to the maximum at none.
                double listed = 0.0;
                for (const auto& part : spec.hits) listed += part.damage;
                const double missing = 1.0 - attacker.health / config_.max_health;
                base += std::max(0.0, *spec.rage_art_max_damage - listed) * std::clamp(missing, 0.0, 1.0);
            }
            if (attacker.move_boosted) base *= config_.ki_charge_damage_scale;
            const double damage = scaled(base, counter);
            if (contact == FullContact::Armored) {
                const bool recoverable = defense_move && move(1 - attacker_index, *defense_move).armor_damage_recoverable;
                deal_damage(1 - attacker_index, damage, recoverable, true, events);
                event.damage = damage;
                break;  // armor: damage only, the defender keeps attacking
            }
            {
                const bool can_ko = !spec.cannot_ko && !spec.recoverable_only;
                const double recoverable_part = spec.recoverable_only ? damage
                    : last_hit ? std::min(damage, spec.recoverable_damage) : 0.0;
                deal_damage(1 - attacker_index, damage - recoverable_part, false, can_ko, events);
                deal_damage(1 - attacker_index, recoverable_part, true, false, events);
            }
            event.damage = damage;
            if (spec.removes_recoverable) defender.recoverable_health = 0.0;
            attacker.move_hit = true;
            if (last_hit && !attacker.resource_gained) {
                attacker.resource_gained = true;
                gain_resource(attacker_index, juggle ? spec.resource_gain_airborne_hit : spec.resource_gain_hit);
            }
            regain(attacker, config_.recoverable_regain_hit * damage + (last_hit ? spec.restore_recoverable_hit : 0.0),
                   last_hit ? spec.restore_health_hit : 0.0, config_.max_health);

            release_throw(1 - attacker_index);  // a throw loses to a strike landing on the same frame
            defender.stance = -1;
            defender.move.reset();
            defender.universal = FullUniversal::Idle;
            defender.action_frame = 0;
            defender.in_blockstun = false;
            defender.heat_dash_from.reset();
            defender.back_turned = false;
            const bool back_turned_hit = defender_back_turned && spec.back_turned_hit_advantage.has_value();
            const FullHitEffect effect = back_turned_hit ? spec.back_turned_hit_effect
                : counter && spec.counter_hit_effect ? *spec.counter_hit_effect : spec.hit_effect;
            const int advantage = back_turned_hit ? *spec.back_turned_hit_advantage
                : counter && spec.counter_hit_advantage ? *spec.counter_hit_advantage : spec.hit_advantage;
            // Attack throw: the strike turns into an unbreakable throw when its condition holds.
            const bool throw_trigger = spec.attack_throw == FullMoveSpec::AttackThrow::OnHit ||
                (spec.attack_throw == FullMoveSpec::AttackThrow::OnCounterHit && counter);
            const bool throw_front = !spec.attack_throw_front_only || !defender_back_turned;
            const bool throw_posture = defender_posture == FullPosture::Airborne ? spec.attack_throw_airborne
                : !spec.attack_throw_standing_only || defender_posture == FullPosture::Standing;
            if (last_hit && throw_trigger && throw_front && throw_posture && !back_turned_hit) {
                const double extra = scaled(spec.attack_throw_damage, false);
                deal_damage(1 - attacker_index, extra, false, !spec.cannot_ko, events);
                event.damage += extra;
            }
            const bool thrown = hit.level == FullHitLevel::Throw;
            if (juggle && hit.spike) {
                ++defender.combo_hits;
                defender.posture = FullPosture::Grounded;
                defender.grounded_elapsed = 0;
                defender.techable = false;
            } else if (juggle) {
                ++defender.combo_hits;
                if (defender.posture == FullPosture::Airborne) {
                    // Kept up until the attacker recovers, plus a window for the next move.
                    const int window = std::max(0, config_.juggle_window_frames -
                                                       config_.juggle_window_decay * (defender.combo_hits - 1));
                    defender.posture_frames = std::max(defender.posture_frames, remaining + window);
                    if (hit.tornado && !defender.tornado_used) {
                        defender.tornado_used = true;
                        defender.posture_frames += config_.tornado_extra_frames;
                    }
                } else {
                    // Hits on a wall-splatted opponent keep it pinned the same way.
                    defender.posture_frames = std::max(defender.posture_frames, remaining + config_.wall_splat_frames);
                }
            } else if (defender.posture == FullPosture::Grounded) {
                ++defender.combo_hits;
                defender.grounded_elapsed = 0;  // hits on the ground keep the fighter down
            } else if (last_hit && effect == FullHitEffect::Launch) {
                defender.posture = FullPosture::Airborne;
                // "+29a": airborne until the attacker has been recovered for 29 frames.
                defender.posture_frames = spec.launch_air_frames.value_or(
                    remaining + advantage > 0 ? remaining + advantage : config_.launch_air_frames);
                defender.combo_hits = 1;
                defender.stun = 0;
            } else if (last_hit && effect == FullHitEffect::Knockdown) {
                defender.posture = FullPosture::Grounded;
                defender.grounded_elapsed = 0;
                defender.techable = !thrown;
                defender.stun = 0;
            } else if (last_hit && effect == FullHitEffect::Crumple) {
                defender.stun = config_.crumple_frames;
                defender.crumpled = true;
            } else {
                defender.stun = std::max(0, remaining + advantage);
                if (defender.posture == FullPosture::Crouching) defender.posture = FullPosture::Standing;
            }
            enable_heat_dash(contact);
            if (last_hit && spec.side_switch_on_hit) std::swap(attacker.x, defender.x);
            push_defender(spec.pushback_hit);

            // Walls and stage transitions, on the side the defender was driven toward.
            const bool right_side = defender.x >= attacker.x;
            const double wall = right_side ? stage.right_wall : stage.left_wall;
            const bool at_wall = std::abs(defender.x - wall) < 1e-9;
            bool& wall_breakable = right_side ? stage.right_wall_breakable : stage.left_wall_breakable;
            bool& balcony = right_side ? stage.right_balcony : stage.left_balcony;
            const auto send_airborne = [&] {
                defender.posture = FullPosture::Airborne;
                defender.posture_frames = config_.stage_break_air_frames;
                defender.stun = 0;
                defender.combo_hits = std::max(1, defender.combo_hits);
            };
            const double extension = right_side ? config_.wall_break_extension : -config_.wall_break_extension;
            if (at_wall && hit.balcony_break && balcony) {
                balcony = false;
                ++stage.level;
                (right_side ? stage.right_wall : stage.left_wall) += extension;
                defender.x += (right_side ? 1.0 : -1.0) * 1.0;
                send_airborne();
                events.transitions.push_back(FullTransition::BalconyBreak);
            } else if (at_wall && hit.wall_break && wall_breakable) {
                wall_breakable = false;
                (right_side ? stage.right_wall : stage.left_wall) += extension;
                send_airborne();
                events.transitions.push_back(FullTransition::WallBreak);
            } else if (at_wall && (hit.wall_splat || juggle) && !defender.wall_splat_used) {
                defender.posture = FullPosture::WallSplat;
                defender.posture_frames = remaining + config_.wall_splat_frames;
                defender.wall_splat_used = true;
                defender.wall_scaled = true;
                defender.stun = 0;
                defender.combo_hits = std::max(1, defender.combo_hits);
            } else if (hit.floor_break && stage.floor_breakable &&
                       (juggle || defender.posture == FullPosture::Grounded)) {
                stage.floor_breakable = false;
                ++stage.level;
                send_airborne();
                events.transitions.push_back(FullTransition::FloorBreak);
            }
            break;
        }
    }
    events.contacts.push_back(event);
}

FullFrameEvents FullCombatEngine::step(const FullInput& p1, const FullInput& p2) {
    FullFrameEvents events{};
    if (state_.winner >= 0) return events;
    auto& stage = state_.stage;
    const std::array<const FullInput*, 2> inputs = {&p1, &p2};
    const auto facing = [&](int player) {
        const auto& own = state_.fighters[static_cast<std::size_t>(player)];
        const auto& other = state_.fighters[static_cast<std::size_t>(1 - player)];
        if (other.x == own.x) return player == 0 ? 1.0 : -1.0;
        return other.x > own.x ? 1.0 : -1.0;
    };

    // 0. The Heat timer ticks first, so Heat activated on any frame lasts exactly its duration.
    for (auto& fighter : state_.fighters) {
        if (fighter.ki_charge_frames > 0 && --fighter.ki_charge_frames == 0) fighter.ki_charge_boost = false;
        if (fighter.heat && --fighter.heat_frames <= 0) {
            fighter.heat = false;
            fighter.heat_frames = 0;
            fighter.heat_dash_from.reset();
        }
    }

    // 1. Start new actions. Legality is judged for both fighters before
    // either acts, so a throw break and a Heat Dash on the same frame both count.
    std::array<bool, 2> accepted{};
    for (int player = 0; player < 2; ++player) accepted[static_cast<std::size_t>(player)] =
        legal(player, *inputs[static_cast<std::size_t>(player)]);
    for (int player = 0; player < 2; ++player) {
        auto& fighter = state_.fighters[static_cast<std::size_t>(player)];
        auto& other = state_.fighters[static_cast<std::size_t>(1 - player)];
        const FullInput& input = *inputs[static_cast<std::size_t>(player)];
        if (!accepted[static_cast<std::size_t>(player)]) continue;
        if (!input.move && is_throw_break(input.universal)) {
            // One attempt: the first break input counts, right or wrong.
            fighter.throw_break_attempted = true;
            if (fighter.throw_break_frames > 0 && other.move && other.pending_throw_hit >= 0) {
                const auto& thrower_move = move(1 - player, *other.move);
                if (breaks(thrower_move.hits[static_cast<std::size_t>(other.pending_throw_hit)].throw_break,
                           input.universal)) {
                    break_throw(1 - player, events);
                }
            }
            continue;
        }
        if (!input.move && input.universal == FullUniversal::HeatDash) {
            const auto& spec = move(player, *fighter.move);
            const bool on_block = *fighter.heat_dash_from == FullContact::Blocked;
            const int frames = config_.heat_dash_frames;
            if (on_block) {
                other.stun = std::max(0, frames + *spec.heat_dash_block_advantage);
                other.in_blockstun = other.stun > 0;
            } else {
                const int advantage = frames + *spec.heat_dash_hit_advantage;
                other.in_blockstun = false;
                // The published outcome replaces the engager's own (a launch becomes "+43d").
                other.posture = FullPosture::Standing;
                other.posture_frames = 0;
                other.crumpled = false;
                switch (spec.heat_dash_hit_effect) {
                    case FullHitEffect::Launch:
                        other.posture = FullPosture::Airborne;
                        other.posture_frames = std::max(1, advantage);
                        other.combo_hits = std::max(1, other.combo_hits);
                        other.stun = 0;
                        break;
                    case FullHitEffect::Knockdown:
                    case FullHitEffect::Crumple:
                        other.stun = std::max(1, advantage);
                        other.crumpled = true;  // goes down when the stun ends
                        break;
                    case FullHitEffect::Stun:
                        other.stun = std::max(0, advantage);
                        break;
                }
            }
            release_throw(player);
            fighter.move.reset();
            fighter.universal = FullUniversal::HeatDash;
            fighter.heat = false;
            fighter.heat_frames = 0;
            fighter.heat_dash_from.reset();
            fighter.action_frame = 0;
            fighter.hits_landed = 0;
            continue;
        }
        fighter.heat_dash_from.reset();
        if (input.move) {
            const auto& spec = move(player, *input.move);
            fighter.move = input.move;
            fighter.universal = FullUniversal::Idle;
            fighter.stance = -1;
            if (spec.engages_heat && fighter.heat_available && !fighter.heat) {
                activate_heat(player, events);
                gain_resource(player, spec.resource_gain_heat_activation);
            }
            if (spec.heat_cost_frames > 0) {
                fighter.heat_frames -= spec.heat_cost_frames;
                if (fighter.heat_frames <= 0) {
                    fighter.heat = false;
                    fighter.heat_frames = 0;
                }
            }
            fighter.move_hit = fighter.move_blocked = fighter.resource_gained = false;
            fighter.move_in_heat = fighter.heat;
            fighter.back_turned = false;
            fighter.move_boosted = fighter.ki_charge_boost && !spec.ki_charge;
            if (!spec.ki_charge) fighter.ki_charge_boost = false;
            if (spec.ki_charge) {
                fighter.ki_charge_frames = config_.ki_charge_frames;
                fighter.ki_charge_boost = true;
            }
            gain_resource(player, spec.resource_gain_start);
            if (spec.consumes_heat) {
                fighter.heat = false;
                fighter.heat_frames = 0;
            }
            if (spec.consumes_rage) fighter.rage = false;
            if (spec.self_damage > 0.0 && !(spec.self_damage_without_heat_only && fighter.heat)) {
                // Self-damage never K.O.s the user.
                deal_damage(player, spec.self_damage - spec.self_recoverable, false, false, events);
                deal_damage(player, spec.self_recoverable, true, false, events);
            }
        } else if (input.universal == FullUniversal::WakeupStand) {
            fighter.posture = FullPosture::Wakeup;
            fighter.posture_frames = config_.wakeup_stand_frames;
            fighter.move.reset();
            fighter.universal = FullUniversal::Idle;
        } else if (input.universal == FullUniversal::WakeupRoll || input.universal == FullUniversal::TechRoll) {
            const bool tech = input.universal == FullUniversal::TechRoll;
            fighter.posture = FullPosture::Wakeup;
            fighter.posture_frames = tech ? config_.tech_roll_frames : config_.wakeup_roll_frames;
            fighter.x -= facing(player) * (tech ? 0.5 : 1.0) * config_.wakeup_roll_distance;
            fighter.move.reset();
            fighter.universal = FullUniversal::Idle;
        } else {
            fighter.move.reset();
            fighter.universal = input.universal;
            if (input.universal != FullUniversal::Idle) {
                fighter.stance = -1;
                fighter.back_turned = false;  // any movement turns a back-turned fighter around
            }
            fighter.posture = input.universal == FullUniversal::Crouch ? FullPosture::Crouching : FullPosture::Standing;
        }
        fighter.action_frame = 0;
        fighter.hits_landed = 0;
    }

    // 2. Advance actions, movement, stun, resource, and posture timers.
    const bool sidestepping = std::any_of(state_.fighters.begin(), state_.fighters.end(), [](const auto& f) {
        return !f.move && (f.universal == FullUniversal::SidestepLeft || f.universal == FullUniversal::SidestepRight);
    });
    for (int player = 0; player < 2; ++player) {
        auto& fighter = state_.fighters[static_cast<std::size_t>(player)];
        const double forward = facing(player);
        if (fighter.move) {
            const auto& spec = move(player, *fighter.move);
            const bool holding_throw = fighter.pending_throw_hit >= 0;
            if (!holding_throw || fighter.action_frame < spec.total_frames()) ++fighter.action_frame;
            const int approach_frames = spec.hits.empty() ? 0 : spec.hits.front().first_active_frame - 1;
            if (approach_frames > 0 && fighter.action_frame <= approach_frames) {
                fighter.x += forward * spec.travel / approach_frames;
            }
            if (fighter.action_frame >= spec.total_frames() && !holding_throw) {
                fighter.stance = fighter.move_hit && spec.result_stance_on_hit >= 0 ? spec.result_stance_on_hit
                    : fighter.move_blocked && spec.result_stance_on_block >= 0 ? spec.result_stance_on_block
                    : spec.result_stance;
                fighter.stance_frames = 0;
                fighter.back_turned = spec.result_back_turned;
                fighter.move.reset();
                fighter.heat_dash_from.reset();
                fighter.action_frame = universal_duration(FullUniversal::Idle, config_);
                if (spec.result_crouching) {
                    fighter.posture = FullPosture::Crouching;
                    fighter.universal = FullUniversal::Crouch;
                } else if (fighter.posture == FullPosture::Crouching && fighter.stance < 0) {
                    fighter.posture = FullPosture::Standing;
                }
            }
        } else if (fighter.posture == FullPosture::Standing || fighter.posture == FullPosture::Crouching) {
            ++fighter.action_frame;
            // Stances: time limit and periodic pulses while idle in the stance.
            if (const auto* rule = stance_rule(player); rule && fighter.stun == 0) {
                ++fighter.stance_frames;
                if (rule->pulse_interval_frames > 0 && fighter.stance_frames % rule->pulse_interval_frames == 0) {
                    regain(fighter, rule->pulse_recoverable, 0.0, config_.max_health);
                    gain_resource(player, rule->pulse_resource);
                }
                if (rule->max_frames && fighter.stance_frames >= *rule->max_frames) {
                    fighter.stance = -1;
                    fighter.stance_frames = 0;
                }
            }
            switch (fighter.universal) {
                case FullUniversal::WalkForward: fighter.x += forward * config_.walk_speed; break;
                case FullUniversal::WalkBack: fighter.x -= forward * config_.walk_speed; break;
                case FullUniversal::DashForward:
                    if (fighter.action_frame <= config_.dash_frames)
                        fighter.x += forward * config_.dash_distance / config_.dash_frames;
                    break;
                case FullUniversal::HeatDash:
                    if (fighter.action_frame <= config_.heat_dash_frames)
                        fighter.x += forward * config_.heat_dash_distance / config_.heat_dash_frames;
                    break;
                case FullUniversal::Backdash:
                    if (fighter.action_frame <= config_.backdash_frames)
                        fighter.x -= forward * config_.backdash_distance / config_.backdash_frames;
                    break;
                case FullUniversal::SidestepLeft:
                case FullUniversal::SidestepRight:
                    if (fighter.action_frame <= config_.sidestep_frames) {
                        // Axis convention (shared with tracking): a fighter facing +x has its left at -axis.
                        const double side = fighter.universal == FullUniversal::SidestepLeft ? -1.0 : 1.0;
                        fighter.axis += side * forward * config_.sidestep_distance / config_.sidestep_frames;
                    }
                    break;
                default: break;
            }
        }
        if (fighter.stun > 0 && --fighter.stun == 0) fighter.in_blockstun = false;
        if (fighter.stun == 0 && fighter.crumpled) {
            fighter.crumpled = false;
            fighter.posture = FullPosture::Grounded;
            fighter.grounded_elapsed = 0;
            fighter.techable = false;
        }
        switch (fighter.posture) {
            case FullPosture::Airborne:
                if (--fighter.posture_frames <= 0) {
                    fighter.posture = FullPosture::Grounded;
                    fighter.grounded_elapsed = 0;
                    fighter.techable = !fighter.tornado_used;
                }
                break;
            case FullPosture::WallSplat:
                if (--fighter.posture_frames <= 0) {
                    fighter.posture = FullPosture::Grounded;
                    fighter.grounded_elapsed = 0;
                    fighter.techable = false;
                }
                break;
            case FullPosture::Grounded:
                ++fighter.grounded_elapsed;
                // Nobody stays down forever: stand up automatically if no wake-up is chosen.
                if (fighter.grounded_elapsed >= 4 * config_.knockdown_frames) {
                    fighter.posture = FullPosture::Wakeup;
                    fighter.posture_frames = config_.wakeup_stand_frames;
                }
                break;
            case FullPosture::Wakeup:
                if (--fighter.posture_frames <= 0) {
                    fighter.posture = FullPosture::Standing;
                    fighter.combo_hits = 0;
                    fighter.tornado_used = false;
                    fighter.wall_splat_used = false;
                    fighter.wall_scaled = false;
                    fighter.action_frame = universal_duration(FullUniversal::Idle, config_);
                    fighter.universal = FullUniversal::Idle;
                }
                break;
            default: break;
        }
    }
    if (!sidestepping) {
        auto& a = state_.fighters[0];
        auto& b = state_.fighters[1];
        const double gap = a.axis - b.axis;
        const double step = std::min(std::abs(gap) / 2.0, config_.axis_realign_per_frame / 2.0);
        a.axis -= std::copysign(step, gap);
        b.axis += std::copysign(step, gap);
    }

    // 3. Walls and body collision.
    const auto clamp_to_stage = [&](FullFighter& fighter) {
        fighter.x = std::clamp(fighter.x, stage.left_wall, stage.right_wall);
    };
    for (auto& fighter : state_.fighters) clamp_to_stage(fighter);
    {
        auto& a = state_.fighters[0];
        auto& b = state_.fighters[1];
        const double minimum = 2.0 * config_.body_radius;
        const double dx = b.x - a.x;
        if (std::abs(a.axis - b.axis) < minimum && std::abs(dx) < minimum) {
            const double push = (minimum - std::abs(dx)) / 2.0;
            const double direction = dx >= 0.0 ? 1.0 : -1.0;
            a.x -= direction * push;
            b.x += direction * push;
            for (auto* fighter : {&a, &b}) {
                const double overflow = fighter->x < stage.left_wall ? stage.left_wall - fighter->x
                    : fighter->x > stage.right_wall ? stage.right_wall - fighter->x : 0.0;
                if (overflow != 0.0) {
                    fighter->x += overflow;
                    (fighter == &a ? b : a).x += overflow;  // the other fighter absorbs it
                }
            }
            for (auto& fighter : state_.fighters) clamp_to_stage(fighter);
        }
    }

    // 4. Decide contacts from one snapshot.
    std::vector<Decision> decisions;
    for (int attacker = 0; attacker < 2; ++attacker) {
        const auto& own = state_.fighters[static_cast<std::size_t>(attacker)];
        const auto& target = state_.fighters[static_cast<std::size_t>(1 - attacker)];
        if (!own.move || own.pending_throw_hit >= 0 || target.throw_break_frames > 0) continue;
        const auto& spec = move(attacker, *own.move);
        const FullMoveSpec* defense = target.move ? &move(1 - attacker, *target.move) : nullptr;
        const int defense_frame = target.action_frame;
        const FullStanceRule* target_rule = target.move ? nullptr : stance_rule(1 - attacker);
        for (std::size_t index = 0; index < spec.hits.size(); ++index) {
            const auto& hit = spec.hits[index];
            if ((own.hits_landed >> index) & 1U) continue;
            if (own.action_frame < hit.first_active_frame ||
                own.action_frame >= hit.first_active_frame + hit.active_frames) continue;
            const double reach = own.installed && spec.install_reach ? *spec.install_reach : hit.reach;
            if (std::abs(target.x - own.x) > reach) continue;
            const double lateral = (target.axis - own.axis) * facing(attacker);
            if (!hit.homing && (lateral < -hit.tracking_left || lateral > hit.tracking_right)) continue;
            const bool target_airborne = target.posture == FullPosture::Airborne ||
                (defense && defense->airborne.contains(defense_frame));
            if (target.posture == FullPosture::Grounded && !(hit.ground_hit || is_low(hit.level))) continue;
            if (target.posture == FullPosture::Crouching &&
                (hit.level == FullHitLevel::High || hit.level == FullHitLevel::Throw)) continue;
            if (target_airborne && hit.level == FullHitLevel::Throw) continue;

            Decision decision{attacker, *own.move, target.move, index, FullContact::Hit};
            const bool high_or_mid_strike = is_strike(hit.level) && !is_low(hit.level);
            if (defense && defense->invincible.contains(defense_frame)) {
                decision.contact = FullContact::Evaded;
            } else if (defense &&
                       (((hit.level == FullHitLevel::High || hit.level == FullHitLevel::Throw) &&
                         defense->high_crush.contains(defense_frame)) ||
                        (is_low(hit.level) && defense->low_crush.contains(defense_frame)))) {
                decision.contact = FullContact::Crushed;
            } else if (defense && high_or_mid_strike && !hit.reversal_break &&
                       defense->reversal.contains(defense_frame)) {
                decision.contact = FullContact::Reversed;
            } else if (defense && !hit.unparryable &&
                       (((defense->parry_levels & level_class(hit.level)) && defense->parry.contains(defense_frame)) ||
                        (target.move_in_heat && (defense->heat_parry_levels & level_class(hit.level)) &&
                         defense->heat_parry.contains(defense_frame)))) {
                decision.contact = FullContact::Parried;
            } else if (!defense && !hit.unparryable && target_rule && target.stun == 0 &&
                       target.posture == FullPosture::Standing && (target_rule->auto_parry & level_class(hit.level))) {
                decision.contact = FullContact::Parried;  // automatic stance parry (Jun: GEN against lows and throws)
            } else if (defense && is_strike(hit.level) &&
                       (defense->armor.contains(defense_frame) ||
                        (high_or_mid_strike && defense->power_crush.contains(defense_frame)))) {
                decision.contact = FullContact::Armored;
            } else {
                const bool stance_guard = (!target_rule || target_rule->can_guard) && !target.back_turned &&
                    target.ki_charge_frames == 0;
                const bool standing_guard = stance_guard && target.posture == FullPosture::Standing && !target.move &&
                    (target.universal == FullUniversal::Idle || target.universal == FullUniversal::WalkBack) &&
                    (target.stun == 0 || target.in_blockstun);
                const bool crouching_guard = stance_guard && target.posture == FullPosture::Crouching && !target.move &&
                    target.universal == FullUniversal::Crouch && (target.stun == 0 || target.in_blockstun);
                const bool blocked = hit.level != FullHitLevel::Throw && !target_airborne &&
                    ((standing_guard && !(hit.level == FullHitLevel::Low)) ||
                     (crouching_guard && hit.level != FullHitLevel::Mid && hit.level != FullHitLevel::High));
                // Getting hit while committed to an attack is a counter-hit.
                const bool counter_hit = !blocked && target.stun == 0 && !target_airborne &&
                    (defense != nullptr || target.ki_charge_frames > 0);
                decision.contact = blocked ? FullContact::Blocked
                    : counter_hit ? FullContact::CounterHit : FullContact::Hit;
                if (hit.level == FullHitLevel::Throw && hit.throw_break != FullThrowBreak::None) {
                    decision.contact = FullContact::ThrowHeld;
                }
            }
            decisions.push_back(decision);
            break;  // at most one new hit per attacker per frame
        }
    }

    // 5. Apply contacts. A held throw whose break window closes unbroken lands
    // first; its stun counts from the thrower's current frame, so hit
    // advantage stays exact however long the window held it.
    for (int player = 0; player < 2; ++player) {
        auto& held = state_.fighters[static_cast<std::size_t>(player)];
        auto& thrower = state_.fighters[static_cast<std::size_t>(1 - player)];
        if (held.throw_break_frames <= 0 || --held.throw_break_frames > 0) continue;
        held.throw_break_attempted = false;
        if (!thrower.move || thrower.pending_throw_hit < 0) continue;
        const auto move_index = *thrower.move;
        const auto hit_index = static_cast<std::size_t>(thrower.pending_throw_hit);
        const auto& spec = move(1 - player, move_index);
        thrower.pending_throw_hit = -1;
        apply_contact(1 - player, move_index, std::nullopt, hit_index, FullContact::Hit,
                      thrower.action_frame - spec.hits[hit_index].first_active_frame, events);
        if (thrower.move && thrower.action_frame >= spec.total_frames()) {
            thrower.stance = spec.result_stance;
            thrower.move.reset();
            thrower.action_frame = universal_duration(FullUniversal::Idle, config_);
        }
    }
    for (const auto& decision : decisions) {
        auto& attacker = state_.fighters[static_cast<std::size_t>(decision.attacker)];
        auto& defender = state_.fighters[static_cast<std::size_t>(1 - decision.attacker)];
        attacker.hits_landed |= 1U << decision.hit_index;
        if (decision.contact == FullContact::ThrowHeld) {
            if (!attacker.move) continue;  // the thrower was hit on the same frame
            attacker.pending_throw_hit = static_cast<int>(decision.hit_index);
            defender.throw_break_frames = config_.throw_break_window;
            defender.throw_break_attempted = false;
            defender.stance = -1;
            defender.move.reset();
            defender.universal = FullUniversal::Idle;
            defender.stun = 0;
            defender.in_blockstun = false;
            defender.heat_dash_from.reset();
            events.contacts.push_back({decision.attacker, decision.hit_index, FullContact::ThrowHeld, 0.0});
            continue;
        }
        apply_contact(decision.attacker, decision.move_index, decision.defense_move, decision.hit_index,
                      decision.contact, 0, events);
    }

    const bool p1_down = state_.fighters[0].health <= 0.0;
    const bool p2_down = state_.fighters[1].health <= 0.0;
    if (p1_down || p2_down) state_.winner = p1_down && p2_down ? 2 : (p1_down ? 1 : 0);
    ++state_.frame;
    return events;
}

}  // namespace t8::v2
