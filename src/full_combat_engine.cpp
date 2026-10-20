#include "t8_v2/full_combat_engine.hpp"

#include "t8_v2/full_combat_core.hpp"

#include <algorithm>
#include <limits>
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
    return static_cast<FullLevelClass>(full_core::level_class_bits(level));
}

std::size_t level_class_index(FullHitLevel level) noexcept {
    return static_cast<std::size_t>(full_core::level_index(level));
}

namespace {

full_core::Window to_window(const FrameWindow& window, const std::string& owner) {
    if (window.spans.size() > static_cast<std::size_t>(full_core::kMaxWindowSpans)) {
        throw std::invalid_argument(owner + " has more than " + std::to_string(full_core::kMaxWindowSpans) +
                                    " spans in one frame window");
    }
    full_core::Window result{};
    result.count = static_cast<int>(window.spans.size());
    for (std::size_t index = 0; index < window.spans.size(); ++index) {
        result.first[index] = window.spans[index].first;
        result.last[index] = window.spans[index].last;
    }
    return result;
}

int outcome_index(const std::vector<FullMoveSpec>& moveset, const std::string& stable_id) {
    if (stable_id.empty()) return full_core::kNoOutcome;
    for (std::size_t index = 0; index < moveset.size(); ++index) {
        if (moveset[index].stable_id == stable_id) return static_cast<int>(index);
    }
    return full_core::kUnknownOutcome;
}

void validate(const std::array<std::vector<FullMoveSpec>, 2>& movesets, const FullEngineConfig& config,
              const std::array<FullCharacterRules, 2>& rules) {
    if (!(config.half_width > 0.0) || !(config.body_radius >= 0.0) || config.dash_frames <= 0 ||
        config.backdash_frames <= 0 || config.sidestep_frames <= 0 || config.launch_air_frames <= 0 ||
        config.throw_break_window <= 0 || config.heat_dash_frames <= 0 || config.heat_duration_frames <= 0 ||
        !(config.max_health > 0.0) || config.rage_health_fraction < 0.0 || config.rage_health_fraction > 1.0) {
        throw std::invalid_argument("full combat engine configuration is invalid");
    }
    for (const auto& moveset : movesets) {
        if (moveset.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 2)) {
            throw std::invalid_argument("moveset is too large");
        }
        for (const auto& move : moveset) {
            // Stance entries have no hits; reactive outcomes need the hit they deal.
            if (move.hits.size() > static_cast<std::size_t>(full_core::kMaxHits) || move.recovery < 0 ||
                (move.hits.empty() && move.recovery <= 0) || (move.reactive && move.hits.empty())) {
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
    for (const auto& rule : rules) {
        if (rule.resource_max < 0.0 || (rule.install_threshold && *rule.install_threshold > rule.resource_max)) {
            throw std::invalid_argument("character resource rules are invalid");
        }
        for (const auto& stance : rule.stances) {
            if (stance.pulse_interval_frames < 0 || (stance.max_frames && *stance.max_frames <= 0)) {
                throw std::invalid_argument("stance " + stance.name + " has invalid timing");
            }
        }
    }
}

full_core::Move compile_move(const FullMoveSpec& spec, const std::vector<FullMoveSpec>& moveset, int hit_offset) {
    using namespace full_core;
    Move move{};
    const auto flag = [&](bool value, MoveFlag bit) {
        if (value) move.flags |= bit;
    };
    flag(spec.reactive, kReactive);
    flag(spec.requires_sidestep, kRequiresSidestep);
    flag(spec.requires_back_to_wall, kRequiresBackToWall);
    flag(spec.requires_back_turned, kRequiresBackTurned);
    flag(spec.requires_opponent_back_turned, kRequiresOpponentBackTurned);
    flag(spec.requires_opponent_left_side, kRequiresOpponentLeft);
    flag(spec.requires_opponent_right_side, kRequiresOpponentRight);
    flag(spec.result_back_turned, kResultBackTurned);
    flag(spec.result_crouching, kResultCrouching);
    flag(spec.attack_throw_front_only, kAttackThrowFrontOnly);
    flag(spec.attack_throw_standing_only, kAttackThrowStandingOnly);
    flag(spec.attack_throw_airborne, kAttackThrowAirborne);
    flag(spec.cannot_ko, kCannotKo);
    flag(spec.ki_charge, kKiCharge);
    flag(spec.requires_heat, kRequiresHeat);
    flag(spec.requires_rage, kRequiresRage);
    flag(spec.engages_heat, kEngagesHeat);
    flag(spec.heat_engager, kHeatEngager);
    flag(spec.consumes_heat, kConsumesHeat);
    flag(spec.consumes_rage, kConsumesRage);
    flag(spec.recoverable_only, kRecoverableOnly);
    flag(spec.removes_recoverable, kRemovesRecoverable);
    flag(spec.armor_damage_recoverable, kArmorDamageRecoverable);
    flag(spec.self_damage_without_heat_only, kSelfDamageWithoutHeatOnly);
    flag(spec.side_switch_on_hit, kSideSwitchOnHit);
    flag(spec.side_switch_on_break, kSideSwitchOnBreak);
    flag(spec.counter_hit_advantage.has_value(), kHasCounterHitAdvantage);
    flag(spec.counter_hit_effect.has_value(), kHasCounterHitEffect);
    flag(spec.launch_air_frames.has_value(), kHasLaunchAirFrames);
    flag(spec.back_turned_hit_advantage.has_value(), kHasBackTurnedHitAdvantage);
    flag(spec.rage_art_max_damage.has_value(), kHasRageArtMax);
    flag(spec.heat_dash_block_advantage.has_value(), kHasHeatDashBlock);
    flag(spec.heat_dash_hit_advantage.has_value(), kHasHeatDashHit);
    flag(spec.install_chip_damage.has_value(), kHasInstallChip);
    flag(spec.install_reach.has_value(), kHasInstallReach);
    move.hit_offset = hit_offset;
    move.hit_count = static_cast<int>(spec.hits.size());
    move.total_frames = spec.total_frames();
    move.recovery = spec.recovery;
    move.block_advantage = spec.block_advantage;
    move.hit_advantage = spec.hit_advantage;
    move.counter_hit_advantage = spec.counter_hit_advantage.value_or(0);
    move.launch_air_frames = spec.launch_air_frames.value_or(0);
    move.back_turned_hit_advantage = spec.back_turned_hit_advantage.value_or(0);
    move.heat_dash_block_advantage = spec.heat_dash_block_advantage.value_or(0);
    move.heat_dash_hit_advantage = spec.heat_dash_hit_advantage.value_or(0);
    move.heat_cost_frames = spec.heat_cost_frames;
    move.required_stance = spec.required_stance;
    move.result_stance = spec.result_stance;
    move.result_stance_on_hit = spec.result_stance_on_hit;
    move.result_stance_on_block = spec.result_stance_on_block;
    for (std::size_t level = 0; level < 4; ++level) {
        move.parry_outcomes[level] = outcome_index(moveset, spec.parry_outcomes[level]);
    }
    move.heat_parry_outcome = outcome_index(moveset, spec.heat_parry_outcome);
    move.hit_effect = spec.hit_effect;
    move.counter_hit_effect = spec.counter_hit_effect.value_or(FullHitEffect::Stun);
    move.back_turned_hit_effect = spec.back_turned_hit_effect;
    move.heat_dash_hit_effect = spec.heat_dash_hit_effect;
    move.required_posture = spec.required_posture;
    move.attack_throw = spec.attack_throw;
    move.parry_levels = spec.parry_levels;
    move.heat_parry_levels = spec.heat_parry_levels;
    move.travel = spec.travel;
    move.pushback_hit = spec.pushback_hit;
    move.pushback_block = spec.pushback_block;
    move.reversal_damage = spec.reversal_damage;
    move.attack_throw_damage = spec.attack_throw_damage;
    move.recoverable_damage = spec.recoverable_damage;
    move.rage_art_max_damage = spec.rage_art_max_damage.value_or(0.0);
    move.self_damage = spec.self_damage;
    move.self_recoverable = spec.self_recoverable;
    move.restore_health_hit = spec.restore_health_hit;
    move.restore_recoverable_hit = spec.restore_recoverable_hit;
    move.restore_recoverable_block = spec.restore_recoverable_block;
    move.resource_gain_start = spec.resource_gain_start;
    move.resource_gain_hit = spec.resource_gain_hit;
    move.resource_gain_airborne_hit = spec.resource_gain_airborne_hit;
    move.resource_gain_block = spec.resource_gain_block;
    move.resource_gain_heat_activation = spec.resource_gain_heat_activation;
    move.install_damage_bonus = spec.install_damage_bonus;
    move.install_chip_damage = spec.install_chip_damage.value_or(0.0);
    move.install_reach = spec.install_reach.value_or(0.0);
    move.high_crush = to_window(spec.high_crush, spec.stable_id);
    move.low_crush = to_window(spec.low_crush, spec.stable_id);
    move.power_crush = to_window(spec.power_crush, spec.stable_id);
    move.armor = to_window(spec.armor, spec.stable_id);
    move.parry = to_window(spec.parry, spec.stable_id);
    move.reversal = to_window(spec.reversal, spec.stable_id);
    move.invincible = to_window(spec.invincible, spec.stable_id);
    move.airborne = to_window(spec.airborne, spec.stable_id);
    move.heat_parry = to_window(spec.heat_parry, spec.stable_id);
    return move;
}

full_core::Hit compile_hit(const FullHitSpec& spec) {
    using namespace full_core;
    Hit hit{};
    hit.level = spec.level;
    hit.throw_break = spec.throw_break;
    const auto flag = [&](bool value, HitFlag bit) {
        if (value) hit.flags = static_cast<std::uint16_t>(hit.flags | bit);
    };
    flag(spec.homing, kHoming);
    flag(spec.ground_hit, kGroundHit);
    flag(spec.wall_splat, kWallSplat);
    flag(spec.wall_break, kWallBreak);
    flag(spec.floor_break, kFloorBreak);
    flag(spec.balcony_break, kBalconyBreak);
    flag(spec.tornado, kTornado);
    flag(spec.spike, kSpike);
    flag(spec.unparryable, kUnparryable);
    flag(spec.reversal_break, kReversalBreak);
    flag(spec.heat_chip_damage.has_value(), kHasHeatChip);
    hit.first_active_frame = spec.first_active_frame;
    hit.active_frames = spec.active_frames;
    hit.damage = spec.damage;
    hit.reach = spec.reach;
    hit.tracking_left = spec.tracking_left;
    hit.tracking_right = spec.tracking_right;
    hit.chip_damage = spec.chip_damage;
    hit.heat_chip_damage = spec.heat_chip_damage.value_or(0.0);
    return hit;
}

}  // namespace

full_core::Tables FullCompiledTables::view() const {
    full_core::Tables tables{};
    for (std::size_t player = 0; player < 2; ++player) {
        tables.side[player].moves = moves[player].data();
        tables.side[player].move_count = static_cast<int>(moves[player].size());
        tables.side[player].hits = hits[player].data();
        tables.side[player].stances = stances[player].data();
        tables.side[player].stance_count = static_cast<int>(stances[player].size());
        tables.side[player].rules = rules[player];
    }
    tables.config = config;
    return tables;
}

FullCompiledTables compile_full_tables(const std::array<std::vector<FullMoveSpec>, 2>& movesets,
                                       const FullEngineConfig& config,
                                       const std::array<FullCharacterRules, 2>& rules) {
    validate(movesets, config, rules);
    FullCompiledTables compiled{};
    compiled.config = config;
    for (std::size_t player = 0; player < 2; ++player) {
        const auto& moveset = movesets[player];
        for (const auto& spec : moveset) {
            compiled.moves[player].push_back(
                compile_move(spec, moveset, static_cast<int>(compiled.hits[player].size())));
            for (const auto& hit : spec.hits) compiled.hits[player].push_back(compile_hit(hit));
        }
        const auto& rule = rules[player];
        for (const auto& stance : rule.stances) {
            full_core::Stance compiled_stance{};
            compiled_stance.can_guard = stance.can_guard;
            compiled_stance.has_max_frames = stance.max_frames.has_value();
            compiled_stance.max_frames = stance.max_frames.value_or(0);
            compiled_stance.auto_parry = stance.auto_parry;
            for (std::size_t level = 0; level < 4; ++level) {
                compiled_stance.parry_outcomes[level] = outcome_index(moveset, stance.parry_outcomes[level]);
            }
            compiled_stance.pulse_interval_frames = stance.pulse_interval_frames;
            compiled_stance.pulse_recoverable = stance.pulse_recoverable;
            compiled_stance.pulse_resource = stance.pulse_resource;
            compiled.stances[player].push_back(compiled_stance);
        }
        compiled.rules[player] = {rule.resource_max, rule.install_threshold.has_value(),
                                  rule.install_threshold.value_or(0.0), rule.resource_persists,
                                  rule.install_consumes};
    }
    return compiled;
}

full_core::State to_core_state(const FullEngineState& state) {
    full_core::State result{};
    for (std::size_t player = 0; player < 2; ++player) {
        const FullFighter& f = state.fighters[player];
        full_core::Fighter& c = result.fighters[player];
        c.health = f.health;
        c.recoverable_health = f.recoverable_health;
        c.x = f.x;
        c.axis = f.axis;
        c.resource = f.resource;
        c.posture = f.posture;
        c.universal = f.universal;
        c.heat_dash_from = f.heat_dash_from ? static_cast<std::int8_t>(*f.heat_dash_from) : full_core::kNoContact;
        c.heat = f.heat;
        c.heat_available = f.heat_available;
        c.rage = f.rage;
        c.rage_available = f.rage_available;
        c.throw_break_attempted = f.throw_break_attempted;
        c.installed = f.installed;
        c.move_hit = f.move_hit;
        c.move_blocked = f.move_blocked;
        c.resource_gained = f.resource_gained;
        c.back_turned = f.back_turned;
        c.move_in_heat = f.move_in_heat;
        c.ki_charge_boost = f.ki_charge_boost;
        c.move_boosted = f.move_boosted;
        c.in_blockstun = f.in_blockstun;
        c.tornado_used = f.tornado_used;
        c.wall_splat_used = f.wall_splat_used;
        c.wall_scaled = f.wall_scaled;
        c.techable = f.techable;
        c.crumpled = f.crumpled;
        c.stance = f.stance;
        c.heat_frames = f.heat_frames;
        c.throw_break_frames = f.throw_break_frames;
        c.pending_throw_hit = f.pending_throw_hit;
        c.pending_throw_elapsed = f.pending_throw_elapsed;
        c.move = f.move ? static_cast<int>(*f.move) : -1;
        c.action_frame = f.action_frame;
        c.hits_landed = f.hits_landed;
        c.stance_frames = f.stance_frames;
        c.ki_charge_frames = f.ki_charge_frames;
        c.stun = f.stun;
        c.posture_frames = f.posture_frames;
        c.grounded_elapsed = f.grounded_elapsed;
        c.combo_hits = f.combo_hits;
    }
    result.stage = state.stage;
    result.frame = state.frame;
    result.winner = state.winner;
    return result;
}

FullEngineState from_core_state(const full_core::State& state) {
    FullEngineState result{};
    for (std::size_t player = 0; player < 2; ++player) {
        const full_core::Fighter& c = state.fighters[player];
        FullFighter& f = result.fighters[player];
        f.health = c.health;
        f.recoverable_health = c.recoverable_health;
        f.x = c.x;
        f.axis = c.axis;
        f.resource = c.resource;
        f.posture = c.posture;
        f.universal = c.universal;
        if (c.heat_dash_from != full_core::kNoContact) f.heat_dash_from = static_cast<FullContact>(c.heat_dash_from);
        f.heat = c.heat;
        f.heat_available = c.heat_available;
        f.rage = c.rage;
        f.rage_available = c.rage_available;
        f.throw_break_attempted = c.throw_break_attempted;
        f.installed = c.installed;
        f.move_hit = c.move_hit;
        f.move_blocked = c.move_blocked;
        f.resource_gained = c.resource_gained;
        f.back_turned = c.back_turned;
        f.move_in_heat = c.move_in_heat;
        f.ki_charge_boost = c.ki_charge_boost;
        f.move_boosted = c.move_boosted;
        f.in_blockstun = c.in_blockstun;
        f.tornado_used = c.tornado_used;
        f.wall_splat_used = c.wall_splat_used;
        f.wall_scaled = c.wall_scaled;
        f.techable = c.techable;
        f.crumpled = c.crumpled;
        f.stance = c.stance;
        f.heat_frames = c.heat_frames;
        f.throw_break_frames = c.throw_break_frames;
        f.pending_throw_hit = c.pending_throw_hit;
        f.pending_throw_elapsed = c.pending_throw_elapsed;
        if (c.move >= 0) f.move = static_cast<std::size_t>(c.move);
        f.action_frame = c.action_frame;
        f.hits_landed = c.hits_landed;
        f.stance_frames = c.stance_frames;
        f.ki_charge_frames = c.ki_charge_frames;
        f.stun = c.stun;
        f.posture_frames = c.posture_frames;
        f.grounded_elapsed = c.grounded_elapsed;
        f.combo_hits = c.combo_hits;
    }
    result.stage = state.stage;
    result.frame = state.frame;
    result.winner = state.winner;
    return result;
}

FullFrameEvents from_core_events(const full_core::Events& events) {
    FullFrameEvents result{};
    for (int index = 0; index < events.contact_count; ++index) {
        const auto& contact = events.contacts[index];
        result.contacts.push_back({contact.attacker, static_cast<std::size_t>(contact.hit_index), contact.contact,
                                   contact.damage});
    }
    for (int index = 0; index < events.transition_count; ++index) result.transitions.push_back(events.transitions[index]);
    result.heat_activated = {events.heat_activated[0], events.heat_activated[1]};
    result.rage_activated = {events.rage_activated[0], events.rage_activated[1]};
    return result;
}

full_core::Input to_core_input(const FullInput& input, std::size_t move_count) {
    full_core::Input result{};
    result.universal = input.universal;
    if (input.move) {
        // An out-of-range index stays out of range (and so illegal) in the compact form.
        result.move = *input.move < move_count ? static_cast<int>(*input.move)
                                               : static_cast<int>(std::min<std::size_t>(move_count, std::numeric_limits<int>::max()));
    }
    return result;
}

FullCombatEngine::FullCombatEngine(std::array<std::vector<FullMoveSpec>, 2> movesets, FullEngineConfig config,
                                   std::array<FullCharacterRules, 2> rules)
    : movesets_(std::move(movesets)), config_(config), rules_(std::move(rules)),
      compiled_(std::make_shared<const FullCompiledTables>(compile_full_tables(movesets_, config_, rules_))) {
    reset();
}

void FullCombatEngine::reset(double p1_x, double p2_x) {
    full_core::State state{};
    full_core::reset(compiled_->view(), state, p1_x, p2_x);
    state_ = from_core_state(state);
}

void FullCombatEngine::next_round(double p1_x, double p2_x) {
    auto state = to_core_state(state_);
    full_core::next_round(compiled_->view(), state, p1_x, p2_x);
    state_ = from_core_state(state);
}

const FullMoveSpec& FullCombatEngine::move(int player, std::size_t index) const {
    return movesets_.at(static_cast<std::size_t>(player)).at(index);
}

bool FullCombatEngine::actionable(int player) const {
    if (player != 0 && player != 1) throw std::out_of_range("player must be 0 or 1");
    return full_core::actionable(compiled_->view(), to_core_state(state_), player);
}

bool FullCombatEngine::legal(int player, const FullInput& input) const {
    if (player != 0 && player != 1) throw std::out_of_range("player must be 0 or 1");
    return full_core::legal(compiled_->view(), to_core_state(state_), player,
                            to_core_input(input, movesets_[static_cast<std::size_t>(player)].size()));
}

FullFrameEvents FullCombatEngine::step(const FullInput& p1, const FullInput& p2) {
    auto state = to_core_state(state_);
    full_core::Events events{};
    full_core::step(compiled_->view(), state, to_core_input(p1, movesets_[0].size()),
                    to_core_input(p2, movesets_[1].size()), events);
    state_ = from_core_state(state);
    return from_core_events(events);
}

}  // namespace t8::v2
