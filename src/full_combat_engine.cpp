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
    }
    return "unknown";
}

namespace {

bool is_strike(FullHitLevel level) noexcept { return level != FullHitLevel::Throw; }
bool is_low(FullHitLevel level) noexcept {
    return level == FullHitLevel::Low || level == FullHitLevel::SpecialLow;
}

int universal_duration(FullUniversal action, const FullEngineConfig& config) {
    switch (action) {
        case FullUniversal::DashForward: return config.dash_frames;
        case FullUniversal::Backdash: return config.backdash_frames;
        case FullUniversal::SidestepLeft:
        case FullUniversal::SidestepRight: return config.sidestep_frames;
        case FullUniversal::ThrowBreak: return config.throw_break_window;
        default: return 1;  // Idle, walking, and crouching are re-chosen every frame.
    }
}

// A decided contact, applied after both fighters' contacts are known so a
// same-frame trade resolves from one snapshot.
struct Decision {
    int attacker = 0;
    std::size_t hit_index = 0;
    FullContact contact = FullContact::Hit;
};

}  // namespace

FullCombatEngine::FullCombatEngine(std::array<std::vector<FullMoveSpec>, 2> movesets, FullEngineConfig config)
    : movesets_(std::move(movesets)), config_(config) {
    if (!(config_.half_width > 0.0) || !(config_.body_radius >= 0.0) || config_.dash_frames <= 0 ||
        config_.backdash_frames <= 0 || config_.sidestep_frames <= 0 || config_.launch_air_frames <= 0 ||
        config_.throw_break_window <= 0) {
        throw std::invalid_argument("full combat engine configuration is invalid");
    }
    for (const auto& moveset : movesets_) {
        for (const auto& move : moveset) {
            if (move.hits.empty() || move.hits.size() > 32 || move.recovery < 0) {
                throw std::invalid_argument("move " + move.stable_id + " needs 1-32 hits and non-negative recovery");
            }
            int previous = 0;
            for (const auto& hit : move.hits) {
                if (hit.first_active_frame <= previous || hit.active_frames <= 0 || !(hit.reach > 0.0) ||
                    hit.damage < 0.0 || hit.tracking_left < 0.0 || hit.tracking_right < 0.0) {
                    throw std::invalid_argument("move " + move.stable_id + " has an invalid or unordered hit");
                }
                previous = hit.first_active_frame;
            }
        }
    }
    reset();
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
    if (fighter.stun > 0 || fighter.crumpled) return false;
    if (fighter.posture != FullPosture::Standing && fighter.posture != FullPosture::Crouching) return false;
    if (fighter.move) return fighter.action_frame >= move(player, *fighter.move).total_frames();
    return fighter.action_frame >= universal_duration(fighter.universal, config_);
}

bool FullCombatEngine::legal(int player, const FullInput& input) const {
    const auto& fighter = state_.fighters.at(static_cast<std::size_t>(player));
    if (input.move) {
        if (*input.move >= movesets_[static_cast<std::size_t>(player)].size() || !actionable(player)) return false;
        const auto& spec = move(player, *input.move);
        return spec.required_posture == fighter.posture &&
            (spec.required_stance < 0 || spec.required_stance == fighter.stance) &&
            (!spec.requires_heat || fighter.heat) && (!spec.requires_rage || fighter.rage);
    }
    switch (input.universal) {
        case FullUniversal::WakeupStand:
        case FullUniversal::WakeupRoll:
            return fighter.posture == FullPosture::Grounded && fighter.grounded_elapsed >= config_.knockdown_frames;
        case FullUniversal::TechRoll:
            return fighter.posture == FullPosture::Grounded && fighter.techable &&
                fighter.grounded_elapsed < config_.tech_window_frames;
        default:
            return actionable(player);
    }
}

FullFrameEvents FullCombatEngine::step(const FullInput& p1, const FullInput& p2) {
    FullFrameEvents events{};
    auto& stage = state_.stage;
    const std::array<const FullInput*, 2> inputs = {&p1, &p2};
    const auto facing = [&](int player) {
        const auto& own = state_.fighters[static_cast<std::size_t>(player)];
        const auto& other = state_.fighters[static_cast<std::size_t>(1 - player)];
        if (other.x == own.x) return player == 0 ? 1.0 : -1.0;
        return other.x > own.x ? 1.0 : -1.0;
    };

    // 1. Start new actions.
    for (int player = 0; player < 2; ++player) {
        auto& fighter = state_.fighters[static_cast<std::size_t>(player)];
        const FullInput& input = *inputs[static_cast<std::size_t>(player)];
        if (!legal(player, input)) continue;
        if (input.move) {
            const auto& spec = move(player, *input.move);
            fighter.move = input.move;
            fighter.universal = FullUniversal::Idle;
            fighter.stance = -1;
            if (spec.engages_heat) fighter.heat = true;
            if (spec.consumes_rage) fighter.rage = false;
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
            if (input.universal != FullUniversal::Idle) fighter.stance = -1;
            fighter.posture = input.universal == FullUniversal::Crouch ? FullPosture::Crouching : FullPosture::Standing;
        }
        fighter.action_frame = 0;
        fighter.hits_landed = 0;
    }

    // 2. Advance actions, movement, stun, and posture timers.
    const bool sidestepping = std::any_of(state_.fighters.begin(), state_.fighters.end(), [](const auto& f) {
        return !f.move && (f.universal == FullUniversal::SidestepLeft || f.universal == FullUniversal::SidestepRight);
    });
    for (int player = 0; player < 2; ++player) {
        auto& fighter = state_.fighters[static_cast<std::size_t>(player)];
        const double forward = facing(player);
        if (fighter.move) {
            const auto& spec = move(player, *fighter.move);
            ++fighter.action_frame;
            const int approach_frames = spec.hits.front().first_active_frame - 1;
            if (approach_frames > 0 && fighter.action_frame <= approach_frames) {
                fighter.x += forward * spec.travel / approach_frames;
            }
            if (fighter.action_frame >= spec.total_frames()) {
                fighter.stance = spec.result_stance;
                fighter.move.reset();
                fighter.action_frame = universal_duration(FullUniversal::Idle, config_);
                if (fighter.posture == FullPosture::Crouching && fighter.stance < 0) {
                    fighter.posture = FullPosture::Standing;
                }
            }
        } else if (fighter.posture == FullPosture::Standing || fighter.posture == FullPosture::Crouching) {
            ++fighter.action_frame;
            switch (fighter.universal) {
                case FullUniversal::WalkForward: fighter.x += forward * config_.walk_speed; break;
                case FullUniversal::WalkBack: fighter.x -= forward * config_.walk_speed; break;
                case FullUniversal::DashForward:
                    if (fighter.action_frame <= config_.dash_frames)
                        fighter.x += forward * config_.dash_distance / config_.dash_frames;
                    break;
                case FullUniversal::Backdash:
                    if (fighter.action_frame <= config_.backdash_frames)
                        fighter.x -= forward * config_.backdash_distance / config_.backdash_frames;
                    break;
                case FullUniversal::SidestepLeft:
                case FullUniversal::SidestepRight:
                    if (fighter.action_frame <= config_.sidestep_frames) {
                        const double side = fighter.universal == FullUniversal::SidestepLeft ? -1.0 : 1.0;
                        fighter.axis += side * config_.sidestep_distance / config_.sidestep_frames;
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
        if (!own.move) continue;
        const auto& spec = move(attacker, *own.move);
        const FullMoveSpec* defense = target.move ? &move(1 - attacker, *target.move) : nullptr;
        const int defense_frame = target.action_frame;
        for (std::size_t index = 0; index < spec.hits.size(); ++index) {
            const auto& hit = spec.hits[index];
            if ((own.hits_landed >> index) & 1U) continue;
            if (own.action_frame < hit.first_active_frame ||
                own.action_frame >= hit.first_active_frame + hit.active_frames) continue;
            if (std::abs(target.x - own.x) > hit.reach) continue;
            const double lateral = (target.axis - own.axis) * facing(attacker);
            if (!hit.homing && (lateral < -hit.tracking_left || lateral > hit.tracking_right)) continue;
            const bool target_airborne = target.posture == FullPosture::Airborne ||
                (defense && defense->airborne.contains(defense_frame));
            if (target.posture == FullPosture::Grounded && !(hit.ground_hit || is_low(hit.level))) continue;
            if (target.posture == FullPosture::Crouching &&
                (hit.level == FullHitLevel::High || hit.level == FullHitLevel::Throw)) continue;
            if (target_airborne && hit.level == FullHitLevel::Throw) continue;

            Decision decision{attacker, index, FullContact::Hit};
            const bool defending_move = defense != nullptr;
            if (defending_move && defense->invincible.contains(defense_frame)) {
                decision.contact = FullContact::Evaded;
            } else if (defending_move &&
                       (((hit.level == FullHitLevel::High || hit.level == FullHitLevel::Throw) &&
                         defense->high_crush.contains(defense_frame)) ||
                        (is_low(hit.level) && defense->low_crush.contains(defense_frame)))) {
                decision.contact = FullContact::Crushed;
            } else if (hit.level == FullHitLevel::Throw && hit.throw_breakable && !target.move &&
                       target.universal == FullUniversal::ThrowBreak) {
                decision.contact = FullContact::ThrowBroken;
            } else if (defending_move && is_strike(hit.level) && !is_low(hit.level) &&
                       defense->parry.contains(defense_frame)) {
                decision.contact = FullContact::Parried;
            } else if (defending_move && is_strike(hit.level) && !is_low(hit.level) &&
                       defense->power_crush.contains(defense_frame)) {
                decision.contact = FullContact::Armored;
            } else {
                const bool standing_guard = target.posture == FullPosture::Standing && !target.move &&
                    (target.universal == FullUniversal::Idle || target.universal == FullUniversal::WalkBack ||
                     target.universal == FullUniversal::ThrowBreak) &&
                    (target.stun == 0 || target.in_blockstun);
                const bool crouching_guard = target.posture == FullPosture::Crouching && !target.move &&
                    target.universal == FullUniversal::Crouch && (target.stun == 0 || target.in_blockstun);
                const bool blocked = hit.level != FullHitLevel::Throw && !target_airborne &&
                    ((standing_guard && !(hit.level == FullHitLevel::Low)) ||
                     (crouching_guard && hit.level != FullHitLevel::Mid && hit.level != FullHitLevel::High));
                // Getting hit while committed to an attack is a counter-hit.
                const bool counter_hit = !blocked && defending_move && target.stun == 0 && !target_airborne;
                decision.contact = blocked ? FullContact::Blocked
                    : counter_hit ? FullContact::CounterHit : FullContact::Hit;
            }
            decisions.push_back(decision);
            break;  // at most one new hit per attacker per frame
        }
    }

    // 5. Apply contacts.
    for (const auto& decision : decisions) {
        const int attacker_index = decision.attacker;
        auto& attacker = state_.fighters[static_cast<std::size_t>(attacker_index)];
        auto& defender = state_.fighters[static_cast<std::size_t>(1 - attacker_index)];
        const auto& spec = move(attacker_index, *attacker.move);
        const auto& hit = spec.hits[decision.hit_index];
        attacker.hits_landed |= 1U << decision.hit_index;
        FullContactEvent event{attacker_index, decision.hit_index, decision.contact, 0.0};
        const double direction = facing(attacker_index);
        // Published advantage assumes contact on the hit's first active frame;
        // stun is fixed from there, so a later (meaty) contact gains frames.
        const int remaining = std::max(0, spec.total_frames() - hit.first_active_frame);
        const auto push_defender = [&](double distance) {
            defender.x += direction * distance;
            const double wall = direction > 0.0 ? stage.right_wall : stage.left_wall;
            const double overflow = direction > 0.0 ? defender.x - wall : wall - defender.x;
            if (overflow > 0.0) {
                defender.x = wall;
                attacker.x -= direction * overflow;  // pushback against a wall moves the attacker
            }
        };
        switch (decision.contact) {
            case FullContact::Evaded:
            case FullContact::Crushed:
                break;
            case FullContact::ThrowBroken:
                attacker.move.reset();
                attacker.action_frame = 0;
                attacker.universal = FullUniversal::Idle;
                attacker.stun = defender.stun = 0;
                push_defender(0.4);
                break;
            case FullContact::Parried:
                attacker.move.reset();
                attacker.universal = FullUniversal::Idle;
                attacker.action_frame = 0;
                attacker.stun = config_.parry_stun_frames;
                break;
            case FullContact::Blocked:
                defender.stun = std::max(0, remaining + spec.block_advantage);
                defender.in_blockstun = defender.stun > 0;
                push_defender(spec.pushback_block);
                break;
            case FullContact::Armored:
            case FullContact::Hit:
            case FullContact::CounterHit: {
                const bool counter = decision.contact == FullContact::CounterHit;
                const bool juggle = defender.posture == FullPosture::Airborne || defender.posture == FullPosture::WallSplat;
                double damage = hit.damage * (counter ? config_.counter_hit_damage_scale : 1.0);
                if (defender.combo_hits > 0) {
                    damage *= std::max(config_.combo_scale_floor,
                                       1.0 - config_.combo_scale_step * defender.combo_hits);
                }
                defender.health = std::max(0.0, defender.health - damage);
                event.damage = damage;
                if (decision.contact == FullContact::Armored) break;  // armor: damage only
                defender.move.reset();
                defender.universal = FullUniversal::Idle;
                defender.action_frame = 0;
                defender.in_blockstun = false;
                const FullHitEffect effect = counter && spec.counter_hit_effect
                    ? *spec.counter_hit_effect : spec.hit_effect;
                const int advantage = counter && spec.counter_hit_advantage
                    ? *spec.counter_hit_advantage : spec.hit_advantage;
                const bool last_hit = decision.hit_index + 1 == spec.hits.size();
                if (juggle) {
                    ++defender.combo_hits;
                    if (defender.posture == FullPosture::Airborne) {
                        const double refresh = config_.juggle_refresh_frames *
                            std::pow(config_.juggle_refresh_decay, defender.combo_hits - 1);
                        defender.posture_frames = std::max(defender.posture_frames, static_cast<int>(refresh));
                        if (hit.tornado && !defender.tornado_used) {
                            defender.tornado_used = true;
                            defender.posture_frames += config_.tornado_extra_frames;
                        }
                    }
                } else if (defender.posture == FullPosture::Grounded) {
                    ++defender.combo_hits;
                    defender.grounded_elapsed = 0;  // hits on the ground keep the fighter down
                } else if (last_hit && effect == FullHitEffect::Launch) {
                    defender.posture = FullPosture::Airborne;
                    defender.posture_frames = config_.launch_air_frames;
                    defender.combo_hits = 1;
                    defender.stun = 0;
                } else if (last_hit && effect == FullHitEffect::Knockdown) {
                    defender.posture = FullPosture::Grounded;
                    defender.grounded_elapsed = 0;
                    defender.techable = true;
                    defender.stun = 0;
                } else if (last_hit && effect == FullHitEffect::Crumple) {
                    defender.stun = config_.crumple_frames;
                    defender.crumpled = true;
                } else {
                    defender.stun = std::max(0, remaining + advantage);
                    if (defender.posture == FullPosture::Crouching) defender.posture = FullPosture::Standing;
                }
                push_defender(spec.pushback_hit);

                // Walls and stage transitions, on the side the defender was driven toward.
                const bool right_side = direction > 0.0;
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
                    defender.posture_frames = config_.wall_splat_frames;
                    defender.wall_splat_used = true;
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
    ++state_.frame;
    return events;
}

}  // namespace t8::v2
