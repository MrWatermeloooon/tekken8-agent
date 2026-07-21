#include "t8_v2/sim.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace t8::v2 {
namespace {

constexpr int move_index_for_action(Action action) noexcept {
    switch (action) {
        case Action::Jab: return 0;
        case Action::Df1: return 1;
        case Action::F2: return 2;
        case Action::Db3: return 3;
        case Action::Hopkick: return 4;
        case Action::Throw: return 5;
        default: return -1;
    }
}

constexpr bool is_lateral(Action action) noexcept {
    return action == Action::SidestepLeft || action == Action::SidestepRight ||
           action == Action::SidewalkLeft || action == Action::SidewalkRight;
}

constexpr bool is_forward(Action action) noexcept {
    return action == Action::WalkForward || action == Action::DashForward;
}

constexpr bool is_late_passive(Action action) noexcept {
    return action == Action::Neutral || action == Action::WalkBack || action == Action::DashBack;
}

constexpr bool is_no_action_passive(Action action) noexcept {
    return action == Action::Neutral || action == Action::Stand || action == Action::Crouch;
}

constexpr bool is_blocked(HitLevel guard, HitLevel attack) noexcept {
    if (attack == HitLevel::Throw) {
        return false;
    }
    if (guard == HitLevel::Low) {
        return attack == HitLevel::Low;
    }
    if (guard == HitLevel::Mid) {
        return attack == HitLevel::High || attack == HitLevel::Mid;
    }
    return false;
}

constexpr bool is_throw_break(Action action) noexcept {
    return action == Action::ThrowBreak1 || action == Action::ThrowBreak2 || action == Action::ThrowBreak12;
}

}  // namespace

struct Simulator::AttackCheck {
    bool in_range = false;
    bool blocked = false;
    double damage = 0.0;
    bool throw_broken = false;
};

struct Simulator::FrameInfo {
    double damage_to_p1 = 0.0;
    double damage_to_p2 = 0.0;
    double p1_throw_damage = 0.0;
    double p2_throw_damage = 0.0;
    bool p1_throw_break = false;
    bool p2_throw_break = false;
    double p1_whiff_punish_bonus = 0.0;
    double p2_whiff_punish_bonus = 0.0;
    bool p1_whiff = false;
    bool p2_whiff = false;
    bool p1_block = false;
    bool p2_block = false;
};

double State::distance() const noexcept {
    return std::abs(p2.x - p1.x);
}

Simulator::Simulator(Config config) : config_(config), state_(initial_state()) {}

const State& Simulator::reset(std::uint64_t seed) noexcept {
    state_ = initial_state(seed);
    return state_;
}

State Simulator::initial_state(std::uint64_t seed) const noexcept {
    State state{};
    state.p1 = FighterRuntime{};
    state.p2 = FighterRuntime{};
    state.p1.health = config_.max_health;
    state.p2.health = config_.max_health;
    state.p1.x = -0.85;
    state.p2.x = 0.85;
    if (config_.randomize_initial_positions) {
        const auto random_unit = [](std::uint64_t value) {
            value += 0x9e3779b97f4a7c15ULL;
            value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
            value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
            value ^= value >> 31;
            return static_cast<double>((value >> 40) & 0xFFFFFFULL) / 16777216.0;
        };
        const double distance = config_.initial_distance_min +
            (config_.initial_distance_max - config_.initial_distance_min) * random_unit(seed);
        const double allowed_center = std::max(0.0, std::min(
            config_.initial_center_jitter,
            config_.stage_half_width - config_.body_radius - distance * 0.5));
        const double center = (random_unit(seed ^ 0xd1b54a32d192ed03ULL) * 2.0 - 1.0) * allowed_center;
        state.p1.x = center - distance * 0.5;
        state.p2.x = center + distance * 0.5;
    }
    return state;
}

FighterRuntime Simulator::start_action(FighterRuntime fighter, Action action) const noexcept {
    if (fighter.busy()) {
        return fighter;
    }
    if (action == Action::BlockHigh || action == Action::Stand || action == Action::Neutral) {
        fighter.guard = HitLevel::Mid;
        fighter.throw_break_active = 0;
        return fighter;
    }
    if (action == Action::BlockLow || action == Action::Crouch || action == Action::LowParry) {
        fighter.guard = HitLevel::Low;
        fighter.throw_break_active = 0;
        return fighter;
    }
    if (action == Action::Jump) {
        fighter.guard = HitLevel::None;
        fighter.airborne = config_.jump_frames;
        fighter.throw_break_active = 0;
        return fighter;
    }
    if (is_throw_break(action)) {
        fighter.guard = HitLevel::None;
        fighter.throw_break_active = config_.throw_break_frames;
        return fighter;
    }
    const int move = move_index_for_action(action);
    if (move >= 0) {
        fighter.guard = HitLevel::None;
        fighter.move = move;
        fighter.move_frame = 0;
        fighter.has_hit = false;
        fighter.throw_break_active = 0;
        return fighter;
    }
    fighter.guard = HitLevel::None;
    fighter.throw_break_active = 0;
    return fighter;
}

FighterRuntime Simulator::tick_timers(FighterRuntime fighter) const noexcept {
    fighter.hitstun = std::max(0, fighter.hitstun - 1);
    fighter.blockstun = std::max(0, fighter.blockstun - 1);
    fighter.airborne = std::max(0, fighter.airborne - 1);
    fighter.throw_break_active = std::max(0, fighter.throw_break_active - 1);
    if (fighter.move >= 0) {
        ++fighter.move_frame;
    }
    if (std::abs(fighter.y) <= config_.lateral_return_speed) {
        fighter.y = 0.0;
    } else if (fighter.y > 0.0) {
        fighter.y -= config_.lateral_return_speed;
    } else {
        fighter.y += config_.lateral_return_speed;
    }
    return fighter;
}

FighterRuntime Simulator::move_fighter(FighterRuntime fighter, Action action, int forward) const noexcept {
    if (action == Action::WalkForward) {
        fighter.x += config_.walk_speed * forward;
    } else if (action == Action::WalkBack) {
        fighter.x -= config_.walk_speed * forward;
    } else if (action == Action::DashForward) {
        fighter.x += config_.dash_speed * forward;
    } else if (action == Action::DashBack) {
        fighter.x -= config_.dash_speed * forward;
    }

    if (action == Action::SidestepLeft) {
        fighter.y += config_.sidestep_speed;
    } else if (action == Action::SidestepRight) {
        fighter.y -= config_.sidestep_speed;
    } else if (action == Action::SidewalkLeft) {
        fighter.y += config_.sidewalk_speed;
    } else if (action == Action::SidewalkRight) {
        fighter.y -= config_.sidewalk_speed;
    }
    fighter.y = std::clamp(fighter.y, -1.0, 1.0);
    return fighter;
}

void Simulator::separate_and_clip(FighterRuntime& p1, FighterRuntime& p2) const noexcept {
    const double min_gap = config_.body_radius * 2.0;
    double p1_x = std::clamp(p1.x, -config_.stage_half_width, config_.stage_half_width);
    double p2_x = std::clamp(p2.x, -config_.stage_half_width, config_.stage_half_width);
    if (p2_x - p1_x < min_gap) {
        const double center = (p1_x + p2_x) / 2.0;
        p1_x = center - min_gap / 2.0;
        p2_x = center + min_gap / 2.0;
    }
    p1.x = std::clamp(p1_x, -config_.stage_half_width, config_.stage_half_width);
    p2.x = std::clamp(p2_x, -config_.stage_half_width, config_.stage_half_width);
}

const MoveSpec* Simulator::active_move(const FighterRuntime& fighter) const noexcept {
    if (fighter.move < 0) {
        return nullptr;
    }
    const auto& move = kMoves[static_cast<std::size_t>(fighter.move)];
    if (fighter.move_frame > move.startup && fighter.move_frame <= move.startup + move.active) {
        return &move;
    }
    return nullptr;
}

Simulator::AttackCheck Simulator::check_attack(
    const FighterRuntime& attacker,
    const FighterRuntime& defender,
    const MoveSpec& move) const noexcept {
    if (std::abs(defender.x - attacker.x) > move.range) {
        return {};
    }
    if (move.hit_level != HitLevel::Throw &&
        std::abs(defender.y - attacker.y) > config_.sidestep_evasion_width) {
        return {};
    }
    if (defender.airborne > 0 && (move.hit_level == HitLevel::Low || move.hit_level == HitLevel::Throw)) {
        return {};
    }
    if (move.hit_level == HitLevel::Throw && defender.throw_break_active > 0) {
        return {true, true, 0.0, true};
    }
    const bool blocked = is_blocked(defender.guard, move.hit_level);
    return {true, blocked, blocked ? 0.0 : move.damage, false};
}

void Simulator::apply_attack(
    FighterRuntime& attacker,
    FighterRuntime& defender,
    const MoveSpec& move,
    const AttackCheck& check,
    int direction) const noexcept {
    if (!check.in_range) {
        return;
    }
    attacker.has_hit = true;
    const double pushback = check.blocked ? move.pushback * 0.6 : move.pushback;
    if (check.blocked) {
        defender.blockstun = std::max(defender.blockstun, move.blockstun);
    } else {
        defender.health = std::max(0.0, defender.health - check.damage);
        defender.hitstun = std::max(defender.hitstun, move.hitstun);
        defender.blockstun = 0;
        defender.guard = HitLevel::None;
        defender.move = -1;
        defender.move_frame = 0;
        defender.has_hit = false;
        defender.launches_taken += static_cast<int>(move.launches);
    }
    defender.x = std::clamp(
        defender.x + pushback * direction,
        -config_.stage_half_width,
        config_.stage_half_width);
}

bool Simulator::finish_move_if_needed(FighterRuntime& fighter, bool active_whiff) const noexcept {
    if (fighter.move < 0) {
        return false;
    }
    const auto& move = kMoves[static_cast<std::size_t>(fighter.move)];
    bool whiffed = false;
    if (active_whiff && !fighter.has_hit && fighter.move_frame == move.startup + move.active) {
        whiffed = true;
        ++fighter.whiffs;
    }
    if (fighter.move_frame >= move.total_frames()) {
        fighter.move = -1;
        fighter.move_frame = 0;
        fighter.has_hit = false;
    }
    return whiffed;
}

void Simulator::check_stalemate(State& state, double damage_to_p1, double damage_to_p2) const noexcept {
    if (state.round_over) {
        return;
    }
    const bool no_damage = damage_to_p1 == 0.0 && damage_to_p2 == 0.0;
    const bool disengaged = state.distance() > config_.stall_distance;
    const bool no_active_attack = state.p1.move < 0 && state.p2.move < 0;
    state.stall_frames = no_damage && disengaged && no_active_attack ? state.stall_frames + 1 : 0;
    if (state.stall_frames >= config_.max_stall_frames) {
        state.round_over = true;
        state.winner = 0;
    }
}

void Simulator::check_round_over(State& state) const noexcept {
    if (state.round_over) {
        return;
    }
    int winner = 0;
    if (state.p1.health <= 0.0) {
        winner = 2;
    } else if (state.p2.health <= 0.0) {
        winner = 1;
    } else if (state.frame >= config_.max_frames) {
        if (config_.timeout_ties_are_draws && state.p1.health == state.p2.health) {
            state.round_over = true;
            state.winner = 0;
            return;
        }
        winner = state.p1.health >= state.p2.health ? 1 : 2;
    }
    if (winner != 0) {
        state.round_over = true;
        state.winner = winner;
    }
}

Simulator::FrameInfo Simulator::advance_frame(State& state, Action p1_action, Action p2_action) const {
    FighterRuntime p1 = tick_timers(state.p1);
    FighterRuntime p2 = tick_timers(state.p2);

    if (!p1.busy()) {
        p1 = move_fighter(p1, p1_action, 1);
    }
    if (!p2.busy()) {
        p2 = move_fighter(p2, p2_action, -1);
    }
    separate_and_clip(p1, p2);

    FrameInfo info{};
    const MoveSpec* p1_attack = active_move(p1);
    const MoveSpec* p2_attack = active_move(p2);
    AttackCheck p1_check{};
    AttackCheck p2_check{};
    const bool has_p1_check = p1_attack != nullptr && !p1.has_hit;
    const bool has_p2_check = p2_attack != nullptr && !p2.has_hit;
    if (has_p1_check) {
        p1_check = check_attack(p1, p2, *p1_attack);
    }
    if (has_p2_check) {
        p2_check = check_attack(p2, p1, *p2_attack);
    }

    const auto in_whiff_recovery = [](const FighterRuntime& fighter) {
        if (fighter.move < 0 || fighter.has_hit) {
            return false;
        }
        const auto& move = kMoves[static_cast<std::size_t>(fighter.move)];
        return fighter.move_frame > move.startup + move.active;
    };
    const bool p1_punishes = has_p1_check && p1_check.damage > 0.0 && in_whiff_recovery(p2);
    const bool p2_punishes = has_p2_check && p2_check.damage > 0.0 && in_whiff_recovery(p1);

    if (has_p1_check) {
        apply_attack(p1, p2, *p1_attack, p1_check, 1);
        info.damage_to_p2 += p1_check.damage;
        if (p1_attack->hit_level == HitLevel::Throw) {
            info.p1_throw_damage += p1_check.damage;
            info.p2_throw_break = p1_check.throw_broken;
        }
        if (p1_punishes) {
            info.p1_whiff_punish_bonus += config_.whiff_punish_reward_scale * p1_check.damage;
        }
        info.p2_block = p1_check.blocked;
    }
    if (has_p2_check) {
        apply_attack(p2, p1, *p2_attack, p2_check, -1);
        info.damage_to_p1 += p2_check.damage;
        if (p2_attack->hit_level == HitLevel::Throw) {
            info.p2_throw_damage += p2_check.damage;
            info.p1_throw_break = p2_check.throw_broken;
        }
        if (p2_punishes) {
            info.p2_whiff_punish_bonus += config_.whiff_punish_reward_scale * p2_check.damage;
        }
        info.p1_block = p2_check.blocked;
    }

    info.p1_whiff = finish_move_if_needed(p1, has_p1_check && !p1_check.in_range);
    info.p2_whiff = finish_move_if_needed(p2, has_p2_check && !p2_check.in_range);

    state.p1 = p1;
    state.p2 = p2;
    ++state.frame;
    check_stalemate(state, info.damage_to_p1, info.damage_to_p2);
    check_round_over(state);
    return info;
}

double Simulator::scaled_win_reward(double health) const noexcept {
    const double ratio = std::clamp(health / config_.max_health, 0.0, 1.0);
    return config_.round_win_base_reward + config_.round_win_health_reward * ratio;
}

double Simulator::scaled_timeout_penalty(double health_margin) const noexcept {
    const double margin = std::clamp(health_margin, -1.0, 1.0);
    if (margin > 0.0) {
        return config_.timeout_ahead_penalty +
               (config_.timeout_even_penalty - config_.timeout_ahead_penalty) * (1.0 - margin);
    }
    if (margin < 0.0) {
        return config_.timeout_even_penalty +
               (config_.timeout_behind_penalty - config_.timeout_even_penalty) * (-margin);
    }
    return config_.timeout_even_penalty;
}

StepResult Simulator::step(Action p1_action, Action p2_action) {
    if (p1_action == Action::Count || p2_action == Action::Count) {
        throw std::invalid_argument("Action::Count is not a valid simulator action");
    }
    if (state_.round_over) {
        StepInfo info{};
        info.frame = state_.frame;
        return {state_, 0.0, 0.0, true, false, info};
    }

    const State previous = state_;
    state_.p1 = start_action(state_.p1, p1_action);
    state_.p2 = start_action(state_.p2, p2_action);

    StepInfo info{};
    for (int frame = 0; frame < config_.decision_frames; ++frame) {
        const FrameInfo current = advance_frame(state_, p1_action, p2_action);
        info.damage_to_p1 += current.damage_to_p1;
        info.damage_to_p2 += current.damage_to_p2;
        info.p1_throw_damage += current.p1_throw_damage;
        info.p2_throw_damage += current.p2_throw_damage;
        info.p1_throw_breaks += static_cast<int>(current.p1_throw_break);
        info.p2_throw_breaks += static_cast<int>(current.p2_throw_break);
        info.p1_whiff_punish_bonus += current.p1_whiff_punish_bonus;
        info.p2_whiff_punish_bonus += current.p2_whiff_punish_bonus;
        info.p1_whiffs += static_cast<int>(current.p1_whiff);
        info.p2_whiffs += static_cast<int>(current.p2_whiff);
        info.p1_blocks += static_cast<int>(current.p1_block);
        info.p2_blocks += static_cast<int>(current.p2_block);
        if (state_.round_over) {
            break;
        }
    }

    const bool no_action =
        info.damage_to_p1 == 0.0 && info.damage_to_p2 == 0.0 &&
        is_no_action_passive(p1_action) && is_no_action_passive(p2_action) &&
        !(previous.p1.busy() || previous.p2.busy() || state_.p1.busy() || state_.p2.busy());
    const int no_action_frames = no_action ? state_.no_action_frames + config_.decision_frames : 0;
    state_.no_action_frames = no_action_frames;
    if (no_action_frames >= config_.no_action_timeout_frames && !state_.round_over) {
        state_.round_over = true;
        state_.winner = 0;
    }

    const bool truncated = state_.frame >= config_.max_frames && !state_.round_over;
    const bool timed_out = state_.round_over && state_.frame >= config_.max_frames &&
                           state_.p1.health > 0.0 && state_.p2.health > 0.0;
    const bool stalemate = state_.round_over && state_.winner == 0;
    const bool no_action_timeout =
        state_.round_over && state_.no_action_frames >= config_.no_action_timeout_frames;

    const double reward_damage_to_p2 =
        (info.damage_to_p2 - info.p1_throw_damage) + config_.throw_reward_scale * info.p1_throw_damage;
    const double reward_damage_to_p1 =
        (info.damage_to_p1 - info.p2_throw_damage) + config_.throw_reward_scale * info.p2_throw_damage;
    double reward_p1 = config_.damage_dealt_scale * reward_damage_to_p2 -
                       config_.damage_taken_scale * info.damage_to_p1;
    double reward_p2 = config_.damage_dealt_scale * reward_damage_to_p1 -
                       config_.damage_taken_scale * info.damage_to_p2;

    if (info.p1_throw_damage > 0.0) reward_p1 += config_.successful_throw_reward;
    if (info.p2_throw_damage > 0.0) reward_p2 += config_.successful_throw_reward;
    reward_p1 += info.p1_whiff_punish_bonus;
    reward_p2 += info.p2_whiff_punish_bonus;

    if (state_.winner == 1 && !timed_out) {
        const double win_reward = scaled_win_reward(state_.p1.health);
        reward_p1 += win_reward;
        reward_p2 -= win_reward;
        reward_p2 -= config_.round_loss_penalty;
    } else if (state_.winner == 2 && !timed_out) {
        const double win_reward = scaled_win_reward(state_.p2.health);
        reward_p1 -= win_reward;
        reward_p1 -= config_.round_loss_penalty;
        reward_p2 += win_reward;
    }

    if (state_.round_over) {
        const double health_margin = (state_.p1.health - state_.p2.health) / config_.max_health;
        if (timed_out) {
            reward_p1 -= scaled_timeout_penalty(health_margin);
            reward_p2 -= scaled_timeout_penalty(-health_margin);
        }
        if (stalemate) {
            reward_p1 -= config_.stalemate_penalty;
            reward_p2 -= config_.stalemate_penalty;
        }
        if (no_action_timeout) {
            reward_p1 -= config_.no_action_timeout_penalty;
            reward_p2 -= config_.no_action_timeout_penalty;
        }
    }

    if (p1_action == Action::Neutral) reward_p1 += config_.idle_penalty;
    if (p2_action == Action::Neutral) reward_p2 += config_.idle_penalty;

    const bool no_contact = info.damage_to_p1 == 0.0 && info.damage_to_p2 == 0.0 &&
                            info.p1_blocks == 0 && info.p2_blocks == 0;
    if (no_contact && is_lateral(p1_action) && info.p2_whiffs == 0) {
        reward_p1 += config_.lateral_passivity_penalty;
    }
    if (no_contact && is_lateral(p2_action) && info.p1_whiffs == 0) {
        reward_p2 += config_.lateral_passivity_penalty;
    }
    if (state_.distance() > 1.5) {
        if (!is_forward(p1_action)) reward_p1 += config_.far_spacing_penalty;
        if (!is_forward(p2_action)) reward_p2 += config_.far_spacing_penalty;
    }
    const double own_wall = config_.stage_half_width - 0.35;
    if (state_.p1.x < -own_wall && !is_forward(p1_action)) reward_p1 += config_.wall_camping_penalty;
    if (state_.p2.x > own_wall && !is_forward(p2_action)) reward_p2 += config_.wall_camping_penalty;
    if (state_.frame > config_.max_frames * 0.65 &&
        info.damage_to_p1 == 0.0 && info.damage_to_p2 == 0.0) {
        if (is_late_passive(p1_action)) reward_p1 += config_.late_round_passivity_penalty;
        if (is_late_passive(p2_action)) reward_p2 += config_.late_round_passivity_penalty;
    }

    reward_p1 += config_.block_reward * info.p1_blocks;
    reward_p2 += config_.block_reward * info.p2_blocks;
    reward_p1 += config_.blocked_attack_penalty * info.p2_blocks;
    reward_p2 += config_.blocked_attack_penalty * info.p1_blocks;
    reward_p1 += config_.throw_break_reward * info.p1_throw_breaks;
    reward_p2 += config_.throw_break_reward * info.p2_throw_breaks;
    reward_p1 += config_.throw_broken_penalty * info.p2_throw_breaks;
    reward_p2 += config_.throw_broken_penalty * info.p1_throw_breaks;
    reward_p1 += config_.whiff_penalty * info.p1_whiffs;
    reward_p2 += config_.whiff_penalty * info.p2_whiffs;

    info.timed_out = timed_out;
    info.stalemate = stalemate;
    info.no_action_timeout = no_action_timeout;
    info.stall_frames = state_.stall_frames;
    info.no_action_frames = state_.no_action_frames;
    info.frame = state_.frame;

    return {
        state_,
        reward_p1,
        reward_p2,
        state_.round_over,
        truncated,
        info,
    };
}

std::array<float, kObservationSize> Simulator::observation(int player) const {
    if (player != 1 && player != 2) {
        throw std::invalid_argument("player must be 1 or 2");
    }
    const FighterRuntime& own = player == 1 ? state_.p1 : state_.p2;
    const FighterRuntime& opponent = player == 1 ? state_.p2 : state_.p1;
    const double forward = player == 1 ? 1.0 : -1.0;
    const double signed_distance = (opponent.x - own.x) * forward;
    const double own_forward_wall =
        player == 1 ? config_.stage_half_width - own.x : own.x + config_.stage_half_width;
    const double own_back_wall =
        player == 1 ? own.x + config_.stage_half_width : config_.stage_half_width - own.x;
    const double stage_width = config_.stage_half_width * 2.0;

    const auto move_remaining = [](const FighterRuntime& fighter) {
        if (fighter.move < 0) return 0.0;
        const auto& move = kMoves[static_cast<std::size_t>(fighter.move)];
        const int remaining = std::max(0, move.total_frames() - fighter.move_frame);
        return std::min(1.0, static_cast<double>(remaining) / move.total_frames());
    };
    const auto throw_threat = [](const FighterRuntime& fighter) {
        if (fighter.move < 0) return false;
        const auto& move = kMoves[static_cast<std::size_t>(fighter.move)];
        return move.hit_level == HitLevel::Throw && fighter.move_frame <= move.startup + move.active;
    };

    return {
        static_cast<float>(own.health / config_.max_health),
        static_cast<float>(opponent.health / config_.max_health),
        static_cast<float>(signed_distance / stage_width),
        static_cast<float>(state_.distance() / stage_width),
        static_cast<float>(own_forward_wall / stage_width),
        static_cast<float>(own_back_wall / stage_width),
        static_cast<float>(static_cast<double>(state_.frame) / config_.max_frames),
        static_cast<float>(std::min(1.0, own.hitstun / 60.0)),
        static_cast<float>(std::min(1.0, opponent.hitstun / 60.0)),
        static_cast<float>(std::min(1.0, own.blockstun / 60.0)),
        static_cast<float>(std::min(1.0, opponent.blockstun / 60.0)),
        own.move >= 0 ? 1.0F : 0.0F,
        opponent.move >= 0 ? 1.0F : 0.0F,
        static_cast<float>(move_remaining(own)),
        static_cast<float>(move_remaining(opponent)),
        own.guard != HitLevel::None ? 1.0F : 0.0F,
        opponent.guard != HitLevel::None ? 1.0F : 0.0F,
        throw_threat(opponent) ? 1.0F : 0.0F,
        1.0F,
    };
}

std::array<float, kVisualObservationSize> Simulator::visual_observation(
    int player,
    const State* previous_state) const {
    if (player != 1 && player != 2) {
        throw std::invalid_argument("player must be 1 or 2");
    }
    const FighterRuntime& own = player == 1 ? state_.p1 : state_.p2;
    const FighterRuntime& opponent = player == 1 ? state_.p2 : state_.p1;
    const FighterRuntime* previous_own = nullptr;
    const FighterRuntime* previous_opponent = nullptr;
    if (previous_state != nullptr) {
        previous_own = player == 1 ? &previous_state->p1 : &previous_state->p2;
        previous_opponent = player == 1 ? &previous_state->p2 : &previous_state->p1;
    }

    const double own_velocity = previous_own == nullptr ? 0.0 : own.x - previous_own->x;
    const double opponent_velocity =
        previous_opponent == nullptr ? 0.0 : opponent.x - previous_opponent->x;
    const bool own_hit = previous_own != nullptr && previous_own->health - own.health > 1.0;
    const bool opponent_hit =
        previous_opponent != nullptr && previous_opponent->health - opponent.health > 1.0;
    const double own_motion = std::min(1.0, std::abs(own_velocity) + (own.move >= 0 ? 0.08 : 0.0));
    const double opponent_motion =
        std::min(1.0, std::abs(opponent_velocity) + (opponent.move >= 0 ? 0.08 : 0.0));

    const auto attack_likelihood = [this](const FighterRuntime& fighter) {
        if (fighter.move < 0) return 0.0;
        const auto& move = kMoves[static_cast<std::size_t>(fighter.move)];
        const double proximity = std::clamp(1.0 - state_.distance() / std::max(move.range + 0.8, 0.1), 0.0, 1.0);
        const bool active_window = fighter.move_frame <= move.startup + move.active;
        return (active_window ? 0.6 : 0.25) + 0.4 * proximity;
    };

    return {
        static_cast<float>(own.health / config_.max_health),
        static_cast<float>(opponent.health / config_.max_health),
        static_cast<float>(own.x),
        static_cast<float>(opponent.x),
        static_cast<float>(state_.distance()),
        static_cast<float>(own_velocity),
        static_cast<float>(opponent_velocity),
        static_cast<float>(own_motion),
        static_cast<float>(opponent_motion),
        own_hit ? 1.0F : 0.0F,
        opponent_hit ? 1.0F : 0.0F,
        static_cast<float>(attack_likelihood(own)),
        static_cast<float>(attack_likelihood(opponent)),
    };
}

std::array<bool, kActionCount> Simulator::legal_action_mask(int player) const {
    if (player != 1 && player != 2) {
        throw std::invalid_argument("player must be 1 or 2");
    }
    std::array<bool, kActionCount> mask{};
    mask.fill(true);
    const FighterRuntime& fighter = player == 1 ? state_.p1 : state_.p2;
    if (state_.round_over || fighter.busy()) {
        mask.fill(false);
        mask[action_index(Action::Neutral)] = true;
    }
    return mask;
}

}  // namespace t8::v2
