#pragma once

// The full-combat engine's frame step as plain data and __host__ __device__
// functions: one source for the scalar reference (FullCombatEngine wraps it)
// and the CUDA batch simulator (GpuFullCombatBatch). Everything here is
// trivially copyable, uses fixed-size arrays and indices instead of vectors,
// optionals, and string lookups, and calls no standard-library functions, so
// the same code compiles for the CPU and the GPU.
//
// Bitwise CPU/GPU parity needs the CUDA translation unit to be compiled with
// --fmad=false (no fused multiply-add contraction), which CMake sets.

#include "t8_v2/full_combat_engine.hpp"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define T8_FULL_HD __host__ __device__
#else
#define T8_FULL_HD
#endif

namespace t8::v2::full_core {

constexpr int kMaxWindowSpans = 8;
constexpr int kMaxHits = 32;
constexpr int kMaxContacts = 8;
constexpr int kMaxTransitions = 4;
constexpr std::int8_t kNoContact = -1;
// Parry outcomes are move indices; a move that names no outcome and one that
// names a move missing from the moveset differ (only the first falls back to
// its stance's outcome), as in the original string-based lookup.
constexpr int kNoOutcome = -1;
constexpr int kUnknownOutcome = -2;

struct Window {
    int count = 0;
    int first[kMaxWindowSpans] = {};
    int last[kMaxWindowSpans] = {};
    T8_FULL_HD bool contains(int frame) const {
        for (int index = 0; index < count; ++index) {
            if (frame >= first[index] && frame <= last[index]) return true;
        }
        return false;
    }
};

enum HitFlag : std::uint16_t {
    kHoming = 1U << 0U, kGroundHit = 1U << 1U, kWallSplat = 1U << 2U, kWallBreak = 1U << 3U,
    kFloorBreak = 1U << 4U, kBalconyBreak = 1U << 5U, kTornado = 1U << 6U, kSpike = 1U << 7U,
    kUnparryable = 1U << 8U, kReversalBreak = 1U << 9U, kHasHeatChip = 1U << 10U,
};

struct Hit {
    FullHitLevel level = FullHitLevel::Mid;
    FullThrowBreak throw_break = FullThrowBreak::OneTwo;
    std::uint16_t flags = 0;
    int first_active_frame = 1;
    int active_frames = 1;
    double damage = 0.0;
    double reach = 1.0;
    double tracking_left = 0.0;
    double tracking_right = 0.0;
    double chip_damage = 0.0;
    double heat_chip_damage = 0.0;
};

enum MoveFlag : std::uint64_t {
    kReactive = 1ULL << 0U, kRequiresSidestep = 1ULL << 1U, kRequiresBackToWall = 1ULL << 2U,
    kRequiresBackTurned = 1ULL << 3U, kRequiresOpponentBackTurned = 1ULL << 4U,
    kRequiresOpponentLeft = 1ULL << 5U, kRequiresOpponentRight = 1ULL << 6U, kResultBackTurned = 1ULL << 7U,
    kResultCrouching = 1ULL << 8U, kAttackThrowFrontOnly = 1ULL << 9U, kAttackThrowStandingOnly = 1ULL << 10U,
    kAttackThrowAirborne = 1ULL << 11U, kCannotKo = 1ULL << 12U, kKiCharge = 1ULL << 13U,
    kRequiresHeat = 1ULL << 14U, kRequiresRage = 1ULL << 15U, kEngagesHeat = 1ULL << 16U,
    kHeatEngager = 1ULL << 17U, kConsumesHeat = 1ULL << 18U, kConsumesRage = 1ULL << 19U,
    kRecoverableOnly = 1ULL << 20U, kRemovesRecoverable = 1ULL << 21U, kArmorDamageRecoverable = 1ULL << 22U,
    kSelfDamageWithoutHeatOnly = 1ULL << 23U, kSideSwitchOnHit = 1ULL << 24U, kSideSwitchOnBreak = 1ULL << 25U,
    kHasCounterHitAdvantage = 1ULL << 26U, kHasCounterHitEffect = 1ULL << 27U, kHasLaunchAirFrames = 1ULL << 28U,
    kHasBackTurnedHitAdvantage = 1ULL << 29U, kHasRageArtMax = 1ULL << 30U, kHasHeatDashBlock = 1ULL << 31U,
    kHasHeatDashHit = 1ULL << 32U, kHasInstallChip = 1ULL << 33U, kHasInstallReach = 1ULL << 34U,
};

struct Move {
    std::uint64_t flags = 0;
    int hit_offset = 0;  // into the side's hit table
    int hit_count = 0;
    int total_frames = 0;
    int recovery = 0;
    int block_advantage = 0;
    int hit_advantage = 0;
    int counter_hit_advantage = 0;
    int launch_air_frames = 0;
    int back_turned_hit_advantage = 0;
    int heat_dash_block_advantage = 0;
    int heat_dash_hit_advantage = 0;
    int heat_cost_frames = 0;
    int required_stance = -1;
    int result_stance = -1;
    int result_stance_on_hit = -1;
    int result_stance_on_block = -1;
    int parry_outcomes[4] = {-1, -1, -1, -1};  // move indices by level class
    int heat_parry_outcome = -1;
    FullHitEffect hit_effect = FullHitEffect::Stun;
    FullHitEffect counter_hit_effect = FullHitEffect::Stun;
    FullHitEffect back_turned_hit_effect = FullHitEffect::Stun;
    FullHitEffect heat_dash_hit_effect = FullHitEffect::Stun;
    FullPosture required_posture = FullPosture::Standing;
    FullMoveSpec::AttackThrow attack_throw = FullMoveSpec::AttackThrow::None;
    std::uint8_t parry_levels = 0;
    std::uint8_t heat_parry_levels = 0;
    double travel = 0.0;
    double pushback_hit = 0.0;
    double pushback_block = 0.0;
    double reversal_damage = 0.0;
    double attack_throw_damage = 0.0;
    double recoverable_damage = 0.0;
    double rage_art_max_damage = 0.0;
    double self_damage = 0.0;
    double self_recoverable = 0.0;
    double restore_health_hit = 0.0;
    double restore_recoverable_hit = 0.0;
    double restore_recoverable_block = 0.0;
    double resource_gain_start = 0.0;
    double resource_gain_hit = 0.0;
    double resource_gain_airborne_hit = 0.0;
    double resource_gain_block = 0.0;
    double resource_gain_heat_activation = 0.0;
    double install_damage_bonus = 0.0;
    double install_chip_damage = 0.0;
    double install_reach = 0.0;
    Window high_crush;
    Window low_crush;
    Window power_crush;
    Window armor;
    Window parry;
    Window reversal;
    Window invincible;
    Window airborne;
    Window heat_parry;

    T8_FULL_HD bool has(MoveFlag flag) const { return (flags & flag) != 0U; }
};

struct Stance {
    bool can_guard = true;
    bool has_max_frames = false;
    int max_frames = 0;
    std::uint8_t auto_parry = 0;
    int parry_outcomes[4] = {-1, -1, -1, -1};
    int pulse_interval_frames = 0;
    double pulse_recoverable = 0.0;
    double pulse_resource = 0.0;
};

struct Rules {
    double resource_max = 0.0;
    bool has_install_threshold = false;
    double install_threshold = 0.0;
    bool resource_persists = false;
    bool install_consumes = false;
};

// One fighter's tables. The pointers address host or device memory.
struct Side {
    const Move* moves = nullptr;
    int move_count = 0;
    const Hit* hits = nullptr;
    const Stance* stances = nullptr;
    int stance_count = 0;
    Rules rules{};
};

struct Tables {
    Side side[2];
    FullEngineConfig config{};
};

struct Fighter {
    double health = 180.0;
    double recoverable_health = 0.0;
    double x = 0.0;
    double axis = 0.0;
    double resource = 0.0;
    FullPosture posture = FullPosture::Standing;
    FullUniversal universal = FullUniversal::Idle;
    std::int8_t heat_dash_from = kNoContact;  // FullContact, or none
    bool heat = false;
    bool heat_available = true;
    bool rage = false;
    bool rage_available = true;
    bool throw_break_attempted = false;
    bool installed = false;
    bool move_hit = false;
    bool move_blocked = false;
    bool resource_gained = false;
    bool back_turned = false;
    bool move_in_heat = false;
    bool ki_charge_boost = false;
    bool move_boosted = false;
    bool in_blockstun = false;
    bool tornado_used = false;
    bool wall_splat_used = false;
    bool wall_scaled = false;
    bool techable = false;
    bool crumpled = false;
    int stance = -1;
    int heat_frames = 0;
    int throw_break_frames = 0;
    int pending_throw_hit = -1;
    int pending_throw_elapsed = 0;
    int move = -1;  // index into the side's moves, or -1 for a universal action
    int action_frame = 0;
    std::uint32_t hits_landed = 0;
    int stance_frames = 0;
    int ki_charge_frames = 0;
    int stun = 0;
    int posture_frames = 0;
    int grounded_elapsed = 0;
    int combo_hits = 0;
};

struct State {
    Fighter fighters[2];
    FullEngineStage stage{};
    int frame = 0;
    int winner = -1;
};

struct Contact {
    int attacker = 0;
    int hit_index = 0;
    FullContact contact = FullContact::Hit;
    double damage = 0.0;
};

struct Events {
    int contact_count = 0;
    Contact contacts[kMaxContacts];
    int transition_count = 0;
    FullTransition transitions[kMaxTransitions] = {};
    bool heat_activated[2] = {false, false};
    bool rage_activated[2] = {false, false};

    T8_FULL_HD void add_contact(int attacker, int hit_index, FullContact contact, double damage) {
        if (contact_count < kMaxContacts) contacts[contact_count++] = {attacker, hit_index, contact, damage};
    }
    T8_FULL_HD void add_transition(FullTransition transition) {
        if (transition_count < kMaxTransitions) transitions[transition_count++] = transition;
    }
};

struct Input {
    int move = -1;  // -1: the universal action
    FullUniversal universal = FullUniversal::Idle;
};

// ---------------------------------------------------------------- helpers

T8_FULL_HD inline double dmin(double a, double b) { return b < a ? b : a; }
T8_FULL_HD inline double dmax(double a, double b) { return a < b ? b : a; }
T8_FULL_HD inline int imin(int a, int b) { return b < a ? b : a; }
T8_FULL_HD inline int imax(int a, int b) { return a < b ? b : a; }
T8_FULL_HD inline double dabs(double value) { return value < 0.0 ? -value : value; }
T8_FULL_HD inline double dclamp(double value, double low, double high) {
    return value < low ? low : (high < value ? high : value);
}
template <typename T>
T8_FULL_HD inline void swap_values(T& a, T& b) {
    T temporary = a;
    a = b;
    b = temporary;
}

T8_FULL_HD inline std::uint8_t level_class_bits(FullHitLevel level) {
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

T8_FULL_HD inline int level_index(FullHitLevel level) {
    switch (level_class_bits(level)) {
        case FullLevelHigh: return 0;
        case FullLevelMid: return 1;
        case FullLevelLow: return 2;
        case FullLevelThrow: return 3;
        default: return 1;
    }
}

T8_FULL_HD inline bool is_strike(FullHitLevel level) { return level != FullHitLevel::Throw; }
T8_FULL_HD inline bool is_low(FullHitLevel level) {
    return level == FullHitLevel::Low || level == FullHitLevel::SpecialLow;
}
T8_FULL_HD inline bool is_throw_break(FullUniversal action) {
    return action == FullUniversal::ThrowBreak || action == FullUniversal::ThrowBreak1 ||
        action == FullUniversal::ThrowBreak2;
}
T8_FULL_HD inline bool breaks(FullThrowBreak rule, FullUniversal input) {
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

T8_FULL_HD inline int universal_duration(FullUniversal action, const FullEngineConfig& config) {
    switch (action) {
        case FullUniversal::DashForward: return config.dash_frames;
        case FullUniversal::Backdash: return config.backdash_frames;
        case FullUniversal::SidestepLeft:
        case FullUniversal::SidestepRight: return config.sidestep_frames;
        case FullUniversal::HeatDash: return config.heat_dash_frames;
        default: return 1;  // Idle, walking, and crouching are re-chosen every frame.
    }
}

T8_FULL_HD inline const Move& move_of(const Tables& tables, int player, int index) {
    return tables.side[player].moves[index];
}
T8_FULL_HD inline const Hit& hit_of(const Tables& tables, int player, const Move& move, int hit) {
    return tables.side[player].hits[move.hit_offset + hit];
}
T8_FULL_HD inline const Stance* stance_rule(const Tables& tables, const State& state, int player) {
    const int stance = state.fighters[player].stance;
    if (stance < 0 || stance >= tables.side[player].stance_count) return nullptr;
    return &tables.side[player].stances[stance];
}

// ---------------------------------------------------------------- engine

T8_FULL_HD inline void gain_resource(const Tables& tables, State& state, int player, double amount) {
    Fighter& fighter = state.fighters[player];
    const Rules& rule = tables.side[player].rules;
    if (!(rule.resource_max > 0.0) || !(amount > 0.0)) return;
    fighter.resource = dmin(rule.resource_max, fighter.resource + amount);
    if (rule.has_install_threshold && fighter.resource >= rule.install_threshold) fighter.installed = true;
}

T8_FULL_HD inline void reset(const Tables& tables, State& state, double p1_x, double p2_x) {
    state = State{};
    state.stage.left_wall = -tables.config.half_width;
    state.stage.right_wall = tables.config.half_width;
    state.fighters[0].x = p1_x;
    state.fighters[1].x = p2_x;
    for (Fighter& fighter : state.fighters) {
        fighter.health = tables.config.max_health;
        fighter.action_frame = universal_duration(FullUniversal::Idle, tables.config);  // actionable at once
    }
}

T8_FULL_HD inline void next_round(const Tables& tables, State& state, double p1_x, double p2_x) {
    const double carried[2] = {state.fighters[0].resource, state.fighters[1].resource};
    reset(tables, state, p1_x, p2_x);
    for (int player = 0; player < 2; ++player) {
        if (tables.side[player].rules.resource_persists) gain_resource(tables, state, player, carried[player]);
    }
}

T8_FULL_HD inline bool actionable(const Tables& tables, const State& state, int player) {
    const Fighter& fighter = state.fighters[player];
    if (fighter.stun > 0 || fighter.crumpled || fighter.throw_break_frames > 0 || fighter.pending_throw_hit >= 0) {
        return false;
    }
    if (fighter.posture != FullPosture::Standing && fighter.posture != FullPosture::Crouching) return false;
    if (fighter.move >= 0) return fighter.action_frame >= move_of(tables, player, fighter.move).total_frames;
    return fighter.action_frame >= universal_duration(fighter.universal, tables.config);
}

T8_FULL_HD inline bool legal(const Tables& tables, const State& state, int player, const Input& input) {
    if (state.winner >= 0) return false;
    const Fighter& fighter = state.fighters[player];
    const FullEngineConfig& config = tables.config;
    if (input.move >= 0) {
        if (input.move >= tables.side[player].move_count) return false;
        const Move& spec = move_of(tables, player, input.move);
        if (spec.has(kReactive)) return false;
        if (spec.has(kRequiresSidestep)) {
            // Performed out of a sidestep in progress.
            const bool sidestepping = fighter.move < 0 && fighter.action_frame >= 1 && fighter.stun == 0 &&
                !fighter.crumpled && fighter.throw_break_frames == 0 && fighter.pending_throw_hit < 0 &&
                (fighter.universal == FullUniversal::SidestepLeft || fighter.universal == FullUniversal::SidestepRight);
            if (!sidestepping) return false;
        } else if (!actionable(tables, state, player)) {
            return false;
        }
        const Fighter& other = state.fighters[1 - player];
        if (spec.has(kRequiresBackTurned) != fighter.back_turned) return false;  // back-turned: only BT moves
        if (spec.has(kRequiresOpponentBackTurned) && !other.back_turned) return false;
        if (spec.has(kRequiresBackToWall)) {
            const bool facing_right = other.x >= fighter.x;
            const double wall = facing_right ? state.stage.left_wall : state.stage.right_wall;
            if (dabs(fighter.x - wall) > config.back_to_wall_distance) return false;
        }
        if (spec.has(kRequiresOpponentLeft) || spec.has(kRequiresOpponentRight)) {
            // Offset of this fighter as seen by the opponent, which faces it: negative is the
            // opponent's left (the tracking convention), so a large negative offset exposes its left side.
            const double toward = fighter.x >= other.x ? 1.0 : -1.0;
            const double distance = dmax(dabs(fighter.x - other.x), 1e-9);
            const double side = (fighter.axis - other.axis) * toward / distance;
            if (spec.has(kRequiresOpponentLeft) && side > -config.side_exposure_ratio) return false;
            if (spec.has(kRequiresOpponentRight) && side < config.side_exposure_ratio) return false;
        }
        return spec.required_posture == fighter.posture &&
            (spec.required_stance < 0 || spec.required_stance == fighter.stance) &&
            (!spec.has(kRequiresHeat) || fighter.heat) && (!spec.has(kConsumesHeat) || fighter.heat) &&
            (spec.heat_cost_frames == 0 || fighter.heat) &&
            (!spec.has(kEngagesHeat) || (fighter.heat_available && !fighter.heat)) &&
            (!(spec.has(kRequiresRage) || spec.has(kConsumesRage)) || fighter.rage);
    }
    switch (input.universal) {
        case FullUniversal::WakeupStand:
        case FullUniversal::WakeupRoll:
            return fighter.posture == FullPosture::Grounded && fighter.grounded_elapsed >= config.knockdown_frames;
        case FullUniversal::TechRoll:
            return fighter.posture == FullPosture::Grounded && fighter.techable &&
                fighter.grounded_elapsed < config.tech_window_frames;
        case FullUniversal::ThrowBreak:
        case FullUniversal::ThrowBreak1:
        case FullUniversal::ThrowBreak2:
            return fighter.throw_break_frames > 0 && !fighter.throw_break_attempted;
        case FullUniversal::HeatDash: {
            if (fighter.move < 0 || !fighter.heat || fighter.heat_dash_from == kNoContact) return false;
            const Move& spec = move_of(tables, player, fighter.move);
            return static_cast<FullContact>(fighter.heat_dash_from) == FullContact::Blocked
                ? spec.has(kHasHeatDashBlock) : spec.has(kHasHeatDashHit);
        }
        default:
            return actionable(tables, state, player);
    }
}

T8_FULL_HD inline void deal_damage(const Tables& tables, State& state, int target, double damage, bool recoverable,
                                   bool can_ko, Events& events) {
    Fighter& fighter = state.fighters[target];
    const FullEngineConfig& config = tables.config;
    if (!can_ko) damage = dmin(damage, dmax(0.0, fighter.health - 1.0));
    damage = dmax(0.0, damage);
    fighter.health -= damage;
    if (recoverable) fighter.recoverable_health += damage;
    fighter.recoverable_health = dclamp(fighter.recoverable_health, 0.0, config.max_health - fighter.health);
    if (fighter.health > 0.0 && fighter.rage_available &&
        fighter.health <= config.rage_health_fraction * config.max_health) {
        fighter.rage = true;
        fighter.rage_available = false;
        events.rage_activated[target] = true;
    }
}

T8_FULL_HD inline void regain(Fighter& fighter, double recoverable, double health, double max_health) {
    const double restored = dmin(dmax(0.0, recoverable), fighter.recoverable_health);
    fighter.recoverable_health -= restored;
    fighter.health = dmin(max_health, fighter.health + restored + dmax(0.0, health));
    fighter.recoverable_health = dmin(fighter.recoverable_health, max_health - fighter.health);
}

T8_FULL_HD inline void activate_heat(const Tables& tables, State& state, int player, Events& events) {
    Fighter& fighter = state.fighters[player];
    if (!fighter.heat_available || fighter.heat) return;
    fighter.heat = true;
    fighter.heat_available = false;
    fighter.heat_frames = tables.config.heat_duration_frames;
    events.heat_activated[player] = true;
}

T8_FULL_HD inline void break_throw(const Tables& tables, State& state, int thrower, Events& events) {
    Fighter& attacker = state.fighters[thrower];
    Fighter& defender = state.fighters[1 - thrower];
    const int hit_index = attacker.pending_throw_hit;
    const bool side_switch = attacker.move >= 0 && move_of(tables, thrower, attacker.move).has(kSideSwitchOnBreak);
    events.add_contact(thrower, hit_index, FullContact::ThrowBroken, 0.0);
    Fighter* both[2] = {&attacker, &defender};
    for (Fighter* fighter : both) {
        fighter->move = -1;
        fighter->universal = FullUniversal::Idle;
        fighter->action_frame = universal_duration(FullUniversal::Idle, tables.config);
        fighter->stun = tables.config.throw_break_recovery;
        fighter->in_blockstun = true;  // guard holds while recovering from the break
        fighter->heat_dash_from = kNoContact;
    }
    attacker.pending_throw_hit = -1;
    defender.throw_break_frames = 0;
    if (side_switch) swap_values(attacker.x, defender.x);
    const double direction = defender.x >= attacker.x ? 1.0 : -1.0;
    attacker.x = dmax(state.stage.left_wall, dmin(state.stage.right_wall, attacker.x - direction * 0.2));
    defender.x = dmax(state.stage.left_wall, dmin(state.stage.right_wall, defender.x + direction * 0.2));
}

T8_FULL_HD inline void release_throw(State& state, int thrower) {
    Fighter& fighter = state.fighters[thrower];
    Fighter& held = state.fighters[1 - thrower];
    if (fighter.pending_throw_hit < 0) return;
    fighter.pending_throw_hit = -1;
    held.throw_break_frames = 0;
    held.throw_break_attempted = false;
}

T8_FULL_HD inline void apply_parry_outcome(const Tables& tables, State& state, int parrier_index, int outcome,
                                           int attacker_index, Events& events) {
    Fighter& parrier = state.fighters[parrier_index];
    Fighter& attacker = state.fighters[attacker_index];
    const FullEngineConfig& config = tables.config;
    const Move& spec = move_of(tables, parrier_index, outcome);
    double damage = 0.0;
    for (int hit = 0; hit < spec.hit_count; ++hit) damage += hit_of(tables, parrier_index, spec, hit).damage;
    deal_damage(tables, state, attacker_index, damage, spec.has(kRecoverableOnly), !spec.has(kRecoverableOnly),
                events);
    release_throw(state, attacker_index);
    attacker.move = -1;
    attacker.universal = FullUniversal::Idle;
    attacker.stance = -1;
    attacker.heat_dash_from = kNoContact;
    attacker.in_blockstun = false;
    const int frames = spec.recovery > 0 ? spec.recovery : config.parry_outcome_frames;
    switch (spec.hit_effect) {
        case FullHitEffect::Launch:
            attacker.stun = 0;
            attacker.posture = FullPosture::Airborne;
            attacker.posture_frames = spec.has(kHasLaunchAirFrames) ? spec.launch_air_frames
                : (frames + spec.hit_advantage > 0 ? frames + spec.hit_advantage : config.launch_air_frames);
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
            attacker.stun = imax(0, frames + spec.hit_advantage);
            break;
    }
    // The parrying fighter plays the outcome: its hits have resolved, `frames` of
    // recovery remain, and it ends in the outcome's stance.
    release_throw(state, parrier_index);
    parrier.move = outcome;
    parrier.universal = FullUniversal::Idle;
    parrier.stance = -1;
    parrier.stun = 0;
    parrier.in_blockstun = false;
    parrier.heat_dash_from = kNoContact;
    parrier.action_frame = imax(0, spec.total_frames - frames);
    parrier.hits_landed = spec.hit_count >= 32 ? ~0U : (1U << static_cast<unsigned>(spec.hit_count)) - 1U;
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
    regain(parrier, spec.restore_recoverable_hit, spec.restore_health_hit, config.max_health);
    gain_resource(tables, state, parrier_index, spec.resource_gain_start + spec.resource_gain_hit);
    if (spec.has(kRemovesRecoverable)) attacker.recoverable_health = 0.0;
}

T8_FULL_HD inline void push_defender(State& state, Fighter& attacker, Fighter& defender, double distance) {
    // Away from the attacker, judged at push time (after any side switch).
    const double direction = defender.x >= attacker.x ? 1.0 : -1.0;
    defender.x += direction * distance;
    const double wall = direction > 0.0 ? state.stage.right_wall : state.stage.left_wall;
    const double overflow = direction > 0.0 ? defender.x - wall : wall - defender.x;
    if (overflow > 0.0) {
        defender.x = wall;
        attacker.x -= direction * overflow;  // pushback against a wall moves the attacker back
    }
}

T8_FULL_HD inline void enable_heat_dash(const Tables& tables, State& state, int attacker_index, const Move& spec,
                                        bool last_hit, FullContact from, Events& events) {
    Fighter& attacker = state.fighters[attacker_index];
    if (!last_hit) return;  // a multi-hit engager activates Heat and allows Heat Dash after its last hit
    if (spec.has(kHeatEngager) && attacker.heat_available && !attacker.heat) {
        activate_heat(tables, state, attacker_index, events);
        gain_resource(tables, state, attacker_index, spec.resource_gain_heat_activation);
    }
    // Engagers, and Heat versions that publish Heat Dash data, can Heat Dash while in Heat.
    const bool has_dash = spec.has(kHasHeatDashBlock) || spec.has(kHasHeatDashHit);
    if (attacker.heat && has_dash) attacker.heat_dash_from = static_cast<std::int8_t>(from);
}

T8_FULL_HD inline double scaled_damage(const FullEngineConfig& config, const Fighter& attacker,
                                       const Fighter& defender, double damage, bool counter) {
    damage *= counter ? config.counter_hit_damage_scale : 1.0;
    if (attacker.rage) damage *= config.rage_damage_scale;
    if (defender.combo_hits > 0) {
        damage *= dmax(config.combo_scale_floor, 1.0 - config.combo_scale_step * defender.combo_hits);
    }
    if (defender.wall_scaled) damage *= config.wall_combo_scale;
    return damage;
}

T8_FULL_HD inline void apply_contact(const Tables& tables, State& state, int attacker_index, int move_index,
                                     int defense_move, int hit_index, FullContact contact, int elapsed,
                                     Events& events) {
    FullEngineStage& stage = state.stage;
    const FullEngineConfig& config = tables.config;
    Fighter& attacker = state.fighters[attacker_index];
    Fighter& defender = state.fighters[1 - attacker_index];
    const int defender_index = 1 - attacker_index;
    const Move& spec = move_of(tables, attacker_index, move_index);
    const Hit& hit = hit_of(tables, attacker_index, spec, hit_index);
    const bool last_hit = hit_index + 1 == spec.hit_count;
    Contact event{attacker_index, hit_index, contact, 0.0};
    // Published advantage assumes contact on the hit's first active frame;
    // stun is fixed from there, so a later (meaty) contact gains frames. A
    // throw resolves `elapsed` frames after it connected (its break window).
    const int remaining = imax(0, spec.total_frames - hit.first_active_frame - elapsed);

    switch (contact) {
        case FullContact::Evaded:
        case FullContact::Crushed:
        case FullContact::ThrowBroken:
        case FullContact::ThrowHeld:
            break;
        case FullContact::Parried: {
            // A parry with a published outcome plays it (Jun: GEN.P); otherwise the attacker is stunned.
            const int level = level_index(hit.level);
            int outcome = -1;
            if (defense_move >= 0) {
                const Move& defense = move_of(tables, defender_index, defense_move);
                const bool heat_parry = defender.move_in_heat && (defense.heat_parry_levels & level_class_bits(hit.level)) &&
                    defense.heat_parry.contains(defender.action_frame);
                outcome = heat_parry ? defense.heat_parry_outcome : defense.parry_outcomes[level];
                const Side& side = tables.side[defender_index];
                // Only a move that names no outcome falls back to its stance's (kNoOutcome, not kUnknownOutcome).
                if (outcome == kNoOutcome && defense.result_stance >= 0 && defense.result_stance < side.stance_count) {
                    outcome = side.stances[defense.result_stance].parry_outcomes[level];
                }
            } else if (const Stance* rule = stance_rule(tables, state, defender_index)) {
                outcome = rule->parry_outcomes[level];
            }
            if (outcome >= 0) {
                const double before = attacker.health;
                apply_parry_outcome(tables, state, defender_index, outcome, attacker_index, events);
                event.damage = before - attacker.health;
                break;
            }
            release_throw(state, attacker_index);
            attacker.stance = -1;
            attacker.move = -1;
            attacker.universal = FullUniversal::Idle;
            attacker.action_frame = universal_duration(FullUniversal::Idle, config);
            attacker.stun = config.parry_stun_frames;
            attacker.in_blockstun = false;
            attacker.heat_dash_from = kNoContact;
            break;
        }
        case FullContact::Reversed: {
            const double damage = defense_move >= 0 ? move_of(tables, defender_index, defense_move).reversal_damage : 0.0;
            deal_damage(tables, state, attacker_index, damage, false, true, events);
            event.damage = damage;
            release_throw(state, attacker_index);
            attacker.move = -1;
            attacker.universal = FullUniversal::Idle;
            attacker.stance = -1;
            attacker.stun = 0;
            attacker.heat_dash_from = kNoContact;
            attacker.posture = FullPosture::Grounded;
            attacker.grounded_elapsed = 0;
            attacker.techable = false;
            // The reversing fighter recovers with its guard up.
            defender.move = -1;
            defender.universal = FullUniversal::Idle;
            defender.action_frame = universal_duration(FullUniversal::Idle, config);
            defender.stun = config.reversal_recovery;
            defender.in_blockstun = true;
            break;
        }
        case FullContact::Blocked: {
            defender.stun = imax(0, remaining + spec.block_advantage);
            defender.in_blockstun = defender.stun > 0;
            attacker.move_blocked = true;
            if (last_hit && !attacker.resource_gained) {
                attacker.resource_gained = true;
                gain_resource(tables, state, attacker_index, spec.resource_gain_block);
            }
            const double chip = attacker.installed && last_hit && spec.has(kHasInstallChip) ? spec.install_chip_damage
                : attacker.heat && (hit.flags & kHasHeatChip) ? hit.heat_chip_damage : hit.chip_damage;
            if (chip > 0.0) deal_damage(tables, state, defender_index, chip, true, false, events);
            event.damage = chip;
            regain(attacker, config.recoverable_regain_block * hit.damage +
                       (last_hit ? spec.restore_recoverable_block : 0.0), 0.0, config.max_health);
            enable_heat_dash(tables, state, attacker_index, spec, last_hit, FullContact::Blocked, events);
            push_defender(state, attacker, defender, spec.pushback_block);
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
            if (last_hit && spec.has(kHasRageArtMax)) {
                // Provisional: grows linearly from the listed damage at full health to the maximum at none.
                double listed = 0.0;
                for (int part = 0; part < spec.hit_count; ++part) listed += hit_of(tables, attacker_index, spec, part).damage;
                const double missing = 1.0 - attacker.health / config.max_health;
                base += dmax(0.0, spec.rage_art_max_damage - listed) * dclamp(missing, 0.0, 1.0);
            }
            if (attacker.move_boosted) base *= config.ki_charge_damage_scale;
            const double damage = scaled_damage(config, attacker, defender, base, counter);
            if (contact == FullContact::Armored) {
                const bool recoverable = defense_move >= 0 &&
                    move_of(tables, defender_index, defense_move).has(kArmorDamageRecoverable);
                deal_damage(tables, state, defender_index, damage, recoverable, true, events);
                event.damage = damage;
                break;  // armor: damage only, the defender keeps attacking
            }
            {
                const bool can_ko = !spec.has(kCannotKo) && !spec.has(kRecoverableOnly);
                const double recoverable_part = spec.has(kRecoverableOnly) ? damage
                    : last_hit ? dmin(damage, spec.recoverable_damage) : 0.0;
                deal_damage(tables, state, defender_index, damage - recoverable_part, false, can_ko, events);
                deal_damage(tables, state, defender_index, recoverable_part, true, false, events);
            }
            event.damage = damage;
            if (spec.has(kRemovesRecoverable)) defender.recoverable_health = 0.0;
            attacker.move_hit = true;
            if (last_hit && !attacker.resource_gained) {
                attacker.resource_gained = true;
                gain_resource(tables, state, attacker_index,
                              juggle ? spec.resource_gain_airborne_hit : spec.resource_gain_hit);
            }
            regain(attacker, config.recoverable_regain_hit * damage + (last_hit ? spec.restore_recoverable_hit : 0.0),
                   last_hit ? spec.restore_health_hit : 0.0, config.max_health);

            release_throw(state, defender_index);  // a throw loses to a strike landing on the same frame
            defender.stance = -1;
            defender.move = -1;
            defender.universal = FullUniversal::Idle;
            defender.action_frame = 0;
            defender.in_blockstun = false;
            defender.heat_dash_from = kNoContact;
            defender.back_turned = false;
            const bool back_turned_hit = defender_back_turned && spec.has(kHasBackTurnedHitAdvantage);
            const FullHitEffect effect = back_turned_hit ? spec.back_turned_hit_effect
                : counter && spec.has(kHasCounterHitEffect) ? spec.counter_hit_effect : spec.hit_effect;
            const int advantage = back_turned_hit ? spec.back_turned_hit_advantage
                : counter && spec.has(kHasCounterHitAdvantage) ? spec.counter_hit_advantage : spec.hit_advantage;
            // Attack throw: the strike turns into an unbreakable throw when its condition holds.
            const bool throw_trigger = spec.attack_throw == FullMoveSpec::AttackThrow::OnHit ||
                (spec.attack_throw == FullMoveSpec::AttackThrow::OnCounterHit && counter);
            const bool throw_front = !spec.has(kAttackThrowFrontOnly) || !defender_back_turned;
            const bool throw_posture = defender_posture == FullPosture::Airborne ? spec.has(kAttackThrowAirborne)
                : !spec.has(kAttackThrowStandingOnly) || defender_posture == FullPosture::Standing;
            if (last_hit && throw_trigger && throw_front && throw_posture && !back_turned_hit) {
                const double extra = scaled_damage(config, attacker, defender, spec.attack_throw_damage, false);
                deal_damage(tables, state, defender_index, extra, false, !spec.has(kCannotKo), events);
                event.damage += extra;
            }
            const bool thrown = hit.level == FullHitLevel::Throw;
            if (juggle && (hit.flags & kSpike)) {
                ++defender.combo_hits;
                defender.posture = FullPosture::Grounded;
                defender.grounded_elapsed = 0;
                defender.techable = false;
            } else if (juggle) {
                ++defender.combo_hits;
                if (defender.posture == FullPosture::Airborne) {
                    // Kept up until the attacker recovers, plus a window for the next move.
                    const int window = imax(0, config.juggle_window_frames -
                                                   config.juggle_window_decay * (defender.combo_hits - 1));
                    defender.posture_frames = imax(defender.posture_frames, remaining + window);
                    if ((hit.flags & kTornado) && !defender.tornado_used) {
                        defender.tornado_used = true;
                        defender.posture_frames += config.tornado_extra_frames;
                    }
                } else {
                    // Hits on a wall-splatted opponent keep it pinned the same way.
                    defender.posture_frames = imax(defender.posture_frames, remaining + config.wall_splat_frames);
                }
            } else if (defender.posture == FullPosture::Grounded) {
                ++defender.combo_hits;
                defender.grounded_elapsed = 0;  // hits on the ground keep the fighter down
            } else if (last_hit && effect == FullHitEffect::Launch) {
                defender.posture = FullPosture::Airborne;
                // "+29a": airborne until the attacker has been recovered for 29 frames.
                defender.posture_frames = spec.has(kHasLaunchAirFrames) ? spec.launch_air_frames
                    : (remaining + advantage > 0 ? remaining + advantage : config.launch_air_frames);
                defender.combo_hits = 1;
                defender.stun = 0;
            } else if (last_hit && effect == FullHitEffect::Knockdown) {
                defender.posture = FullPosture::Grounded;
                defender.grounded_elapsed = 0;
                defender.techable = !thrown;
                defender.stun = 0;
            } else if (last_hit && effect == FullHitEffect::Crumple) {
                defender.stun = config.crumple_frames;
                defender.crumpled = true;
            } else {
                defender.stun = imax(0, remaining + advantage);
                if (defender.posture == FullPosture::Crouching) defender.posture = FullPosture::Standing;
            }
            enable_heat_dash(tables, state, attacker_index, spec, last_hit, contact, events);
            if (last_hit && spec.has(kSideSwitchOnHit)) swap_values(attacker.x, defender.x);
            push_defender(state, attacker, defender, spec.pushback_hit);

            // Walls and stage transitions, on the side the defender was driven toward.
            const bool right_side = defender.x >= attacker.x;
            const double wall = right_side ? stage.right_wall : stage.left_wall;
            const bool at_wall = dabs(defender.x - wall) < 1e-9;
            bool& wall_breakable = right_side ? stage.right_wall_breakable : stage.left_wall_breakable;
            bool& balcony = right_side ? stage.right_balcony : stage.left_balcony;
            double& wall_position = right_side ? stage.right_wall : stage.left_wall;
            const double extension = right_side ? config.wall_break_extension : -config.wall_break_extension;
            const auto send_airborne = [&] {
                defender.posture = FullPosture::Airborne;
                defender.posture_frames = config.stage_break_air_frames;
                defender.stun = 0;
                defender.combo_hits = imax(1, defender.combo_hits);
            };
            if (at_wall && (hit.flags & kBalconyBreak) && balcony) {
                balcony = false;
                ++stage.level;
                wall_position += extension;
                defender.x += (right_side ? 1.0 : -1.0) * 1.0;
                send_airborne();
                events.add_transition(FullTransition::BalconyBreak);
            } else if (at_wall && (hit.flags & kWallBreak) && wall_breakable) {
                wall_breakable = false;
                wall_position += extension;
                send_airborne();
                events.add_transition(FullTransition::WallBreak);
            } else if (at_wall && ((hit.flags & kWallSplat) || juggle) && !defender.wall_splat_used) {
                defender.posture = FullPosture::WallSplat;
                defender.posture_frames = remaining + config.wall_splat_frames;
                defender.wall_splat_used = true;
                defender.wall_scaled = true;
                defender.stun = 0;
                defender.combo_hits = imax(1, defender.combo_hits);
            } else if ((hit.flags & kFloorBreak) && stage.floor_breakable &&
                       (juggle || defender.posture == FullPosture::Grounded)) {
                stage.floor_breakable = false;
                ++stage.level;
                send_airborne();
                events.add_transition(FullTransition::FloorBreak);
            }
            break;
        }
    }
    events.add_contact(event.attacker, event.hit_index, event.contact, event.damage);
}

T8_FULL_HD inline double facing(const State& state, int player) {
    const Fighter& own = state.fighters[player];
    const Fighter& other = state.fighters[1 - player];
    if (other.x == own.x) return player == 0 ? 1.0 : -1.0;
    return other.x > own.x ? 1.0 : -1.0;
}

// A decided contact, applied after both fighters' contacts are known so a
// same-frame trade resolves from one snapshot.
struct Decision {
    int attacker = 0;
    int move_index = 0;
    int defense_move = -1;
    int hit_index = 0;
    FullContact contact = FullContact::Hit;
};

T8_FULL_HD inline void step(const Tables& tables, State& state, const Input& p1, const Input& p2, Events& events) {
    events = Events{};
    if (state.winner >= 0) return;
    FullEngineStage& stage = state.stage;
    const FullEngineConfig& config = tables.config;
    const Input inputs[2] = {p1, p2};

    // 0. The Heat timer ticks first, so Heat activated on any frame lasts exactly its duration.
    for (Fighter& fighter : state.fighters) {
        if (fighter.ki_charge_frames > 0 && --fighter.ki_charge_frames == 0) fighter.ki_charge_boost = false;
        if (fighter.heat && --fighter.heat_frames <= 0) {
            fighter.heat = false;
            fighter.heat_frames = 0;
            fighter.heat_dash_from = kNoContact;
        }
    }

    // 1. Start new actions. Legality is judged for both fighters before
    // either acts, so a throw break and a Heat Dash on the same frame both count.
    bool accepted[2];
    for (int player = 0; player < 2; ++player) accepted[player] = legal(tables, state, player, inputs[player]);
    for (int player = 0; player < 2; ++player) {
        Fighter& fighter = state.fighters[player];
        Fighter& other = state.fighters[1 - player];
        const Input& input = inputs[player];
        if (!accepted[player]) continue;
        if (input.move < 0 && is_throw_break(input.universal)) {
            // One attempt: the first break input counts, right or wrong.
            fighter.throw_break_attempted = true;
            if (fighter.throw_break_frames > 0 && other.move >= 0 && other.pending_throw_hit >= 0) {
                const Move& thrower_move = move_of(tables, 1 - player, other.move);
                if (breaks(hit_of(tables, 1 - player, thrower_move, other.pending_throw_hit).throw_break,
                           input.universal)) {
                    break_throw(tables, state, 1 - player, events);
                }
            }
            continue;
        }
        if (input.move < 0 && input.universal == FullUniversal::HeatDash) {
            const Move& spec = move_of(tables, player, fighter.move);
            const bool on_block = static_cast<FullContact>(fighter.heat_dash_from) == FullContact::Blocked;
            const int frames = config.heat_dash_frames;
            if (on_block) {
                other.stun = imax(0, frames + spec.heat_dash_block_advantage);
                other.in_blockstun = other.stun > 0;
            } else {
                const int advantage = frames + spec.heat_dash_hit_advantage;
                other.in_blockstun = false;
                // The published outcome replaces the engager's own (a launch becomes "+43d").
                other.posture = FullPosture::Standing;
                other.posture_frames = 0;
                other.crumpled = false;
                switch (spec.heat_dash_hit_effect) {
                    case FullHitEffect::Launch:
                        other.posture = FullPosture::Airborne;
                        other.posture_frames = imax(1, advantage);
                        other.combo_hits = imax(1, other.combo_hits);
                        other.stun = 0;
                        break;
                    case FullHitEffect::Knockdown:
                    case FullHitEffect::Crumple:
                        other.stun = imax(1, advantage);
                        other.crumpled = true;  // goes down when the stun ends
                        break;
                    case FullHitEffect::Stun:
                        other.stun = imax(0, advantage);
                        break;
                }
            }
            release_throw(state, player);
            fighter.move = -1;
            fighter.universal = FullUniversal::HeatDash;
            fighter.heat = false;
            fighter.heat_frames = 0;
            fighter.heat_dash_from = kNoContact;
            fighter.action_frame = 0;
            fighter.hits_landed = 0;
            continue;
        }
        fighter.heat_dash_from = kNoContact;
        if (input.move >= 0) {
            const Move& spec = move_of(tables, player, input.move);
            fighter.move = input.move;
            fighter.universal = FullUniversal::Idle;
            fighter.stance = -1;
            if (spec.has(kEngagesHeat) && fighter.heat_available && !fighter.heat) {
                activate_heat(tables, state, player, events);
                gain_resource(tables, state, player, spec.resource_gain_heat_activation);
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
            fighter.move_boosted = fighter.ki_charge_boost && !spec.has(kKiCharge);
            if (!spec.has(kKiCharge)) fighter.ki_charge_boost = false;
            if (spec.has(kKiCharge)) {
                fighter.ki_charge_frames = config.ki_charge_frames;
                fighter.ki_charge_boost = true;
            }
            gain_resource(tables, state, player, spec.resource_gain_start);
            if (spec.has(kConsumesHeat)) {
                fighter.heat = false;
                fighter.heat_frames = 0;
            }
            if (spec.has(kConsumesRage)) fighter.rage = false;
            if (spec.self_damage > 0.0 && !(spec.has(kSelfDamageWithoutHeatOnly) && fighter.heat)) {
                // Self-damage never K.O.s the user.
                deal_damage(tables, state, player, spec.self_damage - spec.self_recoverable, false, false, events);
                deal_damage(tables, state, player, spec.self_recoverable, true, false, events);
            }
        } else if (input.universal == FullUniversal::WakeupStand) {
            fighter.posture = FullPosture::Wakeup;
            fighter.posture_frames = config.wakeup_stand_frames;
            fighter.move = -1;
            fighter.universal = FullUniversal::Idle;
        } else if (input.universal == FullUniversal::WakeupRoll || input.universal == FullUniversal::TechRoll) {
            const bool tech = input.universal == FullUniversal::TechRoll;
            fighter.posture = FullPosture::Wakeup;
            fighter.posture_frames = tech ? config.tech_roll_frames : config.wakeup_roll_frames;
            fighter.x -= facing(state, player) * (tech ? 0.5 : 1.0) * config.wakeup_roll_distance;
            fighter.move = -1;
            fighter.universal = FullUniversal::Idle;
        } else {
            fighter.move = -1;
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
    bool sidestepping = false;
    for (const Fighter& fighter : state.fighters) {
        sidestepping = sidestepping || (fighter.move < 0 && (fighter.universal == FullUniversal::SidestepLeft ||
                                                            fighter.universal == FullUniversal::SidestepRight));
    }
    for (int player = 0; player < 2; ++player) {
        Fighter& fighter = state.fighters[player];
        const double forward = facing(state, player);
        if (fighter.move >= 0) {
            const Move& spec = move_of(tables, player, fighter.move);
            const bool holding_throw = fighter.pending_throw_hit >= 0;
            if (!holding_throw || fighter.action_frame < spec.total_frames) ++fighter.action_frame;
            const int approach_frames = spec.hit_count == 0 ? 0 : hit_of(tables, player, spec, 0).first_active_frame - 1;
            if (approach_frames > 0 && fighter.action_frame <= approach_frames) {
                fighter.x += forward * spec.travel / approach_frames;
            }
            if (fighter.action_frame >= spec.total_frames && !holding_throw) {
                fighter.stance = fighter.move_hit && spec.result_stance_on_hit >= 0 ? spec.result_stance_on_hit
                    : fighter.move_blocked && spec.result_stance_on_block >= 0 ? spec.result_stance_on_block
                    : spec.result_stance;
                fighter.stance_frames = 0;
                fighter.back_turned = spec.has(kResultBackTurned);
                fighter.move = -1;
                fighter.heat_dash_from = kNoContact;
                fighter.action_frame = universal_duration(FullUniversal::Idle, config);
                if (spec.has(kResultCrouching)) {
                    fighter.posture = FullPosture::Crouching;
                    fighter.universal = FullUniversal::Crouch;
                } else if (fighter.posture == FullPosture::Crouching && fighter.stance < 0) {
                    fighter.posture = FullPosture::Standing;
                }
            }
        } else if (fighter.posture == FullPosture::Standing || fighter.posture == FullPosture::Crouching) {
            ++fighter.action_frame;
            // Stances: time limit and periodic pulses while idle in the stance.
            const Stance* rule = stance_rule(tables, state, player);
            if (rule && fighter.stun == 0) {
                ++fighter.stance_frames;
                if (rule->pulse_interval_frames > 0 && fighter.stance_frames % rule->pulse_interval_frames == 0) {
                    regain(fighter, rule->pulse_recoverable, 0.0, config.max_health);
                    gain_resource(tables, state, player, rule->pulse_resource);
                }
                if (rule->has_max_frames && fighter.stance_frames >= rule->max_frames) {
                    fighter.stance = -1;
                    fighter.stance_frames = 0;
                }
            }
            switch (fighter.universal) {
                case FullUniversal::WalkForward: fighter.x += forward * config.walk_speed; break;
                case FullUniversal::WalkBack: fighter.x -= forward * config.walk_speed; break;
                case FullUniversal::DashForward:
                    if (fighter.action_frame <= config.dash_frames)
                        fighter.x += forward * config.dash_distance / config.dash_frames;
                    break;
                case FullUniversal::HeatDash:
                    if (fighter.action_frame <= config.heat_dash_frames)
                        fighter.x += forward * config.heat_dash_distance / config.heat_dash_frames;
                    break;
                case FullUniversal::Backdash:
                    if (fighter.action_frame <= config.backdash_frames)
                        fighter.x -= forward * config.backdash_distance / config.backdash_frames;
                    break;
                case FullUniversal::SidestepLeft:
                case FullUniversal::SidestepRight:
                    if (fighter.action_frame <= config.sidestep_frames) {
                        // Axis convention (shared with tracking): a fighter facing +x has its left at -axis.
                        const double side = fighter.universal == FullUniversal::SidestepLeft ? -1.0 : 1.0;
                        fighter.axis += side * forward * config.sidestep_distance / config.sidestep_frames;
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
                if (fighter.grounded_elapsed >= 4 * config.knockdown_frames) {
                    fighter.posture = FullPosture::Wakeup;
                    fighter.posture_frames = config.wakeup_stand_frames;
                }
                break;
            case FullPosture::Wakeup:
                if (--fighter.posture_frames <= 0) {
                    fighter.posture = FullPosture::Standing;
                    fighter.combo_hits = 0;
                    fighter.tornado_used = false;
                    fighter.wall_splat_used = false;
                    fighter.wall_scaled = false;
                    fighter.action_frame = universal_duration(FullUniversal::Idle, config);
                    fighter.universal = FullUniversal::Idle;
                }
                break;
            default: break;
        }
    }
    if (!sidestepping) {
        Fighter& a = state.fighters[0];
        Fighter& b = state.fighters[1];
        const double gap = a.axis - b.axis;
        const double shift = dmin(dabs(gap) / 2.0, config.axis_realign_per_frame / 2.0);
        a.axis -= copysign(shift, gap);
        b.axis += copysign(shift, gap);
    }

    // 3. Walls and body collision.
    for (Fighter& fighter : state.fighters) fighter.x = dclamp(fighter.x, stage.left_wall, stage.right_wall);
    {
        Fighter& a = state.fighters[0];
        Fighter& b = state.fighters[1];
        const double minimum = 2.0 * config.body_radius;
        const double dx = b.x - a.x;
        if (dabs(a.axis - b.axis) < minimum && dabs(dx) < minimum) {
            const double push = (minimum - dabs(dx)) / 2.0;
            const double direction = dx >= 0.0 ? 1.0 : -1.0;
            a.x -= direction * push;
            b.x += direction * push;
            Fighter* pair[2] = {&a, &b};
            for (int index = 0; index < 2; ++index) {
                Fighter* fighter = pair[index];
                const double overflow = fighter->x < stage.left_wall ? stage.left_wall - fighter->x
                    : fighter->x > stage.right_wall ? stage.right_wall - fighter->x : 0.0;
                if (overflow != 0.0) {
                    fighter->x += overflow;
                    pair[1 - index]->x += overflow;  // the other fighter absorbs it
                }
            }
            for (Fighter& fighter : state.fighters) fighter.x = dclamp(fighter.x, stage.left_wall, stage.right_wall);
        }
    }

    // 4. Decide contacts from one snapshot.
    Decision decisions[2];
    int decision_count = 0;
    for (int attacker = 0; attacker < 2; ++attacker) {
        const Fighter& own = state.fighters[attacker];
        const Fighter& target = state.fighters[1 - attacker];
        if (own.move < 0 || own.pending_throw_hit >= 0 || target.throw_break_frames > 0) continue;
        const Move& spec = move_of(tables, attacker, own.move);
        const Move* defense = target.move >= 0 ? &move_of(tables, 1 - attacker, target.move) : nullptr;
        const int defense_frame = target.action_frame;
        const Stance* target_rule = target.move >= 0 ? nullptr : stance_rule(tables, state, 1 - attacker);
        for (int index = 0; index < spec.hit_count; ++index) {
            const Hit& hit = hit_of(tables, attacker, spec, index);
            if ((own.hits_landed >> static_cast<unsigned>(index)) & 1U) continue;
            if (own.action_frame < hit.first_active_frame ||
                own.action_frame >= hit.first_active_frame + hit.active_frames) continue;
            const double reach = own.installed && spec.has(kHasInstallReach) ? spec.install_reach : hit.reach;
            if (dabs(target.x - own.x) > reach) continue;
            const double lateral = (target.axis - own.axis) * facing(state, attacker);
            if (!(hit.flags & kHoming) && (lateral < -hit.tracking_left || lateral > hit.tracking_right)) continue;
            const bool target_airborne = target.posture == FullPosture::Airborne ||
                (defense && defense->airborne.contains(defense_frame));
            if (target.posture == FullPosture::Grounded && !((hit.flags & kGroundHit) || is_low(hit.level))) continue;
            if (target.posture == FullPosture::Crouching &&
                (hit.level == FullHitLevel::High || hit.level == FullHitLevel::Throw)) continue;
            if (target_airborne && hit.level == FullHitLevel::Throw) continue;

            Decision decision{attacker, own.move, target.move, index, FullContact::Hit};
            const bool high_or_mid_strike = is_strike(hit.level) && !is_low(hit.level);
            if (defense && defense->invincible.contains(defense_frame)) {
                decision.contact = FullContact::Evaded;
            } else if (defense &&
                       (((hit.level == FullHitLevel::High || hit.level == FullHitLevel::Throw) &&
                         defense->high_crush.contains(defense_frame)) ||
                        (is_low(hit.level) && defense->low_crush.contains(defense_frame)))) {
                decision.contact = FullContact::Crushed;
            } else if (defense && high_or_mid_strike && !(hit.flags & kReversalBreak) &&
                       defense->reversal.contains(defense_frame)) {
                decision.contact = FullContact::Reversed;
            } else if (defense && !(hit.flags & kUnparryable) &&
                       (((defense->parry_levels & level_class_bits(hit.level)) && defense->parry.contains(defense_frame)) ||
                        (target.move_in_heat && (defense->heat_parry_levels & level_class_bits(hit.level)) &&
                         defense->heat_parry.contains(defense_frame)))) {
                decision.contact = FullContact::Parried;
            } else if (!defense && !(hit.flags & kUnparryable) && target_rule && target.stun == 0 &&
                       target.posture == FullPosture::Standing && (target_rule->auto_parry & level_class_bits(hit.level))) {
                decision.contact = FullContact::Parried;  // automatic stance parry (Jun: GEN against lows and throws)
            } else if (defense && is_strike(hit.level) &&
                       (defense->armor.contains(defense_frame) ||
                        (high_or_mid_strike && defense->power_crush.contains(defense_frame)))) {
                decision.contact = FullContact::Armored;
            } else {
                const bool stance_guard = (!target_rule || target_rule->can_guard) && !target.back_turned &&
                    target.ki_charge_frames == 0;
                const bool standing_guard = stance_guard && target.posture == FullPosture::Standing && target.move < 0 &&
                    (target.universal == FullUniversal::Idle || target.universal == FullUniversal::WalkBack) &&
                    (target.stun == 0 || target.in_blockstun);
                const bool crouching_guard = stance_guard && target.posture == FullPosture::Crouching &&
                    target.move < 0 && target.universal == FullUniversal::Crouch &&
                    (target.stun == 0 || target.in_blockstun);
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
            decisions[decision_count++] = decision;
            break;  // at most one new hit per attacker per frame
        }
    }

    // 5. Apply contacts. A held throw whose break window closes unbroken lands
    // first; its stun counts from the thrower's current frame, so hit
    // advantage stays exact however long the window held it.
    for (int player = 0; player < 2; ++player) {
        Fighter& held = state.fighters[player];
        Fighter& thrower = state.fighters[1 - player];
        if (held.throw_break_frames <= 0 || --held.throw_break_frames > 0) continue;
        held.throw_break_attempted = false;
        if (thrower.move < 0 || thrower.pending_throw_hit < 0) continue;
        const int move_index = thrower.move;
        const int hit_index = thrower.pending_throw_hit;
        const Move& spec = move_of(tables, 1 - player, move_index);
        thrower.pending_throw_hit = -1;
        apply_contact(tables, state, 1 - player, move_index, -1, hit_index, FullContact::Hit,
                      thrower.action_frame - hit_of(tables, 1 - player, spec, hit_index).first_active_frame, events);
        if (thrower.move >= 0 && thrower.action_frame >= spec.total_frames) {
            thrower.stance = spec.result_stance;
            thrower.move = -1;
            thrower.action_frame = universal_duration(FullUniversal::Idle, config);
        }
    }
    for (int index = 0; index < decision_count; ++index) {
        const Decision& decision = decisions[index];
        Fighter& attacker = state.fighters[decision.attacker];
        Fighter& defender = state.fighters[1 - decision.attacker];
        attacker.hits_landed |= 1U << static_cast<unsigned>(decision.hit_index);
        if (decision.contact == FullContact::ThrowHeld) {
            if (attacker.move < 0) continue;  // the thrower was hit on the same frame
            attacker.pending_throw_hit = decision.hit_index;
            defender.throw_break_frames = config.throw_break_window;
            defender.throw_break_attempted = false;
            defender.stance = -1;
            defender.move = -1;
            defender.universal = FullUniversal::Idle;
            defender.stun = 0;
            defender.in_blockstun = false;
            defender.heat_dash_from = kNoContact;
            events.add_contact(decision.attacker, decision.hit_index, FullContact::ThrowHeld, 0.0);
            continue;
        }
        apply_contact(tables, state, decision.attacker, decision.move_index, decision.defense_move,
                      decision.hit_index, decision.contact, 0, events);
    }

    const bool p1_down = state.fighters[0].health <= 0.0;
    const bool p2_down = state.fighters[1].health <= 0.0;
    if (p1_down || p2_down) state.winner = p1_down && p2_down ? 2 : (p1_down ? 1 : 0);
    ++state.frame;
}

}  // namespace t8::v2::full_core

namespace t8::v2 {

// Host-side owner of a matchup's tables; view() points into its vectors.
struct FullCompiledTables {
    std::array<std::vector<full_core::Move>, 2> moves;
    std::array<std::vector<full_core::Hit>, 2> hits;
    std::array<std::vector<full_core::Stance>, 2> stances;
    std::array<full_core::Rules, 2> rules{};
    FullEngineConfig config{};

    [[nodiscard]] full_core::Tables view() const;
};

// Validates and flattens movesets (stable-id references become indices).
[[nodiscard]] FullCompiledTables compile_full_tables(const std::array<std::vector<FullMoveSpec>, 2>& movesets,
                                                     const FullEngineConfig& config,
                                                     const std::array<FullCharacterRules, 2>& rules);

[[nodiscard]] full_core::State to_core_state(const FullEngineState& state);
[[nodiscard]] FullEngineState from_core_state(const full_core::State& state);
[[nodiscard]] FullFrameEvents from_core_events(const full_core::Events& events);
[[nodiscard]] full_core::Input to_core_input(const FullInput& input, std::size_t move_count);

}  // namespace t8::v2
