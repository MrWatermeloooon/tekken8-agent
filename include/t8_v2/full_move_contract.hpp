#pragma once

// Full-move environment contract shared by the CUDA environment and its CPU
// reference: candidate inputs and legality, the decision rule, the scripted
// opponent, and the observation encoding. All functions are plain data in,
// plain data out, so the GPU and CPU produce identical results.

#include "t8_v2/full_combat_core.hpp"
#include "t8_v2/full_move_roster.hpp"

#include <cstdint>

namespace t8::v2::full_core {

// 64 state features, then the fighter's and the opponent's characters (one-hot
// over the roster), so one policy can play and face every character.
constexpr int kFullMoveStateFeatures = 64;
constexpr int kFullMoveObservationSize = kFullMoveStateFeatures + 2 * static_cast<int>(kRosterCharacterCount);
// Privileged: the full simulator state of both fighters, including the
// opponent's current move. A deployable screen-only contract needs the live
// estimators; this one is for simulator training and is labeled as such.
constexpr const char* kFullMoveObservationContract = "full-move-privileged-148-v2";

// One fighter's candidate list on the device or host.
struct CandidateMap {
    const int* moves = nullptr;  // engine move per candidate (-1: universal or unbound)
    int count = 0;
};

T8_FULL_HD inline bool candidate_input(const CandidateMap& map, int candidate, Input& input) {
    if (candidate < 0 || candidate >= map.count) return false;
    if (candidate < kFullMoveUniversalCount) return universal_candidate_input(candidate, input);
    const int move = map.moves[candidate];
    if (move < 0) return false;  // the engine could not bind this row
    input.move = move;
    input.universal = FullUniversal::Idle;
    return true;
}

// Neutral is always allowed: an explicit "wait", which the engine ignores
// while the fighter cannot act. Every other candidate must be legal now.
T8_FULL_HD inline bool candidate_legal(const Tables& tables, const State& state, int player, const CandidateMap& map,
                                       int candidate) {
    if (candidate == 0) return true;
    Input input{};
    if (!candidate_input(map, candidate, input)) return false;
    if (candidate == 6 && state.fighters[player].posture != FullPosture::Crouching) return false;  // stand
    return legal(tables, state, player, input);
}

// Whether a fighter has a real choice this frame (a policy decision point).
T8_FULL_HD inline bool has_decision(const Tables& tables, const State& state, int player) {
    if (state.winner >= 0) return false;
    if (actionable(tables, state, player)) return true;
    const Fighter& fighter = state.fighters[player];
    if (fighter.move < 0 && fighter.action_frame >= 1 &&
        (fighter.universal == FullUniversal::SidestepLeft || fighter.universal == FullUniversal::SidestepRight) &&
        fighter.stun == 0 && fighter.posture == FullPosture::Standing) {
        return true;  // sidestep moves ("SS.") are performed out of the sidestep
    }
    const Input breaks{-1, FullUniversal::ThrowBreak};
    const Input dash{-1, FullUniversal::HeatDash};
    const Input wake{-1, FullUniversal::WakeupStand};
    const Input tech{-1, FullUniversal::TechRoll};
    return legal(tables, state, player, breaks) || legal(tables, state, player, dash) ||
        legal(tables, state, player, wake) || legal(tables, state, player, tech);
}

// Counter-based random numbers: the same (seed, stream, counter) gives the
// same value on the CPU and the GPU.
T8_FULL_HD inline std::uint64_t mix64(std::uint64_t value) {
    value += 0x9E3779B97F4A7C15ULL;
    value = (value ^ (value >> 30U)) * 0xBF58476D1CE4E5B9ULL;
    value = (value ^ (value >> 27U)) * 0x94D049BB133111EBULL;
    return value ^ (value >> 31U);
}
T8_FULL_HD inline std::uint64_t random_bits(std::uint64_t seed, std::uint64_t stream, std::uint64_t counter) {
    return mix64(mix64(seed ^ mix64(stream)) ^ counter);
}
T8_FULL_HD inline int random_below(std::uint64_t bits, int bound) {
    return static_cast<int>(bits % static_cast<std::uint64_t>(bound));
}
T8_FULL_HD inline double random_unit(std::uint64_t bits) {
    return static_cast<double>(bits >> 11U) * (1.0 / 9007199254740992.0);
}

// Scripted full-move opponent: approaches from range, guards sometimes up
// close, and otherwise plays a random legal candidate (moves preferred up
// close). `aggression` in [0, 1] shifts the move share.
T8_FULL_HD inline int scripted_candidate(const Tables& tables, const State& state, int player, const CandidateMap& map,
                                         double aggression, std::uint64_t seed, std::uint64_t stream,
                                         std::uint64_t counter) {
    const Fighter& own = state.fighters[player];
    const Fighter& other = state.fighters[1 - player];
    const bool close = dabs(own.x - other.x) < 1.6;
    std::uint64_t draw = 0;
    const auto next = [&]() { return random_bits(seed, stream, counter * 64U + draw++); };
    const int roll = random_below(next(), 100);
    if (!close && roll < 55) {
        const int approach = roll < 30 ? 3 : 1;  // dash or walk forward
        if (candidate_legal(tables, state, player, map, approach)) return approach;
    }
    if (close && roll < 15) {
        const int guard = roll < 8 ? 0 : 5;  // standing or crouching guard
        if (candidate_legal(tables, state, player, map, guard)) return guard;
    }
    const int move_share = static_cast<int>((close ? 55.0 : 15.0) + 30.0 * aggression);
    if (map.count > kFullMoveUniversalCount && random_below(next(), 100) < move_share) {
        for (int attempt = 0; attempt < 24; ++attempt) {
            const int candidate = kFullMoveUniversalCount + random_below(next(), map.count - kFullMoveUniversalCount);
            if (candidate_legal(tables, state, player, map, candidate)) return candidate;
        }
    }
    for (int attempt = 0; attempt < 24; ++attempt) {
        const int candidate = random_below(next(), kFullMoveUniversalCount);
        if (candidate_legal(tables, state, player, map, candidate)) return candidate;
    }
    return 0;
}

// ---------------------------------------------------------------------------
// Tiered scripted bots. Each profile is a set of skill dials over the same
// state the policy sees: blocking the incoming hit's level once it has had
// `reaction` frames to see it, sidestepping slow linear hits, punishing
// whiffs and unsafe blocked moves, breaking throws, and picking moves by frame
// data ("smart") instead of at random. Tier 0 is three training dummies; tier 5
// is near-perfect defense; tier 6 plays the archetypal pro styles (movement and
// whiff punishing, frame-trap pressure, patient defense, mix-ups). Rolls that decide a reaction are made once per
// incoming attack (keyed by its start frame), so a skill of 0.6 means 60% of
// attacks are answered, not 60% of frames.
struct ScriptedProfile {
    int tier = 0;
    int fixed = -1;            // dummies: always this universal candidate (-1: a real bot)
    float aggression = 0.0F;   // how readily it attacks when in range
    float block = 0.0F;        // answers a seen attack by guarding the right level
    int reaction = 99;         // frames into an attack before it can respond
    float punish = 0.0F;       // punishes whiffs and blocked unsafe moves
    float throw_break = 0.0F;  // breaks a throw with the right button
    float sidestep = 0.0F;     // sidesteps a seen, slow, non-homing hit
    float smart = 0.0F;        // move choice by frame data instead of at random
    float distance = 1.0F;     // preferred spacing (m)
    float keepaway = 0.0F;     // backs off to its spacing
    float low_bias = 0.0F;     // extra preference for lows
    float throw_bias = 0.0F;   // extra preference for throws
    float heat_use = 0.0F;     // Heat Burst, Heat Smash, Rage Art usage
    float movement = 0.3F;     // sidesteps/steps while waiting
    float pressure = 0.0F;     // keeps attacking while the opponent is in blockstun (frame traps)
};

constexpr int kScriptedHandmadeCount = 30;   // ids 0-29: the hand-tuned bots above
constexpr int kScriptedTierCount = 7;
constexpr int kScriptedGeneratedPerTier = 43;  // ids 30+: generated, tier = (id - 30) / 43
constexpr int kScriptedProfileCount = kScriptedHandmadeCount + kScriptedTierCount * kScriptedGeneratedPerTier;  // 331

T8_FULL_HD inline ScriptedProfile generated_profile(int id);

T8_FULL_HD inline ScriptedProfile scripted_profile(int id) {
    if (id >= kScriptedHandmadeCount) return generated_profile(id);
    ScriptedProfile p{};
    // Shared skill per tier for the real bots (tiers 2-5).
    const auto skill = [&](int tier, float block, int reaction, float punish, float breaks, float side, float smart) {
        p.tier = tier; p.block = block; p.reaction = reaction; p.punish = punish; p.throw_break = breaks;
        p.sidestep = side; p.smart = smart;
    };
    // Styles on top of a tier's skill.
    const auto rushdown = [&]() { p.aggression = 0.85F; p.distance = 0.85F; p.movement = 0.2F; };
    const auto turtle = [&]() { p.aggression = 0.4F; p.distance = 1.15F; p.block = static_cast<float>(dmin(0.98, p.block + 0.1)); p.punish = static_cast<float>(dmin(0.98, p.punish + 0.1)); };
    const auto keepaway = [&]() { p.aggression = 0.45F; p.distance = 1.8F; p.keepaway = 0.7F; };
    const auto sidestepper = [&]() { p.aggression = 0.5F; p.distance = 1.1F; p.sidestep = static_cast<float>(dmin(0.9, p.sidestep + 0.3)); p.movement = 0.6F; };
    const auto mixup = [&]() { p.aggression = 0.65F; p.distance = 0.95F; p.low_bias = 1.5F; p.throw_bias = 0.8F; };
    const auto heat_user = [&]() { p.aggression = 0.6F; p.distance = 1.0F; p.heat_use = 1.0F; };
    switch (id) {
        // Tier 0: dummies.
        case 0: p.tier = 0; p.fixed = 1; break;   // walks forward, never guards: a punching bag
        case 1: p.tier = 0; p.fixed = 0; break;   // standing guard: blocks highs and mids
        case 2: p.tier = 0; p.fixed = 5; break;   // crouching guard: blocks lows, ducks highs
        // Tier 1: beginners with habits and no defense.
        case 3: p.tier = 1; p.aggression = 1.0F; p.distance = 1.0F; break;                       // masher
        case 4: p.tier = 1; p.aggression = 0.8F; p.low_bias = 3.0F; p.smart = 0.3F; break;       // low spammer
        case 5: p.tier = 1; p.aggression = 0.7F; p.distance = 0.8F; p.throw_bias = 3.0F; p.smart = 0.3F; break;  // throw spammer
        case 6: p.tier = 1; p.aggression = 0.4F; p.distance = 2.0F; p.keepaway = 0.8F; break;    // backdasher
        // Tier 2: intermediate.
        case 7: skill(2, 0.42F, 22, 0.30F, 0.35F, 0.02F, 0.50F); rushdown(); break;
        case 8: skill(2, 0.42F, 22, 0.30F, 0.35F, 0.02F, 0.50F); turtle(); break;
        case 9: skill(2, 0.42F, 22, 0.30F, 0.35F, 0.02F, 0.50F); keepaway(); break;
        case 10: skill(2, 0.42F, 22, 0.30F, 0.35F, 0.02F, 0.50F); mixup(); break;
        // Tier 3: advanced.
        case 11: skill(3, 0.68F, 17, 0.60F, 0.58F, 0.10F, 0.75F); rushdown(); break;
        case 12: skill(3, 0.68F, 17, 0.60F, 0.58F, 0.10F, 0.75F); turtle(); break;
        case 13: skill(3, 0.68F, 17, 0.60F, 0.58F, 0.10F, 0.75F); keepaway(); break;
        case 14: skill(3, 0.68F, 17, 0.60F, 0.58F, 0.10F, 0.75F); sidestepper(); break;
        case 15: skill(3, 0.68F, 17, 0.60F, 0.58F, 0.10F, 0.75F); mixup(); break;
        case 16: skill(3, 0.68F, 17, 0.60F, 0.58F, 0.10F, 0.75F); heat_user(); break;
        // Tier 4: expert.
        case 17: skill(4, 0.80F, 15, 0.74F, 0.72F, 0.25F, 0.82F); rushdown(); break;
        case 18: skill(4, 0.80F, 15, 0.74F, 0.72F, 0.25F, 0.82F); turtle(); break;
        case 19: skill(4, 0.80F, 15, 0.74F, 0.72F, 0.25F, 0.82F); keepaway(); break;
        case 20: skill(4, 0.80F, 15, 0.74F, 0.72F, 0.25F, 0.82F); sidestepper(); break;
        case 21: skill(4, 0.80F, 15, 0.74F, 0.72F, 0.25F, 0.82F); mixup(); break;
        case 22: skill(4, 0.80F, 15, 0.74F, 0.72F, 0.25F, 0.82F); heat_user(); break;
        // Tier 5: master.
        case 23: skill(5, 0.91F, 12, 0.90F, 0.87F, 0.38F, 0.95F); rushdown(); break;
        case 24: skill(5, 0.91F, 12, 0.90F, 0.87F, 0.38F, 0.95F); turtle(); break;
        case 25: skill(5, 0.91F, 12, 0.90F, 0.87F, 0.38F, 0.95F); mixup(); p.heat_use = 0.7F; break;  // all-rounder
        // Tier 6: pro styles.
        case 26:  // movement and whiff punishing: backdashes out of range, punishes what whiffs
            skill(6, 0.97F, 9, 0.98F, 0.95F, 0.50F, 1.0F); keepaway(); p.distance = 1.7F; p.pressure = 0.3F; break;
        case 27:  // pressure: frame traps off plus-on-block moves, heavy Heat
            skill(6, 0.97F, 9, 0.98F, 0.95F, 0.50F, 1.0F); rushdown(); p.pressure = 0.85F; p.heat_use = 1.0F; break;
        case 28:  // patient defense: guards, sidesteps, punishes everything
            skill(6, 0.97F, 9, 0.98F, 0.95F, 0.50F, 1.0F); turtle(); p.pressure = 0.4F; break;
        default:  // mix-ups and sidesteps
            skill(6, 0.97F, 9, 0.98F, 0.95F, 0.50F, 1.0F); mixup(); p.sidestep = 0.6F; p.pressure = 0.6F;
            p.heat_use = 0.8F; break;
    }
    return p;
}

// Generated bots: each takes its tier's skill with its own ±10% variation and a
// random style mix, fixed by its id (bot 187 is always the same bot).
T8_FULL_HD inline ScriptedProfile generated_profile(int id) {
    ScriptedProfile p{};
    const int index = id - kScriptedHandmadeCount;
    p.tier = imin(kScriptedTierCount - 1, index / kScriptedGeneratedPerTier);
    std::uint64_t state = 0x9e3779b97f4a7c15ULL * static_cast<std::uint64_t>(id + 1) + 0x632be59bd9b4e5f5ULL;
    const auto next = [&]() {
        state = mix64(state + 0x9e3779b97f4a7c15ULL);
        return static_cast<double>(state >> 11U) * (1.0 / 9007199254740992.0);
    };
    const auto vary = [&](double value, double low, double high) { return dclamp(value * (0.9 + 0.2 * next()), low, high); };
    // Tier skill: block, reaction, punish, throw break, sidestep, smart.
    const double table[kScriptedTierCount][6] = {
        {0.00, 99, 0.00, 0.00, 0.00, 0.10}, {0.05, 40, 0.05, 0.08, 0.00, 0.20}, {0.42, 22, 0.30, 0.35, 0.02, 0.50},
        {0.68, 17, 0.60, 0.58, 0.10, 0.75}, {0.80, 15, 0.74, 0.72, 0.25, 0.82}, {0.91, 12, 0.90, 0.87, 0.38, 0.95},
        {0.97, 9, 0.98, 0.95, 0.50, 1.00}};
    const double* skill = table[p.tier];
    p.block = static_cast<float>(vary(skill[0], 0.0, 0.98));
    p.reaction = static_cast<int>(dclamp(skill[1] + (next() * 4.0 - 2.0), 8.0, 99.0));
    p.punish = static_cast<float>(vary(skill[2], 0.0, 0.99));
    p.throw_break = static_cast<float>(vary(skill[3], 0.0, 0.98));
    p.sidestep = static_cast<float>(vary(skill[4], 0.0, 0.8));
    p.smart = static_cast<float>(vary(skill[5], 0.0, 1.0));
    // Style.
    const int style = static_cast<int>(next() * (p.tier >= 4 ? 9.0 : 8.0));
    p.aggression = static_cast<float>(0.35 + 0.5 * next());
    p.distance = static_cast<float>(0.85 + 0.3 * next());
    p.movement = static_cast<float>(0.1 + 0.5 * next());
    switch (style) {
        case 0: p.aggression = static_cast<float>(0.75 + 0.25 * next()); p.distance = 0.85F; break;          // rushdown
        case 1: p.aggression = static_cast<float>(0.3 + 0.2 * next()); p.distance = 1.15F;                   // turtle
                p.block = static_cast<float>(dmin(0.98, p.block + 0.1)); p.punish = static_cast<float>(dmin(0.99, p.punish + 0.1)); break;
        case 2: p.distance = static_cast<float>(1.6 + 0.4 * next()); p.keepaway = static_cast<float>(0.5 + 0.4 * next()); break;  // keep-away
        case 3: p.sidestep = static_cast<float>(dmin(0.9, p.sidestep + 0.25 + 0.15 * next())); p.movement = 0.7F; break;          // sidestepper
        case 4: p.low_bias = static_cast<float>(0.8 + 1.2 * next()); p.throw_bias = static_cast<float>(0.4 + 0.8 * next()); break; // mix-up
        case 5: p.heat_use = static_cast<float>(0.6 + 0.4 * next()); break;                                                          // Heat user
        case 6: p.throw_bias = static_cast<float>(1.5 + 1.5 * next()); p.distance = 0.85F; break;                                  // throw-heavy
        case 7: p.low_bias = static_cast<float>(1.5 + 1.5 * next()); break;                                                          // low-heavy
        default: p.pressure = static_cast<float>(0.5 + 0.4 * next()); p.aggression = static_cast<float>(0.7 + 0.2 * next()); break; // pressure
    }
    if (p.tier == 0) {  // tier 0: dummies that rarely attack and never defend
        p.aggression = static_cast<float>(0.05 + 0.15 * next());
        p.block = p.punish = p.throw_break = p.sidestep = p.pressure = 0.0F;
        p.reaction = 99;
    }
    if (p.tier == 5 && next() < 0.5) p.pressure = static_cast<float>(dmax(p.pressure, 0.3 + 0.4 * next()));
    if (p.tier == 6) p.pressure = static_cast<float>(dmax(p.pressure, 0.4 + 0.4 * next()));  // pros always frame-trap
    return p;
}

// The opponent-behaviour value carried per environment ("aggression"): values in
// [0, 1] are the legacy random opponent's aggression, -(id + 1) is scripted bot id.
T8_FULL_HD inline float encode_bot(int id) { return -static_cast<float>(id + 1); }
T8_FULL_HD inline int decode_bot(double behaviour) { return behaviour < 0.0 ? static_cast<int>(-behaviour + 0.5) - 1 : -1; }

namespace bot_detail {
constexpr int kNeutral = 0, kWalkForward = 1, kWalkBack = 2, kDashForward = 3, kBackdash = 4, kCrouch = 5;
constexpr int kSidestepLeft = 8, kSidestepRight = 9, kBreak1 = 15, kBreak2 = 16, kBreak12 = 17;
constexpr int kWakeStand = 19, kWakeRoll = 20, kTechRoll = 21;

struct MoveFacts {
    bool valid = false;
    int startup = 99;
    double reach = 0.0;
    double damage = 0.0;
    bool safe = false;
    bool launcher = false;
    bool low = false;
    bool throw_move = false;
    std::uint64_t flags = 0;
};

T8_FULL_HD inline MoveFacts facts(const Tables& tables, int player, const CandidateMap& map, int candidate) {
    MoveFacts out{};
    if (candidate < kFullMoveUniversalCount || candidate >= map.count) return out;
    const int move = map.moves[candidate];
    if (move < 0) return out;
    const Move& spec = move_of(tables, player, move);
    out.flags = spec.flags;
    out.safe = spec.block_advantage >= -9;
    out.launcher = spec.hit_effect == FullHitEffect::Launch;
    for (int hit = 0; hit < spec.hit_count; ++hit) {
        const Hit& h = hit_of(tables, player, spec, hit);
        out.damage += h.damage;
        if (hit == 0) {
            out.startup = h.first_active_frame;
            out.reach = h.reach;
            out.low = h.level == FullHitLevel::Low || h.level == FullHitLevel::SpecialLow;
            out.throw_move = h.level == FullHitLevel::Throw;
        }
    }
    out.valid = spec.hit_count > 0;
    return out;
}

T8_FULL_HD inline double unit(std::uint64_t bits) { return static_cast<double>(bits >> 11U) * (1.0 / 9007199254740992.0); }
}  // namespace bot_detail

// The scripted bot's choice of candidate for `player` this frame.
T8_FULL_HD inline int scripted_bot_candidate(const Tables& tables, const State& state, int player, const CandidateMap& map,
                                             int profile_id, std::uint64_t seed, std::uint64_t stream,
                                             std::uint64_t counter) {
    using namespace bot_detail;
    const ScriptedProfile p = scripted_profile(profile_id);
    const auto ok = [&](int candidate) { return candidate_legal(tables, state, player, map, candidate); };
    if (p.fixed >= 0) return ok(p.fixed) ? p.fixed : kNeutral;
    const Fighter& own = state.fighters[player];
    const Fighter& other = state.fighters[1 - player];
    const double distance = dabs(own.x - other.x);
    std::uint64_t draw = 0;
    const auto roll = [&]() { return unit(random_bits(seed, stream, counter * 64U + draw++)); };
    // One roll per opponent attack and purpose, so a skill is a share of attacks answered.
    const std::uint64_t attack_key = static_cast<std::uint64_t>(state.frame - other.action_frame) * 4096U +
                                     static_cast<std::uint64_t>(other.move + 1);
    const auto attack_roll = [&](std::uint64_t purpose) {
        return unit(random_bits(seed, stream ^ (0x9e3779b97f4a7c15ULL * (purpose + 1)), attack_key));
    };

    // Throw break: one attempt; the right button with probability throw_break.
    if (own.throw_break_frames > 0 && !own.throw_break_attempted && other.move >= 0 && other.pending_throw_hit >= 0) {
        const Move& thrower = move_of(tables, 1 - player, other.move);
        const FullThrowBreak rule = hit_of(tables, 1 - player, thrower, other.pending_throw_hit).throw_break;
        int right = kBreak12;
        if (rule == FullThrowBreak::One || rule == FullThrowBreak::OneOrTwo) right = kBreak1;
        if (rule == FullThrowBreak::Two) right = kBreak2;
        int choice = right;
        if (attack_roll(1) >= p.throw_break) choice = right == kBreak12 ? kBreak1 : (right == kBreak1 ? kBreak2 : kBreak12);
        if (ok(choice)) return choice;
    }
    // Getting up.
    if (own.posture == FullPosture::Grounded) {
        const int options[3] = {kWakeStand, kWakeRoll, kTechRoll};
        const int first = random_below(random_bits(seed, stream, counter * 64U + 63U), 3);
        for (int index = 0; index < 3; ++index) {
            if (ok(options[(first + index) % 3])) return options[(first + index) % 3];
        }
    }
    // Defense: an incoming hit seen in time is guarded at the right level or sidestepped.
    if (other.move >= 0) {
        const Move& spec = move_of(tables, 1 - player, other.move);
        for (int index = 0; index < spec.hit_count; ++index) {
            if ((other.hits_landed >> static_cast<unsigned>(index)) & 1U) continue;
            const Hit& hit = hit_of(tables, 1 - player, spec, index);
            if (other.action_frame >= hit.first_active_frame + hit.active_frames) continue;
            const int until = hit.first_active_frame - other.action_frame;
            const bool seen = other.action_frame >= p.reaction;
            if (!seen || distance > hit.reach + 0.4) break;
            const double answer = attack_roll(2);
            // A sidestep needs about 11 frames to clear 0.3 m of tracking; with less warning, guard.
            if (!(hit.flags & kHoming) && until >= 12 && answer < p.sidestep) {
                const int side = attack_roll(3) < 0.5 ? kSidestepLeft : kSidestepRight;
                if (ok(side)) return side;
            }
            if (answer < p.sidestep + p.block) {
                const bool crouch = hit.level == FullHitLevel::Low || hit.level == FullHitLevel::SpecialLow ||
                                    hit.level == FullHitLevel::Throw ||
                                    (hit.level == FullHitLevel::High && attack_roll(4) < 0.5);
                const int guard = crouch ? kCrouch : kNeutral;
                if (ok(guard)) return guard;
            }
            break;
        }
    }
    // Punish: the opponent whiffed, or is recovering from a blocked unsafe move.
    if (other.move >= 0 && actionable(tables, state, player)) {
        const Move& spec = move_of(tables, 1 - player, other.move);
        int last_active = 0;
        for (int index = 0; index < spec.hit_count; ++index) {
            const Hit& hit = hit_of(tables, 1 - player, spec, index);
            last_active = imax(last_active, hit.first_active_frame + hit.active_frames);
        }
        const int remaining = spec.total_frames - other.action_frame;
        int window = 0;
        if (other.action_frame >= last_active && !other.move_hit && !other.move_blocked) window = remaining;
        if (other.move_blocked && spec.block_advantage <= -10) window = imin(remaining, -spec.block_advantage);
        if (window > 0 && attack_roll(5) < p.punish) {
            int best = -1;
            double best_score = -1.0;
            for (int attempt = 0; attempt < 24; ++attempt) {
                const int candidate = kFullMoveUniversalCount + random_below(
                    random_bits(seed, stream, counter * 64U + 32U + static_cast<std::uint64_t>(attempt)),
                    imax(1, map.count - kFullMoveUniversalCount));
                const MoveFacts move = facts(tables, player, map, candidate);
                if (!move.valid || move.throw_move || move.startup > window || move.reach + 0.1 < distance) continue;
                const double score = move.damage + (move.launcher ? 25.0 : 0.0);
                if (score > best_score && ok(candidate)) {
                    best = candidate;
                    best_score = score;
                }
            }
            if (best >= 0) return best;
        }
    }
    // Heat and Rage: Heat Burst when available, Rage Art when close and in Rage.
    if (p.heat_use > 0.0F && actionable(tables, state, player)) {
        const bool want_heat = own.heat_available && !own.heat && roll() < 0.02 * p.heat_use;
        const bool want_rage = own.rage && distance < 1.3 && roll() < 0.05 * p.heat_use;
        if (want_heat || want_rage) {
            for (int candidate = kFullMoveUniversalCount; candidate < map.count; ++candidate) {
                const MoveFacts move = facts(tables, player, map, candidate);
                const bool wanted = want_rage ? (move.flags & kRequiresRage) != 0 : (move.flags & kEngagesHeat) != 0;
                if (wanted && ok(candidate)) return candidate;
            }
        }
    }
    // Pressure: while the opponent is still in blockstun this bot is plus; a fast move now
    // frame-traps anything slower than it.
    if (p.pressure > 0.0F && other.in_blockstun && actionable(tables, state, player) && roll() < p.pressure) {
        for (int attempt = 0; attempt < 16; ++attempt) {
            const int candidate = kFullMoveUniversalCount + random_below(
                random_bits(seed, stream, counter * 64U + 40U + static_cast<std::uint64_t>(attempt)),
                imax(1, map.count - kFullMoveUniversalCount));
            const MoveFacts move = facts(tables, player, map, candidate);
            if (move.valid && !move.throw_move && move.startup <= 14 && move.reach + 0.05 >= distance && ok(candidate)) {
                return candidate;
            }
        }
    }
    // Fundamentals (in proportion to smart): no button into an incoming attack, and none
    // while the opponent's blocked move left them plus; guard instead. Learned policies
    // otherwise open these bots almost only by counter-hitting their startup.
    if (other.move >= 0) {
        const Move& spec = move_of(tables, 1 - player, other.move);
        bool incoming = false;
        for (int index = 0; index < spec.hit_count; ++index) {
            const Hit& hit = hit_of(tables, 1 - player, spec, index);
            if (!((other.hits_landed >> static_cast<unsigned>(index)) & 1U) &&
                other.action_frame < hit.first_active_frame + hit.active_frames) {
                incoming = true;
            }
        }
        const bool opponent_plus = other.move_blocked && spec.block_advantage > 0;
        if ((incoming || opponent_plus) && distance < 2.0 && roll() < p.smart) {
            return kNeutral;  // standing guard
        }
    }
    // Offense: attack in range; otherwise manage spacing. A careless pick (1 - smart)
    // may be thrown out of range: low tiers whiff, high tiers almost never do.
    const double reach = 1.25;
    if (p.keepaway > 0.0F && distance > reach && distance < p.distance + 0.6 && roll() < 0.025 * p.aggression) {
        if (ok(kDashForward)) return kDashForward;  // keep-away styles go in to attack, then back out
    }
    // Keep-away styles commit once they are in range (they came in to attack).
    const double attack_rate = 0.12 * p.aggression * (p.keepaway > 0.0F ? 3.0 : 1.0);
    if (distance <= reach + 0.25 && roll() < attack_rate) {
        int best = -1;
        double best_score = -1e9;
        for (int attempt = 0; attempt < 12; ++attempt) {
            const int candidate = kFullMoveUniversalCount + random_below(
                random_bits(seed, stream, counter * 64U + 8U + static_cast<std::uint64_t>(attempt)),
                imax(1, map.count - kFullMoveUniversalCount));
            const MoveFacts move = facts(tables, player, map, candidate);
            if (!move.valid || !ok(candidate)) continue;
            if (move.reach + 0.05 < distance && roll() < 0.3 + 0.7 * p.smart) continue;  // out of range
            double score = roll() * 2.0;
            if (roll() < p.smart) {  // judged by frame data
                score += (move.safe ? 2.0 : -1.5) +
                         (move.startup <= 12 ? 2.5 : (move.startup <= 15 ? 0.5 : -2.0)) +
                         (move.reach + 0.05 >= distance ? 1.5 : -3.0) + move.damage / 30.0;
            }
            score += (move.low ? p.low_bias : 0.0) + (move.throw_move ? p.throw_bias : 0.0);
            if (score > best_score) {
                best = candidate;
                best_score = score;
            }
        }
        if (best >= 0) return best;
    }
    const double r = roll();
    if (distance > p.distance + 0.2) {
        const int approach = p.aggression > 0.7F && distance > 2.0 ? kDashForward : kWalkForward;
        if (ok(approach)) return approach;
    }
    if (distance < p.distance - 0.3 && p.keepaway > 0.0F && r < 0.08 * p.keepaway) {
        const int retreat = r < 0.03 * p.keepaway ? kBackdash : kWalkBack;
        if (ok(retreat)) return retreat;
    }
    if (r > 1.0 - 0.05 * p.movement) {
        const int side = r > 1.0 - 0.025 * p.movement ? kSidestepLeft : kSidestepRight;
        if (ok(side)) return side;
    }
    return kNeutral;  // standing guard while waiting
}

T8_FULL_HD inline void encode_posture(FullPosture posture, float* out) {
    for (int index = 0; index < 6; ++index) out[index] = static_cast<int>(posture) == index ? 1.0F : 0.0F;
}

T8_FULL_HD inline float clip01(double value) { return static_cast<float>(dclamp(value, 0.0, 1.0)); }
T8_FULL_HD inline float clip11(double value) { return static_cast<float>(dclamp(value, -1.0, 1.0)); }

// The opponent's or own current move as the observation describes it.
T8_FULL_HD inline void encode_move(const Tables& tables, const State& state, int player, float* out) {
    const Fighter& fighter = state.fighters[player];
    for (int index = 0; index < 8; ++index) out[index] = 0.0F;
    if (fighter.move < 0) return;
    const Move& spec = move_of(tables, player, fighter.move);
    out[0] = 1.0F;
    out[7] = clip01(static_cast<double>(spec.total_frames - fighter.action_frame) / 60.0);
    out[6] = clip11(static_cast<double>(spec.block_advantage) / 30.0);
    for (int hit = 0; hit < spec.hit_count; ++hit) {
        if ((fighter.hits_landed >> static_cast<unsigned>(hit)) & 1U) continue;
        const Hit& next = hit_of(tables, player, spec, hit);
        out[1 + level_index(next.level)] = 1.0F;
        out[5] = clip01(static_cast<double>(next.first_active_frame - fighter.action_frame) / 30.0);
        break;
    }
}

// Player-relative observation, kFullMoveObservationSize floats. own_character and
// opponent_character are roster ids (kRosterCharacterSlugs indices).
// perception_delay: the opponent's move (which move, its next hit's level and timing,
// its block advantage) is only visible once it is that many frames old, as for a human
// or the live screen pipeline; before that the opponent reads as not attacking. Its
// position, posture, stance, and stun are visible at once.
T8_FULL_HD inline void encode_observation(const Tables& tables, const State& state, int player, int round_frame_limit,
                                          int own_character, int opponent_character, float* out,
                                          int perception_delay = 0) {
    for (int index = kFullMoveStateFeatures; index < kFullMoveObservationSize; ++index) out[index] = 0.0F;
    if (own_character >= 0 && own_character < static_cast<int>(kRosterCharacterCount)) {
        out[kFullMoveStateFeatures + own_character] = 1.0F;
    }
    if (opponent_character >= 0 && opponent_character < static_cast<int>(kRosterCharacterCount)) {
        out[kFullMoveStateFeatures + static_cast<int>(kRosterCharacterCount) + opponent_character] = 1.0F;
    }
    const FullEngineConfig& config = tables.config;
    const Fighter& own = state.fighters[player];
    const Fighter& opp = state.fighters[1 - player];
    const double width = state.stage.right_wall - state.stage.left_wall;
    const double toward = facing(state, player);
    const double wall_behind = toward > 0.0 ? own.x - state.stage.left_wall : state.stage.right_wall - own.x;
    const double wall_ahead = toward > 0.0 ? state.stage.right_wall - opp.x : opp.x - state.stage.left_wall;
    out[0] = clip01(own.health / config.max_health);
    out[1] = clip01(opp.health / config.max_health);
    out[2] = clip01(own.recoverable_health / config.max_health);
    out[3] = clip01(opp.recoverable_health / config.max_health);
    out[4] = clip01(dabs(opp.x - own.x) / 4.0);
    out[5] = clip11((opp.axis - own.axis) * toward / 2.0);
    out[6] = clip01(wall_behind / dmax(width, 1e-9));
    out[7] = clip01(wall_ahead / dmax(width, 1e-9));
    encode_posture(own.posture, out + 8);
    encode_posture(opp.posture, out + 14);
    out[20] = clip01(own.stun / 60.0);
    out[21] = clip01(opp.stun / 60.0);
    out[22] = own.in_blockstun ? 1.0F : 0.0F;
    out[23] = opp.in_blockstun ? 1.0F : 0.0F;
    out[24] = actionable(tables, state, player) ? 1.0F : 0.0F;
    out[25] = actionable(tables, state, 1 - player) ? 1.0F : 0.0F;
    out[26] = own.heat ? 1.0F : 0.0F;
    out[27] = clip01(static_cast<double>(own.heat_frames) / dmax(1.0, config.heat_duration_frames));
    out[28] = own.heat_available ? 1.0F : 0.0F;
    out[29] = own.rage ? 1.0F : 0.0F;
    out[30] = own.rage_available ? 1.0F : 0.0F;
    out[31] = opp.heat ? 1.0F : 0.0F;
    out[32] = opp.rage ? 1.0F : 0.0F;
    const int own_stances = tables.side[player].stance_count;
    out[33] = own.stance >= 0 ? 1.0F : 0.0F;
    out[34] = own.stance >= 0 ? static_cast<float>(own.stance + 1) / static_cast<float>(own_stances + 1) : 0.0F;
    out[35] = opp.stance >= 0 ? 1.0F : 0.0F;
    const Rules& rules = tables.side[player].rules;
    out[36] = rules.resource_max > 0.0 ? clip01(own.resource / rules.resource_max) : 0.0F;
    out[37] = own.installed ? 1.0F : 0.0F;
    out[38] = own.back_turned ? 1.0F : 0.0F;
    out[39] = opp.back_turned ? 1.0F : 0.0F;
    out[40] = clip01(own.combo_hits / 10.0);
    out[41] = clip01(opp.combo_hits / 10.0);
    out[42] = own.throw_break_frames > 0 ? 1.0F : 0.0F;
    out[43] = own.pending_throw_hit >= 0 ? 1.0F : 0.0F;
    out[44] = own.heat_dash_from != kNoContact ? 1.0F : 0.0F;
    out[45] = clip01(static_cast<double>(own.grounded_elapsed) / dmax(1.0, config.knockdown_frames));
    out[46] = clip01(static_cast<double>(opp.grounded_elapsed) / dmax(1.0, config.knockdown_frames));
    out[47] = clip01(1.0 - static_cast<double>(state.frame) / dmax(1.0, round_frame_limit));
    out[48] = own.ki_charge_frames > 0 ? 1.0F : 0.0F;
    out[49] = opp.ki_charge_frames > 0 ? 1.0F : 0.0F;
    encode_move(tables, state, 1 - player, out + 50);  // 50..57: opponent's move
    if (opp.move >= 0 && opp.action_frame < perception_delay) {
        for (int index = 50; index < 58; ++index) out[index] = 0.0F;  // not recognized yet
    }
    out[58] = own.move >= 0 ? 1.0F : 0.0F;
    out[59] = own.move >= 0
        ? clip01(static_cast<double>(move_of(tables, player, own.move).total_frames - own.action_frame) / 60.0) : 0.0F;
    out[60] = own.techable ? 1.0F : 0.0F;
    out[61] = clip01(own.posture_frames / 60.0);
    out[62] = clip01(opp.posture_frames / 60.0);
    out[63] = 1.0F;
}

}  // namespace t8::v2::full_core

namespace t8::v2::full_core {

// ---------------------------------------------------------------- memory (--memory)
//
// The observation above is one moment. With --memory a run adds a short history after it
// (kFullMoveMemoryFeatures more inputs): each fighter's last kMemorySlots moves (hit level,
// how they ended, how long ago) and decaying traces of its habits over about a second
// (crouching, backing off, sidestepping, guarding, attacking). The opponent's newest move
// enters its history only once it is perception_delay frames old, like the move itself.
// Memory lives next to the State (it is not part of the engine or of saved states), is
// updated after every frame, and is all zeros at a round's start.
constexpr int kMemorySlots = 4;
constexpr int kMemorySlotFeatures = 10;  // level: high, mid, low, throw, no hit; outcome: hit, counter, blocked, missed; age
constexpr int kMemoryTraces = 5;         // crouching, backing off, sidestepping, guarding, attacking
constexpr int kMemoryFighterFeatures = kMemorySlots * kMemorySlotFeatures + kMemoryTraces;
constexpr int kFullMoveMemoryFeatures = 2 * kMemoryFighterFeatures;  // own, then opponent
constexpr int kFullMoveMemoryObservationSize = kFullMoveObservationSize + kFullMoveMemoryFeatures;
constexpr const char* kFullMoveMemoryObservationContract = "full-move-memory-238-v3";
static_assert(kFullMoveMemoryObservationSize == 238, "update the memory contract name");

// Zero-initialized = empty; Memory{} resets it.
struct MemoryEntry {
    std::int16_t move_plus_one = 0;  // 0: empty slot
    std::int8_t level = 0;           // 1 high, 2 mid, 3 low, 4 throw, 5 no hit
    std::int8_t outcome = 0;         // 0 pending, 1 hit, 2 counter hit, 3 blocked, 4 missed or stopped
    std::int32_t age = 0;            // frames since it started
};

struct FighterMemory {
    MemoryEntry slots[kMemorySlots];  // newest first
    std::int32_t last_move_plus_one = 0;
    std::int32_t last_action_frame = 0;
    float traces[kMemoryTraces];
};

struct Memory {
    FighterMemory fighters[2];
};

T8_FULL_HD inline std::int8_t memory_level(const Tables& tables, int player, int move) {
    const Move& spec = move_of(tables, player, move);
    if (spec.hit_count <= 0) return 5;
    switch (hit_of(tables, player, spec, 0).level) {
        case FullHitLevel::High: return 1;
        case FullHitLevel::Mid:
        case FullHitLevel::SpecialMid: return 2;
        case FullHitLevel::Low:
        case FullHitLevel::SpecialLow: return 3;
        case FullHitLevel::Throw: return 4;
    }
    return 5;
}

// After every engine frame: events are that frame's (step resets them).
T8_FULL_HD inline void memory_update(const Tables& tables, const State& state, const Events& events,
                                     int perception_delay, Memory& memory) {
    for (int player = 0; player < 2; ++player) {
        FighterMemory& m = memory.fighters[player];
        const Fighter& fighter = state.fighters[player];
        for (int slot = 0; slot < kMemorySlots; ++slot) {
            if (m.slots[slot].move_plus_one != 0) ++m.slots[slot].age;
        }
        const bool ended = fighter.move < 0 && m.last_move_plus_one != 0;
        const bool started = fighter.move >= 0 && (fighter.move + 1 != m.last_move_plus_one ||
                                                   fighter.action_frame < m.last_action_frame);
        if ((ended || started) && m.slots[0].move_plus_one != 0 && m.slots[0].outcome == 0) {
            m.slots[0].outcome = 4;  // ended (or was cancelled) without contact: missed
        }
        if (started) {
            for (int slot = kMemorySlots - 1; slot > 0; --slot) m.slots[slot] = m.slots[slot - 1];
            m.slots[0] = MemoryEntry{static_cast<std::int16_t>(fighter.move + 1),
                                     memory_level(tables, player, fighter.move), 0, 0};
        }
        if (m.slots[0].move_plus_one != 0 && m.slots[0].outcome == 0) {
            for (int index = 0; index < events.contact_count; ++index) {
                const Contact& contact = events.contacts[index];
                if (contact.attacker != player) continue;
                switch (contact.contact) {
                    case FullContact::Hit:
                    case FullContact::ThrowHeld: m.slots[0].outcome = 1; break;
                    case FullContact::CounterHit: m.slots[0].outcome = 2; break;
                    case FullContact::Blocked: m.slots[0].outcome = 3; break;
                    default: m.slots[0].outcome = 4; break;  // crushed, parried, armored, evaded, broken, reversed
                }
                break;
            }
        }
        const bool idle = fighter.move < 0;
        const float signals[kMemoryTraces] = {
            fighter.posture == FullPosture::Crouching ? 1.0F : 0.0F,
            idle && (fighter.universal == FullUniversal::WalkBack || fighter.universal == FullUniversal::Backdash)
                ? 1.0F : 0.0F,
            idle && (fighter.universal == FullUniversal::SidestepLeft ||
                     fighter.universal == FullUniversal::SidestepRight) ? 1.0F : 0.0F,
            fighter.in_blockstun ? 1.0F : 0.0F,
            fighter.move >= 0 && fighter.action_frame >= perception_delay ? 1.0F : 0.0F,
        };
        for (int index = 0; index < kMemoryTraces; ++index) {
            m.traces[index] += (signals[index] - m.traces[index]) * (1.0F / 60.0F);
        }
        m.last_move_plus_one = fighter.move + 1;
        m.last_action_frame = fighter.action_frame;
    }
}

T8_FULL_HD inline void encode_fighter_memory(const FighterMemory& m, int hide_younger_than, float* out) {
    for (int index = 0; index < kMemoryFighterFeatures; ++index) out[index] = 0.0F;
    int written = 0;
    for (int slot = 0; slot < kMemorySlots && written < kMemorySlots; ++slot) {
        const MemoryEntry& entry = m.slots[slot];
        if (entry.move_plus_one == 0 || entry.age < hide_younger_than) continue;
        float* at = out + written * kMemorySlotFeatures;
        if (entry.level >= 1 && entry.level <= 5) at[entry.level - 1] = 1.0F;
        if (entry.outcome >= 1 && entry.outcome <= 4) at[4 + entry.outcome] = 1.0F;
        at[9] = clip01(entry.age / 180.0);
        ++written;
    }
    for (int index = 0; index < kMemoryTraces; ++index) {
        out[kMemorySlots * kMemorySlotFeatures + index] = m.traces[index];
    }
}

// kFullMoveMemoryFeatures floats for `player`: its own history, then the opponent's.
T8_FULL_HD inline void encode_memory(const Memory& memory, int player, int perception_delay, float* out) {
    encode_fighter_memory(memory.fighters[player], 0, out);
    encode_fighter_memory(memory.fighters[1 - player], perception_delay, out + kMemoryFighterFeatures);
}

struct EnvConfig {
    int max_round_frames = 3600;        // 60 s rounds; a timeout goes to the healthier fighter
    int hold_frames = 4;                // walking and guarding are held this long per decision
    int max_frames_per_decision = 120;  // returns to the policy at least this often
    float damage_reward = 1.0F;         // shaped: (damage dealt - damage taken) / max health
    float win_reward = 1.0F;            // terminal: +win_reward win, -win_reward loss, 0 draw
    // Anti-stalling (self-play otherwise settles into 0-0 timeouts and throw spam):
    float draw_penalty = 0.0F;          // a draw scores -draw_penalty * win_reward (a loss for both)
    float throw_break_penalty = 0.0F;   // each of the learner's throws that is broken
    // Dense time pressure (Oh et al. 2019, the Blade & Soul pro-level agents): a cost per
    // second of round time, so waiting costs something now instead of only at a round end
    // the discount barely reaches. A full 60 s round at 0.01/s costs 0.6 of a win.
    float time_penalty_per_second = 0.0F;
    // Fighting-style shaping (Oh et al. 2019): the HP ratio (a defensive style weighs its
    // own HP more, 6:4 is dealt 0.8 / taken 1.2) and a distance penalty per metre per
    // second (an aggressive style closes in).
    float damage_dealt_weight = 1.0F;
    float damage_taken_weight = 1.0F;
    float distance_penalty_per_second = 0.0F;
    // Frames before a policy recognizes the opponent's move (encode_observation). 0 sees
    // every move from its first frame, which no human or screen reader can.
    int perception_delay = 0;
    // --memory: the observation adds each fighter's recent moves and habits (Memory).
    bool memory = false;
    // --live-noise: screen-reading noise and controller latency, levels drawn per environment
    // and round from these ranges (NoiseLevels). Uncalibrated guesses until measured live.
    bool live_noise = false;
    float noise_position_max = 0.25F;  // metres of Gaussian error on positions and distance
    float noise_flip_max = 0.10F;      // detection error rate (posture, can-act, the opponent's move)
    int latency_min = 2;               // frames between the observed frame and the input landing
    int latency_max = 6;
    // --screen-observation: only the features the live screen reader provides (apply_screen_view).
    bool screen_view = false;
    double start_center_range = 1.2;    // starting midpoint drawn from [-range, range]
    double start_distance_min = 0.8;
    double start_distance_max = 2.4;
};

// ---------------------------------------------------------------- live noise (--live-noise)
//
// Live, the policy reads the game through screen capture and plays through a virtual
// controller: positions are estimated from pixels, detections (posture, an attack coming out)
// are sometimes wrong, and an input lands a few frames after the frame it was decided on.
// --live-noise trains under that: each environment draws its own levels at every reset
// (NoiseLevels: position noise, detection error rate, input latency), the observation is
// perturbed accordingly, and the levels are three more inputs so the policy learns how far to
// trust what it sees (live, they come from calibration). Noise uses only + - * / on hashed
// bits, so the CPU and GPU produce identical observations.
constexpr int kFullMoveNoiseFeatures = 3;  // position sigma, error rate, latency (absolute scales)

struct NoiseLevels {
    float position_sigma = 0.0F;  // metres
    float flip_rate = 0.0F;       // per detection per observation
    std::int32_t latency = 0;     // frames
    std::uint32_t key = 0;        // noise stream of this environment and round
};

T8_FULL_HD constexpr int full_move_observation_size(bool memory, bool noise) {
    return kFullMoveObservationSize + (memory ? kFullMoveMemoryFeatures : 0) + (noise ? kFullMoveNoiseFeatures : 0);
}
inline const char* full_move_observation_contract(bool memory, bool noise, bool screen = false) {
    if (screen) return noise ? "full-move-screen-noise-151-v4" : "full-move-screen-148-v4";  // no screen memory yet
    if (memory && noise) return "full-move-memory-noise-241-v3";
    if (noise) return "full-move-noise-151-v3";
    if (memory) return kFullMoveMemoryObservationContract;
    return kFullMoveObservationContract;
}

T8_FULL_HD inline NoiseLevels draw_noise_levels(const EnvConfig& config, std::uint64_t seed, std::uint64_t stream,
                                                std::uint64_t counter) {
    const std::uint64_t base = seed ^ 0x6e6f697365ULL;  // "noise"
    NoiseLevels levels{};
    levels.position_sigma = static_cast<float>(config.noise_position_max * random_unit(random_bits(base, stream, counter * 4U)));
    levels.flip_rate = static_cast<float>(config.noise_flip_max * random_unit(random_bits(base, stream, counter * 4U + 1U)));
    const int span = imax(0, config.latency_max - config.latency_min) + 1;
    levels.latency = config.latency_min + random_below(random_bits(base, stream, counter * 4U + 2U), span);
    levels.key = static_cast<std::uint32_t>(random_bits(base, stream, counter * 4U + 3U));
    return levels;
}

// Approximately standard normal from four uniforms (Irwin-Hall), exact on CPU and GPU.
T8_FULL_HD inline double noise_gaussian(std::uint64_t key, std::uint64_t stream, std::uint64_t counter) {
    double sum = 0.0;
    for (std::uint64_t k = 0; k < 4U; ++k) sum += random_unit(random_bits(key, stream, counter * 4U + k));
    return (sum - 2.0) * 1.7320508075688772;
}

// Perturbs the kFullMoveObservationSize features of `player`'s observation (fresh draws per
// frame): HUD health and positions get Gaussian noise; postures, the can-act flags, stun, and the
// opponent's move are misdetected at the error rate (a missed or misread attack, or a false one).
T8_FULL_HD inline void apply_observation_noise(const NoiseLevels& levels, const State& state, int player, float* out) {
    const std::uint64_t key = levels.key;
    const std::uint64_t frame = static_cast<std::uint64_t>(state.frame);
    const std::uint64_t side = static_cast<std::uint64_t>(player) * 4096U;
    const auto gauss = [&](int index) { return noise_gaussian(key, side + static_cast<std::uint64_t>(index), frame); };
    const auto uniform = [&](int index) {
        return random_unit(random_bits(key, side + 2048U + static_cast<std::uint64_t>(index), frame));
    };
    const double sigma = levels.position_sigma;
    const double flip = levels.flip_rate;
    for (int index = 0; index < 4; ++index) out[index] = clip01(out[index] + 0.005 * gauss(index));  // health bars
    out[4] = clip01(out[4] + sigma / 4.0 * gauss(4));   // distance (normalized by 4 m)
    out[5] = clip11(out[5] + sigma / 2.0 * gauss(5));   // sidestep offset (by 2 m)
    out[6] = clip01(out[6] + sigma / 7.2 * gauss(6));   // walls (by the stage width)
    out[7] = clip01(out[7] + sigma / 7.2 * gauss(7));
    for (int fighter = 0; fighter < 2; ++fighter) {  // posture one-hots: own 8-13, opponent 14-19
        if (uniform(8 + fighter) < flip) {
            const int wrong = random_below(random_bits(key, side + 3000U + static_cast<std::uint64_t>(fighter), frame), 6);
            for (int index = 0; index < 6; ++index) out[8 + 6 * fighter + index] = index == wrong ? 1.0F : 0.0F;
        }
    }
    for (int index = 20; index < 24; ++index) {  // stun and blockstun: missed
        if (uniform(index) < flip) out[index] = 0.0F;
    }
    for (int index = 24; index < 26; ++index) {  // can act: misread
        if (uniform(index) < flip) out[index] = 1.0F - out[index];
    }
    // The opponent's move (50 attacking, 51-54 next hit level, 55 frames to it, 56 block
    // advantage, 57 recovery left).
    const double roll = uniform(50);
    if (out[50] > 0.5F) {
        if (roll < flip) {
            for (int index = 50; index < 58; ++index) out[index] = 0.0F;  // missed
        } else if (roll < 2.0 * flip) {
            const int wrong = random_below(random_bits(key, side + 3002U, frame), 4);  // misread level
            for (int index = 0; index < 4; ++index) out[51 + index] = index == wrong ? 1.0F : 0.0F;
        }
        out[55] = clip01(out[55] + 20.0 * flip / 30.0 * gauss(55));  // timing: about 2 frames at a 10% error rate
    } else if (roll < 0.25 * flip) {  // a false alarm
        out[50] = 1.0F;
        const int level = random_below(random_bits(key, side + 3003U, frame), 4);
        for (int index = 0; index < 4; ++index) out[51 + index] = index == level ? 1.0F : 0.0F;
        out[55] = static_cast<float>(uniform(55));
        out[56] = 0.0F;
        out[57] = static_cast<float>(uniform(57));
    }
}

// ---------------------------------------------------------------- screen view (--screen-observation)
//
// The live runtime reads the game from the screen (src/t8_agent/live/own_state.py,
// screen_full_move_observation):
//   HUD          both health bars and recoverable health (0-3; src/t8_agent/live/hud_reader.py),
//                the opponent's Rage by its health (32). The Heat gauges are not read yet: in
//                Practice they misread during hit flashes and Heat activation (measured).
//   pose model   the distance (4), both fighters' postures: standing, crouching, airborne,
//                knocked down (own 8-11, opponent 14-17; src/t8_agent/io/pose_tracker.py)
//   attack cue   the opponent attacking (50)
//   own commands can act (24), Rage (29; by its health), stance (33, 34), the move it is in and
//                how much of it is left (58, 59)
// A screen-view policy trains on exactly those features of the usual layout; every other state
// feature is zero, as it is live. The pose model cannot tell a wakeup from lying down or a wall
// splat from standing, so apply_screen_view folds those postures the same way. Layout and size
// are unchanged; the contract name (v4 since the pose features) says which features are real.
// Keep this list in sync with screen_full_move_observation (tests/test_screen_full_move.py
// reads it from here).
constexpr int kScreenViewFeatures[] = {0, 1, 2, 3, 4, 8, 9, 10, 11, 14, 15, 16, 17, 24, 29, 32, 33, 34, 50, 58, 59, 63};
constexpr int kScreenViewFeatureCount = static_cast<int>(sizeof(kScreenViewFeatures) / sizeof(int));

// The same list as a function, for device code (host arrays are not visible there).
T8_FULL_HD constexpr bool screen_view_feature(int index) {
    return (index >= 0 && index <= 4) || (index >= 8 && index <= 11) || (index >= 14 && index <= 17) ||
        index == 24 || index == 29 || index == 32 || index == 33 || index == 34 || index == 50 || index == 58 ||
        index == 59 || index == 63;
}

// Zeroes every state feature the screen cannot provide (the roster one-hots stay). Applied
// after any noise, so a misdetection only shows in features the screen really has.
T8_FULL_HD inline void apply_screen_view(float* out) {
    // Postures as the pose model sees them: a wakeup lies on the floor (grounded), a wall splat
    // stands against the wall (standing). Posture one-hots: own 8-13, opponent 14-19, in the order
    // standing, crouching, airborne, grounded, wakeup, wall splat.
    const int bases[2] = {8, 14};
    for (int side = 0; side < 2; ++side) {
        const int base = bases[side];
        if (out[base + 4] > 0.5F) out[base + 3] = 1.0F;
        if (out[base + 5] > 0.5F) out[base + 0] = 1.0F;
    }
    for (int index = 0; index < kFullMoveStateFeatures; ++index) {
        if (!screen_view_feature(index)) out[index] = 0.0F;
    }
}

T8_FULL_HD inline void encode_noise_levels(const NoiseLevels& levels, float* out) {
    out[0] = clip01(levels.position_sigma / 0.5);
    out[1] = clip01(levels.flip_rate / 0.3);
    out[2] = clip01(levels.latency / 10.0);
}

struct EnvStepResult {
    float reward = 0.0F;         // shaped
    float sparse_reward = 0.0F;  // terminal only
    bool terminated = false;
    std::int8_t outcome = 0;     // learner-relative when terminated: 1 win, -1 loss, 0 draw
    int frames = 0;
};

// Scenario curricula: situations a round can start in. They change only the
// starting state; the reward is the same damage and win signal, so no
// scenario adds shaping a policy could exploit.
enum Scenario : int {
    kScenarioNeutral = 0,
    kScenarioWall = 1,        // the learner or the opponent starts near a wall
    kScenarioPunish = 2,      // the opponent is recovering from a move that is unsafe on block
    kScenarioThrowBreak = 3,  // the opponent's throw has connected; its break window is open
    kScenarioWakeup = 4,      // the learner is knocked down
    kScenarioJuggle = 5,      // the opponent has just been launched
    kScenarioHeat = 6,        // the learner is in Heat
    kScenarioRage = 7,        // the learner is low on health and in Rage
    kScenarioRange = 8,       // long range: movement and approach
    kScenarioCount = 9,
};

T8_FULL_HD inline const char* scenario_name(int scenario) {
    switch (scenario) {
        case kScenarioWall: return "wall";
        case kScenarioPunish: return "punish";
        case kScenarioThrowBreak: return "throw_break";
        case kScenarioWakeup: return "wakeup";
        case kScenarioJuggle: return "juggle";
        case kScenarioHeat: return "heat";
        case kScenarioRage: return "rage";
        case kScenarioRange: return "range";
        default: return "neutral";
    }
}

// The opponent's candidate move satisfying `want` (tried at random), or -1.
template <typename Predicate>
T8_FULL_HD inline int find_opponent_move(const Tables& tables, int opponent, const CandidateMap& map,
                                         std::uint64_t seed, std::uint64_t stream, std::uint64_t counter,
                                         Predicate want) {
    if (map.count <= kFullMoveUniversalCount) return -1;
    for (int attempt = 0; attempt < 48; ++attempt) {
        const int candidate = kFullMoveUniversalCount +
            random_below(random_bits(seed, stream, counter * 4U + 8U + static_cast<std::uint64_t>(attempt)),
                         map.count - kFullMoveUniversalCount);
        const int move = map.moves[candidate];
        if (move >= 0 && want(move_of(tables, opponent, move))) return move;
    }
    return -1;
}

T8_FULL_HD inline void env_reset(const Tables& tables, State& state, const EnvConfig& config, std::uint64_t seed,
                                 std::uint64_t stream, std::uint64_t counter, int scenario = kScenarioNeutral,
                                 int learner = 0, const CandidateMap* opponent_map = nullptr) {
    double center = (2.0 * random_unit(random_bits(seed, stream, counter * 4U)) - 1.0) * config.start_center_range;
    double distance = config.start_distance_min +
        random_unit(random_bits(seed, stream, counter * 4U + 1U)) * (config.start_distance_max - config.start_distance_min);
    const bool flip = (random_bits(seed, stream, counter * 4U + 2U) & 1U) != 0U;
    const double half = tables.config.half_width;
    if (scenario == kScenarioWall) center = (flip ? 1.0 : -1.0) * (half - 0.9);
    if (scenario == kScenarioRange) distance = 3.0 + random_unit(random_bits(seed, stream, counter * 4U + 3U));
    if (scenario == kScenarioPunish || scenario == kScenarioThrowBreak || scenario == kScenarioJuggle ||
        scenario == kScenarioWakeup) {
        distance = scenario == kScenarioThrowBreak ? 0.6 : 0.9;
    }
    reset(tables, state, center - distance / 2.0, center + distance / 2.0);
    const int opponent = 1 - learner;
    Fighter& own = state.fighters[learner];
    Fighter& other = state.fighters[opponent];
    switch (scenario) {
        case kScenarioPunish: {
            if (opponent_map == nullptr) break;
            const int move = find_opponent_move(tables, opponent, *opponent_map, seed, stream, counter,
                                                [](const Move& spec) {
                                                    return spec.hit_count > 0 && spec.block_advantage <= -10 &&
                                                        spec.required_stance < 0 && !spec.has(kReactive);
                                                });
            if (move < 0) break;
            const Move& spec = move_of(tables, opponent, move);
            // Blocked and recovering: |block advantage| frames until the opponent can act.
            other.move = move;
            other.action_frame = imax(0, spec.total_frames + spec.block_advantage);
            other.hits_landed = spec.hit_count >= 32 ? ~0U : (1U << static_cast<unsigned>(spec.hit_count)) - 1U;
            other.move_blocked = true;
            break;
        }
        case kScenarioThrowBreak: {
            if (opponent_map == nullptr) break;
            const int move = find_opponent_move(tables, opponent, *opponent_map, seed, stream, counter,
                                                [&](const Move& spec) {
                                                    if (spec.hit_count == 0 || spec.required_stance >= 0) return false;
                                                    const Hit& first = tables.side[opponent].hits[spec.hit_offset];
                                                    return first.level == FullHitLevel::Throw &&
                                                        first.throw_break != FullThrowBreak::None &&
                                                        spec.required_posture == FullPosture::Standing &&
                                                        !spec.has(kReactive);
                                                });
            if (move < 0) break;
            const Move& spec = move_of(tables, opponent, move);
            other.move = move;
            other.action_frame = tables.side[opponent].hits[spec.hit_offset].first_active_frame;
            other.hits_landed = 1U;
            other.pending_throw_hit = 0;
            own.throw_break_frames = tables.config.throw_break_window;
            own.throw_break_attempted = false;
            break;
        }
        case kScenarioWakeup:
            own.posture = FullPosture::Grounded;
            own.grounded_elapsed = 0;
            own.techable = false;
            break;
        case kScenarioJuggle:
            other.posture = FullPosture::Airborne;
            other.posture_frames = 45;
            other.combo_hits = 1;
            break;
        case kScenarioHeat:
            own.heat = true;
            own.heat_available = false;
            own.heat_frames = tables.config.heat_duration_frames;
            break;
        case kScenarioRage:
            own.health = tables.config.max_health * tables.config.rage_health_fraction * 0.8;
            own.rage = true;
            own.rage_available = false;
            break;
        default:
            break;
    }
}

// True when a universal input lasts one frame and so is held for hold_frames.
T8_FULL_HD inline bool held_input(const Input& input) {
    return input.move < 0 && (input.universal == FullUniversal::Idle || input.universal == FullUniversal::WalkForward ||
                              input.universal == FullUniversal::WalkBack || input.universal == FullUniversal::Crouch);
}

// One policy decision: applies the learner's candidate (and the opponent's,
// when a policy plays it; otherwise the scripted opponent acts on every frame
// it can), advances frames until the learner (or a policy opponent) has its
// next decision, and scores the step.
T8_FULL_HD inline EnvStepResult env_step(const Tables& tables, State& state, int learner, const CandidateMap& learner_map,
                                         const CandidateMap& opponent_map, int learner_candidate, int opponent_candidate,
                                         double aggression, const EnvConfig& config, std::uint64_t seed,
                                         std::uint64_t stream, std::uint64_t counter, Memory* memory = nullptr,
                                         const NoiseLevels* noise = nullptr) {
    EnvStepResult result{};
    // Input latency: policy inputs land `lag` frames after the observation they were chosen on
    // (one frame more a quarter of the time: the live loop's jitter). Until then nothing new is pressed.
    int lag = 0;
    if (noise != nullptr && noise->latency > 0) {
        lag = noise->latency + (random_unit(random_bits(noise->key, 0x1a7U, counter)) < 0.25 ? 1 : 0);
    }
    const int opponent = 1 - learner;
    const double max_health = tables.config.max_health;
    const double own_before = state.fighters[learner].health;
    const double opponent_before = state.fighters[opponent].health;
    Input learner_input{};
    if (!candidate_legal(tables, state, learner, learner_map, learner_candidate) ||
        !candidate_input(learner_map, learner_candidate, learner_input)) {
        learner_input = Input{};
    }
    const bool policy_opponent = opponent_candidate >= 0;
    Input opponent_input{};
    if (policy_opponent && (!candidate_legal(tables, state, opponent, opponent_map, opponent_candidate) ||
                            !candidate_input(opponent_map, opponent_candidate, opponent_input))) {
        opponent_input = Input{};
    }
    const bool learner_held = held_input(learner_input);
    const bool opponent_held = policy_opponent && held_input(opponent_input);
    Events events{};
    int broken_throws = 0;
    double distance_frames = 0.0;  // metres x frames, for the distance penalty
    for (int frame = 0; frame < config.max_frames_per_decision; ++frame) {
        const bool landing = frame == lag;
        const bool holding = frame > lag && frame < lag + config.hold_frames;
        const Input own = landing || (learner_held && holding) ? learner_input : Input{};
        Input other{};
        if (policy_opponent) {
            other = landing || (opponent_held && holding) ? opponent_input : Input{};
        } else if (has_decision(tables, state, opponent)) {
            // A negative behaviour value names a tiered scripted bot; [0, 1] is the legacy random opponent.
            const int bot = decode_bot(aggression);
            const std::uint64_t at = counter * 256U + static_cast<std::uint64_t>(frame);
            const int choice = bot >= 0 ? scripted_bot_candidate(tables, state, opponent, opponent_map, bot, seed, stream, at)
                                        : scripted_candidate(tables, state, opponent, opponent_map, aggression, seed, stream, at);
            if (!candidate_input(opponent_map, choice, other)) other = Input{};
        }
        step(tables, state, learner == 0 ? own : other, learner == 0 ? other : own, events);
        if (memory != nullptr) memory_update(tables, state, events, config.perception_delay, *memory);
        ++result.frames;
        distance_frames += dabs(state.fighters[0].x - state.fighters[1].x);
        for (int index = 0; index < events.contact_count; ++index) {
            if (events.contacts[index].contact == FullContact::ThrowBroken && events.contacts[index].attacker == learner) {
                ++broken_throws;
            }
        }
        if (state.winner < 0 && state.frame >= config.max_round_frames) {
            const double own_health = state.fighters[learner].health;
            const double opponent_health = state.fighters[opponent].health;
            state.winner = own_health == opponent_health ? 2 : (own_health > opponent_health ? learner : opponent);
        }
        if (state.winner >= 0) break;
        const int minimum = ((learner_held || opponent_held) ? config.hold_frames : 1) + lag;
        if (frame + 1 >= minimum && (has_decision(tables, state, learner) ||
                                     (policy_opponent && has_decision(tables, state, opponent)))) {
            break;
        }
    }
    const double dealt = opponent_before - state.fighters[opponent].health;
    const double taken = own_before - state.fighters[learner].health;
    result.reward = static_cast<float>((dealt * config.damage_dealt_weight - taken * config.damage_taken_weight) /
                                       max_health) * config.damage_reward -
                    static_cast<float>(distance_frames) * (config.distance_penalty_per_second / 60.0F) -
                    static_cast<float>(broken_throws) * config.throw_break_penalty -
                    static_cast<float>(result.frames) * (config.time_penalty_per_second / 60.0F);
    if (state.winner >= 0) {
        result.terminated = true;
        result.outcome = state.winner == learner ? 1 : (state.winner == opponent ? -1 : 0);
        result.sparse_reward = result.outcome == 0 ? -config.draw_penalty * config.win_reward
                                                   : static_cast<float>(result.outcome) * config.win_reward;
        result.reward += result.sparse_reward;
    }
    return result;
}

}  // namespace t8::v2::full_core
