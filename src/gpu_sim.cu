#include "t8_v2/gpu_sim.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace t8::v2 {
namespace {

constexpr int kThreads = 256;
constexpr int kFloatStateFields = 6;
constexpr int kIntStateFields = 27;
constexpr int kFighterIntFields = 11;

enum FloatField : int {
    P1Health,
    P1X,
    P1Y,
    P2Health,
    P2X,
    P2Y,
};

enum FighterIntField : int {
    Guard,
    Move,
    MoveFrame,
    HasHit,
    Hitstun,
    Blockstun,
    Airborne,
    ThrowBreakActive,
    LaunchesTaken,
    Whiffs,
    CharacterId,
};

enum StateIntField : int {
    Frame = 22,
    StallFrames,
    NoActionFrames,
    RoundOver,
    Winner,
};

enum ActionValue : int {
    Neutral = 0,
    WalkForward = 1,
    WalkBack = 2,
    DashForward = 3,
    DashBack = 4,
    Crouch = 5,
    Stand = 6,
    Jump = 7,
    SidestepLeft = 8,
    SidestepRight = 9,
    SidewalkLeft = 10,
    SidewalkRight = 11,
    BlockHigh = 12,
    BlockLow = 13,
    LowParry = 14,
    ThrowBreak1 = 15,
    ThrowBreak2 = 16,
    ThrowBreak12 = 17,
    Jab = 18,
    Df1 = 19,
    F2 = 20,
    Db3 = 21,
    Hopkick = 22,
    Throw = 23,
};

enum HitLevelValue : int {
    HitNone = 0,
    HitHigh = 1,
    HitMid = 2,
    HitLow = 3,
    HitThrow = 4,
};

struct DeviceConfig {
    float max_health;
    float stage_half_width;
    int decision_frames;
    int max_frames;
    int timeout_ties_are_draws;
    int randomize_initial_positions;
    float initial_center_jitter;
    float initial_distance_min;
    float initial_distance_max;
    float walk_speed;
    float dash_speed;
    int jump_frames;
    int throw_break_frames;
    float sidestep_speed;
    float sidewalk_speed;
    float lateral_return_speed;
    float sidestep_evasion_width;
    float body_radius;
    float stall_distance;
    int max_stall_frames;
    int no_action_timeout_frames;
    float no_action_timeout_penalty;
    float round_win_base_reward;
    float round_win_health_reward;
    float round_loss_penalty;
    float timeout_ahead_penalty;
    float timeout_even_penalty;
    float timeout_behind_penalty;
    float stalemate_penalty;
    float damage_dealt_scale;
    float damage_taken_scale;
    float throw_reward_scale;
    float successful_throw_reward;
    float throw_break_reward;
    float throw_broken_penalty;
    float whiff_punish_reward_scale;
    float block_reward;
    float blocked_attack_penalty;
    float idle_penalty;
    float far_spacing_penalty;
    float lateral_passivity_penalty;
    float wall_camping_penalty;
    float late_round_passivity_penalty;
    float whiff_penalty;
};

struct DeviceMove {
    int hit_level;
    int startup;
    int active;
    int recovery;
    float damage;
    float range;
    int hitstun;
    int blockstun;
    float pushback;
    int whiff_recovery;
    int launches;
};

__device__ __constant__ DeviceMove c_moves[6] = {
    {HitHigh, 10, 2, 13, 7.0F, 0.82F, 14, 7, 0.08F, 0, 0},
    {HitMid, 13, 3, 18, 12.0F, 0.95F, 18, 11, 0.11F, 0, 0},
    {HitMid, 16, 3, 22, 18.0F, 1.25F, 24, 14, 0.16F, 0, 0},
    {HitLow, 18, 3, 24, 10.0F, 0.90F, 17, 15, 0.06F, 0, 0},
    {HitMid, 15, 4, 30, 21.0F, 0.88F, 34, 18, 0.18F, 0, 1},
    {HitThrow, 12, 2, 24, 25.0F, 0.48F, 30, 0, 0.22F, 0, 0},
};

__device__ __constant__ DeviceMove
    c_character_moves[kRosterCharacterCount * kCharacterMoveSlotCount];
__device__ __constant__ int c_character_move_catalog_enabled = 0;

struct FighterD {
    float health;
    float x;
    float y;
    int guard;
    int move;
    int move_frame;
    int has_hit;
    int hitstun;
    int blockstun;
    int airborne;
    int throw_break_active;
    int launches_taken;
    int whiffs;
    int character_id;
};

struct StateD {
    FighterD p1;
    FighterD p2;
    int frame;
    int stall_frames;
    int no_action_frames;
    int round_over;
    int winner;
};

struct AttackCheckD {
    int in_range = 0;
    int blocked = 0;
    float damage = 0.0F;
    int throw_broken = 0;
};

struct FrameInfoD {
    float damage_to_p1 = 0.0F;
    float damage_to_p2 = 0.0F;
    float p1_throw_damage = 0.0F;
    float p2_throw_damage = 0.0F;
    int p1_throw_break = 0;
    int p2_throw_break = 0;
    float p1_whiff_punish_bonus = 0.0F;
    float p2_whiff_punish_bonus = 0.0F;
    int p1_whiff = 0;
    int p2_whiff = 0;
    int p1_block = 0;
    int p2_block = 0;
};

struct StepInfoD {
    float damage_to_p1 = 0.0F;
    float damage_to_p2 = 0.0F;
    float p1_throw_damage = 0.0F;
    float p2_throw_damage = 0.0F;
    int p1_throw_breaks = 0;
    int p2_throw_breaks = 0;
    float p1_whiff_punish_bonus = 0.0F;
    float p2_whiff_punish_bonus = 0.0F;
    int p1_whiffs = 0;
    int p2_whiffs = 0;
    int p1_blocks = 0;
    int p2_blocks = 0;
};

inline void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

DeviceConfig to_device_config(const Config& c) {
    return {
        static_cast<float>(c.max_health), static_cast<float>(c.stage_half_width),
        c.decision_frames, c.max_frames, static_cast<int>(c.timeout_ties_are_draws),
        static_cast<int>(c.randomize_initial_positions),
        static_cast<float>(c.initial_center_jitter),
        static_cast<float>(c.initial_distance_min),
        static_cast<float>(c.initial_distance_max),
        static_cast<float>(c.walk_speed),
        static_cast<float>(c.dash_speed), c.jump_frames, c.throw_break_frames,
        static_cast<float>(c.sidestep_speed), static_cast<float>(c.sidewalk_speed),
        static_cast<float>(c.lateral_return_speed), static_cast<float>(c.sidestep_evasion_width),
        static_cast<float>(c.body_radius), static_cast<float>(c.stall_distance),
        c.max_stall_frames, c.no_action_timeout_frames,
        static_cast<float>(c.no_action_timeout_penalty), static_cast<float>(c.round_win_base_reward),
        static_cast<float>(c.round_win_health_reward), static_cast<float>(c.round_loss_penalty),
        static_cast<float>(c.timeout_ahead_penalty), static_cast<float>(c.timeout_even_penalty),
        static_cast<float>(c.timeout_behind_penalty), static_cast<float>(c.stalemate_penalty),
        static_cast<float>(c.damage_dealt_scale), static_cast<float>(c.damage_taken_scale),
        static_cast<float>(c.throw_reward_scale), static_cast<float>(c.successful_throw_reward),
        static_cast<float>(c.throw_break_reward), static_cast<float>(c.throw_broken_penalty),
        static_cast<float>(c.whiff_punish_reward_scale), static_cast<float>(c.block_reward),
        static_cast<float>(c.blocked_attack_penalty), static_cast<float>(c.idle_penalty),
        static_cast<float>(c.far_spacing_penalty), static_cast<float>(c.lateral_passivity_penalty),
        static_cast<float>(c.wall_camping_penalty), static_cast<float>(c.late_round_passivity_penalty),
        static_cast<float>(c.whiff_penalty),
    };
}

__device__ __forceinline__ float clampf(float value, float low, float high) {
    return fminf(high, fmaxf(low, value));
}

__device__ __forceinline__ int maxi(int a, int b) { return a > b ? a : b; }
__device__ __forceinline__ int fighter_base(int player) {
    return player == 1 ? 0 : kFighterIntFields;
}

__device__ FighterD load_fighter(
    const float* state_f,
    const int* state_i,
    std::size_t n,
    std::size_t lane,
    int player) {
    const int float_base = player == 1 ? P1Health : P2Health;
    const int int_base = fighter_base(player);
    return {
        state_f[(float_base + 0) * n + lane],
        state_f[(float_base + 1) * n + lane],
        state_f[(float_base + 2) * n + lane],
        state_i[(int_base + Guard) * n + lane],
        state_i[(int_base + Move) * n + lane],
        state_i[(int_base + MoveFrame) * n + lane],
        state_i[(int_base + HasHit) * n + lane],
        state_i[(int_base + Hitstun) * n + lane],
        state_i[(int_base + Blockstun) * n + lane],
        state_i[(int_base + Airborne) * n + lane],
        state_i[(int_base + ThrowBreakActive) * n + lane],
        state_i[(int_base + LaunchesTaken) * n + lane],
        state_i[(int_base + Whiffs) * n + lane],
        state_i[(int_base + CharacterId) * n + lane],
    };
}

__device__ void store_fighter(
    float* state_f,
    int* state_i,
    std::size_t n,
    std::size_t lane,
    int player,
    const FighterD& f) {
    const int float_base = player == 1 ? P1Health : P2Health;
    const int int_base = fighter_base(player);
    state_f[(float_base + 0) * n + lane] = f.health;
    state_f[(float_base + 1) * n + lane] = f.x;
    state_f[(float_base + 2) * n + lane] = f.y;
    state_i[(int_base + Guard) * n + lane] = f.guard;
    state_i[(int_base + Move) * n + lane] = f.move;
    state_i[(int_base + MoveFrame) * n + lane] = f.move_frame;
    state_i[(int_base + HasHit) * n + lane] = f.has_hit;
    state_i[(int_base + Hitstun) * n + lane] = f.hitstun;
    state_i[(int_base + Blockstun) * n + lane] = f.blockstun;
    state_i[(int_base + Airborne) * n + lane] = f.airborne;
    state_i[(int_base + ThrowBreakActive) * n + lane] = f.throw_break_active;
    state_i[(int_base + LaunchesTaken) * n + lane] = f.launches_taken;
    state_i[(int_base + Whiffs) * n + lane] = f.whiffs;
    state_i[(int_base + CharacterId) * n + lane] = f.character_id;
}

__device__ StateD load_state(
    const float* state_f,
    const int* state_i,
    std::size_t n,
    std::size_t lane) {
    return {
        load_fighter(state_f, state_i, n, lane, 1),
        load_fighter(state_f, state_i, n, lane, 2),
        state_i[Frame * n + lane],
        state_i[StallFrames * n + lane],
        state_i[NoActionFrames * n + lane],
        state_i[RoundOver * n + lane],
        state_i[Winner * n + lane],
    };
}

__device__ void store_state(
    float* state_f,
    int* state_i,
    std::size_t n,
    std::size_t lane,
    const StateD& s) {
    store_fighter(state_f, state_i, n, lane, 1, s.p1);
    store_fighter(state_f, state_i, n, lane, 2, s.p2);
    state_i[Frame * n + lane] = s.frame;
    state_i[StallFrames * n + lane] = s.stall_frames;
    state_i[NoActionFrames * n + lane] = s.no_action_frames;
    state_i[RoundOver * n + lane] = s.round_over;
    state_i[Winner * n + lane] = s.winner;
}

__device__ __forceinline__ bool busy(const FighterD& f) {
    return f.move >= 0 || f.hitstun > 0 || f.blockstun > 0 || f.airborne > 0;
}

__device__ __forceinline__ int move_index_for_action(int action) {
    return action >= Jab && action <= Throw ? action - Jab : -1;
}

__device__ __forceinline__ bool is_lateral(int action) {
    return action >= SidestepLeft && action <= SidewalkRight;
}

__device__ __forceinline__ bool is_forward(int action) {
    return action == WalkForward || action == DashForward;
}

__device__ __forceinline__ bool is_late_passive(int action) {
    return action == Neutral || action == WalkBack || action == DashBack;
}

__device__ __forceinline__ bool is_no_action_passive(int action) {
    return action == Neutral || action == Stand || action == Crouch;
}

__device__ __forceinline__ bool is_throw_break(int action) {
    return action >= ThrowBreak1 && action <= ThrowBreak12;
}

__device__ __forceinline__ int move_total_frames(const DeviceMove& move) {
    return move.startup + move.active + move.recovery + move.whiff_recovery;
}

__device__ __forceinline__ const DeviceMove& move_for(const FighterD& fighter) {
    if (c_character_move_catalog_enabled != 0 && fighter.character_id >= 0 &&
        fighter.character_id < static_cast<int>(kRosterCharacterCount)) {
        return c_character_moves[
            fighter.character_id * static_cast<int>(kCharacterMoveSlotCount) + fighter.move];
    }
    return c_moves[fighter.move];
}

__device__ FighterD initial_fighter(float health, float x) {
    return {health, x, 0.0F, HitNone, -1, 0, 0, 0, 0, 0, 0, 0, 0,
            static_cast<int>(kJunCharacterId)};
}

__device__ float reset_random_unit(unsigned long long value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
    value ^= value >> 31;
    return static_cast<float>((value >> 40) & 0xFFFFFFULL) / 16777216.0F;
}

__device__ StateD initial_state(
    const DeviceConfig& config,
    unsigned long long seed = 0,
    std::size_t lane = 0) {
    float p1_x = -0.85F;
    float p2_x = 0.85F;
    if (config.randomize_initial_positions) {
        const unsigned long long lane_seed = seed ^
            (static_cast<unsigned long long>(lane) * 0x9e3779b97f4a7c15ULL);
        const float distance = config.initial_distance_min +
            (config.initial_distance_max - config.initial_distance_min) * reset_random_unit(lane_seed);
        const float allowed_center = fmaxf(0.0F, fminf(
            config.initial_center_jitter,
            config.stage_half_width - config.body_radius - distance * 0.5F));
        const float center = (reset_random_unit(lane_seed ^ 0xd1b54a32d192ed03ULL) * 2.0F - 1.0F) *
            allowed_center;
        p1_x = center - distance * 0.5F;
        p2_x = center + distance * 0.5F;
    }
    return {initial_fighter(config.max_health, p1_x),
            initial_fighter(config.max_health, p2_x), 0, 0, 0, 0, 0};
}

__device__ FighterD start_action(FighterD fighter, int action, const DeviceConfig& config) {
    if (busy(fighter)) return fighter;
    if (action == BlockHigh || action == Stand || action == Neutral) {
        fighter.guard = HitMid;
        fighter.throw_break_active = 0;
        return fighter;
    }
    if (action == BlockLow || action == Crouch || action == LowParry) {
        fighter.guard = HitLow;
        fighter.throw_break_active = 0;
        return fighter;
    }
    if (action == Jump) {
        fighter.guard = HitNone;
        fighter.airborne = config.jump_frames;
        fighter.throw_break_active = 0;
        return fighter;
    }
    if (is_throw_break(action)) {
        fighter.guard = HitNone;
        fighter.throw_break_active = config.throw_break_frames;
        return fighter;
    }
    const int move = move_index_for_action(action);
    if (move >= 0) {
        fighter.guard = HitNone;
        fighter.move = move;
        fighter.move_frame = 0;
        fighter.has_hit = 0;
        fighter.throw_break_active = 0;
        return fighter;
    }
    fighter.guard = HitNone;
    fighter.throw_break_active = 0;
    return fighter;
}

__device__ FighterD tick_timers(FighterD fighter, const DeviceConfig& config) {
    fighter.hitstun = maxi(0, fighter.hitstun - 1);
    fighter.blockstun = maxi(0, fighter.blockstun - 1);
    fighter.airborne = maxi(0, fighter.airborne - 1);
    fighter.throw_break_active = maxi(0, fighter.throw_break_active - 1);
    if (fighter.move >= 0) ++fighter.move_frame;
    if (fabsf(fighter.y) <= config.lateral_return_speed) {
        fighter.y = 0.0F;
    } else if (fighter.y > 0.0F) {
        fighter.y -= config.lateral_return_speed;
    } else {
        fighter.y += config.lateral_return_speed;
    }
    return fighter;
}

__device__ FighterD move_fighter(FighterD fighter, int action, int forward, const DeviceConfig& config) {
    if (action == WalkForward) fighter.x += config.walk_speed * forward;
    else if (action == WalkBack) fighter.x -= config.walk_speed * forward;
    else if (action == DashForward) fighter.x += config.dash_speed * forward;
    else if (action == DashBack) fighter.x -= config.dash_speed * forward;

    if (action == SidestepLeft) fighter.y += config.sidestep_speed;
    else if (action == SidestepRight) fighter.y -= config.sidestep_speed;
    else if (action == SidewalkLeft) fighter.y += config.sidewalk_speed;
    else if (action == SidewalkRight) fighter.y -= config.sidewalk_speed;
    fighter.y = clampf(fighter.y, -1.0F, 1.0F);
    return fighter;
}

__device__ void separate_and_clip(FighterD& p1, FighterD& p2, const DeviceConfig& config) {
    const float min_gap = config.body_radius * 2.0F;
    float p1_x = clampf(p1.x, -config.stage_half_width, config.stage_half_width);
    float p2_x = clampf(p2.x, -config.stage_half_width, config.stage_half_width);
    if (p2_x - p1_x < min_gap) {
        const float center = (p1_x + p2_x) * 0.5F;
        p1_x = center - min_gap * 0.5F;
        p2_x = center + min_gap * 0.5F;
    }
    p1.x = clampf(p1_x, -config.stage_half_width, config.stage_half_width);
    p2.x = clampf(p2_x, -config.stage_half_width, config.stage_half_width);
}

__device__ __forceinline__ bool is_blocked(int guard, int attack) {
    if (attack == HitThrow) return false;
    if (guard == HitLow) return attack == HitLow;
    if (guard == HitMid) return attack == HitHigh || attack == HitMid;
    return false;
}

__device__ AttackCheckD check_attack(
    const FighterD& attacker,
    const FighterD& defender,
    const DeviceMove& move,
    const DeviceConfig& config) {
    AttackCheckD result{};
    if (fabsf(defender.x - attacker.x) > move.range) return result;
    if (move.hit_level != HitThrow && fabsf(defender.y - attacker.y) > config.sidestep_evasion_width) {
        return result;
    }
    if (defender.airborne > 0 && (move.hit_level == HitLow || move.hit_level == HitThrow)) {
        return result;
    }
    result.in_range = 1;
    if (move.hit_level == HitThrow && defender.throw_break_active > 0) {
        result.blocked = 1;
        result.throw_broken = 1;
        return result;
    }
    result.blocked = is_blocked(defender.guard, move.hit_level) ? 1 : 0;
    result.damage = result.blocked ? 0.0F : move.damage;
    return result;
}

__device__ void apply_attack(
    FighterD& attacker,
    FighterD& defender,
    const DeviceMove& move,
    const AttackCheckD& check,
    int direction,
    const DeviceConfig& config) {
    if (!check.in_range) return;
    attacker.has_hit = 1;
    const float pushback = check.blocked ? move.pushback * 0.6F : move.pushback;
    if (check.blocked) {
        defender.blockstun = maxi(defender.blockstun, move.blockstun);
    } else {
        defender.health = fmaxf(0.0F, defender.health - check.damage);
        defender.hitstun = maxi(defender.hitstun, move.hitstun);
        defender.blockstun = 0;
        defender.guard = HitNone;
        defender.move = -1;
        defender.move_frame = 0;
        defender.has_hit = 0;
        defender.launches_taken += move.launches;
    }
    defender.x = clampf(defender.x + pushback * direction,
                        -config.stage_half_width, config.stage_half_width);
}

__device__ bool finish_move_if_needed(FighterD& fighter, bool active_whiff) {
    if (fighter.move < 0) return false;
    const DeviceMove move = move_for(fighter);
    bool whiffed = false;
    if (active_whiff && !fighter.has_hit && fighter.move_frame == move.startup + move.active) {
        whiffed = true;
        ++fighter.whiffs;
    }
    if (fighter.move_frame >= move_total_frames(move)) {
        fighter.move = -1;
        fighter.move_frame = 0;
        fighter.has_hit = 0;
    }
    return whiffed;
}

__device__ FrameInfoD advance_frame(
    StateD& state,
    int p1_action,
    int p2_action,
    const DeviceConfig& config) {
    FighterD p1 = tick_timers(state.p1, config);
    FighterD p2 = tick_timers(state.p2, config);
    if (!busy(p1)) p1 = move_fighter(p1, p1_action, 1, config);
    if (!busy(p2)) p2 = move_fighter(p2, p2_action, -1, config);
    separate_and_clip(p1, p2, config);

    FrameInfoD info{};
    const int p1_move = p1.move;
    const int p2_move = p2.move;
    DeviceMove p1_move_data{};
    DeviceMove p2_move_data{};
    if (p1_move >= 0) p1_move_data = move_for(p1);
    if (p2_move >= 0) p2_move_data = move_for(p2);
    const bool p1_active = p1_move >= 0 && p1.move_frame > p1_move_data.startup &&
                           p1.move_frame <= p1_move_data.startup + p1_move_data.active;
    const bool p2_active = p2_move >= 0 && p2.move_frame > p2_move_data.startup &&
                           p2.move_frame <= p2_move_data.startup + p2_move_data.active;
    const bool has_p1_check = p1_active && !p1.has_hit;
    const bool has_p2_check = p2_active && !p2.has_hit;
    AttackCheckD p1_check{};
    AttackCheckD p2_check{};
    if (has_p1_check) p1_check = check_attack(p1, p2, p1_move_data, config);
    if (has_p2_check) p2_check = check_attack(p2, p1, p2_move_data, config);

    const bool p2_whiff_recovery = p2.move >= 0 && !p2.has_hit &&
        p2.move_frame > p2_move_data.startup + p2_move_data.active;
    const bool p1_whiff_recovery = p1.move >= 0 && !p1.has_hit &&
        p1.move_frame > p1_move_data.startup + p1_move_data.active;
    const bool p1_punishes = has_p1_check && p1_check.damage > 0.0F && p2_whiff_recovery;
    const bool p2_punishes = has_p2_check && p2_check.damage > 0.0F && p1_whiff_recovery;

    if (has_p1_check) {
        const DeviceMove move = p1_move_data;
        apply_attack(p1, p2, move, p1_check, 1, config);
        info.damage_to_p2 += p1_check.damage;
        if (move.hit_level == HitThrow) {
            info.p1_throw_damage += p1_check.damage;
            info.p2_throw_break = p1_check.throw_broken;
        }
        if (p1_punishes) info.p1_whiff_punish_bonus += config.whiff_punish_reward_scale * p1_check.damage;
        info.p2_block = p1_check.blocked;
    }
    if (has_p2_check) {
        const DeviceMove move = p2_move_data;
        apply_attack(p2, p1, move, p2_check, -1, config);
        info.damage_to_p1 += p2_check.damage;
        if (move.hit_level == HitThrow) {
            info.p2_throw_damage += p2_check.damage;
            info.p1_throw_break = p2_check.throw_broken;
        }
        if (p2_punishes) info.p2_whiff_punish_bonus += config.whiff_punish_reward_scale * p2_check.damage;
        info.p1_block = p2_check.blocked;
    }

    info.p1_whiff = finish_move_if_needed(p1, has_p1_check && !p1_check.in_range) ? 1 : 0;
    info.p2_whiff = finish_move_if_needed(p2, has_p2_check && !p2_check.in_range) ? 1 : 0;

    state.p1 = p1;
    state.p2 = p2;
    ++state.frame;
    if (!state.round_over) {
        const bool no_damage = info.damage_to_p1 == 0.0F && info.damage_to_p2 == 0.0F;
        const bool disengaged = fabsf(state.p2.x - state.p1.x) > config.stall_distance;
        const bool no_active_attack = state.p1.move < 0 && state.p2.move < 0;
        state.stall_frames = no_damage && disengaged && no_active_attack ? state.stall_frames + 1 : 0;
        if (state.stall_frames >= config.max_stall_frames) {
            state.round_over = 1;
            state.winner = 0;
        }
    }
    if (!state.round_over) {
        int winner = 0;
        if (state.p1.health <= 0.0F) winner = 2;
        else if (state.p2.health <= 0.0F) winner = 1;
        else if (state.frame >= config.max_frames) {
            if (config.timeout_ties_are_draws && state.p1.health == state.p2.health) {
                state.round_over = 1;
                state.winner = 0;
                return info;
            }
            winner = state.p1.health >= state.p2.health ? 1 : 2;
        }
        if (winner != 0) {
            state.round_over = 1;
            state.winner = winner;
        }
    }
    return info;
}

__device__ float scaled_win_reward(float health, const DeviceConfig& config) {
    return config.round_win_base_reward +
           config.round_win_health_reward * clampf(health / config.max_health, 0.0F, 1.0F);
}

__device__ float scaled_timeout_penalty(float health_margin, const DeviceConfig& config) {
    const float margin = clampf(health_margin, -1.0F, 1.0F);
    if (margin > 0.0F) {
        return config.timeout_ahead_penalty +
               (config.timeout_even_penalty - config.timeout_ahead_penalty) * (1.0F - margin);
    }
    if (margin < 0.0F) {
        return config.timeout_even_penalty +
               (config.timeout_behind_penalty - config.timeout_even_penalty) * (-margin);
    }
    return config.timeout_even_penalty;
}

__device__ void write_player_outputs(
    const StateD& state,
    const DeviceConfig& config,
    int player,
    std::size_t lane,
    float* observations,
    std::uint8_t* masks) {
    const FighterD& own = player == 1 ? state.p1 : state.p2;
    const FighterD& opponent = player == 1 ? state.p2 : state.p1;
    const float forward = player == 1 ? 1.0F : -1.0F;
    const float signed_distance = (opponent.x - own.x) * forward;
    const float own_forward_wall = player == 1
        ? config.stage_half_width - own.x : own.x + config.stage_half_width;
    const float own_back_wall = player == 1
        ? own.x + config.stage_half_width : config.stage_half_width - own.x;
    const float stage_width = config.stage_half_width * 2.0F;

    float own_move_remaining = 0.0F;
    if (own.move >= 0) {
        const int total = move_total_frames(move_for(own));
        own_move_remaining = fminf(1.0F, static_cast<float>(maxi(0, total - own.move_frame)) / total);
    }
    float opponent_move_remaining = 0.0F;
    if (opponent.move >= 0) {
        const int total = move_total_frames(move_for(opponent));
        opponent_move_remaining = fminf(1.0F, static_cast<float>(maxi(0, total - opponent.move_frame)) / total);
    }
    DeviceMove opponent_move{};
    if (opponent.move >= 0) opponent_move = move_for(opponent);
    const bool throw_threat = opponent.move >= 0 && opponent_move.hit_level == HitThrow &&
        opponent.move_frame <= opponent_move.startup + opponent_move.active;

    const std::size_t base = lane * kObservationSize;
    observations[base + 0] = own.health / config.max_health;
    observations[base + 1] = opponent.health / config.max_health;
    observations[base + 2] = signed_distance / stage_width;
    observations[base + 3] = fabsf(state.p2.x - state.p1.x) / stage_width;
    observations[base + 4] = own_forward_wall / stage_width;
    observations[base + 5] = own_back_wall / stage_width;
    observations[base + 6] = static_cast<float>(state.frame) / config.max_frames;
    observations[base + 7] = fminf(1.0F, own.hitstun / 60.0F);
    observations[base + 8] = fminf(1.0F, opponent.hitstun / 60.0F);
    observations[base + 9] = fminf(1.0F, own.blockstun / 60.0F);
    observations[base + 10] = fminf(1.0F, opponent.blockstun / 60.0F);
    observations[base + 11] = own.move >= 0 ? 1.0F : 0.0F;
    observations[base + 12] = opponent.move >= 0 ? 1.0F : 0.0F;
    observations[base + 13] = own_move_remaining;
    observations[base + 14] = opponent_move_remaining;
    observations[base + 15] = own.guard != HitNone ? 1.0F : 0.0F;
    observations[base + 16] = opponent.guard != HitNone ? 1.0F : 0.0F;
    observations[base + 17] = throw_threat ? 1.0F : 0.0F;
    observations[base + 18] = 1.0F;

    const std::size_t mask_base = lane * kActionCount;
    const bool only_neutral = state.round_over || busy(own);
    for (std::size_t action = 0; action < kActionCount; ++action) {
        masks[mask_base + action] = static_cast<std::uint8_t>(!only_neutral || action == 0);
    }
}

__device__ float visual_attack_likelihood(const FighterD& fighter, float distance) {
    if (fighter.move < 0) return 0.0F;
    const DeviceMove& move = move_for(fighter);
    const float proximity = clampf(1.0F - distance / fmaxf(move.range + 0.8F, 0.1F), 0.0F, 1.0F);
    const bool active_window = fighter.move_frame <= move.startup + move.active;
    return (active_window ? 0.6F : 0.25F) + 0.4F * proximity;
}

__device__ void write_visual_player_outputs(
    const StateD& state,
    const StateD& previous,
    const DeviceConfig& config,
    int player,
    std::size_t lane,
    float* observations) {
    const FighterD& own = player == 1 ? state.p1 : state.p2;
    const FighterD& opponent = player == 1 ? state.p2 : state.p1;
    const FighterD& previous_own = player == 1 ? previous.p1 : previous.p2;
    const FighterD& previous_opponent = player == 1 ? previous.p2 : previous.p1;
    const float own_velocity = own.x - previous_own.x;
    const float opponent_velocity = opponent.x - previous_opponent.x;
    const float distance = fabsf(state.p2.x - state.p1.x);
    const bool own_hit = previous_own.health - own.health > 1.0F;
    const bool opponent_hit = previous_opponent.health - opponent.health > 1.0F;
    const float own_motion = fminf(1.0F, fabsf(own_velocity) + (own.move >= 0 ? 0.08F : 0.0F));
    const float opponent_motion =
        fminf(1.0F, fabsf(opponent_velocity) + (opponent.move >= 0 ? 0.08F : 0.0F));
    const std::size_t base = lane * kVisualObservationSize;
    observations[base + 0] = own.health / config.max_health;
    observations[base + 1] = opponent.health / config.max_health;
    observations[base + 2] = own.x;
    observations[base + 3] = opponent.x;
    observations[base + 4] = distance;
    observations[base + 5] = own_velocity;
    observations[base + 6] = opponent_velocity;
    observations[base + 7] = own_motion;
    observations[base + 8] = opponent_motion;
    observations[base + 9] = own_hit ? 1.0F : 0.0F;
    observations[base + 10] = opponent_hit ? 1.0F : 0.0F;
    observations[base + 11] = visual_attack_likelihood(own, distance);
    observations[base + 12] = visual_attack_likelihood(opponent, distance);
}

__device__ void write_all_outputs(
    const StateD& state,
    const StateD& previous,
    const DeviceConfig& config,
    std::size_t lane,
    float* obs_p1,
    float* obs_p2,
    float* visual_obs_p1,
    float* visual_obs_p2,
    std::uint8_t* masks_p1,
    std::uint8_t* masks_p2) {
    write_player_outputs(state, config, 1, lane, obs_p1, masks_p1);
    write_player_outputs(state, config, 2, lane, obs_p2, masks_p2);
    write_visual_player_outputs(state, previous, config, 1, lane, visual_obs_p1);
    write_visual_player_outputs(state, previous, config, 2, lane, visual_obs_p2);
}

template <typename ActionT>
__global__ void step_kernel(
    float* state_f,
    int* state_i,
    std::size_t n,
    DeviceConfig config,
    const ActionT* p1_actions,
    const ActionT* p2_actions,
    float* obs_p1,
    float* obs_p2,
    float* visual_obs_p1,
    float* visual_obs_p2,
    std::uint8_t* masks_p1,
    std::uint8_t* masks_p2,
    float* rewards_p1,
    float* rewards_p2,
    float* sparse_rewards_p1,
    float* sparse_rewards_p2,
    std::uint8_t* terminated,
    std::uint8_t* truncated) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n) return;
    int p1_action = static_cast<int>(p1_actions[lane]);
    int p2_action = static_cast<int>(p2_actions[lane]);
    if (p1_action < 0 || p1_action >= static_cast<int>(kActionCount)) p1_action = Neutral;
    if (p2_action < 0 || p2_action >= static_cast<int>(kActionCount)) p2_action = Neutral;

    StateD state = load_state(state_f, state_i, n, lane);
    if (state.round_over) {
        rewards_p1[lane] = 0.0F;
        rewards_p2[lane] = 0.0F;
        sparse_rewards_p1[lane] = state.winner == 1 ? 1.0F : (state.winner == 2 ? -1.0F : 0.0F);
        sparse_rewards_p2[lane] = state.winner == 2 ? 1.0F : (state.winner == 1 ? -1.0F : 0.0F);
        terminated[lane] = 1;
        truncated[lane] = 0;
        write_all_outputs(state, state, config, lane, obs_p1, obs_p2,
                          visual_obs_p1, visual_obs_p2, masks_p1, masks_p2);
        return;
    }
    const StateD previous = state;
    state.p1 = start_action(state.p1, p1_action, config);
    state.p2 = start_action(state.p2, p2_action, config);

    StepInfoD info{};
    for (int frame = 0; frame < config.decision_frames; ++frame) {
        const FrameInfoD current = advance_frame(state, p1_action, p2_action, config);
        info.damage_to_p1 += current.damage_to_p1;
        info.damage_to_p2 += current.damage_to_p2;
        info.p1_throw_damage += current.p1_throw_damage;
        info.p2_throw_damage += current.p2_throw_damage;
        info.p1_throw_breaks += current.p1_throw_break;
        info.p2_throw_breaks += current.p2_throw_break;
        info.p1_whiff_punish_bonus += current.p1_whiff_punish_bonus;
        info.p2_whiff_punish_bonus += current.p2_whiff_punish_bonus;
        info.p1_whiffs += current.p1_whiff;
        info.p2_whiffs += current.p2_whiff;
        info.p1_blocks += current.p1_block;
        info.p2_blocks += current.p2_block;
        if (state.round_over) break;
    }

    const bool no_action = info.damage_to_p1 == 0.0F && info.damage_to_p2 == 0.0F &&
        is_no_action_passive(p1_action) && is_no_action_passive(p2_action) &&
        !(busy(previous.p1) || busy(previous.p2) || busy(state.p1) || busy(state.p2));
    state.no_action_frames = no_action ? state.no_action_frames + config.decision_frames : 0;
    if (state.no_action_frames >= config.no_action_timeout_frames && !state.round_over) {
        state.round_over = 1;
        state.winner = 0;
    }

    const bool was_truncated = state.frame >= config.max_frames && !state.round_over;
    const bool timed_out = state.round_over && state.frame >= config.max_frames &&
        state.p1.health > 0.0F && state.p2.health > 0.0F;
    const bool stalemate = state.round_over && state.winner == 0;
    const bool no_action_timeout = state.round_over &&
        state.no_action_frames >= config.no_action_timeout_frames;

    const float reward_damage_to_p2 = (info.damage_to_p2 - info.p1_throw_damage) +
        config.throw_reward_scale * info.p1_throw_damage;
    const float reward_damage_to_p1 = (info.damage_to_p1 - info.p2_throw_damage) +
        config.throw_reward_scale * info.p2_throw_damage;
    float reward_p1 = config.damage_dealt_scale * reward_damage_to_p2 -
        config.damage_taken_scale * info.damage_to_p1;
    float reward_p2 = config.damage_dealt_scale * reward_damage_to_p1 -
        config.damage_taken_scale * info.damage_to_p2;

    if (info.p1_throw_damage > 0.0F) reward_p1 += config.successful_throw_reward;
    if (info.p2_throw_damage > 0.0F) reward_p2 += config.successful_throw_reward;
    reward_p1 += info.p1_whiff_punish_bonus;
    reward_p2 += info.p2_whiff_punish_bonus;

    if (state.winner == 1 && !timed_out) {
        const float win_reward = scaled_win_reward(state.p1.health, config);
        reward_p1 += win_reward;
        reward_p2 -= win_reward;
        reward_p2 -= config.round_loss_penalty;
    } else if (state.winner == 2 && !timed_out) {
        const float win_reward = scaled_win_reward(state.p2.health, config);
        reward_p1 -= win_reward;
        reward_p1 -= config.round_loss_penalty;
        reward_p2 += win_reward;
    }

    if (state.round_over) {
        const float health_margin = (state.p1.health - state.p2.health) / config.max_health;
        if (timed_out) {
            reward_p1 -= scaled_timeout_penalty(health_margin, config);
            reward_p2 -= scaled_timeout_penalty(-health_margin, config);
        }
        if (stalemate) {
            reward_p1 -= config.stalemate_penalty;
            reward_p2 -= config.stalemate_penalty;
        }
        if (no_action_timeout) {
            reward_p1 -= config.no_action_timeout_penalty;
            reward_p2 -= config.no_action_timeout_penalty;
        }
    }

    if (p1_action == Neutral) reward_p1 += config.idle_penalty;
    if (p2_action == Neutral) reward_p2 += config.idle_penalty;
    const bool no_contact = info.damage_to_p1 == 0.0F && info.damage_to_p2 == 0.0F &&
        info.p1_blocks == 0 && info.p2_blocks == 0;
    if (no_contact && is_lateral(p1_action) && info.p2_whiffs == 0) reward_p1 += config.lateral_passivity_penalty;
    if (no_contact && is_lateral(p2_action) && info.p1_whiffs == 0) reward_p2 += config.lateral_passivity_penalty;
    if (fabsf(state.p2.x - state.p1.x) > 1.5F) {
        if (!is_forward(p1_action)) reward_p1 += config.far_spacing_penalty;
        if (!is_forward(p2_action)) reward_p2 += config.far_spacing_penalty;
    }
    const float own_wall = config.stage_half_width - 0.35F;
    if (state.p1.x < -own_wall && !is_forward(p1_action)) reward_p1 += config.wall_camping_penalty;
    if (state.p2.x > own_wall && !is_forward(p2_action)) reward_p2 += config.wall_camping_penalty;
    if (state.frame > config.max_frames * 0.65F &&
        info.damage_to_p1 == 0.0F && info.damage_to_p2 == 0.0F) {
        if (is_late_passive(p1_action)) reward_p1 += config.late_round_passivity_penalty;
        if (is_late_passive(p2_action)) reward_p2 += config.late_round_passivity_penalty;
    }

    reward_p1 += config.block_reward * info.p1_blocks;
    reward_p2 += config.block_reward * info.p2_blocks;
    reward_p1 += config.blocked_attack_penalty * info.p2_blocks;
    reward_p2 += config.blocked_attack_penalty * info.p1_blocks;
    reward_p1 += config.throw_break_reward * info.p1_throw_breaks;
    reward_p2 += config.throw_break_reward * info.p2_throw_breaks;
    reward_p1 += config.throw_broken_penalty * info.p2_throw_breaks;
    reward_p2 += config.throw_broken_penalty * info.p1_throw_breaks;
    reward_p1 += config.whiff_penalty * info.p1_whiffs;
    reward_p2 += config.whiff_penalty * info.p2_whiffs;

    store_state(state_f, state_i, n, lane, state);
    rewards_p1[lane] = reward_p1;
    rewards_p2[lane] = reward_p2;
    sparse_rewards_p1[lane] = state.round_over
        ? (state.winner == 1 ? 1.0F : (state.winner == 2 ? -1.0F : 0.0F)) : 0.0F;
    sparse_rewards_p2[lane] = state.round_over
        ? (state.winner == 2 ? 1.0F : (state.winner == 1 ? -1.0F : 0.0F)) : 0.0F;
    terminated[lane] = static_cast<std::uint8_t>(state.round_over);
    truncated[lane] = static_cast<std::uint8_t>(was_truncated);
    write_all_outputs(state, previous, config, lane, obs_p1, obs_p2,
                      visual_obs_p1, visual_obs_p2, masks_p1, masks_p2);
}

__global__ void reset_kernel(
    float* state_f,
    int* state_i,
    std::size_t n,
    DeviceConfig config,
    float* obs_p1,
    float* obs_p2,
    float* visual_obs_p1,
    float* visual_obs_p2,
    std::uint8_t* masks_p1,
    std::uint8_t* masks_p2,
    float* rewards_p1,
    float* rewards_p2,
    float* sparse_rewards_p1,
    float* sparse_rewards_p2,
    std::uint8_t* terminated,
    std::uint8_t* truncated,
    bool only_done,
    unsigned long long seed) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n || (only_done && !terminated[lane])) return;
    StateD state = initial_state(config, seed, lane);
    if (only_done) {
        const StateD previous = load_state(state_f, state_i, n, lane);
        state.p1.character_id = previous.p1.character_id;
        state.p2.character_id = previous.p2.character_id;
    }
    store_state(state_f, state_i, n, lane, state);
    rewards_p1[lane] = 0.0F;
    rewards_p2[lane] = 0.0F;
    sparse_rewards_p1[lane] = 0.0F;
    sparse_rewards_p2[lane] = 0.0F;
    terminated[lane] = 0;
    truncated[lane] = 0;
    write_all_outputs(state, state, config, lane, obs_p1, obs_p2,
                      visual_obs_p1, visual_obs_p2, masks_p1, masks_p2);
}

__global__ void assign_character_ids_kernel(
    int* state_i,
    const float* state_f,
    std::size_t n,
    DeviceConfig config,
    const OpponentProfileParameters* profiles,
    std::size_t profile_count,
    const std::uint32_t* assignments,
    int learner_player,
    float* obs_p1,
    float* obs_p2,
    float* visual_obs_p1,
    float* visual_obs_p2,
    std::uint8_t* masks_p1,
    std::uint8_t* masks_p2) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n) return;
    const std::uint32_t profile_index = assignments[lane];
    std::uint32_t opponent_character = kJunCharacterId;
    if (profile_index < profile_count &&
        profiles[profile_index].character_id < kRosterCharacterCount) {
        opponent_character = profiles[profile_index].character_id;
    }
    const bool learner_is_p1 = learner_player == 1 ||
        (learner_player == 0 && ((lane / kEvaluationStyleCount) & 1ULL) == 0ULL);
    state_i[(learner_is_p1 ? CharacterId : kFighterIntFields + CharacterId) * n + lane] =
        static_cast<int>(kJunCharacterId);
    state_i[(learner_is_p1 ? kFighterIntFields + CharacterId : CharacterId) * n + lane] =
        static_cast<int>(opponent_character);
    const StateD state = load_state(state_f, state_i, n, lane);
    write_player_outputs(state, config, 1, lane, obs_p1, masks_p1);
    write_player_outputs(state, config, 2, lane, obs_p2, masks_p2);
    const float distance = fabsf(state.p2.x - state.p1.x);
    const std::size_t visual_base = lane * kVisualObservationSize;
    visual_obs_p1[visual_base + 11] = visual_attack_likelihood(state.p1, distance);
    visual_obs_p1[visual_base + 12] = visual_attack_likelihood(state.p2, distance);
    visual_obs_p2[visual_base + 11] = visual_attack_likelihood(state.p2, distance);
    visual_obs_p2[visual_base + 12] = visual_attack_likelihood(state.p1, distance);
}

__global__ void refresh_outputs_kernel(
    const float* state_f,
    const int* state_i,
    std::size_t n,
    DeviceConfig config,
    float* obs_p1,
    float* obs_p2,
    float* visual_obs_p1,
    float* visual_obs_p2,
    std::uint8_t* masks_p1,
    std::uint8_t* masks_p2,
    float* rewards_p1,
    float* rewards_p2,
    float* sparse_rewards_p1,
    float* sparse_rewards_p2,
    std::uint8_t* terminated,
    std::uint8_t* truncated) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n) return;
    const StateD state = load_state(state_f, state_i, n, lane);
    rewards_p1[lane] = 0.0F;
    rewards_p2[lane] = 0.0F;
    sparse_rewards_p1[lane] = 0.0F;
    sparse_rewards_p2[lane] = 0.0F;
    terminated[lane] = static_cast<std::uint8_t>(state.round_over);
    truncated[lane] = 0;
    write_all_outputs(state, state, config, lane, obs_p1, obs_p2,
                      visual_obs_p1, visual_obs_p2, masks_p1, masks_p2);
}

constexpr std::size_t kSummaryScalarCount = 8;
constexpr std::size_t kSummaryStyleStride = 4;
constexpr std::size_t kSummaryCount =
    kSummaryScalarCount + kEvaluationStyleCount * kSummaryStyleStride;

__global__ void summarize_episodes_kernel(
    const float* state_f,
    const int* state_i,
    const std::uint8_t* terminated,
    std::size_t n,
    DeviceConfig config,
    int learner_player,
    unsigned long long* counts,
    double* sums) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= n || !terminated[lane]) return;

    const int winner = state_i[Winner * n + lane];
    const int frame = state_i[Frame * n + lane];
    const int no_action_frames = state_i[NoActionFrames * n + lane];
    const float p1_health = state_f[P1Health * n + lane];
    const float p2_health = state_f[P2Health * n + lane];
    const bool won = winner == learner_player;
    const bool lost = winner != 0 && winner != learner_player;
    const bool draw = winner == 0;
    const bool timed_out = frame >= config.max_frames && p1_health > 0.0F && p2_health > 0.0F;
    const bool no_action_timeout = draw && no_action_frames >= config.no_action_timeout_frames;

    atomicAdd(counts + 0, 1ULL);
    atomicAdd(counts + 1, won ? 1ULL : 0ULL);
    atomicAdd(counts + 2, lost ? 1ULL : 0ULL);
    atomicAdd(counts + 3, draw ? 1ULL : 0ULL);
    atomicAdd(counts + 4, timed_out ? 1ULL : 0ULL);
    atomicAdd(counts + 5, draw ? 1ULL : 0ULL);
    atomicAdd(counts + 6, no_action_timeout ? 1ULL : 0ULL);
    atomicAdd(counts + 7, static_cast<unsigned long long>(frame));

    const std::size_t style = lane % kEvaluationStyleCount;
    const std::size_t style_base = kSummaryScalarCount + style * kSummaryStyleStride;
    atomicAdd(counts + style_base + 0, 1ULL);
    atomicAdd(counts + style_base + 1, won ? 1ULL : 0ULL);
    atomicAdd(counts + style_base + 2, lost ? 1ULL : 0ULL);
    atomicAdd(counts + style_base + 3, draw ? 1ULL : 0ULL);

    const float own_health = learner_player == 1 ? p1_health : p2_health;
    const float opponent_health = learner_player == 1 ? p2_health : p1_health;
    atomicAdd(sums + 0, static_cast<double>(config.max_health - opponent_health));
    atomicAdd(sums + 1, static_cast<double>(config.max_health - own_health));
}

cudaStream_t as_stream(void* stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

int blocks_for(std::size_t count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
}

}  // namespace

struct GpuSimulatorBatch::Impl {
    std::size_t count;
    Config config;
    DeviceConfig device_config;
    float* state_f = nullptr;
    int* state_i = nullptr;
    float* observations_p1 = nullptr;
    float* observations_p2 = nullptr;
    float* visual_observations_p1 = nullptr;
    float* visual_observations_p2 = nullptr;
    std::uint8_t* masks_p1 = nullptr;
    std::uint8_t* masks_p2 = nullptr;
    float* rewards_p1 = nullptr;
    float* rewards_p2 = nullptr;
    float* sparse_rewards_p1 = nullptr;
    float* sparse_rewards_p2 = nullptr;
    std::uint8_t* terminated = nullptr;
    std::uint8_t* truncated = nullptr;
    std::uint8_t* host_actions_p1 = nullptr;
    std::uint8_t* host_actions_p2 = nullptr;
    unsigned long long* summary_counts = nullptr;
    double* summary_sums = nullptr;

    Impl(std::size_t environment_count, Config simulation_config)
        : count(environment_count), config(simulation_config), device_config(to_device_config(config)) {
        if (count == 0) throw std::invalid_argument("environment_count must be greater than zero");
        try {
        check_cuda(cudaMalloc(&state_f, sizeof(float) * kFloatStateFields * count), "allocate GPU float state");
        check_cuda(cudaMalloc(&state_i, sizeof(int) * kIntStateFields * count), "allocate GPU integer state");
        check_cuda(cudaMalloc(&observations_p1, sizeof(float) * kObservationSize * count), "allocate P1 observations");
        check_cuda(cudaMalloc(&observations_p2, sizeof(float) * kObservationSize * count), "allocate P2 observations");
        check_cuda(cudaMalloc(&visual_observations_p1, sizeof(float) * kVisualObservationSize * count),
                   "allocate P1 visual observations");
        check_cuda(cudaMalloc(&visual_observations_p2, sizeof(float) * kVisualObservationSize * count),
                   "allocate P2 visual observations");
        check_cuda(cudaMalloc(&masks_p1, sizeof(std::uint8_t) * kActionCount * count), "allocate P1 masks");
        check_cuda(cudaMalloc(&masks_p2, sizeof(std::uint8_t) * kActionCount * count), "allocate P2 masks");
        check_cuda(cudaMalloc(&rewards_p1, sizeof(float) * count), "allocate P1 rewards");
        check_cuda(cudaMalloc(&rewards_p2, sizeof(float) * count), "allocate P2 rewards");
        check_cuda(cudaMalloc(&sparse_rewards_p1, sizeof(float) * count), "allocate sparse P1 rewards");
        check_cuda(cudaMalloc(&sparse_rewards_p2, sizeof(float) * count), "allocate sparse P2 rewards");
        check_cuda(cudaMalloc(&terminated, sizeof(std::uint8_t) * count), "allocate terminated flags");
        check_cuda(cudaMalloc(&truncated, sizeof(std::uint8_t) * count), "allocate truncated flags");
        check_cuda(cudaMalloc(&host_actions_p1, sizeof(std::uint8_t) * count), "allocate debug P1 actions");
        check_cuda(cudaMalloc(&host_actions_p2, sizeof(std::uint8_t) * count), "allocate debug P2 actions");
        check_cuda(cudaMalloc(&summary_counts, sizeof(unsigned long long) * kSummaryCount),
                   "allocate evaluation counters");
        check_cuda(cudaMalloc(&summary_sums, sizeof(double) * 2), "allocate evaluation sums");
        } catch (...) {
            release();
            throw;
        }
    }

    void release() noexcept {
        cudaFree(summary_sums);
        cudaFree(summary_counts);
        cudaFree(host_actions_p2);
        cudaFree(host_actions_p1);
        cudaFree(truncated);
        cudaFree(terminated);
        cudaFree(rewards_p2);
        cudaFree(rewards_p1);
        cudaFree(sparse_rewards_p2);
        cudaFree(sparse_rewards_p1);
        cudaFree(masks_p2);
        cudaFree(masks_p1);
        cudaFree(visual_observations_p2);
        cudaFree(visual_observations_p1);
        cudaFree(observations_p2);
        cudaFree(observations_p1);
        cudaFree(state_i);
        cudaFree(state_f);
    }

    ~Impl() { release(); }

    void launch_reset(bool only_done, std::uint64_t seed, cudaStream_t stream) {
        reset_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
            state_f, state_i, count, device_config,
            observations_p1, observations_p2, visual_observations_p1, visual_observations_p2,
            masks_p1, masks_p2,
            rewards_p1, rewards_p2, sparse_rewards_p1, sparse_rewards_p2,
            terminated, truncated, only_done, seed);
        check_cuda(cudaGetLastError(), only_done ? "launch reset-done kernel" : "launch reset kernel");
    }

    template <typename ActionT>
    void launch_step(const ActionT* p1_actions, const ActionT* p2_actions, cudaStream_t stream) {
        if (p1_actions == nullptr || p2_actions == nullptr) {
            throw std::invalid_argument("device action pointers cannot be null");
        }
        step_kernel<<<blocks_for(count), kThreads, 0, stream>>>(
            state_f, state_i, count, device_config, p1_actions, p2_actions,
            observations_p1, observations_p2, visual_observations_p1, visual_observations_p2,
            masks_p1, masks_p2,
            rewards_p1, rewards_p2, sparse_rewards_p1, sparse_rewards_p2,
            terminated, truncated);
        check_cuda(cudaGetLastError(), "launch GPU simulator step kernel");
    }
};

GpuSimulatorBatch::GpuSimulatorBatch(std::size_t environment_count, Config config)
    : impl_(std::make_unique<Impl>(environment_count, config)) {
    reset();
    synchronize();
}

GpuSimulatorBatch::~GpuSimulatorBatch() = default;
GpuSimulatorBatch::GpuSimulatorBatch(GpuSimulatorBatch&&) noexcept = default;
GpuSimulatorBatch& GpuSimulatorBatch::operator=(GpuSimulatorBatch&&) noexcept = default;

std::size_t GpuSimulatorBatch::size() const noexcept { return impl_->count; }
const Config& GpuSimulatorBatch::config() const noexcept { return impl_->config; }

GpuBatchDeviceView GpuSimulatorBatch::device_view() const noexcept {
    return {
        impl_->observations_p1, impl_->observations_p2,
        impl_->visual_observations_p1, impl_->visual_observations_p2,
        impl_->masks_p1, impl_->masks_p2,
        impl_->rewards_p1, impl_->rewards_p2,
        impl_->sparse_rewards_p1, impl_->sparse_rewards_p2,
        impl_->terminated, impl_->truncated,
        impl_->state_i + Winner * impl_->count, impl_->count,
    };
}

void GpuSimulatorBatch::reset(void* stream) {
    impl_->launch_reset(false, 0, as_stream(stream));
}

void GpuSimulatorBatch::reset_done(void* stream) {
    impl_->launch_reset(true, 0, as_stream(stream));
}

void GpuSimulatorBatch::reset_seeded(std::uint64_t seed, void* stream) {
    impl_->launch_reset(false, seed, as_stream(stream));
}

void GpuSimulatorBatch::reset_done_seeded(std::uint64_t seed, void* stream) {
    impl_->launch_reset(true, seed, as_stream(stream));
}

void GpuSimulatorBatch::set_character_move_specs(
    std::span<const CharacterMoveParameters> moves,
    void* stream) {
    constexpr std::size_t expected = kRosterCharacterCount * kCharacterMoveSlotCount;
    if (moves.size() != expected) {
        throw std::invalid_argument("character move catalog must contain exactly 42 x 6 rows");
    }
    std::vector<DeviceMove> catalog(expected);
    std::vector<std::uint8_t> seen(expected, 0);
    for (const auto& move : moves) {
        if (move.character_id >= kRosterCharacterCount || move.slot >= kCharacterMoveSlotCount) {
            throw std::invalid_argument("character move catalog contains an invalid character or slot");
        }
        if (move.hit_level < HitHigh || move.hit_level > HitThrow || move.startup <= 0 ||
            move.active <= 0 || move.recovery < 0 || move.damage < 0.0F || move.range <= 0.0F) {
            throw std::invalid_argument("character move catalog contains invalid combat values");
        }
        const std::size_t index =
            move.character_id * kCharacterMoveSlotCount + move.slot;
        if (seen[index] != 0) {
            throw std::invalid_argument("character move catalog contains a duplicate slot");
        }
        seen[index] = 1;
        catalog[index] = {
            move.hit_level, move.startup, move.active, move.recovery,
            move.damage, move.range, move.hitstun, move.blockstun,
            move.pushback, move.whiff_recovery, move.launches,
        };
    }
    if (std::find(seen.begin(), seen.end(), std::uint8_t{0}) != seen.end()) {
        throw std::invalid_argument("character move catalog is missing a required slot");
    }
    const cudaStream_t cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyToSymbolAsync(
        c_character_moves, catalog.data(), sizeof(DeviceMove) * catalog.size(),
        0, cudaMemcpyHostToDevice, cuda_stream), "upload character move catalog");
    const int enabled = 1;
    check_cuda(cudaMemcpyToSymbolAsync(
        c_character_move_catalog_enabled, &enabled, sizeof(enabled),
        0, cudaMemcpyHostToDevice, cuda_stream), "enable character move catalog");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize character move catalog upload");
}

void GpuSimulatorBatch::set_opponent_characters_device(
    const OpponentProfileParameters* device_profiles,
    std::size_t profile_count,
    const std::uint32_t* device_profile_assignments,
    int learner_player,
    void* stream) {
    if (device_profiles == nullptr || device_profile_assignments == nullptr || profile_count == 0) {
        throw std::invalid_argument("profile table and assignments must be non-null and non-empty");
    }
    if (learner_player < 0 || learner_player > 2) {
        throw std::invalid_argument("learner_player must be 0 (mirrored), 1, or 2");
    }
    const cudaStream_t cuda_stream = as_stream(stream);
    assign_character_ids_kernel<<<blocks_for(impl_->count), kThreads, 0, cuda_stream>>>(
        impl_->state_i, impl_->state_f, impl_->count, impl_->device_config,
        device_profiles, profile_count, device_profile_assignments, learner_player,
        impl_->observations_p1, impl_->observations_p2,
        impl_->visual_observations_p1, impl_->visual_observations_p2,
        impl_->masks_p1, impl_->masks_p2);
    check_cuda(cudaGetLastError(), "launch GPU character-assignment kernel");
}

void GpuSimulatorBatch::step_device(
    const std::uint8_t* device_p1_actions,
    const std::uint8_t* device_p2_actions,
    void* stream) {
    impl_->launch_step(device_p1_actions, device_p2_actions, as_stream(stream));
}

void GpuSimulatorBatch::step_device_i64(
    const std::int64_t* device_p1_actions,
    const std::int64_t* device_p2_actions,
    void* stream) {
    impl_->launch_step(device_p1_actions, device_p2_actions, as_stream(stream));
}

void GpuSimulatorBatch::step_host(
    std::span<const std::uint8_t> p1_actions,
    std::span<const std::uint8_t> p2_actions,
    void* stream) {
    if (p1_actions.size() != impl_->count || p2_actions.size() != impl_->count) {
        throw std::invalid_argument("host action spans must match environment_count");
    }
    const cudaStream_t cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(impl_->host_actions_p1, p1_actions.data(), impl_->count,
                               cudaMemcpyHostToDevice, cuda_stream), "upload debug P1 actions");
    check_cuda(cudaMemcpyAsync(impl_->host_actions_p2, p2_actions.data(), impl_->count,
                               cudaMemcpyHostToDevice, cuda_stream), "upload debug P2 actions");
    impl_->launch_step(impl_->host_actions_p1, impl_->host_actions_p2, cuda_stream);
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize host-action step");
}

void GpuSimulatorBatch::synchronize(void* stream) const {
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize GPU simulator");
}

void GpuSimulatorBatch::upload_states(std::span<const State> states, void* stream) {
    if (states.size() != impl_->count) {
        throw std::invalid_argument("state span must match environment_count");
    }
    std::vector<float> floats(kFloatStateFields * impl_->count);
    std::vector<int> ints(kIntStateFields * impl_->count);
    const auto put_fighter = [&](std::size_t lane, int player, const FighterRuntime& fighter) {
        const int float_base = player == 1 ? P1Health : P2Health;
        const int int_base = player == 1 ? 0 : kFighterIntFields;
        floats[(float_base + 0) * impl_->count + lane] = static_cast<float>(fighter.health);
        floats[(float_base + 1) * impl_->count + lane] = static_cast<float>(fighter.x);
        floats[(float_base + 2) * impl_->count + lane] = static_cast<float>(fighter.y);
        ints[(int_base + Guard) * impl_->count + lane] = static_cast<int>(fighter.guard);
        ints[(int_base + Move) * impl_->count + lane] = fighter.move;
        ints[(int_base + MoveFrame) * impl_->count + lane] = fighter.move_frame;
        ints[(int_base + HasHit) * impl_->count + lane] = static_cast<int>(fighter.has_hit);
        ints[(int_base + Hitstun) * impl_->count + lane] = fighter.hitstun;
        ints[(int_base + Blockstun) * impl_->count + lane] = fighter.blockstun;
        ints[(int_base + Airborne) * impl_->count + lane] = fighter.airborne;
        ints[(int_base + ThrowBreakActive) * impl_->count + lane] = fighter.throw_break_active;
        ints[(int_base + LaunchesTaken) * impl_->count + lane] = fighter.launches_taken;
        ints[(int_base + Whiffs) * impl_->count + lane] = fighter.whiffs;
        ints[(int_base + CharacterId) * impl_->count + lane] = static_cast<int>(kJunCharacterId);
    };
    for (std::size_t lane = 0; lane < impl_->count; ++lane) {
        put_fighter(lane, 1, states[lane].p1);
        put_fighter(lane, 2, states[lane].p2);
        ints[Frame * impl_->count + lane] = states[lane].frame;
        ints[StallFrames * impl_->count + lane] = states[lane].stall_frames;
        ints[NoActionFrames * impl_->count + lane] = states[lane].no_action_frames;
        ints[RoundOver * impl_->count + lane] = static_cast<int>(states[lane].round_over);
        ints[Winner * impl_->count + lane] = states[lane].winner;
    }
    const cudaStream_t cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(impl_->state_f, floats.data(), sizeof(float) * floats.size(),
                               cudaMemcpyHostToDevice, cuda_stream), "upload float states");
    check_cuda(cudaMemcpyAsync(impl_->state_i, ints.data(), sizeof(int) * ints.size(),
                               cudaMemcpyHostToDevice, cuda_stream), "upload integer states");
    refresh_outputs_kernel<<<blocks_for(impl_->count), kThreads, 0, cuda_stream>>>(
        impl_->state_f, impl_->state_i, impl_->count, impl_->device_config,
        impl_->observations_p1, impl_->observations_p2,
        impl_->visual_observations_p1, impl_->visual_observations_p2,
        impl_->masks_p1, impl_->masks_p2,
        impl_->rewards_p1, impl_->rewards_p2,
        impl_->sparse_rewards_p1, impl_->sparse_rewards_p2,
        impl_->terminated, impl_->truncated);
    check_cuda(cudaGetLastError(), "launch uploaded-state output refresh");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize state upload");
}

std::vector<State> GpuSimulatorBatch::download_states(void* stream) const {
    std::vector<float> floats(kFloatStateFields * impl_->count);
    std::vector<int> ints(kIntStateFields * impl_->count);
    const cudaStream_t cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(floats.data(), impl_->state_f, sizeof(float) * floats.size(),
                               cudaMemcpyDeviceToHost, cuda_stream), "download float states");
    check_cuda(cudaMemcpyAsync(ints.data(), impl_->state_i, sizeof(int) * ints.size(),
                               cudaMemcpyDeviceToHost, cuda_stream), "download integer states");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize state download");
    std::vector<State> states(impl_->count);
    const auto get_fighter = [&](std::size_t lane, int player) {
        const int float_base = player == 1 ? P1Health : P2Health;
        const int int_base = player == 1 ? 0 : kFighterIntFields;
        FighterRuntime fighter{};
        fighter.health = floats[(float_base + 0) * impl_->count + lane];
        fighter.x = floats[(float_base + 1) * impl_->count + lane];
        fighter.y = floats[(float_base + 2) * impl_->count + lane];
        fighter.guard = static_cast<HitLevel>(ints[(int_base + Guard) * impl_->count + lane]);
        fighter.move = ints[(int_base + Move) * impl_->count + lane];
        fighter.move_frame = ints[(int_base + MoveFrame) * impl_->count + lane];
        fighter.has_hit = ints[(int_base + HasHit) * impl_->count + lane] != 0;
        fighter.hitstun = ints[(int_base + Hitstun) * impl_->count + lane];
        fighter.blockstun = ints[(int_base + Blockstun) * impl_->count + lane];
        fighter.airborne = ints[(int_base + Airborne) * impl_->count + lane];
        fighter.throw_break_active = ints[(int_base + ThrowBreakActive) * impl_->count + lane];
        fighter.launches_taken = ints[(int_base + LaunchesTaken) * impl_->count + lane];
        fighter.whiffs = ints[(int_base + Whiffs) * impl_->count + lane];
        return fighter;
    };
    for (std::size_t lane = 0; lane < impl_->count; ++lane) {
        states[lane].p1 = get_fighter(lane, 1);
        states[lane].p2 = get_fighter(lane, 2);
        states[lane].frame = ints[Frame * impl_->count + lane];
        states[lane].stall_frames = ints[StallFrames * impl_->count + lane];
        states[lane].no_action_frames = ints[NoActionFrames * impl_->count + lane];
        states[lane].round_over = ints[RoundOver * impl_->count + lane] != 0;
        states[lane].winner = ints[Winner * impl_->count + lane];
    }
    return states;
}

template <typename T>
std::vector<T> download_buffer(const T* device, std::size_t count, void* stream, const char* name) {
    std::vector<T> host(count);
    const cudaStream_t cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(host.data(), device, sizeof(T) * count,
                               cudaMemcpyDeviceToHost, cuda_stream), name);
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize GPU download");
    return host;
}

std::vector<float> GpuSimulatorBatch::download_observations(int player, void* stream) const {
    if (player != 1 && player != 2) throw std::invalid_argument("player must be 1 or 2");
    return download_buffer(player == 1 ? impl_->observations_p1 : impl_->observations_p2,
                           impl_->count * kObservationSize, stream, "download observations");
}

std::vector<float> GpuSimulatorBatch::download_visual_observations(int player, void* stream) const {
    if (player != 1 && player != 2) throw std::invalid_argument("player must be 1 or 2");
    return download_buffer(player == 1 ? impl_->visual_observations_p1 : impl_->visual_observations_p2,
                           impl_->count * kVisualObservationSize, stream,
                           "download visual observations");
}

std::vector<std::uint8_t> GpuSimulatorBatch::download_action_masks(int player, void* stream) const {
    if (player != 1 && player != 2) throw std::invalid_argument("player must be 1 or 2");
    return download_buffer(player == 1 ? impl_->masks_p1 : impl_->masks_p2,
                           impl_->count * kActionCount, stream, "download action masks");
}

std::vector<float> GpuSimulatorBatch::download_rewards(int player, void* stream) const {
    if (player != 1 && player != 2) throw std::invalid_argument("player must be 1 or 2");
    return download_buffer(player == 1 ? impl_->rewards_p1 : impl_->rewards_p2,
                           impl_->count, stream, "download rewards");
}

std::vector<float> GpuSimulatorBatch::download_sparse_rewards(int player, void* stream) const {
    if (player != 1 && player != 2) throw std::invalid_argument("player must be 1 or 2");
    return download_buffer(player == 1 ? impl_->sparse_rewards_p1 : impl_->sparse_rewards_p2,
                           impl_->count, stream, "download sparse rewards");
}

std::vector<std::uint8_t> GpuSimulatorBatch::download_terminated(void* stream) const {
    return download_buffer(impl_->terminated, impl_->count, stream, "download terminated flags");
}

std::vector<std::int32_t> GpuSimulatorBatch::download_winners(void* stream) const {
    return download_buffer(impl_->state_i + Winner * impl_->count,
                           impl_->count, stream, "download winners");
}

GpuEpisodeSummary GpuSimulatorBatch::summarize_episodes(int learner_player, void* stream) const {
    if (learner_player != 1 && learner_player != 2) {
        throw std::invalid_argument("learner_player must be 1 or 2");
    }
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaMemsetAsync(impl_->summary_counts, 0,
                               sizeof(unsigned long long) * kSummaryCount, cuda_stream),
               "clear evaluation counters");
    check_cuda(cudaMemsetAsync(impl_->summary_sums, 0, sizeof(double) * 2, cuda_stream),
               "clear evaluation sums");
    summarize_episodes_kernel<<<blocks_for(impl_->count), kThreads, 0, cuda_stream>>>(
        impl_->state_f, impl_->state_i, impl_->terminated, impl_->count,
        impl_->device_config, learner_player, impl_->summary_counts, impl_->summary_sums);
    check_cuda(cudaGetLastError(), "launch evaluation summary kernel");

    std::array<unsigned long long, kSummaryCount> counts{};
    std::array<double, 2> sums{};
    check_cuda(cudaMemcpyAsync(counts.data(), impl_->summary_counts,
                               sizeof(unsigned long long) * counts.size(),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "download evaluation counters");
    check_cuda(cudaMemcpyAsync(sums.data(), impl_->summary_sums, sizeof(double) * sums.size(),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "download evaluation sums");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize evaluation summary");

    GpuEpisodeSummary summary{};
    summary.episodes = counts[0];
    summary.wins = counts[1];
    summary.losses = counts[2];
    summary.draws = counts[3];
    summary.timeouts = counts[4];
    summary.stalemates = counts[5];
    summary.no_action_timeouts = counts[6];
    summary.total_frames = counts[7];
    summary.total_damage_dealt = sums[0];
    summary.total_damage_taken = sums[1];
    for (std::size_t style = 0; style < kEvaluationStyleCount; ++style) {
        const std::size_t base = kSummaryScalarCount + style * kSummaryStyleStride;
        summary.style_episodes[style] = counts[base + 0];
        summary.style_wins[style] = counts[base + 1];
        summary.style_losses[style] = counts[base + 2];
        summary.style_draws[style] = counts[base + 3];
    }
    return summary;
}

int cuda_device_count() noexcept {
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess ? count : 0;
}

}  // namespace t8::v2
