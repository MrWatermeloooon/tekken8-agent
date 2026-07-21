#pragma once

#include <array>
#include <cstddef>
#include <cstdint>
#include <string_view>

namespace t8::v2 {

enum class HitLevel : std::uint8_t {
    None,
    High,
    Mid,
    Low,
    Throw,
};

enum class Action : std::uint8_t {
    Neutral,
    WalkForward,
    WalkBack,
    DashForward,
    DashBack,
    Crouch,
    Stand,
    Jump,
    SidestepLeft,
    SidestepRight,
    SidewalkLeft,
    SidewalkRight,
    BlockHigh,
    BlockLow,
    LowParry,
    ThrowBreak1,
    ThrowBreak2,
    ThrowBreak12,
    Jab,
    Df1,
    F2,
    Db3,
    Hopkick,
    Throw,
    Count,
};

inline constexpr std::size_t kActionCount = static_cast<std::size_t>(Action::Count);
inline constexpr std::size_t kObservationSize = 19;
inline constexpr std::size_t kVisualObservationSize = 13;

inline constexpr std::array<std::string_view, kActionCount> kActionNames = {
    "neutral",
    "walk_forward",
    "walk_back",
    "dash_forward",
    "dash_back",
    "crouch",
    "stand",
    "jump",
    "sidestep_left",
    "sidestep_right",
    "sidewalk_left",
    "sidewalk_right",
    "block_high",
    "block_low",
    "low_parry",
    "throw_break_1",
    "throw_break_2",
    "throw_break_1p2",
    "jab",
    "df1",
    "f2",
    "db3",
    "hopkick",
    "throw",
};

struct Config {
    double max_health = 180.0;
    double stage_half_width = 3.6;
    int decision_frames = 4;
    int max_frames = 60 * 60;
    bool timeout_ties_are_draws = false;
    bool randomize_initial_positions = false;
    double initial_center_jitter = 0.45;
    double initial_distance_min = 1.25;
    double initial_distance_max = 2.25;
    double walk_speed = 0.025;
    double dash_speed = 0.07;
    int jump_frames = 32;
    int throw_break_frames = 9;
    double sidestep_speed = 0.045;
    double sidewalk_speed = 0.075;
    double lateral_return_speed = 0.012;
    double sidestep_evasion_width = 0.33;
    double body_radius = 0.22;
    double stall_distance = 1.5;
    int max_stall_frames = 360;
    int no_action_timeout_frames = 300;
    double no_action_timeout_penalty = 320.0;
    double round_win_base_reward = 40.0;
    double round_win_health_reward = 60.0;
    double round_loss_penalty = 160.0;
    double timeout_ahead_penalty = 60.0;
    double timeout_even_penalty = 120.0;
    double timeout_behind_penalty = 240.0;
    double stalemate_penalty = 260.0;
    double damage_dealt_scale = 0.65;
    double damage_taken_scale = 0.85;
    double throw_reward_scale = 0.70;
    double successful_throw_reward = 1.0;
    double throw_break_reward = 0.35;
    double throw_broken_penalty = -0.20;
    double whiff_punish_reward_scale = 0.15;
    double block_reward = 0.12;
    double blocked_attack_penalty = -0.08;
    double idle_penalty = -0.02;
    double far_spacing_penalty = 0.0;
    double lateral_passivity_penalty = -0.04;
    double wall_camping_penalty = -0.05;
    double late_round_passivity_penalty = -0.03;
    double whiff_penalty = -0.18;
};

struct MoveSpec {
    std::string_view key;
    HitLevel hit_level;
    int startup;
    int active;
    int recovery;
    double damage;
    double range;
    int hitstun;
    int blockstun;
    double pushback;
    int whiff_recovery = 0;
    bool launches = false;

    [[nodiscard]] constexpr int total_frames() const noexcept {
        return startup + active + recovery + whiff_recovery;
    }
};

inline constexpr std::array<MoveSpec, 6> kMoves = {{
    {"jab", HitLevel::High, 10, 2, 13, 7.0, 0.82, 14, 7, 0.08, 0, false},
    {"df1", HitLevel::Mid, 13, 3, 18, 12.0, 0.95, 18, 11, 0.11, 0, false},
    {"f2", HitLevel::Mid, 16, 3, 22, 18.0, 1.25, 24, 14, 0.16, 0, false},
    {"db3", HitLevel::Low, 18, 3, 24, 10.0, 0.90, 17, 15, 0.06, 0, false},
    {"hopkick", HitLevel::Mid, 15, 4, 30, 21.0, 0.88, 34, 18, 0.18, 0, true},
    {"throw", HitLevel::Throw, 12, 2, 24, 25.0, 0.48, 30, 0, 0.22, 0, false},
}};

struct FighterRuntime {
    double health = 180.0;
    double x = 0.0;
    double y = 0.0;
    HitLevel guard = HitLevel::None;
    int move = -1;
    int move_frame = 0;
    bool has_hit = false;
    int hitstun = 0;
    int blockstun = 0;
    int airborne = 0;
    int throw_break_active = 0;
    int launches_taken = 0;
    int whiffs = 0;

    [[nodiscard]] bool busy() const noexcept {
        return move >= 0 || hitstun > 0 || blockstun > 0 || airborne > 0;
    }
};

struct State {
    FighterRuntime p1{};
    FighterRuntime p2{};
    int frame = 0;
    int stall_frames = 0;
    int no_action_frames = 0;
    bool round_over = false;
    int winner = 0;

    [[nodiscard]] double distance() const noexcept;
};

struct StepInfo {
    double damage_to_p1 = 0.0;
    double damage_to_p2 = 0.0;
    double p1_throw_damage = 0.0;
    double p2_throw_damage = 0.0;
    int p1_throw_breaks = 0;
    int p2_throw_breaks = 0;
    double p1_whiff_punish_bonus = 0.0;
    double p2_whiff_punish_bonus = 0.0;
    int p1_whiffs = 0;
    int p2_whiffs = 0;
    int p1_blocks = 0;
    int p2_blocks = 0;
    bool timed_out = false;
    bool stalemate = false;
    bool no_action_timeout = false;
    int stall_frames = 0;
    int no_action_frames = 0;
    int frame = 0;
};

struct StepResult {
    State state{};
    double reward_p1 = 0.0;
    double reward_p2 = 0.0;
    bool terminated = false;
    bool truncated = false;
    StepInfo info{};
};

class Simulator {
public:
    explicit Simulator(Config config = {});

    [[nodiscard]] const Config& config() const noexcept { return config_; }
    [[nodiscard]] const State& state() const noexcept { return state_; }

    const State& reset(std::uint64_t seed = 0) noexcept;
    void set_state(State state) noexcept { state_ = state; }
    [[nodiscard]] StepResult step(Action p1_action, Action p2_action);
    [[nodiscard]] std::array<float, kObservationSize> observation(int player) const;
    [[nodiscard]] std::array<float, kVisualObservationSize> visual_observation(
        int player,
        const State* previous_state = nullptr) const;
    [[nodiscard]] std::array<bool, kActionCount> legal_action_mask(int player) const;

private:
    Config config_{};
    State state_{};

    struct AttackCheck;
    struct FrameInfo;

    [[nodiscard]] State initial_state(std::uint64_t seed = 0) const noexcept;
    [[nodiscard]] FighterRuntime start_action(FighterRuntime fighter, Action action) const noexcept;
    [[nodiscard]] FrameInfo advance_frame(State& state, Action p1_action, Action p2_action) const;
    [[nodiscard]] FighterRuntime tick_timers(FighterRuntime fighter) const noexcept;
    [[nodiscard]] FighterRuntime move_fighter(FighterRuntime fighter, Action action, int forward) const noexcept;
    void separate_and_clip(FighterRuntime& p1, FighterRuntime& p2) const noexcept;
    [[nodiscard]] const MoveSpec* active_move(const FighterRuntime& fighter) const noexcept;
    [[nodiscard]] AttackCheck check_attack(
        const FighterRuntime& attacker,
        const FighterRuntime& defender,
        const MoveSpec& move) const noexcept;
    void apply_attack(
        FighterRuntime& attacker,
        FighterRuntime& defender,
        const MoveSpec& move,
        const AttackCheck& check,
        int direction) const noexcept;
    [[nodiscard]] bool finish_move_if_needed(FighterRuntime& fighter, bool active_whiff) const noexcept;
    void check_stalemate(State& state, double damage_to_p1, double damage_to_p2) const noexcept;
    void check_round_over(State& state) const noexcept;
    [[nodiscard]] double scaled_win_reward(double health) const noexcept;
    [[nodiscard]] double scaled_timeout_penalty(double health_margin) const noexcept;
};

[[nodiscard]] constexpr std::size_t action_index(Action action) noexcept {
    return static_cast<std::size_t>(action);
}

}  // namespace t8::v2
