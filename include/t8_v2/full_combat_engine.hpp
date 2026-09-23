#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace t8::v2 {

// Frame-stepped scalar full-combat engine (60 frames per second).
//
// This is the correctness reference for full-move simulation: one fight,
// advanced one frame at a time, with explicit active windows, reach and
// lateral tracking per hit, per-frame travel, walking/dashing/sidestepping,
// body collision, wall-aware pushback, posture (standing, crouching,
// airborne, grounded, wake-up, wall splat), juggles, and wall/floor/balcony
// breaks with the stage transitions they cause.
//
// Frame advantage is exact by construction: for a hit whose first active
// frame is f in a move lasting T frames, the defender's stun is
// (T - f) + advantage. Contact on frame f leaves both recovering `advantage`
// frames apart, and a later (meaty) contact adds the difference. Per-move
// geometry (reach, tracking, pushback, travel, active frames) is data: the
// engine is exact for whatever measured values it is given. Engine-wide
// constants in FullEngineConfig that Tekken does not publish are marked as
// provisional until measured.

enum class FullHitLevel : std::uint8_t { High, Mid, Low, SpecialMid, SpecialLow, Throw };

// What a clean (non-blocked) hit does beyond stun, from the frame-data suffix.
enum class FullHitEffect : std::uint8_t {
    Stun,       // plain frame advantage
    Launch,     // "a": airborne, juggle
    Knockdown,  // "d": grounded
    Crumple,    // long stun that ends grounded
};

enum class FullPosture : std::uint8_t {
    Standing,
    Crouching,
    Airborne,
    Grounded,
    Wakeup,
    WallSplat,
};

// One or more inclusive, 1-based move-frame spans (a move can, for example,
// float twice).
struct FrameWindow {
    struct Span {
        int first = 0;
        int last = -1;
    };
    std::vector<Span> spans;

    FrameWindow() = default;
    FrameWindow(int first, int last) : spans{{first, last}} {}
    [[nodiscard]] bool contains(int frame) const noexcept {
        for (const auto& span : spans) {
            if (frame >= span.first && frame <= span.last) return true;
        }
        return false;
    }
    [[nodiscard]] bool empty() const noexcept { return spans.empty(); }
};

struct FullHitSpec {
    FullHitLevel level = FullHitLevel::Mid;
    int first_active_frame = 1;
    int active_frames = 1;
    double damage = 0.0;
    double reach = 1.0;            // max center-to-center distance at which it connects
    double tracking_left = 0.0;    // tolerated lateral (axis) offset to each side
    double tracking_right = 0.0;
    bool homing = false;
    bool ground_hit = false;       // connects on grounded opponents regardless of level
    bool wall_splat = false;
    bool wall_break = false;
    bool floor_break = false;
    bool balcony_break = false;
    bool tornado = false;
    bool throw_breakable = true;   // throws only
};

struct FullMoveSpec {
    std::string stable_id;
    std::vector<FullHitSpec> hits;  // ordered by first_active_frame
    int recovery = 0;               // frames after the last active frame
    int block_advantage = 0;
    int hit_advantage = 0;
    FullHitEffect hit_effect = FullHitEffect::Stun;
    std::optional<int> counter_hit_advantage;
    std::optional<FullHitEffect> counter_hit_effect;
    double travel = 0.0;            // forward distance covered before the first active frame
    double pushback_hit = 0.15;
    double pushback_block = 0.25;
    FrameWindow high_crush{};
    FrameWindow low_crush{};
    FrameWindow power_crush{};      // absorbs high/mid strikes, keeps attacking
    FrameWindow parry{};            // stops high/mid strikes
    FrameWindow invincible{};
    FrameWindow airborne{};         // "floating state": hit as airborne
    FullPosture required_posture = FullPosture::Standing;
    int required_stance = -1;
    int result_stance = -1;         // stance entered when the move ends
    bool requires_heat = false;
    bool requires_rage = false;
    bool engages_heat = false;
    bool consumes_rage = false;

    [[nodiscard]] int total_frames() const noexcept;
};

enum class FullUniversal : std::uint8_t {
    Idle,          // standing guard
    WalkForward,
    WalkBack,      // standing guard while moving
    DashForward,
    Backdash,
    Crouch,        // crouching guard
    SidestepLeft,
    SidestepRight,
    ThrowBreak,    // breaks a throw that connects during this action
    WakeupStand,   // from grounded
    WakeupRoll,    // from grounded, rolls back
    TechRoll,      // on landing from a juggle, if within the tech window
};

struct FullInput {
    std::optional<std::size_t> move;  // index into the fighter's move list
    FullUniversal universal = FullUniversal::Idle;
};

struct FullEngineConfig {
    double half_width = 3.6;
    double body_radius = 0.22;
    double walk_speed = 0.025;        // per frame
    double dash_distance = 0.7;
    int dash_frames = 18;
    double backdash_distance = 0.6;
    int backdash_frames = 22;
    double sidestep_distance = 0.5;   // lateral axis offset
    int sidestep_frames = 18;
    double axis_realign_per_frame = 0.03;
    // Provisional (not published by Tekken; to be measured in Practice mode):
    int launch_air_frames = 60;
    int juggle_refresh_frames = 34;
    double juggle_refresh_decay = 0.85;
    int tornado_extra_frames = 40;
    int knockdown_frames = 24;        // grounded frames before wake-up input
    int wakeup_stand_frames = 30;
    int wakeup_roll_frames = 24;
    double wakeup_roll_distance = 0.8;
    int tech_window_frames = 6;       // after landing
    int tech_roll_frames = 20;
    int wall_splat_frames = 40;
    int crumple_frames = 70;
    int stage_break_air_frames = 55;
    double wall_break_extension = 3.6;
    double combo_scale_step = 0.10;
    double combo_scale_floor = 0.30;
    double counter_hit_damage_scale = 1.2;
    int throw_break_window = 20;      // frames a ThrowBreak action stays armed
    int parry_stun_frames = 40;       // attacker stun after being parried
    double max_health = 180.0;
};

struct FullEngineStage {
    double left_wall = -3.6;
    double right_wall = 3.6;
    bool left_wall_breakable = true;
    bool right_wall_breakable = true;
    bool left_balcony = false;
    bool right_balcony = false;
    bool floor_breakable = true;
    int level = 0;  // increases on floor or balcony break
};

struct FullFighter {
    double health = 180.0;
    double recoverable_health = 0.0;
    double x = 0.0;
    double axis = 0.0;
    FullPosture posture = FullPosture::Standing;
    int stance = -1;
    bool heat = false;
    bool rage = false;
    // Current action: a move (index) or a universal action, and its frame.
    std::optional<std::size_t> move;
    FullUniversal universal = FullUniversal::Idle;
    int action_frame = 0;
    std::uint32_t hits_landed = 0;    // bitmask of this move's hits already resolved
    int stun = 0;                     // hitstun/blockstun frames remaining
    bool in_blockstun = false;        // guard holds through blockstun, not hitstun
    int posture_frames = 0;           // frames left in airborne/grounded/wake-up/splat
    int grounded_elapsed = 0;
    int combo_hits = 0;
    bool tornado_used = false;
    bool wall_splat_used = false;
    bool techable = false;
    bool crumpled = false;            // goes grounded when the crumple stun ends
};

struct FullEngineState {
    std::array<FullFighter, 2> fighters{};
    FullEngineStage stage{};
    int frame = 0;
};

enum class FullContact : std::uint8_t {
    Hit, CounterHit, Blocked, Crushed, Parried, Armored, Evaded, ThrowBroken,
};

enum class FullTransition : std::uint8_t { WallBreak, FloorBreak, BalconyBreak };

struct FullContactEvent {
    int attacker = 0;
    std::size_t hit_index = 0;
    FullContact contact = FullContact::Hit;
    double damage = 0.0;
};

struct FullFrameEvents {
    std::vector<FullContactEvent> contacts;
    std::vector<FullTransition> transitions;
};

class FullCombatEngine {
public:
    FullCombatEngine(std::array<std::vector<FullMoveSpec>, 2> movesets, FullEngineConfig config = {});

    void reset(double p1_x = -0.8, double p2_x = 0.8);
    [[nodiscard]] const FullEngineState& state() const noexcept { return state_; }
    [[nodiscard]] FullEngineState& mutable_state() noexcept { return state_; }
    [[nodiscard]] const FullEngineConfig& config() const noexcept { return config_; }
    [[nodiscard]] const FullMoveSpec& move(int player, std::size_t index) const;

    // True when the fighter can start a new action this frame.
    [[nodiscard]] bool actionable(int player) const;
    [[nodiscard]] bool legal(int player, const FullInput& input) const;
    // Advances one frame. Inputs are ignored for fighters that are not
    // actionable (except ThrowBreak, WakeupStand/Roll, and TechRoll, which
    // are accepted in the states where they apply).
    FullFrameEvents step(const FullInput& p1, const FullInput& p2);

private:
    std::array<std::vector<FullMoveSpec>, 2> movesets_;
    FullEngineConfig config_;
    FullEngineState state_{};
};

[[nodiscard]] std::string_view to_string(FullContact contact) noexcept;

}  // namespace t8::v2
