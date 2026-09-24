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
//
// Resources follow the published per-move data where it exists: Heat (Heat
// Burst, Heat Engagers, Heat Dash, Heat Smash, the Heat timer, and chip
// damage), recoverable health (chip, self-damage, restores, removal, and
// recoverable-only damage), and Rage (activation, Rage Arts). Throws resolve
// through a break window in which the defender must press the right button.
//
// Character-specific state comes from FullCharacterRules: stances (guarding,
// time limits, automatic parries and their outcomes, periodic pulses) and a
// resource gauge with an install threshold (Jun: Kazama Essence and Divine
// Aura). Moves enter stances when they end, or on hit or block, and may have
// no hits at all (stance entries).

enum class FullHitLevel : std::uint8_t { High, Mid, Low, SpecialMid, SpecialLow, Throw };

// What a clean (non-blocked) hit does beyond stun, from the frame-data suffix.
enum class FullHitEffect : std::uint8_t {
    Stun,       // plain frame advantage
    Launch,     // "a": airborne, juggle
    Knockdown,  // "d": grounded
    Crumple,    // long stun that ends grounded
};

// Which throw-break input escapes a throw ("Throw break 1 or 2", "1+2", ...).
enum class FullThrowBreak : std::uint8_t { None, One, Two, OneTwo, OneOrTwo };

// Level classes a parry or automatic parry covers (bitmask).
enum FullLevelClass : std::uint8_t {
    FullLevelHigh = 1U << 0U,
    FullLevelMid = 1U << 1U,
    FullLevelLow = 1U << 2U,
    FullLevelThrow = 1U << 3U,
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
    bool spike = false;            // an airborne defender is slammed down, untechable
    bool unparryable = false;
    bool reversal_break = false;   // cannot be reversed
    double chip_damage = 0.0;      // recoverable damage dealt on block
    std::optional<double> heat_chip_damage;  // chip on block while the attacker is in Heat
    FullThrowBreak throw_break = FullThrowBreak::OneTwo;  // throws only
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
    FrameWindow armor{};            // absorbs every strike, lows included
    FrameWindow parry{};            // stops the levels in parry_levels
    std::uint8_t parry_levels = FullLevelHigh | FullLevelMid;
    // Reactive move played when this move's parry succeeds, by level class
    // (high, mid, low, throw); empty = a plain parry.
    std::array<std::string, 4> parry_outcomes{};
    FrameWindow reversal{};         // catches high/mid strikes and counters
    double reversal_damage = 0.0;
    FrameWindow invincible{};
    FrameWindow airborne{};         // "floating state": hit as airborne
    // Airtime of a launch; by default the published advantage ("+29a": airborne until the
    // attacker has recovered for 29 frames).
    std::optional<int> launch_air_frames;
    FullPosture required_posture = FullPosture::Standing;
    int required_stance = -1;
    int result_stance = -1;         // stance entered when the move ends
    int result_stance_on_hit = -1;  // replaces result_stance if the move hit
    int result_stance_on_block = -1;  // ... or was blocked
    bool result_crouching = false;  // ends crouched ("r25 FC")
    bool requires_sidestep = false; // "SS." moves: performed out of a sidestep
    bool reactive = false;          // a parry outcome: played by the engine, never input
    // Situational requirements.
    bool requires_back_to_wall = false;         // "(Back to wall)."
    bool requires_back_turned = false;          // "BT." moves: from the user's own back-turned state
    bool requires_opponent_back_turned = false; // back throws
    bool requires_opponent_left_side = false;   // left throws: the opponent's left side is exposed
    bool requires_opponent_right_side = false;
    bool result_back_turned = false;            // ends back-turned ("r20 BT")
    // Hitting a back-turned opponent ("Hit vs BT +10a (+1)").
    std::optional<int> back_turned_hit_advantage;
    FullHitEffect back_turned_hit_effect = FullHitEffect::Stun;
    // Attack throw: after the last strike lands cleanly, an unbreakable throw follows.
    enum class AttackThrow : std::uint8_t { None, OnHit, OnCounterHit };
    AttackThrow attack_throw = AttackThrow::None;
    bool attack_throw_front_only = false;     // not against a back-turned opponent
    bool attack_throw_standing_only = false;  // not against a crouching opponent (airborne allowed if listed)
    bool attack_throw_airborne = true;        // also against an airborne opponent
    double attack_throw_damage = 0.0;         // added by the throw
    // Heat-only parry ("Power up in Heat (ps5~12)") and the outcome it plays.
    FrameWindow heat_parry{};
    std::uint8_t heat_parry_levels = FullLevelHigh | FullLevelMid;
    std::string heat_parry_outcome;
    bool cannot_ko = false;                   // "Cannot cause a K.O."
    double recoverable_damage = 0.0;          // part of the damage that is recoverable ("Deals 5 recoverable damage")
    std::optional<double> rage_art_max_damage;  // "Damage increases with lower health, maximum 82"
    bool ki_charge = false;                   // 1+2+3+4
    bool requires_heat = false;     // Heat-enhanced version
    bool requires_rage = false;
    bool engages_heat = false;      // Heat Burst: activates Heat when started
    bool heat_engager = false;      // activates Heat on hit or block; enables Heat Dash
    bool consumes_heat = false;     // Heat Smash: spends the remaining Heat
    bool consumes_rage = false;     // Rage Art
    // Heat Dash from this move after contact (spends the remaining Heat).
    std::optional<int> heat_dash_block_advantage;
    std::optional<int> heat_dash_hit_advantage;
    FullHitEffect heat_dash_hit_effect = FullHitEffect::Stun;
    // Recoverable health.
    bool recoverable_only = false;  // damage is recoverable and cannot K.O.
    bool removes_recoverable = false;
    bool armor_damage_recoverable = false;  // damage absorbed by power crush/armor is recoverable
    double self_damage = 0.0;       // dealt to the user when the move starts
    double self_recoverable = 0.0;  // part of self_damage that is recoverable
    bool self_damage_without_heat_only = false;
    double restore_health_hit = 0.0;
    double restore_recoverable_hit = 0.0;
    double restore_recoverable_block = 0.0;
    bool side_switch_on_hit = false;
    bool side_switch_on_break = false;
    int heat_cost_frames = 0;       // Heat time spent when started ("Consumes 450f of heat timer")
    // Character resource gauge (Jun: Kazama Essence), gained once per move.
    double resource_gain_start = 0.0;
    double resource_gain_hit = 0.0;
    double resource_gain_airborne_hit = 0.0;
    double resource_gain_block = 0.0;
    double resource_gain_heat_activation = 0.0;
    // While the install is active (Jun: Divine Aura).
    double install_damage_bonus = 0.0;          // added to the last hit
    std::optional<double> install_chip_damage;  // replaces the last hit's chip
    std::optional<double> install_reach;        // replaces every hit's reach

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
    ThrowBreak,    // 1+2 during a throw's break window
    ThrowBreak1,   // 1 during a throw's break window
    ThrowBreak2,   // 2 during a throw's break window
    HeatDash,      // cancels a Heat Engager after contact
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
    int launch_air_frames = 60;       // only when a launch's published advantage gives no airtime
    // A juggle hit keeps the opponent airborne until the attacker recovers plus this
    // window (shrinking by the decay per combo hit), which is when the next move must start.
    // Bounds: Jun's guide route "DF2: 4 > 1,1 > IZU 1,1 T! > (F1) IZU 1,1" ends with an i13 move
    // on the 8th hit, so the window must still be at least 13 there; the decay (gravity) stops
    // endless juggles, since fast moves (i10) no longer fit after about a dozen hits.
    int juggle_window_frames = 22;
    int juggle_window_decay = 1;
    int tornado_extra_frames = 40;
    int knockdown_frames = 24;        // grounded frames before wake-up input
    int wakeup_stand_frames = 30;
    int wakeup_roll_frames = 24;
    double wakeup_roll_distance = 0.8;
    int tech_window_frames = 6;       // after landing
    int tech_roll_frames = 20;
    int wall_splat_frames = 20;       // held at the wall until the attacker recovers plus this window
    int crumple_frames = 70;
    int stage_break_air_frames = 55;
    double wall_break_extension = 3.6;
    double combo_scale_step = 0.10;
    double combo_scale_floor = 0.30;
    double counter_hit_damage_scale = 1.2;
    int throw_break_window = 20;      // frames after a throw connects in which it can be broken
    int throw_break_recovery = 20;    // both fighters, after a break (neutral)
    int parry_outcome_frames = 30;    // a parry outcome's recovery when the data gives none
    double side_exposure_ratio = 0.5; // lateral offset / distance at which a side is exposed (side throws)
    double back_to_wall_distance = 0.3;  // "back to wall": the wall behind is at most this far
    // Published in the Ki Charge notes: 5 seconds, +10% on the next attack, hits taken are counter-hits.
    int ki_charge_frames = 300;
    double ki_charge_damage_scale = 1.10;
    int parry_stun_frames = 40;       // attacker stun after being parried
    int reversal_recovery = 30;       // the reversing fighter's recovery
    int heat_duration_frames = 600;   // Heat timer
    int heat_dash_frames = 20;
    double heat_dash_distance = 1.2;
    double rage_health_fraction = 0.25;
    double rage_damage_scale = 1.10;
    double wall_combo_scale = 0.70;   // extra scaling after a wall splat
    double recoverable_regain_hit = 0.50;    // recoverable health regained per damage dealt on hit
    double recoverable_regain_block = 0.25;  // ... and on block
    double max_health = 180.0;
};

struct FullStanceRule {
    std::string name;
    bool can_guard = true;
    std::optional<int> max_frames;       // returns to neutral after this long
    std::uint8_t auto_parry = 0;         // level classes parried while idle in the stance
    std::array<std::string, 4> parry_outcomes{};  // stable ids by level class
    int pulse_interval_frames = 0;       // periodic effect while in the stance
    double pulse_recoverable = 0.0;
    double pulse_resource = 0.0;
};

struct FullCharacterRules {
    std::vector<FullStanceRule> stances;  // stance id = index
    double resource_max = 0.0;
    std::optional<double> install_threshold;
    bool resource_persists = false;       // carried into the next round
    bool install_consumes = false;
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

enum class FullContact : std::uint8_t {
    Hit, CounterHit, Blocked, Crushed, Parried, Armored, Evaded, ThrowBroken,
    Reversed,
    ThrowHeld,  // a throw connected and its break window is open
};

struct FullFighter {
    double health = 180.0;
    double recoverable_health = 0.0;
    double x = 0.0;
    double axis = 0.0;
    FullPosture posture = FullPosture::Standing;
    int stance = -1;
    bool heat = false;
    bool heat_available = true;       // one Heat per round
    int heat_frames = 0;              // Heat time remaining
    bool rage = false;
    bool rage_available = true;       // one Rage per round
    // Heat Dash is possible from the current move: which contact enabled it.
    std::optional<FullContact> heat_dash_from;
    // Throws: frames left to break a throw holding this fighter and whether
    // the one break attempt is spent; on the thrower, the pending hit.
    int throw_break_frames = 0;
    bool throw_break_attempted = false;
    int pending_throw_hit = -1;
    int pending_throw_elapsed = 0;
    // Current action: a move (index) or a universal action, and its frame.
    std::optional<std::size_t> move;
    FullUniversal universal = FullUniversal::Idle;
    int action_frame = 0;
    std::uint32_t hits_landed = 0;    // bitmask of this move's hits already resolved
    int stance_frames = 0;            // frames idle in the current stance
    double resource = 0.0;            // character gauge (Jun: Kazama Essence)
    bool installed = false;           // gauge install active (Jun: Divine Aura)
    bool move_hit = false;            // the current move hit (for on-hit transitions)
    bool move_blocked = false;
    bool resource_gained = false;     // this move's resource gain already applied
    bool back_turned = false;
    bool move_in_heat = false;        // Heat was active when the current move started
    int ki_charge_frames = 0;         // Ki Charge: no guard, hits taken are counter-hits
    bool ki_charge_boost = false;     // the next attack deals more damage
    bool move_boosted = false;        // the current move carries the Ki Charge boost
    int stun = 0;                     // hitstun/blockstun frames remaining
    bool in_blockstun = false;        // guard holds through blockstun, not hitstun
    int posture_frames = 0;           // frames left in airborne/grounded/wake-up/splat
    int grounded_elapsed = 0;
    int combo_hits = 0;
    bool tornado_used = false;
    bool wall_splat_used = false;
    bool wall_scaled = false;         // the combo went through a wall splat
    bool techable = false;
    bool crumpled = false;            // goes grounded when the crumple stun ends
};

struct FullEngineState {
    std::array<FullFighter, 2> fighters{};
    FullEngineStage stage{};
    int frame = 0;
    int winner = -1;  // -1 fighting, 0 or 1, 2 for a double K.O.
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
    std::array<bool, 2> heat_activated{};
    std::array<bool, 2> rage_activated{};
};

class FullCombatEngine {
public:
    FullCombatEngine(std::array<std::vector<FullMoveSpec>, 2> movesets, FullEngineConfig config = {},
                     std::array<FullCharacterRules, 2> rules = {});

    void reset(double p1_x = -0.8, double p2_x = 0.8);
    // Starts the next round: like reset, but resources that persist carry over.
    void next_round(double p1_x = -0.8, double p2_x = 0.8);
    [[nodiscard]] const FullCharacterRules& rules(int player) const { return rules_.at(static_cast<std::size_t>(player)); }
    [[nodiscard]] const FullEngineState& state() const noexcept { return state_; }
    [[nodiscard]] FullEngineState& mutable_state() noexcept { return state_; }
    [[nodiscard]] const FullEngineConfig& config() const noexcept { return config_; }
    [[nodiscard]] const FullMoveSpec& move(int player, std::size_t index) const;

    // True when the fighter can start a new action this frame.
    [[nodiscard]] bool actionable(int player) const;
    [[nodiscard]] bool legal(int player, const FullInput& input) const;
    // Advances one frame. Inputs are ignored for fighters that are not
    // actionable (except throw breaks, Heat Dash, WakeupStand/Roll, and
    // TechRoll, which are accepted in the states where they apply). After a
    // K.O. the state no longer changes.
    FullFrameEvents step(const FullInput& p1, const FullInput& p2);

private:
    // move_index and defense_move are the moves at the time the contact was
    // decided, so a same-frame trade applies both hits.
    void apply_contact(int attacker, std::size_t move_index, std::optional<std::size_t> defense_move,
                       std::size_t hit_index, FullContact contact, int elapsed, FullFrameEvents& events);
    void deal_damage(int target, double damage, bool recoverable, bool can_ko, FullFrameEvents& events);
    void activate_heat(int player, FullFrameEvents& events);
    void break_throw(int thrower, FullFrameEvents& events);
    void release_throw(int thrower);
    void gain_resource(int player, double amount);
    void apply_parry_outcome(int parrier, std::size_t outcome, int attacker, FullFrameEvents& events);
    [[nodiscard]] const FullStanceRule* stance_rule(int player) const;
    [[nodiscard]] std::optional<std::size_t> find_move(int player, const std::string& stable_id) const;

    std::array<std::vector<FullMoveSpec>, 2> movesets_;
    FullEngineConfig config_;
    std::array<FullCharacterRules, 2> rules_;
    FullEngineState state_{};
};

[[nodiscard]] std::string_view to_string(FullContact contact) noexcept;
[[nodiscard]] FullLevelClass level_class(FullHitLevel level) noexcept;
[[nodiscard]] std::size_t level_class_index(FullHitLevel level) noexcept;  // high, mid, low, throw

}  // namespace t8::v2
