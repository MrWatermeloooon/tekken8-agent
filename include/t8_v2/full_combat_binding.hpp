#pragma once

#include "t8_v2/full_combat_engine.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace t8::v2 {

// One row of data/generated/full_combat_bindings.csv
// (tools/export_full_combat_bindings.py), kept as source text so every
// binding decision can be traced back to it.
struct FullBindingRow {
    std::string stable_id;
    std::string character;
    std::string command;
    std::string parser_status;
    std::string source_consistency;
    bool reactive = false;
    std::vector<std::string> hit_levels;
    std::vector<double> damages;
    std::optional<int> startup;
    std::optional<int> recovery;
    std::optional<int> block_advantage;
    std::optional<int> hit_advantage;
    std::string hit_effect;
    std::optional<int> counter_hit_advantage;
    std::string counter_hit_effect;
    bool frame_data_variable = false;
    bool homing = false;
    bool tornado = false;
    bool wall_break = false;
    bool floor_break = false;
    bool balcony_break = false;
    bool heat_engager = false;
    bool requires_heat = false;
    bool requires_rage = false;
    bool rage_art = false;
    std::string posture;
    std::vector<std::string> stances;
    std::vector<std::string> situations;
    std::string automatic_transition;
    std::string low_crush;
    std::string high_crush;
    std::string power_crush;
    std::string parry;
    std::string airborne;
    std::string invincible;
    bool heat_burst = false;
    bool heat_smash = false;
    std::optional<int> heat_dash_block;
    std::optional<int> heat_dash_hit;
    std::string heat_dash_effect;
    // Amounts from the notes: "" absent, "?" stated without a number, else
    // one value or one value per hit ("2|7").
    std::string chip_block;
    std::string chip_block_heat;
    bool recoverable_only = false;
    bool removes_recoverable = false;
    bool armor_damage_recoverable = false;
    std::string self_damage;
    std::string self_recoverable;
    bool self_damage_without_heat = false;
    std::string restore_health_hit;
    std::string restore_recoverable_hit;
    std::string restore_recoverable_block;
    std::string throw_break;  // "1", "2", "1+2", "1|2", "none", "?", or "" (not stated)
    bool side_switch_on_hit = false;
    bool side_switch_on_break = false;
    bool spike = false;
    bool unparryable = false;
    bool reversal_break = false;
    std::string variant_of;           // an optional stance branch of this base move
    bool result_crouching = false;
    std::string result_stance;
    std::string result_stance_on_hit;
    std::string result_stance_on_block;
    bool requires_sidestep = false;
    bool requires_running = false;
    std::string parry_levels;         // "high|mid", "?" when not stated, "" without a parry
    std::vector<std::string> parry_outcomes;  // stable ids by level class (high, mid, low, throw)
    int heat_cost_frames = 0;
    double resource_gain_start = 0.0;
    double resource_gain_hit = 0.0;
    double resource_gain_airborne_hit = 0.0;
    double resource_gain_block = 0.0;
    double resource_gain_heat_activation = 0.0;
    double install_damage_bonus = 0.0;
    std::optional<double> install_chip;
    std::optional<double> install_range;  // in the same units as measured_range
    std::string attack_throw;         // "hit", "counter_hit", or ""
    bool attack_throw_front_only = false;
    bool attack_throw_standing_only = false;
    bool attack_throw_airborne = false;
    std::optional<double> attack_throw_damage;
    std::optional<int> back_turned_hit_advantage;
    std::string back_turned_hit_effect;
    bool result_back_turned = false;
    std::string heat_parry;
    std::string heat_parry_levels;
    std::string heat_parry_outcome;
    double recoverable_damage = 0.0;
    bool cannot_ko = false;
    std::optional<double> rage_art_max_damage;
    bool ki_charge = false;
    std::string practice_status;
    std::vector<int> measured_active_frames;
    std::optional<double> measured_range;
    std::optional<double> measured_tracking_left;
    std::optional<double> measured_tracking_right;
    std::optional<double> measured_pushback;
    std::optional<double> measured_travel;
    std::vector<int> measured_hit_frames;
};

// Stand-in geometry for moves that have not been measured yet. Only for
// engine smoke tests and frame-data checks: a move bound with it is marked
// provisional and must never be used for training.
struct FullProvisionalGeometry {
    int active_frames = 2;
    double range = 1.2;
    double tracking = 0.1;
    double pushback = 0.2;
    double travel = 0.0;
    int string_gap = 12;  // frames between successive hits of a string
};

struct FullBindingOptions {
    bool require_practice_validation = true;
    bool allow_variable_frame_data = false;  // use the minimum of a range
    std::optional<FullProvisionalGeometry> provisional;
    std::vector<std::string> stance_names;   // stance id = index in this list
};

struct FullBoundMove {
    std::string stable_id;
    std::string command;
    std::vector<std::string> blockers;  // empty = bound
    bool provisional = false;
    std::optional<FullMoveSpec> spec;
    [[nodiscard]] bool bound() const noexcept { return blockers.empty() && spec.has_value(); }
};

[[nodiscard]] std::vector<FullBindingRow> load_full_combat_bindings(
    const std::filesystem::path& path,
    const std::string& character = {});

// Binds catalog rows to engine moves. Every reason a row cannot be simulated
// exactly is reported as a blocker; nothing is estimated unless
// options.provisional is set, and then the move is marked provisional.
[[nodiscard]] std::vector<FullBoundMove> bind_full_combat_moves(
    const std::vector<FullBindingRow>& rows,
    const FullBindingOptions& options = {});

[[nodiscard]] FrameWindow parse_frame_window(const std::string& text);

// Stance and resource rules exported next to the bindings
// (full_combat_stances.csv, full_combat_resources.csv). Stance ids are the
// row order; pass the names as FullBindingOptions::stance_names.
struct FullCharacterRuleSet {
    FullCharacterRules rules;
    std::vector<std::string> stance_names;
};
[[nodiscard]] FullCharacterRuleSet load_full_combat_character_rules(
    const std::filesystem::path& stances_csv,
    const std::filesystem::path& resources_csv,
    const std::string& character);

}  // namespace t8::v2
