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

}  // namespace t8::v2
