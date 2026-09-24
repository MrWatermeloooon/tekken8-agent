#pragma once

#include "t8_v2/full_combat_engine.hpp"

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

namespace t8::v2 {

// Curated combo routes (data/character_modules/<slug>/routes.yaml, exported to
// data/generated/full_combat_routes.csv) executed in the full-combat engine
// with legal-execution assertions. Passing here is a simulator result; the
// character's route gate passes only after Practice-mode confirmation.

struct FullRouteStep {
    std::string input;  // a stable move id, or "@HeatDash"
    int delay = 0;      // frames to wait once the input is legal
};

struct FullRoute {
    std::string character;
    std::string name;
    std::string category;  // midscreen, wall, heat, counter_hit
    std::string source;    // guide or derived
    double distance = 0.8;
    bool near_wall = false;
    // walk (toward, no guard), crouch (holds position; mids hit), attack (performs opponent_move), guard
    std::string opponent = "walk";
    std::string opponent_move;
    bool heat = false;
    double resource = 0.0;
    std::vector<FullRouteStep> steps;
    std::optional<int> expected_hits;
    bool counter_hit_starter = false;
    bool heat_activated = false;
};

struct FullRouteResult {
    std::vector<std::string> failures;  // empty = the route executes as written
    int hits = 0;
    double damage = 0.0;
    int frames = 0;
    std::vector<int> issue_frames;      // frame each step was input
    FullPosture end_posture = FullPosture::Standing;
    std::vector<std::string> trace;     // inputs and contacts, frame by frame, for diagnosis
    [[nodiscard]] bool passed() const noexcept { return failures.empty(); }
};

[[nodiscard]] std::vector<FullRoute> load_full_combat_routes(const std::filesystem::path& path,
                                                             const std::string& character = {});

// Runs a route: each step is input on the first frame it is legal (plus its
// delay). Fails if a step never becomes legal, a hit whiffs or is blocked,
// crushed, parried, or evaded, the starter is not a counter-hit when the
// route requires one, the opponent can act before a later hit lands (not a
// true combo), the hit count differs, or an expected Heat activation is missing.
[[nodiscard]] FullRouteResult run_full_combat_route(const FullRoute& route,
                                                    const std::vector<FullMoveSpec>& moveset,
                                                    const FullCharacterRules& rules = {},
                                                    const FullEngineConfig& config = {},
                                                    int max_wait_frames = 90);

}  // namespace t8::v2
