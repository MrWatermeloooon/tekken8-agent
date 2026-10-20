#pragma once

// Characters prepared for full-move simulation and training: the policy
// candidate list (data/generated/full_move_candidates.csv, shared with the
// Python runtime), the engine moveset bound from the full-combat binding rows,
// and the mapping from each candidate to an engine input.

#include "t8_v2/full_combat_binding.hpp"
#include "t8_v2/full_combat_core.hpp"
#include "t8_v2/roster.hpp"

#include <array>
#include <filesystem>
#include <string>
#include <vector>

namespace t8::v2 {

constexpr int kFullMoveUniversalCount = 22;
constexpr int kFullMoveMaxCandidates = 368;
constexpr int kFullMoveFeatureSize = 32;

// The 22 universal candidates in contract order (t8_agent.moves.candidates).
extern const std::array<const char*, kFullMoveUniversalCount> kFullMoveUniversalNames;

namespace full_core {

// Engine input of a universal candidate; false when the engine has no such
// mechanic (jump, sidewalks, separate block buttons, a universal low parry),
// so the candidate stays masked rather than becoming a silent no-op.
T8_FULL_HD inline bool universal_candidate_input(int candidate, Input& input) {
    input.move = -1;
    switch (candidate) {
        case 0: input.universal = FullUniversal::Idle; return true;          // neutral (standing guard)
        case 1: input.universal = FullUniversal::WalkForward; return true;
        case 2: input.universal = FullUniversal::WalkBack; return true;
        case 3: input.universal = FullUniversal::DashForward; return true;
        case 4: input.universal = FullUniversal::Backdash; return true;
        case 5: input.universal = FullUniversal::Crouch; return true;
        case 6: input.universal = FullUniversal::Idle; return true;          // stand (legal only when crouching)
        case 8: input.universal = FullUniversal::SidestepLeft; return true;
        case 9: input.universal = FullUniversal::SidestepRight; return true;
        case 15: input.universal = FullUniversal::ThrowBreak1; return true;
        case 16: input.universal = FullUniversal::ThrowBreak2; return true;
        case 17: input.universal = FullUniversal::ThrowBreak; return true;
        case 18: input.universal = FullUniversal::HeatDash; return true;
        case 19: input.universal = FullUniversal::WakeupStand; return true;
        case 20: input.universal = FullUniversal::WakeupRoll; return true;
        case 21: input.universal = FullUniversal::TechRoll; return true;
        default: return false;  // 7 jump, 10/11 sidewalks, 12 block_high, 13 block_low, 14 low_parry
    }
}

}  // namespace full_core

struct FullMoveCharacter {
    std::string slug;
    int roster_id = -1;                         // index in kRosterCharacterSlugs (stable across runs)
    FullCompiledTables tables;                  // side 0 holds this character
    std::vector<std::string> candidate_ids;     // stable ids, universal first
    std::vector<int> candidate_moves;           // engine move index per candidate, -1 universal/unbound
    std::vector<float> candidate_features;      // [candidates, kFullMoveFeatureSize]
    int bound_moves = 0;                        // candidates with an engine move
    bool provisional = false;                   // bound with stand-in geometry: never a release result

    [[nodiscard]] int candidate_count() const noexcept { return static_cast<int>(candidate_ids.size()); }
};

struct FullMoveRosterOptions {
    // Stand-in geometry for unmeasured moves (FullProvisionalGeometry). Without
    // it only measured, Practice-validated moves bind, which today is none.
    bool allow_provisional = false;
    FullEngineConfig engine{};
};

// Loads one character from data/generated.
[[nodiscard]] FullMoveCharacter load_full_move_character(const std::filesystem::path& generated,
                                                         const std::string& slug,
                                                         const FullMoveRosterOptions& options);

}  // namespace t8::v2
