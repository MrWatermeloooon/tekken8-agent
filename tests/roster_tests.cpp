#include "t8_v2/roster.hpp"

#include <algorithm>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
}  // namespace

int main(int argc, char** argv) {
    require(argc == 3, "expected profile and character-move CSV arguments");
    const auto profiles = t8::v2::load_opponent_profiles_csv(std::filesystem::path(argv[1]));
    const auto moves = t8::v2::load_character_move_specs_csv(std::filesystem::path(argv[2]));
    require(profiles.size() == t8::v2::kOpponentProfileCount, "profile count mismatch");
    require(profiles.front().id == 0, "first profile ID mismatch");
    require(profiles.back().id == 2099, "last profile ID mismatch");
    require(moves.size() == t8::v2::kRosterCharacterCount * t8::v2::kCharacterMoveSlotCount,
            "character move count mismatch");
    require(moves[2].startup != moves[8].startup || moves[2].damage != moves[8].damage,
            "character-specific power mids were collapsed to identical data");
    for (std::size_t character = 0; character < t8::v2::kRosterCharacterCount; ++character) {
        const auto count = std::count_if(profiles.begin(), profiles.end(), [character](const auto& profile) {
            return profile.character_id == character;
        });
        require(count == static_cast<std::ptrdiff_t>(t8::v2::kProfilesPerCharacter),
                "per-character profile count mismatch");
    }
    t8::v2::MatchupScheduler scheduler(profiles, 99);
    const auto fundamentals = scheduler.eligible_profile_indices();
    require(!fundamentals.empty(), "fundamentals curriculum is empty");
    require(std::all_of(fundamentals.begin(), fundamentals.end(), [&](const auto index) {
        return (profiles[index].group_mask & t8::v2::CharacterGroup::Fundamentals) != 0 &&
            profiles[index].archetype_id <= 4 && profiles[index].variation_id <= 1;
    }), "fundamentals curriculum admitted an ineligible profile");
    bool rejected = false;
    try {
        scheduler.set_stage(t8::v2::CurriculumStage::CharacterGroups);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    require(rejected, "empty character-group curriculum was accepted");
    scheduler.set_stage(t8::v2::CurriculumStage::CharacterGroups, t8::v2::CharacterGroup::Grappler);
    const auto grapplers = scheduler.eligible_profile_indices();
    require(std::all_of(
        grapplers.begin(), grapplers.end(),
        [&](const auto index) { return (profiles[index].group_mask & t8::v2::CharacterGroup::Grappler) != 0; }),
        "grappler curriculum admitted another group");
    scheduler.set_stage(t8::v2::CurriculumStage::FullRoster);
    const auto assignments = scheduler.sample_profile_indices(32, true);
    require(assignments.size() == 32, "assignment count mismatch");
    for (std::size_t lane = 0; lane < 8; ++lane) {
        require(assignments[lane] == assignments[8 + lane], "first learner-side pair mismatch");
        require(assignments[16 + lane] == assignments[24 + lane], "second learner-side pair mismatch");
    }
    scheduler.record(0, 2, 8);
    scheduler.record(1, 98, 2);
    require(scheduler.priority(0) > scheduler.priority(1), "weak matchup was not prioritized");
    std::cout << "roster profiles=" << profiles.size() << " moves=" << moves.size()
              << " scheduler=ok\n";
}
