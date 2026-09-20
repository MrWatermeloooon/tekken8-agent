#include "t8_v2/roster.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <stdexcept>

namespace {

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 2, "expected the generated full move catalog path");
        const auto catalog = t8::v2::load_full_move_catalog_csv(std::filesystem::path(argv[1]));
        require(catalog.schema_version == t8::v2::kFullMoveCatalogSchemaVersion,
                "full move schema version mismatch");
        require(catalog.moves.size() == 6393, "full move row count mismatch");
        require(catalog.moves_for_character(t8::v2::kJunCharacterId).size() == 149,
                "Jun move count mismatch");
        require(catalog.moves_for_character(41).empty(), "Bob must remain blocked without source data");
        require(catalog.candidate_count(t8::v2::kJunCharacterId) == 167,
                "Jun candidate count must include universal actions");
        const auto jun_features = catalog.action_features_for_character(t8::v2::kJunCharacterId);
        require(jun_features.size() ==
                    catalog.candidate_count(t8::v2::kJunCharacterId) * t8::v2::kMoveActionFeatureSize,
                "Jun candidate feature tensor has the wrong shape");
        bool universal_actions_differ = false;
        for (std::size_t feature = 0; feature < t8::v2::kMoveActionFeatureSize; ++feature) {
            universal_actions_differ = universal_actions_differ ||
                jun_features[feature] != jun_features[t8::v2::kMoveActionFeatureSize + feature];
        }
        require(universal_actions_differ, "universal actions need distinct feature identities");
        for (std::uint32_t character = 0; character < t8::v2::kRosterCharacterCount; ++character) {
            const auto moves = catalog.moves_for_character(character);
            require(moves.size() <= t8::v2::kMaxCharacterMoveCount,
                    "character exceeds candidate bound");
            for (std::size_t local = 0; local < moves.size(); ++local) {
                require(moves[local].character_id == character, "character range is not contiguous");
                require(moves[local].local_id == local, "local move IDs are not contiguous");
                for (float feature : moves[local].action_features) {
                    require(std::isfinite(feature), "move feature is non-finite");
                }
            }
        }
        std::cout << "full move catalog tests passed\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
