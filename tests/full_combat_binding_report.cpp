// Per-character binding report: how many engine rows bind (strictly and with
// stand-in geometry) and what blocks the rest, grouped by blocker kind.
//
//   t8_v2_full_combat_binding_report data/generated [--json]

#include "t8_v2/full_combat_binding.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace t8::v2;
    if (argc < 2) {
        std::cerr << "usage: full_combat_binding_report <data/generated> [--json]\n";
        return 2;
    }
    const std::filesystem::path generated = argv[1];
    const bool json = argc > 2 && std::string(argv[2]) == "--json";
    try {
        const auto rows = load_full_combat_bindings(generated / "full_combat_bindings.csv", "");
        std::map<std::string, std::vector<FullBindingRow>> by_character;
        for (const auto& row : rows) by_character[row.character].push_back(row);
        if (json) std::cout << "{";
        bool first = true;
        for (const auto& [character, character_rows] : by_character) {
            FullBindingOptions strict{}, provisional{};
            provisional.require_practice_validation = false;
            provisional.allow_variable_frame_data = true;
            provisional.provisional = FullProvisionalGeometry{};
            try {
                const auto set = load_full_combat_character_rules(generated / "full_combat_stances.csv",
                                                                  generated / "full_combat_resources.csv", character);
                strict.stance_names = provisional.stance_names = set.stance_names;
            } catch (const std::exception&) {
            }
            const auto strict_bound = bind_full_combat_moves(character_rows, strict);
            const auto provisional_bound = bind_full_combat_moves(character_rows, provisional);
            std::size_t strict_count = 0, provisional_count = 0;
            std::map<std::string, std::size_t> blockers;
            for (const auto& move : strict_bound) strict_count += move.bound();
            for (const auto& move : provisional_bound) {
                provisional_count += move.bound();
                for (const auto& blocker : move.blockers) {
                    // Group "unknown stance: XYZ" and similar by their kind.
                    const auto colon = blocker.find(':');
                    ++blockers[colon == std::string::npos ? blocker : blocker.substr(0, colon)];
                }
            }
            std::vector<std::pair<std::string, std::size_t>> ranked(blockers.begin(), blockers.end());
            std::sort(ranked.begin(), ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
            if (json) {
                std::cout << (first ? "" : ",") << '"' << character << "\":{\"rows\":" << character_rows.size()
                          << ",\"strict\":" << strict_count << ",\"provisional\":" << provisional_count
                          << ",\"stances\":" << provisional.stance_names.size() << ",\"blockers\":{";
                for (std::size_t index = 0; index < ranked.size(); ++index) {
                    std::cout << (index ? "," : "") << '"' << ranked[index].first << "\":" << ranked[index].second;
                }
                std::cout << "}}";
            } else {
                std::cout << character << ": " << provisional_count << '/' << character_rows.size()
                          << " bind with stand-in geometry (" << strict_count << " strictly), "
                          << provisional.stance_names.size() << " stance rules";
                for (std::size_t index = 0; index < std::min<std::size_t>(3, ranked.size()); ++index) {
                    std::cout << (index ? "; " : " | ") << ranked[index].first << " x" << ranked[index].second;
                }
                std::cout << '\n';
            }
            first = false;
        }
        if (json) std::cout << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "binding report error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
