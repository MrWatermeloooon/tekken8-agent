// Why a character can or cannot use its moves.
//
//   t8_v2_full_move_inspect [--character slug | --all] [--data-dir data/generated] [--verbose] [--blockers]
//                           [--list-unbound]
//
// Loads each character as training does (stand-in geometry allowed) and reports how many of
// its candidates bind to an engine move, and how many are legal from a neutral stand at
// mid range, with the reason for every illegal bound move (required posture, stance, Heat,
// Rage, back turned, sidestep, wall, side). A character the policy cannot play shows up here
// as few legal candidates. --blockers lists why unbound moves did not bind (the binder's
// blockers, counted per kind, with examples under --verbose). --list-unbound prints every move
// that did not bind, one per line: "unbound<TAB>stable id<TAB>command<TAB>blockers; ...".

#include "t8_v2/full_combat_binding.hpp"
#include "t8_v2/full_move_contract.hpp"
#include "t8_v2/full_move_roster.hpp"

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace {

using namespace t8::v2;
using namespace t8::v2::full_core;

std::string reason(const Tables& tables, const State& state, int player, int move) {
    const Fighter& fighter = state.fighters[player];
    const Move& spec = move_of(tables, player, move);
    if (spec.has(kReactive)) return "reactive (a parry outcome)";
    if (spec.has(kRequiresSidestep)) return "out of a sidestep only";
    if (spec.has(kRequiresBackTurned) != fighter.back_turned) return spec.has(kRequiresBackTurned) ? "back turned only" : "not while back turned";
    if (spec.has(kRequiresOpponentBackTurned)) return "opponent back turned only";
    if (spec.has(kRequiresBackToWall)) return "back to the wall only";
    if (spec.has(kRequiresOpponentLeft) || spec.has(kRequiresOpponentRight)) return "opponent's side only";
    if (spec.required_posture != fighter.posture) {
        static const char* names[] = {"standing", "crouching", "airborne", "grounded", "wakeup", "wall splat"};
        const int posture = static_cast<int>(spec.required_posture);
        return std::string("needs posture ") + (posture >= 0 && posture < 6 ? names[posture] : "?");
    }
    if (spec.required_stance >= 0 && spec.required_stance != fighter.stance) return "needs a stance";
    if (spec.has(kRequiresHeat) || spec.has(kConsumesHeat) || spec.heat_cost_frames != 0) return "needs Heat";
    if (spec.has(kEngagesHeat)) return "Heat engager (needs Heat available)";
    if (spec.has(kRequiresRage) || spec.has(kConsumesRage)) return "needs Rage";
    return "other";
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path generated = std::filesystem::path(T8_V2_SOURCE_DIR) / "data" / "generated";
    std::vector<std::string> characters;
    bool verbose = false;
    bool blockers = false;
    bool list_unbound = false;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--character" && index + 1 < argc) characters.push_back(argv[++index]);
        else if (argument == "--all") characters.assign(kRosterCharacterSlugs.begin(), kRosterCharacterSlugs.end());
        else if (argument == "--data-dir" && index + 1 < argc) generated = argv[++index];
        else if (argument == "--verbose") verbose = true;
        else if (argument == "--blockers") blockers = true;
        else if (argument == "--list-unbound") blockers = list_unbound = true;
        else {
            std::cerr << "usage: t8_v2_full_move_inspect [--character slug | --all] [--data-dir PATH] [--verbose]\n";
            return 2;
        }
    }
    if (characters.empty()) characters.assign(kRosterCharacterSlugs.begin(), kRosterCharacterSlugs.end());
    FullMoveRosterOptions options{};
    options.allow_provisional = true;
    std::cout << std::left << std::setw(12) << "character" << std::right << std::setw(11) << "candidates"
              << std::setw(8) << "bound" << std::setw(15) << "legal neutral" << "   why the bound moves are not legal\n";
    for (const auto& slug : characters) {
        FullMoveCharacter character;
        try {
            character = load_full_move_character(generated, slug, options);
        } catch (const std::exception& error) {
            std::cout << std::left << std::setw(12) << slug << " cannot load: " << error.what() << '\n';
            continue;
        }
        Tables tables{};
        tables.side[0] = character.tables.view().side[0];
        tables.side[1] = character.tables.view().side[0];
        tables.config = character.tables.config;
        State state{};
        reset(tables, state, -0.6, 0.6);  // a neutral stand at 1.2 m
        const CandidateMap map{character.candidate_moves.data(), character.candidate_count()};
        int legal_count = 0;
        std::map<std::string, int> reasons;
        std::vector<std::string> examples;
        for (int candidate = 0; candidate < character.candidate_count(); ++candidate) {
            if (candidate_legal(tables, state, 0, map, candidate)) {
                ++legal_count;
                continue;
            }
            const int move = character.candidate_moves[static_cast<std::size_t>(candidate)];
            if (move < 0) {
                if (candidate >= kFullMoveUniversalCount) ++reasons["unbound"];
                continue;
            }
            const std::string why = reason(tables, state, 0, move);
            ++reasons[why];
            if (verbose) examples.push_back("    " + character.candidate_ids[static_cast<std::size_t>(candidate)] + ": " + why);
        }
        std::cout << std::left << std::setw(12) << slug << std::right << std::setw(11) << character.candidate_count()
                  << std::setw(8) << character.bound_moves << std::setw(15) << legal_count << "   ";
        bool first = true;
        for (const auto& [why, count] : reasons) {
            std::cout << (first ? "" : ", ") << why << ' ' << count;
            first = false;
        }
        std::cout << '\n';
        for (const auto& line : examples) std::cout << line << '\n';
        if (blockers) {  // why the unbound moves did not bind, as the roster loader binds them
            FullBindingOptions binding{};
            binding.require_practice_validation = false;
            binding.allow_variable_frame_data = true;
            binding.provisional = FullProvisionalGeometry{};
            try {
                binding.stance_names = load_full_combat_character_rules(generated / "full_combat_stances.csv",
                                                                        generated / "full_combat_resources.csv", slug)
                                           .stance_names;
            } catch (const std::exception&) {
            }
            std::map<std::string, int> kinds;
            std::map<std::string, std::string> example;
            const auto rows = load_full_combat_bindings(generated / "full_combat_bindings.csv", slug);
            FullCharacterRules stand_in_rules{};
            add_provisional_stances(rows, stand_in_rules, binding.stance_names);  // as the roster loader does
            for (const auto& move : bind_full_combat_moves(rows, binding)) {
                if (list_unbound && !move.blockers.empty()) {
                    std::cout << "unbound\t" << move.stable_id << '\t' << move.command << '\t';
                    for (std::size_t k = 0; k < move.blockers.size(); ++k) std::cout << (k ? "; " : "") << move.blockers[k];
                    std::cout << '\n';
                }
                for (const auto& blocker : move.blockers) {
                    const std::string kind = blocker.substr(0, blocker.find(':'));
                    ++kinds[kind];
                    if (!example.count(kind)) example[kind] = move.command + " -> " + blocker;
                }
            }
            for (const auto& [kind, count] : kinds) {
                std::cout << "    blocker " << std::setw(5) << count << "  " << kind;
                if (verbose) std::cout << "   e.g. " << example[kind];
                std::cout << '\n';
            }
        }
    }
    return EXIT_SUCCESS;
}
