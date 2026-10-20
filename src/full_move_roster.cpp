#include "t8_v2/full_move_roster.hpp"

#include <algorithm>

#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace t8::v2 {

const std::array<const char*, kFullMoveUniversalCount> kFullMoveUniversalNames = {
    "neutral", "walk_forward", "walk_back", "dash_forward", "dash_back", "crouch", "stand",
    "jump", "sidestep_left", "sidestep_right", "sidewalk_left", "sidewalk_right", "block_high",
    "block_low", "low_parry", "throw_break_1", "throw_break_2", "throw_break_1p2",
    "heat_dash", "wakeup_stand", "wakeup_roll", "tech_roll",
};

namespace {

std::vector<std::string> split_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char character = line[index];
        if (quoted) {
            if (character == '"' && index + 1 < line.size() && line[index + 1] == '"') {
                field += '"';
                ++index;
            } else if (character == '"') {
                quoted = false;
            } else {
                field += character;
            }
        } else if (character == '"') {
            quoted = true;
        } else if (character == ',') {
            fields.push_back(field);
            field.clear();
        } else if (character != '\r') {
            field += character;
        }
    }
    fields.push_back(field);
    return fields;
}

}  // namespace

FullMoveCharacter load_full_move_character(const std::filesystem::path& generated, const std::string& slug,
                                           const FullMoveRosterOptions& options) {
    FullMoveCharacter character;
    character.slug = slug;
    for (std::size_t index = 0; index < kRosterCharacterSlugs.size(); ++index) {
        if (kRosterCharacterSlugs[index] == slug) character.roster_id = static_cast<int>(index);
    }
    if (character.roster_id < 0) throw std::runtime_error("not a roster character: " + slug);
    character.provisional = options.allow_provisional;

    // Engine moveset: every row that binds (reactive outcomes included, since parries play them).
    FullCharacterRules rules{};
    FullBindingOptions binding{};
    if (options.allow_provisional) {
        binding.require_practice_validation = false;
        binding.allow_variable_frame_data = true;
        binding.provisional = FullProvisionalGeometry{};
    }
    try {
        auto set = load_full_combat_character_rules(generated / "full_combat_stances.csv",
                                                    generated / "full_combat_resources.csv", slug);
        rules = set.rules;
        binding.stance_names = set.stance_names;
    } catch (const std::exception&) {
        rules = {};  // no stance or resource rules recorded for this character yet
    }
    const auto rows = load_full_combat_bindings(generated / "full_combat_bindings.csv", slug);
    // Stances the moves use but the stance table lacks get stand-in rules, so their moves bind.
    if (options.allow_provisional) add_provisional_stances(rows, rules, binding.stance_names);
    std::vector<FullMoveSpec> moves;
    for (const auto& move : bind_full_combat_moves(rows, binding)) {
        if (move.bound()) moves.push_back(*move.spec);
    }
    std::unordered_map<std::string, int> engine_index;
    for (std::size_t index = 0; index < moves.size(); ++index) {
        engine_index.emplace(moves[index].stable_id, static_cast<int>(index));
    }
    character.tables = compile_full_tables({moves, moves}, options.engine, {rules, rules});

    // Candidates, in the shared contract order.
    std::ifstream input(generated / "full_move_candidates.csv");
    if (!input) throw std::runtime_error("cannot read full_move_candidates.csv in " + generated.string());
    std::string line;
    std::getline(input, line);
    const auto header = split_csv_line(line);
    std::unordered_map<std::string, std::size_t> column;
    for (std::size_t index = 0; index < header.size(); ++index) column.emplace(header[index], index);
    for (const char* name : {"character", "candidate", "stable_id", "kind", "feature_0"}) {
        if (!column.count(name)) throw std::runtime_error(std::string("full_move_candidates.csv lacks ") + name);
    }
    const std::size_t first_feature = column.at("feature_0");
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto fields = split_csv_line(line);
        if (fields.size() != header.size()) throw std::runtime_error("malformed full_move_candidates.csv row");
        if (fields[column.at("character")] != slug) continue;
        const int index = std::stoi(fields[column.at("candidate")]);
        if (index != character.candidate_count()) {
            throw std::runtime_error(slug + ": candidates are not numbered in order");
        }
        const std::string& stable_id = fields[column.at("stable_id")];
        if (index < kFullMoveUniversalCount &&
            stable_id != std::string("universal:") + kFullMoveUniversalNames[static_cast<std::size_t>(index)]) {
            throw std::runtime_error(slug + ": universal candidates differ from the contract at " + stable_id);
        }
        character.candidate_ids.push_back(stable_id);
        const auto found = engine_index.find(stable_id);
        const int move = index < kFullMoveUniversalCount || found == engine_index.end() ? -1 : found->second;
        character.candidate_moves.push_back(move);
        character.bound_moves += move >= 0;
        for (int feature = 0; feature < kFullMoveFeatureSize; ++feature) {
            character.candidate_features.push_back(std::stof(fields[first_feature + static_cast<std::size_t>(feature)]));
        }
    }
    if (character.candidate_count() < kFullMoveUniversalCount) {
        throw std::runtime_error("no full-move candidates for " + slug);
    }
    if (character.candidate_count() > kFullMoveMaxCandidates) {
        throw std::runtime_error(slug + " exceeds the full-move candidate bound");
    }
    return character;
}

}  // namespace t8::v2
