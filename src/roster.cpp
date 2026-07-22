#include "t8_v2/roster.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>

namespace t8::v2 {
namespace {

std::vector<std::string> split_csv_row(const std::string& line) {
    std::vector<std::string> fields;
    std::string field;
    bool quoted = false;
    for (std::size_t index = 0; index < line.size(); ++index) {
        const char value = line[index];
        if (value == '"') {
            if (quoted && index + 1 < line.size() && line[index + 1] == '"') {
                field.push_back('"');
                ++index;
            } else {
                quoted = !quoted;
            }
        } else if (value == ',' && !quoted) {
            fields.push_back(std::move(field));
            field.clear();
        } else {
            field.push_back(value);
        }
    }
    if (quoted) throw std::runtime_error("unterminated quoted CSV field");
    fields.push_back(std::move(field));
    return fields;
}

template <typename T>
T parse_integer(const std::string& value, std::string_view field, std::size_t row) {
    std::size_t consumed = 0;
    unsigned long long parsed = 0;
    try {
        parsed = std::stoull(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid integer " + std::string(field) + " at profile row " + std::to_string(row));
    }
    if (consumed != value.size() || parsed > static_cast<unsigned long long>(std::numeric_limits<T>::max())) {
        throw std::runtime_error("out-of-range integer " + std::string(field) + " at profile row " + std::to_string(row));
    }
    return static_cast<T>(parsed);
}

float parse_probability(const std::string& value, std::string_view field, std::size_t row) {
    std::size_t consumed = 0;
    float parsed = 0.0F;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid float " + std::string(field) + " at profile row " + std::to_string(row));
    }
    if (consumed != value.size() || !std::isfinite(parsed) || parsed < 0.0F || parsed > 1.0F) {
        throw std::runtime_error("probability " + std::string(field) + " must be in [0, 1] at profile row " + std::to_string(row));
    }
    return parsed;
}

float parse_finite_float(const std::string& value, std::string_view field, std::size_t row) {
    std::size_t consumed = 0;
    float parsed = 0.0F;
    try {
        parsed = std::stof(value, &consumed);
    } catch (const std::exception&) {
        throw std::runtime_error("invalid float " + std::string(field) + " at row " + std::to_string(row));
    }
    if (consumed != value.size() || !std::isfinite(parsed)) {
        throw std::runtime_error("non-finite float " + std::string(field) + " at row " + std::to_string(row));
    }
    return parsed;
}

}  // namespace

double MatchupStats::win_rate() const noexcept {
    return episodes == 0 ? 0.5 :
        (static_cast<double>(wins) + 0.5 * static_cast<double>(draws)) / static_cast<double>(episodes);
}

double MatchupStats::uncertainty() const noexcept {
    return 1.0 / std::sqrt(static_cast<double>(episodes) + 1.0);
}

double MatchupStats::regression() const noexcept {
    return std::max(0.0, best_win_rate - recent_win_rate);
}

std::vector<OpponentProfileParameters> load_opponent_profiles_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open opponent profile catalog: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty opponent profile catalog: " + path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split_csv_row(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) columns.emplace(header[index], index);
    const auto column = [&](std::string_view name) {
        const auto found = columns.find(std::string(name));
        if (found == columns.end()) throw std::runtime_error("profile catalog is missing column: " + std::string(name));
        return found->second;
    };
    column("id");
    column("character_id");
    column("group_mask");
    column("archetype_id");
    column("variation_id");
    column("reaction_min");
    column("reaction_max");
    const std::array probability_names = {
        "aggression", "input_error_rate", "approach", "backdash", "sidestep_left",
        "sidestep_right", "low_frequency", "throw_frequency", "delay_frequency",
        "stance_entry_frequency", "heat_usage", "punish_accuracy", "throw_break_accuracy",
        "low_block_accuracy"};
    std::array<std::size_t, probability_names.size()> probability_columns{};
    for (std::size_t index = 0; index < probability_names.size(); ++index) {
        probability_columns[index] = column(probability_names[index]);
    }

    std::vector<OpponentProfileParameters> profiles;
    std::size_t row = 1;
    while (std::getline(input, line)) {
        ++row;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_csv_row(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("profile catalog row has the wrong field count at row " + std::to_string(row));
        }
        const auto get = [&](std::string_view name) -> const std::string& { return fields[column(name)]; };
        OpponentProfileParameters value{};
        value.id = parse_integer<std::uint32_t>(get("id"), "id", row);
        value.character_id = parse_integer<std::uint32_t>(get("character_id"), "character_id", row);
        value.group_mask = parse_integer<std::uint32_t>(get("group_mask"), "group_mask", row);
        value.archetype_id = parse_integer<std::uint32_t>(get("archetype_id"), "archetype_id", row);
        value.variation_id = parse_integer<std::uint32_t>(get("variation_id"), "variation_id", row);
        value.reaction_min = parse_integer<std::int32_t>(get("reaction_min"), "reaction_min", row);
        value.reaction_max = parse_integer<std::int32_t>(get("reaction_max"), "reaction_max", row);
        float* outputs[] = {
            &value.aggression, &value.input_error_rate, &value.approach, &value.backdash,
            &value.sidestep_left, &value.sidestep_right, &value.low_frequency,
            &value.throw_frequency, &value.delay_frequency, &value.stance_entry_frequency,
            &value.heat_usage, &value.punish_accuracy, &value.throw_break_accuracy,
            &value.low_block_accuracy};
        for (std::size_t index = 0; index < probability_names.size(); ++index) {
            *outputs[index] = parse_probability(fields[probability_columns[index]], probability_names[index], row);
        }
        if (value.id != profiles.size() || value.character_id >= kRosterCharacterCount ||
            value.group_mask == 0 || value.archetype_id >= kOpponentArchetypeCount ||
            value.variation_id >= kVariationsPerArchetype || value.reaction_min <= 0 ||
            value.reaction_max < value.reaction_min) {
            throw std::runtime_error("invalid profile identity or reaction range at row " + std::to_string(row));
        }
        profiles.push_back(value);
    }
    if (input.bad()) throw std::runtime_error("failed reading opponent profile catalog: " + path.string());
    if (profiles.size() != kOpponentProfileCount) {
        throw std::runtime_error("opponent profile catalog must contain exactly " +
                                 std::to_string(kOpponentProfileCount) + " profiles");
    }
    return profiles;
}

std::vector<CharacterMoveParameters> load_character_move_specs_csv(const std::filesystem::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not open character move catalog: " + path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("empty character move catalog: " + path.string());
    if (!line.empty() && line.back() == '\r') line.pop_back();
    const auto header = split_csv_row(line);
    std::unordered_map<std::string, std::size_t> columns;
    for (std::size_t index = 0; index < header.size(); ++index) columns.emplace(header[index], index);
    const auto column = [&](std::string_view name) {
        const auto found = columns.find(std::string(name));
        if (found == columns.end()) throw std::runtime_error("move catalog is missing column: " + std::string(name));
        return found->second;
    };
    constexpr std::array<std::string_view, kCharacterMoveSlotCount> slot_names = {
        "jab", "df1", "f2", "db3", "hopkick", "throw"};
    const auto hit_level = [](std::string_view name) -> std::int32_t {
        if (name == "high") return 1;
        if (name == "mid") return 2;
        if (name == "low") return 3;
        if (name == "throw") return 4;
        return 0;
    };
    std::vector<CharacterMoveParameters> moves;
    std::size_t row = 1;
    while (std::getline(input, line)) {
        ++row;
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        const auto fields = split_csv_row(line);
        if (fields.size() != header.size()) {
            throw std::runtime_error("move catalog row has the wrong field count at row " + std::to_string(row));
        }
        const auto get = [&](std::string_view name) -> const std::string& { return fields[column(name)]; };
        CharacterMoveParameters value{};
        value.character_id = parse_integer<std::uint32_t>(get("character_id"), "character_id", row);
        const auto slot = std::find(slot_names.begin(), slot_names.end(), get("slot"));
        if (slot == slot_names.end()) throw std::runtime_error("unknown move slot at row " + std::to_string(row));
        value.slot = static_cast<std::uint32_t>(std::distance(slot_names.begin(), slot));
        value.hit_level = hit_level(get("hit_level"));
        value.startup = parse_integer<std::int32_t>(get("startup"), "startup", row);
        value.active = parse_integer<std::int32_t>(get("active"), "active", row);
        value.recovery = parse_integer<std::int32_t>(get("recovery"), "recovery", row);
        value.damage = parse_finite_float(get("damage"), "damage", row);
        value.range = parse_finite_float(get("range"), "range", row);
        value.hitstun = parse_integer<std::int32_t>(get("hitstun"), "hitstun", row);
        value.blockstun = parse_integer<std::int32_t>(get("blockstun"), "blockstun", row);
        value.pushback = parse_finite_float(get("pushback"), "pushback", row);
        value.whiff_recovery = parse_integer<std::int32_t>(get("whiff_recovery"), "whiff_recovery", row);
        value.launches = parse_integer<std::int32_t>(get("launches"), "launches", row);
        const std::size_t expected_index = value.character_id * kCharacterMoveSlotCount + value.slot;
        if (value.character_id >= kRosterCharacterCount || expected_index != moves.size() ||
            value.hit_level <= 0 || value.startup <= 0 || value.active <= 0 || value.recovery <= 0 ||
            value.damage < 0.0F || value.range <= 0.0F || value.launches < 0 || value.launches > 1) {
            throw std::runtime_error("invalid or unordered character move at row " + std::to_string(row));
        }
        moves.push_back(value);
    }
    if (moves.size() != kRosterCharacterCount * kCharacterMoveSlotCount) {
        throw std::runtime_error("character move catalog must contain exactly " +
            std::to_string(kRosterCharacterCount * kCharacterMoveSlotCount) + " rows");
    }
    return moves;
}

MatchupScheduler::MatchupScheduler(
    std::span<const OpponentProfileParameters> profiles,
    std::uint64_t seed)
    : profiles_(profiles.begin(), profiles.end()), stats_(profiles.size()), random_state_(seed) {
    if (profiles_.empty()) throw std::invalid_argument("matchup scheduler requires profiles");
}

void MatchupScheduler::set_stage(CurriculumStage stage, std::uint32_t active_group_mask) {
    if (stage == CurriculumStage::CharacterGroups && active_group_mask == 0) {
        throw std::invalid_argument("character-group curriculum requires a nonzero group mask");
    }
    stage_ = stage;
    active_group_mask_ = active_group_mask;
}

std::vector<std::uint32_t> MatchupScheduler::eligible_profile_indices() const {
    std::vector<std::uint32_t> result;
    result.reserve(profiles_.size());
    for (std::size_t index = 0; index < profiles_.size(); ++index) {
        const auto& profile = profiles_[index];
        bool eligible = true;
        if (stage_ == CurriculumStage::JunFundamentals) {
            eligible = (profile.group_mask & CharacterGroup::Fundamentals) != 0 &&
                profile.archetype_id <= 4 && profile.variation_id <= 1;
        } else if (stage_ == CurriculumStage::CharacterGroups) {
            eligible = (profile.group_mask & active_group_mask_) != 0;
        }
        if (eligible) result.push_back(static_cast<std::uint32_t>(index));
    }
    return result;
}

std::vector<std::uint32_t> MatchupScheduler::sample_profile_indices(
    std::size_t count,
    bool mirror_learner_sides) {
    if (count == 0) throw std::invalid_argument("sample count must be positive");
    const auto eligible = eligible_profile_indices();
    if (eligible.empty()) throw std::runtime_error("curriculum has no eligible profiles");
    double total = 0.0;
    for (const auto index : eligible) total += priority(index);
    std::vector<std::uint32_t> result(count);
    if (mirror_learner_sides && count % 16 == 0) {
        for (std::size_t block = 0; block < count; block += 16) {
            for (std::size_t lane = 0; lane < 8; ++lane) {
                const auto selected = sample_one(eligible, total);
                result[block + lane] = selected;
                result[block + 8 + lane] = selected;
            }
        }
    } else {
        for (auto& index : result) index = sample_one(eligible, total);
    }
    return result;
}

void MatchupScheduler::record(
    std::uint32_t profile_index,
    std::uint64_t wins,
    std::uint64_t losses,
    std::uint64_t draws,
    double recent_win_rate,
    double exploit_severity) {
    if (profile_index >= stats_.size() || wins + losses + draws == 0) {
        throw std::invalid_argument("invalid matchup outcome batch");
    }
    auto& value = stats_[profile_index];
    value.episodes += wins + losses + draws;
    value.wins += wins;
    value.losses += losses;
    value.draws += draws;
    const double batch = (static_cast<double>(wins) + 0.5 * static_cast<double>(draws)) /
        static_cast<double>(wins + losses + draws);
    value.recent_win_rate = recent_win_rate < 0.0 ? batch : recent_win_rate;
    value.best_win_rate = std::max(value.best_win_rate, value.recent_win_rate);
    if (exploit_severity >= 0.0) value.exploit_severity = exploit_severity;
}

double MatchupScheduler::priority(std::uint32_t profile_index) const {
    if (profile_index >= stats_.size()) throw std::out_of_range("profile index is out of range");
    const auto& value = stats_[profile_index];
    double result = 1.0 - value.win_rate() + value.uncertainty() +
        value.regression() + value.exploit_severity;
    if (value.win_rate() < 0.25) {
        result *= profiles_[profile_index].variation_id <= 1 ? 1.35 : 0.65;
    } else if (value.win_rate() > 0.95) {
        result *= 0.10;
    }
    return std::max(result, 1e-6);
}

const MatchupStats& MatchupScheduler::stats(std::uint32_t profile_index) const {
    if (profile_index >= stats_.size()) throw std::out_of_range("profile index is out of range");
    return stats_[profile_index];
}

void MatchupScheduler::restore_state(
    std::span<const MatchupStats> stats,
    std::uint64_t random_state) {
    if (stats.size() != stats_.size()) {
        throw std::invalid_argument("scheduler state size does not match profile catalog");
    }
    stats_.assign(stats.begin(), stats.end());
    random_state_ = random_state;
}

std::uint64_t MatchupScheduler::next_random() noexcept {
    random_state_ += 0x9e3779b97f4a7c15ULL;
    auto value = random_state_;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

std::uint32_t MatchupScheduler::sample_one(
    std::span<const std::uint32_t> eligible,
    double total_priority) {
    const double unit = static_cast<double>(next_random() >> 11U) * (1.0 / 9007199254740992.0);
    double target = unit * total_priority;
    for (const auto index : eligible) {
        target -= priority(index);
        if (target <= 0.0) return index;
    }
    return eligible.back();
}

void write_matchup_matrix_json(
    const std::filesystem::path& path,
    std::span<const OpponentProfileParameters> profiles,
    std::span<const MatchupStats> stats,
    std::string_view roster_as_of) {
    if (profiles.size() != stats.size()) {
        throw std::invalid_argument("matchup matrix profiles and stats must have equal length");
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("could not write matchup matrix: " + path.string());
    output << std::setprecision(9) << "{\"roster_as_of\":\"" << roster_as_of << "\",\"characters\":[";
    for (std::size_t character = 0; character < kRosterCharacterCount; ++character) {
        if (character != 0) output << ',';
        std::uint64_t episodes = 0;
        std::uint64_t wins = 0;
        std::uint64_t losses = 0;
        std::uint64_t draws = 0;
        double elo = 0.0;
        std::size_t evaluated_archetypes = 0;
        output << "{\"character_id\":" << character << ",\"archetypes\":[";
        for (std::size_t archetype = 0; archetype < kOpponentArchetypeCount; ++archetype) {
            if (archetype != 0) output << ',';
            MatchupStats aggregate{};
            aggregate.recent_win_rate = 0.0;
            for (std::size_t index = 0; index < profiles.size(); ++index) {
                if (profiles[index].character_id != character || profiles[index].archetype_id != archetype) continue;
                aggregate.episodes += stats[index].episodes;
                aggregate.wins += stats[index].wins;
                aggregate.losses += stats[index].losses;
                aggregate.draws += stats[index].draws;
                aggregate.best_win_rate = std::max(aggregate.best_win_rate, stats[index].best_win_rate);
                aggregate.recent_win_rate += stats[index].recent_win_rate;
                aggregate.exploit_severity = std::max(aggregate.exploit_severity, stats[index].exploit_severity);
            }
            episodes += aggregate.episodes;
            wins += aggregate.wins;
            losses += aggregate.losses;
            draws += aggregate.draws;
            const double win_rate = aggregate.episodes == 0 ? 0.0 :
                static_cast<double>(aggregate.wins) / static_cast<double>(aggregate.episodes);
            const double score_rate = aggregate.episodes == 0 ? 0.5 :
                (static_cast<double>(aggregate.wins) + 0.5 * static_cast<double>(aggregate.draws)) /
                    static_cast<double>(aggregate.episodes);
            const bool forgetting = aggregate.best_win_rate > 0.0 &&
                score_rate + 0.10 < aggregate.best_win_rate;
            output << "{\"archetype_id\":" << archetype
                   << ",\"episodes\":" << aggregate.episodes
                   << ",\"wins\":" << aggregate.wins
                   << ",\"losses\":" << aggregate.losses
                   << ",\"draws\":" << aggregate.draws
                   << ",\"win_rate\":" << win_rate
                   << ",\"score_rate\":" << score_rate
                   << ",\"best_win_rate\":" << aggregate.best_win_rate
                   << ",\"catastrophic_forgetting\":" << (forgetting ? "true" : "false")
                   << '}';
            if (aggregate.episodes != 0) {
                elo += 1500.0 + 400.0 * (score_rate - 0.5);
                ++evaluated_archetypes;
            }
        }
        const double overall = episodes == 0 ? 0.0 :
            static_cast<double>(wins) / static_cast<double>(episodes);
        const double overall_score = episodes == 0 ? 0.5 :
            (static_cast<double>(wins) + 0.5 * static_cast<double>(draws)) /
                static_cast<double>(episodes);
        output << "],\"episodes\":" << episodes
               << ",\"wins\":" << wins << ",\"losses\":" << losses
               << ",\"draws\":" << draws << ",\"overall\":" << overall
               << ",\"overall_score\":" << overall_score
               << ",\"matchup_elo\":" << (evaluated_archetypes == 0
                    ? 1500.0 : elo / static_cast<double>(evaluated_archetypes))
               << '}';
    }
    output << "]}\n";
    if (!output) throw std::runtime_error("matchup matrix write failed: " + path.string());
}

}  // namespace t8::v2
