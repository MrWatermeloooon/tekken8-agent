// Unit coverage for the parts of t8_v2/roster.hpp that roster_tests.cpp does
// not already exercise through the real generated catalog: the MatchupStats
// formulas, MatchupScheduler's outcome-recording/priority/restore contract,
// the matchup-matrix JSON writer, and the CSV loaders' error paths.
#include "t8_v2/roster.hpp"

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        if (failures < 30) std::cerr << "FAIL: " << message << '\n';
    }
}

void near(double actual, double expected, std::string_view message, double tolerance = 1e-9) {
    check(std::abs(actual - expected) <= tolerance, message);
}

std::filesystem::path temp_file(std::string_view name) {
    return std::filesystem::temp_directory_path() / name;
}

// Extracts the number following the first occurrence of `key` in `text`
// (e.g. key = "\"win_rate\":"), stopping at the next ',' or '}'. There is no
// JSON library linked into the native test suite, so the matchup-matrix
// writer is checked this way rather than with brittle substring equality.
double extract_json_number(const std::string& text, std::string_view key, std::size_t from = 0) {
    const auto marker = text.find(key, from);
    if (marker == std::string::npos) throw std::runtime_error("key not found: " + std::string(key));
    const auto begin = marker + key.size();
    const auto end = text.find_first_of(",}", begin);
    return std::stod(text.substr(begin, end - begin));
}

void test_matchup_stats_formulas() {
    t8::v2::MatchupStats fresh{};
    near(fresh.win_rate(), 0.5, "an untouched matchup defaults to a coin-flip win rate");
    check(fresh.uncertainty() == 1.0, "an untouched matchup has maximal uncertainty");
    near(fresh.regression(), 0.0, "an untouched matchup has no forgetting regression");

    t8::v2::MatchupStats played{};
    played.episodes = 20;
    played.wins = 7;
    played.losses = 11;
    played.draws = 2;
    near(played.win_rate(), (7.0 + 0.5 * 2.0) / 20.0, "win rate credits draws as half a win");
    near(played.uncertainty(), 1.0 / std::sqrt(21.0), "uncertainty shrinks with 1/sqrt(episodes+1)");

    played.best_win_rate = 0.80;
    played.recent_win_rate = 0.55;
    near(played.regression(), 0.25, "regression is the drop from a matchup's best recorded win rate");
    played.recent_win_rate = 0.95;
    near(played.regression(), 0.0, "regression never goes negative when recent play improves");
}

void test_scheduler_record_validation_and_priority_thresholds() {
    std::vector<t8::v2::OpponentProfileParameters> profiles(3);
    for (std::size_t index = 0; index < profiles.size(); ++index) {
        profiles[index].id = static_cast<std::uint32_t>(index);
        profiles[index].group_mask = t8::v2::CharacterGroup::Fundamentals;
    }
    profiles[0].variation_id = 0;  // low win rate, gets boosted
    profiles[1].variation_id = 2;  // low win rate, gets damped relative to profile 0
    profiles[2].variation_id = 0;  // near-solved matchup, gets squashed

    t8::v2::MatchupScheduler scheduler(profiles, 7);

    bool rejected_empty_batch = false;
    try {
        scheduler.record(0, 0, 0, 0);
    } catch (const std::invalid_argument&) {
        rejected_empty_batch = true;
    }
    check(rejected_empty_batch, "recording a zero-episode outcome batch is rejected");

    bool rejected_out_of_range = false;
    try {
        scheduler.record(static_cast<std::uint32_t>(profiles.size()), 1, 0, 0);
    } catch (const std::invalid_argument&) {
        rejected_out_of_range = true;
    }
    check(rejected_out_of_range, "recording against an out-of-range profile index is rejected");

    bool rejected_priority_range = false;
    try {
        static_cast<void>(scheduler.priority(static_cast<std::uint32_t>(profiles.size())));
    } catch (const std::out_of_range&) {
        rejected_priority_range = true;
    }
    check(rejected_priority_range, "priority() rejects an out-of-range profile index");

    scheduler.record(0, 1, 19, 0);   // win_rate = 0.05, variation 0 -> boosted 1.35x
    scheduler.record(1, 1, 19, 0);   // win_rate = 0.05, variation 2 -> damped 0.65x
    scheduler.record(2, 39, 1, 0);   // win_rate = 0.975 -> squashed 0.10x

    check(scheduler.priority(0) > scheduler.priority(1),
          "a fundamentals-tier weak matchup outranks a specialist-variation weak matchup");
    check(scheduler.priority(1) > scheduler.priority(2),
          "any struggling matchup still outranks a matchup the learner has nearly solved");
}

void test_scheduler_restore_state_round_trip_and_rejects_size_mismatch() {
    std::vector<t8::v2::OpponentProfileParameters> profiles(4);
    for (std::size_t index = 0; index < profiles.size(); ++index) {
        profiles[index].id = static_cast<std::uint32_t>(index);
        profiles[index].group_mask = t8::v2::CharacterGroup::Fundamentals;
    }

    t8::v2::MatchupScheduler scheduler(profiles, 4242);
    scheduler.set_stage(t8::v2::CurriculumStage::FullRoster);
    scheduler.record(0, 3, 1, 0);
    scheduler.record(2, 1, 3, 0);
    static_cast<void>(scheduler.sample_profile_indices(5, false));

    const std::vector<t8::v2::MatchupStats> snapshot(
        scheduler.all_stats().begin(), scheduler.all_stats().end());
    const auto snapshot_random = scheduler.random_state();
    const auto snapshot_sample = scheduler.sample_profile_indices(6, false);

    // Diverge the live scheduler further so restoring is a meaningful check.
    scheduler.record(1, 10, 0, 0);
    static_cast<void>(scheduler.sample_profile_indices(9, false));

    scheduler.restore_state(snapshot, snapshot_random);
    check(scheduler.random_state() == snapshot_random, "restore_state exactly restores the random cursor");
    bool stats_match = scheduler.all_stats().size() == snapshot.size();
    for (std::size_t index = 0; stats_match && index < snapshot.size(); ++index) {
        const auto& restored = scheduler.stats(static_cast<std::uint32_t>(index));
        const auto& expected = snapshot[index];
        stats_match = restored.episodes == expected.episodes && restored.wins == expected.wins &&
            restored.losses == expected.losses && restored.draws == expected.draws &&
            restored.best_win_rate == expected.best_win_rate &&
            restored.recent_win_rate == expected.recent_win_rate;
    }
    check(stats_match, "restore_state exactly restores every matchup's recorded stats");
    check(scheduler.sample_profile_indices(6, false) == snapshot_sample,
          "restoring stats and the random cursor reproduces the exact future sample sequence");

    bool rejected_size_mismatch = false;
    try {
        scheduler.restore_state(std::span<const t8::v2::MatchupStats>(snapshot.data(), snapshot.size() - 1),
                                snapshot_random);
    } catch (const std::invalid_argument&) {
        rejected_size_mismatch = true;
    }
    check(rejected_size_mismatch, "restore_state rejects a stats vector sized for a different catalog");
}

void test_write_matchup_matrix_json_aggregates_expected_fields() {
    std::vector<t8::v2::OpponentProfileParameters> profiles(2);
    profiles[0].id = 0;
    profiles[0].character_id = 5;
    profiles[0].archetype_id = 1;
    profiles[1].id = 1;
    profiles[1].character_id = 5;
    profiles[1].archetype_id = 1;

    std::vector<t8::v2::MatchupStats> stats(2);
    stats[0].episodes = 10;
    stats[0].wins = 7;
    stats[0].losses = 3;
    stats[0].draws = 0;
    stats[1].episodes = 10;
    stats[1].wins = 1;
    stats[1].losses = 9;
    stats[1].draws = 0;

    const auto path = temp_file("t8_v2_roster_matrix_test.json");
    std::error_code error;
    std::filesystem::remove(path, error);
    t8::v2::write_matchup_matrix_json(path, profiles, stats, "2027-01-01");

    std::ifstream input(path);
    const std::string text((std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    check(text.find("\"roster_as_of\":\"2027-01-01\"") != std::string::npos,
          "matchup matrix writes the roster_as_of stamp verbatim");

    // Both profiles are on character 5, so locate that character's block
    // first: "archetype_id":1 alone would match every other character's
    // (empty) archetype-1 entry too.
    const auto character_marker = text.find("\"character_id\":5");
    check(character_marker != std::string::npos, "matchup matrix reports the character block");
    const auto archetype_marker = text.find("\"archetype_id\":1", character_marker);
    check(archetype_marker != std::string::npos, "matchup matrix reports the aggregated archetype");
    near(extract_json_number(text, "\"episodes\":", archetype_marker), 20.0,
         "aggregated archetype episodes sum across both profiles");
    near(extract_json_number(text, "\"wins\":", archetype_marker), 8.0,
         "aggregated archetype wins sum across both profiles");
    near(extract_json_number(text, "\"win_rate\":", archetype_marker), 8.0 / 20.0,
         "aggregated archetype win rate divides summed wins by summed episodes", 1e-6);

    // The character-level totals are written immediately after the closing
    // "]" of that character's per-archetype array.
    const auto character_total_marker = text.find("],\"episodes\":", character_marker);
    check(character_total_marker != std::string::npos, "matchup matrix closes the archetype array");
    near(extract_json_number(text, "\"episodes\":", character_total_marker), 20.0,
         "character-level episodes match the single populated archetype", 1e-6);
    std::filesystem::remove(path, error);
}

void test_csv_loader_rejects_malformed_rows() {
    {
        const auto path = temp_file("t8_v2_roster_missing_column.csv");
        std::ofstream(path) << "id,character_id,group_mask,archetype_id,variation_id,"
                                "reaction_min\n0,0,1,0,0,12\n";
        bool rejected = false;
        try {
            static_cast<void>(t8::v2::load_opponent_profiles_csv(path));
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()).find("missing column") != std::string_view::npos;
        }
        check(rejected, "profile catalog loader rejects a header missing a required column");
        std::filesystem::remove(path);
    }
    {
        const auto path = temp_file("t8_v2_roster_bad_integer.csv");
        std::ofstream(path) << "id,character_id,group_mask,archetype_id,variation_id,reaction_min,"
                                "reaction_max,aggression,input_error_rate,approach,backdash,"
                                "sidestep_left,sidestep_right,low_frequency,throw_frequency,"
                                "delay_frequency,stance_entry_frequency,heat_usage,punish_accuracy,"
                                "throw_break_accuracy,low_block_accuracy\n"
                                "0,not-a-number,1,0,0,12,20,0.5,0,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,"
                                "0.5,0.5,0.5\n";
        bool rejected = false;
        try {
            static_cast<void>(t8::v2::load_opponent_profiles_csv(path));
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()).find("invalid integer") != std::string_view::npos;
        }
        check(rejected, "profile catalog loader rejects a malformed integer field");
        std::filesystem::remove(path);
    }
    {
        const auto path = temp_file("t8_v2_roster_nonfinite_move.csv");
        std::ofstream(path) << "character_id,slot,hit_level,startup,active,recovery,damage,range,"
                                "hitstun,blockstun,pushback,whiff_recovery,launches\n"
                                "0,jab,high,10,2,13,nan,0.82,14,7,0.08,0,0\n";
        bool rejected = false;
        try {
            static_cast<void>(t8::v2::load_character_move_specs_csv(path));
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()).find("non-finite float") != std::string_view::npos;
        }
        check(rejected, "character move catalog loader rejects a non-finite damage value");
        std::filesystem::remove(path);
    }
    {
        const auto path = temp_file("t8_v2_roster_unterminated_quote.csv");
        std::ofstream(path) << "\"unterminated\n0,0,1,0,0,12,20\n";
        bool rejected = false;
        try {
            static_cast<void>(t8::v2::load_opponent_profiles_csv(path));
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()).find("unterminated quoted") != std::string_view::npos;
        }
        check(rejected, "CSV row splitting rejects an unterminated quoted field");
        std::filesystem::remove(path);
    }
    {
        const auto path = temp_file("t8_v2_roster_empty.csv");
        std::ofstream(path).close();
        bool rejected = false;
        try {
            static_cast<void>(t8::v2::load_opponent_profiles_csv(path));
        } catch (const std::runtime_error& error) {
            rejected = std::string_view(error.what()).find("empty opponent profile catalog") !=
                std::string_view::npos;
        }
        check(rejected, "profile catalog loader rejects a completely empty file");
        std::filesystem::remove(path);
    }
}

}  // namespace

int main() {
    test_matchup_stats_formulas();
    test_scheduler_record_validation_and_priority_thresholds();
    test_scheduler_restore_state_round_trip_and_rejects_size_mismatch();
    test_write_matchup_matrix_json_aggregates_expected_fields();
    test_csv_loader_rejects_malformed_rows();
    if (failures != 0) {
        std::cerr << failures << " roster scheduler assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Matchup stats, scheduler priority/restore, matrix export, and CSV error paths passed\n";
    return EXIT_SUCCESS;
}
