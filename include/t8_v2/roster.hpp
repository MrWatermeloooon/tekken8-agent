#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <string_view>
#include <vector>

namespace t8::v2 {

inline constexpr std::size_t kRosterCharacterCount = 42;
inline constexpr std::size_t kOpponentArchetypeCount = 10;
inline constexpr std::size_t kVariationsPerArchetype = 5;
inline constexpr std::size_t kProfilesPerCharacter = 50;
inline constexpr std::size_t kOpponentProfileCount =
    kRosterCharacterCount * kProfilesPerCharacter;
inline constexpr std::size_t kCharacterMoveSlotCount = 6;
inline constexpr std::uint32_t kJunCharacterId = 3;

enum CharacterGroup : std::uint32_t {
    Fundamentals = 1U << 0U,
    Rushdown = 1U << 1U,
    StanceHeavy = 1U << 2U,
    Grappler = 1U << 3U,
    KeepOut = 1U << 4U,
    Evasive = 1U << 5U,
    Specialist = 1U << 6U,
    CounterHit = 1U << 7U,
    Movement = 1U << 8U,
};

enum class CurriculumStage : std::uint8_t {
    JunFundamentals = 1,
    CharacterGroups = 2,
    FullRoster = 3,
    AdversarialLeague = 4,
};

struct OpponentProfileParameters {
    std::uint32_t id = 0;
    std::uint32_t character_id = 0;
    std::uint32_t group_mask = 0;
    std::uint32_t archetype_id = 0;
    std::uint32_t variation_id = 0;
    float aggression = 0.5F;
    std::int32_t reaction_min = 12;
    std::int32_t reaction_max = 20;
    float input_error_rate = 0.0F;
    float approach = 0.25F;
    float backdash = 0.25F;
    float sidestep_left = 0.25F;
    float sidestep_right = 0.25F;
    float low_frequency = 0.2F;
    float throw_frequency = 0.1F;
    float delay_frequency = 0.2F;
    float stance_entry_frequency = 0.2F;
    float heat_usage = 0.5F;
    float punish_accuracy = 0.5F;
    float throw_break_accuracy = 0.5F;
    float low_block_accuracy = 0.5F;
};

struct CharacterMoveParameters {
    std::uint32_t character_id = 0;
    std::uint32_t slot = 0;
    std::int32_t hit_level = 0;
    std::int32_t startup = 1;
    std::int32_t active = 1;
    std::int32_t recovery = 1;
    float damage = 0.0F;
    float range = 0.0F;
    std::int32_t hitstun = 0;
    std::int32_t blockstun = 0;
    float pushback = 0.0F;
    std::int32_t whiff_recovery = 0;
    std::int32_t launches = 0;
};

struct MatchupStats {
    std::uint64_t episodes = 0;
    std::uint64_t wins = 0;
    std::uint64_t losses = 0;
    std::uint64_t draws = 0;
    double best_win_rate = 0.0;
    double recent_win_rate = 0.5;
    double exploit_severity = 0.0;

    [[nodiscard]] double win_rate() const noexcept;
    [[nodiscard]] double uncertainty() const noexcept;
    [[nodiscard]] double regression() const noexcept;
};

[[nodiscard]] std::vector<OpponentProfileParameters> load_opponent_profiles_csv(
    const std::filesystem::path& path);
[[nodiscard]] std::vector<CharacterMoveParameters> load_character_move_specs_csv(
    const std::filesystem::path& path);

class MatchupScheduler {
public:
    explicit MatchupScheduler(
        std::span<const OpponentProfileParameters> profiles,
        std::uint64_t seed = 2027);

    void set_stage(CurriculumStage stage, std::uint32_t active_group_mask = 0);
    [[nodiscard]] CurriculumStage stage() const noexcept { return stage_; }
    [[nodiscard]] std::uint32_t active_group_mask() const noexcept { return active_group_mask_; }
    [[nodiscard]] std::vector<std::uint32_t> eligible_profile_indices() const;
    [[nodiscard]] std::vector<std::uint32_t> sample_profile_indices(
        std::size_t count,
        bool mirror_learner_sides = true);
    void record(
        std::uint32_t profile_index,
        std::uint64_t wins,
        std::uint64_t losses,
        std::uint64_t draws = 0,
        double recent_win_rate = -1.0,
        double exploit_severity = -1.0);
    [[nodiscard]] double priority(std::uint32_t profile_index) const;
    [[nodiscard]] const MatchupStats& stats(std::uint32_t profile_index) const;
    [[nodiscard]] std::span<const MatchupStats> all_stats() const noexcept { return stats_; }
    [[nodiscard]] std::uint64_t random_state() const noexcept { return random_state_; }
    void restore_state(std::span<const MatchupStats> stats, std::uint64_t random_state);

private:
    std::vector<OpponentProfileParameters> profiles_;
    std::vector<MatchupStats> stats_;
    CurriculumStage stage_ = CurriculumStage::JunFundamentals;
    std::uint32_t active_group_mask_ = 0;
    std::uint64_t random_state_ = 2027;

    [[nodiscard]] std::uint64_t next_random() noexcept;
    [[nodiscard]] std::uint32_t sample_one(
        std::span<const std::uint32_t> eligible,
        double total_priority);
};

void write_matchup_matrix_json(
    const std::filesystem::path& path,
    std::span<const OpponentProfileParameters> profiles,
    std::span<const MatchupStats> stats,
    std::string_view roster_as_of = "2026-07-21");

}  // namespace t8::v2
