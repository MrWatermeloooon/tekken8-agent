#pragma once

// Batched full-move simulation on the GPU.
//
// Each environment is one fight between two roster characters, simulated
// frame by frame by the same code as the scalar engine (full_combat_core.hpp,
// compiled for the device). Two levels of use:
//
//  * engine level (tests, parity): reset fights, feed engine inputs for both
//    fighters, download states and events;
//  * environment level (training): the learner picks one candidate of its
//    character's full candidate list per decision; the opponent is the
//    scripted full-move opponent or a policy's candidate; observations,
//    candidate masks (both sides), rewards, and episode outcomes stay in VRAM.

#include "t8_v2/full_move_contract.hpp"
#include "t8_v2/full_move_roster.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace t8::v2 {

// Everything that defines the environments between steps (exact resume).
struct FullMoveEnvSnapshot {
    std::vector<full_core::State> states;
    std::vector<int> p1_characters;
    std::vector<int> p2_characters;
    std::vector<std::uint8_t> learner_is_p1;
    std::vector<float> aggression;
    std::vector<int> pending_opponents;
    std::vector<float> pending_aggression;
    std::vector<std::uint8_t> terminated;
    std::vector<int> scenarios;
    std::vector<int> pending_scenarios;
    std::vector<int> pending_learners;
};

class GpuFullMoveSimulator {
public:
    GpuFullMoveSimulator(std::vector<FullMoveCharacter> roster, std::size_t environments,
                         full_core::EnvConfig config = {});
    ~GpuFullMoveSimulator();
    GpuFullMoveSimulator(GpuFullMoveSimulator&&) noexcept;
    GpuFullMoveSimulator& operator=(GpuFullMoveSimulator&&) noexcept;
    GpuFullMoveSimulator(const GpuFullMoveSimulator&) = delete;
    GpuFullMoveSimulator& operator=(const GpuFullMoveSimulator&) = delete;

    [[nodiscard]] std::size_t environment_count() const noexcept;
    [[nodiscard]] const std::vector<FullMoveCharacter>& roster() const noexcept;
    [[nodiscard]] const full_core::EnvConfig& config() const noexcept;
    [[nodiscard]] static constexpr int candidate_stride() noexcept { return kFullMoveMaxCandidates; }
    // full_core::full_move_observation_size(config().memory, config().live_noise). Memory and
    // noise levels are not part of snapshots: a restore starts memory empty and redraws levels.
    [[nodiscard]] int observation_size() const noexcept;

    // Characters (roster indices) of P1 and P2 per environment.
    void set_matchups(std::span<const int> p1_characters, std::span<const int> p2_characters);

    // ---- engine level
    void reset_engine(std::span<const double> p1_x, std::span<const double> p2_x);
    // inputs: [environment][2]; new_round: start the next round (resources that
    // persist carry over) at the given positions before stepping.
    void step_engine(std::span<const full_core::Input> inputs, std::span<const std::uint8_t> new_round = {},
                     std::span<const double> p1_x = {}, std::span<const double> p2_x = {});
    [[nodiscard]] std::vector<full_core::State> download_states() const;
    [[nodiscard]] std::vector<full_core::Events> download_events() const;
    void upload_states(std::span<const full_core::State> states);
    // Legal candidates of one player: [environment][candidate_stride()].
    [[nodiscard]] std::vector<std::uint8_t> download_legal_mask(int player) const;

    // ---- environment level
    // learner_is_p1 per environment (side balance); the learner's character is
    // the matchup's character on that side.
    void set_learner_sides(std::span<const std::uint8_t> learner_is_p1);
    // Opponent character, aggression, and starting scenario (full_core::Scenario)
    // applied at each environment's next reset.
    void set_pending_opponents(std::span<const int> characters, std::span<const float> aggression = {},
                               std::span<const int> scenarios = {});
    // The learner's character (roster index) at each environment's next reset;
    // until set, a reset keeps the learner's current character.
    void set_pending_learners(std::span<const int> characters);
    void reset_all(std::uint64_t seed, std::uint64_t counter);
    // Resets environments whose last step ended the round, then refreshes the
    // observations and masks of every environment.
    void reset_done(std::uint64_t seed, std::uint64_t counter);
    // learner_candidates: device int64 [environment]. opponent_candidates:
    // device int64 [environment] from a policy, or null for the scripted opponent.
    void step(const std::int64_t* learner_candidates, const std::int64_t* opponent_candidates,
              std::uint64_t seed, std::uint64_t counter);

    // Device views (owned by the simulator), refreshed by reset_all, step, and reset_done.
    [[nodiscard]] const float* learner_observations() const noexcept;    // [env][observation_size]
    [[nodiscard]] const std::uint8_t* learner_masks() const noexcept;    // [env][candidate_stride]
    [[nodiscard]] const float* opponent_observations() const noexcept;
    [[nodiscard]] const std::uint8_t* opponent_masks() const noexcept;
    [[nodiscard]] const float* rewards() const noexcept;                 // shaped
    [[nodiscard]] const float* sparse_rewards() const noexcept;
    [[nodiscard]] const std::uint8_t* terminated() const noexcept;
    [[nodiscard]] const std::uint8_t* truncated() const noexcept;        // always 0: rounds end by K.O. or time
    [[nodiscard]] const std::int8_t* outcomes() const noexcept;          // learner-relative, valid where terminated
    // Candidate table (= roster index) of each side's character, per environment,
    // for GpuActorCritic::forward and the rollout's feature_tables.
    [[nodiscard]] const std::int32_t* learner_feature_tables() const noexcept;
    // config().live_noise: each environment's current noise levels.
    [[nodiscard]] std::vector<full_core::NoiseLevels> download_noise_levels() const;
    [[nodiscard]] const std::int32_t* opponent_feature_tables() const noexcept;
    [[nodiscard]] std::vector<std::int32_t> download_learner_feature_tables() const;

    [[nodiscard]] std::vector<std::uint8_t> download_terminated() const;
    [[nodiscard]] std::vector<std::int8_t> download_outcomes() const;
    [[nodiscard]] std::vector<float> download_rewards() const;
    [[nodiscard]] std::vector<int> download_opponent_characters() const;
    [[nodiscard]] std::vector<float> download_learner_observations() const;
    [[nodiscard]] std::vector<std::uint8_t> download_learner_masks() const;
    [[nodiscard]] std::uint64_t frames_simulated() const;  // engine frames since construction

    // Finished rounds since the last call, accumulated on the device:
    // [opponent character][aggression bucket 0-4][learner side 0=P1,1=P2][win, loss, draw].
    [[nodiscard]] std::vector<std::uint32_t> download_and_clear_tally();
    // Finished rounds per starting scenario: [scenario][win, loss, draw]; cleared with the tally.
    [[nodiscard]] std::vector<std::uint32_t> download_scenario_tally() const;
    // Finished rounds per learner character: [character][win, loss, draw]; cleared with the tally.
    [[nodiscard]] std::vector<std::uint32_t> download_learner_tally() const;
    // Finished rounds per scripted bot: [bot][win, loss, draw]; cleared with the tally.
    [[nodiscard]] std::vector<std::uint32_t> download_bot_tally() const;
    static constexpr int kAggressionBuckets = 5;

    [[nodiscard]] FullMoveEnvSnapshot snapshot() const;
    // Restores a snapshot and recomputes observations and masks from it.
    void restore(const FullMoveEnvSnapshot& snapshot);

    void synchronize() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
