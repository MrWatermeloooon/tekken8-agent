#pragma once

#include "t8_v2/roster.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace t8::v2 {

inline constexpr std::size_t kCharacterEmbeddingSize = 8;
inline constexpr std::size_t kTemporalHistoryLength = 8;
inline constexpr std::size_t kTemporalFeaturesPerStep = 8;
inline constexpr std::size_t kMatchupContextSize =
    kCharacterEmbeddingSize + kOpponentArchetypeCount +
    kTemporalHistoryLength * kTemporalFeaturesPerStep;
inline constexpr std::size_t kMatchupVisualObservationSize = 13 + kMatchupContextSize;
inline constexpr std::size_t kMatchupPrivilegedObservationSize = 19 + kMatchupContextSize;

struct TemporalEncoderState {
    std::vector<float> history;
    std::vector<std::int64_t> previous_actions;
    std::vector<std::int32_t> repeated_action_frames;
    std::vector<float> previous_own_health;
    std::vector<float> previous_opponent_health;
    std::vector<float> previous_distance;
    std::vector<std::uint8_t> valid;
};

// Frame stacking and matchup conditioning stay entirely in VRAM. One CUDA
// thread updates one lane and produces a row-major augmented observation.
class GpuTemporalMatchupEncoder {
public:
    GpuTemporalMatchupEncoder(std::size_t capacity, std::size_t base_observation_size);
    ~GpuTemporalMatchupEncoder();

    GpuTemporalMatchupEncoder(GpuTemporalMatchupEncoder&&) noexcept;
    GpuTemporalMatchupEncoder& operator=(GpuTemporalMatchupEncoder&&) noexcept;
    GpuTemporalMatchupEncoder(const GpuTemporalMatchupEncoder&) = delete;
    GpuTemporalMatchupEncoder& operator=(const GpuTemporalMatchupEncoder&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t base_observation_size() const noexcept;
    [[nodiscard]] std::size_t observation_size() const noexcept;
    void reset(void* stream = nullptr);
    void reset_done(const std::uint8_t* device_done, std::size_t count, void* stream = nullptr);
    [[nodiscard]] const float* encode(
        const float* base_observations,
        const OpponentProfileParameters* profiles,
        std::size_t profile_count,
        const std::uint32_t* profile_assignments,
        const std::int64_t* previous_opponent_actions,
        std::size_t count,
        void* stream = nullptr);
    void synchronize(void* stream = nullptr) const;
    [[nodiscard]] TemporalEncoderState download_state(void* stream = nullptr) const;
    void upload_state(const TemporalEncoderState& state, void* stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
