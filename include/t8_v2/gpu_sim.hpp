#pragma once

#include "t8_v2/sim.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace t8::v2 {

inline constexpr std::size_t kEvaluationStyleCount = 8;

struct GpuEpisodeSummary {
    std::uint64_t episodes = 0;
    std::uint64_t wins = 0;
    std::uint64_t losses = 0;
    std::uint64_t draws = 0;
    std::uint64_t timeouts = 0;
    std::uint64_t stalemates = 0;
    std::uint64_t no_action_timeouts = 0;
    std::uint64_t total_frames = 0;
    double total_damage_dealt = 0.0;
    double total_damage_taken = 0.0;
    std::array<std::uint64_t, kEvaluationStyleCount> style_episodes{};
    std::array<std::uint64_t, kEvaluationStyleCount> style_wins{};
    std::array<std::uint64_t, kEvaluationStyleCount> style_losses{};
    std::array<std::uint64_t, kEvaluationStyleCount> style_draws{};
};

// Device-resident pointers exposed for zero-copy integration with a CUDA
// policy implementation. Observations are row-major [environment, feature].
struct GpuBatchDeviceView {
    float* observations_p1 = nullptr;
    float* observations_p2 = nullptr;
    float* visual_observations_p1 = nullptr;
    float* visual_observations_p2 = nullptr;
    std::uint8_t* action_masks_p1 = nullptr;
    std::uint8_t* action_masks_p2 = nullptr;
    float* rewards_p1 = nullptr;
    float* rewards_p2 = nullptr;
    float* sparse_rewards_p1 = nullptr;
    float* sparse_rewards_p2 = nullptr;
    std::uint8_t* terminated = nullptr;
    std::uint8_t* truncated = nullptr;
    const std::int32_t* winners = nullptr;
    std::size_t environment_count = 0;
};

// GPU-first simulator. State is stored as structure-of-arrays in VRAM and one
// CUDA thread advances one independent fight for an entire decision step.
// The CPU Simulator in sim.hpp exists only as the deterministic parity oracle.
class GpuSimulatorBatch {
public:
    explicit GpuSimulatorBatch(std::size_t environment_count, Config config = {});
    ~GpuSimulatorBatch();

    GpuSimulatorBatch(GpuSimulatorBatch&&) noexcept;
    GpuSimulatorBatch& operator=(GpuSimulatorBatch&&) noexcept;
    GpuSimulatorBatch(const GpuSimulatorBatch&) = delete;
    GpuSimulatorBatch& operator=(const GpuSimulatorBatch&) = delete;

    [[nodiscard]] std::size_t size() const noexcept;
    [[nodiscard]] const Config& config() const noexcept;
    [[nodiscard]] GpuBatchDeviceView device_view() const noexcept;

    // The stream is a cudaStream_t passed as void* to keep CUDA headers out of
    // public C++ consumers. nullptr selects CUDA's default stream.
    void reset(void* stream = nullptr);
    void reset_done(void* stream = nullptr);
    void reset_seeded(std::uint64_t seed, void* stream = nullptr);
    void reset_done_seeded(std::uint64_t seed, void* stream = nullptr);

    // Fast training path: action tensors already live in VRAM. Each tensor is
    // uint8 and has environment_count elements.
    void step_device(
        const std::uint8_t* device_p1_actions,
        const std::uint8_t* device_p2_actions,
        void* stream = nullptr);

    // Native path for PyTorch/JAX argmax tensors, which are commonly int64.
    void step_device_i64(
        const std::int64_t* device_p1_actions,
        const std::int64_t* device_p2_actions,
        void* stream = nullptr);

    // Test/debug path. This performs host-to-device action copies and should
    // not be used in the PPO hot loop.
    void step_host(
        std::span<const std::uint8_t> p1_actions,
        std::span<const std::uint8_t> p2_actions,
        void* stream = nullptr);

    void synchronize(void* stream = nullptr) const;

    // Parity/debug transfers. Production training reads device_view() directly.
    void upload_states(std::span<const State> states, void* stream = nullptr);
    [[nodiscard]] std::vector<State> download_states(void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_observations(int player, void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_visual_observations(
        int player, void* stream = nullptr) const;
    [[nodiscard]] std::vector<std::uint8_t> download_action_masks(int player, void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_rewards(int player, void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_sparse_rewards(int player, void* stream = nullptr) const;
    [[nodiscard]] std::vector<std::uint8_t> download_terminated(void* stream = nullptr) const;
    [[nodiscard]] std::vector<std::int32_t> download_winners(void* stream = nullptr) const;
    [[nodiscard]] GpuEpisodeSummary summarize_episodes(
        int learner_player,
        void* stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

[[nodiscard]] int cuda_device_count() noexcept;

}  // namespace t8::v2
