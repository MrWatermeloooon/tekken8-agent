#pragma once

#include "t8_v2/sim.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace t8::v2 {

struct ActorCriticConfig {
    int observation_size = static_cast<int>(kObservationSize);
    int action_count = static_cast<int>(kActionCount);
    int hidden_size = 256;
};

struct GpuPolicyOutputView {
    // All pointers remain owned by GpuActorCritic and live in device memory.
    const float* logits = nullptr;       // [environment, action]
    const std::int64_t* actions = nullptr;  // [environment]
    const float* log_probabilities = nullptr;
    const float* values = nullptr;
    const float* entropies = nullptr;
    std::size_t environment_count = 0;
};

// CUDA-native MLP actor-critic. Matrix products use cuBLAS and masked
// categorical sampling runs in CUDA, so simulator observations and selected
// actions never leave VRAM in the production rollout loop.
class GpuActorCritic {
public:
    explicit GpuActorCritic(
        std::size_t capacity,
        ActorCriticConfig config = {},
        std::uint64_t initialization_seed = 2027);
    ~GpuActorCritic();

    GpuActorCritic(GpuActorCritic&&) noexcept;
    GpuActorCritic& operator=(GpuActorCritic&&) noexcept;
    GpuActorCritic(const GpuActorCritic&) = delete;
    GpuActorCritic& operator=(const GpuActorCritic&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] std::size_t parameter_count() const noexcept;
    [[nodiscard]] const ActorCriticConfig& config() const noexcept;

    // observations: row-major [environment_count, observation_size]
    // action_masks: row-major [environment_count, action_count], nonzero=legal
    [[nodiscard]] GpuPolicyOutputView forward(
        const float* device_observations,
        const std::uint8_t* device_action_masks,
        std::size_t environment_count,
        std::uint64_t sampling_seed,
        std::uint64_t sampling_step,
        bool deterministic = false,
        void* stream = nullptr);

    void synchronize(void* stream = nullptr) const;

    // Debug/test transfers only. Training consumes the device view directly.
    [[nodiscard]] std::vector<std::int64_t> download_actions(
        std::size_t environment_count,
        void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_values(
        std::size_t environment_count,
        void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_log_probabilities(
        std::size_t environment_count,
        void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_entropies(
        std::size_t environment_count,
        void* stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
