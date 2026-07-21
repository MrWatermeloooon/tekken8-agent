#pragma once

#include "t8_v2/sim.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
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

struct GpuRolloutView;

struct PpoUpdateConfig {
    int epochs = 4;
    std::size_t minibatch_size = 4096;
    float learning_rate = 3e-4F;
    float clip_range = 0.2F;
    float value_coefficient = 0.5F;
    float entropy_coefficient = 0.01F;
    float max_gradient_norm = 0.5F;
    float adam_beta1 = 0.9F;
    float adam_beta2 = 0.999F;
    float adam_epsilon = 1e-5F;
};

struct PpoUpdateMetrics {
    float policy_loss = 0.0F;
    float value_loss = 0.0F;
    float entropy = 0.0F;
    float approximate_kl = 0.0F;
    float clip_fraction = 0.0F;
    float gradient_norm = 0.0F;
    std::size_t minibatches = 0;
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

    [[nodiscard]] PpoUpdateMetrics update_ppo(
        const GpuRolloutView& rollout,
        const PpoUpdateConfig& update_config = {},
        std::uint64_t shuffle_seed = 2027,
        void* stream = nullptr);

    void save_checkpoint(const std::filesystem::path& path, void* stream = nullptr) const;
    void load_checkpoint(
        const std::filesystem::path& path,
        bool load_optimizer_state = true,
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

struct GpuRolloutView {
    const float* observations = nullptr;          // [step, environment, observation]
    const std::uint8_t* action_masks = nullptr;   // [step, environment, action]
    const std::int64_t* actions = nullptr;        // [step, environment]
    const float* old_log_probabilities = nullptr;
    const float* old_values = nullptr;
    const float* rewards = nullptr;
    const std::uint8_t* terminated = nullptr;
    const float* advantages = nullptr;
    const float* returns = nullptr;
    std::size_t environment_count = 0;
    std::size_t horizon = 0;
    std::size_t sample_count = 0;
};

class GpuRolloutBuffer {
public:
    GpuRolloutBuffer(
        std::size_t environment_count,
        std::size_t horizon,
        ActorCriticConfig config = {});
    ~GpuRolloutBuffer();

    GpuRolloutBuffer(GpuRolloutBuffer&&) noexcept;
    GpuRolloutBuffer& operator=(GpuRolloutBuffer&&) noexcept;
    GpuRolloutBuffer(const GpuRolloutBuffer&) = delete;
    GpuRolloutBuffer& operator=(const GpuRolloutBuffer&) = delete;

    [[nodiscard]] std::size_t environment_count() const noexcept;
    [[nodiscard]] std::size_t horizon() const noexcept;
    [[nodiscard]] std::size_t sample_count() const noexcept;
    [[nodiscard]] GpuRolloutView device_view() const noexcept;

    void record_policy_device(
        std::size_t step,
        const float* observations,
        const std::uint8_t* action_masks,
        const std::int64_t* actions,
        const float* log_probabilities,
        const float* values,
        void* stream = nullptr);

    void record_outcome_device(
        std::size_t step,
        const float* rewards,
        const std::uint8_t* terminated,
        void* stream = nullptr);

    void compute_gae(
        const float* bootstrap_values,
        float gamma = 0.99F,
        float gae_lambda = 0.95F,
        bool normalize_advantages = true,
        void* stream = nullptr);

    void synchronize(void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_advantages(void* stream = nullptr) const;
    [[nodiscard]] std::vector<float> download_returns(void* stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
