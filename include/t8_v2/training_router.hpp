#pragma once

#include "t8_v2/gpu_sim.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>

namespace t8::v2 {

struct GpuRoutedObservationView {
    const float* learner_observations = nullptr;
    const std::uint8_t* learner_action_masks = nullptr;
    const float* opponent_observations = nullptr;
    const std::uint8_t* opponent_action_masks = nullptr;
    std::size_t environment_count = 0;
};

struct GpuRoutedActionView {
    const std::int64_t* p1_actions = nullptr;
    const std::int64_t* p2_actions = nullptr;
    std::size_t environment_count = 0;
};

// Routes blocks of eight opponent styles between learner-as-P1 and
// learner-as-P2 lanes. Each side therefore sees every scripted style while
// rollouts, actions, and rewards remain entirely device resident.
class GpuLearnerSideRouter {
public:
    explicit GpuLearnerSideRouter(std::size_t capacity);
    ~GpuLearnerSideRouter();

    GpuLearnerSideRouter(GpuLearnerSideRouter&&) noexcept;
    GpuLearnerSideRouter& operator=(GpuLearnerSideRouter&&) noexcept;
    GpuLearnerSideRouter(const GpuLearnerSideRouter&) = delete;
    GpuLearnerSideRouter& operator=(const GpuLearnerSideRouter&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept;
    [[nodiscard]] GpuRoutedObservationView select_observations(
        const GpuBatchDeviceView& simulator,
        std::size_t environment_count,
        void* stream = nullptr);
    // Routes the 13-feature screen-compatible tensors to the learner while
    // retaining the 19-feature privileged tensors for scripted opponents.
    [[nodiscard]] GpuRoutedObservationView select_visual_observations(
        const GpuBatchDeviceView& simulator,
        std::size_t environment_count,
        void* stream = nullptr);
    [[nodiscard]] GpuRoutedActionView route_actions(
        const std::int64_t* learner_actions,
        const std::int64_t* opponent_actions,
        std::size_t environment_count,
        void* stream = nullptr);
    [[nodiscard]] const float* select_rewards(
        const float* p1_rewards,
        const float* p2_rewards,
        std::size_t environment_count,
        void* stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
