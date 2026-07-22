#pragma once

#include "t8_v2/roster.hpp"
#include "t8_v2/sim.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

namespace t8::v2 {

enum class ScriptedOpponentSet : std::uint8_t {
    TrainingV1,
    HeldOutV2,
};

// Frozen GPU benchmark/training mixture. Lanes rotate through eight distinct
// pressure, keepout, defense, punishment, throw, low, movement, and adaptive
// styles. The policy is deterministic for (seed, step, lane).
class GpuScriptedOpponent {
public:
    explicit GpuScriptedOpponent(std::size_t capacity);
    ~GpuScriptedOpponent();

    GpuScriptedOpponent(GpuScriptedOpponent&&) noexcept;
    GpuScriptedOpponent& operator=(GpuScriptedOpponent&&) noexcept;
    GpuScriptedOpponent(const GpuScriptedOpponent&) = delete;
    GpuScriptedOpponent& operator=(const GpuScriptedOpponent&) = delete;

    [[nodiscard]] std::size_t capacity() const noexcept;
    // Profile tables and lane assignments are uploaded outside the rollout hot
    // loop. Once configured, the CUDA kernel uses the 2,100-profile catalog
    // instead of the frozen eight-style compatibility mixture.
    void set_profiles(std::span<const OpponentProfileParameters> profiles, void* stream = nullptr);
    void set_profile_assignments(std::span<const std::uint32_t> assignments, void* stream = nullptr);
    void set_action_history(std::span<const std::int64_t> actions, void* stream = nullptr);
    [[nodiscard]] bool uses_profiles() const noexcept;
    [[nodiscard]] const std::uint32_t* profile_assignments_device() const noexcept;
    [[nodiscard]] const OpponentProfileParameters* profiles_device() const noexcept;
    [[nodiscard]] const std::int64_t* actions_buffer_device() const noexcept;
    [[nodiscard]] std::size_t profile_count() const noexcept;
    [[nodiscard]] const std::int64_t* actions_device(
        const float* observations,
        const std::uint8_t* action_masks,
        std::size_t environment_count,
        std::uint64_t seed,
        std::uint64_t step,
        ScriptedOpponentSet opponent_set = ScriptedOpponentSet::TrainingV1,
        void* stream = nullptr);
    void synchronize(void* stream = nullptr) const;
    [[nodiscard]] std::vector<std::int64_t> download_actions(
        std::size_t environment_count,
        void* stream = nullptr) const;
    [[nodiscard]] std::vector<std::uint32_t> download_profile_assignments(
        std::size_t environment_count,
        void* stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
