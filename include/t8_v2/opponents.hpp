#pragma once

#include "t8_v2/sim.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace t8::v2 {

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
    [[nodiscard]] const std::int64_t* actions_device(
        const float* observations,
        const std::uint8_t* action_masks,
        std::size_t environment_count,
        std::uint64_t seed,
        std::uint64_t step,
        void* stream = nullptr);
    void synchronize(void* stream = nullptr) const;
    [[nodiscard]] std::vector<std::int64_t> download_actions(
        std::size_t environment_count,
        void* stream = nullptr) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace t8::v2
