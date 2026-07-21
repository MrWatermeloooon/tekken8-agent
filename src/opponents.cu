#include "t8_v2/opponents.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace t8::v2 {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

cudaStream_t as_stream(void* stream) { return reinterpret_cast<cudaStream_t>(stream); }
int blocks_for(std::size_t count) { return static_cast<int>((count + kThreads - 1) / kThreads); }

__device__ __forceinline__ std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__global__ void scripted_actions_kernel(
    const float* observations,
    const std::uint8_t* masks,
    std::size_t count,
    std::uint64_t seed,
    std::uint64_t step,
    std::int64_t* actions) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    const std::size_t obs = lane * kObservationSize;
    const std::size_t mask = lane * kActionCount;
    if (!masks[mask + static_cast<int>(Action::Jab)]) {
        actions[lane] = static_cast<std::int64_t>(Action::Neutral);
        return;
    }
    const float distance = observations[obs + 3] * 7.2F;
    const bool opponent_attacking = observations[obs + 12] > 0.5F;
    const bool throw_threat = observations[obs + 17] > 0.5F;
    const int random = static_cast<int>(mix(seed ^ mix(step) ^ mix(lane)) % 100ULL);
    const int style = static_cast<int>(lane % 8);
    Action selected = Action::Neutral;

    if (throw_threat) {
        selected = Action::ThrowBreak12;
    } else if (style == 0) {
        if (distance > 1.25F) selected = Action::DashForward;
        else if (distance > 0.95F) selected = random < 55 ? Action::F2 : Action::DashForward;
        else if (random < 30) selected = Action::Jab;
        else if (random < 55) selected = Action::Df1;
        else if (random < 75) selected = Action::Db3;
        else if (random < 90) selected = Action::Throw;
        else selected = Action::Hopkick;
    } else if (style == 1) {
        if (opponent_attacking && distance < 1.3F) selected = Action::BlockHigh;
        else if (distance < 0.62F) selected = Action::DashBack;
        else if (distance < 1.28F) selected = Action::F2;
        else selected = Action::WalkForward;
    } else if (style == 2) {
        if (opponent_attacking) selected = random < 25 ? Action::BlockLow : Action::BlockHigh;
        else if (distance < 0.92F) selected = random < 65 ? Action::Df1 : Action::Hopkick;
        else selected = Action::DashForward;
    } else if (style == 3) {
        if (distance < 0.50F) selected = random < 70 ? Action::Throw : Action::Jab;
        else if (distance < 0.90F) selected = Action::Jab;
        else selected = Action::DashForward;
    } else if (style == 4) {
        if (distance > 0.95F) selected = Action::DashForward;
        else if (random < 50) selected = Action::Db3;
        else if (random < 82) selected = Action::Df1;
        else selected = Action::Hopkick;
    } else if (style == 5) {
        if (opponent_attacking) selected = random < 50 ? Action::SidestepLeft : Action::SidestepRight;
        else if (distance < 0.92F) selected = random < 55 ? Action::Hopkick : Action::Db3;
        else selected = Action::DashForward;
    } else if (style == 6) {
        if (opponent_attacking) selected = random < 20 ? Action::BlockLow : Action::BlockHigh;
        else if (distance < 0.82F) selected = Action::Jab;
        else if (distance > 1.45F) selected = Action::WalkForward;
        else selected = Action::BlockHigh;
    } else {
        if (opponent_attacking && random < 45) selected = random < 12 ? Action::BlockLow : Action::BlockHigh;
        else if (distance > 1.15F) selected = random < 70 ? Action::DashForward : Action::F2;
        else if (random < 22) selected = Action::Jab;
        else if (random < 44) selected = Action::Df1;
        else if (random < 62) selected = Action::Db3;
        else if (random < 74) selected = Action::Throw;
        else if (random < 88) selected = Action::Hopkick;
        else selected = random < 94 ? Action::SidestepLeft : Action::SidestepRight;
    }
    actions[lane] = static_cast<std::int64_t>(selected);
}

}  // namespace

struct GpuScriptedOpponent::Impl {
    std::size_t capacity;
    std::int64_t* actions = nullptr;
    explicit Impl(std::size_t requested) : capacity(requested) {
        if (capacity == 0) throw std::invalid_argument("opponent capacity must be positive");
        check_cuda(cudaMalloc(&actions, sizeof(std::int64_t) * capacity), "allocate scripted actions");
    }
    ~Impl() { cudaFree(actions); }
};

GpuScriptedOpponent::GpuScriptedOpponent(std::size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}
GpuScriptedOpponent::~GpuScriptedOpponent() = default;
GpuScriptedOpponent::GpuScriptedOpponent(GpuScriptedOpponent&&) noexcept = default;
GpuScriptedOpponent& GpuScriptedOpponent::operator=(GpuScriptedOpponent&&) noexcept = default;
std::size_t GpuScriptedOpponent::capacity() const noexcept { return impl_->capacity; }

const std::int64_t* GpuScriptedOpponent::actions_device(
    const float* observations,
    const std::uint8_t* masks,
    std::size_t count,
    std::uint64_t seed,
    std::uint64_t step,
    void* stream) {
    if (!observations || !masks || count == 0 || count > impl_->capacity) {
        throw std::invalid_argument("invalid scripted-opponent input");
    }
    scripted_actions_kernel<<<blocks_for(count), kThreads, 0, as_stream(stream)>>>(
        observations, masks, count, seed, step, impl_->actions);
    check_cuda(cudaGetLastError(), "launch scripted opponent kernel");
    return impl_->actions;
}

void GpuScriptedOpponent::synchronize(void* stream) const {
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize scripted opponent");
}

std::vector<std::int64_t> GpuScriptedOpponent::download_actions(
    std::size_t count,
    void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download exceeds opponent capacity");
    std::vector<std::int64_t> host(count);
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(host.data(), impl_->actions, sizeof(std::int64_t) * count,
                               cudaMemcpyDeviceToHost, cuda_stream), "download scripted actions");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize scripted action download");
    return host;
}

}  // namespace t8::v2
