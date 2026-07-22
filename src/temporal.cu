#include "t8_v2/temporal.hpp"

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

__device__ float hit_level_for_action(std::int64_t action) {
    if (action == 18) return 1.0F / 4.0F;
    if (action == 19 || action == 20 || action == 22) return 2.0F / 4.0F;
    if (action == 21) return 3.0F / 4.0F;
    if (action == 23) return 1.0F;
    return 0.0F;
}

__global__ void reset_all_kernel(
    float* history,
    std::int64_t* previous_actions,
    std::int32_t* repeated_action_frames,
    float* previous_own_health,
    float* previous_opponent_health,
    float* previous_distance,
    std::uint8_t* valid,
    std::size_t count) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    for (std::size_t index = 0; index < kTemporalHistoryLength * kTemporalFeaturesPerStep; ++index) {
        history[lane * kTemporalHistoryLength * kTemporalFeaturesPerStep + index] = 0.0F;
    }
    previous_actions[lane] = 0;
    repeated_action_frames[lane] = 0;
    previous_own_health[lane] = 0.0F;
    previous_opponent_health[lane] = 0.0F;
    previous_distance[lane] = 0.0F;
    valid[lane] = 0;
}

__global__ void reset_done_kernel(
    const std::uint8_t* done,
    float* history,
    std::int64_t* previous_actions,
    std::int32_t* repeated_action_frames,
    float* previous_own_health,
    float* previous_opponent_health,
    float* previous_distance,
    std::uint8_t* valid,
    std::size_t count) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count || done[lane] == 0) return;
    for (std::size_t index = 0; index < kTemporalHistoryLength * kTemporalFeaturesPerStep; ++index) {
        history[lane * kTemporalHistoryLength * kTemporalFeaturesPerStep + index] = 0.0F;
    }
    previous_actions[lane] = 0;
    repeated_action_frames[lane] = 0;
    previous_own_health[lane] = 0.0F;
    previous_opponent_health[lane] = 0.0F;
    previous_distance[lane] = 0.0F;
    valid[lane] = 0;
}

__global__ void encode_kernel(
    const float* base,
    std::size_t base_size,
    const OpponentProfileParameters* profiles,
    std::size_t profile_count,
    const std::uint32_t* assignments,
    const std::int64_t* opponent_actions,
    float* history,
    std::int64_t* previous_actions,
    std::int32_t* repeated_action_frames,
    float* previous_own_health,
    float* previous_opponent_health,
    float* previous_distance,
    std::uint8_t* valid,
    float* output,
    std::size_t count) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    const auto profile_index = assignments[lane];
    if (profile_index >= profile_count) return;
    const auto& profile = profiles[profile_index];
    const std::size_t input_base = lane * base_size;
    const std::size_t output_size = base_size + kMatchupContextSize;
    const std::size_t output_base = lane * output_size;
    for (std::size_t feature = 0; feature < base_size; ++feature) {
        output[output_base + feature] = base[input_base + feature];
    }
    std::size_t cursor = output_base + base_size;
    for (std::size_t bit = 0; bit < 6; ++bit) {
        output[cursor++] = (profile.character_id & (1U << bit)) != 0 ? 1.0F : -1.0F;
    }
    const float angle = 6.283185307179586F * static_cast<float>(profile.character_id % 64U) / 64.0F;
    output[cursor++] = sinf(angle);
    output[cursor++] = cosf(angle);
    for (std::size_t archetype = 0; archetype < kOpponentArchetypeCount; ++archetype) {
        output[cursor++] = profile.archetype_id == archetype ? 1.0F : 0.0F;
    }

    const std::size_t history_base = lane * kTemporalHistoryLength * kTemporalFeaturesPerStep;
    for (std::size_t step = 0; step + 1 < kTemporalHistoryLength; ++step) {
        for (std::size_t feature = 0; feature < kTemporalFeaturesPerStep; ++feature) {
            history[history_base + step * kTemporalFeaturesPerStep + feature] =
                history[history_base + (step + 1) * kTemporalFeaturesPerStep + feature];
        }
    }
    const std::int64_t action = valid[lane] != 0 ? opponent_actions[lane] : 0;
    repeated_action_frames[lane] = valid[lane] != 0 && previous_actions[lane] == action
        ? min(60, repeated_action_frames[lane] + 4) : 0;
    const float own_health = base[input_base + 0];
    const float opponent_health = base[input_base + 1];
    const float distance = base_size == 13 ? base[input_base + 4] / 7.2F : base[input_base + 3];
    const float outcome = valid[lane] == 0 ? 0.0F :
        fminf(1.0F, fmaxf(-1.0F,
            (previous_opponent_health[lane] - opponent_health) -
            (previous_own_health[lane] - own_health)) * 10.0F);
    const float side_movement = base_size == 13
        ? fminf(1.0F, fmaxf(-1.0F, base[input_base + 6]))
        : (valid[lane] == 0 ? 0.0F : fminf(1.0F, fmaxf(-1.0F, distance - previous_distance[lane])));
    const float animation_phase = base_size == 13
        ? fminf(1.0F, fmaxf(0.0F, base[input_base + 12]))
        : 1.0F - fminf(1.0F, fmaxf(0.0F, base[input_base + 14]));
    const bool stance_action = action == 8 || action == 9 || action == 10 || action == 11 || action == 19;
    const float stance = stance_action && profile.stance_entry_frequency >= 0.5F
        ? static_cast<float>(profile.archetype_id + 1U) / 31.0F : 0.0F;
    const std::size_t newest = history_base + (kTemporalHistoryLength - 1) * kTemporalFeaturesPerStep;
    history[newest + 0] = fminf(1.0F, fmaxf(0.0F, static_cast<float>(action) / 23.0F));
    history[newest + 1] = animation_phase;
    history[newest + 2] = stance;
    history[newest + 3] = hit_level_for_action(action);
    history[newest + 4] = static_cast<float>(repeated_action_frames[lane]) / 60.0F;
    history[newest + 5] = outcome;
    history[newest + 6] = fminf(1.0F, fmaxf(0.0F, distance));
    history[newest + 7] = side_movement;
    for (std::size_t index = 0; index < kTemporalHistoryLength * kTemporalFeaturesPerStep; ++index) {
        output[cursor + index] = history[history_base + index];
    }
    previous_actions[lane] = action;
    previous_own_health[lane] = own_health;
    previous_opponent_health[lane] = opponent_health;
    previous_distance[lane] = distance;
    valid[lane] = 1;
}

}  // namespace

struct GpuTemporalMatchupEncoder::Impl {
    std::size_t capacity;
    std::size_t base_size;
    std::size_t output_size;
    float* history = nullptr;
    std::int64_t* previous_actions = nullptr;
    std::int32_t* repeated_action_frames = nullptr;
    float* previous_own_health = nullptr;
    float* previous_opponent_health = nullptr;
    float* previous_distance = nullptr;
    std::uint8_t* valid = nullptr;
    float* output = nullptr;

    Impl(std::size_t requested, std::size_t requested_base)
        : capacity(requested), base_size(requested_base), output_size(requested_base + kMatchupContextSize) {
        if (capacity == 0 || (base_size != 13 && base_size != 19)) {
            throw std::invalid_argument("temporal encoder requires positive capacity and a 13- or 19-feature base");
        }
        try {
            check_cuda(cudaMalloc(&history, sizeof(float) * capacity * kTemporalHistoryLength * kTemporalFeaturesPerStep), "allocate temporal history");
            check_cuda(cudaMalloc(&previous_actions, sizeof(std::int64_t) * capacity), "allocate previous opponent actions");
            check_cuda(cudaMalloc(&repeated_action_frames, sizeof(std::int32_t) * capacity), "allocate opponent delay history");
            check_cuda(cudaMalloc(&previous_own_health, sizeof(float) * capacity), "allocate previous own health");
            check_cuda(cudaMalloc(&previous_opponent_health, sizeof(float) * capacity), "allocate previous opponent health");
            check_cuda(cudaMalloc(&previous_distance, sizeof(float) * capacity), "allocate previous distance");
            check_cuda(cudaMalloc(&valid, sizeof(std::uint8_t) * capacity), "allocate temporal validity flags");
            check_cuda(cudaMalloc(&output, sizeof(float) * capacity * output_size), "allocate matchup observations");
        } catch (...) {
            release();
            throw;
        }
    }

    void release() noexcept {
        cudaFree(output); cudaFree(valid); cudaFree(previous_distance);
        cudaFree(previous_opponent_health); cudaFree(previous_own_health);
        cudaFree(repeated_action_frames); cudaFree(previous_actions); cudaFree(history);
        output = nullptr; valid = nullptr; previous_distance = nullptr;
        previous_opponent_health = nullptr; previous_own_health = nullptr;
        repeated_action_frames = nullptr; previous_actions = nullptr; history = nullptr;
    }
    ~Impl() { release(); }
};

GpuTemporalMatchupEncoder::GpuTemporalMatchupEncoder(std::size_t capacity, std::size_t base_size)
    : impl_(std::make_unique<Impl>(capacity, base_size)) { reset(); synchronize(); }
GpuTemporalMatchupEncoder::~GpuTemporalMatchupEncoder() = default;
GpuTemporalMatchupEncoder::GpuTemporalMatchupEncoder(GpuTemporalMatchupEncoder&&) noexcept = default;
GpuTemporalMatchupEncoder& GpuTemporalMatchupEncoder::operator=(GpuTemporalMatchupEncoder&&) noexcept = default;
std::size_t GpuTemporalMatchupEncoder::capacity() const noexcept { return impl_->capacity; }
std::size_t GpuTemporalMatchupEncoder::base_observation_size() const noexcept { return impl_->base_size; }
std::size_t GpuTemporalMatchupEncoder::observation_size() const noexcept { return impl_->output_size; }

void GpuTemporalMatchupEncoder::reset(void* stream) {
    reset_all_kernel<<<blocks_for(impl_->capacity), kThreads, 0, as_stream(stream)>>>(
        impl_->history, impl_->previous_actions, impl_->repeated_action_frames,
        impl_->previous_own_health, impl_->previous_opponent_health, impl_->previous_distance,
        impl_->valid, impl_->capacity);
    check_cuda(cudaGetLastError(), "launch temporal reset kernel");
}

void GpuTemporalMatchupEncoder::reset_done(
    const std::uint8_t* done, std::size_t count, void* stream) {
    if (!done || count == 0 || count > impl_->capacity) throw std::invalid_argument("invalid temporal reset-done input");
    reset_done_kernel<<<blocks_for(count), kThreads, 0, as_stream(stream)>>>(
        done, impl_->history, impl_->previous_actions, impl_->repeated_action_frames,
        impl_->previous_own_health, impl_->previous_opponent_health, impl_->previous_distance,
        impl_->valid, count);
    check_cuda(cudaGetLastError(), "launch temporal reset-done kernel");
}

const float* GpuTemporalMatchupEncoder::encode(
    const float* base, const OpponentProfileParameters* profiles, std::size_t profile_count,
    const std::uint32_t* assignments, const std::int64_t* actions,
    std::size_t count, void* stream) {
    if (!base || !profiles || profile_count == 0 || !assignments || !actions ||
        count == 0 || count > impl_->capacity) {
        throw std::invalid_argument("invalid temporal encoder input");
    }
    encode_kernel<<<blocks_for(count), kThreads, 0, as_stream(stream)>>>(
        base, impl_->base_size, profiles, profile_count, assignments, actions,
        impl_->history, impl_->previous_actions, impl_->repeated_action_frames,
        impl_->previous_own_health, impl_->previous_opponent_health, impl_->previous_distance,
        impl_->valid, impl_->output, count);
    check_cuda(cudaGetLastError(), "launch temporal encode kernel");
    return impl_->output;
}

void GpuTemporalMatchupEncoder::synchronize(void* stream) const {
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize temporal encoder");
}

TemporalEncoderState GpuTemporalMatchupEncoder::download_state(void* stream) const {
    TemporalEncoderState state{};
    state.history.resize(impl_->capacity * kTemporalHistoryLength * kTemporalFeaturesPerStep);
    state.previous_actions.resize(impl_->capacity);
    state.repeated_action_frames.resize(impl_->capacity);
    state.previous_own_health.resize(impl_->capacity);
    state.previous_opponent_health.resize(impl_->capacity);
    state.previous_distance.resize(impl_->capacity);
    state.valid.resize(impl_->capacity);
    const auto cuda_stream = as_stream(stream);
    const auto copy = [&](void* host, const void* device, std::size_t bytes, const char* name) {
        check_cuda(cudaMemcpyAsync(host, device, bytes, cudaMemcpyDeviceToHost, cuda_stream), name);
    };
    copy(state.history.data(), impl_->history, sizeof(float) * state.history.size(), "download temporal history");
    copy(state.previous_actions.data(), impl_->previous_actions, sizeof(std::int64_t) * state.previous_actions.size(), "download previous actions");
    copy(state.repeated_action_frames.data(), impl_->repeated_action_frames, sizeof(std::int32_t) * state.repeated_action_frames.size(), "download delay history");
    copy(state.previous_own_health.data(), impl_->previous_own_health, sizeof(float) * state.previous_own_health.size(), "download previous own health");
    copy(state.previous_opponent_health.data(), impl_->previous_opponent_health, sizeof(float) * state.previous_opponent_health.size(), "download previous opponent health");
    copy(state.previous_distance.data(), impl_->previous_distance, sizeof(float) * state.previous_distance.size(), "download previous distance");
    copy(state.valid.data(), impl_->valid, sizeof(std::uint8_t) * state.valid.size(), "download temporal validity");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize temporal state download");
    return state;
}

void GpuTemporalMatchupEncoder::upload_state(const TemporalEncoderState& state, void* stream) {
    const std::size_t history_count = impl_->capacity * kTemporalHistoryLength * kTemporalFeaturesPerStep;
    if (state.history.size() != history_count || state.previous_actions.size() != impl_->capacity ||
        state.repeated_action_frames.size() != impl_->capacity || state.previous_own_health.size() != impl_->capacity ||
        state.previous_opponent_health.size() != impl_->capacity || state.previous_distance.size() != impl_->capacity ||
        state.valid.size() != impl_->capacity) {
        throw std::invalid_argument("temporal state dimensions do not match encoder capacity");
    }
    const auto cuda_stream = as_stream(stream);
    const auto copy = [&](void* device, const void* host, std::size_t bytes, const char* name) {
        check_cuda(cudaMemcpyAsync(device, host, bytes, cudaMemcpyHostToDevice, cuda_stream), name);
    };
    copy(impl_->history, state.history.data(), sizeof(float) * state.history.size(), "upload temporal history");
    copy(impl_->previous_actions, state.previous_actions.data(), sizeof(std::int64_t) * state.previous_actions.size(), "upload previous actions");
    copy(impl_->repeated_action_frames, state.repeated_action_frames.data(), sizeof(std::int32_t) * state.repeated_action_frames.size(), "upload delay history");
    copy(impl_->previous_own_health, state.previous_own_health.data(), sizeof(float) * state.previous_own_health.size(), "upload previous own health");
    copy(impl_->previous_opponent_health, state.previous_opponent_health.data(), sizeof(float) * state.previous_opponent_health.size(), "upload previous opponent health");
    copy(impl_->previous_distance, state.previous_distance.data(), sizeof(float) * state.previous_distance.size(), "upload previous distance");
    copy(impl_->valid, state.valid.data(), sizeof(std::uint8_t) * state.valid.size(), "upload temporal validity");
}

}  // namespace t8::v2
