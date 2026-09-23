#include "t8_v2/training_router.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>

namespace t8::v2 {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

cudaStream_t as_stream(void* stream) { return reinterpret_cast<cudaStream_t>(stream); }
int blocks_for(std::size_t count) {
    return static_cast<int>((count + static_cast<std::size_t>(kThreads) - 1) / kThreads);
}

__device__ bool learner_is_p1(std::size_t lane) {
    return ((lane / kEvaluationStyleCount) & 1ULL) == 0ULL;
}

__global__ void select_observations_kernel(
    const float* learner_p1_observations,
    const float* learner_p2_observations,
    const float* opponent_p1_observations,
    const float* opponent_p2_observations,
    std::size_t learner_feature_count,
    std::size_t opponent_feature_count,
    const std::uint8_t* p1_masks,
    const std::uint8_t* p2_masks,
    std::size_t count,
    float* learner_observations,
    float* opponent_observations,
    std::uint8_t* learner_masks,
    std::uint8_t* opponent_masks) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    const bool learner_p1 = learner_is_p1(lane);
    const float* learner_source = learner_p1 ? learner_p1_observations : learner_p2_observations;
    const float* opponent_source = learner_p1 ? opponent_p2_observations : opponent_p1_observations;
    const std::uint8_t* learner_mask_source = learner_p1 ? p1_masks : p2_masks;
    const std::uint8_t* opponent_mask_source = learner_p1 ? p2_masks : p1_masks;
    for (std::size_t feature = 0; feature < learner_feature_count; ++feature) {
        learner_observations[lane * learner_feature_count + feature] =
            learner_source[lane * learner_feature_count + feature];
    }
    for (std::size_t feature = 0; feature < opponent_feature_count; ++feature) {
        opponent_observations[lane * opponent_feature_count + feature] =
            opponent_source[lane * opponent_feature_count + feature];
    }
    for (std::size_t action = 0; action < kActionCount; ++action) {
        learner_masks[lane * kActionCount + action] =
            learner_mask_source[lane * kActionCount + action];
        opponent_masks[lane * kActionCount + action] =
            opponent_mask_source[lane * kActionCount + action];
    }
}

__global__ void route_actions_kernel(
    const std::int64_t* learner_actions,
    const std::int64_t* opponent_actions,
    std::size_t count,
    std::int64_t* p1_actions,
    std::int64_t* p2_actions) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    const bool learner_p1 = learner_is_p1(lane);
    p1_actions[lane] = learner_p1 ? learner_actions[lane] : opponent_actions[lane];
    p2_actions[lane] = learner_p1 ? opponent_actions[lane] : learner_actions[lane];
}

__global__ void mix_self_play_actions_kernel(
    const std::int64_t* latest_actions,
    const std::int64_t* best_older_actions,
    std::size_t count,
    std::int64_t* mixed_actions) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    mixed_actions[lane] = lane % 5 == 0 ? best_older_actions[lane] : latest_actions[lane];
}

__global__ void select_rewards_kernel(
    const float* p1_rewards,
    const float* p2_rewards,
    std::size_t count,
    float* learner_rewards) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    learner_rewards[lane] = learner_is_p1(lane) ? p1_rewards[lane] : p2_rewards[lane];
}

__global__ void tally_outcomes_kernel(
    const float* sparse_rewards_p1,
    const float* sparse_rewards_p2,
    const std::uint8_t* terminated,
    const std::uint32_t* profile_assignments,
    std::size_t profile_count,
    std::size_t count,
    unsigned long long* tally) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count || terminated[lane] == 0) return;
    const std::uint32_t profile = profile_assignments[lane];
    if (profile >= profile_count) return;
    const float outcome = learner_is_p1(lane) ? sparse_rewards_p1[lane] : sparse_rewards_p2[lane];
    const int column = outcome > 0.0F ? 0 : (outcome < 0.0F ? 1 : 2);
    atomicAdd(tally + static_cast<std::size_t>(profile) * 3 + column, 1ULL);
}

}  // namespace

struct GpuLearnerSideRouter::Impl {
    std::size_t capacity;
    float* learner_observations = nullptr;
    float* opponent_observations = nullptr;
    std::uint8_t* learner_masks = nullptr;
    std::uint8_t* opponent_masks = nullptr;
    std::int64_t* p1_actions = nullptr;
    std::int64_t* p2_actions = nullptr;
    std::int64_t* mixed_opponent_actions = nullptr;
    float* learner_rewards = nullptr;
    unsigned long long* outcome_tally = nullptr;
    std::size_t outcome_tally_profiles = 0;

    void ensure_outcome_tally(std::size_t profile_count, cudaStream_t stream) {
        if (profile_count == outcome_tally_profiles) return;
        if (outcome_tally_profiles != 0) {
            throw std::invalid_argument("outcome tally profile count changed while counting");
        }
        check_cuda(cudaMalloc(&outcome_tally, sizeof(unsigned long long) * profile_count * 3),
                   "allocate matchup outcome tally");
        check_cuda(cudaMemsetAsync(outcome_tally, 0, sizeof(unsigned long long) * profile_count * 3,
                                   stream), "clear matchup outcome tally");
        outcome_tally_profiles = profile_count;
    }

    explicit Impl(std::size_t requested_capacity) : capacity(requested_capacity) {
        if (capacity == 0) throw std::invalid_argument("router capacity must be positive");
        try {
        check_cuda(cudaMalloc(&learner_observations, sizeof(float) * capacity * kObservationSize),
                   "allocate routed learner observations");
        check_cuda(cudaMalloc(&opponent_observations, sizeof(float) * capacity * kObservationSize),
                   "allocate routed opponent observations");
        check_cuda(cudaMalloc(&learner_masks, sizeof(std::uint8_t) * capacity * kActionCount),
                   "allocate routed learner masks");
        check_cuda(cudaMalloc(&opponent_masks, sizeof(std::uint8_t) * capacity * kActionCount),
                   "allocate routed opponent masks");
        check_cuda(cudaMalloc(&p1_actions, sizeof(std::int64_t) * capacity),
                   "allocate routed P1 actions");
        check_cuda(cudaMalloc(&p2_actions, sizeof(std::int64_t) * capacity),
                   "allocate routed P2 actions");
        check_cuda(cudaMalloc(&mixed_opponent_actions, sizeof(std::int64_t) * capacity),
                   "allocate mixed self-play actions");
        check_cuda(cudaMalloc(&learner_rewards, sizeof(float) * capacity),
                   "allocate routed learner rewards");
        } catch (...) {
            release();
            throw;
        }
    }

    void release() noexcept {
        cudaFree(outcome_tally);
        cudaFree(learner_rewards);
        cudaFree(mixed_opponent_actions);
        cudaFree(p2_actions);
        cudaFree(p1_actions);
        cudaFree(opponent_masks);
        cudaFree(learner_masks);
        cudaFree(opponent_observations);
        cudaFree(learner_observations);
    }

    ~Impl() { release(); }
};

GpuLearnerSideRouter::GpuLearnerSideRouter(std::size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}
GpuLearnerSideRouter::~GpuLearnerSideRouter() = default;
GpuLearnerSideRouter::GpuLearnerSideRouter(GpuLearnerSideRouter&&) noexcept = default;
GpuLearnerSideRouter& GpuLearnerSideRouter::operator=(GpuLearnerSideRouter&&) noexcept = default;
std::size_t GpuLearnerSideRouter::capacity() const noexcept { return impl_->capacity; }

GpuRoutedObservationView GpuLearnerSideRouter::select_observations(
    const GpuBatchDeviceView& simulator,
    std::size_t environment_count,
    void* stream) {
    if (environment_count == 0 || environment_count > impl_->capacity ||
        !simulator.observations_p1 || !simulator.observations_p2 ||
        !simulator.action_masks_p1 || !simulator.action_masks_p2) {
        throw std::invalid_argument("invalid routed observation input");
    }
    select_observations_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        simulator.observations_p1, simulator.observations_p2,
        simulator.observations_p1, simulator.observations_p2,
        kObservationSize, kObservationSize,
        simulator.action_masks_p1, simulator.action_masks_p2,
        environment_count, impl_->learner_observations, impl_->opponent_observations,
        impl_->learner_masks, impl_->opponent_masks);
    check_cuda(cudaGetLastError(), "launch side-balanced observation router");
    return {impl_->learner_observations, impl_->learner_masks,
            impl_->opponent_observations, impl_->opponent_masks, environment_count};
}

GpuRoutedObservationView GpuLearnerSideRouter::select_visual_observations(
    const GpuBatchDeviceView& simulator,
    std::size_t environment_count,
    void* stream) {
    if (environment_count == 0 || environment_count > impl_->capacity ||
        !simulator.visual_observations_p1 || !simulator.visual_observations_p2 ||
        !simulator.observations_p1 || !simulator.observations_p2 ||
        !simulator.action_masks_p1 || !simulator.action_masks_p2) {
        throw std::invalid_argument("invalid routed visual observation input");
    }
    select_observations_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        simulator.visual_observations_p1, simulator.visual_observations_p2,
        simulator.observations_p1, simulator.observations_p2,
        kVisualObservationSize, kObservationSize,
        simulator.action_masks_p1, simulator.action_masks_p2,
        environment_count, impl_->learner_observations, impl_->opponent_observations,
        impl_->learner_masks, impl_->opponent_masks);
    check_cuda(cudaGetLastError(), "launch side-balanced visual observation router");
    return {impl_->learner_observations, impl_->learner_masks,
            impl_->opponent_observations, impl_->opponent_masks, environment_count};
}

GpuRoutedObservationView GpuLearnerSideRouter::select_self_play_visual_observations(
    const GpuBatchDeviceView& simulator,
    std::size_t environment_count,
    void* stream) {
    if (environment_count == 0 || environment_count > impl_->capacity ||
        !simulator.visual_observations_p1 || !simulator.visual_observations_p2 ||
        !simulator.action_masks_p1 || !simulator.action_masks_p2) {
        throw std::invalid_argument("invalid routed self-play visual observation input");
    }
    select_observations_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        simulator.visual_observations_p1, simulator.visual_observations_p2,
        simulator.visual_observations_p1, simulator.visual_observations_p2,
        kVisualObservationSize, kVisualObservationSize,
        simulator.action_masks_p1, simulator.action_masks_p2,
        environment_count, impl_->learner_observations, impl_->opponent_observations,
        impl_->learner_masks, impl_->opponent_masks);
    check_cuda(cudaGetLastError(), "launch self-play visual observation router");
    return {impl_->learner_observations, impl_->learner_masks,
            impl_->opponent_observations, impl_->opponent_masks, environment_count};
}

GpuRoutedActionView GpuLearnerSideRouter::route_actions(
    const std::int64_t* learner_actions,
    const std::int64_t* opponent_actions,
    std::size_t environment_count,
    void* stream) {
    if (!learner_actions || !opponent_actions || environment_count == 0 ||
        environment_count > impl_->capacity) {
        throw std::invalid_argument("invalid routed action input");
    }
    route_actions_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        learner_actions, opponent_actions, environment_count, impl_->p1_actions, impl_->p2_actions);
    check_cuda(cudaGetLastError(), "launch side-balanced action router");
    return {impl_->p1_actions, impl_->p2_actions, environment_count};
}

const std::int64_t* GpuLearnerSideRouter::mix_self_play_actions(
    const std::int64_t* latest_actions,
    const std::int64_t* best_older_actions,
    std::size_t environment_count,
    void* stream) {
    if (!latest_actions || !best_older_actions || environment_count == 0 ||
        environment_count > impl_->capacity) {
        throw std::invalid_argument("invalid self-play action mixture input");
    }
    mix_self_play_actions_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        latest_actions, best_older_actions, environment_count, impl_->mixed_opponent_actions);
    check_cuda(cudaGetLastError(), "launch self-play action mixture kernel");
    return impl_->mixed_opponent_actions;
}

const float* GpuLearnerSideRouter::select_rewards(
    const float* p1_rewards,
    const float* p2_rewards,
    std::size_t environment_count,
    void* stream) {
    if (!p1_rewards || !p2_rewards || environment_count == 0 ||
        environment_count > impl_->capacity) {
        throw std::invalid_argument("invalid routed reward input");
    }
    select_rewards_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        p1_rewards, p2_rewards, environment_count, impl_->learner_rewards);
    check_cuda(cudaGetLastError(), "launch side-balanced reward router");
    return impl_->learner_rewards;
}

void GpuLearnerSideRouter::tally_outcomes(
    const float* sparse_rewards_p1,
    const float* sparse_rewards_p2,
    const std::uint8_t* terminated,
    const std::uint32_t* profile_assignments,
    std::size_t profile_count,
    std::size_t environment_count,
    void* stream) {
    if (!sparse_rewards_p1 || !sparse_rewards_p2 || !terminated || !profile_assignments ||
        profile_count == 0 || environment_count == 0 || environment_count > impl_->capacity) {
        throw std::invalid_argument("invalid matchup outcome tally input");
    }
    impl_->ensure_outcome_tally(profile_count, as_stream(stream));
    tally_outcomes_kernel<<<blocks_for(environment_count), kThreads, 0, as_stream(stream)>>>(
        sparse_rewards_p1, sparse_rewards_p2, terminated, profile_assignments,
        profile_count, environment_count, impl_->outcome_tally);
    check_cuda(cudaGetLastError(), "launch matchup outcome tally");
}

std::vector<std::uint64_t> GpuLearnerSideRouter::take_outcome_tally(
    std::size_t profile_count,
    void* stream) {
    if (profile_count == 0) throw std::invalid_argument("outcome tally needs at least one profile");
    const auto cuda_stream = as_stream(stream);
    impl_->ensure_outcome_tally(profile_count, cuda_stream);
    std::vector<std::uint64_t> result(profile_count * 3);
    static_assert(sizeof(unsigned long long) == sizeof(std::uint64_t));
    check_cuda(cudaMemcpyAsync(result.data(), impl_->outcome_tally, sizeof(std::uint64_t) * result.size(),
                               cudaMemcpyDeviceToHost, cuda_stream), "download matchup outcome tally");
    check_cuda(cudaMemsetAsync(impl_->outcome_tally, 0, sizeof(std::uint64_t) * result.size(), cuda_stream),
               "clear matchup outcome tally");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize matchup outcome tally");
    return result;
}

}  // namespace t8::v2
