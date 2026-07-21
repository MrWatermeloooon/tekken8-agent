#include "t8_v2/ppo.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace t8::v2 {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void check_cublas(cublasStatus_t result, const char* operation) {
    if (result != CUBLAS_STATUS_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed with cuBLAS status " +
                                 std::to_string(static_cast<int>(result)));
    }
}

cudaStream_t as_stream(void* stream) {
    return reinterpret_cast<cudaStream_t>(stream);
}

int blocks_for(std::size_t count) {
    return static_cast<int>((count + kThreads - 1) / kThreads);
}

__global__ void bias_tanh_kernel(
    float* values,
    const float* bias,
    std::size_t element_count,
    int width) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    values[index] = tanhf(values[index] + bias[index % width]);
}

__global__ void bias_kernel(
    float* values,
    const float* bias,
    std::size_t element_count,
    int width) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    values[index] += bias[index % width];
}

__device__ __forceinline__ std::uint64_t splitmix64(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__device__ __forceinline__ float uniform01(
    std::uint64_t seed,
    std::uint64_t step,
    std::size_t lane) {
    const std::uint64_t bits = splitmix64(seed ^ splitmix64(step) ^ splitmix64(lane));
    return static_cast<float>((bits >> 40U) + 0.5) * (1.0F / 16777216.0F);
}

__global__ void masked_sample_kernel(
    const float* actor_critic_output,
    const std::uint8_t* action_masks,
    std::size_t environment_count,
    int action_count,
    std::uint64_t seed,
    std::uint64_t step,
    bool deterministic,
    float* logits,
    std::int64_t* actions,
    float* log_probabilities,
    float* values,
    float* entropies) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= environment_count) return;
    const int output_width = action_count + 1;
    const std::size_t output_base = lane * output_width;
    const std::size_t action_base = lane * action_count;

    float maximum = -3.402823466e38F;
    int best_action = 0;
    bool any_legal = false;
    for (int action = 0; action < action_count; ++action) {
        const float logit = actor_critic_output[output_base + action];
        logits[action_base + action] = logit;
        if (action_masks[action_base + action]) {
            if (!any_legal || logit > maximum) {
                maximum = logit;
                best_action = action;
            }
            any_legal = true;
        }
    }
    if (!any_legal) {
        maximum = actor_critic_output[output_base];
        best_action = 0;
    }

    float sum = 0.0F;
    for (int action = 0; action < action_count; ++action) {
        if ((any_legal && action_masks[action_base + action]) || (!any_legal && action == 0)) {
            sum += expf(actor_critic_output[output_base + action] - maximum);
        }
    }
    sum = fmaxf(sum, 1e-20F);

    int selected = best_action;
    if (!deterministic) {
        const float target = uniform01(seed, step, lane) * sum;
        float cumulative = 0.0F;
        for (int action = 0; action < action_count; ++action) {
            if ((any_legal && action_masks[action_base + action]) || (!any_legal && action == 0)) {
                cumulative += expf(actor_critic_output[output_base + action] - maximum);
                if (target <= cumulative) {
                    selected = action;
                    break;
                }
            }
        }
    }

    const float log_sum = logf(sum) + maximum;
    float entropy = 0.0F;
    for (int action = 0; action < action_count; ++action) {
        if ((any_legal && action_masks[action_base + action]) || (!any_legal && action == 0)) {
            const float log_probability = actor_critic_output[output_base + action] - log_sum;
            const float probability = expf(log_probability);
            entropy -= probability * log_probability;
        }
    }
    actions[lane] = selected;
    log_probabilities[lane] = actor_critic_output[output_base + selected] - log_sum;
    values[lane] = actor_critic_output[output_base + action_count];
    entropies[lane] = entropy;
}

template <typename T>
std::vector<T> download(
    const T* source,
    std::size_t count,
    cudaStream_t stream,
    const char* operation) {
    std::vector<T> result(count);
    check_cuda(cudaMemcpyAsync(result.data(), source, sizeof(T) * count,
                               cudaMemcpyDeviceToHost, stream), operation);
    check_cuda(cudaStreamSynchronize(stream), "synchronize policy download");
    return result;
}

}  // namespace

struct GpuActorCritic::Impl {
    std::size_t capacity;
    ActorCriticConfig config;
    cublasHandle_t cublas = nullptr;

    float* weights_1 = nullptr;
    float* bias_1 = nullptr;
    float* weights_2 = nullptr;
    float* bias_2 = nullptr;
    float* weights_out = nullptr;
    float* bias_out = nullptr;

    float* hidden_1 = nullptr;
    float* hidden_2 = nullptr;
    float* actor_critic_output = nullptr;
    float* logits = nullptr;
    std::int64_t* actions = nullptr;
    float* log_probabilities = nullptr;
    float* values = nullptr;
    float* entropies = nullptr;

    Impl(std::size_t requested_capacity, ActorCriticConfig network_config, std::uint64_t seed)
        : capacity(requested_capacity), config(network_config) {
        if (capacity == 0) throw std::invalid_argument("policy capacity must be greater than zero");
        if (config.observation_size <= 0 || config.action_count <= 1 || config.hidden_size <= 0) {
            throw std::invalid_argument("actor-critic dimensions must be positive");
        }
        const int output_size = config.action_count + 1;
        check_cublas(cublasCreate(&cublas), "create cuBLAS handle");
        check_cuda(cudaMalloc(&weights_1, sizeof(float) * config.hidden_size * config.observation_size), "allocate layer-1 weights");
        check_cuda(cudaMalloc(&bias_1, sizeof(float) * config.hidden_size), "allocate layer-1 bias");
        check_cuda(cudaMalloc(&weights_2, sizeof(float) * config.hidden_size * config.hidden_size), "allocate layer-2 weights");
        check_cuda(cudaMalloc(&bias_2, sizeof(float) * config.hidden_size), "allocate layer-2 bias");
        check_cuda(cudaMalloc(&weights_out, sizeof(float) * output_size * config.hidden_size), "allocate output weights");
        check_cuda(cudaMalloc(&bias_out, sizeof(float) * output_size), "allocate output bias");
        check_cuda(cudaMalloc(&hidden_1, sizeof(float) * capacity * config.hidden_size), "allocate first activations");
        check_cuda(cudaMalloc(&hidden_2, sizeof(float) * capacity * config.hidden_size), "allocate second activations");
        check_cuda(cudaMalloc(&actor_critic_output, sizeof(float) * capacity * output_size), "allocate actor-critic output");
        check_cuda(cudaMalloc(&logits, sizeof(float) * capacity * config.action_count), "allocate policy logits");
        check_cuda(cudaMalloc(&actions, sizeof(std::int64_t) * capacity), "allocate sampled actions");
        check_cuda(cudaMalloc(&log_probabilities, sizeof(float) * capacity), "allocate log probabilities");
        check_cuda(cudaMalloc(&values, sizeof(float) * capacity), "allocate values");
        check_cuda(cudaMalloc(&entropies, sizeof(float) * capacity), "allocate entropies");

        std::mt19937_64 random(seed);
        initialize_matrix(weights_1, config.hidden_size, config.observation_size, random);
        initialize_matrix(weights_2, config.hidden_size, config.hidden_size, random);
        initialize_matrix(weights_out, output_size, config.hidden_size, random, 0.01F);
        check_cuda(cudaMemset(bias_1, 0, sizeof(float) * config.hidden_size), "zero layer-1 bias");
        check_cuda(cudaMemset(bias_2, 0, sizeof(float) * config.hidden_size), "zero layer-2 bias");
        check_cuda(cudaMemset(bias_out, 0, sizeof(float) * output_size), "zero output bias");
    }

    static void initialize_matrix(
        float* destination,
        int rows,
        int columns,
        std::mt19937_64& random,
        float override_standard_deviation = 0.0F) {
        const float deviation = override_standard_deviation > 0.0F
            ? override_standard_deviation
            : std::sqrt(2.0F / static_cast<float>(rows + columns));
        std::normal_distribution<float> distribution(0.0F, deviation);
        std::vector<float> host(static_cast<std::size_t>(rows) * columns);
        for (float& value : host) value = distribution(random);
        check_cuda(cudaMemcpy(destination, host.data(), sizeof(float) * host.size(),
                              cudaMemcpyHostToDevice), "initialize network matrix");
    }

    ~Impl() {
        cudaFree(entropies);
        cudaFree(values);
        cudaFree(log_probabilities);
        cudaFree(actions);
        cudaFree(logits);
        cudaFree(actor_critic_output);
        cudaFree(hidden_2);
        cudaFree(hidden_1);
        cudaFree(bias_out);
        cudaFree(weights_out);
        cudaFree(bias_2);
        cudaFree(weights_2);
        cudaFree(bias_1);
        cudaFree(weights_1);
        if (cublas != nullptr) cublasDestroy(cublas);
    }

    void linear(
        const float* input,
        const float* weights,
        float* output,
        int batch,
        int input_size,
        int output_size) {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        check_cublas(cublasSgemm(
            cublas, CUBLAS_OP_T, CUBLAS_OP_N,
            output_size, batch, input_size,
            &alpha, weights, input_size,
            input, input_size,
            &beta, output, output_size), "actor-critic matrix multiply");
    }
};

GpuActorCritic::GpuActorCritic(
    std::size_t capacity,
    ActorCriticConfig config,
    std::uint64_t initialization_seed)
    : impl_(std::make_unique<Impl>(capacity, config, initialization_seed)) {}

GpuActorCritic::~GpuActorCritic() = default;
GpuActorCritic::GpuActorCritic(GpuActorCritic&&) noexcept = default;
GpuActorCritic& GpuActorCritic::operator=(GpuActorCritic&&) noexcept = default;

std::size_t GpuActorCritic::capacity() const noexcept { return impl_->capacity; }

std::size_t GpuActorCritic::parameter_count() const noexcept {
    const auto& c = impl_->config;
    const int output_size = c.action_count + 1;
    return static_cast<std::size_t>(c.hidden_size) * c.observation_size + c.hidden_size +
           static_cast<std::size_t>(c.hidden_size) * c.hidden_size + c.hidden_size +
           static_cast<std::size_t>(output_size) * c.hidden_size + output_size;
}

const ActorCriticConfig& GpuActorCritic::config() const noexcept { return impl_->config; }

GpuPolicyOutputView GpuActorCritic::forward(
    const float* device_observations,
    const std::uint8_t* device_action_masks,
    std::size_t environment_count,
    std::uint64_t sampling_seed,
    std::uint64_t sampling_step,
    bool deterministic,
    void* stream) {
    if (device_observations == nullptr || device_action_masks == nullptr) {
        throw std::invalid_argument("policy input pointers cannot be null");
    }
    if (environment_count == 0 || environment_count > impl_->capacity) {
        throw std::invalid_argument("environment_count exceeds policy capacity");
    }
    const auto cuda_stream = as_stream(stream);
    check_cublas(cublasSetStream(impl_->cublas, cuda_stream), "set actor-critic CUDA stream");
    const int batch = static_cast<int>(environment_count);
    const int hidden = impl_->config.hidden_size;
    const int output_size = impl_->config.action_count + 1;
    impl_->linear(device_observations, impl_->weights_1, impl_->hidden_1,
                  batch, impl_->config.observation_size, hidden);
    const std::size_t hidden_elements = environment_count * hidden;
    bias_tanh_kernel<<<blocks_for(hidden_elements), kThreads, 0, cuda_stream>>>(
        impl_->hidden_1, impl_->bias_1, hidden_elements, hidden);
    impl_->linear(impl_->hidden_1, impl_->weights_2, impl_->hidden_2, batch, hidden, hidden);
    bias_tanh_kernel<<<blocks_for(hidden_elements), kThreads, 0, cuda_stream>>>(
        impl_->hidden_2, impl_->bias_2, hidden_elements, hidden);
    impl_->linear(impl_->hidden_2, impl_->weights_out, impl_->actor_critic_output,
                  batch, hidden, output_size);
    const std::size_t output_elements = environment_count * output_size;
    bias_kernel<<<blocks_for(output_elements), kThreads, 0, cuda_stream>>>(
        impl_->actor_critic_output, impl_->bias_out, output_elements, output_size);
    masked_sample_kernel<<<blocks_for(environment_count), kThreads, 0, cuda_stream>>>(
        impl_->actor_critic_output, device_action_masks, environment_count,
        impl_->config.action_count, sampling_seed, sampling_step, deterministic,
        impl_->logits, impl_->actions, impl_->log_probabilities,
        impl_->values, impl_->entropies);
    check_cuda(cudaGetLastError(), "launch actor-critic inference kernels");
    return {impl_->logits, impl_->actions, impl_->log_probabilities,
            impl_->values, impl_->entropies, environment_count};
}

void GpuActorCritic::synchronize(void* stream) const {
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize actor-critic");
}

std::vector<std::int64_t> GpuActorCritic::download_actions(
    std::size_t count, void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download count exceeds capacity");
    return download(impl_->actions, count, as_stream(stream), "download actions");
}

std::vector<float> GpuActorCritic::download_values(std::size_t count, void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download count exceeds capacity");
    return download(impl_->values, count, as_stream(stream), "download values");
}

std::vector<float> GpuActorCritic::download_log_probabilities(
    std::size_t count, void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download count exceeds capacity");
    return download(impl_->log_probabilities, count, as_stream(stream), "download log probabilities");
}

std::vector<float> GpuActorCritic::download_entropies(std::size_t count, void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download count exceeds capacity");
    return download(impl_->entropies, count, as_stream(stream), "download entropies");
}

namespace {

__global__ void gae_kernel(
    const float* rewards,
    const float* values,
    const std::uint8_t* terminated,
    const float* bootstrap_values,
    std::size_t environments,
    std::size_t horizon,
    float gamma,
    float gae_lambda,
    float* advantages,
    float* returns) {
    const std::size_t environment = blockIdx.x * blockDim.x + threadIdx.x;
    if (environment >= environments) return;
    float next_advantage = 0.0F;
    for (std::size_t reverse = 0; reverse < horizon; ++reverse) {
        const std::size_t step = horizon - 1 - reverse;
        const std::size_t index = step * environments + environment;
        const float next_value = step + 1 == horizon
            ? bootstrap_values[environment]
            : values[(step + 1) * environments + environment];
        const float non_terminal = terminated[index] ? 0.0F : 1.0F;
        const float delta = rewards[index] + gamma * next_value * non_terminal - values[index];
        next_advantage = delta + gamma * gae_lambda * non_terminal * next_advantage;
        advantages[index] = next_advantage;
        returns[index] = next_advantage + values[index];
    }
}

__global__ void advantage_stats_kernel(
    const float* advantages,
    std::size_t count,
    float* statistics) {
    __shared__ float shared_sum[kThreads];
    __shared__ float shared_square_sum[kThreads];
    float local_sum = 0.0F;
    float local_square_sum = 0.0F;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const float value = advantages[index];
        local_sum += value;
        local_square_sum += value * value;
    }
    shared_sum[threadIdx.x] = local_sum;
    shared_square_sum[threadIdx.x] = local_square_sum;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            shared_sum[threadIdx.x] += shared_sum[threadIdx.x + stride];
            shared_square_sum[threadIdx.x] += shared_square_sum[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(statistics + 0, shared_sum[0]);
        atomicAdd(statistics + 1, shared_square_sum[0]);
    }
}

__global__ void normalize_advantages_kernel(
    float* advantages,
    std::size_t count,
    const float* statistics) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float mean = statistics[0] / static_cast<float>(count);
    const float second_moment = statistics[1] / static_cast<float>(count);
    const float inverse_standard_deviation = rsqrtf(fmaxf(second_moment - mean * mean, 1e-8F));
    advantages[index] = (advantages[index] - mean) * inverse_standard_deviation;
}

}  // namespace

struct GpuRolloutBuffer::Impl {
    std::size_t environments;
    std::size_t rollout_horizon;
    std::size_t samples;
    ActorCriticConfig config;
    float* observations = nullptr;
    std::uint8_t* action_masks = nullptr;
    std::int64_t* actions = nullptr;
    float* old_log_probabilities = nullptr;
    float* old_values = nullptr;
    float* rewards = nullptr;
    std::uint8_t* terminated = nullptr;
    float* advantages = nullptr;
    float* returns = nullptr;
    float* advantage_statistics = nullptr;

    Impl(std::size_t environment_count, std::size_t horizon, ActorCriticConfig network_config)
        : environments(environment_count), rollout_horizon(horizon),
          samples(environment_count * horizon), config(network_config) {
        if (environments == 0 || rollout_horizon == 0) {
            throw std::invalid_argument("rollout dimensions must be greater than zero");
        }
        check_cuda(cudaMalloc(&observations, sizeof(float) * samples * config.observation_size),
                   "allocate rollout observations");
        check_cuda(cudaMalloc(&action_masks, sizeof(std::uint8_t) * samples * config.action_count),
                   "allocate rollout action masks");
        check_cuda(cudaMalloc(&actions, sizeof(std::int64_t) * samples), "allocate rollout actions");
        check_cuda(cudaMalloc(&old_log_probabilities, sizeof(float) * samples), "allocate rollout log probabilities");
        check_cuda(cudaMalloc(&old_values, sizeof(float) * samples), "allocate rollout values");
        check_cuda(cudaMalloc(&rewards, sizeof(float) * samples), "allocate rollout rewards");
        check_cuda(cudaMalloc(&terminated, sizeof(std::uint8_t) * samples), "allocate rollout terminal flags");
        check_cuda(cudaMalloc(&advantages, sizeof(float) * samples), "allocate rollout advantages");
        check_cuda(cudaMalloc(&returns, sizeof(float) * samples), "allocate rollout returns");
        check_cuda(cudaMalloc(&advantage_statistics, sizeof(float) * 2), "allocate advantage statistics");
    }

    ~Impl() {
        cudaFree(advantage_statistics);
        cudaFree(returns);
        cudaFree(advantages);
        cudaFree(terminated);
        cudaFree(rewards);
        cudaFree(old_values);
        cudaFree(old_log_probabilities);
        cudaFree(actions);
        cudaFree(action_masks);
        cudaFree(observations);
    }
};

GpuRolloutBuffer::GpuRolloutBuffer(
    std::size_t environment_count,
    std::size_t horizon,
    ActorCriticConfig config)
    : impl_(std::make_unique<Impl>(environment_count, horizon, config)) {}

GpuRolloutBuffer::~GpuRolloutBuffer() = default;
GpuRolloutBuffer::GpuRolloutBuffer(GpuRolloutBuffer&&) noexcept = default;
GpuRolloutBuffer& GpuRolloutBuffer::operator=(GpuRolloutBuffer&&) noexcept = default;

std::size_t GpuRolloutBuffer::environment_count() const noexcept { return impl_->environments; }
std::size_t GpuRolloutBuffer::horizon() const noexcept { return impl_->rollout_horizon; }
std::size_t GpuRolloutBuffer::sample_count() const noexcept { return impl_->samples; }

GpuRolloutView GpuRolloutBuffer::device_view() const noexcept {
    return {impl_->observations, impl_->action_masks, impl_->actions,
            impl_->old_log_probabilities, impl_->old_values, impl_->rewards,
            impl_->terminated, impl_->advantages, impl_->returns,
            impl_->environments, impl_->rollout_horizon, impl_->samples};
}

void GpuRolloutBuffer::record_policy_device(
    std::size_t step,
    const float* observations,
    const std::uint8_t* action_masks,
    const std::int64_t* actions,
    const float* log_probabilities,
    const float* values,
    void* stream) {
    if (step >= impl_->rollout_horizon) throw std::out_of_range("rollout step exceeds horizon");
    if (!observations || !action_masks || !actions || !log_probabilities || !values) {
        throw std::invalid_argument("rollout device pointers cannot be null");
    }
    const auto cuda_stream = as_stream(stream);
    const std::size_t sample_offset = step * impl_->environments;
    check_cuda(cudaMemcpyAsync(
        impl_->observations + sample_offset * impl_->config.observation_size,
        observations, sizeof(float) * impl_->environments * impl_->config.observation_size,
        cudaMemcpyDeviceToDevice, cuda_stream), "record rollout observations");
    check_cuda(cudaMemcpyAsync(
        impl_->action_masks + sample_offset * impl_->config.action_count,
        action_masks, sizeof(std::uint8_t) * impl_->environments * impl_->config.action_count,
        cudaMemcpyDeviceToDevice, cuda_stream), "record rollout action masks");
    check_cuda(cudaMemcpyAsync(impl_->actions + sample_offset, actions,
                               sizeof(std::int64_t) * impl_->environments,
                               cudaMemcpyDeviceToDevice, cuda_stream), "record rollout actions");
    check_cuda(cudaMemcpyAsync(impl_->old_log_probabilities + sample_offset, log_probabilities,
                               sizeof(float) * impl_->environments,
                               cudaMemcpyDeviceToDevice, cuda_stream), "record rollout log probabilities");
    check_cuda(cudaMemcpyAsync(impl_->old_values + sample_offset, values,
                               sizeof(float) * impl_->environments,
                               cudaMemcpyDeviceToDevice, cuda_stream), "record rollout values");
}

void GpuRolloutBuffer::record_outcome_device(
    std::size_t step,
    const float* rewards,
    const std::uint8_t* terminated,
    void* stream) {
    if (step >= impl_->rollout_horizon) throw std::out_of_range("rollout step exceeds horizon");
    if (!rewards || !terminated) throw std::invalid_argument("outcome device pointers cannot be null");
    const auto cuda_stream = as_stream(stream);
    const std::size_t sample_offset = step * impl_->environments;
    check_cuda(cudaMemcpyAsync(impl_->rewards + sample_offset, rewards,
                               sizeof(float) * impl_->environments,
                               cudaMemcpyDeviceToDevice, cuda_stream), "record rollout rewards");
    check_cuda(cudaMemcpyAsync(impl_->terminated + sample_offset, terminated,
                               sizeof(std::uint8_t) * impl_->environments,
                               cudaMemcpyDeviceToDevice, cuda_stream), "record rollout terminal flags");
}

void GpuRolloutBuffer::compute_gae(
    const float* bootstrap_values,
    float gamma,
    float gae_lambda,
    bool normalize,
    void* stream) {
    if (!bootstrap_values) throw std::invalid_argument("bootstrap values cannot be null");
    if (!(gamma >= 0.0F && gamma <= 1.0F && gae_lambda >= 0.0F && gae_lambda <= 1.0F)) {
        throw std::invalid_argument("gamma and gae_lambda must be in [0, 1]");
    }
    const auto cuda_stream = as_stream(stream);
    gae_kernel<<<blocks_for(impl_->environments), kThreads, 0, cuda_stream>>>(
        impl_->rewards, impl_->old_values, impl_->terminated, bootstrap_values,
        impl_->environments, impl_->rollout_horizon, gamma, gae_lambda,
        impl_->advantages, impl_->returns);
    if (normalize) {
        check_cuda(cudaMemsetAsync(impl_->advantage_statistics, 0, sizeof(float) * 2, cuda_stream),
                   "clear advantage statistics");
        const int reduction_blocks = std::min(blocks_for(impl_->samples), 1024);
        advantage_stats_kernel<<<reduction_blocks, kThreads, 0, cuda_stream>>>(
            impl_->advantages, impl_->samples, impl_->advantage_statistics);
        normalize_advantages_kernel<<<blocks_for(impl_->samples), kThreads, 0, cuda_stream>>>(
            impl_->advantages, impl_->samples, impl_->advantage_statistics);
    }
    check_cuda(cudaGetLastError(), "launch rollout GAE kernels");
}

void GpuRolloutBuffer::synchronize(void* stream) const {
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize rollout buffer");
}

std::vector<float> GpuRolloutBuffer::download_advantages(void* stream) const {
    return download(impl_->advantages, impl_->samples, as_stream(stream), "download advantages");
}

std::vector<float> GpuRolloutBuffer::download_returns(void* stream) const {
    return download(impl_->returns, impl_->samples, as_stream(stream), "download returns");
}

}  // namespace t8::v2
