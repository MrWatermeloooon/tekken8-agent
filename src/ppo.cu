#include "t8_v2/ppo.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

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

}  // namespace t8::v2
