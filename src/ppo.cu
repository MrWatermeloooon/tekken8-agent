#include "t8_v2/ppo.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace t8::v2 {
namespace {

constexpr int kThreads = 256;

void validate_actor_config(const ActorCriticConfig& config) {
    if (config.observation_size <= 0 || config.action_count <= 1 || config.hidden_size <= 0 ||
        config.action_count == std::numeric_limits<int>::max()) {
        throw std::invalid_argument("actor-critic dimensions must be positive and representable");
    }
}

std::size_t checked_product(std::size_t left, std::size_t right, const char* description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(std::string(description) + " overflows size_t");
    }
    return left * right;
}

void validate_allocation(
    std::size_t rows,
    std::size_t columns,
    std::size_t element_size,
    const char* description) {
    const auto elements = checked_product(rows, columns, description);
    static_cast<void>(checked_product(elements, element_size, description));
}

void validate_ppo_coefficients(const PpoUpdateConfig& config) {
    const bool finite = std::isfinite(config.learning_rate) && std::isfinite(config.clip_range) &&
        std::isfinite(config.value_clip_range) && std::isfinite(config.target_kl) &&
        std::isfinite(config.value_coefficient) && std::isfinite(config.entropy_coefficient) &&
        std::isfinite(config.max_gradient_norm) && std::isfinite(config.adam_beta1) &&
        std::isfinite(config.adam_beta2) && std::isfinite(config.adam_epsilon);
    if (!finite || config.learning_rate <= 0.0F || config.clip_range < 0.0F ||
        config.value_clip_range < 0.0F || config.target_kl < 0.0F ||
        config.value_coefficient < 0.0F || config.entropy_coefficient < 0.0F ||
        config.max_gradient_norm <= 0.0F || config.adam_beta1 < 0.0F ||
        config.adam_beta1 >= 1.0F || config.adam_beta2 < 0.0F ||
        config.adam_beta2 >= 1.0F || config.adam_epsilon <= 0.0F) {
        throw std::invalid_argument("PPO optimizer coefficients are invalid or non-finite");
    }
}

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
    const std::size_t blocks = count / kThreads + (count % kThreads != 0 ? 1 : 0);
    if (blocks == 0 || blocks > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        throw std::invalid_argument("CUDA launch grid is empty or exceeds the supported range");
    }
    return static_cast<int>(blocks);
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

__global__ void gather_minibatch_kernel(
    const float* source_observations,
    const std::uint8_t* source_masks,
    const std::int64_t* source_actions,
    const float* source_old_log_probabilities,
    const float* source_old_values,
    const float* source_advantages,
    const float* source_returns,
    std::size_t sample_count,
    std::size_t batch_offset,
    std::size_t batch_size,
    std::size_t permutation_stride,
    std::size_t permutation_offset,
    int observation_size,
    int action_count,
    float* observations,
    std::uint8_t* masks,
    std::int64_t* actions,
    float* old_log_probabilities,
    float* old_values,
    float* advantages,
    float* returns) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch_size) return;
    const std::size_t source = (permutation_offset +
        (batch_offset + lane) * permutation_stride) % sample_count;
    for (int feature = 0; feature < observation_size; ++feature) {
        observations[lane * observation_size + feature] =
            source_observations[source * observation_size + feature];
    }
    for (int action = 0; action < action_count; ++action) {
        masks[lane * action_count + action] = source_masks[source * action_count + action];
    }
    actions[lane] = source_actions[source];
    old_log_probabilities[lane] = source_old_log_probabilities[source];
    old_values[lane] = source_old_values[source];
    advantages[lane] = source_advantages[source];
    returns[lane] = source_returns[source];
}

__global__ void ppo_output_gradient_kernel(
    const float* actor_critic_output,
    const std::uint8_t* masks,
    const std::int64_t* actions,
    const float* old_log_probabilities,
    const float* old_values,
    const float* advantages,
    const float* returns,
    std::size_t batch_size,
    int action_count,
    float clip_range,
    float value_clip_range,
    float value_coefficient,
    float entropy_coefficient,
    float* output_gradients,
    float* metric_sums) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch_size) return;
    const int width = action_count + 1;
    const std::size_t base = lane * width;
    const std::size_t mask_base = lane * action_count;
    float maximum = -3.402823466e38F;
    for (int action = 0; action < action_count; ++action) {
        if (masks[mask_base + action]) maximum = fmaxf(maximum, actor_critic_output[base + action]);
    }
    float probability_sum = 0.0F;
    for (int action = 0; action < action_count; ++action) {
        if (masks[mask_base + action]) {
            probability_sum += expf(actor_critic_output[base + action] - maximum);
        }
    }
    probability_sum = fmaxf(probability_sum, 1e-20F);
    const float log_sum = logf(probability_sum) + maximum;
    const int selected = static_cast<int>(actions[lane]);
    const float new_log_probability = actor_critic_output[base + selected] - log_sum;
    const float log_ratio = new_log_probability - old_log_probabilities[lane];
    const float ratio = expf(log_ratio);
    const float advantage = advantages[lane];
    const float unclipped = ratio * advantage;
    const float clipped_ratio = fminf(1.0F + clip_range, fmaxf(1.0F - clip_range, ratio));
    const float clipped = clipped_ratio * advantage;
    const bool use_unclipped = unclipped <= clipped;
    const float policy_loss = -fminf(unclipped, clipped);
    const float log_probability_gradient = use_unclipped ? -advantage * ratio : 0.0F;

    float entropy = 0.0F;
    for (int action = 0; action < action_count; ++action) {
        if (!masks[mask_base + action]) {
            output_gradients[base + action] = 0.0F;
            continue;
        }
        const float log_probability = actor_critic_output[base + action] - log_sum;
        const float probability = expf(log_probability);
        entropy -= probability * log_probability;
        const float selected_delta = action == selected ? 1.0F : 0.0F;
        output_gradients[base + action] =
            (log_probability_gradient * (selected_delta - probability)) /
            static_cast<float>(batch_size);
    }
    for (int action = 0; action < action_count; ++action) {
        if (masks[mask_base + action]) {
            const float log_probability = actor_critic_output[base + action] - log_sum;
            const float probability = expf(log_probability);
            output_gradients[base + action] += entropy_coefficient * probability *
                (log_probability + entropy) / static_cast<float>(batch_size);
        }
    }
    const float value = actor_critic_output[base + action_count];
    const float value_error = value - returns[lane];
    float value_gradient = value_error;
    float value_loss = 0.5F * value_error * value_error;
    if (value_clip_range > 0.0F) {
        const float value_delta = value - old_values[lane];
        const float clipped_value = old_values[lane] +
            fminf(value_clip_range, fmaxf(-value_clip_range, value_delta));
        const float clipped_error = clipped_value - returns[lane];
        const float clipped_loss = 0.5F * clipped_error * clipped_error;
        if (clipped_loss > value_loss) {
            value_loss = clipped_loss;
            value_gradient = fabsf(value_delta) <= value_clip_range ? clipped_error : 0.0F;
        }
    }
    output_gradients[base + action_count] =
        value_coefficient * value_gradient / static_cast<float>(batch_size);

    atomicAdd(metric_sums + 0, policy_loss);
    atomicAdd(metric_sums + 1, value_loss);
    atomicAdd(metric_sums + 2, entropy);
    atomicAdd(metric_sums + 3, (ratio - 1.0F) - log_ratio);
    atomicAdd(metric_sums + 4, fabsf(ratio - 1.0F) > clip_range ? 1.0F : 0.0F);
}

template <typename T>
struct ScopedDeviceBuffer {
    T* pointer = nullptr;
    explicit ScopedDeviceBuffer(std::size_t count) {
        check_cuda(cudaMalloc(&pointer, sizeof(T) * count), "allocate PPO diagnostic buffer");
    }
    ~ScopedDeviceBuffer() { cudaFree(pointer); }
    ScopedDeviceBuffer(const ScopedDeviceBuffer&) = delete;
    ScopedDeviceBuffer& operator=(const ScopedDeviceBuffer&) = delete;
};

__global__ void tanh_backward_kernel(
    const float* upstream,
    const float* activation,
    float* gradient,
    std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    gradient[index] = upstream[index] * (1.0F - activation[index] * activation[index]);
}

__global__ void bias_gradient_kernel(
    const float* output_gradient,
    float* bias_gradient,
    std::size_t batch_size,
    int width) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= width) return;
    float sum = 0.0F;
    for (std::size_t lane = 0; lane < batch_size; ++lane) sum += output_gradient[lane * width + feature];
    bias_gradient[feature] = sum;
}

__global__ void gradient_square_sum_kernel(
    const float* weights_1, std::size_t weights_1_count,
    const float* bias_1, std::size_t bias_1_count,
    const float* weights_2, std::size_t weights_2_count,
    const float* bias_2, std::size_t bias_2_count,
    const float* weights_out, std::size_t weights_out_count,
    const float* bias_out, std::size_t bias_out_count,
    float* sum,
    float* metrics) {
    __shared__ float shared[kThreads];
    float local = 0.0F;
    for (std::size_t index = threadIdx.x; index < weights_1_count; index += blockDim.x)
        local += weights_1[index] * weights_1[index];
    for (std::size_t index = threadIdx.x; index < bias_1_count; index += blockDim.x)
        local += bias_1[index] * bias_1[index];
    for (std::size_t index = threadIdx.x; index < weights_2_count; index += blockDim.x)
        local += weights_2[index] * weights_2[index];
    for (std::size_t index = threadIdx.x; index < bias_2_count; index += blockDim.x)
        local += bias_2[index] * bias_2[index];
    for (std::size_t index = threadIdx.x; index < weights_out_count; index += blockDim.x)
        local += weights_out[index] * weights_out[index];
    for (std::size_t index = threadIdx.x; index < bias_out_count; index += blockDim.x)
        local += bias_out[index] * bias_out[index];
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        sum[0] = shared[0];
        atomicAdd(metrics + 5, sqrtf(shared[0]));
    }
}

__global__ void adam_kernel(
    float* parameters,
    const float* gradients,
    float* first_moment,
    float* second_moment,
    std::size_t count,
    const float* gradient_square_sum,
    float max_gradient_norm,
    float learning_rate,
    float beta1,
    float beta2,
    float epsilon,
    float bias_correction1,
    float bias_correction2) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const float norm = sqrtf(gradient_square_sum[0]);
    const float scale = norm > max_gradient_norm ? max_gradient_norm / fmaxf(norm, 1e-12F) : 1.0F;
    const float gradient = gradients[index] * scale;
    const float m = beta1 * first_moment[index] + (1.0F - beta1) * gradient;
    const float v = beta2 * second_moment[index] + (1.0F - beta2) * gradient * gradient;
    first_moment[index] = m;
    second_moment[index] = v;
    parameters[index] -= learning_rate * (m / bias_correction1) /
        (sqrtf(v / bias_correction2) + epsilon);
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

std::vector<float> debug_ppo_objective_gradient(
    std::span<const float> logits_and_value,
    std::span<const std::uint8_t> action_mask,
    std::int64_t action,
    float old_log_probability,
    float old_value,
    float advantage,
    float return_value,
    const PpoUpdateConfig& config) {
    validate_ppo_coefficients(config);
    if (logits_and_value.size() < 3 || action_mask.size() + 1 != logits_and_value.size() ||
        action < 0 || static_cast<std::size_t>(action) >= action_mask.size() ||
        !action_mask[static_cast<std::size_t>(action)] ||
        !std::all_of(logits_and_value.begin(), logits_and_value.end(),
                     [](float value) { return std::isfinite(value); }) ||
        !std::isfinite(old_log_probability) || !std::isfinite(old_value) ||
        !std::isfinite(advantage) || !std::isfinite(return_value)) {
        throw std::invalid_argument("invalid PPO diagnostic objective input");
    }
    const int action_count = static_cast<int>(action_mask.size());
    ScopedDeviceBuffer<float> device_output(logits_and_value.size());
    ScopedDeviceBuffer<std::uint8_t> device_mask(action_mask.size());
    ScopedDeviceBuffer<std::int64_t> device_action(1);
    ScopedDeviceBuffer<float> device_old_log_probability(1);
    ScopedDeviceBuffer<float> device_old_value(1);
    ScopedDeviceBuffer<float> device_advantage(1);
    ScopedDeviceBuffer<float> device_return(1);
    ScopedDeviceBuffer<float> device_gradient(logits_and_value.size());
    ScopedDeviceBuffer<float> device_metrics(6);
    check_cuda(cudaMemcpy(device_output.pointer, logits_and_value.data(),
                          sizeof(float) * logits_and_value.size(), cudaMemcpyHostToDevice),
               "upload PPO diagnostic output");
    check_cuda(cudaMemcpy(device_mask.pointer, action_mask.data(),
                          sizeof(std::uint8_t) * action_mask.size(), cudaMemcpyHostToDevice),
               "upload PPO diagnostic mask");
    check_cuda(cudaMemcpy(device_action.pointer, &action, sizeof(action), cudaMemcpyHostToDevice),
               "upload PPO diagnostic action");
    check_cuda(cudaMemcpy(device_old_log_probability.pointer, &old_log_probability, sizeof(float),
                          cudaMemcpyHostToDevice), "upload PPO diagnostic old log probability");
    check_cuda(cudaMemcpy(device_old_value.pointer, &old_value, sizeof(float), cudaMemcpyHostToDevice),
               "upload PPO diagnostic old value");
    check_cuda(cudaMemcpy(device_advantage.pointer, &advantage, sizeof(float), cudaMemcpyHostToDevice),
               "upload PPO diagnostic advantage");
    check_cuda(cudaMemcpy(device_return.pointer, &return_value, sizeof(float), cudaMemcpyHostToDevice),
               "upload PPO diagnostic return");
    check_cuda(cudaMemset(device_metrics.pointer, 0, sizeof(float) * 6),
               "clear PPO diagnostic metrics");
    ppo_output_gradient_kernel<<<1, 1>>>(
        device_output.pointer, device_mask.pointer, device_action.pointer,
        device_old_log_probability.pointer, device_old_value.pointer,
        device_advantage.pointer, device_return.pointer, 1, action_count,
        config.clip_range, config.value_clip_range, config.value_coefficient,
        config.entropy_coefficient, device_gradient.pointer, device_metrics.pointer);
    check_cuda(cudaGetLastError(), "launch PPO diagnostic objective gradient");
    std::vector<float> gradient(logits_and_value.size());
    check_cuda(cudaMemcpy(gradient.data(), device_gradient.pointer,
                          sizeof(float) * gradient.size(), cudaMemcpyDeviceToHost),
               "download PPO diagnostic gradient");
    return gradient;
}

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

    float* train_observations = nullptr;
    std::uint8_t* train_masks = nullptr;
    std::int64_t* train_actions = nullptr;
    float* train_old_log_probabilities = nullptr;
    float* train_old_values = nullptr;
    float* train_advantages = nullptr;
    float* train_returns = nullptr;
    float* output_gradients = nullptr;
    float* hidden_2_upstream = nullptr;
    float* hidden_2_gradients = nullptr;
    float* hidden_1_upstream = nullptr;
    float* hidden_1_gradients = nullptr;

    float* gradient_weights_1 = nullptr;
    float* gradient_bias_1 = nullptr;
    float* gradient_weights_2 = nullptr;
    float* gradient_bias_2 = nullptr;
    float* gradient_weights_out = nullptr;
    float* gradient_bias_out = nullptr;
    float* moment1_weights_1 = nullptr;
    float* moment1_bias_1 = nullptr;
    float* moment1_weights_2 = nullptr;
    float* moment1_bias_2 = nullptr;
    float* moment1_weights_out = nullptr;
    float* moment1_bias_out = nullptr;
    float* moment2_weights_1 = nullptr;
    float* moment2_bias_1 = nullptr;
    float* moment2_weights_2 = nullptr;
    float* moment2_bias_2 = nullptr;
    float* moment2_weights_out = nullptr;
    float* moment2_bias_out = nullptr;
    float* gradient_square_sum = nullptr;
    float* metric_sums = nullptr;
    std::uint64_t optimizer_step = 0;

    Impl(std::size_t requested_capacity, ActorCriticConfig network_config, std::uint64_t seed)
        : capacity(requested_capacity), config(network_config) {
        if (capacity == 0 || capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("policy capacity must be positive and fit cuBLAS dimensions");
        }
        validate_actor_config(config);
        const auto observations = static_cast<std::size_t>(config.observation_size);
        const auto actions_count = static_cast<std::size_t>(config.action_count);
        const auto hidden_count = static_cast<std::size_t>(config.hidden_size);
        const auto output_count = actions_count + 1;
        validate_allocation(hidden_count, observations, sizeof(float), "layer-1 weights");
        validate_allocation(hidden_count, hidden_count, sizeof(float), "layer-2 weights");
        validate_allocation(output_count, hidden_count, sizeof(float), "output weights");
        validate_allocation(capacity, observations, sizeof(float), "policy observations");
        validate_allocation(capacity, actions_count, sizeof(float), "policy logits");
        validate_allocation(capacity, hidden_count, sizeof(float), "policy hidden activations");
        validate_allocation(capacity, output_count, sizeof(float), "actor-critic output");
        try {
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

        check_cuda(cudaMalloc(&train_observations, sizeof(float) * capacity * config.observation_size), "allocate training observations");
        check_cuda(cudaMalloc(&train_masks, sizeof(std::uint8_t) * capacity * config.action_count), "allocate training masks");
        check_cuda(cudaMalloc(&train_actions, sizeof(std::int64_t) * capacity), "allocate training actions");
        check_cuda(cudaMalloc(&train_old_log_probabilities, sizeof(float) * capacity), "allocate training old log probabilities");
        check_cuda(cudaMalloc(&train_old_values, sizeof(float) * capacity), "allocate training old values");
        check_cuda(cudaMalloc(&train_advantages, sizeof(float) * capacity), "allocate training advantages");
        check_cuda(cudaMalloc(&train_returns, sizeof(float) * capacity), "allocate training returns");
        check_cuda(cudaMalloc(&output_gradients, sizeof(float) * capacity * output_size), "allocate output gradients");
        check_cuda(cudaMalloc(&hidden_2_upstream, sizeof(float) * capacity * config.hidden_size), "allocate hidden-2 upstream gradients");
        check_cuda(cudaMalloc(&hidden_2_gradients, sizeof(float) * capacity * config.hidden_size), "allocate hidden-2 gradients");
        check_cuda(cudaMalloc(&hidden_1_upstream, sizeof(float) * capacity * config.hidden_size), "allocate hidden-1 upstream gradients");
        check_cuda(cudaMalloc(&hidden_1_gradients, sizeof(float) * capacity * config.hidden_size), "allocate hidden-1 gradients");

        allocate_optimizer_tensor(weights_1, gradient_weights_1, moment1_weights_1, moment2_weights_1,
                                  static_cast<std::size_t>(config.hidden_size) * config.observation_size);
        allocate_optimizer_tensor(bias_1, gradient_bias_1, moment1_bias_1, moment2_bias_1, config.hidden_size);
        allocate_optimizer_tensor(weights_2, gradient_weights_2, moment1_weights_2, moment2_weights_2,
                                  static_cast<std::size_t>(config.hidden_size) * config.hidden_size);
        allocate_optimizer_tensor(bias_2, gradient_bias_2, moment1_bias_2, moment2_bias_2, config.hidden_size);
        allocate_optimizer_tensor(weights_out, gradient_weights_out, moment1_weights_out, moment2_weights_out,
                                  static_cast<std::size_t>(output_size) * config.hidden_size);
        allocate_optimizer_tensor(bias_out, gradient_bias_out, moment1_bias_out, moment2_bias_out, output_size);
        check_cuda(cudaMalloc(&gradient_square_sum, sizeof(float)), "allocate gradient norm accumulator");
        check_cuda(cudaMalloc(&metric_sums, sizeof(float) * 6), "allocate PPO metric sums");

        std::mt19937_64 random(seed);
        initialize_matrix(weights_1, config.hidden_size, config.observation_size, random);
        initialize_matrix(weights_2, config.hidden_size, config.hidden_size, random);
        initialize_matrix(weights_out, output_size, config.hidden_size, random, 0.01F);
        check_cuda(cudaMemset(bias_1, 0, sizeof(float) * config.hidden_size), "zero layer-1 bias");
        check_cuda(cudaMemset(bias_2, 0, sizeof(float) * config.hidden_size), "zero layer-2 bias");
        check_cuda(cudaMemset(bias_out, 0, sizeof(float) * output_size), "zero output bias");
        } catch (...) {
            release();
            throw;
        }
    }

    static void allocate_optimizer_tensor(
        float* parameter,
        float*& gradient,
        float*& first_moment,
        float*& second_moment,
        std::size_t count) {
        static_cast<void>(parameter);
        check_cuda(cudaMalloc(&gradient, sizeof(float) * count), "allocate parameter gradient");
        check_cuda(cudaMalloc(&first_moment, sizeof(float) * count), "allocate Adam first moment");
        check_cuda(cudaMalloc(&second_moment, sizeof(float) * count), "allocate Adam second moment");
        check_cuda(cudaMemset(first_moment, 0, sizeof(float) * count), "zero Adam first moment");
        check_cuda(cudaMemset(second_moment, 0, sizeof(float) * count), "zero Adam second moment");
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

    void release() noexcept {
        cudaFree(metric_sums);
        cudaFree(gradient_square_sum);
        cudaFree(moment2_bias_out);
        cudaFree(moment1_bias_out);
        cudaFree(gradient_bias_out);
        cudaFree(moment2_weights_out);
        cudaFree(moment1_weights_out);
        cudaFree(gradient_weights_out);
        cudaFree(moment2_bias_2);
        cudaFree(moment1_bias_2);
        cudaFree(gradient_bias_2);
        cudaFree(moment2_weights_2);
        cudaFree(moment1_weights_2);
        cudaFree(gradient_weights_2);
        cudaFree(moment2_bias_1);
        cudaFree(moment1_bias_1);
        cudaFree(gradient_bias_1);
        cudaFree(moment2_weights_1);
        cudaFree(moment1_weights_1);
        cudaFree(gradient_weights_1);
        cudaFree(hidden_1_gradients);
        cudaFree(hidden_1_upstream);
        cudaFree(hidden_2_gradients);
        cudaFree(hidden_2_upstream);
        cudaFree(output_gradients);
        cudaFree(train_returns);
        cudaFree(train_advantages);
        cudaFree(train_old_log_probabilities);
        cudaFree(train_old_values);
        cudaFree(train_actions);
        cudaFree(train_masks);
        cudaFree(train_observations);
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

    ~Impl() { release(); }

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

    void forward_network(const float* input, int batch, cudaStream_t stream) {
        check_cublas(cublasSetStream(cublas, stream), "set actor-critic CUDA stream");
        const int hidden = config.hidden_size;
        const int output_size = config.action_count + 1;
        linear(input, weights_1, hidden_1, batch, config.observation_size, hidden);
        const std::size_t hidden_elements = static_cast<std::size_t>(batch) * hidden;
        bias_tanh_kernel<<<blocks_for(hidden_elements), kThreads, 0, stream>>>(
            hidden_1, bias_1, hidden_elements, hidden);
        linear(hidden_1, weights_2, hidden_2, batch, hidden, hidden);
        bias_tanh_kernel<<<blocks_for(hidden_elements), kThreads, 0, stream>>>(
            hidden_2, bias_2, hidden_elements, hidden);
        linear(hidden_2, weights_out, actor_critic_output, batch, hidden, output_size);
        const std::size_t output_elements = static_cast<std::size_t>(batch) * output_size;
        bias_kernel<<<blocks_for(output_elements), kThreads, 0, stream>>>(
            actor_critic_output, bias_out, output_elements, output_size);
    }

    void weight_gradient(
        const float* input,
        const float* output_gradient,
        float* weight_gradient,
        int batch,
        int input_size,
        int output_size) {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        check_cublas(cublasSgemm(
            cublas, CUBLAS_OP_N, CUBLAS_OP_T,
            input_size, output_size, batch,
            &alpha, input, input_size,
            output_gradient, output_size,
            &beta, weight_gradient, input_size), "actor-critic weight gradient");
    }

    void input_gradient(
        const float* weights,
        const float* output_gradient,
        float* input_gradient,
        int batch,
        int input_size,
        int output_size) {
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        check_cublas(cublasSgemm(
            cublas, CUBLAS_OP_N, CUBLAS_OP_N,
            input_size, batch, output_size,
            &alpha, weights, input_size,
            output_gradient, output_size,
            &beta, input_gradient, input_size), "actor-critic input gradient");
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
    const int batch = static_cast<int>(environment_count);
    impl_->forward_network(device_observations, batch, cuda_stream);
    masked_sample_kernel<<<blocks_for(environment_count), kThreads, 0, cuda_stream>>>(
        impl_->actor_critic_output, device_action_masks, environment_count,
        impl_->config.action_count, sampling_seed, sampling_step, deterministic,
        impl_->logits, impl_->actions, impl_->log_probabilities,
        impl_->values, impl_->entropies);
    check_cuda(cudaGetLastError(), "launch actor-critic inference kernels");
    return {impl_->logits, impl_->actions, impl_->log_probabilities,
            impl_->values, impl_->entropies, environment_count};
}

PpoUpdateMetrics GpuActorCritic::update_ppo(
    const GpuRolloutView& rollout,
    const PpoUpdateConfig& update_config,
    std::uint64_t shuffle_seed,
    void* stream) {
    if (rollout.sample_count == 0 || !rollout.observations || !rollout.action_masks ||
        !rollout.actions || !rollout.old_log_probabilities || !rollout.old_values ||
        !rollout.advantages || !rollout.returns) {
        throw std::invalid_argument("rollout view is incomplete");
    }
    if (update_config.epochs <= 0 || update_config.minibatch_size == 0 ||
        update_config.minibatch_size > impl_->capacity) {
        throw std::invalid_argument("PPO epochs/minibatch size are invalid for policy capacity");
    }
    validate_ppo_coefficients(update_config);
    const auto cuda_stream = as_stream(stream);
    check_cublas(cublasSetStream(impl_->cublas, cuda_stream), "set PPO CUDA stream");
    check_cuda(cudaMemsetAsync(impl_->metric_sums, 0, sizeof(float) * 6, cuda_stream),
               "clear PPO metric sums");
    const int hidden = impl_->config.hidden_size;
    const int output_size = impl_->config.action_count + 1;
    const std::size_t weights_1_count = static_cast<std::size_t>(hidden) * impl_->config.observation_size;
    const std::size_t weights_2_count = static_cast<std::size_t>(hidden) * hidden;
    const std::size_t weights_out_count = static_cast<std::size_t>(output_size) * hidden;
    std::size_t minibatches = 0;
    int epochs_completed = 0;
    bool early_stopped = false;

    const auto host_mix = [](std::uint64_t value) {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    };
    for (int epoch = 0; epoch < update_config.epochs; ++epoch) {
        std::size_t stride = static_cast<std::size_t>(host_mix(shuffle_seed + epoch) % rollout.sample_count);
        if (stride == 0) stride = 1;
        while (std::gcd(stride, rollout.sample_count) != 1) {
            stride = stride + 1 == rollout.sample_count ? 1 : stride + 1;
        }
        const std::size_t permutation_offset =
            static_cast<std::size_t>(host_mix(shuffle_seed ^ (0xD1B54A32D192ED03ULL + epoch)) % rollout.sample_count);

        for (std::size_t batch_offset = 0; batch_offset < rollout.sample_count;
             batch_offset += update_config.minibatch_size) {
            const std::size_t batch_size = std::min(update_config.minibatch_size,
                                                    rollout.sample_count - batch_offset);
            const int batch = static_cast<int>(batch_size);
            ++minibatches;
            gather_minibatch_kernel<<<blocks_for(batch_size), kThreads, 0, cuda_stream>>>(
                rollout.observations, rollout.action_masks, rollout.actions,
                rollout.old_log_probabilities, rollout.old_values,
                rollout.advantages, rollout.returns,
                rollout.sample_count, batch_offset, batch_size, stride, permutation_offset,
                impl_->config.observation_size, impl_->config.action_count,
                impl_->train_observations, impl_->train_masks, impl_->train_actions,
                impl_->train_old_log_probabilities, impl_->train_old_values,
                impl_->train_advantages, impl_->train_returns);
            impl_->forward_network(impl_->train_observations, batch, cuda_stream);
            ppo_output_gradient_kernel<<<blocks_for(batch_size), kThreads, 0, cuda_stream>>>(
                impl_->actor_critic_output, impl_->train_masks, impl_->train_actions,
                impl_->train_old_log_probabilities, impl_->train_old_values,
                impl_->train_advantages, impl_->train_returns,
                batch_size, impl_->config.action_count, update_config.clip_range,
                update_config.value_clip_range,
                update_config.value_coefficient, update_config.entropy_coefficient,
                impl_->output_gradients, impl_->metric_sums);

            const std::size_t hidden_elements = batch_size * hidden;
            impl_->weight_gradient(impl_->hidden_2, impl_->output_gradients,
                                   impl_->gradient_weights_out, batch, hidden, output_size);
            bias_gradient_kernel<<<blocks_for(output_size), kThreads, 0, cuda_stream>>>(
                impl_->output_gradients, impl_->gradient_bias_out, batch_size, output_size);
            impl_->input_gradient(impl_->weights_out, impl_->output_gradients,
                                  impl_->hidden_2_upstream, batch, hidden, output_size);
            tanh_backward_kernel<<<blocks_for(hidden_elements), kThreads, 0, cuda_stream>>>(
                impl_->hidden_2_upstream, impl_->hidden_2, impl_->hidden_2_gradients, hidden_elements);

            impl_->weight_gradient(impl_->hidden_1, impl_->hidden_2_gradients,
                                   impl_->gradient_weights_2, batch, hidden, hidden);
            bias_gradient_kernel<<<blocks_for(hidden), kThreads, 0, cuda_stream>>>(
                impl_->hidden_2_gradients, impl_->gradient_bias_2, batch_size, hidden);
            impl_->input_gradient(impl_->weights_2, impl_->hidden_2_gradients,
                                  impl_->hidden_1_upstream, batch, hidden, hidden);
            tanh_backward_kernel<<<blocks_for(hidden_elements), kThreads, 0, cuda_stream>>>(
                impl_->hidden_1_upstream, impl_->hidden_1, impl_->hidden_1_gradients, hidden_elements);

            impl_->weight_gradient(impl_->train_observations, impl_->hidden_1_gradients,
                                   impl_->gradient_weights_1, batch,
                                   impl_->config.observation_size, hidden);
            bias_gradient_kernel<<<blocks_for(hidden), kThreads, 0, cuda_stream>>>(
                impl_->hidden_1_gradients, impl_->gradient_bias_1, batch_size, hidden);

            gradient_square_sum_kernel<<<1, kThreads, 0, cuda_stream>>>(
                impl_->gradient_weights_1, weights_1_count,
                impl_->gradient_bias_1, hidden,
                impl_->gradient_weights_2, weights_2_count,
                impl_->gradient_bias_2, hidden,
                impl_->gradient_weights_out, weights_out_count,
                impl_->gradient_bias_out, output_size,
                impl_->gradient_square_sum, impl_->metric_sums);

            ++impl_->optimizer_step;
            const float correction1 = 1.0F - std::pow(update_config.adam_beta1,
                                                       static_cast<float>(impl_->optimizer_step));
            const float correction2 = 1.0F - std::pow(update_config.adam_beta2,
                                                       static_cast<float>(impl_->optimizer_step));
            const auto apply_adam = [&](float* parameters, const float* gradients,
                                        float* moment1, float* moment2, std::size_t count) {
                adam_kernel<<<blocks_for(count), kThreads, 0, cuda_stream>>>(
                    parameters, gradients, moment1, moment2, count, impl_->gradient_square_sum,
                    update_config.max_gradient_norm, update_config.learning_rate,
                    update_config.adam_beta1, update_config.adam_beta2, update_config.adam_epsilon,
                    correction1, correction2);
            };
            apply_adam(impl_->weights_1, impl_->gradient_weights_1,
                       impl_->moment1_weights_1, impl_->moment2_weights_1, weights_1_count);
            apply_adam(impl_->bias_1, impl_->gradient_bias_1,
                       impl_->moment1_bias_1, impl_->moment2_bias_1, hidden);
            apply_adam(impl_->weights_2, impl_->gradient_weights_2,
                       impl_->moment1_weights_2, impl_->moment2_weights_2, weights_2_count);
            apply_adam(impl_->bias_2, impl_->gradient_bias_2,
                       impl_->moment1_bias_2, impl_->moment2_bias_2, hidden);
            apply_adam(impl_->weights_out, impl_->gradient_weights_out,
                       impl_->moment1_weights_out, impl_->moment2_weights_out, weights_out_count);
            apply_adam(impl_->bias_out, impl_->gradient_bias_out,
                       impl_->moment1_bias_out, impl_->moment2_bias_out, output_size);
        }
        epochs_completed = epoch + 1;
        if (update_config.target_kl > 0.0F) {
            float cumulative_kl = 0.0F;
            check_cuda(cudaMemcpyAsync(&cumulative_kl, impl_->metric_sums + 3, sizeof(float),
                                       cudaMemcpyDeviceToHost, cuda_stream),
                       "download PPO KL for early stopping");
            check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize PPO KL early stopping");
            const float mean_kl = cumulative_kl /
                static_cast<float>(rollout.sample_count * static_cast<std::size_t>(epochs_completed));
            if (mean_kl > update_config.target_kl) {
                early_stopped = true;
                break;
            }
        }
    }
    check_cuda(cudaGetLastError(), "launch PPO update kernels");
    std::vector<float> metrics(6);
    check_cuda(cudaMemcpyAsync(metrics.data(), impl_->metric_sums, sizeof(float) * metrics.size(),
                               cudaMemcpyDeviceToHost, cuda_stream), "download PPO metrics");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize PPO update");
    const float sample_visits = static_cast<float>(
        rollout.sample_count * static_cast<std::size_t>(epochs_completed));
    return {metrics[0] / sample_visits, metrics[1] / sample_visits,
            metrics[2] / sample_visits, metrics[3] / sample_visits,
            metrics[4] / sample_visits, metrics[5] / static_cast<float>(minibatches),
            minibatches, epochs_completed, early_stopped};
}

namespace {

struct CheckpointHeader {
    std::array<char, 8> magic{};
    std::uint32_t version = 1;
    std::uint32_t observation_size = 0;
    std::uint32_t action_count = 0;
    std::uint32_t hidden_size = 0;
    std::uint64_t optimizer_step = 0;
    std::uint64_t parameter_count = 0;
};

struct CheckpointIntegrity {
    std::uint64_t payload_bytes = 0;
    std::uint64_t payload_checksum = 0;
};

constexpr std::array<char, 8> kCheckpointMagic = {'T', '8', 'V', '2', 'P', 'P', 'O', '\0'};

std::uint64_t checkpoint_checksum(const std::vector<float>& payload) {
    constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t checksum = offset_basis;
    const auto* bytes = reinterpret_cast<const unsigned char*>(payload.data());
    const std::size_t byte_count = payload.size() * sizeof(float);
    for (std::size_t index = 0; index < byte_count; ++index) {
        checksum ^= bytes[index];
        checksum *= prime;
    }
    return checksum;
}

}  // namespace

void GpuActorCritic::save_checkpoint(const std::filesystem::path& path, void* stream) const {
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize before checkpoint save");
    const auto& c = impl_->config;
    const std::size_t w1 = static_cast<std::size_t>(c.hidden_size) * c.observation_size;
    const std::size_t w2 = static_cast<std::size_t>(c.hidden_size) * c.hidden_size;
    const std::size_t wo = static_cast<std::size_t>(c.action_count + 1) * c.hidden_size;
    const std::size_t bo = static_cast<std::size_t>(c.action_count + 1);
    const std::size_t model_parameter_count = w1 + c.hidden_size + w2 + c.hidden_size + wo + bo;
    std::vector<float> payload;
    payload.reserve(model_parameter_count * 3);
    const auto append_tensor = [&](const float* device, std::size_t count) {
        const std::size_t offset = payload.size();
        payload.resize(offset + count);
        check_cuda(cudaMemcpy(payload.data() + offset, device, sizeof(float) * count,
                              cudaMemcpyDeviceToHost), "download checkpoint tensor");
    };
    append_tensor(impl_->weights_1, w1); append_tensor(impl_->bias_1, c.hidden_size);
    append_tensor(impl_->weights_2, w2); append_tensor(impl_->bias_2, c.hidden_size);
    append_tensor(impl_->weights_out, wo); append_tensor(impl_->bias_out, bo);
    append_tensor(impl_->moment1_weights_1, w1); append_tensor(impl_->moment1_bias_1, c.hidden_size);
    append_tensor(impl_->moment1_weights_2, w2); append_tensor(impl_->moment1_bias_2, c.hidden_size);
    append_tensor(impl_->moment1_weights_out, wo); append_tensor(impl_->moment1_bias_out, bo);
    append_tensor(impl_->moment2_weights_1, w1); append_tensor(impl_->moment2_bias_1, c.hidden_size);
    append_tensor(impl_->moment2_weights_2, w2); append_tensor(impl_->moment2_bias_2, c.hidden_size);
    append_tensor(impl_->moment2_weights_out, wo); append_tensor(impl_->moment2_bias_out, bo);

    CheckpointHeader header{kCheckpointMagic, 2,
                            static_cast<std::uint32_t>(c.observation_size),
                            static_cast<std::uint32_t>(c.action_count),
                            static_cast<std::uint32_t>(c.hidden_size),
                            impl_->optimizer_step, parameter_count()};
    const CheckpointIntegrity integrity{
        static_cast<std::uint64_t>(payload.size() * sizeof(float)),
        checkpoint_checksum(payload)};
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("refusing to overwrite checkpoint: " + path.string());
    }
    auto temporary = path;
    temporary += ".tmp";
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("could not open checkpoint for writing: " + temporary.string());
        output.write(reinterpret_cast<const char*>(&header), sizeof(header));
        output.write(reinterpret_cast<const char*>(&integrity), sizeof(integrity));
        output.write(reinterpret_cast<const char*>(payload.data()),
                     static_cast<std::streamsize>(payload.size() * sizeof(float)));
        output.flush();
        if (!output) throw std::runtime_error("checkpoint write failed: " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

void GpuActorCritic::load_checkpoint(
    const std::filesystem::path& path,
    bool load_optimizer_state,
    void* stream) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("could not open checkpoint for reading: " + path.string());
    CheckpointHeader header{};
    input.read(reinterpret_cast<char*>(&header), sizeof(header));
    const auto& c = impl_->config;
    if (!input || header.magic != kCheckpointMagic || (header.version != 1 && header.version != 2) ||
        header.observation_size != static_cast<std::uint32_t>(c.observation_size) ||
        header.action_count != static_cast<std::uint32_t>(c.action_count) ||
        header.hidden_size != static_cast<std::uint32_t>(c.hidden_size) ||
        header.parameter_count != parameter_count()) {
        throw std::runtime_error("checkpoint architecture/version mismatch: " + path.string());
    }
    const std::size_t w1 = static_cast<std::size_t>(c.hidden_size) * c.observation_size;
    const std::size_t w2 = static_cast<std::size_t>(c.hidden_size) * c.hidden_size;
    const std::size_t wo = static_cast<std::size_t>(c.action_count + 1) * c.hidden_size;
    const std::size_t bo = static_cast<std::size_t>(c.action_count + 1);
    const std::size_t model_parameter_count = w1 + c.hidden_size + w2 + c.hidden_size + wo + bo;
    const std::size_t payload_count = model_parameter_count * 3;
    CheckpointIntegrity integrity{};
    if (header.version == 2) {
        input.read(reinterpret_cast<char*>(&integrity), sizeof(integrity));
        if (!input || integrity.payload_bytes != payload_count * sizeof(float)) {
            throw std::runtime_error("checkpoint payload size mismatch: " + path.string());
        }
    }
    std::vector<float> payload(payload_count);
    input.read(reinterpret_cast<char*>(payload.data()),
               static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!input) throw std::runtime_error("checkpoint tensor is truncated: " + path.string());
    if (header.version == 2 && checkpoint_checksum(payload) != integrity.payload_checksum) {
        throw std::runtime_error("checkpoint checksum mismatch: " + path.string());
    }

    const auto cuda_stream = as_stream(stream);
    std::size_t payload_offset = 0;
    const auto upload_tensor = [&](float* device, std::size_t count, bool upload) {
        if (upload) {
            check_cuda(cudaMemcpyAsync(device, payload.data() + payload_offset, sizeof(float) * count,
                                       cudaMemcpyHostToDevice, cuda_stream), "upload checkpoint tensor");
        }
        payload_offset += count;
    };
    upload_tensor(impl_->weights_1, w1, true); upload_tensor(impl_->bias_1, c.hidden_size, true);
    upload_tensor(impl_->weights_2, w2, true); upload_tensor(impl_->bias_2, c.hidden_size, true);
    upload_tensor(impl_->weights_out, wo, true); upload_tensor(impl_->bias_out, bo, true);
    upload_tensor(impl_->moment1_weights_1, w1, load_optimizer_state);
    upload_tensor(impl_->moment1_bias_1, c.hidden_size, load_optimizer_state);
    upload_tensor(impl_->moment1_weights_2, w2, load_optimizer_state);
    upload_tensor(impl_->moment1_bias_2, c.hidden_size, load_optimizer_state);
    upload_tensor(impl_->moment1_weights_out, wo, load_optimizer_state);
    upload_tensor(impl_->moment1_bias_out, bo, load_optimizer_state);
    upload_tensor(impl_->moment2_weights_1, w1, load_optimizer_state);
    upload_tensor(impl_->moment2_bias_1, c.hidden_size, load_optimizer_state);
    upload_tensor(impl_->moment2_weights_2, w2, load_optimizer_state);
    upload_tensor(impl_->moment2_bias_2, c.hidden_size, load_optimizer_state);
    upload_tensor(impl_->moment2_weights_out, wo, load_optimizer_state);
    upload_tensor(impl_->moment2_bias_out, bo, load_optimizer_state);
    if (load_optimizer_state) {
        impl_->optimizer_step = header.optimizer_step;
    } else {
        impl_->optimizer_step = 0;
        check_cuda(cudaMemsetAsync(impl_->moment1_weights_1, 0, sizeof(float) * w1, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment1_bias_1, 0, sizeof(float) * c.hidden_size, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment1_weights_2, 0, sizeof(float) * w2, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment1_bias_2, 0, sizeof(float) * c.hidden_size, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment1_weights_out, 0, sizeof(float) * wo, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment1_bias_out, 0, sizeof(float) * bo, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment2_weights_1, 0, sizeof(float) * w1, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment2_bias_1, 0, sizeof(float) * c.hidden_size, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment2_weights_2, 0, sizeof(float) * w2, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment2_bias_2, 0, sizeof(float) * c.hidden_size, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment2_weights_out, 0, sizeof(float) * wo, cuda_stream), "reset Adam state");
        check_cuda(cudaMemsetAsync(impl_->moment2_bias_out, 0, sizeof(float) * bo, cuda_stream), "reset Adam state");
    }
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize checkpoint load");
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
    float* block_sums,
    float* block_square_sums) {
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
        block_sums[blockIdx.x] = shared_sum[0];
        block_square_sums[blockIdx.x] = shared_square_sum[0];
    }
}

__global__ void finalize_advantage_stats_kernel(
    const float* block_sums,
    const float* block_square_sums,
    int block_count,
    float* statistics) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    float sum = 0.0F;
    float square_sum = 0.0F;
    for (int block = 0; block < block_count; ++block) {
        sum += block_sums[block];
        square_sum += block_square_sums[block];
    }
    statistics[0] = sum;
    statistics[1] = square_sum;
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
    float* advantage_block_sums = nullptr;
    float* advantage_block_square_sums = nullptr;
    int advantage_reduction_blocks = 0;

    Impl(std::size_t environment_count, std::size_t horizon, ActorCriticConfig network_config)
        : environments(environment_count), rollout_horizon(horizon),
          samples(checked_product(environment_count, horizon, "rollout sample count")),
          config(network_config),
          advantage_reduction_blocks(std::min(blocks_for(samples), 1024)) {
        if (environments == 0 || rollout_horizon == 0) {
            throw std::invalid_argument("rollout dimensions must be greater than zero");
        }
        validate_actor_config(config);
        validate_allocation(samples, static_cast<std::size_t>(config.observation_size),
                            sizeof(float), "rollout observations");
        validate_allocation(samples, static_cast<std::size_t>(config.action_count),
                            sizeof(std::uint8_t), "rollout action masks");
        validate_allocation(samples, 1, sizeof(std::int64_t), "rollout actions");
        try {
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
        check_cuda(cudaMalloc(&advantage_block_sums, sizeof(float) * advantage_reduction_blocks),
                   "allocate advantage block sums");
        check_cuda(cudaMalloc(&advantage_block_square_sums, sizeof(float) * advantage_reduction_blocks),
                   "allocate advantage block square sums");
        } catch (...) {
            release();
            throw;
        }
    }

    void release() noexcept {
        cudaFree(advantage_block_square_sums);
        cudaFree(advantage_block_sums);
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

    ~Impl() { release(); }
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
    if (!std::isfinite(gamma) || !std::isfinite(gae_lambda) ||
        !(gamma >= 0.0F && gamma <= 1.0F && gae_lambda >= 0.0F && gae_lambda <= 1.0F)) {
        throw std::invalid_argument("gamma and gae_lambda must be in [0, 1]");
    }
    const auto cuda_stream = as_stream(stream);
    gae_kernel<<<blocks_for(impl_->environments), kThreads, 0, cuda_stream>>>(
        impl_->rewards, impl_->old_values, impl_->terminated, bootstrap_values,
        impl_->environments, impl_->rollout_horizon, gamma, gae_lambda,
        impl_->advantages, impl_->returns);
    if (normalize) {
        advantage_stats_kernel<<<impl_->advantage_reduction_blocks, kThreads, 0, cuda_stream>>>(
            impl_->advantages, impl_->samples,
            impl_->advantage_block_sums, impl_->advantage_block_square_sums);
        finalize_advantage_stats_kernel<<<1, 1, 0, cuda_stream>>>(
            impl_->advantage_block_sums, impl_->advantage_block_square_sums,
            impl_->advantage_reduction_blocks, impl_->advantage_statistics);
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

std::vector<float> GpuRolloutBuffer::download_rewards(void* stream) const {
    return download(impl_->rewards, impl_->samples, as_stream(stream), "download rollout rewards");
}

std::vector<float> GpuRolloutBuffer::download_values(void* stream) const {
    return download(impl_->old_values, impl_->samples, as_stream(stream), "download rollout values");
}

std::vector<std::uint8_t> GpuRolloutBuffer::download_terminated(void* stream) const {
    return download(impl_->terminated, impl_->samples, as_stream(stream), "download rollout terminal flags");
}

}  // namespace t8::v2
