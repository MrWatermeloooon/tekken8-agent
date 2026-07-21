#include "t8_v2/ppo.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
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

__global__ void gather_minibatch_kernel(
    const float* source_observations,
    const std::uint8_t* source_masks,
    const std::int64_t* source_actions,
    const float* source_old_log_probabilities,
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
    advantages[lane] = source_advantages[source];
    returns[lane] = source_returns[source];
}

__global__ void ppo_output_gradient_kernel(
    const float* actor_critic_output,
    const std::uint8_t* masks,
    const std::int64_t* actions,
    const float* old_log_probabilities,
    const float* advantages,
    const float* returns,
    std::size_t batch_size,
    int action_count,
    float clip_range,
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
    const float value_error = actor_critic_output[base + action_count] - returns[lane];
    output_gradients[base + action_count] =
        value_coefficient * value_error / static_cast<float>(batch_size);

    atomicAdd(metric_sums + 0, policy_loss);
    atomicAdd(metric_sums + 1, 0.5F * value_error * value_error);
    atomicAdd(metric_sums + 2, entropy);
    atomicAdd(metric_sums + 3, (ratio - 1.0F) - log_ratio);
    atomicAdd(metric_sums + 4, fabsf(ratio - 1.0F) > clip_range ? 1.0F : 0.0F);
}

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

__global__ void accumulate_square_kernel(const float* gradient, std::size_t count, float* sum) {
    __shared__ float shared[kThreads];
    float local = 0.0F;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count; index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        local += gradient[index] * gradient[index];
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) atomicAdd(sum, shared[0]);
}

__global__ void accumulate_gradient_norm_metric_kernel(const float* square_sum, float* metrics) {
    if (blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(metrics + 5, sqrtf(square_sum[0]));
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

        check_cuda(cudaMalloc(&train_observations, sizeof(float) * capacity * config.observation_size), "allocate training observations");
        check_cuda(cudaMalloc(&train_masks, sizeof(std::uint8_t) * capacity * config.action_count), "allocate training masks");
        check_cuda(cudaMalloc(&train_actions, sizeof(std::int64_t) * capacity), "allocate training actions");
        check_cuda(cudaMalloc(&train_old_log_probabilities, sizeof(float) * capacity), "allocate training old log probabilities");
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

    ~Impl() {
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
        !rollout.actions || !rollout.old_log_probabilities || !rollout.advantages || !rollout.returns) {
        throw std::invalid_argument("rollout view is incomplete");
    }
    if (update_config.epochs <= 0 || update_config.minibatch_size == 0 ||
        update_config.minibatch_size > impl_->capacity) {
        throw std::invalid_argument("PPO epochs/minibatch size are invalid for policy capacity");
    }
    if (update_config.learning_rate <= 0.0F || update_config.clip_range < 0.0F ||
        update_config.max_gradient_norm <= 0.0F) {
        throw std::invalid_argument("PPO optimizer coefficients must be positive");
    }
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
                rollout.old_log_probabilities, rollout.advantages, rollout.returns,
                rollout.sample_count, batch_offset, batch_size, stride, permutation_offset,
                impl_->config.observation_size, impl_->config.action_count,
                impl_->train_observations, impl_->train_masks, impl_->train_actions,
                impl_->train_old_log_probabilities, impl_->train_advantages, impl_->train_returns);
            impl_->forward_network(impl_->train_observations, batch, cuda_stream);
            ppo_output_gradient_kernel<<<blocks_for(batch_size), kThreads, 0, cuda_stream>>>(
                impl_->actor_critic_output, impl_->train_masks, impl_->train_actions,
                impl_->train_old_log_probabilities, impl_->train_advantages, impl_->train_returns,
                batch_size, impl_->config.action_count, update_config.clip_range,
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

            check_cuda(cudaMemsetAsync(impl_->gradient_square_sum, 0, sizeof(float), cuda_stream),
                       "clear gradient norm");
            const auto add_squares = [&](const float* gradient, std::size_t count) {
                const int blocks = std::min(blocks_for(count), 256);
                accumulate_square_kernel<<<blocks, kThreads, 0, cuda_stream>>>(
                    gradient, count, impl_->gradient_square_sum);
            };
            add_squares(impl_->gradient_weights_1, weights_1_count);
            add_squares(impl_->gradient_bias_1, hidden);
            add_squares(impl_->gradient_weights_2, weights_2_count);
            add_squares(impl_->gradient_bias_2, hidden);
            add_squares(impl_->gradient_weights_out, weights_out_count);
            add_squares(impl_->gradient_bias_out, output_size);
            accumulate_gradient_norm_metric_kernel<<<1, 1, 0, cuda_stream>>>(
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
    }
    check_cuda(cudaGetLastError(), "launch PPO update kernels");
    std::vector<float> metrics(6);
    check_cuda(cudaMemcpyAsync(metrics.data(), impl_->metric_sums, sizeof(float) * metrics.size(),
                               cudaMemcpyDeviceToHost, cuda_stream), "download PPO metrics");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize PPO update");
    const float sample_visits = static_cast<float>(rollout.sample_count * update_config.epochs);
    return {metrics[0] / sample_visits, metrics[1] / sample_visits,
            metrics[2] / sample_visits, metrics[3] / sample_visits,
            metrics[4] / sample_visits, metrics[5] / static_cast<float>(minibatches),
            minibatches};
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
