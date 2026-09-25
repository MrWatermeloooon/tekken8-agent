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
// policy loss, value loss, entropy, KL, clip fraction, gradient norm,
// decision samples, decision entropy.
constexpr std::size_t kMetricCount = 8;

void validate_actor_config(const ActorCriticConfig& config) {
    if (config.observation_size <= 0 || config.action_count <= 1 || config.hidden_size <= 0 ||
        config.action_count == std::numeric_limits<int>::max()) {
        throw std::invalid_argument("actor-critic dimensions must be positive and representable");
    }
    if (config.action_feature_size < 0 || config.universal_action_count < 0 ||
        config.universal_action_count > config.action_count || config.catalog_sha256.size() != 64 ||
        config.roster_version.empty() || config.roster_version.size() >= 24 ||
        config.observation_contract.empty() || config.observation_contract.size() >= 32 ||
        config.action_contract.empty() || config.action_contract.size() >= 32) {
        throw std::invalid_argument("actor-critic policy contract metadata is invalid");
    }
    if (!std::isfinite(config.observation_clip) || config.observation_clip <= 0.0F ||
        !std::isfinite(config.observation_epsilon) || config.observation_epsilon <= 0.0F) {
        throw std::invalid_argument("observation normalizer clip/epsilon must be finite and positive");
    }
}

std::size_t checked_product(std::size_t left, std::size_t right, const char* description) {
    if (right != 0 && left > std::numeric_limits<std::size_t>::max() / right) {
        throw std::invalid_argument(std::string(description) + " overflows size_t");
    }
    return left * right;
}

std::size_t checked_rollout_samples(std::size_t environments, std::size_t horizon) {
    if (environments == 0 || horizon == 0) {
        throw std::invalid_argument("rollout dimensions must be greater than zero");
    }
    return checked_product(environments, horizon, "rollout sample count");
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

__global__ void normalize_observations_kernel(
    const float* input,
    const float* mean,
    const float* inverse_standard_deviation,
    float* output,
    std::size_t element_count,
    int width,
    float clip) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= element_count) return;
    const int feature = static_cast<int>(index % width);
    const float value = (input[index] - mean[feature]) * inverse_standard_deviation[feature];
    output[index] = fminf(clip, fmaxf(-clip, value));
}

// One block per feature. Two passes (mean, then centered squares) in double
// keep the variance stable for large rollouts; fixed-order shared-memory
// reductions keep it deterministic for exact resume.
__global__ void observation_moments_kernel(
    const float* observations,
    std::size_t sample_count,
    int width,
    double* batch_mean,
    double* batch_m2) {
    __shared__ double shared[kThreads];
    const int feature = blockIdx.x;
    if (feature >= width) return;
    double local = 0.0;
    for (std::size_t sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        local += static_cast<double>(observations[sample * width + feature]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    const double mean = shared[0] / static_cast<double>(sample_count);
    __syncthreads();
    local = 0.0;
    for (std::size_t sample = threadIdx.x; sample < sample_count; sample += blockDim.x) {
        const double centered = static_cast<double>(observations[sample * width + feature]) - mean;
        local += centered * centered;
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        batch_mean[feature] = mean;
        batch_m2[feature] = shared[0];
    }
}

// Chan et al. parallel merge of the rollout's moments into the running ones.
__global__ void merge_observation_moments_kernel(
    const double* batch_mean,
    const double* batch_m2,
    double running_count,
    double batch_count,
    float* mean,
    float* variance,
    int width) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= width) return;
    const double total = running_count + batch_count;
    const double delta = batch_mean[feature] - static_cast<double>(mean[feature]);
    const double m2 = static_cast<double>(variance[feature]) * running_count + batch_m2[feature] +
        delta * delta * running_count * batch_count / total;
    mean[feature] = static_cast<float>(static_cast<double>(mean[feature]) + delta * batch_count / total);
    variance[feature] = static_cast<float>(m2 / total);
}

__global__ void refresh_inverse_standard_deviation_kernel(
    const float* variance,
    float* inverse_standard_deviation,
    int width,
    float epsilon) {
    const int feature = blockIdx.x * blockDim.x + threadIdx.x;
    if (feature >= width) return;
    inverse_standard_deviation[feature] = 1.0F / sqrtf(variance[feature] + epsilon);
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

__global__ void parametric_policy_output_kernel(
    const float* query_and_value,
    const float* action_features,
    std::size_t batch_size,
    int action_count,
    int feature_size,
    float* policy_output) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch_size) return;
    const float scale = rsqrtf(static_cast<float>(feature_size));
    const std::size_t query_base = lane * static_cast<std::size_t>(feature_size + 1);
    const std::size_t output_base = lane * static_cast<std::size_t>(action_count + 1);
    for (int action = 0; action < action_count; ++action) {
        float score = 0.0F;
        const std::size_t feature_base = static_cast<std::size_t>(action) * feature_size;
        for (int feature = 0; feature < feature_size; ++feature) {
            score += query_and_value[query_base + feature] *
                action_features[feature_base + feature];
        }
        policy_output[output_base + action] = score * scale;
    }
    policy_output[output_base + action_count] = query_and_value[query_base + feature_size];
}

__global__ void parametric_policy_gradient_kernel(
    const float* policy_gradients,
    const float* action_features,
    std::size_t batch_size,
    int action_count,
    int feature_size,
    float* query_gradients) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    const std::size_t query_width = static_cast<std::size_t>(feature_size + 1);
    if (index >= batch_size * query_width) return;
    const std::size_t lane = index / query_width;
    const int feature = static_cast<int>(index % query_width);
    const std::size_t policy_base = lane * static_cast<std::size_t>(action_count + 1);
    if (feature == feature_size) {
        query_gradients[index] = policy_gradients[policy_base + action_count];
        return;
    }
    const float scale = rsqrtf(static_cast<float>(feature_size));
    float gradient = 0.0F;
    for (int action = 0; action < action_count; ++action) {
        gradient += policy_gradients[policy_base + action] *
            action_features[static_cast<std::size_t>(action) * feature_size + feature];
    }
    query_gradients[index] = gradient * scale;
}

__global__ void gather_minibatch_kernel(
    const float* source_observations,
    const std::uint8_t* source_masks,
    const std::int64_t* source_actions,
    const float* source_old_log_probabilities,
    const float* source_old_values,
    const float* source_advantages,
    const float* source_returns,
    const std::size_t* permutation,
    std::size_t batch_offset,
    std::size_t batch_size,
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
    const std::size_t source = permutation[batch_offset + lane];
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

__global__ void validate_rollout_kernel(
    const float* observations,
    const std::uint8_t* masks,
    const std::int64_t* actions,
    const float* old_log_probabilities,
    const float* old_values,
    const float* advantages,
    const float* returns,
    std::size_t sample_count,
    int observation_size,
    int action_count,
    int* invalid) {
    const std::size_t sample = blockIdx.x * blockDim.x + threadIdx.x;
    if (sample >= sample_count) return;
    bool finite = isfinite(old_log_probabilities[sample]) && isfinite(old_values[sample]) &&
        isfinite(advantages[sample]) && isfinite(returns[sample]);
    for (int feature = 0; feature < observation_size && finite; ++feature) {
        finite = isfinite(observations[sample * observation_size + feature]);
    }
    bool any_legal = false;
    for (int action = 0; action < action_count; ++action) {
        any_legal = any_legal || masks[sample * action_count + action] != 0;
    }
    const std::int64_t selected = actions[sample];
    const bool valid_action = selected >= 0 && selected < action_count &&
        ((any_legal && masks[sample * action_count + selected] != 0) ||
         (!any_legal && selected == 0));
    if (!finite || !valid_action) atomicExch(invalid, 1);
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
    float* metric_sums,
    float* sample_entropy = nullptr,           // [rollout sample] entropy, first epoch only
    std::uint8_t* sample_decision = nullptr,   // [rollout sample] 1 if more than one action was legal
    const std::size_t* permutation = nullptr,  // minibatch lane -> rollout sample
    std::size_t batch_offset = 0) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= batch_size) return;
    const int width = action_count + 1;
    const std::size_t base = lane * width;
    const std::size_t mask_base = lane * action_count;
    bool any_legal = false;
    float maximum = -3.402823466e38F;
    for (int action = 0; action < action_count; ++action) {
        if (masks[mask_base + action]) {
            any_legal = true;
            maximum = fmaxf(maximum, actor_critic_output[base + action]);
        }
    }
    if (!any_legal) maximum = actor_critic_output[base];
    float probability_sum = 0.0F;
    for (int action = 0; action < action_count; ++action) {
        if ((any_legal && masks[mask_base + action]) || (!any_legal && action == 0)) {
            probability_sum += expf(actor_critic_output[base + action] - maximum);
        }
    }
    probability_sum = fmaxf(probability_sum, 1e-20F);
    const float log_sum = logf(probability_sum) + maximum;
    const int recorded = static_cast<int>(actions[lane]);
    const int selected = any_legal ? recorded : 0;
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
    int legal_count = 0;
    for (int action = 0; action < action_count; ++action) {
        const bool legal = (any_legal && masks[mask_base + action]) || (!any_legal && action == 0);
        if (!legal) {
            output_gradients[base + action] = 0.0F;
            continue;
        }
        ++legal_count;
        const float log_probability = actor_critic_output[base + action] - log_sum;
        const float probability = expf(log_probability);
        entropy -= probability * log_probability;
        const float selected_delta = action == selected ? 1.0F : 0.0F;
        output_gradients[base + action] =
            (log_probability_gradient * (selected_delta - probability)) /
            static_cast<float>(batch_size);
    }
    for (int action = 0; action < action_count; ++action) {
        if ((any_legal && masks[mask_base + action]) || (!any_legal && action == 0)) {
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
    if (legal_count > 1) {
        atomicAdd(metric_sums + 6, 1.0F);
        atomicAdd(metric_sums + 7, entropy);
    }
    if (sample_entropy != nullptr) {
        const std::size_t sample = permutation[batch_offset + lane];
        sample_entropy[sample] = entropy;
        sample_decision[sample] = legal_count > 1 ? 1 : 0;
    }
}

// Sums the per-sample entropy of decision samples in a fixed order (one block,
// strided per thread, then a fixed tree), so the result is bit-reproducible;
// atomicAdd order is not, and this value feeds back into training (the entropy target).
__global__ void deterministic_decision_entropy_kernel(
    const float* sample_entropy,
    const std::uint8_t* sample_decision,
    std::size_t count,
    float* result) {  // [entropy sum, decision count]
    __shared__ double entropy_sums[kThreads];
    __shared__ double decision_counts[kThreads];
    double entropy_sum = 0.0;
    double decisions = 0.0;
    for (std::size_t index = threadIdx.x; index < count; index += blockDim.x) {
        if (sample_decision[index]) {
            entropy_sum += sample_entropy[index];
            decisions += 1.0;
        }
    }
    entropy_sums[threadIdx.x] = entropy_sum;
    decision_counts[threadIdx.x] = decisions;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            entropy_sums[threadIdx.x] += entropy_sums[threadIdx.x + stride];
            decision_counts[threadIdx.x] += decision_counts[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        result[0] = static_cast<float>(entropy_sums[0]);
        result[1] = static_cast<float>(decision_counts[0]);
    }
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
    // One block per output feature: threads cooperatively reduce over the
    // batch dimension instead of a single thread scanning the whole batch,
    // which collapsed to `width`-way parallelism regardless of batch size.
    __shared__ float shared[kThreads];
    const int feature = blockIdx.x;
    if (feature >= width) return;
    float local = 0.0F;
    for (std::size_t lane = threadIdx.x; lane < batch_size; lane += blockDim.x) {
        local += output_gradient[lane * width + feature];
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) shared[threadIdx.x] += shared[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) bias_gradient[feature] = shared[0];
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
    if (!isfinite(norm)) return;
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
    const bool any_legal = std::any_of(
        action_mask.begin(), action_mask.end(), [](std::uint8_t value) { return value != 0; });
    const bool valid_selected_action = action >= 0 &&
        static_cast<std::size_t>(action) < action_mask.size() &&
        ((any_legal && action_mask[static_cast<std::size_t>(action)] != 0) ||
         (!any_legal && action == 0));
    if (logits_and_value.size() < 3 || action_mask.size() + 1 != logits_and_value.size() ||
        !valid_selected_action ||
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
    ScopedDeviceBuffer<float> device_metrics(kMetricCount);
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
    check_cuda(cudaMemset(device_metrics.pointer, 0, sizeof(float) * kMetricCount),
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
    float* parametric_policy_output = nullptr;
    float* parametric_policy_gradients = nullptr;
    float* action_features = nullptr;
    bool action_features_initialized = false;
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
    int* invalid_rollout = nullptr;
    std::uint64_t optimizer_step = 0;

    float* observation_mean = nullptr;
    float* observation_variance = nullptr;
    float* observation_inverse_standard_deviation = nullptr;
    double* observation_batch_mean = nullptr;
    double* observation_batch_m2 = nullptr;
    float* normalized_observations = nullptr;
    std::uint64_t observation_count = 0;

    [[nodiscard]] bool normalizing() const noexcept {
        return config.observation_normalization && observation_count > 0;
    }

    [[nodiscard]] bool parametric() const noexcept { return config.action_feature_size > 0; }
    [[nodiscard]] int network_output_size() const noexcept {
        return (parametric() ? config.action_feature_size : config.action_count) + 1;
    }

    Impl(std::size_t requested_capacity, ActorCriticConfig network_config, std::uint64_t seed)
        : capacity(requested_capacity), config(network_config) {
        if (capacity == 0 || capacity > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            throw std::invalid_argument("policy capacity must be positive and fit cuBLAS dimensions");
        }
        validate_actor_config(config);
        const auto observations = static_cast<std::size_t>(config.observation_size);
        const auto actions_count = static_cast<std::size_t>(config.action_count);
        const auto hidden_count = static_cast<std::size_t>(config.hidden_size);
        const auto output_count = static_cast<std::size_t>(network_output_size());
        validate_allocation(hidden_count, observations, sizeof(float), "layer-1 weights");
        validate_allocation(hidden_count, hidden_count, sizeof(float), "layer-2 weights");
        validate_allocation(output_count, hidden_count, sizeof(float), "output weights");
        validate_allocation(capacity, observations, sizeof(float), "policy observations");
        validate_allocation(capacity, actions_count, sizeof(float), "policy logits");
        validate_allocation(capacity, hidden_count, sizeof(float), "policy hidden activations");
        validate_allocation(capacity, output_count, sizeof(float), "actor-critic output");
        try {
        const int output_size = network_output_size();
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
        if (parametric()) {
            check_cuda(cudaMalloc(&parametric_policy_output,
                                  sizeof(float) * capacity * (config.action_count + 1)),
                       "allocate parametric policy output");
            check_cuda(cudaMalloc(&parametric_policy_gradients,
                                  sizeof(float) * capacity * (config.action_count + 1)),
                       "allocate parametric policy gradients");
            check_cuda(cudaMalloc(&action_features,
                                  sizeof(float) * config.action_count * config.action_feature_size),
                       "allocate parametric action features");
        }
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
        check_cuda(cudaMalloc(&metric_sums, sizeof(float) * kMetricCount), "allocate PPO metric sums");
        check_cuda(cudaMalloc(&invalid_rollout, sizeof(int)), "allocate PPO finite guard");
        check_cuda(cudaMalloc(&observation_mean, sizeof(float) * config.observation_size),
                   "allocate observation mean");
        check_cuda(cudaMalloc(&observation_variance, sizeof(float) * config.observation_size),
                   "allocate observation variance");
        check_cuda(cudaMalloc(&observation_inverse_standard_deviation,
                              sizeof(float) * config.observation_size),
                   "allocate observation inverse standard deviation");
        check_cuda(cudaMalloc(&observation_batch_mean, sizeof(double) * config.observation_size),
                   "allocate observation batch mean");
        check_cuda(cudaMalloc(&observation_batch_m2, sizeof(double) * config.observation_size),
                   "allocate observation batch M2");
        check_cuda(cudaMalloc(&normalized_observations,
                              sizeof(float) * capacity * config.observation_size),
                   "allocate normalized observations");
        reset_observation_statistics(nullptr);

        std::mt19937_64 random(seed);
        if (config.orthogonal_initialization) {
            const float hidden_gain = std::sqrt(2.0F);
            upload_matrix(weights_1, orthogonal_matrix(
                config.hidden_size, config.observation_size, hidden_gain, random));
            upload_matrix(weights_2, orthogonal_matrix(
                config.hidden_size, config.hidden_size, hidden_gain, random));
            auto output_weights = orthogonal_matrix(output_size, config.hidden_size, 1.0F, random);
            // Policy (or parametric query) rows start near-uniform; the value
            // row keeps unit gain.
            for (int row = 0; row + 1 < output_size; ++row) {
                for (int column = 0; column < config.hidden_size; ++column) {
                    output_weights[static_cast<std::size_t>(row) * config.hidden_size + column] *= 0.01F;
                }
            }
            upload_matrix(weights_out, output_weights);
        } else {
            initialize_matrix(weights_1, config.hidden_size, config.observation_size, random);
            initialize_matrix(weights_2, config.hidden_size, config.hidden_size, random);
            initialize_matrix(weights_out, output_size, config.hidden_size, random, 0.01F);
        }
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

    // Row-major [rows, columns] with orthonormal rows (rows <= columns) or
    // orthonormal columns (rows > columns), scaled by gain. Modified
    // Gram-Schmidt with one re-orthogonalization pass, in double.
    static std::vector<float> orthogonal_matrix(
        int rows,
        int columns,
        float gain,
        std::mt19937_64& random) {
        const auto tall = static_cast<std::size_t>(std::max(rows, columns));
        const auto narrow = static_cast<std::size_t>(std::min(rows, columns));
        std::normal_distribution<double> distribution(0.0, 1.0);
        std::vector<double> basis(tall * narrow);
        for (double& value : basis) value = distribution(random);
        for (std::size_t vector = 0; vector < narrow; ++vector) {
            double* current = basis.data() + vector * tall;
            for (int pass = 0; pass < 2; ++pass) {
                for (std::size_t previous = 0; previous < vector; ++previous) {
                    const double* other = basis.data() + previous * tall;
                    double dot = 0.0;
                    for (std::size_t index = 0; index < tall; ++index) dot += other[index] * current[index];
                    for (std::size_t index = 0; index < tall; ++index) current[index] -= dot * other[index];
                }
            }
            double norm = 0.0;
            for (std::size_t index = 0; index < tall; ++index) norm += current[index] * current[index];
            norm = std::sqrt(norm);
            for (std::size_t index = 0; index < tall; ++index) current[index] /= norm;
        }
        std::vector<float> result(static_cast<std::size_t>(rows) * columns);
        for (int row = 0; row < rows; ++row) {
            for (int column = 0; column < columns; ++column) {
                const double value = rows >= columns
                    ? basis[static_cast<std::size_t>(column) * tall + row]
                    : basis[static_cast<std::size_t>(row) * tall + column];
                result[static_cast<std::size_t>(row) * columns + column] =
                    gain * static_cast<float>(value);
            }
        }
        return result;
    }

    static void upload_matrix(float* destination, const std::vector<float>& host) {
        check_cuda(cudaMemcpy(destination, host.data(), sizeof(float) * host.size(),
                              cudaMemcpyHostToDevice), "initialize network matrix");
    }

    void reset_observation_statistics(cudaStream_t stream) {
        const std::vector<float> zeros(static_cast<std::size_t>(config.observation_size), 0.0F);
        const std::vector<float> ones(static_cast<std::size_t>(config.observation_size), 1.0F);
        upload_observation_statistics(zeros.data(), ones.data(), stream);
        check_cuda(cudaStreamSynchronize(stream), "reset observation normalizer");
        observation_count = 0;
    }

    // Host pointers must stay valid until the stream is synchronized.
    void upload_observation_statistics(const float* mean, const float* variance, cudaStream_t stream) {
        const std::size_t bytes = sizeof(float) * config.observation_size;
        check_cuda(cudaMemcpyAsync(observation_mean, mean, bytes, cudaMemcpyHostToDevice, stream),
                   "upload observation mean");
        check_cuda(cudaMemcpyAsync(observation_variance, variance, bytes, cudaMemcpyHostToDevice, stream),
                   "upload observation variance");
        refresh_inverse_standard_deviation_kernel<<<blocks_for(config.observation_size), kThreads, 0, stream>>>(
            observation_variance, observation_inverse_standard_deviation,
            config.observation_size, config.observation_epsilon);
        check_cuda(cudaGetLastError(), "refresh observation normalizer");
    }

    void merge_observation_statistics(const float* observations, std::size_t samples, cudaStream_t stream) {
        observation_moments_kernel<<<config.observation_size, kThreads, 0, stream>>>(
            observations, samples, config.observation_size,
            observation_batch_mean, observation_batch_m2);
        merge_observation_moments_kernel<<<blocks_for(config.observation_size), kThreads, 0, stream>>>(
            observation_batch_mean, observation_batch_m2,
            static_cast<double>(observation_count), static_cast<double>(samples),
            observation_mean, observation_variance, config.observation_size);
        refresh_inverse_standard_deviation_kernel<<<blocks_for(config.observation_size), kThreads, 0, stream>>>(
            observation_variance, observation_inverse_standard_deviation,
            config.observation_size, config.observation_epsilon);
        check_cuda(cudaGetLastError(), "merge observation normalizer statistics");
        observation_count += samples;
    }

    void release() noexcept {
        cudaFree(normalized_observations);
        cudaFree(observation_batch_m2);
        cudaFree(observation_batch_mean);
        cudaFree(observation_inverse_standard_deviation);
        cudaFree(observation_variance);
        cudaFree(observation_mean);
        cudaFree(invalid_rollout);
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
        cudaFree(parametric_policy_output);
        cudaFree(parametric_policy_gradients);
        cudaFree(action_features);
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

    // Returns the layer-1 input actually used (normalized copy or the caller's
    // buffer), which the backward pass needs for the layer-1 weight gradient.
    const float* forward_network(const float* input, int batch, cudaStream_t stream) {
        check_cublas(cublasSetStream(cublas, stream), "set actor-critic CUDA stream");
        const int hidden = config.hidden_size;
        const int output_size = network_output_size();
        if (normalizing()) {
            const std::size_t input_elements = static_cast<std::size_t>(batch) * config.observation_size;
            normalize_observations_kernel<<<blocks_for(input_elements), kThreads, 0, stream>>>(
                input, observation_mean, observation_inverse_standard_deviation,
                normalized_observations, input_elements, config.observation_size,
                config.observation_clip);
            input = normalized_observations;
        }
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
        return input;
    }

    float* materialize_policy_output(int batch, cudaStream_t stream) {
        if (!parametric()) return actor_critic_output;
        if (!action_features_initialized) {
            throw std::logic_error("parametric policy action features have not been installed");
        }
        parametric_policy_output_kernel<<<blocks_for(static_cast<std::size_t>(batch)),
                                          kThreads, 0, stream>>>(
            actor_critic_output, action_features, static_cast<std::size_t>(batch),
            config.action_count, config.action_feature_size, parametric_policy_output);
        return parametric_policy_output;
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
    const int output_size = impl_->network_output_size();
    return static_cast<std::size_t>(c.hidden_size) * c.observation_size + c.hidden_size +
           static_cast<std::size_t>(c.hidden_size) * c.hidden_size + c.hidden_size +
           static_cast<std::size_t>(output_size) * c.hidden_size + output_size;
}

const ActorCriticConfig& GpuActorCritic::config() const noexcept { return impl_->config; }

void GpuActorCritic::set_action_features(
    std::span<const float> features,
    void* stream) {
    if (!impl_->parametric()) {
        throw std::logic_error("action features require a parametric actor-critic config");
    }
    const std::size_t expected = static_cast<std::size_t>(impl_->config.action_count) *
        impl_->config.action_feature_size;
    if (features.size() != expected ||
        !std::all_of(features.begin(), features.end(), [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("parametric action feature tensor has the wrong shape or non-finite values");
    }
    check_cuda(cudaMemcpyAsync(impl_->action_features, features.data(), sizeof(float) * features.size(),
                               cudaMemcpyHostToDevice, as_stream(stream)),
               "upload parametric action features");
    impl_->action_features_initialized = true;
}

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
    const float* policy_output = impl_->materialize_policy_output(batch, cuda_stream);
    masked_sample_kernel<<<blocks_for(environment_count), kThreads, 0, cuda_stream>>>(
        policy_output, device_action_masks, environment_count,
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
    check_cuda(cudaMemsetAsync(impl_->invalid_rollout, 0, sizeof(int), cuda_stream),
               "clear PPO finite guard");
    validate_rollout_kernel<<<blocks_for(rollout.sample_count), kThreads, 0, cuda_stream>>>(
        rollout.observations, rollout.action_masks, rollout.actions,
        rollout.old_log_probabilities, rollout.old_values, rollout.advantages, rollout.returns,
        rollout.sample_count, impl_->config.observation_size, impl_->config.action_count,
        impl_->invalid_rollout);
    int invalid_rollout = 0;
    check_cuda(cudaMemcpyAsync(&invalid_rollout, impl_->invalid_rollout, sizeof(int),
                               cudaMemcpyDeviceToHost, cuda_stream),
               "download PPO finite guard");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize PPO finite guard");
    if (invalid_rollout != 0) {
        throw std::runtime_error("PPO rollout contains a non-finite value or invalid action");
    }
    check_cuda(cudaMemsetAsync(impl_->metric_sums, 0, sizeof(float) * kMetricCount, cuda_stream),
               "clear PPO metric sums");
    const int hidden = impl_->config.hidden_size;
    const int output_size = impl_->network_output_size();
    const std::size_t weights_1_count = static_cast<std::size_t>(hidden) * impl_->config.observation_size;
    const std::size_t weights_2_count = static_cast<std::size_t>(hidden) * hidden;
    const std::size_t weights_out_count = static_cast<std::size_t>(output_size) * hidden;
    std::size_t minibatches = 0;
    int epochs_completed = 0;
    bool early_stopped = false;
    ScopedDeviceBuffer<std::size_t> device_permutation(rollout.sample_count);
    ScopedDeviceBuffer<float> sample_entropy(rollout.sample_count);
    ScopedDeviceBuffer<std::uint8_t> sample_decision(rollout.sample_count);
    ScopedDeviceBuffer<float> rollout_entropy(2);
    std::vector<std::size_t> host_permutation(rollout.sample_count);
    std::iota(host_permutation.begin(), host_permutation.end(), std::size_t{0});
    std::mt19937_64 shuffle_random(shuffle_seed);
    float previous_kl_sum = 0.0F;
    for (int epoch = 0; epoch < update_config.epochs; ++epoch) {
        std::shuffle(host_permutation.begin(), host_permutation.end(), shuffle_random);
        check_cuda(cudaMemcpyAsync(device_permutation.pointer, host_permutation.data(),
                                   sizeof(std::size_t) * host_permutation.size(),
                                   cudaMemcpyHostToDevice, cuda_stream),
                   "upload PPO Fisher-Yates permutation");

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
                device_permutation.pointer, batch_offset, batch_size,
                impl_->config.observation_size, impl_->config.action_count,
                impl_->train_observations, impl_->train_masks, impl_->train_actions,
                impl_->train_old_log_probabilities, impl_->train_old_values,
                impl_->train_advantages, impl_->train_returns);
            const float* network_input =
                impl_->forward_network(impl_->train_observations, batch, cuda_stream);
            float* policy_output = impl_->materialize_policy_output(batch, cuda_stream);
            float* policy_gradients = impl_->parametric()
                ? impl_->parametric_policy_gradients : impl_->output_gradients;
            ppo_output_gradient_kernel<<<blocks_for(batch_size), kThreads, 0, cuda_stream>>>(
                policy_output, impl_->train_masks, impl_->train_actions,
                impl_->train_old_log_probabilities, impl_->train_old_values,
                impl_->train_advantages, impl_->train_returns,
                batch_size, impl_->config.action_count, update_config.clip_range,
                update_config.value_clip_range,
                update_config.value_coefficient, update_config.entropy_coefficient,
                policy_gradients, impl_->metric_sums,
                epoch == 0 ? sample_entropy.pointer : nullptr,
                epoch == 0 ? sample_decision.pointer : nullptr,
                device_permutation.pointer, batch_offset);
            if (impl_->parametric()) {
                const std::size_t query_elements = batch_size * output_size;
                parametric_policy_gradient_kernel<<<blocks_for(query_elements), kThreads, 0, cuda_stream>>>(
                    impl_->parametric_policy_gradients, impl_->action_features, batch_size,
                    impl_->config.action_count, impl_->config.action_feature_size,
                    impl_->output_gradients);
            }

            const std::size_t hidden_elements = batch_size * hidden;
            impl_->weight_gradient(impl_->hidden_2, impl_->output_gradients,
                                   impl_->gradient_weights_out, batch, hidden, output_size);
            bias_gradient_kernel<<<output_size, kThreads, 0, cuda_stream>>>(
                impl_->output_gradients, impl_->gradient_bias_out, batch_size, output_size);
            impl_->input_gradient(impl_->weights_out, impl_->output_gradients,
                                  impl_->hidden_2_upstream, batch, hidden, output_size);
            tanh_backward_kernel<<<blocks_for(hidden_elements), kThreads, 0, cuda_stream>>>(
                impl_->hidden_2_upstream, impl_->hidden_2, impl_->hidden_2_gradients, hidden_elements);

            impl_->weight_gradient(impl_->hidden_1, impl_->hidden_2_gradients,
                                   impl_->gradient_weights_2, batch, hidden, hidden);
            bias_gradient_kernel<<<hidden, kThreads, 0, cuda_stream>>>(
                impl_->hidden_2_gradients, impl_->gradient_bias_2, batch_size, hidden);
            impl_->input_gradient(impl_->weights_2, impl_->hidden_2_gradients,
                                  impl_->hidden_1_upstream, batch, hidden, hidden);
            tanh_backward_kernel<<<blocks_for(hidden_elements), kThreads, 0, cuda_stream>>>(
                impl_->hidden_1_upstream, impl_->hidden_1, impl_->hidden_1_gradients, hidden_elements);

            impl_->weight_gradient(network_input, impl_->hidden_1_gradients,
                                   impl_->gradient_weights_1, batch,
                                   impl_->config.observation_size, hidden);
            bias_gradient_kernel<<<hidden, kThreads, 0, cuda_stream>>>(
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
            const float mean_kl = (cumulative_kl - previous_kl_sum) /
                static_cast<float>(rollout.sample_count);
            previous_kl_sum = cumulative_kl;
            if (!std::isfinite(mean_kl)) {
                throw std::runtime_error("PPO KL became non-finite; optimizer update aborted");
            }
            if (mean_kl > update_config.target_kl) {
                early_stopped = true;
                break;
            }
        }
    }
    // Merge only after optimization so this rollout was collected and trained
    // with the same frozen statistics; the next rollout sees the new ones.
    if (impl_->config.observation_normalization) {
        impl_->merge_observation_statistics(rollout.observations, rollout.sample_count, cuda_stream);
    }
    // Every sample is visited once in the first epoch (before the policy changed), so this
    // is the rollout policy's entropy at its choice decisions.
    deterministic_decision_entropy_kernel<<<1, kThreads, 0, cuda_stream>>>(
        sample_entropy.pointer, sample_decision.pointer, rollout.sample_count, rollout_entropy.pointer);
    std::array<float, 2> rollout_entropy_host{};
    check_cuda(cudaMemcpyAsync(rollout_entropy_host.data(), rollout_entropy.pointer, sizeof(float) * 2,
                               cudaMemcpyDeviceToHost, cuda_stream), "download rollout decision entropy");
    check_cuda(cudaGetLastError(), "launch PPO update kernels");
    std::vector<float> metrics(kMetricCount);
    check_cuda(cudaMemcpyAsync(metrics.data(), impl_->metric_sums, sizeof(float) * metrics.size(),
                               cudaMemcpyDeviceToHost, cuda_stream), "download PPO metrics");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize PPO update");
    if (!std::all_of(metrics.begin(), metrics.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("PPO metrics became non-finite; checkpoint was not written");
    }
    const float sample_visits = static_cast<float>(
        rollout.sample_count * static_cast<std::size_t>(epochs_completed));
    return {metrics[0] / sample_visits, metrics[1] / sample_visits,
            metrics[2] / sample_visits, metrics[3] / sample_visits,
            metrics[4] / sample_visits, metrics[5] / static_cast<float>(minibatches),
            metrics[6] / sample_visits,
            metrics[6] > 0.0F ? metrics[7] / metrics[6] : 0.0F,
            rollout_entropy_host[1] > 0.0F ? rollout_entropy_host[0] / rollout_entropy_host[1] : 0.0F,
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

struct CheckpointContract {
    std::array<char, 64> catalog_sha256{};
    std::array<char, 24> roster_version{};
    std::array<char, 32> observation_contract{};
    std::array<char, 32> action_contract{};
    std::uint32_t action_feature_size = 0;
    std::uint32_t universal_action_count = 0;
};

// Version 4+: follows the contract block. The running mean and variance
// (observation_size floats each) are appended to the checksummed payload.
struct CheckpointNormalizer {
    std::uint32_t enabled = 0;
    std::uint32_t reserved = 0;
    std::uint64_t count = 0;
    float clip = 0.0F;
    float epsilon = 0.0F;
};

constexpr std::uint32_t kCheckpointVersion = 4;

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

template <std::size_t Size>
std::array<char, Size> checkpoint_text(std::string_view value) {
    if (value.size() > Size) throw std::invalid_argument("checkpoint contract text is too long");
    std::array<char, Size> result{};
    std::copy(value.begin(), value.end(), result.begin());
    return result;
}

template <std::size_t Size>
std::string checkpoint_text(const std::array<char, Size>& value) {
    const auto end = std::find(value.begin(), value.end(), '\0');
    return std::string(value.begin(), end);
}

}  // namespace

void GpuActorCritic::save_checkpoint(const std::filesystem::path& path, void* stream) const {
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize before checkpoint save");
    const auto& c = impl_->config;
    const std::size_t w1 = static_cast<std::size_t>(c.hidden_size) * c.observation_size;
    const std::size_t w2 = static_cast<std::size_t>(c.hidden_size) * c.hidden_size;
    const std::size_t wo = static_cast<std::size_t>(impl_->network_output_size()) * c.hidden_size;
    const std::size_t bo = static_cast<std::size_t>(impl_->network_output_size());
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
    append_tensor(impl_->observation_mean, c.observation_size);
    append_tensor(impl_->observation_variance, c.observation_size);
    if (!std::all_of(payload.begin(), payload.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("refusing to save checkpoint with non-finite tensors: " + path.string());
    }

    CheckpointHeader header{kCheckpointMagic, kCheckpointVersion,
                            static_cast<std::uint32_t>(c.observation_size),
                            static_cast<std::uint32_t>(c.action_count),
                            static_cast<std::uint32_t>(c.hidden_size),
                            impl_->optimizer_step, parameter_count()};
    const CheckpointContract contract{
        checkpoint_text<64>(c.catalog_sha256), checkpoint_text<24>(c.roster_version),
        checkpoint_text<32>(c.observation_contract), checkpoint_text<32>(c.action_contract),
        static_cast<std::uint32_t>(c.action_feature_size),
        static_cast<std::uint32_t>(c.universal_action_count)};
    const CheckpointNormalizer normalizer{
        c.observation_normalization ? 1U : 0U, 0U, impl_->observation_count,
        c.observation_clip, c.observation_epsilon};
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
        output.write(reinterpret_cast<const char*>(&contract), sizeof(contract));
        output.write(reinterpret_cast<const char*>(&normalizer), sizeof(normalizer));
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
    if (input && header.magic == kCheckpointMagic && header.version < 3) {
        throw std::runtime_error(
            "legacy fixed-action checkpoint is incompatible with the full-roster policy contract: " +
            path.string());
    }
    if (!input || header.magic != kCheckpointMagic ||
        (header.version != 3 && header.version != kCheckpointVersion) ||
        header.observation_size != static_cast<std::uint32_t>(c.observation_size) ||
        header.action_count != static_cast<std::uint32_t>(c.action_count) ||
        header.hidden_size != static_cast<std::uint32_t>(c.hidden_size) ||
        header.parameter_count != parameter_count()) {
        throw std::runtime_error("checkpoint architecture/version mismatch: " + path.string());
    }
    CheckpointContract contract{};
    input.read(reinterpret_cast<char*>(&contract), sizeof(contract));
    if (!input || checkpoint_text(contract.catalog_sha256) != c.catalog_sha256 ||
        checkpoint_text(contract.roster_version) != c.roster_version ||
        checkpoint_text(contract.observation_contract) != c.observation_contract ||
        checkpoint_text(contract.action_contract) != c.action_contract ||
        contract.action_feature_size != static_cast<std::uint32_t>(c.action_feature_size) ||
        contract.universal_action_count != static_cast<std::uint32_t>(c.universal_action_count)) {
        throw std::runtime_error("checkpoint catalog/roster/action contract mismatch: " + path.string());
    }
    // Version 3 predates the normalizer: those weights were trained on raw
    // observations, so they load with normalization disabled.
    const bool has_normalizer = header.version >= 4;
    CheckpointNormalizer normalizer{};
    if (has_normalizer) {
        input.read(reinterpret_cast<char*>(&normalizer), sizeof(normalizer));
        if (!input || normalizer.enabled > 1 || normalizer.reserved != 0 ||
            !std::isfinite(normalizer.clip) || normalizer.clip <= 0.0F ||
            !std::isfinite(normalizer.epsilon) || normalizer.epsilon <= 0.0F) {
            throw std::runtime_error("checkpoint observation normalizer block is invalid: " + path.string());
        }
    }
    const std::size_t w1 = static_cast<std::size_t>(c.hidden_size) * c.observation_size;
    const std::size_t w2 = static_cast<std::size_t>(c.hidden_size) * c.hidden_size;
    const std::size_t wo = static_cast<std::size_t>(impl_->network_output_size()) * c.hidden_size;
    const std::size_t bo = static_cast<std::size_t>(impl_->network_output_size());
    const std::size_t model_parameter_count = w1 + c.hidden_size + w2 + c.hidden_size + wo + bo;
    const std::size_t normalizer_count =
        has_normalizer ? 2 * static_cast<std::size_t>(c.observation_size) : 0;
    const std::size_t payload_count = model_parameter_count * 3 + normalizer_count;
    CheckpointIntegrity integrity{};
    input.read(reinterpret_cast<char*>(&integrity), sizeof(integrity));
    if (!input || integrity.payload_bytes != payload_count * sizeof(float)) {
        throw std::runtime_error("checkpoint payload size mismatch: " + path.string());
    }
    std::vector<float> payload(payload_count);
    input.read(reinterpret_cast<char*>(payload.data()),
               static_cast<std::streamsize>(payload.size() * sizeof(float)));
    if (!input) throw std::runtime_error("checkpoint tensor is truncated: " + path.string());
    if (checkpoint_checksum(payload) != integrity.payload_checksum) {
        throw std::runtime_error("checkpoint checksum mismatch: " + path.string());
    }
    if (!std::all_of(payload.begin(), payload.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::runtime_error("checkpoint contains non-finite tensors: " + path.string());
    }
    if (has_normalizer) {
        const float* variance = payload.data() + model_parameter_count * 3 + c.observation_size;
        if (!std::all_of(variance, variance + c.observation_size,
                         [](float value) { return value >= 0.0F; })) {
            throw std::runtime_error("checkpoint observation variance is negative: " + path.string());
        }
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
    if (has_normalizer) {
        const float* mean = payload.data() + model_parameter_count * 3;
        impl_->config.observation_normalization = normalizer.enabled != 0;
        impl_->config.observation_clip = normalizer.clip;
        impl_->config.observation_epsilon = normalizer.epsilon;
        impl_->upload_observation_statistics(mean, mean + c.observation_size, cuda_stream);
        impl_->observation_count = normalizer.count;
    } else {
        impl_->config.observation_normalization = false;
        impl_->reset_observation_statistics(cuda_stream);
    }
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize checkpoint load");
}

std::uint64_t GpuActorCritic::observation_count() const noexcept {
    return impl_->observation_count;
}

std::vector<float> GpuActorCritic::download_observation_statistics(void* stream) const {
    const auto size = static_cast<std::size_t>(impl_->config.observation_size);
    auto result = download(impl_->observation_mean, size, as_stream(stream), "download observation mean");
    const auto variance = download(
        impl_->observation_variance, size, as_stream(stream), "download observation variance");
    result.insert(result.end(), variance.begin(), variance.end());
    return result;
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
    const std::uint8_t* truncated,
    const float* next_values,
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
        const bool time_limit = truncated[index] != 0;
        const float bootstrap_mask = terminated[index] && !time_limit ? 0.0F : 1.0F;
        const float trace_mask = terminated[index] || time_limit ? 0.0F : 1.0F;
        const float delta = rewards[index] + gamma * next_values[index] * bootstrap_mask - values[index];
        next_advantage = delta + gamma * gae_lambda * trace_mask * next_advantage;
        advantages[index] = next_advantage;
        returns[index] = next_advantage + values[index];
    }
}

__global__ void advantage_stats_kernel(
    const float* advantages,
    std::size_t count,
    double* block_sums,
    double* block_square_sums) {
    __shared__ double shared_sum[kThreads];
    __shared__ double shared_square_sum[kThreads];
    double local_sum = 0.0;
    double local_square_sum = 0.0;
    for (std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        const double value = static_cast<double>(advantages[index]);
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
    const double* block_sums,
    const double* block_square_sums,
    int block_count,
    double* statistics) {
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    double sum = 0.0;
    double square_sum = 0.0;
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
    const double* statistics) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const double mean = statistics[0] / static_cast<double>(count);
    const double second_moment = statistics[1] / static_cast<double>(count);
    const double inverse_standard_deviation =
        1.0 / sqrt(fmax(second_moment - mean * mean, 1e-12));
    advantages[index] = static_cast<float>((static_cast<double>(advantages[index]) - mean) *
                                           inverse_standard_deviation);
}

__global__ void record_outcome_kernel(
    float* destination_rewards,
    std::uint8_t* destination_terminated,
    std::uint8_t* destination_truncated,
    float* destination_next_values,
    const float* rewards,
    const std::uint8_t* terminated,
    const std::uint8_t* truncated,
    const float* next_values,
    float reward_scale,
    std::size_t count) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    destination_rewards[lane] = rewards[lane] * reward_scale;
    destination_terminated[lane] = terminated[lane];
    destination_truncated[lane] = truncated[lane];
    destination_next_values[lane] = next_values[lane];
}

// Carries each environment's discounted return across rollouts (reset at
// episode ends) and records it per sample for the variance estimate.
__global__ void discounted_return_kernel(
    const float* rewards,
    const std::uint8_t* terminated,
    const std::uint8_t* truncated,
    std::size_t environments,
    std::size_t horizon,
    float gamma,
    float* discounted_returns,
    float* return_samples) {
    const std::size_t environment = blockIdx.x * blockDim.x + threadIdx.x;
    if (environment >= environments) return;
    float running = discounted_returns[environment];
    for (std::size_t step = 0; step < horizon; ++step) {
        const std::size_t index = step * environments + environment;
        running = running * gamma + rewards[index];
        return_samples[index] = running;
        if (terminated[index] || truncated[index]) running = 0.0F;
    }
    discounted_returns[environment] = running;
}

__global__ void scale_rewards_kernel(float* rewards, std::size_t count, float scale, float clip) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    rewards[index] = fminf(clip, fmaxf(-clip, rewards[index] * scale));
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
    std::uint8_t* truncated = nullptr;
    float* next_values = nullptr;
    float* advantages = nullptr;
    float* returns = nullptr;
    double* advantage_statistics = nullptr;
    double* advantage_block_sums = nullptr;
    double* advantage_block_square_sums = nullptr;
    int advantage_reduction_blocks = 0;
    float* discounted_returns = nullptr;
    float* return_samples = nullptr;
    std::uint64_t return_count = 0;
    double return_mean = 0.0;
    double return_variance = 1.0;

    Impl(std::size_t environment_count, std::size_t horizon, ActorCriticConfig network_config)
        : environments(environment_count), rollout_horizon(horizon),
          samples(checked_rollout_samples(environment_count, horizon)),
          config(network_config),
          advantage_reduction_blocks(std::min(blocks_for(samples), 1024)) {
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
        check_cuda(cudaMalloc(&truncated, sizeof(std::uint8_t) * samples), "allocate rollout truncation flags");
        check_cuda(cudaMalloc(&next_values, sizeof(float) * samples), "allocate rollout next values");
        check_cuda(cudaMalloc(&advantages, sizeof(float) * samples), "allocate rollout advantages");
        check_cuda(cudaMalloc(&returns, sizeof(float) * samples), "allocate rollout returns");
        check_cuda(cudaMalloc(&advantage_statistics, sizeof(double) * 2), "allocate advantage statistics");
        check_cuda(cudaMalloc(&advantage_block_sums, sizeof(double) * advantage_reduction_blocks),
                   "allocate advantage block sums");
        check_cuda(cudaMalloc(&advantage_block_square_sums, sizeof(double) * advantage_reduction_blocks),
                   "allocate advantage block square sums");
        check_cuda(cudaMalloc(&discounted_returns, sizeof(float) * environments),
                   "allocate running discounted returns");
        check_cuda(cudaMemset(discounted_returns, 0, sizeof(float) * environments),
                   "zero running discounted returns");
        check_cuda(cudaMalloc(&return_samples, sizeof(float) * samples),
                   "allocate discounted return samples");
        } catch (...) {
            release();
            throw;
        }
    }

    void release() noexcept {
        cudaFree(return_samples);
        cudaFree(discounted_returns);
        cudaFree(advantage_block_square_sums);
        cudaFree(advantage_block_sums);
        cudaFree(advantage_statistics);
        cudaFree(returns);
        cudaFree(advantages);
        cudaFree(next_values);
        cudaFree(truncated);
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
            impl_->terminated, impl_->truncated, impl_->next_values,
            impl_->advantages, impl_->returns,
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
    const std::uint8_t* truncated,
    const float* next_values,
    float reward_scale,
    void* stream) {
    if (step >= impl_->rollout_horizon) throw std::out_of_range("rollout step exceeds horizon");
    if (!rewards || !terminated || !truncated || !next_values) {
        throw std::invalid_argument("outcome device pointers cannot be null");
    }
    if (!std::isfinite(reward_scale) || reward_scale <= 0.0F) {
        throw std::invalid_argument("reward scale must be finite and positive");
    }
    const auto cuda_stream = as_stream(stream);
    const std::size_t sample_offset = step * impl_->environments;
    record_outcome_kernel<<<blocks_for(impl_->environments), kThreads, 0, cuda_stream>>>(
        impl_->rewards + sample_offset, impl_->terminated + sample_offset,
        impl_->truncated + sample_offset, impl_->next_values + sample_offset,
        rewards, terminated, truncated, next_values, reward_scale, impl_->environments);
    check_cuda(cudaGetLastError(), "launch rollout outcome recording kernel");
}

void GpuRolloutBuffer::compute_gae(
    float gamma,
    float gae_lambda,
    bool normalize,
    void* stream) {
    if (!std::isfinite(gamma) || !std::isfinite(gae_lambda) ||
        !(gamma >= 0.0F && gamma <= 1.0F && gae_lambda >= 0.0F && gae_lambda <= 1.0F)) {
        throw std::invalid_argument("gamma and gae_lambda must be in [0, 1]");
    }
    const auto cuda_stream = as_stream(stream);
    gae_kernel<<<blocks_for(impl_->environments), kThreads, 0, cuda_stream>>>(
        impl_->rewards, impl_->old_values, impl_->terminated, impl_->truncated, impl_->next_values,
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

float GpuRolloutBuffer::normalize_rewards(float gamma, float clip, void* stream) {
    if (!std::isfinite(gamma) || gamma < 0.0F || gamma > 1.0F) {
        throw std::invalid_argument("return normalization gamma must be in [0, 1]");
    }
    if (!std::isfinite(clip) || clip <= 0.0F) {
        throw std::invalid_argument("return normalization clip must be finite and positive");
    }
    const auto cuda_stream = as_stream(stream);
    discounted_return_kernel<<<blocks_for(impl_->environments), kThreads, 0, cuda_stream>>>(
        impl_->rewards, impl_->terminated, impl_->truncated,
        impl_->environments, impl_->rollout_horizon, gamma,
        impl_->discounted_returns, impl_->return_samples);
    advantage_stats_kernel<<<impl_->advantage_reduction_blocks, kThreads, 0, cuda_stream>>>(
        impl_->return_samples, impl_->samples,
        impl_->advantage_block_sums, impl_->advantage_block_square_sums);
    finalize_advantage_stats_kernel<<<1, 1, 0, cuda_stream>>>(
        impl_->advantage_block_sums, impl_->advantage_block_square_sums,
        impl_->advantage_reduction_blocks, impl_->advantage_statistics);
    check_cuda(cudaGetLastError(), "launch return normalization statistics");
    std::array<double, 2> sums{};
    check_cuda(cudaMemcpyAsync(sums.data(), impl_->advantage_statistics, sizeof(double) * 2,
                               cudaMemcpyDeviceToHost, cuda_stream), "download return statistics");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize return statistics");
    const double batch_count = static_cast<double>(impl_->samples);
    const double batch_mean = sums[0] / batch_count;
    const double batch_m2 = std::max(0.0, sums[1] - sums[0] * batch_mean);
    if (!std::isfinite(batch_mean) || !std::isfinite(batch_m2)) {
        throw std::runtime_error("discounted returns became non-finite during return normalization");
    }
    const double running_count = static_cast<double>(impl_->return_count);
    const double total = running_count + batch_count;
    const double delta = batch_mean - impl_->return_mean;
    impl_->return_variance = (impl_->return_variance * running_count + batch_m2 +
                              delta * delta * running_count * batch_count / total) / total;
    impl_->return_mean += delta * batch_count / total;
    impl_->return_count += impl_->samples;
    const float standard_deviation = static_cast<float>(std::sqrt(impl_->return_variance + 1e-8));
    scale_rewards_kernel<<<blocks_for(impl_->samples), kThreads, 0, cuda_stream>>>(
        impl_->rewards, impl_->samples, 1.0F / standard_deviation, clip);
    check_cuda(cudaGetLastError(), "launch reward normalization");
    return standard_deviation;
}

ReturnNormalizerState GpuRolloutBuffer::return_normalizer_state(void* stream) const {
    return {impl_->return_count, impl_->return_mean, impl_->return_variance,
            download(impl_->discounted_returns, impl_->environments, as_stream(stream),
                     "download running discounted returns")};
}

void GpuRolloutBuffer::restore_return_normalizer_state(
    const ReturnNormalizerState& state,
    void* stream) {
    if (state.discounted_returns.size() != impl_->environments ||
        !std::isfinite(state.mean) || !std::isfinite(state.variance) || state.variance < 0.0 ||
        !std::all_of(state.discounted_returns.begin(), state.discounted_returns.end(),
                     [](float value) { return std::isfinite(value); })) {
        throw std::invalid_argument("return normalizer state is invalid for this rollout buffer");
    }
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(impl_->discounted_returns, state.discounted_returns.data(),
                               sizeof(float) * impl_->environments, cudaMemcpyHostToDevice, cuda_stream),
               "upload running discounted returns");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize return normalizer restore");
    impl_->return_count = state.count;
    impl_->return_mean = state.mean;
    impl_->return_variance = state.variance;
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

std::vector<std::uint8_t> GpuRolloutBuffer::download_truncated(void* stream) const {
    return download(impl_->truncated, impl_->samples, as_stream(stream), "download rollout truncation flags");
}

std::vector<float> GpuRolloutBuffer::download_next_values(void* stream) const {
    return download(impl_->next_values, impl_->samples, as_stream(stream), "download rollout next values");
}

}  // namespace t8::v2
