#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/ppo.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        if (failures < 30) std::cerr << "FAIL: " << message << '\n';
    }
}

void cuda_check(cudaError_t result, std::string_view operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

void test_device_rollout_and_gae() {
    constexpr std::size_t environments = 2048;
    constexpr std::size_t horizon = 64;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuActorCritic learner(environments, {}, 111);
    t8::v2::GpuActorCritic opponent(environments, {}, 222);
    t8::v2::GpuRolloutBuffer rollout(environments, horizon);

    check(rollout.environment_count() == environments, "rollout environment count");
    check(rollout.horizon() == horizon, "rollout horizon");
    check(rollout.sample_count() == environments * horizon, "rollout sample count");

    for (std::size_t step = 0; step < horizon; ++step) {
        const auto before = simulator.device_view();
        const auto p1 = learner.forward(before.observations_p1, before.action_masks_p1,
                                        environments, 1000, step, false);
        const auto p2 = opponent.forward(before.observations_p2, before.action_masks_p2,
                                         environments, 2000, step, false);
        rollout.record_policy_device(step, before.observations_p1, before.action_masks_p1,
                                     p1.actions, p1.log_probabilities, p1.values);
        simulator.step_device_i64(p1.actions, p2.actions);
        const auto after = simulator.device_view();
        const auto next = learner.forward(
            after.observations_p1, after.action_masks_p1,
            environments, 3000, step, true);
        rollout.record_outcome_device(
            step, after.rewards_p1, after.terminated, after.truncated, next.values);
        simulator.reset_done();
    }

    const auto final_state = simulator.device_view();
    [[maybe_unused]] const auto bootstrap = learner.forward(  // computes the bootstrap values
        final_state.observations_p1, final_state.action_masks_p1,
        environments, 1000, horizon, true);
    const auto values_before_update = learner.download_values(environments);
    rollout.compute_gae(0.99F, 0.95F, false);
    const auto raw_advantages = rollout.download_advantages();
    const auto raw_returns = rollout.download_returns();
    const auto rollout_rewards = rollout.download_rewards();
    const auto rollout_values = rollout.download_values();
    const auto rollout_terminated = rollout.download_terminated();
    const auto rollout_truncated = rollout.download_truncated();
    const auto rollout_next_values = rollout.download_next_values();
    for (std::size_t environment = 0; environment < environments; ++environment) {
        float next_advantage = 0.0F;
        for (std::size_t reverse = 0; reverse < horizon; ++reverse) {
            const std::size_t step = horizon - 1 - reverse;
            const std::size_t index = step * environments + environment;
            const bool time_limit = rollout_truncated[index] != 0;
            const float bootstrap_mask = rollout_terminated[index] && !time_limit ? 0.0F : 1.0F;
            const float trace_mask = rollout_terminated[index] || time_limit ? 0.0F : 1.0F;
            const float delta = rollout_rewards[index] +
                0.99F * rollout_next_values[index] * bootstrap_mask -
                rollout_values[index];
            next_advantage = delta + 0.99F * 0.95F * trace_mask * next_advantage;
            check(std::abs(raw_advantages[index] - next_advantage) < 2e-5F,
                  "CUDA GAE matches fixed-order CPU reference");
            check(std::abs(raw_returns[index] - (next_advantage + rollout_values[index])) < 2e-5F,
                  "CUDA return matches fixed-order CPU reference");
        }
    }

    rollout.compute_gae(0.99F, 0.95F, true);
    const auto advantages = rollout.download_advantages();
    const auto returns = rollout.download_returns();

    double sum = 0.0;
    double square_sum = 0.0;
    for (std::size_t sample = 0; sample < advantages.size(); ++sample) {
        check(std::isfinite(advantages[sample]), "advantage is finite");
        check(std::isfinite(returns[sample]), "return is finite");
        sum += advantages[sample];
        square_sum += static_cast<double>(advantages[sample]) * advantages[sample];
    }
    const double mean = sum / advantages.size();
    const double variance = square_sum / advantages.size() - mean * mean;
    check(std::abs(mean) < 2e-4, "normalized advantage mean is approximately zero");
    check(std::abs(variance - 1.0) < 2e-3, "normalized advantage variance is approximately one");

    const auto view = rollout.device_view();
    check(view.observations != nullptr && view.action_masks != nullptr,
          "rollout training inputs remain device-resident");
    check(view.advantages != nullptr && view.returns != nullptr,
          "GAE outputs remain device-resident");

    bool rejected_non_finite_gae = false;
    try {
        rollout.compute_gae(
            std::numeric_limits<float>::quiet_NaN(), 0.95F, true);
    } catch (const std::invalid_argument&) {
        rejected_non_finite_gae = true;
    }
    check(rejected_non_finite_gae, "GAE rejects non-finite coefficients");

    t8::v2::PpoUpdateConfig update_config{};
    update_config.epochs = 2;
    update_config.minibatch_size = environments;
    update_config.target_kl = 0.0F;
    const auto metrics = learner.update_ppo(view, update_config, 333);
    check(std::isfinite(metrics.policy_loss), "PPO policy loss is finite");
    check(std::isfinite(metrics.value_loss) && metrics.value_loss >= 0.0F,
          "PPO value loss is finite and non-negative");
    check(std::isfinite(metrics.entropy) && metrics.entropy > 0.0F,
          "PPO entropy is finite and positive");
    check(std::isfinite(metrics.approximate_kl) && metrics.approximate_kl >= -1e-3F,
          "PPO approximate KL is finite");
    check(metrics.clip_fraction >= 0.0F && metrics.clip_fraction <= 1.0F,
          "PPO clip fraction is a probability");
    check(std::isfinite(metrics.gradient_norm) && metrics.gradient_norm > 0.0F,
          "PPO gradient norm is finite and positive");
    check(metrics.minibatches == 128, "PPO visits every sample for every epoch");
    check(metrics.epochs_completed == 2 && !metrics.early_stopped,
          "disabled target KL completes every configured epoch");

    static_cast<void>(learner.forward(
        final_state.observations_p1, final_state.action_masks_p1,
        environments, 1000, horizon, true));
    const auto values_after_update = learner.download_values(environments);
    bool any_value_changed = false;
    for (std::size_t lane = 0; lane < environments; ++lane) {
        if (std::abs(values_after_update[lane] - values_before_update[lane]) > 1e-6F) {
            any_value_changed = true;
            break;
        }
    }
    check(any_value_changed, "Adam update changes actor-critic parameters");

    update_config.learning_rate = std::numeric_limits<float>::quiet_NaN();
    bool rejected_non_finite_ppo = false;
    try {
        static_cast<void>(learner.update_ppo(view, update_config, 334));
    } catch (const std::invalid_argument&) {
        rejected_non_finite_ppo = true;
    }
    check(rejected_non_finite_ppo, "PPO update rejects non-finite optimizer coefficients");
    update_config.learning_rate = 3e-4F;

    update_config.epochs = 4;
    update_config.target_kl = 1e-12F;
    const auto guarded_metrics = learner.update_ppo(view, update_config, 444);
    check(guarded_metrics.early_stopped && guarded_metrics.epochs_completed == 1,
          "target KL stops PPO after the first excessive epoch");

    bool rejected_zero_environments = false;
    try {
        t8::v2::GpuRolloutBuffer invalid(0, 1);
    } catch (const std::invalid_argument& error) {
        rejected_zero_environments =
            std::string_view(error.what()) == "rollout dimensions must be greater than zero";
    }
    check(rejected_zero_environments, "zero-sized rollout reports the dimension error");
}

void test_parametric_policy_ppo_update() {
    constexpr std::size_t environments = 32;
    constexpr std::size_t horizon = 4;
    constexpr std::size_t samples = environments * horizon;
    t8::v2::ActorCriticConfig config{};
    config.observation_size = 5;
    config.action_count = 4;
    config.hidden_size = 16;
    config.action_feature_size = 3;
    config.universal_action_count = 2;
    config.action_contract = "parametric-update-test-v3";
    t8::v2::GpuActorCritic policy(samples, config, 8080);
    const std::array<float, 12> action_features = {
        1.0F, 0.0F, 0.0F,
        0.0F, 1.0F, 0.0F,
        0.0F, 0.0F, 1.0F,
        -1.0F, -1.0F, -1.0F,
    };
    policy.set_action_features(action_features);
    t8::v2::GpuRolloutBuffer rollout(environments, horizon, config);
    std::vector<float> observations(environments * 5, 0.2F);
    std::vector<std::uint8_t> masks(environments * 4, 1);
    std::vector<float> rewards(environments, 0.1F);
    std::vector<std::uint8_t> flags(environments, 0);
    float* device_observations = nullptr;
    std::uint8_t* device_masks = nullptr;
    float* device_rewards = nullptr;
    std::uint8_t* device_flags = nullptr;
    cuda_check(cudaMalloc(&device_observations, sizeof(float) * observations.size()),
               "allocate parametric rollout observations");
    cuda_check(cudaMalloc(&device_masks, sizeof(std::uint8_t) * masks.size()),
               "allocate parametric rollout masks");
    cuda_check(cudaMalloc(&device_rewards, sizeof(float) * rewards.size()),
               "allocate parametric rollout rewards");
    cuda_check(cudaMalloc(&device_flags, sizeof(std::uint8_t) * flags.size()),
               "allocate parametric rollout flags");
    cuda_check(cudaMemcpy(device_observations, observations.data(), sizeof(float) * observations.size(),
                          cudaMemcpyHostToDevice), "upload parametric rollout observations");
    cuda_check(cudaMemcpy(device_masks, masks.data(), sizeof(std::uint8_t) * masks.size(),
                          cudaMemcpyHostToDevice), "upload parametric rollout masks");
    cuda_check(cudaMemcpy(device_rewards, rewards.data(), sizeof(float) * rewards.size(),
                          cudaMemcpyHostToDevice), "upload parametric rollout rewards");
    cuda_check(cudaMemcpy(device_flags, flags.data(), sizeof(std::uint8_t) * flags.size(),
                          cudaMemcpyHostToDevice), "upload parametric rollout flags");
    for (std::size_t step = 0; step < horizon; ++step) {
        const auto output = policy.forward(
            device_observations, device_masks, environments, 77, step, false);
        rollout.record_policy_device(step, device_observations, device_masks, output.actions,
                                     output.log_probabilities, output.values);
        rollout.record_outcome_device(step, device_rewards, device_flags, device_flags, output.values);
    }
    rollout.compute_gae(0.99F, 0.95F, true);
    t8::v2::PpoUpdateConfig update{};
    update.epochs = 2;
    update.minibatch_size = environments;
    update.target_kl = 0.0F;
    const auto metrics = policy.update_ppo(rollout.device_view(), update, 9090);
    check(std::isfinite(metrics.policy_loss) && std::isfinite(metrics.value_loss),
          "parametric PPO losses are finite");
    check(std::isfinite(metrics.gradient_norm) && metrics.gradient_norm > 0.0F,
          "parametric scorer gradients reach the shared encoder");
    check(metrics.minibatches == 8 && metrics.epochs_completed == 2,
          "parametric PPO visits each rollout sample");
    cudaFree(device_flags);
    cudaFree(device_rewards);
    cudaFree(device_masks);
    cudaFree(device_observations);
}

void test_time_limit_bootstrap_and_reward_scaling() {
    constexpr std::size_t environments = 64;
    t8::v2::Config config{};
    config.max_frames = 1;
    config.timeout_ties_are_draws = true;
    t8::v2::GpuSimulatorBatch simulator(environments, config);
    t8::v2::GpuActorCritic learner(environments, {}, 555);
    t8::v2::GpuActorCritic opponent(environments, {}, 666);
    t8::v2::GpuRolloutBuffer rollout(environments, 1);

    const auto before = simulator.device_view();
    const auto p1 = learner.forward(
        before.observations_p1, before.action_masks_p1, environments, 7000, 0, true);
    rollout.record_policy_device(
        0, before.observations_p1, before.action_masks_p1,
        p1.actions, p1.log_probabilities, p1.values);
    const auto p2 = opponent.forward(
        before.observations_p2, before.action_masks_p2, environments, 8000, 0, true);
    simulator.step_device_i64(p1.actions, p2.actions);
    const auto after = simulator.device_view();
    const auto next = learner.forward(
        after.observations_p1, after.action_masks_p1, environments, 7000, 1, true);
    rollout.record_outcome_device(
        0, after.rewards_p1, after.terminated, after.truncated, next.values, 0.01F);
    rollout.compute_gae(0.997F, 0.95F, false);

    const auto rewards = rollout.download_rewards();
    const auto values = rollout.download_values();
    const auto next_values = rollout.download_next_values();
    const auto advantages = rollout.download_advantages();
    const auto terminated = rollout.download_terminated();
    const auto truncated = rollout.download_truncated();
    for (std::size_t lane = 0; lane < environments; ++lane) {
        check(terminated[lane] != 0 && truncated[lane] != 0,
              "max-frame boundary is terminal for reset and truncated for GAE");
        const float expected = rewards[lane] + 0.997F * next_values[lane] - values[lane];
        check(std::abs(advantages[lane] - expected) < 2e-5F,
              "time-limit GAE bootstraps the post-step critic value");
        check(std::abs(rewards[lane]) < 5.0F,
              "training reward scaling keeps shaped timeout returns near unit scale");
    }
}

void test_record_outcome_rejects_invalid_reward_scale() {
    constexpr std::size_t environments = 4;
    t8::v2::GpuRolloutBuffer rollout(environments, 1);
    float* rewards = nullptr;
    float* next_values = nullptr;
    std::uint8_t* terminated = nullptr;
    std::uint8_t* truncated = nullptr;
    cuda_check(cudaMalloc(&rewards, sizeof(float) * environments), "allocate reward-scale rewards");
    cuda_check(cudaMalloc(&next_values, sizeof(float) * environments), "allocate reward-scale next values");
    cuda_check(cudaMalloc(&terminated, sizeof(std::uint8_t) * environments),
               "allocate reward-scale terminated flags");
    cuda_check(cudaMalloc(&truncated, sizeof(std::uint8_t) * environments),
               "allocate reward-scale truncated flags");
    cuda_check(cudaMemset(rewards, 0, sizeof(float) * environments), "clear reward-scale rewards");
    cuda_check(cudaMemset(next_values, 0, sizeof(float) * environments), "clear reward-scale next values");
    cuda_check(cudaMemset(terminated, 0, sizeof(std::uint8_t) * environments),
               "clear reward-scale terminated flags");
    cuda_check(cudaMemset(truncated, 0, sizeof(std::uint8_t) * environments),
               "clear reward-scale truncated flags");

    bool rejected_zero = false;
    try {
        rollout.record_outcome_device(0, rewards, terminated, truncated, next_values, 0.0F);
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }
    check(rejected_zero, "record_outcome_device rejects a zero reward scale");

    bool rejected_negative = false;
    try {
        rollout.record_outcome_device(0, rewards, terminated, truncated, next_values, -1.0F);
    } catch (const std::invalid_argument&) {
        rejected_negative = true;
    }
    check(rejected_negative, "record_outcome_device rejects a negative reward scale");

    bool rejected_nan = false;
    try {
        rollout.record_outcome_device(
            0, rewards, terminated, truncated, next_values, std::numeric_limits<float>::quiet_NaN());
    } catch (const std::invalid_argument&) {
        rejected_nan = true;
    }
    check(rejected_nan, "record_outcome_device rejects a non-finite reward scale");

    cudaFree(truncated);
    cudaFree(terminated);
    cudaFree(next_values);
    cudaFree(rewards);
}

void test_update_ppo_rejects_non_finite_rollout_and_illegal_action() {
    constexpr std::size_t environments = 8;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuActorCritic learner(environments, {}, 2600);
    t8::v2::GpuRolloutBuffer rollout(environments, 1);

    const auto before = simulator.device_view();
    const auto p1 = learner.forward(
        before.observations_p1, before.action_masks_p1, environments, 4100, 0, false);
    rollout.record_policy_device(
        0, before.observations_p1, before.action_masks_p1,
        p1.actions, p1.log_probabilities, p1.values);
    simulator.step_device_i64(p1.actions, p1.actions);
    const auto after = simulator.device_view();
    const auto next = learner.forward(
        after.observations_p1, after.action_masks_p1, environments, 4100, 1, true);
    rollout.record_outcome_device(0, after.rewards_p1, after.terminated, after.truncated, next.values);
    rollout.compute_gae(0.99F, 0.95F, false);

    t8::v2::PpoUpdateConfig update_config{};
    update_config.epochs = 1;
    update_config.minibatch_size = environments;
    static_cast<void>(learner.update_ppo(rollout.device_view(), update_config, 1));

    float* poisoned_rewards = nullptr;
    cuda_check(cudaMalloc(&poisoned_rewards, sizeof(float) * environments), "allocate poisoned rewards");
    std::vector<float> host_rewards(environments, 0.0F);
    host_rewards[0] = std::numeric_limits<float>::quiet_NaN();
    cuda_check(cudaMemcpy(poisoned_rewards, host_rewards.data(), sizeof(float) * environments,
                          cudaMemcpyHostToDevice), "upload poisoned rewards");
    rollout.record_outcome_device(
        0, poisoned_rewards, after.terminated, after.truncated, next.values);
    rollout.compute_gae(0.99F, 0.95F, false);
    bool rejected_nonfinite = false;
    try {
        static_cast<void>(learner.update_ppo(rollout.device_view(), update_config, 2));
    } catch (const std::runtime_error& error) {
        rejected_nonfinite =
            std::string_view(error.what()) == "PPO rollout contains a non-finite value or invalid action";
    }
    check(rejected_nonfinite, "update_ppo rejects a rollout poisoned with a non-finite reward");
    cudaFree(poisoned_rewards);

    // Restore a healthy reward/GAE state, then corrupt only the recorded
    // action so the finite-value checks above cannot mask this path.
    rollout.record_outcome_device(0, after.rewards_p1, after.terminated, after.truncated, next.values);
    rollout.compute_gae(0.99F, 0.95F, false);

    std::int64_t* illegal_actions = nullptr;
    cuda_check(cudaMalloc(&illegal_actions, sizeof(std::int64_t) * environments),
               "allocate illegal actions buffer");
    const std::vector<std::int64_t> host_illegal(
        environments, static_cast<std::int64_t>(t8::v2::kActionCount));
    cuda_check(cudaMemcpy(illegal_actions, host_illegal.data(), sizeof(std::int64_t) * environments,
                          cudaMemcpyHostToDevice), "upload illegal actions");
    rollout.record_policy_device(
        0, before.observations_p1, before.action_masks_p1,
        illegal_actions, p1.log_probabilities, p1.values);
    bool rejected_illegal_action = false;
    try {
        static_cast<void>(learner.update_ppo(rollout.device_view(), update_config, 3));
    } catch (const std::runtime_error& error) {
        rejected_illegal_action =
            std::string_view(error.what()) == "PPO rollout contains a non-finite value or invalid action";
    }
    check(rejected_illegal_action, "update_ppo rejects a rollout with an out-of-range recorded action");
    cudaFree(illegal_actions);
}

}  // namespace

void test_return_normalization() {
    constexpr std::size_t environments = 32;
    constexpr std::size_t horizon = 6;
    constexpr float gamma = 0.5F;
    t8::v2::GpuRolloutBuffer rollout(environments, horizon);
    // Lane-dependent constant rewards; lane 0 terminates at step 2 so its
    // discounted return restarts, and the carried return survives the rollout.
    std::vector<float> host_rewards(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) host_rewards[lane] = 0.25F * (lane % 5) - 0.5F;
    float* device_rewards = nullptr;
    std::uint8_t* device_terminated = nullptr;
    std::uint8_t* device_truncated = nullptr;
    float* device_values = nullptr;
    cuda_check(cudaMalloc(&device_rewards, sizeof(float) * environments), "allocate rewards");
    cuda_check(cudaMalloc(&device_terminated, environments), "allocate terminated");
    cuda_check(cudaMalloc(&device_truncated, environments), "allocate truncated");
    cuda_check(cudaMalloc(&device_values, sizeof(float) * environments), "allocate values");
    cuda_check(cudaMemcpy(device_rewards, host_rewards.data(), sizeof(float) * environments,
                          cudaMemcpyHostToDevice), "upload rewards");
    cuda_check(cudaMemset(device_truncated, 0, environments), "clear truncated");
    cuda_check(cudaMemset(device_values, 0, sizeof(float) * environments), "clear values");
    std::vector<double> expected_returns;
    std::vector<double> carried(environments, 0.0);
    for (std::size_t step = 0; step < horizon; ++step) {
        std::vector<std::uint8_t> terminated(environments, 0);
        if (step == 2) terminated[0] = 1;
        cuda_check(cudaMemcpy(device_terminated, terminated.data(), environments, cudaMemcpyHostToDevice),
                   "upload terminated");
        rollout.record_outcome_device(step, device_rewards, device_terminated, device_truncated,
                                      device_values, 2.0F);
        for (std::size_t lane = 0; lane < environments; ++lane) {
            carried[lane] = carried[lane] * gamma + host_rewards[lane] * 2.0F;
            expected_returns.push_back(carried[lane]);
            if (terminated[lane]) carried[lane] = 0.0;
        }
    }
    double sum = 0.0;
    for (double value : expected_returns) sum += value;
    const double mean = sum / expected_returns.size();
    double squares = 0.0;
    for (double value : expected_returns) squares += (value - mean) * (value - mean);
    const double expected_std = std::sqrt(squares / expected_returns.size() + 1e-8);

    const float standard_deviation = rollout.normalize_rewards(gamma, 10.0F);
    check(std::abs(standard_deviation - expected_std) < 1e-5 * expected_std,
          "return normalizer uses the discounted-return standard deviation");
    const auto normalized = rollout.download_rewards();
    bool scaled = true;
    for (std::size_t sample = 0; sample < normalized.size(); ++sample) {
        const float expected = host_rewards[sample % environments] * 2.0F / standard_deviation;
        scaled = scaled && std::abs(normalized[sample] - expected) < 1e-5F;
    }
    check(scaled, "rewards are divided by the running return standard deviation");

    const auto state = rollout.return_normalizer_state();
    check(state.count == environments * horizon, "return normalizer counts every sample");
    bool carried_matches = true;
    for (std::size_t lane = 0; lane < environments; ++lane) {
        carried_matches = carried_matches &&
            std::abs(state.discounted_returns[lane] - carried[lane]) < 1e-5;
    }
    check(carried_matches, "discounted returns carry across rollouts and reset on termination");

    t8::v2::GpuRolloutBuffer restored(environments, horizon);
    restored.restore_return_normalizer_state(state);
    const auto round_trip = restored.return_normalizer_state();
    check(round_trip.count == state.count && round_trip.mean == state.mean &&
          round_trip.variance == state.variance &&
          round_trip.discounted_returns == state.discounted_returns,
          "return normalizer state round-trips exactly for resume");

    bool rejected_clip = false;
    try {
        static_cast<void>(rollout.normalize_rewards(gamma, 0.0F));
    } catch (const std::invalid_argument&) {
        rejected_clip = true;
    }
    check(rejected_clip, "return normalizer rejects a non-positive clip");
    cudaFree(device_values);
    cudaFree(device_truncated);
    cudaFree(device_terminated);
    cudaFree(device_rewards);
}

int main() {
    test_device_rollout_and_gae();
    test_return_normalization();
    test_parametric_policy_ppo_update();
    test_time_limit_bootstrap_and_reward_scaling();
    test_record_outcome_rejects_invalid_reward_scale();
    test_update_ppo_rejects_non_finite_rollout_and_illegal_action();
    if (failures != 0) {
        std::cerr << failures << " rollout assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Device rollout recording and normalized CUDA GAE passed\n";
    return EXIT_SUCCESS;
}
