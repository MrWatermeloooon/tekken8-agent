#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/ppo.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string_view>

namespace {

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition) {
        ++failures;
        if (failures < 30) std::cerr << "FAIL: " << message << '\n';
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
        rollout.record_outcome_device(step, after.rewards_p1, after.terminated);
        simulator.reset_done();
    }

    const auto final_state = simulator.device_view();
    const auto bootstrap = learner.forward(
        final_state.observations_p1, final_state.action_masks_p1,
        environments, 1000, horizon, true);
    const auto values_before_update = learner.download_values(environments);
    rollout.compute_gae(bootstrap.values, 0.99F, 0.95F, true);
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

    t8::v2::PpoUpdateConfig update_config{};
    update_config.epochs = 2;
    update_config.minibatch_size = environments;
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
}

}  // namespace

int main() {
    test_device_rollout_and_gae();
    if (failures != 0) {
        std::cerr << failures << " rollout assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Device rollout recording and normalized CUDA GAE passed\n";
    return EXIT_SUCCESS;
}
