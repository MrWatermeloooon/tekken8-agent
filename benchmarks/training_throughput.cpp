#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/training_router.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

std::size_t argument(int argc, char** argv, std::string_view name, std::size_t fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return static_cast<std::size_t>(std::stoull(argv[index + 1]));
    }
    return fallback;
}

bool flag(int argc, char** argv, std::string_view name) {
    for (int index = 1; index < argc; ++index) {
        if (argv[index] == name) return true;
    }
    return false;
}

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t environments = argument(argc, argv, "--envs", 4096);
        const std::size_t horizon = argument(argc, argv, "--horizon", 128);
        const std::size_t updates = argument(argc, argv, "--updates", 10);
        const std::size_t minibatch = argument(argc, argv, "--minibatch", 4096);
        const int epochs = static_cast<int>(argument(argc, argv, "--epochs", 4));
        const bool visual = flag(argc, argv, "--visual");
        if (environments == 0 || environments % (2 * t8::v2::kEvaluationStyleCount) != 0 ||
            horizon == 0 || updates == 0 || minibatch == 0 || minibatch > environments * horizon) {
            throw std::invalid_argument("invalid side-balanced training benchmark dimensions");
        }

        t8::v2::Config simulation_config{};
        simulation_config.timeout_ties_are_draws = true;
        simulation_config.randomize_initial_positions = true;
        t8::v2::ActorCriticConfig actor_config{};
        actor_config.observation_size = static_cast<int>(
            visual ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize);
        t8::v2::GpuSimulatorBatch simulator(environments, simulation_config);
        t8::v2::GpuActorCritic learner(std::max(environments, minibatch), actor_config, 7001);
        t8::v2::GpuScriptedOpponent opponent(environments);
        t8::v2::GpuLearnerSideRouter router(environments);
        t8::v2::GpuRolloutBuffer rollout(environments, horizon, actor_config);
        t8::v2::PpoUpdateConfig update_config{};
        update_config.epochs = epochs;
        update_config.minibatch_size = minibatch;

        double rollout_seconds = 0.0;
        double optimization_seconds = 0.0;
        t8::v2::PpoUpdateMetrics final_metrics{};
        check_cuda(cudaDeviceSynchronize(), "synchronize before training benchmark");
        const auto total_start = std::chrono::steady_clock::now();
        for (std::size_t update = 1; update <= updates; ++update) {
            const auto rollout_start = std::chrono::steady_clock::now();
            for (std::size_t step = 0; step < horizon; ++step) {
                const auto state = simulator.device_view();
                const auto inputs = visual
                    ? router.select_visual_observations(state, environments)
                    : router.select_observations(state, environments);
                const auto policy = learner.forward(
                    inputs.learner_observations, inputs.learner_action_masks,
                    environments, 7001, update * horizon + step, false);
                const auto* scripted = opponent.actions_device(
                    inputs.opponent_observations, inputs.opponent_action_masks,
                    environments, 7002, update * horizon + step,
                    t8::v2::ScriptedOpponentSet::TrainingV1);
                const auto actions = router.route_actions(policy.actions, scripted, environments);
                rollout.record_policy_device(
                    step, inputs.learner_observations, inputs.learner_action_masks,
                    policy.actions, policy.log_probabilities, policy.values);
                simulator.step_device_i64(actions.p1_actions, actions.p2_actions);
                const auto after = simulator.device_view();
                const auto* rewards = router.select_rewards(
                    after.sparse_rewards_p1, after.sparse_rewards_p2, environments);
                rollout.record_outcome_device(step, rewards, after.terminated);
                simulator.reset_done_seeded(update * horizon + step + 1);
            }
            const auto final_state = simulator.device_view();
            const auto final_inputs = visual
                ? router.select_visual_observations(final_state, environments)
                : router.select_observations(final_state, environments);
            const auto bootstrap = learner.forward(
                final_inputs.learner_observations, final_inputs.learner_action_masks,
                environments, 7001, update * horizon + horizon, true);
            rollout.compute_gae(bootstrap.values, 0.99F, 0.95F, true);
            rollout.synchronize();
            rollout_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - rollout_start).count();

            const auto optimization_start = std::chrono::steady_clock::now();
            final_metrics = learner.update_ppo(
                rollout.device_view(), update_config, 7001 + update * 10'000);
            optimization_seconds += std::chrono::duration<double>(
                std::chrono::steady_clock::now() - optimization_start).count();
        }
        check_cuda(cudaDeviceSynchronize(), "synchronize after training benchmark");
        const double total_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - total_start).count();
        const double decisions = static_cast<double>(environments) * horizon * updates;
        const double optimized_samples = decisions * final_metrics.epochs_completed;
        std::cout << std::fixed << std::setprecision(2)
                  << "GPU environments: " << environments << '\n'
                  << "Observation mode: " << (visual ? "visual" : "privileged") << '\n'
                  << "Horizon: " << horizon << '\n'
                  << "Updates: " << updates << '\n'
                  << "Rollout seconds: " << rollout_seconds << '\n'
                  << "PPO seconds: " << optimization_seconds << '\n'
                  << "Total seconds: " << total_seconds << '\n'
                  << "Environment decisions/s: " << decisions / total_seconds << '\n'
                  << "PPO sample-visits/s: " << optimized_samples / optimization_seconds << '\n'
                  << "Final approximate KL: " << final_metrics.approximate_kl << '\n'
                  << "Final epochs completed: " << final_metrics.epochs_completed << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
