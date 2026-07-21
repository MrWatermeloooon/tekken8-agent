#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/training_router.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
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

void test_policy_shapes_and_sampling() {
    constexpr std::size_t environments = 8192;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuActorCritic policy(environments, {}, 12345);
    const auto simulator_view = simulator.device_view();

    check(policy.capacity() == environments, "policy capacity");
    check(policy.parameter_count() == 77337, "256x256 actor-critic parameter count");

    static_cast<void>(policy.forward(simulator_view.observations_p1, simulator_view.action_masks_p1,
                                     environments, 999, 0, false));
    const auto actions = policy.download_actions(environments);
    const auto values = policy.download_values(environments);
    const auto log_probabilities = policy.download_log_probabilities(environments);
    const auto entropies = policy.download_entropies(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) {
        check(actions[lane] >= 0 && actions[lane] < static_cast<std::int64_t>(t8::v2::kActionCount),
              "sampled action is in range");
        check(std::isfinite(values[lane]), "value is finite");
        check(std::isfinite(log_probabilities[lane]) && log_probabilities[lane] <= 0.0F,
              "log probability is finite and non-positive");
        check(std::isfinite(entropies[lane]) && entropies[lane] > 3.0F && entropies[lane] < 3.3F,
              "initial policy entropy is near log(24)");
    }
}

void test_determinism_and_busy_mask() {
    constexpr std::size_t environments = 4096;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuActorCritic policy(environments, {}, 77);
    auto view = simulator.device_view();

    static_cast<void>(policy.forward(
        view.observations_p1, view.action_masks_p1, environments, 10, 0, true));
    const auto first = policy.download_actions(environments);
    static_cast<void>(policy.forward(
        view.observations_p1, view.action_masks_p1, environments, 99999, 991, true));
    const auto second = policy.download_actions(environments);
    check(first == second, "deterministic actions ignore sampling seed and repeat exactly");

    std::vector<std::uint8_t> p1_actions(environments, static_cast<std::uint8_t>(t8::v2::Action::Jab));
    std::vector<std::uint8_t> p2_actions(environments, static_cast<std::uint8_t>(t8::v2::Action::Neutral));
    simulator.step_host(p1_actions, p2_actions);
    view = simulator.device_view();
    static_cast<void>(policy.forward(
        view.observations_p1, view.action_masks_p1, environments, 55, 1, false));
    const auto busy_actions = policy.download_actions(environments);
    for (const auto action : busy_actions) {
        check(action == static_cast<std::int64_t>(t8::v2::Action::Neutral),
              "busy action mask forces neutral entirely on GPU");
    }
}

void test_zero_copy_policy_to_simulator_chain() {
    constexpr std::size_t environments = 16384;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuActorCritic p1_policy(environments, {}, 101);
    t8::v2::GpuActorCritic p2_policy(environments, {}, 202);
    const auto view = simulator.device_view();
    const auto p1 = p1_policy.forward(view.observations_p1, view.action_masks_p1,
                                      environments, 1001, 0, false);
    const auto p2 = p2_policy.forward(view.observations_p2, view.action_masks_p2,
                                      environments, 2002, 0, false);
    simulator.step_device_i64(p1.actions, p2.actions);
    simulator.synchronize();
    const auto states = simulator.download_states();
    for (const auto& state : states) check(state.frame == 4, "GPU policy actions advance GPU simulator");
}

void test_checkpoint_round_trip() {
    constexpr std::size_t environments = 1024;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuActorCritic original(environments, {}, 5150);
    t8::v2::GpuActorCritic restored(environments, {}, 9191);
    const auto view = simulator.device_view();
    static_cast<void>(original.forward(
        view.observations_p1, view.action_masks_p1, environments, 1, 1, true));
    const auto expected_actions = original.download_actions(environments);
    const auto expected_values = original.download_values(environments);
    const auto checkpoint = std::filesystem::temp_directory_path() / "t8_v2_policy_roundtrip.t8ppo";
    std::error_code error;
    std::filesystem::remove(checkpoint, error);
    original.save_checkpoint(checkpoint);
    restored.load_checkpoint(checkpoint);
    static_cast<void>(restored.forward(
        view.observations_p1, view.action_masks_p1, environments, 999, 999, true));
    check(restored.download_actions(environments) == expected_actions,
          "checkpoint restores deterministic actions exactly");
    check(restored.download_values(environments) == expected_values,
          "checkpoint restores values exactly");
    bool refused_overwrite = false;
    try {
        original.save_checkpoint(checkpoint);
    } catch (const std::runtime_error&) {
        refused_overwrite = true;
    }
    check(refused_overwrite, "checkpoint save refuses to overwrite an existing artifact");

    {
        std::fstream corrupt(checkpoint, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekg(-1, std::ios::end);
        char byte = 0;
        corrupt.read(&byte, 1);
        byte ^= 0x5A;
        corrupt.seekp(-1, std::ios::end);
        corrupt.write(&byte, 1);
    }
    bool rejected_corruption = false;
    try {
        restored.load_checkpoint(checkpoint);
    } catch (const std::runtime_error&) {
        rejected_corruption = true;
    }
    check(rejected_corruption, "checkpoint checksum rejects corrupted payloads");
    std::filesystem::remove(checkpoint, error);
}

void test_scripted_opponent_mixture() {
    constexpr std::size_t environments = 8192;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    auto view = simulator.device_view();
    const auto* actions_device = opponent.actions_device(
        view.observations_p2, view.action_masks_p2, environments, 123, 0);
    const auto actions = opponent.download_actions(environments);
    bool saw_attack = false;
    bool saw_movement = false;
    for (const auto action : actions) {
        check(action >= 0 && action < static_cast<std::int64_t>(t8::v2::kActionCount),
              "scripted action is in range");
        saw_attack = saw_attack || action >= static_cast<std::int64_t>(t8::v2::Action::Jab);
        saw_movement = saw_movement || (action >= static_cast<std::int64_t>(t8::v2::Action::WalkForward) &&
                                        action <= static_cast<std::int64_t>(t8::v2::Action::SidewalkRight));
    }
    check(saw_attack && saw_movement, "scripted mixture contains attacks and movement");
    simulator.step_device_i64(actions_device, actions_device);
    view = simulator.device_view();
    static_cast<void>(opponent.actions_device(
        view.observations_p1, view.action_masks_p1, environments, 123, 1));
    const auto busy_actions = opponent.download_actions(environments);
    const auto busy_masks = simulator.download_action_masks(1);
    for (std::size_t lane = 0; lane < environments; ++lane) {
        if (!busy_masks[lane * t8::v2::kActionCount + static_cast<std::size_t>(t8::v2::Action::Jab)]) {
            check(busy_actions[lane] == static_cast<std::int64_t>(t8::v2::Action::Neutral),
                  "scripted opponent respects busy mask");
        }
    }
}

void test_held_out_opponent_and_side_router() {
    constexpr std::size_t environments = 32;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    t8::v2::GpuLearnerSideRouter router(environments);
    const auto simulator_view = simulator.device_view();
    const auto routed = router.select_observations(simulator_view, environments);
    check(routed.learner_observations != nullptr && routed.opponent_observations != nullptr,
          "side router keeps both observation roles on device");
    const auto* training_device = opponent.actions_device(
        routed.opponent_observations, routed.opponent_action_masks,
        environments, 919, 0, t8::v2::ScriptedOpponentSet::TrainingV1);
    static_cast<void>(training_device);
    const auto training_actions = opponent.download_actions(environments);
    const auto* held_out_device = opponent.actions_device(
        routed.opponent_observations, routed.opponent_action_masks,
        environments, 919, 0, t8::v2::ScriptedOpponentSet::HeldOutV2);
    static_cast<void>(held_out_device);
    const auto held_out_actions = opponent.download_actions(environments);
    check(training_actions != held_out_actions, "held-out audit suite differs from training suite");

    const auto routed_visual = router.select_visual_observations(simulator_view, environments);
    t8::v2::ActorCriticConfig visual_config{};
    visual_config.observation_size = static_cast<int>(t8::v2::kVisualObservationSize);
    t8::v2::GpuActorCritic visual_policy(environments, visual_config, 8080);
    static_cast<void>(visual_policy.forward(
        routed_visual.learner_observations, routed_visual.learner_action_masks,
        environments, 44, 0, true));
    for (const auto action : visual_policy.download_actions(environments)) {
        check(action >= 0 && action < static_cast<std::int64_t>(t8::v2::kActionCount),
              "13-feature visual policy runs directly on routed GPU observations");
    }

    std::vector<std::int64_t> learner_actions(environments,
        static_cast<std::int64_t>(t8::v2::Action::Jab));
    std::vector<std::int64_t> opponent_actions(environments,
        static_cast<std::int64_t>(t8::v2::Action::BlockHigh));
    std::int64_t* learner_actions_device = nullptr;
    std::int64_t* opponent_actions_device = nullptr;
    float* p1_rewards_device = nullptr;
    float* p2_rewards_device = nullptr;
    cuda_check(cudaMalloc(&learner_actions_device, sizeof(std::int64_t) * environments),
               "allocate learner action test input");
    cuda_check(cudaMalloc(&opponent_actions_device, sizeof(std::int64_t) * environments),
               "allocate opponent action test input");
    cuda_check(cudaMalloc(&p1_rewards_device, sizeof(float) * environments),
               "allocate P1 reward test input");
    cuda_check(cudaMalloc(&p2_rewards_device, sizeof(float) * environments),
               "allocate P2 reward test input");
    cuda_check(cudaMemcpy(learner_actions_device, learner_actions.data(),
                          sizeof(std::int64_t) * environments, cudaMemcpyHostToDevice),
               "upload learner action test input");
    cuda_check(cudaMemcpy(opponent_actions_device, opponent_actions.data(),
                          sizeof(std::int64_t) * environments, cudaMemcpyHostToDevice),
               "upload opponent action test input");

    const auto routed_actions =
        router.route_actions(learner_actions_device, opponent_actions_device, environments);
    std::vector<std::int64_t> p1_actions(environments);
    std::vector<std::int64_t> p2_actions(environments);
    cuda_check(cudaMemcpy(p1_actions.data(), routed_actions.p1_actions,
                          sizeof(std::int64_t) * environments, cudaMemcpyDeviceToHost),
               "download routed P1 actions");
    cuda_check(cudaMemcpy(p2_actions.data(), routed_actions.p2_actions,
                          sizeof(std::int64_t) * environments, cudaMemcpyDeviceToHost),
               "download routed P2 actions");
    for (std::size_t lane = 0; lane < environments; ++lane) {
        const bool learner_p1 = ((lane / t8::v2::kEvaluationStyleCount) & 1U) == 0U;
        check(p1_actions[lane] == (learner_p1 ? learner_actions[lane] : opponent_actions[lane]),
              "side router assigns P1 action by eight-style block");
        check(p2_actions[lane] == (learner_p1 ? opponent_actions[lane] : learner_actions[lane]),
              "side router assigns P2 action by eight-style block");
    }

    std::vector<float> p1_rewards(environments);
    std::vector<float> p2_rewards(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) {
        p1_rewards[lane] = 100.0F + static_cast<float>(lane);
        p2_rewards[lane] = -100.0F - static_cast<float>(lane);
    }
    cuda_check(cudaMemcpy(p1_rewards_device, p1_rewards.data(), sizeof(float) * environments,
                          cudaMemcpyHostToDevice), "upload P1 reward test input");
    cuda_check(cudaMemcpy(p2_rewards_device, p2_rewards.data(), sizeof(float) * environments,
                          cudaMemcpyHostToDevice), "upload P2 reward test input");
    const float* routed_rewards =
        router.select_rewards(p1_rewards_device, p2_rewards_device, environments);
    std::vector<float> learner_rewards(environments);
    cuda_check(cudaMemcpy(learner_rewards.data(), routed_rewards, sizeof(float) * environments,
                          cudaMemcpyDeviceToHost), "download routed learner rewards");
    for (std::size_t lane = 0; lane < environments; ++lane) {
        const bool learner_p1 = ((lane / t8::v2::kEvaluationStyleCount) & 1U) == 0U;
        check(learner_rewards[lane] == (learner_p1 ? p1_rewards[lane] : p2_rewards[lane]),
              "side router selects reward by eight-style block");
    }

    simulator.step_device_i64(routed_actions.p1_actions, routed_actions.p2_actions);
    simulator.synchronize();
    const auto states = simulator.download_states();
    for (const auto& state : states) check(state.frame == 4, "routed actions advance every lane");
    cudaFree(p2_rewards_device);
    cudaFree(p1_rewards_device);
    cudaFree(opponent_actions_device);
    cudaFree(learner_actions_device);
}

}  // namespace

int main() {
    test_policy_shapes_and_sampling();
    test_determinism_and_busy_mask();
    test_zero_copy_policy_to_simulator_chain();
    test_checkpoint_round_trip();
    test_scripted_opponent_mixture();
    test_held_out_opponent_and_side_router();
    if (failures != 0) {
        std::cerr << failures << " policy assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "GPU actor-critic inference, masking, sampling, and simulator chaining passed\n";
    return EXIT_SUCCESS;
}
