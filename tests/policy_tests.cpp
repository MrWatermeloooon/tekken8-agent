#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
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
    original.save_checkpoint(checkpoint);
    restored.load_checkpoint(checkpoint);
    static_cast<void>(restored.forward(
        view.observations_p1, view.action_masks_p1, environments, 999, 999, true));
    check(restored.download_actions(environments) == expected_actions,
          "checkpoint restores deterministic actions exactly");
    check(restored.download_values(environments) == expected_values,
          "checkpoint restores values exactly");
    std::error_code error;
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

}  // namespace

int main() {
    test_policy_shapes_and_sampling();
    test_determinism_and_busy_mask();
    test_zero_copy_policy_to_simulator_chain();
    test_checkpoint_round_trip();
    test_scripted_opponent_mixture();
    if (failures != 0) {
        std::cerr << failures << " policy assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "GPU actor-critic inference, masking, sampling, and simulator chaining passed\n";
    return EXIT_SUCCESS;
}
