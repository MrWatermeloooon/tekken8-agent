#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/temporal.hpp"
#include "t8_v2/training_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
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

void test_profiled_gpu_opponents() {
    constexpr std::size_t environments = 4096;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    std::vector<t8::v2::OpponentProfileParameters> profiles(2);
    profiles[0].id = 0;
    profiles[0].aggression = 1.0F;
    profiles[0].approach = 1.0F;
    profiles[0].backdash = 0.0F;
    profiles[0].sidestep_left = 0.0F;
    profiles[0].sidestep_right = 0.0F;
    profiles[0].input_error_rate = 0.0F;
    profiles[1] = profiles[0];
    profiles[1].id = 1;
    profiles[1].aggression = 0.0F;
    profiles[1].approach = 0.0F;
    profiles[1].backdash = 1.0F;
    std::vector<std::uint32_t> assignments(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) assignments[lane] = lane & 1U;
    opponent.set_profiles(profiles);
    opponent.set_profile_assignments(assignments);
    check(opponent.uses_profiles() && opponent.profile_count() == 2,
          "profile table stays device resident");
    check(opponent.download_profile_assignments(environments) == assignments,
          "profile assignments round-trip exactly");
    const auto view = simulator.device_view();
    static_cast<void>(opponent.actions_device(
        view.observations_p2, view.action_masks_p2, environments, 515, 0));
    const auto actions = opponent.download_actions(environments);
    std::size_t approach_actions = 0;
    std::size_t retreat_actions = 0;
    for (std::size_t lane = 0; lane < environments; ++lane) {
        if ((lane & 1U) == 0U && actions[lane] == static_cast<std::int64_t>(t8::v2::Action::DashForward)) {
            ++approach_actions;
        }
        if ((lane & 1U) != 0U && actions[lane] == static_cast<std::int64_t>(t8::v2::Action::DashBack)) {
            ++retreat_actions;
        }
    }
    check(approach_actions > environments * 0.45,
          "approach-heavy profiles produce GPU dash-forward behavior");
    check(retreat_actions > environments * 0.45,
          "backdash-heavy profiles produce GPU retreat behavior");
}

void test_gpu_temporal_matchup_encoder() {
    constexpr std::size_t environments = 32;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    std::vector<t8::v2::OpponentProfileParameters> profiles(2);
    profiles[0].id = 0;
    profiles[0].character_id = 0;
    profiles[0].archetype_id = 0;
    profiles[0].approach = 1.0F;
    profiles[1] = profiles[0];
    profiles[1].id = 1;
    profiles[1].character_id = 31;
    profiles[1].archetype_id = 6;
    profiles[1].stance_entry_frequency = 1.0F;
    std::vector<std::uint32_t> assignments(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) assignments[lane] = lane & 1U;
    opponent.set_profiles(profiles);
    opponent.set_profile_assignments(assignments);
    t8::v2::GpuTemporalMatchupEncoder encoder(environments, t8::v2::kVisualObservationSize);
    check(encoder.observation_size() == t8::v2::kMatchupVisualObservationSize,
          "temporal visual contract has 95 features");
    auto view = simulator.device_view();
    const float* encoded = encoder.encode(
        view.visual_observations_p2, opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
    t8::v2::ActorCriticConfig config{};
    config.observation_size = static_cast<int>(t8::v2::kMatchupVisualObservationSize);
    t8::v2::GpuActorCritic policy(environments, config, 42);
    static_cast<void>(policy.forward(encoded, view.action_masks_p2, environments, 1, 0, true));
    for (const auto action : policy.download_actions(environments)) {
        check(action >= 0 && action < static_cast<std::int64_t>(t8::v2::kActionCount),
              "95-feature temporal policy runs directly from GPU encoder output");
    }
    const auto first_state = encoder.download_state();
    check(first_state.valid.size() == environments &&
          std::all_of(first_state.valid.begin(), first_state.valid.end(), [](auto value) { return value == 1; }),
          "temporal encoder marks every lane valid after first frame");
    const auto* opponent_actions = opponent.actions_device(
        view.observations_p2, view.action_masks_p2, environments, 99, 0);
    simulator.step_device_i64(opponent_actions, opponent_actions);
    view = simulator.device_view();
    encoded = encoder.encode(
        view.visual_observations_p2, opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
    std::vector<float> host(environments * encoder.observation_size());
    cuda_check(cudaMemcpy(host.data(), encoded, sizeof(float) * host.size(), cudaMemcpyDeviceToHost),
               "download temporal matchup observations");
    const std::size_t identity = t8::v2::kVisualObservationSize;
    check(host[identity] != host[encoder.observation_size() + identity],
          "character identity embedding differs across assigned fighters");
    const std::size_t archetype = identity + t8::v2::kCharacterEmbeddingSize;
    check(host[archetype] == 1.0F && host[encoder.observation_size() + archetype + 6] == 1.0F,
          "archetype one-hot conditioning follows profile assignments");
    const auto second_state = encoder.download_state();
    check(second_state.history != first_state.history,
          "move, phase, outcome, distance, and movement history advances on GPU");
    encoder.reset();
    encoder.synchronize();
    const auto reset_state = encoder.download_state();
    check(std::all_of(reset_state.valid.begin(), reset_state.valid.end(), [](auto value) { return value == 0; }),
          "temporal reset clears episode history");
    encoder.upload_state(second_state);
    encoder.synchronize();
    check(encoder.download_state().history == second_state.history,
          "temporal history supports exact checkpoint round-trip");
}

void test_character_specific_moves_execute_on_gpu() {
    constexpr std::size_t environments = 2;
    t8::v2::Config config{};
    config.decision_frames = 4;
    config.max_frames = 4;
    t8::v2::GpuSimulatorBatch simulator(environments, config);
    t8::v2::GpuScriptedOpponent opponent(environments);

    std::vector<t8::v2::CharacterMoveParameters> moves(
        t8::v2::kRosterCharacterCount * t8::v2::kCharacterMoveSlotCount);
    for (std::uint32_t character = 0; character < t8::v2::kRosterCharacterCount; ++character) {
        for (std::uint32_t slot = 0; slot < t8::v2::kCharacterMoveSlotCount; ++slot) {
            auto& move = moves[character * t8::v2::kCharacterMoveSlotCount + slot];
            move.character_id = character;
            move.slot = slot;
            move.hit_level = slot == 5 ? 4 : (slot == 3 ? 3 : (slot == 0 ? 1 : 2));
            move.startup = 1;
            move.active = 2;
            move.recovery = 8;
            move.damage = 10.0F;
            move.range = 3.0F;
            move.hitstun = 12;
            move.blockstun = 6;
            move.pushback = 0.1F;
        }
    }
    moves[0].damage = 5.0F;
    moves[t8::v2::kCharacterMoveSlotCount].damage = 25.0F;
    simulator.set_character_move_specs(moves);

    std::vector<t8::v2::OpponentProfileParameters> profiles(2);
    profiles[0].id = 0;
    profiles[0].character_id = 0;
    profiles[1] = profiles[0];
    profiles[1].id = 1;
    profiles[1].character_id = 1;
    const std::vector<std::uint32_t> assignments = {0, 1};
    opponent.set_profiles(profiles);
    opponent.set_profile_assignments(assignments);
    simulator.set_opponent_characters_device(
        opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), 1);

    const std::vector<std::uint8_t> p1_actions(
        environments, static_cast<std::uint8_t>(t8::v2::Action::Crouch));
    const std::vector<std::uint8_t> p2_actions(
        environments, static_cast<std::uint8_t>(t8::v2::Action::Jab));
    simulator.step_host(p1_actions, p2_actions);
    const auto states = simulator.download_states();
    const double expected_character_0 = config.max_health - 5.0;
    const double expected_character_1 = config.max_health - 25.0;
    if (std::fabs(states[0].p1.health - expected_character_0) >= 1e-5 ||
        std::fabs(states[1].p1.health - expected_character_1) >= 1e-5) {
        std::cerr << "character damage healths=" << states[0].p1.health
                  << ',' << states[1].p1.health << '\n';
    }
    check(std::fabs(states[0].p1.health - expected_character_0) < 1e-5,
          "character 0 uses its five-damage jab in the CUDA combat kernel");
    check(std::fabs(states[1].p1.health - expected_character_1) < 1e-5,
          "character 1 uses its twenty-five-damage jab in the CUDA combat kernel");
    simulator.reset_done();
    simulator.step_host(p1_actions, p2_actions);
    const auto reset_states = simulator.download_states();
    check(std::fabs(reset_states[0].p1.health - expected_character_0) < 1e-5 &&
          std::fabs(reset_states[1].p1.health - expected_character_1) < 1e-5,
          "done-lane GPU resets preserve roster character assignments");
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
    test_profiled_gpu_opponents();
    test_gpu_temporal_matchup_encoder();
    test_held_out_opponent_and_side_router();
    test_character_specific_moves_execute_on_gpu();
    if (failures != 0) {
        std::cerr << failures << " policy assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "GPU actor-critic inference, masking, sampling, and simulator chaining passed\n";
    return EXIT_SUCCESS;
}
