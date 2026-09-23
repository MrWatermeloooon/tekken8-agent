#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/temporal.hpp"
#include "t8_v2/training_router.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
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

void test_parametric_move_scorer_masks_variable_candidates() {
    constexpr std::size_t environments = 32;
    t8::v2::ActorCriticConfig config{};
    config.observation_size = 5;
    config.action_count = 4;
    config.hidden_size = 16;
    config.action_feature_size = 2;
    config.universal_action_count = 2;
    config.action_contract = "parametric-test-v3";
    t8::v2::GpuActorCritic policy(environments, config, 31337);
    check(policy.parameter_count() == 419, "parametric head size depends on feature width, not candidates");

    std::vector<float> observations(environments * 5, 0.25F);
    std::vector<std::uint8_t> masks(environments * 4, 0);
    for (std::size_t lane = 0; lane < environments; ++lane) masks[lane * 4 + 2] = 1;
    const std::vector<float> features = {
        1.0F, 0.0F, -1.0F, 0.0F, 0.0F, 1.0F, 0.0F, -1.0F};
    float* device_observations = nullptr;
    std::uint8_t* device_masks = nullptr;
    cuda_check(cudaMalloc(&device_observations, sizeof(float) * observations.size()),
               "allocate parametric observations");
    cuda_check(cudaMalloc(&device_masks, sizeof(std::uint8_t) * masks.size()),
               "allocate parametric masks");
    cuda_check(cudaMemcpy(device_observations, observations.data(), sizeof(float) * observations.size(),
                          cudaMemcpyHostToDevice), "upload parametric observations");
    cuda_check(cudaMemcpy(device_masks, masks.data(), sizeof(std::uint8_t) * masks.size(),
                          cudaMemcpyHostToDevice), "upload parametric masks");
    bool rejected_missing_features = false;
    try {
        static_cast<void>(policy.forward(
            device_observations, device_masks, environments, 1, 0, true));
    } catch (const std::logic_error&) {
        rejected_missing_features = true;
    }
    check(rejected_missing_features, "parametric policy requires an installed move table");
    policy.set_action_features(features);
    static_cast<void>(policy.forward(
        device_observations, device_masks, environments, 1, 0, true));
    for (const auto action : policy.download_actions(environments)) {
        check(action == 2, "parametric scorer samples only legal candidates");
    }
    for (const auto value : policy.download_values(environments)) {
        check(std::isfinite(value), "parametric state-only value is finite");
    }
    cudaFree(device_masks);
    cudaFree(device_observations);
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

    std::vector<t8::v2::OpponentProfileParameters> profiles(1);
    profiles[0].character_id = t8::v2::kJunCharacterId;
    opponent.set_profiles(profiles);
    opponent.set_profile_assignments(std::vector<std::uint32_t>(environments, 0));
    static_cast<void>(opponent.actions_device(
        routed.opponent_observations, routed.opponent_action_masks,
        environments, 919, 0, t8::v2::ScriptedOpponentSet::HeldOutV2));
    check(opponent.download_actions(environments) == held_out_actions,
          "held-out actions remain fixed when roster profiles are installed");

    const auto routed_visual = router.select_visual_observations(simulator_view, environments);
    const auto routed_self_play_visual =
        router.select_self_play_visual_observations(simulator_view, environments);
    check(routed_self_play_visual.opponent_observations != nullptr,
          "self-play router exposes screen-compatible observations for both policies");
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

    const auto* mixed_actions_device = router.mix_self_play_actions(
        learner_actions_device, opponent_actions_device, environments);
    std::vector<std::int64_t> mixed_actions(environments);
    cuda_check(cudaMemcpy(mixed_actions.data(), mixed_actions_device,
                          sizeof(std::int64_t) * environments, cudaMemcpyDeviceToHost),
               "download mixed self-play actions");
    for (std::size_t lane = 0; lane < environments; ++lane) {
        check(mixed_actions[lane] == (lane % 5 == 0 ? opponent_actions[lane] : learner_actions[lane]),
              "self-play lanes use an exact 80/20 latest-to-best mixture");
    }

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

void test_profile_assignments_change_only_for_done_lanes() {
    constexpr std::size_t environments = 2;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    std::vector<t8::v2::OpponentProfileParameters> profiles(2);
    profiles[0].id = 0;
    profiles[1].id = 1;
    opponent.set_profiles(profiles);
    opponent.set_profile_assignments(std::vector<std::uint32_t>{0, 0});

    auto states = simulator.download_states();
    states[0].round_over = true;
    states[0].winner = 1;
    simulator.upload_states(states);
    const std::vector<std::uint32_t> candidates = {1, 1};
    opponent.set_profile_assignments_for_done(candidates, simulator.device_view().terminated);
    opponent.synchronize();
    const auto assignments = opponent.download_profile_assignments(environments);
    check(assignments[0] == 1, "done lane receives its next opponent profile");
    check(assignments[1] == 0, "active lane keeps its current opponent profile");
}

void test_opponent_character_swap_respects_lane_mask() {
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
    opponent.set_profiles(profiles);

    const std::vector<std::uint8_t> p1_actions(
        environments, static_cast<std::uint8_t>(t8::v2::Action::Crouch));
    const std::vector<std::uint8_t> p2_actions(
        environments, static_cast<std::uint8_t>(t8::v2::Action::Jab));

    opponent.set_profile_assignments(std::vector<std::uint32_t>{0, 0});
    simulator.set_opponent_characters_device(
        opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), 1);
    simulator.step_host(p1_actions, p2_actions);
    const auto baseline_states = simulator.download_states();
    check(std::fabs(baseline_states[0].p1.health - (config.max_health - 5.0)) < 1e-5 &&
          std::fabs(baseline_states[1].p1.health - (config.max_health - 5.0)) < 1e-5,
          "both lanes start on the character-0 five-damage jab");

    simulator.reset_done();
    simulator.synchronize();

    std::uint8_t* lane_mask_device = nullptr;
    cuda_check(cudaMalloc(&lane_mask_device, sizeof(std::uint8_t) * environments),
               "allocate character-swap lane mask");
    const std::vector<std::uint8_t> lane_mask_host = {0, 1};
    cuda_check(cudaMemcpy(lane_mask_device, lane_mask_host.data(),
                          sizeof(std::uint8_t) * environments, cudaMemcpyHostToDevice),
               "upload character-swap lane mask");
    const std::vector<std::uint32_t> candidates = {1, 1};
    opponent.set_profile_assignments_for_done(candidates, lane_mask_device);
    opponent.synchronize();
    simulator.set_opponent_characters_device(
        opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), 1, lane_mask_device);

    simulator.step_host(p1_actions, p2_actions);
    const auto masked_states = simulator.download_states();
    check(std::fabs(masked_states[0].p1.health - (config.max_health - 5.0)) < 1e-5,
          "masked-out lane keeps its mid-catalog character instead of swapping opponents");
    check(std::fabs(masked_states[1].p1.health - (config.max_health - 25.0)) < 1e-5,
          "masked-in lane swaps to its newly assigned character");
    cudaFree(lane_mask_device);
}

void test_temporal_preview_does_not_mutate_state() {
    constexpr std::size_t environments = 16;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    std::vector<t8::v2::OpponentProfileParameters> profiles(1);
    profiles[0].character_id = 4;
    opponent.set_profiles(profiles);
    opponent.set_profile_assignments(std::vector<std::uint32_t>(environments, 0));
    t8::v2::GpuTemporalMatchupEncoder encoder(environments, t8::v2::kObservationSize);

    auto view = simulator.device_view();
    static_cast<void>(encoder.encode(
        view.observations_p2, opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments));
    const auto* first_opponent_actions = opponent.actions_device(
        view.observations_p2, view.action_masks_p2, environments, 55, 0);
    simulator.step_device_i64(first_opponent_actions, first_opponent_actions);
    view = simulator.device_view();

    const auto baseline_state = encoder.download_state();
    const std::size_t width = encoder.observation_size();

    const float* preview_output = encoder.preview(
        view.observations_p2, opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
    std::vector<float> preview_values(environments * width);
    cuda_check(cudaMemcpy(preview_values.data(), preview_output, sizeof(float) * preview_values.size(),
                          cudaMemcpyDeviceToHost), "download temporal preview output");
    check(encoder.download_state().history == baseline_state.history,
          "preview does not advance temporal history");
    check(encoder.download_state().valid == baseline_state.valid,
          "preview does not change validity flags");
    check(encoder.download_state().previous_actions == baseline_state.previous_actions,
          "preview does not commit the previewed action");

    const float* preview_again = encoder.preview(
        view.observations_p2, opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
    std::vector<float> preview_again_values(preview_values.size());
    cuda_check(cudaMemcpy(preview_again_values.data(), preview_again,
                          sizeof(float) * preview_again_values.size(), cudaMemcpyDeviceToHost),
               "download repeated temporal preview output");
    check(preview_values == preview_again_values,
          "repeated preview calls from the same state are idempotent");

    const float* encode_output = encoder.encode(
        view.observations_p2, opponent.profiles_device(), opponent.profile_count(),
        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
    std::vector<float> encode_values(preview_values.size());
    cuda_check(cudaMemcpy(encode_values.data(), encode_output, sizeof(float) * encode_values.size(),
                          cudaMemcpyDeviceToHost), "download temporal encode output");
    check(preview_values == encode_values,
          "preview matches the observation a real encode would produce from the same state");
    check(encoder.download_state().history != baseline_state.history,
          "a real encode call still advances history after preview left it untouched");
}

void test_self_play_visual_router_uses_screen_observations_for_opponent() {
    constexpr std::size_t environments = 16;
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuLearnerSideRouter router(environments);
    const auto view = simulator.device_view();
    const auto routed = router.select_self_play_visual_observations(view, environments);

    std::vector<float> opponent_observed(environments * t8::v2::kVisualObservationSize);
    cuda_check(cudaMemcpy(opponent_observed.data(), routed.opponent_observations,
                          sizeof(float) * opponent_observed.size(), cudaMemcpyDeviceToHost),
               "download self-play opponent observations");
    std::vector<float> visual_p1(environments * t8::v2::kVisualObservationSize);
    std::vector<float> visual_p2(environments * t8::v2::kVisualObservationSize);
    cuda_check(cudaMemcpy(visual_p1.data(), view.visual_observations_p1,
                          sizeof(float) * visual_p1.size(), cudaMemcpyDeviceToHost),
               "download raw visual P1 observations");
    cuda_check(cudaMemcpy(visual_p2.data(), view.visual_observations_p2,
                          sizeof(float) * visual_p2.size(), cudaMemcpyDeviceToHost),
               "download raw visual P2 observations");

    bool matches_screen_tensor = true;
    for (std::size_t lane = 0; lane < environments; ++lane) {
        const bool learner_p1 = ((lane / t8::v2::kEvaluationStyleCount) & 1U) == 0U;
        const float* expected =
            (learner_p1 ? visual_p2 : visual_p1).data() + lane * t8::v2::kVisualObservationSize;
        const float* actual = opponent_observed.data() + lane * t8::v2::kVisualObservationSize;
        if (!std::equal(expected, expected + t8::v2::kVisualObservationSize, actual)) {
            matches_screen_tensor = false;
        }
    }
    check(matches_screen_tensor,
          "self-play router gives the opponent the screen-compatible tensor, not privileged state");
}

void test_action_history_device_validation() {
    constexpr std::size_t environments = 4;
    t8::v2::GpuScriptedOpponent opponent(environments);
    std::int64_t* actions_device = nullptr;
    cuda_check(cudaMalloc(&actions_device, sizeof(std::int64_t) * environments),
               "allocate action-history test buffer");

    bool rejected_null = false;
    try {
        opponent.set_action_history_device(nullptr, environments);
    } catch (const std::invalid_argument&) {
        rejected_null = true;
    }
    check(rejected_null, "device action history rejects a null pointer");

    bool rejected_zero = false;
    try {
        opponent.set_action_history_device(actions_device, 0);
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }
    check(rejected_zero, "device action history rejects a zero environment count");

    bool rejected_overflow = false;
    try {
        opponent.set_action_history_device(actions_device, environments + 1);
    } catch (const std::invalid_argument&) {
        rejected_overflow = true;
    }
    check(rejected_overflow, "device action history rejects a count exceeding capacity");
    cudaFree(actions_device);
}

void test_actor_critic_and_router_reject_invalid_configuration() {
    bool rejected_zero_capacity = false;
    try {
        t8::v2::GpuActorCritic invalid(0);
    } catch (const std::invalid_argument&) {
        rejected_zero_capacity = true;
    }
    check(rejected_zero_capacity, "actor-critic rejects zero capacity");

    t8::v2::ActorCriticConfig bad_actions{};
    bad_actions.action_count = 1;
    bool rejected_action_count = false;
    try {
        t8::v2::GpuActorCritic invalid(4, bad_actions);
    } catch (const std::invalid_argument&) {
        rejected_action_count = true;
    }
    check(rejected_action_count, "actor-critic rejects a single-action policy");

    t8::v2::ActorCriticConfig bad_hidden{};
    bad_hidden.hidden_size = 0;
    bool rejected_hidden = false;
    try {
        t8::v2::GpuActorCritic invalid(4, bad_hidden);
    } catch (const std::invalid_argument&) {
        rejected_hidden = true;
    }
    check(rejected_hidden, "actor-critic rejects a zero hidden size");

    bool rejected_router_capacity = false;
    try {
        t8::v2::GpuLearnerSideRouter invalid(0);
    } catch (const std::invalid_argument&) {
        rejected_router_capacity = true;
    }
    check(rejected_router_capacity, "side router rejects zero capacity");

    t8::v2::GpuLearnerSideRouter router(4);
    t8::v2::GpuSimulatorBatch simulator(4);
    bool rejected_environment_overflow = false;
    try {
        static_cast<void>(router.select_observations(simulator.device_view(), 5));
    } catch (const std::invalid_argument&) {
        rejected_environment_overflow = true;
    }
    check(rejected_environment_overflow, "side router rejects an environment count beyond capacity");
}

void test_checkpoint_rejects_architecture_mismatch_and_nonfinite_payload() {
    constexpr std::size_t environments = 8;
    t8::v2::ActorCriticConfig small_config{};
    small_config.hidden_size = 16;
    t8::v2::GpuActorCritic small(environments, small_config, 111);
    const auto checkpoint = std::filesystem::temp_directory_path() / "t8_v2_policy_mismatch.t8ppo";
    std::error_code error;
    std::filesystem::remove(checkpoint, error);
    small.save_checkpoint(checkpoint);

    t8::v2::ActorCriticConfig large_config{};
    large_config.hidden_size = 32;
    t8::v2::GpuActorCritic large(environments, large_config, 222);
    bool rejected_mismatch = false;
    try {
        large.load_checkpoint(checkpoint);
    } catch (const std::runtime_error&) {
        rejected_mismatch = true;
    }
    check(rejected_mismatch, "checkpoint load rejects an architecture mismatch");

    // Hand-craft a checksum-valid checkpoint whose payload contains a NaN, to
    // exercise the finiteness guard independently of the checksum guard.
    struct HeaderLayout {
        std::array<char, 8> magic{};
        std::uint32_t version = 0;
        std::uint32_t observation_size = 0;
        std::uint32_t action_count = 0;
        std::uint32_t hidden_size = 0;
        std::uint64_t optimizer_step = 0;
        std::uint64_t parameter_count = 0;
    };
    struct IntegrityLayout {
        std::uint64_t payload_bytes = 0;
        std::uint64_t payload_checksum = 0;
    };
    struct ContractLayout {
        std::array<char, 64> catalog_sha256{};
        std::array<char, 24> roster_version{};
        std::array<char, 32> observation_contract{};
        std::array<char, 32> action_contract{};
        std::uint32_t action_feature_size = 0;
        std::uint32_t universal_action_count = 0;
    };
    struct NormalizerLayout {
        std::uint32_t enabled = 0;
        std::uint32_t reserved = 0;
        std::uint64_t count = 0;
        float clip = 0.0F;
        float epsilon = 0.0F;
    };
    std::ifstream input(checkpoint, std::ios::binary);
    const std::vector<char> raw(
        (std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>());
    input.close();
    HeaderLayout header{};
    IntegrityLayout integrity{};
    std::memcpy(&header, raw.data(), sizeof(header));
    constexpr std::size_t integrity_offset =
        sizeof(HeaderLayout) + sizeof(ContractLayout) + sizeof(NormalizerLayout);
    std::memcpy(&integrity, raw.data() + integrity_offset, sizeof(integrity));
    check(header.version == 4, "checkpoint format under test is normalizer-bearing version 4");

    std::vector<char> mutated = raw;
    auto* payload = reinterpret_cast<float*>(mutated.data() + integrity_offset + sizeof(integrity));
    payload[0] = std::numeric_limits<float>::quiet_NaN();
    const auto* payload_bytes = reinterpret_cast<const unsigned char*>(payload);
    constexpr std::uint64_t offset_basis = 14695981039346656037ULL;
    constexpr std::uint64_t prime = 1099511628211ULL;
    std::uint64_t checksum = offset_basis;
    for (std::uint64_t index = 0; index < integrity.payload_bytes; ++index) {
        checksum ^= payload_bytes[index];
        checksum *= prime;
    }
    const IntegrityLayout patched_integrity{integrity.payload_bytes, checksum};
    std::memcpy(mutated.data() + integrity_offset, &patched_integrity, sizeof(patched_integrity));

    const auto nonfinite_checkpoint =
        std::filesystem::temp_directory_path() / "t8_v2_policy_nonfinite.t8ppo";
    std::filesystem::remove(nonfinite_checkpoint, error);
    {
        std::ofstream output(nonfinite_checkpoint, std::ios::binary | std::ios::trunc);
        output.write(mutated.data(), static_cast<std::streamsize>(mutated.size()));
    }
    t8::v2::GpuActorCritic restored(environments, small_config, 333);
    bool rejected_nonfinite = false;
    try {
        restored.load_checkpoint(nonfinite_checkpoint);
    } catch (const std::runtime_error&) {
        rejected_nonfinite = true;
    }
    check(rejected_nonfinite, "checkpoint load rejects a checksum-valid but non-finite payload");

    std::filesystem::remove(checkpoint, error);
    std::filesystem::remove(nonfinite_checkpoint, error);
}

// Byte offsets of the V4 checkpoint blocks (header 40, contract 160,
// normalizer 24, integrity 16).
constexpr std::size_t kV4HeaderBytes = 40;
constexpr std::size_t kV4ContractBytes = 160;
constexpr std::size_t kV4NormalizerBytes = 24;
constexpr std::size_t kV4PayloadOffset = kV4HeaderBytes + kV4ContractBytes + kV4NormalizerBytes + 16;

std::vector<char> read_bytes(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    return {(std::istreambuf_iterator<char>(input)), std::istreambuf_iterator<char>()};
}

void write_bytes(const std::filesystem::path& path, const std::vector<char>& bytes) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    output.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

std::uint64_t fnv1a(const char* bytes, std::size_t count) {
    std::uint64_t checksum = 14695981039346656037ULL;
    for (std::size_t index = 0; index < count; ++index) {
        checksum ^= static_cast<unsigned char>(bytes[index]);
        checksum *= 1099511628211ULL;
    }
    return checksum;
}

void test_orthogonal_initialization() {
    t8::v2::ActorCriticConfig config{};
    config.hidden_size = 16;
    t8::v2::GpuActorCritic policy(8, config, 4242);
    const auto checkpoint = std::filesystem::temp_directory_path() / "t8_v2_orthogonal.t8ppo";
    std::error_code error;
    std::filesystem::remove(checkpoint, error);
    policy.save_checkpoint(checkpoint);
    const auto raw = read_bytes(checkpoint);
    std::filesystem::remove(checkpoint, error);
    const int hidden = config.hidden_size;
    const int observations = config.observation_size;
    const int outputs = config.action_count + 1;
    const auto* weights_1 = reinterpret_cast<const float*>(raw.data() + kV4PayloadOffset);
    const auto* weights_2 = weights_1 + hidden * observations + hidden;
    const auto* weights_out = weights_2 + hidden * hidden + hidden;
    // Wide/square layers have orthonormal rows scaled by sqrt(2): W W^T = 2 I.
    const auto gram_error = [](const float* weights, int rows, int columns) {
        double worst = 0.0;
        for (int left = 0; left < rows; ++left) {
            for (int right = 0; right < rows; ++right) {
                double dot = 0.0;
                for (int column = 0; column < columns; ++column) {
                    dot += static_cast<double>(weights[left * columns + column]) *
                        weights[right * columns + column];
                }
                worst = std::max(worst, std::abs(dot - (left == right ? 2.0 : 0.0)));
            }
        }
        return worst;
    };
    check(gram_error(weights_1, hidden, observations) < 1e-4, "layer-1 rows are orthogonal with gain sqrt(2)");
    check(gram_error(weights_2, hidden, hidden) < 1e-4, "layer-2 rows are orthogonal with gain sqrt(2)");
    // Tall output layer: orthonormal columns before per-row gains of 0.01
    // (policy) and 1.0 (value), so undoing the gains restores W^T W = I.
    double worst_column = 0.0;
    float largest_policy_weight = 0.0F;
    for (int row = 0; row + 1 < outputs; ++row) {
        for (int column = 0; column < hidden; ++column) {
            largest_policy_weight = std::max(largest_policy_weight,
                                             std::abs(weights_out[row * hidden + column]));
        }
    }
    for (int left = 0; left < hidden; ++left) {
        for (int right = 0; right < hidden; ++right) {
            double dot = 0.0;
            for (int row = 0; row < outputs; ++row) {
                const double gain = row + 1 < outputs ? 0.01 : 1.0;
                dot += (weights_out[row * hidden + left] / gain) * (weights_out[row * hidden + right] / gain);
            }
            worst_column = std::max(worst_column, std::abs(dot - (left == right ? 1.0 : 0.0)));
        }
    }
    check(worst_column < 1e-3, "output layer is orthogonal before its per-row gains");
    check(largest_policy_weight <= 0.01F, "policy rows use the 0.01 output gain");
}

void test_observation_normalizer_and_v3_compatibility() {
    constexpr std::size_t environments = 64;
    constexpr std::size_t horizon = 4;
    constexpr std::size_t samples = environments * horizon;
    t8::v2::ActorCriticConfig config{};
    config.hidden_size = 16;
    config.observation_normalization = true;
    const int width = config.observation_size;
    t8::v2::GpuActorCritic policy(samples, config, 606);
    t8::v2::GpuRolloutBuffer rollout(environments, horizon, config);
    check(policy.observation_count() == 0, "normalizer starts empty");

    // Heterogeneous scales: feature f ~ 10 f + noise scaled by (f + 1).
    std::vector<float> host_observations(samples * width);
    for (std::size_t sample = 0; sample < samples; ++sample) {
        for (int feature = 0; feature < width; ++feature) {
            host_observations[sample * width + feature] = 10.0F * feature +
                static_cast<float>((sample * 7 + feature * 3) % 11) * (feature + 1);
        }
    }
    std::vector<std::uint8_t> masks(environments * config.action_count, 1);
    std::vector<float> rewards(environments, 0.5F);
    std::vector<std::uint8_t> flags(environments, 0);
    float* device_observations = nullptr;
    std::uint8_t* device_masks = nullptr;
    float* device_rewards = nullptr;
    std::uint8_t* device_flags = nullptr;
    cuda_check(cudaMalloc(&device_observations, sizeof(float) * host_observations.size()), "allocate observations");
    cuda_check(cudaMalloc(&device_masks, masks.size()), "allocate masks");
    cuda_check(cudaMalloc(&device_rewards, sizeof(float) * rewards.size()), "allocate rewards");
    cuda_check(cudaMalloc(&device_flags, flags.size()), "allocate flags");
    cuda_check(cudaMemcpy(device_observations, host_observations.data(),
                          sizeof(float) * host_observations.size(), cudaMemcpyHostToDevice), "upload observations");
    cuda_check(cudaMemcpy(device_masks, masks.data(), masks.size(), cudaMemcpyHostToDevice), "upload masks");
    cuda_check(cudaMemcpy(device_rewards, rewards.data(), sizeof(float) * rewards.size(),
                          cudaMemcpyHostToDevice), "upload rewards");
    cuda_check(cudaMemcpy(device_flags, flags.data(), flags.size(), cudaMemcpyHostToDevice), "upload flags");
    for (std::size_t step = 0; step < horizon; ++step) {
        const float* step_observations = device_observations + step * environments * width;
        const auto output = policy.forward(step_observations, device_masks, environments, 5, step, false);
        rollout.record_policy_device(step, step_observations, device_masks, output.actions,
                                     output.log_probabilities, output.values);
        rollout.record_outcome_device(step, device_rewards, device_flags, device_flags, output.values);
    }
    rollout.compute_gae(0.99F, 0.95F, true);
    t8::v2::PpoUpdateConfig update{};
    update.epochs = 1;
    update.minibatch_size = environments;
    update.target_kl = 0.0F;
    const auto metrics = policy.update_ppo(rollout.device_view(), update, 17);
    check(std::isfinite(metrics.policy_loss) && std::isfinite(metrics.value_loss),
          "PPO update with a normalizer stays finite");
    check(policy.observation_count() == samples, "update merges the whole rollout into the normalizer");

    const auto statistics = policy.download_observation_statistics();
    bool statistics_match = true;
    for (int feature = 0; feature < width; ++feature) {
        double sum = 0.0;
        for (std::size_t sample = 0; sample < samples; ++sample) sum += host_observations[sample * width + feature];
        const double mean = sum / samples;
        double squares = 0.0;
        for (std::size_t sample = 0; sample < samples; ++sample) {
            const double centered = host_observations[sample * width + feature] - mean;
            squares += centered * centered;
        }
        const double variance = squares / samples;
        statistics_match = statistics_match &&
            std::abs(statistics[feature] - mean) <= 1e-4 * std::max(1.0, std::abs(mean)) &&
            std::abs(statistics[width + feature] - variance) <= 1e-4 * std::max(1.0, variance);
    }
    check(statistics_match, "normalizer mean/variance match the rollout's population moments");

    const auto checkpoint = std::filesystem::temp_directory_path() / "t8_v2_normalizer.t8ppo";
    std::error_code error;
    std::filesystem::remove(checkpoint, error);
    policy.save_checkpoint(checkpoint);
    static_cast<void>(policy.forward(device_observations, device_masks, environments, 1, 1, true));
    const auto expected_actions = policy.download_actions(environments);
    const auto expected_values = policy.download_values(environments);

    t8::v2::ActorCriticConfig plain_config = config;
    plain_config.observation_normalization = false;
    t8::v2::GpuActorCritic restored(samples, plain_config, 707);
    restored.load_checkpoint(checkpoint);
    check(restored.config().observation_normalization, "loading adopts the checkpoint's normalizer");
    check(restored.observation_count() == samples, "loading restores the normalizer sample count");
    check(restored.download_observation_statistics() == statistics, "loading restores statistics exactly");
    static_cast<void>(restored.forward(device_observations, device_masks, environments, 1, 1, true));
    check(restored.download_actions(environments) == expected_actions &&
          restored.download_values(environments) == expected_values,
          "normalized checkpoint reproduces actions and values exactly");

    // Version 3 files predate the normalizer: they must still load, raw.
    const auto raw = read_bytes(checkpoint);
    std::uint64_t payload_bytes = 0;
    std::memcpy(&payload_bytes, raw.data() + kV4PayloadOffset - 16, sizeof(payload_bytes));
    const std::size_t v3_payload_bytes = payload_bytes - 2 * sizeof(float) * width;
    std::vector<char> v3(raw.begin(), raw.begin() + kV4HeaderBytes + kV4ContractBytes);
    const std::uint32_t version_3 = 3;
    std::memcpy(v3.data() + 8, &version_3, sizeof(version_3));
    const std::uint64_t v3_integrity[2] = {
        v3_payload_bytes, fnv1a(raw.data() + kV4PayloadOffset, v3_payload_bytes)};
    const auto* integrity_bytes = reinterpret_cast<const char*>(v3_integrity);
    v3.insert(v3.end(), integrity_bytes, integrity_bytes + sizeof(v3_integrity));
    v3.insert(v3.end(), raw.begin() + kV4PayloadOffset,
              raw.begin() + kV4PayloadOffset + static_cast<std::ptrdiff_t>(v3_payload_bytes));
    const auto v3_checkpoint = std::filesystem::temp_directory_path() / "t8_v2_normalizer_v3.t8ppo";
    write_bytes(v3_checkpoint, v3);
    t8::v2::GpuActorCritic legacy(samples, config, 808);
    legacy.load_checkpoint(v3_checkpoint);
    check(!legacy.config().observation_normalization && legacy.observation_count() == 0,
          "version 3 checkpoint loads with normalization disabled");

    // A malformed normalizer block is rejected even with a valid checksum.
    auto bad_block = raw;
    const std::uint32_t invalid_flag = 2;
    std::memcpy(bad_block.data() + kV4HeaderBytes + kV4ContractBytes, &invalid_flag, sizeof(invalid_flag));
    const auto bad_checkpoint = std::filesystem::temp_directory_path() / "t8_v2_normalizer_bad.t8ppo";
    write_bytes(bad_checkpoint, bad_block);
    bool rejected_block = false;
    try {
        legacy.load_checkpoint(bad_checkpoint);
    } catch (const std::runtime_error&) {
        rejected_block = true;
    }
    check(rejected_block, "checkpoint load rejects an invalid normalizer block");

    std::filesystem::remove(checkpoint, error);
    std::filesystem::remove(v3_checkpoint, error);
    std::filesystem::remove(bad_checkpoint, error);
    cudaFree(device_flags);
    cudaFree(device_rewards);
    cudaFree(device_masks);
    cudaFree(device_observations);
}

}  // namespace

int main() {
    test_policy_shapes_and_sampling();
    test_determinism_and_busy_mask();
    test_parametric_move_scorer_masks_variable_candidates();
    test_zero_copy_policy_to_simulator_chain();
    test_checkpoint_round_trip();
    test_scripted_opponent_mixture();
    test_profiled_gpu_opponents();
    test_gpu_temporal_matchup_encoder();
    test_held_out_opponent_and_side_router();
    test_character_specific_moves_execute_on_gpu();
    test_profile_assignments_change_only_for_done_lanes();
    test_opponent_character_swap_respects_lane_mask();
    test_temporal_preview_does_not_mutate_state();
    test_self_play_visual_router_uses_screen_observations_for_opponent();
    test_action_history_device_validation();
    test_actor_critic_and_router_reject_invalid_configuration();
    test_checkpoint_rejects_architecture_mismatch_and_nonfinite_payload();
    test_orthogonal_initialization();
    test_observation_normalizer_and_v3_compatibility();
    if (failures != 0) {
        std::cerr << failures << " policy assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "GPU actor-critic inference, masking, sampling, and simulator chaining passed\n";
    return EXIT_SUCCESS;
}
