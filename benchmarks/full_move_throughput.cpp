// Full-move training throughput and memory at the candidate upper bound.
//
//   t8_v2_full_move_benchmark [--envs N] [--horizon N] [--updates N] [--character yoshimitsu]
//                             [--minibatch N] [--epochs N] [--data-dir PATH]
//
// The learner plays the given character (Yoshimitsu has the most candidates,
// 354 of the 368-slot bound) against the scripted opponent with stand-in
// geometry. Reports simulator-only and complete rollout-plus-PPO decisions/s,
// engine frames/s, and GPU memory in use (cudaMemGetInfo) above the start.

#include "t8_v2/full_move_gpu.hpp"
#include "t8_v2/ppo.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    using namespace t8::v2;
    std::size_t environments = 32768, horizon = 64, updates = 5, minibatch = 131072;
    int epochs = 4;
    std::string character = "yoshimitsu";
    std::string matmul = "fp32";
    int hidden = 256;
    std::filesystem::path generated = std::filesystem::path(T8_V2_SOURCE_DIR) / "data" / "generated";
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string argument = argv[index];
        const std::string value = argv[index + 1];
        if (argument == "--envs") environments = std::stoull(value);
        else if (argument == "--horizon") horizon = std::stoull(value);
        else if (argument == "--updates") updates = std::stoull(value);
        else if (argument == "--minibatch") minibatch = std::stoull(value);
        else if (argument == "--epochs") epochs = std::stoi(value);
        else if (argument == "--character") character = value;
        else if (argument == "--matmul") matmul = value;
        else if (argument == "--hidden") hidden = std::stoi(value);
        else if (argument == "--data-dir") generated = value;
        else {
            std::cerr << "unknown argument " << argument << '\n';
            return EXIT_FAILURE;
        }
    }
    try {
        std::size_t free_start = 0, total = 0;
        cudaMemGetInfo(&free_start, &total);
        FullMoveRosterOptions options{};
        options.allow_provisional = true;
        std::vector<FullMoveCharacter> roster = {load_full_move_character(generated, character, options),
                                                 load_full_move_character(generated, "jun", options)};
        const int candidates = roster.front().candidate_count();
        GpuFullMoveSimulator simulator(roster, environments);
        std::vector<int> p1(environments), p2(environments), opponents(environments, 1);
        std::vector<std::uint8_t> sides(environments);
        for (std::size_t env = 0; env < environments; ++env) {
            sides[env] = env % 2 == 0;
            p1[env] = sides[env] ? 0 : 1;
            p2[env] = sides[env] ? 1 : 0;
        }
        simulator.set_matchups(p1, p2);
        simulator.set_learner_sides(sides);
        simulator.set_pending_opponents(opponents);
        simulator.reset_all(1, 0);

        ActorCriticConfig actor{};
        actor.observation_size = full_core::kFullMoveObservationSize;
        actor.action_count = kFullMoveMaxCandidates;
        actor.action_feature_size = kFullMoveFeatureSize;
        actor.universal_action_count = kFullMoveUniversalCount;
        actor.observation_contract = full_core::kFullMoveObservationContract;
        actor.action_contract = "full-move-v1";
        actor.matmul_precision = matmul;
        actor.hidden_size = hidden;
        actor.observation_normalization = true;
        GpuActorCritic learner(std::max(environments, minibatch), actor, 1);
        std::vector<float> features(static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize, 0.0F);
        std::copy(roster.front().candidate_features.begin(), roster.front().candidate_features.end(), features.begin());
        learner.set_action_features(features);
        GpuRolloutBuffer rollout(environments, horizon, actor);

        // Simulator only: the policy's first forward pass is reused for every step.
        const auto fixed = learner.forward(simulator.learner_observations(), simulator.learner_masks(), environments, 1, 0);
        simulator.synchronize();
        auto frames_before = simulator.frames_simulated();
        auto started = std::chrono::steady_clock::now();
        for (std::uint64_t step = 1; step <= horizon; ++step) {
            simulator.step(fixed.actions, nullptr, 1, step);
            simulator.reset_done(1, step);
        }
        simulator.synchronize();
        double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        const double sim_decisions = static_cast<double>(environments * horizon) / seconds;
        const double sim_frames = static_cast<double>(simulator.frames_simulated() - frames_before) / seconds;

        // Complete rollout plus PPO update, as the full-move trainer runs it.
        frames_before = simulator.frames_simulated();
        started = std::chrono::steady_clock::now();
        PpoUpdateConfig update_config{};
        update_config.epochs = epochs;
        update_config.minibatch_size = minibatch;
        for (std::uint64_t update = 1; update <= updates; ++update) {
            for (std::size_t step = 0; step < horizon; ++step) {
                const std::uint64_t counter = update * horizon + step;
                const auto output = learner.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                    environments, 2, counter);
                rollout.record_policy_device(step, simulator.learner_observations(), simulator.learner_masks(),
                                             output.actions, output.log_probabilities, output.values);
                simulator.step(output.actions, nullptr, 2, counter + 1);
                const auto next = learner.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                  environments, 3, counter, true);
                rollout.record_outcome_device(step, simulator.rewards(), simulator.terminated(), simulator.truncated(),
                                              next.values);
                simulator.reset_done(2, counter + 1);
            }
            static_cast<void>(rollout.normalize_rewards(0.99F));
            rollout.compute_gae(0.99F, 0.95F, true);
            static_cast<void>(learner.update_ppo(rollout.device_view(), update_config, update));
        }
        learner.synchronize();
        seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        std::size_t free_end = 0;
        cudaMemGetInfo(&free_end, &total);
        const double decisions = static_cast<double>(environments * horizon * updates) / seconds;
        const double frames = static_cast<double>(simulator.frames_simulated() - frames_before) / seconds;
        std::cout << "{\"character\":\"" << character << "\",\"candidates\":" << candidates
                  << ",\"candidate_stride\":" << kFullMoveMaxCandidates << ",\"environments\":" << environments
                  << ",\"horizon\":" << horizon << ",\"updates\":" << updates << ",\"minibatch\":" << minibatch
                  << ",\"epochs\":" << epochs
                  << ",\"simulator_decisions_per_second\":" << sim_decisions
                  << ",\"simulator_frames_per_second\":" << sim_frames
                  << ",\"training_decisions_per_second\":" << decisions
                  << ",\"training_frames_per_second\":" << frames
                  << ",\"seconds_per_update\":" << seconds / static_cast<double>(updates)
                  << ",\"gpu_memory_used_mib\":" << static_cast<double>(free_start - free_end) / (1024.0 * 1024.0)
                  << "}\n";
    } catch (const std::exception& error) {
        std::cerr << "full-move benchmark error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
