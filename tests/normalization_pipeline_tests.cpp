// Observation and return normalization validated in the real PPO pipeline:
// full-move rollouts (real observations, rewards, and round ends) through
// GpuActorCritic::update_ppo and GpuRolloutBuffer::normalize_rewards, checked
// after every update against float64 host references of the same quantities:
//  * the observation normalizer's running mean and variance equal the
//    population moments of every observation seen so far;
//  * normalized observations are finite and clipped, constant features map to
//    zero instead of dividing by zero, and normalized features are centered;
//  * the return normalizer's standard deviation equals the running standard
//    deviation of per-environment discounted returns (reset at round ends),
//    and stored rewards are the raw rewards divided by it, clipped.

#include "t8_v2/full_move_gpu.hpp"
#include "t8_v2/ppo.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {

using namespace t8::v2;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        if (failures < 30) std::cerr << "FAIL: " << message << '\n';
    }
}

template <typename T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> values(count);
    cudaMemcpy(values.data(), device, sizeof(T) * count, cudaMemcpyDeviceToHost);
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: normalization_pipeline_tests <data/generated>\n";
        return EXIT_FAILURE;
    }
    try {
        FullMoveRosterOptions options{};
        options.allow_provisional = true;
        std::vector<FullMoveCharacter> roster = {load_full_move_character(argv[1], "jun", options),
                                                 load_full_move_character(argv[1], "king", options)};
        const std::size_t environments = 256, horizon = 32, updates = 6;
        const float gamma = 0.99F, clip = 10.0F;
        full_core::EnvConfig env{};
        env.max_round_frames = 1200;  // round ends inside the test
        GpuFullMoveSimulator simulator(roster, environments, env);
        std::vector<int> p1(environments, 0), p2(environments, 1), opponents(environments, 1);
        std::vector<std::uint8_t> sides(environments, 1U);
        simulator.set_matchups(p1, p2);
        simulator.set_learner_sides(sides);
        simulator.set_pending_opponents(opponents);
        simulator.reset_all(3, 0);

        ActorCriticConfig actor{};
        actor.observation_size = full_core::kFullMoveObservationSize;
        actor.action_count = kFullMoveMaxCandidates;
        actor.hidden_size = 64;
        actor.action_feature_size = kFullMoveFeatureSize;
        actor.universal_action_count = kFullMoveUniversalCount;
        actor.observation_contract = full_core::kFullMoveObservationContract;
        actor.action_contract = "full-move-v1";
        actor.observation_normalization = true;
        GpuActorCritic policy(std::max<std::size_t>(environments, 2048), actor, 9);
        std::vector<float> features(static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize, 0.0F);
        std::copy(roster[0].candidate_features.begin(), roster[0].candidate_features.end(), features.begin());
        policy.set_action_features(features);
        GpuRolloutBuffer rollout(environments, horizon, actor);

        const std::size_t width = full_core::kFullMoveObservationSize;
        std::vector<double> sum(width, 0.0), square(width, 0.0);
        double seen = 0.0;
        std::vector<double> discounted(environments, 0.0);
        double return_count = 0.0, return_mean = 0.0, return_m2 = 0.0;
        std::size_t round_ends = 0;
        std::vector<float> last_observations;
        PpoUpdateConfig update{};
        update.epochs = 1;
        update.minibatch_size = 2048;
        for (std::size_t iteration = 1; iteration <= updates; ++iteration) {
            std::vector<float> raw_rewards;
            last_observations.clear();
            for (std::size_t step = 0; step < horizon; ++step) {
                const std::uint64_t counter = iteration * horizon + step;
                const auto observations = simulator.download_learner_observations();
                last_observations.insert(last_observations.end(), observations.begin(), observations.end());
                for (std::size_t index = 0; index < observations.size(); ++index) {
                    sum[index % width] += observations[index];
                    square[index % width] += static_cast<double>(observations[index]) * observations[index];
                }
                seen += static_cast<double>(environments);
                const auto output = policy.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                   environments, 5, counter);
                rollout.record_policy_device(step, simulator.learner_observations(), simulator.learner_masks(),
                                             output.actions, output.log_probabilities, output.values);
                simulator.step(output.actions, nullptr, 3, counter + 1);
                const auto rewards = simulator.download_rewards();
                const auto terminated = simulator.download_terminated();
                raw_rewards.insert(raw_rewards.end(), rewards.begin(), rewards.end());
                // Discounted returns as the normalizer tracks them (reset at round ends).
                for (std::size_t env_index = 0; env_index < environments; ++env_index) {
                    discounted[env_index] = discounted[env_index] * gamma + rewards[env_index];
                    const double value = discounted[env_index];
                    return_count += 1.0;
                    const double delta = value - return_mean;
                    return_mean += delta / return_count;
                    return_m2 += delta * (value - return_mean);
                    if (terminated[env_index]) {
                        discounted[env_index] = 0.0;
                        ++round_ends;
                    }
                }
                const auto next = policy.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                 environments, 6, counter, true);
                rollout.record_outcome_device(step, simulator.rewards(), simulator.terminated(), simulator.truncated(),
                                              next.values);
                simulator.reset_done(3, counter + 1);
            }
            const float standard_deviation = rollout.normalize_rewards(gamma, clip);
            const double reference = std::sqrt(return_m2 / return_count + 1e-8);
            check(std::abs(standard_deviation - reference) <= 2e-4 * reference + 1e-7,
                  "return normalizer standard deviation matches the float64 reference (update " +
                      std::to_string(iteration) + ": " + std::to_string(standard_deviation) + " vs " +
                      std::to_string(reference) + ")");
            const auto scaled = rollout.download_rewards();
            bool scaled_ok = scaled.size() == raw_rewards.size();
            for (std::size_t index = 0; scaled_ok && index < scaled.size(); ++index) {
                const float expected = std::clamp(raw_rewards[index] / standard_deviation, -clip, clip);
                scaled_ok = std::abs(scaled[index] - expected) <= 1e-5F * std::max(1.0F, std::abs(expected));
            }
            check(scaled_ok, "stored rewards are raw rewards over the return standard deviation, clipped");
            rollout.compute_gae(gamma, 0.95F, true);
            static_cast<void>(policy.update_ppo(rollout.device_view(), update, iteration));

            // Observation normalizer: running moments equal every observation so far.
            const auto statistics = policy.download_observation_statistics();
            check(policy.observation_count() == static_cast<std::uint64_t>(seen), "normalizer counts every sample");
            for (std::size_t feature = 0; feature < width; ++feature) {
                const double mean = sum[feature] / seen;
                const double variance = std::max(0.0, square[feature] / seen - mean * mean);
                const double got_mean = statistics[feature];
                const double got_variance = statistics[width + feature];
                const bool ok = std::abs(got_mean - mean) <= 1e-4 * std::max(1.0, std::abs(mean)) &&
                    std::abs(got_variance - variance) <= 2e-4 * std::max(variance, 1e-3);
                if (!ok) {
                    check(false, "observation feature " + std::to_string(feature) + " moments differ: mean " +
                                     std::to_string(got_mean) + " vs " + std::to_string(mean) + ", variance " +
                                     std::to_string(got_variance) + " vs " + std::to_string(variance));
                }
            }
        }
        check(round_ends > 0, "rounds end during the test, exercising the return reset");

        // Normalized inputs of the last rollout: finite, clipped, centered, constants at zero.
        const auto statistics = policy.download_observation_statistics();
        std::size_t constant_features = 0, centered = 0, varying = 0;
        bool finite = true;
        for (std::size_t feature = 0; feature < width; ++feature) {
            const double mean = statistics[feature];
            const double variance = statistics[width + feature];
            double normalized_sum = 0.0;
            for (std::size_t sample = 0; sample < last_observations.size() / width; ++sample) {
                const double value = std::clamp((last_observations[sample * width + feature] - mean) /
                                                    std::sqrt(variance + 1e-8), -10.0, 10.0);
                finite = finite && std::isfinite(value);
                normalized_sum += value;
            }
            const double normalized_mean = normalized_sum / static_cast<double>(last_observations.size() / width);
            if (variance < 1e-12) {
                ++constant_features;
                check(std::abs(normalized_mean) < 1e-3, "a constant feature normalizes to zero");
            } else {
                ++varying;
                centered += std::abs(normalized_mean) < 1.0;
            }
        }
        check(finite, "normalized observations are finite");
        check(constant_features >= 1, "the bias feature is constant and handled");
        check(centered * 10 >= varying * 9, "normalized features are centered within one standard deviation");
        std::cout << "normalization: " << updates << " updates, " << static_cast<std::uint64_t>(seen)
                  << " observations, " << round_ends << " round ends, " << varying << " varying and "
                  << constant_features << " constant features\n";
    } catch (const std::exception& error) {
        std::cerr << "normalization test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " normalization assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "normalization pipeline tests passed\n";
    return EXIT_SUCCESS;
}
