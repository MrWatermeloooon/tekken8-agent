// Several candidate tables in one parametric policy (one per character):
//  * forward with per-sample table indices equals separate single-table
//    networks with those tables (same weights);
//  * a PPO update on samples that all use table 1 of a two-table network leaves
//    the same weights as the same update on a single-table network holding
//    that table (the gradient reads the sample's own table);
//  * an out-of-range table index makes the sample non-finite instead of
//    scoring another character's moves.

#include "t8_v2/ppo.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace {

using namespace t8::v2;

int failures = 0;
void check(bool condition, const std::string& message) {
    if (!condition && ++failures < 20) std::cerr << "FAIL: " << message << '\n';
}

template <typename T>
T* upload(const std::vector<T>& values) {
    T* device = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&device), sizeof(T) * values.size());
    cudaMemcpy(device, values.data(), sizeof(T) * values.size(), cudaMemcpyHostToDevice);
    return device;
}

template <typename T>
std::vector<T> download(const T* device, std::size_t count) {
    std::vector<T> values(count);
    cudaMemcpy(values.data(), device, sizeof(T) * count, cudaMemcpyDeviceToHost);
    return values;
}

}  // namespace

int main() {
    constexpr int kObservations = 8, kActions = 12, kFeatures = 6, kBatch = 64;
    ActorCriticConfig config{};
    config.observation_size = kObservations;
    config.action_count = kActions;
    config.hidden_size = 16;
    config.action_feature_size = kFeatures;
    config.universal_action_count = 2;
    std::mt19937 random(7);
    std::normal_distribution<float> normal;
    std::vector<float> table_0(kActions * kFeatures), table_1(kActions * kFeatures);
    for (auto& value : table_0) value = normal(random);
    for (auto& value : table_1) value = normal(random);
    std::vector<float> both(table_0);
    both.insert(both.end(), table_1.begin(), table_1.end());
    std::vector<float> observations(kBatch * kObservations);
    for (auto& value : observations) value = normal(random);
    std::vector<std::uint8_t> masks(kBatch * kActions, 1U);
    std::vector<std::int32_t> mixed(kBatch), ones(kBatch, 1), invalid(kBatch, 0);
    for (int lane = 0; lane < kBatch; ++lane) mixed[lane] = lane % 2;
    invalid[3] = 2;
    float* device_observations = upload(observations);
    std::uint8_t* device_masks = upload(masks);
    std::int32_t* device_mixed = upload(mixed);
    std::int32_t* device_ones = upload(ones);
    std::int32_t* device_invalid = upload(invalid);

    GpuActorCritic multi(kBatch, config, 11), single_0(kBatch, config, 11), single_1(kBatch, config, 11);
    multi.set_action_feature_tables(both, 2);
    single_0.set_action_features(table_0);
    single_1.set_action_features(table_1);
    check(multi.action_feature_tables() == 2, "two tables installed");
    const auto multi_logits = download(multi.forward(device_observations, device_masks, kBatch, 1, 1, true, nullptr,
                                                     device_mixed).logits, kBatch * kActions);
    const auto logits_0 = download(single_0.forward(device_observations, device_masks, kBatch, 1, 1, true).logits,
                                   kBatch * kActions);
    const auto logits_1 = download(single_1.forward(device_observations, device_masks, kBatch, 1, 1, true).logits,
                                   kBatch * kActions);
    bool forward_match = true;
    for (int lane = 0; lane < kBatch; ++lane) {
        const auto& expected = lane % 2 == 0 ? logits_0 : logits_1;
        for (int action = 0; action < kActions; ++action) {
            forward_match = forward_match && multi_logits[lane * kActions + action] == expected[lane * kActions + action];
        }
    }
    check(forward_match, "per-sample tables score with each sample's own table");

    const auto bad = download(multi.forward(device_observations, device_masks, kBatch, 1, 1, true, nullptr,
                                            device_invalid).log_probabilities, kBatch);
    check(!std::isfinite(bad[3]) && std::isfinite(bad[2]), "an out-of-range table index is not silently replaced");

    // Same rollout, all samples on table 1: the two-table and single-table updates must agree.
    constexpr std::size_t kHorizon = 4;
    GpuRolloutBuffer rollout_multi(kBatch, kHorizon, config), rollout_single(kBatch, kHorizon, config);
    std::vector<float> rewards(kBatch);
    std::vector<std::uint8_t> flags(kBatch, 0U);
    for (std::size_t step = 0; step < kHorizon; ++step) {
        for (auto& value : rewards) value = normal(random);
        float* device_rewards = upload(rewards);
        std::uint8_t* device_flags = upload(flags);
        const auto a = multi.forward(device_observations, device_masks, kBatch, 3, step, false, nullptr, device_ones);
        rollout_multi.record_policy_device(step, device_observations, device_masks, a.actions, a.log_probabilities,
                                           a.values, nullptr, device_ones);
        rollout_multi.record_outcome_device(step, device_rewards, device_flags, device_flags, a.values);
        const auto b = single_1.forward(device_observations, device_masks, kBatch, 3, step, false);
        rollout_single.record_policy_device(step, device_observations, device_masks, b.actions, b.log_probabilities,
                                            b.values);
        rollout_single.record_outcome_device(step, device_rewards, device_flags, device_flags, b.values);
        cudaFree(device_rewards);
        cudaFree(device_flags);
    }
    rollout_multi.compute_gae();
    rollout_single.compute_gae();
    PpoUpdateConfig update{};
    update.epochs = 2;
    update.minibatch_size = 64;
    const auto multi_metrics = multi.update_ppo(rollout_multi.device_view(), update, 5);
    const auto single_metrics = single_1.update_ppo(rollout_single.device_view(), update, 5);
    // Loss metrics are atomic sums (order-dependent in the last bits); the weights below are exact.
    const auto close = [](float a, float b) { return std::abs(a - b) <= 1e-5F * std::max(1.0F, std::abs(b)); };
    check(close(multi_metrics.policy_loss, single_metrics.policy_loss) &&
              close(multi_metrics.value_loss, single_metrics.value_loss),
          "the two-table update reads table 1 like a single-table network");
    const auto after_multi = download(multi.forward(device_observations, device_masks, kBatch, 1, 9, true, nullptr,
                                                    device_ones).logits, kBatch * kActions);
    const auto after_single = download(single_1.forward(device_observations, device_masks, kBatch, 1, 9, true).logits,
                                       kBatch * kActions);
    check(after_multi == after_single, "weights after the update are identical");

    cudaFree(device_observations);
    cudaFree(device_masks);
    cudaFree(device_mixed);
    cudaFree(device_ones);
    cudaFree(device_invalid);
    if (failures != 0) {
        std::cerr << failures << " feature-table assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "per-sample candidate tables passed\n";
    return EXIT_SUCCESS;
}
