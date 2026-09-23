#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/screen_observation.hpp"
#include "t8_v2/temporal.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
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
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

constexpr std::size_t kVisual = t8::v2::kVisualObservationSize;

t8::v2::Config screen_config(float sigma_min, float sigma_max, float error_min, float error_max,
                             float health_sigma = 0.0F) {
    t8::v2::Config config{};
    config.screen_observations = true;
    config.screen_noise.position_sigma_min = sigma_min;
    config.screen_noise.position_sigma_max = sigma_max;
    config.screen_noise.event_error_min = error_min;
    config.screen_noise.event_error_max = error_max;
    config.screen_noise.health_sigma = health_sigma;
    return config;
}

// P2 is 7 frames into a move; the lanes differ only in which move (hidden).
std::vector<t8::v2::State> hidden_move_states(std::size_t lanes, int first_move, int second_move) {
    std::vector<t8::v2::State> states(lanes);
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        auto& state = states[lane];
        state.frame = 40;
        state.p1.x = -0.6;
        state.p2.x = 0.4;
        state.p2.move = lane % 2 == 0 ? first_move : second_move;
        state.p2.move_frame = 7;
    }
    return states;
}

void test_screen_observations_hide_move_identity() {
    // Jab (slot 0) and f2 (slot 2) are both in startup at frame 7 but differ in
    // range, so the original attack-likelihood feature tells them apart.
    constexpr std::size_t lanes = 2;
    t8::v2::GpuSimulatorBatch visual(lanes);
    visual.upload_states(hidden_move_states(lanes, 0, 2));
    const auto leaky = visual.download_visual_observations(1);
    check(leaky[12] != leaky[kVisual + 12], "baseline visual features leak the hidden move through attack likelihood");

    // Same states in two lanes at the same key would get different noise, so
    // compare one lane uploaded twice with each move instead.
    t8::v2::GpuSimulatorBatch screen(1, screen_config(0.3F, 0.3F, 0.2F, 0.2F, 0.01F));
    screen.upload_states(hidden_move_states(1, 0, 0));
    const auto with_jab = screen.download_visual_observations(1);
    screen.upload_states(hidden_move_states(1, 2, 2));
    const auto with_f2 = screen.download_visual_observations(1);
    check(with_jab == with_f2, "screen observations are identical when only the hidden move differs");
}

void test_noise_free_screen_features_equal_truth() {
    constexpr std::size_t lanes = 4;
    t8::v2::GpuSimulatorBatch screen(lanes, screen_config(0.0F, 0.0F, 0.0F, 0.0F));
    auto states = hidden_move_states(lanes, 0, 2);
    states[1].p2.move = -1;                  // idle: no activity, no attack cue
    states[2].p2.move_frame = 40;            // f2 recovering: activity, no attack cue
    states[3].p2.move = -1;
    states[3].p2.hitstun = 5;                // being hit: activity, no attack cue
    screen.upload_states(states);
    const auto view = screen.download_visual_observations(1);
    const auto at = [&](std::size_t lane, std::size_t feature) { return view[lane * kVisual + feature]; };
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        check(at(lane, 0) == 1.0F && at(lane, 1) == 1.0F, "noise-free screen health equals HUD truth");
        check(std::abs(at(lane, 2) + 0.6F) < 1e-6F && std::abs(at(lane, 3) - 0.4F) < 1e-6F,
              "noise-free screen positions equal truth");
        check(std::abs(at(lane, 4) - 1.0F) < 1e-6F, "screen distance is derived from measured positions");
    }
    check(at(0, 8) == 1.0F && at(0, 12) == 1.0F, "an attack in startup is visible activity and an attack cue");
    check(at(1, 8) == 0.0F && at(1, 12) == 0.0F, "an idle fighter shows neither");
    check(at(2, 8) == 1.0F && at(2, 12) == 0.0F, "recovery is activity without an attack cue");
    check(at(3, 8) == 1.0F && at(3, 12) == 0.0F, "hitstun is activity without an attack cue");
}

void test_noise_levels_match_configuration() {
    constexpr std::size_t lanes = 20000;
    constexpr float sigma = 0.3F;
    constexpr float error = 0.2F;
    t8::v2::GpuSimulatorBatch screen(lanes, screen_config(sigma, sigma, error, error));
    auto states = hidden_move_states(lanes, 0, 0);
    for (std::size_t lane = 0; lane < lanes; ++lane) states[lane].frame = static_cast<int>(lane % 997);
    screen.upload_states(states);
    const auto view = screen.download_visual_observations(1);
    double sum = 0.0;
    double squares = 0.0;
    std::size_t flips = 0;
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        const double position_error = view[lane * kVisual + 3] - 0.4;
        sum += position_error;
        squares += position_error * position_error;
        flips += view[lane * kVisual + 12] == 0.0F;  // truth is an attack cue in every lane
    }
    const double mean = sum / lanes;
    const double measured_sigma = std::sqrt(squares / lanes - mean * mean);
    const double flip_rate = static_cast<double>(flips) / lanes;
    check(std::abs(mean) < 0.01, "position noise is unbiased");
    check(std::abs(measured_sigma - sigma) < 0.01, "position noise has the configured standard deviation");
    check(std::abs(flip_rate - error) < 0.015, "detections flip at the configured error rate");

    bool rejected = false;
    try {
        t8::v2::GpuSimulatorBatch invalid(1, screen_config(0.5F, 0.2F, 0.0F, 0.0F));
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "screen noise rejects an inverted position range");
}

void test_screen_temporal_encoder_ignores_hidden_actions() {
    constexpr std::size_t lanes = 64;
    t8::v2::ScreenObservationNoise noise{};
    t8::v2::GpuTemporalMatchupEncoder encoder(lanes, kVisual, noise);
    t8::v2::GpuTemporalMatchupEncoder oracle(lanes, kVisual);
    check(encoder.screen_only() && !oracle.screen_only(), "encoder reports its observation mode");
    t8::v2::GpuScriptedOpponent opponent(lanes);
    t8::v2::GpuSimulatorBatch simulator(lanes, screen_config(0.2F, 0.8F, 0.05F, 0.25F));
    opponent.set_profiles(std::vector<t8::v2::OpponentProfileParameters>(1));
    opponent.set_profile_assignments(std::vector<std::uint32_t>(lanes, 0));
    const float* base = simulator.device_view().visual_observations_p1;

    std::int64_t* actions = nullptr;
    cuda_check(cudaMalloc(&actions, sizeof(std::int64_t) * lanes), "allocate actions");
    const auto encode_with = [&](t8::v2::GpuTemporalMatchupEncoder& target, std::int64_t action) {
        const std::vector<std::int64_t> host(lanes, action);
        cuda_check(cudaMemcpy(actions, host.data(), sizeof(std::int64_t) * lanes, cudaMemcpyHostToDevice),
                   "upload actions");
        std::vector<float> result;
        for (int step = 0; step < 3; ++step) {
            const float* output = target.encode(base, opponent.profiles_device(), opponent.profile_count(),
                                                opponent.profile_assignments_device(), actions, lanes);
            result.resize(lanes * target.observation_size());
            cuda_check(cudaMemcpy(result.data(), output, sizeof(float) * result.size(), cudaMemcpyDeviceToHost),
                       "download encoding");
        }
        return result;
    };
    const auto with_jab = encode_with(encoder, 18);
    encoder.reset();
    const auto with_f2 = encode_with(encoder, 20);
    check(with_jab == with_f2, "screen encoding does not depend on the opponent's hidden action");
    const auto oracle_jab = encode_with(oracle, 18);
    oracle.reset();
    const auto oracle_f2 = encode_with(oracle, 20);
    check(oracle_jab != oracle_f2, "the oracle encoding does depend on it (control)");

    const std::size_t width = encoder.observation_size();
    const std::size_t newest = kVisual + t8::v2::kCharacterEmbeddingSize + t8::v2::kOpponentArchetypeCount +
        (t8::v2::kTemporalHistoryLength - 1) * t8::v2::kTemporalFeaturesPerStep;
    bool uncertainty_matches = true;
    for (std::size_t lane = 0; lane < lanes; ++lane) {
        const float sigma = t8::v2::screen_normalized_position_sigma(t8::v2::screen_lane_position_sigma(noise, lane));
        const float error = t8::v2::screen_normalized_event_error(t8::v2::screen_lane_event_error(noise, lane));
        uncertainty_matches = uncertainty_matches &&
            std::abs(with_jab[lane * width + newest + 2] - sigma) < 1e-6F &&
            std::abs(with_jab[lane * width + newest + 3] - error) < 1e-6F;
    }
    check(uncertainty_matches, "uncertainty inputs equal each lane's noise levels");
    cudaFree(actions);

    bool rejected = false;
    try {
        t8::v2::GpuTemporalMatchupEncoder invalid(4, t8::v2::kObservationSize, noise);
    } catch (const std::invalid_argument&) {
        rejected = true;
    }
    check(rejected, "screen encoding rejects the privileged 19-feature base");
}

void test_lane_uncertainty_covers_configured_range() {
    t8::v2::ScreenObservationNoise noise{};
    float low = 1e9F;
    float high = -1e9F;
    for (std::uint64_t lane = 0; lane < 4096; ++lane) {
        const float sigma = t8::v2::screen_lane_position_sigma(noise, lane);
        low = std::min(low, sigma);
        high = std::max(high, sigma);
    }
    check(low >= noise.position_sigma_min && high <= noise.position_sigma_max, "lane sigma stays in range");
    check(low < 0.02F && high > 0.98F, "lanes span the configured sigma range");
}

}  // namespace

int main() {
    try {
        test_screen_observations_hide_move_identity();
        test_noise_free_screen_features_equal_truth();
        test_noise_levels_match_configuration();
        test_screen_temporal_encoder_ignores_hidden_actions();
        test_lane_uncertainty_covers_configured_range();
    } catch (const std::exception& error) {
        std::cerr << "screen observation test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " screen observation assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Screen-only observations hide move identity and match configured noise\n";
    return EXIT_SUCCESS;
}
