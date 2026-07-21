#include "t8_v2/gpu_sim.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

using t8::v2::Action;
using t8::v2::FighterRuntime;
using t8::v2::GpuSimulatorBatch;
using t8::v2::Simulator;
using t8::v2::State;

int failures = 0;

void check(bool condition, std::string_view message) {
    if (!condition && failures < 30) std::cerr << "FAIL: " << message << '\n';
    if (!condition) ++failures;
}

void near(double actual, double expected, std::string_view message, double tolerance = 4e-4) {
    check(std::abs(actual - expected) <= tolerance, message);
}

void compare_fighter(const FighterRuntime& gpu, const FighterRuntime& cpu) {
    near(gpu.health, cpu.health, "fighter health");
    near(gpu.x, cpu.x, "fighter x");
    near(gpu.y, cpu.y, "fighter y");
    check(gpu.guard == cpu.guard, "fighter guard");
    check(gpu.move == cpu.move, "fighter move");
    check(gpu.move_frame == cpu.move_frame, "fighter move frame");
    check(gpu.has_hit == cpu.has_hit, "fighter has-hit");
    check(gpu.hitstun == cpu.hitstun, "fighter hitstun");
    check(gpu.blockstun == cpu.blockstun, "fighter blockstun");
    check(gpu.airborne == cpu.airborne, "fighter airborne");
    check(gpu.throw_break_active == cpu.throw_break_active, "fighter throw-break timer");
    check(gpu.launches_taken == cpu.launches_taken, "fighter launches taken");
    check(gpu.whiffs == cpu.whiffs, "fighter whiffs");
}

void compare_state(const State& gpu, const State& cpu) {
    compare_fighter(gpu.p1, cpu.p1);
    compare_fighter(gpu.p2, cpu.p2);
    check(gpu.frame == cpu.frame, "state frame");
    check(gpu.stall_frames == cpu.stall_frames, "state stall frames");
    check(gpu.no_action_frames == cpu.no_action_frames, "state no-action frames");
    check(gpu.round_over == cpu.round_over, "state round-over");
    check(gpu.winner == cpu.winner, "state winner");
}

void test_many_lane_trace_parity() {
    constexpr std::size_t lanes = 128;
    GpuSimulatorBatch gpu(lanes);
    std::vector<Simulator> cpu;
    cpu.reserve(lanes);
    for (std::size_t lane = 0; lane < lanes; ++lane) cpu.emplace_back();
    std::vector<std::uint8_t> p1_actions(lanes);
    std::vector<std::uint8_t> p2_actions(lanes);
    std::vector<bool> done(lanes, false);

    for (int decision = 0; decision < 180; ++decision) {
        gpu.reset_done();
        for (std::size_t lane = 0; lane < lanes; ++lane) {
            if (done[lane]) cpu[lane].reset(static_cast<std::uint64_t>(decision * lanes + lane));
            // Deterministic, broad action coverage without CPU-side RNG overhead.
            p1_actions[lane] = static_cast<std::uint8_t>((lane * 7 + decision * 5) % t8::v2::kActionCount);
            p2_actions[lane] = static_cast<std::uint8_t>((lane * 11 + decision * 3 + 1) % t8::v2::kActionCount);
        }
        gpu.step_host(p1_actions, p2_actions);
        const auto gpu_states = gpu.download_states();
        const auto gpu_rewards_p1 = gpu.download_rewards(1);
        const auto gpu_rewards_p2 = gpu.download_rewards(2);
        const auto gpu_obs_p1 = gpu.download_observations(1);
        const auto gpu_obs_p2 = gpu.download_observations(2);
        const auto gpu_visual_p1 = gpu.download_visual_observations(1);
        const auto gpu_visual_p2 = gpu.download_visual_observations(2);
        const auto gpu_masks_p1 = gpu.download_action_masks(1);
        const auto gpu_masks_p2 = gpu.download_action_masks(2);
        const auto gpu_done = gpu.download_terminated();

        for (std::size_t lane = 0; lane < lanes; ++lane) {
            const State previous = cpu[lane].state();
            const auto result = cpu[lane].step(static_cast<Action>(p1_actions[lane]),
                                               static_cast<Action>(p2_actions[lane]));
            compare_state(gpu_states[lane], result.state);
            near(gpu_rewards_p1[lane], result.reward_p1, "P1 reward", 2e-3);
            near(gpu_rewards_p2[lane], result.reward_p2, "P2 reward", 2e-3);
            check((gpu_done[lane] != 0) == result.terminated, "terminated flag");
            const auto cpu_obs_p1 = cpu[lane].observation(1);
            const auto cpu_obs_p2 = cpu[lane].observation(2);
            const auto cpu_visual_p1 = cpu[lane].visual_observation(1, &previous);
            const auto cpu_visual_p2 = cpu[lane].visual_observation(2, &previous);
            const auto cpu_mask_p1 = cpu[lane].legal_action_mask(1);
            const auto cpu_mask_p2 = cpu[lane].legal_action_mask(2);
            for (std::size_t feature = 0; feature < t8::v2::kObservationSize; ++feature) {
                near(gpu_obs_p1[lane * t8::v2::kObservationSize + feature], cpu_obs_p1[feature],
                     "P1 observation", 5e-4);
                near(gpu_obs_p2[lane * t8::v2::kObservationSize + feature], cpu_obs_p2[feature],
                     "P2 observation", 5e-4);
            }
            for (std::size_t feature = 0; feature < t8::v2::kVisualObservationSize; ++feature) {
                near(gpu_visual_p1[lane * t8::v2::kVisualObservationSize + feature],
                     cpu_visual_p1[feature], "P1 visual observation", 5e-4);
                near(gpu_visual_p2[lane * t8::v2::kVisualObservationSize + feature],
                     cpu_visual_p2[feature], "P2 visual observation", 5e-4);
            }
            for (std::size_t action = 0; action < t8::v2::kActionCount; ++action) {
                check((gpu_masks_p1[lane * t8::v2::kActionCount + action] != 0) == cpu_mask_p1[action],
                      "P1 action mask");
                check((gpu_masks_p2[lane * t8::v2::kActionCount + action] != 0) == cpu_mask_p2[action],
                      "P2 action mask");
            }
            done[lane] = result.terminated;
        }
        if (failures > 200) break;
    }
}

void test_uploaded_timeout_fixture() {
    GpuSimulatorBatch gpu(1);
    Simulator cpu;
    State state{};
    state.p1.health = 180.0;
    state.p2.health = 170.0;
    state.p1.x = 0.0;
    state.p2.x = 0.82;
    state.frame = cpu.config().max_frames - 1;
    cpu.set_state(state);
    gpu.upload_states(std::span<const State>(&state, 1));
    const std::vector<std::uint8_t> neutral(1, 0);
    gpu.step_host(neutral, neutral);
    const auto expected = cpu.step(Action::Neutral, Action::Neutral);
    compare_state(gpu.download_states().front(), expected.state);
    near(gpu.download_rewards(1).front(), expected.reward_p1, "timeout P1 reward", 2e-3);
    near(gpu.download_rewards(2).front(), expected.reward_p2, "timeout P2 reward", 2e-3);
    near(gpu.download_sparse_rewards(1).front(), 1.0, "timeout sparse P1 win reward");
    near(gpu.download_sparse_rewards(2).front(), -1.0, "timeout sparse P2 loss reward");
    check(gpu.download_winners().front() == 1, "GPU winner tensor exposes timeout winner");
}

void test_upload_refreshes_derived_outputs() {
    GpuSimulatorBatch gpu(2);
    Simulator first;
    Simulator second;
    State states[2]{};
    states[0].p1.health = 143.0;
    states[0].p2.health = 88.0;
    states[0].p1.x = -0.61;
    states[0].p2.x = 0.14;
    states[0].p1.hitstun = 4;
    states[0].p2.blockstun = 6;
    states[0].frame = 321;
    states[1].p1.health = 9.0;
    states[1].p2.health = 199.0;
    states[1].p1.x = 0.73;
    states[1].p2.x = -0.22;
    states[1].p1.airborne = true;
    states[1].p2.throw_break_active = 3;
    states[1].frame = 777;
    first.set_state(states[0]);
    second.set_state(states[1]);

    gpu.upload_states(states);
    const auto p1_observations = gpu.download_observations(1);
    const auto p2_observations = gpu.download_observations(2);
    const auto visual_p1_observations = gpu.download_visual_observations(1);
    const auto visual_p2_observations = gpu.download_visual_observations(2);
    const auto p1_masks = gpu.download_action_masks(1);
    const auto p2_masks = gpu.download_action_masks(2);
    const Simulator* oracles[] = {&first, &second};
    for (std::size_t lane = 0; lane < 2; ++lane) {
        const auto expected_p1 = oracles[lane]->observation(1);
        const auto expected_p2 = oracles[lane]->observation(2);
        const auto expected_visual_p1 = oracles[lane]->visual_observation(1);
        const auto expected_visual_p2 = oracles[lane]->visual_observation(2);
        const auto expected_p1_mask = oracles[lane]->legal_action_mask(1);
        const auto expected_p2_mask = oracles[lane]->legal_action_mask(2);
        for (std::size_t feature = 0; feature < t8::v2::kObservationSize; ++feature) {
            near(p1_observations[lane * t8::v2::kObservationSize + feature], expected_p1[feature],
                 "uploaded P1 observation refresh", 5e-4);
            near(p2_observations[lane * t8::v2::kObservationSize + feature], expected_p2[feature],
                 "uploaded P2 observation refresh", 5e-4);
        }
        for (std::size_t feature = 0; feature < t8::v2::kVisualObservationSize; ++feature) {
            near(visual_p1_observations[lane * t8::v2::kVisualObservationSize + feature],
                 expected_visual_p1[feature], "uploaded P1 visual observation refresh", 5e-4);
            near(visual_p2_observations[lane * t8::v2::kVisualObservationSize + feature],
                 expected_visual_p2[feature], "uploaded P2 visual observation refresh", 5e-4);
        }
        for (std::size_t action = 0; action < t8::v2::kActionCount; ++action) {
            check((p1_masks[lane * t8::v2::kActionCount + action] != 0) == expected_p1_mask[action],
                  "uploaded P1 mask refresh");
            check((p2_masks[lane * t8::v2::kActionCount + action] != 0) == expected_p2_mask[action],
                  "uploaded P2 mask refresh");
        }
    }
}

void test_fair_timeout_draw_is_absorbing_and_summarized() {
    t8::v2::Config config{};
    config.timeout_ties_are_draws = true;
    GpuSimulatorBatch gpu(1, config);
    Simulator cpu(config);
    State state{};
    state.p1.health = 180.0;
    state.p2.health = 180.0;
    state.p1.x = 0.0;
    state.p2.x = 0.82;
    state.frame = config.max_frames - 1;
    cpu.set_state(state);
    gpu.upload_states(std::span<const State>(&state, 1));
    const std::vector<std::uint8_t> neutral(1, 0);

    const auto expected = cpu.step(Action::Neutral, Action::Neutral);
    gpu.step_host(neutral, neutral);
    const auto terminal = gpu.download_states().front();
    compare_state(terminal, expected.state);
    check(terminal.round_over && terminal.winner == 0, "fair equal-health timeout is a draw");
    near(gpu.download_sparse_rewards(1).front(), 0.0, "draw has zero sparse P1 reward");
    near(gpu.download_sparse_rewards(2).front(), 0.0, "draw has zero sparse P2 reward");

    const auto summary = gpu.summarize_episodes(1);
    check(summary.episodes == 1 && summary.draws == 1, "GPU summary counts timeout draw");
    check(summary.timeouts == 1 && summary.total_frames == static_cast<std::uint64_t>(config.max_frames),
          "GPU summary records timeout and frame count");

    gpu.step_host(neutral, neutral);
    const auto after_repeat = gpu.download_states().front();
    check(after_repeat.frame == terminal.frame && after_repeat.winner == terminal.winner,
          "terminal GPU state is absorbing until reset");
    const auto cpu_repeat = cpu.step(Action::Neutral, Action::Neutral);
    check(cpu_repeat.state.frame == expected.state.frame && cpu_repeat.state.winner == 0,
          "terminal CPU oracle state is absorbing until reset");
}

void test_seeded_randomized_resets() {
    t8::v2::Config config{};
    config.randomize_initial_positions = true;
    GpuSimulatorBatch gpu(64, config);
    gpu.reset_seeded(424242);
    const auto first = gpu.download_states();
    gpu.reset_seeded(424242);
    const auto repeated = gpu.download_states();
    gpu.reset_seeded(424243);
    const auto changed = gpu.download_states();
    bool saw_lane_diversity = false;
    bool saw_seed_difference = false;
    for (std::size_t lane = 0; lane < first.size(); ++lane) {
        near(repeated[lane].p1.x, first[lane].p1.x, "same reset seed reproduces P1 position");
        near(repeated[lane].p2.x, first[lane].p2.x, "same reset seed reproduces P2 position");
        const double distance = first[lane].distance();
        check(distance >= config.initial_distance_min - 1e-5 &&
              distance <= config.initial_distance_max + 1e-5,
              "randomized reset distance stays in configured range");
        check(first[lane].p1.x > -config.stage_half_width &&
              first[lane].p2.x < config.stage_half_width && first[lane].p1.x < first[lane].p2.x,
              "randomized reset positions stay ordered inside stage");
        if (lane > 0 && std::abs(first[lane].p1.x - first[0].p1.x) > 1e-5) saw_lane_diversity = true;
        if (std::abs(first[lane].p1.x - changed[lane].p1.x) > 1e-5) saw_seed_difference = true;
    }
    check(saw_lane_diversity, "seeded GPU reset diversifies lanes");
    check(saw_seed_difference, "different reset seeds change starts");

    Simulator cpu(config);
    cpu.reset(424242);
    near(first[0].p1.x, cpu.state().p1.x, "CPU/GPU randomized reset P1 parity", 2e-5);
    near(first[0].p2.x, cpu.state().p2.x, "CPU/GPU randomized reset P2 parity", 2e-5);
}

}  // namespace

int main() {
    check(t8::v2::cuda_device_count() > 0, "CUDA device is available");
    if (failures == 0) {
        test_many_lane_trace_parity();
        test_uploaded_timeout_fixture();
        test_upload_refreshes_derived_outputs();
        test_fair_timeout_draw_is_absorbing_and_summarized();
        test_seeded_randomized_resets();
    }
    if (failures != 0) {
        std::cerr << failures << " GPU parity assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "Full CUDA step, reward, termination, observation, and mask parity passed\n";
    return EXIT_SUCCESS;
}
