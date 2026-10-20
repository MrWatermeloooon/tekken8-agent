// Full-roster full-move smoke test: every sourced fighter plays every other on
// both sides through the CUDA simulator, with random legal candidates for both
// (stand-in geometry). Every observation and reward must stay finite, every
// state physically sane (health, positions, frame counters), masks must always
// allow neutral, and rounds must keep ending and resetting. Run it under
// compute-sanitizer for invalid memory access (.github/workflows/gpu.yml).

#include "t8_v2/full_move_gpu.hpp"
#include "t8_v2/roster.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    using namespace t8::v2;
    if (argc < 2) {
        std::cerr << "usage: full_roster_smoke <data/generated> [steps]\n";
        return EXIT_FAILURE;
    }
    const int steps = argc > 2 ? std::atoi(argv[2]) : 150;
    int failures = 0;
    const auto check = [&](bool condition, const std::string& message) {
        if (!condition && ++failures < 20) std::cerr << "FAIL: " << message << '\n';
    };
    try {
        FullMoveRosterOptions options{};
        options.allow_provisional = true;
        std::vector<FullMoveCharacter> roster;
        for (const auto slug : kRosterCharacterSlugs) {
            auto character = load_full_move_character(argv[1], std::string(slug), options);
            if (character.bound_moves > 0) roster.push_back(std::move(character));
        }
        const std::size_t characters = roster.size();
        const std::size_t count = characters * characters * 2;  // every ordered pairing, learner on both sides
        full_core::EnvConfig config{};
        config.max_round_frames = 1800;
        GpuFullMoveSimulator simulator(roster, count, config);
        std::vector<int> p1(count), p2(count), opponents(count), scenarios(count);
        std::vector<std::uint8_t> sides(count);
        for (std::size_t env = 0; env < count; ++env) {
            const std::size_t pair = env / 2;
            const int learner = static_cast<int>(pair / characters);
            opponents[env] = static_cast<int>(pair % characters);
            sides[env] = env % 2 == 0;
            p1[env] = sides[env] ? learner : opponents[env];
            p2[env] = sides[env] ? opponents[env] : learner;
            scenarios[env] = static_cast<int>(env % full_core::kScenarioCount);
        }
        simulator.set_matchups(p1, p2);
        simulator.set_learner_sides(sides);
        simulator.set_pending_opponents(opponents, {}, scenarios);
        simulator.reset_all(11, 0);
        std::int64_t* actions = nullptr;
        std::int64_t* opponent_actions = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&actions), sizeof(std::int64_t) * count);
        cudaMalloc(reinterpret_cast<void**>(&opponent_actions), sizeof(std::int64_t) * count);
        std::uint64_t random = 0x2545F4914F6CDD1DULL;
        const auto next = [&]() {
            random ^= random << 13U;
            random ^= random >> 7U;
            random ^= random << 17U;
            return random;
        };
        std::size_t rounds = 0;
        for (int step = 1; step <= steps; ++step) {
            const auto masks = simulator.download_learner_masks();
            const auto observations = simulator.download_learner_observations();
            std::vector<std::int64_t> chosen(count), chosen_opponent(count);
            for (std::size_t env = 0; env < count; ++env) {
                const std::uint8_t* mask = masks.data() + env * kFullMoveMaxCandidates;
                check(mask[0] != 0, "neutral is always legal");
                std::vector<int> legal;
                for (int candidate = 0; candidate < kFullMoveMaxCandidates; ++candidate) {
                    if (mask[candidate]) legal.push_back(candidate);
                }
                chosen[env] = legal[next() % legal.size()];
            }
            for (const float value : observations) check(std::isfinite(value), "observations are finite");
            // Half the steps also drive the opponent as a policy with random legal candidates.
            const bool policy_opponent = step % 2 == 0;
            if (policy_opponent) {
                std::vector<std::uint8_t> opponent_masks(count * kFullMoveMaxCandidates);
                cudaMemcpy(opponent_masks.data(), simulator.opponent_masks(), opponent_masks.size(), cudaMemcpyDeviceToHost);
                for (std::size_t env = 0; env < count; ++env) {
                    std::vector<int> legal;
                    for (int candidate = 0; candidate < kFullMoveMaxCandidates; ++candidate) {
                        if (opponent_masks[env * kFullMoveMaxCandidates + static_cast<std::size_t>(candidate)]) {
                            legal.push_back(candidate);
                        }
                    }
                    chosen_opponent[env] = legal.empty() ? 0 : legal[next() % legal.size()];
                }
                cudaMemcpy(opponent_actions, chosen_opponent.data(), sizeof(std::int64_t) * count, cudaMemcpyHostToDevice);
            }
            cudaMemcpy(actions, chosen.data(), sizeof(std::int64_t) * count, cudaMemcpyHostToDevice);
            simulator.step(actions, policy_opponent ? opponent_actions : nullptr, 11, static_cast<std::uint64_t>(step));
            for (const float reward : simulator.download_rewards()) {
                check(std::isfinite(reward) && std::abs(reward) <= 3.0F, "rewards are finite and bounded");
            }
            for (const auto terminated : simulator.download_terminated()) rounds += terminated;
            for (const auto& state : simulator.download_states()) {
                for (const auto& fighter : state.fighters) {
                    check(std::isfinite(fighter.health) && fighter.health <= 180.0 + 1e-9, "health is sane");
                    check(fighter.recoverable_health >= -1e-9 && fighter.recoverable_health <= 180.0 + 1e-9,
                          "recoverable health is sane");
                    check(std::isfinite(fighter.x) && fighter.x >= state.stage.left_wall - 1e-9 &&
                              fighter.x <= state.stage.right_wall + 1e-9,
                          "fighters stay inside the stage");
                    check(std::isfinite(fighter.axis) && fighter.stun >= 0 && fighter.action_frame >= 0,
                          "axis and counters are sane");
                }
            }
            simulator.reset_done(11, static_cast<std::uint64_t>(step));
        }
        cudaFree(actions);
        cudaFree(opponent_actions);
        check(cudaGetLastError() == cudaSuccess, "no CUDA error");
        check(rounds > (steps >= 100 ? count / 4 : 0), "rounds end and reset across the roster");
        std::cout << "full-roster smoke: " << characters << " fighters, " << count << " environments (every pairing, "
                  << "both sides, all scenarios), " << steps << " decisions, " << rounds << " rounds, "
                  << simulator.frames_simulated() << " engine frames\n";
    } catch (const std::exception& error) {
        std::cerr << "full-roster smoke error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " full-roster smoke assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "full-roster smoke passed\n";
    return EXIT_SUCCESS;
}
