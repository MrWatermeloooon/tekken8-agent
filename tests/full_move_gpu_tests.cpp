// CUDA full-move simulator tests:
//  1. engine parity: random fights recorded with the scalar engine are replayed
//     on the GPU, one fight per environment, and every frame's state and
//     events must match bit for bit (randomized parity over every move the
//     random players reach, across several matchups);
//  2. candidate masks equal the scalar engine's legality;
//  3. the training environment (decision steps, scripted opponent, rewards,
//     resets, observations, masks) equals the same contract run on the CPU.

#include "full_combat_random_play.hpp"
#include "t8_v2/full_move_gpu.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <map>
#include <string>

namespace {

using namespace t8::v2;
using namespace t8::v2::testing;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (!condition) {
        ++failures;
        if (failures < 30) std::cerr << "FAIL: " << message << '\n';
    }
}

bool same_bytes(const full_core::State& a, const full_core::State& b) {
    return dump(from_core_state(a)) == dump(from_core_state(b));
}

std::string events_text(const full_core::Events& events) { return dump(from_core_events(events)); }

void test_engine_parity(const std::filesystem::path& generated, int frames) {
    const std::vector<std::pair<std::string, std::string>> matchups = {
        {"jun", "jun"}, {"jun", "king"}, {"yoshimitsu", "lidia"}, {"kazuya", "alisa"},
        {"armor-king", "jack-8"}, {"leroy", "zafina"}, {"lili", "hwoarang"}, {"bryan", "devil-jin"},
    };
    FullMoveRosterOptions options{};
    options.allow_provisional = true;
    std::map<std::string, int> roster_index;
    std::vector<FullMoveCharacter> roster;
    for (const auto& [first, second] : matchups) {
        for (const auto& slug : {first, second}) {
            if (!roster_index.count(slug)) {
                roster_index[slug] = static_cast<int>(roster.size());
                roster.push_back(load_full_move_character(generated, slug, options));
            }
        }
    }
    struct Fight {
        int p1 = 0;
        int p2 = 0;
        FullRecordedFight record;
    };
    std::vector<Fight> fights;
    for (const auto& [first, second] : matchups) {
        const auto fixture = make_fixture(generated, first, second);
        check(fixture.moves[0].size() == roster[roster_index[first]].tables.moves[0].size(),
              first + ": the roster loader binds the same moveset as the reference");
        for (std::uint64_t seed = 1; seed <= 4; ++seed) {
            fights.push_back({roster_index[first], roster_index[second], record_random_fight(fixture, seed * 104729, frames)});
        }
    }
    const std::size_t count = fights.size();
    GpuFullMoveSimulator gpu(roster, count);
    std::vector<int> p1(count), p2(count);
    std::vector<double> x1(count), x2(count);
    for (std::size_t env = 0; env < count; ++env) {
        p1[env] = fights[env].p1;
        p2[env] = fights[env].p2;
        x1[env] = fights[env].record.start;
        x2[env] = fights[env].record.start + 0.9;
    }
    gpu.set_matchups(p1, p2);
    gpu.reset_engine(x1, x2);
    std::vector<full_core::Input> inputs(2 * count);
    std::vector<std::uint8_t> new_round(count);
    std::size_t mismatched_fights = 0;
    std::vector<bool> diverged(count, false);
    std::size_t contacts = 0;
    for (int frame = 0; frame < frames; ++frame) {
        for (std::size_t env = 0; env < count; ++env) {
            const auto& record = fights[env].record;
            for (int player = 0; player < 2; ++player) {
                inputs[2 * env + static_cast<std::size_t>(player)] =
                    to_core_input(record.inputs[static_cast<std::size_t>(frame)][static_cast<std::size_t>(player)],
                                  roster[static_cast<std::size_t>(player == 0 ? p1[env] : p2[env])].tables.moves[0].size());
            }
            new_round[env] = record.new_round[static_cast<std::size_t>(frame)];
        }
        gpu.step_engine(inputs, new_round, x1, x2);
        const auto states = gpu.download_states();
        const auto events = gpu.download_events();
        for (std::size_t env = 0; env < count; ++env) {
            if (diverged[env]) continue;
            const auto& record = fights[env].record;
            contacts += static_cast<std::size_t>(events[env].contact_count);
            const bool state_ok = dump(from_core_state(states[env])) == record.states[static_cast<std::size_t>(frame)];
            const bool events_ok = events_text(events[env]) == record.events[static_cast<std::size_t>(frame)];
            if (!state_ok || !events_ok) {
                diverged[env] = true;
                ++mismatched_fights;
                std::cerr << "GPU/CPU divergence in fight " << env << " at frame " << frame << "\n  cpu: "
                          << record.states[static_cast<std::size_t>(frame)] << "\n  gpu: "
                          << dump(from_core_state(states[env])) << '\n';
            }
        }
    }
    check(mismatched_fights == 0, "every GPU fight matches the scalar engine bit for bit");
    check(contacts > static_cast<std::size_t>(frames) / 6, "the parity fights exercise many contacts");  // 1000 at 6000 frames
    std::cout << "engine parity: " << count << " fights x " << frames << " frames, " << contacts
              << " contacts, " << mismatched_fights << " divergent\n";
}

// memory: --memory observations (recent moves and habits) with a perception delay, which
// must match a CPU replay that keeps its own full_core::Memory per environment.
void test_masks_and_environment(const std::filesystem::path& generated, bool memory = false, bool noise = false,
                                bool screen = false) {
    FullMoveRosterOptions options{};
    options.allow_provisional = true;
    std::vector<FullMoveCharacter> roster = {load_full_move_character(generated, "jun", options),
                                             load_full_move_character(generated, "king", options),
                                             load_full_move_character(generated, "yoshimitsu", options)};
    check(roster[0].candidate_count() == 182, "Jun has 182 candidates");
    check(roster[0].bound_moves > 100, "most of Jun's candidates bind with stand-in geometry");
    const std::size_t count = 96;
    full_core::EnvConfig config{};
    config.max_round_frames = 900;  // short rounds so resets happen often
    config.draw_penalty = 0.5F;     // the anti-stalling terms must match on both sides too
    config.throw_break_penalty = 0.03F;
    config.time_penalty_per_second = 0.01F;
    config.damage_dealt_weight = 0.8F;
    config.damage_taken_weight = 1.2F;
    config.distance_penalty_per_second = 0.002F;
    config.memory = memory;
    config.live_noise = noise;
    config.screen_view = screen;
    if (memory || noise) config.perception_delay = 12;
    GpuFullMoveSimulator gpu(roster, count, config);
    const int width = gpu.observation_size();
    check(width == full_core::full_move_observation_size(memory, noise), "the observation size follows the options");
    std::vector<full_core::Memory> memories(count);
    std::vector<full_core::NoiseLevels> levels(count);
    std::size_t memory_inputs_used = 0;
    std::vector<int> p1(count), p2(count), opponents(count);
    std::vector<std::uint8_t> learner_is_p1(count);
    std::vector<float> aggression(count);
    std::vector<int> scenarios(count);
    for (std::size_t env = 0; env < count; ++env) {
        scenarios[env] = static_cast<int>(env % full_core::kScenarioCount);
        learner_is_p1[env] = env % 2 == 0;
        opponents[env] = static_cast<int>(env % roster.size());
        p1[env] = learner_is_p1[env] ? 0 : opponents[env];
        p2[env] = learner_is_p1[env] ? opponents[env] : 0;
        aggression[env] = static_cast<float>(env % 5) / 4.0F;
    }
    gpu.set_matchups(p1, p2);
    gpu.set_learner_sides(learner_is_p1);
    gpu.set_pending_opponents(opponents, aggression, scenarios);
    const std::uint64_t seed = 99;
    gpu.reset_all(seed, 0);

    // CPU reference running the same contract.
    std::vector<full_core::State> states(count);
    const auto tables_for = [&](std::size_t env) {
        full_core::Tables tables{};
        tables.side[0] = roster[static_cast<std::size_t>(p1[env])].tables.view().side[0];
        tables.side[1] = roster[static_cast<std::size_t>(p2[env])].tables.view().side[0];
        tables.config = roster[0].tables.config;
        return tables;
    };
    const auto map_for = [&](int character) {
        return full_core::CandidateMap{roster[static_cast<std::size_t>(character)].candidate_moves.data(),
                                       roster[static_cast<std::size_t>(character)].candidate_count()};
    };
    const auto reset_reference = [&](std::size_t env, std::uint64_t counter) {
        const int learner = learner_is_p1[env] ? 0 : 1;
        const auto opponent_map = map_for(learner == 0 ? p2[env] : p1[env]);
        full_core::env_reset(tables_for(env), states[env], config, seed, env, counter, scenarios[env], learner,
                             &opponent_map);
        memories[env] = full_core::Memory{};
        if (noise) levels[env] = full_core::draw_noise_levels(config, seed, env, counter);
    };
    for (std::size_t env = 0; env < count; ++env) reset_reference(env, 0);
    // Every scenario actually changes the starting state (or keeps neutral).
    const auto initial = gpu.download_states();
    std::map<int, int> applied;
    for (std::size_t env = 0; env < count; ++env) {
        check(same_bytes(initial[env], states[env]), "scenario resets match on the CPU and GPU");
        const int learner = learner_is_p1[env] ? 0 : 1;
        const auto& own = states[env].fighters[learner];
        const auto& other = states[env].fighters[1 - learner];
        switch (scenarios[env]) {
            case full_core::kScenarioPunish: applied[2] += other.move >= 0 && other.move_blocked; break;
            case full_core::kScenarioThrowBreak: applied[3] += own.throw_break_frames > 0 && other.pending_throw_hit == 0; break;
            case full_core::kScenarioWakeup: applied[4] += own.posture == FullPosture::Grounded; break;
            case full_core::kScenarioJuggle: applied[5] += other.posture == FullPosture::Airborne; break;
            case full_core::kScenarioHeat: applied[6] += own.heat; break;
            case full_core::kScenarioRage: applied[7] += own.rage; break;
            default: break;
        }
    }
    for (int scenario = 2; scenario <= 7; ++scenario) {
        check(applied[scenario] > 0, std::string("scenario ") + full_core::scenario_name(scenario) + " sets up its situation");
    }

    SplitMix rng{7};
    std::size_t terminations = 0, legal_checked = 0;
    std::map<int, int> outcomes;
    bool environment_ok = true;
    for (std::uint64_t step = 1; step <= 1500 && environment_ok; ++step) {
        const auto gpu_masks = gpu.download_learner_masks();
        const auto gpu_observations = gpu.download_learner_observations();
        std::vector<std::int64_t> actions(count);
        for (std::size_t env = 0; env < count && environment_ok; ++env) {
            const int learner = learner_is_p1[env] ? 0 : 1;
            const int learner_character = learner == 0 ? p1[env] : p2[env];
            const auto tables = tables_for(env);
            std::vector<int> legal;
            for (int candidate = 0; candidate < kFullMoveMaxCandidates; ++candidate) {
                const bool cpu_legal = candidate < roster[static_cast<std::size_t>(learner_character)].candidate_count() &&
                    full_core::candidate_legal(tables, states[env], learner, map_for(learner_character), candidate);
                if (cpu_legal != (gpu_masks[env * kFullMoveMaxCandidates + static_cast<std::size_t>(candidate)] != 0)) {
                    environment_ok = false;
                    std::cerr << "mask mismatch env " << env << " candidate " << candidate << " step " << step << '\n';
                    break;
                }
                if (cpu_legal) legal.push_back(candidate);
                ++legal_checked;
            }
            float observation[full_core::full_move_observation_size(true, true)] = {};
            const int opponent_character = learner == 0 ? p2[env] : p1[env];
            full_core::encode_observation(tables, states[env], learner, config.max_round_frames,
                                          roster[static_cast<std::size_t>(learner_character)].roster_id,
                                          roster[static_cast<std::size_t>(opponent_character)].roster_id, observation,
                                          config.perception_delay);
            if (noise) full_core::apply_observation_noise(levels[env], states[env], learner, observation);
            if (screen) full_core::apply_screen_view(observation);
            if (noise) {
                full_core::encode_noise_levels(levels[env], observation + full_core::full_move_observation_size(memory, false));
            }
            if (memory) {
                full_core::encode_memory(memories[env], learner, config.perception_delay,
                                         observation + full_core::kFullMoveObservationSize);
                for (int index = full_core::kFullMoveObservationSize;
                     index < full_core::full_move_observation_size(true, false); ++index) {
                    memory_inputs_used += observation[index] != 0.0F;
                }
            }
            if (std::memcmp(observation, gpu_observations.data() + env * static_cast<std::size_t>(width),
                            sizeof(float) * static_cast<std::size_t>(width)) != 0) {
                environment_ok = false;
                std::cerr << "observation mismatch env " << env << " step " << step << '\n';
            }
            actions[env] = legal.empty() ? 0 : legal[rng.below(legal.size())];
        }
        if (!environment_ok) break;
        // Learner candidates live on the device for the simulator.
        std::int64_t* device_actions = nullptr;
        cudaMalloc(reinterpret_cast<void**>(&device_actions), sizeof(std::int64_t) * count);
        cudaMemcpy(device_actions, actions.data(), sizeof(std::int64_t) * count, cudaMemcpyHostToDevice);
        gpu.step(device_actions, nullptr, seed, step);
        cudaFree(device_actions);
        const auto rewards = gpu.download_rewards();
        const auto terminated = gpu.download_terminated();
        const auto gpu_outcomes = gpu.download_outcomes();
        for (std::size_t env = 0; env < count; ++env) {
            const int learner = learner_is_p1[env] ? 0 : 1;
            const int learner_character = learner == 0 ? p1[env] : p2[env];
            const int opponent_character = learner == 0 ? p2[env] : p1[env];
            const auto result = full_core::env_step(tables_for(env), states[env], learner, map_for(learner_character),
                                                    map_for(opponent_character), static_cast<int>(actions[env]), -1,
                                                    aggression[env], config, seed, env, step,
                                                    memory ? &memories[env] : nullptr, noise ? &levels[env] : nullptr);
            if (result.reward != rewards[env] || result.terminated != (terminated[env] != 0) ||
                (result.terminated && result.outcome != gpu_outcomes[env])) {
                environment_ok = false;
                std::cerr << "step result mismatch env " << env << " step " << step << '\n';
            }
            if (result.terminated) {
                ++terminations;
                ++outcomes[result.outcome];
            }
        }
        const auto gpu_states = gpu.download_states();
        for (std::size_t env = 0; env < count && environment_ok; ++env) {
            if (!same_bytes(states[env], gpu_states[env])) {
                environment_ok = false;
                std::cerr << "state mismatch env " << env << " step " << step << '\n';
            }
        }
        gpu.reset_done(seed, step);
        for (std::size_t env = 0; env < count; ++env) {
            if (terminated[env]) reset_reference(env, step);
        }
    }
    check(environment_ok, screen ? "with the screen view and live noise, the GPU environment matches the CPU contract"
                          : noise ? "with memory and live noise, the GPU environment matches the CPU contract"
                          : memory ? "with memory, the GPU environment matches the CPU contract step for step"
                                   : "the GPU environment matches the CPU contract step for step");
    if (noise) {
        const auto gpu_levels = gpu.download_noise_levels();
        check(gpu_levels.size() == count &&
                  std::memcmp(gpu_levels.data(), levels.data(), sizeof(full_core::NoiseLevels) * count) == 0,
              "every environment's noise levels match");
    }
    if (memory) {
        check(memory_inputs_used > 10000, "the memory inputs carry history");
        std::cout << "memory parity: " << memory_inputs_used << " nonzero memory inputs compared\n";
    }
    check(terminations > 50, "rounds end and reset during the test");
    std::cout << "environment parity: " << terminations << " rounds (wins " << outcomes[1] << ", losses "
              << outcomes[-1] << ", draws " << outcomes[0] << "), " << legal_checked << " mask entries checked\n";

    // Engine-level legal mask against the scalar engine on the same states.
    const auto legal_p2 = gpu.download_legal_mask(1);
    for (std::size_t env = 0; env < count; ++env) {
        const auto tables = tables_for(env);
        for (int candidate = 0; candidate < roster[static_cast<std::size_t>(p2[env])].candidate_count(); ++candidate) {
            const bool cpu_legal = full_core::candidate_legal(tables, states[env], 1, map_for(p2[env]), candidate);
            check(cpu_legal == (legal_p2[env * kFullMoveMaxCandidates + static_cast<std::size_t>(candidate)] != 0),
                  "engine-level legal masks match");
        }
    }
}

// What --memory records: P2 throws one move at P1 (who stands guarding). The move enters P2's
// own history at once and P1's view of P2 only after the perception delay; a whiff from far
// away ends as missed, the same move up close as a contact (blocked or hit).
void test_memory_semantics(const std::filesystem::path& generated) {
    FullMoveRosterOptions options{};
    options.allow_provisional = true;
    const auto jun = load_full_move_character(generated, "jun", options);
    full_core::Tables tables{};
    tables.side[0] = jun.tables.view().side[0];
    tables.side[1] = jun.tables.view().side[0];
    tables.config = jun.tables.config;
    int attack = -1;  // a quick single-hit strike
    for (int move = 0; move < tables.side[1].move_count && attack < 0; ++move) {
        const auto& spec = tables.side[1].moves[move];
        if (spec.hit_count == 1 && full_core::memory_level(tables, 1, move) <= 3 &&
            full_core::hit_of(tables, 1, spec, 0).first_active_frame <= 15 && spec.total_frames < 50) {
            full_core::State probe{};  // usable from a neutral stand
            full_core::reset(tables, probe, -0.4, 3.0);
            full_core::Input strike{};
            strike.move = move;
            full_core::Events events{};
            full_core::step(tables, probe, full_core::Input{}, strike, events);
            if (probe.fighters[1].move == move) attack = move;
        }
    }
    check(attack >= 0, "Jun has a quick single-hit strike for the memory test");
    if (attack < 0) return;
    const int delay = 12;
    const auto play = [&](double p2_x, int& visible_at, std::int8_t& outcome) {
        full_core::State state{};
        full_core::reset(tables, state, -0.4, p2_x);
        full_core::Memory memory{};
        visible_at = -1;
        outcome = -1;
        for (int frame = 0; frame < 120; ++frame) {
            full_core::Input strike{};
            if (frame == 0) strike.move = attack;
            full_core::Events events{};
            full_core::step(tables, state, full_core::Input{}, strike, events);
            full_core::memory_update(tables, state, events, delay, memory);
            float own_view[full_core::kFullMoveMemoryFeatures];
            float p1_view[full_core::kFullMoveMemoryFeatures];
            full_core::encode_memory(memory, 1, delay, own_view);
            full_core::encode_memory(memory, 0, delay, p1_view);
            const float* p2_seen_by_p1 = p1_view + full_core::kMemoryFighterFeatures;
            bool own_has = false, p1_has = false;
            for (int index = 0; index < 5; ++index) {
                own_has = own_has || own_view[index] != 0.0F;
                p1_has = p1_has || p2_seen_by_p1[index] != 0.0F;
            }
            if (frame == 0) check(own_has, "a move enters its own fighter's history at once");
            if (p1_has && visible_at < 0) visible_at = memory.fighters[1].slots[0].age;
            if (state.fighters[1].move < 0 && frame > 0) {
                outcome = memory.fighters[1].slots[0].outcome;
                break;
            }
        }
    };
    int visible_at = -1;
    std::int8_t outcome = -1;
    play(3.0, visible_at, outcome);
    check(visible_at == delay, "the opponent sees a move in its history only once it is perception_delay frames old");
    check(outcome == 4, "a whiff from far away is remembered as missed");
    play(0.4, visible_at, outcome);
    check(outcome == 1 || outcome == 2 || outcome == 3, "the same move up close is remembered as a contact");
    std::cout << "memory semantics: visible after " << visible_at << " frames, close-range outcome "
              << static_cast<int>(outcome) << '\n';
}

// What --live-noise does: zero levels change nothing; the position noise has the drawn spread;
// detections flip at about the drawn rate; and a policy's input lands `latency` frames late
// (one more a quarter of the time).
void test_noise_semantics(const std::filesystem::path& generated) {
    FullMoveRosterOptions options{};
    options.allow_provisional = true;
    const auto jun = load_full_move_character(generated, "jun", options);
    full_core::Tables tables{};
    tables.side[0] = jun.tables.view().side[0];
    tables.side[1] = jun.tables.view().side[0];
    tables.config = jun.tables.config;
    full_core::State state{};
    full_core::reset(tables, state, -0.6, 0.6);
    float clean[full_core::kFullMoveObservationSize];
    full_core::encode_observation(tables, state, 0, 3600, 0, 1, clean);
    full_core::NoiseLevels zero{};
    zero.key = 12345U;
    float copy[full_core::kFullMoveObservationSize];
    std::memcpy(copy, clean, sizeof copy);
    full_core::apply_observation_noise(zero, state, 0, copy);
    bool health_only = true;  // zero levels: only the tiny HUD health noise remains
    for (int index = 4; index < full_core::kFullMoveObservationSize; ++index) health_only = health_only && copy[index] == clean[index];
    check(health_only, "zero noise levels leave everything but the health bars untouched");
    full_core::NoiseLevels levels{};
    levels.position_sigma = 0.25F;
    levels.flip_rate = 0.10F;
    levels.key = 777U;
    double sum = 0.0, squares = 0.0;
    int posture_changes = 0;
    const int samples = 20000;
    for (int frame = 0; frame < samples; ++frame) {
        state.frame = frame;
        std::memcpy(copy, clean, sizeof copy);
        full_core::apply_observation_noise(levels, state, 0, copy);
        const double error = (copy[4] - clean[4]) * 4.0;  // metres
        sum += error;
        squares += error * error;
        bool changed = false;
        for (int index = 14; index < 20; ++index) changed = changed || copy[index] != clean[index];
        posture_changes += changed;
    }
    const double spread = std::sqrt(squares / samples - (sum / samples) * (sum / samples));
    const double change_rate = static_cast<double>(posture_changes) / samples;
    check(spread > 0.22 && spread < 0.28, "distance noise has the drawn spread (0.25 m)");
    check(change_rate > 0.07 && change_rate < 0.10, "the opponent's posture is misread at about the error rate (10% x 5/6)");

    // Latency: the same attack from the same state starts `lag` frames later.
    int attack = -1;
    for (int candidate = kFullMoveUniversalCount; candidate < jun.candidate_count() && attack < 0; ++candidate) {
        full_core::State probe{};
        full_core::reset(tables, probe, -0.6, 0.6);
        const full_core::CandidateMap map{jun.candidate_moves.data(), jun.candidate_count()};
        if (full_core::candidate_legal(tables, probe, 0, map, candidate)) attack = candidate;
    }
    check(attack >= 0, "a legal attack candidate for the latency test");
    if (attack < 0) return;
    const full_core::CandidateMap map{jun.candidate_moves.data(), jun.candidate_count()};
    full_core::EnvConfig config{};
    const auto start_frame = [&](const full_core::NoiseLevels* noise, std::uint64_t counter) {
        full_core::State fight{};
        full_core::reset(tables, fight, -0.6, 0.6);
        full_core::Memory memory{};
        const auto result = full_core::env_step(tables, fight, 0, map, map, attack, 0, 0.0, config, 1, 0, counter,
                                                &memory, noise);
        return result.frames - 1 - memory.fighters[0].slots[0].age;
    };
    check(start_frame(nullptr, 1) == 0, "without latency the input lands on the decision frame");
    full_core::NoiseLevels late{};
    late.latency = 4;
    late.key = 99U;
    int fours = 0, fives = 0;
    for (std::uint64_t counter = 1; counter <= 200; ++counter) {
        const int start = start_frame(&late, counter);
        fours += start == 4;
        fives += start == 5;
    }
    check(fours + fives == 200, "with latency 4 every input lands 4 or 5 frames late");
    check(fives > 30 && fives < 70, "the extra frame of jitter comes about a quarter of the time");
    std::cout << "noise semantics: distance spread " << spread << " m, posture misread " << change_rate
              << ", latency 4/5 frames " << fours << '/' << fives << '\n';
}

// The screen view keeps exactly kScreenViewFeatures of the state features (and the roster).
void test_screen_view() {
    bool same = true;
    int listed = 0;
    for (int index = 0; index < full_core::kFullMoveStateFeatures; ++index) {
        bool in_list = false;
        for (const int kept : full_core::kScreenViewFeatures) in_list = in_list || kept == index;
        same = same && in_list == full_core::screen_view_feature(index);
        listed += in_list;
    }
    check(same && listed == full_core::kScreenViewFeatureCount, "the screen-view list and function agree");
    float out[full_core::kFullMoveObservationSize];
    for (float& value : out) value = 0.5F;
    full_core::apply_screen_view(out);
    bool kept_only = true;
    for (int index = 0; index < full_core::kFullMoveObservationSize; ++index) {
        const bool kept = index >= full_core::kFullMoveStateFeatures || full_core::screen_view_feature(index);
        kept_only = kept_only && (out[index] == (kept ? 0.5F : 0.0F));
    }
    check(kept_only, "the screen view zeroes every state feature the screen cannot read, and nothing else");
}

void report_throughput(const std::filesystem::path& generated) {
    FullMoveRosterOptions options{};
    options.allow_provisional = true;
    std::vector<FullMoveCharacter> roster = {load_full_move_character(generated, "jun", options),
                                             load_full_move_character(generated, "yoshimitsu", options)};
    const std::size_t count = 8192;
    GpuFullMoveSimulator gpu(roster, count);
    std::vector<int> p1(count, 0), p2(count, 1), opponents(count, 1);
    std::vector<std::uint8_t> sides(count, 1U);
    gpu.set_matchups(p1, p2);
    gpu.set_learner_sides(sides);
    gpu.set_pending_opponents(opponents);
    gpu.reset_all(5, 0);
    std::int64_t* actions = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&actions), sizeof(std::int64_t) * count);
    cudaMemset(actions, 0, sizeof(std::int64_t) * count);  // neutral: the scripted opponent drives the fight
    gpu.synchronize();
    const auto frames_before = gpu.frames_simulated();
    const auto started = std::chrono::steady_clock::now();
    for (std::uint64_t step = 1; step <= 50; ++step) {
        gpu.step(actions, nullptr, 5, step);
        gpu.reset_done(5, step);
    }
    gpu.synchronize();
    const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    cudaFree(actions);
    const double frames = static_cast<double>(gpu.frames_simulated() - frames_before);
    std::cout << "throughput: " << count * 50 / seconds / 1e6 << "M decisions/s, " << frames / seconds / 1e6
              << "M engine frames/s (" << count << " environments, includes 368-candidate masks for both sides)\n";
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: full_move_gpu_tests <data/generated> [frames]\n";
        return EXIT_FAILURE;
    }
    try {
        test_engine_parity(argv[1], argc > 2 ? std::atoi(argv[2]) : 6000);
        test_masks_and_environment(argv[1]);
        test_masks_and_environment(argv[1], true);
        test_memory_semantics(argv[1]);
        test_masks_and_environment(argv[1], true, true);
        test_noise_semantics(argv[1]);
        test_masks_and_environment(argv[1], false, true, true);
        test_screen_view();
        report_throughput(argv[1]);
    } catch (const std::exception& error) {
        std::cerr << "full-move GPU test error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    if (failures != 0) {
        std::cerr << failures << " full-move GPU assertion(s) failed\n";
        return EXIT_FAILURE;
    }
    std::cout << "full-move GPU tests passed\n";
    return EXIT_SUCCESS;
}
