// Head-to-head full-move checkpoint arena.
//
//   t8_v2_full_move_arena --output arena.csv [--rounds-per-pair N] [--seed N] [--allow-provisional]
//                         [--character slug | --mixed] [--sample] [--perception-delay N] [--hold-frames N]
//                         a.t8ppo b.t8ppo c.t8ppo ...
//
// --perception-delay / --hold-frames: the environment the entrants trained in (defaults 0 / 4).
// --live-noise [--noise-position M --noise-flip P --latency-min F --latency-max F]: play under
// screen-reading noise and input latency (a robustness check for any checkpoint).
// --sample: moves drawn from the policies' probabilities (as training and champion matches play)
// instead of greedy.
//
//   t8_v2_full_move_arena --character-ranking --output ranking.csv [--rounds-per-pair N] [--sample]
//                         [--perception-delay N] [--hold-frames N] --allow-provisional champion.t8ppo
//
// --character-ranking: one multi-character checkpoint plays itself in every matchup of the
// characters it learned (both sides, N rounds per ordered pair, default 32), so each
// character's score says how well the policy plays it. The CSV has one row per ordered pair
// (character, opponent, rounds, wins, losses, draws); tools/character_ranking.py ranks it.
//
// --mixed: multi-character policies play every pairing of the characters all
// entrants were trained on (each round picks both sides' characters in turn),
// instead of one character each.
//
// Every pair of checkpoints plays N rounds (half with each on P1), both as
// policies over their characters' full candidate lists, deterministically
// (greedy) from a fixed seed, so the table is reproducible. Each checkpoint's
// character comes from its .t8fmstate. The CSV (one row per pair: wins of each
// and draws) feeds tools/rank_checkpoints.py, which fits draw-aware ratings.

#include "t8_v2/full_move_gpu.hpp"
#include "t8_v2/ppo.hpp"

#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

namespace {

using namespace t8::v2;

struct Entrant {
    std::filesystem::path checkpoint;
    std::string character;
    bool provisional = false;
    std::string trained;  // every character the checkpoint learned
};

std::string read_string(std::istream& input) {
    std::uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof count);
    if (!input || count > 4096) throw std::runtime_error("corrupt full-move trainer state");
    std::string value(count, '\0');
    input.read(value.data(), static_cast<std::streamsize>(count));
    return value;
}

Entrant read_entrant(const std::filesystem::path& checkpoint) {
    auto state = checkpoint;
    state.replace_extension(".t8fmstate");
    std::ifstream input(state, std::ios::binary);
    if (!input) throw std::runtime_error("missing " + state.string() + " (only full-move checkpoints can play)");
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    if (std::string(magic.data(), 6) != "T8FMV1") throw std::runtime_error("not a full-move trainer state: " + state.string());
    std::uint32_t version = 0;
    std::uint64_t update = 0;
    input.read(reinterpret_cast<char*>(&version), sizeof version);
    input.read(reinterpret_cast<char*>(&update), sizeof update);
    read_string(input);  // catalog
    read_string(input);  // roster
    // A multi-character policy records its characters comma-separated; it plays the first
    // one here unless --character picks another it was trained on.
    const std::string learners = read_string(input);
    Entrant entrant{checkpoint, learners.substr(0, learners.find(',')), false};
    entrant.trained = learners;
    char provisional = 0;
    input.read(&provisional, 1);
    entrant.provisional = provisional != 0;
    return entrant;
}

// The architecture and contract stored in a .t8ppo header (see ppo.cu).
ActorCriticConfig config_from_checkpoint(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    struct Header {
        char magic[8];
        std::uint32_t version, observations, actions, hidden;
        std::uint64_t step, parameters;
    } header{};
    struct Contract {
        char catalog[64], roster[24], observation[32], action[32];
        std::uint32_t feature_size, universal_count;
    } contract{};
    input.read(reinterpret_cast<char*>(&header), sizeof header);
    input.read(reinterpret_cast<char*>(&contract), sizeof contract);
    if (!input || std::string(header.magic, 7) != "T8V2PPO" || header.version < 3) {
        throw std::runtime_error("not a contract-bound checkpoint: " + path.string());
    }
    const auto text = [](const char* data, std::size_t size) { return std::string(data, strnlen(data, size)); };
    ActorCriticConfig config{};
    config.observation_size = static_cast<int>(header.observations);
    config.action_count = static_cast<int>(header.actions);
    config.hidden_size = static_cast<int>(header.hidden);
    config.catalog_sha256 = text(contract.catalog, sizeof contract.catalog);
    config.roster_version = text(contract.roster, sizeof contract.roster);
    config.observation_contract = text(contract.observation, sizeof contract.observation);
    config.action_contract = text(contract.action, sizeof contract.action);
    config.action_feature_size = static_cast<int>(contract.feature_size);
    config.universal_action_count = static_cast<int>(contract.universal_count);
    if (header.version >= 4) {
        std::uint32_t enabled = 0;
        std::uint32_t architecture = 0;  // depth (ppo.cu architecture_code); 0 = 2 shared, no heads
        input.read(reinterpret_cast<char*>(&enabled), sizeof enabled);
        input.read(reinterpret_cast<char*>(&architecture), sizeof architecture);
        config.observation_normalization = enabled != 0;
        if (architecture != 0) {
            config.shared_layers = static_cast<int>(architecture & 0xFFU);
            config.head_layers = static_cast<int>((architecture >> 8U) & 0xFFU);
        }
    }
    if (config.action_contract != "full-move-v1") {
        throw std::runtime_error(path.string() + " is not a full-move checkpoint (" + config.action_contract + ")");
    }
    return config;
}

// The simulator's observations for a policy. The simulator lays out the base inputs, then the
// memory block (when any entrant has memory), then the noise levels (when any has noise or
// --live-noise is on); each policy gets the blocks it was trained with, gathered on the device.
struct InputLayout {
    bool memory = false;
    bool noise = false;
};

InputLayout layout_of(int observation_size) {
    for (const bool memory : {false, true}) {
        for (const bool noise : {false, true}) {
            if (full_core::full_move_observation_size(memory, noise) == observation_size) return {memory, noise};
        }
    }
    throw std::runtime_error("unknown observation size " + std::to_string(observation_size));
}

class ObservationView {
public:
    ObservationView(std::size_t environments, InputLayout simulator) : environments_(environments), simulator_(simulator) {}
    ~ObservationView() {
        if (buffer_ != nullptr) cudaFree(buffer_);
    }
    ObservationView(const ObservationView&) = delete;
    ObservationView& operator=(const ObservationView&) = delete;
    // screen: the policy trained with --screen-observation sees only full_core::kScreenViewFeatures
    // of the state features (the rest zeroed on the device, as apply_screen_view does).
    const float* operator()(const float* source, int policy_size, bool screen = false) {
        const InputLayout policy = layout_of(policy_size);
        if (policy.memory == simulator_.memory && policy.noise == simulator_.noise && !screen) return source;
        if ((policy.memory && !simulator_.memory) || (policy.noise && !simulator_.noise)) {
            throw std::runtime_error("a policy needs inputs the simulator does not give");
        }
        const int width = full_core::full_move_observation_size(simulator_.memory, simulator_.noise);
        if (buffer_ == nullptr) cudaMalloc(reinterpret_cast<void**>(&buffer_), sizeof(float) * environments_ * width);
        const auto copy = [&](int to, int from, int count) {
            cudaMemcpy2D(buffer_ + to, sizeof(float) * policy_size, source + from, sizeof(float) * width,
                         sizeof(float) * count, environments_, cudaMemcpyDeviceToDevice);
        };
        copy(0, 0, full_core::kFullMoveObservationSize);
        if (policy.memory) copy(full_core::kFullMoveObservationSize, full_core::kFullMoveObservationSize,
                                full_core::kFullMoveMemoryFeatures);
        if (policy.noise) {
            copy(full_core::full_move_observation_size(policy.memory, false),
                 full_core::full_move_observation_size(simulator_.memory, false), full_core::kFullMoveNoiseFeatures);
        }
        if (screen) {  // zero each run of state features between the kept ones
            int start = 0;
            for (int kept = 0; kept <= full_core::kScreenViewFeatureCount; ++kept) {
                const int end = kept < full_core::kScreenViewFeatureCount ? full_core::kScreenViewFeatures[kept]
                                                                           : full_core::kFullMoveStateFeatures;
                if (end > start) {
                    cudaMemset2D(buffer_ + start, sizeof(float) * policy_size, 0, sizeof(float) * (end - start),
                                 environments_);
                }
                start = end + 1;
            }
        }
        return buffer_;
    }

private:
    std::size_t environments_;
    InputLayout simulator_;
    float* buffer_ = nullptr;
};

// The environment for these entrants: memory and noise blocks when any of them needs one.
// Noise-trained entrants on a clean evaluation get zero noise levels.
void fit_environment(full_core::EnvConfig& config, const std::vector<int>& input_sizes, bool live_noise) {
    for (const int size : input_sizes) {
        const InputLayout layout = layout_of(size);
        config.memory = config.memory || layout.memory;
        config.live_noise = config.live_noise || layout.noise;
    }
    if (live_noise) {
        config.live_noise = true;
    } else if (config.live_noise) {
        config.noise_position_max = 0.0F;
        config.noise_flip_max = 0.0F;
        config.latency_min = 0;
        config.latency_max = 0;
    }
}

// Every ordered pair of the checkpoint's characters, the checkpoint on both sides.
int character_ranking(const std::filesystem::path& checkpoint, const std::filesystem::path& generated,
                      const std::filesystem::path& output, std::size_t rounds_per_pair, std::uint64_t seed,
                      bool sample, full_core::EnvConfig env_config, bool live_noise) {
    const Entrant entrant = read_entrant(checkpoint);
    const auto checkpoint_config = config_from_checkpoint(checkpoint);
    fit_environment(env_config, {checkpoint_config.observation_size}, live_noise);
    env_config.screen_view = checkpoint_config.observation_contract.rfind("full-move-screen", 0) == 0;
    FullMoveRosterOptions roster_options{};
    roster_options.allow_provisional = true;
    std::vector<FullMoveCharacter> roster;
    std::istringstream list(entrant.trained);
    for (std::string slug; std::getline(list, slug, ',');) {
        if (slug.empty()) continue;
        auto character = load_full_move_character(generated, slug, roster_options);
        if (character.bound_moves > 0) roster.push_back(std::move(character));
    }
    const std::size_t count = roster.size();
    if (count < 2) throw std::runtime_error("--character-ranking needs a checkpoint trained on several characters");
    const std::size_t table_size = static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize;
    std::vector<float> all_tables(count * table_size, 0.0F);
    for (std::size_t index = 0; index < count; ++index) {
        std::copy(roster[index].candidate_features.begin(), roster[index].candidate_features.end(),
                  all_tables.begin() + static_cast<std::ptrdiff_t>(index * table_size));
    }
    // One pass plays every ordered pair once with the "learner" on P1 and once on P2.
    const std::size_t environments = count * count * 2;
    GpuFullMoveSimulator simulator(roster, environments, env_config);
    GpuActorCritic first(environments, config_from_checkpoint(checkpoint), 1);
    GpuActorCritic second(environments, config_from_checkpoint(checkpoint), 2);
    first.load_checkpoint(checkpoint, false);
    second.load_checkpoint(checkpoint, false);
    first.set_action_feature_tables(all_tables, static_cast<int>(count));
    second.set_action_feature_tables(all_tables, static_cast<int>(count));
    std::vector<int> own(environments), other(environments), p1(environments), p2(environments);
    std::vector<std::uint8_t> sides(environments);
    for (std::size_t env = 0; env < environments; ++env) {
        const std::size_t pair = env / 2;
        own[env] = static_cast<int>(pair / count);
        other[env] = static_cast<int>(pair % count);
        sides[env] = env % 2 == 0;
        p1[env] = sides[env] ? own[env] : other[env];
        p2[env] = sides[env] ? other[env] : own[env];
    }
    // wins, losses, draws of character i against character j
    std::vector<std::array<std::size_t, 3>> results(count * count, {0, 0, 0});
    const std::size_t passes = std::max<std::size_t>(1, rounds_per_pair / 2);
    for (std::size_t pass = 0; pass < passes; ++pass) {
        simulator.set_matchups(p1, p2);
        simulator.set_learner_sides(sides);
        simulator.set_pending_opponents(other);
        simulator.set_pending_learners(own);
        const std::uint64_t pass_seed = seed + pass * 7919;
        simulator.reset_all(pass_seed, 0);
        std::vector<bool> finished(environments, false);
        std::size_t remaining = environments;
        for (std::uint64_t step = 1; step <= 40'000 && remaining > 0; ++step) {
            const auto first_out = first.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                 environments, pass_seed, step, !sample, nullptr,
                                                 simulator.learner_feature_tables());
            const auto second_out = second.forward(simulator.opponent_observations(), simulator.opponent_masks(),
                                                   environments, pass_seed + 1, step, !sample, nullptr,
                                                   simulator.opponent_feature_tables());
            simulator.step(first_out.actions, second_out.actions, pass_seed, step);
            const auto terminated = simulator.download_terminated();
            const auto outcomes = simulator.download_outcomes();
            for (std::size_t env = 0; env < environments; ++env) {
                if (finished[env] || !terminated[env]) continue;
                finished[env] = true;
                --remaining;
                auto& cell = results[static_cast<std::size_t>(own[env]) * count + static_cast<std::size_t>(other[env])];
                ++cell[outcomes[env] > 0 ? 0 : (outcomes[env] < 0 ? 1 : 2)];
            }
            simulator.reset_done(pass_seed, step);
        }
        std::cerr << "\rpass " << pass + 1 << '/' << passes << std::flush;
    }
    std::cerr << '\n';
    std::ofstream table(output);
    table << "character,opponent,rounds,wins,losses,draws\n";
    for (std::size_t i = 0; i < count; ++i) {
        for (std::size_t j = 0; j < count; ++j) {
            const auto& cell = results[i * count + j];
            table << roster[i].slug << ',' << roster[j].slug << ',' << cell[0] + cell[1] + cell[2] << ',' << cell[0]
                  << ',' << cell[1] << ',' << cell[2] << '\n';
        }
    }
    std::cout << "wrote " << output.string() << " (" << count << " characters, " << passes * 2
              << " rounds per ordered pair)\n";
    return EXIT_SUCCESS;
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::filesystem::path> checkpoints;
    std::filesystem::path output = "arena.csv";
    std::size_t rounds = 256;
    std::uint64_t seed = 31337;
    bool allow_provisional = false;
    std::string forced_character;
    bool mixed = false;
    bool sample = false;
    bool ranking = false;
    bool live_noise = false;  // --live-noise: evaluate under screen noise and input latency
    bool rounds_given = false;
    full_core::EnvConfig env_config{};
    std::filesystem::path generated = std::filesystem::path(T8_V2_SOURCE_DIR) / "data" / "generated";
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--output" && index + 1 < argc) output = argv[++index];
        else if (argument == "--rounds-per-pair" && index + 1 < argc) {
            rounds = std::stoull(argv[++index]);
            rounds_given = true;
        }
        else if (argument == "--seed" && index + 1 < argc) seed = std::stoull(argv[++index]);
        else if (argument == "--data-dir" && index + 1 < argc) generated = argv[++index];
        else if (argument == "--allow-provisional") allow_provisional = true;
        else if (argument == "--character" && index + 1 < argc) forced_character = argv[++index];
        else if (argument == "--mixed") mixed = true;
        else if (argument == "--sample") sample = true;
        else if (argument == "--character-ranking") ranking = true;
        else if (argument == "--live-noise") live_noise = true;
        else if (argument == "--noise-position" && index + 1 < argc) env_config.noise_position_max = std::stof(argv[++index]);
        else if (argument == "--noise-flip" && index + 1 < argc) env_config.noise_flip_max = std::stof(argv[++index]);
        else if (argument == "--latency-min" && index + 1 < argc) env_config.latency_min = std::stoi(argv[++index]);
        else if (argument == "--latency-max" && index + 1 < argc) env_config.latency_max = std::stoi(argv[++index]);
        else if (argument == "--perception-delay" && index + 1 < argc) env_config.perception_delay = std::stoi(argv[++index]);
        else if (argument == "--hold-frames" && index + 1 < argc) env_config.hold_frames = std::stoi(argv[++index]);
        else checkpoints.emplace_back(argument);
    }
    if (ranking) {
        if (checkpoints.size() != 1) {
            std::cerr << "--character-ranking takes exactly one checkpoint\n";
            return 2;
        }
        try {
            return character_ranking(checkpoints.front(), generated, output, rounds_given ? rounds : 32, seed, sample,
                                     env_config, live_noise);
        } catch (const std::exception& error) {
            std::cerr << "arena error: " << error.what() << '\n';
            return EXIT_FAILURE;
        }
    }
    if (checkpoints.size() < 2 || rounds == 0 || rounds % 2 != 0) {
        std::cerr << "usage: t8_v2_full_move_arena --output arena.csv [--rounds-per-pair EVEN] [--seed N] "
                     "[--allow-provisional] a.t8ppo b.t8ppo ...\n";
        return 2;
    }
    try {
        std::vector<Entrant> entrants;
        for (const auto& path : checkpoints) {
            auto entrant = read_entrant(path);
            if (!forced_character.empty()) {
                if (("," + entrant.trained + ",").find("," + forced_character + ",") == std::string::npos) {
                    throw std::runtime_error(path.string() + " was not trained to play " + forced_character);
                }
                entrant.character = forced_character;
            }
            entrants.push_back(entrant);
        }
        FullMoveRosterOptions roster_options{};
        roster_options.allow_provisional = allow_provisional;
        std::vector<FullMoveCharacter> roster;
        std::map<std::string, int> character_index;
        // --mixed: the characters every entrant learned.
        std::vector<std::string> common;
        if (mixed) {
            std::istringstream first_list(entrants.front().trained);
            for (std::string slug; std::getline(first_list, slug, ',');) {
                bool shared = true;
                for (const auto& entrant : entrants) {
                    shared = shared && ("," + entrant.trained + ",").find("," + slug + ",") != std::string::npos;
                }
                if (shared) common.push_back(slug);
            }
            if (common.empty()) throw std::runtime_error("--mixed: the entrants share no trained character");
        }
        for (const auto& entrant : entrants) {
            if (entrant.provisional && !allow_provisional) {
                throw std::runtime_error(entrant.checkpoint.string() + " was trained with stand-in geometry; "
                                         "pass --allow-provisional to rank development checkpoints");
            }
        }
        std::vector<std::string> needed = mixed ? common : std::vector<std::string>{};
        if (!mixed) {
            for (const auto& entrant : entrants) needed.push_back(entrant.character);
        }
        for (const auto& slug : needed) {
            if (!character_index.count(slug)) {
                auto character = load_full_move_character(generated, slug, roster_options);
                if (character.bound_moves == 0) {
                    if (mixed) continue;
                    throw std::runtime_error(slug + " has no bound move");
                }
                character_index[slug] = static_cast<int>(roster.size());
                roster.push_back(std::move(character));
            }
        }
        std::vector<int> mixed_roster;
        for (const auto& slug : common) {
            if (character_index.count(slug)) mixed_roster.push_back(character_index.at(slug));
        }
        // One candidate table per loaded character, for the mixed mode.
        const std::size_t table_size = static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize;
        std::vector<float> all_tables(roster.size() * table_size, 0.0F);
        for (std::size_t index = 0; index < roster.size(); ++index) {
            std::copy(roster[index].candidate_features.begin(), roster[index].candidate_features.end(),
                      all_tables.begin() + static_cast<std::ptrdiff_t>(index * table_size));
        }
        const auto features_of = [&](const std::string& character) {
            std::vector<float> features(static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize, 0.0F);
            const auto& source = roster[static_cast<std::size_t>(character_index.at(character))].candidate_features;
            std::copy(source.begin(), source.end(), features.begin());
            return features;
        };
        std::vector<int> input_sizes;
        std::vector<bool> screen_inputs;  // --screen-observation entrants
        for (const auto& entrant : entrants) {
            const auto entrant_config = config_from_checkpoint(entrant.checkpoint);
            input_sizes.push_back(entrant_config.observation_size);
            screen_inputs.push_back(entrant_config.observation_contract.rfind("full-move-screen", 0) == 0);
        }
        fit_environment(env_config, input_sizes, live_noise);
        GpuFullMoveSimulator simulator(roster, rounds, env_config);
        const InputLayout simulator_layout{env_config.memory, env_config.live_noise};
        ObservationView first_view(rounds, simulator_layout);
        ObservationView second_view(rounds, simulator_layout);
        std::ofstream table(output);
        table << "a,b,character_a,character_b,rounds,wins_a,wins_b,draws\n";
        for (std::size_t a = 0; a < entrants.size(); ++a) {
            for (std::size_t b = a + 1; b < entrants.size(); ++b) {
                GpuActorCritic first(rounds, config_from_checkpoint(entrants[a].checkpoint), 1);
                GpuActorCritic second(rounds, config_from_checkpoint(entrants[b].checkpoint), 2);
                first.load_checkpoint(entrants[a].checkpoint, false);
                second.load_checkpoint(entrants[b].checkpoint, false);
                std::vector<int> own(rounds), other(rounds), p1(rounds), p2(rounds);
                std::vector<std::uint8_t> sides(rounds);
                if (mixed) {
                    first.set_action_feature_tables(all_tables, static_cast<int>(roster.size()));
                    second.set_action_feature_tables(all_tables, static_cast<int>(roster.size()));
                } else {
                    first.set_action_features(features_of(entrants[a].character));
                    second.set_action_features(features_of(entrants[b].character));
                }
                const std::size_t count = mixed_roster.size();
                for (std::size_t env = 0; env < rounds; ++env) {
                    const std::size_t pair = env / 2;
                    own[env] = mixed ? mixed_roster[pair % count] : character_index.at(entrants[a].character);
                    other[env] = mixed ? mixed_roster[(pair / count + pair) % count]
                                       : character_index.at(entrants[b].character);
                    sides[env] = env % 2 == 0;  // "learner" = checkpoint a, on P1 in even environments
                    p1[env] = sides[env] ? own[env] : other[env];
                    p2[env] = sides[env] ? other[env] : own[env];
                }
                simulator.set_matchups(p1, p2);
                simulator.set_learner_sides(sides);
                simulator.set_pending_opponents(other);
                simulator.set_pending_learners(own);
                const std::uint64_t pair_seed = seed + a * 1000 + b;
                simulator.reset_all(pair_seed, 0);
                std::vector<bool> finished(rounds, false);
                std::array<std::size_t, 3> tally{};  // a wins, b wins, draws
                std::size_t remaining = rounds;
                for (std::uint64_t step = 1; step <= 40'000 && remaining > 0; ++step) {
                    const auto first_out = first.forward(first_view(simulator.learner_observations(), input_sizes[a],
                                                                    screen_inputs[a]),
                                                         simulator.learner_masks(), rounds, pair_seed, step, !sample,
                                                         nullptr,
                                                         mixed ? simulator.learner_feature_tables() : nullptr);
                    const auto second_out = second.forward(second_view(simulator.opponent_observations(), input_sizes[b],
                                                                      screen_inputs[b]),
                                                           simulator.opponent_masks(), rounds, pair_seed + 1, step,
                                                           !sample, nullptr,
                                                           mixed ? simulator.opponent_feature_tables() : nullptr);
                    simulator.step(first_out.actions, second_out.actions, pair_seed, step);
                    const auto terminated = simulator.download_terminated();
                    const auto outcomes = simulator.download_outcomes();
                    for (std::size_t env = 0; env < rounds; ++env) {
                        if (finished[env] || !terminated[env]) continue;
                        finished[env] = true;
                        --remaining;
                        ++tally[outcomes[env] > 0 ? 0 : (outcomes[env] < 0 ? 1 : 2)];
                    }
                    simulator.reset_done(pair_seed, step);
                }
                table << a << ',' << b << ',' << (mixed ? std::string("mixed") : entrants[a].character) << ','
                      << (mixed ? std::string("mixed") : entrants[b].character) << ','
                      << rounds - remaining << ',' << tally[0] << ',' << tally[1] << ',' << tally[2] << '\n';
                std::cout << entrants[a].checkpoint.filename().string() << " vs "
                          << entrants[b].checkpoint.filename().string() << ": " << tally[0] << '-' << tally[1]
                          << " (" << tally[2] << " draws)\n";
            }
        }
        std::ofstream names(std::filesystem::path(output).replace_extension(".entrants.csv"));
        names << "index,checkpoint,character,provisional\n";
        for (std::size_t index = 0; index < entrants.size(); ++index) {
            names << index << ',' << entrants[index].checkpoint.generic_string() << ',' << entrants[index].character
                  << ',' << (entrants[index].provisional ? 1 : 0) << '\n';
        }
    } catch (const std::exception& error) {
        std::cerr << "arena error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
