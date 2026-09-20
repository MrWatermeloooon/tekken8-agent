#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/roster.hpp"
#include "t8_v2/sim.hpp"
#include "t8_v2/temporal.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Options {
    std::filesystem::path checkpoint;
    std::filesystem::path opponent_catalog = "data/generated/opponent_profiles.csv";
    std::filesystem::path character_moves = "data/generated/character_move_specs.csv";
    std::size_t profile_index = 0;
    std::size_t steps = 0;
    int interval_ms = 67;
    std::uint64_t seed = 2027;
    bool visual_observations = true;
};

[[noreturn]] void print_help_and_exit() {
    std::cout
        << "V2 CUDA simulator state feed for scripts/visualize_v2.py\n\n"
        << "Options:\n"
        << "  --checkpoint PATH          Optional native .t8ppo learner checkpoint\n"
        << "  --observation-mode MODE    visual (default) or privileged\n"
        << "  --profile-index N          Opponent profile row, 0 through 2099\n"
        << "  --opponent-catalog PATH    Generated opponent profile CSV\n"
        << "  --character-moves PATH     Generated character move CSV\n"
        << "  --steps N                  Stop after N decisions; 0 runs continuously\n"
        << "  --interval-ms N            Delay between decisions; 67 is about real time\n"
        << "  --seed N                   Deterministic simulator seed\n";
    std::exit(EXIT_SUCCESS);
}

std::size_t parse_size(const char* text, std::string_view option) {
    try {
        return static_cast<std::size_t>(std::stoull(text));
    } catch (const std::exception&) {
        throw std::invalid_argument(std::string(option) + " requires a non-negative integer");
    }
}

Options parse_options(int argc, char** argv) {
    Options options{};
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto next = [&]() -> const char* {
            if (++index >= argc) throw std::invalid_argument("missing value for " + std::string(argument));
            return argv[index];
        };
        if (argument == "--help" || argument == "-h") print_help_and_exit();
        if (argument == "--checkpoint") options.checkpoint = next();
        else if (argument == "--opponent-catalog") options.opponent_catalog = next();
        else if (argument == "--character-moves") options.character_moves = next();
        else if (argument == "--profile-index") options.profile_index = parse_size(next(), argument);
        else if (argument == "--steps") options.steps = parse_size(next(), argument);
        else if (argument == "--interval-ms") {
            options.interval_ms = std::stoi(next());
            if (options.interval_ms < 0) throw std::invalid_argument("--interval-ms cannot be negative");
        } else if (argument == "--seed") {
            options.seed = static_cast<std::uint64_t>(parse_size(next(), argument));
        } else if (argument == "--observation-mode") {
            const std::string_view mode = next();
            if (mode != "visual" && mode != "privileged") {
                throw std::invalid_argument("--observation-mode must be visual or privileged");
            }
            options.visual_observations = mode == "visual";
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    return options;
}

std::string_view action_name(std::int64_t action) {
    if (action < 0 || static_cast<std::size_t>(action) >= t8::v2::kActionNames.size()) return "invalid";
    return t8::v2::kActionNames[static_cast<std::size_t>(action)];
}

std::string_view move_name(int move) {
    if (move < 0 || static_cast<std::size_t>(move) >= t8::v2::kMoves.size()) return "-";
    return t8::v2::kMoves[static_cast<std::size_t>(move)].key;
}

void write_fighter(std::ostream& output, const t8::v2::FighterRuntime& fighter) {
    output << "{\"health\":" << fighter.health
           << ",\"x\":" << fighter.x
           << ",\"y\":" << fighter.y
           << ",\"guard\":" << static_cast<int>(fighter.guard)
           << ",\"move\":\"" << move_name(fighter.move) << "\""
           << ",\"move_frame\":" << fighter.move_frame
           << ",\"hitstun\":" << fighter.hitstun
           << ",\"blockstun\":" << fighter.blockstun
           << ",\"airborne\":" << fighter.airborne
           << ",\"whiffs\":" << fighter.whiffs
           << ",\"launches_taken\":" << fighter.launches_taken
           << '}';
}

void write_state(
    std::size_t step,
    std::size_t episode,
    const t8::v2::State& state,
    const t8::v2::Config& config,
    const t8::v2::OpponentProfileParameters& profile,
    std::int64_t p1_action,
    std::int64_t p2_action,
    float reward,
    bool terminated) {
    std::cout << std::fixed << std::setprecision(5)
              << "{\"type\":\"state\""
              << ",\"step\":" << step
              << ",\"episode\":" << episode
              << ",\"frame\":" << state.frame
              << ",\"max_frames\":" << config.max_frames
              << ",\"max_health\":" << config.max_health
              << ",\"stage_half_width\":" << config.stage_half_width
              << ",\"profile_id\":" << profile.id
              << ",\"character_id\":" << profile.character_id
              << ",\"archetype_id\":" << profile.archetype_id
              << ",\"variation_id\":" << profile.variation_id
              << ",\"p1_action\":\"" << action_name(p1_action) << "\""
              << ",\"p2_action\":\"" << action_name(p2_action) << "\""
              << ",\"reward_p1\":" << reward
              << ",\"distance\":" << std::abs(state.p2.x - state.p1.x)
              << ",\"round_over\":" << (state.round_over ? "true" : "false")
              << ",\"terminated\":" << (terminated ? "true" : "false")
              << ",\"winner\":" << state.winner
              << ",\"p1\":";
    write_fighter(std::cout, state.p1);
    std::cout << ",\"p2\":";
    write_fighter(std::cout, state.p2);
    std::cout << "}\n" << std::flush;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        if (t8::v2::cuda_device_count() <= 0) {
            throw std::runtime_error("the V2 visualizer feed requires an NVIDIA CUDA device");
        }

        const auto profiles = t8::v2::load_opponent_profiles_csv(options.opponent_catalog);
        const auto moves = t8::v2::load_character_move_specs_csv(options.character_moves);
        if (options.profile_index >= profiles.size()) {
            throw std::out_of_range("--profile-index exceeds the generated opponent catalog");
        }

        // Sixteen lanes retain the same CUDA launch/routing assumptions used by
        // training. The GUI displays lane zero, while the remaining lanes make
        // this a faithful GPU execution path rather than a second simulator.
        constexpr std::size_t kEnvironmentCount = 16;
        t8::v2::Config config{};
        config.timeout_ties_are_draws = true;
        config.randomize_initial_positions = true;
        t8::v2::GpuSimulatorBatch simulator(kEnvironmentCount, config);
        t8::v2::GpuScriptedOpponent opponent(kEnvironmentCount);
        t8::v2::GpuScriptedOpponent scripted_learner(kEnvironmentCount);
        std::vector<std::uint32_t> assignments(kEnvironmentCount,
                                               static_cast<std::uint32_t>(options.profile_index));

        simulator.set_character_move_specs(moves);
        opponent.set_profiles(profiles);
        opponent.set_profile_assignments(assignments);
        simulator.reset_seeded(options.seed);
        simulator.set_opponent_characters_device(
            opponent.profiles_device(), opponent.profile_count(),
            opponent.profile_assignments_device(), 1);

        const int observation_size = static_cast<int>(options.visual_observations
            ? t8::v2::kMatchupVisualObservationSize
            : t8::v2::kMatchupPrivilegedObservationSize);
        std::unique_ptr<t8::v2::GpuActorCritic> learner;
        std::unique_ptr<t8::v2::GpuTemporalMatchupEncoder> temporal;
        if (!options.checkpoint.empty()) {
            t8::v2::ActorCriticConfig actor_config{};
            actor_config.observation_size = observation_size;
            learner = std::make_unique<t8::v2::GpuActorCritic>(kEnvironmentCount, actor_config, options.seed);
            learner->load_checkpoint(options.checkpoint, false);
            temporal = std::make_unique<t8::v2::GpuTemporalMatchupEncoder>(
                kEnvironmentCount,
                options.visual_observations ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize);
        }

        std::size_t episode = 1;
        for (std::size_t step = 0; options.steps == 0 || step < options.steps; ++step) {
            const auto before = simulator.device_view();
            const float* policy_observations = options.visual_observations
                ? before.visual_observations_p1 : before.observations_p1;
            const std::int64_t* p1_actions = nullptr;
            if (learner) {
                policy_observations = temporal->encode(
                    policy_observations, opponent.profiles_device(), opponent.profile_count(),
                    opponent.profile_assignments_device(), opponent.actions_buffer_device(),
                    kEnvironmentCount);
                p1_actions = learner->forward(
                    policy_observations, before.action_masks_p1, kEnvironmentCount,
                    options.seed + 1, step, true).actions;
            } else {
                p1_actions = scripted_learner.actions_device(
                    before.observations_p1, before.action_masks_p1, kEnvironmentCount,
                    options.seed + 1, step, t8::v2::ScriptedOpponentSet::TrainingV1);
            }
            const auto* p2_actions = opponent.actions_device(
                before.observations_p2, before.action_masks_p2, kEnvironmentCount,
                options.seed + 2, step, t8::v2::ScriptedOpponentSet::TrainingV1);

            simulator.step_device_i64(p1_actions, p2_actions);
            const auto states = simulator.download_states();
            const auto rewards = simulator.download_rewards(1);
            const auto terminated = simulator.download_terminated();
            const auto learner_actions = learner
                ? learner->download_actions(kEnvironmentCount)
                : scripted_learner.download_actions(kEnvironmentCount);
            const auto opponent_actions = opponent.download_actions(kEnvironmentCount);
            write_state(step, episode, states[0], config, profiles[options.profile_index],
                        learner_actions[0], opponent_actions[0], rewards[0], terminated[0] != 0);
            // If the GUI or its launcher exits abruptly, the stdout pipe closes.
            // End this helper instead of leaving a GPU process orphaned.
            if (!std::cout.good()) break;

            if (temporal) temporal->reset_done(simulator.device_view().terminated, kEnvironmentCount);
            if (terminated[0] != 0) ++episode;
            simulator.reset_done_seeded(options.seed + step + 1);
            if (options.interval_ms > 0) {
                std::this_thread::sleep_for(std::chrono::milliseconds(options.interval_ms));
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "visualizer feed error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
