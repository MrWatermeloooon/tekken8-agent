#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/roster.hpp"
#include "t8_v2/temporal.hpp"
#include "t8_v2/training_router.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#endif

namespace {

struct Options {
    std::size_t environments = 4096;
    std::size_t horizon = 128;
    std::size_t updates = 100;
    int epochs = 4;
    std::size_t minibatch_size = 4096;
    float learning_rate = 3e-4F;
    float gamma = 0.99F;
    float gae_lambda = 0.95F;
    std::uint64_t seed = 2027;
    std::size_t checkpoint_interval = 1;
    std::size_t evaluation_interval = 10;
    std::size_t evaluation_episodes = 256;
    bool sparse_reward = false;
    bool visual_observations = false;
    bool full_roster = true;
    int curriculum_stage = 0;
    std::filesystem::path opponent_catalog = "data/generated/opponent_profiles.csv";
    std::filesystem::path character_move_catalog = "data/generated/character_move_specs.csv";
    std::filesystem::path run_directory;
    std::filesystem::path resume_checkpoint;
    bool run_directory_explicit = false;
};

std::size_t parse_size(const char* text, std::string_view option) {
    const auto value = std::stoull(text);
    if (value == 0) throw std::invalid_argument(std::string(option) + " must be positive");
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options{};
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        const auto next = [&]() -> const char* {
            if (++index >= argc) throw std::invalid_argument("missing value for " + std::string(argument));
            return argv[index];
        };
        if (argument == "--envs") options.environments = parse_size(next(), argument);
        else if (argument == "--horizon") options.horizon = parse_size(next(), argument);
        else if (argument == "--updates") options.updates = parse_size(next(), argument);
        else if (argument == "--epochs") {
            const auto epochs = parse_size(next(), argument);
            if (epochs > static_cast<std::size_t>(std::numeric_limits<int>::max())) {
                throw std::invalid_argument("--epochs exceeds the supported range");
            }
            options.epochs = static_cast<int>(epochs);
        }
        else if (argument == "--minibatch") options.minibatch_size = parse_size(next(), argument);
        else if (argument == "--learning-rate") options.learning_rate = std::stof(next());
        else if (argument == "--gamma") options.gamma = std::stof(next());
        else if (argument == "--gae-lambda") options.gae_lambda = std::stof(next());
        else if (argument == "--seed") options.seed = std::stoull(next());
        else if (argument == "--checkpoint-interval") options.checkpoint_interval = parse_size(next(), argument);
        else if (argument == "--eval-interval") options.evaluation_interval = parse_size(next(), argument);
        else if (argument == "--eval-episodes") options.evaluation_episodes = parse_size(next(), argument);
        else if (argument == "--opponents") {
            const std::string_view mode = next();
            if (mode != "roster" && mode != "legacy") {
                throw std::invalid_argument("--opponents must be roster or legacy");
            }
            options.full_roster = mode == "roster";
        }
        else if (argument == "--opponent-catalog") options.opponent_catalog = next();
        else if (argument == "--character-moves") options.character_move_catalog = next();
        else if (argument == "--curriculum-stage") {
            const std::string_view stage = next();
            if (stage == "auto") options.curriculum_stage = 0;
            else {
                const auto parsed = std::stoi(std::string(stage));
                if (parsed < 1 || parsed > 4) {
                    throw std::invalid_argument("--curriculum-stage must be auto or 1 through 4");
                }
                options.curriculum_stage = parsed;
            }
        }
        else if (argument == "--observation-mode") {
            const std::string_view mode = next();
            if (mode != "privileged" && mode != "visual") {
                throw std::invalid_argument("--observation-mode must be privileged or visual");
            }
            options.visual_observations = mode == "visual";
        }
        else if (argument == "--reward") {
            const std::string_view reward = next();
            if (reward != "shaped" && reward != "sparse") {
                throw std::invalid_argument("--reward must be shaped or sparse");
            }
            options.sparse_reward = reward == "sparse";
        } else if (argument == "--run-dir") {
            options.run_directory = next();
            options.run_directory_explicit = true;
        }
        else if (argument == "--resume") options.resume_checkpoint = next();
        else if (argument == "--smoke") {
            options.environments = 512;
            options.horizon = 32;
            options.updates = 2;
            options.epochs = 2;
            options.minibatch_size = 1024;
            options.checkpoint_interval = 1;
            options.evaluation_interval = 1;
            options.evaluation_episodes = 32;
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    if (!std::isfinite(options.learning_rate) || options.learning_rate <= 0.0F) {
        throw std::invalid_argument("--learning-rate must be finite and positive");
    }
    if (!std::isfinite(options.gamma) || options.gamma < 0.0F || options.gamma > 1.0F ||
        !std::isfinite(options.gae_lambda) || options.gae_lambda < 0.0F ||
        options.gae_lambda > 1.0F) {
        throw std::invalid_argument("--gamma and --gae-lambda must be finite and in [0, 1]");
    }
    if (options.environments > std::numeric_limits<std::size_t>::max() / options.horizon) {
        throw std::invalid_argument("rollout sample count overflows size_t");
    }
    const std::size_t rollout_samples = options.environments * options.horizon;
    if (options.minibatch_size > rollout_samples) {
        throw std::invalid_argument("minibatch cannot exceed rollout sample count");
    }
    if (options.updates > std::numeric_limits<std::uint64_t>::max() / rollout_samples) {
        throw std::invalid_argument("environment step counter overflows uint64_t");
    }
    const std::size_t balanced_group = 2 * t8::v2::kEvaluationStyleCount;
    if (options.environments % balanced_group != 0) {
        throw std::invalid_argument("--envs must be a multiple of 16 for side/style-balanced training");
    }
    if (options.evaluation_episodes % balanced_group != 0) {
        throw std::invalid_argument("--eval-episodes must be a multiple of 16 for side/style-balanced evaluation");
    }
    if (options.run_directory.empty() && !options.resume_checkpoint.empty()) {
        options.run_directory = options.resume_checkpoint.parent_path().parent_path();
    }
    if (options.run_directory.empty()) {
        options.run_directory = std::filesystem::path("runs") /
            (std::string(options.full_roster ? "roster_curriculum_" : "phase0_scripted_") +
             (options.visual_observations ? "visual_" : "privileged_") +
             (options.sparse_reward ? "sparse" : "shaped") +
             "_seed" + std::to_string(options.seed));
    }
    return options;
}

std::pair<t8::v2::CurriculumStage, std::uint32_t> curriculum_for_update(
    const Options& options,
    std::size_t update) {
    if (options.curriculum_stage != 0) {
        const auto stage = static_cast<t8::v2::CurriculumStage>(options.curriculum_stage);
        return {stage, stage == t8::v2::CurriculumStage::CharacterGroups
            ? static_cast<std::uint32_t>(t8::v2::Rushdown | t8::v2::StanceHeavy |
                                         t8::v2::Grappler | t8::v2::KeepOut |
                                         t8::v2::Evasive | t8::v2::Specialist)
            : 0U};
    }
    const std::size_t stage_span = std::max<std::size_t>(1, (options.updates + 3) / 4);
    if (update <= stage_span) return {t8::v2::CurriculumStage::JunFundamentals, 0U};
    if (update <= 2 * stage_span) {
        constexpr std::array<std::uint32_t, 7> groups = {
            t8::v2::Fundamentals, t8::v2::Rushdown, t8::v2::StanceHeavy,
            t8::v2::Grappler, t8::v2::KeepOut, t8::v2::Evasive,
            t8::v2::Specialist};
        return {t8::v2::CurriculumStage::CharacterGroups,
                groups[(update - stage_span - 1) % groups.size()]};
    }
    if (update <= 3 * stage_span) return {t8::v2::CurriculumStage::FullRoster, 0U};
    return {t8::v2::CurriculumStage::AdversarialLeague, 0U};
}

std::filesystem::path resolve_generated_catalog(
    const std::filesystem::path& requested,
    const std::filesystem::path& filename,
    const char* executable) {
    if (std::filesystem::exists(requested)) return requested;
    const std::array candidates = {
        std::filesystem::current_path().parent_path() / filename,
        std::filesystem::absolute(executable).parent_path().parent_path().parent_path() / filename,
    };
    for (const auto& candidate : candidates) {
        if (std::filesystem::exists(candidate)) return candidate;
    }
    return requested;
}

struct Evaluation {
    t8::v2::GpuEpisodeSummary total{};
    t8::v2::GpuEpisodeSummary as_p1{};
    t8::v2::GpuEpisodeSummary as_p2{};
    double win_rate = 0.0;
};

void merge_summary(t8::v2::GpuEpisodeSummary& target, const t8::v2::GpuEpisodeSummary& source) {
    target.episodes += source.episodes;
    target.wins += source.wins;
    target.losses += source.losses;
    target.draws += source.draws;
    target.timeouts += source.timeouts;
    target.stalemates += source.stalemates;
    target.no_action_timeouts += source.no_action_timeouts;
    target.total_frames += source.total_frames;
    target.total_damage_dealt += source.total_damage_dealt;
    target.total_damage_taken += source.total_damage_taken;
    for (std::size_t style = 0; style < t8::v2::kEvaluationStyleCount; ++style) {
        target.style_episodes[style] += source.style_episodes[style];
        target.style_wins[style] += source.style_wins[style];
        target.style_losses[style] += source.style_losses[style];
        target.style_draws[style] += source.style_draws[style];
    }
}

t8::v2::GpuEpisodeSummary evaluate_side(
    t8::v2::GpuActorCritic& learner,
    std::size_t requested_episodes,
    std::uint64_t seed,
    int learner_player,
    bool visual_observations,
    bool full_roster,
    std::span<const t8::v2::OpponentProfileParameters> profiles,
    std::span<const t8::v2::CharacterMoveParameters> character_moves,
    t8::v2::MatchupScheduler* scheduler) {
    t8::v2::GpuEpisodeSummary aggregate{};
    std::size_t completed = 0;
    while (completed < requested_episodes) {
        const std::size_t environments = std::min(
            {requested_episodes - completed, learner.capacity(), std::size_t{256}});
        t8::v2::Config evaluation_config{};
        evaluation_config.timeout_ties_are_draws = true;
        evaluation_config.randomize_initial_positions = true;
        t8::v2::GpuSimulatorBatch simulator(environments, evaluation_config);
        simulator.reset_seeded(seed);
        t8::v2::GpuScriptedOpponent opponent(environments);
        std::vector<std::uint32_t> assignments;
        std::unique_ptr<t8::v2::GpuTemporalMatchupEncoder> temporal;
        if (full_roster) {
            if (profiles.empty() || character_moves.empty() || scheduler == nullptr) {
                throw std::logic_error(
                    "full-roster evaluation requires profiles, character moves, and a scheduler");
            }
            simulator.set_character_move_specs(character_moves);
            opponent.set_profiles(profiles);
            assignments = scheduler->sample_profile_indices(environments, false);
            opponent.set_profile_assignments(assignments);
            simulator.set_opponent_characters_device(
                opponent.profiles_device(), opponent.profile_count(),
                opponent.profile_assignments_device(), learner_player);
            temporal = std::make_unique<t8::v2::GpuTemporalMatchupEncoder>(
                environments, visual_observations ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize);
        }
        const std::size_t max_decisions =
            (simulator.config().max_frames + simulator.config().decision_frames - 1) /
            simulator.config().decision_frames;
        for (std::size_t decision = 0; decision < max_decisions; ++decision) {
            const auto before = simulator.device_view();
            const float* learner_p1_observations = visual_observations
                ? before.visual_observations_p1 : before.observations_p1;
            const float* learner_p2_observations = visual_observations
                ? before.visual_observations_p2 : before.observations_p2;
            if (learner_player == 1) {
                const float* policy_observations = learner_p1_observations;
                if (temporal) {
                    policy_observations = temporal->encode(
                        policy_observations, opponent.profiles_device(), opponent.profile_count(),
                        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
                }
                const auto p1 = learner.forward(
                    policy_observations, before.action_masks_p1,
                    environments, seed, decision, true);
                const auto* p2 = opponent.actions_device(
                    before.observations_p2, before.action_masks_p2,
                    environments, seed + 1, decision,
                    t8::v2::ScriptedOpponentSet::HeldOutV2);
                simulator.step_device_i64(p1.actions, p2);
            } else {
                const auto* p1 = opponent.actions_device(
                    before.observations_p1, before.action_masks_p1,
                    environments, seed + 1, decision,
                    t8::v2::ScriptedOpponentSet::HeldOutV2);
                const float* policy_observations = learner_p2_observations;
                if (temporal) {
                    policy_observations = temporal->encode(
                        policy_observations, opponent.profiles_device(), opponent.profile_count(),
                        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
                }
                const auto p2 = learner.forward(
                    policy_observations, before.action_masks_p2,
                    environments, seed, decision, true);
                simulator.step_device_i64(p1, p2.actions);
            }
        }
        const auto summary = simulator.summarize_episodes(learner_player);
        if (summary.episodes != environments) {
            throw std::runtime_error("held-out evaluation did not terminate every GPU lane");
        }
        merge_summary(aggregate, summary);
        if (full_roster) {
            const auto winners = simulator.download_winners();
            for (std::size_t lane = 0; lane < environments; ++lane) {
                const bool win = winners[lane] == learner_player;
                const bool draw = winners[lane] == 0;
                scheduler->record(assignments[lane], win ? 1U : 0U,
                                  !win && !draw ? 1U : 0U, draw ? 1U : 0U);
            }
        }
        completed += environments;
        seed += 0x9e3779b97f4a7c15ULL;
    }
    return aggregate;
}

Evaluation evaluate(
    t8::v2::GpuActorCritic& learner,
    std::size_t requested_episodes,
    std::uint64_t seed,
    bool visual_observations,
    bool full_roster,
    std::span<const t8::v2::OpponentProfileParameters> profiles,
    std::span<const t8::v2::CharacterMoveParameters> character_moves,
    t8::v2::MatchupScheduler* scheduler) {
    Evaluation result{};
    t8::v2::CurriculumStage previous_stage = t8::v2::CurriculumStage::FullRoster;
    std::uint32_t previous_groups = 0;
    if (full_roster) {
        if (scheduler == nullptr) throw std::logic_error("full-roster evaluation requires a scheduler");
        previous_stage = scheduler->stage();
        previous_groups = scheduler->active_group_mask();
        scheduler->set_stage(t8::v2::CurriculumStage::FullRoster);
    }
    const std::size_t p1_episodes = requested_episodes / 2;
    const std::size_t p2_episodes = requested_episodes - p1_episodes;
    result.as_p1 = evaluate_side(learner, p1_episodes, seed, 1, visual_observations,
                                 full_roster, profiles, character_moves, scheduler);
    result.as_p2 = evaluate_side(learner, p2_episodes, seed + 100'000, 2, visual_observations,
                                 full_roster, profiles, character_moves, scheduler);
    if (full_roster) scheduler->set_stage(previous_stage, previous_groups);
    merge_summary(result.total, result.as_p1);
    merge_summary(result.total, result.as_p2);
    result.win_rate = result.total.episodes == 0 ? 0.0 :
        static_cast<double>(result.total.wins) / static_cast<double>(result.total.episodes);
    return result;
}

void append_summary_json(
    std::ostream& output,
    std::string_view name,
    const t8::v2::GpuEpisodeSummary& summary) {
    const double denominator = summary.episodes == 0 ? 1.0 : static_cast<double>(summary.episodes);
    output << '"' << name << "\":{\"episodes\":" << summary.episodes
           << ",\"wins\":" << summary.wins
           << ",\"losses\":" << summary.losses
           << ",\"draws\":" << summary.draws
           << ",\"win_rate\":" << static_cast<double>(summary.wins) / denominator
           << ",\"timeouts\":" << summary.timeouts
           << ",\"stalemates\":" << summary.stalemates
           << ",\"no_action_timeouts\":" << summary.no_action_timeouts
           << ",\"mean_frames\":" << static_cast<double>(summary.total_frames) / denominator
           << ",\"mean_damage_dealt\":" << summary.total_damage_dealt / denominator
           << ",\"mean_damage_taken\":" << summary.total_damage_taken / denominator
           << ",\"styles\":[";
    for (std::size_t style = 0; style < t8::v2::kEvaluationStyleCount; ++style) {
        if (style != 0) output << ',';
        const double style_denominator = summary.style_episodes[style] == 0
            ? 1.0 : static_cast<double>(summary.style_episodes[style]);
        output << "{\"style\":" << style
               << ",\"episodes\":" << summary.style_episodes[style]
               << ",\"wins\":" << summary.style_wins[style]
               << ",\"losses\":" << summary.style_losses[style]
               << ",\"draws\":" << summary.style_draws[style]
               << ",\"win_rate\":" << static_cast<double>(summary.style_wins[style]) / style_denominator
               << '}';
    }
    output << "]}";
}

void append_metrics(
    const std::filesystem::path& path,
    std::size_t update,
    std::uint64_t environment_steps,
    std::string_view reward_mode,
    std::string_view observation_mode,
    const t8::v2::PpoUpdateMetrics& metrics,
    double elapsed_seconds,
    const std::optional<Evaluation>& evaluation) {
    std::ostringstream row;
    row << std::setprecision(9)
           << "{\"update\":" << update
           << ",\"environment_steps\":" << environment_steps
           << ",\"reward_mode\":\"" << reward_mode << "\""
           << ",\"observation_mode\":\"" << observation_mode << "\""
           << ",\"benchmark\":\"heldout_v2_side_balanced\""
           << ",\"elapsed_seconds\":" << elapsed_seconds
           << ",\"policy_loss\":" << metrics.policy_loss
           << ",\"value_loss\":" << metrics.value_loss
           << ",\"entropy\":" << metrics.entropy
           << ",\"approximate_kl\":" << metrics.approximate_kl
           << ",\"clip_fraction\":" << metrics.clip_fraction
           << ",\"gradient_norm\":" << metrics.gradient_norm
           << ",\"minibatches\":" << metrics.minibatches
           << ",\"epochs_completed\":" << metrics.epochs_completed
           << ",\"kl_early_stop\":" << (metrics.early_stopped ? "true" : "false");
    if (evaluation) {
        row << ",\"evaluation\":{";
        append_summary_json(row, "total", evaluation->total);
        row << ',';
        append_summary_json(row, "as_p1", evaluation->as_p1);
        row << ',';
        append_summary_json(row, "as_p2", evaluation->as_p2);
        row << '}';
    }
    row << "}\n";

    // Rewrite the small JSONL ledger through a temporary file. A trainer killed
    // during an append must leave either the previous complete ledger or the new
    // complete ledger, never a partially allocated/null-filled final record.
    auto temporary = path;
    temporary += ".tmp";
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("could not write metrics: " + temporary.string());
    if (std::filesystem::exists(path)) {
        std::ifstream existing(path, std::ios::binary);
        if (!existing) throw std::runtime_error("could not read metrics: " + path.string());
        output << existing.rdbuf();
        if (existing.bad()) throw std::runtime_error("metrics read failed: " + path.string());
    }
    const auto completed_row = row.str();
    output.write(completed_row.data(), static_cast<std::streamsize>(completed_row.size()));
    output.flush();
    if (!output) throw std::runtime_error("metrics write failed: " + temporary.string());
    output.close();
#ifdef _WIN32
    if (!MoveFileExW(
            temporary.c_str(), path.c_str(),
            MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
        throw std::runtime_error("could not atomically replace metrics: " + path.string());
    }
#else
    std::filesystem::rename(temporary, path);
#endif
}

struct ResumeState {
    std::size_t completed_update = 0;
    std::uint64_t environment_steps = 0;
    double elapsed_seconds = 0.0;
    std::vector<t8::v2::State> simulator_states;
    std::vector<std::uint32_t> profile_assignments;
    std::vector<std::int64_t> opponent_actions;
    std::vector<t8::v2::MatchupStats> scheduler_stats;
    std::uint64_t scheduler_random_state = 0;
    std::optional<t8::v2::TemporalEncoderState> temporal_state;
};

template <typename T>
void write_value(std::ostream& output, const T& value) {
    static_assert(std::is_trivially_copyable_v<T>);
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

template <typename T>
T read_value(std::istream& input, const std::filesystem::path& path) {
    static_assert(std::is_trivially_copyable_v<T>);
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("truncated trainer state: " + path.string());
    return value;
}

template <typename T>
void write_vector(std::ostream& output, std::span<const T> values) {
    static_assert(std::is_trivially_copyable_v<T>);
    write_value(output, static_cast<std::uint64_t>(values.size()));
    output.write(reinterpret_cast<const char*>(values.data()),
                 static_cast<std::streamsize>(sizeof(T) * values.size()));
}

template <typename T>
std::vector<T> read_vector(std::istream& input, const std::filesystem::path& path) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto count = read_value<std::uint64_t>(input, path);
    if (count > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        throw std::runtime_error("trainer-state vector is too large: " + path.string());
    }
    std::vector<T> result(static_cast<std::size_t>(count));
    input.read(reinterpret_cast<char*>(result.data()),
               static_cast<std::streamsize>(sizeof(T) * result.size()));
    if (!input) throw std::runtime_error("truncated trainer-state vector: " + path.string());
    return result;
}

void write_fighter(std::ostream& output, const t8::v2::FighterRuntime& fighter) {
    write_value(output, fighter.health);
    write_value(output, fighter.x);
    write_value(output, fighter.y);
    write_value(output, static_cast<std::uint8_t>(fighter.guard));
    write_value(output, static_cast<std::int32_t>(fighter.move));
    write_value(output, static_cast<std::int32_t>(fighter.move_frame));
    write_value(output, static_cast<std::uint8_t>(fighter.has_hit));
    write_value(output, static_cast<std::int32_t>(fighter.hitstun));
    write_value(output, static_cast<std::int32_t>(fighter.blockstun));
    write_value(output, static_cast<std::int32_t>(fighter.airborne));
    write_value(output, static_cast<std::int32_t>(fighter.throw_break_active));
    write_value(output, static_cast<std::int32_t>(fighter.launches_taken));
    write_value(output, static_cast<std::int32_t>(fighter.whiffs));
}

t8::v2::FighterRuntime read_fighter(std::istream& input, const std::filesystem::path& path) {
    t8::v2::FighterRuntime fighter{};
    fighter.health = read_value<double>(input, path);
    fighter.x = read_value<double>(input, path);
    fighter.y = read_value<double>(input, path);
    fighter.guard = static_cast<t8::v2::HitLevel>(read_value<std::uint8_t>(input, path));
    fighter.move = read_value<std::int32_t>(input, path);
    fighter.move_frame = read_value<std::int32_t>(input, path);
    fighter.has_hit = read_value<std::uint8_t>(input, path) != 0;
    fighter.hitstun = read_value<std::int32_t>(input, path);
    fighter.blockstun = read_value<std::int32_t>(input, path);
    fighter.airborne = read_value<std::int32_t>(input, path);
    fighter.throw_break_active = read_value<std::int32_t>(input, path);
    fighter.launches_taken = read_value<std::int32_t>(input, path);
    fighter.whiffs = read_value<std::int32_t>(input, path);
    return fighter;
}

std::filesystem::path trainer_state_path(const std::filesystem::path& checkpoint) {
    auto path = checkpoint;
    path.replace_extension(".t8state");
    return path;
}

void save_trainer_state(
    const std::filesystem::path& path,
    const Options& options,
    std::size_t completed_update,
    std::uint64_t environment_steps,
    double elapsed_seconds,
    const std::vector<t8::v2::State>& states,
    std::span<const std::uint32_t> profile_assignments = {},
    std::span<const std::int64_t> opponent_actions = {},
    std::span<const t8::v2::MatchupStats> scheduler_stats = {},
    std::uint64_t scheduler_random_state = 0,
    const t8::v2::TemporalEncoderState* temporal_state = nullptr) {
    if (std::filesystem::exists(path)) {
        throw std::runtime_error("refusing to overwrite trainer state: " + path.string());
    }
    auto temporary = path;
    temporary += ".tmp";
    std::error_code remove_error;
    std::filesystem::remove(temporary, remove_error);
    std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
    if (!output) throw std::runtime_error("could not write trainer state: " + temporary.string());
    constexpr std::array<char, 8> magic = {'T', '8', 'R', 'U', 'N', 'V', '2', '\0'};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));
    write_value(output, std::uint32_t{3});
    write_value(output, static_cast<std::uint64_t>(completed_update));
    write_value(output, environment_steps);
    write_value(output, elapsed_seconds);
    write_value(output, options.seed);
    write_value(output, static_cast<std::uint64_t>(options.environments));
    write_value(output, static_cast<std::uint64_t>(options.horizon));
    write_value(output, static_cast<std::int32_t>(options.epochs));
    write_value(output, static_cast<std::uint64_t>(options.minibatch_size));
    write_value(output, options.learning_rate);
    write_value(output, options.gamma);
    write_value(output, options.gae_lambda);
    write_value(output, static_cast<std::uint64_t>(options.checkpoint_interval));
    write_value(output, static_cast<std::uint64_t>(options.evaluation_interval));
    write_value(output, static_cast<std::uint64_t>(options.evaluation_episodes));
    write_value(output, static_cast<std::uint8_t>(options.sparse_reward));
    write_value(output, static_cast<std::uint8_t>(options.visual_observations));
    write_value(output, static_cast<std::uint8_t>(options.full_roster));
    write_value(output, static_cast<std::int32_t>(options.curriculum_stage));
    write_value(output, static_cast<std::uint64_t>(states.size()));
    for (const auto& state : states) {
        write_fighter(output, state.p1);
        write_fighter(output, state.p2);
        write_value(output, static_cast<std::int32_t>(state.frame));
        write_value(output, static_cast<std::int32_t>(state.stall_frames));
        write_value(output, static_cast<std::int32_t>(state.no_action_frames));
        write_value(output, static_cast<std::uint8_t>(state.round_over));
        write_value(output, static_cast<std::int32_t>(state.winner));
    }
    if (options.full_roster) {
        if (profile_assignments.size() != options.environments ||
            opponent_actions.size() != options.environments ||
            scheduler_stats.size() != t8::v2::kOpponentProfileCount || temporal_state == nullptr) {
            throw std::invalid_argument("full-roster trainer state is incomplete");
        }
        write_vector(output, profile_assignments);
        write_vector(output, opponent_actions);
        write_value(output, scheduler_random_state);
        write_value(output, static_cast<std::uint64_t>(scheduler_stats.size()));
        for (const auto& value : scheduler_stats) {
            write_value(output, value.episodes); write_value(output, value.wins);
            write_value(output, value.losses); write_value(output, value.draws);
            write_value(output, value.best_win_rate); write_value(output, value.recent_win_rate);
            write_value(output, value.exploit_severity);
        }
        write_vector(output, std::span<const float>(temporal_state->history));
        write_vector(output, std::span<const std::int64_t>(temporal_state->previous_actions));
        write_vector(output, std::span<const std::int32_t>(temporal_state->repeated_action_frames));
        write_vector(output, std::span<const float>(temporal_state->previous_own_health));
        write_vector(output, std::span<const float>(temporal_state->previous_opponent_health));
        write_vector(output, std::span<const float>(temporal_state->previous_distance));
        write_vector(output, std::span<const std::uint8_t>(temporal_state->valid));
    }
    output.flush();
    if (!output) throw std::runtime_error("trainer-state write failed: " + temporary.string());
    output.close();
    std::filesystem::rename(temporary, path);
}

ResumeState load_trainer_state(const std::filesystem::path& path, const Options& options) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("missing trainer state for exact resume: " + path.string());
    std::array<char, 8> magic{};
    input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    constexpr std::array<char, 8> expected_magic = {'T', '8', 'R', 'U', 'N', 'V', '2', '\0'};
    const auto version = read_value<std::uint32_t>(input, path);
    if (!input || magic != expected_magic || (version < 1 || version > 3)) {
        throw std::runtime_error("unsupported trainer state: " + path.string());
    }
    ResumeState result{};
    result.completed_update = static_cast<std::size_t>(read_value<std::uint64_t>(input, path));
    result.environment_steps = read_value<std::uint64_t>(input, path);
    result.elapsed_seconds = read_value<double>(input, path);
    const auto seed = read_value<std::uint64_t>(input, path);
    const auto environments = read_value<std::uint64_t>(input, path);
    const auto horizon = read_value<std::uint64_t>(input, path);
    const auto epochs = read_value<std::int32_t>(input, path);
    const auto minibatch = read_value<std::uint64_t>(input, path);
    const auto learning_rate = read_value<float>(input, path);
    const auto gamma = read_value<float>(input, path);
    const auto gae_lambda = read_value<float>(input, path);
    const auto checkpoint_interval = read_value<std::uint64_t>(input, path);
    const auto evaluation_interval = read_value<std::uint64_t>(input, path);
    const auto evaluation_episodes = read_value<std::uint64_t>(input, path);
    const auto sparse_reward = read_value<std::uint8_t>(input, path) != 0;
    const bool visual_observations = version >= 2
        ? read_value<std::uint8_t>(input, path) != 0 : false;
    const bool full_roster = version >= 3
        ? read_value<std::uint8_t>(input, path) != 0 : false;
    const int curriculum_stage = version >= 3
        ? read_value<std::int32_t>(input, path) : 0;
    const auto state_count = read_value<std::uint64_t>(input, path);
    if (seed != options.seed || environments != options.environments || horizon != options.horizon ||
        epochs != options.epochs || minibatch != options.minibatch_size ||
        learning_rate != options.learning_rate || gamma != options.gamma ||
        gae_lambda != options.gae_lambda || checkpoint_interval != options.checkpoint_interval ||
        evaluation_interval != options.evaluation_interval ||
        evaluation_episodes != options.evaluation_episodes || sparse_reward != options.sparse_reward ||
        visual_observations != options.visual_observations ||
        full_roster != options.full_roster || curriculum_stage != options.curriculum_stage ||
        state_count != options.environments) {
        throw std::runtime_error("resume options do not match saved trainer state: " + path.string());
    }
    result.simulator_states.resize(static_cast<std::size_t>(state_count));
    for (auto& state : result.simulator_states) {
        state.p1 = read_fighter(input, path);
        state.p2 = read_fighter(input, path);
        state.frame = read_value<std::int32_t>(input, path);
        state.stall_frames = read_value<std::int32_t>(input, path);
        state.no_action_frames = read_value<std::int32_t>(input, path);
        state.round_over = read_value<std::uint8_t>(input, path) != 0;
        state.winner = read_value<std::int32_t>(input, path);
    }
    if (full_roster) {
        result.profile_assignments = read_vector<std::uint32_t>(input, path);
        result.opponent_actions = read_vector<std::int64_t>(input, path);
        result.scheduler_random_state = read_value<std::uint64_t>(input, path);
        const auto stats_count = read_value<std::uint64_t>(input, path);
        if (stats_count != t8::v2::kOpponentProfileCount) {
            throw std::runtime_error("trainer-state scheduler profile count mismatch: " + path.string());
        }
        result.scheduler_stats.resize(static_cast<std::size_t>(stats_count));
        for (auto& value : result.scheduler_stats) {
            value.episodes = read_value<std::uint64_t>(input, path);
            value.wins = read_value<std::uint64_t>(input, path);
            value.losses = read_value<std::uint64_t>(input, path);
            value.draws = read_value<std::uint64_t>(input, path);
            value.best_win_rate = read_value<double>(input, path);
            value.recent_win_rate = read_value<double>(input, path);
            value.exploit_severity = read_value<double>(input, path);
        }
        t8::v2::TemporalEncoderState temporal{};
        temporal.history = read_vector<float>(input, path);
        temporal.previous_actions = read_vector<std::int64_t>(input, path);
        temporal.repeated_action_frames = read_vector<std::int32_t>(input, path);
        temporal.previous_own_health = read_vector<float>(input, path);
        temporal.previous_opponent_health = read_vector<float>(input, path);
        temporal.previous_distance = read_vector<float>(input, path);
        temporal.valid = read_vector<std::uint8_t>(input, path);
        result.temporal_state = std::move(temporal);
        if (result.profile_assignments.size() != options.environments ||
            result.opponent_actions.size() != options.environments) {
            throw std::runtime_error("trainer-state roster lane count mismatch: " + path.string());
        }
    }
    char trailing = 0;
    if (input.read(&trailing, 1)) {
        throw std::runtime_error("trainer state contains trailing data: " + path.string());
    }
    if (!input.eof()) throw std::runtime_error("trainer-state read failed: " + path.string());
    return result;
}

void validate_metrics_for_resume(const std::filesystem::path& path, std::size_t completed_update) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("missing metrics ledger for exact resume: " + path.string());
    }
    std::ifstream input(path);
    if (!input) throw std::runtime_error("could not validate metrics for resume: " + path.string());
    std::size_t previous = 0;
    std::string line;
    while (std::getline(input, line)) {
        if (line.size() < 2 || line.front() != '{' || line.back() != '}' ||
            line.find('\0') != std::string::npos) {
            throw std::runtime_error("incomplete metrics row: " + path.string());
        }
        const std::string marker = "\"update\":";
        const auto position = line.find(marker);
        if (position == std::string::npos) throw std::runtime_error("invalid metrics row: " + path.string());
        const auto number_begin = position + marker.size();
        auto number_end = number_begin;
        while (number_end < line.size() &&
               std::isdigit(static_cast<unsigned char>(line[number_end]))) {
            ++number_end;
        }
        if (number_end == number_begin ||
            (line[number_end] != ',' && line[number_end] != '}')) {
            throw std::runtime_error("invalid metrics update field: " + path.string());
        }
        const auto update = static_cast<std::size_t>(
            std::stoull(line.substr(number_begin, number_end - number_begin)));
        if (update != previous + 1) {
            throw std::runtime_error("metrics updates are not contiguous: " + path.string());
        }
        if (update > completed_update) {
            throw std::runtime_error("metrics are newer than the resume checkpoint: " + path.string());
        }
        previous = update;
    }
    if (input.bad()) throw std::runtime_error("metrics read failed: " + path.string());
    if (previous != completed_update) {
        throw std::runtime_error("metrics do not end at the resume checkpoint update: " + path.string());
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        Options options = parse_options(argc, argv);
        if (options.full_roster) {
            options.opponent_catalog = resolve_generated_catalog(
                options.opponent_catalog, "data/generated/opponent_profiles.csv", argv[0]);
            options.character_move_catalog = resolve_generated_catalog(
                options.character_move_catalog, "data/generated/character_move_specs.csv", argv[0]);
        }
        const auto metrics_path = options.run_directory / "metrics.jsonl";
        std::optional<ResumeState> resume_state;
        if (!options.resume_checkpoint.empty()) {
            resume_state = load_trainer_state(trainer_state_path(options.resume_checkpoint), options);
            validate_metrics_for_resume(metrics_path, resume_state->completed_update);
            if (resume_state->completed_update >= options.updates) {
                throw std::invalid_argument("--updates must exceed the completed checkpoint update");
            }
        } else {
            const auto checkpoint_directory = options.run_directory / "checkpoints";
            const bool has_checkpoints = std::filesystem::exists(checkpoint_directory) &&
                std::filesystem::directory_iterator(checkpoint_directory) !=
                    std::filesystem::directory_iterator{};
            if (std::filesystem::exists(metrics_path) || has_checkpoints) {
                throw std::runtime_error(
                    "run directory already contains training artifacts; use --resume or a new --run-dir");
            }
        }
        std::filesystem::create_directories(options.run_directory / "checkpoints");
        std::vector<t8::v2::OpponentProfileParameters> roster_profiles;
        std::vector<t8::v2::CharacterMoveParameters> roster_character_moves;
        std::unique_ptr<t8::v2::MatchupScheduler> matchup_scheduler;
        if (options.full_roster) {
            roster_profiles = t8::v2::load_opponent_profiles_csv(options.opponent_catalog);
            roster_character_moves =
                t8::v2::load_character_move_specs_csv(options.character_move_catalog);
            matchup_scheduler = std::make_unique<t8::v2::MatchupScheduler>(roster_profiles, options.seed + 700'000);
        }
        const std::size_t policy_capacity = std::max(options.environments, options.minibatch_size);
        t8::v2::ActorCriticConfig actor_config{};
        if (options.full_roster) {
            actor_config.observation_size = static_cast<int>(options.visual_observations
                ? t8::v2::kMatchupVisualObservationSize : t8::v2::kMatchupPrivilegedObservationSize);
        } else {
            actor_config.observation_size = static_cast<int>(options.visual_observations
                ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize);
        }
        t8::v2::Config training_config{};
        training_config.timeout_ties_are_draws = true;
        training_config.randomize_initial_positions = true;
        t8::v2::GpuSimulatorBatch simulator(options.environments, training_config);
        t8::v2::GpuActorCritic learner(policy_capacity, actor_config, options.seed);
        t8::v2::GpuScriptedOpponent opponent(options.environments);
        t8::v2::GpuLearnerSideRouter side_router(options.environments);
        t8::v2::GpuRolloutBuffer rollout(options.environments, options.horizon, actor_config);
        std::unique_ptr<t8::v2::GpuTemporalMatchupEncoder> temporal;
        if (options.full_roster) {
            simulator.set_character_move_specs(roster_character_moves);
            opponent.set_profiles(roster_profiles);
            temporal = std::make_unique<t8::v2::GpuTemporalMatchupEncoder>(
                options.environments,
                options.visual_observations ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize);
        }
        if (resume_state) {
            learner.load_checkpoint(options.resume_checkpoint);
            simulator.upload_states(resume_state->simulator_states);
            if (options.full_roster) {
                opponent.set_profile_assignments(resume_state->profile_assignments);
                simulator.set_opponent_characters_device(
                    opponent.profiles_device(), opponent.profile_count(),
                    opponent.profile_assignments_device(), 0);
                opponent.set_action_history(resume_state->opponent_actions);
                matchup_scheduler->restore_state(
                    resume_state->scheduler_stats, resume_state->scheduler_random_state);
                temporal->upload_state(*resume_state->temporal_state);
            }
        }

        t8::v2::PpoUpdateConfig update_config{};
        update_config.epochs = options.epochs;
        update_config.minibatch_size = options.minibatch_size;
        update_config.learning_rate = options.learning_rate;
        const auto started = std::chrono::steady_clock::now();
        const double elapsed_before_resume = resume_state ? resume_state->elapsed_seconds : 0.0;
        const std::string reward_mode = options.sparse_reward ? "sparse" : "shaped";
        const std::string observation_mode = options.full_roster
            ? (options.visual_observations ? "visual_matchup_temporal" : "privileged_matchup_temporal")
            : (options.visual_observations ? "visual" : "privileged");
        const std::size_t first_update = resume_state ? resume_state->completed_update + 1 : 1;

        for (std::size_t update = first_update; update <= options.updates; ++update) {
            if (options.full_roster) {
                const auto [stage, group_mask] = curriculum_for_update(options, update);
                matchup_scheduler->set_stage(stage, group_mask);
                opponent.set_profile_assignments(
                    matchup_scheduler->sample_profile_indices(options.environments, true));
                simulator.set_opponent_characters_device(
                    opponent.profiles_device(), opponent.profile_count(),
                    opponent.profile_assignments_device(), 0);
            }
            for (std::size_t step = 0; step < options.horizon; ++step) {
                const auto before = simulator.device_view();
                const auto routed_inputs = options.visual_observations
                    ? side_router.select_visual_observations(before, options.environments)
                    : side_router.select_observations(before, options.environments);
                const float* policy_observations = routed_inputs.learner_observations;
                if (temporal) {
                    policy_observations = temporal->encode(
                        policy_observations, opponent.profiles_device(), opponent.profile_count(),
                        opponent.profile_assignments_device(), opponent.actions_buffer_device(),
                        options.environments);
                }
                const auto learner_output = learner.forward(
                    policy_observations, routed_inputs.learner_action_masks,
                    options.environments, options.seed + 1,
                    update * options.horizon + step, false);
                const auto* opponent_actions = opponent.actions_device(
                    routed_inputs.opponent_observations, routed_inputs.opponent_action_masks,
                    options.environments, options.seed + 2,
                    update * options.horizon + step,
                    t8::v2::ScriptedOpponentSet::TrainingV1);
                const auto routed_actions = side_router.route_actions(
                    learner_output.actions, opponent_actions, options.environments);
                rollout.record_policy_device(
                    step, policy_observations, routed_inputs.learner_action_masks,
                    learner_output.actions, learner_output.log_probabilities, learner_output.values);
                simulator.step_device_i64(routed_actions.p1_actions, routed_actions.p2_actions);
                const auto after = simulator.device_view();
                const float* rewards = side_router.select_rewards(
                    options.sparse_reward ? after.sparse_rewards_p1 : after.rewards_p1,
                    options.sparse_reward ? after.sparse_rewards_p2 : after.rewards_p2,
                    options.environments);
                rollout.record_outcome_device(step, rewards, after.terminated);
                if (temporal) temporal->reset_done(after.terminated, options.environments);
                simulator.reset_done_seeded(
                    options.seed + update * options.horizon + step);
            }
            const auto final_state = simulator.device_view();
            const auto routed_final = options.visual_observations
                ? side_router.select_visual_observations(final_state, options.environments)
                : side_router.select_observations(final_state, options.environments);
            const float* final_policy_observations = routed_final.learner_observations;
            if (temporal) {
                final_policy_observations = temporal->encode(
                    final_policy_observations, opponent.profiles_device(), opponent.profile_count(),
                    opponent.profile_assignments_device(), opponent.actions_buffer_device(),
                    options.environments);
            }
            const auto bootstrap = learner.forward(
                final_policy_observations, routed_final.learner_action_masks,
                options.environments, options.seed, update * options.horizon + options.horizon, true);
            rollout.compute_gae(bootstrap.values, options.gamma, options.gae_lambda, true);
            const auto metrics = learner.update_ppo(
                rollout.device_view(), update_config, options.seed + update * 10'000);

            std::optional<Evaluation> evaluation;
            if (update % options.evaluation_interval == 0 || update == options.updates) {
                // Frozen benchmark weights and stochastic sequence for every
                // update, shared by shaped/sparse runs with the same seed.
                evaluation = evaluate(
                    learner, options.evaluation_episodes, options.seed + 500'000,
                    options.visual_observations, options.full_roster,
                    roster_profiles, roster_character_moves, matchup_scheduler.get());
                if (options.full_roster) {
                    t8::v2::write_matchup_matrix_json(
                        options.run_directory / "matchup_matrix.json",
                        roster_profiles, matchup_scheduler->all_stats());
                }
            }
            const double elapsed = elapsed_before_resume + std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const std::uint64_t environment_steps = static_cast<std::uint64_t>(update) *
                static_cast<std::uint64_t>(options.environments) *
                static_cast<std::uint64_t>(options.horizon);
            append_metrics(metrics_path, update, environment_steps, reward_mode, observation_mode,
                           metrics, elapsed, evaluation);
            if (update % options.checkpoint_interval == 0 || update == options.updates) {
                const auto checkpoint = options.run_directory / "checkpoints" /
                    ("update_" + std::to_string(update) + ".t8ppo");
                learner.save_checkpoint(checkpoint);
                if (options.full_roster) {
                    const auto assignments = opponent.download_profile_assignments(options.environments);
                    const auto actions = opponent.download_actions(options.environments);
                    const auto temporal_state = temporal->download_state();
                    save_trainer_state(
                        trainer_state_path(checkpoint), options, update, environment_steps,
                        elapsed, simulator.download_states(), assignments, actions,
                        matchup_scheduler->all_stats(), matchup_scheduler->random_state(),
                        &temporal_state);
                } else {
                    save_trainer_state(
                        trainer_state_path(checkpoint), options, update, environment_steps,
                        elapsed, simulator.download_states());
                }
            }
            std::cout << "update=" << update << '/' << options.updates
                      << " steps=" << environment_steps
                      << " reward=" << reward_mode
                      << " observations=" << observation_mode
                      << " opponents=" << (options.full_roster ? "roster" : "legacy")
                      << " policy_loss=" << metrics.policy_loss
                      << " value_loss=" << metrics.value_loss
                      << " entropy=" << metrics.entropy;
            if (evaluation) std::cout << " eval_win_rate=" << evaluation->win_rate;
            std::cout << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "training error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
