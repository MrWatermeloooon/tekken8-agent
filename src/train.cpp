#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/roster.hpp"
#include "t8_v2/temporal.hpp"
#include "t8_v2/training_router.hpp"

#include <cuda_runtime.h>

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
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
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
    float final_learning_rate = 3e-5F;
    std::size_t anneal_updates = 100;
    float gamma = 0.997F;
    float gae_lambda = 0.95F;
    float reward_scale = 0.01F;
    float clip_range = 0.2F;
    float value_clip_range = 0.2F;
    float target_kl = 0.02F;
    float value_coefficient = 0.5F;
    float entropy_coefficient = 0.01F;
    float final_entropy_coefficient = 0.001F;
    float max_gradient_norm = 0.5F;
    std::size_t curriculum_updates = 100;
    std::uint64_t seed = 2027;
    std::size_t checkpoint_interval = 1;
    std::size_t evaluation_interval = 10;
    std::size_t evaluation_episodes = 256;
    bool sparse_reward = false;
    bool visual_observations = false;
    // Screen-only contract (implies visual): see t8_v2/screen_observation.hpp.
    bool screen_observations = false;
    t8::v2::ScreenObservationNoise screen_noise{};
    bool full_roster = true;
    int curriculum_stage = 0;
    std::filesystem::path opponent_catalog = "data/generated/opponent_profiles.csv";
    std::filesystem::path character_move_catalog = "data/generated/character_move_specs.csv";
    std::filesystem::path full_move_catalog = "data/generated/full_move_catalog.csv";
    std::string learner_character = "jun";
    std::filesystem::path run_directory;
    std::filesystem::path resume_checkpoint;
    bool run_directory_explicit = false;
    bool reward_scale_explicit = false;
    bool observation_normalization = true;
    bool return_normalization = true;
    // Largest held-out |P1 - P2| win-rate gap allowed for best-older promotion;
    // 1 disables the filter.
    float promotion_max_side_gap = 0.10F;
    // Older checkpoints scoring within this many binomial standard errors of
    // the best count as tied, and the newest of them is promoted; 0 disables.
    float promotion_tie_band = 1.0F;
    // Regression guard, checked on every deterministic held-out evaluation.
    std::uint8_t regression_guard = 1;  // 0 off, 1 rollback (then pause), 2 pause
    float regression_score_drop = 0.15F;
    float regression_style_drop = 0.30F;
    float regression_max_side_gap = 0.20F;
    std::uint32_t regression_patience = 3;
    std::uint32_t regression_max_rollbacks = 2;
    std::filesystem::path probe_checkpoint;
    std::size_t probe_episodes = 1024;
};

bool parse_on_off(std::string_view value, std::string_view option) {
    if (value == "on") return true;
    if (value == "off") return false;
    throw std::invalid_argument(std::string(option) + " must be on or off");
}

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
        else if (argument == "--final-learning-rate") options.final_learning_rate = std::stof(next());
        else if (argument == "--anneal-updates") options.anneal_updates = parse_size(next(), argument);
        else if (argument == "--gamma") options.gamma = std::stof(next());
        else if (argument == "--gae-lambda") options.gae_lambda = std::stof(next());
        else if (argument == "--reward-scale") {
            options.reward_scale = std::stof(next());
            options.reward_scale_explicit = true;
        }
        else if (argument == "--clip-range") options.clip_range = std::stof(next());
        else if (argument == "--value-clip-range") options.value_clip_range = std::stof(next());
        else if (argument == "--target-kl") options.target_kl = std::stof(next());
        else if (argument == "--value-coefficient") options.value_coefficient = std::stof(next());
        else if (argument == "--entropy-coefficient") options.entropy_coefficient = std::stof(next());
        else if (argument == "--final-entropy-coefficient") {
            options.final_entropy_coefficient = std::stof(next());
        }
        else if (argument == "--max-gradient-norm") options.max_gradient_norm = std::stof(next());
        else if (argument == "--curriculum-updates") {
            options.curriculum_updates = parse_size(next(), argument);
        }
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
        else if (argument == "--full-move-catalog") options.full_move_catalog = next();
        else if (argument == "--learner-character") options.learner_character = next();
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
            if (mode != "privileged" && mode != "visual" && mode != "screen") {
                throw std::invalid_argument("--observation-mode must be privileged, visual, or screen");
            }
            if (mode == "screen") {
                options.visual_observations = true;
                options.screen_observations = true;
            } else {
                options.visual_observations = mode == "visual";
                options.screen_observations = false;
            }
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
        else if (argument == "--observation-norm") {
            options.observation_normalization = parse_on_off(next(), argument);
        } else if (argument == "--return-norm") {
            options.return_normalization = parse_on_off(next(), argument);
        } else if (argument == "--promotion-max-side-gap") {
            options.promotion_max_side_gap = std::stof(next());
        } else if (argument == "--screen-position-sigma-max") {
            options.screen_noise.position_sigma_max = std::stof(next());
        } else if (argument == "--screen-event-error-max") {
            options.screen_noise.event_error_max = std::stof(next());
        } else if (argument == "--promotion-tie-band") {
            options.promotion_tie_band = std::stof(next());
        } else if (argument == "--regression-guard") {
            const std::string_view mode = next();
            if (mode == "off") options.regression_guard = 0;
            else if (mode == "rollback") options.regression_guard = 1;
            else if (mode == "pause") options.regression_guard = 2;
            else throw std::invalid_argument("--regression-guard must be off, rollback, or pause");
        } else if (argument == "--regression-score-drop") {
            options.regression_score_drop = std::stof(next());
        } else if (argument == "--regression-style-drop") {
            options.regression_style_drop = std::stof(next());
        } else if (argument == "--regression-max-side-gap") {
            options.regression_max_side_gap = std::stof(next());
        } else if (argument == "--regression-patience") {
            options.regression_patience = static_cast<std::uint32_t>(
                std::min<std::size_t>(parse_size(next(), argument), 1'000'000));
        } else if (argument == "--regression-max-rollbacks") {
            options.regression_max_rollbacks = static_cast<std::uint32_t>(
                std::min<unsigned long long>(std::stoull(next()), 1'000'000));
        } else if (argument == "--probe-checkpoint") {
            options.probe_checkpoint = next();
        } else if (argument == "--probe-episodes") {
            options.probe_episodes = parse_size(next(), argument);
        }
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
    if (!std::isfinite(options.final_learning_rate) || options.final_learning_rate <= 0.0F) {
        throw std::invalid_argument("--final-learning-rate must be finite and positive");
    }
    if (!std::isfinite(options.gamma) || options.gamma < 0.0F || options.gamma > 1.0F ||
        !std::isfinite(options.gae_lambda) || options.gae_lambda < 0.0F ||
        options.gae_lambda > 1.0F) {
        throw std::invalid_argument("--gamma and --gae-lambda must be finite and in [0, 1]");
    }
    if (options.sparse_reward && !options.reward_scale_explicit) options.reward_scale = 1.0F;
    const bool invalid_ppo = !std::isfinite(options.reward_scale) || options.reward_scale <= 0.0F ||
        !std::isfinite(options.clip_range) || options.clip_range < 0.0F ||
        !std::isfinite(options.value_clip_range) || options.value_clip_range < 0.0F ||
        !std::isfinite(options.target_kl) || options.target_kl < 0.0F ||
        !std::isfinite(options.value_coefficient) || options.value_coefficient < 0.0F ||
        !std::isfinite(options.entropy_coefficient) || options.entropy_coefficient < 0.0F ||
        !std::isfinite(options.final_entropy_coefficient) || options.final_entropy_coefficient < 0.0F ||
        !std::isfinite(options.max_gradient_norm) || options.max_gradient_norm <= 0.0F;
    if (invalid_ppo) throw std::invalid_argument("PPO and reward coefficients are invalid or non-finite");
    if (!std::isfinite(options.promotion_max_side_gap) || options.promotion_max_side_gap < 0.0F ||
        options.promotion_max_side_gap > 1.0F) {
        throw std::invalid_argument("--promotion-max-side-gap must be in [0, 1]");
    }
    if (!std::isfinite(options.promotion_tie_band) || options.promotion_tie_band < 0.0F ||
        options.promotion_tie_band > 10.0F) {
        throw std::invalid_argument("--promotion-tie-band must be in [0, 10]");
    }
    const auto unit_interval = [](float value) {
        return std::isfinite(value) && value >= 0.0F && value <= 1.0F;
    };
    if (!unit_interval(options.regression_score_drop) || !unit_interval(options.regression_style_drop) ||
        !unit_interval(options.regression_max_side_gap)) {
        throw std::invalid_argument("--regression-* drops and side gap must be in [0, 1]");
    }
    if (options.screen_observations && !options.full_roster) {
        throw std::invalid_argument("--observation-mode screen requires --opponents roster (the temporal encoder carries the uncertainty inputs)");
    }
    if (!std::isfinite(options.screen_noise.position_sigma_max) || options.screen_noise.position_sigma_max < 0.0F ||
        options.screen_noise.position_sigma_max > 10.0F || !std::isfinite(options.screen_noise.event_error_max) ||
        options.screen_noise.event_error_max < 0.0F || options.screen_noise.event_error_max > 0.5F) {
        throw std::invalid_argument("--screen-position-sigma-max must be in [0, 10] and --screen-event-error-max in [0, 0.5]");
    }
    if (options.probe_episodes % (2 * t8::v2::kEvaluationStyleCount) != 0) {
        throw std::invalid_argument("--probe-episodes must be a multiple of 16");
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
             (options.screen_observations ? "screen_" : options.visual_observations ? "visual_" : "privileged_") +
             (options.sparse_reward ? "sparse" : "shaped") +
             "_seed" + std::to_string(options.seed));
    }
    return options;
}

std::optional<t8::v2::ScreenObservationNoise> screen_noise_for(const Options& options) {
    if (!options.screen_observations) return std::nullopt;
    return options.screen_noise;
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
    const std::size_t stage_span = std::max<std::size_t>(1, (options.curriculum_updates + 3) / 4);
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

struct SelfPlaySelection {
    std::filesystem::path latest_checkpoint;
    std::filesystem::path best_older_checkpoint;
    std::size_t latest_update = 0;
    std::size_t best_older_update = 0;
    std::optional<double> best_older_side_gap;
    bool side_gap_fallback = false;
};

struct EvaluationRecord {
    std::uint64_t episodes = 256;
    double win_rate = 0.0;
    double p1_win_rate = 0.0;
    double p2_win_rate = 0.0;
    [[nodiscard]] double side_gap() const noexcept { return std::abs(p1_win_rate - p2_win_rate); }
};

// Held-out results parsed from the metrics ledger. The ledger only ever grows
// (it is rewritten with an unchanged prefix), so each refresh parses just the
// bytes appended since the last one instead of the whole multi-megabyte file.
class EvaluationLedger {
public:
    explicit EvaluationLedger(std::filesystem::path path) : path_(std::move(path)) {}

    const std::map<std::size_t, EvaluationRecord>& refresh() {
        std::ifstream input(path_, std::ios::binary);
        if (!input) return records_;
        input.seekg(0, std::ios::end);
        const auto size = static_cast<std::uint64_t>(input.tellg());
        if (size < consumed_) {
            records_.clear();
            consumed_ = 0;
        }
        input.seekg(static_cast<std::streamoff>(consumed_));
        std::string line;
        // A final line without its newline is still being written; leave it.
        while (std::getline(input, line) && !input.eof()) {
            consumed_ += line.size() + 1;
            parse(line);
        }
        return records_;
    }

private:
    static std::optional<double> win_rate_after(const std::string& line, std::size_t from) {
        const auto marker = line.find("\"win_rate\":", from);
        if (marker == std::string::npos) return std::nullopt;
        const auto begin = marker + 11;
        const auto end = line.find_first_of(",}", begin);
        if (end == std::string::npos) return std::nullopt;
        return std::stod(line.substr(begin, end - begin));
    }

    void parse(const std::string& line) {
        const auto update_marker = line.find("\"update\":");
        const auto evaluation_marker = line.find("\"evaluation\":{\"total\":");
        if (update_marker == std::string::npos || evaluation_marker == std::string::npos) return;
        const auto update_begin = update_marker + 9;
        const auto update_end = line.find(',', update_begin);
        const auto total = win_rate_after(line, evaluation_marker);
        if (update_end == std::string::npos || !total) return;
        // as_p1/as_p2 follow the deterministic total, before any stochastic block.
        const auto p1_marker = line.find("\"as_p1\":", evaluation_marker);
        const auto p2_marker = line.find("\"as_p2\":", evaluation_marker);
        const auto p1 = p1_marker == std::string::npos ? total : win_rate_after(line, p1_marker);
        const auto p2 = p2_marker == std::string::npos ? total : win_rate_after(line, p2_marker);
        std::uint64_t episodes = 256;
        const auto episodes_marker = line.find("\"episodes\":", evaluation_marker);
        if (episodes_marker != std::string::npos) {
            episodes = std::max<std::uint64_t>(1, std::stoull(line.substr(episodes_marker + 11)));
        }
        records_[static_cast<std::size_t>(
            std::stoull(line.substr(update_begin, update_end - update_begin)))] =
            {episodes, *total, p1.value_or(*total), p2.value_or(*total)};
    }

    std::filesystem::path path_;
    std::uint64_t consumed_ = 0;
    std::map<std::size_t, EvaluationRecord> records_;
};

// Latest-self plus the strongest older checkpoint. An older checkpoint is
// promotable only if its held-out P1/P2 win-rate gap is within
// max_side_gap; if none qualifies, the most side-balanced scored one is used.
// Scores within tie_band_standard_errors binomial standard errors of the best
// are treated as tied and the newest wins: a single noisy 256-episode
// evaluation otherwise freezes a lucky checkpoint in place (update 100 held
// the slot for 11,700 updates after beating the truly stronger update 200 by
// one game).
std::optional<SelfPlaySelection> select_self_play_checkpoint(
    const std::filesystem::path& checkpoint_directory,
    EvaluationLedger& ledger,
    std::size_t current_update,
    double max_side_gap,
    double tie_band_standard_errors) {
    struct Candidate {
        std::filesystem::path checkpoint;
        std::size_t update = 0;
    };
    std::vector<Candidate> candidates;
    if (!std::filesystem::exists(checkpoint_directory)) return std::nullopt;
    for (const auto& entry : std::filesystem::directory_iterator(checkpoint_directory)) {
        if (!entry.is_regular_file() || entry.path().extension() != ".t8ppo") continue;
        const std::string stem = entry.path().stem().string();
        constexpr std::string_view prefix = "update_";
        if (!stem.starts_with(prefix)) continue;
        const std::string digits = stem.substr(prefix.size());
        if (digits.empty() || !std::all_of(digits.begin(), digits.end(), [](unsigned char value) {
                return std::isdigit(value) != 0;
            })) continue;
        const auto update = static_cast<std::size_t>(std::stoull(digits));
        if (update < current_update) candidates.push_back({entry.path(), update});
    }
    if (candidates.empty()) return std::nullopt;
    std::sort(candidates.begin(), candidates.end(), [](const auto& left, const auto& right) {
        return left.update < right.update;
    });
    const Candidate latest = candidates.back();
    Candidate best = latest;
    if (candidates.size() == 1) {
        return SelfPlaySelection{
            latest.checkpoint, latest.checkpoint, latest.update, latest.update};
    }

    const auto& records = ledger.refresh();
    const Candidate* promoted = nullptr;
    const EvaluationRecord* promoted_record = nullptr;
    const Candidate* balanced = nullptr;
    const EvaluationRecord* balanced_record = nullptr;
    for (const auto& candidate : candidates) {
        if (candidate.update == latest.update) continue;
        const auto found = records.find(candidate.update);
        if (found == records.end()) continue;
        const auto& record = found->second;
        // Candidates ascend by update, so >= prefers the newer one on ties.
        if (record.side_gap() <= max_side_gap + 1e-9 &&
            (!promoted || record.win_rate >= promoted_record->win_rate)) {
            promoted = &candidate;
            promoted_record = &record;
        }
        if (!balanced || record.side_gap() < balanced_record->side_gap() ||
            (record.side_gap() == balanced_record->side_gap() &&
             record.win_rate >= balanced_record->win_rate)) {
            balanced = &candidate;
            balanced_record = &record;
        }
    }
    if (promoted && tie_band_standard_errors > 0.0) {
        const double best_score = promoted_record->win_rate;
        const double band = tie_band_standard_errors *
            std::sqrt(best_score * (1.0 - best_score) / static_cast<double>(promoted_record->episodes));
        for (const auto& candidate : candidates) {
            if (candidate.update == latest.update || candidate.update <= promoted->update) continue;
            const auto found = records.find(candidate.update);
            if (found == records.end()) continue;
            if (found->second.side_gap() <= max_side_gap + 1e-9 &&
                found->second.win_rate >= best_score - band - 1e-12) {
                promoted = &candidate;
                promoted_record = &found->second;
            }
        }
    }
    if (!promoted) {
        promoted = balanced;
        promoted_record = balanced_record;
    }
    if (promoted) best = *promoted;
    else best = candidates[candidates.size() - 2];
    return SelfPlaySelection{
        latest.checkpoint, best.checkpoint, latest.update, best.update,
        promoted_record ? std::optional<double>(promoted_record->side_gap()) : std::nullopt,
        promoted_record != nullptr && promoted_record->side_gap() > max_side_gap + 1e-9};
}

// Feeds one update's scripted-opponent outcomes ([profile][win, loss, draw])
// into matchmaking. recent_win_rate is a sample-weighted moving score with a
// window of roughly 200 episodes, so best_win_rate and the forgetting flag are
// not set by a single small, lucky batch.
void record_training_outcomes(
    t8::v2::MatchupScheduler& scheduler,
    std::span<const std::uint64_t> tally) {
    constexpr double kRecentWindowEpisodes = 200.0;
    for (std::size_t profile = 0; profile * 3 < tally.size(); ++profile) {
        const auto wins = tally[profile * 3];
        const auto losses = tally[profile * 3 + 1];
        const auto draws = tally[profile * 3 + 2];
        const auto episodes = wins + losses + draws;
        if (episodes == 0) continue;
        const double batch_score =
            (static_cast<double>(wins) + 0.5 * static_cast<double>(draws)) / static_cast<double>(episodes);
        const auto& previous = scheduler.stats(static_cast<std::uint32_t>(profile));
        const double weight = std::min(1.0, static_cast<double>(episodes) / kRecentWindowEpisodes);
        const double recent = previous.episodes == 0 ? batch_score
            : previous.recent_win_rate + weight * (batch_score - previous.recent_win_rate);
        scheduler.record(static_cast<std::uint32_t>(profile), wins, losses, draws, recent);
    }
}

std::uint32_t jun_profile_index(
    std::span<const t8::v2::OpponentProfileParameters> profiles) {
    const auto found = std::find_if(profiles.begin(), profiles.end(), [](const auto& profile) {
        return profile.character_id == t8::v2::kJunCharacterId &&
            (profile.group_mask & t8::v2::Fundamentals) != 0;
    });
    if (found == profiles.end()) throw std::runtime_error("opponent catalog has no Jun profile");
    return static_cast<std::uint32_t>(std::distance(profiles.begin(), found));
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

// Learner [win, loss, draw] per opponent roster character.
using CharacterOutcomes = std::array<std::array<std::uint64_t, 3>, t8::v2::kRosterCharacterCount>;

struct Evaluation {
    t8::v2::GpuEpisodeSummary total{};
    t8::v2::GpuEpisodeSummary as_p1{};
    t8::v2::GpuEpisodeSummary as_p2{};
    double win_rate = 0.0;
    CharacterOutcomes characters{};  // full-roster evaluations only
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

// Probe-only behaviour statistics over live (unfinished) learner decisions.
// "Choice" decisions have more than one legal action.
struct DecisionStats {
    std::uint64_t live_decisions = 0;
    std::uint64_t choice_decisions = 0;
    double choice_entropy = 0.0;
    std::array<std::uint64_t, t8::v2::kActionCount> choice_actions{};
};

void accumulate_decision_stats(
    DecisionStats& stats,
    const t8::v2::GpuActorCritic& learner,
    const std::uint8_t* device_masks,
    std::span<const t8::v2::State> states,
    std::size_t environments) {
    std::vector<std::uint8_t> masks(environments * t8::v2::kActionCount);
    if (cudaMemcpy(masks.data(), device_masks, masks.size(), cudaMemcpyDeviceToHost) != cudaSuccess) {
        throw std::runtime_error("probe could not download action masks");
    }
    const auto actions = learner.download_actions(environments);
    const auto entropies = learner.download_entropies(environments);
    for (std::size_t lane = 0; lane < environments; ++lane) {
        if (states[lane].round_over) continue;
        ++stats.live_decisions;
        std::size_t legal = 0;
        for (std::size_t action = 0; action < t8::v2::kActionCount; ++action) {
            legal += masks[lane * t8::v2::kActionCount + action] != 0;
        }
        if (legal <= 1) continue;
        ++stats.choice_decisions;
        stats.choice_entropy += entropies[lane];
        ++stats.choice_actions[static_cast<std::size_t>(actions[lane])];
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
    std::uint32_t learner_character_id,
    bool deterministic,
    DecisionStats* decision_stats = nullptr,
    CharacterOutcomes* character_outcomes = nullptr,
    std::optional<t8::v2::ScreenObservationNoise> screen = std::nullopt) {
    t8::v2::GpuEpisodeSummary aggregate{};
    std::size_t completed = 0;
    while (completed < requested_episodes) {
        const std::size_t environments = std::min(
            {requested_episodes - completed, learner.capacity(), std::size_t{256}});
        t8::v2::Config evaluation_config{};
        evaluation_config.timeout_ties_are_draws = true;
        evaluation_config.screen_observations = screen.has_value();
        if (screen) evaluation_config.screen_noise = *screen;
        evaluation_config.randomize_initial_positions = true;
        t8::v2::GpuSimulatorBatch simulator(environments, evaluation_config);
        simulator.reset_seeded(seed);
        t8::v2::GpuScriptedOpponent opponent(environments);
        std::vector<std::uint32_t> assignments;
        std::unique_ptr<t8::v2::GpuTemporalMatchupEncoder> temporal;
        if (full_roster) {
            if (profiles.empty() || character_moves.empty()) {
                throw std::logic_error(
                    "full-roster evaluation requires profiles and character moves");
            }
            simulator.set_character_move_specs(character_moves);
            opponent.set_profiles(profiles);
            assignments.resize(environments);
            for (std::size_t lane = 0; lane < environments; ++lane) {
                const std::size_t global_lane = completed + lane;
                assignments[lane] = static_cast<std::uint32_t>(
                    (global_lane * profiles.size()) / requested_episodes);
            }
            opponent.set_profile_assignments(assignments);
            simulator.set_opponent_characters_device(
                opponent.profiles_device(), opponent.profile_count(),
                opponent.profile_assignments_device(), learner_player, nullptr, nullptr,
                learner_character_id);
            temporal = std::make_unique<t8::v2::GpuTemporalMatchupEncoder>(
                environments, visual_observations ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize,
                screen);
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
                    environments, seed, decision, deterministic);
                if (decision_stats) {
                    accumulate_decision_stats(*decision_stats, learner, before.action_masks_p1,
                                              simulator.download_states(), environments);
                }
                const auto* p2 = opponent.actions_device(
                    before.observations_p2, before.action_masks_p2,
                    environments, seed + 1, decision,
                    t8::v2::ScriptedOpponentSet::HeldOutV2);
                simulator.step_device_i64(p1.actions, p2);
            } else {
                const float* policy_observations = learner_p2_observations;
                if (temporal) {
                    policy_observations = temporal->encode(
                        policy_observations, opponent.profiles_device(), opponent.profile_count(),
                        opponent.profile_assignments_device(), opponent.actions_buffer_device(), environments);
                }
                const auto p2 = learner.forward(
                    policy_observations, before.action_masks_p2,
                    environments, seed, decision, deterministic);
                if (decision_stats) {
                    accumulate_decision_stats(*decision_stats, learner, before.action_masks_p2,
                                              simulator.download_states(), environments);
                }
                const auto* p1 = opponent.actions_device(
                    before.observations_p1, before.action_masks_p1,
                    environments, seed + 1, decision,
                    t8::v2::ScriptedOpponentSet::HeldOutV2);
                simulator.step_device_i64(p1, p2.actions);
            }
        }
        const auto summary = simulator.summarize_episodes(learner_player);
        if (summary.episodes != environments) {
            throw std::runtime_error("held-out evaluation did not terminate every GPU lane");
        }
        if (character_outcomes && full_roster) {
            const auto states = simulator.download_states();
            for (std::size_t lane = 0; lane < environments; ++lane) {
                const auto character = profiles[assignments[lane]].character_id;
                if (character >= t8::v2::kRosterCharacterCount) continue;
                const int winner = states[lane].winner;
                const std::size_t column = winner == learner_player ? 0 : (winner == 0 ? 2 : 1);
                ++(*character_outcomes)[character][column];
            }
        }
        merge_summary(aggregate, summary);
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
    std::uint32_t learner_character_id,
    bool deterministic,
    DecisionStats* p1_decisions = nullptr,
    DecisionStats* p2_decisions = nullptr,
    std::optional<t8::v2::ScreenObservationNoise> screen = std::nullopt) {
    Evaluation result{};
    const std::size_t p1_episodes = requested_episodes / 2;
    const std::size_t p2_episodes = requested_episodes - p1_episodes;
    result.as_p1 = evaluate_side(learner, p1_episodes, seed, 1, visual_observations,
                                 full_roster, profiles, character_moves,
                                 learner_character_id, deterministic, p1_decisions,
                                 &result.characters, screen);
    result.as_p2 = evaluate_side(learner, p2_episodes, seed + 100'000, 2, visual_observations,
                                 full_roster, profiles, character_moves,
                                 learner_character_id, deterministic, p2_decisions,
                                 &result.characters, screen);
    merge_summary(result.total, result.as_p1);
    merge_summary(result.total, result.as_p2);
    result.win_rate = result.total.episodes == 0 ? 0.0 :
        static_cast<double>(result.total.wins) / static_cast<double>(result.total.episodes);
    return result;
}

// Held-out regression guard. The reference is the best evaluation so far that
// passed every check and has a saved checkpoint. `patience` consecutive
// regressed evaluations trigger a rollback to it (up to max_rollbacks per
// reference) or a pause. Exploitability is approximated by the weakest
// held-out style's win rate; no best-response exploiter is trained.
struct RegressionGuardState {
    std::uint64_t reference_update = 0;  // 0 = no healthy reference yet
    double reference_win_rate = 0.0;
    double reference_worst_style = 0.0;
    std::uint32_t consecutive_regressions = 0;
    std::uint32_t rollbacks = 0;
    // A trainer state saved at this update must restore the reference weights
    // on resume, because the checkpoint holds the pre-rollback weights.
    std::uint64_t rollback_update = 0;
};

enum class GuardAction { None, Rollback, Pause };

struct GuardDecision {
    std::vector<std::string_view> reasons;
    GuardAction action = GuardAction::None;
    double worst_style_win_rate = 0.0;
    double side_gap = 0.0;
    std::uint64_t reference_update = 0;
    std::uint32_t consecutive_regressions = 0;
    bool new_reference = false;
};

double worst_style_win_rate(const t8::v2::GpuEpisodeSummary& summary) {
    double worst = 1.0;
    for (std::size_t style = 0; style < t8::v2::kEvaluationStyleCount; ++style) {
        if (summary.style_episodes[style] == 0) continue;
        worst = std::min(worst, static_cast<double>(summary.style_wins[style]) /
                                    static_cast<double>(summary.style_episodes[style]));
    }
    return worst;
}

double side_win_rate(const t8::v2::GpuEpisodeSummary& summary) {
    return summary.episodes == 0 ? 0.0 :
        static_cast<double>(summary.wins) / static_cast<double>(summary.episodes);
}

GuardDecision regression_guard_step(
    RegressionGuardState& guard,
    const Options& options,
    std::size_t update,
    const Evaluation& evaluation,
    bool checkpoint_saved) {
    GuardDecision decision{};
    decision.worst_style_win_rate = worst_style_win_rate(evaluation.total);
    decision.side_gap = std::abs(side_win_rate(evaluation.as_p1) - side_win_rate(evaluation.as_p2));
    if (options.regression_guard != 0) {
        const bool has_reference = guard.reference_update != 0;
        if (has_reference && evaluation.win_rate < guard.reference_win_rate - options.regression_score_drop) {
            decision.reasons.push_back("held_out_score");
        }
        if (has_reference && decision.worst_style_win_rate <
                guard.reference_worst_style - options.regression_style_drop) {
            decision.reasons.push_back("worst_style");
        }
        if (decision.side_gap > options.regression_max_side_gap + 1e-9) {
            decision.reasons.push_back("side_balance");
        }
        if (decision.reasons.empty()) {
            guard.consecutive_regressions = 0;
            if (checkpoint_saved && (!has_reference || evaluation.win_rate > guard.reference_win_rate)) {
                guard.reference_update = update;
                guard.reference_win_rate = evaluation.win_rate;
                guard.reference_worst_style = decision.worst_style_win_rate;
                guard.rollbacks = 0;
                decision.new_reference = true;
            }
        } else if (++guard.consecutive_regressions >= options.regression_patience && has_reference) {
            guard.consecutive_regressions = 0;
            guard.rollback_update = update;
            if (options.regression_guard == 1 && guard.rollbacks < options.regression_max_rollbacks) {
                ++guard.rollbacks;
                decision.action = GuardAction::Rollback;
            } else {
                // A resumed run gets a fresh rollback budget.
                guard.rollbacks = 0;
                decision.action = GuardAction::Pause;
            }
        }
    }
    decision.reference_update = guard.reference_update;
    decision.consecutive_regressions = guard.consecutive_regressions;
    return decision;
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

// Moves a completed temporary file over `path`. On Windows, antivirus,
// indexers, sync clients, and monitoring readers can briefly open the target
// without delete sharing, so transient failures are retried instead of ending
// a long run.
void replace_file_atomically(const std::filesystem::path& temporary, const std::filesystem::path& path) {
#ifdef _WIN32
    constexpr int kReplaceAttempts = 600;
    DWORD replace_error = ERROR_SUCCESS;
    bool replaced = false;
    for (int attempt = 0; attempt < kReplaceAttempts; ++attempt) {
        if (MoveFileExW(
                temporary.c_str(), path.c_str(),
                MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
            replaced = true;
            break;
        }
        replace_error = GetLastError();
        const bool transient = replace_error == ERROR_ACCESS_DENIED ||
            replace_error == ERROR_SHARING_VIOLATION ||
            replace_error == ERROR_LOCK_VIOLATION;
        if (!transient) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    if (!replaced) {
        throw std::runtime_error(
            "could not atomically replace file (Windows error " +
            std::to_string(replace_error) + "): " + path.string());
    }
#else
    std::filesystem::rename(temporary, path);
#endif
}

void append_metrics(
    const std::filesystem::path& path,
    std::size_t update,
    std::uint64_t environment_steps,
    std::string_view reward_mode,
    std::string_view observation_mode,
    const t8::v2::PpoUpdateMetrics& metrics,
    double elapsed_seconds,
    const std::optional<Evaluation>& deterministic_evaluation,
    const std::optional<Evaluation>& stochastic_evaluation,
    const std::optional<SelfPlaySelection>& self_play_selection,
    std::optional<float> return_standard_deviation,
    std::uint64_t observation_normalizer_count,
    const std::optional<GuardDecision>& guard_decision) {
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
           << ",\"kl_early_stop\":" << (metrics.early_stopped ? "true" : "false")
           << ",\"decision_fraction\":" << metrics.decision_fraction
           << ",\"decision_entropy\":" << metrics.decision_entropy
           << ",\"training_opponent\":\""
           << (self_play_selection ? "self_play_80_latest_20_best" : "scripted")
           << '"'
           << ",\"observation_normalizer_count\":" << observation_normalizer_count;
    if (return_standard_deviation) row << ",\"return_std\":" << *return_standard_deviation;
    if (guard_decision) {
        row << ",\"regression_guard\":{\"reasons\":[";
        for (std::size_t index = 0; index < guard_decision->reasons.size(); ++index) {
            row << (index == 0 ? "" : ",") << '"' << guard_decision->reasons[index] << '"';
        }
        const auto action = guard_decision->action == GuardAction::Rollback ? "rollback"
            : guard_decision->action == GuardAction::Pause ? "pause" : "none";
        row << "],\"action\":\"" << action << '"'
            << ",\"consecutive_regressions\":" << guard_decision->consecutive_regressions
            << ",\"reference_update\":" << guard_decision->reference_update
            << ",\"new_reference\":" << (guard_decision->new_reference ? "true" : "false")
            << ",\"worst_style_win_rate\":" << guard_decision->worst_style_win_rate
            << ",\"side_gap\":" << guard_decision->side_gap << '}';
    }
    if (self_play_selection) {
        row << ",\"latest_checkpoint_update\":" << self_play_selection->latest_update
            << ",\"best_older_checkpoint_update\":" << self_play_selection->best_older_update;
        if (self_play_selection->best_older_side_gap) {
            row << ",\"best_older_side_gap\":" << *self_play_selection->best_older_side_gap
                << ",\"best_older_side_gap_fallback\":"
                << (self_play_selection->side_gap_fallback ? "true" : "false");
        }
    }
    if (deterministic_evaluation) {
        row << ",\"evaluation\":{";
        append_summary_json(row, "total", deterministic_evaluation->total);
        row << ',';
        append_summary_json(row, "as_p1", deterministic_evaluation->as_p1);
        row << ',';
        append_summary_json(row, "as_p2", deterministic_evaluation->as_p2);
        row << '}';
    }
    if (stochastic_evaluation) {
        row << ",\"evaluation_stochastic\":{";
        append_summary_json(row, "total", stochastic_evaluation->total);
        row << ',';
        append_summary_json(row, "as_p1", stochastic_evaluation->as_p1);
        row << ',';
        append_summary_json(row, "as_p2", stochastic_evaluation->as_p2);
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
    replace_file_atomically(temporary, path);
}

// Evaluation exports: flat CSVs written directly at each held-out evaluation,
// so curves and breakdowns never have to be recovered from nested metrics
// records or console logs. The first column of every row is the update, which
// lets a resume drop rows written after the resume checkpoint.
constexpr std::string_view kEvaluationsCsv = "evaluations.csv";
constexpr std::string_view kEvaluationStylesCsv = "evaluation_styles.csv";
constexpr std::string_view kEvaluationCharactersCsv = "evaluation_characters.csv";

void write_csv_atomically(
    const std::filesystem::path& path,
    std::string_view header,
    const std::vector<std::string>& rows) {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("could not write evaluation export: " + temporary.string());
        output << header << '\n';
        for (const auto& row : rows) output << row << '\n';
        output.flush();
        if (!output) throw std::runtime_error("evaluation export write failed: " + temporary.string());
    }
    replace_file_atomically(temporary, path);
}

// Data rows of `path` whose leading update is <= max_update (all if absent).
std::vector<std::string> read_csv_rows(
    const std::filesystem::path& path,
    std::size_t max_update = std::numeric_limits<std::size_t>::max()) {
    std::vector<std::string> rows;
    std::ifstream input(path, std::ios::binary);
    std::string line;
    bool header = true;
    while (std::getline(input, line)) {
        if (header) {
            header = false;
            continue;
        }
        if (line.empty() || !std::isdigit(static_cast<unsigned char>(line.front()))) continue;
        if (static_cast<std::size_t>(std::stoull(line)) <= max_update) rows.push_back(line);
    }
    return rows;
}

void append_csv_rows(
    const std::filesystem::path& path,
    std::string_view header,
    const std::vector<std::string>& new_rows) {
    auto rows = read_csv_rows(path);
    rows.insert(rows.end(), new_rows.begin(), new_rows.end());
    write_csv_atomically(path, header, rows);
}

void truncate_evaluation_exports(const std::filesystem::path& run_directory, std::size_t completed_update) {
    for (const auto name : {kEvaluationsCsv, kEvaluationStylesCsv, kEvaluationCharactersCsv}) {
        const auto path = run_directory / name;
        if (!std::filesystem::exists(path)) continue;
        std::ifstream input(path, std::ios::binary);
        std::string header;
        std::getline(input, header);
        input.close();
        write_csv_atomically(path, header, read_csv_rows(path, completed_update));
    }
}

std::string csv_rate(std::uint64_t numerator, std::uint64_t denominator) {
    std::ostringstream value;
    value << std::setprecision(9)
          << (denominator == 0 ? 0.0 : static_cast<double>(numerator) / static_cast<double>(denominator));
    return value.str();
}

void export_evaluation(
    const std::filesystem::path& run_directory,
    std::size_t update,
    std::uint64_t environment_steps,
    double elapsed_seconds,
    const Evaluation& deterministic,
    const Evaluation& stochastic,
    const std::optional<SelfPlaySelection>& self_play_selection,
    const std::optional<GuardDecision>& guard_decision,
    bool full_roster) {
    const auto summary_columns = [](const Evaluation& evaluation) {
        std::ostringstream columns;
        columns << std::setprecision(9)
                << csv_rate(evaluation.total.wins, evaluation.total.episodes) << ','
                << csv_rate(evaluation.as_p1.wins, evaluation.as_p1.episodes) << ','
                << csv_rate(evaluation.as_p2.wins, evaluation.as_p2.episodes) << ','
                << csv_rate(evaluation.total.draws, evaluation.total.episodes) << ','
                << worst_style_win_rate(evaluation.total) << ','
                << (evaluation.total.episodes == 0 ? 0.0
                        : evaluation.total.total_damage_dealt / static_cast<double>(evaluation.total.episodes)) << ','
                << (evaluation.total.episodes == 0 ? 0.0
                        : evaluation.total.total_damage_taken / static_cast<double>(evaluation.total.episodes));
        return columns.str();
    };
    std::ostringstream row;
    row << std::setprecision(12) << update << ',' << environment_steps << ',' << elapsed_seconds << ','
        << deterministic.total.episodes << ','
        << (self_play_selection ? "self_play" : "scripted") << ','
        << (self_play_selection ? std::to_string(self_play_selection->latest_update) : "") << ','
        << (self_play_selection ? std::to_string(self_play_selection->best_older_update) : "") << ','
        << summary_columns(deterministic) << ',' << summary_columns(stochastic) << ','
        << (!guard_decision ? "" : guard_decision->action == GuardAction::Rollback ? "rollback"
            : guard_decision->action == GuardAction::Pause ? "pause" : "none");
    append_csv_rows(
        run_directory / kEvaluationsCsv,
        "update,environment_steps,elapsed_seconds,episodes,training_opponent,latest_checkpoint_update,"
        "best_older_checkpoint_update,"
        "win_rate,p1_win_rate,p2_win_rate,draw_rate,worst_style_win_rate,mean_damage_dealt,mean_damage_taken,"
        "stochastic_win_rate,stochastic_p1_win_rate,stochastic_p2_win_rate,stochastic_draw_rate,"
        "stochastic_worst_style_win_rate,stochastic_mean_damage_dealt,stochastic_mean_damage_taken,"
        "guard_action",
        {row.str()});

    std::vector<std::string> style_rows;
    std::vector<std::string> character_rows;
    for (const auto& [policy, evaluation] :
         {std::pair<std::string_view, const Evaluation*>{"deterministic", &deterministic},
          std::pair<std::string_view, const Evaluation*>{"stochastic", &stochastic}}) {
        for (const auto& [side, summary] :
             {std::pair<std::string_view, const t8::v2::GpuEpisodeSummary*>{"total", &evaluation->total},
              std::pair<std::string_view, const t8::v2::GpuEpisodeSummary*>{"p1", &evaluation->as_p1},
              std::pair<std::string_view, const t8::v2::GpuEpisodeSummary*>{"p2", &evaluation->as_p2}}) {
            for (std::size_t style = 0; style < t8::v2::kEvaluationStyleCount; ++style) {
                std::ostringstream style_row;
                style_row << update << ',' << policy << ',' << side << ',' << style << ','
                          << summary->style_episodes[style] << ',' << summary->style_wins[style] << ','
                          << summary->style_losses[style] << ',' << summary->style_draws[style] << ','
                          << csv_rate(summary->style_wins[style], summary->style_episodes[style]);
                style_rows.push_back(style_row.str());
            }
        }
        if (!full_roster) continue;
        for (std::size_t character = 0; character < t8::v2::kRosterCharacterCount; ++character) {
            const auto& outcome = evaluation->characters[character];
            const auto episodes = outcome[0] + outcome[1] + outcome[2];
            if (episodes == 0) continue;
            std::ostringstream character_row;
            character_row << update << ',' << policy << ',' << character << ','
                          << t8::v2::kRosterCharacterSlugs[character] << ',' << episodes << ','
                          << outcome[0] << ',' << outcome[1] << ',' << outcome[2] << ','
                          << csv_rate(outcome[0], episodes);
            character_rows.push_back(character_row.str());
        }
    }
    append_csv_rows(run_directory / kEvaluationStylesCsv,
                    "update,policy,side,style,episodes,wins,losses,draws,win_rate", style_rows);
    if (full_roster) {
        append_csv_rows(run_directory / kEvaluationCharactersCsv,
                        "update,policy,character_id,character,episodes,wins,losses,draws,win_rate",
                        character_rows);
    }
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
    std::vector<std::int64_t> learner_actions;
    bool self_play_active = false;
    std::optional<t8::v2::TemporalEncoderState> self_play_temporal_state;
    std::optional<t8::v2::ReturnNormalizerState> return_normalizer;
    RegressionGuardState guard;
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

void write_string(std::ostream& output, std::string_view value) {
    write_vector(output, std::span<const char>(value.data(), value.size()));
}

std::string read_string(std::istream& input, const std::filesystem::path& path) {
    const auto bytes = read_vector<char>(input, path);
    return std::string(bytes.begin(), bytes.end());
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

std::filesystem::path checkpoint_path(const std::filesystem::path& run_directory, std::uint64_t update) {
    return run_directory / "checkpoints" / ("update_" + std::to_string(update) + ".t8ppo");
}

// Distinguishes a deliberate regression-guard pause from a crash (exit 1).
constexpr int kRegressionPauseExitCode = 3;

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
    const t8::v2::TemporalEncoderState* temporal_state = nullptr,
    std::span<const std::int64_t> learner_actions = {},
    bool self_play_active = false,
    const t8::v2::TemporalEncoderState* self_play_temporal_state = nullptr,
    const t8::v2::ReturnNormalizerState* return_normalizer = nullptr,
    const RegressionGuardState& guard = {}) {
    if (options.return_normalization && return_normalizer == nullptr) {
        throw std::invalid_argument("return-normalized trainer state is missing normalizer statistics");
    }
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
    write_value(output, std::uint32_t{7});
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
    write_value(output, options.final_learning_rate);
    write_value(output, static_cast<std::uint64_t>(options.anneal_updates));
    write_value(output, options.reward_scale);
    write_value(output, options.clip_range);
    write_value(output, options.value_clip_range);
    write_value(output, options.target_kl);
    write_value(output, options.value_coefficient);
    write_value(output, options.entropy_coefficient);
    write_value(output, options.final_entropy_coefficient);
    write_value(output, options.max_gradient_norm);
    write_value(output, static_cast<std::uint64_t>(options.curriculum_updates));
    write_value(output, static_cast<std::uint64_t>(options.checkpoint_interval));
    write_value(output, static_cast<std::uint64_t>(options.evaluation_interval));
    write_value(output, static_cast<std::uint64_t>(options.evaluation_episodes));
    write_value(output, static_cast<std::uint8_t>(options.sparse_reward));
    write_value(output, static_cast<std::uint8_t>(options.visual_observations));
    write_value(output, static_cast<std::uint8_t>(options.full_roster));
    write_value(output, static_cast<std::int32_t>(options.curriculum_stage));
    if (options.full_roster) {
        const auto catalog = t8::v2::load_full_move_catalog_csv(options.full_move_catalog);
        write_string(output, catalog.catalog_sha256);
        write_string(output, catalog.roster_version);
    } else {
        write_string(output, std::string(64, '0'));
        write_string(output, "compatibility");
    }
    write_string(output, options.learner_character);
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
            learner_actions.size() != options.environments ||
            scheduler_stats.size() != t8::v2::kOpponentProfileCount || temporal_state == nullptr ||
            (self_play_active && self_play_temporal_state == nullptr)) {
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
        write_value(output, static_cast<std::uint8_t>(self_play_active));
        write_vector(output, learner_actions);
        if (self_play_active) {
            write_vector(output, std::span<const float>(self_play_temporal_state->history));
            write_vector(output, std::span<const std::int64_t>(self_play_temporal_state->previous_actions));
            write_vector(output, std::span<const std::int32_t>(self_play_temporal_state->repeated_action_frames));
            write_vector(output, std::span<const float>(self_play_temporal_state->previous_own_health));
            write_vector(output, std::span<const float>(self_play_temporal_state->previous_opponent_health));
            write_vector(output, std::span<const float>(self_play_temporal_state->previous_distance));
            write_vector(output, std::span<const std::uint8_t>(self_play_temporal_state->valid));
        }
    }
    write_value(output, static_cast<std::uint8_t>(options.observation_normalization));
    write_value(output, static_cast<std::uint8_t>(options.return_normalization));
    write_value(output, options.promotion_max_side_gap);
    write_value(output, options.promotion_tie_band);
    write_value(output, static_cast<std::uint8_t>(options.screen_observations));
    write_value(output, options.screen_noise.position_sigma_max);
    write_value(output, options.screen_noise.event_error_max);
    write_value(output, options.regression_guard);
    write_value(output, options.regression_score_drop);
    write_value(output, options.regression_style_drop);
    write_value(output, options.regression_max_side_gap);
    write_value(output, options.regression_patience);
    write_value(output, options.regression_max_rollbacks);
    write_value(output, guard.reference_update);
    write_value(output, guard.reference_win_rate);
    write_value(output, guard.reference_worst_style);
    write_value(output, guard.consecutive_regressions);
    write_value(output, guard.rollbacks);
    write_value(output, guard.rollback_update);
    if (options.return_normalization) {
        write_value(output, return_normalizer->count);
        write_value(output, return_normalizer->mean);
        write_value(output, return_normalizer->variance);
        write_vector(output, std::span<const float>(return_normalizer->discounted_returns));
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
    if (input && magic == expected_magic && version < 6) {
        throw std::runtime_error(
            "legacy trainer state is incompatible with the full-roster move contract: " +
            path.string());
    }
    if (!input || magic != expected_magic || (version != 6 && version != 7)) {
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
    const float final_learning_rate = version >= 4
        ? read_value<float>(input, path) : learning_rate;
    const std::uint64_t anneal_updates = version >= 4
        ? read_value<std::uint64_t>(input, path) : 1;
    const float reward_scale = version >= 4 ? read_value<float>(input, path) : 1.0F;
    const float clip_range = version >= 4 ? read_value<float>(input, path) : 0.2F;
    const float value_clip_range = version >= 4 ? read_value<float>(input, path) : 0.2F;
    const float target_kl = version >= 4 ? read_value<float>(input, path) : 0.02F;
    const float value_coefficient = version >= 4 ? read_value<float>(input, path) : 0.5F;
    const float entropy_coefficient = version >= 4 ? read_value<float>(input, path) : 0.01F;
    const float final_entropy_coefficient = version >= 4
        ? read_value<float>(input, path) : entropy_coefficient;
    const float max_gradient_norm = version >= 4 ? read_value<float>(input, path) : 0.5F;
    const std::uint64_t curriculum_updates = version >= 5
        ? read_value<std::uint64_t>(input, path) : 100;
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
    const auto catalog_sha256 = read_string(input, path);
    const auto roster_version = read_string(input, path);
    const auto learner_character = read_string(input, path);
    const auto state_count = read_value<std::uint64_t>(input, path);
    const auto current_catalog = options.full_roster
        ? t8::v2::load_full_move_catalog_csv(options.full_move_catalog)
        : t8::v2::FullMoveCatalog{0, std::string(64, '0'), "compatibility"};
    const bool extended_ppo_mismatch = version >= 4 &&
        (final_learning_rate != options.final_learning_rate ||
         anneal_updates != options.anneal_updates ||
         reward_scale != options.reward_scale || clip_range != options.clip_range ||
         value_clip_range != options.value_clip_range || target_kl != options.target_kl ||
         value_coefficient != options.value_coefficient ||
         entropy_coefficient != options.entropy_coefficient ||
         final_entropy_coefficient != options.final_entropy_coefficient ||
         max_gradient_norm != options.max_gradient_norm ||
         curriculum_updates != options.curriculum_updates);
    const bool legacy_ppo_mismatch = version < 4 &&
        (options.final_learning_rate != options.learning_rate ||
         options.reward_scale != 1.0F || options.clip_range != 0.2F ||
         options.value_clip_range != 0.2F || options.target_kl != 0.02F ||
         options.value_coefficient != 0.5F || options.entropy_coefficient != 0.01F ||
         options.final_entropy_coefficient != options.entropy_coefficient ||
         options.max_gradient_norm != 0.5F);
    if (seed != options.seed || environments != options.environments || horizon != options.horizon ||
        epochs != options.epochs || minibatch != options.minibatch_size ||
        learning_rate != options.learning_rate || gamma != options.gamma ||
        gae_lambda != options.gae_lambda || extended_ppo_mismatch || legacy_ppo_mismatch ||
        checkpoint_interval != options.checkpoint_interval ||
        evaluation_interval != options.evaluation_interval ||
        evaluation_episodes != options.evaluation_episodes || sparse_reward != options.sparse_reward ||
        visual_observations != options.visual_observations ||
        full_roster != options.full_roster || curriculum_stage != options.curriculum_stage ||
        catalog_sha256 != current_catalog.catalog_sha256 ||
        roster_version != current_catalog.roster_version ||
        learner_character != options.learner_character ||
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
        if (version >= 5) {
            result.self_play_active = read_value<std::uint8_t>(input, path) != 0;
            result.learner_actions = read_vector<std::int64_t>(input, path);
            if (result.self_play_active) {
                t8::v2::TemporalEncoderState self_play_temporal{};
                self_play_temporal.history = read_vector<float>(input, path);
                self_play_temporal.previous_actions = read_vector<std::int64_t>(input, path);
                self_play_temporal.repeated_action_frames = read_vector<std::int32_t>(input, path);
                self_play_temporal.previous_own_health = read_vector<float>(input, path);
                self_play_temporal.previous_opponent_health = read_vector<float>(input, path);
                self_play_temporal.previous_distance = read_vector<float>(input, path);
                self_play_temporal.valid = read_vector<std::uint8_t>(input, path);
                result.self_play_temporal_state = std::move(self_play_temporal);
            }
        } else {
            result.learner_actions.assign(options.environments, 0);
        }
        if (result.profile_assignments.size() != options.environments ||
            result.opponent_actions.size() != options.environments ||
            result.learner_actions.size() != options.environments) {
            throw std::runtime_error("trainer-state roster lane count mismatch: " + path.string());
        }
    }
    // Version 6 runs predate both normalizers.
    const bool observation_normalization = version >= 7 && read_value<std::uint8_t>(input, path) != 0;
    const bool return_normalization = version >= 7 && read_value<std::uint8_t>(input, path) != 0;
    // Version 6 promoted purely on held-out score, which a gap limit of 1 reproduces.
    const float promotion_max_side_gap = version >= 7 ? read_value<float>(input, path) : 1.0F;
    const float promotion_tie_band = version >= 7 ? read_value<float>(input, path) : 0.0F;
    const bool screen_observations = version >= 7 && read_value<std::uint8_t>(input, path) != 0;
    const float screen_sigma_max = version >= 7
        ? read_value<float>(input, path) : options.screen_noise.position_sigma_max;
    const float screen_error_max = version >= 7
        ? read_value<float>(input, path) : options.screen_noise.event_error_max;
    // Version 6 had no regression guard.
    const std::uint8_t regression_guard = version >= 7 ? read_value<std::uint8_t>(input, path) : 0;
    const float score_drop = version >= 7 ? read_value<float>(input, path) : options.regression_score_drop;
    const float style_drop = version >= 7 ? read_value<float>(input, path) : options.regression_style_drop;
    const float guard_side_gap = version >= 7 ? read_value<float>(input, path) : options.regression_max_side_gap;
    const std::uint32_t patience = version >= 7 ? read_value<std::uint32_t>(input, path) : options.regression_patience;
    const std::uint32_t max_rollbacks = version >= 7
        ? read_value<std::uint32_t>(input, path) : options.regression_max_rollbacks;
    if (version >= 7) {
        result.guard.reference_update = read_value<std::uint64_t>(input, path);
        result.guard.reference_win_rate = read_value<double>(input, path);
        result.guard.reference_worst_style = read_value<double>(input, path);
        result.guard.consecutive_regressions = read_value<std::uint32_t>(input, path);
        result.guard.rollbacks = read_value<std::uint32_t>(input, path);
        result.guard.rollback_update = read_value<std::uint64_t>(input, path);
    }
    static constexpr std::array<const char*, 3> guard_modes = {"off", "rollback", "pause"};
    if (observation_normalization != options.observation_normalization ||
        return_normalization != options.return_normalization ||
        promotion_max_side_gap != options.promotion_max_side_gap ||
        promotion_tie_band != options.promotion_tie_band ||
        screen_observations != options.screen_observations ||
        screen_sigma_max != options.screen_noise.position_sigma_max ||
        screen_error_max != options.screen_noise.event_error_max ||
        regression_guard != options.regression_guard || score_drop != options.regression_score_drop ||
        style_drop != options.regression_style_drop || guard_side_gap != options.regression_max_side_gap ||
        patience != options.regression_patience || max_rollbacks != options.regression_max_rollbacks) {
        throw std::runtime_error(
            std::string("resume options do not match saved trainer state (saved: ") +
            "--observation-norm " + (observation_normalization ? "on" : "off") +
            " --return-norm " + (return_normalization ? "on" : "off") +
            " --promotion-max-side-gap " + std::to_string(promotion_max_side_gap) +
            " --promotion-tie-band " + std::to_string(promotion_tie_band) +
            " --regression-guard " + guard_modes[std::min<std::size_t>(regression_guard, 2)] +
            " --regression-score-drop " + std::to_string(score_drop) +
            " --regression-style-drop " + std::to_string(style_drop) +
            " --regression-max-side-gap " + std::to_string(guard_side_gap) +
            " --regression-patience " + std::to_string(patience) +
            " --regression-max-rollbacks " + std::to_string(max_rollbacks) + "): " + path.string());
    }
    if (return_normalization) {
        t8::v2::ReturnNormalizerState normalizer{};
        normalizer.count = read_value<std::uint64_t>(input, path);
        normalizer.mean = read_value<double>(input, path);
        normalizer.variance = read_value<double>(input, path);
        normalizer.discounted_returns = read_vector<float>(input, path);
        if (normalizer.discounted_returns.size() != options.environments) {
            throw std::runtime_error("trainer-state return normalizer lane count mismatch: " + path.string());
        }
        result.return_normalizer = std::move(normalizer);
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

void append_decision_stats_json(std::ostream& output, std::string_view name, const DecisionStats& stats) {
    const double choices = std::max<double>(1.0, static_cast<double>(stats.choice_decisions));
    output << '"' << name << "\":{\"live_decisions\":" << stats.live_decisions
           << ",\"choice_fraction\":"
           << static_cast<double>(stats.choice_decisions) /
                  std::max<double>(1.0, static_cast<double>(stats.live_decisions))
           << ",\"choice_entropy\":" << stats.choice_entropy / choices << ",\"choice_actions\":{";
    bool first = true;
    for (std::size_t action = 0; action < t8::v2::kActionCount; ++action) {
        if (stats.choice_actions[action] == 0) continue;
        output << (first ? "" : ",") << '"' << t8::v2::kActionNames[action] << "\":"
               << static_cast<double>(stats.choice_actions[action]) / choices;
        first = false;
    }
    output << "}}";
}

// --probe-checkpoint: held-out evaluation of one checkpoint with behaviour
// statistics (how often it has a choice, how random it is then, and what it
// picks), per side and for deterministic and sampled play. Read-only.
int probe_checkpoint(
    const Options& options,
    const t8::v2::ActorCriticConfig& actor_config,
    std::span<const t8::v2::OpponentProfileParameters> profiles,
    std::span<const t8::v2::CharacterMoveParameters> character_moves,
    std::uint32_t learner_character_id) {
    t8::v2::GpuActorCritic learner(256, actor_config, options.seed);
    learner.load_checkpoint(options.probe_checkpoint, false);
    std::cout << std::setprecision(6) << "{\"checkpoint\":\""
              << options.probe_checkpoint.generic_string() << "\",\"episodes\":" << options.probe_episodes;
    for (const bool deterministic : {true, false}) {
        DecisionStats p1{};
        DecisionStats p2{};
        const auto result = evaluate(
            learner, options.probe_episodes, options.seed + 500'000, options.visual_observations,
            options.full_roster, profiles, character_moves, learner_character_id, deterministic,
            &p1, &p2, screen_noise_for(options));
        std::cout << ",\"" << (deterministic ? "deterministic" : "stochastic") << "\":{";
        append_summary_json(std::cout, "total", result.total);
        std::cout << ',';
        append_summary_json(std::cout, "as_p1", result.as_p1);
        std::cout << ',';
        append_summary_json(std::cout, "as_p2", result.as_p2);
        std::cout << ',';
        append_decision_stats_json(std::cout, "decisions_p1", p1);
        std::cout << ',';
        append_decision_stats_json(std::cout, "decisions_p2", p2);
        std::cout << '}';
    }
    std::cout << "}\n";
    return EXIT_SUCCESS;
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
            options.full_move_catalog = resolve_generated_catalog(
                options.full_move_catalog, "data/generated/full_move_catalog.csv", argv[0]);
        }
        const bool sample_all_learner_characters = options.learner_character == "all";
        const std::uint32_t learner_character_id = sample_all_learner_characters
            ? t8::v2::kJunCharacterId
            : t8::v2::character_id_from_slug(options.learner_character);
        const auto metrics_path = options.run_directory / "metrics.jsonl";
        const bool probing = !options.probe_checkpoint.empty();
        std::optional<ResumeState> resume_state;
        if (probing) {
            // Read-only: never touches a run directory.
        } else if (!options.resume_checkpoint.empty()) {
            resume_state = load_trainer_state(trainer_state_path(options.resume_checkpoint), options);
            validate_metrics_for_resume(metrics_path, resume_state->completed_update);
            truncate_evaluation_exports(options.run_directory, resume_state->completed_update);
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
        if (!probing) std::filesystem::create_directories(options.run_directory / "checkpoints");
        std::vector<t8::v2::OpponentProfileParameters> roster_profiles;
        std::vector<t8::v2::CharacterMoveParameters> roster_character_moves;
        std::optional<t8::v2::FullMoveCatalog> full_move_catalog;
        std::unique_ptr<t8::v2::MatchupScheduler> matchup_scheduler;
        if (options.full_roster) {
            roster_profiles = t8::v2::load_opponent_profiles_csv(options.opponent_catalog);
            roster_character_moves =
                t8::v2::load_character_move_specs_csv(options.character_move_catalog);
            full_move_catalog = t8::v2::load_full_move_catalog_csv(options.full_move_catalog);
            if (sample_all_learner_characters) {
                throw std::runtime_error(
                    "--learner-character all is gated until at least one full-move character "
                    "passes measured CPU/CUDA parity and Practice validation");
            }
            if (!sample_all_learner_characters &&
                full_move_catalog->moves_for_character(learner_character_id).empty()) {
                throw std::runtime_error(
                    "learner character has no compiled move data: " + options.learner_character);
            }
            matchup_scheduler = std::make_unique<t8::v2::MatchupScheduler>(roster_profiles, options.seed + 700'000);
        }
        const std::size_t policy_capacity = std::max(options.environments, options.minibatch_size);
        t8::v2::ActorCriticConfig actor_config{};
        if (options.full_roster) {
            actor_config.observation_size = static_cast<int>(options.visual_observations
                ? t8::v2::kMatchupVisualObservationSize : t8::v2::kMatchupPrivilegedObservationSize);
            actor_config.catalog_sha256 = full_move_catalog->catalog_sha256;
            actor_config.roster_version = full_move_catalog->roster_version;
            actor_config.observation_contract = options.screen_observations ? "screen-matchup-95-v1"
                : options.visual_observations ? "visual-matchup-95-v2" : "privileged-matchup-101-v2";
            actor_config.action_contract = "compatibility-six-v3";
        } else {
            actor_config.observation_size = static_cast<int>(options.visual_observations
                ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize);
            actor_config.observation_contract = options.visual_observations
                ? "visual-13-v2" : "privileged-19-v2";
        }
        actor_config.observation_normalization = options.observation_normalization;
        if (probing) {
            return probe_checkpoint(options, actor_config, roster_profiles, roster_character_moves,
                                    learner_character_id);
        }
        t8::v2::Config training_config{};
        training_config.timeout_ties_are_draws = true;
        training_config.screen_observations = options.screen_observations;
        training_config.screen_noise = options.screen_noise;
        training_config.randomize_initial_positions = true;
        t8::v2::GpuSimulatorBatch simulator(options.environments, training_config);
        t8::v2::GpuActorCritic learner(policy_capacity, actor_config, options.seed);
        std::unique_ptr<t8::v2::GpuActorCritic> latest_self_play_opponent;
        std::unique_ptr<t8::v2::GpuActorCritic> best_self_play_opponent;
        t8::v2::GpuScriptedOpponent opponent(options.environments);
        t8::v2::GpuScriptedOpponent learner_action_history(options.environments);
        t8::v2::GpuLearnerSideRouter side_router(options.environments);
        t8::v2::GpuRolloutBuffer rollout(options.environments, options.horizon, actor_config);
        std::unique_ptr<t8::v2::GpuTemporalMatchupEncoder> temporal;
        std::unique_ptr<t8::v2::GpuTemporalMatchupEncoder> self_play_temporal;
        if (options.full_roster) {
            simulator.set_character_move_specs(roster_character_moves);
            opponent.set_profiles(roster_profiles);
            temporal = std::make_unique<t8::v2::GpuTemporalMatchupEncoder>(
                options.environments,
                options.visual_observations ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize,
                screen_noise_for(options));
            self_play_temporal = std::make_unique<t8::v2::GpuTemporalMatchupEncoder>(
                options.environments,
                options.visual_observations ? t8::v2::kVisualObservationSize : t8::v2::kObservationSize,
                screen_noise_for(options));
            latest_self_play_opponent = std::make_unique<t8::v2::GpuActorCritic>(
                policy_capacity, actor_config, options.seed + 900'000);
            best_self_play_opponent = std::make_unique<t8::v2::GpuActorCritic>(
                policy_capacity, actor_config, options.seed + 900'001);
        }
        bool self_play_active = false;
        if (resume_state) {
            learner.load_checkpoint(options.resume_checkpoint);
            if (learner.config().observation_normalization != options.observation_normalization) {
                throw std::runtime_error(
                    "resume checkpoint observation normalization does not match --observation-norm: " +
                    options.resume_checkpoint.string());
            }
            if (resume_state->return_normalizer) {
                rollout.restore_return_normalizer_state(*resume_state->return_normalizer);
            }
            // The checkpoint holds the evaluated (regressed) weights; the run had
            // already rolled back to the reference when this state was written.
            if (resume_state->guard.rollback_update != 0 &&
                resume_state->guard.rollback_update == resume_state->completed_update) {
                learner.load_checkpoint(checkpoint_path(
                    options.run_directory, resume_state->guard.reference_update));
            }
            simulator.upload_states(resume_state->simulator_states);
            if (options.full_roster) {
                opponent.set_profile_assignments(resume_state->profile_assignments);
                simulator.set_opponent_characters_device(
                    opponent.profiles_device(), opponent.profile_count(),
                    opponent.profile_assignments_device(), 0, nullptr, nullptr,
                    learner_character_id);
                opponent.set_action_history(resume_state->opponent_actions);
                learner_action_history.set_action_history(resume_state->learner_actions);
                matchup_scheduler->restore_state(
                    resume_state->scheduler_stats, resume_state->scheduler_random_state);
                temporal->upload_state(*resume_state->temporal_state);
                self_play_active = resume_state->self_play_active;
                if (self_play_active) {
                    if (!resume_state->self_play_temporal_state) {
                        throw std::runtime_error("self-play resume is missing opponent temporal state");
                    }
                    self_play_temporal->upload_state(*resume_state->self_play_temporal_state);
                }
            }
        } else if (options.full_roster) {
            const auto [initial_stage, initial_groups] = curriculum_for_update(options, 1);
            matchup_scheduler->set_stage(initial_stage, initial_groups);
            opponent.set_profile_assignments(
                matchup_scheduler->sample_profile_indices(options.environments, true));
            simulator.set_opponent_characters_device(
                opponent.profiles_device(), opponent.profile_count(),
                opponent.profile_assignments_device(), 0, nullptr, nullptr,
                learner_character_id);
        }

        t8::v2::PpoUpdateConfig update_config{};
        update_config.epochs = options.epochs;
        update_config.minibatch_size = options.minibatch_size;
        update_config.learning_rate = options.learning_rate;
        update_config.clip_range = options.clip_range;
        update_config.value_clip_range = options.value_clip_range;
        update_config.target_kl = options.target_kl;
        update_config.value_coefficient = options.value_coefficient;
        update_config.entropy_coefficient = options.entropy_coefficient;
        update_config.max_gradient_norm = options.max_gradient_norm;
        const auto started = std::chrono::steady_clock::now();
        const double elapsed_before_resume = resume_state ? resume_state->elapsed_seconds : 0.0;
        const std::string reward_mode = options.sparse_reward ? "sparse" : "shaped";
        const std::string observation_mode = options.full_roster
            ? (options.screen_observations ? "screen_matchup_temporal"
               : options.visual_observations ? "visual_matchup_temporal" : "privileged_matchup_temporal")
            : (options.visual_observations ? "visual" : "privileged");
        const std::size_t first_update = resume_state ? resume_state->completed_update + 1 : 1;
        EvaluationLedger evaluation_ledger(metrics_path);
        RegressionGuardState regression_guard = resume_state ? resume_state->guard : RegressionGuardState{};
        // Opponent weights only change when a new checkpoint is selected;
        // reloading identical files each update is pure overhead.
        std::filesystem::path loaded_latest_checkpoint;
        std::filesystem::path loaded_best_checkpoint;

        for (std::size_t update = first_update; update <= options.updates; ++update) {
            std::vector<std::uint32_t> pending_profile_assignments;
            std::optional<SelfPlaySelection> self_play_selection;
            bool use_self_play = false;
            if (options.full_roster) {
                const auto [stage, group_mask] = curriculum_for_update(options, update);
                matchup_scheduler->set_stage(stage, group_mask);
                if (stage == t8::v2::CurriculumStage::AdversarialLeague) {
                    self_play_selection = select_self_play_checkpoint(
                        options.run_directory / "checkpoints", evaluation_ledger, update,
                        options.promotion_max_side_gap, options.promotion_tie_band);
                    use_self_play = self_play_selection.has_value();
                }
                if (use_self_play) {
                    if (!self_play_active) {
                        simulator.reset_seeded(options.seed + update * options.horizon);
                        temporal->reset();
                        self_play_temporal->reset();
                        const std::vector<std::uint32_t> jun_assignments(
                            options.environments, jun_profile_index(roster_profiles));
                        opponent.set_profile_assignments(jun_assignments);
                        simulator.set_opponent_characters_device(
                            opponent.profiles_device(), opponent.profile_count(),
                            opponent.profile_assignments_device(), 0, nullptr, nullptr,
                            learner_character_id);
                        opponent.set_action_history(
                            std::vector<std::int64_t>(options.environments, 0));
                    }
                    if (self_play_selection->latest_checkpoint != loaded_latest_checkpoint) {
                        latest_self_play_opponent->load_checkpoint(
                            self_play_selection->latest_checkpoint, false);
                        loaded_latest_checkpoint = self_play_selection->latest_checkpoint;
                    }
                    if (self_play_selection->best_older_checkpoint != loaded_best_checkpoint) {
                        best_self_play_opponent->load_checkpoint(
                            self_play_selection->best_older_checkpoint, false);
                        loaded_best_checkpoint = self_play_selection->best_older_checkpoint;
                    }
                } else {
                    pending_profile_assignments =
                        matchup_scheduler->sample_profile_indices(options.environments, true);
                    if (self_play_active) {
                        simulator.reset_seeded(options.seed + update * options.horizon);
                        temporal->reset();
                        self_play_temporal->reset();
                        opponent.set_profile_assignments(pending_profile_assignments);
                        simulator.set_opponent_characters_device(
                            opponent.profiles_device(), opponent.profile_count(),
                            opponent.profile_assignments_device(), 0, nullptr, nullptr,
                            learner_character_id);
                    }
                }
                self_play_active = use_self_play;
            }
            for (std::size_t step = 0; step < options.horizon; ++step) {
                const auto before = simulator.device_view();
                const auto routed_inputs = options.visual_observations
                    ? (use_self_play
                        ? side_router.select_self_play_visual_observations(before, options.environments)
                        : side_router.select_visual_observations(before, options.environments))
                    : side_router.select_observations(before, options.environments);
                const float* policy_observations = routed_inputs.learner_observations;
                const float* opponent_policy_observations = routed_inputs.opponent_observations;
                if (temporal) {
                    policy_observations = temporal->encode(
                        policy_observations, opponent.profiles_device(), opponent.profile_count(),
                        opponent.profile_assignments_device(), opponent.actions_buffer_device(),
                        options.environments);
                    if (use_self_play) {
                        opponent_policy_observations = self_play_temporal->encode(
                            opponent_policy_observations,
                            opponent.profiles_device(), opponent.profile_count(),
                            opponent.profile_assignments_device(),
                            learner_action_history.actions_buffer_device(),
                            options.environments);
                    }
                }
                const auto learner_output = learner.forward(
                    policy_observations, routed_inputs.learner_action_masks,
                    options.environments, options.seed + 1,
                    update * options.horizon + step, false);
                learner_action_history.set_action_history_device(
                    learner_output.actions, options.environments);
                const std::int64_t* opponent_actions = nullptr;
                if (use_self_play) {
                    const auto latest_output = latest_self_play_opponent->forward(
                        opponent_policy_observations, routed_inputs.opponent_action_masks,
                        options.environments, options.seed + 2,
                        update * options.horizon + step, false);
                    const auto best_output = best_self_play_opponent->forward(
                        opponent_policy_observations, routed_inputs.opponent_action_masks,
                        options.environments, options.seed + 4,
                        update * options.horizon + step, false);
                    opponent_actions = side_router.mix_self_play_actions(
                        latest_output.actions, best_output.actions, options.environments);
                    opponent.set_action_history_device(opponent_actions, options.environments);
                } else {
                    opponent_actions = opponent.actions_device(
                        routed_inputs.opponent_observations, routed_inputs.opponent_action_masks,
                        options.environments, options.seed + 2,
                        update * options.horizon + step,
                        t8::v2::ScriptedOpponentSet::TrainingV1);
                }
                const auto routed_actions = side_router.route_actions(
                    learner_output.actions, opponent_actions, options.environments);
                rollout.record_policy_device(
                    step, policy_observations, routed_inputs.learner_action_masks,
                    learner_output.actions, learner_output.log_probabilities, learner_output.values);
                simulator.step_device_i64(routed_actions.p1_actions, routed_actions.p2_actions);
                const auto after = simulator.device_view();
                if (options.full_roster && !use_self_play) {
                    // Count before finished lanes are reassigned to new profiles.
                    side_router.tally_outcomes(
                        after.sparse_rewards_p1, after.sparse_rewards_p2, after.terminated,
                        opponent.profile_assignments_device(), opponent.profile_count(),
                        options.environments);
                }
                const float* rewards = side_router.select_rewards(
                    options.sparse_reward ? after.sparse_rewards_p1 : after.rewards_p1,
                    options.sparse_reward ? after.sparse_rewards_p2 : after.rewards_p2,
                    options.environments);
                const auto routed_next = options.visual_observations
                    ? side_router.select_visual_observations(after, options.environments)
                    : side_router.select_observations(after, options.environments);
                const float* next_policy_observations = routed_next.learner_observations;
                if (temporal) {
                    next_policy_observations = temporal->preview(
                        next_policy_observations, opponent.profiles_device(), opponent.profile_count(),
                        opponent.profile_assignments_device(), opponent.actions_buffer_device(),
                        options.environments);
                }
                const auto next_output = learner.forward(
                    next_policy_observations, routed_next.learner_action_masks,
                    options.environments, options.seed + 3,
                    update * options.horizon + step, true);
                rollout.record_outcome_device(
                    step, rewards, after.terminated, after.truncated,
                    next_output.values, options.reward_scale);
                if (temporal) {
                    temporal->reset_done(after.terminated, options.environments);
                    if (use_self_play) {
                        self_play_temporal->reset_done(after.terminated, options.environments);
                    } else {
                        opponent.set_profile_assignments_for_done(
                            pending_profile_assignments, after.terminated);
                        simulator.set_opponent_characters_device(
                            opponent.profiles_device(), opponent.profile_count(),
                            opponent.profile_assignments_device(), 0, after.terminated, nullptr,
                            learner_character_id);
                    }
                }
                simulator.reset_done_seeded(
                    options.seed + update * options.horizon + step);
            }
            std::optional<float> return_standard_deviation;
            if (options.return_normalization) {
                return_standard_deviation = rollout.normalize_rewards(options.gamma);
            }
            rollout.compute_gae(options.gamma, options.gae_lambda, true);
            const float schedule_progress = options.anneal_updates <= 1 ? 1.0F :
                std::min(1.0F, static_cast<float>(update - 1) /
                                   static_cast<float>(options.anneal_updates - 1));
            update_config.learning_rate = options.learning_rate +
                (options.final_learning_rate - options.learning_rate) * schedule_progress;
            update_config.entropy_coefficient = options.entropy_coefficient +
                (options.final_entropy_coefficient - options.entropy_coefficient) * schedule_progress;
            const auto metrics = learner.update_ppo(
                rollout.device_view(), update_config, options.seed + update * 10'000);
            if (options.full_roster && !use_self_play) {
                record_training_outcomes(
                    *matchup_scheduler, side_router.take_outcome_tally(opponent.profile_count()));
            }

            std::optional<Evaluation> deterministic_evaluation;
            std::optional<Evaluation> stochastic_evaluation;
            if (update % options.evaluation_interval == 0 || update == options.updates) {
                // Frozen benchmark weights and stochastic sequence for every
                // update, shared by shaped/sparse runs with the same seed.
                deterministic_evaluation = evaluate(
                    learner, options.evaluation_episodes, options.seed + 500'000,
                    options.visual_observations, options.full_roster,
                    roster_profiles, roster_character_moves, learner_character_id, true,
                    nullptr, nullptr, screen_noise_for(options));
                stochastic_evaluation = evaluate(
                    learner, options.evaluation_episodes, options.seed + 500'000,
                    options.visual_observations, options.full_roster,
                    roster_profiles, roster_character_moves, learner_character_id, false,
                    nullptr, nullptr, screen_noise_for(options));
                if (options.full_roster) {
                    t8::v2::write_matchup_matrix_json(
                        options.run_directory / "matchup_matrix.json",
                        roster_profiles, matchup_scheduler->all_stats());
                }
            }
            bool save_checkpoint = update % options.checkpoint_interval == 0 || update == options.updates;
            std::optional<GuardDecision> guard_decision;
            if (deterministic_evaluation) {
                guard_decision = regression_guard_step(
                    regression_guard, options, update, *deterministic_evaluation, save_checkpoint);
                // A pause must leave a resumable checkpoint behind.
                if (guard_decision->action == GuardAction::Pause) save_checkpoint = true;
            }
            const double elapsed = elapsed_before_resume + std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const std::uint64_t environment_steps = static_cast<std::uint64_t>(update) *
                static_cast<std::uint64_t>(options.environments) *
                static_cast<std::uint64_t>(options.horizon);
            append_metrics(metrics_path, update, environment_steps, reward_mode, observation_mode,
                           metrics, elapsed, deterministic_evaluation, stochastic_evaluation,
                           self_play_selection, return_standard_deviation,
                           learner.observation_count(), guard_decision);
            if (deterministic_evaluation) {
                export_evaluation(options.run_directory, update, environment_steps, elapsed,
                                  *deterministic_evaluation, *stochastic_evaluation,
                                  self_play_selection, guard_decision, options.full_roster);
            }
            if (save_checkpoint) {
                const auto checkpoint = checkpoint_path(options.run_directory, update);
                learner.save_checkpoint(checkpoint);
                std::optional<t8::v2::ReturnNormalizerState> return_normalizer;
                if (options.return_normalization) return_normalizer = rollout.return_normalizer_state();
                if (options.full_roster) {
                    const auto assignments = opponent.download_profile_assignments(options.environments);
                    const auto actions = opponent.download_actions(options.environments);
                    const auto learner_actions =
                        learner_action_history.download_actions(options.environments);
                    const auto temporal_state = temporal->download_state();
                    std::optional<t8::v2::TemporalEncoderState> opponent_temporal_state;
                    if (self_play_active) {
                        opponent_temporal_state = self_play_temporal->download_state();
                    }
                    save_trainer_state(
                        trainer_state_path(checkpoint), options, update, environment_steps,
                        elapsed, simulator.download_states(), assignments, actions,
                        matchup_scheduler->all_stats(), matchup_scheduler->random_state(),
                        &temporal_state, learner_actions, self_play_active,
                        opponent_temporal_state ? &*opponent_temporal_state : nullptr,
                        return_normalizer ? &*return_normalizer : nullptr, regression_guard);
                } else {
                    save_trainer_state(
                        trainer_state_path(checkpoint), options, update, environment_steps,
                        elapsed, simulator.download_states(), {}, {}, {}, 0, nullptr, {}, false,
                        nullptr, return_normalizer ? &*return_normalizer : nullptr, regression_guard);
                }
            }
            if (guard_decision && guard_decision->action != GuardAction::None) {
                // Checkpoint `update` keeps the evaluated weights for inspection;
                // training continues (or resumes) from the healthy reference.
                learner.load_checkpoint(checkpoint_path(options.run_directory, guard_decision->reference_update));
                std::cout << "regression_guard=" << (guard_decision->action == GuardAction::Pause
                                                         ? "pause" : "rollback")
                          << " update=" << update << " reference=" << guard_decision->reference_update
                          << " reasons=";
                for (std::size_t index = 0; index < guard_decision->reasons.size(); ++index) {
                    std::cout << (index == 0 ? "" : ",") << guard_decision->reasons[index];
                }
                std::cout << '\n';
                if (guard_decision->action == GuardAction::Pause) {
                    std::cout << "training paused by the regression guard; inspect metrics, then resume "
                                 "from update_" << update << ".t8ppo\n";
                    return kRegressionPauseExitCode;
                }
            }
            std::cout << "update=" << update << '/' << options.updates
                      << " steps=" << environment_steps
                      << " reward=" << reward_mode
                      << " observations=" << observation_mode
                      << " opponents=" << (options.full_roster ? "roster" : "legacy")
                      << " policy_loss=" << metrics.policy_loss
                      << " value_loss=" << metrics.value_loss
                      << " entropy=" << metrics.entropy
                      << " decision_entropy=" << metrics.decision_entropy;
            if (return_standard_deviation) std::cout << " return_std=" << *return_standard_deviation;
            if (self_play_selection) {
                std::cout << " self_play=80%latest:" << self_play_selection->latest_update
                          << "/20%best:" << self_play_selection->best_older_update;
            }
            if (deterministic_evaluation) {
                std::cout << " eval_win_rate=" << deterministic_evaluation->win_rate
                          << " stochastic_eval_win_rate=" << stochastic_evaluation->win_rate;
            }
            std::cout << '\n';
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "training error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
