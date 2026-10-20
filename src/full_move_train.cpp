// Full-move PPO trainer.
//
// One policy learns to play every chosen character (--learner-character jun,
// a comma list, or all), each with its complete candidate list (the 22
// universal actions plus every catalog move and stance branch), through the
// CUDA full-combat simulator. Every fight picks the learner's character and the
// opponent's; the policy scores each fight's own character's candidates (one
// candidate table per character) and sees both characters in its observation.
// Opponents are the scripted full-move opponent across the roster or, in
// self-play updates, a frozen checkpoint of the same policy playing another of
// the learner characters, so the opponents improve along with the learner.
//
// Champion: every --champion-interval-steps the latest policy plays the
// current champion head-to-head (--champion-rounds rounds, every learner
// pairing, both sides). A round share of at least --champion-promote crowns it
// (saved under <run>/champion/); under --champion-rollback the loss streak
// grows, and after --champion-rollback-after such matches in a row the policy is
// reloaded from the champion (policy and value share one network, so both come
// back together, with the champion's return normalizer and entropy boost; the
// optimizer restarts and the step count keeps going). The
// first match crowns the current policy. The visualizer and play use the
// champion: the best policy found, not the newest.
//
// Entropy floor: while the policy's per-choice entropy is below --entropy-target,
// the entropy bonus is raised (up to --max-entropy-coefficient), so the policy
// keeps exploring instead of collapsing onto a few moves. The
// policy is the parametric scorer (state query dot candidate features), legal
// masks are computed on the GPU every decision, and nothing crosses PCIe in the
// rollout loop.
//
// League (self-play updates): opponents are drawn from a pool of the learner's
// own checkpoints, weighted toward the latest, toward comparable ratings
// (ratings follow the self-play results), and uniformly for diversity; extra
// fixed policies (for example exploiters) can join the pool. --exploit-target
// trains a fresh best response against one frozen checkpoint instead, whose
// final win rate measures that checkpoint's exploitability. A checkpoint is
// promoted only if it keeps its held-out score, stays side balanced, and does
// not lose to the recently promoted ones (catastrophic-forgetting gate).
//
// Until a character passes its gate, simulation needs stand-in geometry for
// unmeasured moves (--allow-provisional). Such runs are development runs: the
// run directory, metrics, and checkpoint state record that they are provisional.

#include "t8_v2/full_move_gpu.hpp"
#include "t8_v2/ppo.hpp"
#include "t8_v2/roster.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <cctype>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace t8::v2;

struct Options {
    std::vector<std::string> learner_characters = {"jun"};  // or {"all"}
    std::vector<std::string> opponents = {"roster"};
    bool allow_provisional = false;
    std::size_t environments = 4096;
    std::size_t horizon = 64;
    std::size_t updates = 100;
    int epochs = 4;
    std::size_t minibatch_size = 8192;
    int hidden_size = 256;
    std::string matmul_precision = "fp32";  // fp32 | tf32 | bf16
    float learning_rate = 3e-4F;
    float final_learning_rate = 3e-5F;
    std::size_t anneal_updates = 0;  // 0: the run length
    float gamma = 0.99F;
    float gae_lambda = 0.95F;
    float reward_scale = 1.0F;
    float clip_range = 0.2F;
    float value_clip_range = 0.2F;
    float target_kl = 0.02F;
    float value_coefficient = 0.5F;
    float entropy_coefficient = 0.01F;
    float final_entropy_coefficient = 0.001F;
    float entropy_target = 1.0F;           // nats per choice; 0 = no floor
    float entropy_target_gain = 0.5F;      // multiplicative response per nat of shortfall
    float max_entropy_coefficient = 0.1F;
    float max_gradient_norm = 0.5F;
    bool sparse_reward = false;
    bool return_normalization = true;
    bool observation_normalization = true;
    float self_play_share = 0.0F;          // fraction of updates in mirror self-play
    std::size_t self_play_start = 50;      // first update that may use self-play
    // Adaptive scripted share (0 = off: a fixed 1 - self_play_share). The share of
    // scripted updates starts at 1 - self_play_share and shrinks toward this floor
    // while the learner wins at least 95% of its scripted rounds; under 90% it grows
    // back, so the scripted roster stays a regression check without dominating.
    float scripted_floor = 0.0F;
    float draw_penalty = 0.0F;             // a 0-0 or equal-health timeout scores -draw_penalty (a loss for both)
    float throw_break_penalty = 0.0F;      // per broken throw of the learner
    float time_penalty = 0.0F;             // per second of round time (dense time pressure)
    // Learner characters: 0 = every one equally often; otherwise this share of the
    // environments goes to characters in proportion to how much they lose in self-play
    // (1 - their score), so the weakest get the most practice.
    float learner_weakness_weight = 0.0F;
    // Tiered scripted bots (full_move_contract.hpp, 331 bots in 7 tiers from dummies to
    // pro styles) instead of the random scripted opponent. The ladder starts at tier 0;
    // scripted rounds draw 60% from the current tier, 25% from lower tiers, 15% from the
    // next; at a score of tier_promote over tier_min_updates scripted updates it moves up,
    // under tier_demote it moves down. The scripted share only shrinks once the top tier
    // is beaten.
    bool scripted_tiers = false;
    // Self-play opponent pool: the newest checkpoint's share anneals from 0.8 to 0.1 over
    // the schedule (Oh et al. 2019) instead of a fixed 0.4.
    bool pool_anneal = false;
    int hold_frames = 4;                   // movement and guard inputs are held this long
    int perception_delay = 0;              // frames before the opponent's move is visible
    bool memory = false;                   // observation adds recent moves and habits (full_core::Memory)
    bool live_noise = false;               // screen-reading noise and input latency (full_core::NoiseLevels)
    bool screen_view = false;              // only the features the live screen reader provides
    float noise_position = 0.25F;          // their ranges, drawn per environment and round
    float noise_flip = 0.10F;
    int latency_min = 2;
    int latency_max = 6;
    int shared_layers = 2;                 // network depth (ActorCriticConfig)
    int head_layers = 0;
    float damage_dealt_weight = 1.0F;      // fighting-style HP ratio (defensive 0.8 / 1.2)
    float damage_taken_weight = 1.0F;
    float distance_penalty = 0.0F;         // per metre per second (aggressive style)
    // Other runs' champion directories: their newest champions join this run's pool as
    // they appear (a shared pool across fighting styles). Not reproduced by exact resume.
    std::vector<std::filesystem::path> league_follow;
    float tier_promote = 0.8F;
    float tier_demote = 0.45F;
    std::size_t tier_min_updates = 20;
    std::size_t stop_at_update = 0;        // tests: behave as if save-and-quit was requested at this update
    float weakness_weight = 0.5F;          // opponent sampling: uniform vs weakest-first
    float scenario_share = 0.25F;          // rounds starting in a scenario curriculum situation
    bool self_play_scenarios = false;      // scenario starts in self-play rounds too (else neutral there)
    std::size_t league_pool_size = 8;      // kept non-promoted checkpoints
    std::vector<std::filesystem::path> league_extras;  // fixed extra opponents (same character)
    std::optional<std::filesystem::path> exploit_target;
    float promotion_tolerance = 0.05F;     // held-out score may not drop more than this below the best
    float promotion_max_side_gap = 0.10F;
    float promotion_min_score = 0.45F;     // against each recently promoted checkpoint
    std::size_t regression_set_size = 4;
    std::uint64_t champion_interval_steps = 500'000'000;  // 0 = off
    std::size_t champion_rounds = 1024;
    float champion_promote = 0.53F;
    float champion_rollback = 0.45F;
    int champion_rollback_after = 2;
    int max_round_frames = 3600;
    std::size_t checkpoint_interval = 10;
    std::size_t evaluation_interval = 10;
    std::size_t evaluation_episodes = 512;
    std::uint64_t seed = 2027;
    std::filesystem::path run_directory = "runs/full_move";
    std::filesystem::path data_directory;
    std::optional<std::filesystem::path> resume;
    bool fork = false;  // --fork: resume a checkpoint's whole state under changed settings (an A/B branch)
    bool smoke = false;
};

[[noreturn]] void usage(const std::string& message) {
    std::cerr << "error: " << message << "\n"
              << "usage: t8_v2_full_move_train --learner-character jun|all|slug,slug [--opponents roster|slug,slug]\n"
              << "  --entropy-target NATS --entropy-target-gain F --max-entropy-coefficient F\n"
              << "  --champion-interval-steps N --champion-rounds N --champion-promote F --champion-rollback F\n"
              << "  --champion-rollback-after N\n"
              << "  --shared-layers N --head-layers N (network depth: shared layers, then separate policy/value layers)\n"
              << "  --allow-provisional --envs N --horizon N --updates N --epochs N --minibatch N\n"
              << "  --draw-penalty F (draws score -F: stalling is a loss for both)  --throw-break-penalty F (per broken throw)\n"
              << "  --time-penalty F (per second of round time, dense pressure to act)\n"
              << "  --learner-weakness-weight F (share of environments given to characters that lose most in self-play)\n"
              << "  --scripted-tiers [--tier-promote F --tier-demote F --tier-min-updates N] (tiered scripted bots)\n"
              << "  --pool-anneal --hold-frames N --damage-dealt-weight F --damage-taken-weight F --distance-penalty F\n"
              << "  --perception-delay N (frames before the opponent's move is visible to the policy)\n"
              << "  --memory (the policy also sees each fighter's last moves and recent habits)\n"
              << "  --live-noise [--noise-position M --noise-flip P --latency-min F --latency-max F]\n"
              << "    (train as the screen reader and controller play: noisy readings, late inputs)\n"
              << "  --screen-observation (only what the live screen reader sees: full_core::apply_screen_view)\n"
              << "  --league-follow DIR (another run's champion directory; its newest champions join the pool)\n"
              << "  --scripted-floor F (adaptive scripted share: shrinks to F while scripted rounds are >=95% won)\n"
              << "  --matmul fp32|tf32|bf16 (tensor-core matrix multiplies; FP32 storage and optimizer)\n"
              << "  --learning-rate F --final-learning-rate F --anneal-updates N --gamma F --gae-lambda F\n"
              << "  --entropy-coefficient F --final-entropy-coefficient F --reward shaped|sparse\n"
              << "  --self-play-share F --self-play-start N --weakness-weight F --max-round-frames N\n"
              << "  --league-pool-size N --league-extra CHECKPOINT --exploit-target CHECKPOINT --scenario-share F\n"
              << "  --self-play-scenarios (self-play rounds also start in scenarios: punishes, throw breaks, ...)\n"
              << "  --promotion-tolerance F --promotion-max-side-gap F --promotion-min-score F\n"
              << "  --checkpoint-interval N --eval-interval N --eval-episodes N --seed N --run-dir PATH\n"
              << "  --data-dir PATH --resume CHECKPOINT --fork CHECKPOINT --smoke\n"
              << "  (--fork continues a checkpoint's full state, league and champion included, under changed\n"
              << "   settings; copy the run directory to the new --run-dir first)\n";
    std::exit(2);
}

std::vector<std::string> split(const std::string& text, char separator) {
    std::vector<std::string> parts;
    std::stringstream stream(text);
    std::string part;
    while (std::getline(stream, part, separator)) {
        if (!part.empty()) parts.push_back(part);
    }
    return parts;
}

Options parse(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) usage("missing value for " + argument);
            return argv[++index];
        };
        if (argument == "--learner-character" || argument == "--learner-characters") {
            options.learner_characters = split(next(), ',');
        }
        else if (argument == "--entropy-target") options.entropy_target = std::stof(next());
        else if (argument == "--entropy-target-gain") options.entropy_target_gain = std::stof(next());
        else if (argument == "--max-entropy-coefficient") options.max_entropy_coefficient = std::stof(next());
        else if (argument == "--opponents") options.opponents = split(next(), ',');
        else if (argument == "--allow-provisional") options.allow_provisional = true;
        else if (argument == "--envs") options.environments = std::stoull(next());
        else if (argument == "--horizon") options.horizon = std::stoull(next());
        else if (argument == "--updates") options.updates = std::stoull(next());
        else if (argument == "--epochs") options.epochs = std::stoi(next());
        else if (argument == "--minibatch") options.minibatch_size = std::stoull(next());
        else if (argument == "--hidden") options.hidden_size = std::stoi(next());
        else if (argument == "--matmul") options.matmul_precision = next();
        else if (argument == "--learning-rate") options.learning_rate = std::stof(next());
        else if (argument == "--final-learning-rate") options.final_learning_rate = std::stof(next());
        else if (argument == "--anneal-updates") options.anneal_updates = std::stoull(next());
        else if (argument == "--gamma") options.gamma = std::stof(next());
        else if (argument == "--gae-lambda") options.gae_lambda = std::stof(next());
        else if (argument == "--entropy-coefficient") options.entropy_coefficient = std::stof(next());
        else if (argument == "--final-entropy-coefficient") options.final_entropy_coefficient = std::stof(next());
        else if (argument == "--target-kl") options.target_kl = std::stof(next());
        else if (argument == "--reward") {
            const auto value = next();
            if (value != "shaped" && value != "sparse") usage("--reward must be shaped or sparse");
            options.sparse_reward = value == "sparse";
        } else if (argument == "--no-return-normalization") options.return_normalization = false;
        else if (argument == "--no-observation-normalization") options.observation_normalization = false;
        else if (argument == "--self-play-share") options.self_play_share = std::stof(next());
        else if (argument == "--self-play-start") options.self_play_start = std::stoull(next());
        else if (argument == "--scripted-floor") options.scripted_floor = std::stof(next());
        else if (argument == "--draw-penalty") options.draw_penalty = std::stof(next());
        else if (argument == "--throw-break-penalty") options.throw_break_penalty = std::stof(next());
        else if (argument == "--time-penalty") options.time_penalty = std::stof(next());
        else if (argument == "--learner-weakness-weight") options.learner_weakness_weight = std::stof(next());
        else if (argument == "--scripted-tiers") options.scripted_tiers = true;
        else if (argument == "--pool-anneal") options.pool_anneal = true;
        else if (argument == "--hold-frames") options.hold_frames = std::stoi(next());
        else if (argument == "--perception-delay") options.perception_delay = std::stoi(next());
        else if (argument == "--memory") options.memory = true;
        else if (argument == "--live-noise") options.live_noise = true;
        else if (argument == "--screen-observation") options.screen_view = true;
        else if (argument == "--noise-position") options.noise_position = std::stof(next());
        else if (argument == "--noise-flip") options.noise_flip = std::stof(next());
        else if (argument == "--latency-min") options.latency_min = std::stoi(next());
        else if (argument == "--latency-max") options.latency_max = std::stoi(next());
        else if (argument == "--shared-layers") options.shared_layers = std::stoi(next());
        else if (argument == "--head-layers") options.head_layers = std::stoi(next());
        else if (argument == "--damage-dealt-weight") options.damage_dealt_weight = std::stof(next());
        else if (argument == "--damage-taken-weight") options.damage_taken_weight = std::stof(next());
        else if (argument == "--distance-penalty") options.distance_penalty = std::stof(next());
        else if (argument == "--league-follow") options.league_follow.emplace_back(next());
        else if (argument == "--tier-promote") options.tier_promote = std::stof(next());
        else if (argument == "--tier-demote") options.tier_demote = std::stof(next());
        else if (argument == "--tier-min-updates") options.tier_min_updates = std::stoull(next());
        else if (argument == "--stop-at-update") options.stop_at_update = std::stoull(next());
        else if (argument == "--weakness-weight") options.weakness_weight = std::stof(next());
        else if (argument == "--scenario-share") options.scenario_share = std::stof(next());
        else if (argument == "--self-play-scenarios") options.self_play_scenarios = true;
        else if (argument == "--champion-interval-steps") options.champion_interval_steps = std::stoull(next());
        else if (argument == "--champion-rounds") options.champion_rounds = std::stoull(next());
        else if (argument == "--champion-promote") options.champion_promote = std::stof(next());
        else if (argument == "--champion-rollback") options.champion_rollback = std::stof(next());
        else if (argument == "--champion-rollback-after") options.champion_rollback_after = std::stoi(next());
        else if (argument == "--league-pool-size") options.league_pool_size = std::stoull(next());
        else if (argument == "--league-extra") options.league_extras.emplace_back(next());
        else if (argument == "--exploit-target") options.exploit_target = std::filesystem::path(next());
        else if (argument == "--promotion-tolerance") options.promotion_tolerance = std::stof(next());
        else if (argument == "--promotion-max-side-gap") options.promotion_max_side_gap = std::stof(next());
        else if (argument == "--promotion-min-score") options.promotion_min_score = std::stof(next());
        else if (argument == "--max-round-frames") options.max_round_frames = std::stoi(next());
        else if (argument == "--checkpoint-interval") options.checkpoint_interval = std::stoull(next());
        else if (argument == "--eval-interval") options.evaluation_interval = std::stoull(next());
        else if (argument == "--eval-episodes") options.evaluation_episodes = std::stoull(next());
        else if (argument == "--seed") options.seed = std::stoull(next());
        else if (argument == "--run-dir") options.run_directory = next();
        else if (argument == "--data-dir") options.data_directory = next();
        else if (argument == "--resume") options.resume = std::filesystem::path(next());
        else if (argument == "--fork") {
            options.resume = std::filesystem::path(next());
            options.fork = true;
        }
        else if (argument == "--smoke") options.smoke = true;
        else usage("unknown argument " + argument);
    }
    if (options.smoke) {
        options.champion_rounds = 64;
        options.environments = 256;
        options.horizon = 16;
        options.updates = std::min<std::size_t>(options.updates, 4);
        options.minibatch_size = 1024;
        options.epochs = 2;
        options.checkpoint_interval = 2;
        options.evaluation_interval = 2;
        options.evaluation_episodes = 64;
        options.hidden_size = 64;
    }
    if (options.anneal_updates == 0) options.anneal_updates = options.updates;
    if (options.environments == 0 || options.environments % 2 != 0) usage("--envs must be even and positive");
    if (options.horizon == 0 || options.updates == 0 || options.epochs <= 0 || options.minibatch_size == 0 ||
        options.minibatch_size > options.environments * options.horizon) {
        usage("invalid rollout shape (minibatch must fit in envs x horizon)");
    }
    if (options.evaluation_episodes == 0 || options.evaluation_episodes % 2 != 0) {
        usage("--eval-episodes must be even and positive");
    }
    if (options.learner_characters.empty()) usage("--learner-character needs at least one character");
    if (options.champion_rounds == 0 || options.champion_rounds % 2 != 0 || options.champion_rollback_after < 1 ||
        !(options.champion_rollback <= options.champion_promote) || !(options.champion_promote <= 1.0F) ||
        !(options.champion_rollback >= 0.0F)) {
        usage("champion rounds must be even and positive, and 0 <= rollback <= promote <= 1");
    }
    if (!(options.entropy_target >= 0.0F && options.entropy_target <= 6.0F) ||
        !(options.entropy_target_gain >= 0.0F && options.entropy_target_gain <= 10.0F) ||
        !(options.max_entropy_coefficient >= std::max(options.entropy_coefficient, options.final_entropy_coefficient))) {
        usage("entropy target must be in [0, 6] nats, gain in [0, 10], and the maximum coefficient at least the schedule");
    }
    if (!(options.scripted_floor >= 0.0F && options.scripted_floor <= 1.0F - options.self_play_share)) {
        usage("--scripted-floor must be in [0, 1 - self-play share]");
    }
    if (!(options.self_play_share >= 0.0F && options.self_play_share <= 1.0F) ||
        !(options.weakness_weight >= 0.0F && options.weakness_weight <= 1.0F)) {
        usage("--self-play-share and --weakness-weight must be in [0, 1]");
    }
    if (!(options.noise_position >= 0.0F && options.noise_position <= 2.0F) ||
        !(options.noise_flip >= 0.0F && options.noise_flip <= 0.5F) || options.latency_min < 0 ||
        options.latency_max < options.latency_min || options.latency_max > 30) {
        usage("noise: position in [0, 2] m, flip in [0, 0.5], 0 <= latency min <= max <= 30 frames");
    }
    if (options.screen_view && options.memory) usage("--screen-observation has no memory variant yet");
    return options;
}

std::filesystem::path find_generated(const Options& options) {
    if (!options.data_directory.empty()) return options.data_directory;
    for (auto directory = std::filesystem::current_path(); ; directory = directory.parent_path()) {
        if (std::filesystem::exists(directory / "data" / "generated" / "full_move_candidates.csv")) {
            return directory / "data" / "generated";
        }
        if (directory == directory.root_path()) break;
    }
#ifdef T8_V2_SOURCE_DIR
    return std::filesystem::path(T8_V2_SOURCE_DIR) / "data" / "generated";
#else
    throw std::runtime_error("cannot find data/generated; pass --data-dir");
#endif
}

// ---------------------------------------------------------------- state file

constexpr std::array<char, 8> kStateMagic = {'T', '8', 'F', 'M', 'V', '1', '\0', '\0'};
constexpr std::uint32_t kStateVersion = 8;

template <typename T>
void write_value(std::ostream& output, const T& value) {
    output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}
template <typename T>
T read_value(std::istream& input) {
    T value{};
    input.read(reinterpret_cast<char*>(&value), sizeof(T));
    if (!input) throw std::runtime_error("truncated full-move trainer state");
    return value;
}
template <typename T>
void write_vector(std::ostream& output, const std::vector<T>& values) {
    write_value(output, static_cast<std::uint64_t>(values.size()));
    output.write(reinterpret_cast<const char*>(values.data()), static_cast<std::streamsize>(sizeof(T) * values.size()));
}
template <typename T>
std::vector<T> read_vector(std::istream& input) {
    const auto count = read_value<std::uint64_t>(input);
    if (count > (1ULL << 32U)) throw std::runtime_error("corrupt full-move trainer state");
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(sizeof(T) * count));
    if (!input) throw std::runtime_error("truncated full-move trainer state");
    return values;
}
void write_string(std::ostream& output, const std::string& value) {
    write_value(output, static_cast<std::uint64_t>(value.size()));
    output.write(value.data(), static_cast<std::streamsize>(value.size()));
}
std::string read_string(std::istream& input) {
    const auto count = read_value<std::uint64_t>(input);
    if (count > 4096) throw std::runtime_error("corrupt full-move trainer state string");
    std::string value(count, '\0');
    input.read(value.data(), static_cast<std::streamsize>(count));
    if (!input) throw std::runtime_error("truncated full-move trainer state");
    return value;
}

// A resume continues from its checkpoint, so log rows written after it (a run stopped
// between checkpoints) are dropped instead of duplicated: metrics.jsonl lines whose
// "update" is newer, and CSV rows whose first column (the update) is newer.
void truncate_logs(const std::filesystem::path& run_directory, std::uint64_t update) {
    const auto row_update = [](const std::string& line, bool json) -> std::optional<std::uint64_t> {
        std::size_t begin = 0;
        if (json) {
            const auto marker = line.find("\"update\":");
            if (marker == std::string::npos) return std::nullopt;
            begin = marker + 9;
        }
        std::size_t end = begin;
        while (end < line.size() && std::isdigit(static_cast<unsigned char>(line[end]))) ++end;
        if (end == begin) return std::nullopt;
        return std::stoull(line.substr(begin, end - begin));
    };
    for (const auto& entry : std::filesystem::directory_iterator(run_directory)) {
        const auto& path = entry.path();
        const bool json = path.filename() == "metrics.jsonl";
        if (!entry.is_regular_file() || !(json || path.extension() == ".csv")) continue;
        std::ifstream input(path);
        std::vector<std::string> kept;
        bool dropped = false;
        for (std::string line; std::getline(input, line);) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const auto value = row_update(line, json);
            // Headers and other non-numbered lines stay; an unterminated last JSON line goes.
            const bool torn = json && (line.empty() || line.back() != '}');
            if (torn || (value && *value > update)) {
                dropped = true;
                continue;
            }
            kept.push_back(line);
        }
        input.close();
        if (!dropped) continue;
        auto temporary = path;
        temporary += ".tmp";
        {
            std::ofstream output(temporary, std::ios::trunc);
            for (const auto& line : kept) output << line << '\n';
        }
        std::filesystem::rename(temporary, path);
    }
}

// Every option that changes training results; resume requires them unchanged.
std::string join(const std::vector<std::string>& values) {
    std::string out;
    for (const auto& value : values) out += (out.empty() ? "" : ",") + value;
    return out;
}

std::string run_identity(const Options& options, const std::vector<std::string>& learners,
                         const std::vector<std::string>& opponents) {
    std::ostringstream identity;
    identity << std::setprecision(9) << "learners=" << join(learners) << ";opponents=";
    for (const auto& slug : opponents) identity << slug << ',';
    identity << ";provisional=" << options.allow_provisional << ";envs=" << options.environments
             << ";horizon=" << options.horizon << ";updates_anneal=" << options.anneal_updates
             << ";epochs=" << options.epochs << ";minibatch=" << options.minibatch_size
             << ";hidden=" << options.hidden_size
             << (options.matmul_precision == "fp32" ? "" : ";matmul=" + options.matmul_precision) << ";lr=" << options.learning_rate << ',' << options.final_learning_rate
             << ";gamma=" << options.gamma << ";lambda=" << options.gae_lambda << ";entropy=" << options.entropy_coefficient
             << ',' << options.final_entropy_coefficient << ";entropy_floor=" << options.entropy_target << ','
             << options.entropy_target_gain << ',' << options.max_entropy_coefficient << ";kl=" << options.target_kl
             << ";reward=" << (options.sparse_reward ? "sparse" : "shaped") << ";return_norm=" << options.return_normalization
             << ";obs_norm=" << options.observation_normalization << ";self_play=" << options.self_play_share << ','
             << options.self_play_start
             << (options.scripted_floor > 0.0F ? ";scripted_floor=" + std::to_string(options.scripted_floor) : "")
             << (options.draw_penalty > 0.0F ? ";draw_penalty=" + std::to_string(options.draw_penalty) : "")
             << (options.throw_break_penalty > 0.0F ? ";throw_break_penalty=" + std::to_string(options.throw_break_penalty) : "")
             << (options.time_penalty > 0.0F ? ";time_penalty=" + std::to_string(options.time_penalty) : "")
             << (options.learner_weakness_weight > 0.0F
                     ? ";learner_weakness=" + std::to_string(options.learner_weakness_weight) : "")
             << (options.pool_anneal ? ";pool_anneal" : "")
             << (options.hold_frames != 4 ? ";hold=" + std::to_string(options.hold_frames) : "")
             << (options.perception_delay > 0 ? ";perception=" + std::to_string(options.perception_delay) : "")
             << (options.memory ? ";memory" : "") << (options.screen_view ? ";screen" : "")
             << (options.live_noise ? ";noise=" + std::to_string(options.noise_position) + "," +
                                          std::to_string(options.noise_flip) + "," + std::to_string(options.latency_min) +
                                          "," + std::to_string(options.latency_max)
                                    : std::string())
             << (options.shared_layers != 2 || options.head_layers != 0
                     ? ";layers=" + std::to_string(options.shared_layers) + ',' + std::to_string(options.head_layers) : "")
             << (options.damage_dealt_weight != 1.0F || options.damage_taken_weight != 1.0F
                     ? ";hp_ratio=" + std::to_string(options.damage_dealt_weight) + ',' +
                           std::to_string(options.damage_taken_weight) : "")
             << (options.distance_penalty > 0.0F ? ";distance_penalty=" + std::to_string(options.distance_penalty) : "")
             << (options.scripted_tiers ? ";tiers=" + std::to_string(options.tier_promote) + ',' +
                                              std::to_string(options.tier_demote) + ',' +
                                              std::to_string(options.tier_min_updates) : "")
             << ";weakness=" << options.weakness_weight
             << ";round_frames=" << options.max_round_frames << ";seed=" << options.seed
             << ";scenarios=" << options.scenario_share << (options.self_play_scenarios ? ";sp_scenarios" : "")
             << ";champion=" << options.champion_interval_steps << ','
             << options.champion_rounds << ',' << options.champion_promote << ',' << options.champion_rollback << ','
             << options.champion_rollback_after << ";league=" << options.league_pool_size << ',' << options.promotion_tolerance << ','
             << options.promotion_max_side_gap << ',' << options.promotion_min_score << ";extras=";
    for (const auto& extra : options.league_extras) identity << extra.generic_string() << ',';
    if (options.exploit_target) identity << ";exploit=" << options.exploit_target->generic_string();
    return identity.str();
}

struct PoolEntry {
    std::string checkpoint;
    std::uint64_t update = 0;
    double rating = 1000.0;
    bool promoted = false;
    bool external = false;  // a fixed extra policy, never trimmed
};

struct TrainerState {
    std::uint64_t update = 0;
    std::uint64_t environment_steps = 0;
    std::uint64_t engine_frames = 0;
    double elapsed_seconds = 0.0;
    bool self_play_active = false;
    std::vector<double> win_rate_estimate;  // per opponent, for weakness sampling
    std::vector<double> learner_score_estimate;  // per learner character, self-play score
    int scripted_tier = 0;                // tier ladder position (--scripted-tiers)
    double tier_score = 0.5;              // score against the current tier (moving average)
    std::uint64_t updates_at_tier = 0;    // scripted updates since the last tier change
    ReturnNormalizerState returns;
    FullMoveEnvSnapshot environments;
    std::vector<PoolEntry> pool;
    double learner_rating = 1000.0;
    double best_scripted_win_rate = 0.0;
    float entropy_boost = 1.0F;  // entropy floor controller
    float scripted_share = -1.0F;  // adaptive scripted share (-1: not started)
    // Champion
    std::string champion;               // checkpoint path, empty before the first match
    std::uint64_t champion_update = 0;
    double champion_share = 0.0;        // share it won when crowned
    int champion_loss_streak = 0;
    std::uint64_t next_champion_step = 0;
    std::uint64_t champion_matches = 0;
    std::uint64_t rollbacks = 0;
};

void save_state(const std::filesystem::path& path, const std::string& catalog_sha256, const std::string& roster_version,
                const Options& options, const std::string& learners, const std::string& identity,
                const TrainerState& state) {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream output(temporary, std::ios::binary | std::ios::trunc);
        if (!output) throw std::runtime_error("cannot write " + temporary.string());
        output.write(kStateMagic.data(), kStateMagic.size());
        write_value(output, kStateVersion);
        write_value(output, state.update);
        // Identity first, so the live runtime can verify a checkpoint's character cheaply.
        write_string(output, catalog_sha256);
        write_string(output, roster_version);
        write_string(output, learners);  // comma-separated learner characters
        write_value(output, static_cast<std::uint8_t>(options.allow_provisional));
        write_string(output, identity);
        write_value(output, state.environment_steps);
        write_value(output, state.engine_frames);
        write_value(output, state.elapsed_seconds);
        write_value(output, static_cast<std::uint8_t>(state.self_play_active));
        write_vector(output, state.win_rate_estimate);
        write_value(output, state.returns.count);
        write_value(output, state.returns.mean);
        write_value(output, state.returns.variance);
        write_vector(output, state.returns.discounted_returns);
        const auto& env = state.environments;
        write_vector(output, env.states);
        write_vector(output, env.p1_characters);
        write_vector(output, env.p2_characters);
        write_vector(output, env.learner_is_p1);
        write_vector(output, env.aggression);
        write_vector(output, env.pending_opponents);
        write_vector(output, env.pending_aggression);
        write_vector(output, env.terminated);
        write_vector(output, env.scenarios);
        write_vector(output, env.pending_scenarios);
        write_vector(output, env.pending_learners);
        write_value(output, static_cast<std::uint64_t>(state.pool.size()));
        for (const auto& entry : state.pool) {
            write_string(output, entry.checkpoint);
            write_value(output, entry.update);
            write_value(output, entry.rating);
            write_value(output, static_cast<std::uint8_t>(entry.promoted));
            write_value(output, static_cast<std::uint8_t>(entry.external));
        }
        write_value(output, state.learner_rating);
        write_value(output, state.best_scripted_win_rate);
        write_value(output, state.entropy_boost);
        write_value(output, state.scripted_share);
        write_vector(output, state.learner_score_estimate);
        write_value(output, state.scripted_tier);
        write_value(output, state.tier_score);
        write_value(output, state.updates_at_tier);
        write_string(output, state.champion);
        write_value(output, state.champion_update);
        write_value(output, state.champion_share);
        write_value(output, state.champion_loss_streak);
        write_value(output, state.next_champion_step);
        write_value(output, state.champion_matches);
        write_value(output, state.rollbacks);
        if (!output) throw std::runtime_error("failed writing " + temporary.string());
    }
    std::filesystem::rename(temporary, path);
}

TrainerState load_state(const std::filesystem::path& path, const std::string& catalog_sha256,
                        const std::string& learners, const std::string& identity, bool fork = false) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot read " + path.string());
    std::array<char, 8> magic{};
    input.read(magic.data(), magic.size());
    const auto version = read_value<std::uint32_t>(input);
    if (magic != kStateMagic || version < 5 || version > kStateVersion) {
        throw std::runtime_error("not a full-move trainer state: " + path.string());
    }
    TrainerState state;
    state.update = read_value<std::uint64_t>(input);
    if (read_string(input) != catalog_sha256) throw std::runtime_error("resume: the move catalog changed");
    read_string(input);  // roster version (checked through the catalog hash)
    if (read_string(input) != learners) throw std::runtime_error("resume: different learner characters");
    read_value<std::uint8_t>(input);
    if (read_string(input) != identity && !fork) {
        throw std::runtime_error("resume: run-defining options differ from the original run (--fork branches off)");
    }
    state.environment_steps = read_value<std::uint64_t>(input);
    state.engine_frames = read_value<std::uint64_t>(input);
    state.elapsed_seconds = read_value<double>(input);
    state.self_play_active = read_value<std::uint8_t>(input) != 0;
    state.win_rate_estimate = read_vector<double>(input);
    state.returns.count = read_value<std::uint64_t>(input);
    state.returns.mean = read_value<double>(input);
    state.returns.variance = read_value<double>(input);
    state.returns.discounted_returns = read_vector<float>(input);
    auto& env = state.environments;
    env.states = read_vector<full_core::State>(input);
    env.p1_characters = read_vector<int>(input);
    env.p2_characters = read_vector<int>(input);
    env.learner_is_p1 = read_vector<std::uint8_t>(input);
    env.aggression = read_vector<float>(input);
    env.pending_opponents = read_vector<int>(input);
    env.pending_aggression = read_vector<float>(input);
    env.terminated = read_vector<std::uint8_t>(input);
    env.scenarios = read_vector<int>(input);
    env.pending_scenarios = read_vector<int>(input);
    env.pending_learners = read_vector<int>(input);
    const auto pool_size = read_value<std::uint64_t>(input);
    if (pool_size > 4096) throw std::runtime_error("corrupt league pool");
    for (std::uint64_t index = 0; index < pool_size; ++index) {
        PoolEntry entry;
        entry.checkpoint = read_string(input);
        entry.update = read_value<std::uint64_t>(input);
        entry.rating = read_value<double>(input);
        entry.promoted = read_value<std::uint8_t>(input) != 0;
        entry.external = read_value<std::uint8_t>(input) != 0;
        state.pool.push_back(entry);
    }
    state.learner_rating = read_value<double>(input);
    state.best_scripted_win_rate = read_value<double>(input);
    state.entropy_boost = read_value<float>(input);
    if (version >= 6) state.scripted_share = read_value<float>(input);
    if (version >= 7) state.learner_score_estimate = read_vector<double>(input);
    if (version >= 8) {
        state.scripted_tier = read_value<int>(input);
        state.tier_score = read_value<double>(input);
        state.updates_at_tier = read_value<std::uint64_t>(input);
    }
    state.champion = read_string(input);
    state.champion_update = read_value<std::uint64_t>(input);
    state.champion_share = read_value<double>(input);
    state.champion_loss_streak = read_value<int>(input);
    state.next_champion_step = read_value<std::uint64_t>(input);
    state.champion_matches = read_value<std::uint64_t>(input);
    state.rollbacks = read_value<std::uint64_t>(input);
    return state;
}

// ---------------------------------------------------------------- scheduling

std::uint64_t mix(std::uint64_t value) { return full_core::mix64(value); }

// The starting scenario of an environment pair: --scenario-share of pairs start in one,
// rotating through every scenario so each gets an equal share.
int scenario_for(const Options& options, std::uint64_t update, std::uint64_t pair) {
    const double draw = static_cast<double>(mix(options.seed ^ mix(update * 2'000'029ULL + pair)) % 10'000ULL);
    return draw < options.scenario_share * 10'000.0
        ? 1 + static_cast<int>((pair + update) % (full_core::kScenarioCount - 1)) : full_core::kScenarioNeutral;
}

// Opponents for the next rollout: a blend of uniform coverage and the
// opponents the learner currently beats least, with every aggression level and
// both sides equally represented (side = environment parity, aggression rotates).
// The learner's character for each environment pair: every learner character
// equally often, rotating each update.
int learner_for_pair(const std::vector<int>& learner_roster, std::uint64_t pair, std::uint64_t update) {
    return learner_roster[static_cast<std::size_t>((pair + update * 7ULL) % learner_roster.size())];
}

// Learner characters per environment pair, weighted: a (1 - weight) share split evenly,
// the rest in proportion to each character's self-play losses (1 - score). Exact counts
// by largest remainder, then rotated each update; deterministic, so resume is exact.
std::vector<int> weighted_learners(const std::vector<int>& learner_roster, const std::vector<double>& score,
                                   double weight, std::size_t pairs, std::uint64_t update) {
    const std::size_t count = learner_roster.size();
    std::vector<double> share(count, (1.0 - weight) / static_cast<double>(count));
    double weakness_total = 0.0;
    for (std::size_t index = 0; index < count; ++index) weakness_total += std::max(1e-3, 1.0 - score[index]);
    for (std::size_t index = 0; index < count; ++index) {
        share[index] += weight * std::max(1e-3, 1.0 - score[index]) / weakness_total;
    }
    std::vector<std::size_t> pairs_of(count);
    std::vector<std::pair<double, std::size_t>> remainders;
    std::size_t assigned = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const double exact = share[index] * static_cast<double>(pairs);
        pairs_of[index] = static_cast<std::size_t>(exact);
        assigned += pairs_of[index];
        remainders.emplace_back(exact - static_cast<double>(pairs_of[index]), index);
    }
    std::stable_sort(remainders.begin(), remainders.end(), [](const auto& a, const auto& b) { return a.first > b.first; });
    for (std::size_t index = 0; assigned < pairs; ++index, ++assigned) ++pairs_of[remainders[index % count].second];
    std::vector<int> learners;
    learners.reserve(pairs);
    for (std::size_t index = 0; index < count; ++index) {
        learners.insert(learners.end(), pairs_of[index], learner_roster[index]);
    }
    std::rotate(learners.begin(), learners.begin() + static_cast<std::ptrdiff_t>((update * 7919ULL) % pairs),
                learners.end());
    return learners;
}

void schedule_opponents(const Options& options, const std::vector<int>& opponent_roster,
                        const std::vector<double>& win_rate, std::uint64_t update, std::size_t environments,
                        std::vector<int>& characters, std::vector<float>& aggression, std::vector<int>& scenarios) {
    // Each part is a distribution, so the option sets their exact mix.
    std::vector<double> weights(opponent_roster.size());
    double weakness_total = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) weakness_total += 1.0 - win_rate[index] + 0.05;
    double total = 0.0;
    for (std::size_t index = 0; index < weights.size(); ++index) {
        const double weakness = (1.0 - win_rate[index] + 0.05) / weakness_total;
        weights[index] = (1.0 - options.weakness_weight) / static_cast<double>(weights.size()) +
            options.weakness_weight * weakness;
        total += weights[index];
    }
    characters.resize(environments);
    aggression.resize(environments);
    scenarios.resize(environments);
    for (std::size_t env = 0; env < environments; ++env) {
        // Pairs of environments (P1 and P2 learner) share an opponent and aggression.
        const std::uint64_t pair = env / 2;
        const double draw = static_cast<double>(mix(options.seed ^ mix(update * 1'000'003ULL + pair)) >> 11U) *
            (1.0 / 9007199254740992.0) * total;
        double running = 0.0;
        std::size_t chosen = weights.size() - 1;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            running += weights[index];
            if (draw < running) {
                chosen = index;
                break;
            }
        }
        characters[env] = opponent_roster[chosen];
        aggression[env] = static_cast<float>((pair + update) % GpuFullMoveSimulator::kAggressionBuckets) / 4.0F;
        scenarios[env] = scenario_for(options, update, pair);
    }
}

// The scripted bots of each tier (ids per full_core::scripted_profile).
const std::vector<std::vector<int>>& bots_by_tier() {
    static const std::vector<std::vector<int>> tiers = [] {
        std::vector<std::vector<int>> out(full_core::kScriptedTierCount);
        for (int bot = 0; bot < full_core::kScriptedProfileCount; ++bot) {
            out[static_cast<std::size_t>(full_core::scripted_profile(bot).tier)].push_back(bot);
        }
        return out;
    }();
    return tiers;
}

const char* tier_name(int tier) {
    static const char* names[] = {"dummy", "beginner", "intermediate", "advanced", "expert", "master", "pro"};
    return names[std::clamp(tier, 0, full_core::kScriptedTierCount - 1)];
}

// Scripted opponents from the ladder: 60% the current tier, 25% a lower tier (the current
// one at tier 0), 15% the next (the current one at the top). Pairs share a bot.
void assign_tier_bots(const Options& options, int tier, std::uint64_t update, std::vector<float>& behaviour) {
    const auto& tiers = bots_by_tier();
    const int top = full_core::kScriptedTierCount - 1;
    for (std::size_t env = 0; env < behaviour.size(); ++env) {
        const std::uint64_t pair = env / 2;
        const std::uint64_t bits = mix(options.seed ^ mix(update * 3'000'017ULL + pair));
        const int roll = static_cast<int>(bits % 100ULL);
        int chosen_tier = tier;
        if (roll >= 60 && roll < 85 && tier > 0) chosen_tier = static_cast<int>((bits >> 8U) % static_cast<std::uint64_t>(tier));
        if (roll >= 85 && tier < top) chosen_tier = tier + 1;
        const auto& bots = tiers[static_cast<std::size_t>(chosen_tier)];
        const int bot = bots[static_cast<std::size_t>((bits >> 24U) % bots.size())];
        behaviour[env] = full_core::encode_bot(bot);
    }
}

// The newest update_N.t8ppo with its .t8fmstate in a directory, if any.
std::optional<std::filesystem::path> newest_champion(const std::filesystem::path& directory) {
    std::optional<std::filesystem::path> best;
    std::uint64_t best_update = 0;
    std::error_code error;
    if (!std::filesystem::is_directory(directory, error)) return best;
    for (const auto& entry : std::filesystem::directory_iterator(directory, error)) {
        const auto name = entry.path().filename().string();
        if (entry.path().extension() != ".t8ppo" || name.rfind("update_", 0) != 0) continue;
        auto state = entry.path();
        state.replace_extension(".t8fmstate");
        if (!std::filesystem::exists(state)) continue;
        try {
            const auto update = std::stoull(name.substr(7, name.size() - 7 - 6));
            if (!best || update > best_update) {
                best = entry.path();
                best_update = update;
            }
        } catch (const std::exception&) {
        }
    }
    return best;
}

// ---------------------------------------------------------------- league

// Self-play opponent: 40% the latest checkpoint, 40% by rating closeness to
// the learner, 20% uniform over the pool (diversity).
std::size_t choose_pool_opponent(const std::vector<PoolEntry>& pool, double learner_rating, std::uint64_t seed,
                                 std::uint64_t update, double latest_share = 0.4) {
    std::size_t latest = 0;
    for (std::size_t index = 0; index < pool.size(); ++index) {
        if (!pool[index].external && pool[index].update >= pool[latest].update) latest = index;
    }
    std::vector<double> closeness(pool.size());
    double closeness_total = 0.0;
    for (std::size_t index = 0; index < pool.size(); ++index) {
        closeness[index] = std::exp(-std::abs(pool[index].rating - learner_rating) / 200.0);
        closeness_total += closeness[index];
    }
    std::vector<double> weights(pool.size());
    for (std::size_t index = 0; index < pool.size(); ++index) {
        // The rest keeps the default 2:1 split between rating closeness and uniform.
        const double rest = 1.0 - latest_share;
        weights[index] = latest_share * (index == latest ? 1.0 : 0.0) +
            rest * (2.0 / 3.0) * closeness[index] / closeness_total + rest / 3.0 / static_cast<double>(pool.size());
    }
    const double draw = static_cast<double>(full_core::mix64(seed ^ full_core::mix64(update * 104'729ULL)) >> 11U) *
        (1.0 / 9007199254740992.0);
    double running = 0.0;
    for (std::size_t index = 0; index < pool.size(); ++index) {
        running += weights[index];
        if (draw < running) return index;
    }
    return pool.size() - 1;
}

// Keeps promoted and external entries, the newest checkpoint, and otherwise
// the most spread-out ratings: the entry closest in rating to a neighbor goes.
void trim_pool(std::vector<PoolEntry>& pool, std::size_t keep) {
    while (true) {
        std::vector<std::size_t> candidates;
        std::size_t newest = 0;
        for (std::size_t index = 0; index < pool.size(); ++index) {
            if (pool[index].update > pool[newest].update) newest = index;
        }
        for (std::size_t index = 0; index < pool.size(); ++index) {
            if (!pool[index].promoted && !pool[index].external) candidates.push_back(index);
        }
        if (candidates.size() <= keep) return;
        std::size_t victim = candidates.front();
        double smallest = 1e300;
        for (const std::size_t index : candidates) {
            if (index == newest) continue;
            double gap = 1e300;
            for (std::size_t other = 0; other < pool.size(); ++other) {
                if (other != index) gap = std::min(gap, std::abs(pool[other].rating - pool[index].rating));
            }
            if (gap < smallest) {
                smallest = gap;
                victim = index;
            }
        }
        pool.erase(pool.begin() + static_cast<std::ptrdiff_t>(victim));
    }
}

// ---------------------------------------------------------------- evaluation

struct Evaluation {
    std::size_t episodes = 0;
    double win_rate = 0.0;
    double p1_win_rate = 0.0;
    double p2_win_rate = 0.0;
    double draw_rate = 0.0;
    double worst_character_win_rate = 1.0;
    std::string worst_character;
    std::map<std::string, std::array<std::size_t, 4>> characters;  // by opponent: wins, losses, draws, episodes
    std::map<std::string, std::array<std::size_t, 4>> learners;    // by the learner's own character
    std::array<std::array<std::size_t, 4>, 5> styles{};            // per aggression bucket
};

Evaluation evaluate(GpuActorCritic& policy, GpuFullMoveSimulator& simulator, const std::vector<int>& learner_roster,
                    const std::vector<int>& opponent_roster, bool deterministic, std::uint64_t seed) {
    const std::size_t count = simulator.environment_count();
    std::vector<int> opponents(count), learners(count), p1(count), p2(count);
    std::vector<float> aggression(count);
    std::vector<std::uint8_t> sides(count);
    for (std::size_t env = 0; env < count; ++env) {
        const std::size_t pair = env / 2;
        learners[env] = learner_roster[pair % learner_roster.size()];
        opponents[env] = opponent_roster[(pair / learner_roster.size() + pair) % opponent_roster.size()];
        aggression[env] = static_cast<float>((pair / opponent_roster.size()) % 5) / 4.0F;
        sides[env] = env % 2 == 0;
        p1[env] = sides[env] ? learners[env] : opponents[env];
        p2[env] = sides[env] ? opponents[env] : learners[env];
    }
    simulator.set_matchups(p1, p2);
    simulator.set_learner_sides(sides);
    simulator.set_pending_opponents(opponents, aggression);
    simulator.set_pending_learners(learners);
    simulator.reset_all(seed, 0);
    std::vector<bool> finished(count, false);
    std::vector<std::int8_t> result(count, 0);
    std::size_t remaining = count;
    for (std::uint64_t step = 1; step <= 20'000 && remaining > 0; ++step) {
        const auto output = policy.forward(simulator.learner_observations(), simulator.learner_masks(), count,
                                           seed + 17, step, deterministic, nullptr, simulator.learner_feature_tables());
        simulator.step(output.actions, nullptr, seed, step);
        const auto terminated = simulator.download_terminated();
        const auto outcomes = simulator.download_outcomes();
        for (std::size_t env = 0; env < count; ++env) {
            if (!finished[env] && terminated[env]) {
                finished[env] = true;
                result[env] = outcomes[env];
                --remaining;
            }
        }
        // Finished environments keep playing new rounds; only their first counts.
        simulator.reset_done(seed, step);
    }
    Evaluation evaluation;
    std::array<std::size_t, 2> side_wins{}, side_episodes{};
    std::size_t wins = 0, draws = 0;
    const auto& roster = simulator.roster();
    for (std::size_t env = 0; env < count; ++env) {
        if (!finished[env]) continue;  // counted as neither; reported through `episodes`
        ++evaluation.episodes;
        const int side = sides[env] ? 0 : 1;
        ++side_episodes[static_cast<std::size_t>(side)];
        auto& row = evaluation.characters[roster[static_cast<std::size_t>(opponents[env])].slug];
        auto& learner_row = evaluation.learners[roster[static_cast<std::size_t>(learners[env])].slug];
        auto& style = evaluation.styles[static_cast<std::size_t>(std::lround(aggression[env] * 4.0F))];
        const std::size_t column = result[env] > 0 ? 0 : (result[env] < 0 ? 1 : 2);
        ++row[column];
        ++row[3];
        ++learner_row[column];
        ++learner_row[3];
        ++style[column];
        ++style[3];
        wins += result[env] > 0;
        draws += result[env] == 0;
        side_wins[static_cast<std::size_t>(side)] += result[env] > 0;
    }
    const auto rate = [](std::size_t part, std::size_t whole) {
        return whole == 0 ? 0.0 : static_cast<double>(part) / static_cast<double>(whole);
    };
    evaluation.win_rate = rate(wins, evaluation.episodes);
    evaluation.draw_rate = rate(draws, evaluation.episodes);
    evaluation.p1_win_rate = rate(side_wins[0], side_episodes[0]);
    evaluation.p2_win_rate = rate(side_wins[1], side_episodes[1]);
    for (const auto& [slug, row] : evaluation.characters) {
        const double character_rate = rate(row[0], row[3]);
        if (character_rate <= evaluation.worst_character_win_rate) {
            evaluation.worst_character_win_rate = character_rate;
            evaluation.worst_character = slug;
        }
    }
    return evaluation;
}

// Learner against a frozen policy, both playing learner characters (every
// pairing in turn): one round per environment, half on each side, greedy, from
// a fixed seed. {wins, losses, draws}.
std::array<std::size_t, 3> head_to_head(GpuActorCritic& learner, GpuActorCritic& opponent,
                                        GpuFullMoveSimulator& simulator, const std::vector<int>& learner_roster,
                                        std::uint64_t seed, bool deterministic = true) {
    const std::size_t count = simulator.environment_count();
    const std::size_t characters = learner_roster.size();
    std::vector<int> own(count), other(count), p1(count), p2(count);
    std::vector<std::uint8_t> sides(count);
    for (std::size_t env = 0; env < count; ++env) {
        const std::size_t pair = env / 2;
        own[env] = learner_roster[pair % characters];
        other[env] = learner_roster[(pair / characters + pair) % characters];
        sides[env] = env % 2 == 0;
        p1[env] = sides[env] ? own[env] : other[env];
        p2[env] = sides[env] ? other[env] : own[env];
    }
    simulator.set_matchups(p1, p2);
    simulator.set_learner_sides(sides);
    simulator.set_pending_opponents(other);
    simulator.set_pending_learners(own);
    simulator.reset_all(seed, 0);
    std::vector<bool> finished(count, false);
    std::array<std::size_t, 3> tally{};
    std::size_t remaining = count;
    for (std::uint64_t step = 1; step <= 40'000 && remaining > 0; ++step) {
        const auto mine = learner.forward(simulator.learner_observations(), simulator.learner_masks(), count, seed, step,
                                          deterministic, nullptr, simulator.learner_feature_tables());
        const auto theirs = opponent.forward(simulator.opponent_observations(), simulator.opponent_masks(), count,
                                             seed + 1, step, deterministic, nullptr, simulator.opponent_feature_tables());
        simulator.step(mine.actions, theirs.actions, seed, step);
        const auto terminated = simulator.download_terminated();
        const auto outcomes = simulator.download_outcomes();
        for (std::size_t env = 0; env < count; ++env) {
            if (finished[env] || !terminated[env]) continue;
            finished[env] = true;
            --remaining;
            ++tally[outcomes[env] > 0 ? 0 : (outcomes[env] < 0 ? 1 : 2)];
        }
        simulator.reset_done(seed, step);
    }
    return tally;
}

void append_line(const std::filesystem::path& path, const std::string& header, const std::string& line) {
    const bool fresh = !std::filesystem::exists(path);
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("cannot append to " + path.string());
    if (fresh && !header.empty()) output << header << '\n';
    output << line << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse(argc, argv);
        const auto generated = find_generated(options);
        const auto catalog = load_full_move_catalog_csv(generated / "full_move_catalog.csv");

        // Roster: the learner characters first, then each further opponent once.
        std::vector<std::string> learner_slugs;
        for (const auto& value : options.learner_characters) {
            if (value == "all") {
                for (const auto slug : kRosterCharacterSlugs) learner_slugs.emplace_back(slug);
            } else {
                learner_slugs.push_back(value);
            }
        }
        std::vector<std::string> opponent_slugs;
        for (const auto& value : options.opponents) {
            if (value == "roster") {
                for (const auto slug : kRosterCharacterSlugs) opponent_slugs.emplace_back(slug);
            } else {
                opponent_slugs.push_back(value);
            }
        }
        FullMoveRosterOptions roster_options{};
        roster_options.allow_provisional = options.allow_provisional;
        std::vector<FullMoveCharacter> roster;
        // Without stand-in geometry a fighter takes part only once its whole moveset binds
        // (measured, Practice-validated, consistent data): its character gate.
        const auto gate_passed = [&](const FullMoveCharacter& character) {
            return options.allow_provisional ? character.bound_moves > 0
                : character.bound_moves == character.candidate_count() - kFullMoveUniversalCount;
        };
        std::vector<int> learner_roster;
        std::vector<std::string> used_learners;
        std::string gate_failures;
        for (const auto& slug : learner_slugs) {
            if (std::find(used_learners.begin(), used_learners.end(), slug) != used_learners.end()) continue;
            auto character = load_full_move_character(generated, slug, roster_options);
            if (!gate_passed(character)) {
                gate_failures += (gate_failures.empty() ? "" : "; ") + slug + " " + std::to_string(character.bound_moves) +
                    "/" + std::to_string(character.candidate_count() - kFullMoveUniversalCount);
                continue;
            }
            learner_roster.push_back(static_cast<int>(roster.size()));
            used_learners.push_back(slug);
            roster.push_back(std::move(character));
        }
        if (learner_roster.empty()) {
            throw std::runtime_error(
                "no learner character has passed its character gate (moves bound with measured, Practice-validated "
                "data: " + gate_failures + "). --allow-provisional trains with stand-in geometry (development only; "
                "results are not game-accurate).");
        }
        if (!gate_failures.empty()) std::cerr << "skipping learners without a passed gate: " << gate_failures << '\n';
        std::vector<int> opponent_roster;
        std::vector<std::string> used_opponents;
        for (const auto& slug : opponent_slugs) {
            int index = -1;
            for (std::size_t existing = 0; existing < roster.size(); ++existing) {
                if (roster[existing].slug == slug) index = static_cast<int>(existing);
            }
            if (index < 0) {
                auto character = load_full_move_character(generated, slug, roster_options);
                if (!gate_passed(character)) {
                    std::cerr << "skipping opponent " << slug << ": "
                              << (character.bound_moves == 0 ? "no move binds" : "character gate not passed") << '\n';
                    continue;
                }
                index = static_cast<int>(roster.size());
                roster.push_back(std::move(character));
            }
            if (std::find(opponent_roster.begin(), opponent_roster.end(), index) == opponent_roster.end()) {
                opponent_roster.push_back(index);
                used_opponents.push_back(slug);
            }
        }
        if (opponent_roster.empty()) throw std::runtime_error("no usable opponent");
        const std::string learners_text = join(used_learners);
        const std::string identity = run_identity(options, used_learners, used_opponents);

        full_core::EnvConfig env_config{};
        env_config.max_round_frames = options.max_round_frames;
        env_config.draw_penalty = options.draw_penalty;
        env_config.throw_break_penalty = options.throw_break_penalty;
        env_config.time_penalty_per_second = options.time_penalty;
        env_config.hold_frames = options.hold_frames;
        env_config.perception_delay = options.perception_delay;
        env_config.memory = options.memory;
        env_config.screen_view = options.screen_view;
        env_config.live_noise = options.live_noise;
        env_config.noise_position_max = options.noise_position;
        env_config.noise_flip_max = options.noise_flip;
        env_config.latency_min = options.latency_min;
        env_config.latency_max = options.latency_max;
        env_config.damage_dealt_weight = options.damage_dealt_weight;
        env_config.damage_taken_weight = options.damage_taken_weight;
        env_config.distance_penalty_per_second = options.distance_penalty;
        GpuFullMoveSimulator simulator(roster, options.environments, env_config);
        GpuFullMoveSimulator evaluator(roster, options.evaluation_episodes, env_config);
        std::optional<GpuFullMoveSimulator> champion_arena;
        if (options.champion_interval_steps > 0) champion_arena.emplace(roster, options.champion_rounds, env_config);

        ActorCriticConfig actor{};
        actor.observation_size = simulator.observation_size();
        actor.action_count = kFullMoveMaxCandidates;
        actor.hidden_size = options.hidden_size;
        actor.matmul_precision = options.matmul_precision;
        actor.shared_layers = options.shared_layers;
        actor.head_layers = options.head_layers;
        actor.action_feature_size = kFullMoveFeatureSize;
        actor.universal_action_count = kFullMoveUniversalCount;
        actor.catalog_sha256 = catalog.catalog_sha256;
        actor.roster_version = catalog.roster_version;
        actor.observation_contract =
            full_core::full_move_observation_contract(options.memory, options.live_noise, options.screen_view);
        actor.action_contract = "full-move-v1";
        actor.observation_normalization = options.observation_normalization;
        const std::size_t capacity = std::max({options.environments, options.minibatch_size, options.evaluation_episodes,
                                               options.champion_rounds});
        GpuActorCritic learner(capacity, actor, options.seed);
        // One candidate table per roster character (table index = roster index).
        const std::size_t table_size = static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize;
        std::vector<float> features(roster.size() * table_size, 0.0F);
        for (std::size_t index = 0; index < roster.size(); ++index) {
            std::copy(roster[index].candidate_features.begin(), roster[index].candidate_features.end(),
                      features.begin() + static_cast<std::ptrdiff_t>(index * table_size));
        }
        const int tables = static_cast<int>(roster.size());
        learner.set_action_feature_tables(features, tables);
        GpuRolloutBuffer rollout(options.environments, options.horizon, actor);

        std::filesystem::create_directories(options.run_directory / "checkpoints");
        const auto metrics_path = options.run_directory / "metrics.jsonl";
        TrainerState state;
        state.win_rate_estimate.assign(opponent_roster.size(), 0.5);
        state.learner_score_estimate.assign(learner_roster.size(), 0.5);
        std::vector<std::uint8_t> sides(options.environments);
        std::vector<int> p1(options.environments), p2(options.environments);
        for (std::size_t env = 0; env < options.environments; ++env) sides[env] = env % 2 == 0;
        simulator.set_learner_sides(sides);

        if (options.resume) {
            learner.load_checkpoint(*options.resume, true);
            learner.set_action_feature_tables(features, tables);
            auto state_path = *options.resume;
            state_path.replace_extension(".t8fmstate");
            state = load_state(state_path, catalog.catalog_sha256, learners_text, identity, options.fork);
            simulator.restore(state.environments);
            rollout.restore_return_normalizer_state(state.returns);
            truncate_logs(options.run_directory, state.update);
            std::cout << (options.fork ? "forked at update " : "resumed at update ") << state.update << '\n';
        } else {
            if (std::filesystem::exists(metrics_path)) {
                throw std::runtime_error("run directory already has metrics; pass --resume or choose a new --run-dir");
            }
            std::vector<int> opponents;
            std::vector<float> aggression;
            std::vector<int> scenarios;
            schedule_opponents(options, opponent_roster, state.win_rate_estimate, 0, options.environments, opponents,
                               aggression, scenarios);
            if (options.scripted_tiers) assign_tier_bots(options, state.scripted_tier, 0, aggression);
            std::vector<int> learners(options.environments);
            for (std::size_t env = 0; env < options.environments; ++env) {
                learners[env] = learner_for_pair(learner_roster, env / 2, 0);
                p1[env] = sides[env] ? learners[env] : opponents[env];
                p2[env] = sides[env] ? opponents[env] : learners[env];
            }
            simulator.set_matchups(p1, p2);
            simulator.set_pending_opponents(opponents, aggression, scenarios);
            simulator.set_pending_learners(learners);
            simulator.reset_all(options.seed, 0);
        }
        if (!options.resume || options.fork) {
            std::ofstream(options.run_directory / "run.json")
                << "{\"trainer\":\"t8_v2_full_move_train\",\"learner_characters\":\"" << learners_text
                << "\",\"provisional_geometry\":" << (options.allow_provisional ? "true" : "false")
                << ",\"catalog_sha256\":\"" << catalog.catalog_sha256
                << "\",\"observation_contract\":\"" << actor.observation_contract
                << "\",\"action_contract\":\"full-move-v1\",\"identity\":\"" << identity << "\"}\n";
        }
        std::cout << "full-move training: " << learner_roster.size() << " learner character"
                  << (learner_roster.size() == 1 ? " (" + learners_text + ")" : "s") << " vs " << opponent_roster.size()
                  << " opponents, " << options.environments << " environments"
                  << (options.allow_provisional ? ", provisional geometry" : "") << "; entropy floor "
                  << options.entropy_target << " nats\n";

        std::optional<GpuActorCritic> self_play_opponent;
        std::optional<GpuActorCritic> champion_policy;
        if (!options.resume) state.next_champion_step = options.champion_interval_steps;
        if (!options.resume) {
            for (const auto& extra : options.league_extras) {
                state.pool.push_back({extra.generic_string(), 0, 1000.0, false, true});
            }
            if (options.exploit_target) {
                state.pool.clear();
                state.pool.push_back({options.exploit_target->generic_string(), 0, 1000.0, false, true});
            }
        }
        const auto load_opponent = [&](const std::string& path) -> GpuActorCritic& {
            if (!self_play_opponent) self_play_opponent.emplace(capacity, actor, options.seed + 1);
            self_play_opponent->load_checkpoint(path, false);
            self_play_opponent->set_action_feature_tables(features, tables);
            return *self_play_opponent;
        };
        const auto started = std::chrono::steady_clock::now();
        const double elapsed_before = state.elapsed_seconds;
        const std::uint64_t frames_before = state.engine_frames;
        // Save and quit: a stop_request file in the run directory (T8Train's Q key)
        // checkpoints after the current update and exits. A stale one is ignored.
        const auto stop_request = options.run_directory / "stop_request";
        std::filesystem::remove(stop_request);
        for (std::uint64_t update = state.update + 1; update <= options.updates; ++update) {
            // --league-follow: the other runs' newest champions join the pool (three per run).
            if (!options.league_follow.empty() && update % options.checkpoint_interval == 1) {
                for (const auto& directory : options.league_follow) {
                    const auto champion = newest_champion(directory);
                    if (!champion) continue;
                    const auto path = champion->generic_string();
                    const auto prefix = directory.generic_string();
                    const bool known = std::any_of(state.pool.begin(), state.pool.end(),
                                                   [&](const PoolEntry& entry) { return entry.checkpoint == path; });
                    if (known) continue;
                    state.pool.push_back({path, 0, state.learner_rating, false, true});
                    std::vector<std::size_t> followed;
                    for (std::size_t index = 0; index < state.pool.size(); ++index) {
                        if (state.pool[index].external && state.pool[index].checkpoint.rfind(prefix, 0) == 0) {
                            followed.push_back(index);
                        }
                    }
                    if (followed.size() > 3) state.pool.erase(state.pool.begin() + static_cast<std::ptrdiff_t>(followed.front()));
                }
            }
            const double progress = std::min(1.0, static_cast<double>(update - 1) /
                                                      static_cast<double>(std::max<std::size_t>(1, options.anneal_updates)));
            PpoUpdateConfig update_config{};
            update_config.epochs = options.epochs;
            update_config.minibatch_size = options.minibatch_size;
            update_config.learning_rate = static_cast<float>(options.learning_rate +
                (options.final_learning_rate - options.learning_rate) * progress);
            // The schedule is a floor; the entropy target can only raise it.
            const float scheduled_entropy = static_cast<float>(options.entropy_coefficient +
                (options.final_entropy_coefficient - options.entropy_coefficient) * progress);
            update_config.entropy_coefficient = std::min(options.max_entropy_coefficient,
                                                         state.entropy_boost * scheduled_entropy);
            update_config.clip_range = options.clip_range;
            update_config.value_clip_range = options.value_clip_range;
            update_config.target_kl = options.target_kl;
            update_config.value_coefficient = options.value_coefficient;
            update_config.max_gradient_norm = options.max_gradient_norm;

            // Self-play against the league pool for a share of updates (always when exploiting).
            if (options.scripted_floor > 0.0F && state.scripted_share < 0.0F) {
                state.scripted_share = 1.0F - options.self_play_share;
            }
            const double self_play_share = options.scripted_floor > 0.0F ? 1.0 - state.scripted_share
                                                                         : options.self_play_share;
            const bool self_play = !state.pool.empty() &&
                (options.exploit_target.has_value() ||
                 (self_play_share > 0.0 && update >= options.self_play_start &&
                  static_cast<double>(mix(options.seed ^ (update * 7919ULL)) % 10'000ULL) <
                      self_play_share * 10'000.0));
            std::optional<std::size_t> pool_opponent;
            std::vector<int> opponents;
            std::vector<float> aggression;
            std::vector<int> scenarios;
            std::vector<int> learners(options.environments);
            if (options.learner_weakness_weight > 0.0F) {
                if (state.learner_score_estimate.size() != learner_roster.size()) {
                    state.learner_score_estimate.assign(learner_roster.size(), 0.5);
                }
                const auto pairs = weighted_learners(learner_roster, state.learner_score_estimate,
                                                     options.learner_weakness_weight, options.environments / 2, update);
                for (std::size_t env = 0; env < options.environments; ++env) learners[env] = pairs[env / 2];
            } else {
                for (std::size_t env = 0; env < options.environments; ++env) {
                    learners[env] = learner_for_pair(learner_roster, env / 2, update);
                }
            }
            if (self_play) {
                // The frozen policy plays another learner character (any pairing, mirrors included).
                opponents.resize(options.environments);
                for (std::size_t env = 0; env < options.environments; ++env) {
                    opponents[env] = learner_roster[static_cast<std::size_t>(
                        mix(options.seed ^ mix(update * 3'000'017ULL + env / 2)) % learner_roster.size())];
                }
                aggression.assign(options.environments, 0.5F);
                // Self-play rounds start neutral unless --self-play-scenarios: then the same share of
                // them starts in a scenario (a punish, a throw to break, a wakeup, ...) against the
                // frozen policy instead of a scripted bot.
                scenarios.assign(options.environments, full_core::kScenarioNeutral);
                if (options.self_play_scenarios) {
                    for (std::size_t env = 0; env < options.environments; ++env) {
                        scenarios[env] = scenario_for(options, update, env / 2);
                    }
                }
                double latest_share = 0.4;
                if (options.pool_anneal) {
                    const double pool_progress = std::min(1.0, static_cast<double>(update) /
                                                              static_cast<double>(std::max<std::size_t>(1, options.anneal_updates)));
                    latest_share = 0.8 + (0.1 - 0.8) * pool_progress;
                }
                pool_opponent = choose_pool_opponent(state.pool, state.learner_rating, options.seed, update, latest_share);
                load_opponent(state.pool[*pool_opponent].checkpoint);
            } else {
                schedule_opponents(options, opponent_roster, state.win_rate_estimate, update, options.environments,
                                   opponents, aggression, scenarios);
                if (options.scripted_tiers) assign_tier_bots(options, state.scripted_tier, update, aggression);
            }
            simulator.set_pending_opponents(opponents, aggression, scenarios);
            simulator.set_pending_learners(learners);
            if (self_play != state.self_play_active) {
                // Switching between scripted and mirror opponents restarts every round.
                simulator.reset_all(options.seed, update * options.horizon);
                state.self_play_active = self_play;
            }

            const auto collection_started = std::chrono::steady_clock::now();
            const std::uint64_t frames_at_start = simulator.frames_simulated();
            for (std::size_t step = 0; step < options.horizon; ++step) {
                const std::uint64_t counter = update * options.horizon + step;
                const auto output = learner.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                    options.environments, options.seed + 1, counter, false, nullptr,
                                                    simulator.learner_feature_tables());
                rollout.record_policy_device(step, simulator.learner_observations(), simulator.learner_masks(),
                                             output.actions, output.log_probabilities, output.values, nullptr,
                                             simulator.learner_feature_tables(), output.entropies);
                const std::int64_t* opponent_actions = nullptr;
                if (self_play) {
                    opponent_actions = self_play_opponent->forward(
                        simulator.opponent_observations(), simulator.opponent_masks(), options.environments,
                        options.seed + 2, counter, false, nullptr, simulator.opponent_feature_tables()).actions;
                }
                simulator.step(output.actions, opponent_actions, options.seed, counter + 1);
                const auto next = learner.forward(simulator.learner_observations(), simulator.learner_masks(),
                                                  options.environments, options.seed + 3, counter, true, nullptr,
                                                  simulator.learner_feature_tables());
                rollout.record_outcome_device(step,
                                              options.sparse_reward ? simulator.sparse_rewards() : simulator.rewards(),
                                              simulator.terminated(), simulator.truncated(), next.values,
                                              options.reward_scale);
                simulator.reset_done(options.seed, counter + 1);
            }
            const double reward_mean = rollout.mean_reward();  // synchronizes: collection is done
            const double collection_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - collection_started).count();
            const auto learning_started = std::chrono::steady_clock::now();
            std::optional<float> return_std;
            if (options.return_normalization) return_std = rollout.normalize_rewards(options.gamma);
            rollout.compute_gae(options.gamma, options.gae_lambda, true);
            const auto metrics = learner.update_ppo(rollout.device_view(), update_config, options.seed + update);
            learner.synchronize();
            const double learning_seconds =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - learning_started).count();
            if (options.entropy_target > 0.0F && std::isfinite(metrics.rollout_decision_entropy)) {
                state.entropy_boost *= std::exp(options.entropy_target_gain *
                                                (options.entropy_target - metrics.rollout_decision_entropy));
                const float ceiling = options.max_entropy_coefficient / std::max(scheduled_entropy, 1e-8F);
                state.entropy_boost = std::clamp(state.entropy_boost, 1.0F, std::max(1.0F, ceiling));
            }

            // Outcomes of finished rounds update the weakness estimate (scripted opponents only).
            const auto tally = simulator.download_and_clear_tally();
            const auto scenario_tally = simulator.download_scenario_tally();
            std::size_t wins = 0, losses = 0, draws = 0;
            for (std::size_t index = 0; index + 2 < tally.size(); index += 3) {
                wins += tally[index];
                losses += tally[index + 1];
                draws += tally[index + 2];
            }
            for (std::size_t index = 0; index < opponent_roster.size(); ++index) {
                std::size_t character_wins = 0, character_total = 0;
                for (int bucket = 0; bucket < GpuFullMoveSimulator::kAggressionBuckets; ++bucket) {
                    for (int side = 0; side < 2; ++side) {
                        const std::size_t base = ((static_cast<std::size_t>(opponent_roster[index]) * 5 +
                                                   static_cast<std::size_t>(bucket)) * 2 + static_cast<std::size_t>(side)) * 3;
                        character_wins += tally[base];
                        character_total += tally[base] + tally[base + 1] + tally[base + 2];
                    }
                }
                if (!self_play && character_total > 0) {
                    const double observed = static_cast<double>(character_wins) / static_cast<double>(character_total);
                    state.win_rate_estimate[index] = 0.9 * state.win_rate_estimate[index] + 0.1 * observed;
                }
            }
            // Each learner character's self-play score (draws half), for --learner-weakness-weight.
            if (self_play && state.learner_score_estimate.size() == learner_roster.size()) {
                const auto learner_tally = simulator.download_learner_tally();
                for (std::size_t index = 0; index < learner_roster.size(); ++index) {
                    const std::size_t base = static_cast<std::size_t>(learner_roster[index]) * 3;
                    const double total = static_cast<double>(learner_tally[base] + learner_tally[base + 1] +
                                                             learner_tally[base + 2]);
                    if (total <= 0.0) continue;
                    const double observed = (static_cast<double>(learner_tally[base]) +
                                             0.5 * static_cast<double>(learner_tally[base + 2])) / total;
                    state.learner_score_estimate[index] = 0.9 * state.learner_score_estimate[index] + 0.1 * observed;
                }
            }
            if (options.scripted_tiers && !self_play) {
                const auto bot_tally = simulator.download_bot_tally();
                double score = 0.0, rounds = 0.0;
                for (const int bot : bots_by_tier()[static_cast<std::size_t>(state.scripted_tier)]) {
                    const std::size_t base = static_cast<std::size_t>(bot) * 3;
                    score += static_cast<double>(bot_tally[base]) + 0.5 * static_cast<double>(bot_tally[base + 2]);
                    rounds += static_cast<double>(bot_tally[base] + bot_tally[base + 1] + bot_tally[base + 2]);
                }
                if (rounds > 0.0) {
                    state.tier_score = 0.8 * state.tier_score + 0.2 * (score / rounds);
                    ++state.updates_at_tier;
                    const bool settled = state.updates_at_tier >= options.tier_min_updates;
                    if (settled && state.tier_score >= options.tier_promote &&
                        state.scripted_tier < full_core::kScriptedTierCount - 1) {
                        ++state.scripted_tier;
                        state.tier_score = 0.5;
                        state.updates_at_tier = 0;
                    } else if (settled && state.tier_score < options.tier_demote && state.scripted_tier > 0) {
                        --state.scripted_tier;
                        state.tier_score = 0.5;
                        state.updates_at_tier = 0;
                    }
                }
            }
            const bool ladder_done = !options.scripted_tiers || state.scripted_tier == full_core::kScriptedTierCount - 1;
            if (ladder_done && options.scripted_floor > 0.0F && !self_play && update >= options.self_play_start &&
                !state.pool.empty() && wins + losses + draws > 0) {
                const double scripted_score = (static_cast<double>(wins) + 0.5 * static_cast<double>(draws)) /
                    static_cast<double>(wins + losses + draws);
                if (scripted_score >= 0.95) {
                    state.scripted_share = std::max(options.scripted_floor, state.scripted_share * 0.97F);
                } else if (scripted_score < 0.90) {
                    state.scripted_share = std::min(1.0F - options.self_play_share, state.scripted_share + 0.05F);
                }
            }
            if (pool_opponent && wins + losses + draws > 0) {
                // Elo step from this update's self-play rounds; the opponent's rating moves oppositely.
                auto& entry = state.pool[*pool_opponent];
                const double score = (static_cast<double>(wins) + 0.5 * static_cast<double>(draws)) /
                    static_cast<double>(wins + losses + draws);
                const double expected = 1.0 / (1.0 + std::pow(10.0, (entry.rating - state.learner_rating) / 400.0));
                const double delta = 24.0 * (score - expected);
                state.learner_rating += delta;
                entry.rating -= delta;
            }
            state.update = update;
            state.environment_steps += options.environments * options.horizon;
            const std::uint64_t frames = frames_before + simulator.frames_simulated();
            const double elapsed = elapsed_before +
                std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
            state.elapsed_seconds = elapsed;

            // Champion match.
            std::string champion_event;
            std::optional<double> champion_match_share;
            const auto crown = [&](double share) {
                const auto directory = options.run_directory / "champion";
                std::filesystem::create_directories(directory);
                const auto path = directory / ("update_" + std::to_string(update) + ".t8ppo");
                // A champion crowned after the resumed checkpoint (the run stopped without saving)
                // is crowned again identically: replace it.
                std::filesystem::remove(path);
                learner.save_checkpoint(path);
                state.champion = path.generic_string();
                state.champion_update = update;
                state.champion_share = share;
                state.champion_loss_streak = 0;
                state.environments = simulator.snapshot();
                state.returns = rollout.return_normalizer_state();
                auto state_path = path;
                state_path.replace_extension(".t8fmstate");
                save_state(state_path, catalog.catalog_sha256, catalog.roster_version, options, learners_text, identity,
                           state);
            };
            if (champion_arena && state.environment_steps >= state.next_champion_step) {
                state.next_champion_step += options.champion_interval_steps;
                ++state.champion_matches;
                if (state.champion.empty()) {
                    crown(0.5);
                    champion_event = "first champion crowned";
                } else {
                    if (!champion_policy) champion_policy.emplace(capacity, actor, options.seed + 7);
                    champion_policy->load_checkpoint(state.champion, false);
                    champion_policy->set_action_feature_tables(features, tables);
                    // Sampled, as the policies actually play (greedy play stalls into draws).
                    const auto result = head_to_head(learner, *champion_policy, *champion_arena, learner_roster,
                                                     options.seed + 950'000 + state.champion_matches, false);
                    const std::size_t rounds = result[0] + result[1] + result[2];
                    const double share = rounds ? (static_cast<double>(result[0]) + 0.5 * static_cast<double>(result[2])) /
                        static_cast<double>(rounds) : 0.0;
                    champion_match_share = share;
                    std::ostringstream event;
                    event << std::fixed << std::setprecision(3) << "share " << share << " vs update "
                          << state.champion_update << ": ";
                    if (share >= options.champion_promote) {
                        crown(share);
                        event << "new champion";
                    } else if (share >= options.champion_rollback) {
                        state.champion_loss_streak = 0;
                        event << "champion kept";
                    } else if (++state.champion_loss_streak >= options.champion_rollback_after) {
                        learner.load_checkpoint(state.champion, false);
                        learner.set_action_feature_tables(features, tables);
                        // The champion's value head was trained against its own return scale, and its
                        // entropy bonus against its own entropy: restore both with the weights.
                        auto champion_state_path = std::filesystem::path(state.champion);
                        champion_state_path.replace_extension(".t8fmstate");
                        const auto champion_state =
                            load_state(champion_state_path, catalog.catalog_sha256, learners_text, identity, options.fork);
                        rollout.restore_return_normalizer_state(champion_state.returns);
                        state.entropy_boost = champion_state.entropy_boost;
                        state.champion_loss_streak = 0;
                        ++state.rollbacks;
                        event << "ROLLED BACK to the champion";
                    } else {
                        event << "worse (loss streak " << state.champion_loss_streak << ")";
                    }
                    champion_event = event.str();
                }
                std::ostringstream line;
                line << std::setprecision(6) << update << ',' << state.environment_steps << ','
                     << (champion_match_share ? *champion_match_share : -1.0) << ',' << state.champion_update << ','
                     << state.champion_loss_streak << ",\"" << champion_event << "\"";
                append_line(options.run_directory / "champion_history.csv",
                            "update,environment_steps,share,champion_update,loss_streak,event", line.str());
            }

            std::optional<Evaluation> deterministic_eval;
            std::optional<Evaluation> stochastic_eval;
            if (update % options.evaluation_interval == 0 || update == options.updates) {
                deterministic_eval = evaluate(learner, evaluator, learner_roster, opponent_roster, true, options.seed + 900'000);
                stochastic_eval = evaluate(learner, evaluator, learner_roster, opponent_roster, false, options.seed + 900'001);
            }
            const bool periodic = update % options.checkpoint_interval == 0 || update == options.updates;
            const bool stopping = std::filesystem::exists(stop_request) || update == options.stop_at_update;
            const bool saving = periodic || stopping;
            bool promoted = false;
            std::optional<double> exploit_win_rate;
            if (deterministic_eval && options.exploit_target) {
                const auto result = head_to_head(learner, load_opponent(options.exploit_target->generic_string()),
                                                 evaluator, learner_roster, options.seed + 900'002);
                const std::size_t rounds = result[0] + result[1] + result[2];
                exploit_win_rate = rounds ? static_cast<double>(result[0]) / static_cast<double>(rounds) : 0.0;
            } else if (deterministic_eval && saving) {
                // Promotion gate against held-out regression and forgetting.
                const double side_gap = std::abs(deterministic_eval->p1_win_rate - deterministic_eval->p2_win_rate);
                std::vector<std::string> reasons;
                if (deterministic_eval->win_rate + options.promotion_tolerance < state.best_scripted_win_rate) {
                    reasons.push_back("held-out win rate below the best promoted");
                }
                if (side_gap > options.promotion_max_side_gap) reasons.push_back("side gap too large");
                double worst_score = 1.0;
                std::size_t checked = 0;
                for (auto entry = state.pool.rbegin(); entry != state.pool.rend() && checked < options.regression_set_size;
                     ++entry) {
                    if (!entry->promoted) continue;
                    ++checked;
                    const auto result = head_to_head(learner, load_opponent(entry->checkpoint), evaluator, learner_roster,
                                                     options.seed + 900'003 + entry->update);
                    const std::size_t rounds = result[0] + result[1] + result[2];
                    const double score = rounds ? (static_cast<double>(result[0]) + 0.5 * static_cast<double>(result[2])) /
                        static_cast<double>(rounds) : 0.0;
                    worst_score = std::min(worst_score, score);
                }
                if (worst_score < options.promotion_min_score) reasons.push_back("loses to a recently promoted checkpoint");
                promoted = reasons.empty();
                if (promoted) state.best_scripted_win_rate = std::max(state.best_scripted_win_rate, deterministic_eval->win_rate);
                std::ostringstream reason_text;
                for (std::size_t index = 0; index < reasons.size(); ++index) reason_text << (index ? "; " : "") << reasons[index];
                std::ostringstream promotion_line;
                promotion_line << std::setprecision(6) << update << ',' << (promoted ? 1 : 0) << ','
                               << deterministic_eval->win_rate << ',' << state.best_scripted_win_rate << ',' << side_gap
                               << ',' << checked << ',' << (checked ? worst_score : 1.0) << ",\"" << reason_text.str() << "\"";
                append_line(options.run_directory / "promotions.csv",
                            "update,promoted,heldout_win_rate,best_promoted_win_rate,side_gap,regression_opponents,"
                            "worst_regression_score,reasons",
                            promotion_line.str());
            }
            std::ostringstream row;
            row << std::setprecision(9) << "{\"update\":" << update << ",\"environment_steps\":" << state.environment_steps
                << ",\"engine_frames\":" << frames << ",\"elapsed_seconds\":" << elapsed
                << ",\"reward_mode\":\"" << (options.sparse_reward ? "sparse" : "shaped") << "\""
                << ",\"observation_mode\":\"full_move_privileged\",\"provisional_geometry\":"
                << (options.allow_provisional ? "true" : "false")
                << ",\"policy_loss\":" << metrics.policy_loss << ",\"value_loss\":" << metrics.value_loss
                << ",\"entropy\":" << metrics.entropy << ",\"approximate_kl\":" << metrics.approximate_kl
                << ",\"clip_fraction\":" << metrics.clip_fraction << ",\"gradient_norm\":" << metrics.gradient_norm
                << ",\"minibatches\":" << metrics.minibatches << ",\"epochs_completed\":" << metrics.epochs_completed
                << ",\"kl_early_stop\":" << (metrics.early_stopped ? "true" : "false")
                << ",\"decision_fraction\":" << metrics.decision_fraction
                << ",\"decision_entropy\":" << metrics.decision_entropy
                << ",\"rollout_decision_entropy\":" << metrics.rollout_decision_entropy
                << ",\"entropy_coefficient\":" << update_config.entropy_coefficient
                << ",\"entropy_boost\":" << state.entropy_boost << ",\"scripted_share\":" << state.scripted_share
                << ",\"scripted_tier\":" << state.scripted_tier << ",\"tier_score\":" << state.tier_score
                << ",\"learning_rate\":" << update_config.learning_rate
                << ",\"training_opponent\":\"" << (self_play ? "self_play_mirror" : "scripted_full_move") << "\""
                << ",\"rollout_reward_mean\":" << reward_mean
                << ",\"learner_rating\":" << state.learner_rating << ",\"league_pool\":" << state.pool.size()
                << ",\"training_rounds\":{\"wins\":" << wins << ",\"losses\":" << losses << ",\"draws\":" << draws << "}"
                << ",\"scenario_rounds\":{";
            for (int scenario = 0; scenario < full_core::kScenarioCount; ++scenario) {
                const std::size_t base = static_cast<std::size_t>(scenario) * 3;
                row << (scenario ? "," : "") << '"' << full_core::scenario_name(scenario) << "\":[" << scenario_tally[base]
                    << ',' << scenario_tally[base + 1] << ',' << scenario_tally[base + 2] << ']';
            }
            row << '}';
            if (return_std) row << ",\"return_std\":" << *return_std;
            if (pool_opponent) {
                row << ",\"self_play_opponent\":\"" << state.pool[*pool_opponent].checkpoint << "\"";
            }
            if (exploit_win_rate) row << ",\"exploiter_win_rate\":" << *exploit_win_rate;
            row << ",\"champion_update\":" << state.champion_update << ",\"champion_loss_streak\":"
                << state.champion_loss_streak << ",\"rollbacks\":" << state.rollbacks;
            if (champion_match_share) row << ",\"champion_match_share\":" << *champion_match_share;
            row << ",\"collection_seconds\":" << collection_seconds << ",\"learning_seconds\":" << learning_seconds;
            if (deterministic_eval && saving && !options.exploit_target) {
                row << ",\"promoted\":" << (promoted ? "true" : "false");
            }
            if (deterministic_eval) {
                row << ",\"evaluation\":{\"episodes\":" << deterministic_eval->episodes
                    << ",\"win_rate\":" << deterministic_eval->win_rate << ",\"draw_rate\":" << deterministic_eval->draw_rate
                    << ",\"worst_character\":\"" << deterministic_eval->worst_character << "\"}";
            }
            row << '}';
            append_line(metrics_path, "", row.str());
            if (deterministic_eval) {
                const auto& d = *deterministic_eval;
                const auto& s = *stochastic_eval;
                std::ostringstream line;
                line << std::setprecision(9) << update << ',' << state.environment_steps << ',' << elapsed << ','
                     << d.episodes << ',' << (self_play ? "self_play" : "scripted") << ",0,0," << d.win_rate << ','
                     << d.p1_win_rate << ',' << d.p2_win_rate << ',' << d.draw_rate << ',' << d.worst_character_win_rate
                     << ",0,0," << s.win_rate << ',' << s.p1_win_rate << ',' << s.p2_win_rate << ',' << s.draw_rate << ','
                     << s.worst_character_win_rate << ",0,0,none";
                append_line(options.run_directory / "evaluations.csv",
                            "update,environment_steps,elapsed_seconds,episodes,training_opponent,"
                            "latest_checkpoint_update,best_older_checkpoint_update,win_rate,p1_win_rate,p2_win_rate,"
                            "draw_rate,worst_style_win_rate,mean_damage_dealt,mean_damage_taken,stochastic_win_rate,"
                            "stochastic_p1_win_rate,stochastic_p2_win_rate,stochastic_draw_rate,"
                            "stochastic_worst_style_win_rate,stochastic_mean_damage_dealt,stochastic_mean_damage_taken,"
                            "guard_action",
                            line.str());
                for (const auto& [slug, counts] : d.characters) {
                    std::ostringstream character_line;
                    character_line << update << ",deterministic,0," << slug << ',' << counts[3] << ',' << counts[0] << ','
                                   << counts[1] << ',' << counts[2] << ','
                                   << (counts[3] ? static_cast<double>(counts[0]) / static_cast<double>(counts[3]) : 0.0);
                    append_line(options.run_directory / "evaluation_characters.csv",
                                "update,policy,character_id,character,episodes,wins,losses,draws,win_rate",
                                character_line.str());
                }
                for (const auto& [slug, counts] : d.learners) {
                    std::ostringstream learner_line;
                    learner_line << update << ",deterministic," << slug << ',' << counts[3] << ',' << counts[0] << ','
                                 << counts[1] << ',' << counts[2] << ','
                                 << (counts[3] ? static_cast<double>(counts[0]) / static_cast<double>(counts[3]) : 0.0);
                    append_line(options.run_directory / "evaluation_learners.csv",
                                "update,policy,learner_character,episodes,wins,losses,draws,win_rate", learner_line.str());
                }
                for (std::size_t style = 0; style < d.styles.size(); ++style) {
                    const auto& counts = d.styles[style];
                    std::ostringstream style_line;
                    style_line << update << ",deterministic,total," << style << ',' << counts[3] << ',' << counts[0]
                               << ',' << counts[1] << ',' << counts[2] << ','
                               << (counts[3] ? static_cast<double>(counts[0]) / static_cast<double>(counts[3]) : 0.0);
                    append_line(options.run_directory / "evaluation_styles.csv",
                                "update,policy,side,style,episodes,wins,losses,draws,win_rate", style_line.str());
                }
            }
            {
                const std::uint64_t update_frames = simulator.frames_simulated() - frames_at_start;
                const double update_seconds = collection_seconds + learning_seconds;
                const std::size_t decided = options.environments * options.horizon;
                const auto share_of = [](std::size_t won, std::size_t lost, std::size_t drawn) {
                    const std::size_t total = won + lost + drawn;
                    return total ? (static_cast<double>(won) + 0.5 * static_cast<double>(drawn)) / static_cast<double>(total)
                                 : 0.0;
                };
                const auto grouped = [](double value) {
                    std::string digits = std::to_string(static_cast<long long>(std::llround(value)));
                    for (int position = static_cast<int>(digits.size()) - 3; position > 0; position -= 3) {
                        digits.insert(static_cast<std::size_t>(position), ",");
                    }
                    return digits;
                };
                const auto clock = [](double seconds) {
                    const auto total = static_cast<long long>(seconds);
                    std::ostringstream out;
                    out << total / 3600 << ':' << std::setw(2) << std::setfill('0') << (total / 60) % 60 << ':'
                        << std::setw(2) << std::setfill('0') << total % 60;
                    return out.str();
                };
                const double remaining_seconds = update_seconds * static_cast<double>(options.updates - update);
                std::ostringstream report;
                report << std::fixed;
                report << "========================================\n";
                report << "Update " << update << " / " << options.updates << std::setprecision(1) << "  ("
                       << 100.0 * static_cast<double>(update) / static_cast<double>(options.updates) << "%)   Elapsed "
                       << clock(elapsed) << "   ETA " << clock(remaining_seconds) << '\n';
                report << "Opponent: " << (self_play ? "self-play vs " + std::filesystem::path(
                                               state.pool[*pool_opponent].checkpoint).stem().string()
                                                    : std::string("scripted roster"));
                if (options.scripted_floor > 0.0F && state.scripted_share >= 0.0F) {
                    if (options.scripted_tiers) {
                    report << "\nScripted Tier: " << state.scripted_tier << ' ' << tier_name(state.scripted_tier)
                           << std::setprecision(2) << "   score " << state.tier_score << " over "
                           << state.updates_at_tier << " scripted updates";
                }
                report << std::setprecision(0) << "   (mix: scripted " << 100.0 * state.scripted_share
                           << "%, self-play " << 100.0 * (1.0 - state.scripted_share) << "%)";
                }
                report << '\n';
                report << '\n' << std::setprecision(4);
                report << "Average Step Reward: " << reward_mean << '\n';
                report << "Policy Entropy (per choice): " << metrics.rollout_decision_entropy << " nats   floor "
                       << std::setprecision(2) << options.entropy_target << '\n';
                report << "Entropy Coefficient: " << std::setprecision(5) << update_config.entropy_coefficient
                       << std::setprecision(1) << "  (floor boost x" << state.entropy_boost << ")\n";
                report << std::setprecision(5) << "Post-Update KL: " << metrics.approximate_kl << "   Clip Fraction: "
                       << metrics.clip_fraction << '\n';
                report << std::scientific << std::setprecision(3) << "Policy LR: " << update_config.learning_rate
                       << std::fixed << '\n';
                report << std::setprecision(4) << "Value Loss: " << metrics.value_loss << '\n';
                report << '\n' << "Training Rounds: " << wins << " W / " << losses << " L / " << draws << " D   share "
                       << std::setprecision(3) << share_of(wins, losses, draws) << '\n';
                report << "Scenario Share:";
                for (int scenario = 0; scenario < full_core::kScenarioCount; ++scenario) {
                    const std::size_t base = static_cast<std::size_t>(scenario) * 3;
                    if (scenario_tally[base] + scenario_tally[base + 1] + scenario_tally[base + 2] == 0) continue;
                    report << ' ' << full_core::scenario_name(scenario) << ' ' << std::setprecision(2)
                           << share_of(scenario_tally[base], scenario_tally[base + 1], scenario_tally[base + 2]);
                }
                report << '\n';
                if (deterministic_eval) {
                    std::string worst_learner;
                    double worst_learner_rate = 2.0;
                    for (const auto& [slug, counts] : deterministic_eval->learners) {
                        const double rate = counts[3] ? static_cast<double>(counts[0]) / static_cast<double>(counts[3]) : 0.0;
                        if (rate < worst_learner_rate) {
                            worst_learner_rate = rate;
                            worst_learner = slug;
                        }
                    }
                    report << std::setprecision(3) << "Held-Out Win Rate: " << deterministic_eval->win_rate << "  (P1 "
                           << deterministic_eval->p1_win_rate << " / P2 " << deterministic_eval->p2_win_rate
                           << ")   worst learner " << worst_learner << ' ' << worst_learner_rate << ", worst opponent "
                           << deterministic_eval->worst_character << ' ' << deterministic_eval->worst_character_win_rate
                           << '\n';
                }
                if (exploit_win_rate) report << "Exploiter Win Rate vs Target: " << *exploit_win_rate << '\n';
                report << '\n' << std::setprecision(0) << "Learner Rating: " << state.learner_rating
                       << "   League Pool: " << state.pool.size() << '\n';
                report << "Champion: " << (state.champion.empty() ? std::string("none yet")
                                                                  : "update " + std::to_string(state.champion_update))
                       << "   Loss Streak: " << state.champion_loss_streak << "   Rollbacks: " << state.rollbacks;
                if (options.champion_interval_steps > 0) {
                    report << "   Next Match: " << grouped(static_cast<double>(state.next_champion_step)) << " steps";
                }
                report << '\n';
                if (!champion_event.empty()) report << "Champion Match: " << champion_event << '\n';
                report << '\n' << "Steps/Second: " << grouped(static_cast<double>(decided) / update_seconds)
                       << "   Frames/Second: " << grouped(static_cast<double>(update_frames) / update_seconds) << '\n';
                report << std::setprecision(3) << "Collection Time: " << collection_seconds << "   Learn Time: "
                       << learning_seconds << '\n';
                report << '\n' << "Total Steps: " << grouped(static_cast<double>(state.environment_steps))
                       << "   Total Frames: " << grouped(static_cast<double>(frames)) << '\n';
                report << "========================================";
                std::cout << report.str() << std::endl;
            }

            if (saving) {
                const auto checkpoint = options.run_directory / "checkpoints" / ("update_" + std::to_string(update) + ".t8ppo");
                learner.save_checkpoint(checkpoint);
                // Only periodic checkpoints join the league pool, so a save-and-quit resumes
                // exactly as if training had never stopped.
                if (!options.exploit_target && periodic) {
                    state.pool.push_back({checkpoint.generic_string(), update, state.learner_rating, promoted, false});
                    trim_pool(state.pool, options.league_pool_size);
                }
                state.returns = rollout.return_normalizer_state();
                state.environments = simulator.snapshot();
                state.engine_frames = frames;
                auto state_path = checkpoint;
                state_path.replace_extension(".t8fmstate");
                save_state(state_path, catalog.catalog_sha256, catalog.roster_version, options, learners_text, identity,
                           state);
            }
            if (stopping) {
                std::filesystem::remove(stop_request);
                std::cout << "Saved update " << update << " (" << state.environment_steps
                          << " steps); stopping on request." << std::endl;
                break;
            }
        }
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "full-move training error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
