// Frame-by-frame JSON feed of a full-move fight for scripts/visualizer.py.
//
//   t8_v2_full_move_feed --character jun --opponent king [--checkpoint update_40.t8ppo]
//                        [--opponent-checkpoint other.t8ppo] [--frames N] [--interval-ms N]
//                        [--seed N] [--allow-provisional] [--follow-dir runs/x/checkpoints] [--sample]
//                        [--bot N] [--opponent-bot N] [--perception-delay N] [--hold-frames N] [--emit-observation]   (tiered scripted bots, 0-29, for sides without a checkpoint)
//
// One fight runs on the CPU with the shared frame step (full_combat_core.hpp).
// With a checkpoint, the learner's decisions come from that full-move policy
// (GPU inference; the line includes its top candidates and probabilities),
// otherwise both fighters are scripted. Each line describes one frame: both
// fighters' complete state, current moves, legal candidate counts, the last
// contacts, the checkpoint's identity, and where the opponent's decisions come
// from. --follow-dir reloads the newest checkpoint in that directory each round.

#include "t8_v2/full_move_contract.hpp"
#include "t8_v2/full_move_roster.hpp"
#include "t8_v2/ppo.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace t8::v2;
using namespace t8::v2::full_core;

std::string json_escape(const std::string& value) {
    std::string out;
    for (const char character : value) {
        if (character == '"' || character == '\\') out += '\\';
        out += character;
    }
    return out;
}

struct Identity {
    std::string path;
    std::uint64_t update = 0;
    std::string catalog;
    bool provisional = false;
};

std::string read_string(std::istream& input) {
    std::uint64_t count = 0;
    input.read(reinterpret_cast<char*>(&count), sizeof count);
    if (!input || count > 4096) return {};
    std::string value(count, '\0');
    input.read(value.data(), static_cast<std::streamsize>(count));
    return value;
}

Identity read_identity(const std::filesystem::path& checkpoint) {
    Identity identity{checkpoint.generic_string()};
    auto state = checkpoint;
    state.replace_extension(".t8fmstate");
    std::ifstream input(state, std::ios::binary);
    if (!input) return identity;
    char magic[8] = {};
    input.read(magic, sizeof magic);
    std::uint32_t version = 0;
    input.read(reinterpret_cast<char*>(&version), sizeof version);
    input.read(reinterpret_cast<char*>(&identity.update), sizeof identity.update);
    identity.catalog = read_string(input);
    read_string(input);
    read_string(input);
    char provisional = 0;
    input.read(&provisional, 1);
    identity.provisional = provisional != 0;
    return identity;
}

ActorCriticConfig config_from_checkpoint(const std::filesystem::path& path) {
    std::ifstream input(path, std::ios::binary);
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
    if (config.action_contract != "full-move-v1") throw std::runtime_error(path.string() + " is not a full-move checkpoint");
    return config;
}

// Greedy (the most likely move every time) unless --sample: then moves are drawn
// from the policy's probabilities, as in training.
bool sample_policy = false;

// A full-move policy evaluated one observation at a time.
class Policy {
public:
    Policy(const std::filesystem::path& checkpoint, const FullMoveCharacter& character, std::uint64_t seed = 1)
        : seed_(seed), identity_(read_identity(checkpoint)),
          network_(1, config_from_checkpoint(checkpoint), 1),
          observation_size_(config_from_checkpoint(checkpoint).observation_size),
          screen_(config_from_checkpoint(checkpoint).observation_contract.rfind("full-move-screen", 0) == 0) {
        network_.load_checkpoint(checkpoint, false);
        std::vector<float> features(static_cast<std::size_t>(kFullMoveMaxCandidates) * kFullMoveFeatureSize, 0.0F);
        std::copy(character.candidate_features.begin(), character.candidate_features.end(), features.begin());
        network_.set_action_features(features);
        cudaMalloc(reinterpret_cast<void**>(&observation_), sizeof(float) * observation_size_);
        cudaMalloc(reinterpret_cast<void**>(&mask_), kFullMoveMaxCandidates);
    }
    ~Policy() {
        cudaFree(observation_);
        cudaFree(mask_);
    }
    Policy(const Policy&) = delete;
    Policy& operator=(const Policy&) = delete;

    // The choice (greedy or sampled) plus the policy's probabilities over the legal candidates.
    int choose(const float* observation, const std::uint8_t* mask, std::vector<float>& probabilities) {
        cudaMemcpy(observation_, observation, sizeof(float) * observation_size_, cudaMemcpyHostToDevice);
        cudaMemcpy(mask_, mask, kFullMoveMaxCandidates, cudaMemcpyHostToDevice);
        const auto output = network_.forward(observation_, mask_, 1, seed_, step_++, !sample_policy);
        std::vector<float> logits(kFullMoveMaxCandidates);
        cudaMemcpy(logits.data(), output.logits, sizeof(float) * kFullMoveMaxCandidates, cudaMemcpyDeviceToHost);
        std::int64_t action = 0;
        cudaMemcpy(&action, output.actions, sizeof action, cudaMemcpyDeviceToHost);
        float maximum = -1e30F;
        for (int candidate = 0; candidate < kFullMoveMaxCandidates; ++candidate) {
            if (mask[candidate]) maximum = std::max(maximum, logits[static_cast<std::size_t>(candidate)]);
        }
        probabilities.assign(kFullMoveMaxCandidates, 0.0F);
        float total = 0.0F;
        for (int candidate = 0; candidate < kFullMoveMaxCandidates; ++candidate) {
            if (!mask[candidate]) continue;
            probabilities[static_cast<std::size_t>(candidate)] = std::exp(logits[static_cast<std::size_t>(candidate)] - maximum);
            total += probabilities[static_cast<std::size_t>(candidate)];
        }
        for (auto& value : probabilities) value /= total;
        return static_cast<int>(action);
    }
    [[nodiscard]] const Identity& identity() const noexcept { return identity_; }
    [[nodiscard]] int observation_size() const noexcept { return observation_size_; }
    [[nodiscard]] bool memory() const noexcept {
        return observation_size_ == full_move_observation_size(true, false) ||
               observation_size_ == full_move_observation_size(true, true);
    }
    [[nodiscard]] bool screen() const noexcept { return screen_; }
    [[nodiscard]] bool noise() const noexcept {
        return observation_size_ == full_move_observation_size(false, true) ||
               observation_size_ == full_move_observation_size(true, true);
    }

private:
    std::uint64_t seed_;
    Identity identity_;
    GpuActorCritic network_;
    int observation_size_ = kFullMoveObservationSize;
    bool screen_ = false;  // a --screen-observation checkpoint: sees apply_screen_view's features only
    float* observation_ = nullptr;
    std::uint8_t* mask_ = nullptr;
    std::uint64_t step_ = 0;
};

std::optional<std::filesystem::path> newest_checkpoint(const std::filesystem::path& directory) {
    std::optional<std::filesystem::path> best;
    std::uint64_t best_update = 0;
    if (!std::filesystem::exists(directory)) return best;
    for (const auto& entry : std::filesystem::directory_iterator(directory)) {
        const auto name = entry.path().filename().string();
        if (entry.path().extension() != ".t8ppo" || name.rfind("update_", 0) != 0) continue;
        auto state = entry.path();
        state.replace_extension(".t8fmstate");
        if (!std::filesystem::exists(state)) continue;
        const auto update = std::stoull(name.substr(7, name.size() - 13));
        if (!best || update > best_update) {
            best = entry.path();
            best_update = update;
        }
    }
    return best;
}

void write_fighter(std::ostream& out, const Tables& tables, const full_core::State& state, int player,
                   const FullMoveCharacter& character, const CandidateMap& map) {
    const Fighter& f = state.fighters[player];
    int legal_count = 0;
    for (int candidate = 0; candidate < map.count; ++candidate) legal_count += candidate_legal(tables, state, player, map, candidate);
    std::string move_id;
    if (f.move >= 0) {
        for (int candidate = 0; candidate < map.count; ++candidate) {
            if (map.moves[candidate] == f.move) {
                move_id = character.candidate_ids[static_cast<std::size_t>(candidate)];
                break;
            }
        }
        if (move_id.empty()) move_id = "engine:" + std::to_string(f.move);  // a reactive parry outcome
    }
    out << "{\"character\":\"" << character.slug << "\",\"health\":" << f.health
        << ",\"recoverable\":" << f.recoverable_health << ",\"x\":" << f.x << ",\"axis\":" << f.axis
        << ",\"posture\":" << static_cast<int>(f.posture) << ",\"stance\":" << f.stance
        << ",\"heat\":" << (f.heat ? 1 : 0) << ",\"heat_frames\":" << f.heat_frames
        << ",\"heat_available\":" << (f.heat_available ? 1 : 0) << ",\"rage\":" << (f.rage ? 1 : 0)
        << ",\"rage_available\":" << (f.rage_available ? 1 : 0) << ",\"resource\":" << f.resource
        << ",\"installed\":" << (f.installed ? 1 : 0) << ",\"move\":\"" << json_escape(move_id) << "\""
        << ",\"action_frame\":" << f.action_frame
        << ",\"move_frames\":" << (f.move >= 0 ? move_of(tables, player, f.move).total_frames : 0)
        << ",\"hits\":[";
    // The current move's hits as [level, first active frame, active frames, reach]
    // (levels: 0 high, 1 mid, 2 low, 3 special mid, 4 special low, 5 throw).
    if (f.move >= 0) {
        const auto& current = move_of(tables, player, f.move);
        for (int hit = 0; hit < current.hit_count; ++hit) {
            const auto& spec = hit_of(tables, player, current, hit);
            out << (hit ? "," : "") << '[' << static_cast<int>(spec.level) << ',' << spec.first_active_frame << ','
                << spec.active_frames << ',' << spec.reach << ']';
        }
    }
    out << "],\"move_hit\":" << (f.move_hit ? 1 : 0) << ",\"move_blocked\":" << (f.move_blocked ? 1 : 0)
        << ",\"heat_dash\":" << (f.heat_dash_from >= 0 ? 1 : 0) << ",\"crumpled\":" << (f.crumpled ? 1 : 0)
        << ",\"universal\":" << static_cast<int>(f.universal) << ",\"stun\":" << f.stun
        << ",\"blockstun\":" << (f.in_blockstun ? 1 : 0) << ",\"combo_hits\":" << f.combo_hits
        << ",\"back_turned\":" << (f.back_turned ? 1 : 0) << ",\"throw_break_frames\":" << f.throw_break_frames
        << ",\"break_attempted\":" << (f.throw_break_attempted ? 1 : 0)
        << ",\"block_advantage\":" << (f.move >= 0 ? move_of(tables, player, f.move).block_advantage : 0)
        // The button that breaks this fighter's connected throw (FullThrowBreak: 0 none, 1 one,
        // 2 two, 3 one+two, 4 one or two), -1 when no throw of this fighter is pending.
        << ",\"throw_rule\":"
        << (f.move >= 0 && f.pending_throw_hit >= 0
                ? static_cast<int>(hit_of(tables, player, move_of(tables, player, f.move), f.pending_throw_hit).throw_break)
                : -1)
        << ",\"legal_candidates\":" << legal_count << ",\"actionable\":" << (actionable(tables, state, player) ? 1 : 0)
        << '}';
}

}  // namespace

// A setting from the run.json identity of the checkpoint's run (e.g. "perception=" -> 12), or fallback.
int run_setting(const std::filesystem::path& checkpoint, const std::string& key, int fallback) {
    for (const auto& run_json : {checkpoint.parent_path().parent_path() / "run.json", checkpoint.parent_path() / "run.json"}) {
        std::ifstream in(run_json);
        if (!in) continue;
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto at = text.find(key);
        return at == std::string::npos ? fallback : std::atoi(text.c_str() + at + key.size());
    }
    return fallback;
}

// The live noise ranges a checkpoint's run trained with (run.json ";noise=p,f,min,max").
std::optional<std::array<float, 4>> run_noise(const std::filesystem::path& checkpoint) {
    for (const auto& run_json : {checkpoint.parent_path().parent_path() / "run.json", checkpoint.parent_path() / "run.json"}) {
        std::ifstream in(run_json);
        if (!in) continue;
        const std::string text((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
        const auto at = text.find(";noise=");
        if (at == std::string::npos) return std::nullopt;
        std::array<float, 4> values{};
        std::istringstream fields(text.substr(at + 7));
        for (auto& value : values) {
            std::string field;
            if (!std::getline(fields, field, ',')) return std::nullopt;
            value = std::stof(field);
        }
        return values;
    }
    return std::nullopt;
}

int main(int argc, char** argv) {
    std::string character_slug = "jun", opponent_slug = "jun";
    std::optional<std::filesystem::path> checkpoint, opponent_checkpoint, follow_dir;
    long long frames = -1;
    int interval_ms = 16;
    std::uint64_t seed = 1;
    bool allow_provisional = false;
    int bots[2] = {-1, -1};    // tiered scripted bot per side without a checkpoint (-1: legacy random)
    int perception_delay = -1;  // must match the checkpoint's training; -1: read it from the run's run.json
    int hold_frames = -1;       // likewise (--hold-frames)
    int live_noise = -1;        // --live-noise / --no-live-noise; -1: as the run trained (run.json)
    std::optional<float> noise_position, noise_flip;
    std::optional<int> latency_min, latency_max;
    bool emit_observation = false;  // --emit-observation: P1's policy input in every frame line
    std::filesystem::path generated = std::filesystem::path(T8_V2_SOURCE_DIR) / "data" / "generated";
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) throw std::runtime_error("missing value for " + argument);
            return argv[++index];
        };
        if (argument == "--character") character_slug = next();
        else if (argument == "--opponent") opponent_slug = next();
        else if (argument == "--checkpoint") checkpoint = std::filesystem::path(next());
        else if (argument == "--opponent-checkpoint") opponent_checkpoint = std::filesystem::path(next());
        else if (argument == "--follow-dir") follow_dir = std::filesystem::path(next());
        else if (argument == "--frames") frames = std::stoll(next());
        else if (argument == "--interval-ms") interval_ms = std::stoi(next());
        else if (argument == "--seed") seed = std::stoull(next());
        else if (argument == "--allow-provisional") allow_provisional = true;
        else if (argument == "--sample") sample_policy = true;
        else if (argument == "--bot") bots[0] = std::stoi(next());
        else if (argument == "--perception-delay") perception_delay = std::stoi(next());
        else if (argument == "--hold-frames") hold_frames = std::stoi(next());
        else if (argument == "--live-noise") live_noise = 1;
        else if (argument == "--no-live-noise") live_noise = 0;
        else if (argument == "--noise-position") noise_position = std::stof(next());
        else if (argument == "--noise-flip") noise_flip = std::stof(next());
        else if (argument == "--latency-min") latency_min = std::stoi(next());
        else if (argument == "--latency-max") latency_max = std::stoi(next());
        else if (argument == "--emit-observation") emit_observation = true;
        else if (argument == "--opponent-bot") bots[1] = std::stoi(next());
        else if (argument == "--data-dir") generated = next();
        else {
            std::cerr << "unknown argument " << argument << '\n';
            return 2;
        }
    }
    try {
        FullMoveRosterOptions options{};
        options.allow_provisional = allow_provisional;
        const auto own = load_full_move_character(generated, character_slug, options);
        const auto other = load_full_move_character(generated, opponent_slug, options);
        if (own.bound_moves == 0 || other.bound_moves == 0) {
            throw std::runtime_error("no move binds without stand-in geometry; pass --allow-provisional");
        }
        Tables tables{};
        tables.side[0] = own.tables.view().side[0];
        tables.side[1] = other.tables.view().side[0];
        tables.config = own.tables.config;
        const CandidateMap own_map{own.candidate_moves.data(), own.candidate_count()};
        const CandidateMap other_map{other.candidate_moves.data(), other.candidate_count()};
        std::unique_ptr<Policy> policy, opponent_policy;
        if (follow_dir) {
            if (const auto newest = newest_checkpoint(*follow_dir)) checkpoint = newest;
        }
        if (checkpoint) policy = std::make_unique<Policy>(*checkpoint, own, seed * 2 + 1);
        if (opponent_checkpoint) opponent_policy = std::make_unique<Policy>(*opponent_checkpoint, other, seed * 2 + 2);
        EnvConfig env{};
        // Play the way the checkpoint was trained: its run's perception delay and hold frames.
        const auto& reference = checkpoint ? checkpoint : opponent_checkpoint;
        if (perception_delay < 0) perception_delay = reference ? run_setting(*reference, ";perception=", 0) : 0;
        if (hold_frames < 0) hold_frames = reference ? run_setting(*reference, ";hold=", env.hold_frames) : env.hold_frames;
        env.hold_frames = hold_frames;
        // Live noise as the run trained (";noise=position,flip,latency_min,latency_max"), unless told.
        if (live_noise < 0) {
            const auto trained = reference ? run_noise(*reference) : std::nullopt;
            live_noise = trained ? 1 : 0;
            if (trained) {
                env.noise_position_max = (*trained)[0];
                env.noise_flip_max = (*trained)[1];
                env.latency_min = static_cast<int>((*trained)[2]);
                env.latency_max = static_cast<int>((*trained)[3]);
            }
        }
        env.live_noise = live_noise == 1;
        if (noise_position) env.noise_position_max = *noise_position;
        if (noise_flip) env.noise_flip_max = *noise_flip;
        if (latency_min) env.latency_min = *latency_min;
        if (latency_max) env.latency_max = *latency_max;
        std::cerr << "perception delay " << perception_delay << ", hold frames " << hold_frames;
        if (env.live_noise) {
            std::cerr << ", live noise (position " << env.noise_position_max << " m, errors " << env.noise_flip_max
                      << ", latency " << env.latency_min << '-' << env.latency_max << " frames)";
        }
        std::cerr << std::endl;
        full_core::State state{};
        std::uint64_t round = 0;
        env_reset(tables, state, env, seed, 0, round);
        Input held[2] = {};
        int hold_left[2] = {0, 0};
        std::vector<float> probabilities;
        std::array<std::vector<std::pair<int, float>>, 2> top;
        std::array<int, 2> last_choice = {-1, -1};
        // The exact input of each side's latest policy decision (what --emit-observation reports).
        constexpr int kWidest = full_move_observation_size(true, true);
        std::array<std::array<float, kWidest>, 2> decision_observation{};
        // --live-noise: this round's levels, and policy inputs waiting out the latency.
        NoiseLevels levels = env.live_noise ? draw_noise_levels(env, seed, 0, round) : NoiseLevels{};
        Input pending[2] = {};
        int delay_left[2] = {0, 0};
        // --memory checkpoints: recent moves and habits, updated every frame (full_core::Memory).
        Memory memory{};
        const auto uses_memory = [&]() {
            return (policy && policy->memory()) || (opponent_policy && opponent_policy->memory());
        };
        std::array<long long, 2> decisions = {0, 0};
        std::string recent_contacts;
        int contact_age = 0;
        for (long long frame = 0; frames < 0 || frame < frames; ++frame) {
            if (state.winner >= 0 || state.frame >= env.max_round_frames) {
                ++round;
                if (follow_dir) {
                    const auto newest = newest_checkpoint(*follow_dir);
                    if (newest && (!policy || policy->identity().path != newest->generic_string())) {
                        policy = std::make_unique<Policy>(*newest, own, seed * 2 + 1);
                    }
                }
                env_reset(tables, state, env, seed, 0, round);
                memory = Memory{};
                if (env.live_noise) levels = draw_noise_levels(env, seed, 0, round);
                hold_left[0] = hold_left[1] = 0;
                delay_left[0] = delay_left[1] = 0;
            }
            Input inputs[2] = {};
            for (int player = 0; player < 2; ++player) {
                const CandidateMap& map = player == 0 ? own_map : other_map;
                Policy* chooser = player == 0 ? policy.get() : opponent_policy.get();
                if (delay_left[player] > 0) {  // the decided input is still on its way
                    if (--delay_left[player] == 0) {
                        inputs[player] = pending[player];
                        held[player] = pending[player];
                        hold_left[player] = held_input(pending[player]) ? env.hold_frames - 1 : 0;
                    }
                    continue;
                }
                if (hold_left[player] > 0) {
                    inputs[player] = held[player];
                    --hold_left[player];
                    continue;
                }
                if (!has_decision(tables, state, player)) continue;
                int choice = 0;
                if (chooser) {
                    float observation[kWidest] = {};
                    encode_observation(tables, state, player, env.max_round_frames,
                                       player == 0 ? own.roster_id : other.roster_id,
                                       player == 0 ? other.roster_id : own.roster_id, observation,
                                       perception_delay);
                    if (env.live_noise) apply_observation_noise(levels, state, player, observation);
                    if (chooser->screen()) apply_screen_view(observation);
                    if (chooser->memory()) {
                        encode_memory(memory, player, perception_delay, observation + kFullMoveObservationSize);
                    }
                    if (chooser->noise()) {
                        encode_noise_levels(levels, observation + full_move_observation_size(chooser->memory(), false));
                    }
                    std::uint8_t mask[kFullMoveMaxCandidates] = {};
                    for (int candidate = 0; candidate < map.count; ++candidate) {
                        mask[candidate] = candidate_legal(tables, state, player, map, candidate);
                    }
                    choice = chooser->choose(observation, mask, probabilities);
                    std::copy(observation, observation + chooser->observation_size(),
                              decision_observation[static_cast<std::size_t>(player)].begin());
                    ++decisions[static_cast<std::size_t>(player)];
                    top[static_cast<std::size_t>(player)].clear();
                    for (int candidate = 0; candidate < map.count; ++candidate) {
                        if (mask[candidate]) top[static_cast<std::size_t>(player)].emplace_back(candidate, probabilities[static_cast<std::size_t>(candidate)]);
                    }
                    auto& ranked = top[static_cast<std::size_t>(player)];
                    std::partial_sort(ranked.begin(), ranked.begin() + std::min<std::ptrdiff_t>(5, static_cast<std::ptrdiff_t>(ranked.size())),
                                      ranked.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
                    if (ranked.size() > 5) ranked.resize(5);
                } else {
                    const int bot = bots[static_cast<std::size_t>(player)];
                    choice = bot >= 0 ? scripted_bot_candidate(tables, state, player, map, bot, seed,
                                                               static_cast<std::uint64_t>(player), static_cast<std::uint64_t>(frame))
                                      : scripted_candidate(tables, state, player, map, 0.5, seed,
                                                           static_cast<std::uint64_t>(player), static_cast<std::uint64_t>(frame));
                    top[static_cast<std::size_t>(player)].clear();
                }
                last_choice[static_cast<std::size_t>(player)] = choice;
                Input input{};
                if (!candidate_input(map, choice, input)) input = Input{};
                if (chooser && env.live_noise && levels.latency > 0) {  // lands `lag` frames from now
                    const auto at = static_cast<std::uint64_t>(decisions[static_cast<std::size_t>(player)]);
                    pending[player] = input;
                    delay_left[player] = levels.latency + (random_unit(random_bits(levels.key, 0x1a7U, at)) < 0.25 ? 1 : 0);
                    hold_left[player] = 0;
                    continue;
                }
                inputs[player] = input;
                held[player] = input;
                hold_left[player] = held_input(input) ? env.hold_frames - 1 : 0;
            }
            Events events{};
            step(tables, state, inputs[0], inputs[1], events);
            if (uses_memory()) memory_update(tables, state, events, perception_delay, memory);
            if (events.contact_count > 0) {
                std::ostringstream contacts;
                for (int index = 0; index < events.contact_count; ++index) {
                    const auto& contact = events.contacts[index];
                    const auto& attacker = state.fighters[contact.attacker];
                    int level = -1;
                    if (attacker.move >= 0) {
                        const auto& spec = move_of(tables, contact.attacker, attacker.move);
                        if (contact.hit_index >= 0 && contact.hit_index < spec.hit_count) {
                            level = static_cast<int>(hit_of(tables, contact.attacker, spec, contact.hit_index).level);
                        }
                    }
                    contacts << (index ? "," : "") << "{\"attacker\":" << contact.attacker << ",\"contact\":\""
                             << to_string(contact.contact) << "\",\"damage\":" << contact.damage
                             << ",\"hit\":" << contact.hit_index << ",\"level\":" << level << '}';
                }
                recent_contacts = contacts.str();
                contact_age = 0;
            } else if (++contact_age > 30) {
                recent_contacts.clear();
            }
            std::ostringstream line;
            line.precision(6);
            line << "{\"frame\":" << state.frame << ",\"round\":" << round << ",\"winner\":" << state.winner
                 << ",\"max_frames\":" << env.max_round_frames << ",\"contact_age\":" << contact_age
                 << ",\"stage\":{\"left\":" << state.stage.left_wall << ",\"right\":" << state.stage.right_wall
                 << ",\"level\":" << state.stage.level << "},\"fighters\":[";
            write_fighter(line, tables, state, 0, own, own_map);
            line << ',';
            write_fighter(line, tables, state, 1, other, other_map);
            line << "],\"choices\":[";
            for (int player = 0; player < 2; ++player) {
                const auto& character = player == 0 ? own : other;
                const int choice = last_choice[static_cast<std::size_t>(player)];
                line << (player ? "," : "") << "{\"last\":\""
                     << (choice >= 0 ? json_escape(character.candidate_ids[static_cast<std::size_t>(choice)]) : "")
                     << "\",\"top\":[";
                const auto& ranked = top[static_cast<std::size_t>(player)];
                for (std::size_t index = 0; index < ranked.size(); ++index) {
                    line << (index ? "," : "") << "[\""
                         << json_escape(character.candidate_ids[static_cast<std::size_t>(ranked[index].first)]) << "\","
                         << ranked[index].second << ']';
                }
                line << "]}";
            }
            line << "],\"contacts\":[" << recent_contacts << ']';
            if (emit_observation) {  // the input of P1's latest decision, exactly (the network visualizer)
                line << ",\"decision\":" << decisions[0] << ",\"observation\":[";
                const int emitted = policy ? policy->observation_size() : kFullMoveObservationSize;
                for (int index = 0; index < emitted; ++index) {
                    line << (index ? "," : "") << decision_observation[0][static_cast<std::size_t>(index)];
                }
                line << ']';
            }
            line << ",\"policy\":";
            if (policy) {
                const auto& identity = policy->identity();
                line << "{\"checkpoint\":\"" << json_escape(identity.path) << "\",\"update\":" << identity.update
                     << ",\"catalog\":\"" << identity.catalog.substr(0, 12) << "\",\"provisional\":"
                     << (identity.provisional ? "true" : "false") << '}';
            } else {
                line << "null";
            }
            line << ",\"opponent_source\":\""
                 << (opponent_policy ? "checkpoint:" + json_escape(opponent_policy->identity().path) : std::string("scripted"))
                 << "\",\"provisional_geometry\":" << (allow_provisional ? "true" : "false") << '}';
            std::cout << line.str() << '\n' << std::flush;
            if (interval_ms > 0) std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
        }
    } catch (const std::exception& error) {
        std::cerr << "full-move feed error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
