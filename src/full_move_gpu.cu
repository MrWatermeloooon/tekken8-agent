#include "t8_v2/full_move_gpu.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <utility>

namespace t8::v2 {
namespace {

using namespace full_core;

constexpr int kThreads = 128;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

unsigned blocks_for(std::size_t count) {
    return static_cast<unsigned>((count + kThreads - 1) / kThreads);
}

template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(std::size_t count) { allocate(count); }
    ~DeviceArray() { release(); }
    DeviceArray(DeviceArray&& other) noexcept : data_(std::exchange(other.data_, nullptr)),
                                                count_(std::exchange(other.count_, 0)) {}
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            release();
            data_ = std::exchange(other.data_, nullptr);
            count_ = std::exchange(other.count_, 0);
        }
        return *this;
    }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;

    void allocate(std::size_t count) {
        release();
        count_ = count;
        if (count > 0) {
            check_cuda(cudaMalloc(reinterpret_cast<void**>(&data_), sizeof(T) * count), "allocate full-move buffer");
            check_cuda(cudaMemset(data_, 0, sizeof(T) * count), "clear full-move buffer");
        }
    }
    void upload(const T* values, std::size_t count) {
        if (count != count_) throw std::invalid_argument("full-move upload size mismatch");
        if (count > 0) check_cuda(cudaMemcpy(data_, values, sizeof(T) * count, cudaMemcpyHostToDevice), "upload");
    }
    [[nodiscard]] std::vector<T> download() const {
        std::vector<T> values(count_);
        if (count_ > 0) {
            check_cuda(cudaMemcpy(values.data(), data_, sizeof(T) * count_, cudaMemcpyDeviceToHost), "download");
        }
        return values;
    }
    [[nodiscard]] T* data() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return count_; }

private:
    void release() noexcept {
        if (data_ != nullptr) cudaFree(data_);
        data_ = nullptr;
        count_ = 0;
    }
    T* data_ = nullptr;
    std::size_t count_ = 0;
};

// One roster character in device memory.
struct DeviceCharacter {
    Side side{};
    CandidateMap candidates{};
    int roster_id = -1;
};

__device__ Tables matchup_tables(const DeviceCharacter* characters, int p1, int p2, const FullEngineConfig& config) {
    Tables tables{};
    tables.side[0] = characters[p1].side;
    tables.side[1] = characters[p2].side;
    tables.config = config;
    return tables;
}

__global__ void engine_reset_kernel(const DeviceCharacter* characters, const int* p1_characters,
                                    const int* p2_characters, FullEngineConfig config, const double* p1_x,
                                    const double* p2_x, State* states, std::size_t count) {
    const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) return;
    const Tables tables = matchup_tables(characters, p1_characters[env], p2_characters[env], config);
    reset(tables, states[env], p1_x[env], p2_x[env]);
}

__global__ void engine_step_kernel(const DeviceCharacter* characters, const int* p1_characters,
                                   const int* p2_characters, FullEngineConfig config, const Input* inputs,
                                   const std::uint8_t* new_round, const double* p1_x, const double* p2_x,
                                   State* states, Events* events, unsigned long long* frames, std::size_t count) {
    const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) return;
    const Tables tables = matchup_tables(characters, p1_characters[env], p2_characters[env], config);
    State state = states[env];
    if (new_round != nullptr && new_round[env]) next_round(tables, state, p1_x[env], p2_x[env]);
    Events frame_events{};
    step(tables, state, inputs[2 * env], inputs[2 * env + 1], frame_events);
    states[env] = state;
    events[env] = frame_events;
    atomicAdd(frames, 1ULL);
}

__global__ void legal_mask_kernel(const DeviceCharacter* characters, const int* p1_characters,
                                  const int* p2_characters, FullEngineConfig config, const State* states, int player,
                                  std::uint8_t* masks, std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count * kFullMoveMaxCandidates) return;
    const std::size_t env = index / kFullMoveMaxCandidates;
    const int candidate = static_cast<int>(index % kFullMoveMaxCandidates);
    const int own_character = player == 0 ? p1_characters[env] : p2_characters[env];
    const Tables tables = matchup_tables(characters, p1_characters[env], p2_characters[env], config);
    masks[index] = candidate < characters[own_character].candidates.count &&
        candidate_legal(tables, states[env], player, characters[own_character].candidates, candidate);
}

__global__ void observe_kernel(const DeviceCharacter* characters, const int* p1_characters, const int* p2_characters,
                               FullEngineConfig config, EnvConfig env_config, const State* states,
                               const std::uint8_t* learner_is_p1, const Memory* memories,
                               const NoiseLevels* noise, int stride,
                               float* learner_observations, float* opponent_observations,
                               std::int32_t* learner_tables, std::int32_t* opponent_tables, std::size_t count) {
    const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) return;
    const Tables tables = matchup_tables(characters, p1_characters[env], p2_characters[env], config);
    const int learner = learner_is_p1[env] ? 0 : 1;
    const int own = learner == 0 ? p1_characters[env] : p2_characters[env];
    const int other = learner == 0 ? p2_characters[env] : p1_characters[env];
    float* learner_out = learner_observations + env * stride;
    float* opponent_out = opponent_observations + env * stride;
    encode_observation(tables, states[env], learner, env_config.max_round_frames, characters[own].roster_id,
                       characters[other].roster_id, learner_out, env_config.perception_delay);
    encode_observation(tables, states[env], 1 - learner, env_config.max_round_frames, characters[other].roster_id,
                       characters[own].roster_id, opponent_out, env_config.perception_delay);
    if (memories != nullptr) {
        encode_memory(memories[env], learner, env_config.perception_delay, learner_out + kFullMoveObservationSize);
        encode_memory(memories[env], 1 - learner, env_config.perception_delay, opponent_out + kFullMoveObservationSize);
    }
    if (noise != nullptr) {
        const int offset = full_move_observation_size(memories != nullptr, false);
        apply_observation_noise(noise[env], states[env], learner, learner_out);
        apply_observation_noise(noise[env], states[env], 1 - learner, opponent_out);
        encode_noise_levels(noise[env], learner_out + offset);
        encode_noise_levels(noise[env], opponent_out + offset);
    }
    if (env_config.screen_view) {
        apply_screen_view(learner_out);
        apply_screen_view(opponent_out);
    }
    learner_tables[env] = own;
    opponent_tables[env] = other;
}

// One thread per (environment, candidate, side): masks for both fighters.
__global__ void candidate_mask_kernel(const DeviceCharacter* characters, const int* p1_characters,
                                      const int* p2_characters, FullEngineConfig config, const State* states,
                                      const std::uint8_t* learner_is_p1, std::uint8_t* learner_masks,
                                      std::uint8_t* opponent_masks, std::size_t count) {
    const std::size_t index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= 2 * count * kFullMoveMaxCandidates) return;
    const bool for_opponent = index >= count * kFullMoveMaxCandidates;
    const std::size_t local = for_opponent ? index - count * kFullMoveMaxCandidates : index;
    const std::size_t env = local / kFullMoveMaxCandidates;
    const int candidate = static_cast<int>(local % kFullMoveMaxCandidates);
    const int learner = learner_is_p1[env] ? 0 : 1;
    const int player = for_opponent ? 1 - learner : learner;
    const int own_character = player == 0 ? p1_characters[env] : p2_characters[env];
    const Tables tables = matchup_tables(characters, p1_characters[env], p2_characters[env], config);
    const std::uint8_t legal_candidate = candidate < characters[own_character].candidates.count &&
        candidate_legal(tables, states[env], player, characters[own_character].candidates, candidate);
    (for_opponent ? opponent_masks : learner_masks)[local] = legal_candidate;
}

__global__ void env_step_kernel(const DeviceCharacter* characters, const int* p1_characters,
                                const int* p2_characters, FullEngineConfig config, EnvConfig env_config,
                                const std::uint8_t* learner_is_p1, const float* aggression,
                                const std::int64_t* learner_candidates, const std::int64_t* opponent_candidates,
                                State* states, Memory* memories, const NoiseLevels* noise, float* rewards,
                                float* sparse_rewards,
                                std::uint8_t* terminated,
                                std::int8_t* outcomes, unsigned long long* frames, unsigned int* tally,
                                const int* scenarios, unsigned int* scenario_tally, unsigned int* learner_tally,
                                unsigned int* bot_tally,
                                std::uint64_t seed, std::uint64_t counter, std::size_t count) {
    const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) return;
    const int p1 = p1_characters[env];
    const int p2 = p2_characters[env];
    const Tables tables = matchup_tables(characters, p1, p2, config);
    const int learner = learner_is_p1[env] ? 0 : 1;
    const CandidateMap learner_map = characters[learner == 0 ? p1 : p2].candidates;
    const CandidateMap opponent_map = characters[learner == 0 ? p2 : p1].candidates;
    State state = states[env];
    const EnvStepResult result = env_step(
        tables, state, learner, learner_map, opponent_map, static_cast<int>(learner_candidates[env]),
        opponent_candidates != nullptr ? static_cast<int>(opponent_candidates[env]) : -1, aggression[env],
        env_config, seed, env, counter, memories != nullptr ? memories + env : nullptr,
        noise != nullptr ? noise + env : nullptr);
    states[env] = state;
    rewards[env] = result.reward;
    sparse_rewards[env] = result.sparse_reward;
    terminated[env] = result.terminated ? 1U : 0U;
    outcomes[env] = result.outcome;
    atomicAdd(frames, static_cast<unsigned long long>(result.frames));
    if (result.terminated) {
        const int opponent_character = learner == 0 ? p2 : p1;
        const int bucket = min(4, max(0, static_cast<int>(aggression[env] * 4.0F + 0.5F)));
        const int column = result.outcome > 0 ? 0 : (result.outcome < 0 ? 1 : 2);
        atomicAdd(&tally[((opponent_character * 5 + bucket) * 2 + learner) * 3 + column], 1U);
        atomicAdd(&scenario_tally[scenarios[env] * 3 + column], 1U);
        atomicAdd(&learner_tally[(learner == 0 ? p1 : p2) * 3 + column], 1U);
        const int bot = decode_bot(aggression[env]);
        if (bot >= 0 && bot < kScriptedProfileCount) atomicAdd(&bot_tally[bot * 3 + column], 1U);
    }
}

__global__ void env_reset_kernel(const DeviceCharacter* characters, int* p1_characters, int* p2_characters,
                                 FullEngineConfig config, EnvConfig env_config, const std::uint8_t* learner_is_p1,
                                 const int* pending_opponents, const float* pending_aggression, float* aggression,
                                 const int* pending_scenarios, int* scenarios, const int* pending_learners,
                                 const std::uint8_t* only_done, State* states, Memory* memories,
                                 NoiseLevels* noise, std::uint64_t seed, std::uint64_t counter, std::size_t count) {
    const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env >= count) return;
    if (only_done != nullptr && !only_done[env]) return;
    if (memories != nullptr) memories[env] = Memory{};
    if (noise != nullptr) noise[env] = draw_noise_levels(env_config, seed, env, counter);
    if (learner_is_p1[env]) {
        p2_characters[env] = pending_opponents[env];
        if (pending_learners[env] >= 0) p1_characters[env] = pending_learners[env];
    } else {
        p1_characters[env] = pending_opponents[env];
        if (pending_learners[env] >= 0) p2_characters[env] = pending_learners[env];
    }
    aggression[env] = pending_aggression[env];
    scenarios[env] = pending_scenarios[env];
    const Tables tables = matchup_tables(characters, p1_characters[env], p2_characters[env], config);
    const int learner = learner_is_p1[env] ? 0 : 1;
    const CandidateMap opponent_map = characters[learner == 0 ? p2_characters[env] : p1_characters[env]].candidates;
    env_reset(tables, states[env], env_config, seed, env, counter, scenarios[env], learner, &opponent_map);
}

__global__ void noise_draw_kernel(EnvConfig env_config, NoiseLevels* noise, std::uint64_t seed, std::size_t count) {
    const std::size_t env = blockIdx.x * blockDim.x + threadIdx.x;
    if (env < count) noise[env] = draw_noise_levels(env_config, seed, env, 0);
}

}  // namespace

struct GpuFullMoveSimulator::Impl {
    std::vector<FullMoveCharacter> roster;
    full_core::EnvConfig config{};
    FullEngineConfig engine{};
    std::size_t environments = 0;

    // Roster tables (concatenated per kind; each character's Side points into them).
    DeviceArray<Move> moves;
    DeviceArray<Hit> hits;
    DeviceArray<Stance> stances;
    DeviceArray<int> candidate_moves;
    DeviceArray<DeviceCharacter> characters;

    DeviceArray<int> p1_characters;
    DeviceArray<int> p2_characters;
    DeviceArray<State> states;
    DeviceArray<Events> events;
    DeviceArray<Input> inputs;
    DeviceArray<std::uint8_t> new_round;
    DeviceArray<double> p1_x;
    DeviceArray<double> p2_x;
    DeviceArray<std::uint8_t> legal_masks;
    DeviceArray<unsigned long long> frames;

    DeviceArray<std::uint8_t> learner_is_p1;
    DeviceArray<int> pending_opponents;
    DeviceArray<float> pending_aggression;
    DeviceArray<float> aggression;
    DeviceArray<float> learner_observations;
    DeviceArray<float> opponent_observations;
    DeviceArray<Memory> memories;  // config.memory only
    DeviceArray<NoiseLevels> noise;  // config.live_noise only
    [[nodiscard]] NoiseLevels* noise_data() const noexcept { return config.live_noise ? noise.data() : nullptr; }
    int observation_size = kFullMoveObservationSize;
    [[nodiscard]] Memory* memory_data() const noexcept { return config.memory ? memories.data() : nullptr; }
    DeviceArray<std::uint8_t> learner_masks;
    DeviceArray<std::uint8_t> opponent_masks;
    DeviceArray<float> rewards;
    DeviceArray<float> sparse_rewards;
    DeviceArray<std::uint8_t> terminated;
    DeviceArray<std::uint8_t> truncated;
    DeviceArray<std::int8_t> outcomes;
    DeviceArray<unsigned int> tally;
    DeviceArray<unsigned int> scenario_tally;
    DeviceArray<unsigned int> learner_tally;
    std::vector<unsigned int> last_learner_tally;
    DeviceArray<unsigned int> bot_tally;
    std::vector<unsigned int> last_bot_tally;
    DeviceArray<int> scenarios;
    DeviceArray<int> pending_scenarios;
    DeviceArray<int> pending_learners;
    DeviceArray<std::int32_t> learner_tables;
    DeviceArray<std::int32_t> opponent_tables;
    std::vector<unsigned int> last_scenario_tally;

    void check_launch(const char* operation) const { check_cuda(cudaGetLastError(), operation); }

    void observe() {
        observe_kernel<<<blocks_for(environments), kThreads>>>(
            characters.data(), p1_characters.data(), p2_characters.data(), engine, config, states.data(),
            learner_is_p1.data(), memory_data(), noise_data(), observation_size, learner_observations.data(),
            opponent_observations.data(), learner_tables.data(), opponent_tables.data(), environments);
        check_launch("full-move observations");
        candidate_mask_kernel<<<blocks_for(2 * environments * kFullMoveMaxCandidates), kThreads>>>(
            characters.data(), p1_characters.data(), p2_characters.data(), engine, states.data(),
            learner_is_p1.data(), learner_masks.data(), opponent_masks.data(), environments);
        check_launch("full-move candidate masks");
    }
};

GpuFullMoveSimulator::GpuFullMoveSimulator(std::vector<FullMoveCharacter> roster, std::size_t environments,
                                           full_core::EnvConfig config)
    : impl_(std::make_unique<Impl>()) {
    if (roster.empty() || environments == 0) throw std::invalid_argument("full-move simulator needs characters and environments");
    if (config.max_round_frames <= 0 || config.hold_frames <= 0 || config.max_frames_per_decision < config.hold_frames ||
        !(config.start_distance_max >= config.start_distance_min) || !(config.start_distance_min > 0.0)) {
        throw std::invalid_argument("full-move environment configuration is invalid");
    }
    auto& impl = *impl_;
    impl.roster = std::move(roster);
    impl.config = config;
    impl.engine = impl.roster.front().tables.config;
    impl.environments = environments;

    std::vector<Move> moves;
    std::vector<Hit> hits;
    std::vector<Stance> stances;
    std::vector<int> candidate_moves;
    struct Offsets { std::size_t move, hit, stance, candidate; };
    std::vector<Offsets> offsets;
    for (const auto& character : impl.roster) {
        if (std::memcmp(&character.tables.config, &impl.engine, sizeof(FullEngineConfig)) != 0) {
            throw std::invalid_argument("every roster character must share one engine configuration");
        }
        offsets.push_back({moves.size(), hits.size(), stances.size(), candidate_moves.size()});
        moves.insert(moves.end(), character.tables.moves[0].begin(), character.tables.moves[0].end());
        hits.insert(hits.end(), character.tables.hits[0].begin(), character.tables.hits[0].end());
        stances.insert(stances.end(), character.tables.stances[0].begin(), character.tables.stances[0].end());
        candidate_moves.insert(candidate_moves.end(), character.candidate_moves.begin(), character.candidate_moves.end());
    }
    impl.moves.allocate(moves.size());
    impl.moves.upload(moves.data(), moves.size());
    impl.hits.allocate(hits.size());
    impl.hits.upload(hits.data(), hits.size());
    impl.stances.allocate(stances.size());
    impl.stances.upload(stances.data(), stances.size());
    impl.candidate_moves.allocate(candidate_moves.size());
    impl.candidate_moves.upload(candidate_moves.data(), candidate_moves.size());
    std::vector<DeviceCharacter> characters;
    for (std::size_t index = 0; index < impl.roster.size(); ++index) {
        const auto& character = impl.roster[index];
        DeviceCharacter device{};
        device.side.moves = impl.moves.data() + offsets[index].move;
        device.side.move_count = static_cast<int>(character.tables.moves[0].size());
        device.side.hits = impl.hits.data() + offsets[index].hit;
        device.side.stances = impl.stances.data() + offsets[index].stance;
        device.side.stance_count = static_cast<int>(character.tables.stances[0].size());
        device.side.rules = character.tables.rules[0];
        device.candidates.moves = impl.candidate_moves.data() + offsets[index].candidate;
        device.candidates.count = character.candidate_count();
        device.roster_id = character.roster_id;
        characters.push_back(device);
    }
    impl.characters.allocate(characters.size());
    impl.characters.upload(characters.data(), characters.size());

    const std::size_t n = environments;
    impl.p1_characters.allocate(n);
    impl.p2_characters.allocate(n);
    impl.states.allocate(n);
    impl.events.allocate(n);
    impl.inputs.allocate(2 * n);
    impl.new_round.allocate(n);
    impl.p1_x.allocate(n);
    impl.p2_x.allocate(n);
    impl.legal_masks.allocate(n * kFullMoveMaxCandidates);
    impl.frames.allocate(1);
    impl.learner_is_p1.allocate(n);
    impl.pending_opponents.allocate(n);
    impl.pending_aggression.allocate(n);
    impl.aggression.allocate(n);
    impl.observation_size = full_move_observation_size(config.memory, config.live_noise);
    impl.learner_observations.allocate(n * static_cast<std::size_t>(impl.observation_size));
    impl.opponent_observations.allocate(n * static_cast<std::size_t>(impl.observation_size));
    if (config.memory) impl.memories.allocate(n);
    if (config.live_noise) impl.noise.allocate(n);
    impl.learner_masks.allocate(n * kFullMoveMaxCandidates);
    impl.opponent_masks.allocate(n * kFullMoveMaxCandidates);
    impl.rewards.allocate(n);
    impl.sparse_rewards.allocate(n);
    impl.terminated.allocate(n);
    impl.truncated.allocate(n);
    impl.outcomes.allocate(n);
    impl.tally.allocate(impl.roster.size() * kAggressionBuckets * 2 * 3);
    impl.scenario_tally.allocate(static_cast<std::size_t>(kScenarioCount) * 3);
    impl.learner_tally.allocate(impl.roster.size() * 3);
    impl.bot_tally.allocate(static_cast<std::size_t>(full_core::kScriptedProfileCount) * 3);
    impl.scenarios.allocate(n);
    impl.pending_scenarios.allocate(n);
    impl.pending_learners.allocate(n);
    impl.learner_tables.allocate(n);
    impl.opponent_tables.allocate(n);
    const std::vector<int> keep_learners(n, -1);
    impl.pending_learners.upload(keep_learners.data(), n);
    const std::vector<std::uint8_t> p1_learners(n, 1U);
    impl.learner_is_p1.upload(p1_learners.data(), n);
    std::vector<State> initial(n);
    for (auto& state : initial) {
        const Tables tables = impl.roster.front().tables.view();
        reset(tables, state, -0.8, 0.8);
    }
    impl.states.upload(initial.data(), n);
}

GpuFullMoveSimulator::~GpuFullMoveSimulator() = default;
GpuFullMoveSimulator::GpuFullMoveSimulator(GpuFullMoveSimulator&&) noexcept = default;
GpuFullMoveSimulator& GpuFullMoveSimulator::operator=(GpuFullMoveSimulator&&) noexcept = default;

std::size_t GpuFullMoveSimulator::environment_count() const noexcept { return impl_->environments; }
const std::vector<FullMoveCharacter>& GpuFullMoveSimulator::roster() const noexcept { return impl_->roster; }
const full_core::EnvConfig& GpuFullMoveSimulator::config() const noexcept { return impl_->config; }
int GpuFullMoveSimulator::observation_size() const noexcept { return impl_->observation_size; }

void GpuFullMoveSimulator::set_matchups(std::span<const int> p1_characters, std::span<const int> p2_characters) {
    auto& impl = *impl_;
    if (p1_characters.size() != impl.environments || p2_characters.size() != impl.environments) {
        throw std::invalid_argument("one matchup per environment is required");
    }
    for (const int character : p1_characters) {
        if (character < 0 || static_cast<std::size_t>(character) >= impl.roster.size()) throw std::out_of_range("P1 character");
    }
    for (const int character : p2_characters) {
        if (character < 0 || static_cast<std::size_t>(character) >= impl.roster.size()) throw std::out_of_range("P2 character");
    }
    impl.p1_characters.upload(p1_characters.data(), p1_characters.size());
    impl.p2_characters.upload(p2_characters.data(), p2_characters.size());
}

void GpuFullMoveSimulator::reset_engine(std::span<const double> p1_x, std::span<const double> p2_x) {
    auto& impl = *impl_;
    impl.p1_x.upload(p1_x.data(), p1_x.size());
    impl.p2_x.upload(p2_x.data(), p2_x.size());
    engine_reset_kernel<<<blocks_for(impl.environments), kThreads>>>(
        impl.characters.data(), impl.p1_characters.data(), impl.p2_characters.data(), impl.engine,
        impl.p1_x.data(), impl.p2_x.data(), impl.states.data(), impl.environments);
    impl.check_launch("full-move engine reset");
}

void GpuFullMoveSimulator::step_engine(std::span<const full_core::Input> inputs,
                                       std::span<const std::uint8_t> new_round, std::span<const double> p1_x,
                                       std::span<const double> p2_x) {
    auto& impl = *impl_;
    impl.inputs.upload(inputs.data(), inputs.size());
    const bool rounds = !new_round.empty();
    if (rounds) {
        impl.new_round.upload(new_round.data(), new_round.size());
        impl.p1_x.upload(p1_x.data(), p1_x.size());
        impl.p2_x.upload(p2_x.data(), p2_x.size());
    }
    engine_step_kernel<<<blocks_for(impl.environments), kThreads>>>(
        impl.characters.data(), impl.p1_characters.data(), impl.p2_characters.data(), impl.engine,
        impl.inputs.data(), rounds ? impl.new_round.data() : nullptr, impl.p1_x.data(), impl.p2_x.data(),
        impl.states.data(), impl.events.data(), impl.frames.data(), impl.environments);
    impl.check_launch("full-move engine step");
}

std::vector<full_core::State> GpuFullMoveSimulator::download_states() const { return impl_->states.download(); }
std::vector<full_core::Events> GpuFullMoveSimulator::download_events() const { return impl_->events.download(); }

void GpuFullMoveSimulator::upload_states(std::span<const full_core::State> states) {
    impl_->states.upload(states.data(), states.size());
}

std::vector<std::uint8_t> GpuFullMoveSimulator::download_legal_mask(int player) const {
    auto& impl = *impl_;
    if (player != 0 && player != 1) throw std::out_of_range("player must be 0 or 1");
    legal_mask_kernel<<<blocks_for(impl.environments * kFullMoveMaxCandidates), kThreads>>>(
        impl.characters.data(), impl.p1_characters.data(), impl.p2_characters.data(), impl.engine,
        impl.states.data(), player, impl.legal_masks.data(), impl.environments);
    impl.check_launch("full-move legal mask");
    return impl.legal_masks.download();
}

void GpuFullMoveSimulator::set_learner_sides(std::span<const std::uint8_t> learner_is_p1) {
    impl_->learner_is_p1.upload(learner_is_p1.data(), learner_is_p1.size());
}

void GpuFullMoveSimulator::set_pending_opponents(std::span<const int> characters, std::span<const float> aggression,
                                                 std::span<const int> scenarios) {
    auto& impl = *impl_;
    for (const int character : characters) {
        if (character < 0 || static_cast<std::size_t>(character) >= impl.roster.size()) throw std::out_of_range("opponent");
    }
    impl.pending_opponents.upload(characters.data(), characters.size());
    if (aggression.empty()) {
        const std::vector<float> zero(impl.environments, 0.5F);
        impl.pending_aggression.upload(zero.data(), zero.size());
    } else {
        impl.pending_aggression.upload(aggression.data(), aggression.size());
    }
    if (scenarios.empty()) {
        const std::vector<int> neutral(impl.environments, kScenarioNeutral);
        impl.pending_scenarios.upload(neutral.data(), neutral.size());
    } else {
        for (const int scenario : scenarios) {
            if (scenario < 0 || scenario >= kScenarioCount) throw std::out_of_range("scenario");
        }
        impl.pending_scenarios.upload(scenarios.data(), scenarios.size());
    }
}

void GpuFullMoveSimulator::set_pending_learners(std::span<const int> characters) {
    auto& impl = *impl_;
    if (characters.size() != impl.environments) throw std::invalid_argument("one learner per environment is required");
    for (const int character : characters) {
        if (character < -1 || character >= static_cast<int>(impl.roster.size())) throw std::out_of_range("learner");
    }
    impl.pending_learners.upload(characters.data(), characters.size());
}

const std::int32_t* GpuFullMoveSimulator::learner_feature_tables() const noexcept { return impl_->learner_tables.data(); }
const std::int32_t* GpuFullMoveSimulator::opponent_feature_tables() const noexcept {
    return impl_->opponent_tables.data();
}
std::vector<std::int32_t> GpuFullMoveSimulator::download_learner_feature_tables() const {
    return impl_->learner_tables.download();
}

void GpuFullMoveSimulator::reset_all(std::uint64_t seed, std::uint64_t counter) {
    auto& impl = *impl_;
    env_reset_kernel<<<blocks_for(impl.environments), kThreads>>>(
        impl.characters.data(), impl.p1_characters.data(), impl.p2_characters.data(), impl.engine, impl.config,
        impl.learner_is_p1.data(), impl.pending_opponents.data(), impl.pending_aggression.data(),
        impl.aggression.data(), impl.pending_scenarios.data(), impl.scenarios.data(), impl.pending_learners.data(),
        nullptr, impl.states.data(), impl.memory_data(), impl.noise_data(), seed, counter, impl.environments);
    impl.check_launch("full-move reset");
    check_cuda(cudaMemset(impl.terminated.data(), 0, impl.environments), "clear terminated");
    impl.observe();
}

void GpuFullMoveSimulator::reset_done(std::uint64_t seed, std::uint64_t counter) {
    auto& impl = *impl_;
    env_reset_kernel<<<blocks_for(impl.environments), kThreads>>>(
        impl.characters.data(), impl.p1_characters.data(), impl.p2_characters.data(), impl.engine, impl.config,
        impl.learner_is_p1.data(), impl.pending_opponents.data(), impl.pending_aggression.data(),
        impl.aggression.data(), impl.pending_scenarios.data(), impl.scenarios.data(), impl.pending_learners.data(),
        impl.terminated.data(), impl.states.data(), impl.memory_data(), impl.noise_data(), seed, counter,
        impl.environments);
    impl.check_launch("full-move reset done");
    impl.observe();
}

void GpuFullMoveSimulator::step(const std::int64_t* learner_candidates, const std::int64_t* opponent_candidates,
                                std::uint64_t seed, std::uint64_t counter) {
    auto& impl = *impl_;
    env_step_kernel<<<blocks_for(impl.environments), kThreads>>>(
        impl.characters.data(), impl.p1_characters.data(), impl.p2_characters.data(), impl.engine, impl.config,
        impl.learner_is_p1.data(), impl.aggression.data(), learner_candidates, opponent_candidates,
        impl.states.data(), impl.memory_data(), impl.noise_data(), impl.rewards.data(), impl.sparse_rewards.data(),
        impl.terminated.data(),
        impl.outcomes.data(), impl.frames.data(), impl.tally.data(), impl.scenarios.data(),
        impl.scenario_tally.data(), impl.learner_tally.data(), impl.bot_tally.data(), seed, counter,
        impl.environments);
    impl.check_launch("full-move step");
    impl.observe();
}

const float* GpuFullMoveSimulator::learner_observations() const noexcept { return impl_->learner_observations.data(); }
std::vector<full_core::NoiseLevels> GpuFullMoveSimulator::download_noise_levels() const {
    return impl_->config.live_noise ? impl_->noise.download() : std::vector<full_core::NoiseLevels>{};
}
const std::uint8_t* GpuFullMoveSimulator::learner_masks() const noexcept { return impl_->learner_masks.data(); }
const float* GpuFullMoveSimulator::opponent_observations() const noexcept { return impl_->opponent_observations.data(); }
const std::uint8_t* GpuFullMoveSimulator::opponent_masks() const noexcept { return impl_->opponent_masks.data(); }
const float* GpuFullMoveSimulator::rewards() const noexcept { return impl_->rewards.data(); }
const float* GpuFullMoveSimulator::sparse_rewards() const noexcept { return impl_->sparse_rewards.data(); }
const std::uint8_t* GpuFullMoveSimulator::terminated() const noexcept { return impl_->terminated.data(); }
const std::uint8_t* GpuFullMoveSimulator::truncated() const noexcept { return impl_->truncated.data(); }
const std::int8_t* GpuFullMoveSimulator::outcomes() const noexcept { return impl_->outcomes.data(); }

std::vector<std::uint8_t> GpuFullMoveSimulator::download_terminated() const { return impl_->terminated.download(); }
std::vector<std::int8_t> GpuFullMoveSimulator::download_outcomes() const { return impl_->outcomes.download(); }
std::vector<float> GpuFullMoveSimulator::download_rewards() const { return impl_->rewards.download(); }
std::vector<float> GpuFullMoveSimulator::download_learner_observations() const {
    return impl_->learner_observations.download();
}
std::vector<std::uint8_t> GpuFullMoveSimulator::download_learner_masks() const { return impl_->learner_masks.download(); }

std::vector<int> GpuFullMoveSimulator::download_opponent_characters() const {
    const auto p1 = impl_->p1_characters.download();
    const auto p2 = impl_->p2_characters.download();
    const auto sides = impl_->learner_is_p1.download();
    std::vector<int> result(p1.size());
    for (std::size_t env = 0; env < p1.size(); ++env) result[env] = sides[env] ? p2[env] : p1[env];
    return result;
}

std::uint64_t GpuFullMoveSimulator::frames_simulated() const { return impl_->frames.download().front(); }

std::vector<std::uint32_t> GpuFullMoveSimulator::download_and_clear_tally() {
    const auto values = impl_->tally.download();
    impl_->last_scenario_tally = impl_->scenario_tally.download();
    impl_->last_learner_tally = impl_->learner_tally.download();
    impl_->last_bot_tally = impl_->bot_tally.download();
    check_cuda(cudaMemset(impl_->bot_tally.data(), 0, sizeof(unsigned int) * impl_->bot_tally.size()),
               "clear bot tally");
    check_cuda(cudaMemset(impl_->learner_tally.data(), 0, sizeof(unsigned int) * impl_->learner_tally.size()),
               "clear learner tally");
    check_cuda(cudaMemset(impl_->tally.data(), 0, sizeof(unsigned int) * impl_->tally.size()), "clear tally");
    check_cuda(cudaMemset(impl_->scenario_tally.data(), 0, sizeof(unsigned int) * impl_->scenario_tally.size()),
               "clear scenario tally");
    return {values.begin(), values.end()};
}

std::vector<std::uint32_t> GpuFullMoveSimulator::download_bot_tally() const {
    return {impl_->last_bot_tally.begin(), impl_->last_bot_tally.end()};
}

std::vector<std::uint32_t> GpuFullMoveSimulator::download_learner_tally() const {
    return {impl_->last_learner_tally.begin(), impl_->last_learner_tally.end()};
}

std::vector<std::uint32_t> GpuFullMoveSimulator::download_scenario_tally() const {
    return {impl_->last_scenario_tally.begin(), impl_->last_scenario_tally.end()};
}

FullMoveEnvSnapshot GpuFullMoveSimulator::snapshot() const {
    const auto& impl = *impl_;
    return {impl.states.download(), impl.p1_characters.download(), impl.p2_characters.download(),
            impl.learner_is_p1.download(), impl.aggression.download(), impl.pending_opponents.download(),
            impl.pending_aggression.download(), impl.terminated.download(), impl.scenarios.download(),
            impl.pending_scenarios.download(), impl.pending_learners.download()};
}

void GpuFullMoveSimulator::restore(const FullMoveEnvSnapshot& snapshot) {
    auto& impl = *impl_;
    const std::size_t n = impl.environments;
    if (snapshot.states.size() != n || snapshot.p1_characters.size() != n || snapshot.p2_characters.size() != n ||
        snapshot.learner_is_p1.size() != n || snapshot.aggression.size() != n ||
        snapshot.pending_opponents.size() != n || snapshot.pending_aggression.size() != n ||
        snapshot.terminated.size() != n || snapshot.scenarios.size() != n || snapshot.pending_scenarios.size() != n ||
        snapshot.pending_learners.size() != n) {
        throw std::invalid_argument("full-move snapshot does not match the environment count");
    }
    set_matchups(snapshot.p1_characters, snapshot.p2_characters);
    impl.states.upload(snapshot.states.data(), n);
    impl.learner_is_p1.upload(snapshot.learner_is_p1.data(), n);
    impl.aggression.upload(snapshot.aggression.data(), n);
    set_pending_opponents(snapshot.pending_opponents, snapshot.pending_aggression, snapshot.pending_scenarios);
    impl.scenarios.upload(snapshot.scenarios.data(), n);
    impl.pending_learners.upload(snapshot.pending_learners.data(), n);
    impl.terminated.upload(snapshot.terminated.data(), n);
    if (impl.config.memory) check_cuda(cudaMemset(impl.memories.data(), 0, sizeof(Memory) * n), "clear memory");
    if (impl.config.live_noise) {
        noise_draw_kernel<<<blocks_for(n), kThreads>>>(impl.config, impl.noise.data(), 0x5eedULL, n);
        impl.check_launch("draw noise levels");
    }
    impl.observe();
}

void GpuFullMoveSimulator::synchronize() const { check_cuda(cudaDeviceSynchronize(), "synchronize full-move simulator"); }

}  // namespace t8::v2
