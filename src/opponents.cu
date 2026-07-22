#include "t8_v2/opponents.hpp"

#include <cuda_runtime.h>

#include <stdexcept>
#include <string>
#include <utility>

namespace t8::v2 {
namespace {

constexpr int kThreads = 256;

void check_cuda(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
    }
}

cudaStream_t as_stream(void* stream) { return reinterpret_cast<cudaStream_t>(stream); }
int blocks_for(std::size_t count) { return static_cast<int>((count + kThreads - 1) / kThreads); }

__device__ __forceinline__ std::uint64_t mix(std::uint64_t value) {
    value += 0x9e3779b97f4a7c15ULL;
    value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
    return value ^ (value >> 31U);
}

__device__ __forceinline__ float random_unit(std::uint64_t key, std::uint64_t salt) {
    return static_cast<float>(mix(key ^ mix(salt)) & 0xFFFFFFULL) / 16777216.0F;
}

__device__ __forceinline__ float clamp_probability(float value) {
    return fminf(1.0F, fmaxf(0.0F, value));
}

__device__ Action profiled_action(
    const float* observations,
    std::size_t obs,
    const OpponentProfileParameters& profile,
    std::uint64_t seed,
    std::uint64_t step,
    std::size_t lane) {
    const float distance = observations[obs + 3] * 7.2F;
    const bool opponent_attacking = observations[obs + 12] > 0.5F;
    const bool throw_threat = observations[obs + 17] > 0.5F;
    const float opponent_move_remaining = observations[obs + 14];
    const auto frame_decision = static_cast<std::uint64_t>(fmaxf(0.0F, observations[obs + 6]) * 900.0F);
    const std::uint64_t episode_key = step >= frame_decision ? step - frame_decision : 0;
    const std::uint64_t key = seed ^ mix(lane) ^ mix(episode_key) ^ mix(profile.id);
    const float episode_jitter = (random_unit(key, 991) - 0.5F) * 0.07F;
    const auto probability = [&](float base) { return clamp_probability(base + episode_jitter); };

    if (random_unit(key ^ mix(step), 1) < probability(profile.input_error_rate)) {
        const float error = random_unit(key ^ mix(step), 2);
        return error < 0.34F ? Action::Neutral :
            (error < 0.67F ? Action::SidestepLeft : Action::SidestepRight);
    }
    if (throw_threat) {
        if (random_unit(key ^ mix(step), 3) < probability(profile.throw_break_accuracy)) {
            const auto break_roll = static_cast<int>(mix(key ^ mix(step) ^ 4ULL) % 3ULL);
            return break_roll == 0 ? Action::ThrowBreak1 :
                (break_roll == 1 ? Action::ThrowBreak2 : Action::ThrowBreak12);
        }
        return Action::BlockHigh;
    }
    if (opponent_attacking) {
        const float reaction_span = static_cast<float>(max(1, profile.reaction_max - profile.reaction_min + 1));
        const float sampled_reaction = static_cast<float>(profile.reaction_min) +
            random_unit(key ^ mix(step), 5) * reaction_span;
        const float observed_attack_frames = (1.0F - opponent_move_remaining) * 30.0F;
        if (observed_attack_frames < sampled_reaction) return Action::Neutral;
        if (random_unit(key ^ mix(step), 6) < probability(profile.delay_frequency) * 0.25F) {
            return Action::Neutral;
        }
        if (distance < 1.05F && random_unit(key ^ mix(step), 7) < probability(profile.punish_accuracy) * 0.35F) {
            return random_unit(key ^ mix(step), 8) < profile.heat_usage ? Action::Hopkick : Action::Df1;
        }
        const float defense = random_unit(key ^ mix(step), 9);
        if (defense < probability(profile.low_block_accuracy) * 0.35F) return Action::BlockLow;
        if (defense < probability(profile.low_block_accuracy) * 0.45F) return Action::LowParry;
        return Action::BlockHigh;
    }
    if (distance > 1.15F) {
        const float total = profile.approach + profile.backdash +
            profile.sidestep_left + profile.sidestep_right;
        float movement = random_unit(key ^ mix(step), 10) * fmaxf(total, 1e-6F);
        movement -= profile.approach;
        if (movement <= 0.0F) {
            return profile.aggression > 0.65F ? Action::DashForward : Action::WalkForward;
        }
        movement -= profile.backdash;
        if (movement <= 0.0F) return Action::DashBack;
        movement -= profile.sidestep_left;
        if (movement <= 0.0F) return Action::SidestepLeft;
        return Action::SidestepRight;
    }
    const float offense = random_unit(key ^ mix(step), 11);
    float boundary = probability(profile.delay_frequency) * 0.16F;
    if (offense < boundary) return Action::Neutral;
    boundary += probability(profile.throw_frequency) * 0.34F;
    if (offense < boundary && distance < 0.65F) return Action::Throw;
    boundary += probability(profile.low_frequency) * 0.42F;
    if (offense < boundary) return Action::Db3;
    boundary += probability(profile.stance_entry_frequency) * 0.22F;
    if (offense < boundary) {
        return random_unit(key ^ mix(step), 12) < 0.5F ? Action::SidestepLeft : Action::Df1;
    }
    if (random_unit(key ^ mix(step), 13) < probability(profile.heat_usage) * 0.42F) {
        return random_unit(key ^ mix(step), 14) < 0.55F ? Action::F2 : Action::Hopkick;
    }
    return random_unit(key ^ mix(step), 15) < profile.aggression ? Action::Jab : Action::BlockHigh;
}

__global__ void scripted_actions_kernel(
    const float* observations,
    const std::uint8_t* masks,
    std::size_t count,
    std::uint64_t seed,
    std::uint64_t step,
    ScriptedOpponentSet opponent_set,
    const OpponentProfileParameters* profiles,
    const std::uint32_t* profile_assignments,
    std::size_t profile_count,
    std::int64_t* actions) {
    const std::size_t lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= count) return;
    const std::size_t obs = lane * kObservationSize;
    const std::size_t mask = lane * kActionCount;
    if (!masks[mask + static_cast<int>(Action::Jab)]) {
        actions[lane] = static_cast<std::int64_t>(Action::Neutral);
        return;
    }
    if (profiles != nullptr && profile_assignments != nullptr) {
        const std::uint32_t profile_index = profile_assignments[lane];
        if (profile_index >= profile_count) {
            actions[lane] = static_cast<std::int64_t>(Action::Neutral);
            return;
        }
        Action selected = profiled_action(
            observations, obs, profiles[profile_index], seed, step, lane);
        if (!masks[mask + static_cast<std::size_t>(selected)]) selected = Action::Neutral;
        actions[lane] = static_cast<std::int64_t>(selected);
        return;
    }
    const float distance = observations[obs + 3] * 7.2F;
    const bool opponent_attacking = observations[obs + 12] > 0.5F;
    const bool throw_threat = observations[obs + 17] > 0.5F;
    const int random = static_cast<int>(mix(seed ^ mix(step) ^ mix(lane)) % 100ULL);
    const int style = static_cast<int>(lane % 8);
    Action selected = Action::Neutral;

    if (throw_threat) {
        selected = Action::ThrowBreak12;
    } else if (opponent_set == ScriptedOpponentSet::HeldOutV2) {
        // Deliberately distinct thresholds and mixtures from the training suite.
        // These styles are audit-only so evaluation measures transfer rather
        // than replaying the behavior distribution used for PPO rollouts.
        if (style == 0) {
            if (distance < 0.68F) selected = Action::DashBack;
            else if (distance > 1.42F) selected = random < 45 ? Action::F2 : Action::DashForward;
            else selected = random < 38 ? Action::F2 : (random < 72 ? Action::Db3 : Action::SidestepLeft);
        } else if (style == 1) {
            if (opponent_attacking) selected = random < 55 ? Action::BlockHigh : Action::SidestepRight;
            else if (distance < 0.88F) selected = random < 60 ? Action::Jab : Action::Df1;
            else selected = Action::WalkForward;
        } else if (style == 2) {
            if (distance > 0.78F) selected = Action::DashForward;
            else selected = random < 34 ? Action::Throw : (random < 68 ? Action::Db3 : Action::Df1);
        } else if (style == 3) {
            if (opponent_attacking) selected = random < 50 ? Action::SidestepLeft : Action::SidewalkRight;
            else if (distance < 0.72F) selected = Action::DashBack;
            else if (distance < 1.12F) selected = Action::Hopkick;
            else selected = Action::DashForward;
        } else if (style == 4) {
            if (opponent_attacking) selected = random < 18 ? Action::BlockLow : Action::BlockHigh;
            else if (distance < 0.76F) selected = random < 70 ? Action::Jab : Action::Throw;
            else selected = Action::BlockHigh;
        } else if (style == 5) {
            if (opponent_attacking) selected = random < 40 ? Action::LowParry : Action::Jump;
            else if (distance < 0.94F) selected = random < 55 ? Action::Hopkick : Action::Db3;
            else selected = Action::WalkForward;
        } else if (style == 6) {
            if (distance < 0.58F) selected = Action::DashBack;
            else if (distance < 1.36F) selected = random < 65 ? Action::F2 : Action::SidewalkLeft;
            else selected = Action::WalkForward;
        } else {
            if (opponent_attacking && random < 52) {
                selected = random < 16 ? Action::BlockLow : (random < 38 ? Action::SidestepRight : Action::BlockHigh);
            } else if (distance > 1.20F) {
                selected = random < 62 ? Action::DashForward : Action::F2;
            } else if (random < 18) selected = Action::Throw;
            else if (random < 39) selected = Action::Db3;
            else if (random < 61) selected = Action::Df1;
            else if (random < 79) selected = Action::Hopkick;
            else selected = Action::Jab;
        }
    } else if (style == 0) {
        if (distance > 1.25F) selected = Action::DashForward;
        else if (distance > 0.95F) selected = random < 55 ? Action::F2 : Action::DashForward;
        else if (random < 30) selected = Action::Jab;
        else if (random < 55) selected = Action::Df1;
        else if (random < 75) selected = Action::Db3;
        else if (random < 90) selected = Action::Throw;
        else selected = Action::Hopkick;
    } else if (style == 1) {
        if (opponent_attacking && distance < 1.3F) selected = Action::BlockHigh;
        else if (distance < 0.62F) selected = Action::DashBack;
        else if (distance < 1.28F) selected = Action::F2;
        else selected = Action::WalkForward;
    } else if (style == 2) {
        if (opponent_attacking) selected = random < 25 ? Action::BlockLow : Action::BlockHigh;
        else if (distance < 0.92F) selected = random < 65 ? Action::Df1 : Action::Hopkick;
        else selected = Action::DashForward;
    } else if (style == 3) {
        if (distance < 0.50F) selected = random < 70 ? Action::Throw : Action::Jab;
        else if (distance < 0.90F) selected = Action::Jab;
        else selected = Action::DashForward;
    } else if (style == 4) {
        if (distance > 0.95F) selected = Action::DashForward;
        else if (random < 50) selected = Action::Db3;
        else if (random < 82) selected = Action::Df1;
        else selected = Action::Hopkick;
    } else if (style == 5) {
        if (opponent_attacking) selected = random < 50 ? Action::SidestepLeft : Action::SidestepRight;
        else if (distance < 0.92F) selected = random < 55 ? Action::Hopkick : Action::Db3;
        else selected = Action::DashForward;
    } else if (style == 6) {
        if (opponent_attacking) selected = random < 20 ? Action::BlockLow : Action::BlockHigh;
        else if (distance < 0.82F) selected = Action::Jab;
        else if (distance > 1.45F) selected = Action::WalkForward;
        else selected = Action::BlockHigh;
    } else {
        if (opponent_attacking && random < 45) selected = random < 12 ? Action::BlockLow : Action::BlockHigh;
        else if (distance > 1.15F) selected = random < 70 ? Action::DashForward : Action::F2;
        else if (random < 22) selected = Action::Jab;
        else if (random < 44) selected = Action::Df1;
        else if (random < 62) selected = Action::Db3;
        else if (random < 74) selected = Action::Throw;
        else if (random < 88) selected = Action::Hopkick;
        else selected = random < 94 ? Action::SidestepLeft : Action::SidestepRight;
    }
    if (!masks[mask + static_cast<std::size_t>(selected)]) selected = Action::Neutral;
    actions[lane] = static_cast<std::int64_t>(selected);
}

}  // namespace

struct GpuScriptedOpponent::Impl {
    std::size_t capacity;
    std::int64_t* actions = nullptr;
    OpponentProfileParameters* profiles = nullptr;
    std::uint32_t* profile_assignments = nullptr;
    std::size_t profile_count = 0;
    explicit Impl(std::size_t requested) : capacity(requested) {
        if (capacity == 0) throw std::invalid_argument("opponent capacity must be positive");
        try {
            check_cuda(cudaMalloc(&actions, sizeof(std::int64_t) * capacity), "allocate scripted actions");
            check_cuda(cudaMemset(actions, 0, sizeof(std::int64_t) * capacity),
                       "initialize scripted actions");
            check_cuda(cudaMalloc(&profile_assignments, sizeof(std::uint32_t) * capacity),
                       "allocate profile assignments");
            check_cuda(cudaMemset(profile_assignments, 0, sizeof(std::uint32_t) * capacity),
                       "initialize profile assignments");
        } catch (...) {
            cudaFree(profile_assignments);
            cudaFree(actions);
            throw;
        }
    }
    ~Impl() {
        cudaFree(profiles);
        cudaFree(profile_assignments);
        cudaFree(actions);
    }
};

GpuScriptedOpponent::GpuScriptedOpponent(std::size_t capacity)
    : impl_(std::make_unique<Impl>(capacity)) {}
GpuScriptedOpponent::~GpuScriptedOpponent() = default;
GpuScriptedOpponent::GpuScriptedOpponent(GpuScriptedOpponent&&) noexcept = default;
GpuScriptedOpponent& GpuScriptedOpponent::operator=(GpuScriptedOpponent&&) noexcept = default;
std::size_t GpuScriptedOpponent::capacity() const noexcept { return impl_->capacity; }

void GpuScriptedOpponent::set_profiles(
    std::span<const OpponentProfileParameters> profiles,
    void* stream) {
    if (profiles.empty()) throw std::invalid_argument("profile table cannot be empty");
    OpponentProfileParameters* uploaded = nullptr;
    check_cuda(cudaMalloc(&uploaded, sizeof(OpponentProfileParameters) * profiles.size()),
               "allocate opponent profiles");
    try {
        check_cuda(cudaMemcpyAsync(uploaded, profiles.data(),
                                   sizeof(OpponentProfileParameters) * profiles.size(),
                                   cudaMemcpyHostToDevice, as_stream(stream)),
                   "upload opponent profiles");
        check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize opponent profile upload");
    } catch (...) {
        cudaFree(uploaded);
        throw;
    }
    cudaFree(impl_->profiles);
    impl_->profiles = uploaded;
    impl_->profile_count = profiles.size();
}

void GpuScriptedOpponent::set_profile_assignments(
    std::span<const std::uint32_t> assignments,
    void* stream) {
    if (impl_->profiles == nullptr) throw std::logic_error("set profiles before profile assignments");
    if (assignments.size() != impl_->capacity) {
        throw std::invalid_argument("profile assignments must match opponent capacity");
    }
    for (const auto index : assignments) {
        if (index >= impl_->profile_count) throw std::out_of_range("profile assignment is out of range");
    }
    check_cuda(cudaMemcpyAsync(impl_->profile_assignments, assignments.data(),
                               sizeof(std::uint32_t) * assignments.size(),
                               cudaMemcpyHostToDevice, as_stream(stream)),
               "upload profile assignments");
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize profile assignment upload");
}

void GpuScriptedOpponent::set_action_history(
    std::span<const std::int64_t> actions,
    void* stream) {
    if (actions.size() != impl_->capacity) {
        throw std::invalid_argument("action history must match opponent capacity");
    }
    check_cuda(cudaMemcpyAsync(impl_->actions, actions.data(),
                               sizeof(std::int64_t) * actions.size(),
                               cudaMemcpyHostToDevice, as_stream(stream)),
               "upload opponent action history");
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize opponent action history upload");
}

bool GpuScriptedOpponent::uses_profiles() const noexcept { return impl_->profiles != nullptr; }
const std::uint32_t* GpuScriptedOpponent::profile_assignments_device() const noexcept {
    return impl_->profile_assignments;
}
const OpponentProfileParameters* GpuScriptedOpponent::profiles_device() const noexcept {
    return impl_->profiles;
}
const std::int64_t* GpuScriptedOpponent::actions_buffer_device() const noexcept {
    return impl_->actions;
}
std::size_t GpuScriptedOpponent::profile_count() const noexcept { return impl_->profile_count; }

const std::int64_t* GpuScriptedOpponent::actions_device(
    const float* observations,
    const std::uint8_t* masks,
    std::size_t count,
    std::uint64_t seed,
    std::uint64_t step,
    ScriptedOpponentSet opponent_set,
    void* stream) {
    if (!observations || !masks || count == 0 || count > impl_->capacity) {
        throw std::invalid_argument("invalid scripted-opponent input");
    }
    scripted_actions_kernel<<<blocks_for(count), kThreads, 0, as_stream(stream)>>>(
        observations, masks, count, seed, step, opponent_set,
        impl_->profiles, impl_->profile_assignments, impl_->profile_count, impl_->actions);
    check_cuda(cudaGetLastError(), "launch scripted opponent kernel");
    return impl_->actions;
}

std::vector<std::uint32_t> GpuScriptedOpponent::download_profile_assignments(
    std::size_t count,
    void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download exceeds opponent capacity");
    std::vector<std::uint32_t> host(count);
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(host.data(), impl_->profile_assignments,
                               sizeof(std::uint32_t) * count,
                               cudaMemcpyDeviceToHost, cuda_stream),
               "download profile assignments");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize profile assignment download");
    return host;
}

void GpuScriptedOpponent::synchronize(void* stream) const {
    check_cuda(cudaStreamSynchronize(as_stream(stream)), "synchronize scripted opponent");
}

std::vector<std::int64_t> GpuScriptedOpponent::download_actions(
    std::size_t count,
    void* stream) const {
    if (count > impl_->capacity) throw std::invalid_argument("download exceeds opponent capacity");
    std::vector<std::int64_t> host(count);
    const auto cuda_stream = as_stream(stream);
    check_cuda(cudaMemcpyAsync(host.data(), impl_->actions, sizeof(std::int64_t) * count,
                               cudaMemcpyDeviceToHost, cuda_stream), "download scripted actions");
    check_cuda(cudaStreamSynchronize(cuda_stream), "synchronize scripted action download");
    return host;
}

}  // namespace t8::v2
