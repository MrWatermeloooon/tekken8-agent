#include "t8_v2/gpu_sim.hpp"
#include "t8_v2/opponents.hpp"
#include "t8_v2/ppo.hpp"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

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
    std::size_t checkpoint_interval = 10;
    std::size_t evaluation_interval = 10;
    std::size_t evaluation_episodes = 256;
    bool sparse_reward = false;
    std::filesystem::path run_directory;
    std::filesystem::path resume_checkpoint;
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
        else if (argument == "--epochs") options.epochs = static_cast<int>(parse_size(next(), argument));
        else if (argument == "--minibatch") options.minibatch_size = parse_size(next(), argument);
        else if (argument == "--learning-rate") options.learning_rate = std::stof(next());
        else if (argument == "--gamma") options.gamma = std::stof(next());
        else if (argument == "--gae-lambda") options.gae_lambda = std::stof(next());
        else if (argument == "--seed") options.seed = std::stoull(next());
        else if (argument == "--checkpoint-interval") options.checkpoint_interval = parse_size(next(), argument);
        else if (argument == "--eval-interval") options.evaluation_interval = parse_size(next(), argument);
        else if (argument == "--eval-episodes") options.evaluation_episodes = parse_size(next(), argument);
        else if (argument == "--reward") {
            const std::string_view reward = next();
            if (reward != "shaped" && reward != "sparse") {
                throw std::invalid_argument("--reward must be shaped or sparse");
            }
            options.sparse_reward = reward == "sparse";
        } else if (argument == "--run-dir") options.run_directory = next();
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
    if (options.minibatch_size > options.environments * options.horizon) {
        throw std::invalid_argument("minibatch cannot exceed rollout sample count");
    }
    if (options.run_directory.empty()) {
        options.run_directory = std::filesystem::path("runs") /
            (std::string("phase0_scripted_") + (options.sparse_reward ? "sparse" : "shaped") +
             "_seed" + std::to_string(options.seed));
    }
    return options;
}

struct Evaluation {
    std::size_t episodes = 0;
    std::size_t wins = 0;
    std::size_t losses = 0;
    std::size_t draws = 0;
    double win_rate = 0.0;
};

Evaluation evaluate(
    t8::v2::GpuActorCritic& learner,
    std::size_t requested_episodes,
    std::uint64_t seed) {
    const std::size_t environments = std::min({requested_episodes, learner.capacity(), std::size_t{256}});
    t8::v2::GpuSimulatorBatch simulator(environments);
    t8::v2::GpuScriptedOpponent opponent(environments);
    Evaluation result{};
    std::size_t decision = 0;
    while (result.episodes < requested_episodes) {
        const auto before = simulator.device_view();
        const auto p1 = learner.forward(before.observations_p1, before.action_masks_p1,
                                        environments, seed, decision, true);
        const auto* p2_actions = opponent.actions_device(
            before.observations_p2, before.action_masks_p2,
            environments, seed + 1, decision);
        simulator.step_device_i64(p1.actions, p2_actions);
        const auto done = simulator.download_terminated();
        const auto winners = simulator.download_winners();
        for (std::size_t lane = 0; lane < environments && result.episodes < requested_episodes; ++lane) {
            if (!done[lane]) continue;
            ++result.episodes;
            if (winners[lane] == 1) ++result.wins;
            else if (winners[lane] == 2) ++result.losses;
            else ++result.draws;
        }
        simulator.reset_done();
        if (++decision > requested_episodes * 2000) {
            throw std::runtime_error("evaluation exceeded safety decision limit");
        }
    }
    result.win_rate = result.episodes == 0 ? 0.0 :
        static_cast<double>(result.wins) / static_cast<double>(result.episodes);
    return result;
}

void append_metrics(
    const std::filesystem::path& path,
    std::size_t update,
    std::uint64_t environment_steps,
    std::string_view reward_mode,
    const t8::v2::PpoUpdateMetrics& metrics,
    double elapsed_seconds,
    const std::optional<Evaluation>& evaluation) {
    std::ofstream output(path, std::ios::app);
    if (!output) throw std::runtime_error("could not append metrics: " + path.string());
    output << std::setprecision(9)
           << "{\"update\":" << update
           << ",\"environment_steps\":" << environment_steps
           << ",\"reward_mode\":\"" << reward_mode << "\""
           << ",\"benchmark\":\"scripted_v1\""
           << ",\"elapsed_seconds\":" << elapsed_seconds
           << ",\"policy_loss\":" << metrics.policy_loss
           << ",\"value_loss\":" << metrics.value_loss
           << ",\"entropy\":" << metrics.entropy
           << ",\"approximate_kl\":" << metrics.approximate_kl
           << ",\"clip_fraction\":" << metrics.clip_fraction
           << ",\"gradient_norm\":" << metrics.gradient_norm
           << ",\"minibatches\":" << metrics.minibatches;
    if (evaluation) {
        output << ",\"evaluation\":{\"episodes\":" << evaluation->episodes
               << ",\"wins\":" << evaluation->wins
               << ",\"losses\":" << evaluation->losses
               << ",\"draws\":" << evaluation->draws
               << ",\"win_rate\":" << evaluation->win_rate << '}';
    }
    output << "}\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::filesystem::create_directories(options.run_directory / "checkpoints");
        const auto metrics_path = options.run_directory / "metrics.jsonl";
        const std::size_t policy_capacity = std::max(options.environments, options.minibatch_size);
        t8::v2::GpuSimulatorBatch simulator(options.environments);
        t8::v2::GpuActorCritic learner(policy_capacity, {}, options.seed);
        t8::v2::GpuScriptedOpponent opponent(options.environments);
        t8::v2::GpuRolloutBuffer rollout(options.environments, options.horizon);
        if (!options.resume_checkpoint.empty()) learner.load_checkpoint(options.resume_checkpoint);

        t8::v2::PpoUpdateConfig update_config{};
        update_config.epochs = options.epochs;
        update_config.minibatch_size = options.minibatch_size;
        update_config.learning_rate = options.learning_rate;
        const auto started = std::chrono::steady_clock::now();
        const std::string reward_mode = options.sparse_reward ? "sparse" : "shaped";

        for (std::size_t update = 1; update <= options.updates; ++update) {
            for (std::size_t step = 0; step < options.horizon; ++step) {
                const auto before = simulator.device_view();
                const auto p1 = learner.forward(before.observations_p1, before.action_masks_p1,
                                                options.environments, options.seed, update * options.horizon + step, false);
                const auto* p2_actions = opponent.actions_device(
                    before.observations_p2, before.action_masks_p2,
                    options.environments, options.seed + 1,
                    update * options.horizon + step);
                rollout.record_policy_device(step, before.observations_p1, before.action_masks_p1,
                                             p1.actions, p1.log_probabilities, p1.values);
                simulator.step_device_i64(p1.actions, p2_actions);
                const auto after = simulator.device_view();
                const float* rewards = options.sparse_reward
                    ? after.sparse_rewards_p1 : after.rewards_p1;
                rollout.record_outcome_device(step, rewards, after.terminated);
                simulator.reset_done();
            }
            const auto final_state = simulator.device_view();
            const auto bootstrap = learner.forward(
                final_state.observations_p1, final_state.action_masks_p1,
                options.environments, options.seed, update * options.horizon + options.horizon, true);
            rollout.compute_gae(bootstrap.values, options.gamma, options.gae_lambda, true);
            const auto metrics = learner.update_ppo(
                rollout.device_view(), update_config, options.seed + update * 10'000);

            std::optional<Evaluation> evaluation;
            if (update % options.evaluation_interval == 0 || update == options.updates) {
                // Frozen benchmark weights and stochastic sequence for every
                // update, shared by shaped/sparse runs with the same seed.
                evaluation = evaluate(learner, options.evaluation_episodes,
                                      options.seed + 500'000);
            }
            if (update % options.checkpoint_interval == 0 || update == options.updates) {
                const auto checkpoint = options.run_directory / "checkpoints" /
                    ("update_" + std::to_string(update) + ".t8ppo");
                learner.save_checkpoint(checkpoint);
            }
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            const std::uint64_t environment_steps = static_cast<std::uint64_t>(
                update * options.environments * options.horizon);
            append_metrics(metrics_path, update, environment_steps, reward_mode,
                           metrics, elapsed, evaluation);
            std::cout << "update=" << update << '/' << options.updates
                      << " steps=" << environment_steps
                      << " reward=" << reward_mode
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
