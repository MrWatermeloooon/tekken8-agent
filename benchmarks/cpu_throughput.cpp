#include "t8_v2/sim.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <vector>

namespace {

struct Options {
    unsigned threads = 1;
    std::size_t envs = 256;
    std::size_t steps = 20'000;
};

std::size_t parse_positive(const char* text, std::string_view option) {
    const auto value = std::stoull(text);
    if (value == 0) {
        throw std::invalid_argument(std::string(option) + " must be positive");
    }
    return static_cast<std::size_t>(value);
}

Options parse_options(int argc, char** argv) {
    Options options{};
    for (int index = 1; index < argc; ++index) {
        const std::string_view argument = argv[index];
        if (index + 1 >= argc) {
            throw std::invalid_argument("missing value for " + std::string(argument));
        }
        if (argument == "--threads") {
            options.threads = static_cast<unsigned>(parse_positive(argv[++index], argument));
        } else if (argument == "--envs") {
            options.envs = parse_positive(argv[++index], argument);
        } else if (argument == "--steps") {
            options.steps = parse_positive(argv[++index], argument);
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }
    options.threads = std::max(1U, std::min(options.threads, static_cast<unsigned>(options.envs)));
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const Options options = parse_options(argc, argv);
        std::vector<double> checksums(options.threads, 0.0);
        std::vector<std::thread> workers;
        workers.reserve(options.threads);

        const auto start = std::chrono::steady_clock::now();
        for (unsigned thread_index = 0; thread_index < options.threads; ++thread_index) {
            const std::size_t begin = options.envs * thread_index / options.threads;
            const std::size_t end = options.envs * (thread_index + 1) / options.threads;
            workers.emplace_back([&, thread_index, begin, end] {
                std::vector<t8::v2::Simulator> simulators(end - begin);
                double checksum = 0.0;
                for (std::size_t step = 0; step < options.steps; ++step) {
                    for (std::size_t local = 0; local < simulators.size(); ++local) {
                        auto& sim = simulators[local];
                        const auto p1_index = (step + local + begin) % t8::v2::kActionCount;
                        const auto p2_index = (step * 7 + local + begin + 3) % t8::v2::kActionCount;
                        const auto result = sim.step(
                            static_cast<t8::v2::Action>(p1_index),
                            static_cast<t8::v2::Action>(p2_index));
                        checksum += result.reward_p1 + result.reward_p2 + result.state.p1.health * 1e-6;
                        if (result.terminated || result.truncated) {
                            sim.reset(step + local + begin);
                        }
                    }
                }
                checksums[thread_index] = checksum;
            });
        }
        for (auto& worker : workers) {
            worker.join();
        }
        const auto end = std::chrono::steady_clock::now();

        const double seconds = std::chrono::duration<double>(end - start).count();
        const auto decisions = static_cast<double>(options.envs) * static_cast<double>(options.steps);
        double checksum = 0.0;
        for (double value : checksums) checksum += value;

        std::cout << std::fixed << std::setprecision(3)
                  << "threads=" << options.threads << '\n'
                  << "envs=" << options.envs << '\n'
                  << "decisions=" << static_cast<std::uint64_t>(decisions) << '\n'
                  << "seconds=" << seconds << '\n'
                  << "decisions_per_second=" << decisions / seconds << '\n'
                  << "simulated_frames_per_second="
                  << decisions * t8::v2::Config{}.decision_frames / seconds << '\n'
                  << "checksum=" << checksum << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "benchmark error: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
