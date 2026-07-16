#include "t8_v2/gpu_sim.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string_view>

namespace {

void cuda_check(cudaError_t result, const char* operation) {
    if (result != cudaSuccess) throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(result));
}

std::size_t argument(int argc, char** argv, std::string_view name, std::size_t fallback) {
    for (int index = 1; index + 1 < argc; ++index) {
        if (argv[index] == name) return static_cast<std::size_t>(std::stoull(argv[index + 1]));
    }
    return fallback;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const std::size_t envs = argument(argc, argv, "--envs", 262144);
        const std::size_t steps = argument(argc, argv, "--steps", 2000);
        t8::v2::GpuSimulatorBatch simulator(envs);
        std::uint8_t* p1_actions = nullptr;
        std::uint8_t* p2_actions = nullptr;
        cuda_check(cudaMalloc(&p1_actions, envs), "allocate P1 actions");
        cuda_check(cudaMalloc(&p2_actions, envs), "allocate P2 actions");
        cuda_check(cudaMemset(p1_actions, static_cast<int>(t8::v2::Action::Jab), envs), "fill P1 actions");
        cuda_check(cudaMemset(p2_actions, static_cast<int>(t8::v2::Action::Db3), envs), "fill P2 actions");

        for (int warmup = 0; warmup < 100; ++warmup) {
            simulator.step_device(p1_actions, p2_actions);
            simulator.reset_done();
        }
        simulator.synchronize();

        cudaEvent_t begin{};
        cudaEvent_t end{};
        cuda_check(cudaEventCreate(&begin), "create begin event");
        cuda_check(cudaEventCreate(&end), "create end event");
        cuda_check(cudaEventRecord(begin), "record begin event");
        for (std::size_t step = 0; step < steps; ++step) {
            simulator.step_device(p1_actions, p2_actions);
            simulator.reset_done();
        }
        cuda_check(cudaEventRecord(end), "record end event");
        cuda_check(cudaEventSynchronize(end), "wait for benchmark");
        float elapsed_ms = 0.0F;
        cuda_check(cudaEventElapsedTime(&elapsed_ms, begin, end), "measure benchmark");

        const double decisions = static_cast<double>(envs) * steps;
        const double seconds = elapsed_ms / 1000.0;
        std::cout << std::fixed << std::setprecision(2)
                  << "GPU environments: " << envs << '\n'
                  << "Decision batches: " << steps << '\n'
                  << "Elapsed seconds: " << seconds << '\n'
                  << "Environment decisions/s: " << decisions / seconds << '\n'
                  << "Simulated frames/s: " << decisions * simulator.config().decision_frames / seconds << '\n';

        cudaEventDestroy(end);
        cudaEventDestroy(begin);
        cudaFree(p2_actions);
        cudaFree(p1_actions);
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
