cmake_minimum_required(VERSION 3.24)

project(tekken_lite_v2 LANGUAGES CXX CUDA)

option(T8_V2_BUILD_TESTS "Build CPU-oracle and CUDA parity tests" ON)
option(T8_V2_BUILD_BENCHMARKS "Build GPU throughput benchmarks" ON)

find_package(CUDAToolkit REQUIRED)

# This target is deliberately an oracle, not the production environment.
add_library(t8_v2_cpu_oracle STATIC src/sim.cpp)
target_include_directories(t8_v2_cpu_oracle PUBLIC include)
target_compile_features(t8_v2_cpu_oracle PUBLIC cxx_std_20)

add_library(t8_v2_gpu STATIC src/gpu_sim.cu)
target_include_directories(t8_v2_gpu PUBLIC include)
target_compile_features(t8_v2_gpu PUBLIC cxx_std_20)
target_link_libraries(t8_v2_gpu PUBLIC CUDA::cudart)
set_target_properties(t8_v2_gpu PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES native
)
target_compile_options(t8_v2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

add_library(t8_v2_ppo STATIC src/ppo.cu)
target_include_directories(t8_v2_ppo PUBLIC include)
target_compile_features(t8_v2_ppo PUBLIC cxx_std_20)
target_link_libraries(t8_v2_ppo PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(t8_v2_ppo PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
    CUDA_ARCHITECTURES native
)
target_compile_options(t8_v2_ppo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

if(MSVC)
    target_compile_options(t8_v2_cpu_oracle PRIVATE /W4 /permissive-)
else()
    target_compile_options(t8_v2_cpu_oracle PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(T8_V2_BUILD_TESTS)
    enable_testing()

    add_executable(t8_v2_parity_tests tests/parity_tests.cpp)
    target_link_libraries(t8_v2_parity_tests PRIVATE t8_v2_cpu_oracle)
    add_test(NAME t8_v2_cpu_contract COMMAND t8_v2_parity_tests)

    add_executable(t8_v2_gpu_parity_tests tests/gpu_parity_tests.cpp)
    target_link_libraries(t8_v2_gpu_parity_tests PRIVATE t8_v2_gpu t8_v2_cpu_oracle)
    add_test(NAME t8_v2_gpu_parity COMMAND t8_v2_gpu_parity_tests)

    add_executable(t8_v2_policy_tests tests/policy_tests.cpp)
    target_link_libraries(t8_v2_policy_tests PRIVATE t8_v2_ppo t8_v2_gpu)
    add_test(NAME t8_v2_gpu_policy COMMAND t8_v2_policy_tests)
endif()

if(T8_V2_BUILD_BENCHMARKS)
    add_executable(t8_v2_gpu_benchmark benchmarks/gpu_throughput.cpp)
    target_link_libraries(t8_v2_gpu_benchmark PRIVATE t8_v2_gpu CUDA::cudart)
endif()
