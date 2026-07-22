cmake_minimum_required(VERSION 3.24)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    # Avoid CMake's `native` probe as the default: it can fail on headless builders and
    # during Visual Studio regeneration. Ship native code for NVIDIA's supported
    # Turing-through-Blackwell generations, with PTX in the newest image.
    set(CMAKE_CUDA_ARCHITECTURES "75-real;86-real;89-real;120" CACHE STRING
        "CUDA architectures to build (for example: 120, 89-real;120, or native)")
endif()

project(tekken_lite_v2 LANGUAGES CXX CUDA)

option(T8_V2_BUILD_TESTS "Build CPU-oracle and CUDA parity tests" ON)
option(T8_V2_BUILD_BENCHMARKS "Build GPU throughput benchmarks" ON)
option(T8_V2_PPO_FAST_MATH "Use approximate CUDA math in PPO probability calculations" OFF)

find_package(CUDAToolkit REQUIRED)

# This target is deliberately an oracle, not the production environment.
add_library(t8_v2_cpu_oracle STATIC src/sim.cpp)
target_include_directories(t8_v2_cpu_oracle PUBLIC include)
target_compile_features(t8_v2_cpu_oracle PUBLIC cxx_std_20)

add_library(t8_v2_roster STATIC src/roster.cpp)
target_include_directories(t8_v2_roster PUBLIC include)
target_compile_features(t8_v2_roster PUBLIC cxx_std_20)

add_library(t8_v2_gpu STATIC src/gpu_sim.cu)
target_include_directories(t8_v2_gpu PUBLIC include)
target_compile_features(t8_v2_gpu PUBLIC cxx_std_20)
target_link_libraries(t8_v2_gpu PUBLIC CUDA::cudart)
set_target_properties(t8_v2_gpu PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_gpu PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

add_library(t8_v2_ppo STATIC src/ppo.cu src/training_router.cu)
target_include_directories(t8_v2_ppo PUBLIC include)
target_compile_features(t8_v2_ppo PUBLIC cxx_std_20)
target_link_libraries(t8_v2_ppo PUBLIC CUDA::cudart CUDA::cublas)
set_target_properties(t8_v2_ppo PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_ppo PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>
)
if(T8_V2_PPO_FAST_MATH)
    target_compile_options(t8_v2_ppo PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
endif()

add_library(t8_v2_opponents STATIC src/opponents.cu src/temporal.cu)
target_include_directories(t8_v2_opponents PUBLIC include)
target_compile_features(t8_v2_opponents PUBLIC cxx_std_20)
target_link_libraries(t8_v2_opponents PUBLIC CUDA::cudart)
set_target_properties(t8_v2_opponents PROPERTIES
    CUDA_STANDARD 20
    CUDA_STANDARD_REQUIRED ON
    CUDA_SEPARABLE_COMPILATION OFF
)
target_compile_options(t8_v2_opponents PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math;-lineinfo>
)

if(T8_V2_BUILD_TESTS)
    enable_testing()

    add_executable(t8_v2_parity_tests tests/parity_tests.cpp)
    target_link_libraries(t8_v2_parity_tests PRIVATE t8_v2_cpu_oracle)
    add_test(NAME t8_v2_cpu_contract COMMAND t8_v2_parity_tests)

    add_executable(t8_v2_gpu_parity_tests tests/gpu_parity_tests.cpp)
    target_link_libraries(t8_v2_gpu_parity_tests PRIVATE t8_v2_gpu t8_v2_cpu_oracle)
    add_test(NAME t8_v2_gpu_parity COMMAND t8_v2_gpu_parity_tests)

    add_executable(t8_v2_policy_tests tests/policy_tests.cpp)
    target_link_libraries(t8_v2_policy_tests PRIVATE t8_v2_ppo t8_v2_gpu t8_v2_opponents)
    add_test(NAME t8_v2_gpu_policy COMMAND t8_v2_policy_tests)

    add_executable(t8_v2_rollout_tests tests/rollout_tests.cpp)
    target_link_libraries(t8_v2_rollout_tests PRIVATE t8_v2_ppo t8_v2_gpu)
    add_test(NAME t8_v2_gpu_rollout COMMAND t8_v2_rollout_tests)

    add_executable(t8_v2_ppo_math_tests tests/ppo_math_tests.cpp)
    target_link_libraries(t8_v2_ppo_math_tests PRIVATE t8_v2_ppo)
    add_test(NAME t8_v2_ppo_math COMMAND t8_v2_ppo_math_tests)

    add_executable(t8_v2_roster_tests tests/roster_tests.cpp)
    target_link_libraries(t8_v2_roster_tests PRIVATE t8_v2_roster)
    add_test(NAME t8_v2_roster_catalog COMMAND t8_v2_roster_tests
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated/opponent_profiles.csv
             ${CMAKE_CURRENT_SOURCE_DIR}/data/generated/character_move_specs.csv)
endif()

if(T8_V2_BUILD_BENCHMARKS)
    add_executable(t8_v2_gpu_benchmark benchmarks/gpu_throughput.cpp)
    target_link_libraries(t8_v2_gpu_benchmark PRIVATE t8_v2_gpu CUDA::cudart)

    add_executable(t8_v2_training_benchmark benchmarks/training_throughput.cpp)
    target_link_libraries(t8_v2_training_benchmark PRIVATE
        t8_v2_ppo t8_v2_gpu t8_v2_opponents CUDA::cudart)
endif()

add_executable(t8_v2_train src/train.cpp)
target_link_libraries(t8_v2_train PRIVATE t8_v2_ppo t8_v2_gpu t8_v2_opponents t8_v2_roster)

if(T8_V2_BUILD_TESTS AND WIN32)
    add_test(
        NAME t8_v2_exact_resume
        COMMAND powershell -NoProfile -ExecutionPolicy Bypass
                -File ${CMAKE_CURRENT_SOURCE_DIR}/tests/resume_smoke.ps1
                $<TARGET_FILE:t8_v2_train>
    )
endif()

if(T8_V2_BUILD_TESTS)
    find_package(Python3 COMPONENTS Interpreter)
    if(Python3_Interpreter_FOUND)
        add_test(
            NAME t8_v2_live_policy
            COMMAND ${CMAKE_COMMAND} -E env
                    PYTHONPATH=${CMAKE_CURRENT_SOURCE_DIR}/src
                    ${Python3_EXECUTABLE} -m pytest -q
                    -p no:cacheprovider --basetemp=${CMAKE_CURRENT_BINARY_DIR}/pytest-tmp
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_v2_live_policy.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_live_backends.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_vision.py
                    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_roster_system.py
        )
    endif()
endif()

function(t8_v2_enable_warnings target)
    if(MSVC)
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:/W4;/permissive->
            # nvcc 13.1 emits C stub declarations that trigger MSVC C4211;
            # suppress that generated-code warning while retaining /W4.
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/wd4211>
        )
    else()
        target_compile_options(${target} PRIVATE
            $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>
            $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall,-Wextra,-Wpedantic>
        )
    endif()
endfunction()

set(T8_V2_WARNING_TARGETS
    t8_v2_cpu_oracle t8_v2_roster t8_v2_gpu t8_v2_ppo t8_v2_opponents t8_v2_train)
if(T8_V2_BUILD_TESTS)
    list(APPEND T8_V2_WARNING_TARGETS
        t8_v2_parity_tests t8_v2_gpu_parity_tests t8_v2_policy_tests
        t8_v2_rollout_tests t8_v2_ppo_math_tests)
    list(APPEND T8_V2_WARNING_TARGETS t8_v2_roster_tests)
endif()
if(T8_V2_BUILD_BENCHMARKS)
    list(APPEND T8_V2_WARNING_TARGETS
        t8_v2_gpu_benchmark t8_v2_training_benchmark)
endif()
foreach(target IN LISTS T8_V2_WARNING_TARGETS)
    t8_v2_enable_warnings(${target})
endforeach()
